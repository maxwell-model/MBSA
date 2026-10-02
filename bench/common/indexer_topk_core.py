#!/usr/bin/env python3
"""① indexer score + ② top-k benchmark core (DeepGEMM fp8 + DeepSeek-V4 topk).

Two stages, measured in isolation from the sparse-attn kernel:

  1. idx score : PREFILL uses deep_gemm.fp8_mqa_logits (NON-paged, contiguous
                 full-sequence KV); DECODE uses fp8_paged_mqa_logits (block table).
                 This mirrors the production DSA path — the paged variant is the
                 decode kernel, NOT the prefill one.
  2. topk      : DeepSeek-V4 `deepseek_v4_topk_transform_512`, fused select +
                 paged transform.  This is the only timed top-k implementation.

Timing follows bench/common/bench_utils.py: decode times one invocation per
sample; prefill sums per-chunk kernel events over PREFILL_REP complete passes;
both report median + p10/p90.
"""

from __future__ import annotations

import os

import torch

from common.bench_utils import (
    PassTimer,
    Stats,
    bench_call,
    bench_pass,
    print_config,
)

DEV = "cuda"
DT = torch.bfloat16
FP8 = torch.float8_e4m3fn


class Config:
    """Env-overridable knobs; per-model wrapper passes the fallbacks."""

    def __init__(self, model, hidx, topk, topk_backends, didx=128, page=64,
                 chunk=4096, decode_bs=32):
        self.model = model
        self.HIDX = int(os.environ.get("HIDX", str(hidx)))
        self.DIDX = int(os.environ.get("DIDX", str(didx)))
        self.TOPK = int(os.environ.get("TOPK", str(topk)))
        self.PAGE = int(os.environ.get("PAGE", str(page)))
        self.CHUNK = int(os.environ.get("CHUNK", str(chunk)))
        self.DECODE_BS = int(os.environ.get("DECODE_BS", str(decode_bs)))
        self.SEED = int(os.environ.get("SEED", "1"))
        # Fixed decode workload: 32 requests each own a physical KV cache.
        self.INDEPENDENT_KV = int(os.environ.get("INDEPENDENT_KV", "1"))
        self.PREFILL_SYNC_EACH_CHUNK = int(
            os.environ.get("PREFILL_SYNC_EACH_CHUNK", "0")
        )
        self.TOPK_BACKENDS = [
            b for b in os.environ.get("TOPK_BACKENDS", topk_backends).replace(" ", "").split(",") if b
        ]
        self.PREFILL_SEQS = [
            int(x) for x in os.environ.get(
                "PREFILL_SEQS", "32768,65536,131072,262144,524288,1048576"
            ).replace(" ", "").split(",") if x
        ]
        self.DECODE_KVS = [
            int(x) for x in os.environ.get(
                "DECODE_KVS", "32768,65536,131072,262144,524288,1048576"
            ).replace(" ", "").split(",") if x
        ]


# ----------------------------- score kernel -----------------------------

def _make_identity_page_table(cfg, rows, kv_len):
    num_pages = (kv_len + cfg.PAGE - 1) // cfg.PAGE
    base = torch.arange(num_pages, dtype=torch.int32, device=DEV)
    return base.unsqueeze(0).expand(rows, -1).contiguous()


def _make_independent_page_table(cfg, rows, kv_len):
    num_pages = (kv_len + cfg.PAGE - 1) // cfg.PAGE
    base = torch.arange(num_pages, dtype=torch.int32, device=DEV)
    offs = (torch.arange(rows, dtype=torch.int32, device=DEV) * num_pages).unsqueeze(1)
    return (base.unsqueeze(0) + offs).contiguous()


def _make_fp8_k_cache(cfg, num_pages):
    k_fp8 = torch.randn(num_pages, cfg.PAGE, cfg.DIDX, device=DEV, dtype=DT).to(FP8)
    scale = torch.ones(num_pages, cfg.PAGE, dtype=torch.float32, device=DEV)
    k_u8 = k_fp8.view(torch.uint8).reshape(num_pages, cfg.PAGE * cfg.DIDX)
    s_u8 = scale.view(torch.uint8).reshape(num_pages, cfg.PAGE * 4)
    return torch.cat([k_u8, s_u8], dim=1).contiguous()


def _prepare_score(cfg, kcache_u8, seq_lens):
    """Build immutable launch metadata outside the measured operator region."""
    import deep_gemm
    try:
        from sglang.jit_kernel.dsa import deepgemm_paged_mqa_logits_split
    except ImportError:
        # Fallback to the repository-local tensor-layout wrapper.
        from sglang_dsa_paged_mqa_logits import deepgemm_paged_mqa_logits_split  # noqa: F811

    kv_fused = kcache_u8.view(kcache_u8.shape[0], cfg.PAGE, 1, cfg.DIDX + 4)
    ctx_lens_2d = seq_lens.unsqueeze(-1)
    schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
        ctx_lens_2d, cfg.PAGE, torch.cuda.get_device_properties(0).multi_processor_count,
    )
    torch.cuda.synchronize()
    return (
        deepgemm_paged_mqa_logits_split,
        deep_gemm.fp8_paged_mqa_logits,
        kv_fused,
        ctx_lens_2d,
        schedule_metadata,
    )


def _run_score(prepared, q_fp8, weights, page_table, max_seq_len):
    """Time only fp8_paged_mqa_logits, excluding static schedule preparation."""
    wrapper, kernel, kv_fused, ctx_lens_2d, schedule_metadata = prepared
    return wrapper(
        kernel, q_fp8, kv_fused, weights, ctx_lens_2d, page_table,
        schedule_metadata, max_seq_len, q_offset=q_fp8.shape[0],
    )


# ----------------------------- topk backends -----------------------------

_V4_PAGE_TABLES = {}


def _v4_page_table(cfg, rows, width):
    """SHARED identity logical->physical page table, cached per (rows, width).

    Every row maps page p -> p (no per-batch offset), so the kernel's
    page_to_slot(i) = (pt[i>>page_bits]<<page_bits)|(i&mask) == i: page_indices
    == raw_indices == the logical positions the sparse-attn kernels consume.
    The table maps every logical page to the same physical page.

    Built outside the timed region (the first, warmup call populates it) so the
    arange/expand never lands in a measured sample.
    """
    key = (rows, width)
    pt = _V4_PAGE_TABLES.get(key)
    if pt is None:
        num_pages = (width + cfg.PAGE - 1) // cfg.PAGE
        pt = torch.arange(num_pages, dtype=torch.int32, device=DEV).unsqueeze(0).expand(rows, -1).contiguous()
        _V4_PAGE_TABLES[key] = pt
    return pt


def _topk_v4(cfg, logits, seq_lens, out):
    """DeepSeek-V4 fused select + paged transform, exact top-`cfg.TOPK`.

    `out` receives the *physical slot* ids. With the identity page table above
    those equal the logical positions, which is what the sparse-attn kernel wants
    and what makes the result directly comparable to the other backends.
    """
    from v4_topk.v4_topk import topk_transform_512

    lg = logits if logits.stride(1) == 1 else logits.contiguous()
    page_table = _v4_page_table(cfg, lg.shape[0], lg.shape[1])
    topk_transform_512(lg, seq_lens.to(torch.int32), page_table, out, cfg.PAGE, None)


_TOPK_FNS = {"v4": _topk_v4}


# ----------------------------- decode -----------------------------

def _bench_decode(cfg, kv_len):
    rows = cfg.DECODE_BS
    num_pages = (kv_len + cfg.PAGE - 1) // cfg.PAGE
    padded_kv = num_pages * cfg.PAGE
    phys_pages = num_pages * (rows if cfg.INDEPENDENT_KV else 1)
    kcache_u8 = _make_fp8_k_cache(cfg, phys_pages)
    page_table = (_make_independent_page_table if cfg.INDEPENDENT_KV else _make_identity_page_table)(cfg, rows, kv_len)
    seq_lens = torch.full((rows,), kv_len, dtype=torch.int32, device=DEV)
    q_idx = torch.randn(rows, cfg.HIDX, cfg.DIDX, dtype=DT, device=DEV).to(FP8)
    weights = torch.randn(rows, cfg.HIDX, dtype=torch.float32, device=DEV) * 0.25
    out = torch.empty(rows, cfg.TOPK, dtype=torch.int32, device=DEV)
    prepared = _prepare_score(cfg, kcache_u8, seq_lens)
    latest = [None]

    def score_fn():
        latest[0] = _run_score(prepared, q_idx, weights, page_table, padded_kv)

    score_st = bench_call(score_fn)
    logits = latest[0]
    if logits is None:
        raise RuntimeError("fp8_paged_mqa_logits returned no output")

    topk_st = {}
    for name in cfg.TOPK_BACKENDS:
        fn = _TOPK_FNS[name]
        topk_st[name] = bench_call(lambda: fn(cfg, logits, seq_lens, out))
    return score_st, topk_st


# ----------------------------- prefill -----------------------------
# NOTE: DeepSeek/V3.2 (and LongCat) PREFILL indexer uses the NON-paged
# deep_gemm.fp8_mqa_logits (contiguous full-sequence KV), NOT fp8_paged_mqa_logits.
# The paged variant is the DECODE kernel. Confirmed against production DSA path.

def _make_fp8_kv_prefill(cfg, seq_len):
    """non-paged fp8 KV: (kv_data [S,DIDX] fp8, kv_sf [S] f32) for fp8_mqa_logits."""
    kv_bf16 = torch.randn(seq_len, cfg.DIDX, dtype=DT, device=DEV)
    sf = kv_bf16.abs().amax(-1, keepdim=True).clamp(1e-4) / 448.0
    kv_data = (kv_bf16 / sf).to(FP8)
    kv_sf = sf.squeeze(-1).float().contiguous()
    del kv_bf16
    return kv_data, kv_sf


def _bench_prefill(cfg, seq_len):
    if seq_len % cfg.CHUNK != 0:
        raise ValueError(f"seq_len={seq_len} must be divisible by CHUNK={cfg.CHUNK}")
    nchunks = seq_len // cfg.CHUNK

    import deep_gemm
    kv_data, kv_sf = _make_fp8_kv_prefill(cfg, seq_len)
    q_idx = torch.randn(cfg.CHUNK, cfg.HIDX, cfg.DIDX, dtype=DT, device=DEV).to(FP8)
    weights = torch.randn(cfg.CHUNK, cfg.HIDX, dtype=torch.float32, device=DEV) * 0.25
    out = torch.empty(cfg.CHUNK, cfg.TOPK, dtype=torch.int32, device=DEV)

    def one_pass():
        pt = PassTimer()
        for c in range(nchunks):
            q0 = c * cfg.CHUNK
            q1 = q0 + cfg.CHUNK
            kv_len = q1
            # fp8_mqa_logits uses chunk-local (ks, ke); ke = absolute end of each
            # row's causal prefix, kv_data[:kv_len] = the prefix KV.
            ks = torch.zeros(cfg.CHUNK, dtype=torch.int32, device=DEV)
            ke = torch.arange(q0 + 1, q1 + 1, dtype=torch.int32, device=DEV)
            with pt.section("score"):
                lg = deep_gemm.fp8_mqa_logits(
                    q_idx, (kv_data[:kv_len], kv_sf[:kv_len]), weights,
                    ks, ke, False, kv_len)
            for name in cfg.TOPK_BACKENDS:
                fn = _TOPK_FNS[name]
                with pt.section(name):
                    fn(cfg, lg, ke, out)
            if cfg.PREFILL_SYNC_EACH_CHUNK:
                # Diagnostic-only compatibility switch. Canonical full-table
                # timing keeps this disabled and synchronizes once per pass.
                torch.cuda.synchronize()
        pt.finish()
        return pt.totals()

    stats = bench_pass(one_pass)  # {"score": Stats, <backend>: Stats, ...}
    del kv_data, kv_sf
    torch.cuda.empty_cache()
    return nchunks, stats


# ----------------------------- driver -----------------------------

def main(model, hidx, topk, topk_backends, **kw):
    cfg = Config(model, hidx, topk, topk_backends, **kw)
    torch.manual_seed(cfg.SEED)
    print(
        f"# {cfg.model} idx score + topk | DeepGEMM fp8_mqa_logits(nonpaged prefill) / "
        f"fp8_paged_mqa_logits(decode) | GPU={torch.cuda.get_device_name(0)} | "
        f"Hidx={cfg.HIDX} Didx={cfg.DIDX} topk={cfg.TOPK} page={cfg.PAGE} chunk={cfg.CHUNK} "
        f"bs={cfg.DECODE_BS} independent_kv={cfg.INDEPENDENT_KV} "
        f"prefill_sync_each_chunk={cfg.PREFILL_SYNC_EACH_CHUNK} seed={cfg.SEED} "
        f"| topk_backends={cfg.TOPK_BACKENDS}",
        flush=True,
    )
    print_config()

    print("\n## Decode B=%d Sq=1" % cfg.DECODE_BS, flush=True)
    cols = ["idx_score"] + [f"topk_{n}" for n in cfg.TOPK_BACKENDS]
    print("kv_len | stage | " + Stats.header(), flush=True)
    for kv in cfg.DECODE_KVS:
        try:
            s, t = _bench_decode(cfg, kv)
            print(f"{kv} | idx_score | {s.fmt()}", flush=True)
            for n in cfg.TOPK_BACKENDS:
                print(f"{kv} | topk_{n} | {t[n].fmt()}", flush=True)
        except Exception as exc:
            print(f"{kv} | ERROR | {type(exc).__name__}: {exc}", flush=True)
            torch.cuda.empty_cache()

    print("\n## Prefill B=1 chunked", flush=True)
    print("seq | chunks | stage | " + Stats.header(), flush=True)
    for seq in cfg.PREFILL_SEQS:
        try:
            chunks, st = _bench_prefill(cfg, seq)
            print(f"{seq} | {chunks} | idx_score | {st['score'].fmt()}", flush=True)
            for n in cfg.TOPK_BACKENDS:
                print(f"{seq} | {chunks} | topk_{n} | {st[n].fmt()}", flush=True)
        except Exception as exc:
            print(f"{seq} | ERROR | {type(exc).__name__}: {exc}", flush=True)
            torch.cuda.empty_cache()
