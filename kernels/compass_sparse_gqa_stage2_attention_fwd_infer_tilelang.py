# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Inference-only TileLang forward for sparse_gqa stage-2 sparse attention.

This is the single-pass counterpart of
``compass_sparse_gqa_stage2_attention_fwd_tilelang.py``.  That kernel is the
*training* forward: it runs two K-tile passes (pass 1 computes the final
online-softmax (m, l); pass 2 recomputes scores, produces the output, and writes
the ``teacher_probs`` side-output needed by the distillation backward).  The
two-pass structure exists solely so the teacher probabilities can be normalized
by the final ``l``.

At inference time there is no teacher and no backward, so the standard
single-pass FlashAttention-2 online-softmax is sufficient and strictly cheaper:
each K/V slot is gathered once and ``Q@K^T`` is computed once, with the output
accumulator rescaled on the fly.  Compared with the training kernel this drops
one redundant K gather and one redundant ``Q@K^T`` GEMM per tile.

Layout matches the training kernel's default (per-KV-head grid):
  Grid = (S_LOCAL, BATCH, NUM_KV_HEADS)
  Each program owns GROUP_SIZE Q heads that share one KV head.

Numerics: same FA-2 math as the training forward; verified elsewhere against the
Triton reference.  Returns only the attention output (no LSE, no teacher).
"""

import math
import os
from typing import Optional

import torch

try:
    import tilelang
    from tilelang import language as T

    _TILELANG_OK = True
except ImportError:  # pragma: no cover - env without tilelang
    tilelang = None
    T = None
    _TILELANG_OK = False


def is_tilelang_available() -> bool:
    return _TILELANG_OK


if _TILELANG_OK:

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
    )
    def _stage2_attn_fwd_infer_main(
        Q,
        K,
        V,
        Topk,
        QGlobal,
        Out,
        softmax_scale,
        block_K: int = 64,
        num_stages: int = 1,
        threads: int = 256,
        dtype=T.bfloat16,
        accum_dtype=T.float32,
    ):
        """Single-pass FA-2 forward for sparse-topk attention, per-KV-head grid.

        One K-tile loop maintains running (m, l) via online softmax, rescales the
        output accumulator each time the row-max grows, and accumulates P@V.
        """
        # ---- compile-time constants ----
        s_local = T.const("s_local")
        batch = T.const("batch")
        num_q_heads = T.const("num_q_heads")
        head_dim = T.const("head_dim")
        s_full = T.const("s_full")
        num_kv_heads = T.const("num_kv_heads")
        value_dim = T.const("value_dim")
        topk = T.const("topk")
        group_size = num_q_heads // num_kv_heads
        group_block = 16  # padded group dim for shared-memory alignment

        # ---- tensor declarations ----
        Q: T.Tensor([s_local, batch, num_q_heads, head_dim], dtype)  # type: ignore[name-defined]
        K: T.Tensor([s_full, batch, num_kv_heads, head_dim], dtype)  # type: ignore[name-defined]
        V: T.Tensor([s_full, batch, num_kv_heads, value_dim], dtype)  # type: ignore[name-defined]
        Topk: T.Tensor([batch, s_local, topk], T.int32)  # type: ignore[name-defined]
        QGlobal: T.Tensor([s_local], T.int32)  # type: ignore[name-defined]
        Out: T.Tensor([s_local, batch, num_q_heads, value_dim], dtype)  # type: ignore[name-defined]

        with T.Kernel(s_local, batch, num_kv_heads, threads=threads) as (
            q_i,
            b_i,
            hk_i,
        ):
            # ---- shared memory ----
            q_shared = T.alloc_shared([group_block, head_dim], dtype)
            k_shared = T.alloc_shared([block_K, head_dim], dtype)
            v_shared = T.alloc_shared([block_K, value_dim], dtype)
            cand = T.alloc_shared([block_K], T.int32)
            valid = T.alloc_shared([block_K], "bool")
            prob_cast = T.alloc_shared([group_block, block_K], dtype)

            # ---- register fragments (reused across tiles) ----
            scores = T.alloc_fragment([group_block, block_K], accum_dtype)
            prob = T.alloc_fragment([group_block, block_K], accum_dtype)
            acc = T.alloc_fragment([group_block, value_dim], accum_dtype)

            # online-softmax state (per "real" head in the group)
            m = T.alloc_fragment([group_block], accum_dtype)
            l = T.alloc_fragment([group_block], accum_dtype)
            s_max = T.alloc_fragment([group_block], accum_dtype)
            m_new = T.alloc_fragment([group_block], accum_dtype)
            alpha = T.alloc_fragment([group_block], accum_dtype)
            l_inc = T.alloc_fragment([group_block], accum_dtype)

            q_global = QGlobal[q_i]

            # ----- load Q for this KV-head group -----
            for g_i, d_i in T.Parallel(group_block, head_dim):
                if g_i < group_size:
                    q_shared[g_i, d_i] = Q[
                        q_i, b_i, hk_i * group_size + g_i, d_i
                    ]
                else:
                    q_shared[g_i, d_i] = 0.0

            # ----- init online-softmax state + output accumulator -----
            for g_i in T.Parallel(group_block):
                m[g_i] = -3.4028234663852886e38  # -FLT_MAX
                l[g_i] = 0.0
            T.clear(acc)

            # ================================================================
            # Single pass — online softmax with on-the-fly acc rescale
            # ================================================================
            for tile_i in T.Pipelined(
                T.ceildiv(topk, block_K), num_stages=num_stages
            ):
                # -- load topk positions + causal mask --
                for k_i in T.Parallel(block_K):
                    slot = tile_i * block_K + k_i
                    if slot < topk:
                        raw = Topk[b_i, q_i, slot]
                        valid[k_i] = (
                            raw >= 0 and raw < s_full and raw <= q_global
                        )
                        cand[k_i] = T.if_then_else(valid[k_i], raw, 0)
                    else:
                        valid[k_i] = False
                        cand[k_i] = 0

                # -- gather K, V --
                for k_i, d_i in T.Parallel(block_K, head_dim):
                    k_shared[k_i, d_i] = T.if_then_else(
                        valid[k_i], K[cand[k_i], b_i, hk_i, d_i], 0
                    )
                for k_i, d_i in T.Parallel(block_K, value_dim):
                    v_shared[k_i, d_i] = T.if_then_else(
                        valid[k_i], V[cand[k_i], b_i, hk_i, d_i], 0
                    )

                # -- scores = Q @ K^T  [group_block, block_K] --
                T.clear(scores)
                T.gemm(
                    q_shared,
                    k_shared,
                    scores,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                # -- scale + causal mask (invalid slots -> -inf) --
                for g_i, k_i in T.Parallel(group_block, block_K):
                    scores[g_i, k_i] = T.if_then_else(
                        valid[k_i] and g_i < group_size,
                        scores[g_i, k_i] * softmax_scale,
                        -3.4028234663852886e38,
                    )

                # -- per-head row max of this tile --
                T.reduce_max(scores, s_max, dim=1, clear=True)

                # -- online update: m_new, rescale factor alpha --
                for g_i in T.Parallel(group_block):
                    m_new[g_i] = T.max(m[g_i], s_max[g_i])
                    alpha[g_i] = T.exp(m[g_i] - m_new[g_i])
                    m[g_i] = m_new[g_i]

                # -- rescale running sum l and output accumulator acc --
                for g_i in T.Parallel(group_block):
                    l[g_i] = l[g_i] * alpha[g_i]
                for g_i, d_i in T.Parallel(group_block, value_dim):
                    acc[g_i, d_i] = acc[g_i, d_i] * alpha[g_i]

                # -- p = exp(scores - m), accumulate l --
                for g_i, k_i in T.Parallel(group_block, block_K):
                    prob[g_i, k_i] = T.if_then_else(
                        valid[k_i] and g_i < group_size,
                        T.exp(scores[g_i, k_i] - m[g_i]),
                        0.0,
                    )
                T.reduce_sum(prob, l_inc, dim=1, clear=True)
                for g_i in T.Parallel(group_block):
                    l[g_i] = l[g_i] + l_inc[g_i]

                # -- acc += p @ V  [group_block, value_dim] --
                T.copy(prob, prob_cast)
                T.gemm(
                    prob_cast,
                    v_shared,
                    acc,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            # ================================================================
            # Normalize by l and store output
            # ================================================================
            for g_i, d_i in T.Parallel(group_size, value_dim):
                l_safe = T.max(l[g_i], 1.0e-20)
                Out[q_i, b_i, hk_i * group_size + g_i, d_i] = (
                    acc[g_i, d_i] / l_safe
                )


def sparse_gqa_stage2_attn_fwd_infer_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_indices: torch.Tensor,
    q_global: torch.Tensor,
    softmax_scale: float,
    out: Optional[torch.Tensor] = None,
):
    """Run the single-pass inference forward kernel and return ``output``.

    Same interface as ``sparse_gqa_stage2_attn_fwd_tilelang`` minus the
    ``teacher_probs`` side-output and the LSE return (neither is needed for
    inference).
    """
    if not _TILELANG_OK:
        raise ImportError(
            "TileLang is required for sparse_gqa_stage2_attn_fwd_infer_tilelang."
        )

    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
    ):
        raise ValueError(
            "TileLang stage2 attention infer forward supports bf16 q/k/v only."
        )

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    topk_i32 = topk_indices.to(torch.int32).contiguous()
    qg_i32 = q_global.to(torch.int32).contiguous()

    s_local, batch, num_q_heads, head_dim = q_c.shape
    num_kv_heads = k_c.shape[2]
    value_dim = v_c.shape[3]
    group_size = num_q_heads // num_kv_heads

    if group_size > 16:
        raise ValueError(
            "TileLang stage2 attention infer forward supports GQA group_size <= 16."
        )

    expected_out_shape = (s_local, batch, num_q_heads, value_dim)
    if out is None:
        out = torch.empty(
            expected_out_shape,
            device=q_c.device,
            dtype=torch.bfloat16,
        )
    elif (
        out.shape != expected_out_shape
        or out.device != q_c.device
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
    ):
        raise ValueError(
            f"preallocated out must be contiguous bf16 {expected_out_shape} "
            f"on {q_c.device}"
        )

    block_k = int(os.environ.get("SPARSE_GQA_TL_FWD_BLOCK_K", "64"))
    num_stages = int(os.environ.get("SPARSE_GQA_TL_FWD_STAGES", "1"))
    threads = int(os.environ.get("SPARSE_GQA_TL_FWD_THREADS", "256"))

    _stage2_attn_fwd_infer_main(
        q_c,
        k_c,
        v_c,
        topk_i32,
        qg_i32,
        out,
        float(softmax_scale),
        block_K=block_k,
        num_stages=num_stages,
        threads=threads,
    )

    return out
