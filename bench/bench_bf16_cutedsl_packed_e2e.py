#!/usr/bin/env python3
"""Packed variable-length opt3 vs CuTeDSL Router correctness gate."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import _paths  # noqa: F401,E402

import torch  # noqa: E402

from kernels.msa_indexer_score import build_prefill_plan  # noqa: E402
from kernels.unified_block_sparse_plan import (  # noqa: E402
    FwdTilePlan,
    PrefillPolicy,
)
from kernels.unified_block_sparse_prefill import (  # noqa: E402
    prepare_unified_c123_aligned_chain,
)


BLOCK = 128
HQ = 32
HKV = 4
DIM = 128


def _prefix(lengths: tuple[int, ...], device: torch.device) -> torch.Tensor:
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int32, device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--q-lens", type=int, nargs=2, default=(777, 1153))
    parser.add_argument("--kv-lens", type=int, nargs=2, default=(33007, 16641))
    parser.add_argument("--shuffle-index-pages", action="store_true")
    parser.add_argument("--full-chain", action="store_true")
    parser.add_argument("--trace-stages", action="store_true")
    parser.add_argument("--overlap-fixed-score", action="store_true")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--pairs", type=int, default=0)
    parser.add_argument("--stage-pairs", type=int, default=0)
    parser.add_argument("--l2-flush-mib", type=int, default=64)
    args = parser.parse_args()
    if (args.pairs or args.stage_pairs) and not args.full_chain:
        raise ValueError("timing requires --full-chain")
    q_lens = tuple(args.q_lens)
    kv_lens = tuple(args.kv_lens)
    if any(
        q_len <= 0 or q_len > kv_len
        for q_len, kv_len in zip(q_lens, kv_lens)
    ):
        raise ValueError("require 0 < request Q <= request KV")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    total_q = sum(q_lens)
    total_k = sum(kv_lens)
    request_blocks = tuple((length + BLOCK - 1) // BLOCK for length in kv_lens)
    total_pages = sum(request_blocks)

    cu_q = _prefix(q_lens, device)
    cu_k = _prefix(kv_lens, device)
    prefix_lens = torch.tensor(
        [k_len - q_len for q_len, k_len in zip(q_lens, kv_lens)],
        dtype=torch.int32,
        device=device,
    )
    index_q = torch.randn(
        (total_q, HKV, DIM), dtype=torch.bfloat16, device=device
    )
    logical_index_k = torch.randn(
        (total_pages, 1, BLOCK, DIM), dtype=torch.bfloat16, device=device
    )
    if args.shuffle_index_pages:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + 10_003)
        logical_to_physical = torch.randperm(
            total_pages,
            dtype=torch.int64,
            device=device,
            generator=generator,
        )
        index_k = torch.empty_like(logical_index_k)
        index_k.index_copy_(0, logical_to_physical, logical_index_k)
        index_pages = logical_to_physical.to(torch.int32)
    else:
        index_k = logical_index_k
        index_pages = torch.arange(
            total_pages, dtype=torch.int32, device=device
        )

    index_plan = build_prefill_plan(
        torch.tensor(q_lens, dtype=torch.int32),
        torch.tensor(kv_lens, dtype=torch.int32),
        qo_offset=torch.tensor(
            [k_len - q_len for q_len, k_len in zip(q_lens, kv_lens)],
            dtype=torch.int32,
        ),
        num_idx_heads=HKV,
        num_idx_kv_heads=1,
        page_size=BLOCK,
    )
    score_rows = max(32, int(index_plan[3]["max_k_tiles"]))

    q = torch.randn((total_q, HQ, DIM), dtype=torch.bfloat16, device=device)
    k = torch.randn((total_k, HKV, DIM), dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    valid_kv_tokens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    placeholder_page_table = torch.empty(
        (len(q_lens), 0), dtype=torch.int32, device=device
    )

    def prepare(
        *,
        use_cutedsl_router: bool,
        overlap_fixed_score: bool | None = False,
    ):
        tile_plan = FwdTilePlan(
            dynamic_route=torch.empty(
                (HKV, total_q, 16), dtype=torch.int32, device=device
            ),
            page_table=placeholder_page_table,
            valid_kv_tokens=valid_kv_tokens,
            q_offset_tokens=tuple(
                k_len - q_len for q_len, k_len in zip(q_lens, kv_lens)
            ),
            policy=PrefillPolicy.ALIGNED_CONTIGUOUS,
        )
        return prepare_unified_c123_aligned_chain(
            index_q=index_q,
            index_k=index_k,
            index_plan=index_plan,
            index_page_indices=index_pages,
            candidate_storage=torch.empty(
                (HKV, score_rows, total_q), dtype=torch.float32, device=device
            ),
            prefix_lens=prefix_lens,
            q=q,
            k=k,
            v=v,
            tile_plan=tile_plan,
            out=torch.empty_like(q),
            fuse_fixed_state_in_dynamic_combine=False,
            overlap_fixed_score=overlap_fixed_score,
            compact_batch1_schedule=False,
            dynamic_full_score_tiles=False,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            use_cutedsl_router=use_cutedsl_router,
        )

    baseline = prepare(use_cutedsl_router=False)
    candidate = prepare(
        use_cutedsl_router=True,
        overlap_fixed_score=True if args.overlap_fixed_score else None,
    )
    baseline.run_score()
    baseline.run_plan()
    candidate.run_score()
    candidate.run_plan()
    torch.cuda.synchronize()

    expected_route = baseline.attention.tile_plan.dynamic_route
    actual_route = candidate.attention.tile_plan.dynamic_route
    route_mismatch = int((expected_route != actual_route).sum().item())
    if route_mismatch:
        first = (expected_route != actual_route).nonzero()[0].cpu().tolist()
        raise AssertionError(
            f"packed route mismatch count={route_mismatch}, first={first}"
        )
    for expected, actual, name in (
        (
            baseline.attention.dynamic_state[0],
            candidate.attention.dynamic_state[0],
            "row_ptr",
        ),
        (
            baseline.attention.dynamic_state[1],
            candidate.attention.dynamic_state[1],
            "q_indices",
        ),
    ):
        if not torch.equal(expected, actual):
            mismatch = int((expected != actual).sum().item())
            raise AssertionError(f"packed CSR {name} mismatch count={mismatch}")

    result: dict[str, object] = {
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "index_page_table": (
            "permuted" if args.shuffle_index_pages else "contiguous"
        ),
        "opt3_overlap_fixed_score": baseline.overlap_fixed_score,
        "cutedsl_overlap_fixed_score": candidate.overlap_fixed_score,
        "dynamic_route_exact_match": True,
        "csr_exact_match": True,
    }
    if args.full_chain:
        def run_attention(plan, label: str):
            if not args.trace_stages:
                return plan.attention.run()
            print(f"{label}: fixed launch", flush=True)
            plan.attention.run_fixed()
            torch.cuda.synchronize()
            print(f"{label}: fixed done", flush=True)
            print(f"{label}: dynamic launch", flush=True)
            plan.attention.run_dynamic()
            torch.cuda.synchronize()
            print(f"{label}: dynamic done", flush=True)
            print(f"{label}: merge launch", flush=True)
            result = plan.attention.run_merge()
            torch.cuda.synchronize()
            print(f"{label}: merge done", flush=True)
            return result

        expected_o, expected_lse = run_attention(baseline, "opt3")
        expected_o = expected_o.clone()
        expected_lse = expected_lse.clone()
        actual_o, actual_lse = run_attention(candidate, "cutedsl")
        actual_o = actual_o.clone()
        actual_lse = actual_lse.clone()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual_o, expected_o, atol=0.0, rtol=0.0)
        torch.testing.assert_close(actual_lse, expected_lse, atol=0.0, rtol=0.0)
        result.update(
            {
                "o_bitwise_match": True,
                "lse_bitwise_match": True,
                "max_abs_o_diff": float(
                    (actual_o.float() - expected_o.float()).abs().max()
                ),
                "max_abs_lse_diff": float(
                    (actual_lse - expected_lse).abs().max()
                ),
            }
        )
        if args.pairs:
            flush_buffer = torch.empty(
                args.l2_flush_mib * 1024 * 1024,
                dtype=torch.uint8,
                device=device,
            )
            for index in range(args.warmups):
                flush_buffer.add_(1)
                (candidate if index % 2 else baseline).run()
            torch.cuda.synchronize()

            baseline_samples: list[float] = []
            candidate_samples: list[float] = []
            for index in range(args.pairs):
                order = (
                    ((baseline, baseline_samples), (candidate, candidate_samples))
                    if index % 2 == 0
                    else ((candidate, candidate_samples), (baseline, baseline_samples))
                )
                for plan, samples in order:
                    flush_buffer.add_(1)
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    plan.run()
                    end.record()
                    end.synchronize()
                    samples.append(float(begin.elapsed_time(end)))
            baseline_median = statistics.median(baseline_samples)
            candidate_median = statistics.median(candidate_samples)
            result.update(
                {
                    "l2_flush_mib": args.l2_flush_mib,
                    "opt3_samples_ms": baseline_samples,
                    "cutedsl_samples_ms": candidate_samples,
                    "opt3_median_ms": baseline_median,
                    "cutedsl_median_ms": candidate_median,
                    "regression_percent": (
                        candidate_median / baseline_median - 1.0
                    )
                    * 100.0,
                }
            )
        if args.stage_pairs:
            flush_buffer = torch.empty(
                args.l2_flush_mib * 1024 * 1024,
                dtype=torch.uint8,
                device=device,
            )
            stage_samples: dict[str, dict[str, list[float]]] = {
                "opt3": {},
                "cutedsl": {},
            }
            plans = (("opt3", baseline), ("cutedsl", candidate))
            for index in range(args.stage_pairs):
                order = plans if index % 2 == 0 else tuple(reversed(plans))
                for plan_name, plan in order:
                    stages = (
                        ("router", plan.run_score),
                        ("csr_lpt", plan.run_plan),
                        ("fixed", plan.attention.run_fixed),
                        ("dynamic", plan.attention.run_dynamic),
                        ("merge", plan.attention.run_merge),
                    )
                    for stage_name, stage in stages:
                        flush_buffer.add_(1)
                        begin = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        begin.record()
                        stage()
                        end.record()
                        end.synchronize()
                        stage_samples[plan_name].setdefault(
                            stage_name, []
                        ).append(float(begin.elapsed_time(end)))
            result["stage_samples_ms"] = stage_samples
            result["stage_medians_ms"] = {
                plan_name: {
                    stage_name: statistics.median(samples)
                    for stage_name, samples in per_stage.items()
                }
                for plan_name, per_stage in stage_samples.items()
            }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
