#!/usr/bin/env python3
"""End-to-end opt3 vs CuTeDSL router gate with cold-L2 measurements."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import _paths  # noqa: F401,E402

import torch  # noqa: E402

from bench.common.sparse_gqa_workloads import Case, build_overlap  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--pairs", type=int, default=11)
    parser.add_argument("--l2-flush-mib", type=int, default=256)
    args = parser.parse_args()

    case = Case(
        args.kv,
        args.seed,
        fused_state=True,
        partial_dtype=torch.bfloat16,
    )
    use_lpt = args.kv < 32768
    baseline = build_overlap(case, prepare_lpt_schedule=use_lpt)
    candidate = build_overlap(
        case,
        prepare_lpt_schedule=use_lpt,
        use_cutedsl_router=True,
    )
    flush_buffer = torch.empty(
        args.l2_flush_mib * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda",
    )

    with torch.cuda.stream(case.stream):
        baseline.run_score()
        baseline.run_plan()
        candidate.run_score()
        candidate.run_plan()
    case.stream.synchronize()
    baseline_route = baseline.attention.tile_plan.dynamic_route
    candidate_route = candidate.attention.tile_plan.dynamic_route
    if not torch.equal(baseline_route, candidate_route):
        mismatch = int((baseline_route != candidate_route).sum().item())
        raise AssertionError(f"dynamic route mismatch in {mismatch} entries")
    for baseline_tensor, candidate_tensor, name in (
        (baseline.attention.dynamic_state[0], candidate.attention.dynamic_state[0], "row_ptr"),
        (baseline.attention.dynamic_state[1], candidate.attention.dynamic_state[1], "q_indices"),
    ):
        if not torch.equal(baseline_tensor, candidate_tensor):
            mismatch = int((baseline_tensor != candidate_tensor).sum().item())
            raise AssertionError(f"CSR {name} mismatch in {mismatch} entries")

    with torch.cuda.stream(case.stream):
        expected_o, expected_lse = baseline.run()
        expected_o = expected_o.clone()
        expected_lse = expected_lse.clone()
    case.stream.synchronize()
    with torch.cuda.stream(case.stream):
        actual_o, actual_lse = candidate.run()
        actual_o = actual_o.clone()
        actual_lse = actual_lse.clone()
    case.stream.synchronize()
    max_o_diff = float((actual_o.float() - expected_o.float()).abs().max())
    max_lse_diff = float((actual_lse - expected_lse).abs().max())
    torch.testing.assert_close(actual_o, expected_o, atol=0.0, rtol=0.0)
    torch.testing.assert_close(actual_lse, expected_lse, atol=0.0, rtol=0.0)

    for index in range(args.warmups):
        with torch.cuda.stream(case.stream):
            flush_buffer.add_(1)
            if index % 2:
                candidate.run()
            else:
                baseline.run()
    case.stream.synchronize()

    baseline_samples: list[float] = []
    candidate_samples: list[float] = []
    for index in range(args.pairs):
        order = (
            ((baseline, baseline_samples), (candidate, candidate_samples))
            if index % 2 == 0
            else ((candidate, candidate_samples), (baseline, baseline_samples))
        )
        for plan, samples in order:
            with torch.cuda.stream(case.stream):
                flush_buffer.add_(1)
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record(case.stream)
                plan.run()
                end.record(case.stream)
            end.synchronize()
            samples.append(float(begin.elapsed_time(end)))

    baseline_median = statistics.median(baseline_samples)
    candidate_median = statistics.median(candidate_samples)
    print(
        json.dumps(
            {
                "kv": args.kv,
                "sq": case.sq,
                "seed": args.seed,
                "l2_flush_mib": args.l2_flush_mib,
                "score_partition_blocks": candidate.candidate_partition_blocks,
                "score_partitions": (
                    1
                    if candidate.cutedsl_partition_scores is None
                    else int(candidate.cutedsl_partition_scores.shape[1])
                ),
                "dynamic_route_exact_match": True,
                "csr_exact_match": True,
                "o_bitwise_match": True,
                "lse_bitwise_match": True,
                "max_abs_o_diff": max_o_diff,
                "max_abs_lse_diff": max_lse_diff,
                "opt3_samples_ms": baseline_samples,
                "cutedsl_samples_ms": candidate_samples,
                "opt3_median_ms": baseline_median,
                "cutedsl_median_ms": candidate_median,
                "cutedsl_over_opt3": candidate_median / baseline_median,
                "regression_percent": (
                    candidate_median / baseline_median - 1.0
                )
                * 100.0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
