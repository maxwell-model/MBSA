#!/usr/bin/env python3
"""Cold-L2 stage attribution for opt3 and the CuTeDSL router chain."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from collections.abc import Callable

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

    case = Case(args.kv, args.seed, fused_state=True, partial_dtype=torch.bfloat16)
    use_lpt = args.kv < 32768
    plans = {
        "opt3": build_overlap(case, prepare_lpt_schedule=use_lpt),
        "cutedsl": build_overlap(
            case,
            prepare_lpt_schedule=use_lpt,
            use_cutedsl_router=True,
        ),
    }
    flush = torch.empty(
        args.l2_flush_mib * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda",
    )

    def enqueue_prefix(plan: object) -> None:
        current = torch.cuda.current_stream(case.q.device)
        plan.inputs_ready.record(current)
        if plan.fixed_first_score_overlap:
            plan.fixed_stream.wait_event(plan.inputs_ready)
            with torch.cuda.stream(plan.fixed_stream):
                plan.attention.run_fixed()
                plan.fixed_done.record(plan.fixed_stream)
            plan.run_score()
        else:
            plan.run_score()
            plan.fixed_stream.wait_event(plan.inputs_ready)
            with torch.cuda.stream(plan.fixed_stream):
                plan.attention.run_fixed()
                plan.fixed_done.record(plan.fixed_stream)
        plan.run_plan()
        current.wait_event(plan.fixed_done)

    def enqueue_score_csr(plan: object) -> None:
        plan.run_score()
        plan.run_plan()

    stages: dict[str, Callable[[object], None]] = {
        "score": lambda plan: plan.run_score(),
        "csr_lpt": lambda plan: plan.run_plan(),
        "score_csr": enqueue_score_csr,
        "fixed": lambda plan: plan.attention.run_fixed(),
        "score_csr_fixed_overlap": enqueue_prefix,
        "dynamic_merge": lambda plan: (
            plan.attention.run_dynamic(),
            plan.attention.run_merge(),
        ),
        "full_chain": lambda plan: plan.run(),
    }

    # Compile every stage and seed all route/fixed state before measurement.
    with torch.cuda.stream(case.stream):
        for plan in plans.values():
            plan.run()
    case.stream.synchronize()

    results: dict[str, object] = {
        "kv": args.kv,
        "sq": case.sq,
        "seed": args.seed,
        "l2_flush_mib": args.l2_flush_mib,
        "opt3_launch_order": "score_first",
        "cutedsl_launch_order": (
            "fixed_first"
            if plans["cutedsl"].fixed_first_score_overlap
            else "score_first"
        ),
    }
    for stage_name, enqueue in stages.items():
        samples = {name: [] for name in plans}
        for index in range(args.warmups):
            name = "cutedsl" if index % 2 else "opt3"
            with torch.cuda.stream(case.stream):
                flush.add_(1)
                enqueue(plans[name])
        case.stream.synchronize()
        for index in range(args.pairs):
            order = ("opt3", "cutedsl") if index % 2 == 0 else ("cutedsl", "opt3")
            for name in order:
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(case.stream):
                    flush.add_(1)
                    begin.record(case.stream)
                    enqueue(plans[name])
                    end.record(case.stream)
                end.synchronize()
                samples[name].append(float(begin.elapsed_time(end)))
        opt3_ms = statistics.median(samples["opt3"])
        cutedsl_ms = statistics.median(samples["cutedsl"])
        results[stage_name] = {
            "opt3_median_ms": opt3_ms,
            "cutedsl_median_ms": cutedsl_ms,
            "regression_percent": (cutedsl_ms / opt3_ms - 1.0) * 100.0,
            "opt3_samples_ms": samples["opt3"],
            "cutedsl_samples_ms": samples["cutedsl"],
        }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
