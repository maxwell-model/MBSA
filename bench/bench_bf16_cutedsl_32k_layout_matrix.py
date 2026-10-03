#!/usr/bin/env python3
"""Controlled 32K Index-K layout/resource/launch-order matrix."""

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
from kernels.bf16_indexer_cutedsl import (  # noqa: E402
    bf16_indexer_local_top16,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--pairs", type=int, default=21)
    parser.add_argument("--l2-flush-mib", type=int, default=256)
    args = parser.parse_args()

    kv = 32768
    case = Case(kv, args.seed, fused_state=True, partial_dtype=torch.bfloat16)
    logical_index_k = case.index_k
    identity_pages = case.page_ids

    baseline = build_overlap(case, prepare_lpt_schedule=False)
    identity = build_overlap(
        case,
        prepare_lpt_schedule=False,
        use_cutedsl_router=True,
    )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed + 10_003)
    logical_to_physical = torch.randperm(
        case.pages,
        dtype=torch.int64,
        device="cuda",
        generator=generator,
    )
    permuted_index_k = torch.empty_like(logical_index_k)
    permuted_index_k.index_copy_(
        0,
        logical_to_physical,
        logical_index_k,
    )
    case.index_k = permuted_index_k
    case.page_ids = logical_to_physical.to(torch.int32)
    permuted = build_overlap(
        case,
        prepare_lpt_schedule=False,
        use_cutedsl_router=True,
    )
    case.index_k = logical_index_k
    case.page_ids = identity_pages

    def run_score(
        plan: object,
        *,
        use_page_table: bool,
        reduced_acc_pipeline: bool,
    ) -> None:
        bf16_indexer_local_top16(
            plan.index_q,
            plan.cutedsl_index_k,
            partition_blocks=plan.candidate_partition_blocks,
            page_indices=plan.cutedsl_index_pages,
            logical_k_tokens=plan.cutedsl_logical_k_tokens,
            use_page_table=use_page_table,
            reduced_acc_pipeline=reduced_acc_pipeline,
            out_compact_ids=plan.candidate_storage.view(torch.int32),
        )

    def run_chain(
        plan: object,
        score: Callable[[], None],
        *,
        score_first: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current = torch.cuda.current_stream(case.q.device)
        plan.inputs_ready.record(current)
        if score_first:
            score()
            plan.fixed_stream.wait_event(plan.inputs_ready)
            with torch.cuda.stream(plan.fixed_stream):
                plan.attention.run_fixed()
                plan.fixed_done.record(plan.fixed_stream)
        else:
            plan.fixed_stream.wait_event(plan.inputs_ready)
            with torch.cuda.stream(plan.fixed_stream):
                plan.attention.run_fixed()
                plan.fixed_done.record(plan.fixed_stream)
            score()
        plan.run_plan()
        current.wait_event(plan.fixed_done)
        plan.attention.run_dynamic()
        return plan.attention.run_merge()

    variants: dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]] = {
        "opt3": baseline.run,
    }
    layout_variants = (
        ("identity_direct", identity, False),
        ("identity_indirect", identity, True),
        ("permuted_indirect", permuted, True),
    )
    for layout_name, plan, use_page_table in layout_variants:
        for footprint_name, reduced in (("full", False), ("reduced", True)):
            for order_name, score_first in (
                ("fixed_first", False),
                ("score_first", True),
            ):
                name = f"{layout_name}_{footprint_name}_{order_name}"

                def call(
                    plan: object = plan,
                    use_page_table: bool = use_page_table,
                    reduced: bool = reduced,
                    score_first: bool = score_first,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    return run_chain(
                        plan,
                        lambda: run_score(
                            plan,
                            use_page_table=use_page_table,
                            reduced_acc_pipeline=reduced,
                        ),
                        score_first=score_first,
                    )

                variants[name] = call

    with torch.cuda.stream(case.stream):
        baseline.run_score()
        baseline.run_plan()
        reference_route = baseline.attention.tile_plan.dynamic_route.clone()
        reference_o, reference_lse = baseline.run()
        reference_o = reference_o.clone()
        reference_lse = reference_lse.clone()
    case.stream.synchronize()

    correctness: dict[str, object] = {}
    for name, call in variants.items():
        if name == "opt3":
            continue
        with torch.cuda.stream(case.stream):
            out, lse = call()
            out = out.clone()
            lse = lse.clone()
        case.stream.synchronize()
        layout_plan = permuted if name.startswith("permuted") else identity
        route = layout_plan.attention.tile_plan.dynamic_route
        torch.testing.assert_close(route, reference_route, rtol=0.0, atol=0.0)
        torch.testing.assert_close(out, reference_o, rtol=0.0, atol=0.0)
        torch.testing.assert_close(lse, reference_lse, rtol=0.0, atol=0.0)
        correctness[name] = "bitwise"

    flush = torch.empty(
        args.l2_flush_mib * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda",
    )
    names = list(variants)
    for warmup in range(args.warmups):
        order = names[warmup % len(names) :] + names[: warmup % len(names)]
        with torch.cuda.stream(case.stream):
            for name in order:
                flush.add_(1)
                variants[name]()
    case.stream.synchronize()

    samples: dict[str, list[float]] = {name: [] for name in names}
    for pair in range(args.pairs):
        offset = pair % len(names)
        order = names[offset:] + names[:offset]
        if pair % 2:
            order.reverse()
        for name in order:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(case.stream):
                flush.add_(1)
                begin.record(case.stream)
                variants[name]()
                end.record(case.stream)
            end.synchronize()
            samples[name].append(float(begin.elapsed_time(end)))

    medians = {name: statistics.median(values) for name, values in samples.items()}
    opt3_ms = medians["opt3"]
    result = {
        "kv": kv,
        "sq": case.sq,
        "seed": args.seed,
        "warmups": args.warmups,
        "pairs": args.pairs,
        "l2_flush_mib": args.l2_flush_mib,
        "correctness": correctness,
        "results": {
            name: {
                "median_ms": medians[name],
                "delta_vs_opt3_percent": (medians[name] / opt3_ms - 1.0) * 100.0,
                "samples_ms": samples[name],
            }
            for name in names
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
