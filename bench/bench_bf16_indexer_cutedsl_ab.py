#!/usr/bin/env python3
"""Paired opt3-vs-CuTeDSL router gate with all buffers preallocated."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import _paths  # noqa: F401,E402

import torch  # noqa: E402

from kernels.bf16_indexer_cutedsl import (  # noqa: E402
    bf16_indexer_local_top16,
)
from kernels.msa_indexer_score import build_prefill_plan, score_prefill  # noqa: E402


BLOCK = 128
TOPK = 16


def _summary(samples: list[float]) -> dict[str, float | list[float]]:
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": ordered[0],
        "p90_ms": ordered[int(0.9 * (len(ordered) - 1))],
        "samples_ms": samples,
    }


def _paired(
    first,
    second,
    *,
    flush_l2,
    warmups: int,
    pairs: int,
) -> tuple[list[float], list[float]]:
    for index in range(warmups):
        if index % 2:
            flush_l2()
            second()
            flush_l2()
            first()
        else:
            flush_l2()
            first()
            flush_l2()
            second()
    torch.cuda.synchronize()
    first_samples: list[float] = []
    second_samples: list[float] = []
    for index in range(pairs):
        order = ((first, first_samples), (second, second_samples))
        if index % 2:
            order = tuple(reversed(order))
        events = []
        for fn, destination in order:
            flush_l2()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            fn()
            end.record()
            events.append((begin, end, destination))
        events[-1][1].synchronize()
        for begin, end, destination in events:
            destination.append(float(begin.elapsed_time(end)))
    return first_samples, second_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=int, default=8192)
    parser.add_argument("--kv", type=int, default=8192)
    parser.add_argument("--q-heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--partition-blocks", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--pairs", type=int, default=31)
    parser.add_argument("--l2-flush-mib", type=int, default=256)
    args = parser.parse_args()

    if args.q != 8192 or args.kv < args.q or args.kv % BLOCK:
        raise ValueError("the paired gate requires Q=8192 and block-aligned KV>=Q")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    q = torch.randn(
        (args.q, args.q_heads, 128),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k = torch.randn(
        (args.kv, args.kv_heads, 128),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    blocks = args.kv // BLOCK
    partitions = (blocks + args.partition_blocks - 1) // args.partition_blocks

    new_scores = torch.empty(
        (args.q_heads, partitions, args.q, TOPK),
        dtype=torch.float32,
        device="cuda",
    )
    new_ids = torch.empty_like(new_scores, dtype=torch.int32)
    new_route = torch.empty(
        (args.q_heads, args.q, TOPK), dtype=torch.int32, device="cuda"
    )
    l2_flush = torch.empty(
        args.l2_flush_mib * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda",
    )

    def flush_l2() -> None:
        # The write is ordered before the start event on the same stream, but
        # remains outside the measured interval.  The buffer is deliberately
        # larger than B300 L2.
        # Read-modify-write rather than a write-only memset: write-only CUDA
        # paths may bypass/stream through L2 and fail to evict the prior K
        # working set, which shows up as an order-dependent bimodal result.
        l2_flush.add_(1)

    def new_score() -> None:
        bf16_indexer_local_top16(
            q,
            k,
            partition_blocks=args.partition_blocks,
            out_scores=new_scores,
            out_ids=new_ids,
        )

    q_lens = torch.tensor([args.q], dtype=torch.int32)
    kv_lens = torch.tensor([args.kv], dtype=torch.int32)
    q_offset = torch.tensor([args.kv - args.q], dtype=torch.int32)
    plan = build_prefill_plan(
        q_lens,
        kv_lens,
        qo_offset=q_offset,
        num_idx_heads=args.q_heads,
        num_idx_kv_heads=args.kv_heads,
        page_size=BLOCK,
    )
    paged_k = (
        k.view(blocks, BLOCK, args.kv_heads, 128)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    page_ids = torch.arange(blocks, dtype=torch.int32, device="cuda")
    score_options = {
        "local_top16": True,
        "local_top16_int32": True,
        "local_top16_partition2048": args.partition_blocks == 2048,
        "local_top16_partition4096": args.partition_blocks == 4096,
        "local_top16_partition8192": args.partition_blocks == 8192,
        "early_s_release": args.kv >= 32768,
        "eight_way_reduction": args.kv >= 32768,
        "rolling_reduction": args.kv >= 32768,
        "rolling_prefetch3": args.kv >= 32768,
        "tmem_load_reduce": args.kv >= 32768,
        "tmem_load_reduce_x64": args.kv >= 32768,
        "single_correction_warp": args.kv >= 131072,
        "onlyscore_no_o_barrier": args.kv >= 262144,
        "early_c_commit": args.kv >= 262144,
    }
    opt3_storage = score_prefill(
        q,
        paged_k,
        paged_k,
        plan=plan,
        kv_indices=page_ids,
        sm_scale=128**-0.5,
        **score_options,
    )
    def opt3_score() -> None:
        score_prefill(
            q,
            paged_k,
            paged_k,
            plan=plan,
            kv_indices=page_ids,
            sm_scale=128**-0.5,
            out=opt3_storage,
            **score_options,
        )

    def new_chain() -> None:
        bf16_indexer_local_top16(
            q,
            k,
            partition_blocks=args.partition_blocks,
            out_scores=new_scores,
            out_ids=new_ids,
            out_route=new_route,
        )

    def opt3_chain() -> None:
        opt3_score()

    new_chain()
    opt3_chain()
    torch.cuda.synchronize()
    opt3_route = (
        opt3_storage.view(torch.int32)[:, 16:32, :]
        .permute(0, 2, 1)
        .contiguous()
    )
    opt3_route = torch.where(
        opt3_route == torch.iinfo(torch.int32).max,
        torch.full_like(opt3_route, -1),
        opt3_route,
    )
    if not torch.equal(new_route, opt3_route):
        mismatch = int((new_route != opt3_route).sum().item())
        raise AssertionError(f"dynamic Top-16 route mismatch in {mismatch} entries")

    opt3_score_samples, new_score_samples = _paired(
        opt3_score,
        new_score,
        flush_l2=flush_l2,
        warmups=args.warmups,
        pairs=args.pairs,
    )
    opt3_chain_samples, new_chain_samples = _paired(
        opt3_chain,
        new_chain,
        flush_l2=flush_l2,
        warmups=args.warmups,
        pairs=args.pairs,
    )
    opt3_median = statistics.median(opt3_chain_samples)
    new_median = statistics.median(new_chain_samples)
    print(
        json.dumps(
            {
                "shape": vars(args),
                "route_exact_match": True,
                "opt3": {
                    "score": _summary(opt3_score_samples),
                    "chain": _summary(opt3_chain_samples),
                },
                "cutedsl": {
                    "score": _summary(new_score_samples),
                    "chain": _summary(new_chain_samples),
                },
                "chain_ratio": new_median / opt3_median,
                "chain_regression_percent": (new_median / opt3_median - 1.0)
                * 100.0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
