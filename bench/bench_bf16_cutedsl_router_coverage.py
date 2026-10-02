#!/usr/bin/env python3
"""Correctness gate for CuTeDSL Router page tables and logical KV tails."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import _paths  # noqa: F401,E402

import torch  # noqa: E402

from kernels.bf16_indexer_cutedsl import bf16_indexer_exact_route  # noqa: E402
from kernels.msa_indexer_score import build_prefill_plan, score_prefill  # noqa: E402


BLOCK = 128
HEADS = 4
DIM = 128


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv", type=int, default=33007)
    parser.add_argument("--q", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--shuffle-index-pages", action="store_true")
    args = parser.parse_args()
    if args.q <= 0 or args.q % BLOCK or args.kv < args.q:
        raise ValueError("require block-aligned positive Q <= KV")

    torch.manual_seed(args.seed)
    blocks = (args.kv + BLOCK - 1) // BLOCK
    q = torch.randn(
        (args.q, HEADS, DIM), dtype=torch.bfloat16, device="cuda"
    )
    logical_k = torch.randn(
        (blocks, 1, BLOCK, DIM), dtype=torch.bfloat16, device="cuda"
    )
    if args.shuffle_index_pages:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(args.seed + 10_003)
        pages = torch.randperm(
            blocks, dtype=torch.int64, device="cuda", generator=generator
        )
        physical_k = torch.empty_like(logical_k)
        physical_k.index_copy_(0, pages, logical_k)
        page_indices = pages.to(torch.int32)
    else:
        physical_k = logical_k
        page_indices = torch.arange(blocks, dtype=torch.int32, device="cuda")

    plan = build_prefill_plan(
        torch.tensor([args.q], dtype=torch.int32),
        torch.tensor([args.kv], dtype=torch.int32),
        qo_offset=torch.tensor([args.kv - args.q], dtype=torch.int32),
        num_idx_heads=HEADS,
        num_idx_kv_heads=1,
        page_size=BLOCK,
    )
    oracle_storage = score_prefill(
        q,
        physical_k,
        physical_k,
        plan=plan,
        kv_indices=page_indices,
        sm_scale=DIM**-0.5,
        local_top16=True,
        local_top16_int32=True,
        early_s_release=True,
        eight_way_reduction=True,
        rolling_reduction=True,
        rolling_prefetch3=True,
        tmem_load_reduce=True,
        tmem_load_reduce_x64=True,
    )
    expected = (
        oracle_storage.view(torch.int32)[:, 16:32, :]
        .permute(0, 2, 1)
        .contiguous()
    )
    actual = bf16_indexer_exact_route(
        q,
        physical_k.view(-1, 1, DIM),
        partition_blocks=1024,
        page_indices=page_indices,
        logical_k_tokens=args.kv,
        use_page_table=args.shuffle_index_pages,
    )
    torch.cuda.synchronize()
    mismatch = int((actual != expected).sum().item())
    if mismatch:
        first = (actual != expected).nonzero()[0].cpu().tolist()
        raise AssertionError(f"route mismatch count={mismatch}, first={first}")
    print(
        json.dumps(
            {
                "kv": args.kv,
                "q": args.q,
                "logical_blocks": blocks,
                "partial_tail_tokens": args.kv % BLOCK,
                "index_page_table": (
                    "permuted" if args.shuffle_index_pages else "identity"
                ),
                "dynamic_route_exact_match": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
