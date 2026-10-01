#!/usr/bin/env python3
"""Correctness and latency gate for the independent Opt4 GPU revision."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compass_bsa import (
    build_kv_grouped_schedule,
    gpu_dual_path_attention,
    gpu_dual_path_attention_v2,
    gpu_dynamic_grouped_partials_v2,
    gpu_fixed_attention_v2,
    gpu_hierarchical_exact_topk,
    gpu_merge_states_v2,
    gpu_pipeline_v2,
)


BLOCK = 128
PREFIX = 8
LOCAL = 8
TOPK = 16
INDEX_HEADS = 4
QUERY_HEADS = 32
DIM = 128


def make_inputs(q_tokens: int, kv_tokens: int, seed: int):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    def randn(shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    return (
        randn((q_tokens, INDEX_HEADS, DIM)),
        randn((kv_tokens, 1, DIM)),
        randn((q_tokens, QUERY_HEADS, DIM)),
        randn((kv_tokens, INDEX_HEADS, DIM)),
        randn((kv_tokens, INDEX_HEADS, DIM)),
    )


def dense_route_reference(index_q: torch.Tensor, index_k: torch.Tensor):
    q_tokens = index_q.shape[0]
    kv_tokens = index_k.shape[0]
    q_offset = max(kv_tokens - q_tokens, 0)
    blocks = math.ceil(kv_tokens / BLOCK)
    padded = blocks * BLOCK
    key = index_k.float().permute(1, 0, 2).expand(INDEX_HEADS, -1, -1)
    if padded != kv_tokens:
        key = torch.nn.functional.pad(key, (0, 0, 0, padded - kv_tokens))
    key = key.view(INDEX_HEADS, blocks, BLOCK, DIM)
    query = index_q.float().permute(1, 0, 2)
    score = torch.einsum("hqd,hbtd->hqbt", query, key)
    if padded != kv_tokens:
        valid_tokens = torch.arange(padded, device="cuda") < kv_tokens
        score = score.masked_fill(~valid_tokens.view(1, 1, blocks, BLOCK), -torch.inf)
    score = score.amax(-1)
    q_position = q_offset + torch.arange(q_tokens, device="cuda")
    q_block = q_position // BLOCK
    block = torch.arange(blocks, device="cuda").view(1, 1, blocks)
    local_begin = torch.clamp(q_block - LOCAL + 1, min=PREFIX)
    allowed = (
        (block <= q_block.view(1, -1, 1))
        & (block >= PREFIX)
        & ~(
            (block >= local_begin.view(1, -1, 1))
            & (block <= q_block.view(1, -1, 1))
        )
    )
    score = score.masked_fill(~allowed, -torch.inf)
    values, indices = torch.topk(score, TOPK, dim=-1, sorted=True)
    indices = torch.where(torch.isfinite(values), indices, -1).to(torch.int32)
    return indices, values


def union_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    route: torch.Tensor,
    rows: int,
):
    q_tokens = q.shape[0]
    kv_tokens = k.shape[0]
    offset = max(kv_tokens - q_tokens, 0)
    group = QUERY_HEADS // INDEX_HEADS
    output = torch.empty((rows, QUERY_HEADS, DIM), device="cuda", dtype=torch.float32)
    lse = torch.empty((rows, QUERY_HEADS), device="cuda", dtype=torch.float32)
    for qi in range(rows):
        position = offset + qi
        q_block = position // BLOCK
        fixed = set(range(min(PREFIX, q_block + 1)))
        fixed.update(range(max(PREFIX, q_block - LOCAL + 1), q_block + 1))
        for qh in range(QUERY_HEADS):
            kvh = qh // group
            blocks = set(fixed)
            blocks.update(int(x) for x in route[kvh, qi].tolist() if int(x) >= 0)
            token_ids = []
            for block in sorted(blocks):
                begin = block * BLOCK
                end = min(begin + BLOCK, kv_tokens, position + 1)
                token_ids.extend(range(begin, end))
            ids = torch.tensor(token_ids, device="cuda", dtype=torch.long)
            key = k[:, kvh].index_select(0, ids).float()
            value = v[:, kvh].index_select(0, ids).float()
            scores = key @ q[qi, qh].float() * (DIM ** -0.5)
            probabilities = torch.softmax(scores, dim=0)
            output[qi, qh] = probabilities @ value
            lse[qi, qh] = torch.logsumexp(scores, dim=0)
    return output, lse


def median_ms(function, warmups: int, samples: int) -> float:
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    measurements = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        measurements.append(start.elapsed_time(end))
    return statistics.median(measurements)


def correctness(seed: int, revision: int):
    q_tokens = 128 if revision == 2 else 16
    index_q, index_k, q, k, v = make_inputs(q_tokens, 4096, seed)
    route, route_scores = gpu_hierarchical_exact_topk(index_q, index_k)
    reference_route, reference_scores = dense_route_reference(index_q, index_k)
    route_ok = torch.equal(route, reference_route)
    score_error = (route_scores - reference_scores).abs()
    score_error = score_error[torch.isfinite(score_error)].max().item()
    attention = gpu_dual_path_attention_v2 if revision == 2 else gpu_dual_path_attention
    output, lse = attention(q, k, v, route)
    reference_output, reference_lse = union_attention_reference(q, k, v, route, rows=4)
    return {
        "route_exact": route_ok,
        "route_score_max_abs": score_error,
        "output_max_abs": (output[:4].float() - reference_output).abs().max().item(),
        "lse_max_abs": (lse[:4] - reference_lse).abs().max().item(),
    }


def benchmark_case(kv_tokens: int, warmups: int, samples: int, seed: int, revision: int):
    q_tokens = min(8192, kv_tokens)
    index_q, index_k, q, k, v = make_inputs(q_tokens, kv_tokens, seed)
    route, _ = gpu_hierarchical_exact_topk(index_q, index_k)
    attention = gpu_dual_path_attention_v2 if revision == 2 else gpu_dual_path_attention
    order = build_kv_grouped_schedule(route, kv_tokens) if revision == 2 else None
    if revision == 2:
        attention(q, k, v, route, schedule=order)
    else:
        attention(q, k, v, route)
    torch.cuda.synchronize()

    router_ms = median_ms(
        lambda: gpu_hierarchical_exact_topk(index_q, index_k), warmups, samples
    )
    order_ms = (
        median_ms(lambda: build_kv_grouped_schedule(route, kv_tokens), warmups, samples)
        if revision == 2 else 0.0
    )
    component = {}
    if revision == 2:
        fixed_state = gpu_fixed_attention_v2(q, k, v)
        dynamic_state = gpu_dynamic_grouped_partials_v2(q, k, v, route, order)
        component["fixed_ms"] = median_ms(
            lambda: gpu_fixed_attention_v2(q, k, v), warmups, samples
        )
        component["dynamic_ms"] = median_ms(
            lambda: gpu_dynamic_grouped_partials_v2(q, k, v, route, order),
            warmups,
            samples,
        )
        component["merge_ms"] = median_ms(
            lambda: gpu_merge_states_v2(
                fixed_state[0], fixed_state[1], dynamic_state[0], dynamic_state[1],
                k.shape[1],
            ),
            warmups,
            samples,
        )
    if revision == 2:
        attention_call = lambda: attention(q, k, v, route, schedule=order)
    else:
        attention_call = lambda: attention(q, k, v, route)
    attention_ms = median_ms(attention_call, warmups, samples)

    def end_to_end():
        if revision == 2:
            return gpu_pipeline_v2(
                index_q, index_k, q, k, v, fixed_stream=fixed_stream
            )
        current_route, _ = gpu_hierarchical_exact_topk(index_q, index_k)
        return attention(q, k, v, current_route)

    fixed_stream = torch.cuda.Stream() if revision == 2 else None
    total_ms = median_ms(end_to_end, warmups, samples)
    return {
        "kv": kv_tokens,
        "q": q_tokens,
        "router_ms": router_ms,
        "order_ms": order_ms,
        "attention_ms": attention_ms,
        "e2e_ms": total_ms,
        **component,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv", default="8192,32768,65536")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--revision", type=int, choices=(1, 2), default=1)
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    payload = {
        "kind": f"opt4_independent_gpu_v{args.revision}",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "correctness": None if args.skip_correctness else correctness(args.seed, args.revision),
        "results": [],
    }
    print(json.dumps({key: value for key, value in payload.items() if key != "results"}), flush=True)
    for kv_tokens in (int(value) for value in args.kv.split(",")):
        result = benchmark_case(
            kv_tokens, args.warmups, args.samples, args.seed, args.revision
        )
        payload["results"].append(result)
        print(json.dumps(result), flush=True)
    print("RESULT_JSON=" + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
