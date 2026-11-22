#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decode-like step timing: one packed step of one-token requests, eager and CUDA Graph.

Every request has one query token and its own KV pages; all requests of a step have the same KV length.
Inputs start at the Index/attention operands (BF16 attention Q/K/V). The plan is prepared outside
timing. Eager timing encloses plan.run() in one CUDA event pair on the current stream and therefore
includes host launch latency; the CUDA Graph column replays the captured step and is the figure for
callers that run decode inside a graph.

By default every KV length gets its own plan and its own captured graph. With --capacity (optimized
backend) one plan is prepared, and one graph captured, at that KV capacity; every length is then
published with KvLengths.set() before it is timed, as a decode loop does from token to token.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))


def timed(torch, run, samples):
    values = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('default', 'optimized'), default='optimized')
    parser.add_argument('--batch', type=int, default=32, help='one-token requests in the step')
    parser.add_argument('--lengths', default='8192,16384,32768,65536,131072,262144,524288,1048576',
                        help='KV tokens per request')
    parser.add_argument('--index-dtype', choices=('bf16', 'fp8'), default='bf16')
    parser.add_argument('--seeds', default='17,29,43')
    parser.add_argument('--warmups', type=int, default=20)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--no-graph', action='store_true', help='skip the CUDA Graph capture and replay')
    parser.add_argument('--capacity', type=int, default=0,
                        help='KV tokens per request of one plan and one captured graph that serve every length '
                             '(optimized backend); 0 prepares a plan per length')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lengths = [int(x) for x in args.lengths.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    if not lengths or any(not 0 < n <= 1048576 for n in lengths) or args.batch < 1:
        parser.error('lengths are KV tokens per request through 1M; batch is positive')
    if args.warmups < 1 or args.samples < 1:
        parser.error('warmups and samples must be positive')
    if args.capacity and (args.backend != 'optimized' or not max(lengths) <= args.capacity <= 1048576):
        parser.error('--capacity needs the optimized backend and must cover every length, through 1M')
    if args.output.exists():
        parser.error('output exists; use a new evidence path')
    import torch
    from mbsa import prepare_cutedsl_pipeline
    props = torch.cuda.get_device_properties(0)
    result = dict(contract=dict(backend=args.backend, batch=args.batch, query_tokens_per_request=1,
        index_dtype=args.index_dtype, attention_dtype='bf16', seeds=seeds, warmups=args.warmups,
        samples=args.samples, gpu=props.name, sm_count=props.multi_processor_count, uuid=str(props.uuid),
        torch=str(torch.__version__), kv_capacity=args.capacity or None,
        timing='p50 of direct same-stream outer CUDA events per seed, then the median over seeds; '
               'eager plan.run() and replay of the captured CUDA Graph',
        pages='every request owns its KV and Index pages'), rows=[])
    print(json.dumps(result['contract']), flush=True)
    pages_per_request = (max(lengths + [args.capacity]) + 127) // 128
    generator = torch.Generator(device='cuda').manual_seed(1234)
    pool = torch.empty((args.batch * pages_per_request, 128, 4, 2, 128), device='cuda', dtype=torch.bfloat16)
    index_k = torch.empty((args.batch * pages_per_request, 1, 128, 128), device='cuda', dtype=torch.bfloat16)
    for begin in range(0, pool.shape[0], 4096):
        for tensor in (pool, index_k):
            part = tensor[begin:begin + 4096]
            part.copy_(torch.randn(part.shape, device='cuda', dtype=torch.bfloat16, generator=generator))
    k, v = (pool[:, :, :, half].permute(0, 2, 1, 3) for half in (0, 1))
    if args.index_dtype == 'fp8':
        index_k = index_k.to(torch.float8_e4m3fn)
    def step_kwargs(length, packed):
        count = (length + 127) // 128
        rows = [request * pages_per_request + torch.arange(count, device='cuda', dtype=torch.int32)
                for request in range(args.batch)]
        kwargs = dict(index_page_indices=torch.cat(rows).contiguous(), page_table=torch.stack(rows).contiguous(),
                      backend=args.backend)
        if packed:
            kwargs.update(
                cu_seqlens_q=torch.arange(args.batch + 1, device='cuda', dtype=torch.int32),
                cu_seqlens_k=torch.arange(0, args.batch * length + 1, length, device='cuda', dtype=torch.int32),
                request_lens=[(1, length)] * args.batch)
        else:
            kwargs['logical_k_tokens'] = length
        return kwargs

    def operands(seed):
        seeded = torch.Generator(device='cuda').manual_seed(seed)
        q = torch.randn((args.batch, 32, 128), device='cuda', dtype=torch.bfloat16, generator=seeded)
        index_q = torch.randn((args.batch, 4, 128), device='cuda', dtype=torch.bfloat16, generator=seeded)
        if args.index_dtype == 'fp8':
            index_q = index_q.to(torch.float8_e4m3fn)
        return index_q, index_k, q, k, v

    def measure(run):
        for _ in range(args.warmups):
            run()
        torch.cuda.synchronize()
        return statistics.median(timed(torch, run, args.samples))

    eager = {length: [] for length in lengths}
    graph = {length: [] for length in lengths}

    def report(length):
        row = dict(kv_tokens=length, eager_ms=statistics.median(eager[length]), eager_ms_per_seed=eager[length])
        if graph[length]:
            row.update(cuda_graph_ms=statistics.median(graph[length]), cuda_graph_ms_per_seed=graph[length])
        result['rows'].append(row)
        print(json.dumps({key: value for key, value in row.items() if not key.endswith('per_seed')}), flush=True)
        args.output.write_text(json.dumps(result, indent=1))

    if args.capacity:
        from mbsa.serving import KvLengths
        # The same order as below (lengths outer, seeds inner): a length is timed after the shorter ones.
        steps = []
        for seed in seeds:
            current = KvLengths('cuda')
            plan = prepare_cutedsl_pipeline(*operands(seed), **step_kwargs(args.capacity, True), kv_lengths=current)
            if not args.no_graph:
                current.set((lengths[0],) * args.batch)
                plan.capture_mega_graph()
            steps.append((current, plan))
        for length in lengths:
            for current, plan in steps:
                current.set((length,) * args.batch)
                eager[length].append(measure(plan.run))
                if not args.no_graph:
                    graph[length].append(measure(plan.run_mega))
            report(length)
    else:
        for length in lengths:
            kwargs = step_kwargs(length, args.batch > 1)
            for seed in seeds:
                plan = prepare_cutedsl_pipeline(*operands(seed), **kwargs)
                eager[length].append(measure(plan.run))
                if not args.no_graph:
                    plan.capture_mega_graph()
                    graph[length].append(measure(plan.run_mega))
                del plan
            report(length)


if __name__ == '__main__':
    main()
