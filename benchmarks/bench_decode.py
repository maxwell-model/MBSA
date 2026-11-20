#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decode-like step timing: one packed step of one-token requests, eager and CUDA Graph.

Every request has one query token and its own KV pages; all requests of a step have the same KV length.
Inputs start at the Index/attention operands (BF16 attention Q/K/V). The plan is prepared outside
timing. Eager timing encloses plan.run() in one CUDA event pair on the current stream and therefore
includes host launch latency; the CUDA Graph column replays the captured step and is the figure for
callers that run decode inside a graph.
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
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lengths = [int(x) for x in args.lengths.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    if not lengths or any(not 0 < n <= 1048576 for n in lengths) or args.batch < 1:
        parser.error('lengths are KV tokens per request through 1M; batch is positive')
    if args.warmups < 1 or args.samples < 1:
        parser.error('warmups and samples must be positive')
    if args.output.exists():
        parser.error('output exists; use a new evidence path')
    import torch
    from mbsa import prepare_cutedsl_pipeline
    props = torch.cuda.get_device_properties(0)
    result = dict(contract=dict(backend=args.backend, batch=args.batch, query_tokens_per_request=1,
        index_dtype=args.index_dtype, attention_dtype='bf16', seeds=seeds, warmups=args.warmups,
        samples=args.samples, gpu=props.name, sm_count=props.multi_processor_count, uuid=str(props.uuid),
        torch=str(torch.__version__),
        timing='p50 of direct same-stream outer CUDA events per seed, then the median over seeds; '
               'eager plan.run() and replay of the captured CUDA Graph',
        pages='every request owns its KV and Index pages'), rows=[])
    print(json.dumps(result['contract']), flush=True)
    pages_per_request = (max(lengths) + 127) // 128
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
    for length in lengths:
        count = (length + 127) // 128
        rows = [request * pages_per_request + torch.arange(count, device='cuda', dtype=torch.int32)
                for request in range(args.batch)]
        table, index_pages = torch.stack(rows).contiguous(), torch.cat(rows).contiguous()
        kwargs = dict(index_page_indices=index_pages, page_table=table, backend=args.backend)
        if args.batch == 1:
            kwargs['logical_k_tokens'] = length
        else:
            kwargs.update(
                cu_seqlens_q=torch.arange(args.batch + 1, device='cuda', dtype=torch.int32),
                cu_seqlens_k=torch.arange(0, args.batch * length + 1, length, device='cuda', dtype=torch.int32),
                request_lens=[(1, length)] * args.batch)
        eager, graph = [], []
        for seed in seeds:
            seeded = torch.Generator(device='cuda').manual_seed(seed)
            q = torch.randn((args.batch, 32, 128), device='cuda', dtype=torch.bfloat16, generator=seeded)
            index_q = torch.randn((args.batch, 4, 128), device='cuda', dtype=torch.bfloat16, generator=seeded)
            if args.index_dtype == 'fp8':
                index_q = index_q.to(torch.float8_e4m3fn)
            plan = prepare_cutedsl_pipeline(index_q, index_k, q, k, v, **kwargs)
            for _ in range(args.warmups):
                plan.run()
            torch.cuda.synchronize()
            eager.append(statistics.median(timed(torch, plan.run, args.samples)))
            if not args.no_graph:
                plan.capture_mega_graph()
                for _ in range(args.warmups):
                    plan.run_mega()
                torch.cuda.synchronize()
                graph.append(statistics.median(timed(torch, plan.run_mega, args.samples)))
            del plan
        row = dict(kv_tokens=length, eager_ms=statistics.median(eager), eager_ms_per_seed=eager)
        if graph:
            row.update(cuda_graph_ms=statistics.median(graph), cuda_graph_ms_per_seed=graph)
        result['rows'].append(row)
        print(json.dumps({key: value for key, value in row.items() if not key.endswith('per_seed')}), flush=True)
        args.output.write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
