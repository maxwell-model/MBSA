#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Direct eager single-chunk or ordered full-prefill timing (not a sum of medians).

Inputs start at the Index/attention operands. Projection, RMSNorm, RoPE and
attention-cache writes are excluded. FP8 preparation includes Index Q casting
and incremental Index K insertion; attention remains BF16.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))


def build_prompt(torch, prepare, *, length, chunk, backend, index_dtype, scope, seed,
                 lean192="auto"):
    """Prepare every legal chunk outside timing; each plan sees only its own KV extent."""
    gen = torch.Generator(device='cuda').manual_seed(seed)

    def rand(shape):
        return torch.randn(shape, device='cuda', dtype=torch.bfloat16, generator=gen)

    q, iq = rand((length, 32, 128)), rand((length, 4, 128))
    k, v, ik = rand((length, 4, 128)), rand((length, 4, 128)), rand((length, 1, 128))
    if index_dtype == 'fp8':
        prepared_iq, prepared_ik = iq.to(torch.float8_e4m3fn), ik.to(torch.float8_e4m3fn)
    else:
        prepared_iq, prepared_ik = iq, ik
    ranges = [(end - chunk, end) for end in range(chunk, length + 1, chunk)]
    if scope == 'single':
        ranges = ranges[-1:]
    plans = [prepare(prepared_iq[begin:end], prepared_ik[:end], q[begin:end], k[:end], v[:end],
                     backend=backend, partial_dtype=torch.bfloat16, lean192=lean192)
             for begin, end in ranges]
    # Construction also compiles and warms all selected helpers. Retain operands
    # and every plan for the entire sample; do not alias in-flight workspace.
    retained = (q, iq, k, v, ik, prepared_iq, prepared_ik, plans)

    def run(include_preparation=False):
        for (begin, end), plan in zip(ranges, plans):
            if include_preparation and index_dtype == 'fp8':
                prepared_iq[begin:end].copy_(iq[begin:end])
                # A single chunk starts with an already populated prefix cache.
                prepared_ik[begin:end].copy_(ik[begin:end])
            plan.run()
        return retained

    return run, plans


def measure(torch, run, *, warmups, samples, include_preparation=False):
    for _ in range(warmups):
        run(include_preparation)
    torch.cuda.current_stream().synchronize()
    values, walls = [], []
    import time
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # No queued flush or profiler work before start: CPU enqueue gaps are
        # part of this eager enclosing event span.
        wall_start = time.perf_counter()
        start.record()
        run(include_preparation)
        end.record()
        end.synchronize()
        walls.append((time.perf_counter() - wall_start) * 1000)
        values.append(start.elapsed_time(end))
    return dict(samples_ms=values, p50_ms=statistics.median(values),
                host_wall_ms=walls, p50_host_wall_ms=statistics.median(walls))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('default', 'optimized'), default='default')
    parser.add_argument('--lean192', choices=('auto', 'on', 'off'), default='auto',
                        help='optimized BF16 Index policy; forced on rejects unsupported geometry')
    parser.add_argument('--lengths', default='8192,16384,32768,65536,131072,262144,524288,1048576')
    parser.add_argument('--chunk', type=int, default=8192)
    parser.add_argument('--scope', choices=('single', 'full'), default='full')
    parser.add_argument('--index-dtype', choices=('bf16', 'fp8'), default='bf16')
    parser.add_argument('--include-preparation', action='store_true')
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--samples', type=int, default=15)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.lean192 != 'auto' and args.backend != 'optimized':
        parser.error('--lean192 on/off requires --backend optimized')
    if args.lean192 == 'on' and args.index_dtype != 'bf16':
        parser.error('--lean192 on requires --index-dtype bf16')
    lengths = [int(x) for x in args.lengths.split(',')]
    if (not lengths or not 0 < args.chunk <= 8192 or args.chunk % 128
            or any(n < args.chunk or n > 1048576 or n % args.chunk for n in lengths)):
        parser.error('lengths must be chunk multiples through 1M; chunk is a multiple of 128 through 8192')
    if args.warmups < 1 or args.samples < 1:
        parser.error('warmups and samples must be positive')
    if args.output.exists():
        parser.error('output exists; use a new evidence path')
    import torch
    from mbsa import prepare_cutedsl_pipeline
    props = torch.cuda.get_device_properties(0)
    result = dict(contract=dict(backend=args.backend, lean192_requested=args.lean192,
        lean192_resolved_location='rows[].lean192_per_chunk', scope=args.scope, chunk=args.chunk,
        index_dtype=args.index_dtype, attention_dtype='bf16', seed=args.seed,
        include_preparation=args.include_preparation, gpu=props.name,
        sm_count=props.multi_processor_count, uuid=str(props.uuid), torch=str(torch.__version__),
        timing='direct same-stream outer CUDA events, eager, warm cache',
        preparation='Index Q casting and new Index K slice insertion only; producer and attention cache writes excluded'),
        rows=[])
    print(json.dumps(result['contract']), flush=True)
    for length in lengths:
        run, plans = build_prompt(torch, prepare_cutedsl_pipeline, length=length, chunk=args.chunk,
            backend=args.backend, index_dtype=args.index_dtype, scope=args.scope, seed=args.seed,
            lean192=args.lean192)
        resolved = [dict(q_offset=length-args.chunk if args.scope == 'single' else i*args.chunk,
                         visible_kv=length if args.scope == 'single' else (i+1)*args.chunk,
                         requested=plan.lean192, enabled=plan.lean192_enabled)
                    for i, plan in enumerate(plans)]
        print(json.dumps(dict(length=length, lean192_per_chunk=resolved)), flush=True)
        row = dict(length=length, executed_chunks=len(plans), lean192_per_chunk=resolved,
                   **measure(torch, run,
            warmups=args.warmups, samples=args.samples, include_preparation=args.include_preparation))
        result['rows'].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(row), flush=True)
        del run, plans


if __name__ == '__main__':
    main()
