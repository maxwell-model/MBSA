#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decode-like step against a reference decode chain, paired in one process on one GPU.

The reference is the decode chain of another checkout (--reference-root): its
bench/bench_top32_chain.py provides DecodeStep, a batch of one-token requests with its own tensors,
seq_lens and run_chain(). MBSA binds views of those tensors, so both sides read the same Q, K/V,
Index K and pages, and every row reports how far the two outputs are apart.

Timing alternates blocks of back-to-back runs (MBSA x block, reference x block, the next round in
the opposite order), every run between one CUDA event pair on its own stream. A row holds the median
and the minimum of each side and the speedup, reference median / MBSA median: above 1 is faster.

Modes
  capacity_graph, capacity_eager  both sides are prepared (and captured) once at --capacity. Before a
      case is timed the reference's seq_lens is written in place and MBSA's lengths are published
      with KvLengths.set(), as a decode loop does from token to token.
  fixed_graph, fixed_eager        MBSA is prepared for the case's own lengths; the reference keeps its
      capacity chain with seq_lens set to the case.

Cases are batches of equal lengths (--lengths, KV tokens per request) and, with --ragged, four
batches of unequal lengths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))

LADDER = (8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576)
MODES = ('capacity_graph', 'fixed_graph', 'capacity_eager', 'fixed_eager')


def ragged_cases(batch, capacity):
    """Four batches of unequal KV lengths, none above the capacity."""
    rng = random.Random(4242)
    cases = (
        ('ladder', [LADDER[i % len(LADDER)] for i in range(batch)]),            # every length, in turn
        ('one_long', [capacity] + [8192] * (batch - 1)),                        # one long request among short ones
        ('random', [rng.randint(1024, 262144) for _ in range(batch)]),          # not page aligned
        ('halves', [16384 if i % 2 else 131072 for i in range(batch)]),         # half short, half long
    )
    return [(name, [min(n, capacity) for n in lens]) for name, lens in cases]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reference-root', type=Path, required=True,
                        help='checkout whose bench/bench_top32_chain.py provides DecodeStep')
    parser.add_argument('--batch', type=int, default=32, help='one-token requests in the step')
    parser.add_argument('--capacity', type=int, default=1048576, help='KV tokens per request the step is built for')
    parser.add_argument('--lengths', default=','.join(str(n) for n in LADDER),
                        help='KV tokens per request of the equal-length cases; those above the capacity are skipped')
    parser.add_argument('--ragged', action='store_true', help='add the four batches of unequal lengths')
    parser.add_argument('--modes', default='capacity_graph', help='comma-separated: ' + ', '.join(MODES))
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--warmups', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--block', type=int, default=10, help='back-to-back runs of one side per round')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    modes = args.modes.split(',')
    lengths = [int(x) for x in args.lengths.split(',') if x]
    if not set(modes) <= set(MODES):
        parser.error('modes are ' + ', '.join(MODES))
    if args.batch < 1 or not 0 < args.capacity <= 1048576 or any(n < 1 for n in lengths):
        parser.error('batch and lengths are positive; the capacity is KV tokens per request through 1M')
    if min(args.warmups, args.rounds, args.block) < 1:
        parser.error('warmups, rounds and block must be positive')
    if not (args.reference_root / 'bench/bench_top32_chain.py').is_file():
        parser.error('--reference-root has no bench/bench_top32_chain.py')
    if args.output.exists():
        parser.error('output exists; use a new evidence path')
    sys.path.insert(0, str(args.reference_root.resolve()))
    import _paths  # noqa: F401  (the reference checkout's import bootstrap)
    import torch
    from bench.bench_top32_chain import DecodeStep
    from mbsa import prepare_cutedsl_pipeline
    from mbsa.serving import KvLengths

    cases = [(f'{n}', [n] * args.batch) for n in lengths if n <= args.capacity]
    if args.ragged:
        cases += ragged_cases(args.batch, args.capacity)
    props = torch.cuda.get_device_properties(0)
    result = dict(contract=dict(batch=args.batch, query_tokens_per_request=1, kv_capacity=args.capacity,
        seed=args.seed, warmups=args.warmups, rounds=args.rounds, block=args.block,
        reference=str(args.reference_root.resolve()), gpu=props.name, sm_count=props.multi_processor_count,
        uuid=str(props.uuid), torch=str(torch.__version__),
        timing='alternating blocks of back-to-back runs, one CUDA event pair per run on the side\'s own stream; '
               'median and minimum per side; speedup = reference median / MBSA median',
        tensors='the reference step owns Q, K/V, Index K and the page table; MBSA binds views of them'), rows=[])
    print(json.dumps(result['contract']), flush=True)

    reference = DecodeStep(args.capacity, args.batch, args.seed)
    reference.stream.synchronize()
    k, v = (reference.main_kv[:, half].permute(0, 2, 1, 3) for half in (0, 1))
    operands = (reference.index_q, reference.index_k.view(-1, 1, 128, 128), reference.main_q, k, v)

    def prepare(lens, kv_lengths=None):
        counts = [(n + 127) // 128 for n in lens]
        ends = [0]
        for n in lens:
            ends.append(ends[-1] + n)
        return prepare_cutedsl_pipeline(
            *operands, backend='optimized', kv_lengths=kv_lengths, request_lens=[(1, n) for n in lens],
            cu_seqlens_q=torch.arange(len(lens) + 1, device='cuda', dtype=torch.int32),
            cu_seqlens_k=torch.tensor(ends, device='cuda', dtype=torch.int32),
            page_table=reference.block_table[:, :max(counts)].contiguous(),
            index_page_indices=torch.cat([reference.block_table[r, :c] for r, c in enumerate(counts)]).contiguous())

    def reference_lengths(lens):
        reference.seq_lens.copy_(torch.tensor(lens, dtype=torch.int32, device=reference.seq_lens.device))

    def reference_graph():
        for _ in range(5):
            reference.run_chain()
        reference.stream.synchronize()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=reference.stream, capture_error_mode='global'):
            reference.run_chain()
        reference.stream.synchronize()
        return graph

    def timed(run, stream=None):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        if stream is None:
            start.record()
            run()
            end.record()
        else:
            with torch.cuda.stream(stream):
                start.record(stream)
                run()
                end.record(stream)
        end.synchronize()
        return start.elapsed_time(end)

    def measure(mode, name, ours, theirs):
        """Time one case, then run both sides once more and compare their outputs."""
        for _ in range(args.warmups):
            ours()
            with torch.cuda.stream(reference.stream):
                theirs()
        torch.cuda.synchronize()
        a, b = [], []
        for round_ in range(args.rounds):
            sides = [(a, lambda: timed(ours)), (b, lambda: timed(theirs, reference.stream))]
            for values, sample in (sides if round_ % 2 == 0 else sides[::-1]):
                values.extend(sample() for _ in range(args.block))
        with torch.cuda.stream(reference.stream):
            theirs()
        out, lse, _ = ours()
        torch.cuda.synchronize()
        row = dict(mode=mode, case=name, mbsa_ms=statistics.median(a), reference_ms=statistics.median(b),
                   mbsa_min_ms=min(a), reference_min_ms=min(b), speedup=statistics.median(b) / statistics.median(a),
                   out_max_abs=float((out.float() - reference.output.float()).abs().max()),
                   lse_max_abs=float((lse.float() - reference.lse.float()).abs().max()))
        result['rows'].append(row)
        print(json.dumps({key: round(value, 5) if isinstance(value, float) else value for key, value in row.items()}),
              flush=True)
        args.output.write_text(json.dumps(result, indent=1))

    graph = reference_graph() if any(mode.endswith('graph') for mode in modes) else None
    if any(mode.startswith('capacity') for mode in modes):
        current = KvLengths('cuda')
        plan = prepare([args.capacity] * args.batch, current)
        if 'capacity_eager' in modes:
            for name, lens in cases:
                reference_lengths(lens)
                current.set(lens)
                measure('capacity_eager', name, plan.run, reference.run_chain)
        if 'capacity_graph' in modes:
            plan.capture_mega_graph()
            for name, lens in cases:
                reference_lengths(lens)
                current.set(lens)
                measure('capacity_graph', name, plan.run_mega, graph.replay)
        del plan
    if any(mode.startswith('fixed') for mode in modes):
        for name, lens in cases:
            reference_lengths(lens)
            plan = prepare(lens)
            if 'fixed_eager' in modes:
                measure('fixed_eager', name, plan.run, reference.run_chain)
            if 'fixed_graph' in modes:
                plan.capture_mega_graph()
                measure('fixed_graph', name, plan.run_mega, graph.replay)
            del plan


if __name__ == '__main__':
    main()
