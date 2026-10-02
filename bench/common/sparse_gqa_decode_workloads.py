"""Prepared decode workloads and paired timing for the unified harness."""
from __future__ import annotations

import statistics
import inspect
from functools import partial
import torch
from kernels.msa_topk_select import select_decode_top32_per_group
from msa_block_sparse.minimax_sparse_attn_triton import minimax_m3_sparse_attn_decode


class DecodeCase:
    def __init__(self, kv, batch, seed, split_k=None, ragged=False, public_runtime=True,
                 backend='both', seq_lens=None):
        if backend not in ('both', 'triton', 'cutedsl'):
            raise ValueError('backend must be both, triton, or cutedsl')
        if type(kv) is not int or kv <= 0 or type(batch) is not int or batch <= 0:
            raise ValueError('kv and batch must be positive integers')
        if seq_lens is not None and (
            not isinstance(seq_lens, list) or len(seq_lens) != batch
            or any(type(n) is not int or not 0 <= n <= kv for n in seq_lens)
        ):
            raise ValueError('seq_lens must contain batch integers in [0, kv]')
        if seq_lens is not None and ragged:
            raise ValueError('choose explicit seq_lens or generated ragged lengths')
        if backend == 'triton' and split_k is not None:
            raise ValueError('split_k selects the CuTe DSL backend only')
        self.stream = torch.cuda.Stream()
        self.kv, self.batch = kv, batch
        self.backend = backend
        self.public_fwd = None
        self.public_parent = None
        self.hmma = None
        pages = max(1, (kv + 127)//128)
        with torch.cuda.stream(self.stream):
            torch.manual_seed(seed)
            lengths = seq_lens if seq_lens is not None else [max(1, kv - ((i*7919) % max(1, kv*3//4))) if ragged else kv for i in range(batch)]
            self.seq_lens_cpu = list(lengths)
            self.lengths = torch.tensor(lengths, device='cuda', dtype=torch.int32)
            self.table = torch.arange(batch*pages, device='cuda', dtype=torch.int32).reshape(batch, pages).flip(1).contiguous()
            self.q = torch.randn(batch, 32, 128, device='cuda', dtype=torch.bfloat16)
            self.cache = torch.randn(batch*pages, 2, 128, 4, 128, device='cuda', dtype=torch.bfloat16)
            self.iq = torch.randn(batch, 4, 128, device='cuda', dtype=torch.bfloat16)
            self.ik = torch.randn(batch*pages, 128, 128, device='cuda', dtype=torch.bfloat16)
            self.route = torch.empty(4, batch, 32, device='cuda', dtype=torch.int32)
            self.scores = torch.empty(4, batch, ((pages+15)//16)*16, device='cuda', dtype=torch.float32)
            target = max(1, min(16, 64 // (batch*4)))
            chunks = 1 << (target.bit_length()-1)
            shape = (chunks, 4, batch, 32)
            self.select_workspace = (torch.empty(shape, device='cuda', dtype=torch.float32),
                                     torch.empty(shape, device='cuda', dtype=torch.int32))
            # Freeze the owning Triton heuristic independently of HMMA tuning.
            target = max(1, min(32, 256 // (batch * 4)))
            baseline_splits = 1 << (target.bit_length() - 1)
            self.baseline_splits = baseline_splits
            self.baseline_out = torch.empty_like(self.q)
            self.baseline_lse = torch.empty(batch, 32, device='cuda', dtype=torch.float32)
            self.baseline_workspace = (
                torch.empty(baseline_splits, batch, 32, 128, device='cuda', dtype=torch.bfloat16),
                torch.empty(baseline_splits, batch, 32, device='cuda', dtype=torch.float32),
            )
            self.select()
            if public_runtime:
                from msa_sparse_attn_decode import prepare_sparse_gqa_decode
                self.public_parent = prepare_sparse_gqa_decode(
                    self.q, self.cache, self.route, self.table, self.lengths,
                    num_kv_heads=4, softmax_scale=128**-0.5, backend='triton',
                    out=self.baseline_out, lse_out=self.baseline_lse,
                    workspace=self.baseline_workspace)
            if public_runtime and backend != 'triton':
                self.public_fwd = prepare_sparse_gqa_decode(
                    self.q, self.cache, self.route, self.table, self.lengths,
                    num_kv_heads=4, softmax_scale=128**-0.5, split_k=split_k,
                    backend='hmma')
                if self.public_fwd.backend != 'hmma':
                    raise ValueError('public-runtime candidate did not select HMMA')
                self.hmma = self.public_fwd._hmma
            elif backend != 'triton':
                from kernels.msa_hmma_decode import PreparedHmmaDecode
                self.hmma = PreparedHmmaDecode(self.q, self.cache, self.route, self.table,
                                               self.lengths, split_k=split_k)
            self.parent()
            if self.hmma is not None:
                self.hmma()
                torch.testing.assert_close(self.hmma.out, self.baseline_out, atol=3e-2, rtol=3e-2)
                torch.testing.assert_close(self.hmma.lse, self.baseline_lse, atol=2e-3, rtol=2e-3)
        self.stream.synchronize()

    def select(self):
        return select_decode_top32_per_group(
            self.iq, self.ik, block_table=self.table, seq_lens=self.lengths,
            max_seq_len=self.kv, sm_scale=128**-0.5, score_out=self.scores,
            topk_out=self.route, workspace=self.select_workspace,
        )

    def parent(self):
        if self.public_parent is not None:
            return self.public_parent.run()
        minimax_m3_sparse_attn_decode(
            self.q, self.cache, self.route, self.table, self.lengths, 4, 128**-0.5,
            self.baseline_out, 1, lse_out=self.baseline_lse, workspace=self.baseline_workspace,
        )

    def parent_chain(self):
        if self.public_parent is not None:
            return self._public_chain(self.public_parent)
        self.select()
        self.parent()

    def _public_chain(self, plan):
        from kernels.minimax_top32 import decode_top32_chain
        return decode_top32_chain(
            index_q=self.iq, index_k_cache=self.ik, main_q=self.q,
            main_kv_cache=self.cache, block_table=self.table, seq_lens=self.lengths,
            max_seq_len=self.kv, softmax_scale=128**-0.5,
            score_out=self.scores, route_out=self.route,
            selection_workspace=self.select_workspace, prepared_fwd=plan)

    def candidate_chain(self):
        if self.public_fwd is not None:
            return self._public_chain(self.public_fwd)
        else:
            self.select()
            self.hmma()

    def selector_components(self):
        """Bind the owning selector's exact launches in this isolated worker.

        Record during preparation, restore globals before timing, and avoid
        duplicating its launch heuristics or argument/stride construction.
        """
        from kernels.msa_topk_select import minimax_m3_index_decode
        namespace = inspect.unwrap(minimax_m3_index_decode).__globals__
        names = {'_decode_index_score_kernel': 'selector_score',
                 '_topk_index_partial_kernel': 'selector_local_topk',
                 '_topk_index_merge_kernel': 'selector_route_merge'}
        original = {name: namespace[name] for name in names}
        calls = {}

        class Recorder:
            def __init__(self, kernel, label):
                self.kernel, self.label = kernel, label

            def __getitem__(self, grid):
                launch = self.kernel[grid]

                def record(*args, **kwargs):
                    if self.label in calls:
                        raise RuntimeError('selector component launched more than once')
                    calls[self.label] = partial(launch, *args, **kwargs)
                    return launch(*args, **kwargs)
                return record

        with torch.cuda.stream(self.stream):
            expected = self.route.clone()
            try:
                for name, label in names.items():
                    namespace[name] = Recorder(original[name], label)
                self.select()
            finally:
                namespace.update(original)
            if set(calls) != set(names.values()):
                raise RuntimeError('selector component launch set changed')
            for call in calls.values():
                call()
            torch.testing.assert_close(self.route, expected, atol=0, rtol=0)
        return calls

    def measure_pairs(self, warmups, samples, mode, *, selector_components=False):
        if mode not in ('graph', 'eager'):
            raise ValueError('mode must be graph or eager')
        calls = {'selector': self.select}
        if self.backend != 'cutedsl':
            calls.update(triton_attention=self.parent, triton_chain=self.parent_chain)
        if self.hmma is not None:
            calls.update(hmma_attention=(self.public_fwd.run if self.public_fwd is not None else self.hmma),
                         hmma_chain=self.candidate_chain, hmma_partial=self.hmma.attention)
            if self.hmma.combine is not None:
                calls['hmma_merge'] = self.hmma.reduce
        if selector_components:
            calls.update(self.selector_components())
        graphs = {}
        with torch.cuda.stream(self.stream):
            for call in calls.values():
                for _ in range(warmups):
                    call()
            if mode == 'graph':
                for name, call in calls.items():
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=self.stream):
                        call()
                    graphs[name] = graph
                calls = {name: graph.replay for name, graph in graphs.items()}
                for call in calls.values():
                    for _ in range(warmups):
                        call()
                for parent, candidate in [('triton_attention', 'hmma_attention'),
                                          ('triton_chain', 'hmma_chain')]:
                    if parent not in calls or candidate not in calls:
                        continue
                    calls[parent]()
                    calls[candidate]()
                    torch.testing.assert_close(self.hmma.out, self.baseline_out, atol=3e-2, rtol=3e-2)
                    torch.testing.assert_close(self.hmma.lse, self.baseline_lse, atol=2e-3, rtol=2e-3)
            starts = [torch.cuda.Event(enable_timing=True) for _ in calls]
            ends = [torch.cuda.Event(enable_timing=True) for _ in calls]
        self.stream.synchronize()
        before = torch.cuda.memory_stats()['allocation.all.allocated']
        result = {name: [] for name in calls}
        names = list(calls)
        for sample in range(samples):
            order = names if sample % 2 == 0 else list(reversed(names))
            for index, name in enumerate(order):
                with torch.cuda.stream(self.stream):
                    starts[index].record(self.stream)
                    calls[name]()
                    ends[index].record(self.stream)
                ends[index].synchronize()
                result[name].append(starts[index].elapsed_time(ends[index]))
        after = torch.cuda.memory_stats()['allocation.all.allocated']
        if before != after:
            raise RuntimeError(f'allocation count changed {before} -> {after}')
        properties = torch.cuda.get_device_properties(self.q.device)
        return {'samples_ms': result['triton_chain' if self.backend == 'triton' else 'hmma_chain'], 'paired_ms': result,
                'medians_ms': {name: statistics.median(values) for name, values in result.items()},
                'backend_requested': self.backend,
                'backends_timed': (['triton', 'cutedsl'] if self.backend == 'both' else [self.backend]),
                'public_runtime': self.public_parent is not None,
                'actual_backends': {'triton': ('triton' if self.public_parent is None else self.public_parent.backend),
                                    'cutedsl': (None if self.hmma is None else
                                                ('hmma' if self.public_fwd is None else self.public_fwd.backend))},
                'torch_version': torch.__version__, 'torch_cuda_build': torch.version.cuda,
                'seq_lens': self.lengths.cpu().tolist(),
                'shape': {'batch': self.batch, 'max_kv': self.kv, 'hq': 32, 'hkv': 4, 'd': 128,
                          'page_size': 128, 'topk': 32, 'decode_query_len': 1},
                'device': {'name': properties.name, 'sm_count': properties.multi_processor_count,
                           'capability': list(torch.cuda.get_device_capability(self.q.device)),
                           'total_memory': properties.total_memory},
                'allocation_count_before': before, 'allocation_count_after': after,
                'split_k': None if self.hmma is None else self.hmma.split_k, 'mode': mode,
                'triton_split_k': self.baseline_splits,
                'triton_workspace_bytes': sum(t.numel()*t.element_size() for t in self.baseline_workspace),
                'triton_attention_launch_count': 2,
                'reset_bytes': 0,
                'workspace_bytes': None if self.hmma is None else self.hmma.workspace.numel()*self.hmma.workspace.element_size(),
                'attention_launch_count': None if self.hmma is None else 1 + int(self.hmma.combine is not None)}
