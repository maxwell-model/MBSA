# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serving follow-up: optimized FP8 K/V pages, merge compilation, Q>8192 splitting, workspace bound,
default warmup coverage."""
import math
import unittest
from unittest.mock import patch

import torch

FP8 = torch.float8_e4m3fn


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class ServingFollowupRegression(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest('SM100/SM103 required')
        self.device = torch.device('cuda', torch.cuda.current_device())

    def case(self, lengths, extra_pages=0, *, q_dtype=torch.bfloat16, index_dtype=FP8, seed=31):
        """Paged FP8 cache with ``extra_pages`` unowned pages holding NaN bytes."""
        gen = torch.Generator(device=self.device).manual_seed(seed)
        counts = [math.ceil(kv / 128) for _, kv in lengths]
        used = sum(counts)
        kv = (torch.randn((used + extra_pages, 4, 128, 256), generator=gen, device=self.device) * .4).to(FP8)
        order = torch.randperm(used + extra_pages, generator=gen, device=self.device)
        kv.view(torch.uint8)[order[used:]] = 0x7F
        table = torch.zeros(len(lengths), max(counts), dtype=torch.int32, device=self.device)
        start = 0
        for row, count in enumerate(counts):
            table[row, :count] = order[start:start + count].int()
            start += count
        compact = torch.cat([table[row, :count] for row, count in enumerate(counts)])
        tokens = sum(q for q, _ in lengths)
        q = (torch.randn((tokens, 32, 128), generator=gen, device=self.device) * .4).to(q_dtype)
        iq = (torch.randn((tokens, 4, 128), generator=gen, device=self.device) * .4).to(index_dtype)
        ik = (torch.randn((used + extra_pages + 1, 128, 128), generator=gen, device=self.device) * .4).to(index_dtype)
        return kv, table, compact, q, iq, ik

    def run_step(self, lengths, kv, table, index_table, q, iq, ik, backend='optimized'):
        from mbsa.serving import Prefill
        k, v = kv.split(128, dim=-1)
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=self.device)
        step = Prefill(self.device, backend=backend).stage(lengths, table, index_table)
        step.run(q, k, v, index_q=iq, index_k_cache=ik, out=out)
        return out, step

    @staticmethod
    def as_bf16(kv):
        # Unowned NaN pages are never read; zero them so equality checks see only real data.
        return kv.to(torch.bfloat16).nan_to_num(0.0)

    def test_fp8_paged_kv_converts_only_request_pages(self):
        """Paged FP8 K/V converts the step's logical pages, bit-identical to BF16 K/V."""
        for lengths, extra in ((((256, 4113),), 3000), (((64, 33000), (17, 4113), (300, 9000)), 2000)):
            for q_dtype in (torch.bfloat16, FP8):
                with self.subTest(lengths=lengths, q_dtype=q_dtype):
                    kv, table, compact, q, iq, ik = self.case(lengths, extra, q_dtype=q_dtype)
                    index_table = compact if len(lengths) > 1 else table
                    out, step = self.run_step(lengths, kv, table, index_table, q, iq, ik)
                    attention = step.plan.attention
                    self.assertEqual(attention.k.shape[0], sum(math.ceil(kv / 128) for _, kv in lengths))
                    self.assertEqual(attention.k.dtype, torch.bfloat16)
                    lse = attention.out_lse.clone()
                    expected, reference = self.run_step(lengths, self.as_bf16(kv), table, index_table,
                                                        q.to(torch.bfloat16), iq, ik)
                    torch.testing.assert_close(out, expected, atol=0, rtol=0)
                    torch.testing.assert_close(lse, reference.plan.attention.out_lse, atol=0, rtol=0)
                    # A second layer binds new caches through the same prepared step.
                    kv2 = (kv.float() * -.5).to(FP8)
                    k2, v2 = kv2.split(128, dim=-1)
                    out2 = torch.empty_like(out)
                    step.run(q, k2, v2, index_q=iq, index_k_cache=ik, out=out2)
                    expected2, _ = self.run_step(lengths, self.as_bf16(kv2), table, index_table,
                                                 q.to(torch.bfloat16), iq, ik)
                    torch.testing.assert_close(out2, expected2, atol=0, rtol=0)

    def test_fp8_shared_pages_convert_the_cache_once(self):
        """Requests sharing pages own as many logical pages as the cache: convert the cache."""
        lengths = ((64, 9000), (80, 9000))
        kv, table, compact, q, iq, ik = self.case(lengths)
        table[1] = table[0]
        count = table.shape[1]
        compact = torch.cat((table[0], table[0]))
        out, step = self.run_step(lengths, kv, table, compact, q, iq, ik)
        self.assertEqual(step.plan.attention.k.shape[0], kv.shape[0])
        self.assertEqual(kv.shape[0], 2 * count)
        expected, _ = self.run_step(lengths, self.as_bf16(kv), table, compact, q, iq, ik)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)

    def test_fp8_paged_kv_rebind_reads_new_page_values(self):
        """Rebinding moved pages refreshes the compact copy from the new page table."""
        lengths = ((128, 9000),)
        kv, table, compact, q, iq, ik = self.case(lengths, 64)
        out, step = self.run_step(lengths, kv, table, table, q, iq, ik)
        moved = torch.empty_like(kv)
        moved.view(torch.uint8).fill_(0x7F)
        count = math.ceil(lengths[0][1] / 128)
        gen = torch.Generator(device=self.device).manual_seed(7)
        target = torch.randperm(kv.shape[0], generator=gen, device=self.device)[:count].int()
        moved[target.long()] = kv[table[0, :count].long()]
        table2 = table.clone()
        table2[0, :count] = target
        k, v = moved.split(128, dim=-1)
        out2 = torch.empty_like(out)
        step.plan.rebind(iq, ik.view(-1, 1, 128, 128), q, k, v, index_page_indices=table,
                         page_table=table2, out=out2)
        step.plan.run()
        torch.testing.assert_close(out2, out, atol=0, rtol=0)

    def test_fp8_paged_step_drop_without_cyclic_gc(self):
        """The compact page copy adds no reference cycle: dropping a step frees it."""
        import gc
        import weakref
        lengths = ((64, 9000),)
        kv, table, compact, q, iq, ik = self.case(lengths, 64)
        self.run_step(lengths, kv, table, table, q, iq, ik)
        torch.cuda.synchronize()
        gc.collect()
        enabled = gc.isenabled()
        gc.disable()
        try:
            out, step = self.run_step(lengths, kv, table, table, q, iq, ik)
            torch.cuda.synchronize()
            page_ref = weakref.ref(step.plan.attention.k)
            plan_ref = weakref.ref(step.plan)
            del step
            self.assertIsNone(plan_ref())
            self.assertIsNone(page_ref())
        finally:
            if enabled:
                gc.enable()

    def test_fp8_page_conversion_is_exact(self):
        from mbsa.backends.optimized.kernels.kv_pages import PagedKVConversion
        codes = torch.arange(256, dtype=torch.uint8, device=self.device)
        kv = codes.repeat(4 * 4 * 128 * 256 // 256).view(4, 4, 128, 256).view(FP8)
        kv[1] = kv[1].flip(-1)
        k, v = kv.split(128, dim=-1)
        table = torch.tensor([[3, 1], [2, 0]], dtype=torch.int32, device=self.device)
        conversion = PagedKVConversion.prepare(k, v, table, (2, 1))
        torch.cuda.synchronize()
        for converted, source in ((conversion.k, k), (conversion.v, v)):
            expected = source[torch.tensor([3, 1, 2], device=self.device)].to(torch.bfloat16)
            nan = torch.isnan(expected)
            self.assertTrue(torch.equal(torch.isnan(converted), nan))
            self.assertTrue(torch.equal(converted.view(torch.int16)[~nan], expected.view(torch.int16)[~nan]))
        self.assertEqual(conversion.table.tolist(), [[0, 1], [2, 0]])

    def test_fp8_page_tables_from_a_wide_block_table(self):
        """K/V page tables are rows of a wider block table (any start alignment), also for split requests."""
        for lengths in (((256, 4113),), ((64, 33000), (17, 4113)), ((9000, 12000),)):
            with self.subTest(lengths=lengths):
                kv, table, compact, q, iq, ik = self.case(lengths, 500)
                block = torch.zeros((len(lengths) + 3, table.shape[1] + 37), dtype=torch.int32, device=self.device)
                block[2:2 + len(lengths), :table.shape[1]] = table
                index_table = compact if len(lengths) > 1 else table
                out, _ = self.run_step(lengths, kv, block[2:2 + len(lengths)], index_table, q, iq, ik)
                expected, _ = self.run_step(lengths, self.as_bf16(kv), table, index_table, q, iq, ik)
                torch.testing.assert_close(out, expected, atol=0, rtol=0)

    def test_fp8_warmup_compiles_the_page_conversion(self):
        """After an FP8 warmup, a step on a larger cache (which converts only its pages) compiles nothing,
        also when its page table is a block-table slice that is not 16-byte aligned."""
        from mbsa.backends.optimized.kernels import kv_pages
        from mbsa.serving import Prefill, warmup

        def compiled(fn):
            caches = getattr(fn, 'device_caches', None)
            if caches is not None:
                return sum(len(entry[0]) for entry in caches.values())
            return sum(len(cache) for cache in fn.cache.values())

        # Start from no compiled conversion (other tests in this process compile it).
        caches = getattr(kv_pages._gather_bf16, 'device_caches', None)
        (caches if caches is not None else kv_pages._gather_bf16.cache).clear()
        lengths = ((256, 4113),)
        warmup(Prefill(self.device, backend='optimized'), lengths, index_format='fp8', kv_dtype=FP8)
        torch.cuda.synchronize()
        before = compiled(kv_pages._gather_bf16)
        self.assertGreater(before, 0)
        kv, table, compact, q, iq, ik = self.case(lengths, 3000)
        out, step = self.run_step(lengths, kv, table, table, q, iq, ik)
        self.assertEqual(step.plan.attention.k.shape[0], math.ceil(4113 / 128))   # the compact page copy
        shifted = torch.empty(table.numel() + 1, dtype=table.dtype, device=self.device)[1:].view_as(table)
        shifted.copy_(table)
        self.assertNotEqual(shifted.data_ptr() % 16, 0)
        out_shifted, _ = self.run_step(lengths, kv, shifted, table, q, iq, ik)
        torch.cuda.synchronize()
        self.assertEqual(compiled(kv_pages._gather_bf16), before)
        torch.testing.assert_close(out_shifted, out, atol=0, rtol=0)

    def test_split_rejects_a_narrow_index_table(self):
        lengths = ((9000, 12000),)
        kv, table, compact, q, iq, ik = self.case(lengths, 16)
        with self.assertRaisesRegex(ValueError, 'cover every request'):
            self.run_step(lengths, self.as_bf16(kv), table, table[:, :50].contiguous(), q, iq, ik)

    def test_workspace_bytes_validates_its_arguments(self):
        from mbsa.serving import Limits
        limits = Limits(max_tokens=8192, max_kv_tokens=65536, max_requests=4, max_total_kv_tokens=131072)
        for kwargs in (dict(backend='optimised'), dict(backend='optimized', kv_dtype=torch.float16)):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}), self.assertRaises(ValueError):
                limits.workspace_bytes(partial_dtype=torch.bfloat16, index_format='fp8', **kwargs)

    def test_merge_compiles_once_for_new_row_counts(self):
        from mbsa.backends.optimized.kernels.merge import _MERGE_KERNELS
        with patch.dict(_MERGE_KERNELS, clear=True):
            for lengths in (((16, 4113), (17, 4113)), ((16, 4113), (18, 4113)),
                            ((100, 4113), (17, 9000), (5, 4113))):
                kv, table, compact, q, iq, ik = self.case(lengths, index_dtype=torch.bfloat16)
                self.run_step(lengths, self.as_bf16(kv), table, compact, q, iq, ik)
            torch.cuda.synchronize()
            self.assertEqual(len(_MERGE_KERNELS), 1)

    def test_optimized_warmup_accepts_mixed_request_lengths(self):
        """Warmup binds a compact Index map: padded rows of shorter requests would leave gaps."""
        from mbsa.serving import Prefill, warmup
        prefill = Prefill(self.device, backend='optimized')
        for index_format in ('bf16', 'fp8'):
            with self.subTest(index_format=index_format):
                warmup(prefill, ((17, 4113), (23, 2195), (41, 127)), index_format=index_format)
                warmup(prefill, ((300, 9000), (9000, 12000)), index_format=index_format)   # split pieces
        torch.cuda.synchronize()

    def test_default_warmup_compiles_every_kvouter_replica_count(self):
        """Default KV-outer index kernels compile per atomic replica count. After warmup over
        warmup_request_sets, wave steps that select 32 and 64 replicas compile nothing."""
        from mbsa.cute.kvouter import build_kvouter_index as index
        from mbsa.serving import Limits, Prefill, compiled_kernel_count, warmup, warmup_request_sets
        limits = Limits(max_tokens=8192, max_kv_tokens=32768, max_requests=2, max_total_kv_tokens=65536)
        # Start from no compiled index kernel (other tests in this process compile some).
        with patch.dict(index._init_slots_compile_cache, clear=True), \
                patch.dict(index._count_edges_compile_cache, clear=True), \
                patch.dict(index._reduce_replicas_compile_cache, clear=True), \
                patch.dict(index._scatter_compile_cache, clear=True):
            prefill = Prefill(self.device)
            for lengths in warmup_request_sets(limits, index_format='fp8'):
                warmup(prefill, lengths, index_format='fp8')
            torch.cuda.synchronize()
            compiled = compiled_kernel_count()
            # A prompt tail behind decodes, and a 4-query tail packed beside a new prompt.
            for lengths, replicas in ((((466, 8192),), 32), (((4, 32768), (8187, 8187)), 64)):
                with self.subTest(lengths=lengths):
                    kv, table, compact, q, iq, ik = self.case(lengths)
                    _, step = self.run_step(lengths, self.as_bf16(kv), table, table, q, iq, ik,
                                            backend='default')
                    self.assertEqual(step.plan.attention.dynamic_kvouter.replicas, replicas)
            torch.cuda.synchronize()
            self.assertEqual(compiled_kernel_count(), compiled)

    def test_optimized_splits_queries_above_8192(self):
        """Pieces equal explicit piece staging bit for bit and match the default backend."""
        for lengths, map_kind in ((((9000, 9000),), '1d'), (((9000, 12000),), '2d'),
                                  (((17, 4113), (16500, 20000)), '1d')):
            with self.subTest(lengths=lengths, map_kind=map_kind):
                kv, table, compact, q, iq, ik = self.case(lengths, 16)
                kv = self.as_bf16(kv)
                out, step = self.run_step(lengths, kv, table, compact if map_kind == '1d' else table,
                                          q, iq, ik)
                self.assertGreater(len(step.pieces), len(lengths))
                rows = list(step._rows)
                piece_table = table[rows].contiguous()
                piece_map = torch.cat([piece_table[i, :math.ceil(kv_len / 128)]
                                       for i, (_, kv_len) in enumerate(step.pieces)])
                explicit, _ = self.run_step(step.pieces, kv, piece_table, piece_map, q, iq, ik)
                torch.testing.assert_close(out, explicit, atol=0, rtol=0)
                default, _ = self.run_step(lengths, kv, table, table, q, iq, ik, backend='default')
                torch.testing.assert_close(out.float(), default.float(), atol=4e-3, rtol=1e-2)

    def test_workspace_bytes_bounds_optimized_steps(self):
        """One prepared step with two layers stays within Limits.workspace_bytes."""
        import gc
        from mbsa.serving import Limits
        # The short-Q long-KV step is dominated by the compact FP8 page copy.
        for lengths in (((8192, 65536),), ((300, 9000), (17, 4113)), ((9000, 9000),), ((16, 262144),)):
            for kv_dtype in (torch.bfloat16, FP8):
                with self.subTest(lengths=lengths, kv_dtype=kv_dtype):
                    kv, table, compact, q, iq, ik = self.case(lengths, 256)
                    if kv_dtype == torch.bfloat16:
                        kv = self.as_bf16(kv)
                    index_table = compact if len(lengths) > 1 else table
                    limits = Limits(max_tokens=sum(n for n, _ in lengths),
                                    max_kv_tokens=max(n for _, n in lengths),
                                    max_requests=len(lengths),
                                    max_total_kv_tokens=sum(n for _, n in lengths))
                    budget = limits.workspace_bytes(partial_dtype=torch.bfloat16, index_format='fp8',
                                                    backend='optimized', kv_dtype=kv_dtype)
                    torch.cuda.synchronize()
                    gc.collect()
                    base = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    out, step = self.run_step(lengths, kv, table, index_table, q, iq, ik)
                    k, v = kv.split(128, dim=-1)
                    step.run(q, k, v, index_q=iq, index_k_cache=ik, out=out)
                    torch.cuda.synchronize()
                    # The caller's output is not part of the workspace.
                    peak = torch.cuda.max_memory_allocated() - base - out.numel() * out.element_size()
                    self.assertLessEqual(peak, budget)
                    del out, step


if __name__ == '__main__':
    unittest.main()
