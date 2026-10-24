# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent oracle and live lifecycle gates for the optimized attention backend."""
import inspect
import itertools
import math
import unittest
from unittest import mock
import torch

from mbsa import prepare_cutedsl_pipeline
from tests.common import BackendCaseMixin
from tests.regression import test_backends as backend_support
from tests.reference import reference_pipeline


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class OptimizedBackend(BackendCaseMixin, unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("SM100/SM103 required")

    def test_fixed_ragged_tma_excludes_physical_page_padding(self):
        """Logical page bounds follow packed lengths and mappings on replay."""
        from mbsa.backends.optimized.bsa.cute_dsl.sparse.bsa_attn_sm100_blk128 import bsa_attn_sm100_blk128_fwd
        from tests.regression.test_fixed_fa4 import fixed_oracle

        for use_clc, dtype, width in itertools.product(
                (False, True), (torch.bfloat16, torch.float16), (128, 144)):
            with self.subTest(clc=use_clc, dtype=dtype, row_stride=width):
                gen = torch.Generator(device='cuda').manual_seed(491)
                def rand(shape):
                    return (torch.randn(shape, device='cuda', generator=gen) * .2).to(dtype)
                q = rand((51, 32, 128))
                k, v = (rand((8, 4, 128, width))[..., :128] for _ in range(2))
                original = (k.clone(), v.clone())
                table = torch.tensor([[3, 1], [5, 2], [7, 4]], device='cuda', dtype=torch.int32)
                rows = [[b, m, b * 17, (b + 1) * 17, 0, n]
                        for b, n in enumerate((127, 128, 129)) for m in range(2)]
                tiles = torch.tensor(rows, device='cuda', dtype=torch.int32)
                routes = torch.zeros((1,), device='cuda', dtype=torch.int32)
                out = torch.empty_like(q)
                lse = torch.empty((32, 51), device='cuda', dtype=torch.float32)
                def run():
                    return bsa_attn_sm100_blk128_fwd(
                        q, k, v, routes, 0, pack_gqa=True, return_lse=True,
                        out=out, lse=lse, causal=True, implicit_fixed_support=True,
                        page_table=table, packed_tiles=tiles, use_clc_scheduler=use_clc)
                def reset_and_reference(lengths, changed=False):
                    k.copy_(original[0])
                    v.copy_(original[1])
                    if changed:
                        k.mul_(.75)
                        v.add_(3)
                    expected_o, expected_l = [], []
                    for b, n in enumerate(lengths):
                        tiles[b * 2:(b + 1) * 2, 5] = n
                        ids = table[b].long()
                        logical = [x[ids].permute(0, 2, 1, 3).reshape(-1, 4, 128)[:n]
                                   for x in (k, v)]
                        oo, ll = fixed_oracle(q[b * 17:(b + 1) * 17], *logical)
                        expected_o.append(oo)
                        expected_l.append(ll)
                        # This storage is physically in bounds but logically outside
                        # the request; only the request-bounded TMA may read it safely.
                        for page in range((n + 127) // 128):
                            valid = min(128, n - page * 128)
                            k[ids[page], :, valid:].fill_(torch.nan)
                            v[ids[page], :, valid:].fill_(torch.nan)
                    return torch.cat(expected_o), torch.cat(expected_l, dim=1)
                expected = reset_and_reference((127, 128, 129))
                run()
                torch.testing.assert_close(out.float(), expected[0], atol=.025, rtol=.025)
                torch.testing.assert_close(lse, expected[1], atol=.002, rtol=.002)
                baseline = (out.clone(), lse.clone())
                out.fill_(torch.nan)
                lse.fill_(torch.nan)
                # A skipped replay must fail the same oracle used for live capture.
                with self.assertRaises(AssertionError):
                    torch.testing.assert_close(out.float(), expected[0], atol=.025, rtol=.025)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        run()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                torch.cuda.current_stream().wait_stream(stream)
                table.copy_(table.roll(1, 0))
                q.mul_(.875)
                expected = reset_and_reference((126, 127, 128), changed=True)
                with self.assertRaises(AssertionError):
                    torch.testing.assert_close(baseline[0].float(), expected[0], atol=.025, rtol=.025)
                out.fill_(torch.nan)
                lse.fill_(torch.nan)
                graph.replay()
                torch.testing.assert_close(out.float(), expected[0], atol=.025, rtol=.025)
                torch.testing.assert_close(lse, expected[1], atol=.002, rtol=.002)
                live = (out.clone(), lse.clone())
                out.fill_(torch.nan)
                lse.fill_(torch.nan)
                graph.replay()
                for actual, saved in zip((out, lse), live):
                    torch.testing.assert_close(actual, saved, atol=0, rtol=0)
                run()
                for actual, saved in zip((out, lse), live):
                    torch.testing.assert_close(actual, saved, atol=0, rtol=0)

    def test_live_index_page_views_rebind_and_capture(self):
        lengths = ((17, 8192),)
        for layout in ('offset', 'identity', 'padded-row', 'unaligned', 'strided'):
            with self.subTest(layout=layout):
                data, inputs, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
                count = data['pages'].numel()
                stride = 2 if layout == 'strided' else 1
                offset = 4 if layout in ('offset', 'identity', 'padded-row') else 1
                direct = layout in ('offset', 'identity', 'padded-row')
                backing = torch.full((count * stride + offset + 5,), -1,
                                     dtype=torch.int32, device='cuda')
                pages = (backing[offset:offset + count + 5].view(1, -1) if layout == 'padded-row'
                         else backing[offset:offset + count * stride:stride])
                active = pages[0, :count] if pages.ndim == 2 else pages
                active.copy_(torch.arange(count, device='cuda') if layout == 'identity'
                             else data['pages'])
                if layout != 'identity':
                    self.assertFalse(torch.equal(active, torch.arange(count, device='cuda')))
                kwargs['index_page_indices'] = pages
                if not direct:
                    with self.assertRaisesRegex(ValueError, 'direct Index page map'):
                        prepare_cutedsl_pipeline(*inputs, backend='optimized', **kwargs)
                    continue
                plan = prepare_cutedsl_pipeline(*inputs, backend='optimized', **kwargs)

                def oracle():
                    row = pages[0, :count] if pages.ndim == 2 else pages
                    return backend_support.BackendLifecycle.reference(
                        self, dict(data, pages=row), inputs, lengths)

                index = plan.plan._state.index
                # Even an initially identity caller map remains mutable.
                self.assertTrue(index.cutedsl_use_page_table)
                self.assertEqual(index.cutedsl_index_pages.data_ptr() == active.data_ptr(), direct)
                saved = tuple(t.clone() for t in plan.run())
                self.compare(saved, oracle())
                old_state, old_args = plan.plan._state, index._local_top16_args
                replacement = torch.empty_strided(pages.shape, pages.stride(),
                                                  dtype=pages.dtype, device=pages.device)
                replacement.copy_(pages)
                row = replacement[0, :count] if replacement.ndim == 2 else replacement
                row.copy_(row.roll(1))
                out = torch.full_like(inputs[2], 123)
                bad = torch.empty_strided(pages.shape, pages.stride(),
                                          dtype=pages.dtype, device=pages.device)
                bad.copy_(replacement)
                bad[(0,) * bad.ndim] = -1
                with self.assertRaises(ValueError):
                    plan.rebind(*inputs, index_page_indices=bad,
                                page_table=data['table'], out=out)
                self.assertIs(plan.plan._state, old_state)
                self.assertTrue(bool((out == 123).all()))
                for actual, previous in zip(plan.run(), saved):
                    torch.testing.assert_close(actual, previous, rtol=0, atol=0)
                with mock.patch('mbsa.backends.optimized.index.indexer_local_top16',
                                side_effect=RuntimeError('map ABI compile failure')):
                    with self.assertRaisesRegex(RuntimeError, 'map ABI compile failure'):
                        plan.rebind(*inputs, index_page_indices=replacement,
                                    page_table=data['table'], out=out)
                self.assertIs(plan.plan._state, old_state)
                plan.rebind(*inputs, index_page_indices=replacement,
                            page_table=data['table'], out=out)
                pages = replacement
                expected = oracle()
                self.assertFalse(torch.equal(saved[2].sort(-1).values, expected[2].sort(-1).values))
                self.compare(plan.run(), expected)
                index = plan.plan._state.index
                self.assertEqual(index.cutedsl_index_pages.data_ptr() == row.data_ptr(), stride == 1)
                if stride == 1:
                    # The actual cached ABI must follow the new map address.
                    current_args = index._local_top16_args
                    stale = list(current_args)
                    stale[2] = old_args[2]
                    index._local_top16_args = tuple(stale)
                    try:
                        observed = tuple(t.clone() for t in plan.run())
                        self.assertFalse(torch.equal(observed[2].sort(-1).values,
                                                     expected[2].sort(-1).values))
                    finally:
                        index._local_top16_args = current_args
                self.compare(plan.run(), expected)
                if stride == 1:
                    weak_backing = torch.empty(pages.numel() + 1,
                        dtype=pages.dtype, device=pages.device)
                    weak_pages = weak_backing[1:].view(pages.shape)
                    weak_pages.copy_(pages)
                    previous = plan.plan._state
                    with self.assertRaisesRegex(ValueError, 'direct Index page map'):
                        plan.rebind(*inputs, index_page_indices=weak_pages,
                                    page_table=data['table'], out=out)
                    self.assertIs(plan.plan._state, previous)
                    self.compare(plan.run(), expected)
                plan.capture_mega_graph()
                row.copy_(row.roll(2))
                inputs[1].copy_(inputs[1].roll(1, dims=0))
                inputs[4].add_(3)
                refreshed = oracle()
                self.assertFalse(torch.equal(expected[2].sort(-1).values,
                                             refreshed[2].sort(-1).values))
                plan.plan.attention.out.fill_(torch.nan)
                plan.plan.attention.out_lse.fill_(torch.nan)
                plan._owner.dynamic_route.fill_(-99)
                index.candidate_storage.view(torch.int32)[:, 16:].fill_(-99)
                with self.assertRaises(AssertionError):
                    self.compare((plan.plan.attention.out, plan.plan.attention.out_lse,
                                  plan._owner.dynamic_route), refreshed)
                live = tuple(t.clone() for t in plan.run_mega())
                self.compare(live, refreshed)
                for actual, observed in zip(plan.run(), live):
                    torch.testing.assert_close(actual, observed, rtol=0, atol=0)

    def test_packed_index_page_views_reject_gaps_and_strides(self):
        lengths = ((17, 8192), (23, 4113), (19, 127))
        for layout in ('active', 'padded', 'strided'):
            with self.subTest(layout=layout):
                data, inputs, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
                counts = data['counts']
                if layout == 'padded':
                    pages = torch.full((len(lengths), max(counts) + 3), -1,
                                       dtype=torch.int32, device='cuda')
                    start = 0
                    for row, count in enumerate(counts):
                        pages[row, :count].copy_(data['pages'][start:start + count])
                        start += count
                elif layout == 'strided':
                    pages = torch.empty(data['pages'].numel() * 2,
                                        dtype=torch.int32, device='cuda')[::2]
                    pages.copy_(data['pages'])
                else:
                    pages = data['pages']
                kwargs['index_page_indices'] = pages
                if layout != 'active':
                    with self.assertRaisesRegex(ValueError, 'direct Index page map'):
                        prepare_cutedsl_pipeline(*inputs, backend='optimized', **kwargs)
                    continue
                plan = prepare_cutedsl_pipeline(*inputs, backend='optimized', **kwargs)
                index = plan.plan._state.index
                self.assertEqual(index.cutedsl_packed_pages.data_ptr() == pages.data_ptr(), layout == 'active')

                def oracle():
                    active = (torch.cat([pages[r, :n] for r, n in enumerate(counts)])
                              if pages.ndim == 2 else pages)
                    return backend_support.BackendLifecycle.reference(
                        self, dict(data, pages=active), inputs, lengths)

                self.compare(plan.run(), oracle())
                plan.capture_mega_graph()
                if pages.ndim == 2:
                    for row, count in enumerate(counts):
                        pages[row, :count].copy_(pages[row, :count].roll(1))
                else:
                    pages.copy_(pages.roll(1))
                inputs[4].add_(3)
                expected = oracle()
                plan.plan.attention.out.fill_(torch.nan)
                plan.plan.attention.out_lse.fill_(torch.nan)
                plan._owner.dynamic_route.fill_(-99)
                live = tuple(t.clone() for t in plan.run_mega())
                self.compare(live, expected)
                for actual, observed in zip(plan.run(), live):
                    torch.testing.assert_close(actual, observed, rtol=0, atol=0)

    def test_direct_index_page_async_drop_lifetime(self):
        import gc
        import weakref

        def launch():
            lengths = ((17, 8192),)
            data, inputs, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
            backing = torch.empty(data['pages'].numel() + 4, dtype=torch.int32, device='cuda')
            pages = backing[4:]
            pages.copy_(data['pages'])
            kwargs['index_page_indices'] = pages
            plan = prepare_cutedsl_pipeline(*inputs, backend='optimized', **kwargs)
            self.assertEqual(plan.plan._state.index.cutedsl_index_pages.data_ptr(), pages.data_ptr())
            before, after = pages.clone(), pages.roll(1)
            original_v, changed_v = inputs[4].clone(), inputs[4] + 3
            pages.copy_(after)
            inputs[4].copy_(changed_v)
            expected = tuple(t.clone() for t in plan.run())
            self.compare(expected, backend_support.BackendLifecycle.reference(
                self, dict(data, pages=after), inputs, lengths))
            pages.copy_(before)
            inputs[4].copy_(original_v)
            producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
            producer.wait_stream(torch.cuda.current_stream())
            ready = torch.cuda.Event()
            with torch.cuda.stream(producer):
                torch.cuda._sleep(10_000_000)
                pages.copy_(after)
                inputs[4].copy_(changed_v)
                ready.record()
            ref = weakref.ref(pages)
            del kwargs, data, backing, pages
            gc.collect()
            self.assertIsNotNone(ref())
            consumer.wait_event(ready)
            with torch.cuda.stream(consumer):
                torch.cuda._sleep(10_000_000)
                final = tuple(t.clone() for t in plan.run())
            # Drop plan, maps and operands while the consumer is still queued.
            # Only independent copy sources and expected/final outputs survive.
            return final, expected, producer, consumer, (after, changed_v, ready)

        final, expected, producer, consumer, pending = launch()
        gc.collect()
        pressure = [torch.empty(512, dtype=torch.uint8, device='cuda').fill_(77)
                    for _ in range(128)]
        torch.cuda.current_stream().wait_stream(consumer)
        for actual, observed in zip(final, expected):
            torch.testing.assert_close(actual, observed, rtol=0, atol=0)
        self.assertEqual(len(pressure), 128)

    def test_single_partition_direct_ids_ties_pages_and_live_graph(self):
        from mbsa.backends.optimized.src.sm100.router import indexer_local_top16
        from tests.reference import MaxwellBlockSparseAttentionReference

        oracle = MaxwellBlockSparseAttentionReference(index_head_dim=128, head_dim=128)
        for kn in (8192, 3297):
            with self.subTest(logical_k=kn):
                qn, heads = 35, 4
                pages = (kn + 127) // 128
                ids = torch.arange(pages - 1, -1, -1, device='cuda', dtype=torch.int32)
                iq = torch.zeros((qn, heads, 128), device='cuda', dtype=torch.bfloat16)
                ik = (torch.arange(pages, device='cuda') % 3).to(torch.bfloat16)
                ik = ik[:, None, None, None].expand(pages, 128, 1, 128).contiguous()
                logical = ik[ids.long()].flatten(0, 1)[:kn]
                routes = [torch.full((heads, 32, qn), -99, device='cuda', dtype=torch.int32)
                          for _ in range(2)]

                def run(fused):
                    return indexer_local_top16(
                        iq, ik.flatten(0, 1), page_indices=ids, logical_k_tokens=kn,
                        partition_blocks=1024,
                        use_page_table=True, out_compact_ids=routes[int(fused)],
                        _fuse_single_partition_compact=fused)

                def expected():
                    positions = torch.arange(kn - qn, kn, device='cuda')[None]
                    allowed = oracle._allowed_keys(query_positions=positions,
                        key_len=kn, attention_mask=None)
                    selected = oracle._select_blocks(iq[None], logical[None], positions, allowed)
                    return selected[0].permute(1, 2, 0).to(torch.int32)

                run(False)
                run(True)
                baseline = routes[0][:, 16:].clone()
                torch.testing.assert_close(baseline, expected(), rtol=0, atol=0)
                torch.testing.assert_close(routes[1], routes[0], rtol=0, atol=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run(True)
                iq.fill_(1)
                refreshed = expected()
                self.assertFalse(torch.equal(baseline, refreshed))
                routes[1][:, 16:].fill_(-99)
                self.assertFalse(torch.equal(routes[1][:, 16:], refreshed))
                graph.replay()
                live = routes[1].clone()
                torch.testing.assert_close(live[:, 16:], refreshed, rtol=0, atol=0)
                self.assertTrue(bool((live[:, :16] == -99).all()))
                run(False)
                torch.testing.assert_close(live, routes[0], rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, 'one KV partition'):
                    indexer_local_top16(iq, ik.flatten(0, 1), page_indices=ids,
                        logical_k_tokens=kn, partition_blocks=1024,
                        _fuse_single_partition_compact=True)
        # Compact output alone does not permit bypassing a real partition merge.
        many_k = torch.empty((1025 * 128, 1, 128), device='cuda', dtype=torch.bfloat16)
        many_pages = torch.arange(1025, device='cuda', dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, 'one KV partition'):
            indexer_local_top16(iq, many_k, page_indices=many_pages,
                logical_k_tokens=many_k.shape[0], partition_blocks=1024,
                out_compact_ids=routes[1], _fuse_single_partition_compact=True)

    def test_direct_ids_partition_boundaries_and_fp8(self):
        from mbsa.backends.optimized.src.sm100.router import indexer_local_top16
        from tests.reference import MaxwellBlockSparseAttentionReference

        oracle = MaxwellBlockSparseAttentionReference(index_head_dim=128, head_dim=128)
        for dtype, capacity in ((torch.bfloat16, 1024), (torch.float8_e4m3fn, 2048)):
            for pages in (capacity, capacity + 1):
                with self.subTest(dtype=dtype, pages=pages):
                    qn, kn = 35, pages * 128
                    iq = torch.zeros((qn, 4, 128), device='cuda', dtype=dtype)
                    ik = (torch.arange(pages, device='cuda') % 3).to(dtype)
                    ik = ik[:, None, None, None].expand(pages, 128, 1, 128).contiguous()
                    ids = torch.arange(pages - 1, -1, -1, device='cuda', dtype=torch.int32)
                    routes = [torch.full((4, 32, qn), -99, device='cuda', dtype=torch.int32)
                              for _ in range(2)]
                    fused = pages == capacity

                    def run(which):
                        return indexer_local_top16(iq, ik.flatten(0, 1),
                            partition_blocks=capacity, page_indices=ids,
                            logical_k_tokens=kn, use_page_table=True,
                            out_compact_ids=routes[which],
                            _fuse_single_partition_compact=fused and which == 1)

                    def expected():
                        positions = torch.arange(kn - qn, kn, device='cuda')[None]
                        allowed = oracle._allowed_keys(query_positions=positions,
                            key_len=kn, attention_mask=None)
                        logical = ik[ids.long()].flatten(0, 1)
                        selected = oracle._select_blocks(iq[None], logical[None], positions, allowed)
                        return selected[0].permute(1, 2, 0).to(torch.int32)

                    run(0)
                    run(1)
                    baseline = routes[0][:, 16:].clone()
                    torch.testing.assert_close(baseline, expected(), rtol=0, atol=0)
                    torch.testing.assert_close(routes[0], routes[1], rtol=0, atol=0)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run(1)
                    iq.fill_(1)
                    refreshed = expected()
                    self.assertFalse(torch.equal(baseline, refreshed))
                    routes[1][:, 16:].fill_(-99)
                    graph.replay()
                    live = routes[1].clone()
                    torch.testing.assert_close(live[:, 16:], refreshed, rtol=0, atol=0)
                    self.assertTrue(bool((live[:, :16] == -99).all()))
                    run(0)
                    torch.testing.assert_close(live, routes[0], rtol=0, atol=0)
                    if not fused:
                        with self.assertRaisesRegex(ValueError, 'one KV partition'):
                            indexer_local_top16(iq, ik.flatten(0, 1),
                                partition_blocks=capacity, page_indices=ids,
                                logical_k_tokens=kn, out_compact_ids=routes[1],
                                _fuse_single_partition_compact=True)

    def test_public_single_partition_boundary_reference_and_live_graph(self):
        operands = self.inputs(17, 131072)
        plan = prepare_cutedsl_pipeline(*operands, backend='optimized',
                                       partial_dtype=torch.bfloat16)
        index = plan.plan._state.index
        self.assertEqual(index.candidate_partition_blocks, 1024)
        self.assertIs(plan.plan.attention.fixed_tmem_load_reduce, False)
        self.assertEqual(index._local_top16_key[-2:], ('fuse_single_partition_compact', True))
        expected = reference_pipeline(*operands)
        baseline = tuple(t.clone() for t in plan.run())
        self.compare(baseline, expected)
        plan.capture_mega_graph()
        operands[4].add_(3)
        expected = reference_pipeline(*operands)
        plan.plan.attention.out.fill_(torch.nan)
        plan.plan.attention.out_lse.fill_(torch.nan)
        plan.plan.attention.tile_plan.dynamic_route.fill_(-99)
        index.candidate_storage.view(torch.int32)[:, 16:].fill_(-99)
        live = tuple(t.clone() for t in plan.run_mega())
        self.compare(live, expected)
        self.assertFalse(torch.allclose(live[0], baseline[0], atol=.03, rtol=.03))
        for actual, observed in zip(plan.run(), live):
            torch.testing.assert_close(actual, observed, rtol=0, atol=0)

    def test_fixed_reduce_alignment_guard_is_request_local(self):
        aligned = ((16, 16400),)
        _, operands, kwargs = backend_support.BackendLifecycle.fixture(self, aligned)
        plan = prepare_cutedsl_pipeline(*operands, backend='optimized', **kwargs)
        self.assertIs(plan.plan.attention.fixed_tmem_load_reduce,
                      torch.cuda.get_device_capability() == (10, 3))
        # The second batch also has an aligned aggregate offset: each request
        # must independently satisfy the collective-load precondition.
        for lengths in (((16, 16400), (17, 16384)),
                        ((17, 16384), (111, 16384))):
            with self.subTest(lengths=lengths):
                _, operands, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
                plan = prepare_cutedsl_pipeline(*operands, backend='optimized', **kwargs)
                self.assertIs(plan.plan.attention.fixed_tmem_load_reduce, False)

    def test_public_fp8_single_partition_2048_selection_and_live_graph(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest('public FP8 PB2048 policy is SM103-only; SM100 uses PB1024')
        operands = self.inputs(8192, 196608)
        operands = (*(t.to(torch.float8_e4m3fn) for t in operands[:2]), *operands[2:])
        plan = prepare_cutedsl_pipeline(*operands, backend='optimized',
                                       partial_dtype=torch.bfloat16)
        index = plan.plan._state.index
        self.assertEqual(index.candidate_partition_blocks, 2048)
        self.assertIs(plan.plan.attention.fixed_tmem_load_reduce, True)
        self.assertEqual(index._local_top16_key[-2:], ('fuse_single_partition_compact', True))
        self.assertEqual(index._local_top16_key[index._local_top16_key.index('query_tile') + 1], 256)

        def check(observed):
            for row in (0, 127, 4096, 8191):
                expected = reference_pipeline(operands[0][row:row+1], operands[1],
                    operands[2][row:row+1], *operands[3:], query_offset=196608-8192+row)
                self.compare(tuple(t[row:row+1] if i < 2 else t[:, row:row+1]
                                   for i, t in enumerate(observed)), expected)

        baseline = tuple(t.clone() for t in plan.run())
        check(baseline)
        plan.capture_mega_graph()
        plan.plan.attention.out.fill_(torch.nan)
        plan.plan.attention.out_lse.fill_(torch.nan)
        plan.plan.attention.tile_plan.dynamic_route.fill_(-99)
        index.candidate_storage.view(torch.int32)[:, 16:].fill_(-99)
        for actual, observed in zip(plan.run_mega(), baseline):
            torch.testing.assert_close(actual, observed, rtol=0, atol=0)
        operands[4].add_(3)
        plan.plan.attention.out.fill_(torch.nan)
        plan.plan.attention.out_lse.fill_(torch.nan)
        plan.plan.attention.tile_plan.dynamic_route.fill_(-99)
        index.candidate_storage.view(torch.int32)[:, 16:].fill_(-99)
        live = tuple(t.clone() for t in plan.run_mega())
        check(live)
        self.assertFalse(torch.allclose(live[0], baseline[0], atol=.03, rtol=.03))
        for actual, observed in zip(plan.run(), live):
            torch.testing.assert_close(actual, observed, rtol=0, atol=0)

    def test_public_fp8_pb2048_context_views_and_rebind(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest('automatic FP8 PB2048 selection is SM103-only')
        for kn in (8192, 131072, 139264, 262144, 270336, 1048576):
            with self.subTest(kv=kn):
                iq, ik, q, k, v = self.inputs(8192, kn)
                iq, ik = iq.to(torch.float8_e4m3fn), ik.to(torch.float8_e4m3fn)
                pages = ik.view(-1, 1, 128, 128)
                plans = [prepare_cutedsl_pipeline(iq, keys, q, k, v,
                    backend='optimized', partial_dtype=torch.bfloat16)
                    for keys in (ik, pages)]
                indices = [p.plan._state.index for p in plans]
                for index in indices:
                    self.assertEqual(index.candidate_partition_blocks, 2048)
                    self.assertFalse(index.cutedsl_use_page_table)
                    self.assertFalse(index._lean192_early_router)
                    self.assertEqual(index.cutedsl_index_k.data_ptr(), ik.data_ptr())
                    key = index._local_top16_key
                    actual_query_tile = (key[key.index('query_tile') + 1]
                                         if 'query_tile' in key else 128)
                    self.assertEqual(actual_query_tile, 128 if kn == 8192 else 256)
                    self.assertEqual('fuse_single_partition_compact' in key,
                                     kn <= 262144)
                self.assertEqual(indices[0]._local_top16_key, indices[1]._local_top16_key)
                self.assertIs(indices[0]._local_top16_compiled, indices[1]._local_top16_compiled)
                first = tuple(t.clone() for t in plans[0].run())
                second = tuple(t.clone() for t in plans[1].run())
                self.compare(second, first)
                torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
                if kn > 262144:
                    for name in ('cutedsl_partition_scores', 'cutedsl_partition_ids'):
                        torch.testing.assert_close(getattr(indices[0], name),
                            getattr(indices[1], name), rtol=0, atol=0)

                def check(actual, active_iq=iq, active_ik=ik):
                    for row in (0, 127, 128, 8191):
                        expected = reference_pipeline(active_iq[row:row+1].float(),
                            active_ik.view(-1, 1, 128).float(), q[row:row+1], k, v,
                            query_offset=kn-8192+row)
                        self.compare((actual[0][row:row+1], actual[1][row:row+1],
                                      actual[2][:, row:row+1]), expected)

                check(second)
                plans[1].capture_mega_graph()
                attention = plans[1].plan._state.attention

                def replay():
                    attention.out.fill_(torch.nan)
                    attention.out_lse.fill_(torch.nan)
                    attention.tile_plan.dynamic_route.fill_(-99)
                    indices[1].candidate_storage.view(torch.int32)[:, 16:].fill_(-99)
                    if kn > 262144:
                        indices[1].cutedsl_partition_ids.fill_(-999)
                    self.assertFalse(bool(torch.isfinite(attention.out).all()))
                    actual = tuple(t.clone() for t in plans[1].run_mega())
                    torch.cuda.synchronize()
                    if kn > 262144:
                        self.assertFalse(bool((indices[1].cutedsl_partition_ids == -999).any()))
                    return actual

                for actual, expected in zip(replay(), second):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for operand in (iq, pages):
                    operand.copy_((-operand.float()).to(operand.dtype))
                    changed = replay()
                    self.assertFalse(torch.equal(changed[2], second[2]))
                    check(changed)
                    for actual, expected in zip(plans[1].run(), changed):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    operand.copy_((-operand.float()).to(operand.dtype))
                for actual, expected in zip(replay(), second):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                # Rebind the uncaptured plan. A rejected replacement preserves
                # its state and output; valid new addresses select the same PB.
                before = plans[0].plan._state
                retained = (before.attention.out, before.attention.out_lse,
                            before.attention.tile_plan.dynamic_route, q)
                rejected_baseline = tuple(t.clone() for t in retained)
                with self.assertRaises(ValueError):
                    plans[0].rebind(iq, ik, q, k, v,
                                    index_page_indices=before.bindings['index_page_indices'],
                                    page_table=None, out=q)
                self.assertIs(plans[0].plan._state, before)
                for actual, expected in zip(retained, rejected_baseline):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.compare(plans[0].run(), first)
                replacement_iq = iq.float().roll(1, -1).to(iq.dtype)
                replacement_ik = ik.float().roll(2, -1).to(ik.dtype)
                paged_plan = prepare_cutedsl_pipeline(iq, pages, q, k, v,
                    backend='optimized', partial_dtype=torch.bfloat16)
                for target, replacement_keys in (
                    (plans[0], replacement_ik),
                    (paged_plan, replacement_ik.view_as(pages)),
                ):
                    old_state = target.plan._state
                    replacement_out = torch.empty_like(q)
                    target.rebind(replacement_iq, replacement_keys, q, k, v,
                        index_page_indices=old_state.bindings['index_page_indices'],
                        page_table=None, out=replacement_out)
                    self.assertIsNot(target.plan._state, old_state)
                    self.assertEqual(target.plan._state.index.candidate_partition_blocks, 2048)
                    check(tuple(t.clone() for t in target.run()),
                          replacement_iq, replacement_keys)

    def test_lean192_standalone_ninth_partition_live_graph(self):
        """The Index ABI supports nine partitions beyond the attention API cap."""
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("192-thread Index requires SM103")
        from mbsa.backends.optimized.src.sm100.router import indexer_local_top16
        from tests.reference import MaxwellBlockSparseAttentionReference

        kn, qn = 1056768, 8192
        iq = torch.zeros((qn, 4, 128), device='cuda', dtype=torch.bfloat16)
        ik = torch.zeros((kn, 1, 128), device='cuda', dtype=torch.bfloat16)
        values = (torch.arange(kn // 128, device='cuda') % 37).to(ik.dtype)
        # Once queries become nonzero, the ninth partition must contribute to
        # the global winners rather than merely publishing unused partials.
        values[-64:] += 64
        ik[:, 0, 0] = values.repeat_interleave(128)
        routes = [torch.full((4, 32, qn), -99, device='cuda', dtype=torch.int32)
                  for _ in range(2)]
        oracle = MaxwellBlockSparseAttentionReference(index_head_dim=128, head_dim=128)

        def run(which):
            return indexer_local_top16(iq, ik, partition_blocks=1024,
                logical_k_tokens=kn, use_page_table=False,
                out_compact_ids=routes[which], _lean192_early=bool(which))

        def check_reference(actual):
            for row in (0, 127, 128, qn - 1):
                positions = torch.tensor([[kn - qn + row]], device='cuda')
                allowed = oracle._allowed_keys(query_positions=positions,
                    key_len=kn, attention_mask=None)
                expected = oracle._select_blocks(iq[row:row+1][None].float(),
                    ik[None].float(), positions, allowed)
                torch.testing.assert_close(actual[:, 16:, row],
                    expected[0, 0].to(torch.int32), rtol=0, atol=0)

        regular = run(0)
        lean = run(1)
        self.assertEqual(lean[0].shape[1], 9)
        for left, right in zip(regular, lean):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(routes[0], routes[1], rtol=0, atol=0)
        tied = routes[1][:, 16:].clone()
        wanted = torch.arange(8, 24, device='cuda', dtype=torch.int32).expand(4, 16)
        torch.testing.assert_close(tied[:, :, -1], wanted, rtol=0, atol=0)
        check_reference(routes[1])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(1)
        for changed in ('query', 'key'):
            iq[:, :, 0] = 1
            if changed == 'key':
                ik.neg_()
            routes[1][:, 16:].fill_(-99)
            captured[0].fill_(torch.nan)
            captured[1].fill_(-99)
            self.assertTrue(bool((routes[1][:, 16:] == -99).all()))
            graph.replay()
            live = routes[1].clone()
            self.assertFalse(torch.equal(live[:, 16:], tied))
            check_reference(live)
            if changed == 'query':
                self.assertTrue(bool((live[:, 16:, -1] >= 8192).all()))
            else:
                self.assertTrue(bool((live[:, 16:, -1] < 8192).all()))
            refreshed = run(0)
            for actual, expected in zip(captured, refreshed):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(live, routes[0], rtol=0, atol=0)
            routes[1][:, 16:].fill_(-99)
            graph.replay()
            torch.testing.assert_close(routes[1], live, rtol=0, atol=0)

    def test_canonical_4d_index_lean192_live_dispatch(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("192-thread Index requires SM103")
        for kn in (8192, 262144, 524288, 786432, 1048576):
            with self.subTest(kv=kn):
                iq, ik, q, k, v = self.inputs(8192, kn)
                pages = ik.view(-1, 1, 128, 128)
                dense = prepare_cutedsl_pipeline(iq, ik, q, k, v,
                    backend='optimized', partial_dtype=torch.bfloat16)
                paged = prepare_cutedsl_pipeline(iq, pages, q, k, v,
                    backend='optimized', partial_dtype=torch.bfloat16)
                p3, p4 = dense.plan._state.index, paged.plan._state.index
                for index in (p3, p4):
                    self.assertTrue(index._lean192_early_router)
                    self.assertFalse(index.cutedsl_use_page_table)
                    self.assertEqual(index.candidate_partition_blocks, 1024)
                    self.assertEqual(index.cutedsl_index_k.data_ptr(), ik.data_ptr())
                    self.assertEqual(index.cutedsl_index_k.stride(), ik.stride())
                self.assertEqual(p3._local_top16_key, p4._local_top16_key)
                self.assertIs(p3._local_top16_compiled, p4._local_top16_compiled)
                first = tuple(t.clone() for t in dense.run())
                second = tuple(t.clone() for t in paged.run())
                self.compare(second, first)
                torch.testing.assert_close(second[2], first[2], rtol=0, atol=0)
                if kn > 131072:
                    torch.testing.assert_close(p3.cutedsl_partition_scores,
                        p4.cutedsl_partition_scores, rtol=0, atol=0)
                    torch.testing.assert_close(p3.cutedsl_partition_ids,
                        p4.cutedsl_partition_ids, rtol=0, atol=0)

                def check_rows(actual, rows):
                    for row in rows:
                        expected = reference_pipeline(iq[row:row+1].float(), ik.float(),
                            q[row:row+1], k, v, query_offset=kn-8192+row)
                        self.compare((actual[0][row:row+1], actual[1][row:row+1],
                                      actual[2][:, row:row+1]), expected)

                check_rows(second, (0, 127, 128, 8191))
                paged.capture_mega_graph()
                attention = paged.plan._state.attention

                def replay_poisoned():
                    attention.out.fill_(torch.nan)
                    attention.out_lse.fill_(torch.nan)
                    attention.tile_plan.dynamic_route.fill_(-123)
                    self.assertFalse(bool(torch.isfinite(attention.out).all()))
                    result = tuple(t.clone() for t in paged.run_mega())
                    torch.cuda.synchronize()
                    return result

                replay = replay_poisoned()
                for actual, expected in zip(replay, second):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                # The canonical view must read the caller's live allocation,
                # not a preparation-time clone of either Index operand.
                for operand in (iq, pages):
                    with self.subTest(changed_index_shape=tuple(operand.shape)):
                        operand.neg_()
                        changed = replay_poisoned()
                        self.assertFalse(torch.equal(changed[2], second[2]))
                        check_rows(changed, (8191,))
                        operand.neg_()
                restored = replay_poisoned()
                for actual, expected in zip(restored, second):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_canonical_4d_index_selection_fallbacks_and_rebind(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("192-thread Index requires SM103")
        iq, ik, q, k, v = self.inputs(8192, 8192)
        pages = ik.view(-1, 1, 128, 128)
        plan = prepare_cutedsl_pipeline(iq, pages, q, k, v,
            backend='optimized', partial_dtype=torch.bfloat16)
        old = plan.plan._state
        baseline = tuple(t.clone() for t in plan.run())
        with self.assertRaises(ValueError):
            plan.rebind(iq, pages, q, k, v,
                        index_page_indices=old.bindings['index_page_indices'],
                        page_table=None, out=q)
        self.assertIs(plan.plan._state, old)
        self.compare(plan.run(), baseline)
        # Rebind retains the existing mutable-map/lifetime contract. It may
        # select the table-backed fallback rather than the initial lean path.
        stream, current = torch.cuda.Stream(), torch.cuda.current_stream()
        replacement_iq, replacement_ik = iq.roll(1, -1), pages.roll(2, -1)
        replacement_map = old.bindings['index_page_indices'].clone()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            plan.rebind(replacement_iq, replacement_ik, q, k, v,
                        index_page_indices=replacement_map, page_table=None,
                        out=torch.empty_like(q))
            rebound = tuple(t.clone() for t in plan.run())
        current.wait_stream(stream)
        self.assertIsNot(plan.plan._state, old)
        reference = reference_pipeline(replacement_iq[-1:].float(),
            replacement_ik.view(-1, 1, 128).float(), q[-1:], k, v,
            query_offset=8191)
        self.compare((rebound[0][-1:], rebound[1][-1:], rebound[2][:, -1:]), reference)
        explicit = prepare_cutedsl_pipeline(iq, pages, q, k, v,
            backend='optimized', partial_dtype=torch.bfloat16,
            index_page_indices=torch.arange(
                64, device='cuda', dtype=torch.int32))
        self.assertFalse(explicit.plan._state.index._lean192_early_router)
        self.assertTrue(explicit.plan._state.index.cutedsl_use_page_table)
        padded = torch.cat((pages, pages[:1]), dim=0)
        extra = prepare_cutedsl_pipeline(iq, padded, q, k, v,
            backend='optimized', partial_dtype=torch.bfloat16)
        self.assertFalse(extra.plan._state.index._lean192_early_router)
        strided = torch.empty((64, 1, 128, 144), device='cuda',
                              dtype=torch.bfloat16)[..., :128]
        with self.assertRaises(ValueError):
            prepare_cutedsl_pipeline(iq, strided, q, k, v,
                backend='optimized', partial_dtype=torch.bfloat16)

    def test_prepared_index_operands_rebind_stream_and_live_capture(self):
        import gc
        lengths = ((17, 4113),)
        data, operands, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
        # Keep the conversion allocations in the same binding ownership test.
        inputs = (*operands[:2], *(t.to(torch.float8_e4m3fn) for t in operands[2:]))
        changed = (inputs[0].roll(1, dims=-1), inputs[1].roll(2, dims=-1), inputs[2].clone(),
                   inputs[3].clone(), (inputs[4].to(torch.bfloat16) + 3).to(torch.float8_e4m3fn))
        expected = [backend_support.BackendLifecycle.reference(
            self, data, (*values[:2], *(t.to(torch.bfloat16) for t in values[2:])), lengths)
                    for values in (inputs, changed)]
        plan = prepare_cutedsl_pipeline(*inputs, backend='optimized', **kwargs)
        old = plan.plan._state
        bundle = old.index._local_top16_args
        self.assertIsNotNone(bundle)
        self.compare(plan.run(), expected[0])
        self.assertIs(old.index._local_top16_args, bundle)
        # Cold resolution is private to the candidate state. A compiler error
        # during replacement must preserve the published binding and output.
        compiled = old.index._local_top16_compiled
        self.assertIsNotNone(compiled)
        with mock.patch('mbsa.backends.optimized.index.indexer_local_top16',
                        side_effect=RuntimeError('injected Index compilation failure')):
            with self.assertRaisesRegex(RuntimeError, 'Index compilation failure'):
                plan.rebind(*changed, index_page_indices=data['pages'],
                            page_table=data['table'], out=torch.empty_like(operands[2]))
        self.assertIs(plan.plan._state, old)
        self.assertIs(old.index._local_top16_compiled, compiled)
        self.compare(plan.run(), expected[0])
        with self.assertRaises(ValueError):
            plan.rebind(*changed, index_page_indices=data['pages'],
                        page_table=data['table'], out=inputs[2])
        self.assertIs(plan.plan._state, old)
        self.assertIs(old.index._local_top16_args, bundle)
        stream, current = torch.cuda.Stream(), torch.cuda.current_stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            first = tuple(t.clone() for t in plan.run())
            plan.rebind(*changed, index_page_indices=data['pages'].clone(),
                        page_table=data['table'].clone(), out=torch.empty_like(operands[2]))
            self.assertIsNot(plan.plan._state.index._local_top16_args, bundle)
            second = tuple(t.clone() for t in plan.run())
        current.wait_stream(stream)
        self.compare(first, expected[0])
        self.compare(second, expected[1])
        # A uniform score scale preserves TopK. Perturb Q and K separately
        # and prove that either stale cached pointer fails the route oracle.
        rebound = plan.plan._state.index
        rebound_bundle = rebound._local_top16_args
        for operand in (0, 1):
            with self.subTest(stale_index_operand=operand):
                stale = list(rebound_bundle)
                stale[operand] = bundle[operand]
                rebound._local_top16_args = tuple(stale)
                try:
                    observed = tuple(t.clone() for t in plan.run())
                    self.assertFalse(torch.equal(
                        observed[2].sort(-1).values,
                        expected[1][2].sort(-1).values))
                finally:
                    rebound._local_top16_args = rebound_bundle
        self.compare(plan.run(), expected[1])
        plan.capture_mega_graph()
        plan.plan.attention.out.fill_(torch.nan)
        plan.plan.attention.out_lse.fill_(torch.nan)
        plan.run_mega()
        live = tuple(t.clone() for t in (plan.plan.attention.out,
            plan.plan.attention.out_lse, plan.plan.attention.tile_plan.dynamic_route))
        self.compare(live, expected[1])
        with torch.cuda.stream(stream):
            torch.cuda._sleep(10_000_000)
            final = tuple(t.clone() for t in plan.run())
        del plan, old, bundle, compiled, rebound, rebound_bundle, stale, inputs, changed, operands, data, kwargs
        gc.collect()
        pressure = torch.empty(1 << 20, device='cuda').fill_(123)
        current.wait_stream(stream)
        self.compare(final, expected[1])
        self.assertEqual(pressure.numel(), 1 << 20)

    def test_storage_offsets_and_transactional_alignment_rejection(self):
        lengths = ((17, 4096),)
        data, operands, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
        plan = prepare_cutedsl_pipeline(*operands, backend='optimized', **kwargs)
        saved = tuple(t.clone() for t in plan.run())
        state = plan.plan._state

        def offset_copy(tensor, byte_offset):
            offset = byte_offset // tensor.element_size()
            extent = 1 + sum((n - 1) * s for n, s in
                             zip(tensor.shape, tensor.stride()))
            result = torch.empty(extent + offset, dtype=tensor.dtype,
                                 device=tensor.device).as_strided(
                                     tensor.shape, tensor.stride(), offset)
            result.copy_(tensor)
            self.assertEqual(result.data_ptr() % 16, byte_offset % 16)
            return result

        # Each operand retains the prepared metadata, but its base violates
        # the direct consumer's alignment. Reject before publishing or launch.
        for i in range(5):
            with self.subTest(operand=i):
                changed = list(operands)
                changed[i] = offset_copy(operands[i], 2)
                with self.assertRaisesRegex(ValueError, 'aligned'):
                    prepare_cutedsl_pipeline(*changed, backend='optimized', **kwargs)
                with self.assertRaisesRegex(ValueError, 'aligned'):
                    plan.rebind(*changed, index_page_indices=data['pages'],
                                page_table=data['table'], out=torch.empty_like(operands[2]))
                self.assertIs(plan.plan._state, state)
        bad_out = offset_copy(torch.zeros_like(operands[2]), 2)
        with self.assertRaisesRegex(ValueError, 'aligned'):
            plan.rebind(*operands, index_page_indices=data['pages'],
                        page_table=data['table'], out=bad_out)
        self.assertIs(plan.plan._state, state)
        self.assertTrue(torch.all(bad_out == 0).item())
        for actual, previous in zip(plan.run(), saved):
            torch.testing.assert_close(actual, previous, atol=0, rtol=0)
        aligned = tuple(offset_copy(t, 16) for t in operands)
        aligned_out = offset_copy(torch.zeros_like(operands[2]), 16)
        plan.rebind(*aligned, index_page_indices=data['pages'],
                    page_table=data['table'], out=aligned_out)
        self.assertEqual(plan.run()[0].data_ptr(), aligned_out.data_ptr())
        for actual, previous in zip(plan.run(), saved):
            torch.testing.assert_close(actual, previous, atol=0, rtol=0)
        self.compare(plan.run(), backend_support.BackendLifecycle.reference(
            self, data, aligned, lengths))

    def test_uniform_fixed_scheduler_lengths_and_overlap(self):
        # Public policy stays static across the measured range and old cutoffs.
        for kn in (8192, 16384, 24575, 24576, 32768, 32769, 40960, 49152,
                   49153, 65536, 131072, 262144, 524288, 786432, 1048576):
            with self.subTest(kv=kn):
                plan = prepare_cutedsl_pipeline(*self.inputs(8192, kn),
                                               backend='optimized')
                self.assertIs(plan.plan.attention.fixed_use_clc_scheduler, False)
        operands = self.inputs(8192, 32768)
        plan = prepare_cutedsl_pipeline(*operands, backend='optimized',
                                       overlap_fixed_score=False)
        self.assertIs(plan.plan.attention.fixed_use_clc_scheduler, False)
        self.assertIsNone(plan.plan._state.fixed_stream)
        for lengths in (((8192, 32768),), ((8064, 32768),),
                        ((8192, 32768), (17, 4113))):
            with self.subTest(paged_requests=lengths):
                _, operands, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
                plan = prepare_cutedsl_pipeline(*operands, backend='optimized',
                                               overlap_fixed_score=True, **kwargs)
                self.assertIs(plan.plan.attention.fixed_use_clc_scheduler, False)

    def test_mixed_paged_tail_live_graph_and_transactional_rebind(self):
        lengths = ((17, 4113), (23, 2195), (41, 127))
        data, operands, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
        expected = backend_support.BackendLifecycle.reference(self, data, operands, lengths)
        plan = prepare_cutedsl_pipeline(*operands, backend="optimized", **kwargs)
        fixed = plan.plan.attention.fixed_out
        self.assertEqual(fixed.shape, operands[2].shape)
        self.assertTrue(fixed.is_contiguous())
        self.assertNotEqual(fixed.data_ptr(), plan.plan.attention.out.data_ptr())
        self.assertEqual(plan.backend, "optimized")
        actual = plan.run()
        self.compare(actual, expected)
        self.assertEqual(actual[0].dtype, torch.bfloat16)
        self.assertEqual(actual[1].dtype, torch.float32)
        self.assertEqual(actual[2].dtype, torch.int32)
        saved = tuple(t.clone() for t in actual)
        bad = data['table'].clone()
        bad[0, 0] = -1
        replacement = torch.full_like(operands[2], 123)
        state = plan.plan._state
        with self.assertRaises(ValueError):
            plan.rebind(*operands, index_page_indices=data['pages'],
                        page_table=bad, out=replacement)
        self.assertIs(plan.plan._state, state)
        self.assertTrue(torch.all(replacement == 123).item())
        for a, b in zip(plan.run(), saved):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        plan.rebind(*operands, index_page_indices=data['pages'],
                    page_table=data['table'].clone(), out=replacement)
        self.assertIs(plan.plan.attention.fixed_out, fixed)
        self.assertNotEqual(fixed.data_ptr(), replacement.data_ptr())
        self.compare(plan.run(), expected)
        self.assertEqual(plan.run()[0].data_ptr(), replacement.data_ptr())
        plan.capture_mega_graph()
        # A no-op replay must fail against poisoned live outputs.
        def check_live(replay):
            plan.plan.attention.out.fill_(torch.nan)
            plan.plan.attention.out_lse.fill_(torch.nan)
            replay()
            return tuple(t.clone() for t in (plan.plan.attention.out,
                          plan.plan.attention.out_lse, plan._owner.dynamic_route))
        with self.assertRaises(AssertionError):
            self.compare(check_live(lambda: None), expected)
        observed = check_live(plan.run_mega)
        self.compare(observed, expected)
        operands[4].add_(3)
        changed = backend_support.BackendLifecycle.reference(self, data, operands, lengths)
        observed = check_live(plan.run_mega)
        self.compare(observed, changed)
        with self.assertRaises(AssertionError):
            self.compare(observed, expected)
        # Save immediate graph output before any eager stage overwrite.
        for a, b in zip(observed, plan.run()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        with self.assertRaisesRegex(RuntimeError, "captured"):
            plan.rebind(*operands, index_page_indices=data['pages'],
                        page_table=data['table'], out=replacement)

    def test_two_backends_independent_streams_and_external_capture(self):
        lengths = ((19, 4096),)
        data, operands, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
        expected = backend_support.BackendLifecycle.reference(self, data, operands, lengths)
        plans, streams = [], [torch.cuda.Stream() for _ in range(2)]
        current = torch.cuda.current_stream()
        for backend, stream in zip(('default', 'optimized'), streams):
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                plan = prepare_cutedsl_pipeline(*operands, backend=backend, **kwargs)
                plan.run()
            plans.append(plan)
        for stream in streams:
            current.wait_stream(stream)
        self.assertEqual(len({p.plan.attention.out.data_ptr() for p in plans}), 2)
        self.assertEqual(len({p.plan.candidate_storage.data_ptr() for p in plans}), 2)
        self.assertEqual(len({p._owner.dynamic_route.data_ptr() for p in plans}), 2)
        observed = []
        for plan, stream in zip(plans, streams):
            with torch.cuda.stream(stream):
                observed.append(tuple(t.clone() for t in plan.run()))
        for stream in streams:
            current.wait_stream(stream)
        for result in observed:
            self.compare(result, expected)

        og, stream = plans[1], streams[1]
        graph = torch.cuda.CUDAGraph()
        stream.wait_stream(current)
        with torch.cuda.graph(graph, stream=stream):
            og.plan.run()
        current.wait_stream(stream)
        with self.assertRaisesRegex(RuntimeError, 'captured'):
            og.rebind(*operands, index_page_indices=data['pages'],
                      page_table=data['table'], out=torch.empty_like(operands[2]))
        og.plan.attention.out.fill_(torch.nan)
        og.plan.attention.out_lse.fill_(torch.nan)
        graph.replay()
        live = tuple(t.clone() for t in (og.plan.attention.out,
                                        og.plan.attention.out_lse, og._owner.dynamic_route))
        self.compare(live, expected)
        # Inspect this live output before any other plan can overwrite its state.
        for a, b in zip(live, og.run()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_serving_layer_reuse_and_single_mixed_bsa_launch(self):
        from mbsa.serving import Prefill
        lengths = ((17, 4113), (23, 2195), (41, 127))
        data, operands, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
        iq, ik, q, k, v = operands
        expected = backend_support.BackendLifecycle.reference(self, data, operands, lengths)
        step = Prefill(q.device, backend='optimized').stage(lengths, data['table'], data['pages'])
        first = step.run(q, k, v, index_q=iq, index_k_cache=ik, out=torch.empty_like(q))
        self.compare((first, step.plan.attention.out_lse, step.plan.dynamic_route), expected)
        pipeline = step.plan
        fixed = pipeline.plan.attention.fixed_out
        self.assertEqual(fixed.shape, q.shape)
        self.assertTrue(fixed.is_contiguous())
        self.assertNotEqual(fixed.data_ptr(), first.data_ptr())
        pointers = tuple(t.data_ptr() for t in pipeline.plan.attention.dynamic_workspace)
        from mbsa.backends.optimized.bsa.cute_dsl.sparse import bsa_attn_sm100_blk128 as bsa
        cache = bsa._sm100_blk128_compile_cache.cache
        wrapped = {key: mock.Mock(wraps=value) for key, value in cache.items()}
        with mock.patch.dict(cache, wrapped):
            second = step.run(q, k, v, index_q=iq, index_k_cache=ik,
                              out=torch.empty_like(q))
            self.compare((second, step.plan.attention.out_lse, step.plan.dynamic_route), expected)
            self.assertEqual(sum(value.call_count for value in wrapped.values()), 1)
        self.assertIs(pipeline.plan.attention.fixed_out, fixed)
        self.assertNotEqual(fixed.data_ptr(), second.data_ptr())
        self.assertEqual(tuple(t.data_ptr() for t in pipeline.plan.attention.dynamic_workspace), pointers)

    def test_contiguous_fixed_fused_output_rebind(self):
        for kn in (127, 4113):
            with self.subTest(kv=kn):
                operands = self.inputs(17, kn)
                plan = prepare_cutedsl_pipeline(*operands, backend='optimized')
                fixed = plan.plan.attention.fixed_out
                self.assertEqual(fixed.shape, operands[2].shape)
                self.assertEqual(fixed.stride(), (4096, 128, 1))
                self.assertNotEqual(fixed.data_ptr(), plan.plan.attention.out.data_ptr())
                expected = reference_pipeline(*operands)
                saved = tuple(t.clone() for t in plan.run())
                self.compare(saved, expected)
                replacement = torch.full_like(operands[2], 123)
                pages = plan.plan.index_page_indices
                with self.assertRaises(ValueError):
                    plan.rebind(*operands, index_page_indices=pages,
                                page_table=None, out=operands[2])
                for actual, previous in zip(plan.run(), saved):
                    torch.testing.assert_close(actual, previous, atol=0, rtol=0)
                plan.rebind(*operands, index_page_indices=pages,
                            page_table=None, out=replacement)
                self.assertIs(plan.plan.attention.fixed_out, fixed)
                self.assertNotEqual(fixed.data_ptr(), replacement.data_ptr())
                plan.capture_mega_graph()
                operands[4].add_(3)
                expected = reference_pipeline(*operands)
                replacement.fill_(torch.nan)
                plan.plan.attention.out_lse.fill_(torch.nan)
                live = tuple(t.clone() for t in plan.run_mega())
                self.compare(live, expected)
                for actual, observed in zip(plan.run(), live):
                    torch.testing.assert_close(actual, observed, rtol=0, atol=0)

    def test_index_and_attention_formats_with_live_graph(self):
        inputs = self.inputs(17, 4096)
        for index_dtype, attention_dtype, partial_dtype in (
            (torch.float16, torch.bfloat16, torch.bfloat16),
            (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.bfloat16),
            (torch.bfloat16, torch.bfloat16, torch.float8_e4m3fn),
        ):
            with self.subTest(index_dtype=index_dtype, attention_dtype=attention_dtype,
                              partial_dtype=partial_dtype):
                operands = tuple(t.to(index_dtype if i < 2 else attention_dtype)
                                 for i, t in enumerate(inputs))
                reference = reference_pipeline(*(t.to(torch.bfloat16) if i >= 2 else t
                                                  for i, t in enumerate(operands)))
                plan = prepare_cutedsl_pipeline(*operands, backend='optimized',
                                               partial_dtype=partial_dtype)
                self.compare(plan.run(), reference)
                plan.capture_mega_graph()
                plan.plan.attention.out.fill_(torch.nan)
                plan.plan.attention.out_lse.fill_(torch.nan)
                live = tuple(t.clone() for t in plan.run_mega())
                self.compare(live, reference)
                for a, b in zip(live, plan.run()):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_mx_scales_public_rebind_and_serving(self):
        from mbsa.index_formats import (quantize_q_mxfp8, quantize_index_k_mxfp6,
                                       dequantize_index_q_mxfp8, dequantize_index_k_mxfp6)
        from mbsa.serving import Prefill
        lengths = ((17, 4113),)
        data, inputs, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
        codes, scales = quantize_q_mxfp8(inputs[0])
        iq = codes[:, :4].view(torch.float8_e4m3fn)
        ik = quantize_index_k_mxfp6(inputs[1])
        operands = iq, ik, *inputs[2:]

        def reference(bound_scales):
            decoded = (dequantize_index_q_mxfp8(iq, bound_scales),
                       dequantize_index_k_mxfp6(ik), *inputs[2:])
            return backend_support.BackendLifecycle.reference(self, data, decoded, lengths)

        plan = prepare_cutedsl_pipeline(*operands, backend='optimized',
                                       index_q_scales=scales, **kwargs)
        saved = tuple(t.clone() for t in plan.run())
        self.compare(saved, reference(scales))
        out = torch.full_like(inputs[2], 123)
        with self.assertRaisesRegex(ValueError, 'index_q_scales'):
            plan.rebind(*operands, index_page_indices=data['pages'],
                        page_table=data['table'], out=out)
        for actual, previous in zip(plan.run(), saved):
            torch.testing.assert_close(actual, previous, atol=0, rtol=0)
        changed = scales.clone().bitwise_xor_(1 << 8)
        plan.rebind(*operands, index_q_scales=changed, index_page_indices=data['pages'],
                    page_table=data['table'], out=out)
        self.assertIs(plan.plan.index_q_scales, changed)
        self.assertIs(plan.plan.mxfp6_router.tensors[1], changed)
        self.compare(plan.run(), reference(changed))
        step = Prefill(inputs[2].device, backend='optimized').stage(
            lengths, data['table'], data['pages'].view(1, -1))
        for bound_scales in (scales, changed):
            step.run(inputs[2], *inputs[3:], index_q=iq,
                     index_k_cache=ik.view(-1, 128, 100), out=out,
                     index_q_scales=bound_scales)
            self.assertIs(step.plan.index_q_scales, bound_scales)
            self.compare((out, step.plan.attention.out_lse, step.plan.dynamic_route),
                         reference(bound_scales))

    def test_short_kv_empty_dynamic_work_and_warmup_reuse(self):
        from mbsa.serving import Prefill, warmup, compiled_kernel_count
        for kn in (127, 2048, 4096):
            with self.subTest(kv=kn):
                operands = self.inputs(17, kn)
                expected = reference_pipeline(*operands)
                plan = prepare_cutedsl_pipeline(*operands, backend='optimized')
                self.compare(plan.run(), expected)
                plan.capture_mega_graph()
                plan.plan.attention.out.fill_(torch.nan)
                plan.plan.attention.out_lse.fill_(torch.nan)
                self.compare(tuple(t.clone() for t in plan.run_mega()), expected)
        prefill = Prefill(torch.device('cuda'), backend='optimized')
        warmup(prefill, ((17, 4113),), index_format='bf16')
        compiled = compiled_kernel_count()
        from mbsa.backends.optimized.msa.cute import interface
        from mbsa.backends.optimized.msa.cute.src.sm100.fwd import combine
        from mbsa.backends.optimized.bsa.cute_dsl.sparse import bsa_attn_sm100_blk128 as bsa
        caches = (interface._compile_cache, combine._combine_compile_cache,
                  bsa._sm100_blk128_compile_cache.cache)
        selected = tuple(dict(cache) for cache in caches)
        self.assertTrue(all(selected), 'selected optimized CuTe caches must be nonempty')
        warmup(prefill, ((17, 4113),), index_format='bf16')
        self.assertEqual(compiled_kernel_count(), compiled)
        for cache, previous in zip(caches, selected):
            self.assertEqual(cache.keys(), previous.keys())
            for key, entry in previous.items():
                self.assertIs(cache[key], entry)

    def test_independent_optimized_plans_and_eager_drop_lifetime(self):
        import gc
        operands = self.inputs(17, 4113)
        changed = (*operands[:4], operands[4].clone().add_(3))
        expected = (reference_pipeline(*operands), reference_pipeline(*changed))
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        current = torch.cuda.current_stream()
        plans, live = [], []
        for inputs, stream in zip((operands, changed), streams):
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                plan = prepare_cutedsl_pipeline(*inputs, backend='optimized')
                plan.run()
            plans.append(plan)
        for stream in streams:
            current.wait_stream(stream)
        self.assertNotEqual(plans[0].plan.attention.out.data_ptr(),
                            plans[1].plan.attention.out.data_ptr())
        # Queue warmed work behind a delay, drop the plan, then reuse allocator
        # storage on the preparation stream before inspecting the live outputs.
        # Caller-owned outputs alone do not keep CSR/partial workspaces alive.
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                torch.cuda._sleep(10_000_000)
                live.append(plans[i].run())
        del plan
        plans.clear()
        gc.collect()
        pressure = [torch.empty(1 << 20, dtype=torch.bfloat16, device='cuda').fill_(123)
                    for _ in range(8)]
        for stream in streams:
            current.wait_stream(stream)
        for actual, oracle in zip(live, expected):
            self.compare(actual, oracle)
        self.assertEqual(len(pressure), 8)

    def test_explicit_sm90_selection_rejects_before_dispatch(self):
        operands = self.inputs(1, 128)
        with mock.patch('torch.cuda.get_device_capability', return_value=(9, 0)):
            with self.assertRaisesRegex(ValueError, 'optimized.*SM100.*SM103'):
                prepare_cutedsl_pipeline(*operands, backend='optimized')

    def test_eager_rebind_conversion_and_drop_lifetime(self):
        import gc
        from mbsa.backends.optimized.runtime import _backing_tensors
        lengths = ((17, 4113), (9, 4225))
        data, original, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
        operands = (*original[:2], *(t.to(torch.float8_e4m3fn) for t in original[2:]))
        changed = (*[t.clone() for t in operands[:4]],
                   (operands[4].to(torch.bfloat16) + 3).to(torch.float8_e4m3fn))
        expected = [backend_support.BackendLifecycle.reference(
            self, data, (*inputs[:2], *(t.to(torch.bfloat16) for t in inputs[2:])), lengths)
                    for inputs in (operands, changed)]
        plan = prepare_cutedsl_pipeline(*operands, backend='optimized', **kwargs)

        def check_backings(plan):
            state = plan.plan._state
            fresh = _backing_tensors(state.bindings, state.conversions,
                                     state.attention, state.index)
            # CSR may replace tuple/view objects while reusing their storage.
            self.assertEqual({t.untyped_storage()._cdata for t in fresh},
                             {t.untyped_storage()._cdata for t in state.backing_tensors})
            self.assertEqual(len({id(t) for t in state.backing_tensors}),
                             len(state.backing_tensors))
            for destination, source in state.conversions:
                self.assertTrue(any(t is source for t in state.backing_tensors))
                self.assertTrue(any(t is destination for t in state.backing_tensors))

        check_backings(plan)
        self.compare(plan.run(), expected[0])
        check_backings(plan)
        old_state = plan.plan._state
        bad_table = data['table'].clone()
        bad_table[0, 0] = -1
        with self.assertRaises(ValueError):
            plan.rebind(*changed, index_page_indices=data['pages'],
                        page_table=bad_table, out=torch.empty_like(original[2]))
        self.assertIs(plan.plan._state, old_state)
        del old_state, bad_table
        stream, current = torch.cuda.Stream(), torch.cuda.current_stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(10_000_000)
            first = tuple(t.clone() for t in plan.run())
            plan.rebind(*changed, index_page_indices=data['pages'].clone(),
                        page_table=data['table'].clone(), out=torch.empty_like(original[2]))
            check_backings(plan)
            second = tuple(t.clone() for t in plan.run())
            check_backings(plan)
        del plan, operands, changed, original, kwargs, data
        gc.collect()
        pressure = [torch.empty(1 << 20, dtype=torch.bfloat16, device='cuda').fill_(123)
                    for _ in range(8)]
        current.wait_stream(stream)
        for actual, oracle in zip((first, second), expected):
            self.compare(actual, oracle)
        self.assertEqual(len(pressure), 8)

    def test_packed_large_ragged_rebind_distinct_inputs_and_live_pages(self):
        lengths = ((512, 8192), (1024, 12289), (65, 4225), (193, 8321))
        data, inputs, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
        plan = prepare_cutedsl_pipeline(*inputs, backend='optimized', **kwargs)
        original = plan.run()
        self.compare(original, backend_support.BackendLifecycle.reference(self, data, inputs, lengths))
        previous_out, previous_value = original[0], original[0].clone()
        rebound = tuple(torch.empty_strided(t.shape, t.stride(), device=t.device,
                                            dtype=t.dtype).copy_(t).add_(.125)
                        for t in inputs)
        self.assertTrue(all(a.data_ptr() != b.data_ptr() for a, b in zip(inputs, rebound)))
        pages, table = data['pages'].roll(1), data['table'].clone()
        offset = 0
        for row, (_, kn) in enumerate(lengths):
            count = (kn + 127)//128
            table[row, :count].copy_(pages[offset:offset+count])
            offset += count
        mapped = dict(data, pages=pages, table=table)
        out = torch.full_like(rebound[2], 123)
        plan.rebind(*rebound, index_page_indices=pages, page_table=table, out=out)
        for old in inputs:
            old.fill_(torch.nan)
        expected = backend_support.BackendLifecycle.reference(self, mapped, rebound, lengths)
        self.compare(plan.run(), expected)
        torch.testing.assert_close(previous_out, previous_value, rtol=0, atol=0)
        plan.capture_mega_graph()
        # Captured addresses stay fixed; live Index and physical KV maps change.
        pages.copy_(pages.roll(1))
        offset = 0
        for row, (_, kn) in enumerate(lengths):
            count = (kn + 127)//128
            table[row, :count].copy_(pages[offset:offset+count])
            offset += count
        rebound[0].copy_(rebound[0].roll(1, dims=-1))
        rebound[2].mul_(.75)
        rebound[4].add_(3)
        expected = backend_support.BackendLifecycle.reference(self, mapped, rebound, lengths)
        out.fill_(torch.nan)
        plan.plan.attention.out_lse.fill_(torch.nan)
        live = tuple(t.clone() for t in plan.run_mega())
        self.compare(live, expected)
        for actual, observed in zip(plan.run(), live):
            torch.testing.assert_close(actual, observed, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class SparsePagedTail(unittest.TestCase):
    def test_masked_kv_conversion_retains_dequantization_roles(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("SM100/SM103 required")
        from mbsa.backends.optimized.msa.cute import interface
        from mbsa.backends.optimized.msa.cute.sparse_index_utils import build_k2q_csr

        torch.manual_seed(619)
        q = torch.randn((17, 32, 128), device='cuda', dtype=torch.bfloat16) * .25
        k = (torch.randn((129, 4, 128), device='cuda') * .25).to(torch.float8_e4m3fn)
        v = torch.randn((129, 4, 128), device='cuda').to(torch.float8_e4m3fn)
        cu_q = torch.tensor([0, 17], device='cuda', dtype=torch.int32)
        cu_k = torch.tensor([0, 129], device='cuda', dtype=torch.int32)
        routes = torch.full((4, 17, 16), -1, device='cuda', dtype=torch.int32)
        routes[:, :, 0], routes[:, :, 1] = 0, 1
        rp, qi, schedule = build_k2q_csr(
            routes, cu_q, cu_k, 128, total_k=129, max_seqlen_k=129,
            max_seqlen_q=17, qhead_per_kv=8, return_schedule=True)
        out = torch.empty_like(q)

        def run(kernel_cfg=None):
            return interface.sparse_atten_func(
                q, k, v, rp, qi, 16, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                max_seqlen_q=17, max_seqlen_k=129, schedule=schedule,
                full_score_tiles=False, out=out, partial_dtype=torch.bfloat16,
                kernel_cfg=kernel_cfg,
                return_softmax_lse=True)

        keys = k.float().repeat_interleave(8, dim=1)
        values = v.float().repeat_interleave(8, dim=1)
        scores = torch.einsum('qhd,khd->qhk', q.float(), keys) / math.sqrt(128)
        expected = (torch.einsum('qhk,khd->qhd', scores.softmax(-1), values),
                    torch.logsumexp(scores, dim=-1))

        def check(actual):
            torch.testing.assert_close(actual[0].float(), expected[0], rtol=.03, atol=.02)
            torch.testing.assert_close(actual[1], expected[1], rtol=.002, atol=.01)

        selector = interface.select_msa_cfg
        inner = interface._call_sparse_forward_sm100_csr_varlen
        with mock.patch.object(interface, 'select_msa_cfg', wraps=selector) as selected:
            with mock.patch.object(interface, '_call_sparse_forward_sm100_csr_varlen',
                                   wraps=inner) as tapped:
                check(run())
        direct = inspect.signature(inner).bind(*tapped.call_args.args,
                                              **tapped.call_args.kwargs)
        self.assertTrue(selected.call_args_list)
        for call in selected.call_args_list:
            self.assertTrue(call.kwargs['requires_kv_staging'])
            self.assertEqual(selector(**call.kwargs).num_wgs, 4)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        captured[0].fill_(torch.nan)
        captured[1].fill_(torch.nan)
        with self.assertRaises(AssertionError):
            check(captured)
        graph.replay()
        live = tuple(t.clone() for t in captured)
        check(live)
        for actual, eager in zip(live, run()):
            torch.testing.assert_close(actual, eager, rtol=0, atol=0)

        from mbsa.backends.optimized.msa.cute.src.sm100.fwd.config import (
            MSA_5WG_STATIC_MASKED, MSA_5WG_CLC_MASKED,
        )
        for cfg in (MSA_5WG_STATIC_MASKED, MSA_5WG_CLC_MASKED):
            with self.subTest(rejected_config=cfg.symbol_suffix):
                out.fill_(.125)
                unchanged = out.clone()
                with mock.patch.object(
                    interface, '_call_sparse_forward_sm100_csr_varlen',
                    side_effect=AssertionError('rejection must precede forward submission'),
                ) as forward:
                    with self.assertRaisesRegex(ValueError, 'dequantization roles'):
                        run(kernel_cfg=cfg)
                forward.assert_not_called()
                torch.testing.assert_close(out, unchanged, rtol=0, atol=0)
                direct.arguments['kernel_cfg'] = cfg
                partials = [direct.arguments[name] for name in ('O_partial', 'LSE_partial')]
                saved = [tensor.clone() for tensor in partials]
                with mock.patch.object(interface, '_compile_cache', {}), \
                     mock.patch.object(interface.cute, 'compile',
                                       side_effect=AssertionError('must reject before compile')) as compile_op, \
                     mock.patch('mbsa.backends.optimized.msa.cute.src.common.aot_cache.try_load_aot',
                                side_effect=AssertionError('must reject before cached launch')) as load_op:
                    with self.assertRaisesRegex(ValueError, 'dequantization roles'):
                        inner(*direct.args, **direct.kwargs)
                compile_op.assert_not_called()
                load_op.assert_not_called()
                for tensor, snapshot in zip(partials, saved):
                    torch.testing.assert_close(tensor.view(torch.uint8), snapshot.view(torch.uint8),
                                               rtol=0, atol=0)

    def test_masked_query_transport_selection_and_live_dispatch(self):
        self._check_query_transport_selection_and_live_dispatch()

    def test_automatic_gather4_static_and_clc_live_dispatch(self):
        for length, full in ((8191, False), (8192, True),
                             (32767, False), (32768, True), (32769, False)):
            with self.subTest(length=length, full=full):
                self._check_query_transport_selection_and_live_dispatch(
                    kv_length=length, full_score_tiles=full)

    def test_automatic_gather4_native_fp8_live_dispatch(self):
        for length, full in ((127, False), (32768, True), (32769, False)):
            with self.subTest(length=length, full=full):
                self._check_query_transport_selection_and_live_dispatch(
                    kv_length=length, full_score_tiles=full,
                    storage_dtype=torch.float8_e4m3fn, ratios=(1, 2, 4))

    def test_ragged_tma_output_descriptor_live_dispatch(self):
        from mbsa.backends.optimized.msa.cute.src.sm100.fwd.config import (
            OutputLayout, OutputStore, QLoad, Scheduler, WorkOrder,
            make_msa_variant,
        )
        cfg = make_msa_variant(
            q_load=QLoad.TMA, output_store=OutputStore.TMA,
            output_layout=OutputLayout.SHUFFLED, num_wgs=5,
            scheduler=Scheduler.STATIC, work_order=WorkOrder.NATIVE).cfg
        self.assertTrue(cfg.ragged_tma_store)
        if torch.cuda.get_device_capability() == (10, 0):
            self._check_query_transport_selection_and_live_dispatch(
                cfg, 17, ratios=(8,), expected_error='sm103')
            return
        for q_length in (17, 129):
            with self.subTest(q_length=q_length):
                self._check_query_transport_selection_and_live_dispatch(
                    cfg, q_length, ratios=(8,))

    def test_nvfp4_paged_gather4_descriptor_live_dispatch(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("SM100/SM103 required")
        from mbsa.backends.optimized.msa.cute import interface
        from mbsa.backends.optimized.msa.cute.quantize import swizzle_nvfp4_scale_to_128x4
        from mbsa.backends.optimized.msa.cute.sparse_index_utils import build_k2q_csr

        torch.manual_seed(177)
        pages, heads, block, dim, length, nq, ratio = 4, 4, 128, 128, 385, 17, 4
        shape = (pages, heads, block, dim)
        positions = torch.arange(math.prod(shape), device='cuda').reshape(shape)
        codes = ((positions // 7 + positions // 128) % 16).to(torch.uint8)
        packed_k = (codes[..., ::2] | (codes[..., 1::2] << 4)).contiguous()
        v_codes = (codes + 3) % 16
        packed_v = (v_codes[..., ::2] | (v_codes[..., 1::2] << 4)).contiguous()
        lookup = torch.tensor(
            [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
            device='cuda')
        rows, cols = pages * heads * block, dim // 16
        scale_values = torch.where(
            torch.arange(rows * cols, device='cuda').reshape(rows, cols) % 3 == 0,
            .125, .25)
        scale_bytes = scale_values.to(torch.float8_e4m3fn).view(torch.uint8)
        ks = swizzle_nvfp4_scale_to_128x4(scale_bytes, rows=rows, cols=cols)
        vs = swizzle_nvfp4_scale_to_128x4(scale_bytes, rows=rows, cols=cols)
        kg = torch.tensor([.5], device='cuda')
        vg = torch.tensor([2.], device='cuda')
        q = torch.randn((nq, heads * ratio, dim), device='cuda', dtype=torch.bfloat16) * .125
        cuq = torch.tensor([0, nq], device='cuda', dtype=torch.int32)
        cuk = torch.tensor([0, length], device='cuda', dtype=torch.int32)
        table = torch.tensor([[2, 0, 3, 1]], device='cuda', dtype=torch.int32)
        used = torch.tensor([length], device='cuda', dtype=torch.int32)
        routes = torch.arange(pages, device='cuda', dtype=torch.int32).expand(heads, nq, pages).contiguous()
        rp, qi, schedule = build_k2q_csr(
            routes, cuq, cuk, block, total_k=length, max_seqlen_k=length,
            max_seqlen_q=nq, qhead_per_kv=ratio, return_schedule=True)

        def run():
            return interface.sparse_atten_nvfp4_kv_func(
                q, packed_k, packed_v, ks, vs, kg, vg, rp, qi, pages,
                cu_seqlens_q=cuq, cu_seqlens_k=cuk, max_seqlen_q=nq,
                max_seqlen_k=length, page_table=table, seqused_k=used,
                schedule=schedule, partial_dtype=torch.bfloat16, return_softmax_lse=True)

        def oracle():
            # Dyadic scales make the represented BF16 K/V values exact. The
            # reference decodes nibbles and logical pages independently of
            # descriptor construction and the device scale-address helper.
            sc = scale_values.reshape(pages, heads, block, cols).repeat_interleave(16, -1)
            kd = (lookup[(packed_k[..., None] >> torch.tensor([0, 4], device='cuda'))
                         .reshape(shape).long() & 15] * sc * kg).to(torch.bfloat16)
            vd = (lookup[(packed_v[..., None] >> torch.tensor([0, 4], device='cuda'))
                         .reshape(shape).long() & 15] * sc).to(torch.bfloat16)
            keys = kd[table[0].long()].permute(0, 2, 1, 3).reshape(-1, heads, dim)[:int(used.item())]
            values = vd[table[0].long()].permute(0, 2, 1, 3).reshape(-1, heads, dim)[:int(used.item())]
            keys = keys.float().repeat_interleave(ratio, 1)
            values = values.float().repeat_interleave(ratio, 1)
            scores = torch.einsum('qhd,khd->qhk', q.float(), keys) / math.sqrt(dim)
            return (torch.einsum('qhk,khd->qhd', scores.softmax(-1), values) * vg,
                    torch.logsumexp(scores, -1))

        def check(actual):
            expected = oracle()
            # The same BF16 partial-attention reference contract used by the
            # ordinary storage tests applies to these exactly decoded inputs.
            torch.testing.assert_close(actual[0].float(), expected[0], rtol=.03, atol=.02)
            torch.testing.assert_close(actual[1], expected[1], rtol=.002, atol=.01)

        check(run())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        previous = None
        for changed in (False, True):
            if changed:
                q.add_(.25)
                kg.fill_(1.)
                vg.fill_(4.)
                used.fill_(384)
                table.copy_(table.roll(1, -1))
                with self.assertRaises(AssertionError):
                    check(previous)
            for tensor in captured:
                tensor.fill_(torch.nan)
            with self.assertRaises(AssertionError):
                check(captured)
            graph.replay()
            live = tuple(t.clone() for t in captured)
            check(live)
            for actual, eager in zip(live, run()):
                torch.testing.assert_close(actual, eager, rtol=0, atol=0)
            previous = live

    def test_five_group_gather4_static_and_clc_live_dispatch(self):
        self._check_five_group_gather4_static_and_clc_live_dispatch(17)

    def test_five_group_gather4_long_rows_and_partial_groups(self):
        self._check_five_group_gather4_static_and_clc_live_dispatch(2051)

    def _check_five_group_gather4_static_and_clc_live_dispatch(self, q_length):
        from mbsa.backends.optimized.msa.cute.src.sm100.fwd.config import (
            MSA_5WG_CLC_MASKED,
            MSA_5WG_CLC_UTMALDG_SCATTERED,
            MSA_5WG_STATIC_MASKED,
            MSA_5WG_STATIC_UTMALDG_SCATTERED,
        )
        for cfg in (MSA_5WG_STATIC_MASKED, MSA_5WG_CLC_MASKED,
                    MSA_5WG_STATIC_UTMALDG_SCATTERED,
                    MSA_5WG_CLC_UTMALDG_SCATTERED):
            with self.subTest(config=cfg.symbol_suffix):
                self._check_query_transport_selection_and_live_dispatch(cfg, q_length)

    def _check_query_transport_selection_and_live_dispatch(
        self, kernel_cfg=None, q_length=17, *, kv_length=None,
        full_score_tiles=False, storage_dtype=torch.bfloat16,
        ratios=(1, 2, 4, 8, 16), expected_error=None,
    ):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("SM100/SM103 required")
        from mbsa.backends.optimized.msa.cute import interface
        from mbsa.backends.optimized.msa.cute.sparse_index_utils import build_k2q_csr

        torch.manual_seed(173)
        full = (kernel_cfg.full_kv_blocks if kernel_cfg is not None
                else full_score_tiles)
        length = kv_length if kv_length is not None else (128 if full else 127)
        selected_page = (length - 1) // 128
        selected_begin = selected_page * 128
        for ratio in ratios:
            with self.subTest(qhead_per_kv=ratio):
                q = (torch.randn((q_length, 4 * ratio, 128), device='cuda', dtype=torch.bfloat16) * 0.25).to(storage_dtype)
                k = (torch.randn((length, 4, 128), device='cuda', dtype=torch.bfloat16) * 0.25).to(storage_dtype)
                v = torch.randn((length, 4, 128), device='cuda', dtype=torch.bfloat16).to(storage_dtype)
                cu_q = torch.tensor([0, q_length], device='cuda', dtype=torch.int32)
                cu_k = torch.tensor([0, length], device='cuda', dtype=torch.int32)
                routes = torch.full((4, q_length, 16), -1, device='cuda', dtype=torch.int32)
                routes[:, :, 0] = selected_page
                row_ptr, indices, schedule = build_k2q_csr(
                    routes, cu_q, cu_k, 128, total_k=length, max_seqlen_k=length,
                    max_seqlen_q=q_length, qhead_per_kv=ratio, return_schedule=True)
                if q_length > 2048 and kernel_cfg is not None and not kernel_cfg.use_clc:
                    # Exercise a whole long CSR row in one static work item.
                    # This changes descriptor chunking, not CSR membership or
                    # the launch grid derived from the supplied descriptors.
                    from dataclasses import replace
                    metadata = torch.tensor(
                        [[head, 0, 0, q_length, 0, 0] for head in range(4)],
                        device='cuda', dtype=torch.int32)
                    schedule = replace(
                        schedule, scheduler_metadata=metadata,
                        work_count=torch.tensor([4], device='cuda', dtype=torch.int32),
                        target_q_per_cta=q_length)
                out = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)

                def run():
                    return interface.sparse_atten_func(
                        q, k, v, row_ptr, indices, 16, cu_seqlens_q=cu_q,
                        cu_seqlens_k=cu_k, max_seqlen_q=q_length, max_seqlen_k=length,
                        schedule=schedule, full_score_tiles=full,
                        kernel_cfg=kernel_cfg,
                        out=out, partial_dtype=torch.bfloat16, return_softmax_lse=True)

                def oracle():
                    keys = k[selected_begin:length].float().repeat_interleave(ratio, dim=1)
                    values = v[selected_begin:length].float().repeat_interleave(ratio, dim=1)
                    scores = torch.einsum('qhd,khd->qhk', q.float(), keys) / math.sqrt(128)
                    return (torch.einsum('qhk,khd->qhd', scores.softmax(-1), values),
                            torch.logsumexp(scores, dim=-1))

                def check(actual, expected):
                    torch.testing.assert_close(actual[0].float(), expected[0], rtol=0.03, atol=0.02)
                    torch.testing.assert_close(actual[1], expected[1], rtol=0.002, atol=0.01)

                if expected_error is not None:
                    out.fill_(123)
                    with self.assertRaisesRegex(ValueError, expected_error):
                        run()
                    self.assertTrue(bool((out == 123).all()))
                    continue

                selector = interface.select_msa_cfg
                inner = interface._call_sparse_forward_sm100_csr_varlen
                with mock.patch.object(interface, 'select_msa_cfg', wraps=selector) as selected:
                    with mock.patch.object(interface, '_call_sparse_forward_sm100_csr_varlen',
                                           wraps=inner) as tapped:
                        check(run(), oracle())
                    bound = inspect.signature(inner).bind(*tapped.call_args.args,
                                                          **tapped.call_args.kwargs)
                    if kernel_cfg is None:
                        self.assertEqual(bound.arguments['kernel_cfg'].num_wgs, 5)
                        self.assertEqual(bound.arguments['kernel_cfg'].use_clc,
                                         length >= 32768)
                        self.assertEqual(bound.arguments['kernel_cfg'].full_kv_blocks, full)
                        bound.arguments['kernel_cfg'] = None
                        inner(*bound.args, **bound.kwargs)
                    else:
                        self.assertEqual(bound.arguments['kernel_cfg'], kernel_cfg)
                        self.assertEqual(kernel_cfg.num_wgs, 5)
                if kernel_cfg is None:
                    self.assertGreaterEqual(len(selected.call_args_list), 2)
                    self.assertEqual(
                        selector(**selected.call_args_list[-1].kwargs).use_clc,
                        length >= 32768)
                for call in selected.call_args_list:
                    self.assertEqual(call.kwargs['qhead_per_kv'], ratio)
                    self.assertEqual(selector(**call.kwargs).num_wgs, 5)
                    self.assertEqual(selector(**call.kwargs).full_kv_blocks, full)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = run()
                captured[0].fill_(torch.nan)
                captured[1].fill_(torch.nan)
                with self.assertRaises(AssertionError):
                    check(captured, oracle())
                graph.replay()
                live = tuple(t.clone() for t in captured)
                check(live, oracle())
                for actual, eager in zip(live, run()):
                    torch.testing.assert_close(actual, eager, rtol=0, atol=0)
                q.copy_((q.float() + 0.5).to(q.dtype))
                k.copy_((k.float() * 0.5).to(k.dtype))
                v.copy_((v.float() + 3).to(v.dtype))
                with self.assertRaises(AssertionError):
                    check(live, oracle())
                captured[0].fill_(torch.nan)
                captured[1].fill_(torch.nan)
                graph.replay()
                live = tuple(t.clone() for t in captured)
                check(live, oracle())
                for actual, eager in zip(live, run()):
                    torch.testing.assert_close(actual, eager, rtol=0, atol=0)

    def test_runtime_tail_and_changed_length_live_graph(self):
        self._check_runtime_tail_and_changed_length_live_graph()

    def test_clc_runtime_tail_and_changed_length_live_graph(self):
        from mbsa.backends.optimized.msa.cute.src.sm100.fwd.config import MSA_5WG_CLC_MASKED
        self._check_runtime_tail_and_changed_length_live_graph(MSA_5WG_CLC_MASKED)

    def _check_runtime_tail_and_changed_length_live_graph(self, kernel_cfg=None):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("SM100/SM103 required")
        from mbsa.backends.optimized.msa.cute import interface
        from mbsa.backends.optimized.msa.cute.sparse_index_utils import build_k2q_csr

        q = torch.zeros((17, 32, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.zeros((1, 4, 128, 128), device="cuda", dtype=torch.bfloat16)
        v = torch.ones_like(k)
        v[:, :, 126] = 127
        v[:, :, 127] = 4096
        route = torch.full((4, 17, 16), -1, device="cuda", dtype=torch.int32)
        route[:, :, 0] = 0
        cu_q = torch.tensor([0, 17], device="cuda", dtype=torch.int32)
        cu_k = torch.tensor([0, 128], device="cuda", dtype=torch.int32)
        used = torch.tensor([127], device="cuda", dtype=torch.int32)
        pages = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
        row_ptr, indices, schedule = build_k2q_csr(
            route, cu_q, cu_k, 128, total_k=128, max_seqlen_k=128,
            max_seqlen_q=17, qhead_per_kv=8, return_schedule=True)
        out = torch.empty_like(q)

        def run():
            return interface.sparse_atten_func(
                q, k, v, row_ptr, indices, 16, cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k, max_seqlen_q=17, max_seqlen_k=128,
                page_table=pages, seqused_k=used, schedule=schedule,
                kernel_cfg=kernel_cfg, full_score_tiles=False,
                out=out, return_softmax_lse=True)

        def check(actual, length):
            expected = v[0, :, :length].float().mean(dim=1).repeat_interleave(8, dim=0)
            torch.testing.assert_close(actual[0].float(), expected[None].expand_as(q),
                                       rtol=0.03, atol=0.03)
            torch.testing.assert_close(actual[1], torch.full_like(actual[1], math.log(length)),
                                       rtol=0.002, atol=0.01)

        check(run(), 127)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        for length in (127, 126):
            used.fill_(length)
            captured[0].fill_(torch.nan)
            captured[1].fill_(torch.nan)
            with self.assertRaises(AssertionError):
                check(captured, length)
            graph.replay()
            live = tuple(t.clone() for t in captured)
            check(live, length)
            for actual, expected in zip(live, run()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_logical_tail_boundaries_mixed_offsets_and_reordered_pages(self):
        self._check_logical_tail_boundaries_mixed_offsets_and_reordered_pages()

    def test_clc_logical_tail_boundaries_mixed_offsets_and_reordered_pages(self):
        from mbsa.backends.optimized.msa.cute.src.sm100.fwd.config import MSA_5WG_CLC_MASKED
        self._check_logical_tail_boundaries_mixed_offsets_and_reordered_pages(
            MSA_5WG_CLC_MASKED)

    def _check_logical_tail_boundaries_mixed_offsets_and_reordered_pages(
        self, kernel_cfg=None
    ):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("SM100/SM103 required")
        from mbsa.backends.optimized.msa.cute import interface
        from mbsa.backends.optimized.msa.cute.sparse_index_utils import build_k2q_csr

        torch.manual_seed(731)
        cases = ((127,), (128,), (129,), (127, 128, 129), (1048575,))
        for lengths in cases:
            for paged in (False, True):
                if not paged and lengths == (1048575,):
                    continue  # Long dense coverage belongs to the paired workload.
                for causal in (False, True):
                    if kernel_cfg is not None and causal:
                        continue  # Causal execution retains its general static path.
                    with self.subTest(lengths=lengths, paged=paged, causal=causal):
                        qlens = (17, 19, 9)[:len(lengths)]
                        capacities = ([((n + 127) // 128 + 1) * 128 for n in lengths]
                                      if paged else list(lengths))
                        cu_q = torch.tensor([0] + list(itertools.accumulate(qlens)),
                                            device='cuda', dtype=torch.int32)
                        cu_k = torch.tensor([0] + list(itertools.accumulate(capacities)),
                                            device='cuda', dtype=torch.int32)
                        used = torch.tensor(lengths, device='cuda', dtype=torch.int32)
                        q = torch.randn((sum(qlens), 32, 128), device='cuda',
                                        dtype=torch.bfloat16) * 0.25
                        table = None
                        if paged:
                            # Logical pages are independent of physical order. Long
                            # requests alias unselected pages but retain distinct
                            # first/tail pages, avoiding a large oracle allocation.
                            width = max(capacities) // 128
                            table = torch.zeros((len(lengths), width), device='cuda', dtype=torch.int32)
                            k = torch.randn((3 * len(lengths), 4, 128, 128),
                                            device='cuda', dtype=torch.bfloat16) * 0.25
                            v = torch.randn_like(k)
                            for b, length in enumerate(lengths):
                                table[b].fill_(3*b + 1)
                                table[b, 0] = 3*b + 2
                                if length > 128:
                                    table[b, (length - 1) // 128] = 3*b
                        else:
                            k = torch.randn((sum(capacities), 4, 128), device='cuda',
                                            dtype=torch.bfloat16) * 0.25
                            v = torch.randn_like(k)
                        routes = torch.full((4, sum(qlens), 16), -1, device='cuda', dtype=torch.int32)
                        for b, length in enumerate(lengths):
                            begin, end = int(cu_q[b]), int(cu_q[b+1])
                            blocks = {0, (length - 1) // 128}
                            if paged:
                                blocks.add((length + 127) // 128)
                            blocks = sorted(blocks)
                            for slot, block in enumerate(blocks):
                                routes[:, begin:end, slot] = block
                        saved_routes = routes.clone()
                        row_ptr, indices, schedule = build_k2q_csr(
                            routes, cu_q, cu_k, 128, total_k=sum(capacities),
                            max_seqlen_k=max(capacities), max_seqlen_q=max(qlens),
                            qhead_per_kv=8, return_schedule=True)
                        for b, length in enumerate(lengths):
                            selected_blocks = {0, (length - 1) // 128}
                            if paged:
                                selected_blocks.add((length + 127) // 128)
                            selected_count = len(selected_blocks)
                            begin, end = int(cu_q[b]), int(cu_q[b+1])
                            torch.testing.assert_close(
                                schedule.split_counts[begin:end],
                                torch.full_like(schedule.split_counts[begin:end], selected_count),
                                rtol=0, atol=0)
                        out = torch.empty_like(q)

                        def run():
                            return interface.sparse_atten_func(
                                q, k, v, row_ptr, indices, 16,
                                cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                                max_seqlen_q=max(qlens), max_seqlen_k=max(capacities),
                                page_table=table, seqused_k=used if paged else None, schedule=schedule,
                                causal=causal, full_score_tiles=False, out=out,
                                kernel_cfg=kernel_cfg,
                                return_softmax_lse=True, partial_dtype=torch.bfloat16)

                        def oracle():
                            expected_o = torch.zeros_like(q, dtype=torch.float32)
                            expected_lse = torch.full(q.shape[:2], -torch.inf, device='cuda')
                            for b, qlen in enumerate(qlens):
                                length = (int(used[b]) if paged
                                          else int(cu_k[b+1] - cu_k[b]))
                                begin = int(cu_q[b])
                                for row in range(qlen):
                                    blocks = saved_routes[0, begin + row]
                                    tokens = torch.cat([torch.arange(int(block)*128, min((int(block)+1)*128, length),
                                                        device='cuda') for block in blocks
                                                        if 0 <= int(block)*128 < length])
                                    if causal:
                                        tokens = tokens[tokens <= length - qlen + row]
                                    if not tokens.numel():
                                        continue
                                    if paged:
                                        physical = table[b, tokens // 128].long()
                                        keys = k[physical, :, tokens % 128, :]
                                        values = v[physical, :, tokens % 128, :]
                                    else:
                                        keys = k[int(cu_k[b]) + tokens]
                                        values = v[int(cu_k[b]) + tokens]
                                    keys = keys.float().repeat_interleave(8, dim=1)
                                    values = values.float().repeat_interleave(8, dim=1)
                                    scores = torch.einsum('hd,nhd->hn', q[begin+row].float(), keys) / math.sqrt(128)
                                    expected_lse[begin+row] = torch.logsumexp(scores, dim=-1)
                                    expected_o[begin+row] = torch.einsum('hn,nhd->hd', scores.softmax(-1), values)
                            return expected_o, expected_lse

                        def check(actual, expected):
                            torch.testing.assert_close(actual[0].float(), expected[0], rtol=0.03, atol=0.02)
                            torch.testing.assert_close(actual[1], expected[1], rtol=0.002, atol=0.01)
                            torch.testing.assert_close(routes, saved_routes, rtol=0, atol=0)

                        check(run(), oracle())
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            captured = run()
                        # Same-address mutation can remove an entire selected tail.
                        if paged:
                            used.sub_(1)
                        else:
                            cu_k[-1].sub_(1)
                        expected = oracle()
                        captured[0].fill_(torch.nan)
                        captured[1].fill_(torch.nan)
                        with self.assertRaises(AssertionError):
                            check(captured, expected)  # No-op replay negative control.
                        graph.replay()
                        live = tuple(t.clone() for t in captured)
                        check(live, expected)
                        for actual, eager in zip(live, run()):
                            torch.testing.assert_close(actual, eager, rtol=0, atol=0)
                        q.add_(0.5)
                        k.mul_(0.5)
                        v.add_(3)
                        expected = oracle()
                        with self.assertRaises(AssertionError):
                            check(live, expected)  # Stale output must reject changed operands.
                        captured[0].fill_(torch.nan)
                        captured[1].fill_(torch.nan)
                        graph.replay()
                        live = tuple(t.clone() for t in captured)
                        check(live, expected)
                        for actual, eager in zip(live, run()):
                            torch.testing.assert_close(actual, eager, rtol=0, atol=0)

@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class SparseFullTileOverride(unittest.TestCase):
    def test_mask_override_and_explicit_config_consistency(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("SM100/SM103 required")
        from mbsa.backends.optimized.msa.cute import interface
        from mbsa.backends.optimized.msa.cute.sparse_index_utils import build_k2q_csr
        from mbsa.backends.optimized.msa.cute.src.sm100.fwd.config import MSA_5WG_STATIC_UTMALDG_SCATTERED

        q = torch.zeros((17, 32, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.zeros((128, 4, 128), device="cuda", dtype=torch.bfloat16)
        v = torch.ones_like(k)
        route = torch.full((4, 17, 16), -1, device="cuda", dtype=torch.int32)
        route[:, :, 0] = 0
        cu_q = torch.tensor([0, 17], device="cuda", dtype=torch.int32)
        cu_k = torch.tensor([0, 128], device="cuda", dtype=torch.int32)
        row_ptr, indices, schedule = build_k2q_csr(
            route, cu_q, cu_k, 128, total_k=128, max_seqlen_k=128,
            max_seqlen_q=17, qhead_per_kv=8, return_schedule=True)
        common = dict(cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                      max_seqlen_q=17, max_seqlen_k=128, schedule=schedule,
                      return_softmax_lse=True)

        def run(**kwargs):
            return interface.sparse_atten_func(q, k, v, row_ptr, indices, 16,
                                               **common, **kwargs)

        selector = interface.select_msa_cfg
        inner = interface._call_sparse_forward_sm100_csr_varlen
        with mock.patch.object(interface, "select_msa_cfg", wraps=selector) as selected:
            with mock.patch.object(interface, "_call_sparse_forward_sm100_csr_varlen",
                                   wraps=inner) as tapped:
                output = run(full_score_tiles=False)
        direct = inspect.signature(inner).bind(*tapped.call_args.args,
                                               **tapped.call_args.kwargs)
        direct.arguments['kernel_cfg'] = MSA_5WG_STATIC_UTMALDG_SCATTERED
        direct.arguments['full_score_tiles'] = False
        with self.assertRaisesRegex(ValueError, "full"):
            inner(*direct.args, **direct.kwargs)
        self.assertTrue(selected.call_args_list)
        self.assertTrue(all(call.kwargs['full_kv_blocks'] is False
                            for call in selected.call_args_list))
        torch.testing.assert_close(output[0], torch.ones_like(q), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "full"):
            run(full_score_tiles=False, kernel_cfg=MSA_5WG_STATIC_UTMALDG_SCATTERED)

        # Explicit full configuration on paged inputs needs an explicit proof;
        # allocation geometry alone does not establish runtime valid lengths.
        kp = k.view(1, 128, 4, 128).permute(0, 2, 1, 3)
        vp = v.view(1, 128, 4, 128).permute(0, 2, 1, 3)
        page = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
        used = torch.tensor([127], device="cuda", dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "full"):
            interface.sparse_atten_func(q, kp, vp, row_ptr, indices, 16,
                **common, page_table=page, seqused_k=used,
                kernel_cfg=MSA_5WG_STATIC_UTMALDG_SCATTERED)
        with mock.patch.object(interface, "_call_sparse_forward_sm100_csr_varlen",
                               wraps=inner) as tapped:
            interface.sparse_atten_func(q, kp, vp, row_ptr, indices, 16,
                **common, page_table=page, seqused_k=used)
        direct = inspect.signature(inner).bind(*tapped.call_args.args,
                                               **tapped.call_args.kwargs)
        direct.arguments['kernel_cfg'] = MSA_5WG_STATIC_UTMALDG_SCATTERED
        direct.arguments['full_score_tiles'] = None
        with self.assertRaisesRegex(ValueError, "full"):
            inner(*direct.args, **direct.kwargs)
        used.fill_(128)
        proven = interface.sparse_atten_func(q, kp, vp, row_ptr, indices, 16,
            **common, page_table=page, seqused_k=used, full_score_tiles=True,
            kernel_cfg=MSA_5WG_STATIC_UTMALDG_SCATTERED)
        torch.testing.assert_close(proven[0], torch.ones_like(q), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
