# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Targeted independent regressions for customer integration review findings."""
import math
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class CustomerReviewRegression(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest('SM100/SM103 required')

    def test_combine_reused_tma_partial_stages(self):
        """Mixed valid counts must not consume the next reuse of an SMEM stage."""
        from mbsa.backends.optimized.msa.cute.src.sm100.fwd.combine import combine
        q, heads, dim, splits = 257, 32, 128, 16
        gen = torch.Generator(device='cuda').manual_seed(9128)
        counts = (torch.arange(q * 4, device='cuda').reshape(q, 4) % 17).int()
        count_heads = counts.repeat_interleave(8, dim=1)
        valid = torch.arange(splits, device='cuda')[:, None, None] < count_heads
        values = torch.randn((splits, q, heads, 1), device='cuda', generator=gen).to(torch.bfloat16)
        # Constant columns make this oracle independent of the internal fake-column layout.
        partial = values.expand(splits, q, heads, dim).contiguous()
        partial.masked_fill_(~valid[..., None], torch.nan)
        lse = torch.randn((splits, q, heads), device='cuda', generator=gen)
        lse.masked_fill_(~valid, torch.nan)
        fixed = torch.randn((q, heads, dim), device='cuda', generator=gen).to(torch.bfloat16)
        fixed_lse = torch.randn((1, heads, q), device='cuda', generator=gen)
        all_lse = torch.cat((fixed_lse[0].T[None], lse.masked_fill(~valid, -torch.inf)))
        weights = all_lse.softmax(0)
        expected = weights[0, ..., None] * fixed.float()
        expected += (weights[1:, ..., None] * partial.float().masked_fill(~valid[..., None], 0)).sum(0)
        expected_lse = all_lse.logsumexp(0)
        cu = torch.tensor([0, q], device='cuda', dtype=torch.int32)
        outputs = []
        for tma in (False, True):
            guarded = torch.full((q + 2, heads, dim), 19, device='cuda', dtype=torch.bfloat16)
            out = guarded[1:-1]
            out_lse = torch.empty((q, heads), device='cuda')

            def run():
                combine(partial, lse, out, out_lse, initial_o=fixed, initial_lse=fixed_lse,
                        cu_seqlens=cu, split_counts=counts, use_tma_partial=tma)

            run()
            torch.testing.assert_close(out.float(), expected, atol=.02, rtol=.02)
            torch.testing.assert_close(out_lse, expected_lse, atol=.002, rtol=.002)
            for _ in range(100):
                run()
                torch.testing.assert_close(out.float(), expected, atol=.02, rtol=.02)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            for _ in range(10):
                out.fill_(torch.nan)
                out_lse.fill_(torch.nan)
                graph.replay()
            torch.testing.assert_close(out.float(), expected, atol=.02, rtol=.02)
            torch.testing.assert_close(out_lse, expected_lse, atol=.002, rtol=.002)
            self.assertTrue(bool((guarded[[0, -1]] == 19).all()))
            outputs.append((out.clone(), out_lse.clone()))
        for actual, reference in zip(outputs[1], outputs[0]):
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)

    def test_fixed_collective_load_with_mixed_nonfinite_rows(self):
        """A row-dependent result must never gate a warp-collective TMEM load."""
        from mbsa.backends.optimized.bsa.cute_dsl.sparse.bsa_attn_sm100_blk128 import bsa_attn_sm100_blk128_fwd
        from tests.regression.test_fixed_fa4 import fixed_oracle
        nq, nk = 129, 2177
        gen = torch.Generator(device='cuda').manual_seed(9130)
        rand = lambda shape: (torch.randn(shape, device='cuda', generator=gen) * .2).to(torch.bfloat16)
        q, k, v = rand((nq, 32, 128)), rand((18, 4, 128, 128)), rand((18, 4, 128, 128))
        table = torch.arange(18, device='cuda', dtype=torch.int32).flip(0)[None]
        tiles = torch.tensor([[0, m, 0, nq, 0, nk] for m in range(math.ceil(nq * 8 / 128))],
                             device='cuda', dtype=torch.int32)
        sentinel = torch.zeros(1, device='cuda', dtype=torch.int32)
        guard = torch.full((nq + 2, 32, 128), 23, device='cuda', dtype=torch.bfloat16)
        out, lse = guard[1:-1], torch.empty((32, nq), device='cuda')

        def run():
            return bsa_attn_sm100_blk128_fwd(q, k, v, sentinel, 0, pack_gqa=True,
                return_lse=True, out=out, lse=lse, causal=True, implicit_fixed_support=True,
                page_table=table, packed_tiles=tiles, use_clc_scheduler=False)

        logical = [x[table[0].long()].permute(0, 2, 1, 3).reshape(-1, 4, 128)[:nk]
                   for x in (k, v)]
        expected_o, expected_l = fixed_oracle(q, *logical)
        run()
        torch.testing.assert_close(out.float(), expected_o, atol=.025, rtol=.025)
        torch.testing.assert_close(lse, expected_l, atol=.002, rtol=.002)
        clean_q, clean_out = q.clone(), out.clone()
        for value, rows in ((torch.nan, (0,)), (torch.nan, (0, 1, 31, 32)), (torch.inf, (1,))):
            q.copy_(clean_q)
            mask = torch.zeros((nq, 32), device='cuda', dtype=torch.bool)
            for row in rows:
                q[row, 0].fill_(value)
                mask[row, 0] = True
            run()
            torch.testing.assert_close(out[~mask], clean_out[~mask], atol=0, rtol=0)
            self.assertTrue(bool(torch.isfinite(out[~mask]).all()))
            if math.isnan(value):
                self.assertTrue(bool(torch.isnan(out[mask]).all()))
                self.assertTrue(bool(torch.isnan(lse.T[mask]).all()))
            # Nonfinite logical operands are diagnostic; masked-input isolation is not imposed.
            self.assertTrue(bool((guard[[0, -1]] == 23).all()))

    def test_non_aligned_query_tile_dynamic_support(self):
        """Per-row remote exclusion must hold across an absolute page boundary."""
        from mbsa import prepare_cutedsl_pipeline
        from tests.regression.test_backends import BackendLifecycle
        from tests.common import BackendCaseMixin
        fixture = BackendLifecycle()
        lengths = ((129, 4273),)
        for fp8 in (False, True):
            data, inputs, kwargs = fixture.fixture(lengths)
            if fp8:
                inputs = (inputs[0].to(torch.float8_e4m3fn), inputs[1].to(torch.float8_e4m3fn), *inputs[2:])
            expected = fixture.reference(data, inputs, lengths)
            plan = prepare_cutedsl_pipeline(*inputs, backend='optimized', **kwargs)
            actual = plan.run()
            BackendCaseMixin.compare(self, actual, expected)
            positions = torch.arange(129, device='cuda') + 4273 - 129
            local_begin = (positions // 128 + 1 - 8).clamp_min(8)
            route = actual[2]
            valid = route >= 0
            self.assertTrue(bool(((~valid) | ((route >= 8) & (route < local_begin[None, :, None]))).all()))

    def test_mx_compact_non_aligned_request_rows(self):
        from mbsa import prepare_cutedsl_pipeline
        from mbsa.index_formats import (quantize_q_mxfp8, quantize_index_k_mxfp6,
            dequantize_index_q_mxfp8, dequantize_index_k_mxfp6)
        from tests.regression.test_backends import BackendLifecycle
        from tests.common import BackendCaseMixin
        fixture = BackendLifecycle()
        lengths = tuple((q, pages * 128 - 17) for q, pages in zip((17, 23, 29, 31), (41, 58, 73, 30)))
        data, inputs, kwargs = fixture.fixture(lengths)
        codes, scales = quantize_q_mxfp8(inputs[0])
        iq = codes[:, :4].view(torch.float8_e4m3fn)
        ik = quantize_index_k_mxfp6(inputs[1])
        decoded = (dequantize_index_q_mxfp8(iq, scales), dequantize_index_k_mxfp6(ik), *inputs[2:])
        expected = fixture.reference(data, decoded, lengths)
        compact = data['pages']
        self.assertNotEqual(compact[41:].data_ptr() % 16, 0)
        saved = tuple(t.clone() for t in (compact, *inputs[2:]))
        plan = prepare_cutedsl_pipeline(iq, ik, *inputs[2:], backend='optimized',
            index_q_scales=scales, **kwargs)
        BackendCaseMixin.compare(self, plan.run(), expected)
        plan.capture_mega_graph()
        plan.plan.attention.out.fill_(torch.nan)
        BackendCaseMixin.compare(self, plan.run_mega(), expected)
        for actual, before in zip((compact, *inputs[2:]), saved):
            torch.testing.assert_close(actual, before, atol=0, rtol=0)

    def test_actual_plan_drop_without_cyclic_gc(self):
        import gc
        import weakref
        from mbsa import prepare_cutedsl_pipeline
        from tests.regression.test_backends import BackendLifecycle
        fixture = BackendLifecycle()
        for backend in ('default', 'optimized'):
            _, inputs, kwargs = fixture.fixture(((17, 4113),))
            # Isolate plan ownership from one-time compiler traceback frames.
            warm = prepare_cutedsl_pipeline(*inputs, backend=backend, **kwargs)
            warm.run()
            torch.cuda.synchronize()
            del warm
            gc.collect()
            enabled = gc.isenabled()
            gc.disable()
            try:
                owner = prepare_cutedsl_pipeline(*inputs, backend=backend, **kwargs)
                owner.run()
                torch.cuda.synchronize()
                plan_ref = weakref.ref(owner.plan)
                out_ref = weakref.ref(owner.plan.attention.out)
                del owner
                self.assertIsNone(plan_ref(), backend)
                self.assertIsNone(out_ref(), backend)
            finally:
                if enabled:
                    gc.enable()

    def test_csr_sorted_duplicate_edges_across_warp_cta_ranges(self):
        from mbsa.backends.optimized.msa.cute.sparse_index_utils import build_k2q_csr
        qlens, blocks = (1025, 513), (9, 7)
        total = sum(qlens)
        cuq = torch.tensor([0, qlens[0], total], device='cuda', dtype=torch.int32)
        cuk = torch.tensor([0, blocks[0] * 128, sum(blocks) * 128], device='cuda', dtype=torch.int32)
        gen = torch.Generator(device='cuda').manual_seed(9131)
        route = torch.randint(-1, 10, (4, total, 16), device='cuda', generator=gen, dtype=torch.int32)
        route[:, :, :4] = 0  # Duplicate same-query edges stress rank ordering.
        rows = [(b, r) for r in range(max(blocks)) for b in range(2) if r < blocks[b]]
        expected = [[[] for _ in rows] for _ in range(4)]
        for h in range(4):
            for row, (b, r) in enumerate(rows):
                start = 0 if b == 0 else qlens[0]
                expected[h][row] = torch.where(route[h, start:start + qlens[b]] == r)[0].int()
        for _ in range(20):
            rowptr, indices = build_k2q_csr(route, cuq, cuk, 128, total_k=sum(blocks) * 128,
                max_seqlen_k=max(blocks) * 128, max_seqlen_q=max(qlens), total_rows=sum(blocks))
            for h in range(4):
                offset = 0
                for row, target in enumerate(expected[h]):
                    self.assertEqual(int(rowptr[h, row]), offset)
                    torch.testing.assert_close(indices[h, offset:offset + target.numel()], target,
                                               atol=0, rtol=0)
                    offset += target.numel()
                    self.assertEqual(int(rowptr[h, row + 1]), offset)
                self.assertTrue(bool((indices[h, offset:] == -1).all()))

    def test_mixed_serving_warmup_compiles_merge(self):
        from unittest.mock import patch
        from mbsa.serving import Prefill, warmup
        from mbsa.backends.optimized.kernels.merge import _MERGE_KERNELS
        prefill = Prefill(torch.device('cuda'), backend='optimized')
        lengths = ((16, 4113), (17, 4113))
        torch.cuda.synchronize()
        with patch.dict(_MERGE_KERNELS, clear=True):
            warmup(prefill, lengths, index_format='bf16')
            torch.cuda.synchronize()
            self.assertTrue(_MERGE_KERNELS)
            before = dict(_MERGE_KERNELS)
            warmup(prefill, lengths, index_format='bf16')
            # Row counts are dynamic: another mixed batch size reuses the warmed kernel.
            warmup(prefill, ((16, 4113), (18, 4113)), index_format='bf16')
            torch.cuda.synchronize()
            self.assertEqual(before.keys(), _MERGE_KERNELS.keys())
            for key, entry in before.items():
                self.assertIs(entry, _MERGE_KERNELS[key])

    def test_full_query_poisoned_physical_tail(self):
        from mbsa import prepare_cutedsl_pipeline
        from tests.regression.test_backends import BackendLifecycle
        from tests.reference import reference_pipeline
        data, inputs, kwargs = BackendLifecycle().fixture(((8192, 32769),))
        physical = int(data['table'][0, -1])
        inputs[1][physical, :, 1:].fill_(torch.nan)
        for tensor in inputs[3:]:
            tensor[physical, :, 1:].fill_(torch.nan)
        saved = [t.clone() for t in inputs]
        p = prepare_cutedsl_pipeline(*inputs, backend='optimized', **kwargs)
        out, lse, route = p.run()
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertTrue(bool(torch.isfinite(lse).all()))
        ids = data['table'][0].long()
        ik = inputs[1][data['pages'].long()].reshape(-1, 1, 128)[:32769]
        k, v = (t[ids].permute(0, 2, 1, 3).reshape(-1, 4, 128)[:32769] for t in inputs[3:])
        for row in (0, 1, 15, 16, 31, 32, 127, 128, 4095, 4096, 8063, 8064, 8175, 8176, 8190, 8191):
            eo, el, er = reference_pipeline(inputs[0][row:row+1], ik, inputs[2][row:row+1],
                                           k, v, query_offset=32769 - 8192 + row)
            torch.testing.assert_close(out[row:row+1], eo, atol=.03, rtol=.03)
            torch.testing.assert_close(lse[row:row+1], el, atol=.01, rtol=.002)
            torch.testing.assert_close(route[:, row:row+1].sort(-1).values,
                                       er.sort(-1).values, atol=0, rtol=0)
        for tensor, before in zip(inputs, saved):
            torch.testing.assert_close(tensor, before, atol=0, rtol=0, equal_nan=True)

    def test_fp8_partial_producer_to_combine_stage_reuse(self):
        """FP8 is intermediate storage here; Index and attention operands are BF16."""
        import os
        from unittest.mock import patch
        from mbsa import prepare_cutedsl_pipeline
        from tests.reference import reference_pipeline
        qn, nk = 6194, 7184
        gen = torch.Generator(device='cuda').manual_seed(9137)

        def rand(shape):
            return (torch.randn(shape, device='cuda', generator=gen) * .3).to(torch.bfloat16)

        inputs = (rand((qn, 4, 128)), rand((nk, 1, 128)), rand((qn, 32, 128)),
                  rand((nk, 4, 128)), rand((nk, 4, 128)))
        expected = reference_pipeline(*inputs)
        with patch.dict(os.environ, {'FLASHINFER_K2_TMA_PARTIAL': '0'}):
            p = prepare_cutedsl_pipeline(*inputs, backend='optimized', partial_dtype=torch.float8_e4m3fn)
            actual = p.run()
            torch.testing.assert_close(actual[0], expected[0], atol=.03, rtol=.03)
            torch.testing.assert_close(actual[1], expected[1], atol=.01, rtol=.002)
            torch.testing.assert_close(actual[2].sort(-1).values, expected[2].sort(-1).values, atol=0, rtol=0)
            safe = tuple(t.clone() for t in actual[:2])
        with patch.dict(os.environ, {'FLASHINFER_K2_TMA_PARTIAL': '1'}):
            for _ in range(256):
                for tensor in p.plan.attention.dynamic_workspace[:2]:
                    tensor.fill_(torch.nan)
                p.plan.attention.out.fill_(torch.nan)
                p.plan.attention.out_lse.fill_(torch.nan)
                actual = p.run()
                torch.testing.assert_close(actual[0], safe[0], atol=0, rtol=0)
                torch.testing.assert_close(actual[1], safe[1], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
