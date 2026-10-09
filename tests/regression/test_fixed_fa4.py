# SPDX-License-Identifier: Apache-2.0
"""Fixed FA4 dispatch, complete-output oracle, bind-layer and graph contracts."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))


def fixed_oracle(q, k, v):
    """Independent FP32 token-mask attention, covering every query/head."""
    nq, nk = q.shape[0], k.shape[0]
    tokens = torch.arange(nk, device=q.device)
    output = torch.empty_like(q, dtype=torch.float32)
    lse = torch.empty(32, nq, device=q.device)
    for begin in range(0, nq, 64):
        end = min(begin + 64, nq)
        queries = q[begin:end].float().reshape(-1, 4, 8, 128)
        scores = torch.einsum('qhgd,khd->hqgk', queries, k.float()) * 128 ** -.5
        positions = torch.arange(begin, end, device=q.device) + nk - nq
        visible = ((tokens[None] <= positions[:, None]) &
                   ((tokens[None] // 128 < 8) |
                    (tokens[None] // 128 >= positions[:, None] // 128 - 7)))
        scores.masked_fill_(~visible[None, :, None], -torch.inf)
        output[begin:end] = torch.einsum('hqgk,khd->qhgd', scores.softmax(-1), v.float()).reshape(-1, 32, 128)
        lse[:, begin:end] = scores.logsumexp(-1).permute(1, 0, 2).reshape(-1, 32).T
    return output, lse


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
class FixedFA4Regression(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest('SM100/SM103 is required')
        try:
            available = importlib.util.find_spec('flash_attn.cute.interface') is not None
        except (ImportError, ValueError):
            available = False
        if not available:
            self.skipTest('FlashAttention-4 CuTe is required')

    def _inputs(self, nq=1025, nk=2048):
        gen = torch.Generator(device='cuda').manual_seed(337)
        def rand(shape):
            return (torch.randn(shape, generator=gen, device='cuda') * .2).bfloat16()
        return rand((nq, 4, 128)), rand((nk, 1, 128)), rand((nq, 32, 128)), rand((nk, 4, 128)), rand((nk, 4, 128))

    def _pipeline(self, nq=1025, nk=2048):
        from mbsa import prepare_cutedsl_pipeline
        data = self._inputs(nq=nq, nk=nk)
        pipeline = prepare_cutedsl_pipeline(*data, partial_dtype=torch.bfloat16)
        return pipeline, data

    def _check(self, pipeline, data):
        _, _, q, k, v = data
        expected_out, expected_lse = fixed_oracle(q, k, v)
        a = pipeline.plan.attention
        torch.testing.assert_close(a.fixed_out.float(), expected_out, atol=.025, rtol=.025)
        torch.testing.assert_close(a.fixed_lse[0], expected_lse, atol=.002, rtol=.002)
        torch.testing.assert_close(a.out, a.fixed_out, atol=0, rtol=0)
        torch.testing.assert_close(a.out_lse, a.fixed_lse[0].T, atol=0, rtol=0)

    def test_auto_fixed_only_query_tail_exact_output_and_merge(self):
        pipeline, data = self._pipeline()
        a = pipeline.plan.attention
        self.assertEqual(a.fixed_backend, 'fa4')
        self.assertTrue(a.fixed_only)
        self.assertTrue(a.fixed_attention.immutable_table)
        self.assertTrue(a.fixed_attention.identity_table)
        self.assertTrue(a.fixed_attention.pack_gqa)
        self.assertFalse(a.fixed_attention.static_identity_mask)
        self.assertEqual(getattr(a.fixed_attention._mask_mod, '__vec_size__', 1), 1)
        pipeline.run()
        self._check(pipeline, data)

    def test_private_compilation_reuse_keeps_geometry_and_operands_independent(self):
        """New plans reuse compatible kernels without stale operands or public cache edits."""
        from flash_attn.cute.interface import _flash_attn_fwd

        public_store = getattr(_flash_attn_fwd.compile_cache, 'cache', _flash_attn_fwd.compile_cache)
        public_cache = dict(public_store)
        first, first_data = self._pipeline(nq=634)
        second, second_data = self._pipeline(nq=635)
        other, other_data = self._pipeline(nq=16)
        a, b, c = (p.plan.attention.fixed_attention for p in (first, second, other))
        self.assertIs(a._prepared_forward, b._prepared_forward)
        self.assertIsNot(a._prepared_forward, c._prepared_forward)
        for fixed in (a, b, c):
            self.assertEqual(len(fixed._forward.compile_cache), 1)
            self.assertIsNot(fixed._forward.compile_cache, _flash_attn_fwd.compile_cache)
        self.assertEqual(public_store, public_cache)
        for pipeline, data in ((first, first_data), (second, second_data), (other, other_data)):
            pipeline.run()
            self._check(pipeline, data)

    def test_auto_static_identity_exact_output_and_merge(self):
        pipeline, data = self._pipeline(nq=8192, nk=8192)
        a = pipeline.plan.attention
        fixed = a.fixed_attention
        self.assertTrue(fixed.static_identity_mask)
        self.assertEqual(fixed._mask_mod.__vec_size__, 32)
        self.assertFalse(a.fixed_only)
        self.assertEqual(pipeline.plan._lean192_early_router,
                         torch.cuda.get_device_capability() == (10, 3))
        pipeline.run()
        _, _, q, k, v = data
        expected_fixed_out, expected_fixed_lse = fixed_oracle(q, k, v)
        torch.testing.assert_close(a.fixed_out.float(), expected_fixed_out, atol=.025, rtol=.025)
        torch.testing.assert_close(a.fixed_lse[0], expected_fixed_lse, atol=.002, rtol=.002)

        # An independent FP32 formula verifies the full fixed/dynamic merge.
        fixed_lse, dynamic_lse = a.fixed_lse[0].T, a.dynamic_workspace[2]
        expected_lse = torch.logaddexp(fixed_lse, dynamic_lse)
        fixed_weight = (fixed_lse - expected_lse).exp()[..., None]
        dynamic_weight = (dynamic_lse - expected_lse).exp()[..., None]
        expected_out = fixed_weight * a.fixed_out.float() + torch.where(
            dynamic_weight > 0, dynamic_weight * a.dynamic_out.float(), 0)
        torch.testing.assert_close(a.out.float(), expected_out, atol=.025, rtol=.025)
        torch.testing.assert_close(a.out_lse, expected_lse, atol=.002, rtol=.002)

    def _check_bind_layer_live_values_and_graph_query_refresh(self, nq):
        pipeline, data = self._pipeline(nq=nq)
        a = pipeline.plan.attention
        fixed = a.fixed_attention
        old_fixed_addresses = fixed.out.data_ptr(), fixed.lse.data_ptr(), fixed.block_tables.data_ptr()
        iq, ik, q, k, v = (tensor.clone() for tensor in data)
        q.add_(.125)
        k.mul_(.75)
        v.add_(.25)
        out = torch.empty_like(q)
        with mock.patch.object(fixed, '_refresh_mappings', wraps=fixed._refresh_mappings) as refresh:
            pipeline.plan.bind_layer(iq, ik, q, k, v, out)
            pipeline.run()
            self.assertEqual(refresh.call_count, 0)
        self.assertEqual(fixed.q.data_ptr(), q.data_ptr())
        self.assertEqual(fixed.k_cache.data_ptr(), k.data_ptr())
        self.assertEqual(fixed.v_cache.data_ptr(), v.data_ptr())
        self.assertEqual(a.out.data_ptr(), out.data_ptr())
        self.assertEqual(old_fixed_addresses, (fixed.out.data_ptr(), fixed.lse.data_ptr(), fixed.block_tables.data_ptr()))
        rebound = iq, ik, q, k, v
        self._check(pipeline, rebound)
        pipeline.plan.capture_mega_graph()
        q.add_(.125)
        v.add_(.5)
        pipeline.plan.run_mega()
        self._check(pipeline, rebound)
        with self.assertRaisesRegex(RuntimeError, 'captured mega graph'):
            pipeline.plan.bind_layer(iq, ik, q, k, v, out)

    def test_bind_layer_live_values_and_graph_query_refresh(self):
        self._check_bind_layer_live_values_and_graph_query_refresh(1025)

    def test_static_identity_bind_layer_live_values_and_graph_query_refresh(self):
        self._check_bind_layer_live_values_and_graph_query_refresh(1024)

    def test_direct_scalar_metadata_and_static_identity_contract(self):
        from mbsa.cute.fixed_attention import PreparedFixedAttention
        _, _, q, k, v = self._inputs(nq=1024)
        cache_k = k.view(-1, 128, 4, 128).permute(0, 2, 1, 3)
        cache_v = v.view(-1, 128, 4, 128).permute(0, 2, 1, 3)
        table = torch.arange(16, device=q.device, dtype=torch.int32)[None]
        used = torch.tensor([2048], device=q.device, dtype=torch.int32)

        def prepare(query=q, lengths=((1024, 2048),), **kwargs):
            return PreparedFixedAttention(query, cache_k, cache_v, table, used,
                                          lengths, warmup=False, **kwargs)

        direct = prepare()
        self.assertFalse(direct.static_identity_mask)
        self.assertFalse(direct.immutable_table)
        self.assertEqual(getattr(direct._mask_mod, '__vec_size__', 1), 1)
        with mock.patch.object(direct, '_forward') as forward, \
                mock.patch.object(direct, '_refresh_mappings', wraps=direct._refresh_mappings) as refresh:
            direct._run()
            used.fill_(1920)
            table.copy_(table.flip(-1))
            direct._run()
            self.assertEqual(refresh.call_count, 2)
            aux = forward.call_args.kwargs['aux_tensors']
            self.assertIsNotNone(aux)
            self.assertEqual(aux[1].data_ptr(), used.data_ptr())
            self.assertEqual(aux[1].item(), 1920)
        used.fill_(2048)
        table.copy_(torch.arange(16, device=q.device, dtype=torch.int32)[None])
        default_identity = prepare(identity_table=True, immutable_table=True)
        self.assertFalse(default_identity.static_identity_mask)
        static = prepare(identity_table=True, immutable_table=True, static_identity_mask=True)
        with mock.patch.object(static, '_forward') as forward:
            static._run()
            self.assertIsNone(forward.call_args.kwargs['aux_tensors'])
        for kwargs in ({}, {'identity_table': True}, {'immutable_table': True}):
            with self.subTest(**kwargs), self.assertRaisesRegex(ValueError, 'static_identity_mask'):
                prepare(static_identity_mask=True, **kwargs)
        for query, lengths in ((torch.cat((q, q[:1])), ((1025, 2048),)),
                               (q, ((1024, 2047),))):
            with self.subTest(lengths=lengths), self.assertRaisesRegex(ValueError, 'static_identity_mask'):
                prepare(query, lengths, identity_table=True, immutable_table=True,
                        static_identity_mask=True)
        used.fill_(1920)
        with self.assertRaisesRegex(ValueError, 'fixed used-KV lengths'):
            prepare(identity_table=True, immutable_table=True, static_identity_mask=True)

    def _attention(self, *, nq=1024, nk=2048, batch=1, paged=False, partial=torch.bfloat16):
        from mbsa.cute.plan import FwdTilePlan, PrefillPolicy
        from mbsa.cute.prefill import prepare_attention
        gen = torch.Generator(device='cuda').manual_seed(341)
        q = torch.randn(batch * nq, 32, 128, generator=gen, device='cuda').bfloat16()
        pages = (nk + 127) // 128
        if paged:
            k = torch.zeros(batch * pages, 4, 128, 128, device='cuda', dtype=torch.bfloat16)
            v = torch.zeros_like(k)
        else:
            k = torch.zeros(batch * nk, 4, 128, device='cuda', dtype=torch.bfloat16)
            v = torch.zeros_like(k)
        table = torch.arange(batch * pages, device='cuda', dtype=torch.int32).reshape(batch, pages)
        tile = FwdTilePlan(torch.full((4, batch * nq, 16), -1, dtype=torch.int32, device='cuda'),
                           table, torch.full((batch,), nk, dtype=torch.int32, device='cuda'),
                           nk - nq if batch == 1 else (nk - nq,) * batch,
                           PrefillPolicy.ALIGNED_PAGED if paged else PrefillPolicy.ALIGNED_CONTIGUOUS)
        packed = {}
        if batch != 1:
            packed = dict(cu_seqlens_q=torch.arange(batch + 1, device='cuda', dtype=torch.int32) * nq,
                          cu_seqlens_k=torch.arange(batch + 1, device='cuda', dtype=torch.int32) * nk,
                          request_lens=((nq, nk),) * batch)
        return prepare_attention(q, k, v, tile, out=torch.empty_like(q), partial_dtype=partial, **packed)

    def test_packed_single_request_retains_scalar_fixed(self):
        from mbsa import prepare_cutedsl_pipeline
        data = self._inputs(nq=1024)
        pipeline = prepare_cutedsl_pipeline(
            *data, partial_dtype=torch.bfloat16,
            cu_seqlens_q=torch.tensor([0, 1024], device='cuda', dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 2048], device='cuda', dtype=torch.int32),
            request_lens=((1024, 2048),))
        fixed = pipeline.plan.attention.fixed_attention
        self.assertFalse(fixed.static_identity_mask)
        pipeline.run()
        self._check(pipeline, data)

    def test_serving_interleaved_pages_live_tables_rebind_and_graph(self):
        """vLLM NHD K/V views must use FA4 with independent, mutable page maps."""
        from tests.reference import reference_pipeline
        from mbsa.serving import Prefill

        iq, ik, q, k, v = self._inputs(nq=1025, nk=4096)
        pages, pool_pages = 32, 41
        gen = torch.Generator(device='cuda').manual_seed(53)
        table = torch.randperm(pool_pages, device='cuda', generator=gen)[:pages].int()[None]
        index_table = torch.randperm(pool_pages, device='cuda', generator=gen)[:pages].int()[None]
        pool = torch.zeros(pool_pages, 128, 4, 256, device='cuda', dtype=q.dtype)
        pool[table[0].long()] = torch.cat((k, v), -1).view(pages, 128, 4, 256)
        cache = pool.permute(0, 2, 1, 3)
        kc, vc = cache.split(128, -1)
        index_cache = torch.zeros(pool_pages, 128, 128, device='cuda', dtype=iq.dtype)
        index_cache[index_table[0].long()] = ik.view(pages, 128, 128)
        step = Prefill(q.device).stage(((1025, 4096),), table, index_table)
        out = torch.empty_like(q)
        def run():
            step.run(q, kc, vc, index_q=iq, index_k_cache=index_cache, out=out)
        run()
        a = step.plan.attention
        self.assertEqual(a.fixed_backend, 'fa4')
        self.assertEqual(a.dynamic_backend, 'fireworks')
        self.assertTrue(step.plan.overlap_fixed_score)
        self.assertFalse(a.fixed_attention.immutable_table)

        def check():
            logical = cache[table[0].long()].permute(0, 2, 1, 3).reshape(-1, 4, 256)
            logical_k, logical_v = logical.split(128, -1)
            logical_ik = index_cache[index_table[0].long()].reshape(-1, 1, 128)
            fixed_out, fixed_lse = fixed_oracle(q, logical_k, logical_v)
            torch.testing.assert_close(a.fixed_out.float(), fixed_out, atol=.025, rtol=.025)
            torch.testing.assert_close(a.fixed_lse[0], fixed_lse, atol=.002, rtol=.002)
            for row in (0, 127, 128, 512, 1024):
                expected = reference_pipeline(iq[row:row+1].float(), logical_ik.float(),
                    q[row:row+1], logical_k, logical_v, query_offset=4096-1025+row)
                torch.testing.assert_close(out[row:row+1], expected[0], atol=.025, rtol=.025)
                route = a.tile_plan.dynamic_route[:, row:row+1]
                torch.testing.assert_close(route.sort(-1).values, expected[2].sort(-1).values,
                                           atol=0, rtol=0)
        check()
        # Page values change in place; the step owns the prepared table addresses.
        table.copy_(table.flip(-1))
        index_table.copy_(index_table.flip(-1))
        cache = cache.clone(memory_format=torch.preserve_format)
        kc, vc = cache.split(128, -1)
        vc.add_(.25)
        q = q.clone().add_(.125)
        run()
        check()
        step.plan.capture_mega_graph()
        table.copy_(table.flip(-1))
        index_table.copy_(index_table.flip(-1))
        vc.add_(.25)
        step.plan.run_mega()
        check()


    def _check_fixed_page_helpers(self, lengths, map_sizes):
        from mbsa.cute.cache_pages import gather_pages
        from mbsa.cute.paged_fixed_maps import refresh_maps

        cache = torch.arange(8 * 4 * 128 * 256, device='cuda', dtype=torch.float32)
        cache = cache.reshape(8, 4, 128, 256).to(torch.bfloat16)
        k, v = cache.split(128, -1)
        table = torch.tensor([7, 2, 6, 0, 5, 1, 4, 3], device='cuda', dtype=torch.int32)
        logical = torch.tensor([0, 2, 4], device='cuda', dtype=torch.int32)
        ko = torch.empty(3, 128, 4, 128, device='cuda', dtype=k.dtype).permute(0, 2, 1, 3)
        vo = torch.empty_like(ko.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        for length in lengths:
            gather_pages(k, v, table, logical, ko, vo, length)
            valid = logical[:, None].long() * 128 + torch.arange(128, device='cuda')[None] < length
            for source, actual in ((k, ko), (v, vo)):
                expected = source[table[logical.long()].long()].clone()
                expected.masked_fill_(~valid[:, None, :, None], 0)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for pages, masked_count, full_count in map_sizes:
            masked = torch.arange(masked_count, device='cuda', dtype=torch.int64)
            full = torch.arange(full_count, device='cuda', dtype=torch.int64)
            pm = torch.empty_like(masked, dtype=torch.int32)
            pf = torch.empty_like(full, dtype=torch.int32)
            inverse = torch.empty(8, device='cuda', dtype=torch.int32)
            positions = torch.arange(pages, device='cuda', dtype=torch.int32)
            refresh_maps(table, (masked, full, pm, pf, inverse, None, positions))
            torch.testing.assert_close(pm, table[masked], atol=0, rtol=0)
            torch.testing.assert_close(pf, table[full], atol=0, rtol=0)
            torch.testing.assert_close(inverse[table[:pages].long()], positions, atol=0, rtol=0)

    def test_fixed_page_helpers_preserve_variable_tails_and_map_sizes(self):
        """Changing chunk lengths keeps staged KV padding and page maps exact."""
        self._check_fixed_page_helpers(
            (640, 639, 513, 512), ((8, 1, 0), (7, 3, 2), (5, 0, 4)))

    def test_serving_warmup_covers_changing_page_helper_shapes(self):
        """Mixed chunk sizes do not compile page helpers after serving warmup."""
        import triton
        from mbsa.serving import Prefill, warmup

        warmup(Prefill(torch.device('cuda')), ((17, 4096),), index_format='fp8')
        compiled = []
        def record(**kwargs):
            compiled.append(getattr(kwargs.get('fn'), 'name', None))
        with mock.patch.object(triton.knobs.runtime, 'jit_post_compile_hook', record):
            self._check_fixed_page_helpers(
                (637, 629, 615, 603), ((8, 5, 6), (7, 6, 3), (5, 2, 0)))
        self.assertFalse(set(compiled) & {'_gather', '_refresh'}, compiled)

    def test_packed_requests_use_paged_fixed_with_shared_prefix_pages(self):
        """Packed serving requests retain separate masks when their cache pages overlap."""
        from mbsa.serving import Prefill
        from tests.reference import reference_pipeline
        iq, ik, q, k, v = self._inputs(nq=1024, nk=4096)
        cache = torch.cat((k, v), -1).view(32, 128, 4, 256).permute(0, 2, 1, 3)
        kc, vc = cache.split(128, -1)
        table = torch.arange(32, device='cuda', dtype=torch.int32)[None].repeat(2, 1)
        table[1, 8:] = table[1, 8:].flip(-1)
        index_table = torch.arange(32, device='cuda', dtype=torch.int32)[None].repeat(2, 1)
        q = torch.cat((q, q[:13])).contiguous()
        iq = torch.cat((iq, iq[:13])).contiguous()
        step = Prefill(q.device).stage(((1024, 4096), (13, 4096)), table, index_table)
        out = torch.empty_like(q)
        step.run(q, kc, vc, index_q=iq, index_k_cache=ik.view(32, 128, 128), out=out)
        a = step.plan.attention
        self.assertEqual(a.fixed_backend, 'fa4')
        self.assertEqual(a.dynamic_backend, 'fireworks')
        begin = 0
        for row, nq in enumerate((1024, 13)):
            logical = cache[table[row].long()].permute(0, 2, 1, 3).reshape(-1, 4, 256)
            logical_k, logical_v = logical.split(128, -1)
            expected, expected_lse = fixed_oracle(q[begin:begin+nq], logical_k, logical_v)
            torch.testing.assert_close(a.fixed_out[begin:begin+nq].float(), expected,
                                       atol=.025, rtol=.025)
            torch.testing.assert_close(a.fixed_lse[0, :, begin:begin+nq], expected_lse,
                                       atol=.002, rtol=.002)
            for query in (0, nq - 1):
                pos = begin + query
                reference = reference_pipeline(iq[pos:pos+1].float(), ik.float(),
                    q[pos:pos+1], logical_k, logical_v, query_offset=4096-nq+query)
                torch.testing.assert_close(out[pos:pos+1], reference[0], atol=.025, rtol=.025)
                route = a.tile_plan.dynamic_route[:, pos:pos+1]
                torch.testing.assert_close(route.sort(-1).values, reference[2].sort(-1).values,
                                           atol=0, rtol=0)
            begin += nq
        expected = out.clone()
        step.plan.capture_mega_graph()
        step.plan.run_mega()
        torch.testing.assert_close(out, expected, atol=.025, rtol=.025)



if __name__ == '__main__':
    unittest.main()
