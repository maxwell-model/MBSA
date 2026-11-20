# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Index fixed-block skip: no UMMA for blocks that cannot be dynamic candidates of any tile row.

Prefix blocks [0, 8) and the local window of a tile's last row are never dynamic candidates, so the router may drop
them from the K loop without changing any selection. Covers the binding (IndexDispatch.skip_fixed) and its auto
policy, the router flag on every BF16 Index kernel (regular, Q256, Lean192, two_sm, packed requests), cache keys,
bit-identity with the unskipped kernels, the full FP32 oracle, short contexts without candidates, page tables, logical
tails, multi-partition merges and the public pipeline.
"""
import unittest
from unittest import mock

import torch

from mbsa import prepare_cutedsl_pipeline
from mbsa.backends.optimized.src.sm100 import router
from tests.common import BackendCaseMixin, index_dispatch
from tests.regression import test_backends as backend_support  # module imports: do not re-collect tests
from tests.regression import test_two_sm as two_sm_support

SENTINEL = two_sm_support.SENTINEL
KERNELS = {
    'off': dict(),
    'q256': dict(_query_tile=256),
    'on': dict(_lean192_early=True),
    'two_sm': dict(_query_tile=256, _two_sm=True, _force_early_acc_release=True, _chunked_epilogue=True),
}


def _route(iq, k, pages, kv, kernel, skip):
    qn = iq.shape[0]
    partitions = ((kv + 127) // 128 + 1023) // 1024
    scores = torch.empty((4, partitions, qn, 16), device='cuda', dtype=torch.float32)
    ids = torch.empty((4, partitions, qn, 16), device='cuda', dtype=torch.int32)
    compact = torch.full((4, 32, qn), SENTINEL, device='cuda', dtype=torch.int32)
    mapped = not torch.equal(pages, torch.arange(pages.numel(), device='cuda', dtype=torch.int32))
    _, key = router.indexer_local_top16(
        iq, k, partition_blocks=1024, page_indices=pages, logical_k_tokens=kv, use_page_table=mapped,
        out_scores=scores, out_ids=ids, out_compact_ids=compact, _prepare_only=True,
        _fuse_single_partition_compact=partitions == 1, _skip_fixed_blocks=skip, **KERNELS[kernel])
    router.indexer_local_top16(
        iq, k, partition_blocks=1024, page_indices=pages, logical_k_tokens=kv, use_page_table=mapped,
        out_scores=scores, out_ids=ids, out_compact_ids=compact,
        _fuse_single_partition_compact=partitions == 1, _skip_fixed_blocks=skip, **KERNELS[kernel])
    torch.cuda.synchronize()
    return compact, key


def _bind(env, dtype=torch.bfloat16, kv=32768, lean_router=True):
    from mbsa.backends.optimized import index
    plan = mock.MagicMock()
    plan.attention.fixed_only = False
    plan.overlap_fixed_score = False
    plan.mxfp6_router = None
    plan.cutedsl_packed_tile_metadata = None
    plan.cutedsl_logical_k_tokens = kv
    plan.candidate_partition_blocks = 1024
    plan.lean192_mode = 'auto'
    plan._lean192_early_router = lean_router
    plan.index_q.dtype = plan.cutedsl_index_k.dtype = dtype
    with index_dispatch(**{'skip_fixed': '0', 'two_sm': '0', 'partitions': '0', 'fp8_q256': '0', **env}), \
         mock.patch.object(index, 'indexer_local_top16', return_value=(None, ('k',))) as call, \
         mock.patch.object(index, '_local_top16_arguments', return_value=()), \
         mock.patch.object(index, 'lean192_compiler_supported', return_value=True):
        index.bind_score_operands(plan)
    return plan, call.call_args.kwargs


class SkipFixedGuards(unittest.TestCase):
    def test_binding_opt_in(self):
        for env, skip, two_sm in (({}, False, False), ({'skip_fixed': '0'}, False, False),
                                  ({'skip_fixed': '1'}, True, False),
                                  ({'skip_fixed': '1', 'two_sm': '1'}, True, True)):
            with self.subTest(env=env):
                plan, kwargs = _bind(env)
                self.assertEqual(kwargs.get('_skip_fixed_blocks', False), skip)
                self.assertEqual(plan.skip_fixed_blocks_bound, skip)
                self.assertEqual(kwargs.get('_two_sm', False), two_sm)
                self.assertEqual(kwargs['_lean192_early'], not two_sm)

    def test_auto_binds_inside_measured_domain(self):
        from mbsa.backends.optimized.index_policy import SKIP_FIXED_MAX_KV_TOKENS as limit
        for two_sm, dtype, kv, lean_router, skip in (
                ('0', torch.bfloat16, 1048576, True, True),       # Lean192
                ('1', torch.bfloat16, 1048576, False, True),      # two_sm
                ('0', torch.bfloat16, limit, False, True),        # Q128 regular
                ('0', torch.bfloat16, limit + 1, False, False),
                ('0', torch.float8_e4m3fn, 131072, False, True),  # E4M3 Q256
                ('0', torch.float8_e4m3fn, 1048576, False, False)):
            with self.subTest(two_sm=two_sm, dtype=dtype, kv=kv, lean_router=lean_router):
                plan, kwargs = _bind({'skip_fixed': 'auto', 'two_sm': two_sm},
                                     dtype=dtype, kv=kv, lean_router=lean_router)
                self.assertEqual(kwargs.get('_skip_fixed_blocks', False), skip)
                self.assertEqual(plan.skip_fixed_blocks_bound, skip)
                self.assertEqual(plan.two_sm_bound, two_sm == '1')
                self.assertEqual(plan.lean192_bound, lean_router and two_sm == '0')

    def test_packed_plans_key_auto_on_longest_request(self):
        from mbsa.backends.optimized import index
        from mbsa.backends.optimized.index_policy import SKIP_FIXED_MAX_KV_TOKENS as limit
        for mode, longest, skip in (('0', 4096, False), ('1', 1048576, True), ('auto', limit, True),
                                    ('auto', limit + 1, False)):
            with self.subTest(mode=mode, longest=longest):
                plan = mock.MagicMock()
                plan.attention.fixed_only = False
                plan.overlap_fixed_score = False
                plan.mxfp6_router = None
                plan.cutedsl_packed_requests = ((128, 4096), (64, longest))
                plan.cutedsl_packed_partitions = 1
                plan.cutedsl_partition_scores = None
                plan.index_q = torch.empty((192, 4, 128), dtype=torch.bfloat16, device='meta')
                with index_dispatch(skip_fixed=mode), \
                     mock.patch.object(index, 'indexer_local_top16') as call:
                    index.bind_score_operands(plan)
                call.assert_not_called()
                self.assertIsNone(plan._local_top16_compiled)
                self.assertEqual(plan.skip_fixed_blocks_bound, skip)
                self.assertFalse(plan.two_sm_bound)

    def test_skip_is_dtype_independent_in_binding(self):
        plan, kwargs = _bind({'skip_fixed': '1'}, dtype=torch.float8_e4m3fn)
        self.assertTrue(kwargs['_skip_fixed_blocks'])
        self.assertTrue(plan.skip_fixed_blocks_bound)

    def test_invalid_mode_is_rejected(self):
        from mbsa.backends.optimized.index_policy import IndexDispatch
        for value in ('on', 'true', '2', ''):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'IndexDispatch.skip_fixed'):
                IndexDispatch(skip_fixed=value)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class SkipFixedRouter(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest('UMMA Index requires SM100/SM103')
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, 'allow_tf32', old)

    def test_bit_identity_and_full_oracle(self):
        # short contexts (no candidates, one candidate block, partial windows), the Lean192 domain,
        # PB boundaries, page tables, logical tails and multi-partition merges
        cases = ((1, 1152, 'identity'), (17, 1280, 'identity'), (300, 2048, 'identity'),
                 (129, 1281, 'identity_tail'), (8192, 8192, 'identity'), (8192, 9216, 'identity'),
                 (8192, 16384, 'identity'), (8192, 16385, 'identity_tail'), (8192, 24576, 'permuted'),
                 (8192, 131200, 'identity'), (257, 8192, 'identity'), (8192, 1048576, 'identity'),
                 (8192, 1048575, 'permuted'), (4096, 1048576, 'identity'))
        for qn, kv, mapping in cases:
            with self.subTest(q=qn, kv=kv, mapping=mapping):
                iq, k, pages = two_sm_support._case(qn, kv, mapping)
                reference, _ = _route(iq, k, pages, kv, 'q256', False)
                lean = qn == 8192 and kv % 8192 == 0 and mapping == 'identity'
                for kernel in ('off', 'q256', 'two_sm') + (('on',) if lean else ()):
                    got, _ = _route(iq, k, pages, kv, kernel, True)
                    torch.testing.assert_close(got, reference, rtol=0, atol=0, msg=kernel)
                self.assertTrue(bool((got[:, :16] == SENTINEL).all()))
                self.assertEqual(two_sm_support._oracle_violations(iq, k, pages, kv, got), 0)

    def test_packed_bit_identity(self):
        # request-local prefix/local windows: short, empty, tail, multi-partition and mixed requests
        for reqs in (((17, 1280), (129, 1281), (300, 2048), (1, 1152), (1024, 9000)),
                     ((1, 129), (255, 8191), (4097, 12289)), ((513, 131200), (64, 1088))):
            with self.subTest(reqs=reqs):
                total_q = sum(q for q, _ in reqs)
                blocks = [(kv + 127) // 128 for _, kv in reqs]
                parts = [(b + 1023) // 1024 for b in blocks]
                g = torch.Generator(device='cuda').manual_seed(7)
                iq = torch.randn((total_q, 4, 128), device='cuda', dtype=torch.bfloat16, generator=g)
                k = torch.randn((sum(blocks) * 128, 1, 128), device='cuda', dtype=torch.bfloat16, generator=g)
                meta, q0, page = [], 0, 0
                for (q, kv), b, p in zip(reqs, blocks, parts):
                    for tile in range((q + 127) // 128):
                        meta += [q0, q0 + q, tile, page, kv, p]
                    q0 += q
                    page += b
                metadata = torch.tensor(meta, dtype=torch.int32, device='cuda').view(-1, 6).contiguous()
                pages = torch.arange(sum(blocks), dtype=torch.int32, device='cuda')
                outputs = []
                for skip in (False, True):
                    bufs = (torch.full((4, max(parts), total_q, 16), float('nan'), device='cuda'),
                            torch.full((4, max(parts), total_q, 16), SENTINEL, device='cuda', dtype=torch.int32),
                            torch.full((4, 32, total_q), SENTINEL, device='cuda', dtype=torch.int32))
                    router.indexer_packed_top16(
                        iq, k, partition_blocks=1024, page_indices=pages, tile_metadata=metadata,
                        partitions=max(parts), out_scores=bufs[0], out_ids=bufs[1], out_compact_ids=bufs[2],
                        _skip_fixed_blocks=skip)
                    torch.cuda.synchronize()
                    outputs.append((bufs[0].view(torch.int32), bufs[1], bufs[2]))
                for got, expected in zip(outputs[1], outputs[0]):
                    torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_cache_key_isolates_skip(self):
        iq, k, pages = two_sm_support._case(8192, 65536, 'identity')
        _, plain = _route(iq, k, pages, 65536, 'two_sm', False)
        _, skip = _route(iq, k, pages, 65536, 'two_sm', True)
        self.assertNotIn('skip_fixed_blocks', plain)
        self.assertIn('skip_fixed_blocks', skip)
        self.assertEqual(skip[-2:], ('fuse_single_partition_compact', True))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class SkipFixedPublicPipeline(BackendCaseMixin, unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest('UMMA Index requires SM100/SM103')

    @staticmethod
    def prepare(inputs, kwargs, skip, two_sm='0'):
        env = {'skip_fixed': skip if isinstance(skip, str) else '1' if skip else '0',
               'two_sm': two_sm}
        with index_dispatch(**env):
            return prepare_cutedsl_pipeline(*inputs, backend='optimized', **kwargs)

    def assert_same(self, actual, expected):
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=0, atol=0)

    def test_opt_in_matches_default_eager_and_graph(self):
        shapes = [('dense', ((8192, 16384),)), ('dense', ((8192, 1048576),)), ('dense', ((300, 4096),)),
                  ('paged', ((8192, 16385),)), ('paged', ((17, 4113),))]
        for kind, lengths in shapes:
            for two_sm in ('0', '1', 'auto'):
                with self.subTest(kind=kind, lengths=lengths, two_sm=two_sm):
                    if kind == 'dense':
                        inputs, kwargs = self.inputs(*lengths[0]), {}
                    else:
                        _, inputs, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
                    default = tuple(t.clone() for t in self.prepare(inputs, kwargs, False).run())
                    plan = self.prepare(inputs, kwargs, True, two_sm)
                    self.assertTrue(plan.plan._state.index.skip_fixed_blocks_bound)
                    self.assert_same(plan.run(), default)
                    plan.capture_mega_graph()
                    self.assert_same(plan.run_mega(), default)

    def test_auto_follows_policy_and_matches_default(self):
        # Lean192 (skip), Q128 regular past SKIP_FIXED_MAX_KV_TOKENS (no skip), two_sm (skip)
        for (qn, kv), two_sm, skip in (((8192, 16384), '0', True), ((1024, 524288), '0', False),
                                       ((8192, 1048576), 'auto', True)):
            with self.subTest(q=qn, kv=kv, two_sm=two_sm):
                inputs = self.inputs(qn, kv)
                default = tuple(t.clone() for t in self.prepare(inputs, {}, False).run())
                plan = self.prepare(inputs, {}, 'auto', two_sm)
                self.assertEqual(plan.plan._state.index.skip_fixed_blocks_bound, skip)
                self.assert_same(plan.run(), default)

    def test_fixed_only_plan_binds_no_router(self):
        inputs = self.inputs(300, 2048)  # every row's dynamic range is empty
        plan = self.prepare(inputs, {}, True)
        self.assertIsNone(plan.plan._state.index._local_top16_compiled)
        self.assertFalse(plan.plan._state.index.skip_fixed_blocks_bound)
        self.assert_same(plan.run(), tuple(t.clone() for t in self.prepare(inputs, {}, False).run()))

    def test_packed_requests_match_default(self):
        for lengths, mode, skip in ((((4096, 65536), (17, 4113)), '1', True),
                                    (((4096, 65536), (17, 4113)), 'auto', True),
                                    (((1024, 524288), (17, 4113)), 'auto', False)):
            with self.subTest(lengths=lengths, mode=mode):
                _, inputs, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
                default = tuple(t.clone() for t in self.prepare(inputs, kwargs, False).run())
                plan = self.prepare(inputs, kwargs, mode)
                self.assertEqual(plan.plan._state.index.skip_fixed_blocks_bound, skip)
                self.assert_same(plan.run(), default)
                plan.capture_mega_graph()
                self.assert_same(plan.run_mega(), default)


if __name__ == '__main__':
    unittest.main()
