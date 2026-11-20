# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""two_sm Index routing: Q256 + 2-SM UMMA (cta_group::2) + early accumulator release.

Covers the IndexDispatch.two_sm binding and the router entry: guards, cache keys,
bit-identity with every existing BF16 Index kernel, a full FP32 block-max oracle over all rows and
heads, odd query extents (masked CTA pairs), page tables, logical K tails, multi-partition merges,
the public pipeline (eager, CUDA mega-graph, rebind), and packed requests (request-local CTA pairs in
one launch, their auto policy and bit-identity with the Q128 packed launch).
"""
import unittest
from unittest import mock

import torch

from mbsa import prepare_cutedsl_pipeline
from mbsa.backends.optimized.src.sm100 import router
from tests.common import BackendCaseMixin, index_dispatch
from tests.regression import test_backends as backend_support  # module import: do not re-collect its tests

SENTINEL = -7
VARIANTS = {
    'off': dict(),
    'q256': dict(_query_tile=256),
    'on': dict(_lean192_early=True),
    # the configuration the binding uses (32-column load-reduce epilogue on SM103)
    'two_sm': dict(_query_tile=256, _two_sm=True, _force_early_acc_release=True, _chunked_epilogue=True),
    'two_sm_full_row': dict(_query_tile=256, _two_sm=True, _force_early_acc_release=True),
}


def _case(qn, kv, mapping, seed=101):
    gen = torch.Generator(device='cuda').manual_seed(seed)
    blocks = (kv + 127) // 128
    extra = 3 if mapping == 'permuted' else 0
    iq = torch.randn((qn, 4, 128), device='cuda', dtype=torch.bfloat16, generator=gen)
    storage = torch.randn(((blocks + extra) * 128, 1, 128), device='cuda', dtype=torch.bfloat16, generator=gen)
    if mapping == 'permuted':
        pages = torch.randperm(blocks + extra, device='cuda', generator=gen)[:blocks].to(torch.int32).contiguous()
    else:
        pages = torch.arange(blocks, device='cuda', dtype=torch.int32)
    k = storage[:kv] if mapping == 'identity' and kv % 128 == 0 else storage
    return iq, k, pages


def _route(iq, k, pages, kv, variant):
    qn = iq.shape[0]
    partitions = ((kv + 127) // 128 + 1023) // 1024
    scores = torch.empty((4, partitions, qn, 16), device='cuda', dtype=torch.float32)
    ids = torch.empty((4, partitions, qn, 16), device='cuda', dtype=torch.int32)
    compact = torch.full((4, 32, qn), SENTINEL, device='cuda', dtype=torch.int32)
    mapped = not torch.equal(pages, torch.arange(pages.numel(), device='cuda', dtype=torch.int32))
    router.indexer_local_top16(
        iq, k, partition_blocks=1024, page_indices=pages, logical_k_tokens=kv, use_page_table=mapped,
        out_scores=scores, out_ids=ids, out_compact_ids=compact,
        _fuse_single_partition_compact=partitions == 1, **VARIANTS[variant])
    torch.cuda.synchronize()
    return compact


def _oracle_violations(iq, k, pages, kv, compact, tol=1e-4):
    """Rows (row, head) whose dynamic Top-16 is not a valid FP32 block-max Top-16 within `tol`."""
    qn = iq.shape[0]
    pos = torch.arange(qn, device='cuda') + (kv - qn)
    local_begin = torch.clamp(pos // 128 + 1 - 8, min=8)
    want = (local_begin - 8).clamp(min=0, max=16)
    width = int(local_begin.max())
    blocks = (kv + 127) // 128
    cols = torch.arange(width, device='cuda')
    candidate = (cols[None] >= 8) & (cols[None] < local_begin[:, None])
    paged = k.view(-1, 128, 128)
    bad = 0
    for head in range(4):
        q = iq[:, head].float()
        block_max = torch.full((qn, width), -float('inf'), device='cuda')
        for c0 in range(0, min(width, blocks), 256):
            c1 = min(width, blocks, c0 + 256)
            keys = paged[pages[c0:c1].long()].reshape(-1, 128).float()
            block_max[:, c0:c1] = (q @ keys.T).view(qn, c1 - c0, 128).amax(-1)
        score = torch.where(candidate, block_max * (128 ** -0.5), torch.full_like(block_max, -float('inf')))
        got = compact[head, 16:32].T.long()
        valid = got >= 0
        clamped = got.clamp(0, width - 1)
        in_range = (got >= 8) & (got < local_begin[:, None])
        ordered = torch.sort(torch.where(valid, got, torch.full_like(got, -2 ** 40)), -1).values
        duplicate = ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any(-1)
        chosen = torch.where(valid, torch.gather(score, 1, clamped), torch.full((qn, 16), float('inf'), device='cuda'))
        rest = score.scatter(1, torch.where(valid, clamped, torch.zeros_like(clamped)), -float('inf'))
        margin = chosen.min(-1).values - rest.max(-1).values
        wrong = (valid.sum(-1) != want) | (valid & ~in_range).any(-1) | duplicate
        wrong |= torch.isfinite(rest.max(-1).values) & (valid.sum(-1) > 0) & (margin < -tol)
        bad += int(wrong.sum())
    return bad


class TwoSmGuards(unittest.TestCase):
    def test_descriptor_rejects_lean192(self):
        with self.assertRaisesRegex(ValueError, 'two_sm'):
            router.IndexerLocalTop16Sm100(partition_blocks=1024, two_sm=True, lean192_early=True)
        router.IndexerLocalTop16Sm100(partition_blocks=1024, two_sm=True, packed_requests=True, query_tile=256)

    def test_binding_opt_in_policy(self):
        from mbsa.backends.optimized import index
        cases = (({}, 'auto', torch.bfloat16, False), ({'two_sm': '0'}, 'auto', torch.bfloat16, False),
                 ({'two_sm': '1'}, 'auto', torch.bfloat16, True),
                 ({'two_sm': '1'}, 'off', torch.bfloat16, True),
                 ({'two_sm': '1'}, 'on', torch.bfloat16, False),
                 ({'two_sm': '1'}, 'auto', torch.float8_e4m3fn, False))
        for env, mode, dtype, expected in cases:
            with self.subTest(env=env, mode=mode, dtype=dtype):
                plan = mock.MagicMock()
                plan.attention.fixed_only = False
                plan.attention.query_outer = False
                plan.overlap_fixed_score = False
                plan.mxfp6_router = None
                plan.cutedsl_packed_tile_metadata = None
                plan.cutedsl_logical_k_tokens = 8192
                plan.candidate_partition_blocks = 1024
                plan.lean192_mode = mode
                plan._lean192_early_router = True
                plan.index_q.dtype = plan.cutedsl_index_k.dtype = dtype
                with index_dispatch(**env), \
                     mock.patch.object(index, 'indexer_local_top16', return_value=(None, ('k',))) as call, \
                     mock.patch.object(index, '_local_top16_arguments', return_value=()), \
                     mock.patch.object(index, 'lean192_compiler_supported', return_value=True):
                    index.bind_score_operands(plan)
                kwargs = call.call_args.kwargs
                self.assertEqual(kwargs.get('_two_sm', False), expected)
                self.assertEqual(plan.two_sm_bound, expected)
                self.assertEqual(plan.lean192_bound, not expected)
                self.assertEqual(kwargs['_lean192_early'], not expected)
                if expected:
                    self.assertEqual((kwargs['_query_tile'], kwargs['_force_early_acc_release'],
                                      kwargs['_chunked_epilogue']), (256, True, True))


    def test_auto_policy_matches_measured_domain(self):
        from mbsa.backends.optimized.index_policy import two_sm_index_preferred

        def preferred(qn, kv, dtype=torch.bfloat16, partition_blocks=1024):
            q = torch.empty((qn, 4, 128), dtype=dtype, device='meta')
            k = torch.empty((kv, 1, 128), dtype=dtype, device='meta')
            return two_sm_index_preferred(q, k, partition_blocks=partition_blocks, logical_k_tokens=kv)

        # B300 steady-state measurements: faster / slower or tied against the skip
        # binding. (6144, 196608) and (4096, 196608) have a half-filled last KV partition.
        for qn, kv in ((8192, 12288), (8192, 16384), (8192, 131200), (8192, 262144), (8192, 1048576),
                       (7168, 262144), (6144, 16384), (6144, 65536), (6144, 196608), (6144, 393216),
                       (6144, 1048576), (5120, 65536), (4096, 262144), (4096, 1048576), (3072, 262144),
                       (3072, 393216), (2048, 524288), (2048, 1048576), (1024, 1048576)):
            self.assertTrue(preferred(qn, kv), (qn, kv))
        for qn, kv in ((8192, 8192), (8192, 8320), (6144, 262144), (5120, 262144), (4096, 393216),
                       (4096, 196608), (4096, 131072), (4096, 16384), (2048, 262144), (1536, 1048576),
                       (1024, 524288), (512, 1048576), (1000, 131200), (17, 131072), (1, 128)):
            self.assertFalse(preferred(qn, kv), (qn, kv))
        self.assertFalse(preferred(8192, 1048576, dtype=torch.float8_e4m3fn))
        self.assertFalse(preferred(8192, 1048576, partition_blocks=2048))

    def test_packed_binding_opt_in_policy(self):
        from mbsa.backends.optimized import index
        inside = ((2048, 32768),) * 4    # one 2-SM wave against two Q128 waves
        outside = ((4096, 131072),) + ((1024, 4096),) * 4   # the long request fits one Q128 wave
        cases = (('0', inside, torch.bfloat16, False), ('1', outside, torch.bfloat16, True),
                 ('auto', inside, torch.bfloat16, True), ('auto', outside, torch.bfloat16, False),
                 ('1', inside, torch.float8_e4m3fn, False))
        for mode, requests, dtype, expected in cases:
            with self.subTest(mode=mode, requests=requests, dtype=dtype):
                plan = mock.MagicMock()
                plan.attention.fixed_only = False
                plan.attention.query_outer = False
                plan.overlap_fixed_score = False
                plan.mxfp6_router = None
                plan.cutedsl_packed_requests = requests
                plan.cutedsl_packed_two_sm_tile_metadata = None
                plan.candidate_partition_blocks = 1024
                plan.index_q.device = torch.device('meta')   # policy on the B300 SM count
                plan.index_q.dtype = plan.cutedsl_index_k.dtype = dtype
                with index_dispatch(two_sm=mode), \
                     mock.patch.object(index, 'indexer_local_top16') as call, \
                     mock.patch.object(index, '_stage_packed_tile_metadata', return_value=('pairs',)) as stage:
                    index.bind_score_operands(plan)
                    index.bind_score_operands(plan)   # staged rebind reuses the 2-SM rows
                call.assert_not_called()
                self.assertEqual(plan.two_sm_bound, expected)
                self.assertFalse(plan.skip_fixed_blocks_bound)
                if expected:
                    stage.assert_called_once_with(plan.index_q.device, requests, 1024, 256)
                    self.assertEqual(plan.cutedsl_packed_two_sm_tile_metadata, 'pairs')
                else:
                    stage.assert_not_called()

    def test_packed_auto_policy_matches_measured_domain(self):
        from mbsa.backends.optimized.index_policy import packed_two_sm_index_preferred

        def preferred(*requests):
            return packed_two_sm_index_preferred(requests, partition_blocks=1024)

        # B300 steady-state measurements: 5..22 % faster / tied or slower than the skipping
        # Q128 packed launch. 9 and 16 x (1024, 16K) fill whole 2-SM waves; 10..12 leave a tail wave.
        for requests in (((1024, 16384),) * 9, ((1024, 16384),) * 16, ((1024, 131072),) * 5,
                         ((4096, 131072),) * 2, ((2048, 32768),) * 4, ((2048, 8192),) * 4, ((512, 4096),) * 16,
                         ((1536, 16384),) * 4, ((4096, 524288),) * 2, ((4096, 262144),) + ((1024, 8192),) * 4,
                         ((1024, 262144),) * 3 + ((4096, 16384),)):
            self.assertTrue(preferred(*requests), requests)
        for requests in (((1024, 16384),) * 2, ((1024, 16384),) * 10, ((1024, 16384),) * 12,
                         ((2048, 65536),) * 6, ((2048, 262144),) * 3, ((4096, 131072),) + ((1024, 4096),) * 4,
                         ((2048, 131072),) * 2 + ((512, 4096),) * 8, ((1024, 131072),) * 4,
                         ((128, 65536),) * 16, ((640, 16384),) * 12, ((768, 65536),) * 6):
            self.assertFalse(preferred(*requests), requests)
        self.assertFalse(packed_two_sm_index_preferred(((2048, 32768),) * 4, partition_blocks=2048))

    def test_invalid_mode_is_rejected(self):
        from mbsa.backends.optimized.index_policy import IndexDispatch
        for value in ('on', 'true', '2', ''):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'IndexDispatch.two_sm'):
                IndexDispatch(two_sm=value)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class TwoSmRouter(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest('2-SM UMMA requires SM100/SM103')
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, 'allow_tf32', old)

    def test_non_bf16_is_rejected_before_compilation(self):
        iq, k, pages = _case(512, 8192, 'identity')
        with mock.patch.object(router.cute, 'compile', side_effect=AssertionError('compiled')), \
             self.assertRaisesRegex(ValueError, 'BF16'):
            _route(iq.half(), k.half(), pages, 8192, 'two_sm')

    def test_two_sm_requires_the_q256_pair_tiles(self):
        iq, k, pages = _case(512, 8192, 'identity')
        with mock.patch.object(router.cute, 'compile', side_effect=AssertionError('compiled')), \
             self.assertRaisesRegex(ValueError, '_query_tile=256'):
            router.indexer_local_top16(iq, k, partition_blocks=1024, page_indices=pages, logical_k_tokens=8192,
                                       _two_sm=True)

    def test_compact_partition_buffers_must_match_the_launch(self):
        iq, k, pages = _case(512, 65536, 'identity')
        buffers = dict(out_scores=torch.empty((4, 3, 512, 16), device='cuda'),
                       out_ids=torch.empty((4, 3, 512, 16), device='cuda', dtype=torch.int32),
                       out_compact_ids=torch.empty((4, 32, 512), device='cuda', dtype=torch.int32))
        with mock.patch.object(router.cute, 'compile', side_effect=AssertionError('compiled')), \
             self.assertRaisesRegex(ValueError, 'out_scores / out_ids'):
            router.indexer_local_top16(iq, k, partition_blocks=256, page_indices=pages, logical_k_tokens=65536,
                                       **buffers)   # 65536 tokens at PB256 launch 2 partitions, not 3

    def test_bit_identity_and_full_oracle_across_kv_and_query_extents(self):
        cases = ((8192, 8192, 'identity'), (8192, 131072, 'identity'), (8192, 131200, 'identity'),
                 (8192, 1048576, 'identity'), (8192, 1048575, 'permuted'), (8192, 32768, 'permuted'),
                 (8192, 65537, 'identity_tail'), (1, 128, 'identity'), (17, 4113, 'identity_tail'),
                 (257, 8192, 'identity'), (513, 65536, 'permuted'), (1000, 131200, 'identity'),
                 (4096, 1048576, 'identity'))
        for qn, kv, mapping in cases:
            with self.subTest(q=qn, kv=kv, mapping=mapping):
                iq, k, pages = _case(qn, kv, mapping)
                got = _route(iq, k, pages, kv, 'two_sm')
                for variant in ('off', 'q256', 'on', 'two_sm_full_row'):
                    torch.testing.assert_close(got, _route(iq, k, pages, kv, variant), rtol=0, atol=0)
                torch.testing.assert_close(got, _route(iq, k, pages, kv, 'two_sm'), rtol=0, atol=0)
                self.assertTrue(bool((got[:, :16] == SENTINEL).all()))
                self.assertEqual(_oracle_violations(iq, k, pages, kv, got), 0)

    def test_cache_key_isolates_two_sm(self):
        iq, k, pages = _case(8192, 65536, 'identity')
        keys = {}
        for variant in ('q256', 'two_sm'):
            compact = torch.full((4, 32, 8192), SENTINEL, device='cuda', dtype=torch.int32)
            scores = torch.empty((4, 1, 8192, 16), device='cuda')
            ids = torch.empty((4, 1, 8192, 16), device='cuda', dtype=torch.int32)
            _, keys[variant] = router.indexer_local_top16(
                iq, k, partition_blocks=1024, page_indices=pages, logical_k_tokens=65536, use_page_table=False,
                out_scores=scores, out_ids=ids, out_compact_ids=compact, _prepare_only=True,
                _fuse_single_partition_compact=True, **VARIANTS[variant])
        self.assertNotIn('two_sm', keys['q256'])
        self.assertIn('two_sm', keys['two_sm'])
        self.assertEqual(keys['two_sm'][-2:], ('fuse_single_partition_compact', True))


    def test_packed_bit_identity_with_q128_launch(self):
        from mbsa.backends.optimized.index import _stage_packed_tile_metadata
        # odd/short/empty-peer requests, logical tails, multi-partition and mapped pages in one launch
        for reqs, mapped in ((((17, 1280), (129, 1281), (300, 2048), (1, 1152), (1024, 9000)), False),
                             (((1, 129), (255, 8191), (4097, 12289)), True),
                             (((513, 131200), (64, 1088), (2048, 32768)), False),
                             (((1000, 263000), (256, 4096)), True)):
            with self.subTest(reqs=reqs, mapped=mapped):
                total_q = sum(q for q, _ in reqs)
                blocks = [(kv + 127) // 128 for _, kv in reqs]
                partitions = max((b + 1023) // 1024 for b in blocks)
                g = torch.Generator(device='cuda').manual_seed(11)
                iq = torch.randn((total_q, 4, 128), device='cuda', dtype=torch.bfloat16, generator=g)
                k = torch.randn(((sum(blocks) + 5) * 128, 1, 128), device='cuda', dtype=torch.bfloat16, generator=g)
                pages = (torch.randperm(sum(blocks) + 5, device='cuda', generator=g)[:sum(blocks)] if mapped
                         else torch.arange(sum(blocks), device='cuda')).to(torch.int32).contiguous()
                outputs = {}
                for name, tile, options in (('q128', 128, {}), ('two_sm', 256, VARIANTS['two_sm']),
                                            ('two_sm_skip', 256, dict(VARIANTS['two_sm'], _skip_fixed_blocks=True))):
                    (metadata,) = _stage_packed_tile_metadata(iq.device, reqs, 1024, tile)
                    bufs = (torch.full((4, partitions, total_q, 16), float('nan'), device='cuda'),
                            torch.full((4, partitions, total_q, 16), SENTINEL, device='cuda', dtype=torch.int32),
                            torch.full((4, 32, total_q), SENTINEL, device='cuda', dtype=torch.int32))
                    router.indexer_packed_top16(
                        iq, k, partition_blocks=1024, page_indices=pages, tile_metadata=metadata,
                        partitions=partitions, out_scores=bufs[0], out_ids=bufs[1], out_compact_ids=bufs[2],
                        **options)
                    torch.cuda.synchronize()
                    outputs[name] = (bufs[0].view(torch.int32), bufs[1], bufs[2])
                for name in ('two_sm', 'two_sm_skip'):
                    for got, expected in zip(outputs[name], outputs['q128']):
                        torch.testing.assert_close(got, expected, rtol=0, atol=0, msg=name)

    def test_packed_two_sm_requires_whole_pairs(self):
        iq, k, _ = _case(256, 4096, 'identity')
        metadata = torch.tensor([[0, 256, 0, 0, 4096, 1]], device='cuda', dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, 'pair'):
            router.indexer_packed_top16(
                iq, k, partition_blocks=1024, page_indices=torch.arange(32, device='cuda', dtype=torch.int32),
                tile_metadata=metadata, partitions=1, **VARIANTS['two_sm'])


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class TwoSmPublicPipeline(BackendCaseMixin, unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest('2-SM UMMA requires SM100/SM103')

    @staticmethod
    def prepare(inputs, kwargs, opt_in, **extra):
        with index_dispatch(two_sm='1' if opt_in else '0'):
            return prepare_cutedsl_pipeline(*inputs, backend='optimized', **kwargs, **extra)

    def assert_same(self, actual, expected):
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=0, atol=0)

    def test_opt_in_matches_default_eager_and_graph(self):
        shapes = [('dense', ((8192, 32768),)), ('dense', ((8192, 1048576),)), ('dense', ((257, 8192),)),
                  ('paged', ((8192, 32769),)), ('paged', ((17, 4113),))]
        for kind, lengths in shapes:
            with self.subTest(kind=kind, lengths=lengths):
                if kind == 'dense':
                    inputs, kwargs = self.inputs(*lengths[0]), {}
                else:
                    _, inputs, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
                default = tuple(t.clone() for t in self.prepare(inputs, kwargs, False).run())
                plan = self.prepare(inputs, kwargs, True)
                self.assertTrue(plan.plan._state.index.two_sm_bound)
                self.assertFalse(plan.plan._state.index.lean192_bound)
                self.assert_same(plan.run(), default)
                plan.capture_mega_graph()
                self.assert_same(plan.run_mega(), default)

    def test_auto_mode_follows_policy(self):
        for (qn, kn), expected in (((8192, 8192), False), ((8192, 32768), True), ((2048, 65536), False)):
            with self.subTest(q=qn, kv=kn):
                inputs = self.inputs(qn, kn)
                default = tuple(t.clone() for t in self.prepare(inputs, {}, False).run())
                with index_dispatch(two_sm='auto'):
                    plan = prepare_cutedsl_pipeline(*inputs, backend='optimized')
                self.assertEqual(plan.plan._state.index.two_sm_bound, expected)
                self.assert_same(plan.run(), default)

    def test_forced_lean192_on_keeps_lean192(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest('lean192="on" requires SM103')
        inputs = self.inputs(8192, 32768)
        plan = self.prepare(inputs, {}, True, lean192='on')
        self.assertFalse(plan.plan._state.index.two_sm_bound)
        auto = self.prepare(inputs, {}, True)
        self.assertTrue(auto.plan._state.index.two_sm_bound)
        self.assert_same(auto.run(), tuple(t.clone() for t in plan.run()))

    def test_rebind_with_new_page_mapping(self):
        lengths = ((8192, 32769),)
        data, inputs, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
        plan = self.prepare(inputs, kwargs, True)
        before = tuple(t.clone() for t in plan.run())
        flipped = kwargs['index_page_indices'].flip(0).contiguous()
        with index_dispatch(two_sm='1'):
            plan.rebind(*inputs, index_page_indices=flipped, page_table=kwargs['page_table'],
                        out=torch.empty_like(inputs[2]))
        self.assertTrue(plan.plan._state.index.two_sm_bound)
        after = tuple(t.clone() for t in plan.run())
        reference = self.prepare(inputs, dict(kwargs, index_page_indices=flipped), False)
        self.assert_same(after, tuple(t.clone() for t in reference.run()))
        self.assertFalse(torch.equal(before[2], after[2]))

    def test_packed_requests_match_default(self):
        for lengths, mode, expected in ((((4096, 65536), (17, 4113)), '1', True),
                                        (((2048, 32768),) * 4, 'auto', True),
                                        (((4096, 131072), (1024, 4096), (1024, 4096)), 'auto', False)):
            with self.subTest(lengths=lengths, mode=mode):
                _, inputs, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
                default = tuple(t.clone() for t in self.prepare(inputs, kwargs, False).run())
                with index_dispatch(two_sm=mode):
                    plan = prepare_cutedsl_pipeline(*inputs, backend='optimized', **kwargs)
                self.assertEqual(plan.plan._state.index.two_sm_bound, expected)
                self.assert_same(plan.run(), default)
                plan.capture_mega_graph()
                self.assert_same(plan.run_mega(), default)

    def test_packed_rebind_with_new_page_mapping(self):
        lengths = ((4096, 65536), (17, 4113))
        data, inputs, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
        plan = self.prepare(inputs, kwargs, True)
        pairs = plan.plan._state.index.cutedsl_packed_two_sm_tile_metadata
        before = tuple(t.clone() for t in plan.run())
        flipped = kwargs['index_page_indices'].flip(0).contiguous()
        with index_dispatch(two_sm='1'):
            plan.rebind(*inputs, index_page_indices=flipped, page_table=kwargs['page_table'],
                        out=torch.empty_like(inputs[2]))
        self.assertTrue(plan.plan._state.index.two_sm_bound)
        self.assertIs(plan.plan._state.index.cutedsl_packed_two_sm_tile_metadata, pairs)
        after = tuple(t.clone() for t in plan.run())
        reference = self.prepare(inputs, dict(kwargs, index_page_indices=flipped), False)
        self.assert_same(after, tuple(t.clone() for t in reference.run()))
        self.assertFalse(torch.equal(before[2], after[2]))


if __name__ == '__main__':
    unittest.main()
