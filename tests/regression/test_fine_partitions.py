# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Finer Index KV partitions (512 / 256 blocks) for single requests whose PB1024 grid leaves SMs idle.

A KV partition is an independent K range; the merge keeps each row's Top-16 by (score, smaller id), so any
split selects the same blocks. Covers the binding (IndexDispatch.partitions), its makespan
policy, bit-identity of PB512/PB256 with the PB1024 router (regular, skip and two_sm kernels, page tables,
logical tails, odd query extents) and the public pipeline (eager, CUDA mega-graph, rebind).
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
    'off_skip': dict(_skip_fixed_blocks=True),
    'two_sm': dict(two_sm_support.VARIANTS['two_sm'], _skip_fixed_blocks=True),
}


def _route(iq, k, pages, kv, kernel, partition_blocks):
    qn = iq.shape[0]
    partitions = ((kv + 127) // 128 + partition_blocks - 1) // partition_blocks
    scores = torch.empty((4, partitions, qn, 16), device='cuda', dtype=torch.float32)
    ids = torch.empty((4, partitions, qn, 16), device='cuda', dtype=torch.int32)
    compact = torch.full((4, 32, qn), SENTINEL, device='cuda', dtype=torch.int32)
    mapped = not torch.equal(pages, torch.arange(pages.numel(), device='cuda', dtype=torch.int32))
    router.indexer_local_top16(
        iq, k, partition_blocks=partition_blocks, page_indices=pages, logical_k_tokens=kv, use_page_table=mapped,
        out_scores=scores, out_ids=ids, out_compact_ids=compact,
        _fuse_single_partition_compact=partitions == 1, **KERNELS[kernel])
    torch.cuda.synchronize()
    return compact


def _bind(env, qn=1024, kv=65536, dtype=torch.bfloat16, lean_router=False, lean192_mode='auto', beside_fixed=False):
    from mbsa.backends.optimized import index
    plan = mock.MagicMock()
    plan.attention.fixed_only = False
    plan.attention.query_outer = False
    plan.overlap_fixed_score = beside_fixed
    plan.mxfp6_router = None
    plan.cutedsl_packed_tile_metadata = None
    plan.cutedsl_partition_scores = None
    plan.cutedsl_logical_k_tokens = kv
    plan.candidate_partition_blocks = 1024
    plan.lean192_mode = lean192_mode
    plan._lean192_early_router = lean_router
    plan.index_q = torch.empty((qn, 4, 128), dtype=dtype, device='meta')
    plan.cutedsl_index_k = torch.empty((kv, 1, 128), dtype=dtype, device='meta')
    with index_dispatch(**{'two_sm': '0', 'skip_fixed': '0', **env}), \
         mock.patch.object(index, 'indexer_local_top16', return_value=(None, ('k',))) as call, \
         mock.patch.object(index, '_local_top16_arguments', return_value=()), \
         mock.patch.object(index, 'lean192_compiler_supported', return_value=True):
        index.bind_score_operands(plan)
    return plan, call.call_args.kwargs


class FinePartitionGuards(unittest.TestCase):
    def test_binding_opt_in(self):
        from mbsa.backends.optimized.index_policy import index_launch_shape
        shape = lambda mode: index_launch_shape(((1024, 65536),), two_sm=mode)
        cases = (({}, '0', 1024, False), ({'partitions': '0'}, '0', 1024, False),
                 ({'partitions': 'auto'}, '0', shape('0')[1], shape('0')[0]),
                 ({'partitions': 'auto'}, 'auto', shape('auto')[1], shape('auto')[0]),
                 ({'partitions': 'auto'}, '1', shape('1')[1], True))
        self.assertTrue(all(blocks < 1024 for _, _, blocks, _ in cases[2:]))
        for env, two_sm, blocks, paired in cases:
            with self.subTest(env=env, two_sm=two_sm):
                plan, kwargs = _bind(dict(env, two_sm=two_sm))
                self.assertEqual(kwargs['partition_blocks'], blocks)
                self.assertEqual(plan.router_partition_blocks, blocks)
                self.assertEqual(plan.two_sm_bound, paired)
                partitions = -(-(65536 // 128) // blocks)
                self.assertEqual(tuple(plan.cutedsl_partition_scores.shape), (4, partitions, 1024, 16))
                self.assertEqual(kwargs['_fuse_single_partition_compact'], partitions == 1)

    def test_auto_keeps_pb1024_where_it_does_not_help(self):
        auto = {'partitions': 'auto', 'two_sm': 'auto'}
        for kwargs in (dict(qn=8192, kv=1048576),                      # two full waves already
                       dict(qn=8192, kv=65536, lean_router=True),      # Lean192 keeps PB1024
                       dict(qn=1024, kv=65536, lean192_mode='on'),
                       dict(qn=300, kv=2048)):                          # no candidates
            # (E4M3 single requests take finer widths too: tests/regression/test_fp8_policy.py)
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                plan, call = _bind(auto, **kwargs)
                self.assertEqual(call['partition_blocks'], 1024)
                self.assertEqual(plan.router_partition_blocks, 1024)

    def test_beside_fixed_attention_keeps_pb1024_where_fixed_hides_the_index(self):
        from mbsa.backends.optimized.index_policy import index_launch_shape
        # B300 layers (the Index beside Fixed attention): (2048, 16K) BF16 at PB64 is 33 % faster alone but
        # 8.9 % slower per layer, (8192, 16K) 1.1 % and (4096, 32K) 1.9 %; their PB1024 choices are -3.9 / -1.3 %.
        # (17, 256K) BF16 at PB64 is 73 % faster per layer, (1024, 256K) E4M3 at PB256 38 %.
        for pairs, (qn, kv), alone, beside in ((True, (2048, 16385), (False, 64), (False, 1024)),
                                               (True, (8192, 16385), (False, 64), (True, 1024)),
                                               (True, (4096, 32769), (True, 128), (False, 1024)),
                                               (False, (4096, 16385), (True, 64), (False, 1024)),
                                               (True, (17, 262144), (False, 64), (False, 64)),
                                               (False, (1024, 262144), (True, 256), (True, 256))):
            for beside_fixed, expected in ((False, alone), (True, beside)):
                with self.subTest(pairs=pairs, q=qn, kv=kv, beside_fixed=beside_fixed):
                    self.assertEqual(index_launch_shape(((qn, kv),), two_sm='auto', pairs=pairs,
                                                        beside_fixed=beside_fixed), expected)
        auto = {'partitions': 'auto', 'two_sm': 'auto'}
        for beside_fixed, blocks in ((False, 64), (True, 1024)):
            with self.subTest(beside_fixed=beside_fixed):
                plan, call = _bind(auto, qn=2048, kv=16385, beside_fixed=beside_fixed)
                self.assertEqual(call['partition_blocks'], blocks)
                self.assertEqual(plan.router_partition_blocks, blocks)

    def test_invalid_opt_in_value_is_rejected(self):
        for value in ('1', 'on', '512', ''):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'partitions'):
                _bind({'partitions': value})

    def test_policy_matches_measured_domain(self):
        from mbsa.backends.optimized.index_policy import index_launch_shape
        # B300 steady-state measurements: (two_sm mode, (q, kv)) -> fastest measured choice
        for mode, (qn, kv), expected in MEASURED_CHOICES:
            with self.subTest(mode=mode, q=qn, kv=kv):
                self.assertEqual(index_launch_shape(((qn, kv),), two_sm=mode), expected)


# B300 (Index kernel, 2 seeds): the measured-fastest of Q128 / 2-SM at PB1024 / 512 / 256, e.g.
# (1024, 64K) Q128 PB256 -46.8 %, (1024, 256K) 2-SM PB256 -44.5 %, (2048, 128K) 2-SM PB256 -42.9 % against
# Q128 PB1024. Q128 PB256 is 4.1..4.9 % slower wherever the model sees no makespan gain.
MEASURED_CHOICES = (
    # Widths 128 / 64 make (1024, 64K) 2-SM PB64 -34.9 % / Q128 PB128 -33.9 % vs Q128 PB256
    ('auto', (1024, 65536), (True, 64)), ('auto', (1024, 262144), (True, 256)),
    ('auto', (1024, 524288), (True, 512)), ('auto', (2048, 131072), (True, 256)),
    ('auto', (2048, 262144), (True, 512)), ('auto', (4096, 65536), (True, 256)),
    ('auto', (4096, 196608), (True, 256)), ('1', (1024, 65536), (True, 64)), ('1', (4096, 196608), (True, 256)),
    ('0', (1024, 65536), (False, 128)), ('0', (1024, 524288), (False, 1024)), ('0', (2048, 262144), (False, 1024)),
    ('0', (4096, 65536), (False, 1024)), ('0', (4096, 196608), (False, 1024)), ('0', (4096, 1048576), (False, 1024)),
    # Against the PB1024 makespan policy: T=1..3072, -6.5..-73.5 %
    # (T=1 PB64 -63.9 %, T=17 PB128 -35.8 % against PB256)
    ('auto', (1, 131072), (False, 64)), ('auto', (17, 524288), (False, 128)), ('auto', (512, 524288), (True, 256)),
    ('auto', (256, 1048576), (False, 512)), ('auto', (768, 1048576), (False, 512)),
    # (3072, 128K) 2-SM PB128 -7.7 %, the fastest; (1536, 128K) 2-SM PB64 -9.9 % against Q128
    # PB256 is 2.1 % behind the fastest 2-SM PB256 -11.8 %: a known model regret
    ('auto', (1536, 131072), (True, 64)), ('auto', (3072, 131072), (True, 128)), ('auto', (3072, 524288), (True, 512)),
    # Grids of 2..4 PB1024 waves gain 3..5 points; T=8192 keeps PB1024 (a finer split is 0.7..2.4 % slower
    # at 256K, model 0.88; 3.5 % faster at 512K, model 0.87, below the threshold)
    ('auto', (5120, 262144), (True, 256)), ('auto', (6144, 262144), (True, 512)), ('auto', (2048, 786432), (True, 512)),
    ('auto', (5120, 524288), (True, 512)), ('auto', (5120, 786432), (True, 256)), ('auto', (7168, 393216), (True, 512)),
    ('auto', (7168, 524288), (True, 256)), ('auto', (8192, 262144), (True, 1024)), ('auto', (8192, 524288), (True, 1024)),
    ('0', (5120, 262144), (False, 512)), ('0', (7168, 262144), (False, 256)),
    # T=128 / 512 at 128K PB64 -61.3 / -35.2 %; T=17 at 1M keeps PB256 (PB128 +14.5 %: 64-partition merge)
    ('auto', (128, 131072), (False, 64)), ('auto', (512, 131072), (True, 64)), ('auto', (17, 1048576), (False, 256)))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class FinePartitionRouter(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest('UMMA Index requires SM100/SM103')

    def test_bit_identity_with_pb1024(self):
        cases = ((1024, 65536, 'identity'), (1024, 262144, 'identity'), (2048, 131072, 'permuted'),
                 (1000, 131200, 'identity_tail'), (17, 1048575, 'permuted'), (257, 33024, 'identity'),
                 (1, 65536, 'identity'), (4096, 196608, 'identity'))
        for qn, kv, mapping in cases:
            with self.subTest(q=qn, kv=kv, mapping=mapping):
                iq, k, pages = two_sm_support._case(qn, kv, mapping)
                reference = _route(iq, k, pages, kv, 'off', 1024)
                for kernel in KERNELS:
                    for blocks in (512, 256):
                        torch.testing.assert_close(_route(iq, k, pages, kv, kernel, blocks), reference,
                                                   rtol=0, atol=0, msg=f'{kernel} PB{blocks}')
                got = _route(iq, k, pages, kv, 'two_sm', 256)
                self.assertTrue(bool((got[:, :16] == SENTINEL).all()))
                self.assertEqual(two_sm_support._oracle_violations(iq, k, pages, kv, got), 0)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class TreeMerge(unittest.TestCase):
    """More than 16 partitions merge with the multi-warp tree; the selection is order independent."""

    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest('UMMA Index requires SM100/SM103')

    def test_local_tree_merge_is_bit_identical(self):
        for qn, kv, mapping, blocks in ((17, 1048576, 'identity', 64), (64, 262145, 'identity_tail', 64),
                                        (300, 524288, 'permuted', 128), (1, 131072, 'identity', 64)):
            with self.subTest(q=qn, kv=kv, mapping=mapping, blocks=blocks):
                iq, k, pages = two_sm_support._case(qn, kv, mapping)
                reference = _route(iq, k, pages, kv, 'off', 1024)
                got = _route(iq, k, pages, kv, 'off_skip', blocks)
                torch.testing.assert_close(got, reference, rtol=0, atol=0)
                partitions = -(-((kv + 127) // 128) // blocks)
                compact = torch.full((4, 32, qn), SENTINEL, device='cuda', dtype=torch.int32)
                _, key = router.indexer_local_top16(
                    iq, k, partition_blocks=blocks, page_indices=pages, logical_k_tokens=kv,
                    use_page_table=mapping == 'permuted', out_compact_ids=compact, _prepare_only=True)
                self.assertEqual('tree_merge' in key, partitions > 16)
        self.assertEqual(two_sm_support._oracle_violations(iq, k, pages, kv, got), 0)

    def test_packed_tree_merge_is_bit_identical(self):
        from mbsa.backends.optimized.index import _stage_packed_tile_metadata
        reqs = ((17, 262144), (300, 131073), (1, 129))
        total_q = sum(q for q, _ in reqs)
        blocks = [(kv + 127) // 128 for _, kv in reqs]
        g = torch.Generator(device='cuda').manual_seed(17)
        iq = torch.randn((total_q, 4, 128), device='cuda', dtype=torch.bfloat16, generator=g)
        k = torch.randn((sum(blocks) * 128, 1, 128), device='cuda', dtype=torch.bfloat16, generator=g)
        pages = torch.arange(sum(blocks), device='cuda', dtype=torch.int32)
        routes = []
        for width in (1024, 64):
            (metadata,) = _stage_packed_tile_metadata(iq.device, reqs, width, 128)
            partitions = max(-(-b // width) for b in blocks)
            compact = torch.full((4, 32, total_q), SENTINEL, device='cuda', dtype=torch.int32)
            router.indexer_packed_top16(
                iq, k, partition_blocks=width, page_indices=pages, tile_metadata=metadata, partitions=partitions,
                out_compact_ids=compact, out_scores=torch.empty((4, partitions, total_q, 16), device='cuda'),
                out_ids=torch.empty((4, partitions, total_q, 16), device='cuda', dtype=torch.int32))
            torch.cuda.synchronize()
            routes.append(compact)
        self.assertGreater(partitions, 16)
        torch.testing.assert_close(routes[1], routes[0], rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class FinePartitionPublicPipeline(BackendCaseMixin, unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest('UMMA Index requires SM100/SM103')

    @staticmethod
    def prepare(inputs, kwargs, partitions, two_sm='auto', skip='auto'):
        env = {'partitions': partitions, 'two_sm': two_sm, 'skip_fixed': skip}
        with index_dispatch(**env):
            return prepare_cutedsl_pipeline(*inputs, backend='optimized', **kwargs)

    def assert_same(self, actual, expected):
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=0, atol=0)

    def test_auto_matches_default_eager_and_graph(self):
        shapes = [('dense', ((1024, 65536),)), ('dense', ((1024, 262144),)), ('dense', ((2048, 131072),)),
                  ('paged', ((1000, 131201),)), ('paged', ((17, 65536),))]
        for kind, lengths in shapes:
            for two_sm in ('0', 'auto'):
                with self.subTest(kind=kind, lengths=lengths, two_sm=two_sm):
                    if kind == 'dense':
                        inputs, kwargs = self.inputs(*lengths[0]), {}
                    else:
                        _, inputs, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
                    default = tuple(t.clone() for t in self.prepare(inputs, kwargs, '0', '0', '0').run())
                    plan = self.prepare(inputs, kwargs, 'auto', two_sm)
                    self.assertLess(plan.plan._state.index.router_partition_blocks, 1024)
                    self.assertEqual(plan.plan._state.index.candidate_partition_blocks, 1024)
                    self.assert_same(plan.run(), default)
                    plan.capture_mega_graph()
                    self.assert_same(plan.run_mega(), default)

    def test_overlapped_fixed_attention_keeps_pb1024(self):
        # Batch one runs Fixed attention beside the Index, which hides a PB1024 Index at (2048, 16K).
        lengths = ((2048, 16385),)
        _, inputs, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
        default = tuple(t.clone() for t in self.prepare(inputs, kwargs, '0', '0', '0').run())
        for overlap, blocks in ((None, 1024), (False, 64)):
            with self.subTest(overlap_fixed_score=overlap):
                plan = self.prepare(inputs, dict(kwargs, overlap_fixed_score=overlap), 'auto')
                self.assertEqual(plan.plan._state.index.overlap_fixed_score, overlap is None)
                self.assertEqual(plan.plan._state.index.router_partition_blocks, blocks)
                self.assert_same(plan.run(), default)

    def test_rebind_keeps_fine_partitions(self):
        lengths = ((1024, 262145),)
        _, inputs, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
        plan = self.prepare(inputs, kwargs, 'auto')
        blocks = plan.plan._state.index.router_partition_blocks
        self.assertLess(blocks, 1024)
        flipped = kwargs['index_page_indices'].flip(0).contiguous()
        with index_dispatch(partitions='auto', two_sm='auto', skip_fixed='auto'):
            plan.rebind(*inputs, index_page_indices=flipped, page_table=kwargs['page_table'],
                        out=torch.empty_like(inputs[2]))
        self.assertEqual(plan.plan._state.index.router_partition_blocks, blocks)
        reference = self.prepare(inputs, dict(kwargs, index_page_indices=flipped), '0', '0', '0')
        self.assert_same(plan.run(), tuple(t.clone() for t in reference.run()))

    # Packed requests: tests/regression/test_packed_partitions.py.

if __name__ == '__main__':
    unittest.main()
