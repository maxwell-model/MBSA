# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Finer Index KV partitions (512 / 256 blocks) for packed requests (one launch over several requests).

Packed tile rows carry request-local partition counts, so the router width only changes how many K ranges
each request is split into; the merge keeps each row's Top-16 by (score, smaller id), so the published
candidate IDs do not depend on it. Covers the binding (IndexDispatch.partitions) on packed plans
(joint 2-SM / width choice, staged rows, merge buffers), bit-identity of the packed router at PB512/PB256
with the PB1024 launch, and the public pipeline (eager, CUDA mega-graph, rebind).
"""
import unittest
from unittest import mock

import torch

from mbsa import prepare_cutedsl_pipeline
from mbsa.backends.optimized.index_policy import index_launch_shape
from mbsa.backends.optimized.src.sm100 import router
from tests.common import BackendCaseMixin, index_dispatch
from tests.regression import test_backends as backend_support  # module imports: do not re-collect tests
from tests.regression import test_two_sm as two_sm_support

SENTINEL = two_sm_support.SENTINEL
UNDERFILLED = ((512, 131072),) * 4             # PB1024: 64 Q128 CTAs on 148 SMs
FULL = ((2048, 32768),) * 4                     # one 2-SM wave already: keeps PB1024


def _bind(env, requests, dtype=torch.bfloat16, scores=None, packed_partitions=1):
    from mbsa.backends.optimized import index
    plan = mock.MagicMock()
    plan.attention.fixed_only = False
    plan.attention.query_outer = False
    plan.overlap_fixed_score = False
    plan.mxfp6_router = None
    plan.cutedsl_packed_requests = requests
    plan.cutedsl_packed_two_sm_tile_metadata = None
    plan.cutedsl_packed_fine_tile_metadata = None
    plan.cutedsl_packed_partitions = packed_partitions
    plan.cutedsl_partition_scores = scores
    plan.candidate_partition_blocks = 1024
    plan.index_q = torch.empty((sum(q for q, _ in requests), 4, 128), dtype=dtype, device='meta')
    plan.cutedsl_index_k = torch.empty((128, 1, 128), dtype=dtype, device='meta')
    with index_dispatch(**{'two_sm': '0', 'skip_fixed': '0', 'partitions': '0', 'fp8_q256': '0', **env}), \
         mock.patch.object(index, 'indexer_local_top16') as call, \
         mock.patch.object(index, '_stage_packed_tile_metadata',
                           side_effect=lambda device, reqs, blocks, tile: (('rows', blocks, tile),)) as stage:
        index.bind_score_operands(plan)
    call.assert_not_called()
    return plan, stage


class PackedPartitionBinding(unittest.TestCase):
    def test_opt_in_takes_the_joint_launch_shape(self):
        for two_sm in ('0', '1', 'auto'):
            with self.subTest(two_sm=two_sm):
                paired, blocks = index_launch_shape(UNDERFILLED, two_sm=two_sm)
                self.assertLess(blocks, 1024)
                plan, _ = _bind({'partitions': 'auto', 'two_sm': two_sm}, UNDERFILLED)
                self.assertEqual(plan.two_sm_bound, paired)
                self.assertEqual(plan.router_partition_blocks, blocks)
                self.assertEqual(plan.cutedsl_packed_router_tile_metadata, ('rows', blocks, 256 if paired else 128))
                partitions = -(-(131072 // 128) // blocks)
                self.assertEqual(plan.cutedsl_packed_router_partitions, partitions)
                self.assertEqual(tuple(plan.cutedsl_partition_scores.shape), (4, partitions, 2048, 16))
                self.assertEqual(tuple(plan.cutedsl_partition_ids.shape), (4, partitions, 2048, 16))

    def test_default_and_full_launches_keep_pb1024(self):
        for env, requests in (({}, UNDERFILLED), ({'partitions': '0'}, UNDERFILLED),
                              ({'partitions': 'auto'}, FULL)):
            with self.subTest(env=env, requests=requests[:1]):
                plan, stage = _bind(env, requests)
                self.assertIsNone(plan.cutedsl_packed_router_tile_metadata)
                self.assertIsNone(plan.cutedsl_packed_router_partitions)
                self.assertIsNone(plan.router_partition_blocks)
                for call in stage.call_args_list:
                    self.assertEqual(call.args[2], 1024)
        # E4M3 packed requests take finer widths too, Q128 rows unless fp8_q256 is "auto"
        q256, blocks = index_launch_shape(UNDERFILLED, two_sm='0', pairs=False)
        plan, _ = _bind({'partitions': 'auto'}, UNDERFILLED, dtype=torch.float8_e4m3fn)
        self.assertFalse(q256)
        self.assertEqual(plan.router_partition_blocks, blocks)
        self.assertEqual(plan.cutedsl_packed_router_tile_metadata, ('rows', blocks, 128))
        self.assertFalse(plan.fp8_q256_bound)

    def test_rebind_reuses_rows_and_restores_the_prepared_buffers(self):
        from mbsa.backends.optimized import index
        plan, stage = _bind({'partitions': 'auto'}, UNDERFILLED)
        rows, fine = plan.cutedsl_packed_router_tile_metadata, plan.cutedsl_partition_scores
        with index_dispatch(partitions='auto'), mock.patch.object(index, '_stage_packed_tile_metadata') as again:
            index.bind_score_operands(plan)      # staged rebind: same rows, same buffers
            again.assert_not_called()
            self.assertIs(plan.cutedsl_packed_router_tile_metadata, rows)
            self.assertIs(plan.cutedsl_partition_scores, fine)
            with index_dispatch(partitions='0'):
                index.bind_score_operands(plan)  # back to the prepared single-partition launch
        self.assertIsNone(plan.cutedsl_packed_router_tile_metadata)
        # The single-partition launch writes its own partition buffers, bound with the plan's backing tensors.
        for buffer in (plan.cutedsl_partition_scores, plan.cutedsl_partition_ids):
            self.assertEqual(tuple(buffer.shape), (4, 1, 2048, 16))
        self.assertIsNot(plan.cutedsl_partition_scores, fine)
        prepared = torch.empty((4, 2, 2048, 16))
        plan, _ = _bind({}, UNDERFILLED, scores=prepared, packed_partitions=2)
        self.assertIs(plan.cutedsl_partition_scores, prepared)   # the prepared layout is left alone

    def test_invalid_mode_is_rejected(self):
        from mbsa.backends.optimized.index_policy import IndexDispatch
        for value in ('1', 'on', '256'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'IndexDispatch.partitions'):
                IndexDispatch(partitions=value)

    def test_mixed_requests_keep_pb1024_while_most_ctas_would_be_empty(self):
        # B300: (8, 1M) + (8184, 8K) at PB256 launches 8032 empty of 8320 CTAs and is 3.7..5.5 % slower than
        # PB1024; (3000, 32K) + (5192, 200K) at PB256 (288 of 952 empty) is 6..15 % faster.
        mixed = ((8, 1 << 20), (8184, 8184))
        for pairs in (True, False):
            self.assertEqual(index_launch_shape(mixed, two_sm='auto', pairs=pairs), (False, 1024))
        self.assertEqual(index_launch_shape(((3000, 32768), (5192, 200000)), two_sm='auto'), (True, 256))
        self.assertLess(index_launch_shape(UNDERFILLED, two_sm='0')[1], 1024)   # equal requests: no empty CTA


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class PackedPartitionRouter(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest('UMMA Index requires SM100/SM103')

    def test_bit_identity_with_pb1024(self):
        from mbsa.backends.optimized.index import _stage_packed_tile_metadata
        for reqs, mapped in ((((512, 131072),) * 3, False), (((2048, 131072), (513, 131200), (1, 129)), True),
                             (((17, 65537), (300, 262144), (255, 8191)), True), (((1000, 263000), (256, 4096)), False)):
            with self.subTest(reqs=reqs, mapped=mapped):
                total_q = sum(q for q, _ in reqs)
                blocks = [(kv + 127) // 128 for _, kv in reqs]
                g = torch.Generator(device='cuda').manual_seed(13)
                iq = torch.randn((total_q, 4, 128), device='cuda', dtype=torch.bfloat16, generator=g)
                k = torch.randn(((sum(blocks) + 5) * 128, 1, 128), device='cuda', dtype=torch.bfloat16, generator=g)
                pages = (torch.randperm(sum(blocks) + 5, device='cuda', generator=g)[:sum(blocks)] if mapped
                         else torch.arange(sum(blocks), device='cuda')).to(torch.int32).contiguous()
                outputs = {}
                for name, tile, width, options in (
                        ('q128', 128, 1024, {}), ('q128_pb512', 128, 512, dict(_skip_fixed_blocks=True)),
                        ('q128_pb256', 128, 256, {}),
                        ('two_sm_pb256', 256, 256, dict(two_sm_support.VARIANTS['two_sm'], _skip_fixed_blocks=True))):
                    (metadata,) = _stage_packed_tile_metadata(iq.device, reqs, width, tile)
                    partitions = max(-(-b // width) for b in blocks)
                    compact = torch.full((4, 32, total_q), SENTINEL, device='cuda', dtype=torch.int32)
                    router.indexer_packed_top16(
                        iq, k, partition_blocks=width, page_indices=pages, tile_metadata=metadata,
                        partitions=partitions, out_compact_ids=compact,
                        out_scores=torch.empty((4, partitions, total_q, 16), device='cuda'),
                        out_ids=torch.empty((4, partitions, total_q, 16), device='cuda', dtype=torch.int32),
                        **options)
                    torch.cuda.synchronize()
                    outputs[name] = compact
                for name, got in outputs.items():
                    torch.testing.assert_close(got, outputs['q128'], rtol=0, atol=0, msg=name)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class PackedPartitionPublicPipeline(BackendCaseMixin, unittest.TestCase):
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

    def test_auto_matches_default_eager_graph_and_rebind(self):
        for lengths in (((1024, 65536), (17, 4113)), ((300, 131201), (17, 65537), (1, 129), (513, 262144))):
            for two_sm in ('0', 'auto'):
                with self.subTest(lengths=lengths, two_sm=two_sm):
                    _, inputs, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
                    default = tuple(t.clone() for t in self.prepare(inputs, kwargs, '0', '0', '0').run())
                    plan = self.prepare(inputs, kwargs, 'auto', two_sm)
                    state = plan.plan._state.index
                    expected = index_launch_shape(lengths, two_sm=two_sm)
                    self.assertEqual((state.two_sm_bound, state.router_partition_blocks or 1024), expected)
                    self.assertLess(expected[1], 1024)
                    self.assertEqual(state.candidate_partition_blocks, 1024)
                    self.assert_same(plan.run(), default)
                    plan.capture_mega_graph()
                    self.assert_same(plan.run_mega(), default)
                    # rebind an uncaptured plan (a captured mega graph is immutable by design)
                    plan = self.prepare(inputs, kwargs, 'auto', two_sm)
                    flipped = kwargs['index_page_indices'].flip(0).contiguous()
                    with index_dispatch(partitions='auto', two_sm=two_sm, skip_fixed='auto'):
                        plan.rebind(*inputs, index_page_indices=flipped, page_table=kwargs['page_table'],
                                    out=torch.empty_like(inputs[2]))
                    self.assertEqual(plan.plan._state.index.router_partition_blocks or 1024, expected[1])
                    reference = self.prepare(inputs, dict(kwargs, index_page_indices=flipped), '0', '0', '0')
                    self.assert_same(plan.run(), tuple(t.clone() for t in reference.run()))


if __name__ == '__main__':
    unittest.main()
