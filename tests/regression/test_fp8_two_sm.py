# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""E4M3 Q256 rows as 2-SM CTA pairs (IndexDispatch.fp8_two_sm).

With the queued Top-16 insertion bound (IndexDispatch.queued_insert), "auto" runs the E4M3 Q256 router (the dense
T=8192 geometry, or the Q256 choice of IndexDispatch.fp8_q256) as 2-SM CTA pairs: one M256 UMMA per pair, so each SM
receives half of every K tile. Same math, same per-row selection; under the power cap the saved delivery energy is
time. Covers the mode and its scope (E4M3 + queue only, local and packed), the policy, the router's dtype/combination
guards, bit-identity of local and packed launches across geometries (pair padding, causal tails, page tables, skip,
fine partitions with the tree merge, score ties, monotone scores), and the public pipeline (eager, CUDA mega-graph).
"""
import unittest

import torch

from mbsa import prepare_cutedsl_pipeline
from mbsa.backends.optimized.index_policy import (fp8_two_sm_index_preferred, FP8_TWO_SM_MIN_KV_TOKENS,
                                                  FP8_TWO_SM_MIN_PARTITION_BLOCKS)
from mbsa.backends.optimized.src.sm100 import router
from tests.common import BackendCaseMixin, baseline_index_dispatch, index_dispatch
from tests.regression import test_backends as backend_support  # module imports: do not re-collect tests
from tests.regression import test_two_sm as two_sm_support
from tests.regression.test_fine_partitions import _bind
from tests.regression.test_packed_partitions import _bind as bind_packed
from tests.regression.test_parallel_insert import _local, _tie
from tests.regression.test_queued_insert import _monotone

SENTINEL = two_sm_support.SENTINEL
E4M3 = torch.float8_e4m3fn
Q256_QI = dict(_query_tile=256, _force_early_acc_release=True, _queued_insert=True)
PAIRS_QI = dict(Q256_QI, _two_sm=True)


class Fp8TwoSmBinding(unittest.TestCase):
    def test_switch_reaches_e4m3_local_routers_with_the_queue_only(self):
        for value, queued, dtype, bound in ((None, True, E4M3, False), ('0', True, E4M3, False),
                                            ('auto', True, E4M3, True), ('auto', False, E4M3, False),
                                            ('auto', True, torch.bfloat16, False)):
            with self.subTest(value=value, queued=queued, dtype=str(dtype)):
                env = {'queued_insert': queued, **({} if value is None else {'fp8_two_sm': value})}
                with baseline_index_dispatch():
                    plan, kwargs = _bind(env, qn=8192, kv=131072, dtype=dtype)
                self.assertEqual(kwargs.get('_fp8_two_sm', False), bound)

    def test_switch_reaches_e4m3_packed_q256_launches_with_the_queue_only(self):
        requests = ((4096, 131072), (4096, 131072))
        self.assertTrue(fp8_two_sm_index_preferred(requests, partition_blocks=1024))
        on = {'fp8_two_sm': 'auto', 'queued_insert': True, 'fp8_q256': 'auto'}
        for env, dtype, bound in ((on, E4M3, True), (dict(on, fp8_two_sm='0'), E4M3, False),
                                  (dict(on, queued_insert=False), E4M3, False),
                                  (dict(on, fp8_q256='0'), E4M3, False), (on, torch.bfloat16, False)):
            with self.subTest(env=env, dtype=str(dtype)):
                plan, _ = bind_packed(env, requests, dtype=dtype)
                self.assertEqual(plan.fp8_two_sm_bound, bound)
                self.assertFalse(plan.two_sm_bound)   # the BF16 2-SM binding is separate

    def test_invalid_mode_is_rejected(self):
        from mbsa.backends.optimized.index_policy import IndexDispatch
        for value in ('1', 'on', 'yes'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'IndexDispatch.fp8_two_sm'):
                IndexDispatch(fp8_two_sm=value)

    def test_policy(self):
        self.assertTrue(fp8_two_sm_index_preferred(((8192, 1048576),), partition_blocks=2048))
        self.assertTrue(fp8_two_sm_index_preferred(((8192, FP8_TWO_SM_MIN_KV_TOKENS),), partition_blocks=2048))
        self.assertFalse(fp8_two_sm_index_preferred(((8192, FP8_TWO_SM_MIN_KV_TOKENS - 128),), partition_blocks=2048))
        self.assertFalse(fp8_two_sm_index_preferred(((8192, 8192),), partition_blocks=1024))
        self.assertTrue(fp8_two_sm_index_preferred(((2048, 131072),), partition_blocks=FP8_TWO_SM_MIN_PARTITION_BLOCKS))
        self.assertFalse(fp8_two_sm_index_preferred(((2048, 131072),),
                                                    partition_blocks=FP8_TWO_SM_MIN_PARTITION_BLOCKS // 2))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class Fp8TwoSmRouter(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest('UMMA Index requires SM100/SM103')

    @staticmethod
    def _operands(qn, kv, mapping, tie=None):
        iq, k, pages = two_sm_support._case(qn, kv, mapping)
        if tie is not None:
            k = _tie(k, tie)
        return iq.to(E4M3).contiguous(), k.to(E4M3).contiguous(), pages

    def assert_same(self, got, expected):
        for g, e in zip(got, expected):
            torch.testing.assert_close(g, e, rtol=0, atol=0, equal_nan=True)

    def test_local_pairs_are_bit_identical(self):
        for name, qn, kv, mapping, width, extra, tie in (
                ('dense_1m', 8192, 1048576, 'identity', 2048, {}, None),
                ('dense_128k_pb1024', 8192, 131072, 'identity', 1024, {}, None),
                ('k_tail', 8192, 1048575, 'identity_tail', 2048, {}, None),
                ('paged_4096', 4096, 524288, 'permuted', 2048, {}, None),
                ('odd_tiles_padding_peer', 2304, 262144, 'permuted', 1024, {}, None),
                ('short_t_one_pair', 300, 131072, 'identity', 2048, {}, None),
                ('skip_pb256', 3000, 98304, 'permuted', 256, dict(_skip_fixed_blocks=True), None),
                ('pb64_tree', 513, 1048575, 'permuted', 64, {}, None),
                ('pairs_tied', 2048, 131072, 'identity', 1024, {}, 'pairs'),
                ('all_tied', 1024, 65536, 'identity', 1024, {}, 'all')):
            for parallel in (False, True):
                with self.subTest(name=name, parallel=parallel):
                    iq, k, pages = self._operands(qn, kv, mapping, tie)
                    base = dict(extra, **({'_parallel_insert': True} if parallel else {}))
                    single, key = _local(iq, k, pages, kv, width, dict(Q256_QI, **base))
                    paired, paired_key = _local(iq, k, pages, kv, width, dict(PAIRS_QI, **base))
                    self.assertNotIn('two_sm', key)
                    self.assertIn('two_sm', paired_key)
                    self.assertIn('reg_split', paired_key)
                    self.assertEqual(int((single[0][:, 16:] == SENTINEL).sum()), 0)   # dynamic half published
                    self.assert_same(paired, single)

    def test_monotone_scores_are_bit_identical(self):
        for falling in (False, True):
            for parallel in (False, True):
                with self.subTest(falling=falling, parallel=parallel):
                    iq, k, pages = _monotone(2048, 262144, falling)
                    base = dict(_parallel_insert=True) if parallel else {}
                    reference, _ = _local(iq, k, pages, 262144, 1024, base)   # Q128, serial order of insertions
                    paired, _ = _local(iq, k, pages, 262144, 1024, dict(PAIRS_QI, **base))
                    self.assert_same(paired, reference)

    def test_wrapper_opt_in_takes_pairs_on_the_q256_geometry_only(self):
        # The dense PB2048 rule resolves Q256 at 131072 KV tokens on SM103 only (SM100: 196608 to 262144).
        sm103 = torch.cuda.get_device_capability() == (10, 3)
        for name, qn, kv, mapping, width, extra, paired in (
                ('dense_default_q256', 8192, 131072, 'identity', 2048, {}, sm103),
                ('fp8_q256_opt_in', 4096, 262144, 'permuted', 1024, dict(_fp8_q256=True), True),
                ('q128_geometry', 1024, 131072, 'permuted', 1024, {}, False),
                ('below_kv_floor', 8192, 8192, 'identity', 1024, dict(_query_tile=256), False)):
            with self.subTest(name=name):
                iq, k, pages = self._operands(qn, kv, mapping)
                base = dict(extra, _parallel_insert=True, _queued_insert=True)
                plain, key = _local(iq, k, pages, kv, width, base)
                opted, opted_key = _local(iq, k, pages, kv, width, dict(base, _fp8_two_sm=True))
                self.assertNotIn('two_sm', key)
                self.assertEqual('two_sm' in opted_key, paired)
                self.assert_same(opted, plain)
                if paired:
                    _, without_queue = _local(iq, k, pages, kv, width, dict(extra, _parallel_insert=True,
                                                                            _fp8_two_sm=True))
                    self.assertNotIn('two_sm', without_queue)   # the pairs need the queued insertion

    def test_router_guards(self):
        iq, k, pages = self._operands(2048, 131072, 'identity')
        with self.assertRaisesRegex(ValueError, 'two_sm'):   # E4M3 2-SM is validated with the queue only
            _local(iq, k, pages, 131072, 1024, dict(PAIRS_QI, _queued_insert=False))
        bf16 = two_sm_support._case(2048, 131072, 'identity')
        with self.assertRaisesRegex(ValueError, 'queued_insert'):   # BF16 2-SM keeps its chunked serial epilogue
            _local(*bf16, 131072, 1024, PAIRS_QI)
        with self.assertRaisesRegex(ValueError, 'queued_insert'):
            router.IndexerLocalTop16Sm100(partition_blocks=1024, queued_insert=True, lean192_early=True)

    def test_packed_pairs_are_bit_identical(self):
        from mbsa.backends.optimized.index import _stage_packed_tile_metadata
        for reqs in (((2048, 131072), (513, 131200), (1, 129)), ((4096, 262144), (300, 65537), (255, 8191))):
            total_q = sum(q for q, _ in reqs)
            blocks = [(kv + 127) // 128 for _, kv in reqs]
            g = torch.Generator(device='cuda').manual_seed(17)
            iq = torch.randn((total_q, 4, 128), device='cuda', dtype=torch.bfloat16, generator=g).to(E4M3)
            k = torch.randn(((sum(blocks) + 5) * 128, 1, 128), device='cuda', dtype=torch.bfloat16,
                            generator=g).to(E4M3)
            pages = torch.randperm(sum(blocks) + 5, device='cuda', generator=g)[:sum(blocks)].to(torch.int32)
            for width in (1024, 256):
                with self.subTest(reqs=reqs, width=width):
                    (metadata,) = _stage_packed_tile_metadata(iq.device, reqs, width, 256)
                    partitions = max(-(-b // width) for b in blocks)
                    outputs = []
                    for options in (Q256_QI, PAIRS_QI):
                        compact = torch.full((4, 32, total_q), SENTINEL, device='cuda', dtype=torch.int32)
                        scores = torch.full((4, partitions, total_q, 16), float('nan'), device='cuda')
                        ids = torch.full((4, partitions, total_q, 16), SENTINEL, device='cuda', dtype=torch.int32)
                        router.indexer_packed_top16(
                            iq, k, partition_blocks=width, page_indices=pages.contiguous(), tile_metadata=metadata,
                            partitions=partitions, out_compact_ids=compact, out_scores=scores, out_ids=ids,
                            _parallel_insert=True, **options)
                        torch.cuda.synchronize()
                        outputs.append((compact, scores, ids))
                    self.assert_same(outputs[1], outputs[0])


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class Fp8TwoSmPublicPipeline(BackendCaseMixin, unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest('UMMA Index requires SM100/SM103')

    def assert_same(self, actual, expected):
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=0, atol=0)

    @staticmethod
    def prepare(inputs, kwargs, **env):
        with index_dispatch(**env):
            return prepare_cutedsl_pipeline(*inputs, backend='optimized', **kwargs)

    def test_single_request_matches_default_eager_and_graph(self):
        torch.manual_seed(11)
        r = lambda s: torch.randn(s, device='cuda', dtype=torch.bfloat16)
        env = dict(fp8_q256='auto', parallel_insert=True, queued_insert=True)
        for qn, kn in ((8192, 131072), (4096, 524288)):
            with self.subTest(qn=qn, kn=kn):
                inputs = (r((qn, 4, 128)).to(E4M3), r((kn, 1, 128)).to(E4M3), r((qn, 32, 128)), r((kn, 4, 128)),
                          r((kn, 4, 128)))
                default = tuple(t.clone() for t in self.prepare(inputs, {}, fp8_two_sm='0').run())
                plan = self.prepare(inputs, {}, **env, fp8_two_sm='auto')
                self.assertTrue(plan.plan._state.index.fp8_two_sm_bound)
                self.assert_same(plan.run(), default)
                plan.capture_mega_graph()
                self.assert_same(plan.run_mega(), default)

    def test_packed_requests_match_default(self):
        lengths = ((4096, 131072), (4096, 131072))   # fp8_q256 "auto" binds the packed Q256 launch
        _, inputs, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
        inputs = (inputs[0].to(E4M3), inputs[1].to(E4M3)) + tuple(inputs[2:])
        env = dict(fp8_q256='auto', parallel_insert=True, queued_insert=True)
        default = tuple(t.clone() for t in self.prepare(inputs, kwargs, fp8_two_sm='0').run())
        plan = self.prepare(inputs, kwargs, **env, fp8_two_sm='auto')
        self.assertTrue(plan.plan._state.index.fp8_q256_bound)
        self.assertTrue(plan.plan._state.index.fp8_two_sm_bound)
        self.assert_same(plan.run(), default)
        plan.capture_mega_graph()
        self.assert_same(plan.run_mega(), default)


if __name__ == '__main__':
    unittest.main()
