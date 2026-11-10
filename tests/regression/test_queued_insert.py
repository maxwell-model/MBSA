# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Queued Top-16 insertion (IndexDispatch.queued_insert).

A candidate that beats the row's current 16th entry joins a per-row queue (depth 3) instead of entering the list at
once. The warp runs the insertion network at most once per KV block, when some row's queue is full, inserting every
row's queue head after re-checking it against the current list, and drains the queues at the end of the partition.
The 16th entry only rises, so a candidate it rejects can never reach the final Top-16: the list is the same by
construction, with far fewer warp-wide network passes (E4M3 random scores: about 13 % of the blocks instead of 48 %).
On the Q256 router the queue and the full-row TMEM load do not fit the 168-register cap of ten warps, so the router
pads to three warpgroups and moves registers with setmaxnreg (epilogue 232, MMA/TMA/idle 40). The binding takes it on
E4M3 routers only (BF16 measures flat) and never with the BF16 2-SM router or Lean192 (the E4M3 2-SM pairs are covered
by test_fp8_two_sm). Covers the binding and its scope,
bit-identity of local and packed routers across geometries, score ties and monotone scores (every block enters the
Top-16, so the queues fill every block; or only the first blocks do), serial and parallel networks, and the public
pipeline (eager, CUDA mega-graph).
"""
import unittest

import torch

from mbsa import prepare_cutedsl_pipeline
from mbsa.backends.optimized.src.sm100 import router
from tests.common import BackendCaseMixin, baseline_index_dispatch, index_dispatch
from tests.regression import test_backends as backend_support  # module imports: do not re-collect tests
from tests.regression import test_two_sm as two_sm_support
from tests.regression.test_fine_partitions import _bind
from tests.regression.test_packed_partitions import _bind as bind_packed
from tests.regression.test_parallel_insert import _local, _tie

SENTINEL = two_sm_support.SENTINEL
E4M3 = torch.float8_e4m3fn
Q256_EARLY = dict(_query_tile=256, _force_early_acc_release=True)


def _monotone(qn, kv, falling):
    """Q = 1 and the K rows of logical block b sum to b (or nb - 1 - b) exactly in E4M3, so every (head, row) sees
    strictly rising (falling) block scores."""
    nb = kv // 128
    b = torch.arange(nb, device='cuda')
    v = nb - 1 - b if falling else b
    n256, n16, s = v // 256, v % 256 // 16, v % 16
    d = torch.arange(128, device='cuda')[None]
    row = (256 * (d < n256[:, None]) + 16 * ((d >= n256[:, None]) & (d < (n256 + n16)[:, None]))
           + s[:, None] * (d == (n256 + n16)[:, None])).float()
    k = row[:, None, None, :].expand(nb, 128, 1, 128).reshape(kv, 1, 128)
    iq = torch.ones((qn, 4, 128), device='cuda')
    return iq.to(E4M3), k.to(E4M3).contiguous(), torch.arange(nb, device='cuda', dtype=torch.int32)


class QueuedInsertBinding(unittest.TestCase):
    def test_switch_reaches_e4m3_local_routers_only(self):
        for value, dtype, bound in ((None, E4M3, False), (False, E4M3, False), (True, E4M3, True),
                                    (True, torch.bfloat16, False)):
            with self.subTest(value=value, dtype=str(dtype)), baseline_index_dispatch():
                plan, kwargs = _bind({} if value is None else {'queued_insert': value}, dtype=dtype)
                self.assertEqual(kwargs.get('_queued_insert', False), bound)

    def test_switch_reaches_e4m3_packed_launches_only(self):
        requests = ((512, 131072),) * 3
        for value, dtype, bound in ((False, E4M3, False), (True, E4M3, True), (True, torch.bfloat16, False)):
            with self.subTest(value=value, dtype=str(dtype)):
                plan, _ = bind_packed({'queued_insert': value}, requests, dtype=dtype)
                self.assertEqual(plan.queued_insert_bound, bound)

    def test_invalid_value_is_rejected(self):
        from mbsa.backends.optimized.index_policy import IndexDispatch
        for value in ('auto', 'on', '1', 2):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'IndexDispatch.queued_insert'):
                IndexDispatch(queued_insert=value)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class QueuedInsertRouter(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest('UMMA Index requires SM100/SM103')

    def test_local_router_is_bit_identical(self):
        for name, qn, kv, mapping, width, options, tie in (
                ('q128_tail', 256, 131073, 'identity_tail', 1024, {}, None),
                ('q128_small_t_fused', 300, 131072, 'identity', 2048, {}, None),
                ('q128_pb64_tree', 17, 524288, 'identity', 64, dict(_skip_fixed_blocks=True), None),
                ('q256_early_permuted', 2048, 262144, 'permuted', 1024, Q256_EARLY, None),
                ('q256_dense_1m', 8192, 1048576, 'identity', 2048, {}, None),
                ('q256_pairs_tied', 2048, 131072, 'identity', 1024, Q256_EARLY, 'pairs'),
                ('q128_all_tied', 1024, 65536, 'identity', 1024, {}, 'all')):
            for parallel in (False, True):
                with self.subTest(name=name, parallel=parallel):
                    iq, k, pages = two_sm_support._case(qn, kv, mapping)
                    if tie is not None:
                        k = _tie(k, tie)
                    iq, k = iq.to(E4M3).contiguous(), k.to(E4M3).contiguous()
                    base = dict(options, **({'_parallel_insert': True} if parallel else {}))
                    direct, key = _local(iq, k, pages, kv, width, base)
                    queued, queued_key = _local(iq, k, pages, kv, width, dict(base, _queued_insert=True))
                    self.assertNotIn('queued_insert', key)
                    self.assertIn('queued_insert', queued_key)
                    self.assertEqual('reg_split' in queued_key, ('query_tile', 256) in tuple(zip(key, key[1:])))
                    self.assertEqual(int((direct[0][:, 16:] == SENTINEL).sum()), 0)   # dynamic half published
                    for got, expected in zip(queued, direct):
                        torch.testing.assert_close(got, expected, rtol=0, atol=0, equal_nan=True)

    def test_monotone_scores_are_bit_identical(self):
        for name, qn, kv, width, options in (('q128', 300, 262144, 2048, {}), ('q128_pb1024', 1024, 131072, 1024, {}),
                                             ('q256_early', 2048, 262144, 1024, Q256_EARLY)):
            for falling in (False, True):
                for parallel in (False, True):
                    with self.subTest(name=name, falling=falling, parallel=parallel):
                        iq, k, pages = _monotone(qn, kv, falling)
                        base = dict(options, **({'_parallel_insert': True} if parallel else {}))
                        direct, _ = _local(iq, k, pages, kv, width, base)
                        queued, _ = _local(iq, k, pages, kv, width, dict(base, _queued_insert=True))
                        for got, expected in zip(queued, direct):
                            torch.testing.assert_close(got, expected, rtol=0, atol=0, equal_nan=True)

    def test_router_rejects_lean192(self):
        with self.assertRaisesRegex(ValueError, 'queued_insert'):
            router.IndexerLocalTop16Sm100(partition_blocks=1024, queued_insert=True, lean192_early=True)

    def test_opt_in_drops_the_chunked_epilogue(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest('the chunked load-reduce epilogue is an SM103 path')
        qn, kv = 8192, 131072
        iq, k, pages = two_sm_support._case(qn, kv, 'identity')
        iq, k = iq.to(E4M3).contiguous(), k.to(E4M3).contiguous()
        chunked, key = _local(iq, k, pages, kv, 1024, dict(_fp8_epilogue=True))
        queued, queued_key = _local(iq, k, pages, kv, 1024, dict(_fp8_epilogue=True, _queued_insert=True))
        self.assertIn('chunked_epilogue', key)
        self.assertNotIn('chunked_epilogue', queued_key)
        torch.testing.assert_close(queued[0], chunked[0], rtol=0, atol=0)

    def test_packed_router_is_bit_identical(self):
        from mbsa.backends.optimized.index import _stage_packed_tile_metadata
        for reqs in (((2048, 131072), (513, 131200), (1, 129)), ((17, 65537), (300, 262144), (255, 8191))):
            total_q = sum(q for q, _ in reqs)
            blocks = [(kv + 127) // 128 for _, kv in reqs]
            g = torch.Generator(device='cuda').manual_seed(13)
            iq = torch.randn((total_q, 4, 128), device='cuda', dtype=torch.bfloat16, generator=g).to(E4M3)
            k = torch.randn(((sum(blocks) + 5) * 128, 1, 128), device='cuda', dtype=torch.bfloat16,
                            generator=g).to(E4M3)
            pages = torch.randperm(sum(blocks) + 5, device='cuda', generator=g)[:sum(blocks)].to(torch.int32)
            for name, tile, width, options in (('q128', 128, 1024, {}), ('q128_pb64_tree', 128, 64, {}),
                                               ('q256', 256, 1024, Q256_EARLY)):
                with self.subTest(reqs=reqs, name=name):
                    (metadata,) = _stage_packed_tile_metadata(iq.device, reqs, width, tile)
                    partitions = max(-(-b // width) for b in blocks)
                    outputs = []
                    for queued in (False, True):
                        compact = torch.full((4, 32, total_q), SENTINEL, device='cuda', dtype=torch.int32)
                        scores = torch.full((4, partitions, total_q, 16), float('nan'), device='cuda')
                        ids = torch.full((4, partitions, total_q, 16), SENTINEL, device='cuda', dtype=torch.int32)
                        router.indexer_packed_top16(
                            iq, k, partition_blocks=width, page_indices=pages.contiguous(), tile_metadata=metadata,
                            partitions=partitions, out_compact_ids=compact, out_scores=scores, out_ids=ids,
                            _parallel_insert=True, **options, **({'_queued_insert': True} if queued else {}))
                        torch.cuda.synchronize()
                        outputs.append((compact, scores, ids))
                    for got, expected in zip(*reversed(outputs)):
                        torch.testing.assert_close(got, expected, rtol=0, atol=0, equal_nan=True)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class QueuedInsertPublicPipeline(BackendCaseMixin, unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest('UMMA Index requires SM100/SM103')

    def assert_same(self, actual, expected):
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=0, atol=0)

    @staticmethod
    def prepare(inputs, kwargs, queued, **env):
        with index_dispatch(queued_insert=queued, **env):
            return prepare_cutedsl_pipeline(*inputs, backend='optimized', **kwargs)

    def test_single_request_matches_default_eager_and_graph(self):
        torch.manual_seed(7)
        r = lambda s: torch.randn(s, device='cuda', dtype=torch.bfloat16)
        for qn, env in ((8192, dict(fp8_q256='auto', parallel_insert=True)),
                        (1024, dict(fp8_q256='auto', parallel_insert=False))):
            with self.subTest(qn=qn):
                kn = 131072
                inputs = (r((qn, 4, 128)).to(E4M3), r((kn, 1, 128)).to(E4M3), r((qn, 32, 128)), r((kn, 4, 128)),
                          r((kn, 4, 128)))
                default = tuple(t.clone() for t in self.prepare(inputs, {}, False, **env).run())
                plan = self.prepare(inputs, {}, True, **env)
                self.assertTrue(plan.plan._state.index.queued_insert_bound)
                self.assert_same(plan.run(), default)
                plan.capture_mega_graph()
                self.assert_same(plan.run_mega(), default)

    def test_packed_requests_match_default(self):
        lengths = ((300, 131201), (17, 65537), (1, 129), (513, 262144))
        _, inputs, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
        inputs = (inputs[0].to(E4M3), inputs[1].to(E4M3)) + tuple(inputs[2:])
        env = dict(partitions='auto', fp8_q256='auto', parallel_insert=True)
        default = tuple(t.clone() for t in self.prepare(inputs, kwargs, False, **env).run())
        plan = self.prepare(inputs, kwargs, True, **env)
        self.assertTrue(plan.plan._state.index.queued_insert_bound)
        self.assert_same(plan.run(), default)
        plan.capture_mega_graph()
        self.assert_same(plan.run_mega(), default)


if __name__ == '__main__':
    unittest.main()
