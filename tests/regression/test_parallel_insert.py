# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Parallel Top-16 insertion (IndexDispatch.parallel_insert).

Every epilogue row keeps its Top-16 sorted under one strict order (score descending, then the smaller valid block id).
The default inserts a block with a serial bubble: slot s compares with the candidate left over by slot s-1. The
parallel insertion compares every slot with the original candidate; the beaten slots form a suffix, the first one
takes the candidate and the later ones shift down by one. The list is the same by construction, without the 16-step
dependency chain that the E4M3 512-cycle MMA period cannot hide. The binding takes it on routers with the full-row
epilogue only (every E4M3 router, BF16 single-request Q128/Q256): the 2-SM router and Lean192 read 32-column chunks,
with which the parallel network compiles to 64 registers and spills, and keep the serial bubble. Covers the binding
and its scope, bit-identity of local and packed routers across geometries and score ties (including the chunked
epilogue), and the public pipeline (eager, CUDA mega-graph).
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

SENTINEL = two_sm_support.SENTINEL
E4M3 = torch.float8_e4m3fn
Q256_EARLY = dict(_query_tile=256, _force_early_acc_release=True)


class ParallelInsertBinding(unittest.TestCase):
    def test_switch_reaches_local_routers_of_every_format(self):
        for value, dtype in ((None, torch.bfloat16), (False, torch.bfloat16), (True, torch.bfloat16), (True, E4M3),
                             (False, E4M3)):
            with self.subTest(value=value, dtype=str(dtype)), baseline_index_dispatch():
                plan, kwargs = _bind({} if value is None else {'parallel_insert': value}, dtype=dtype)
                self.assertEqual(kwargs.get('_parallel_insert', False), value is True)

    def test_two_sm_and_lean192_routers_keep_the_serial_insertion(self):
        with baseline_index_dispatch():
            _, kwargs = _bind({'parallel_insert': True, 'two_sm': '1'}, qn=8192, kv=131072)
            self.assertTrue(kwargs['_two_sm'])
            self.assertNotIn('_parallel_insert', kwargs)
            _, kwargs = _bind({'parallel_insert': True}, qn=8192, kv=131072, lean_router=True)
            self.assertTrue(kwargs['_lean192_early'])
            self.assertNotIn('_parallel_insert', kwargs)

    def test_switch_reaches_e4m3_packed_launches(self):
        requests = ((512, 131072),) * 3
        for value in (False, True):
            with self.subTest(value=value):
                plan, _ = bind_packed({'parallel_insert': value}, requests, dtype=E4M3)
                self.assertEqual(plan.parallel_insert_bound, value)
        # BF16 packed launches keep the serial insertion, 2-SM pairs or not.
        for two_sm in ('0', '1'):
            plan, _ = bind_packed({'parallel_insert': True, 'two_sm': two_sm}, requests)
            self.assertEqual(plan.two_sm_bound, two_sm == '1')
            self.assertFalse(plan.parallel_insert_bound)

    def test_invalid_value_is_rejected(self):
        from mbsa.backends.optimized.index_policy import IndexDispatch
        for value in ('auto', 'on', '1', 2):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'IndexDispatch.parallel_insert'):
                IndexDispatch(parallel_insert=value)

    def test_e4m3_chunked_epilogue_yields_to_parallel_insertion(self):
        from mbsa.backends.optimized.index_policy import fp8_chunked_epilogue_preferred
        self.assertTrue(fp8_chunked_epilogue_preferred(query_tile=256, logical_k_tokens=131072))
        self.assertFalse(fp8_chunked_epilogue_preferred(query_tile=256, logical_k_tokens=131072, parallel_insert=True))


def _tie(k, mode):
    """Copy 128-token blocks so that block maxima tie: 'all' (every block equals block 0) or 'pairs'."""
    blocks = k.view(-1, 128, 1, 128)
    if mode == 'all':
        blocks[1:] = blocks[:1]
    elif mode == 'pairs':
        blocks[1::2] = blocks[0:blocks.shape[0] - blocks.shape[0] % 2:2]
    return k


def _local(iq, k, pages, kv, partition_blocks, options):
    qn = iq.shape[0]
    partitions = ((kv + 127) // 128 + partition_blocks - 1) // partition_blocks
    scores = torch.full((4, partitions, qn, 16), float('nan'), device='cuda')
    ids = torch.full((4, partitions, qn, 16), SENTINEL, device='cuda', dtype=torch.int32)
    compact = torch.full((4, 32, qn), SENTINEL, device='cuda', dtype=torch.int32)
    mapped = not torch.equal(pages, torch.arange(pages.numel(), device='cuda', dtype=torch.int32))
    compiled, key = router.indexer_local_top16(
        iq, k, partition_blocks=partition_blocks, page_indices=pages, logical_k_tokens=kv, use_page_table=mapped,
        out_scores=scores, out_ids=ids, out_compact_ids=compact, _fuse_single_partition_compact=partitions == 1,
        _prepare_only=True, **options)
    compiled(*router._local_top16_arguments(iq, k, pages, scores, ids, compact, kv, 2),
             router.cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    return (compact,) if partitions == 1 else (compact, scores, ids), key


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class ParallelInsertRouter(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest('UMMA Index requires SM100/SM103')

    def test_local_router_is_bit_identical(self):
        two_sm = dict(two_sm_support.VARIANTS['two_sm'], _skip_fixed_blocks=True)
        for name, dtype, qn, kv, mapping, width, options, tie in (
                ('bf16_q128', torch.bfloat16, 1024, 65536, 'permuted', 1024, {}, None),
                ('bf16_q256', torch.bfloat16, 2048, 131072, 'identity', 1024, dict(_query_tile=256), None),
                ('bf16_two_sm_skip', torch.bfloat16, 2048, 262144, 'permuted', 1024, two_sm, None),
                ('bf16_lean192', torch.bfloat16, 8192, 65536, 'identity', 1024, dict(_lean192_early=True), None),
                ('bf16_pb64_tree', torch.bfloat16, 17, 524288, 'identity', 64, dict(_skip_fixed_blocks=True), None),
                ('e4m3_q128_tail', E4M3, 256, 131073, 'identity_tail', 1024, {}, None),
                ('e4m3_q256_early', E4M3, 2048, 262144, 'permuted', 1024, Q256_EARLY, None),
                ('e4m3_q256_ce', E4M3, 8192, 131072, 'identity', 1024, dict(Q256_EARLY, _chunked_epilogue=True), None),
                ('e4m3_dense_1m', E4M3, 8192, 1048576, 'identity', 2048, {}, None),
                ('bf16_all_tied', torch.bfloat16, 1024, 65536, 'identity', 1024, {}, 'all'),
                ('bf16_pairs_tied_two_sm', torch.bfloat16, 2048, 131072, 'permuted', 1024, two_sm, 'pairs'),
                ('e4m3_pairs_tied_q256', E4M3, 2048, 131072, 'identity', 1024, Q256_EARLY, 'pairs')):
            with self.subTest(name=name):
                iq, k, pages = two_sm_support._case(qn, kv, mapping)
                if tie is not None:
                    k = _tie(k, tie)
                iq, k = iq.to(dtype).contiguous(), k.to(dtype).contiguous()
                serial, key = _local(iq, k, pages, kv, width, options)
                parallel, parallel_key = _local(iq, k, pages, kv, width, dict(options, _parallel_insert=True))
                self.assertNotIn('parallel_insert', key)
                self.assertIn('parallel_insert', parallel_key)
                self.assertEqual('chunked_epilogue' in parallel_key, 'chunked_epilogue' in key)
                self.assertEqual(int((serial[0][:, 16:] == SENTINEL).sum()), 0)   # dynamic half published
                for got, expected in zip(parallel, serial):
                    torch.testing.assert_close(got, expected, rtol=0, atol=0, equal_nan=True)

    def test_e4m3_opt_in_drops_the_chunked_epilogue(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest('the chunked load-reduce epilogue is an SM103 path')
        qn, kv = 8192, 131072
        iq, k, pages = two_sm_support._case(qn, kv, 'identity')
        iq, k = iq.to(E4M3).contiguous(), k.to(E4M3).contiguous()
        serial, key = _local(iq, k, pages, kv, 1024, dict(_fp8_epilogue=True))
        parallel, parallel_key = _local(iq, k, pages, kv, 1024, dict(_fp8_epilogue=True, _parallel_insert=True))
        self.assertIn('chunked_epilogue', key)
        self.assertNotIn('chunked_epilogue', parallel_key)
        torch.testing.assert_close(parallel[0], serial[0], rtol=0, atol=0)

    def test_packed_router_is_bit_identical(self):
        from mbsa.backends.optimized.index import _stage_packed_tile_metadata
        two_sm = dict(two_sm_support.VARIANTS['two_sm'], _skip_fixed_blocks=True)
        for reqs, mapped in ((((2048, 131072), (513, 131200), (1, 129)), True),
                             (((17, 65537), (300, 262144), (255, 8191)), True)):
            total_q = sum(q for q, _ in reqs)
            blocks = [(kv + 127) // 128 for _, kv in reqs]
            g = torch.Generator(device='cuda').manual_seed(13)
            iq = torch.randn((total_q, 4, 128), device='cuda', dtype=torch.bfloat16, generator=g)
            k = torch.randn(((sum(blocks) + 5) * 128, 1, 128), device='cuda', dtype=torch.bfloat16, generator=g)
            pages = (torch.randperm(sum(blocks) + 5, device='cuda', generator=g)[:sum(blocks)] if mapped
                     else torch.arange(sum(blocks), device='cuda')).to(torch.int32).contiguous()
            for name, dtype, tile, width, options in (
                    ('bf16_q128', torch.bfloat16, 128, 1024, {}),
                    ('bf16_q128_pb64_tree', torch.bfloat16, 128, 64, dict(_skip_fixed_blocks=True)),
                    ('bf16_two_sm_pb256', torch.bfloat16, 256, 256, two_sm),
                    ('e4m3_q256', E4M3, 256, 1024, Q256_EARLY)):
                with self.subTest(reqs=reqs, name=name):
                    (metadata,) = _stage_packed_tile_metadata(iq.device, reqs, width, tile)
                    partitions = max(-(-b // width) for b in blocks)
                    q, kk = iq.to(dtype).contiguous(), k.to(dtype).contiguous()
                    outputs = []
                    for parallel in (False, True):
                        compact = torch.full((4, 32, total_q), SENTINEL, device='cuda', dtype=torch.int32)
                        scores = torch.full((4, partitions, total_q, 16), float('nan'), device='cuda')
                        ids = torch.full((4, partitions, total_q, 16), SENTINEL, device='cuda', dtype=torch.int32)
                        router.indexer_packed_top16(
                            q, kk, partition_blocks=width, page_indices=pages, tile_metadata=metadata,
                            partitions=partitions, out_compact_ids=compact, out_scores=scores, out_ids=ids,
                            **options, **({'_parallel_insert': True} if parallel else {}))
                        torch.cuda.synchronize()
                        outputs.append((compact, scores, ids))
                    for got, expected in zip(*reversed(outputs)):
                        torch.testing.assert_close(got, expected, rtol=0, atol=0, equal_nan=True)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class ParallelInsertPublicPipeline(BackendCaseMixin, unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest('UMMA Index requires SM100/SM103')

    def assert_same(self, actual, expected):
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=0, atol=0)

    @staticmethod
    def prepare(inputs, kwargs, parallel, **env):
        with index_dispatch(parallel_insert=parallel, **env):
            return prepare_cutedsl_pipeline(*inputs, backend='optimized', **kwargs)

    def test_single_request_matches_default_eager_and_graph(self):
        torch.manual_seed(7)
        r = lambda s: torch.randn(s, device='cuda', dtype=torch.bfloat16)
        # BF16 Q128 (T=4096: no Lean192, 2-SM off) and E4M3 with fp8_q256 "auto".
        for dtype, qn, env in ((torch.bfloat16, 4096, dict(two_sm='0', skip_fixed='auto')),
                               (E4M3, 8192, dict(fp8_q256='auto'))):
            with self.subTest(dtype=str(dtype)):
                kn = 131072
                inputs = (r((qn, 4, 128)).to(dtype), r((kn, 1, 128)).to(dtype), r((qn, 32, 128)), r((kn, 4, 128)),
                          r((kn, 4, 128)))
                default = tuple(t.clone() for t in self.prepare(inputs, {}, False, **env).run())
                plan = self.prepare(inputs, {}, True, **env)
                self.assertTrue(plan.plan._state.index.parallel_insert_bound)
                self.assert_same(plan.run(), default)
                plan.capture_mega_graph()
                self.assert_same(plan.run_mega(), default)

    def test_packed_requests_match_default(self):
        lengths = ((300, 131201), (17, 65537), (1, 129), (513, 262144))
        _, inputs, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
        inputs = (inputs[0].to(E4M3), inputs[1].to(E4M3)) + tuple(inputs[2:])   # BF16 packed stays serial
        env = dict(partitions='auto', two_sm='0', skip_fixed='auto')
        default = tuple(t.clone() for t in self.prepare(inputs, kwargs, False, **env).run())
        plan = self.prepare(inputs, kwargs, True, **env)
        self.assertTrue(plan.plan._state.index.parallel_insert_bound)
        self.assert_same(plan.run(), default)
        plan.capture_mega_graph()
        self.assert_same(plan.run_mega(), default)


if __name__ == '__main__':
    unittest.main()
