# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""E4M3 Index Q256 geometry and chunked epilogue (IndexDispatch.fp8_q256).

Outside the dense T=8192 range (page tables, logical tails, other query extents) the router binds its Q128
kernel; "auto" takes the Q256 + early-release kernel where its makespan allows. The E4M3 Q256 router reads
each accumulator row with one x128 load-reduce and spills at its 168-register
cap; the 32-column chunked epilogue is spill-free and computes the same block maximum. Under the B300 power
cap (KV >= 128K) it is faster; "auto" binds it only there. Covers the binding mode, the policy domain,
bit-identity of the bound router and the public pipeline (eager, CUDA mega-graph).
"""
import unittest

import torch

from mbsa import prepare_cutedsl_pipeline
from mbsa.backends.optimized.index_policy import fp8_chunked_epilogue_preferred, fp8_q256_index_preferred
from mbsa.backends.optimized.src.sm100 import router
from tests.common import index_dispatch
from tests.regression import test_two_sm as two_sm_support
from tests.regression.test_fine_partitions import _bind


class Fp8EpilogueBinding(unittest.TestCase):
    def test_switch_passes_the_opt_in_to_e4m3_routers_only(self):
        for env, dtype, expected in (({}, torch.float8_e4m3fn, False),
                                     ({'fp8_q256': '0'}, torch.float8_e4m3fn, False),
                                     ({'fp8_q256': 'auto'}, torch.float8_e4m3fn, True),
                                     ({'fp8_q256': 'auto'}, torch.bfloat16, False)):
            with self.subTest(env=env, dtype=str(dtype)):
                with index_dispatch(**env):
                    plan, kwargs = _bind({}, qn=8192, kv=1048576, dtype=dtype)
                self.assertEqual(kwargs.get('_fp8_epilogue', False), expected)
                self.assertEqual(kwargs.get('_fp8_q256', False), expected)   # (8192, 1M): Q256 fits
                self.assertFalse(plan.fp8_epilogue_bound)   # mocked router key has no chunked epilogue

    def test_invalid_mode_is_rejected(self):
        from mbsa.backends.optimized.index_policy import IndexDispatch
        for value in ('1', 'on', ''):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'IndexDispatch.fp8_q256'):
                IndexDispatch(fp8_q256=value)

    def test_finer_partitions_choose_q128_or_single_sm_q256(self):
        from mbsa.backends.optimized.index_policy import index_launch_shape
        for fp8_mode, qn, kv in (('auto', 1024, 131072), ('auto', 17, 524288), ('0', 1024, 131072),
                                 ('auto', 4096, 1048576)):
            with self.subTest(fp8=fp8_mode, q=qn, kv=kv):
                q256, blocks = index_launch_shape(((qn, kv),), two_sm='auto' if fp8_mode == 'auto' else '0',
                                                  pairs=False)
                with index_dispatch(fp8_q256=fp8_mode):
                    plan, kwargs = _bind({'partitions': 'auto'}, qn=qn, kv=kv, dtype=torch.float8_e4m3fn)
                self.assertEqual(kwargs['partition_blocks'], blocks)
                self.assertEqual(plan.router_partition_blocks, blocks)
                self.assertEqual(kwargs.get('_fp8_q256', False), q256)
                self.assertNotIn('_two_sm', kwargs)

    def test_packed_e4m3_requests_bind_q256_rows(self):
        from tests.regression.test_packed_partitions import _bind as bind_packed
        requests = ((4096, 131072),) * 2
        self.assertTrue(fp8_q256_index_preferred(requests))
        plan, stage = bind_packed({'fp8_q256': 'auto'}, requests, dtype=torch.float8_e4m3fn)
        self.assertTrue(plan.fp8_q256_bound)
        self.assertFalse(plan.two_sm_bound)
        self.assertEqual(plan.cutedsl_packed_router_tile_metadata, ('rows', 1024, 256))
        stage.assert_called_once()
        plan, _ = bind_packed({}, requests, dtype=torch.float8_e4m3fn)
        self.assertFalse(plan.fp8_q256_bound)
        self.assertIsNone(plan.cutedsl_packed_router_tile_metadata)

    def test_q256_needs_a_filled_grid(self):
        for qn, kv in ((8192, 131072), (8192, 1048576), (4096, 1048576)):
            self.assertTrue(fp8_q256_index_preferred(((qn, kv),)), (qn, kv))
        for qn, kv in ((256, 131072), (1, 1048576)):
            self.assertFalse(fp8_q256_index_preferred(((qn, kv),)), (qn, kv))
        with index_dispatch(fp8_q256='auto'):
            plan, kwargs = _bind({}, qn=256, kv=131072, dtype=torch.float8_e4m3fn)
        self.assertNotIn('_fp8_q256', kwargs)
        self.assertTrue(kwargs['_fp8_epilogue'])

    def test_policy_domain(self):
        # B300: faster at T=8192 with early release from 16K to 256K; slower without
        # early release, at T=1024..6144, and around even beyond 256K.
        for query_tile, kv, early, rows, expected in ((256, 131072, True, 8192, True), (256, 262144, True, 8192, True),
                                                      (256, 16384, True, 8192, True), (256, 32768, False, 8192, False),
                                                      (256, 65536, False, 8192, False), (256, 393216, True, 8192, False),
                                                      (256, 1048576, True, 8192, False), (256, 8192, True, 8192, False),
                                                      (256, 262144, True, 3072, False), (256, 131072, True, 6144, False),
                                                      (128, 131072, True, 8192, False)):
            with self.subTest(query_tile=query_tile, kv=kv, early=early, rows=rows):
                self.assertEqual(fp8_chunked_epilogue_preferred(query_tile=query_tile, logical_k_tokens=kv,
                                                                early_acc_release=early, query_rows=rows), expected)
        self.assertFalse(fp8_chunked_epilogue_preferred(query_tile=256, logical_k_tokens=131072, partition_blocks=256))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class Fp8EpilogueRouter(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest('the chunked load-reduce epilogue is an SM103 path')

    def test_bound_router_is_bit_identical(self):
        for qn, kv in ((8192, 131072), (8192, 1048576), (8192, 32768)):
            with self.subTest(q=qn, kv=kv):
                iq, k, pages = two_sm_support._case(qn, kv, 'identity')
                iq, k = iq.to(torch.float8_e4m3fn).contiguous(), k.to(torch.float8_e4m3fn).contiguous()
                routes, keys = [], []
                for opt_in in (False, True):
                    compact = torch.full((4, 32, qn), two_sm_support.SENTINEL, device='cuda', dtype=torch.int32)
                    compiled, key = router.indexer_local_top16(
                        iq, k, partition_blocks=2048, page_indices=pages, logical_k_tokens=kv, use_page_table=False,
                        out_compact_ids=compact, _fuse_single_partition_compact=kv <= 262144, _prepare_only=True,
                        _fp8_epilogue=opt_in)
                    compiled(*router._local_top16_arguments(iq, k, pages, *_buffers(qn, kv), compact, kv, 2),
                             router.cuda.CUstream(torch.cuda.current_stream().cuda_stream))
                    torch.cuda.synchronize()
                    routes.append(compact)
                    keys.append(key)
                torch.testing.assert_close(routes[1], routes[0], rtol=0, atol=0)
                self.assertEqual('chunked_epilogue' in keys[1], kv == 131072)
                self.assertNotIn('chunked_epilogue', keys[0])


def _buffers(qn, kv):
    partitions = ((kv + 127) // 128 + 2047) // 2048
    return (torch.empty((4, partitions, qn, 16), device='cuda'),
            torch.empty((4, partitions, qn, 16), device='cuda', dtype=torch.int32))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class Fp8EpiloguePublicPipeline(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest('the chunked load-reduce epilogue is an SM103 path')

    def test_auto_matches_default_eager_and_graph(self):
        torch.manual_seed(7)
        qn, kn = 8192, 131072
        r = lambda s: torch.randn(s, device='cuda', dtype=torch.bfloat16)
        inputs = (r((qn, 4, 128)).to(torch.float8_e4m3fn), r((kn, 1, 128)).to(torch.float8_e4m3fn),
                  r((qn, 32, 128)), r((kn, 4, 128)), r((kn, 4, 128)))
        # the parallel and the queued Top-16 insertion supersede the chunked epilogue: pin them off here
        with index_dispatch(fp8_q256='0', parallel_insert=False, queued_insert=False):
            default = tuple(t.clone() for t in prepare_cutedsl_pipeline(*inputs, backend='optimized').run())
        with index_dispatch(fp8_q256='auto', parallel_insert=False, queued_insert=False):
            plan = prepare_cutedsl_pipeline(*inputs, backend='optimized')
        self.assertTrue(plan.plan._state.index.fp8_epilogue_bound)
        for got, expected in zip(plan.run(), default):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
        plan.capture_mega_graph()
        for got, expected in zip(plan.run_mega(), default):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
