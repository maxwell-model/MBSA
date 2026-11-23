# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Optimized Index dispatch: MBSA_INDEX_DISPATCH, the SM103 default and the bound plans.

The dispatch only selects among exact Index variants. Covers mode resolution (environment, device, override
precedence and nesting), IndexDispatch validation and, on SM103, the public pipeline: an unset environment binds the
AUTO_INDEX_DISPATCH variant, "baseline" binds the baseline one, and both give bit-identical results for dense, paged
and packed BF16 / E4M3 Index operands.
"""
import contextlib
import os
import unittest
from unittest import mock

import torch

from mbsa import prepare_cutedsl_pipeline
from mbsa.backends.optimized.index_policy import (AUTO_INDEX_DISPATCH, BASELINE_INDEX_DISPATCH, IndexDispatch,
                                                  index_dispatch_override, resolve_index_dispatch)
from tests.common import BackendCaseMixin, index_dispatch
from tests.regression import test_backends as backend_support  # module import: do not re-collect its tests

E4M3 = torch.float8_e4m3fn


@contextlib.contextmanager
def _environment(value):
    """Set MBSA_INDEX_DISPATCH to `value`, or unset it for None, inside the block."""
    with mock.patch.dict(os.environ):
        os.environ.pop('MBSA_INDEX_DISPATCH', None)
        if value is not None:
            os.environ['MBSA_INDEX_DISPATCH'] = value
        yield


class DispatchResolution(unittest.TestCase):
    def resolve(self, capability, value=None):
        with _environment(value), mock.patch('torch.cuda.get_device_capability', return_value=capability):
            return resolve_index_dispatch(torch.device('cuda', 0))

    def test_default_is_auto_on_sm103_only(self):
        self.assertIs(self.resolve((10, 3)), AUTO_INDEX_DISPATCH)
        self.assertIs(self.resolve((10, 3), 'auto'), AUTO_INDEX_DISPATCH)
        self.assertIs(self.resolve((10, 3), 'baseline'), BASELINE_INDEX_DISPATCH)
        for capability in ((10, 0), (9, 0), (12, 0)):
            with self.subTest(capability=capability):
                self.assertIs(self.resolve(capability), BASELINE_INDEX_DISPATCH)

    def test_invalid_environment_is_rejected(self):
        for value in ('', 'off', '0', 'AUTO', 'default'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'MBSA_INDEX_DISPATCH'):
                self.resolve((10, 3), value)

    def test_host_only_devices_use_the_b300_policy(self):
        with _environment(None):
            self.assertIs(resolve_index_dispatch(torch.device('meta')), AUTO_INDEX_DISPATCH)

    def test_override_takes_precedence_and_nests(self):
        forced = IndexDispatch(two_sm='1')
        with index_dispatch_override(forced):
            self.assertIs(self.resolve((10, 0), 'baseline'), forced)
            with index_dispatch(partitions='0'):
                nested = self.resolve((10, 3))
                self.assertEqual((nested.two_sm, nested.partitions), ('1', '0'))
            self.assertIs(self.resolve((10, 3)), forced)
        self.assertIs(self.resolve((10, 3)), AUTO_INDEX_DISPATCH)
        with index_dispatch():
            self.assertEqual(self.resolve((10, 3)), BASELINE_INDEX_DISPATCH)
        # The baseline selection takes the partition count at run time too: serving warmup compiles its kernels.
        self.assertEqual(BASELINE_INDEX_DISPATCH, IndexDispatch(
            two_sm='0', skip_fixed='0', partitions='0', fp8_q256='0', parallel_insert=False,
            queued_insert=False, fp8_two_sm='0', runtime_partitions=True))
        with self.assertRaises(TypeError), index_dispatch_override({'two_sm': '1'}):
            pass

    def test_modes_are_validated(self):
        for field, value in (('two_sm', 'on'), ('skip_fixed', 'yes'), ('partitions', '1'), ('fp8_q256', '1'),
                             ('parallel_insert', '1'), ('queued_insert', 1), ('fp8_two_sm', '1'),
                             ('runtime_partitions', 'auto')):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, f'IndexDispatch.{field}'):
                IndexDispatch(**{field: value})
        self.assertEqual(AUTO_INDEX_DISPATCH, IndexDispatch())


class LaunchPolicyCost(unittest.TestCase):
    """Serving rebinds every layer: the launch policies are memoized and the list schedule groups equal CTAs."""

    @staticmethod
    def per_cta_schedule(requests, query_tile, heads, partition_blocks, sm_count, pairs):
        import heapq
        group = 2 if query_tile == 256 and pairs else 1
        visible, partitions = [], 1
        for q_len, kv_len in requests:
            tiles = -(-q_len // (query_tile * group)) * group
            visible += [(kv_len - q_len + min(q_len, (tile + 1) * query_tile) - 1) // 128 + 1
                        for tile in range(group - 1, tiles, group)]
            partitions = max(partitions, ((kv_len + 127) // 128 + partition_blocks - 1) // partition_blocks)
        free = [0] * (sm_count // group)
        for partition in range(partitions):
            begin, end = max(partition * partition_blocks, 8), (partition + 1) * partition_blocks
            for _ in range(heads):
                for blocks in visible:
                    if min(end, blocks - 8) - begin > 0:
                        heapq.heapreplace(free, free[0] + (min(end, blocks - 8) - begin) * query_tile)
        return max(free)

    def test_grouped_schedule_matches_one_cta_at_a_time(self):
        import random
        from mbsa.backends.optimized.index_policy import _list_schedule
        rng = random.Random(11)
        cases = [((8192, 1 << 20),), ((17, 1 << 20),), ((256, 65536),) * 32, ((4096, 131072),) * 2, ((1, 129),)]
        for _ in range(60):
            count = rng.choice((1, 2, 3, 8, 32))
            cases.append(tuple((q, max(q, rng.randint(1, 1 << 20)))
                               for q in (rng.randint(1, 8192 // count) for _ in range(count))))
        for requests in cases:
            for query_tile, pairs in ((128, True), (256, True), (256, False)):
                for blocks in (1024, 256, 64):
                    args = (requests, query_tile, 4, blocks, 148, pairs)
                    self.assertEqual(_list_schedule.__wrapped__(*args), self.per_cta_schedule(*args), args[1:])

    def test_policies_accept_lists_and_reuse_results(self):
        from mbsa.backends.optimized.index_policy import index_launch_shape
        index_launch_shape.cache_clear()
        first = index_launch_shape([[1024, 262144]], two_sm='auto')
        hits = index_launch_shape.cache_info().hits
        self.assertEqual(index_launch_shape(((1024, 262144),), two_sm='auto'), first)
        self.assertEqual(index_launch_shape.cache_info().hits, hits + 1)


def _local(iq, k, pages, kv, partition_blocks, options, runtime_partitions):
    """Compact IDs and partition buffers of one batch-one router launch, and its compile key."""
    from mbsa.backends.optimized.src.sm100 import router
    qn = iq.shape[0]
    partitions = ((kv + 127) // 128 + partition_blocks - 1) // partition_blocks
    scores = torch.full((4, partitions, qn, 16), float('nan'), device='cuda')
    ids = torch.full((4, partitions, qn, 16), -7, device='cuda', dtype=torch.int32)
    compact = torch.full((4, 32, qn), -7, device='cuda', dtype=torch.int32)
    compiled, key = router.indexer_local_top16(
        iq, k, partition_blocks=partition_blocks, page_indices=pages, logical_k_tokens=kv, use_page_table=True,
        out_scores=scores, out_ids=ids, out_compact_ids=compact, _fuse_single_partition_compact=partitions == 1,
        _prepare_only=True, _runtime_partitions=runtime_partitions, **options)
    compiled(*router._local_top16_arguments(iq, k, pages, scores, ids, compact, kv, 2, runtime_partitions),
             router.cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    return (compact, scores, ids), key


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class RuntimePartitionRouter(unittest.TestCase):
    """The default dispatch compiles the router with a runtime partition count: same selection, one kernel."""

    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest('UMMA Index requires SM100/SM103')

    @staticmethod
    def operands(qn, kv, dtype, seed=3):
        g = torch.Generator(device='cuda').manual_seed(seed)
        blocks = (kv + 127) // 128
        iq = torch.randn((qn, 4, 128), device='cuda', generator=g).to(dtype)
        k = torch.randn(((blocks + 3) * 128, 1, 128), device='cuda', generator=g).to(dtype)
        pages = torch.randperm(blocks + 3, device='cuda', generator=g)[:blocks].to(torch.int32).contiguous()
        return iq, k, pages

    def test_local_router_matches_static_partition_counts(self):
        from mbsa.backends.optimized.index_base import FP8_Q256_ROUTER_OPTIONS, TWO_SM_ROUTER_OPTIONS
        for dtype, qn, kv, width, options in (
                (torch.bfloat16, 1024, 262145, 256, dict(_parallel_insert=True, _skip_fixed_blocks=True)),
                (torch.bfloat16, 17, 524288, 128, dict(_parallel_insert=True)),      # 32 partitions: tree merge
                (torch.bfloat16, 2048, 393216, 512, dict(TWO_SM_ROUTER_OPTIONS, _skip_fixed_blocks=True)),
                (E4M3, 300, 131073, 64, dict(_parallel_insert=True, _queued_insert=True)),   # 17: tree merge
                (E4M3, 2048, 655360, 1024, dict(FP8_Q256_ROUTER_OPTIONS, _parallel_insert=True, _queued_insert=True))):
            with self.subTest(dtype=str(dtype), qn=qn, kv=kv, width=width):
                iq, k, pages = self.operands(qn, kv, dtype)
                static, static_key = _local(iq, k, pages, kv, width, options, False)
                runtime, runtime_key = _local(iq, k, pages, kv, width, options, True)
                self.assertNotIn('runtime_partitions', static_key)
                self.assertIn('runtime_partitions', runtime_key)
                for got, expected in zip(runtime, static):
                    torch.testing.assert_close(got, expected, rtol=0, atol=0, equal_nan=True)

    def test_one_kernel_serves_every_partition_count_of_a_width(self):
        from mbsa.backends.optimized.src.sm100 import router
        keys, sizes = set(), []
        for kv in (65536 * 3 + 1, 65536 * 5, 65536 * 7 - 128):   # 3, 5 and 7 partitions of 512 blocks
            iq, k, pages = self.operands(1024, kv, torch.bfloat16)
            keys.add(_local(iq, k, pages, kv, 512, dict(_parallel_insert=True), True)[1])
            sizes.append(len(router._COMPILE_CACHE))
        self.assertEqual(len(keys), 1)
        self.assertEqual(sizes[1:], sizes[:1] * 2)

    def test_packed_router_matches_static_partition_counts(self):
        from mbsa.backends.optimized.index import _stage_packed_tile_metadata
        from mbsa.backends.optimized.index_base import FP8_TWO_SM_ROUTER_OPTIONS
        from mbsa.backends.optimized.src.sm100 import router
        reqs = ((2048, 131072), (513, 262272), (1, 129), (17, 65537))
        blocks = [(kv + 127) // 128 for _, kv in reqs]
        for dtype, tile, width, options in (
                (torch.bfloat16, 128, 64, dict(_parallel_insert=True, _skip_fixed_blocks=True)),   # tree merge
                (E4M3, 256, 256, dict(FP8_TWO_SM_ROUTER_OPTIONS, _parallel_insert=True, _queued_insert=True))):
            with self.subTest(dtype=str(dtype), width=width):
                iq, k, _ = self.operands(sum(q for q, _ in reqs), 128, dtype)
                g = torch.Generator(device='cuda').manual_seed(11)
                k = torch.randn(((sum(blocks) + 5) * 128, 1, 128), device='cuda', generator=g).to(dtype)
                pages = torch.randperm(sum(blocks) + 5, device='cuda', generator=g)[:sum(blocks)].to(torch.int32)
                (metadata,) = _stage_packed_tile_metadata(iq.device, reqs, width, tile)
                partitions = max(-(-b // width) for b in blocks)
                outputs = []
                for runtime in (False, True):
                    compact = torch.full((4, 32, iq.shape[0]), -7, device='cuda', dtype=torch.int32)
                    scores = torch.full((4, partitions, iq.shape[0], 16), float('nan'), device='cuda')
                    ids = torch.full((4, partitions, iq.shape[0], 16), -7, device='cuda', dtype=torch.int32)
                    router.indexer_packed_top16(
                        iq, k, partition_blocks=width, page_indices=pages, tile_metadata=metadata,
                        partitions=partitions, out_compact_ids=compact, out_scores=scores, out_ids=ids,
                        _runtime_partitions=runtime, **options)
                    torch.cuda.synchronize()
                    outputs.append((compact, scores, ids))
                for got, expected in zip(*outputs[::-1]):
                    torch.testing.assert_close(got, expected, rtol=0, atol=0, equal_nan=True)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class DefaultDispatchPipeline(BackendCaseMixin, unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest('the Index dispatch defaults to its variants on SM103 only')

    @staticmethod
    def variant(plan):
        index = plan.plan._state.index
        return tuple(getattr(index, name) for name in (
            '_local_top16_key', 'two_sm_bound', 'fp8_q256_bound', 'fp8_two_sm_bound', 'router_partition_blocks',
            'skip_fixed_blocks_bound', 'parallel_insert_bound', 'queued_insert_bound', 'lean192_bound'))

    def prepare(self, inputs, kwargs, value=None, dispatch=None):
        with _environment(value):
            if dispatch is None:
                return prepare_cutedsl_pipeline(*inputs, backend='optimized', **kwargs)
            with index_dispatch_override(dispatch):
                return prepare_cutedsl_pipeline(*inputs, backend='optimized', **kwargs)

    def cases(self):
        torch.manual_seed(5)
        r = lambda s: torch.randn(s, device='cuda', dtype=torch.bfloat16)  # noqa: E731
        for qn, kn, dtype in ((8192, 131072, E4M3), (8192, 65536, torch.bfloat16)):
            inputs = (r((qn, 4, 128)).to(dtype), r((kn, 1, 128)).to(dtype), r((qn, 32, 128)), r((kn, 4, 128)),
                      r((kn, 4, 128)))
            yield f'dense {qn}/{kn} {dtype}', inputs, {}
        for lengths, dtype in ((((1024, 65536),), torch.bfloat16), (((4096, 131072),) * 2, E4M3)):
            _, inputs, kwargs = backend_support.BackendLifecycle.fixture(self, lengths)
            inputs = (inputs[0].to(dtype), inputs[1].to(dtype)) + tuple(inputs[2:])
            yield f'paged {lengths} {dtype}', inputs, kwargs

    def test_default_binds_auto_and_baseline_binds_baseline(self):
        for name, inputs, kwargs in self.cases():
            with self.subTest(case=name):
                default = self.prepare(inputs, kwargs)
                baseline = self.prepare(inputs, kwargs, 'baseline')
                self.assertEqual(self.variant(default),
                                 self.variant(self.prepare(inputs, kwargs, 'baseline', AUTO_INDEX_DISPATCH)))
                self.assertEqual(self.variant(baseline),
                                 self.variant(self.prepare(inputs, kwargs, None, BASELINE_INDEX_DISPATCH)))
                self.assertNotEqual(self.variant(default), self.variant(baseline))
                expected = tuple(t.clone() for t in baseline.run())
                for actual, reference in zip(default.run(), expected):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                default.capture_mega_graph()
                for actual, reference in zip(default.run_mega(), expected):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_fp16_index_keeps_the_baseline_selection(self):
        # The variants are measured, and compiled by serving warmup, for BF16 and E4M3 Index only.
        _, inputs, kwargs = backend_support.BackendLifecycle.fixture(self, ((1024, 65536),))
        inputs = (inputs[0].half(), inputs[1].half()) + tuple(inputs[2:])
        self.assertEqual(self.variant(self.prepare(inputs, kwargs)),
                         self.variant(self.prepare(inputs, kwargs, 'baseline')))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class ServingWarmupCoverage(unittest.TestCase):
    """Serving warmup compiles the router kernels of every variant the dispatch binds for paged steps."""

    # Short and long queries, finer partitions with serial and tree merges, packed steps and split pieces.
    STEPS = (((17, 4113),), ((300, 131073),), ((1024, 262144),), ((2048, 655360),), ((8192, 1048576),),
             ((17, 1048576),), ((128, 32768),), ((4096, 1000000),), ((17, 4113), (23, 2195), (41, 127)),
             ((1024, 262144),) * 4, ((256, 65536),) * 16, ((3000, 32768), (5192, 200000)), ((9000, 40960),),
             ((1, 524288),) * 32)

    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest('the optimized backend requires SM100 or SM103')

    def steps_compile_no_router_kernel(self, value, formats):
        from mbsa.backends.optimized.src.sm100 import router
        from mbsa.serving import Prefill, warmup
        with _environment(value):
            for index_format in formats:
                with self.subTest(index_format=index_format):
                    prefill = Prefill(torch.device('cuda'), backend='optimized')
                    warmup(prefill, ((16, 4096),), index_format=index_format)
                    before = set(router._COMPILE_CACHE)
                    for lengths in self.STEPS:
                        warmup(prefill, lengths, index_format=index_format)
                    self.assertEqual(set(router._COMPILE_CACHE) - before, set())

    def test_steps_after_warmup_compile_no_router_kernel(self):
        # The device's own dispatch: the default variants on SM103, the baseline selection on SM100.
        self.steps_compile_no_router_kernel(None, ('bf16', 'fp8'))

    def test_baseline_steps_after_warmup_compile_no_router_kernel(self):
        # Every partition count of the baseline selection (FP16 Index always binds it) runs the warmed-up kernels.
        self.steps_compile_no_router_kernel('baseline', ('bf16', 'fp16', 'fp8'))


class DeviceKeys(unittest.TestCase):
    """A bare "cuda" names the current GPU: runner and warmup caches are per GPU."""

    @staticmethod
    def current(index):
        """torch.cuda as an initialized runtime whose current device is index."""
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch('torch.cuda.is_initialized', return_value=True))
        stack.enter_context(mock.patch('torch.cuda.current_device', return_value=index))
        return stack

    def test_bare_cuda_resolves_to_the_current_device(self):
        from mbsa.backends._binding import explicit_device
        with self.current(3):
            for device in ('cuda', torch.device('cuda')):
                self.assertEqual(explicit_device(device), torch.device('cuda', 3))
            self.assertEqual(explicit_device('cuda:1'), torch.device('cuda', 1))
        self.assertEqual(explicit_device('cpu'), torch.device('cpu'))

    def test_a_runner_is_built_without_initializing_cuda(self):
        # Until CUDA is initialized the current device is 0; asking torch for it would initialize CUDA in
        # a process that only builds its runner (before forking workers, say).
        from mbsa import serving
        from mbsa.backends._binding import explicit_device
        with mock.patch('torch.cuda.is_initialized', return_value=False), \
                mock.patch('torch.cuda.current_device', side_effect=AssertionError('initializes CUDA')), \
                mock.patch.dict(serving._RUNNERS, clear=True):
            self.assertEqual(explicit_device('cuda'), torch.device('cuda', 0))
            self.assertEqual(serving.runner('cuda', backend='optimized').device, torch.device('cuda', 0))
            self.assertEqual(serving.Prefill('cuda', backend='optimized').device, torch.device('cuda', 0))

    def test_runner_cache_is_per_gpu(self):
        from mbsa import serving
        with mock.patch.dict(serving._RUNNERS, clear=True):
            with self.current(0):
                first = serving.runner('cuda', backend='optimized')
                self.assertIs(first, serving.runner('cuda:0', backend='optimized'))
            with self.current(1):
                second = serving.runner('cuda', backend='optimized')
                self.assertIs(second, serving.runner(torch.device('cuda'), backend='optimized'))
            self.assertIsNot(first, second)
            self.assertEqual((first.device, second.device), (torch.device('cuda', 0), torch.device('cuda', 1)))

    def test_warmup_record_is_per_gpu(self):
        from mbsa.backends.optimized import index
        calls = []

        def variants(dispatch, fp8, max_kv_tokens):
            calls.append(torch.cuda.current_device())
            return iter(())

        with mock.patch.object(index, '_COMPILED_SERVING_VARIANTS', set()) as done, \
                mock.patch.object(index, '_serving_router_variants', variants), \
                index_dispatch_override(AUTO_INDEX_DISPATCH):
            for current in (0, 1, 1, 0):
                with self.current(current):
                    index.compile_serving_index_variants('cuda', torch.bfloat16, max_kv_tokens=1 << 20)
            self.assertEqual(calls, [0, 1])
            self.assertEqual({key[0] for key in done}, {0, 1})

    @unittest.skipUnless(torch.cuda.device_count() >= 2, 'two GPUs required')
    def test_second_gpu_compiles_its_own_router_kernels(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest('the optimized backend requires SM100 or SM103')
        from mbsa.backends.optimized.src.sm100 import router
        from mbsa.serving import Prefill, warmup
        with _environment(None):
            for ordinal in (0, 1):
                with torch.cuda.device(ordinal):
                    warmup(Prefill('cuda', backend='optimized'), ((16, 4096),), index_format='bf16')
        # Earlier tests of the process may have compiled more kernels for GPU 0: compare the variants.
        compiled = [{key[2:] for key in router._COMPILE_CACHE if key[0] == ordinal and key[1] == torch.bfloat16}
                    for ordinal in (0, 1)]
        self.assertGreater(len(compiled[1]), 1)
        self.assertLessEqual(compiled[1], compiled[0])


if __name__ == '__main__':
    unittest.main()
