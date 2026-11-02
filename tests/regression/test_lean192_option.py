# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Public policy, actual execution, and binding/capture isolation regressions."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from mbsa import prepare_cutedsl_pipeline
from mbsa.api import prepare_sm100
from tests.common import BackendCaseMixin
from tests.reference import reference_pipeline


class Lean192APIValidation(unittest.TestCase):
    def test_incompatible_target_rejects_on_before_attention_preparation(self):
        q = SimpleNamespace(is_cuda=True, device='cuda:0')
        for prepare in (prepare_cutedsl_pipeline, prepare_sm100):
            with self.subTest(api=prepare.__name__), \
                 mock.patch('torch.cuda.get_device_capability', return_value=(10, 3)), \
                 mock.patch('mbsa.backends.optimized.index.lean192_compiler_supported', return_value=False), \
                 mock.patch('mbsa.backends.optimized.kernels.attention.prepare_unified_aligned_prefill') as attention:
                with self.assertRaisesRegex(ValueError, 'SM103 compiler target'):
                    prepare(None, None, q, None, None, backend='optimized', lean192='on')
                attention.assert_not_called()

    def test_invalid_policy_is_rejected_before_operand_or_backend_access(self):
        for prepare in (prepare_cutedsl_pipeline, prepare_sm100):
            for invalid in (None, True, False, 'ON', 'yes', ''):
                with self.subTest(api=prepare.__name__, mode=invalid):
                    with self.assertRaisesRegex(ValueError, 'lean192 must be'):
                        prepare(None, None, None, None, None,
                                backend='optimized', lean192=invalid)
            for mode in ('on', 'off'):
                with self.assertRaisesRegex(ValueError, 'requires backend="optimized"'):
                    prepare(None, None, None, None, None, lean192=mode)

    def test_runner_cache_separates_requested_modes_without_gpu_work(self):
        from mbsa import serving
        with mock.patch.dict(serving._RUNNERS, clear=True):
            owners = {mode: serving.runner('cuda:0', backend='optimized', lean192=mode)
                      for mode in ('auto', 'on', 'off')}
            self.assertEqual(len({id(x) for x in owners.values()}), 3)
            for mode, owner in owners.items():
                self.assertIs(owner, serving.runner('cuda:0', backend='optimized', lean192=mode))
                self.assertEqual(owner.lean192, mode)
                with self.assertRaises(AttributeError):
                    owner.lean192 = 'off'
            default = serving.runner('cuda:0')
            self.assertEqual(default.backend, 'default')
            self.assertEqual(default.lean192, 'auto')
            self.assertEqual(len(serving._RUNNERS), 4)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class Lean192Execution(BackendCaseMixin, unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest('actual Lean192 ON/OFF requires SM103')
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, 'allow_tf32', old_tf32)

    @staticmethod
    def binary_hash(plan):
        value = plan.plan._state.index._local_top16_compiled.__cubin__
        if isinstance(value, str):
            value = Path(value).read_bytes()
        return hashlib.sha256(value).hexdigest()

    def check_rows(self, actual, inputs, *, kv, rows=(0, 127, 128, 8191)):
        iq, ik, q, k, v = inputs
        for row in rows:
            expected = reference_pipeline(iq[row:row+1].float(), ik.view(-1, 1, 128).float(),
                q[row:row+1], k, v, query_offset=kv-8192+row)
            self.compare((actual[0][row:row+1], actual[1][row:row+1],
                          actual[2][:, row:row+1]), expected)

    def test_public_modes_bind_distinct_kernels_with_identical_results(self):
        inputs = self.inputs(8192, 32768)
        snapshots = tuple(t.clone() for t in inputs)
        for prepare in (prepare_cutedsl_pipeline, prepare_sm100):
            plans = {mode: prepare(*inputs, backend='optimized', lean192=mode)
                     for mode in ('auto', 'on', 'off')}
            values = {}
            for mode, plan in plans.items():
                self.assertEqual(plan.lean192, mode)
                self.assertEqual(plan.lean192_enabled, mode != 'off')
                self.assertEqual(plan.plan._state.index.candidate_partition_blocks, 1024)
                with self.assertRaises(AttributeError):
                    plan.lean192 = 'off'
                with mock.patch('mbsa.backends.optimized.index.indexer_local_top16',
                                side_effect=AssertionError('unexpected runtime compilation')):
                    values[mode] = tuple(t.clone() for t in plan.run())
                self.check_rows(values[mode], inputs, kv=32768)
            self.assertEqual(self.binary_hash(plans['auto']), self.binary_hash(plans['on']))
            self.assertNotEqual(self.binary_hash(plans['on']), self.binary_hash(plans['off']))
            for mode in ('on', 'off'):
                for actual, expected in zip(values[mode], values['auto']):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for actual, expected in zip(inputs, snapshots):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_forced_on_rebind_rejection_preserves_original_binding(self):
        inputs = self.inputs(8192, 8192)
        plan = prepare_cutedsl_pipeline(*inputs, backend='optimized', lean192='on')
        original = plan.plan._state
        expected = tuple(t.clone() for t in plan.run())
        pages = original.bindings['index_page_indices'].clone()
        for mapping in (pages, pages.flip(0)):
            rejected_out = torch.full_like(inputs[2], 91)
            with self.assertRaisesRegex(ValueError, 'lean192="on".*rebind'):
                plan.rebind(*inputs, index_page_indices=mapping,
                            page_table=None, out=rejected_out)
            self.assertIs(plan.plan._state, original)
            self.assertTrue(bool((rejected_out == 91).all()))
            for actual, before in zip(plan.run(), expected):
                torch.testing.assert_close(actual, before, atol=0, rtol=0)
        replacement = tuple(t.clone() for t in inputs)
        replacement[4].add_(1)
        new_plan = prepare_cutedsl_pipeline(*replacement, backend='optimized', lean192='on')
        self.assertTrue(new_plan.lean192_enabled)
        self.assertEqual(new_plan.lean192, 'on')
        self.assertFalse(new_plan.plan._state.index.cutedsl_use_page_table)
        self.check_rows(tuple(t.clone() for t in new_plan.run()), replacement, kv=8192)
        # Old storage may become nonfinite once ordered old work is complete.
        torch.cuda.current_stream().synchronize()
        for t in inputs:
            t.fill_(torch.nan)
        self.check_rows(tuple(t.clone() for t in new_plan.run()), replacement, kv=8192)

    def test_supplied_maps_remain_live_for_auto_off_and_reject_forced_on(self):
        inputs = self.inputs(8192, 8192)
        pages = torch.arange(64, device='cuda', dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, 'SM103 BF16 direct-addressing'):
            prepare_cutedsl_pipeline(*inputs, backend='optimized', lean192='on',
                                    index_page_indices=pages)
        plans = {mode: prepare_cutedsl_pipeline(*inputs, backend='optimized', lean192=mode,
                                               index_page_indices=pages)
                 for mode in ('auto', 'off')}
        before = {mode: tuple(t.clone() for t in p.run()) for mode, p in plans.items()}
        pages.copy_(pages.flip(0))
        after = {mode: tuple(t.clone() for t in p.run()) for mode, p in plans.items()}
        self.assertFalse(torch.equal(before['auto'][2], after['auto'][2]))
        logical_ik = inputs[1].view(64, 128, 1, 128)[pages.long()].reshape(8192, 1, 128)
        for mode, p in plans.items():
            self.assertFalse(p.lean192_enabled)
            self.assertTrue(p.plan._state.index.cutedsl_use_page_table)
            self.check_rows(after[mode], (inputs[0], logical_ik, *inputs[2:]), kv=8192)
            for actual, expected in zip(after[mode], after['auto']):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            p.capture_mega_graph()
        pages.copy_(pages.flip(0))
        for mode, p in plans.items():
            for actual, expected in zip(p.run_mega(), before[mode]):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_graph_live_inputs_and_independent_modes_on_two_streams(self):
        inputs = self.inputs(8192, 8192)
        plans = {mode: prepare_cutedsl_pipeline(*inputs, backend='optimized', lean192=mode)
                 for mode in ('on', 'off')}
        golden = {mode: tuple(t.clone() for t in plan.run()) for mode, plan in plans.items()}
        streams = {mode: torch.cuda.Stream() for mode in plans}
        current = torch.cuda.current_stream()
        for mode, plan in plans.items():
            streams[mode].wait_stream(current)
            with torch.cuda.stream(streams[mode]):
                for _ in range(10):
                    plan.run()
        for stream in streams.values():
            current.wait_stream(stream)
        # Read live results after both joins; no serial rerun substitutes for concurrency.
        for mode, plan in plans.items():
            live = (plan.plan.attention.out, plan.plan.attention.out_lse,
                    plan.plan.attention.tile_plan.dynamic_route)
            for actual, expected in zip(live, golden[mode]):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            plan.capture_mega_graph()
            with self.assertRaises(RuntimeError):
                plan.rebind(*inputs, index_page_indices=plan.plan._state.bindings['index_page_indices'],
                            page_table=None, out=torch.empty_like(inputs[2]))
            plan.plan.attention.out.fill_(torch.nan)
            plan.plan.attention.out_lse.fill_(torch.nan)
            plan.plan.attention.tile_plan.dynamic_route.fill_(-99)
            for _ in range(1000):
                plan.run_mega()
            for actual, expected in zip(plan.run_mega(), golden[mode]):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        inputs[4].neg_()
        for mode, plan in plans.items():
            actual = plan.run_mega()
            torch.testing.assert_close(actual[0], -golden[mode][0], atol=.03, rtol=.03)
            torch.testing.assert_close(actual[1], golden[mode][1], atol=0, rtol=0)
            torch.testing.assert_close(actual[2], golden[mode][2], atol=0, rtol=0)

    def test_unsupported_forced_on_is_not_a_silent_fallback(self):
        from tests.regression.test_backends import BackendLifecycle
        fixture = BackendLifecycle()
        for lengths in (((129, 4273),), ((17, 4113), (23, 8192))):
            data, inputs, kwargs = fixture.fixture(lengths)
            expected = fixture.reference(data, inputs, lengths)
            for mode in ('auto', 'off'):
                plan = prepare_cutedsl_pipeline(*inputs, backend='optimized', lean192=mode, **kwargs)
                self.assertFalse(plan.lean192_enabled)
                self.compare(plan.run(), expected)
            with self.assertRaisesRegex(ValueError, 'SM103 BF16 direct-addressing'):
                prepare_cutedsl_pipeline(*inputs, backend='optimized', lean192='on', **kwargs)
        inputs = self.inputs(8192, 8192)
        for changed in ((inputs[0].to(torch.float8_e4m3fn), inputs[1].to(torch.float8_e4m3fn), *inputs[2:]),
                        (*inputs[:2], *(t.to(torch.float8_e4m3fn) for t in inputs[2:]))):
            with self.assertRaisesRegex(ValueError, 'SM103 BF16 direct-addressing'):
                prepare_cutedsl_pipeline(*changed, backend='optimized', lean192='on')
        # A supplied identity tensor is still mutable, not a permanent proof.
        identity = torch.arange(64, device='cuda', dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, 'SM103 BF16 direct-addressing'):
            prepare_cutedsl_pipeline(*inputs, backend='optimized', lean192='on',
                                    index_page_indices=identity)


if __name__ == '__main__':
    unittest.main()
