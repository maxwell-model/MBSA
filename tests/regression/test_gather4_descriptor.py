# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schedule-owned TMA metadata lifetime; real FWD oracles stay in test_optimized."""
from dataclasses import replace
import gc
import unittest
from unittest import mock
import weakref
import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Gather4DescriptorLifecycle(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("SM100/SM103 required")

    @staticmethod
    def schedule():
        from mbsa.backends.optimized.msa.cute.src.sm100.prepare_scheduler import SparseAttentionSchedule
        return SparseAttentionSchedule(False, None, None)

    def test_binding_keys_schedule_isolation_and_direct_release(self):
        from mbsa.backends.optimized.msa.cute import interface
        q = torch.zeros((16, 128), device='cuda', dtype=torch.bfloat16)
        schedule = self.schedule()
        get = interface._get_q_gather4_tma_desc
        desc = get(schedule, q, box_x=64)
        self.assertIs(get(schedule, q.view_as(q), box_x=64), desc)
        self.assertIsNot(get(schedule, q[:8], box_x=64), desc)
        other = replace(schedule)
        self.assertIsNot(other._q_gather4_desc_cache, schedule._q_gather4_desc_cache)
        self.assertIsNot(get(other, q, box_x=64), desc)
        qref, dref = weakref.ref(q), weakref.ref(desc)
        previous = gc.isenabled()
        gc.disable()
        try:
            del q, desc, schedule, other
            self.assertIsNone(qref(), "metadata must not retain input tensors")
            self.assertIsNone(dref(), "schedule metadata must not form an ownership cycle")
        finally:
            if previous:
                gc.enable()

    def test_warmed_capture_reuses_metadata_and_miss_does_not_poison_capture(self):
        from mbsa.backends.optimized.msa.cute import interface
        q = torch.zeros((16, 128), device='cuda', dtype=torch.bfloat16)
        schedule = self.schedule()
        desc = interface._get_q_gather4_tma_desc(schedule, q, box_x=64)
        mirror = torch.empty_like(desc)
        graph = torch.cuda.CUDAGraph()
        with mock.patch.object(interface, 'create_q_gather4_tma_desc',
                               side_effect=AssertionError("capture must not rebuild descriptors")):
            with torch.cuda.graph(graph):
                warmed = interface._get_q_gather4_tma_desc(schedule, q, box_x=64)
                mirror.copy_(warmed)
            for _ in range(10):
                graph.replay()
        torch.testing.assert_close(mirror, desc, atol=0, rtol=0)
        missing, out = self.schedule(), torch.empty_like(q)
        safe = torch.cuda.CUDAGraph()
        with torch.cuda.graph(safe):
            out.copy_(q)
            with self.assertRaisesRegex(RuntimeError, "warm gather4"):
                interface._get_q_gather4_tma_desc(missing, q, box_x=64)
            out.add_(1)
        safe.replay()
        torch.testing.assert_close(out, q + 1, atol=0, rtol=0)
        self.assertEqual(missing._q_gather4_desc_cache, {})

    def test_private_ragged_partials_binding_and_release(self):
        from mbsa.backends.optimized.msa.cute import interface
        q = torch.zeros((17, 32, 128), device='cuda', dtype=torch.bfloat16)
        schedule, other = self.schedule(), self.schedule()
        get = interface._get_ragged_partials
        kwargs = dict(tile_rows=256, dim=128, partial_dtype=torch.bfloat16)
        partials = get(schedule, q, **kwargs)
        self.assertIs(get(schedule, q, **kwargs), partials)
        self.assertIsNot(get(other, q, **kwargs)[0], partials[0])
        self.assertIsNot(get(schedule, q, **dict(kwargs, tile_rows=512))[0], partials[0])
        self.assertEqual(partials[0].shape, (256, 128))
        self.assertEqual(partials[1].dtype, torch.float32)
        qref, pref = weakref.ref(q), weakref.ref(partials[0])
        previous = gc.isenabled()
        gc.disable()
        try:
            del q, partials, schedule, other
            self.assertIsNone(qref())
            self.assertIsNone(pref())
        finally:
            if previous:
                gc.enable()
