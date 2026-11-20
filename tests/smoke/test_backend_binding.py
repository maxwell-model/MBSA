# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend selection and binding ownership checks without a CUDA runtime."""

import importlib.util
import itertools
import gc
import weakref
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch, sentinel

import torch

from mbsa.backends._binding import publish_plan, stage_plan, validate_output_aliases
from mbsa.backends._lifecycle import PreparedOwnerBase


class _BindingObject:
    pass


class _LifetimePlan:
    def __init__(self):
        self.payload = torch.empty(256)
        self.attention = _BindingObject()
        self.attention.tile_plan = _BindingObject()
        self.calls = 0

    def run(self):
        self.calls += 1
        return self.calls


class BindingTest(unittest.TestCase):
    def test_owner_releases_plan_and_operands_without_cyclic_gc(self):
        enabled = gc.isenabled()
        gc.disable()
        try:
            plan = _LifetimePlan()
            owner = PreparedOwnerBase(plan)
            refs = weakref.ref(plan), weakref.ref(plan.payload)
            del plan, owner
            self.assertTrue(all(ref() is None for ref in refs))
        finally:
            if enabled:
                gc.enable()

    def test_capture_tracking_survives_stage_and_publish(self):
        plan = _LifetimePlan()
        owner = PreparedOwnerBase(plan)
        staged = stage_plan(plan)
        self.assertIs(type(staged), type(plan))
        self.assertNotIn('run', vars(plan))
        with patch.object(torch.cuda, 'is_current_stream_capturing', return_value=True):
            self.assertEqual(staged.run(), 1)
        self.assertFalse(plan._binding_external_capture)
        self.assertTrue(staged._binding_external_capture)
        publish_plan(plan, staged)
        self.assertIs(owner.plan, plan)
        with patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False):
            self.assertEqual(owner.run(), 2)
        self.assertTrue(plan._binding_external_capture)
        self.assertIs(plan.run.__self__, plan)

    def test_capture_wrapper_type_is_cached_and_not_nested(self):
        first, second = _LifetimePlan(), _LifetimePlan()
        owners = PreparedOwnerBase(first), PreparedOwnerBase(second)
        self.assertIs(type(first), type(second))
        before = type(first)
        another_owner = PreparedOwnerBase(first)
        self.assertIs(type(first), before)
        self.assertIs(owners[0].plan, another_owner.plan)

    def test_optimized_selection_uses_replacement_factory(self):
        from mbsa.api import prepare_sm100, _validate_backend
        from mbsa.backends.optimized import runtime
        q = SimpleNamespace(is_cuda=True, device=torch.device("cuda:0"))
        with patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)), \
             patch("mbsa.backends._binding.validate_output_aliases"), \
             patch.object(runtime, "prepare_sm100", return_value=sentinel.plan) as factory:
            result = prepare_sm100(None, None, q, None, None, backend="optimized")
        self.assertIs(result, sentinel.plan)
        factory.assert_called_once()
        self.assertEqual(runtime.PreparedOwner.backend, "optimized")
        with self.assertRaisesRegex(ValueError, "backend"):
            _validate_backend("unsupported")

    def test_output_alias_contract(self):
        storage = torch.arange(64)
        # Inputs may overlap one another; non-overlapping views may be outputs.
        validate_output_aliases(storage[32:], storage[:16], storage[8:24])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_output_aliases(storage[16:40], storage[8:24])
        validate_output_aliases(torch.empty(0), storage)

    def test_staged_mapping_aliases_are_isolated(self):
        packed = torch.arange(32)
        table = torch.zeros((1, 2), dtype=torch.int32)
        fixed = SimpleNamespace(
            block_tables=table, mappings=[(packed[4:8], packed[8:12])],
            packed_metadata={"indices": packed}, _prepared_forward=None,
        )
        plan = SimpleNamespace(attention=SimpleNamespace(
            tile_plan=SimpleNamespace(page_table=table), fixed_attention=fixed,
        ))
        staged = stage_plan(plan, page_table=table)
        staged_fixed = staged.attention.fixed_attention
        staged_fixed.mappings[0][0].fill_(-1)
        self.assertEqual(staged_fixed.packed_metadata["indices"][4:8].tolist(), [-1] * 4)
        self.assertEqual(packed[4:8].tolist(), list(range(4, 8)))
        self.assertIs(plan.attention.fixed_attention, fixed)


class MsaConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = (Path(__file__).parents[2] /
                "python/mbsa/backends/optimized/msa/cute/src/sm100/fwd/config.py")
        spec = importlib.util.spec_from_file_location("_msa_config_test", path)
        cls.config = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.config
        spec.loader.exec_module(cls.config)

    def test_full_blocks_select_same_geometry(self):
        for sm in ((10, 0), (10, 3)):
            for tokens, persistent in ((8192, False), (16384, False),
                                       (32768, True), (1048576, True)):
                with self.subTest(sm=sm, tokens=tokens):
                    cfg = self.config.select_msa_cfg(
                        causal=False, full_kv_blocks=True, sm=sm,
                        max_seqlen_k=tokens)
                    self.assertEqual(cfg.num_wgs, 5)
                    self.assertEqual((cfg.q_stages, cfg.kv_stages),
                                     (2, 2 if persistent else 1))
                    self.assertTrue(cfg.full_kv_blocks)
                    self.assertEqual(cfg.use_clc, persistent)
                    self.assertEqual(cfg.regs,
                                     (184, 104, 104, 24, 64))

    def test_causal_blocks_preserve_fallback(self):
        for sm, ratio in itertools.product(((10, 0), (10, 3)), (1, 2, 4, 8, 16)):
            for causal, full in ((True, True), (True, False)):
                with self.subTest(sm=sm, causal=causal, full=full):
                    cfg = self.config.select_msa_cfg(
                        causal=causal, full_kv_blocks=full, sm=sm,
                        max_seqlen_k=1048576, qhead_per_kv=ratio)
                    self.assertEqual(cfg.num_wgs, 4)
                    self.assertFalse(cfg.full_kv_blocks or cfg.use_clc)
                    self.assertEqual(cfg.kv_stages, 1)

    def test_noncausal_masked_blocks_preserve_scheduler_and_validity(self):
        for sm in ((10, 0), (10, 3)):
            for tokens in (127, 128, 129, 8192, 32768, 1048575, 1048576):
                with self.subTest(sm=sm, tokens=tokens):
                    cfg = self.config.select_msa_cfg(
                        causal=False, full_kv_blocks=False, sm=sm,
                        max_seqlen_k=tokens, paged_kv=True)
                    self.assertEqual(cfg.num_wgs, 5)
                    persistent = tokens >= 32768
                    self.assertFalse(cfg.full_kv_blocks)
                    self.assertEqual(cfg.use_clc, persistent)
                    self.assertEqual((cfg.q_stages, cfg.kv_stages),
                                     (2, 2 if persistent else 1))
                    self.assertEqual(
                        cfg.regs,
                        (200, 96, 96, 24, 64) if persistent
                        else (184, 104, 104, 24, 64))
                    self.assertEqual(sum(cfg.regs), 480)
                    full_cfg = self.config.select_msa_cfg(
                        causal=False, full_kv_blocks=True, sm=sm,
                        max_seqlen_k=tokens)
                    self.assertEqual(full_cfg.regs, (184, 104, 104, 24, 64))

    def test_query_transport_controls_five_group_eligibility(self):
        for sm in ((10, 0), (10, 3)):
            for ratio in (1, 2, 4, 8, 16):
                for full in (False, True):
                    for length in (8192, 32768, 1048576):
                        with self.subTest(sm=sm, ratio=ratio, full=full, length=length):
                            cfg = self.config.select_msa_cfg(
                                causal=False, full_kv_blocks=full, sm=sm,
                                max_seqlen_k=length, qhead_per_kv=ratio)
                            self.assertEqual(cfg.num_wgs, 5)
                            self.assertEqual(cfg.full_kv_blocks, full)
                            self.assertEqual(cfg.use_clc, length >= 32768)
        for sm, ratio in (((9, 0), 1), ((10, 0), 3), ((10, 3), 32)):
            with self.subTest(unsupported_sm=sm, unsupported_ratio=ratio):
                cfg = self.config.select_msa_cfg(
                    causal=False, full_kv_blocks=False, sm=sm,
                    max_seqlen_k=1048576, qhead_per_kv=ratio)
                self.assertEqual(cfg, self.config.MSA_4WG_STATIC_GENERIC)

    def test_storage_conversion_retains_dequantization_roles(self):
        for sm in ((10, 0), (10, 3)):
            for ratio in (1, 2, 4, 8, 16):
                for full in (False, True):
                    with self.subTest(sm=sm, ratio=ratio, full=full):
                        cfg = self.config.select_msa_cfg(
                            causal=False, full_kv_blocks=full, sm=sm,
                            max_seqlen_k=1048576, qhead_per_kv=ratio,
                            requires_kv_staging=True)
                        self.assertEqual(cfg, self.config.MSA_4WG_STATIC_GENERIC)

    def test_explicit_conversion_config_rejects_before_execution(self):
        validate = self.config._validate_msa_operand_roles
        for cfg in (self.config.MSA_5WG_STATIC_MASKED, self.config.MSA_5WG_CLC_MASKED,
                    self.config.MSA_4WG_CLC):
            with self.subTest(config=cfg.symbol_suffix):
                with self.assertRaisesRegex(ValueError, 'dequantization roles'):
                    validate(cfg, True)
                validate(cfg, False)
        validate(None, True)
        validate(self.config.MSA_4WG_STATIC_GENERIC, True)


class CompiledKernelCount(unittest.TestCase):
    def test_counts_the_separate_mixed_merge_cache(self):
        from mbsa.serving import compiled_kernel_count
        module = SimpleNamespace(_MERGE_KERNELS={})
        with patch.dict(sys.modules, {'mbsa.backends.optimized.kernels.merge': module}):
            before = compiled_kernel_count()
            module._MERGE_KERNELS.update({0: sentinel.first, 1: sentinel.second})
            self.assertEqual(compiled_kernel_count(), before + 2)


if __name__ == "__main__":
    unittest.main()
