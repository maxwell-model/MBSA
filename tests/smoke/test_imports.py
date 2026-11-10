#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Lazy import boundaries and public input contracts."""

from __future__ import annotations
import os
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

CUTEDSL_ARCHS = ((9, 0), (10, 0), (10, 3))


class ImportBoundaryTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('cutlass'), 'CUTLASS DSL is required')
    def test_optimized_import_under_vllm_vendored_namespace(self):
        """Optimized imports must not require an unrelated top-level mbsa alias."""
        script = r'''
import importlib, importlib.util, sys, types
from pathlib import Path
root = Path(sys.argv[1])
for name in ('vllm', 'vllm.third_party'):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
name = 'vllm.third_party.mbsa'
spec = importlib.util.spec_from_file_location(name, root / '__init__.py',
    submodule_search_locations=[str(root)])
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
for suffix in ('.backends.optimized.runtime', '.backends.optimized.kernels.attention',
               '.backends.optimized.msa.cute.interface', '.backends.optimized.kernels.kv_pages',
               '.backends.optimized.kernels.merge'):
    imported = importlib.import_module(name + suffix)
    assert imported.__name__.startswith(name + '.')
assert 'mbsa' not in sys.modules
'''
        result = subprocess.run(
            [sys.executable, '-I', '-c', script, str(ROOT / 'python/mbsa')],
            text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(importlib.util.find_spec('cutlass'), 'CUTLASS DSL is required')
    def test_runtime_import_under_vllm_vendored_namespace(self):
        """The CUDA runtime must load without a standalone mbsa package alias."""
        script = r'''
import importlib, importlib.util, sys, types
from pathlib import Path
root = Path(sys.argv[1])
for name in ('vllm', 'vllm.third_party'):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
name = 'vllm.third_party.mbsa'
spec = importlib.util.spec_from_file_location(name, root / '__init__.py',
    submodule_search_locations=[str(root)])
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
importlib.import_module(name + '.cute.prefill')
assert 'mbsa' not in sys.modules
'''
        result = subprocess.run(
            [sys.executable, '-I', '-c', script, str(ROOT / 'python/mbsa')],
            text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_partial_precision_switch_and_explicit_override(self):
        import torch
        from mbsa.api import _resolve_partial_dtype
        from mbsa.serving import Prefill, runner

        for mode, expected in (("0", torch.bfloat16), ("1", torch.float8_e4m3fn)):
            with (
                self.subTest(mode=mode),
                patch.dict(os.environ, {"MBSA_FP8_PARTIAL_OUT": mode}),
            ):
                self.assertEqual(_resolve_partial_dtype(None), expected)
                self.assertEqual(_resolve_partial_dtype(torch.bfloat16), torch.bfloat16)
                self.assertEqual(Prefill(torch.device("cpu")).partial_dtype, expected)
                self.assertEqual(runner(torch.device("cpu")).partial_dtype, expected)
                self.assertEqual(Prefill(torch.device("cpu"),
                                        partial_dtype=torch.bfloat16).partial_dtype,
                                 torch.bfloat16)
        with patch.dict(os.environ, {"MBSA_FP8_PARTIAL_OUT": "invalid"}):
            with self.assertRaisesRegex(ValueError, "MBSA_FP8_PARTIAL_OUT"):
                _resolve_partial_dtype(None)
        with self.assertRaisesRegex(ValueError, "partial_dtype"):
            _resolve_partial_dtype(torch.float32)

    def test_public_import_is_lazy(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "python")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import mbsa; "
                "assert mbsa.__all__ == ['prepare_cutedsl_pipeline']; "
                "assert not {'torch','triton','cutlass','fmha_sm100'}.intersection(sys.modules)",
            ],
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cutedsl_rejects_cpu_before_loading_runtime(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "python")
        script = r"""
import sys, torch
from mbsa import prepare_cutedsl_pipeline
q = torch.zeros((1, 32, 128))
try:
    prepare_cutedsl_pipeline(q, q, q, q, q)
except ValueError as error:
    assert 'CUDA' in str(error), str(error)
else:
    raise AssertionError('CPU input must be rejected')
assert 'fmha_sm100' not in sys.modules
assert 'cutlass' not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", script], env=env, text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
