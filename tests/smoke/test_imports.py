#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU/GPU contracts and full-query comparison against pinned HF Attention.

The shared config owns every shape and gate. Decode reserves scheduler budget;
only prefill enters the operator. A strict mismatch is a failure, including ties.
"""

from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

CUTEDSL_ARCHS = ((9, 0), (10, 0), (10, 3))


class ImportBoundaryTest(unittest.TestCase):
    def test_partial_precision_switch_and_explicit_override(self):
        import torch
        from mbsa.api import _resolve_partial_dtype

        for mode, expected in (("0", torch.bfloat16), ("1", torch.float8_e4m3fn)):
            with (
                self.subTest(mode=mode),
                patch.dict(os.environ, {"MBSA_FP8_PARTIAL_OUT": mode}),
            ):
                self.assertEqual(_resolve_partial_dtype(None), expected)
                self.assertEqual(_resolve_partial_dtype(torch.bfloat16), torch.bfloat16)
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
                "assert mbsa.__all__ == ['reference_pipeline', 'triton_pipeline', 'prepare_cutedsl_pipeline']; "
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

    def test_bundled_runtime_does_not_take_over_global_modules(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "python")
        script = r"""
import importlib, sys, types
from mbsa.api import _load_runtime
globals_before = {}
for name in ('src', 'interface', 'sparse_index_utils', 'fmha_sm100'):
    globals_before[name] = sys.modules[name] = types.ModuleType(name)
base = 'mbsa.cute'
sentinel = object()
for suffix, names in (
    ('.interface', ('SparseForwardPhase', 'sparse_atten_func')),
    ('.sparse_index_utils', ('build_k2q_csr',)),
    ('.src.sm100.prepare_k2q_csr', ('SparseK2qCsrBuilderSm100', 'allocate_k2q_csr_scratch')),
):
    module = types.ModuleType(base + suffix)
    for name in names:
        setattr(module, name, sentinel)
    sys.modules[module.__name__] = module
path_before = sys.path[:]
_load_runtime()
owned = importlib.import_module(base + ".sm100_sparse")
assert owned.sparse_atten_func is sentinel
assert sys.path == path_before
assert all(sys.modules[name] is value for name, value in globals_before.items())
"""
        result = subprocess.run(
            [sys.executable, "-c", script], env=env, text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
