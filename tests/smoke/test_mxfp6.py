#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts of the MXFP6 index kernels: import, sources and build errors.

The CUDA sources must build from ``csrc/`` plus CUTLASS alone, and a missing
CUTLASS or a non-SM100 device must fail with a message naming the fix.
"""

from __future__ import annotations
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

CSRC = ROOT / "python/mbsa/mxfp6/csrc"
SYSTEM_PREFIXES = ("cute/", "cutlass/", "cuda/", "torch/", "c10/")


class Mxfp6PackageTest(unittest.TestCase):
    def test_import_loads_neither_triton_nor_the_extensions(self):
        env = {**os.environ, "PYTHONPATH": str(ROOT / "python")}
        script = (
            "import sys; import mbsa.mxfp6 as m; "
            "assert (m.INDEX_HEADS, m.MAX_DECODE_REQS, m.MAX_DECODE_QUERY_LEN) "
            "== (4, 256, 4); "
            "assert 'triton' not in sys.modules; "
            "assert 'torch.utils.cpp_extension' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], env=env, text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_sources_need_only_their_directory_and_cutlass(self):
        include = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.M)
        for path in sorted(CSRC.iterdir()):
            for kind, name in include.findall(path.read_text()):
                with self.subTest(file=path.name, include=name):
                    if kind == '"':
                        self.assertTrue((CSRC / name).is_file())
                    else:
                        self.assertFalse(name.startswith("deep_gemm/"))
                        self.assertTrue(
                            name.startswith(SYSTEM_PREFIXES) or "/" not in name
                        )

    def test_a_jit_build_without_cutlass_names_the_environment_variable(self):
        """Forced to JIT with every CUTLASS candidate gone; ``_load`` is not
        cached, so an extension another test already loaded cannot mask it."""
        from mbsa import _cuda_jit
        from mbsa.mxfp6 import scoring

        env = {
            k: v
            for k, v in os.environ.items()
            if k not in (_cuda_jit.CUTLASS_INCLUDE_ENV, "CUTLASS_HOME")
        }
        env[_cuda_jit.EXTENSIONS_ENV] = "jit"
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(_cuda_jit, "cutlass_include_candidates", return_value=[]),
        ):
            with self.assertRaisesRegex(RuntimeError, _cuda_jit.CUTLASS_INCLUDE_ENV):
                scoring._load("fp6_ext.cu", "mbsa_mxfp6_prefill")

    def test_non_sm100_devices_are_rejected(self):
        from mbsa.mxfp6 import scoring

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            with self.assertRaisesRegex(RuntimeError, "SM100"):
                scoring._arch()

    def test_shared_schedule_is_computed_once_per_owner(self):
        import torch
        from mbsa.mxfp6 import scoring

        calls = []

        def fake(seq_lens, next_n):
            calls.append(next_n)
            return torch.tensor([len(calls)])

        lens = torch.tensor([3, 5], dtype=torch.int32)
        with patch.object(scoring, "schedule_metadata", side_effect=fake):
            step = SimpleNamespace()
            first = scoring.shared_schedule(lens, 1, step)
            self.assertIs(scoring.shared_schedule(lens, 1, step), first)
            self.assertIsNot(scoring.shared_schedule(lens, 2, step), first)
            self.assertIsNot(scoring.shared_schedule(lens, 1, SimpleNamespace()), first)
            scoring.shared_schedule(lens, 1)
        self.assertEqual(calls, [1, 2, 1, 1])


if __name__ == "__main__":
    unittest.main()
