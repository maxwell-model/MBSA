#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Loading and include resolution for the CUDA C++ extensions.

A prebuilt module must win over a JIT build unless told otherwise, and never
be silently replaced by one when only prebuilt modules are allowed. CUTLASS/
CuTe headers are not vendored, so a missing location must fail with the
environment variable to set, and an explicit location must win.
"""

from __future__ import annotations
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from mbsa import _cuda_jit


def _cutlass_tree(root):
    for name in ("cute", "cutlass"):
        (Path(root) / name).mkdir()
    return root


class CutlassIncludeTest(unittest.TestCase):
    def setUp(self):
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in (_cuda_jit.CUTLASS_INCLUDE_ENV, "CUTLASS_HOME")
        }
        patches = (
            patch.dict(os.environ, env, clear=True),
            patch.object(_cuda_jit, "_package_dir", return_value=None),
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_nothing_found_names_the_environment_variable(self):
        with self.assertRaisesRegex(RuntimeError, _cuda_jit.CUTLASS_INCLUDE_ENV):
            _cuda_jit.cutlass_include_dir()

    def test_an_invalid_explicit_directory_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as empty, tempfile.TemporaryDirectory() as home:
            os.mkdir(os.path.join(home, "include"))
            _cutlass_tree(os.path.join(home, "include"))
            os.environ["CUTLASS_HOME"] = home
            os.environ[_cuda_jit.CUTLASS_INCLUDE_ENV] = empty
            with self.assertRaisesRegex(RuntimeError, _cuda_jit.CUTLASS_INCLUDE_ENV):
                _cuda_jit.cutlass_include_dir()

    def test_explicit_directory_wins_over_cutlass_home(self):
        with tempfile.TemporaryDirectory() as explicit, tempfile.TemporaryDirectory() as home:
            os.mkdir(os.path.join(home, "include"))
            _cutlass_tree(os.path.join(home, "include"))
            os.environ["CUTLASS_HOME"] = home
            self.assertEqual(
                _cuda_jit.cutlass_include_dir(), os.path.join(home, "include")
            )
            os.environ[_cuda_jit.CUTLASS_INCLUDE_ENV] = _cutlass_tree(explicit)
            self.assertEqual(_cuda_jit.cutlass_include_dir(), explicit)

    def test_package_locations_follow_the_environment(self):
        with tempfile.TemporaryDirectory() as pkg:
            include = Path(pkg, "third_party", "deep_gemm", "include")
            include.mkdir(parents=True)
            _cutlass_tree(include)
            dirs = {"vllm": pkg}
            with patch.object(_cuda_jit, "_package_dir", side_effect=dirs.get):
                self.assertEqual(_cuda_jit.cutlass_include_dir(), str(include))


class LoadExtensionTest(unittest.TestCase):
    PACKAGE = "mbsa"

    def _load(self, mode, name):
        calls = []

        def jit():
            calls.append(name)
            return f"jit:{name}"

        with patch.dict(os.environ, {_cuda_jit.EXTENSIONS_ENV: mode}):
            return _cuda_jit.load_extension(self.PACKAGE, name, jit), calls

    def test_a_prebuilt_module_is_used_without_compiling(self):
        prebuilt = ModuleType(f"{self.PACKAGE}._fake_prebuilt_ext")
        with patch.dict(sys.modules, {prebuilt.__name__: prebuilt}):
            module, calls = self._load("auto", "_fake_prebuilt_ext")
            self.assertIs(module, prebuilt)
            self.assertEqual(calls, [])
            self.assertEqual(_cuda_jit.LOADED["_fake_prebuilt_ext"], "prebuilt")
            module, calls = self._load("jit", "_fake_prebuilt_ext")
            self.assertEqual(calls, ["_fake_prebuilt_ext"])
            self.assertEqual(_cuda_jit.LOADED["_fake_prebuilt_ext"], "jit")

    def test_a_missing_prebuilt_module_is_compiled_unless_forbidden(self):
        module, calls = self._load("auto", "_no_such_ext")
        self.assertEqual((module, calls), ("jit:_no_such_ext", ["_no_such_ext"]))
        with self.assertRaisesRegex(RuntimeError, _cuda_jit.EXTENSIONS_ENV):
            self._load("prebuilt", "_no_such_ext")
        with self.assertRaisesRegex(ValueError, _cuda_jit.EXTENSIONS_ENV):
            self._load("sometimes", "_no_such_ext")


if __name__ == "__main__":
    unittest.main()
