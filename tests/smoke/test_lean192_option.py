# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pure host validation of the explicit policy, without importing CUDA backends."""
import importlib.util
from pathlib import Path
import unittest


def options_module():
    path = Path(__file__).resolve().parents[2] / 'python/mbsa/backends/_index_options.py'
    spec = importlib.util.spec_from_file_location('lean192_host_options', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Lean192Option(unittest.TestCase):
    def test_auto_preserves_existing_eligibility(self):
        option = options_module()
        for eligible in (False, True):
            self.assertEqual(option.resolve_lean192('auto', eligible), eligible)

    def test_off_changes_only_the_selected_implementation(self):
        option = options_module()
        for eligible in (False, True):
            self.assertFalse(option.resolve_lean192('off', eligible))

    def test_on_never_silently_falls_back(self):
        option = options_module()
        self.assertTrue(option.resolve_lean192('on', True))
        with self.assertRaisesRegex(ValueError, 'SM103 BF16 direct-addressing'):
            option.resolve_lean192('on', False)

    def test_invalid_options_fail_without_gpu_import(self):
        option = options_module()
        for invalid in (None, True, False, 0, 1, [], {}, 'ON', 'OFF', 'yes', ''):
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaisesRegex(ValueError, 'lean192 must be'):
                    option.validate_lean192(invalid)
                with self.assertRaises(ValueError):
                    option.resolve_lean192(invalid, True)

    def test_nonoptimized_override_is_not_a_fake_control(self):
        option = options_module()
        for backend in ('default', 'optimized'):
            self.assertEqual(option.validate_lean192('auto', backend=backend), 'auto')
        for mode in ('on', 'off'):
            self.assertEqual(option.validate_lean192(mode, backend='optimized'), mode)
            with self.assertRaisesRegex(ValueError, 'requires backend="optimized"'):
                option.validate_lean192(mode, backend='default')


if __name__ == '__main__':
    unittest.main()
