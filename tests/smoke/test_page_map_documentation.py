# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep public page-map docstrings consistent with compact MX row support."""
import ast
from pathlib import Path
import unittest


class PageMapDocumentation(unittest.TestCase):
    def test_compact_maps_require_aligned_base_not_every_row(self):
        root = Path(__file__).resolve().parents[2] / 'python/mbsa'
        counts = {'api.py': 3, 'backends/default.py': 2,
                  'backends/optimized/runtime.py': 1, 'backends/optimized/index.py': 1}
        for name, count in counts.items():
            with self.subTest(path=name):
                tree = ast.parse((root / name).read_text())
                docs = [ast.get_docstring(n) for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef)]
                docs = [' '.join(d.split()) for d in docs
                        if d and 'Optimized binds active rows' in d]
                self.assertEqual(len(docs), count)
                for doc in docs:
                    self.assertIn('base alignment', doc)
                    self.assertIn('need not be 16-byte aligned', doc)
                    self.assertNotIn('each row aligned to 16 bytes', doc)
                    self.assertIn('Gaps/strides are rejected', doc)
