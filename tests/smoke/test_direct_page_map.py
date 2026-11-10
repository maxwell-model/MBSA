# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct page-map ownership and rejection without a CUDA runtime."""
import unittest
from types import SimpleNamespace
import torch
from mbsa.backends.optimized.page_map import page_operand_view, bind_page_operands


class DirectPageMapTest(unittest.TestCase):
    def test_live_compact_rows_and_aligned_offset(self):
        storage = torch.arange(20, dtype=torch.int32)
        rows = (storage[4:8], storage[8:13])
        view = page_operand_view(rows, 16)
        self.assertEqual(view.data_ptr(), rows[0].data_ptr())
        self.assertEqual(view.tolist(), list(range(4, 13)))
        storage[8] = 99
        self.assertEqual(view[4].item(), 99)
        view[0] = 77
        self.assertEqual(storage[4].item(), 77)

    def test_packed_four_byte_alignment(self):
        storage = torch.arange(16, dtype=torch.int32)
        view = page_operand_view((storage[1:4], storage[4:8]), 4)
        self.assertEqual(view.data_ptr(), storage[1:].data_ptr())
        self.assertEqual(view.tolist(), list(range(1, 8)))

    def test_compact_mx_rows_require_only_flat_operand_alignment(self):
        counts = (41, 58, 73, 30)
        storage = torch.arange(sum(counts), dtype=torch.int32)
        rows = storage.split(counts)
        view = page_operand_view(rows, 16)
        self.assertEqual(view.data_ptr(), storage.data_ptr())
        self.assertEqual(view.numel(), sum(counts))
        self.assertEqual(view.tolist(), storage.tolist())
        self.assertNotEqual(rows[1].data_ptr() % 16, 0)
        rows[2][0] = 999
        self.assertEqual(view[sum(counts[:2])].item(), 999)

    def test_reject_incompatible_active_rows(self):
        storage = torch.arange(32, dtype=torch.int32)
        cases = [((storage[::2],), 4), ((storage[1:9],), 16),
                 ((storage[:4], storage[8:12]), 4),
                 ((storage[:4], storage[4:8].clone()), 4), ((), 4)]
        for rows, alignment in cases:
            with self.subTest(alignment=alignment, lengths=[r.numel() for r in rows]):
                with self.assertRaisesRegex(ValueError, "direct Index page map"):
                    page_operand_view(rows, alignment)

    def test_rebind_failure_preserves_previous_operand(self):
        for packed, mx, alignment in ((False, False, 16), (True, False, 4), (True, True, 16)):
            with self.subTest(packed=packed, mx=mx):
                old = torch.arange(8, dtype=torch.int32)
                name = "cutedsl_packed_pages" if packed else "cutedsl_index_pages"
                plan = SimpleNamespace(cutedsl_index_pages=None, cutedsl_packed_pages=None,
                                       mxfp6_router=object() if mx else None)
                setattr(plan, name, old)
                backing = torch.arange(32, dtype=torch.int32)
                with self.assertRaisesRegex(ValueError, "direct Index page map"):
                    bind_page_operands(plan, (backing[::2],))
                self.assertIs(getattr(plan, name), old)
                replacement = backing[4:12]
                bind_page_operands(plan, (replacement,))
                self.assertEqual(getattr(plan, name).data_ptr(), replacement.data_ptr())
                replacement[0] = 101
                self.assertEqual(getattr(plan, name)[0].item(), 101)
                self.assertEqual(old[0].item(), 0)

    def test_serving_preserves_compact_map(self):
        from mbsa.serving import PrefillStep
        pages = torch.arange(51, dtype=torch.int32)
        table = torch.zeros((2, 33), dtype=torch.int32)
        owner = SimpleNamespace(backend="optimized")
        step = PrefillStep(owner, ((17, 4113), (23, 2195)), table, pages)
        self.assertIs(step._index_page_table, pages)
        # Q > 8192 is split into pieces; the pieces' map is gathered when the step is prepared.
        for index_map in (pages, pages.view(1, -1)):
            split = PrefillStep(owner, ((8193, 16384),), table, index_map)
            self.assertEqual(split.pieces, ((8192, 16383), (1, 16384)))
            self.assertEqual(split._index_page_table.data_ptr(), index_map.data_ptr())

    def test_fixed_only_has_no_index_operand(self):
        plan = SimpleNamespace(cutedsl_index_pages=None, cutedsl_packed_pages=None)
        bind_page_operands(plan, ())
        self.assertIsNone(plan.cutedsl_index_pages)
        self.assertIsNone(plan.cutedsl_packed_pages)


if __name__ == "__main__":
    unittest.main()
