# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Index tuning boundaries without a CUDA runtime or large allocations."""
import unittest
from types import SimpleNamespace
import torch
from mbsa.backends.optimized.index_policy import (
    select_partition_blocks, lean192_index_eligible, _validate_index_token_extent,
)


def operand(shape, dtype, contiguous=True):
    return SimpleNamespace(shape=shape, dtype=dtype,
                           is_contiguous=lambda: contiguous)


class IndexPolicyTest(unittest.TestCase):
    def test_partition_rounding_int32_domain(self):
        for blocks in (1024, 2048, 4096, 8192):
            limit = (1 << 31) - 128 * blocks
            with self.subTest(partition_blocks=blocks):
                self.assertIsNone(_validate_index_token_extent(limit, blocks))
                self.assertIsNone(_validate_index_token_extent(1056768, blocks))
                for invalid in (0, -1, limit + 1, 1 << 31):
                    with self.assertRaisesRegex(ValueError, "signed int32"):
                        _validate_index_token_extent(invalid, blocks)

    def test_fp8_partition_boundaries(self):
        for kv, expected in ((8192, 2048), (131072, 2048), (139264, 2048),
                             (188416, 2048), (196608, 2048), (229376, 2048),
                             (262144, 2048), (270336, 2048), (524288, 2048),
                             (1048576, 2048), (196609, 1024)):
            for shape in ((kv, 1, 128), (kv // 128, 1, 128, 128)):
                with self.subTest(kv=kv, shape=shape):
                    args = dict(index_q=operand((8192, 4, 128), torch.float8_e4m3fn),
                                index_k=operand(shape, torch.float8_e4m3fn),
                                batch=1, contiguous_unpacked=True, kv_tokens=kv,
                                identity_pages=True, capability=(10, 3))
                    self.assertEqual(select_partition_blocks(**args), expected)
                    for changed in ({"capability": (10, 0)}, {"capability": (9, 0)},
                                    {"identity_pages": False}, {"identity_pages": None},
                                    {"batch": 2}, {"contiguous_unpacked": False},
                                    {"index_k": operand(shape, torch.float8_e4m3fn, False)},
                                    {"index_k": operand((kv + 128, 1, 128), torch.float8_e4m3fn)},
                                    {"index_k": operand((kv // 128 + 1, 1, 128, 128), torch.float8_e4m3fn)},
                                    {"index_q": operand((8192, 4, 128), torch.bfloat16)},
                                    {"index_q": operand((4096, 4, 128), torch.float8_e4m3fn)}):
                        self.assertEqual(select_partition_blocks(**(args | changed)), 1024)

    def test_lean192_geometry_and_tail(self):
        for kv, expected in ((4096, False), (8192, True), (32768, True),
                             (262144, True), (270336, True), (524288, True),
                             (786432, True), (1048576, True), (1056768, True),
                             (2147352576, True), (8193, False)):
            args = dict(q=operand((8192, 4, 128), torch.bfloat16),
                        k=operand((kv, 1, 128), torch.bfloat16),
                        partition_blocks=1024, logical_k_tokens=kv,
                        use_page_table=False)
            with self.subTest(kv=kv):
                self.assertEqual(lean192_index_eligible(**args), expected)
                for changed in ({"partition_blocks": 2048}, {"use_page_table": True},
                                {"k": operand((kv + 128, 1, 128), torch.bfloat16)},
                                {"q": operand((8192, 4, 128), torch.float16)},
                                {"q": operand((4096, 4, 128), torch.bfloat16)}):
                    self.assertFalse(lean192_index_eligible(**(args | changed)))

    def test_lean192_canonical_page_shaped_storage(self):
        pages = torch.empty((64, 1, 128, 128), dtype=torch.bfloat16)
        dense = pages.view(-1, 1, 128)
        self.assertEqual(pages.untyped_storage().data_ptr(),
                         dense.untyped_storage().data_ptr())
        args = dict(q=operand((8192, 4, 128), torch.bfloat16),
                    partition_blocks=1024, logical_k_tokens=8192,
                    use_page_table=False)
        self.assertTrue(lean192_index_eligible(k=dense, **args))
        # The helper consumes the validated canonical view, not arbitrary
        # page geometry. Extra physical pages and mutable maps stay excluded.
        self.assertFalse(lean192_index_eligible(k=pages, **args))
        self.assertFalse(lean192_index_eligible(
            k=dense, **(args | {"logical_k_tokens": 8191})))
        self.assertFalse(lean192_index_eligible(
            k=dense, **(args | {"use_page_table": True})))


if __name__ == "__main__":
    unittest.main()
