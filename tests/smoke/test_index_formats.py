#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Index codecs, page layouts and quantization reference contracts."""

from __future__ import annotations
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import torch

CUTEDSL_ARCHS = ((9, 0), (10, 0), (10, 3))


class IndexFormatCodecTest(unittest.TestCase):
    def test_mx_oracle_keeps_subnormal_fp32_values(self):
        from mbsa.index_formats import (
            dequantize_index_q_mxfp8,
            dequantize_index_k_mxfp6,
            pack_codes,
        )

        # E4M3's smallest subnormal times 2^-127 survives in FP32 but not BF16.
        q = torch.ones((1, 4, 128), dtype=torch.uint8)
        scales = torch.zeros((1, 8), dtype=torch.int32)
        dq = dequantize_index_q_mxfp8(q, scales)
        self.assertEqual(dq.dtype, torch.float32)
        self.assertTrue(bool((dq > 0).all()))
        self.assertTrue(bool((dq.bfloat16() == 0).all()))
        cache = torch.zeros((1, 1, 128, 100), dtype=torch.uint8)
        cache.flatten()[:12288] = pack_codes(torch.ones(16384, dtype=torch.uint8))
        dk = dequantize_index_k_mxfp6(cache)
        self.assertEqual(dk.dtype, torch.float32)
        torch.testing.assert_close(dk, torch.full_like(dk, 2.0**-130), rtol=0, atol=0)

    def test_mxfp6_all_codes_and_page_scale_permutation(self):
        from mbsa.index_formats import (
            build_page,
            decode_e2m3,
            dequantize_index_k_mxfp6,
            encode_e2m3,
            pack_codes,
            quantize_index_k_mxfp6,
            read_page,
            scale_slot,
            unpack_codes,
        )

        codes = torch.arange(64, dtype=torch.uint8).repeat(2)
        torch.testing.assert_close(unpack_codes(pack_codes(codes)), codes)
        encoded = encode_e2m3(decode_e2m3(codes))
        codes[codes == 32] = 0  # negative zero is canonicalized by the codec.
        torch.testing.assert_close(encoded, codes)
        slots = torch.tensor([scale_slot(i) for i in range(128)])
        torch.testing.assert_close(slots.sort().values, torch.arange(128))
        # Distinct token and feature-block exponents catch scale permutations.
        token_scale = torch.exp2((torch.arange(128) % 11 - 5).float())
        block_scale = torch.tensor([0.125, 0.5, 2.0, 16.0]).repeat_interleave(32)
        key = (
            torch.linspace(-7.5, 7.5, 128)[None] * token_scale[:, None] * block_scale
        )[None, None].repeat(3, 1, 1, 1)
        key[1] *= 0.25
        key[2] *= 4.0
        cache = quantize_index_k_mxfp6(key, rotate_first=False)
        for page in range(3):
            expected = build_page(key[page, 0])
            torch.testing.assert_close(cache[page].flatten(), expected, rtol=0, atol=0)
            torch.testing.assert_close(
                dequantize_index_k_mxfp6(cache)[page, 0],
                read_page(expected),
                rtol=0,
                atol=0,
            )

    def test_mxfp8_native_padded_head_slice_and_scale_words(self):
        from mbsa.index_formats import (
            dequantize_index_q_mxfp8,
            quantize_q_mxfp8,
            fake_quant_q,
        )

        torch.manual_seed(71)
        q = torch.randn(5, 4, 128) * torch.exp2(
            torch.tensor([-9.0, -2.0, 3.0, 8.0])
        ).repeat_interleave(32)
        codes, scales = quantize_q_mxfp8(q)
        view = codes[:, :4].view(torch.float8_e4m3fn)
        self.assertFalse(view.is_contiguous())
        torch.testing.assert_close(
            dequantize_index_q_mxfp8(view, scales), fake_quant_q(q), rtol=0, atol=0
        )
        torch.testing.assert_close(codes[:, 4:], torch.zeros_like(codes[:, 4:]))
        torch.testing.assert_close(
            scales[:, 4:], torch.full_like(scales[:, 4:], 0x7F7F7F7F)
        )
        torch.testing.assert_close(
            dequantize_index_q_mxfp8(codes, scales)[:, 4:],
            torch.zeros(5, 4, 128),
            rtol=0,
            atol=0,
        )
