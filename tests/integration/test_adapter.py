#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU/GPU contracts and full-query comparison against pinned HF Attention.

The shared config owns every shape and gate. Decode reserves scheduler budget;
only prefill enters the operator. A strict mismatch is a failure, including ties.
"""

from __future__ import annotations
import gc
import itertools
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import torch
from mbsa import reference_pipeline

CUTEDSL_ARCHS = ((9, 0), (10, 0), (10, 3))


from tests.common import BackendCaseMixin


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PagedPrefillIntegration(BackendCaseMixin, unittest.TestCase):
    def test_vllm_adapter_reuses_packed_plan_with_new_layer_and_page_maps(self):
        """Packed routing and output ownership survive a real layer rebind."""
        self._check_vllm_adapter_rebind(use_graphs=False)

    def test_vllm_graph_adapter_replays_updated_queries_cache_and_page_maps(self):
        """Captured kernels must consume live values through stable staging."""
        self._check_vllm_adapter_rebind(use_graphs=True)

    def test_vllm_adapter_quantized_index_formats_and_partial_dtypes(self):
        """Native index layouts and scale staging survive layer and graph reuse."""
        if torch.cuda.get_device_capability() not in CUTEDSL_ARCHS:
            self.skipTest("quantized Index Router requires SM100/SM103")
        for fmt, graphs, partial in itertools.product(
            ("fp8",)
            if torch.cuda.get_device_capability() == (9, 0)
            else ("fp8", "mxfp8_mxfp6"),
            (False, True),
            (torch.bfloat16, torch.float8_e4m3fn),
        ):
            with self.subTest(index_format=fmt, graphs=graphs, partial=partial):
                self._check_vllm_adapter_rebind(graphs, fmt, partial)
            gc.collect()
            torch.cuda.empty_cache()

    def _check_vllm_adapter_rebind(
        self, use_graphs, index_format="bf16", partial_dtype=torch.bfloat16
    ):
        if torch.cuda.get_device_capability() not in CUTEDSL_ARCHS:
            self.skipTest("CuTe DSL runtime requires SM90/SM100/SM103")
        from mbsa.sparse_fmha_adapter import PagedPrefill
        from mbsa.index_formats import (
            dequantize_index_k_mxfp6,
            dequantize_index_q_mxfp8,
            quantize_index_k_mxfp6,
            quantize_q_mxfp8,
        )

        lengths = ((1, 130), (1, 4098), (17, 8193))
        counts = [(kn + 127) // 128 for _, kn in lengths]
        adapter = PagedPrefill(max_cached_plans=1, use_graphs=use_graphs)
        rounds = 3 if use_graphs else 2
        runtime_ik = None
        for iteration in range(rounds):
            operands = [
                tuple(t.add(iteration * 0.125) for t in self.inputs(qn, kn))
                for qn, kn in lengths
            ]
            iq, q = (torch.cat([row[i] for row in operands]) for i in (0, 2))
            if iteration != 2:
                pool = torch.zeros(
                    (sum(counts), 128, 4, 2, 128), device="cuda", dtype=q.dtype
                )
                ik = torch.zeros(
                    (sum(counts), 1, 128, 128), device="cuda", dtype=q.dtype
                )
            pages = torch.arange(sum(counts), device="cuda", dtype=torch.int32)
            if iteration % 2:
                pages = pages.flip(0)
            table = torch.zeros(
                (len(lengths), max(counts)), device="cuda", dtype=torch.int32
            )
            page_start = 0
            for slot, (row, count) in enumerate(zip(operands, counts)):
                ids = pages[page_start : page_start + count]
                table[slot, :count].copy_(ids)
                for half, tensor in enumerate(row[3:5]):
                    padded = torch.zeros(
                        (count * 128, 4, 128), device="cuda", dtype=q.dtype
                    )
                    padded[: tensor.shape[0]].copy_(tensor)
                    pool[..., half, :].index_copy_(
                        0, ids.long(), padded.view(count, 128, 4, 128)
                    )
                padded = torch.zeros(
                    (count * 128, 1, 128), device="cuda", dtype=q.dtype
                )
                padded[: row[1].shape[0]].copy_(row[1])
                ik.index_copy_(0, ids.long(), padded.view(count, 1, 128, 128))
                page_start += count
            scales = None
            if index_format == "fp8":
                runtime_iq = iq.to(torch.float8_e4m3fn)
                encoded_k = ik.to(torch.float8_e4m3fn)
                oracle_iq = runtime_iq.float()
                oracle_ik = encoded_k.float()
            elif index_format == "mxfp8_mxfp6":
                codes, scales = quantize_q_mxfp8(iq)
                runtime_iq = codes[:, :4]
                encoded_k = quantize_index_k_mxfp6(ik)
                oracle_iq = dequantize_index_q_mxfp8(runtime_iq, scales)
                oracle_ik = dequantize_index_k_mxfp6(encoded_k)
            else:
                runtime_iq, encoded_k, oracle_iq, oracle_ik = iq, ik, iq, ik
            if iteration == 2:
                self.assertIsNotNone(runtime_ik)
                runtime_ik.copy_(encoded_k)
            else:
                runtime_ik = encoded_k
            out = torch.empty_like(q)
            actual = adapter(
                q,
                pool[..., 0, :],
                pool[..., 1, :],
                index_q=runtime_iq,
                index_q_scales=scales,
                index_k_cache=runtime_ik[:, 0],
                index_page_indices=pages,
                page_table=table,
                request_lens=lengths,
                out=out,
                partial_dtype=partial_dtype,
            )
            self.assertEqual(actual.data_ptr(), out.data_ptr())
            pipeline = next(iter(adapter.plans.values()))
            lse = pipeline.plan.attention.out_lse
            route = pipeline.plan.attention.tile_plan.dynamic_route
            self.assertEqual(
                pipeline.plan.attention.dynamic_workspace[0].dtype, partial_dtype
            )
            start = 0
            for slot, (row, (qn, kn)) in enumerate(zip(operands, lengths)):
                qs = slice(start, start + qn)
                physical = table[slot, : counts[slot]].long()
                expected_ik = oracle_ik.index_select(0, physical).reshape(-1, 1, 128)
                self.compare(
                    (actual[qs], lse[qs], route[:, qs]),
                    reference_pipeline(oracle_iq[qs], expected_ik[:kn], *row[2:]),
                )
                start += qn
        self.assertEqual((adapter.hits, adapter.misses), (rounds - 1, 1))
        if use_graphs:
            self.assertEqual(len(pipeline._vllm_graphs), 2)
