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
from mbsa import prepare_cutedsl_pipeline

CUTEDSL_ARCHS = ((9, 0), (10, 0), (10, 3))


from tests.common import select_cases, make_inputs


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class FixedWindowRegression(unittest.TestCase):
    def test_graph_capture_waits_for_queued_eager_work(self):
        if torch.cuda.get_device_capability() not in CUTEDSL_ARCHS:
            self.skipTest("SM103 capture stream regression")
        torch.manual_seed(17)
        qn, kn = 128, 4096
        operands = [
            torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            for shape in (
                (qn, 4, 128),
                (kn, 1, 128),
                (qn, 32, 128),
                (kn, 4, 128),
                (kn, 4, 128),
            )
        ]
        p = prepare_cutedsl_pipeline(*operands, overlap_fixed_score=True)
        p.run()
        torch.cuda.synchronize()
        torch.cuda._sleep(100_000_000)
        operands[2].add_(0.125)
        p.run()
        # No host synchronization between the eager launch and capture.
        p.plan.capture_mega_graph()
        p.plan.run_mega()
        graph_out = p.plan.attention.out.clone()
        eager_out = p.run()[0]
        torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)

    def test_fp8_shape_sequence_capture_and_live_query_updates(self):
        if torch.cuda.get_device_capability() not in CUTEDSL_ARCHS:
            self.skipTest("SM103 FP8 shape sequence regression")
        case = select_cases(ids=["p1-d31-kv8192-q4096-boundary"])[0]
        # Reuse compiled kernels across small/tail/full Q shapes and return to
        # the failing shape after releasing each plan and its graph workspace.
        for qn, overlap in itertools.product(
            (127, 129, 4096, 257, 4096), (False, True)
        ):
            with self.subTest(q=qn, overlap=overlap):
                case = {**case, "requests": [{"phase": "prefill", "q": qn, "kv": 8192}]}
                data = make_inputs(case, "permuted")
                pipeline = prepare_cutedsl_pipeline(
                    data["iq"],
                    data["ik"],
                    data["q"],
                    data["pool"][..., 0, :].permute(0, 2, 1, 3),
                    data["pool"][..., 1, :].permute(0, 2, 1, 3),
                    cu_seqlens_q=data["cu_q"],
                    cu_seqlens_k=data["cu_k"],
                    index_page_indices=data["pages"],
                    page_table=data["table"],
                    partial_dtype=torch.float8_e4m3fn,
                    overlap_fixed_score=overlap,
                )
                pipeline.run()
                torch.cuda.synchronize()
                pipeline.plan.capture_mega_graph()
                for shift in (0.125, -0.25, 0.125):
                    data["q"].add_(shift)
                    pipeline.plan.run_mega()
                    graph_out = pipeline.plan.attention.out.clone()
                    graph_lse = pipeline.plan.attention.out_lse.clone()
                    eager_out, eager_lse, _ = pipeline.run()
                    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)
                    torch.testing.assert_close(graph_lse, eager_lse, atol=0, rtol=0)
                del pipeline, data
                gc.collect()
                torch.cuda.empty_cache()

    def test_unaligned_prefix_does_not_include_old_swa_page(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("SM103 TMEM regression")
        qn, kn = 8160, 65536
        q = torch.zeros((qn, 32, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.zeros((kn, 4, 128), device="cuda", dtype=torch.bfloat16)
        v = torch.zeros_like(k)
        v[:, :, 0] = (torch.arange(kn, device="cuda") // 128)[:, None] / 512
        p = prepare_cutedsl_pipeline(
            q[:, :4].contiguous(),
            k[:, :1].contiguous(),
            q,
            k,
            v,
            overlap_fixed_score=False,
        )
        a = p.plan.attention
        a.run_fixed()
        position = torch.arange(kn - qn, kn, device="cuda")
        block = position // 128
        live = position % 128 + 1
        count = 1024 + 7 * 128 + live
        local_sum = (7 * block - 28) * 128 + block * live
        expected = (128 * 28 + local_sum).float() / count / 512
        actual = a.fixed_out
        if actual.ndim == 4:
            actual = actual.permute(1, 0, 2, 3).reshape(qn, 32, 128)
        torch.testing.assert_close(
            actual[:, :, 0].float(),
            expected[:, None].expand(qn, 32),
            atol=0.002,
            rtol=0.002,
        )
        torch.testing.assert_close(
            a.fixed_lse[0].T,
            count.float().log()[:, None].expand(qn, 32),
            atol=1e-5,
            rtol=1e-6,
        )
