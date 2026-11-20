# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public backend selection, independent plans and transactional lifecycle."""

import unittest

import torch

from mbsa import prepare_cutedsl_pipeline
from mbsa.serving import Prefill, runner
from tests.common import BackendCaseMixin, make_inputs
from tests.reference import reference_pipeline


class BackendSelection(unittest.TestCase):
    def test_invalid_selection_and_runner_identity(self):
        with self.assertRaisesRegex(ValueError, "backend"):
            prepare_cutedsl_pipeline(*([None] * 5), backend="typo")
        with self.assertRaisesRegex(ValueError, "backend"):
            prepare_cutedsl_pipeline(*([None] * 5), backend="unsupported")
        a = runner(torch.device("cuda:0"), backend="default")
        b = runner(torch.device("cuda:0"), backend="optimized")
        self.assertIsNot(a, b)
        self.assertIs(b, runner(torch.device("cuda:0"), backend="optimized"))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Sm90BackendLifecycle(BackendCaseMixin, unittest.TestCase):
    def test_default_public_rebind_graph_and_explicit_rejection(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("SM90 required")
        inputs = self.inputs(17, 4096)
        iq, ik, q, k, v = inputs
        pages = torch.arange(32, device=q.device, dtype=torch.int32)
        table = pages[None]
        kp, vp = (t.view(32, 128, 4, 128).permute(0, 2, 1, 3) for t in (k, v))
        operands = iq, ik.view(32, 1, 128, 128), q, kp, vp
        kwargs = dict(index_page_indices=pages, page_table=table, logical_k_tokens=4096)
        with self.assertRaisesRegex(ValueError, "optimized backend requires"):
            prepare_cutedsl_pipeline(*operands, backend="optimized", **kwargs)
        with self.assertRaisesRegex(ValueError, "optimized backend requires"):
            Prefill(q.device, backend="optimized").stage(((17, 4096),), table, table).run(
                q, kp, vp, index_q=iq, index_k_cache=ik.view(32, 128, 128),
                out=torch.empty_like(q))
        p = prepare_cutedsl_pipeline(*operands, **kwargs)
        self.assertEqual(p.backend, "default")
        expected = reference_pipeline(*inputs)
        self.compare(p.run(), expected)
        bad = table.clone()
        bad[0, 0] = -1
        new_out = torch.full_like(q, 123)
        with self.assertRaises(ValueError):
            p.rebind(*operands, index_page_indices=pages, page_table=bad, out=new_out)
        self.compare(p.run(), expected)
        p.rebind(*operands, index_page_indices=pages, page_table=table.clone(), out=new_out)
        self.assertIs(p.plan.run.__self__, p.plan)
        self.compare(p.run(), expected)
        self.assertEqual(p.run()[0].data_ptr(), new_out.data_ptr())
        p.capture_mega_graph()
        q.mul_(0.875)
        expected = reference_pipeline(*inputs)
        self.compare(p.run_mega(), expected)
        with self.assertRaisesRegex(RuntimeError, "captured"):
            p.rebind(*operands, index_page_indices=pages, page_table=table, out=new_out)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class BackendLifecycle(BackendCaseMixin, unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("SM100/SM103 required")

    def fixture(self, lengths):
        data = make_inputs({"requests": [dict(q=q, kv=kv) for q, kv in lengths]}, "permuted")
        k = data["pool"][:, :, :, 0].permute(0, 2, 1, 3)
        v = data["pool"][:, :, :, 1].permute(0, 2, 1, 3)
        inputs = data["iq"], data["ik"], data["q"], k, v
        kwargs = dict(index_page_indices=data["pages"], page_table=data["table"],
                      partial_dtype=torch.bfloat16)
        if len(lengths) == 1:
            kwargs["logical_k_tokens"] = lengths[0][1]
        else:
            kwargs.update(cu_seqlens_q=data["cu_q"], cu_seqlens_k=data["cu_k"],
                          request_lens=lengths)
        return data, inputs, kwargs

    def reference(self, data, inputs, lengths):
        results = []
        q_start = 0
        page_start = 0
        for row, (qn, kn) in enumerate(lengths):
            ids = data["table"][row, : (kn + 127) // 128].long()
            index_ids = data["pages"][page_start:page_start + ids.numel()].long()
            ik = inputs[1][index_ids].reshape(-1, 1, 128)[:kn]
            k, v = (x[ids].permute(0, 2, 1, 3).reshape(-1, 4, 128)[:kn]
                    for x in inputs[3:])
            qs = slice(q_start, q_start + qn)
            results.append(reference_pipeline(inputs[0][qs], ik, inputs[2][qs], k, v))
            q_start += qn
            page_start += ids.numel()
        return tuple(torch.cat([r[i] for r in results], dim=1 if i == 2 else 0)
                     for i in range(3))

    def test_paged_mixed_requests_coexistence_and_rejected_rebind(self):
        lengths = ((17, 4113), (23, 2195), (41, 127))
        data, inputs, kwargs = self.fixture(lengths)
        expected = self.reference(data, inputs, lengths)
        plans = [prepare_cutedsl_pipeline(*inputs, backend=backend, **kwargs)
                 for backend in ("default", "optimized")]
        self.assertIsNot(plans[0].plan.attention, plans[1].plan.attention)
        for p in plans:
            self.compare(p.run(), expected)
            saved = tuple(t.clone() for t in p.run())
            bad = data["table"].clone()
            bad[0, 0] = -1
            out = torch.full_like(inputs[2], 123)
            with self.assertRaises(ValueError):
                p.rebind(*inputs, index_page_indices=data["pages"], page_table=bad, out=out)
            for actual, original in zip(p.run(), saved):
                torch.testing.assert_close(actual, original, rtol=0, atol=0)
            self.assertTrue(bool(torch.all(out == 123)))
            p.rebind(*inputs, index_page_indices=data["pages"],
                     page_table=data["table"].clone(), out=out)
            self.compare(p.run(), expected)
            self.assertEqual(p.run()[0].data_ptr(), out.data_ptr())
            # The address is unchanged but Fixed's physical mapping must follow
            # the new page IDs, independently of the Index page order.
            table = p.plan.attention.tile_plan.page_table
            for row, count in enumerate(data["counts"]):
                table[row, :count].copy_(table[row, :count].roll(1))
            p.rebind(*inputs, index_page_indices=data["pages"], page_table=table, out=out)
            expected_now = self.reference(dict(data, table=table), inputs, lengths)
            self.compare(p.run(), expected_now)
            p.capture_mega_graph()
            self.compare(p.run_mega(), expected_now)
            with self.assertRaisesRegex(RuntimeError, "captured"):
                p.rebind(*inputs, index_page_indices=data["pages"],
                         page_table=data["table"], out=out)

    def test_independent_streams_and_compatibility_external_capture(self):
        lengths = ((17, 4113),)
        data, inputs, kwargs = self.fixture(lengths)
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        plans, bound_inputs, expected = [], [], []
        for offset, (stream, backend) in enumerate(zip(streams, ("default", "optimized"))):
            operands = tuple(torch.empty_strided(t.shape, t.stride(), device=t.device,
                                                dtype=t.dtype).copy_(t).add_(offset * 0.125)
                             for t in inputs)
            bound_inputs.append(operands)
            expected.append(self.reference(data, operands, lengths))
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                plans.append(prepare_cutedsl_pipeline(*operands, backend=backend, **kwargs))
        for stream, plan in zip(streams, plans):
            with torch.cuda.stream(stream):
                plan.run()
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        self.assertNotEqual(plans[0].plan.attention.out.data_ptr(),
                            plans[1].plan.attention.out.data_ptr())
        for plan, oracle in zip(plans, expected):
            self.compare((plan.plan.attention.out, plan.plan.attention.out_lse,
                          plan.plan.attention.tile_plan.dynamic_route), oracle)
        legacy_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(legacy_graph):
            plans[0].plan.run()
        legacy_graph.replay()
        self.compare((plans[0].plan.attention.out, plans[0].plan.attention.out_lse,
                      plans[0].plan.attention.tile_plan.dynamic_route), expected[0])
        public_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(public_graph):
            live = plans[1].run()
        addresses = tuple(t.data_ptr() for t in live)
        operands = bound_inputs[1]
        for _ in range(2):
            operands[0].copy_(operands[0].roll(1, dims=-1).mul_(0.875))
            operands[2].add_(0.125)
            operands[4].sub_(0.0625)
            oracle = self.reference(data, operands, lengths)
            public_graph.replay()
            self.assertEqual(tuple(t.data_ptr() for t in live), addresses)
            self.compare(live, oracle)
        for plan, operands in zip(plans, bound_inputs):
            with self.assertRaisesRegex(RuntimeError, "captured"):
                plan.rebind(*operands, index_page_indices=data["pages"],
                            page_table=data["table"], out=torch.empty_like(operands[2]))

    def test_serving_backend_layer_reuse(self):
        lengths = ((17, 4113),)
        for backend in ("default", "optimized"):
            data, inputs, _ = self.fixture(lengths)
            step = Prefill(inputs[2].device, backend=backend).stage(
                lengths, data["table"], data["pages"].view(1, -1))
            out = torch.full_like(inputs[2], 123)
            step.run(inputs[2], *inputs[3:], index_q=inputs[0],
                     index_k_cache=inputs[1].view(-1, 128, 128), out=out)
            replacement = tuple(torch.empty_strided(t.shape, t.stride(), device=t.device,
                                                   dtype=t.dtype).copy_(t).add_(0.125)
                                for t in inputs)
            new_out = torch.full_like(out, 123)
            self.assertTrue(all(a.data_ptr() != b.data_ptr() for a, b in zip(inputs, replacement)))
            actual = step.run(replacement[2], *replacement[3:], index_q=replacement[0],
                              index_k_cache=replacement[1].view(-1, 128, 128), out=new_out)
            self.assertIs(actual, new_out)
            expected = self.reference(data, replacement, lengths)
            self.compare((actual, step.plan.attention.out_lse,
                          step.plan.attention.tile_plan.dynamic_route), expected)
            # Keep and poison every previous operand, then execute without a
            # second bind. Any stale prepared launch argument must now fail.
            for old in inputs:
                old.fill_(torch.nan)
            self.compare((*step.plan.run(), step.plan.dynamic_route), expected)
            self.assertEqual(step.plan.attention.out.data_ptr(), new_out.data_ptr())

    def test_optimized_mx_scale_binding_public_and_serving(self):
        from mbsa.index_formats import (
            quantize_q_mxfp8, quantize_index_k_mxfp6,
            dequantize_index_q_mxfp8, dequantize_index_k_mxfp6,
        )
        lengths = ((17, 4113),)
        data, inputs, kwargs = self.fixture(lengths)
        codes, scales = quantize_q_mxfp8(inputs[0])
        iq = codes[:, :4].view(torch.float8_e4m3fn)
        ik = quantize_index_k_mxfp6(inputs[1])
        mx_inputs = iq, ik, *inputs[2:]

        def expected(bound_scales):
            decoded = (dequantize_index_q_mxfp8(iq, bound_scales),
                       dequantize_index_k_mxfp6(ik), *inputs[2:])
            return self.reference(data, decoded, lengths)

        for backend in ("default", "optimized"):
            p = prepare_cutedsl_pipeline(*mx_inputs, backend=backend,
                                         index_q_scales=scales, **kwargs)
            if backend == "optimized":
                self.assertFalse(p.plan.attention.dynamic_route_fully_visible)
                self.assertEqual(p.plan.attention.dynamic_kernel_cfg.num_wgs, 4)
                self.assertFalse(p.plan.attention.dynamic_kernel_cfg.use_clc)
            saved = tuple(t.clone() for t in p.run())
            self.compare(saved, expected(scales))
            new_scales = scales.clone().bitwise_xor_(1 << 8)
            out = torch.full_like(inputs[2], 123)
            with self.assertRaisesRegex(ValueError, "requires index_q_scales"):
                p.rebind(*mx_inputs, index_page_indices=data["pages"],
                         page_table=data["table"], out=out)
            for actual, original in zip(p.run(), saved):
                torch.testing.assert_close(actual, original, rtol=0, atol=0)
            p.rebind(*mx_inputs, index_q_scales=new_scales,
                     index_page_indices=data["pages"], page_table=data["table"], out=out)
            self.assertIs(p.plan.index_q_scales, new_scales)
            self.assertIs(p.plan.mxfp6_router.tensors[1], new_scales)
            self.compare(p.run(), expected(new_scales))
            step = Prefill(inputs[2].device, backend=backend).stage(
                lengths, data["table"], data["pages"].view(1, -1))
            for bound_scales in (scales, new_scales):
                step.run(inputs[2], *inputs[3:], index_q=iq,
                         index_k_cache=ik.view(-1, 128, 100), out=out,
                         index_q_scales=bound_scales)
                self.assertIs(step.plan.index_q_scales, bound_scales)
                self.compare((out, step.plan.attention.out_lse,
                              step.plan.attention.tile_plan.dynamic_route), expected(bound_scales))

    def test_optimized_native_fp8_layer_reuse(self):
        from tests.regression.test_kvfp8 import KVFP8Regression, FP8
        fixture = KVFP8Regression()
        fixture.device = torch.device("cuda", torch.cuda.current_device())
        lengths = ((17, 4096), (5, 2177))
        operands, table = fixture.case(lengths, q_dtype=FP8, index_format="bf16")
        step = Prefill(fixture.device, backend="optimized").stage(lengths, table, table)
        step.run(**operands)
        fixture.check(step, operands, table, lengths)
        rebound, _ = fixture.case(lengths, q_dtype=FP8, index_format="bf16")
        rebound["q"].copy_(rebound["q"].float().mul_(0.75).to(FP8))
        step.run(**rebound)
        fixture.check(step, rebound, table, lengths)
        step.plan.capture_mega_graph()
        rebound["q"].copy_(rebound["q"].float().mul_(0.75).to(FP8))
        step.plan.run_mega()
        fixture.check(step, rebound, table, lengths)


if __name__ == "__main__":
    unittest.main()
