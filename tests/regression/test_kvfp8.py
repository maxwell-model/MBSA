# SPDX-License-Identifier: Apache-2.0
"""Native FP8 Q/K/V attention against an independent paged token-mask oracle."""
import math
import unittest

import torch


FP8 = torch.float8_e4m3fn


def oracle(q, k, v, table, lengths, route):
    """FP32 arithmetic on represented FP8 values and the actual selected pages."""
    q = q.to(FP8).float()
    k, v = k.float(), v.float()
    routes = route.transpose(0, 1).cpu() if route.shape[0] == 4 else route.cpu()
    output = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], device=q.device)
    offset = 0
    for row, (nq, nk) in enumerate(lengths):
        pages = table[row, :math.ceil(nk / 128)].long()
        key = k.index_select(0, pages).permute(0, 2, 1, 3).reshape(-1, 4, 128)[:nk]
        value = v.index_select(0, pages).permute(0, 2, 1, 3).reshape(-1, 4, 128)[:nk]
        for qi in range(nq):
            position = nk - nq + qi
            block = position // 128
            for hi in range(4):
                selected = set(range(8)) | set(range(max(0, block - 7), block + 1))
                selected |= {int(p) for p in routes[offset + qi, hi] if p >= 0}
                ids = [t for p in sorted(selected)
                       for t in range(p * 128, min((p + 1) * 128, position + 1, nk))]
                heads = slice(hi * 8, (hi + 1) * 8)
                scores = q[offset + qi, heads] @ key[ids, hi].T / math.sqrt(128)
                output[offset + qi, heads] = scores.softmax(-1) @ value[ids, hi]
                lse[offset + qi, heads] = scores.logsumexp(-1)
        offset += nq
    return output, lse


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class KVFP8Regression(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("Blackwell is required")
        self.device = torch.device("cuda", torch.cuda.current_device())

    def case(self, lengths, q_dtype=torch.bfloat16, index_format="fp8"):
        gen = torch.Generator(device=self.device).manual_seed(791)
        def rand(shape):
            return (torch.randn(shape, generator=gen, device=self.device) * .4).bfloat16()
        counts = [math.ceil(nk / 128) for _, nk in lengths]
        width = max(counts)
        total_pages = sum(counts)
        storage = rand((total_pages, 4, 128, 256)).to(FP8)
        k, v = storage.split(128, dim=-1)
        mapping = torch.randperm(total_pages, generator=gen, device=self.device).int()
        table = torch.zeros(len(lengths), width, dtype=torch.int32, device=self.device)
        start = 0
        for row, count in enumerate(counts):
            table[row, :count] = mapping[start:start + count]
            start += count
        tokens = sum(nq for nq, _ in lengths)
        q = rand((tokens, 32, 128)).to(q_dtype)
        iq, ik = rand((tokens, 4, 128)), rand((total_pages, 128, 128))
        if index_format == "mxfp6":
            from mbsa.mxfp6.write import write_index_k, quantize_q_mxfp8_fused
            packed = torch.zeros(total_pages, 128, 100, dtype=torch.uint8, device=self.device)
            slots = torch.arange(total_pages * 128, dtype=torch.int32, device=self.device)
            write_index_k(ik.flatten(0, 1), slots, packed.flatten())
            iq, scales = quantize_q_mxfp8_fused(iq, pad_to=8)
            iq, ik = iq[:, :4], packed
        else:
            dtype = FP8 if index_format == "fp8" else torch.bfloat16
            iq, ik, scales = iq.to(dtype), ik.to(dtype), None
        return dict(q=q, key_cache=k, value_cache=v, index_q=iq, index_k_cache=ik,
                    index_q_scales=scales, out=torch.empty_like(q, dtype=torch.bfloat16)), table

    def run_case(self, lengths, *, q_dtype=torch.bfloat16, index_format="fp8",
                 partial_dtype=torch.bfloat16):
        from mbsa.serving import Prefill
        operands, table = self.case(lengths, q_dtype, index_format)
        step = Prefill(self.device, partial_dtype=partial_dtype).stage(lengths, table, table)
        result = step.run(**operands)
        self.assertEqual(result.dtype, torch.bfloat16)
        a = step.plan.attention
        actual_q = getattr(a, "compute_q", None)
        if actual_q is None:
            actual_q = a.attention_q
        self.assertEqual(actual_q.dtype, FP8)
        self.assertEqual(a.k.dtype, FP8)
        self.assertEqual(a.v.dtype, FP8)
        self.check(step, operands, table, lengths)
        return step, operands, table

    def check(self, step, operands, table, lengths):
        a = step.plan.attention
        expected, expected_lse = oracle(operands["q"], operands["key_cache"], operands["value_cache"],
                                        table, lengths, a.tile_plan.dynamic_route)
        torch.testing.assert_close(operands["out"].float(), expected, atol=.025, rtol=.08)
        torch.testing.assert_close(a.out_lse, expected_lse, atol=.015, rtol=.003)

    def test_fixed_online_softmax_rescales_before_fp8_probability_saturation(self):
        """A later fixed page raises scores by<4log2 but would overflow FP8 P."""
        from flash_attn.cute.softmax import SoftmaxSm100
        from mbsa.cute.fixed_attention import PreparedFixedAttention

        create = SoftmaxSm100.create
        q = torch.zeros(16, 32, 128, device=self.device, dtype=torch.bfloat16)
        q[..., 0] = 8
        # FA4 consumes masked page63 first (score0), then full pages62..56.
        # Those pages raise score to16/sqrt(128), so holding the first maximum
        # produces P=256*exp(16/sqrt(128))>448, below the old rescale threshold.
        key = torch.zeros(64, 128, 4, 128, device=self.device, dtype=torch.bfloat16)
        value = torch.zeros_like(key)
        key[56:63, ..., 0] = 2
        value[56:63, ..., 0] = 1
        k, v = (t.to(FP8).permute(0, 2, 1, 3) for t in (key, value))
        table = torch.arange(64, device=self.device, dtype=torch.int32)[None]
        used = torch.tensor([8192], device=self.device, dtype=torch.int32)
        fixed = PreparedFixedAttention(
            q.to(FP8), k, v, table, used, ((16, 8192),),
            pack_gqa=True, immutable_table=True, identity_table=True,
            prepared_launch=True,
        )
        actual, lse = fixed.run()
        routes = torch.full((4, 16, 16), -1, dtype=torch.int32, device=self.device)
        expected, expected_lse = oracle(q, k, v, table, ((16, 8192),), routes)
        torch.testing.assert_close(actual.float(), expected, atol=.03, rtol=.03)
        torch.testing.assert_close(lse.T, expected_lse, atol=.001, rtol=.001)
        self.assertIs(SoftmaxSm100.create, create)

    def test_fixed_only_accepts_native_fp8_q_and_default_bf16_output(self):
        self.run_case(((17, 256),), q_dtype=FP8)

    def test_dynamic_tail_with_bf16_q_and_index(self):
        self.run_case(((17, 2176),), index_format="bf16")

    def test_long_context_clc_refreshes_bf16_q(self):
        lengths = ((17, 32768),)
        step, operands, table = self.run_case(lengths, index_format="fp8")
        self.assertTrue(step.plan.overlap_fixed_score)
        attention = step.plan.attention
        # Garena exposes the selected MSA configuration; public KV-Outer has
        # another scheduler and must not be assigned the same CLC contract.
        if hasattr(attention, "dynamic_kernel_cfg"):
            cfg = attention.dynamic_kernel_cfg
            self.assertIsNotNone(cfg)
            if torch.cuda.get_device_capability(self.device) == (10, 3):
                self.assertTrue(cfg.use_clc)
                self.assertEqual(cfg.num_wgs, 5)
        operands["q"].mul_(1.5)
        step.run(**operands)
        self.check(step, operands, table, lengths)

    def test_packed_rebind_live_q_and_graph_replay(self):
        lengths = ((17, 4096), (5, 2177))
        step, operands, table = self.run_case(lengths)
        pointer = step.plan.attention.q_compute.data_ptr() if hasattr(step.plan.attention, "q_compute") else step.plan.attention.compute_q.data_ptr()
        # Refresh must happen at execution, including in-place changes.
        operands["q"].mul_(1.5)
        step.run(**operands)
        self.check(step, operands, table, lengths)
        # A staged plan must rebind another layer with live strided FP8 K/V.
        rebound, _ = self.case(lengths)
        step.run(**rebound)
        self.check(step, rebound, table, lengths)
        actual_q = getattr(step.plan.attention, "compute_q", None)
        if actual_q is None:
            actual_q = step.plan.attention.attention_q
        self.assertEqual(pointer, actual_q.data_ptr())
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(self.device)
        with torch.cuda.graph(graph):
            step.plan.run()
        rebound["q"].mul_(.5)
        graph.replay()
        torch.cuda.synchronize(self.device)
        self.check(step, rebound, table, lengths)

    def test_mxfp6_index_with_native_fp8_main_qkv(self):
        self.run_case(((17, 4096),), q_dtype=FP8, index_format="mxfp6")

    def test_fp8_partial_output_with_native_fp8_main_qkv(self):
        self.run_case(((17, 4096),), q_dtype=FP8, partial_dtype=FP8)


if __name__ == "__main__":
    unittest.main()
