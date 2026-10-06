#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU/GPU contracts and full-query comparison against pinned HF Attention.

The shared config owns every shape and gate. Decode reserves scheduler budget;
only prefill enters the operator. A strict mismatch is a failure, including ties.
"""

from __future__ import annotations
import gc
import importlib.util
import itertools
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import torch
from mbsa import prepare_cutedsl_pipeline, reference_pipeline
from mbsa.reference import MaxwellBlockSparseAttentionReference

CUTEDSL_ARCHS = ((9, 0), (10, 0), (10, 3))


from tests.common import (
    select_cases,
    make_inputs,
    format_index_inputs,
    oracle_checks,
    compare,
)

from tests.common import BackendCaseMixin


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class BackendAlignment(BackendCaseMixin, unittest.TestCase):
    def test_triton(self):
        if importlib.util.find_spec("triton") is None:
            self.skipTest("Triton is required")
        from mbsa import triton_pipeline

        for qn, kn in ((128, 4096), (17, 4113), (128, 128)):
            with self.subTest(q=qn, kv=kn):
                operands = self.inputs(qn, kn)
                self.compare(triton_pipeline(*operands), reference_pipeline(*operands))

    def test_triton_rejects_unaligned_offset(self):
        if importlib.util.find_spec("triton") is None:
            self.skipTest("Triton is required")
        from mbsa import triton_pipeline

        with self.assertRaisesRegex(ValueError, "block-aligned"):
            triton_pipeline(*self.inputs(16, 4096))

    def test_cutedsl(self):
        if torch.cuda.get_device_capability() not in CUTEDSL_ARCHS:
            self.skipTest("CuTe DSL runtime requires SM90/SM100/SM103")
        from mbsa import prepare_cutedsl_pipeline

        for qn, kn in ((16, 4096), (16, 4113), (128, 128)):
            with self.subTest(q=qn, kv=kn):
                operands = self.inputs(qn, kn)
                self.compare(
                    prepare_cutedsl_pipeline(*operands).run(),
                    reference_pipeline(*operands),
                )

    def test_mxfp6_page_mma_uses_native_packing_and_block_scales(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("MXFP6 page MMA requires SM100/SM103")
        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        from mbsa.cute.src.sm100.mx_router import Mxfp6PageScoreSm103
        from mbsa.index_formats import (
            dequantize_index_k_mxfp6,
            dequantize_index_q_mxfp8,
            quantize_index_k_mxfp6,
            quantize_q_mxfp8,
        )

        compiled = None
        for seed in (0, 7, 29, 71):
            torch.manual_seed(seed)
            q = torch.randn(128, 4, 128, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, 1, 128, 128, device="cuda", dtype=torch.bfloat16)
            q *= torch.exp2(
                torch.tensor([-6.0, -1.0, 2.0, 5.0], device="cuda")
            ).repeat_interleave(32)
            k *= torch.exp2((torch.arange(128, device="cuda") % 7 - 3).float())[
                None, None, :, None
            ]
            codes, scales = quantize_q_mxfp8(q)
            qp = codes[:, :4].view(torch.float8_e4m3fn)
            kp = quantize_index_k_mxfp6(k)
            output = torch.empty(128, device="cuda", dtype=torch.float32)
            args = [from_dlpack(x) for x in (qp, scales, kp.flatten(), output)]
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            if compiled is None:
                compiled = cute.compile(Mxfp6PageScoreSm103(), *args, stream)
            compiled(*args, stream)
            qr = dequantize_index_q_mxfp8(qp, scales)[:, 0]
            kr = dequantize_index_k_mxfp6(kp)[0, 0]
            expected = (qr @ kr.T).amax(-1)
            with self.subTest(seed=seed):
                torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    def test_cutedsl_fp8_index_cache_switch_and_live_graph(self):
        if torch.cuda.get_device_capability() not in CUTEDSL_ARCHS:
            self.skipTest("native FP8 Index requires SM90, SM100 or SM103")
        for qn, kn in ((128, 8192), (17, 4113)):
            operands = self.inputs(qn, kn)
            for partial in (torch.bfloat16, torch.float8_e4m3fn):
                # Identical shapes reuse the cache across BF16 -> FP8 -> BF16.
                for dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.bfloat16):
                    with self.subTest(q=qn, kv=kn, index=dtype, partial=partial):
                        iq, ik = (x.to(dtype).clone() for x in operands[:2])
                        main = operands[2:]
                        expected = reference_pipeline(
                            iq.bfloat16(), ik.bfloat16(), *main
                        )
                        prepared = prepare_cutedsl_pipeline(
                            iq, ik, *main, partial_dtype=partial
                        )
                        self.compare(prepared.run(), expected)
                        self.assertEqual(
                            prepared.plan.attention.dynamic_workspace[0].dtype, partial
                        )
                        prepared.plan.capture_mega_graph()

                        def replay(prepared=prepared):
                            out, lse = prepared.plan.run_mega()
                            return (
                                out,
                                lse,
                                prepared.plan.attention.tile_plan.dynamic_route,
                            )

                        self.compare(replay(), expected)
                        iq.copy_(iq.float().roll(1, dims=1).to(dtype))
                        expected = reference_pipeline(
                            iq.bfloat16(), ik.bfloat16(), *main
                        )
                        self.compare(replay(), expected)
                        del prepared, expected, iq, ik
                        gc.collect()

    def test_cutedsl_mx_pipeline_rebind_and_live_scales(self):
        """Native padded Q, page rebinding and live scales with both partial types."""
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("MXFP6 Index requires SM100/SM103")
        from mbsa.index_formats import (
            dequantize_index_k_mxfp6,
            dequantize_index_q_mxfp8,
            quantize_index_k_mxfp6,
            quantize_q_mxfp8,
        )

        for kn in (8192, 128):
            for partial in (torch.bfloat16, torch.float8_e4m3fn):
                with self.subTest(kv=kn, partial=partial):
                    blocks = kn // 128

                    def physical(operands, permute):
                        iq, ik, q, k, v = operands
                        pages = torch.arange(blocks, device="cuda", dtype=torch.int32)
                        if permute:
                            pages = pages.flip(0)
                        codes, scales = quantize_q_mxfp8(iq)
                        packed_k = quantize_index_k_mxfp6(ik.view(blocks, 1, 128, 128))
                        ip = torch.empty_like(packed_k)
                        ip.index_copy_(0, pages.long(), packed_k)
                        pool = torch.empty(
                            (blocks, 128, 4, 2, 128), device="cuda", dtype=q.dtype
                        )
                        for half, values in enumerate((k, v)):
                            pool[..., half, :].index_copy_(
                                0, pages.long(), values.view(blocks, 128, 4, 128)
                            )
                        # Preserve the native row stride and exercise raw-byte Q.
                        return codes[:, :4], ip, q, pool, scales, pages

                    first = self.inputs(17, kn)
                    iq, ik, q, pool, scales, pages = physical(first, False)
                    p = prepare_cutedsl_pipeline(
                        iq,
                        ik,
                        q,
                        pool[..., 0, :].permute(0, 2, 1, 3),
                        pool[..., 1, :].permute(0, 2, 1, 3),
                        index_q_scales=scales,
                        index_page_indices=pages,
                        page_table=pages.unsqueeze(0),
                        logical_k_tokens=kn,
                        partial_dtype=partial,
                    )

                    def expected(main):
                        dq = dequantize_index_q_mxfp8(iq, scales)
                        dk = dequantize_index_k_mxfp6(ik).index_select(0, pages.long())
                        return reference_pipeline(dq, dk.view(kn, 1, 128), *main)

                    self.compare(p.run(), expected(first[2:]))
                    second = tuple(t.roll(1, 0).add(0.125) for t in first)
                    iq, ik, q, pool, scales, pages = physical(second, True)
                    out = torch.empty_like(q)
                    with self.assertRaisesRegex(ValueError, "requires index_q_scales"):
                        p.rebind(
                            iq,
                            ik,
                            q,
                            pool[..., 0, :].permute(0, 2, 1, 3),
                            pool[..., 1, :].permute(0, 2, 1, 3),
                            index_page_indices=pages,
                            page_table=pages.unsqueeze(0),
                            out=out,
                        )
                    p.rebind(
                        iq,
                        ik,
                        q,
                        pool[..., 0, :].permute(0, 2, 1, 3),
                        pool[..., 1, :].permute(0, 2, 1, 3),
                        index_q_scales=scales,
                        index_page_indices=pages,
                        page_table=pages.unsqueeze(0),
                        out=out,
                    )
                    actual = p.run()
                    self.assertEqual(actual[0].data_ptr(), out.data_ptr())
                    self.compare(actual, expected(second[2:]))
                    p.plan.capture_mega_graph()
                    scales.bitwise_xor_(1 << 8)
                    ik.view(blocks, 12800)[:, 12288:].bitwise_xor_(1)
                    go, gl = p.plan.run_mega()
                    self.compare(
                        (go, gl, p.plan.attention.tile_plan.dynamic_route),
                        expected(second[2:]),
                    )
                    self.assertEqual(
                        p.plan.attention.dynamic_workspace[0].dtype, partial
                    )
                    del p
                    gc.collect()

    def test_fp8_index_stages_survive_repeated_two_partition_prefill(self):
        """Stress the SMEM overwrite race observed at the second K partition."""
        if torch.cuda.get_device_capability() not in CUTEDSL_ARCHS:
            self.skipTest("FP8 Index requires SM100/SM103")
        case = select_cases(ids=["p17-d15-kv139264-balanced_full"])[0]
        ref = MaxwellBlockSparseAttentionReference(index_head_dim=128, head_dim=128)
        for mapping in ("identity", "permuted"):
            original = make_inputs(case, mapping)
            data, oracle = format_index_inputs(original, "fp8")
            probes = oracle_checks(case, oracle, ref)
            p = prepare_cutedsl_pipeline(
                data["iq"],
                data["ik"],
                data["q"],
                data["pool"][..., 0, :].permute(0, 2, 1, 3),
                data["pool"][..., 1, :].permute(0, 2, 1, 3),
                index_page_indices=data["pages"],
                page_table=data["table"],
                cu_seqlens_q=data["cu_q"],
                cu_seqlens_k=data["cu_k"],
                overlap_fixed_score=False,
            )
            for iteration in range(20):
                with self.subTest(
                    mapping=mapping, execution="eager", iteration=iteration
                ):
                    self.assertEqual(compare(p.run(), probes)["status"], "passed")
            p.plan.capture_mega_graph()
            for iteration in range(20):
                with self.subTest(
                    mapping=mapping, execution="graph", iteration=iteration
                ):
                    out, lse = p.plan.run_mega()
                    actual = out, lse, p.plan.attention.tile_plan.dynamic_route
                    self.assertEqual(compare(actual, probes)["status"], "passed")
            del p, data, original, oracle, probes
            gc.collect()
            torch.cuda.empty_cache()

    def test_index_cutoff_ties_match_reference_route_sets(self):
        if torch.cuda.get_device_capability() not in CUTEDSL_ARCHS:
            self.skipTest("FP8 and MXFP6 Index require SM103")
        ref = MaxwellBlockSparseAttentionReference(index_head_dim=128, head_dim=128)
        for fmt, (qn, kn) in itertools.product(
            ("bf16", "fp8")
            if torch.cuda.get_device_capability() == (9, 0)
            else ("bf16", "fp8", "mxfp8_mxfp6"),
            ((17, 8192), (1, 139264)),
        ):
            with self.subTest(index_format=fmt, q=qn, kv=kn):
                case = {
                    "config_id": f"tie-q{qn}-kv{kn}",
                    "requests": [{"phase": "prefill", "q": qn, "kv": kn}],
                    "seed": 17,
                }
                original = make_inputs(case, "identity")
                original["iq"].zero_()
                data, oracle = format_index_inputs(original, fmt)
                probes = oracle_checks(case, oracle, ref)
                p = prepare_cutedsl_pipeline(
                    data["iq"],
                    data["ik"],
                    data["q"],
                    data["pool"][..., 0, :].permute(0, 2, 1, 3),
                    data["pool"][..., 1, :].permute(0, 2, 1, 3),
                    index_q_scales=data.get("index_q_scales"),
                    index_page_indices=data["pages"],
                    page_table=data["table"],
                    cu_seqlens_q=data["cu_q"],
                    cu_seqlens_k=data["cu_k"],
                )
                self.assertEqual(compare(p.run(), probes)["status"], "passed")
                del p, data, original, oracle, probes
                gc.collect()
                torch.cuda.empty_cache()

    def test_native_top16_merge_ties_invalid_slots_and_live_graph(self):
        """Merge the direct UMMA score lists, including many partitions and tails."""
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("native score merge requires SM100/SM103")
        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        from mbsa.cute.src.sm100.router import merge_native_top16

        class Merge:
            @cute.jit
            def __call__(self, scores, ids, route, stream):
                merge_native_top16(scores, ids, route, True).launch(
                    grid=(scores.shape[0], cute.ceil_div(scores.shape[2], 4), 1),
                    block=(128, 1, 1),
                    stream=stream,
                )

        for parts in (1, 2, 8, 32):
            with self.subTest(partitions=parts):
                ids = torch.arange(parts * 16, device="cuda", dtype=torch.int32)
                ids = ids.reshape(1, parts, 1, 16).expand(4, parts, 35, 16).clone()
                # Repeated scores exercise deterministic ID tie-breaking. Each
                # partition remains sorted while its invalid tail grows live.
                scores = (15 - ids % 16).float()
                route = torch.full((4, 32, 35), -99, device="cuda", dtype=torch.int32)
                args = [from_dlpack(x, assumed_align=16) for x in (scores, ids, route)]
                stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
                run = cute.compile(Merge(), *args, stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
                for invalid_tail in (0, 7, 16):
                    if invalid_tail:
                        ids[..., -invalid_tail:] = -1
                        scores[..., -invalid_tail:] = -torch.inf
                    graph.replay()
                    score_rows = scores.permute(0, 2, 1, 3).flatten(2)
                    id_rows = ids.permute(0, 2, 1, 3).flatten(2)
                    order = torch.argsort(
                        score_rows, dim=-1, descending=True, stable=True
                    )
                    expected = id_rows.gather(-1, order[..., :16])
                    torch.testing.assert_close(
                        route[:, 16:].permute(0, 2, 1), expected, rtol=0, atol=0
                    )
                    self.assertTrue(bool((route[:, :16] == -99).all()))

    def test_cutedsl_fp8_partial_storage_matches_reference(self):
        if torch.cuda.get_device_capability() not in CUTEDSL_ARCHS:
            self.skipTest("CuTe DSL runtime requires SM90/SM100/SM103")
        from mbsa import prepare_cutedsl_pipeline

        for qn, kn in ((128, 8192), (17, 4113)):
            with self.subTest(q=qn, kv=kn):
                operands = self.inputs(qn, kn)
                pipeline = prepare_cutedsl_pipeline(
                    *operands, partial_dtype=torch.float8_e4m3fn
                )
                self.assertEqual(
                    pipeline.plan.attention.dynamic_workspace[0].dtype,
                    torch.float8_e4m3fn,
                )
                self.compare(pipeline.run(), reference_pipeline(*operands))

    def test_cutedsl_packed_permuted_index_pages(self):
        if torch.cuda.get_device_capability() not in CUTEDSL_ARCHS:
            self.skipTest("CuTe DSL runtime requires SM90/SM100/SM103")
        from mbsa import prepare_cutedsl_pipeline

        lengths = ((17, 4113), (23, 2195))
        operands = [self.inputs(qn, kn) for qn, kn in lengths]
        iq, q, k, v = (
            torch.cat([request[i] for request in operands]) for i in (0, 2, 3, 4)
        )
        pages = []
        for request in operands:
            ik = request[1]
            padded = torch.zeros(
                ((ik.shape[0] + 127) // 128 * 128, 1, 128),
                device="cuda",
                dtype=torch.bfloat16,
            )
            padded[: ik.shape[0]].copy_(ik)
            pages.append(padded.view(-1, 1, 128, 128))
        logical_ik = torch.cat(pages)
        permutation = torch.arange(logical_ik.shape[0] - 1, -1, -1, device="cuda")
        physical_ik = torch.empty_like(logical_ik)
        physical_ik.index_copy_(0, permutation, logical_ik)
        cu_q = torch.tensor([0, 17, 40], dtype=torch.int32, device="cuda")
        cu_k = torch.tensor([0, 4113, 6308], dtype=torch.int32, device="cuda")
        actual = prepare_cutedsl_pipeline(
            iq,
            physical_ik,
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            index_page_indices=permutation.to(torch.int32),
        ).run()
        start = 0
        for request, (qn, _) in zip(operands, lengths):
            qs = slice(start, start + qn)
            self.compare(
                (actual[0][qs], actual[1][qs], actual[2][:, qs]),
                reference_pipeline(*request),
            )
            start += qn

    def test_cutedsl_permuted_kv_pages_with_tail(self):
        if torch.cuda.get_device_capability() not in CUTEDSL_ARCHS:
            self.skipTest("CuTe DSL runtime requires SM90/SM100/SM103")
        from mbsa import prepare_cutedsl_pipeline

        operands = self.inputs(17, 4113)
        iq, ik, q, k, v = operands
        blocks = (k.shape[0] + 127) // 128
        permutation = torch.arange(blocks - 1, -1, -1, device="cuda")

        def paged(tensor):
            padded = torch.zeros(
                (blocks * 128, 4, 128), dtype=tensor.dtype, device=tensor.device
            )
            padded[: tensor.shape[0]].copy_(tensor)
            logical = padded.view(blocks, 128, 4, 128).permute(0, 2, 1, 3).contiguous()
            physical = torch.empty_like(logical)
            physical.index_copy_(0, permutation, logical)
            return physical

        actual = prepare_cutedsl_pipeline(
            iq,
            ik,
            q,
            paged(k),
            paged(v),
            page_table=permutation.to(torch.int32).unsqueeze(0),
            logical_k_tokens=k.shape[0],
        ).run()
        self.compare(actual, reference_pipeline(*operands))

    def test_cutedsl_vllm_interleaved_cache_and_single_query_chunk(self):
        """Use the actual split/permuted cache strides, including a KV tail."""
        if torch.cuda.get_device_capability() not in CUTEDSL_ARCHS:
            self.skipTest("CuTe DSL runtime requires SM90/SM100/SM103")
        from mbsa import prepare_cutedsl_pipeline

        for qn, kn in (
            (1, 4113),
            (17, 4113),
            (1, 16384),
            (17, 16384),
            (16, 16384),
            (16, 16385),
        ):
            operands = self.inputs(qn, kn)
            iq, ik, q, k, v = operands
            blocks = (k.shape[0] + 127) // 128
            pool = torch.zeros((blocks, 128, 4, 2, 128), device="cuda", dtype=q.dtype)
            permutation = torch.arange(blocks - 1, -1, -1, device="cuda")
            for half, tensor in enumerate((k, v)):
                padded = torch.zeros(
                    (blocks * 128, 4, 128), device="cuda", dtype=q.dtype
                )
                padded[: tensor.shape[0]].copy_(tensor)
                pool[..., half, :].index_copy_(
                    0, permutation, padded.view(blocks, 128, 4, 128)
                )
            kp, vp = (pool[..., half, :].permute(0, 2, 1, 3) for half in (0, 1))
            self.assertEqual(kp.permute(0, 2, 1, 3).stride(), (131072, 1024, 256, 1))
            self.assertEqual(vp.data_ptr() - kp.data_ptr(), 128 * q.element_size())
            actual = prepare_cutedsl_pipeline(
                iq,
                ik,
                q,
                kp,
                vp,
                page_table=permutation.to(torch.int32).unsqueeze(0),
                logical_k_tokens=k.shape[0],
            ).run()
            self.compare(actual, reference_pipeline(*operands))

    def test_cutedsl_rebind_refreshes_index_pages_and_fused_output(self):
        """A reused plan must read the new layer and identity-to-permuted map."""
        if torch.cuda.get_device_capability() not in CUTEDSL_ARCHS:
            self.skipTest("CuTe DSL runtime requires SM90/SM100/SM103")
        from mbsa import prepare_cutedsl_pipeline

        first = self.inputs(17, 8192)
        second = tuple(tensor.roll(1, 0).add(0.125) for tensor in first)
        blocks = 64

        def physical(operands, permute):
            iq, ik, q, k, v = operands
            pages = torch.arange(blocks, dtype=torch.int32, device="cuda")
            if permute:
                pages = pages.flip(0)
            pool = torch.empty((blocks, 128, 4, 2, 128), device="cuda", dtype=q.dtype)
            for half, tensor in enumerate((k, v)):
                pool[..., half, :].index_copy_(
                    0, pages.long(), tensor.view(blocks, 128, 4, 128)
                )
            ip = torch.empty((blocks, 1, 128, 128), device="cuda", dtype=q.dtype)
            ip.index_copy_(0, pages.long(), ik.view(blocks, 1, 128, 128))
            return (
                iq,
                ip,
                q,
                pool[..., 0, :].permute(0, 2, 1, 3),
                pool[..., 1, :].permute(0, 2, 1, 3),
                pages,
            )

        iq, ik, q, k, v, pages = physical(first, False)
        pipeline = prepare_cutedsl_pipeline(
            iq,
            ik,
            q,
            k,
            v,
            index_page_indices=pages,
            page_table=pages.unsqueeze(0),
            logical_k_tokens=8192,
        )
        self.compare(pipeline.run(), reference_pipeline(*first))
        iq, ik, q, k, v, pages = physical(second, True)
        out = torch.empty_like(q)
        pipeline.rebind(
            iq,
            ik,
            q,
            k,
            v,
            index_page_indices=pages,
            page_table=pages.unsqueeze(0),
            out=out,
        )
        actual = pipeline.run()
        self.assertEqual(actual[0].data_ptr(), out.data_ptr())
        self.compare(actual, reference_pipeline(*second))
