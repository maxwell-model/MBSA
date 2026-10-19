#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Core index precision, paged input, native Top-16 and graph regressions."""

from __future__ import annotations
import gc
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import torch
from mbsa import prepare_cutedsl_pipeline
from tests.reference import reference_pipeline

CUTEDSL_ARCHS = ((10, 0), (10, 3))


from tests.common import BackendCaseMixin, make_inputs


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class BackendAlignment(BackendCaseMixin, unittest.TestCase):
    def test_cutedsl_fp8_index_cache_switch_and_live_graph(self):
        if torch.cuda.get_device_capability() not in CUTEDSL_ARCHS:
            self.skipTest("native FP8 Index requires SM90, SM100 or SM103")
        for qn, kn in ((17, 4113), (1024, 4096)):
            operands = self.inputs(qn, kn)
            for partial in (torch.bfloat16, torch.float8_e4m3fn):
                # Identical shapes reuse the cache across BF16 -> FP8 -> BF16.
                for dtype in (
                    torch.bfloat16,
                    torch.float16,
                    torch.float8_e4m3fn,
                    torch.bfloat16,
                ):
                    with self.subTest(q=qn, kv=kn, index=dtype, partial=partial):
                        iq, ik = (x.to(dtype).clone() for x in operands[:2])
                        main = operands[2:]
                        expected = reference_pipeline(iq.float(), ik.float(), *main)
                        prepared = prepare_cutedsl_pipeline(
                            iq, ik, *main, partial_dtype=partial
                        )
                        self.compare(prepared.run(), expected)
                        attention = prepared.plan.attention
                        if attention.dynamic_workspace:
                            self.assertEqual(
                                attention.dynamic_workspace[0].dtype, partial
                            )
                        if attention.fixed_attention is not None:
                            self.assertEqual(
                                attention.fixed_attention.static_identity_mask,
                                qn % 128 == 0,
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
                        expected = reference_pipeline(iq.float(), ik.float(), *main)
                        self.compare(replay(), expected)
                        del prepared, expected, iq, ik
                        gc.collect()

    def test_sm90_paged_index_formats_fp8_partials_rebind_and_graph(self):
        """Hopper retains native routing and scaled partials for live paged caches."""
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("SM90 is required")
        from mbsa.serving import Prefill

        qn, kn = 17, 4113
        dense_iq, dense_ik, q, k, v = self.inputs(qn, kn)
        pages = (kn + 127) // 128
        table = torch.arange(pages - 1, -1, -1, device="cuda", dtype=torch.int32)[None]
        cache = torch.zeros(pages, 128, 4, 256, device="cuda", dtype=q.dtype)
        for half, values in enumerate((k, v)):
            padded = torch.zeros(pages * 128, 4, 128, device="cuda", dtype=q.dtype)
            padded[:kn].copy_(values)
            cache[..., half * 128 : (half + 1) * 128].index_copy_(
                0, table[0].long(), padded.view(pages, 128, 4, 128)
            )
        # Poison the masked tail without changing valid tokens.
        cache[table[0, -1], kn % 128 :].fill_(4096)
        pk, pv = cache.permute(0, 2, 1, 3).split(128, dim=-1)
        for dtype in (torch.bfloat16, torch.float16, torch.float8_e4m3fn):
            iq = dense_iq.to(dtype)
            ik = torch.zeros(pages * 128, 1, 128, device="cuda", dtype=dtype)
            ik[:kn].copy_(dense_ik.to(dtype))
            for partial in (torch.bfloat16, torch.float8_e4m3fn):
                with self.subTest(index=dtype, partial=partial):
                    step = Prefill(q.device, partial_dtype=partial).stage(
                        ((qn, kn),),
                        table,
                        torch.arange(pages, device="cuda", dtype=torch.int32)[None],
                    )
                    out = torch.empty_like(q)
                    step.run(
                        q,
                        pk,
                        pv,
                        index_q=iq,
                        index_k_cache=ik.view(pages, 128, 128),
                        out=out,
                    )
                    self.compare(
                        (out, step.plan.attention.out_lse, step.plan.route),
                        reference_pipeline(iq.float(), ik[:kn].float(), q, k, v),
                    )
                    live_q = q.clone().mul_(0.5)
                    step.run(
                        live_q,
                        pk,
                        pv,
                        index_q=iq,
                        index_k_cache=ik.view(pages, 128, 128),
                        out=out,
                    )
                    step.plan.capture_mega_graph()
                    live_q.mul_(0.5)
                    actual = step.plan.run_mega()
                    self.compare(
                        (*actual, step.plan.route),
                        reference_pipeline(iq.float(), ik[:kn].float(), live_q, k, v),
                    )

    def test_sm90_fp8_partials_large_values_empty_routes_and_graph(self):
        """Scaled Hopper partials preserve mass for large V and live empty routes."""
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("SM90 is required")
        import math

        iq, ik, q, k, v = self.inputs(17, 4096)
        q.zero_()
        k.zero_()
        values = torch.linspace(-4096, 4096, 128, device="cuda").bfloat16()
        v.copy_(values)
        p = prepare_cutedsl_pipeline(
            iq,
            ik,
            q,
            k,
            v,
            partial_dtype=torch.float8_e4m3fn,
            overlap_fixed_score=False,
        ).plan
        p.route.fill_(-1)
        p.route[:, :, :3] = torch.tensor([8, 9, 10], device="cuda")
        p.workspace.fixed_out.zero_()
        p.workspace.fixed_lse.fill_(-torch.inf)

        def run():
            p.run_plan()
            p.attention.run_dynamic()
            p.attention.run_merge()

        def check():
            expected = v[8 * 128 : 11 * 128].float().mean(0).repeat_interleave(8, 0)
            err = (p.attention.out.float() - expected[None]).abs()
            self.assertLessEqual(err.max().item(), expected.abs().max().item() * 0.04)
            torch.testing.assert_close(
                p.attention.out_lse,
                torch.full_like(p.attention.out_lse, math.log(384)),
                atol=0.002,
                rtol=0.002,
            )
            self.assertTrue(bool(torch.isfinite(p.attention.out).all()))

        run()
        check()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
        v.mul_(4)
        graph.replay()
        check()
        p.route.fill_(-1)
        graph.replay()
        self.assertTrue(bool(torch.all(p.attention.out == 0)))
        self.assertTrue(bool(torch.all(p.attention.out_lse == -torch.inf)))

    def test_sm90_packed_shared_prefix_pages_and_kv_tails(self):
        """Packed requests retain independent Index maps and shared attention pages."""
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("SM90 is required")
        lengths = ((1, 4113), (17, 2195))
        operands = [self.inputs(qn, kn) for qn, kn in lengths]
        counts = [(kn + 127) // 128 for _, kn in lengths]
        iq = torch.cat([x[0] for x in operands]).half()
        q = torch.cat([x[2] for x in operands])
        logical_index = []
        for x, nb in zip(operands, counts):
            padded = torch.zeros(nb * 128, 1, 128, device="cuda", dtype=torch.float16)
            padded[: x[1].shape[0]].copy_(x[1])
            logical_index.append(padded.view(nb, 128, 128))
        logical_index = torch.cat(logical_index)
        index_pages = torch.arange(
            sum(counts) - 1, -1, -1, device="cuda", dtype=torch.int32
        )
        index_cache = torch.empty_like(logical_index)
        index_cache.index_copy_(0, index_pages.long(), logical_index)
        index_cache = index_cache.unsqueeze(1)
        cu_q = torch.tensor([0, 1, 18], device="cuda", dtype=torch.int32)
        cu_k = torch.tensor([0, 4113, 6308], device="cuda", dtype=torch.int32)
        for paged in (False, True):
            with self.subTest(paged=paged):
                table = None
                main = [x[2:] for x in operands]
                if paged:
                    pool = torch.randn(
                        sum(counts), 128, 4, 256, device="cuda"
                    ).bfloat16()
                    table = torch.zeros(
                        2, max(counts), device="cuda", dtype=torch.int32
                    )
                    table[0, : counts[0]] = torch.arange(counts[0], device="cuda")
                    table[1, :8] = table[0, :8]
                    table[1, 8 : counts[1]] = torch.arange(
                        counts[0], sum(counts) - 8, device="cuda"
                    )
                    main = []
                    for row, ((_, kn), x, nb) in enumerate(
                        zip(lengths, operands, counts)
                    ):
                        page_values = pool.index_select(
                            0, table[row, :nb].long()
                        ).reshape(-1, 4, 256)[:kn]
                        main.append((x[2], *page_values.split(128, dim=-1)))
                        pool[table[row, nb - 1], kn % 128 :].fill_(4096)
                    k, v = pool.permute(0, 2, 1, 3).split(128, dim=-1)
                else:
                    k, v = (torch.cat([x[i] for x in operands]) for i in (3, 4))
                p = prepare_cutedsl_pipeline(
                    iq,
                    index_cache,
                    q,
                    k,
                    v,
                    cu_seqlens_q=cu_q,
                    cu_seqlens_k=cu_k,
                    index_page_indices=index_pages,
                    page_table=table,
                    partial_dtype=torch.float8_e4m3fn,
                )
                actual = p.run()
                begin = 0
                for x, (qn, _), mkv in zip(operands, lengths, main):
                    rows = slice(begin, begin + qn)
                    self.compare(
                        (actual[0][rows], actual[1][rows], actual[2][:, rows]),
                        reference_pipeline(
                            x[0].half().float(), x[1].half().float(), *mkv
                        ),
                    )
                    begin += qn

    def test_cutedsl_mx_pipeline_rebind_and_live_scales(self):
        """MX Index with paged/contiguous attention, rebinding and live graph metadata."""
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("MXFP6 Index requires SM100/SM103")
        from mbsa.index_formats import (
            dequantize_index_k_mxfp6,
            dequantize_index_q_mxfp8,
            quantize_index_k_mxfp6,
            quantize_q_mxfp8,
        )

        kn = 8192
        for qn, paged_attention, native_q in (
            (17, True, False),
            (1024, True, False),
            (1024, True, True),
            (1024, False, False),
            (1024, False, True),
        ):
            for partial in (torch.bfloat16, torch.float8_e4m3fn):
                with self.subTest(
                    q=qn,
                    kv=kn,
                    paged=paged_attention,
                    native_q=native_q,
                    partial=partial,
                ):
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
                        # Preserve the native pad8 token stride for raw/native Q.
                        index_q = codes[:, :4]
                        if native_q:
                            index_q = index_q.view(torch.float8_e4m3fn)
                        return index_q, ip, q, pool, scales, pages

                    first = self.inputs(qn, kn)
                    iq, ik, q, pool, scales, pages = physical(first, False)
                    attention_pages = pages.clone()
                    kp, vp = (
                        (pool[..., half, :].permute(0, 2, 1, 3) for half in (0, 1))
                        if paged_attention
                        else first[3:]
                    )
                    p = prepare_cutedsl_pipeline(
                        iq,
                        ik,
                        q,
                        kp,
                        vp,
                        index_q_scales=scales,
                        index_page_indices=pages,
                        page_table=attention_pages.unsqueeze(0)
                        if paged_attention
                        else None,
                        logical_k_tokens=kn,
                        partial_dtype=partial,
                    )

                    def expected(main):
                        dq = dequantize_index_q_mxfp8(iq, scales)
                        dk = dequantize_index_k_mxfp6(ik).index_select(0, pages.long())
                        if paged_attention:
                            logical_pool = pool.index_select(0, attention_pages.long())
                            main = (
                                q,
                                logical_pool[..., 0, :].reshape(kn, 4, 128),
                                logical_pool[..., 1, :].reshape(kn, 4, 128),
                            )
                        return reference_pipeline(dq, dk.view(kn, 1, 128), *main)

                    attention = p.plan.attention
                    self.assertEqual(attention.fixed_backend, "fa4")
                    self.assertEqual(
                        attention.fixed_attention.static_identity_mask,
                        not paged_attention and qn % 128 == 0,
                    )
                    self.compare(p.run(), expected(first[2:]))
                    second = tuple(t.roll(1, 0).add(0.125) for t in first)
                    iq, ik, q, pool, scales, pages = physical(second, paged_attention)
                    attention_pages = pages.clone()
                    kp, vp = (
                        (pool[..., half, :].permute(0, 2, 1, 3) for half in (0, 1))
                        if paged_attention
                        else second[3:]
                    )
                    out = torch.empty_like(q)
                    if paged_attention:
                        with self.assertRaisesRegex(
                            ValueError, "requires index_q_scales"
                        ):
                            p.rebind(
                                iq,
                                ik,
                                q,
                                kp,
                                vp,
                                index_page_indices=pages,
                                page_table=attention_pages.unsqueeze(0),
                                out=out,
                            )
                        p.rebind(
                            iq,
                            ik,
                            q,
                            kp,
                            vp,
                            index_q_scales=scales,
                            index_page_indices=pages,
                            page_table=attention_pages.unsqueeze(0),
                            out=out,
                        )
                    else:
                        # Layer rebinding retains the prepared identity Index
                        # map and swaps live values/scales and contiguous K/V.
                        p.plan.bind_layer(iq, ik, q, kp, vp, out, index_q_scales=scales)
                    actual = p.run()
                    self.assertEqual(actual[0].data_ptr(), out.data_ptr())
                    self.compare(actual, expected(second[2:]))
                    p.plan.capture_mega_graph()
                    before = expected(second[2:])
                    raw_q = iq.view(torch.uint8)
                    raw_q.copy_(raw_q.roll(1, dims=1))
                    live = expected(second[2:])
                    self.assertFalse(
                        torch.equal(before[2], live[2]), "fixture must change routes"
                    )
                    go, gl = p.plan.run_mega()
                    self.compare(
                        (go, gl, p.plan.attention.tile_plan.dynamic_route), live
                    )
                    scales.bitwise_xor_(1 << 8)
                    ik.view(blocks, 12800)[:, 12288:].bitwise_xor_(1)
                    if paged_attention:
                        # Attention and Index page maps are independent. The
                        # attention table is live in the captured kernel;
                        # Index-map address/value changes use explicit rebind.
                        attention_pages.copy_(attention_pages.flip(0))
                        pool[..., 1, :].add_(0.125)
                    else:
                        vp.add_(0.125)
                    q.add_(0.0625)
                    go, gl = p.plan.run_mega()
                    self.compare(
                        (go, gl, p.plan.attention.tile_plan.dynamic_route),
                        expected(second[2:]),
                    )
                    with self.assertRaisesRegex(RuntimeError, "mega graph"):
                        if paged_attention:
                            p.rebind(
                                iq,
                                ik,
                                q,
                                kp,
                                vp,
                                index_q_scales=scales,
                                index_page_indices=pages,
                                page_table=attention_pages.unsqueeze(0),
                                out=out,
                            )
                        else:
                            p.plan.bind_layer(
                                iq, ik, q, kp, vp, out, index_q_scales=scales
                            )
                    del p, before, live
                    gc.collect()
                    torch.cuda.empty_cache()

    def test_fp8_index_stages_survive_repeated_two_partition_prefill(self):
        """Stress the SMEM overwrite race observed at the second K partition."""
        if torch.cuda.get_device_capability() not in CUTEDSL_ARCHS:
            self.skipTest("FP8 Index requires SM100/SM103")
        case = dict(requests=[dict(phase="prefill", q=17, kv=139264)], seed=17)
        for mapping in ("identity", "permuted"):
            original = make_inputs(case, mapping)
            data = {
                **original,
                "iq": original["iq"].to(torch.float8_e4m3fn),
                "ik": original["ik"].to(torch.float8_e4m3fn),
            }
            pages = data["table"][0, : data["counts"][0]].long()
            pool = data["pool"].index_select(0, pages)
            expected = reference_pipeline(
                data["iq"].bfloat16(),
                data["ik"].index_select(0, pages).reshape(-1, 1, 128).bfloat16(),
                data["q"],
                pool[..., 0, :].reshape(-1, 4, 128),
                pool[..., 1, :].reshape(-1, 4, 128),
            )
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
            for iteration in range(3):
                with self.subTest(
                    mapping=mapping, execution="eager", iteration=iteration
                ):
                    self.compare(p.run(), expected)
            p.plan.capture_mega_graph()
            for iteration in range(3):
                with self.subTest(
                    mapping=mapping, execution="graph", iteration=iteration
                ):
                    out, lse = p.plan.run_mega()
                    actual = out, lse, p.plan.attention.tile_plan.dynamic_route
                    self.compare(actual, expected)
            del p, data, original, expected, pool
            gc.collect()
            torch.cuda.empty_cache()

    def test_mx_padded_q_live_values_eager_graph_and_rebind(self):
        """Raw/native pad8 Q remains live through the compact MX kernel ABI."""
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("MXFP6 Index requires SM100/SM103")
        from mbsa.index_formats import (
            dequantize_index_k_mxfp6,
            dequantize_index_q_mxfp8,
            quantize_index_k_mxfp6,
            quantize_q_mxfp8,
        )

        nq, nk = 17, 8192
        for native in (False, True):
            with self.subTest(native=native):
                iq_dense, ik_dense, q, k, v = self.inputs(nq, nk)
                codes, scales = quantize_q_mxfp8(iq_dense)
                iq = codes[:, :4]
                if native:
                    iq = iq.view(torch.float8_e4m3fn)
                self.assertEqual(iq.stride(), (1024, 128, 1))
                ik = quantize_index_k_mxfp6(ik_dense.view(-1, 1, 128, 128))
                pool = torch.stack((k, v), dim=2).view(-1, 128, 4, 2, 128)
                pages = torch.arange(nk // 128, dtype=torch.int32, device=q.device)
                kp, vp = (pool[..., half, :].permute(0, 2, 1, 3) for half in (0, 1))
                p = prepare_cutedsl_pipeline(
                    iq,
                    ik,
                    q,
                    kp,
                    vp,
                    index_q_scales=scales,
                    index_page_indices=pages,
                    page_table=pages.unsqueeze(0),
                    logical_k_tokens=nk,
                    partial_dtype=torch.bfloat16,
                )

                def expected():
                    return reference_pipeline(
                        dequantize_index_q_mxfp8(iq, scales),
                        dequantize_index_k_mxfp6(ik).view(nk, 1, 128),
                        q,
                        k,
                        v,
                    )

                before = expected()
                self.compare(p.run(), before)
                compact_pointer = p.plan.mxfp6_router.tensors[0].data_ptr()
                codes[:, :4].copy_(codes[:, :4].roll(1, dims=1))
                live = expected()
                self.assertFalse(
                    torch.equal(before[2], live[2]), "fixture must change routes"
                )
                self.compare(p.run(), live)
                # Rebind new padded addresses while preserving internal compact storage.
                codes, scales = quantize_q_mxfp8(iq_dense.roll(1, dims=1).add(0.125))
                iq = codes[:, :4]
                if native:
                    iq = iq.view(torch.float8_e4m3fn)
                out = torch.empty_like(q)
                p.rebind(
                    iq,
                    ik,
                    q,
                    kp,
                    vp,
                    index_q_scales=scales,
                    index_page_indices=pages,
                    page_table=pages.unsqueeze(0),
                    out=out,
                )
                self.assertEqual(
                    p.plan.mxfp6_router.tensors[0].data_ptr(), compact_pointer
                )
                self.compare(p.run(), expected())
                p.plan.capture_mega_graph()
                before = expected()
                codes[:, :4].copy_(codes[:, :4].roll(1, dims=1))
                live = expected()
                self.assertFalse(
                    torch.equal(before[2], live[2]), "fixture must change routes"
                )
                for _ in range(2):
                    go, gl = p.plan.run_mega()
                    self.compare(
                        (go, gl, p.plan.attention.tile_plan.dynamic_route), live
                    )
                del p, before, live
                gc.collect()
                torch.cuda.empty_cache()

    def test_public_mx_writers_and_native_scores_live_graph(self):
        """Public MX codecs/scorers preserve bytes, causal tails and live graph inputs."""
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("native MX CUDA scorers require SM100/SM103")
        from mbsa.index_formats import (
            dequantize_index_k_mxfp6,
            dequantize_index_q_mxfp8,
            quantize_index_k_mxfp6,
            quantize_q_mxfp8,
        )
        from mbsa.mxfp6 import decode_score, prefill_score, shared_schedule
        from mbsa.mxfp6.write import write_index_k, quantize_q_mxfp8_fused

        lengths, next_n = (257, 4113), 4
        counts = tuple((n + 127) // 128 for n in lengths)
        pages = sum(counts)
        torch.manual_seed(71)
        dense_k = torch.randn(pages, 1, 128, 128, device="cuda", dtype=torch.bfloat16)
        mapping = torch.arange(pages, device="cuda", dtype=torch.int32).flip(0)
        packed = quantize_index_k_mxfp6(dense_k)
        # One unused physical page also checks that the writer preserves bytes.
        expected_cache = torch.full(
            (pages + 1, 1, 128, 100), 0xA5, device="cuda", dtype=torch.uint8
        )
        expected_cache.index_copy_(0, mapping.long(), packed)
        cache = torch.full_like(expected_cache.flatten(), 0xA5)
        slots = (
            mapping.long()[:, None] * 128 + torch.arange(128, device="cuda")[None]
        ).flatten()
        slots = torch.cat((slots, slots.new_tensor([-1])))
        writer_k = torch.cat((dense_k.view(-1, 128), dense_k.new_zeros(1, 128)))
        write_index_k(writer_k, slots, cache)
        torch.testing.assert_close(cache, expected_cache.flatten(), rtol=0, atol=0)
        iq_dense = torch.randn(2 * next_n, 4, 128, device="cuda", dtype=torch.bfloat16)
        iq_dense[0].zero_()
        iq, scales = quantize_q_mxfp8_fused(iq_dense)
        expected_q, expected_scales = quantize_q_mxfp8(iq_dense)
        torch.testing.assert_close(iq, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
        table = torch.zeros(2, max(counts), device="cuda", dtype=torch.int32)
        offset = 0
        for row, count in enumerate(counts):
            table[row, :count].copy_(mapping[offset : offset + count])
            offset += count
        seq = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        cu_q = torch.tensor([0, next_n, 2 * next_n], device="cuda", dtype=torch.int32)
        context = seq - next_n
        owner = type("ScheduleOwner", (), {})()
        schedule = shared_schedule(seq, next_n, owner)
        self.assertEqual(
            schedule.data_ptr(), shared_schedule(seq, next_n, owner).data_ptr()
        )

        def expected(score_sum, init, local):
            dq = dequantize_index_q_mxfp8(iq, scales)[:, :4]
            dk = dequantize_index_k_mxfp6(cache.view(-1, 1, 128, 100))
            rows = []
            for request, length in enumerate(lengths):
                keys = dk.index_select(
                    0, table[request, : counts[request]].long()
                ).view(-1, 128)
                for token in range(next_n):
                    bound = length - next_n + token + 1
                    query = dq[request * next_n + token]
                    dots = (query @ keys[:bound].T) * (128**-0.5)
                    blocks = (bound + 127) // 128
                    values, forced = [], []
                    for block in range(blocks):
                        part = dots[:, block * 128 : min((block + 1) * 128, bound)]
                        value = part.sum(-1) if score_sum else part.max(-1).values
                        sentinel = (
                            1e29
                            if block + local >= blocks
                            else 1e30
                            if block < init
                            else None
                        )
                        if sentinel is not None:
                            value = torch.full_like(value, sentinel)
                        values.append(value)
                        forced.append(sentinel is not None)
                    rows.append(
                        (torch.stack(values, -1), torch.tensor(forced, device="cuda"))
                    )
            return rows

        def compare_scores(results, oracle):
            # Native scorers deliberately leave columns beyond each causal
            # page count unwritten. Compare the complete valid domain only.
            for scores in results:
                for token, (values, forced) in enumerate(oracle):
                    actual = scores[:, token, : values.shape[-1]]
                    torch.testing.assert_close(
                        actual[:, ~forced], values[:, ~forced], rtol=0.001, atol=0.003
                    )
                    torch.testing.assert_close(
                        actual[:, forced], values[:, forced], rtol=0, atol=0
                    )

        for score_sum, init, local in ((False, 0, 0), (True, 0, 0), (False, 1, 1)):
            with self.subTest(score_sum=score_sum, init=init, local=local):

                def run():
                    decode = decode_score(
                        iq,
                        scales,
                        cache,
                        table,
                        seq,
                        init_blocks=init,
                        local_blocks=local,
                        score_sum=score_sum,
                        sm_scale=128**-0.5,
                        next_n=next_n,
                        schedule=schedule,
                    )
                    prefill = prefill_score(
                        iq,
                        scales,
                        cache,
                        table,
                        cu_q,
                        context,
                        max_seq_len=max(lengths),
                        init_blocks=init,
                        local_blocks=local,
                        score_sum=score_sum,
                        sm_scale=128**-0.5,
                    )
                    return decode, prefill

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    eager = run()  # Warm exact descriptor addresses outside capture.
                torch.cuda.current_stream().wait_stream(stream)
                before = expected(score_sum, init, local)
                compare_scores(eager, before)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = run()
                torch.cuda.current_stream().wait_stream(stream)
                iq[:, :4].copy_(iq[:, :4].roll(1, dims=1))
                scales[:, :4].bitwise_xor_(1 << 8)
                words = cache.view(-1, 12800)
                words[:pages, :12288].bitwise_xor_(1)
                words[:pages, 12288:].bitwise_xor_(1)
                for row, count in enumerate(counts):
                    table[row, : count - 1].copy_(table[row, : count - 1].flip(0))
                live = expected(score_sum, init, local)
                self.assertTrue(
                    any(not torch.equal(a[0], b[0]) for a, b in zip(before, live)),
                    "fixture must change valid scores",
                )
                graph.replay()
                compare_scores(captured, live)
                del graph, captured, eager, before, live

    def test_public_mx_prefill_many_ctas_live_graph(self):
        """Varied causal chunks retire TMEM under a grid of hundreds of CTAs."""
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("native MX CUDA scorers require SM100/SM103")
        from mbsa.index_formats import (
            dequantize_index_k_mxfp6,
            dequantize_index_q_mxfp8,
            quantize_index_k_mxfp6,
            quantize_q_mxfp8,
        )
        from mbsa.mxfp6 import prefill_score

        torch.manual_seed(73)
        query_counts, lengths = (17, 129), (257, 4113)
        counts = tuple((n + 127) // 128 for n in lengths)
        dense_k = torch.randn(
            sum(counts), 1, 128, 128, device="cuda", dtype=torch.bfloat16
        )
        cache = quantize_index_k_mxfp6(dense_k).flatten()
        iq, scales = quantize_q_mxfp8(
            torch.randn(sum(query_counts), 4, 128, device="cuda", dtype=torch.bfloat16)
        )
        table = torch.zeros(2, max(counts), device="cuda", dtype=torch.int32)
        table[0, : counts[0]] = torch.arange(counts[0], device="cuda")
        table[1, : counts[1]] = torch.arange(counts[0], sum(counts), device="cuda")
        cu_q = torch.tensor([0, 17, 146], device="cuda", dtype=torch.int32)
        context = torch.tensor(
            [n - q for n, q in zip(lengths, query_counts)],
            device="cuda",
            dtype=torch.int32,
        )

        def oracle(score_sum):
            # The reference has no CUDA arithmetic, so sanitizer failures come
            # from the production path rather than an oracle GEMM.
            dq = dequantize_index_q_mxfp8(iq.cpu(), scales.cpu())[:, :4]
            dk = dequantize_index_k_mxfp6(cache.cpu().view(-1, 1, 128, 100))
            rows, offset = [], 0
            for request, (nq, length) in enumerate(zip(query_counts, lengths)):
                keys = dk.index_select(
                    0, table[request, : counts[request]].cpu().long()
                ).view(-1, 128)
                for token in range(nq):
                    bound = length - nq + token + 1
                    dots = (dq[offset + token] @ keys[:bound].T) * (128**-0.5)
                    values = []
                    for block in range((bound + 127) // 128):
                        part = dots[:, block * 128 : min((block + 1) * 128, bound)]
                        values.append(
                            part.sum(-1) if score_sum else part.max(-1).values
                        )
                    rows.append(torch.stack(values, -1))
                offset += nq
            return rows

        def compare(scores, expected):
            actual = scores.cpu()
            for token, values in enumerate(expected):
                torch.testing.assert_close(
                    actual[:, token, : values.shape[-1]], values, rtol=0.001, atol=0.003
                )

        for score_sum in (False, True):
            with self.subTest(score_sum=score_sum):

                def run():
                    return prefill_score(
                        iq,
                        scales,
                        cache,
                        table,
                        cu_q,
                        context,
                        max_seq_len=max(lengths),
                        score_sum=score_sum,
                        sm_scale=128**-0.5,
                    )

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    eager = run()
                stream.synchronize()
                before = oracle(score_sum)
                compare(eager, before)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = run()
                stream.synchronize()
                iq[:, :4].copy_(iq[:, :4].roll(1, dims=1))
                scales[:, :4].bitwise_xor_(1 << 8)
                cache.view(-1, 12800)[:, :12288].bitwise_xor_(1)
                live = oracle(score_sum)
                self.assertTrue(
                    any(not torch.equal(a, b) for a, b in zip(before, live))
                )
                graph.replay()
                compare(captured, live)
                del graph, captured, eager

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

        for parts in (1, 2):
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

    def test_cutedsl_vllm_interleaved_cache_and_single_query_chunk(self):
        """Use the actual split/permuted cache strides, including a KV tail."""
        if torch.cuda.get_device_capability() not in CUTEDSL_ARCHS:
            self.skipTest("CuTe DSL runtime requires SM90/SM100/SM103")
        from mbsa import prepare_cutedsl_pipeline

        for qn, kn in (
            (1, 4113),
            (17, 4113),
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
