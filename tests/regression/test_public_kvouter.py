# SPDX-License-Identifier: Apache-2.0
"""Public KV-outer live cache, rebind, phase separation and graph route refresh."""
import math
import importlib.util
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def _oracle(q, k, v, selected, table, kv_len):
    """Independent FP32 attention over distinct selected logical blocks."""
    tq, hq, dim = q.shape
    hkv = k.shape[1]
    group = hq // hkv
    logical_k = k.index_select(0, table[0].long()).permute(0, 2, 1, 3).reshape(-1, hkv, dim)
    logical_v = v.index_select(0, table[0].long()).permute(0, 2, 1, 3).reshape(-1, hkv, dim)
    out = torch.zeros(tq, hq, dim, device=q.device)
    lse = torch.full((hq, tq), -torch.inf, device=q.device)
    routes = selected.cpu()
    for qi in range(tq):
        limit = kv_len - tq + qi
        for hi in range(hkv):
            blocks = sorted(set(int(x) for x in routes[qi, hi] if x >= 0))
            ids = [token for block in blocks
                   for token in range(block * 128, min((block + 1) * 128, kv_len))
                   if token <= limit]
            if not ids:
                continue
            heads = slice(hi * group, (hi + 1) * group)
            scores = q[qi, heads].float() @ logical_k[ids, hi].float().t() / math.sqrt(dim)
            out[qi, heads] = scores.softmax(-1) @ logical_v[ids, hi].float()
            lse[heads, qi] = scores.logsumexp(-1)
    return out, lse


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PublicKVOuterRegression(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("SM100/SM103 is required")
        try:
            available = importlib.util.find_spec("flash_attn.cute.flash_fwd_sm100") is not None
        except (ImportError, ValueError):
            available = False
        if not available:
            self.skipTest("FlashAttention-4 CuTe is required")

    def _case(self, page_size):
        tq, kv_len, hkv, hq, dim, topk = 17, 8193, 4, 32, 128, 4
        pages = (kv_len + page_size - 1) // page_size
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(193)
        q = (torch.randn(tq, hq, dim, generator=generator, device=device) * 0.2).bfloat16()
        # Interleaved live K/V storage, matching a paged allocator's strided views.
        pool = torch.randn(pages, page_size, hkv, 2, dim, generator=generator, device=device).bfloat16()
        k = pool[:, :, :, 0].permute(0, 2, 1, 3)
        v = pool[:, :, :, 1].permute(0, 2, 1, 3)
        table = torch.randperm(pages, generator=generator, device=device).int().unsqueeze(0)
        selected = torch.full((hkv, tq, topk), -1, device=device, dtype=torch.int32).permute(1, 0, 2)
        selected[:, :, :3] = torch.tensor([0, 2, 64], device=device, dtype=torch.int32)
        used = torch.tensor([kv_len], device=device, dtype=torch.int32)
        cuq = torch.tensor([0, tq], device=device, dtype=torch.int64)
        # Finite poison isolates real-length masking; NaN-padding handling is a
        # separate property and is not supplied by this public Fireworks kernel.
        last_page = int(table[0, -1])
        k[last_page, :, kv_len % page_size:] = 1000
        v[last_page, :, kv_len % page_size:] = 1000
        return q, k, v, pool, selected, table, cuq, used

    def _check(self, plan, inputs):
        q, k, v, _, selected, table, _, used = inputs
        expected = _oracle(q, k, v, selected, table, int(used[0]))
        torch.testing.assert_close(plan.out.float(), expected[0], atol=0.025, rtol=0.025)
        torch.testing.assert_close(plan.lse, expected[1], atol=0.002, rtol=0.002)

    def test_live_strided_cache_rebind_stages_and_route_graph(self):
        from mbsa.cute.kvouter import PreparedKVOuter
        inputs = self._case(128)
        q, k, v, pool, selected, table, cuq, used = inputs
        plan = PreparedKVOuter(q, k, v, selected, table, cuq, used)
        plan.run()
        self._check(plan, inputs)
        pointers = tuple(t.data_ptr() for t in (*plan.buffers.values(), plan.out, plan.lse))

        # Combine must consume existing partials even if Q changes after forward.
        previous = plan.out.clone(), plan.lse.clone()
        q.zero_()
        plan.run_combine()
        torch.testing.assert_close(plan.out, previous[0], atol=0, rtol=0)
        torch.testing.assert_close(plan.lse, previous[1], atol=0, rtol=0)
        plan.run_forward()
        plan.run_combine()
        self._check(plan, inputs)

        v.add_(0.5)
        plan.run()
        self._check(plan, inputs)

        # Rebind must use all four new addresses without a stale copied cache.
        new_pool = pool.clone()
        new_k = new_pool[:, :, :, 0].permute(0, 2, 1, 3)
        new_v = new_pool[:, :, :, 1].permute(0, 2, 1, 3)
        new_v.add_(0.75)
        new_table = table.flip(1).contiguous()
        new_q = q.clone().normal_(0, 0.2)
        plan.rebind(new_q, new_k, new_v, new_table)
        rebound = (new_q, new_k, new_v, new_pool, selected, new_table, cuq, used)
        plan.run()
        self._check(plan, rebound)
        with self.assertRaisesRegex(ValueError, "strides"):
            plan.rebind(new_q, new_k.contiguous(), new_v, new_table)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                plan.run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
        selected[:, :, :3] = torch.tensor([0, 4, 63], device=q.device, dtype=torch.int32)
        graph.replay()
        self._check(plan, rebound)
        new_v.add_(0.25)
        graph.replay()
        self._check(plan, rebound)
        selected.fill_(-1)
        graph.replay()
        self.assertTrue(bool(torch.all(plan.out == 0)))
        self.assertTrue(bool(torch.all(plan.lse == -torch.inf)))
        self.assertEqual(pointers, tuple(t.data_ptr() for t in (*plan.buffers.values(), plan.out, plan.lse)))

    def test_page64_half_full_last_block(self):
        from mbsa.cute.kvouter import PreparedKVOuter
        inputs = self._case(64)
        q, k, v, _, selected, table, cuq, used = inputs
        plan = PreparedKVOuter(q, k, v, selected, table, cuq, used)
        plan.run()
        self._check(plan, inputs)

    def _pipeline(self, kv_len, seed, mapping, partial_dtype=torch.bfloat16):
        from tests.common import make_inputs
        from mbsa import prepare_cutedsl_pipeline
        case = dict(requests=[dict(phase="prefill", q=17, kv=kv_len)], seed=seed)
        data = make_inputs(case, mapping)
        data["k"] = data["pool"][..., 0, :].permute(0, 2, 1, 3)
        data["v"] = data["pool"][..., 1, :].permute(0, 2, 1, 3)
        data["out"] = torch.empty_like(data["q"])
        pipeline = prepare_cutedsl_pipeline(
            data["iq"], data["ik"], data["q"], data["k"], data["v"],
            cu_seqlens_q=data["cu_q"], cu_seqlens_k=data["cu_k"],
            index_page_indices=data["pages"], page_table=data["table"],
            out=data["out"], partial_dtype=partial_dtype,
        )
        return pipeline, data

    def _check_pipeline(self, pipeline, data, kv_len, atol=0.03):
        route = pipeline.plan.attention.tile_plan.dynamic_route
        tq = data["q"].shape[0]
        selected = torch.full((tq, 4, 32), -1, device="cuda", dtype=torch.int32)
        selected[:, :, 16:] = route.permute(1, 0, 2)
        for qi in range(tq):
            visible = (kv_len - tq + qi) // 128 + 1
            fixed = sorted(set(range(min(8, visible))) | set(range(max(8, visible - 8), visible)))
            selected[qi, :, :len(fixed)] = torch.tensor(fixed, device="cuda", dtype=torch.int32)
        expected = _oracle(data["q"], data["k"], data["v"], selected, data["table"], kv_len)
        attention = pipeline.plan.attention
        torch.testing.assert_close(attention.out.float(), expected[0], atol=atol, rtol=atol)
        torch.testing.assert_close(attention.out_lse.T, expected[1], atol=0.01, rtol=0.002)

    def test_auto_full_pipeline_layer_rebind_and_mega_graph(self):
        pipeline, data = self._pipeline(8192, 71, "identity")
        self.assertEqual(pipeline.plan.attention.dynamic_backend, "fireworks")
        if torch.cuda.get_device_capability() == (10, 3):
            self.assertEqual(pipeline.plan.attention.dynamic_kvouter.q_load_stage, 3)
        pipeline.run()
        self._check_pipeline(pipeline, data, 8192)

        # Serving bind_layer keeps metadata and switches live layer tensors.
        other, layer = self._pipeline(8192, 72, "identity")
        del other
        pipeline.plan.bind_layer(layer["iq"], layer["ik"], layer["q"],
                                 layer["k"], layer["v"], layer["out"])
        pipeline.run()
        self._check_pipeline(pipeline, layer, 8192)

        # Public paged rebind also switches the page table and Index-page map.
        other, rebound = self._pipeline(8192, 73, "permuted")
        del other
        pipeline.rebind(rebound["iq"], rebound["ik"], rebound["q"],
                        rebound["k"], rebound["v"],
                        index_page_indices=rebound["pages"], page_table=rebound["table"],
                        out=rebound["out"])
        pipeline.run()
        self._check_pipeline(pipeline, rebound, 8192)
        prepared = pipeline.plan.attention.dynamic_kvouter
        pointers = tuple(t.data_ptr() for t in (*prepared.buffers.values(), prepared.out, prepared.lse))
        pipeline.plan.capture_mega_graph()
        old_route = pipeline.plan.attention.tile_plan.dynamic_route.clone()
        rebound["iq"].mul_(-1)
        rebound["v"].add_(0.25)
        pipeline.plan.run_mega()
        self._check_pipeline(pipeline, rebound, 8192)
        self.assertFalse(torch.equal(old_route, pipeline.plan.attention.tile_plan.dynamic_route))
        self.assertEqual(pointers, tuple(t.data_ptr() for t in (*prepared.buffers.values(), prepared.out, prepared.lse)))
        with self.assertRaisesRegex(RuntimeError, "mega graph"):
            pipeline.rebind(rebound["iq"], rebound["ik"], rebound["q"],
                            rebound["k"], rebound["v"],
                            index_page_indices=rebound["pages"], page_table=rebound["table"],
                            out=rebound["out"])

    def test_fast_path_nan_kv_tail(self):
        for kv_len, partial_dtype in ((8193, torch.bfloat16),):
            with self.subTest(kv_len=kv_len, partial_dtype=partial_dtype):
                pipeline, data = self._pipeline(kv_len, 74, "permuted", partial_dtype)
                self.assertEqual(pipeline.plan.attention.dynamic_backend, "fireworks")
                self.assertIsNotNone(pipeline.plan.attention.dynamic_kvouter)
                if kv_len % 128:
                    last = int(data["table"][0, -1])
                    data["pool"][last, kv_len % 128:] = torch.nan
                pipeline.run()
                self._check_pipeline(pipeline, data, kv_len, atol=0.04)

    def test_fp8_partials_scale_large_values_and_live_graph(self):
        """FP8 storage must preserve mass and avoid overflow for large finite V."""
        from mbsa.cute.kvouter import PreparedKVOuter
        q = torch.zeros(17, 32, 128, device='cuda', dtype=torch.bfloat16)
        k = torch.zeros(32, 4, 128, 128, device='cuda', dtype=q.dtype)
        v = torch.empty_like(k)
        values = torch.linspace(-4096, 4096, 128, device=q.device).bfloat16()
        v.copy_(values)
        selected = torch.tensor([8, 9, 10], device=q.device, dtype=torch.int32).expand(17, 4, 3).contiguous()
        table = torch.arange(32, device=q.device, dtype=torch.int32)[None]
        used = torch.tensor([4096], device=q.device, dtype=torch.int32)
        cuq = torch.tensor([0, 17], device=q.device, dtype=torch.int64)
        p = PreparedKVOuter(q, k, v, selected, table, cuq, used,
                            partial_dtype=torch.float8_e4m3fn)

        def check():
            expected = v[8:11].float().mean((0, 2)).repeat_interleave(8, 0)
            err = (p.out.float() - expected[None]).abs()
            self.assertLessEqual(err.max().item(), expected.abs().max().item() * .04)
            torch.testing.assert_close(p.lse, torch.full_like(p.lse, math.log(384)), atol=.002, rtol=.002)
            self.assertTrue(bool(torch.isfinite(p.out).all()))

        p.run()
        check()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                p.run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            p.run()
        v.mul_(4)
        graph.replay()
        check()
        selected.fill_(-1)
        graph.replay()
        self.assertEqual(torch.count_nonzero(p.out).item(), 0)
        self.assertTrue(bool(torch.isneginf(p.lse).all()))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class HeadMajorMergeRegression(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("SM100/SM103 is required")

    def test_head_major_lse_empty_states_inplace_and_graph(self):
        from mbsa.cute.src.sm100.merge_vectorized import _merge_two_attention_states_vectorized as _merge_two_attention_states_cutedsl
        from mbsa.cute.src.sm100.merge_vectorized import _merge_two_attention_states_vectorized

        for merge in (_merge_two_attention_states_cutedsl, _merge_two_attention_states_vectorized):
            with self.subTest(merge=merge.__name__):
                generator = torch.Generator(device="cuda").manual_seed(791)
                a = torch.randn(17, 32, 128, generator=generator, device="cuda").bfloat16()
                b = torch.randn(a.shape, generator=generator, device="cuda").bfloat16()
                la = torch.randn(1, 32, 17, generator=generator, device="cuda")
                native_lb = torch.randn(32, 17, generator=generator, device="cuda")
                lb = native_lb.T
                # Distinct token/head masks catch accidental flattening in the wrong order.
                la[0, 2, 3] = -torch.inf
                a[3, 2] = torch.nan
                native_lb[5, 7] = -torch.inf
                b[7, 5] = torch.nan
                la[0, 9, 11] = native_lb[9, 11] = -torch.inf
                a[11, 9] = b[11, 9] = torch.nan
                out = torch.empty_like(a)
                out_lse = torch.empty(17, 32, device="cuda")

                def expected():
                    log_total = torch.logaddexp(la[0].T, lb)
                    wa = torch.nan_to_num(torch.exp(la[0].T - log_total))
                    wb = torch.nan_to_num(torch.exp(lb - log_total))
                    value = torch.nan_to_num(a.float()) * wa[..., None] + torch.nan_to_num(b.float()) * wb[..., None]
                    return value, log_total

                merge(a, la, b, lb, out, out_lse)
                torch.testing.assert_close(out.float(), expected()[0], atol=0.02, rtol=0.02)
                torch.testing.assert_close(out_lse, expected()[1], atol=0.0001, rtol=0.0001)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    merge(a, la, b, lb, out, out_lse)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    merge(a, la, b, lb, out, out_lse)
                native_lb.add_(0.5)
                graph.replay()
                torch.testing.assert_close(out.float(), expected()[0], atol=0.02, rtol=0.02)
                torch.testing.assert_close(out_lse, expected()[1], atol=0.0001, rtol=0.0001)
                reference = expected()
                merge(a, la, b, lb, b, out_lse)
                torch.testing.assert_close(b.float(), reference[0], atol=0.02, rtol=0.02)
                with self.assertRaisesRegex(ValueError, "contiguous"):
                    merge(a, la, b, lb, out, torch.empty(32, 17, device="cuda").T)
                with self.assertRaisesRegex(ValueError, "contiguous"):
                    merge(a, torch.empty(1, 17, 32, device="cuda").transpose(1, 2), b, lb, out, out_lse)
                with self.assertRaisesRegex(ValueError, "head-major"):
                    merge(a, la, b, torch.empty(17, 64, device="cuda")[:, ::2], out, out_lse)
