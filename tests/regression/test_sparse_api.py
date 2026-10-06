# SPDX-License-Identifier: Apache-2.0
"""MSA-compatible ``mbsa.sparse`` entry points against an eager reference."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import torch


def _cuda_arch():
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else None


def _q2k(q_lens, kv_lens, heads, topk, generator, block_size=128):
    """Distinct batch-local blocks per (head, query); some trailing -1 padding."""
    rows = []
    for qn, kn in zip(q_lens, kv_lens):
        blocks = (kn + block_size - 1) // block_size
        for _ in range(qn):
            per_head = []
            for _ in range(heads):
                count = min(blocks, topk) - int(
                    torch.randint(0, 2, (1,), generator=generator)
                )
                picked = torch.randperm(blocks, generator=generator)[: max(count, 1)]
                per_head.append(
                    torch.cat([picked, picked.new_full((topk - picked.numel(),), -1)])
                )
            rows.append(torch.stack(per_head))
    return torch.stack(rows).permute(1, 0, 2).contiguous().to(torch.int32)


def _reference(
    q, k, v, q2k, q_lens, kv_lens, causal, initial=None, block_size=128, temperature=1.0
):
    """Attention over the union of selected blocks per (query, KV head)."""
    total_q, q_heads, dim = q.shape
    heads = k.shape[1]
    group = q_heads // heads
    out = torch.zeros(total_q, q_heads, dim, dtype=torch.float32, device=q.device)
    lse = torch.full((total_q, q_heads), float("-inf"), device=q.device)
    q_start = k_start = 0
    for qn, kn in zip(q_lens, kv_lens):
        for i in range(qn):
            row = q_start + i
            limit = kn - qn + i if causal else kn - 1
            for h in range(heads):
                blocks = sorted({int(b) for b in q2k[h, row] if b >= 0})
                tokens = [
                    t
                    for b in blocks
                    for t in range(b * block_size, min(kn, (b + 1) * block_size))
                    if t <= limit
                ]
                heads_q = slice(h * group, (h + 1) * group)
                scores = torch.empty(group, 0, device=q.device)
                if tokens:
                    idx = torch.tensor(tokens, device=q.device) + k_start
                    scores = (
                        q[row, heads_q].float()
                        @ k[idx, h].float().t()
                        * (128**-0.5 / temperature)
                    )
                    values = v[idx, h].float()
                if initial is not None:
                    init_out, init_lse = initial
                    scores = torch.cat([scores, init_lse[row, heads_q, None]], 1)
                if scores.shape[1] == 0:
                    continue
                m = scores.max(1, keepdim=True).values
                p = torch.exp(scores - m)
                total = p.sum(1, keepdim=True)
                acc = torch.zeros(group, dim, device=q.device)
                if tokens:
                    acc = p[:, : len(tokens)] @ values
                if initial is not None:
                    acc = acc + p[:, -1:] * init_out[row, heads_q].float()
                out[row, heads_q] = acc / total
                lse[row, heads_q] = (m + total.log()).squeeze(1)
        q_start += qn
        k_start += kn
    return out, lse


class MaxwellGeometryContracts(unittest.TestCase):
    def test_rejects_non_maxwell_geometry(self):
        from mbsa.sparse import sparse_atten_func, build_k2q_csr

        metadata = torch.empty(4, 0, dtype=torch.int32)
        for q_heads, kv_heads, dim in (
            (16, 2, 128),
            (32, 8, 128),
            (64, 4, 128),
            (32, 4, 64),
        ):
            with self.subTest(q_heads=q_heads, kv_heads=kv_heads, dim=dim):
                q = torch.empty(1, q_heads, dim, dtype=torch.bfloat16)
                k = torch.empty(1, kv_heads, dim, dtype=torch.bfloat16)
                with self.assertRaisesRegex(NotImplementedError, "Maxwell attention"):
                    sparse_atten_func(q, k, k, metadata, metadata, 16)
        cu = torch.tensor([0, 1], dtype=torch.int32)
        with self.assertRaisesRegex(NotImplementedError, "4 KV heads"):
            build_k2q_csr(torch.empty(2, 1, 16, dtype=torch.int32), cu, cu, 128)
        with self.assertRaisesRegex(NotImplementedError, "GQA=8"):
            build_k2q_csr(
                torch.empty(4, 1, 16, dtype=torch.int32), cu, cu, 128, qhead_per_kv=4
            )


@unittest.skipUnless(
    _cuda_arch() in ((9, 0), (10, 0), (10, 3)), "SM90/SM100 GPU required"
)
class SparseApiTest(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(7)

    def rand(self, *shape):
        return torch.randn(*shape, generator=self.generator).to("cuda", torch.bfloat16)

    def check(self, actual, expected, out_tol=0.03):
        out, lse = actual
        eo, el = expected
        finite = torch.isfinite(el)
        torch.testing.assert_close(out.float(), eo, rtol=out_tol, atol=out_tol)
        self.assertTrue(torch.equal(torch.isfinite(lse), finite))
        torch.testing.assert_close(lse[finite], el[finite], rtol=0.002, atol=0.01)

    def run_case(
        self,
        q_lens,
        kv_lens,
        *,
        q_heads=32,
        heads=4,
        topk=16,
        causal=False,
        partial_dtype=torch.bfloat16,
        paged=False,
        initial=False,
        out_tol=0.03,
        block_size=128,
    ):
        from mbsa.sparse import build_k2q_csr, sparse_atten_func

        q = self.rand(sum(q_lens), q_heads, 128)
        k, v = self.rand(sum(kv_lens), heads, 128), self.rand(sum(kv_lens), heads, 128)
        q2k = _q2k(q_lens, kv_lens, heads, topk, self.generator, block_size).cuda()
        cu = lambda xs: torch.tensor(
            [0] + list(torch.tensor(xs).cumsum(0)), dtype=torch.int32, device="cuda"
        )
        cu_q, cu_k = cu(q_lens), cu(kv_lens)
        row_ptr, q_idx = build_k2q_csr(
            q2k,
            cu_q,
            cu_k,
            block_size,
            total_k=sum(kv_lens),
            max_seqlen_k=max(kv_lens),
            max_seqlen_q=max(q_lens),
            qhead_per_kv=q_heads // heads,
        )
        kwargs = dict(
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max(q_lens),
            max_seqlen_k=max(kv_lens),
            blk_kv=block_size,
            causal=causal,
            partial_dtype=partial_dtype,
            return_softmax_lse=True,
        )
        kk, vv = k, v
        if paged:
            blocks = [(n + block_size - 1) // block_size for n in kv_lens]
            perm = torch.randperm(sum(blocks), generator=self.generator).cuda()
            pool = torch.zeros(
                sum(blocks),
                block_size,
                heads,
                2,
                128,
                dtype=torch.bfloat16,
                device="cuda",
            )
            table = torch.zeros(
                len(blocks), max(blocks), dtype=torch.int32, device="cuda"
            )
            start = page = 0
            for r, (kn, nb) in enumerate(zip(kv_lens, blocks)):
                for half, src in enumerate((k, v)):
                    padded = torch.zeros(
                        nb * block_size, heads, 128, dtype=torch.bfloat16, device="cuda"
                    )
                    padded[:kn] = src[start : start + kn]
                    pool[perm[page : page + nb], :, :, half] = padded.view(
                        nb, block_size, heads, 128
                    )
                table[r, :nb] = perm[page : page + nb].to(torch.int32)
                start += kn
                page += nb
            kk, vv = (pool[..., h, :].permute(0, 2, 1, 3) for h in (0, 1))
            kwargs.update(
                page_table=table,
                seqused_k=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
            )
        reference_initial = None
        if initial:
            init_out = self.rand(sum(q_lens), q_heads, 128)
            init_lse = (
                torch.randn(sum(q_lens), q_heads, generator=self.generator).cuda() + 3
            )
            kwargs.update(
                initial_out=init_out, initial_lse=init_lse.t().contiguous()[None]
            )
            reference_initial = (init_out, init_lse)
        actual = sparse_atten_func(q, kk, vv, row_ptr, q_idx, topk, **kwargs)
        expected = _reference(
            q,
            k,
            v,
            q2k,
            q_lens,
            kv_lens,
            causal,
            reference_initial,
            block_size=block_size,
        )
        self.check(actual, expected, out_tol)

    def test_dense_batch_one(self):
        self.run_case((96,), (2048,))

    def test_packed_causal_tails(self):
        self.run_case((33, 1, 70), (700, 129, 1000), topk=8, causal=True)

    def test_paged_fp8_initial_state(self):
        self.run_case(
            (57,),
            (2013,),
            topk=32,
            causal=True,
            paged=True,
            partial_dtype=torch.float8_e4m3fn,
            initial=True,
            out_tol=0.05,
        )
        if _cuda_arch() == (9, 0):
            self.run_case(
                (17, 40),
                (1500, 513),
                topk=32,
                causal=True,
                paged=True,
                partial_dtype=torch.float8_e4m3fn,
                initial=True,
                out_tol=0.05,
            )

    def test_maxwell_fp32_and_fp16_partials(self):
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                self.run_case(
                    (50, 9),
                    (640, 300),
                    q_heads=32,
                    heads=4,
                    topk=4,
                    partial_dtype=dtype,
                )

    def test_csr_matches_definition(self):
        from mbsa.sparse import build_k2q_csr

        q_lens, kv_lens = (5, 3, 7), (300, 1000, 129)
        q2k = _q2k(q_lens, kv_lens, 4, 4, self.generator).cuda()
        cu = lambda xs: torch.tensor(
            [0] + list(torch.tensor(xs).cumsum(0)), dtype=torch.int32, device="cuda"
        )
        row_ptr, q_idx = build_k2q_csr(
            q2k,
            cu(q_lens),
            cu(kv_lens),
            128,
            total_k=sum(kv_lens),
            max_seqlen_k=max(kv_lens),
        )
        blocks = [(n + 127) // 128 for n in kv_lens]
        rows = [
            (b, level)
            for level in range(max(blocks))
            for b in range(len(blocks))
            if level < blocks[b]
        ]
        for h in range(4):
            for r, (b, level) in enumerate(rows):
                q0 = sum(q_lens[:b])
                expected = [
                    i for i in range(q_lens[b]) if level in q2k[h, q0 + i].tolist()
                ]
                got = q_idx[h, row_ptr[h, r] : row_ptr[h, r + 1]].tolist()
                self.assertEqual(got, expected, (h, r))

    def test_matches_mbsa_dynamic_path(self):
        """Routes and fixed state from the MBSA plan reproduce its output."""
        from mbsa import prepare_cutedsl_pipeline
        from mbsa.sparse import build_k2q_csr, sparse_atten_func

        if _cuda_arch() != (9, 0):
            self.skipTest("plan internals are SM90-specific")
        qn, kn = 256, 8192
        iq, q = self.rand(qn, 4, 128), self.rand(qn, 32, 128)
        k, v, ik = self.rand(kn, 4, 128), self.rand(kn, 4, 128), self.rand(kn, 1, 128)
        pipeline = prepare_cutedsl_pipeline(iq, ik, q, k, v)
        out, lse, route = pipeline.run()
        ws = pipeline.plan.workspace
        cu_q = torch.tensor([0, qn], dtype=torch.int32, device="cuda")
        cu_k = torch.tensor([0, kn], dtype=torch.int32, device="cuda")
        row_ptr, q_idx = build_k2q_csr(
            route.contiguous(), cu_q, cu_k, 128, total_k=kn, max_seqlen_k=kn
        )
        actual = sparse_atten_func(
            q,
            k,
            v,
            row_ptr,
            q_idx,
            16,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=qn,
            max_seqlen_k=kn,
            return_softmax_lse=True,
            initial_out=ws.fixed_out,
            initial_lse=ws.fixed_lse.t().contiguous()[None],
        )
        torch.testing.assert_close(actual[0], out, rtol=0, atol=2**-7)
        torch.testing.assert_close(actual[1], lse, rtol=0, atol=1e-5)

    def test_paged_block_sizes_and_partial_types(self):
        for block_size in (32, 64, 128):
            for dtype in (
                torch.float32,
                torch.bfloat16,
                torch.float16,
                torch.float8_e4m3fn,
            ):
                with self.subTest(block_size=block_size, partial=dtype):
                    self.run_case(
                        (12,),
                        (block_size * 8 + 4,),
                        q_heads=32,
                        heads=4,
                        topk=8,
                        causal=True,
                        paged=True,
                        initial=True,
                        partial_dtype=dtype,
                        block_size=block_size,
                        out_tol=0.05,
                    )

                    self.run_case(
                        (3, 9),
                        (block_size * 6 + 3, block_size * 2 + 1),
                        q_heads=32,
                        heads=4,
                        topk=8,
                        causal=True,
                        paged=True,
                        partial_dtype=dtype,
                        block_size=block_size,
                        out_tol=0.05,
                    )

    def test_supported_parameter_grid(self):
        """Every legal CSR block/Top-K/partial/storage choice, Maxwell heads."""
        for block_size in (32, 64, 128):
            for topk in (4, 8, 16, 32):
                for dtype in (
                    torch.float32,
                    torch.bfloat16,
                    torch.float16,
                    torch.float8_e4m3fn,
                ):
                    for paged in (False, True):
                        with self.subTest(
                            block=block_size, topk=topk, partial=dtype, paged=paged
                        ):
                            self.run_case(
                                (9, 23),
                                (block_size * 9 + 3, block_size * 2 + 1),
                                topk=topk,
                                causal=True,
                                partial_dtype=dtype,
                                paged=paged,
                                block_size=block_size,
                                out_tol=0.05 if dtype == torch.float8_e4m3fn else 0.03,
                            )

    def test_block_sizes_workspace_split_and_temperature(self):
        from mbsa.sparse import build_k2q_csr, sparse_atten_func, SparseForwardPhase

        for block_size in (32, 64, 128):
            for partial_dtype in (torch.bfloat16, torch.float8_e4m3fn):
                with self.subTest(block_size=block_size, partial=partial_dtype):
                    q_lens, kv_lens = (5, 9), (block_size * 5 + 3, block_size * 3 + 1)
                    q = self.rand(14, 32, 128)
                    k, v = (
                        self.rand(sum(kv_lens), 4, 128),
                        self.rand(sum(kv_lens), 4, 128),
                    )
                    q2k = _q2k(q_lens, kv_lens, 4, 4, self.generator, block_size).cuda()
                    cu_q = torch.tensor([0, 5, 14], device="cuda", dtype=torch.int32)
                    cu_k = torch.tensor(
                        [0, kv_lens[0], sum(kv_lens)], device="cuda", dtype=torch.int32
                    )
                    row_ptr, indices, schedule = build_k2q_csr(
                        q2k,
                        cu_q,
                        cu_k,
                        block_size,
                        total_k=sum(kv_lens),
                        max_seqlen_k=max(kv_lens),
                        max_seqlen_q=9,
                        qhead_per_kv=8,
                        return_schedule=True,
                    )
                    ws = (
                        torch.empty(4, 14, 32, 128, device="cuda", dtype=partial_dtype),
                        torch.empty(4, 14, 32, device="cuda"),
                        torch.empty(14, 32, device="cuda"),
                    )
                    kwargs = dict(
                        cu_seqlens_q=cu_q,
                        cu_seqlens_k=cu_k,
                        max_seqlen_q=9,
                        max_seqlen_k=max(kv_lens),
                        blk_kv=block_size,
                        causal=True,
                        partial_dtype=partial_dtype,
                        return_softmax_lse=True,
                        schedule=schedule,
                        workspace=ws,
                    )
                    expected = _reference(
                        q, k, v, q2k, q_lens, kv_lens, True, block_size=block_size
                    )
                    full = sparse_atten_func(q, k, v, row_ptr, indices, 4, **kwargs)
                    self.check(full, expected, 0.05)
                    if _cuda_arch() == (9, 0):
                        with self.assertRaisesRegex(ValueError, "completed MSA"):
                            other = type(schedule)(
                                4, 14, 4, row_ptr.shape[1] - 1, 8, block_size, q.device
                            )
                            sparse_atten_func(
                                q,
                                k,
                                v,
                                row_ptr,
                                indices,
                                4,
                                **{**kwargs, "schedule": other},
                                phase=SparseForwardPhase.COMBINE,
                            )
                    sparse_atten_func(
                        q,
                        k,
                        v,
                        row_ptr,
                        indices,
                        4,
                        **kwargs,
                        phase=SparseForwardPhase.MSA,
                    )
                    # Changing Q must not cause COMBINE to rerun the producer.
                    old_q = q.clone()
                    q.add_(1)
                    split = sparse_atten_func(
                        q,
                        k,
                        v,
                        row_ptr,
                        indices,
                        4,
                        **kwargs,
                        phase=SparseForwardPhase.COMBINE,
                    )
                    self.check(split, expected, 0.05)
                    q.copy_(old_q)
                    result = sparse_atten_func(
                        q,
                        k,
                        v,
                        row_ptr,
                        indices,
                        4,
                        **kwargs,
                        return_temperature_lse=True,
                        lse_temperature_scale=2,
                    )
                    self.check(result[:2], expected, 0.05)
                    _, temp_lse = _reference(
                        q,
                        k,
                        v,
                        q2k,
                        q_lens,
                        kv_lens,
                        True,
                        block_size=block_size,
                        temperature=2,
                    )
                    torch.testing.assert_close(
                        result[2], temp_lse, rtol=0.002, atol=0.01
                    )
                    unit = sparse_atten_func(
                        q,
                        k,
                        v,
                        row_ptr,
                        indices,
                        4,
                        **kwargs,
                        return_temperature_lse=True,
                        lse_temperature_scale=1,
                    )
                    self.assertEqual(unit[1].data_ptr(), unit[2].data_ptr())


if __name__ == "__main__":
    unittest.main()
