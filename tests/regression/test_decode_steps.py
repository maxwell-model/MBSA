# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Decode-like steps on the optimized backend: every request has one query token.

Such a step runs query-outer: the Index stage, then one BSA pass over each token's prefix, local-window and
routed Dynamic blocks, which writes the final output and LSE. Nothing in it is sized by the KV blocks of the
whole batch (the KV-outer path builds a CSR whose histogram is), so 32 requests of 512K and 1M KV tokens run.
Packed steps bind the heads-in-rows Index launch (IndexDispatch.heads_in_rows): a tile row is one
(token, Index head) pair and one CTA scores all heads of a request, reading its Index K once.

Covers the reference comparison (ragged KV with partial tail pages, fewer than 16 Dynamic candidates, long KV,
batch one, FP8 K/V), the bound launch, bit-identity of the heads-in-rows launch with one CTA per head for
BF16 / FP16 / E4M3 Index (serial and tree merges), the CUDA mega-graph, layer rebinding and serving steps.
"""
import dataclasses
import unittest

import torch

from mbsa import prepare_cutedsl_pipeline
from mbsa.backends.optimized.index_policy import (AUTO_INDEX_DISPATCH, BASELINE_INDEX_DISPATCH,
                                                  SERIAL_MERGE_MAX_PARTITIONS, index_dispatch_override)
from tests.common import BackendCaseMixin, baseline_index_dispatch
from tests.reference import reference_pipeline

E4M3 = torch.float8_e4m3fn
# Requests without Dynamic candidates (up to 16 visible pages, where prefix and local window overlap), partial
# tail pages, 1 to 16 Dynamic candidates (17 to 32 visible pages), and longer KV.
RAGGED = (300, 1024, 2048, 2049, 2176, 2304, 3000, 4096, 4097, 8192, 33000, 65536, 131072)


def decode_fixture(kv_lens, *, seed=17, index_dtype=torch.bfloat16, kv_dtype=torch.bfloat16, q_lens=None):
    """Requests (one query token each unless q_lens is given) over one shared physical cache; each request
    maps its pages differently."""
    q_lens = [1] * len(kv_lens) if q_lens is None else list(q_lens)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def rand(shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)

    counts = [(n + 127) // 128 for n in kv_lens]
    physical = max(counts) + 3
    pool = rand((physical, 128, 4, 2, 128))
    k = pool[:, :, :, 0].permute(0, 2, 1, 3).to(kv_dtype)
    v = pool[:, :, :, 1].permute(0, 2, 1, 3).to(kv_dtype)
    index_k = rand((physical, 1, 128, 128)).to(index_dtype)
    q = rand((sum(q_lens), 32, 128))
    index_q = rand((sum(q_lens), 4, 128)).to(index_dtype)
    order = torch.randperm(physical, device="cuda", generator=generator).to(torch.int32)
    rows = [order.roll(7 * request)[:count] for request, count in enumerate(counts)]
    table = torch.zeros((len(kv_lens), max(counts)), device="cuda", dtype=torch.int32)
    for request, pages in enumerate(rows):
        table[request, :pages.numel()].copy_(pages)

    def prefix(values):
        sums = [0]
        for value in values:
            sums.append(sums[-1] + value)
        return torch.tensor(sums, device="cuda", dtype=torch.int32)

    inputs = (index_q, index_k, q, k, v)
    kwargs = dict(index_page_indices=torch.cat(rows).contiguous(), page_table=table, backend="optimized")
    if len(kv_lens) == 1:
        kwargs["logical_k_tokens"] = kv_lens[0]
    else:
        kwargs.update(cu_seqlens_q=prefix(q_lens), cu_seqlens_k=prefix(kv_lens),
                      request_lens=list(zip(q_lens, kv_lens)))
    return inputs, kwargs, rows


def decode_reference(inputs, rows, kv_lens, requests=None, q_lens=None):
    """Reference (O, LSE, route) of the chosen requests, in packed order."""
    index_q, index_k, q, k, v = inputs
    q_lens = [1] * len(kv_lens) if q_lens is None else list(q_lens)
    results = []
    for request in range(len(kv_lens)) if requests is None else requests:
        ids, length = rows[request].long(), kv_lens[request]
        rk, rv = (x[ids].to(torch.bfloat16).permute(0, 2, 1, 3).reshape(-1, 4, 128)[:length] for x in (k, v))
        rik = index_k[ids].reshape(-1, 1, 128)[:length]
        tokens = slice(sum(q_lens[:request]), sum(q_lens[:request + 1]))
        results.append(reference_pipeline(index_q[tokens], rik, q[tokens], rk, rv))
    return tuple(torch.cat([r[i] for r in results], dim=1 if i == 2 else 0) for i in range(3))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DecodeSteps(BackendCaseMixin, unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("SM100/SM103 required")

    def run_step(self, kv_lens, **options):
        inputs, kwargs, rows = decode_fixture(kv_lens, **options)
        pipeline = prepare_cutedsl_pipeline(*inputs, **kwargs)
        return pipeline, inputs, kwargs, rows

    def subset(self, actual, requests):
        out, lse, route = actual
        return out[requests], lse[requests], route[:, requests]

    def test_ragged_packed_step_matches_the_reference(self):
        for index_dtype in (torch.bfloat16, torch.float16, E4M3):
            with self.subTest(index_dtype=index_dtype):
                pipeline, inputs, _, rows = self.run_step(RAGGED, index_dtype=index_dtype)
                state = pipeline.plan._state
                self.assertTrue(state.attention.query_outer)
                self.assertTrue(state.index.heads_in_rows_bound and state.index.route_published)
                self.assertIsNone(state.attention.dynamic_state)
                self.compare(pipeline.run(), decode_reference(inputs, rows, RAGGED))

    def test_32_requests_of_512k_and_1m_tokens_run(self):
        # The KV-outer path cannot prepare these steps: its CSR histogram is sized by the 131072 / 262144 KV
        # blocks of the batch.
        for kv in (524288, 1048576):
            with self.subTest(kv=kv):
                kv_lens = (kv,) * 32
                pipeline, inputs, _, rows = self.run_step(kv_lens)
                checked = [0, 13, 31]
                self.compare(self.subset(pipeline.run(), checked),
                             decode_reference(inputs, rows, kv_lens, checked))

    def test_multi_token_requests_beyond_one_csr_row_range(self):
        # Not a decode-like step: the KV-outer path builds its CSR over the 131072 KV blocks of the batch,
        # more than one histogram CTA's shared memory holds; the builder splits the rows into ranges.
        kv_lens, q_lens = (524288,) * 32, (2,) * 32
        inputs, kwargs, rows = decode_fixture(kv_lens, q_lens=q_lens)
        pipeline = prepare_cutedsl_pipeline(*inputs, **kwargs)
        self.assertFalse(pipeline.plan._state.attention.query_outer)
        out, lse, route = pipeline.run()
        checked = [0, 17, 31]
        tokens = [t for request in checked for t in (2 * request, 2 * request + 1)]
        self.compare((out[tokens], lse[tokens], route[:, tokens]),
                     decode_reference(inputs, rows, kv_lens, checked, q_lens))

    def test_batch_one_and_fixed_only_steps(self):
        # One request takes the same heads-in-rows launch as a batch; up to 2048 KV tokens there is no
        # Dynamic stage.
        for kv in (2048, 2049, 4096, 70000):
            with self.subTest(kv=kv):
                pipeline, inputs, _, rows = self.run_step((kv,))
                state = pipeline.plan._state
                self.assertEqual(state.attention.query_outer, kv > 2048)
                self.assertEqual(state.index.heads_in_rows_bound, kv > 2048)
                self.compare(pipeline.run(), decode_reference(inputs, rows, (kv,)))
        pipeline, inputs, _, rows = self.run_step((1024, 2048, 300))
        self.assertFalse(pipeline.plan._state.attention.query_outer)
        self.compare(pipeline.run(), decode_reference(inputs, rows, (1024, 2048, 300)))

    def test_fp8_kv_cache(self):
        kv_lens = (4096, 33000, 8192)
        pipeline, inputs, _, rows = self.run_step(kv_lens, kv_dtype=E4M3)
        self.assertTrue(pipeline.plan._state.attention.query_outer)
        self.compare(pipeline.run(), decode_reference(inputs, rows, kv_lens))

    def test_routed_blocks_require_packed_gqa_and_one_token_tiles(self):
        # The route is per KV head; without packed GQA the pass is scheduled by query head. A tile's route row
        # is its first token: a tile of two tokens would give the second one the first one's blocks.
        from mbsa.backends.optimized.bsa.cute_dsl.sparse.bsa_attn_sm100_blk128 import bsa_attn_sm100_blk128_fwd
        inputs, kwargs, _ = decode_fixture((4096, 4096))
        _, _, q, k, v = inputs
        route = torch.full((4, 2, 16), -1, dtype=torch.int32, device="cuda")
        tiles = torch.tensor([[0, 0, 0, 1, 0, 4096], [1, 0, 1, 2, 4096, 8192]], dtype=torch.int32, device="cuda")
        options = dict(q2k_block_index=route, block_sparse_num=16, causal=True, implicit_fixed_support=True,
                       implicit_fixed_route_blocks=16, page_table=kwargs["page_table"], return_lse=True)
        with self.assertRaisesRegex(ValueError, "packed GQA"):
            bsa_attn_sm100_blk128_fwd(q, k, v, packed_tiles=tiles, pack_gqa=False, **options)
        two_tokens = torch.tensor([[0, 0, 0, 2, 0, 4096]], dtype=torch.int32, device="cuda")
        with self.assertRaisesRegex(ValueError, "one packed tile per query token"):
            bsa_attn_sm100_blk128_fwd(q, k, v, packed_tiles=two_tokens, **options)

    def test_query_outer_plans_have_no_stages(self):
        # The plan holds no Fixed / Dynamic state: its stage entry points say so instead of failing on None.
        pipeline, _, _, _ = self.run_step((4096, 8192))
        attention = pipeline.plan._state.attention
        self.assertTrue(attention.query_outer)
        for stage in (attention.run_fixed, attention.run_dynamic, attention.run_dynamic_msa,
                      attention.run_dynamic_combine, attention.run_merge):
            with self.subTest(stage=stage.__name__), self.assertRaisesRegex(RuntimeError, "query-outer"):
                stage()
        staged, _, _, _ = self.run_step((4096, 8192), q_lens=(2, 1))
        with self.assertRaisesRegex(RuntimeError, "query_outer"):
            staged.plan._state.attention.run_query_outer()

    def test_mxfp6_index(self):
        # The MX scorer keeps its own launch; its merged candidate plane is copied into the route.
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("the MXFP6 Index scorer requires SM103")
        from mbsa.index_formats import (dequantize_index_k_mxfp6, dequantize_index_q_mxfp8,
                                        quantize_index_k_mxfp6, quantize_q_mxfp8)
        kv_lens = (300, 2304, 4096, 33000, 8192)
        inputs, kwargs, rows = decode_fixture(kv_lens)
        index_q, index_k, q, k, v = inputs
        codes, scales = quantize_q_mxfp8(index_q)
        mx_q, mx_k = codes[:, :4], quantize_index_k_mxfp6(index_k)
        pipeline = prepare_cutedsl_pipeline(mx_q, mx_k, q, k, v, index_q_scales=scales, **kwargs)
        state = pipeline.plan._state
        self.assertTrue(state.attention.query_outer)
        self.assertFalse(state.index.route_published)
        decoded = (dequantize_index_q_mxfp8(mx_q, scales),
                   dequantize_index_k_mxfp6(mx_k).view(-1, 1, 128, 128), q, k, v)
        self.compare(pipeline.run(), decode_reference(decoded, rows, kv_lens))

    def test_heads_in_rows_is_bit_identical_to_one_cta_per_head(self):
        # The default dispatch and the baseline selection (FP16 Index always binds it). Serial merges (up to 16
        # partitions per request) and tree merges: the ragged step gives its longest request more than 16 of
        # its work rows, two requests of 1M tokens 63 each.
        cases = [(AUTO_INDEX_DISPATCH, dtype, kv_lens, tree)
                 for dtype in (torch.bfloat16, E4M3)
                 for kv_lens, tree in ((RAGGED, True), ((8192,) * 32, False), ((262144,) * 32, False),
                                       ((262144,) * 8, False), ((1048576, 1048576), True))]
        cases += [(BASELINE_INDEX_DISPATCH, dtype, RAGGED, True) for dtype in (torch.bfloat16, torch.float16, E4M3)]
        for base, index_dtype, kv_lens, tree in cases:
            with self.subTest(index_dtype=index_dtype, kv=kv_lens[0], requests=len(kv_lens),
                              baseline=base is BASELINE_INDEX_DISPATCH):
                inputs, kwargs, _ = decode_fixture(kv_lens, index_dtype=index_dtype)
                results = []
                for heads_in_rows in (True, False):
                    with index_dispatch_override(dataclasses.replace(base, heads_in_rows=heads_in_rows)):
                        pipeline = prepare_cutedsl_pipeline(*inputs, **kwargs)
                        index = pipeline.plan._state.index
                        self.assertEqual(index.heads_in_rows_bound, heads_in_rows)
                        if heads_in_rows:
                            self.assertEqual(
                                index.cutedsl_packed_router_partitions > SERIAL_MERGE_MAX_PARTITIONS, tree)
                        results.append(tuple(t.clone() for t in pipeline.run()))
                for name, a, b in zip(("out", "lse", "route"), *results):
                    self.assertTrue(torch.equal(a, b), name)

    def test_mega_graph_and_rebind(self):
        kv_lens = (4096, 33000, 131072, 2304)
        pipeline, inputs, kwargs, rows = self.run_step(kv_lens)
        self.compare(pipeline.run(), decode_reference(inputs, rows, kv_lens))
        # New Index and attention queries, Index K and output; same requests, maps and K/V cache.
        other, _, _ = decode_fixture(kv_lens, seed=29)
        layer = other[:3] + inputs[3:]
        new_out = torch.empty_like(inputs[2])
        pipeline.rebind(*layer, index_page_indices=kwargs["index_page_indices"],
                        page_table=kwargs["page_table"], out=new_out)
        out, lse, route = pipeline.run()
        self.assertIs(out, new_out)
        self.compare((out, lse, route), decode_reference(layer, rows, kv_lens))
        pipeline.capture_mega_graph()
        layer[2].mul_(0.875)
        self.compare(pipeline.run_mega(), decode_reference(layer, rows, kv_lens))

    def test_serving_step(self):
        from mbsa.serving import Prefill
        kv_lens = (4096, 33000, 131072, 2304, 8192)
        for dispatch in (None, baseline_index_dispatch):
            with self.subTest(baseline=dispatch is not None):
                inputs, kwargs, rows = decode_fixture(kv_lens)
                index_q, index_k, q, k, v = inputs
                step = Prefill(torch.device("cuda"), backend="optimized").stage(
                    [(1, n) for n in kv_lens], kwargs["page_table"], kwargs["index_page_indices"])
                out = torch.empty_like(q)
                if dispatch is None:
                    step.run(q, k, v, index_q=index_q, index_k_cache=index_k, out=out)
                else:
                    with dispatch():
                        step.run(q, k, v, index_q=index_q, index_k_cache=index_k, out=out)
                expected = decode_reference(inputs, rows, kv_lens)
                torch.testing.assert_close(out, expected[0], rtol=0.03, atol=0.03)


if __name__ == "__main__":
    unittest.main()
