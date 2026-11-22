# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Decode-like steps whose KV lengths change from step to step, and small batches.

A plan prepared with kv_lengths=KvLengths(...) takes its request_lens as KV capacities and reads the lengths
KvLengths.set() publishes: one prepared plan, and one captured CUDA graph, serve every decode step of the
requests. The Index launch is a list of work rows shared among the requests in proportion to their KV blocks
(index_policy), so the result does not depend on the capacity. Batches that leave SMs without an attention
tile split every tile's blocks among several CTAs and combine their states.

Covers bit-identity with a plan prepared for the lengths themselves (BF16 / FP16 / E4M3 Index; ragged, tiny and
page-crossing lengths; eager, CUDA graph, a second plan of the step, a rebound layer), the reference
comparison of split tiles, the row-sharing policy and the argument errors.
"""
import dataclasses
import unittest

import torch

from mbsa import prepare_cutedsl_pipeline
from mbsa.serving import KvLengths
from mbsa.backends.optimized.index_policy import (HEADS_IN_ROWS_CTAS_PER_SM, HEADS_IN_ROWS_MERGE_PARTITIONS,
                                                  HEADS_IN_ROWS_PARTITION_MIN_BLOCKS,
                                                  heads_in_rows_partition_counts, heads_in_rows_work_rows,
                                                  index_dispatch_override, resolve_index_dispatch)
from mbsa.backends.optimized.kernels.attention import QUERY_OUTER_MAX_SPLITS, query_outer_splits
from mbsa.backends.optimized.kv_lengths import index_work_rows
from tests.common import BackendCaseMixin
from tests.regression.test_decode_steps import decode_fixture, decode_reference

E4M3 = torch.float8_e4m3fn
CAPACITY = 40000
# Requests without Dynamic candidates, partial tail pages, a page crossed token by token, and the capacity.
STEPS = (
    (300, 1, 2049, 4096, 33000, 40000, 127, 129, 8191, 2048, 20000, 39999, 128),
    (40000,) * 13,
    (4095,) * 13, (4096,) * 13, (4097,) * 13,
    (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
)


def step_kwargs(rows, kv_lens):
    """Packed preparation arguments of one-token requests over the first pages of each request's row."""
    counts = [(n + 127) // 128 for n in kv_lens]
    table = torch.zeros((len(kv_lens), max(counts)), device="cuda", dtype=torch.int32)
    for request, count in enumerate(counts):
        table[request, :count].copy_(rows[request][:count])
    prefix = [0]
    for n in kv_lens:
        prefix.append(prefix[-1] + n)
    return dict(index_page_indices=torch.cat([rows[r][:c] for r, c in enumerate(counts)]).contiguous(),
                page_table=table, backend="optimized",
                cu_seqlens_q=torch.arange(len(kv_lens) + 1, device="cuda", dtype=torch.int32),
                cu_seqlens_k=torch.tensor(prefix, device="cuda", dtype=torch.int32),
                request_lens=[(1, n) for n in kv_lens])


def snapshot(result):
    return tuple(x.clone() for x in result)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DecodeLengths(BackendCaseMixin, unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("SM100/SM103 required")

    def capacity_step(self, batch=13, capacity=CAPACITY, **options):
        inputs, _, rows = decode_fixture((capacity,) * batch, **options)
        lengths = KvLengths("cuda")
        pipeline = prepare_cutedsl_pipeline(*inputs, **step_kwargs(rows, (capacity,) * batch), kv_lengths=lengths)
        return pipeline, lengths, inputs, rows

    def fresh(self, inputs, rows, kv_lens):
        pipeline = prepare_cutedsl_pipeline(*inputs, **step_kwargs(rows, kv_lens))
        result = snapshot(pipeline.run())
        torch.cuda.synchronize()
        return result

    def assert_identical(self, actual, expected):
        for name, a, b in zip(("out", "lse", "route"), actual, expected):
            self.assertTrue(torch.equal(a, b), name)

    def test_lengths_follow_set_like_a_plan_prepared_for_them(self):
        for index_dtype in (torch.bfloat16, torch.float16, E4M3):
            with self.subTest(index_dtype=index_dtype):
                pipeline, lengths, inputs, rows = self.capacity_step(index_dtype=index_dtype)
                state = pipeline.plan._state
                self.assertTrue(state.index.heads_in_rows_bound and state.index.even_split_bound)
                self.assertEqual(lengths.capacities, (CAPACITY,) * 13)
                for kv_lens in STEPS:
                    lengths.set(kv_lens)
                    self.assertEqual(lengths.lengths, kv_lens)
                    self.assert_identical(pipeline.run(), self.fresh(inputs, rows, kv_lens))
                self.compare(pipeline.run(), decode_reference(inputs, rows, STEPS[-1]))

    def test_one_captured_graph_serves_every_step(self):
        pipeline, lengths, inputs, rows = self.capacity_step()
        pipeline.capture_mega_graph()
        for kv_lens in STEPS + (STEPS[0],):
            lengths.set(kv_lens)
            self.assert_identical(pipeline.run_mega(), self.fresh(inputs, rows, kv_lens))
        self.compare(pipeline.run_mega(), decode_reference(inputs, rows, STEPS[0]))

    def test_the_plans_of_a_step_share_its_lengths(self):
        pipeline, lengths, inputs, rows = self.capacity_step()
        layer = decode_fixture((CAPACITY,) * 13, seed=29)[0]
        operands = (layer[0], inputs[1], layer[2], inputs[3], inputs[4])     # another layer's Index Q and Q
        second = prepare_cutedsl_pipeline(*operands, **step_kwargs(rows, (CAPACITY,) * 13), kv_lengths=lengths)
        for table in (lambda p: p.plan._state.attention.fixed_packed_tiles,
                      lambda p: p.plan._state.index.cutedsl_packed_router_tile_metadata):
            self.assertIs(table(second), table(pipeline))
        lengths.set(STEPS[0])
        self.assert_identical(pipeline.run(), self.fresh(inputs, rows, STEPS[0]))
        self.assert_identical(second.run(), self.fresh(operands, rows, STEPS[0]))

    def test_a_rebound_layer_keeps_the_lengths(self):
        pipeline, lengths, inputs, rows = self.capacity_step()
        lengths.set(STEPS[0])
        layer = decode_fixture((CAPACITY,) * 13, seed=29)[0]
        operands = (layer[0], inputs[1], layer[2], inputs[3], inputs[4])
        kwargs = step_kwargs(rows, (CAPACITY,) * 13)
        out = torch.empty_like(operands[2])
        pipeline.rebind(*operands, index_page_indices=kwargs["index_page_indices"],
                        page_table=kwargs["page_table"], out=out)
        self.assert_identical(pipeline.run(), self.fresh(operands, rows, STEPS[0]))
        lengths.set(STEPS[2])
        self.assert_identical(pipeline.run(), self.fresh(operands, rows, STEPS[2]))

    def test_a_rebind_keeps_the_heads_in_rows_launch(self):
        # The lengths reach the Index through the work rows of that launch: a rebind that would bind another
        # one is rejected and leaves the binding as it was.
        pipeline, lengths, inputs, rows = self.capacity_step()
        lengths.set(STEPS[0])
        expected = snapshot(pipeline.run())
        kwargs = step_kwargs(rows, (CAPACITY,) * 13)
        other = dataclasses.replace(resolve_index_dispatch(torch.device("cuda", 0)), heads_in_rows=False)
        with index_dispatch_override(other):
            with self.assertRaisesRegex(ValueError, "kv_lengths requires"):
                pipeline.rebind(*inputs, index_page_indices=kwargs["index_page_indices"],
                                page_table=kwargs["page_table"], out=torch.empty_like(inputs[2]))
        self.assert_identical(pipeline.run(), expected)

    def test_split_tiles_need_the_scattered_output_layout(self):
        from mbsa.backends.optimized.bsa.cute_dsl.sparse.bsa_attn_sm100_blk128 import (BsaOutputLayout,
                                                                                    bsa_attn_sm100_blk128_fwd)
        inputs, kwargs, _ = decode_fixture((4096, 4096))
        _, _, q, k, v = inputs
        route = torch.full((4, 2, 16), -1, dtype=torch.int32, device="cuda")
        tiles = torch.tensor([[0, 0, 0, 1, 0, 4096], [1, 0, 1, 2, 4096, 8192]], dtype=torch.int32, device="cuda")
        with self.assertRaisesRegex(ValueError, "scattered"):
            bsa_attn_sm100_blk128_fwd(
                q, k, v, q2k_block_index=route, block_sparse_num=16, causal=True, implicit_fixed_support=True,
                implicit_fixed_route_blocks=16, page_table=kwargs["page_table"], return_lse=True, packed_tiles=tiles,
                num_splits=2, output_layout=BsaOutputLayout.NON_SCATTERED)

    def test_small_batches_split_their_tiles(self):
        sm_count = torch.cuda.get_device_properties(0).multi_processor_count
        # Batches that share a split count (1 and 2; 5 and 8; 13 and 16) share the compiled combine. From the
        # second request on the lengths include lists shorter than the split count.
        order = (4096, 300, 1, 2049, 127, 33000, 128, 40000, 129, 8191, 2048, 20000, 39999)
        for batch in (1, 2, 4, 5, 8, 13, 16, 32):
            with self.subTest(batch=batch):
                kv_lens = tuple(order[i % 13] for i in range(batch))
                pipeline, lengths, inputs, rows = self.capacity_step(batch=batch)
                attention = pipeline.plan._state.attention
                self.assertEqual(attention.query_outer_splits, query_outer_splits(batch, sm_count))
                lengths.set(kv_lens)
                result = snapshot(pipeline.run())
                self.compare(result, decode_reference(inputs, rows, kv_lens))
                self.assert_identical(result, self.fresh(inputs, rows, kv_lens))
                pipeline.capture_mega_graph()
                self.assert_identical(pipeline.run_mega(), result)

    def test_warmup_steps_reach_every_split_count(self):
        # serving.warmup runs these steps: whatever the request limit, one of them has each split count a
        # decode step of the limits can take (and the unsplit pass, when a batch is large enough for it).
        from mbsa.serving import Limits, warmup_request_sets
        for max_requests in (1, 2, 3, 9, 10, 15, 18, 19, 24, 40, 64, 200):
            limits = Limits(max_tokens=8192, max_kv_tokens=65536, max_requests=max_requests,
                            max_total_kv_tokens=max_requests * 65536)
            batches = {len(lens) for lens in warmup_request_sets(limits, index_format="bf16")
                       if all(q_len == 1 for q_len, _ in lens)}
            for sm_count in (72, 148):
                with self.subTest(max_requests=max_requests, sm_count=sm_count):
                    reachable = {query_outer_splits(batch, sm_count) for batch in range(1, max_requests + 1)}
                    self.assertEqual({query_outer_splits(batch, sm_count) for batch in batches}, reachable)

    def test_split_count_policy(self):
        self.assertEqual([query_outer_splits(t, 148) for t in (1, 2, 4, 8, 16, 18, 19, 32, 64)],
                         [16, 16, 8, 4, 2, 2, 1, 1, 1])
        for tokens in range(1, 200):
            splits = query_outer_splits(tokens, 148)
            self.assertLessEqual(splits, QUERY_OUTER_MAX_SPLITS)
            self.assertTrue(splits == 1 or tokens * 4 * splits <= 148)

    def test_work_rows_share_the_launch_in_proportion_to_kv_blocks(self):
        for batch in (1, 2, 3, 5, 13, 32, 48, 64, 126, 127, 300):
            rows = heads_in_rows_work_rows(batch, sm_count=148)
            # The fewest whole rows per request that put HEADS_IN_ROWS_CTAS_PER_SM CTAs on every SM.
            self.assertEqual(rows % batch, 0)
            self.assertGreaterEqual(rows, max(batch, HEADS_IN_ROWS_CTAS_PER_SM * 148))
            self.assertTrue(rows == batch or rows - batch < HEADS_IN_ROWS_CTAS_PER_SM * 148)
        self.assertEqual(heads_in_rows_partition_counts((8192,) * 32, 128), [4] * 32)
        skewed = heads_in_rows_partition_counts((1048576,) + (8192,) * 31, 128)
        self.assertEqual((sum(skewed), skewed[0], min(skewed)), (128, 78, 1))
        self.assertEqual(heads_in_rows_partition_counts((300, 129, 128, 1, 4096), 130), [3, 2, 1, 1, 32])
        # One request: 32 partitions until they hold 8 blocks each, then all its rows.
        # A request keeps only partitions that hold a block (33 blocks in partitions of 2: 17 of them).
        self.assertEqual([heads_in_rows_partition_counts((n,), 126)[0]
                          for n in (1, 300, 4097, 8192, 32768, 65536, 131072, 1048576)],
                         [1, 3, 17, 32, 32, 64, 114, 125])
        generator = torch.Generator().manual_seed(7)
        for _ in range(200):
            batch = int(torch.randint(1, 70, (1,), generator=generator))
            kv_lens = [int(n) for n in torch.randint(1, 1048577, (batch,), generator=generator)]
            rows = heads_in_rows_work_rows(batch, sm_count=148)
            counts = heads_in_rows_partition_counts(kv_lens, rows)
            blocks = [(n + 127) // 128 for n in kv_lens]
            limits = [min(b, max(HEADS_IN_ROWS_MERGE_PARTITIONS, -(-b // HEADS_IN_ROWS_PARTITION_MIN_BLOCKS)))
                      for b in blocks]
            self.assertTrue(all(1 <= c <= limit for c, limit in zip(counts, limits)))
            self.assertLessEqual(sum(counts), rows)
            self.assertTrue(all((c - 1) * -(-b // c) < b for c, b in zip(counts, blocks)))    # none is empty
            flat = index_work_rows((4,) * batch, kv_lens, kv_lens, counts, rows)
            table = [flat[i:i + 7] for i in range(0, len(flat), 7)]
            self.assertEqual([row[6] for row in table[:batch]], [0] * batch)
            self.assertEqual([row[5] for row in table[:batch]], counts)
            self.assertEqual(sorted((row[0] // 4, row[6]) for row in table if row[5]),
                             sorted((r, p) for r, c in enumerate(counts) for p in range(c)))

    def test_argument_errors(self):
        inputs, kwargs, rows = decode_fixture((CAPACITY,) * 3)
        packed = step_kwargs(rows, (CAPACITY,) * 3)
        with self.assertRaisesRegex(RuntimeError, "prepare a plan"):
            KvLengths("cuda").set((1, 2, 3))
        with self.assertRaisesRegex(ValueError, "optimized backend"):
            prepare_cutedsl_pipeline(*inputs, **{**packed, "backend": "default"}, kv_lengths=KvLengths("cuda"))
        fp8 = decode_fixture((CAPACITY,) * 3, kv_dtype=E4M3)[0]
        with self.assertRaisesRegex(ValueError, "BF16 Q/K/V"):
            prepare_cutedsl_pipeline(*fp8, **packed, kv_lengths=KvLengths("cuda"))
        with self.assertRaisesRegex(ValueError, "packed one-token requests"):
            single = decode_fixture((CAPACITY,))
            prepare_cutedsl_pipeline(*single[0], **single[1], kv_lengths=KvLengths("cuda"))
        with self.assertRaisesRegex(ValueError, "above 2048"):
            short = decode_fixture((2048,) * 3)
            prepare_cutedsl_pipeline(*short[0], **step_kwargs(short[2], (2048,) * 3), kv_lengths=KvLengths("cuda"))
        lengths = KvLengths("cuda")
        with self.assertRaisesRegex(ValueError, "above 2048"):      # a failed first preparation binds nothing
            prepare_cutedsl_pipeline(*short[0], **step_kwargs(short[2], (2048,) * 3), kv_lengths=lengths)
        self.assertIsNone(lengths.capacities)
        prepare_cutedsl_pipeline(*inputs, **packed, kv_lengths=lengths)
        self.assertEqual(lengths.capacities, (CAPACITY,) * 3)
        with self.assertRaisesRegex(ValueError, "same requests"):
            other = decode_fixture((CAPACITY, CAPACITY, 30000))
            prepare_cutedsl_pipeline(*other[0], **step_kwargs(other[2], (CAPACITY, CAPACITY, 30000)),
                                     kv_lengths=lengths)
        for bad in ((1, 2), (1, 2, 0), (1, 2, CAPACITY + 1)):
            with self.assertRaisesRegex(ValueError, "one length per request"):
                lengths.set(bad)
        self.assertEqual(lengths.lengths, (CAPACITY,) * 3)


if __name__ == "__main__":
    unittest.main()
