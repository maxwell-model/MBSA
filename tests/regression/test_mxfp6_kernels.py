#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MXFP6 index kernels against the ``index_formats`` reference.

The Triton writers must produce the reference pages byte for byte. The CUDA
scorers must return the page reduction of the dequantized Q.K, and decode and
prefill must agree bit for bit on the same query. Scores are compared at the
score level: a top-k cannot see a uniform scale error.
"""

from __future__ import annotations
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import torch
from mbsa import index_formats as fmt

HEADS, DIM, PAGE = 4, 128, 128
PAGE_BYTES = fmt.PAGE_BYTES


def _written(seq_lens, width):
    """Pages a request has; nothing past them is written."""
    pages = (seq_lens.long() + PAGE - 1) // PAGE
    cols = torch.arange(width, device=seq_lens.device)
    return (cols[None, :] < pages[:, None]).expand(HEADS, -1, -1)


def _reduce(page, q, visible_tokens, sm_scale, score_sum):
    """One page's per-head score from dequantized operands."""
    s = (fmt.read_page(page) @ q.T) * sm_scale
    visible = (torch.arange(PAGE, device=s.device) < visible_tokens)[:, None]
    s = torch.where(visible, s, 0.0 if score_sum else float("-inf"))
    return s.sum(0) if score_sum else s.amax(0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Mxfp6KernelRegression(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("MXFP6 index kernels require SM100/SM103")
        try:
            import triton  # noqa: F401
        except ImportError:
            self.skipTest("Triton is required")

    def _cache_and_q(self, lens, seed=0):
        """A packed cache of random pages, one block-table row per request."""
        from mbsa.mxfp6.write import write_index_k

        torch.manual_seed(seed)
        max_blocks = max(1, max(-(-n // PAGE) for n in lens))
        pages = len(lens) * max_blocks
        keys = torch.randn(pages * PAGE, DIM, device="cuda")
        cache = torch.empty(pages * PAGE_BYTES, dtype=torch.uint8, device="cuda")
        # The reference page layout byte for byte, as tested below.
        write_index_k(
            keys, torch.arange(pages * PAGE, device="cuda"), cache, rotate=False
        )
        q = torch.randn(len(lens), HEADS, DIM, device="cuda")
        block_table = torch.arange(pages, dtype=torch.int32, device="cuda").view(
            len(lens), max_blocks
        )
        seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
        return cache, q, block_table, seq_lens

    # ---- Triton writers ----------------------------------------------------

    def test_the_scatter_writes_reference_pages_byte_for_byte(self):
        """Tokens arrive in scattered slot order; the pages must come out as
        the reference lays them out, permuted scale words included."""
        from mbsa.mxfp6.write import write_index_k

        for rotate in (True, False):
            with self.subTest(rotate=rotate):
                torch.manual_seed(0)
                pages = 5
                keys = torch.randn(pages * PAGE, DIM, device="cuda")
                slots = torch.randperm(pages * PAGE, device="cuda").to(torch.int32)
                cache = torch.zeros(
                    pages * PAGE_BYTES, dtype=torch.uint8, device="cuda"
                )
                write_index_k(keys[slots.long()], slots, cache, rotate=rotate)
                want = fmt.quantize_index_k_mxfp6(
                    keys.view(pages, 1, PAGE, DIM), rotate_first=rotate
                )
                self.assertTrue(torch.equal(cache, want.view(-1)))

    def test_negative_slots_write_nothing(self):
        from mbsa.mxfp6.write import write_index_k

        cache = torch.full((PAGE_BYTES,), 0xAB, dtype=torch.uint8, device="cuda")
        write_index_k(
            torch.randn(4, DIM, device="cuda"),
            torch.full((4,), -1, dtype=torch.int32, device="cuda"),
            cache,
        )
        self.assertTrue(bool((cache == 0xAB).all()))

    def test_the_q_quantizer_matches_the_reference_at_extreme_block_scales(self):
        """Padding rows included, and blocks scaled by 2^-60..2^60, all-zero
        blocks and one past E4M3's range: none of them may drift or turn NaN."""
        from mbsa.mxfp6.write import quantize_q_mxfp8_fused

        torch.manual_seed(4)
        q = torch.randn(64, HEADS, 4, 32, device="cuda")
        q = q * 2.0 ** torch.randint(-60, 61, (64, HEADS, 4, 1), device="cuda").float()
        q[::7, :, 1] = 0.0
        q[3, 2, 0, 5] = 3.0e4
        q = q.flatten(-2).to(torch.bfloat16)
        # Dirty the allocator so unwritten padding rows cannot pass by luck.
        dirt = torch.full((64, 8, DIM), 0xFF, dtype=torch.uint8, device="cuda")
        del dirt
        codes, scales = quantize_q_mxfp8_fused(q)
        want_codes, want_scales = fmt.quantize_q_mxfp8(q)
        self.assertTrue(torch.equal(codes, want_codes))
        self.assertTrue(torch.equal(scales, want_scales))

    # ---- CUDA scorers ------------------------------------------------------

    def test_decode_and_prefill_kernels_score_bit_for_bit_alike(self):
        """A one-token-per-request prefill is a decode step: same bytes in, same
        reduction order, so anything but equality is a bug."""
        from mbsa.mxfp6 import decode_score, prefill_score

        cases = {
            "ragged": [900, 1500, 257, 4096, 130],
            "odd-full-row": [896] * 16,
            "padding-rows": [900, 0, 1500, 0],
            "b1-one-token": [1],
            "b1-183k": [183000],
        }
        for name, lens in cases.items():
            for score_sum in (False, True):
                with self.subTest(case=name, score_sum=score_sum):
                    cache, q, block_table, seq_lens = self._cache_and_q(lens)
                    q_bytes, sf_q = fmt.quantize_q_mxfp8(q)
                    kw = dict(
                        init_blocks=2, local_blocks=2, score_sum=score_sum,
                        sm_scale=0.3,
                    )  # fmt: skip
                    dg = decode_score(
                        q_bytes, sf_q, cache, block_table, seq_lens, **kw
                    )
                    hand = prefill_score(
                        q_bytes,
                        sf_q,
                        cache,
                        block_table,
                        torch.arange(len(lens) + 1, dtype=torch.int32, device="cuda"),
                        seq_lens - 1,
                        max_seq_len=max(lens),
                        **kw,
                    )
                    width = min(dg.shape[2], hand.shape[2])
                    m = _written(seq_lens, width)
                    self.assertTrue(
                        torch.equal(dg[..., :width][m], hand[..., :width][m])
                    )

    def test_decode_scores_are_the_page_reduction_of_q_dot_k(self):
        """``sm_scale`` is not 1/sqrt(d) on purpose: a kernel that ignored it
        would still rank pages correctly."""
        from mbsa.mxfp6 import decode_score

        lens, sm_scale = [900, 1500, 257, 130], 0.3
        cache, q, block_table, seq_lens = self._cache_and_q(lens)
        q_bytes, sf_q = fmt.quantize_q_mxfp8(q)
        qd = fmt.dequantize_index_q_mxfp8(q_bytes[:, :HEADS], sf_q)
        pages = cache.view(-1, PAGE_BYTES)
        for score_sum in (False, True):
            with self.subTest(score_sum=score_sum):
                got = decode_score(
                    q_bytes,
                    sf_q,
                    cache,
                    block_table,
                    seq_lens,
                    init_blocks=0,
                    local_blocks=0,
                    score_sum=score_sum,
                    sm_scale=sm_scale,
                )
                for r, length in enumerate(lens):
                    for b in range(-(-length // PAGE)):
                        want = _reduce(
                            pages[int(block_table[r, b])],
                            qd[r],
                            length - b * PAGE,
                            sm_scale,
                            score_sum,
                        )
                        torch.testing.assert_close(
                            got[:, r, b], want, rtol=1e-4, atol=1e-4
                        )

    def test_prefill_scores_are_the_causal_page_reduction_of_q_dot_k(self):
        """Each prefill token sees its own causal bound: context plus its
        position, so the last page it scores is partial and moves per token."""
        from mbsa.mxfp6 import prefill_score

        q_lens, ctx_lens, sm_scale = [3, 130, 200], [700, 0, 128], 0.3
        seq = [c + n for c, n in zip(ctx_lens, q_lens)]
        cache, _, block_table, _ = self._cache_and_q(seq)
        total_q = sum(q_lens)
        q = torch.randn(total_q, HEADS, DIM, device="cuda")
        q_bytes, sf_q = fmt.quantize_q_mxfp8(q)
        qd = fmt.dequantize_index_q_mxfp8(q_bytes[:, :HEADS], sf_q)
        cu = torch.tensor(
            [0, *torch.tensor(q_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )
        ctx = torch.tensor(ctx_lens, dtype=torch.int32, device="cuda")
        pages = cache.view(-1, PAGE_BYTES)
        for score_sum in (False, True):
            with self.subTest(score_sum=score_sum):
                got = prefill_score(
                    q_bytes,
                    sf_q,
                    cache,
                    block_table,
                    cu,
                    ctx,
                    max_seq_len=max(seq),
                    score_sum=score_sum,
                    sm_scale=sm_scale,
                )
                t = 0
                for r, (n, c) in enumerate(zip(q_lens, ctx_lens)):
                    for pos in range(n):
                        kv_len = c + pos + 1
                        for b in range(-(-kv_len // PAGE)):
                            want = _reduce(
                                pages[int(block_table[r, b])],
                                qd[t],
                                kv_len - b * PAGE,
                                sm_scale,
                                score_sum,
                            )
                            torch.testing.assert_close(
                                got[:, t, b], want, rtol=1e-4, atol=1e-4
                            )
                        t += 1

    def test_forced_pages_carry_the_sentinels_and_local_wins_over_init(self):
        """900 tokens are 8 pages: with init=3 and local=3, pages 0-2 are init,
        5-7 are local and 3-4 are scored. 257 tokens are 3 pages: with init=2
        and local=2, page 1 is both, and must come out local."""
        from mbsa.mxfp6 import decode_score

        cache, q, block_table, seq_lens = self._cache_and_q([900, 257])
        q_bytes, sf_q = fmt.quantize_q_mxfp8(q)

        def score(blocks):
            return decode_score(
                q_bytes, sf_q, cache, block_table, seq_lens,
                init_blocks=blocks, local_blocks=blocks, score_sum=False,
                sm_scale=1.0,
            )  # fmt: skip

        got = score(3)
        self.assertTrue(bool((got[:, 0, :3] == 1e30).all()))
        self.assertTrue(bool((got[:, 0, 5:8] == 1e29).all()))
        self.assertTrue(bool((got[:, 0, 3:5].abs() < 1e28).all()))
        got = score(2)
        self.assertTrue(bool((got[:, 1, 0] == 1e30).all()))
        self.assertTrue(bool((got[:, 1, 1:3] == 1e29).all()))

    def test_a_shared_schedule_scores_what_a_fresh_one_does(self):
        """Every layer of a step reuses one schedule; it must be the one the
        scorer would have built, and be built once per owner."""
        from types import SimpleNamespace

        from mbsa.mxfp6 import decode_score, shared_schedule

        lens = [131072, 8192, 1, 0, 65536, 20 * PAGE]
        cache, q, block_table, seq_lens = self._cache_and_q(lens)
        q_bytes, sf_q = fmt.quantize_q_mxfp8(q)
        step = SimpleNamespace()
        schedule = shared_schedule(seq_lens, 1, step)
        self.assertIs(shared_schedule(seq_lens, 1, step), schedule)
        kw = dict(init_blocks=8, local_blocks=8, score_sum=False, sm_scale=1.0)
        fresh = decode_score(q_bytes, sf_q, cache, block_table, seq_lens, **kw)
        shared = decode_score(
            q_bytes, sf_q, cache, block_table, seq_lens, schedule=schedule, **kw
        )
        m = _written(seq_lens, block_table.shape[1])
        self.assertTrue(torch.equal(fresh[m], shared[m]))

    def test_a_captured_scorer_replays_what_eager_scores(self):
        """Each layer scores a different Q against one schedule, so a replay
        that reused stale descriptors or inputs would not match."""
        from mbsa.mxfp6 import decode_score, schedule_metadata

        lens = [131072, 8192, 1, 0, 65536, 20 * PAGE]
        cache, _, block_table, seq_lens = self._cache_and_q(lens)
        schedule = schedule_metadata(seq_lens)
        qs = [
            fmt.quantize_q_mxfp8(torch.randn(len(lens), HEADS, DIM, device="cuda"))
            for _ in range(6)
        ]

        def layer(i):
            return decode_score(
                *qs[i], cache, block_table, seq_lens, init_blocks=8,
                local_blocks=8, score_sum=False, sm_scale=1.0, schedule=schedule,
            )  # fmt: skip

        m = _written(seq_lens, block_table.shape[1])
        want = [layer(i)[m].clone() for i in range(len(qs))]
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            got = [layer(i) for i in range(len(qs))]
        for _ in range(2):
            graph.replay()
            torch.cuda.synchronize()
            for w, g in zip(want, got):
                self.assertTrue(torch.equal(g[m], w))


if __name__ == "__main__":
    unittest.main()
