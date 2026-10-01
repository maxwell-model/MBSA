from __future__ import annotations

import pathlib
import sys
import unittest

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from compass_bsa import (  # noqa: E402
    IndependentBsaV1,
    RouterConfig,
    build_kv_centric_csr,
    hierarchical_exact_topk,
)
from compass_bsa.router import dynamic_candidate_mask  # noqa: E402


class IndependentV1Test(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = RouterConfig(
            block_size=4,
            dynamic_topk=2,
            prefix_blocks=1,
            local_blocks=1,
            partition_blocks=2,
            query_chunk=3,
            score_scale=0.5,
        )
        self.queries = 6
        self.kv_tokens = 32
        self.kv_heads = 2
        self.q_heads = 4
        self.dim = 8
        self.positions = torch.arange(20, 26)
        self.index_q = torch.randn(self.queries, self.kv_heads, self.dim)
        self.index_k = torch.randn(self.kv_tokens, self.kv_heads, self.dim)

    def test_hierarchical_topk_matches_dense_oracle(self) -> None:
        result = hierarchical_exact_topk(
            self.index_q,
            self.index_k,
            query_positions=self.positions,
            config=self.config,
        )
        q = self.index_q.transpose(0, 1)
        k = self.index_k.transpose(0, 1).view(
            self.kv_heads, -1, self.config.block_size, self.dim
        )
        scores = torch.einsum("hqd,hbtd->hqbt", q, k).amax(-1)
        scores *= self.config.score_scale
        blocks = torch.arange(k.shape[1])
        allowed = dynamic_candidate_mask(self.positions, blocks, self.config)
        scores = scores.masked_fill(~allowed.unsqueeze(0), -torch.inf)
        expected_scores, expected_indices = torch.topk(
            scores, self.config.dynamic_topk, dim=-1, sorted=True
        )
        expected_indices = torch.where(
            torch.isfinite(expected_scores),
            expected_indices,
            torch.full_like(expected_indices, -1),
        )
        self.assertTrue(torch.equal(result.indices.long(), expected_indices))
        torch.testing.assert_close(result.scores, expected_scores)

    def test_csr_contains_every_valid_route_once(self) -> None:
        route = hierarchical_exact_topk(
            self.index_q,
            self.index_k,
            query_positions=self.positions,
            config=self.config,
        ).indices
        csr = build_kv_centric_csr(
            route,
            num_kv_blocks=self.kv_tokens // self.config.block_size,
            queries_per_record=2,
        )
        self.assertEqual(int(csr.indptr[-1]), int((route >= 0).sum()))
        if csr.work_records.numel():
            counts = csr.work_records[:, 3]
            ordered = counts[csr.lpt_order.long()]
            self.assertTrue(bool(torch.all(ordered[:-1] >= ordered[1:])))

    def test_dual_path_merge_matches_union_attention(self) -> None:
        q = torch.randn(self.queries, self.q_heads, self.dim)
        k = torch.randn(self.kv_tokens, self.kv_heads, self.dim)
        v = torch.randn_like(k)
        pipeline = IndependentBsaV1(self.config, queries_per_record=2)
        result = pipeline(
            self.index_q,
            self.index_k,
            q,
            k,
            v,
            query_positions=self.positions,
        )
        expected = torch.empty_like(q)
        expected_lse = torch.empty(self.queries, self.q_heads)
        group = self.q_heads // self.kv_heads
        num_blocks = self.kv_tokens // self.config.block_size
        for qi, position_tensor in enumerate(self.positions):
            position = int(position_tensor)
            q_block = position // self.config.block_size
            fixed = set(range(min(self.config.prefix_blocks, q_block + 1)))
            fixed.update(
                range(
                    max(self.config.prefix_blocks, q_block - self.config.local_blocks + 1),
                    min(q_block + 1, num_blocks),
                )
            )
            for qh in range(self.q_heads):
                kvh = qh // group
                dynamic = {
                    int(block)
                    for block in result.router.indices[kvh, qi].tolist()
                    if int(block) >= 0
                }
                blocks = sorted(fixed | dynamic)
                token_ids = []
                for block in blocks:
                    begin = block * self.config.block_size
                    end = min(begin + self.config.block_size, position + 1, self.kv_tokens)
                    token_ids.extend(range(begin, end))
                ids = torch.tensor(token_ids, dtype=torch.long)
                scores = torch.mv(k[:, kvh].index_select(0, ids), q[qi, qh])
                scores *= self.dim ** -0.5
                expected[qi, qh] = torch.mv(
                    v[:, kvh].index_select(0, ids).transpose(0, 1),
                    torch.softmax(scores, dim=0),
                )
                expected_lse[qi, qh] = torch.logsumexp(scores, dim=0)
        self.assertEqual(tuple(result.attention.output.shape), tuple(q.shape))
        torch.testing.assert_close(result.attention.output, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(result.attention.lse, expected_lse, atol=1e-5, rtol=1e-5)
        self.assertTrue(bool(torch.isfinite(result.attention.output).all()))
        self.assertTrue(bool(torch.isfinite(result.attention.lse).all()))


if __name__ == "__main__":
    unittest.main()
