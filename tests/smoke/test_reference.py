#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU/GPU contracts and full-query comparison against pinned HF Attention.

The shared config owns every shape and gate. Decode reserves scheduler budget;
only prefill enters the operator. A strict mismatch is a failure, including ties.
"""

from __future__ import annotations
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import torch
from mbsa import reference_pipeline
from mbsa.reference import MaxwellBlockSparseAttentionReference

CUTEDSL_ARCHS = ((9, 0), (10, 0), (10, 3))


class UpstreamReferenceTest(unittest.TestCase):
    def test_raw_negative_scores_select_distinct_group_routes(self):
        # ReLU would turn both eligible remote scores into ties. The upstream
        # oracle selects each group's least negative block independently.
        index_q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        index_k = torch.tensor(
            [
                [-10.0, -10.0],
                [-10.0, -10.0],
                [-4.0, -1.0],
                [-4.0, -1.0],
                [-2.0, -3.0],
                [-2.0, -3.0],
                [100.0, 100.0],
                [100.0, 100.0],
            ]
        )
        q = torch.zeros(1, 4, 2)
        k = torch.zeros(8, 2, 2)
        v = torch.arange(8 * 2 * 2).reshape(8, 2, 2).float()
        output, lse, route = reference_pipeline(
            index_q,
            index_k,
            q,
            k,
            v,
            block_size=2,
            dynamic_topk=1,
            prefix_blocks=1,
            local_blocks=1,
        )
        self.assertEqual(route.tolist(), [[[2]], [[1]]])
        for head in range(4):
            group = head // 2
            selected = [0, 1, 4, 5, 6, 7] if group == 0 else [0, 1, 2, 3, 6, 7]
            torch.testing.assert_close(output[0, head], v[selected, group].mean(0))
        torch.testing.assert_close(lse, torch.full((1, 4), 6.0).log())

    def test_partial_block_cached_chunk_matches_dense_masked_gqa(self):
        torch.manual_seed(11)
        queries, tokens, groups, heads, dim = 3, 11, 2, 4, 4
        index_q = torch.randn(queries, groups, dim)
        index_k = torch.randn(tokens, 1, dim)
        q = torch.randn(queries, heads, dim)
        k = torch.randn(tokens, groups, dim)
        v = torch.randn_like(k)
        # Top-K covers every remote block, so complete support is dense causal.
        output, lse, route = reference_pipeline(
            index_q,
            index_k,
            q,
            k,
            v,
            block_size=3,
            dynamic_topk=8,
            prefix_blocks=1,
            local_blocks=1,
            query_offset=7,
        )
        self.assertEqual(route.shape, (groups, queries, 8))
        self.assertTrue(bool((route[..., 3:] == -1).all()))
        for qi in range(queries):
            for head in range(heads):
                group = head // (heads // groups)
                scores = k[: qi + 8, group] @ q[qi, head] * dim**-0.5
                expected = scores.softmax(-1) @ v[: qi + 8, group]
                torch.testing.assert_close(output[qi, head], expected)
                torch.testing.assert_close(lse[qi, head], scores.logsumexp(-1))

    def test_fixed_prefix_local_support_has_no_duplicates(self):
        oracle = MaxwellBlockSparseAttentionReference(
            index_head_dim=2,
            head_dim=2,
            block_size=4,
            prefix_blocks=2,
            local_blocks=3,
            dynamic_topk=1,
        )
        positions = torch.arange(32).unsqueeze(0)
        remote = torch.full((1, 32, 2, 1), -1)
        route = oracle._build_block_aligned_route(remote, positions)
        for query, position in enumerate(positions[0].tolist()):
            current = position // 4
            expected = set(range(min(2, current + 1)))
            expected.update(range(max(2, current - 2), current + 1))
            for group in range(2):
                actual = [b for b in route[0, query, group].tolist() if b >= 0]
                self.assertEqual(set(actual), expected)
                self.assertEqual(len(actual), len(set(actual)))

    def test_bf16_probability_cast_matches_upstream(self):
        torch.manual_seed(42)
        q = torch.randn(1, 2, 4).bfloat16()
        k = torch.randn(7, 1, 4).bfloat16()
        v = torch.randn_like(k)
        output, lse, _ = reference_pipeline(
            torch.ones(1, 1, 4),
            torch.ones(7, 1, 4),
            q,
            k,
            v,
            block_size=2,
            dynamic_topk=4,
            prefix_blocks=1,
            local_blocks=1,
        )
        for head in range(2):
            scores = k[:, 0].float() @ q[0, head].float() * 0.5
            expected = scores.softmax(-1).bfloat16() @ v[:, 0]
            torch.testing.assert_close(output[0, head], expected, rtol=0, atol=0)
            torch.testing.assert_close(lse[0, head], scores.logsumexp(-1))

    def test_input_contract_rejects_group_specific_index_keys(self):
        with self.assertRaisesRegex(ValueError, "shared index key"):
            reference_pipeline(
                torch.zeros(2, 2, 4),
                torch.zeros(2, 2, 4),
                torch.zeros(2, 4, 4),
                torch.zeros(2, 2, 4),
                torch.zeros(2, 2, 4),
            )
