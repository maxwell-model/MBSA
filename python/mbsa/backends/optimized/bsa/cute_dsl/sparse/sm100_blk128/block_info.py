# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from dataclasses import dataclass
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr


@dataclass(frozen=True)
class BlockInfo:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int]
    q_token_offset: Int32
    q_seqlen: Int32
    implicit_fixed_support: cutlass.Constexpr[bool]
    # Implicit fixed blocks followed by up to `route_blocks` routed blocks of the tile's query token:
    # block_index[head, route_row, :] holds its block IDs, valid IDs first and negative padding after
    # them. The tile holds one query token, so the tile's fixed union is that token's exact fixed set.
    route_blocks: cutlass.Constexpr[int] = 0
    route_row: Int32 = 0

    @cute.jit
    def get_implicit_min_valid_blocks(self, m_block: Int32) -> Int32:
        local_q_block = m_block // self.qhead_per_kvhead_packgqa
        tile_start = self.q_token_offset + local_q_block * self.tile_m
        return tile_start // self.tile_n + 1

    @cute.jit
    def get_implicit_max_valid_blocks(self, m_block: Int32) -> Int32:
        local_q_block = m_block // self.qhead_per_kvhead_packgqa
        tile_last_local = cutlass.min(
            (local_q_block + 1) * self.tile_m, self.q_seqlen
        ) - 1
        tile_last = self.q_token_offset + tile_last_local
        return tile_last // self.tile_n + 1

    @cute.jit
    def get_implicit_fixed_count(self, m_block: Int32) -> Int32:
        min_valid = self.get_implicit_min_valid_blocks(m_block)
        max_valid = self.get_implicit_max_valid_blocks(m_block)
        prefix_count = cutlass.min(max_valid, Int32(8))
        local_begin = cutlass.max(min_valid - 8, Int32(0))
        return (
            max_valid
            if local_begin <= prefix_count
            else prefix_count + max_valid - local_begin
        )

    @cute.jit
    def get_route_count(self, block_index: cute.Tensor, head_idx: Int32) -> Int32:
        """Number of routed blocks of this tile's query token (its non-negative route entries)."""
        count = Int32(0)
        for slot in cutlass.range_constexpr(self.route_blocks):
            count += Int32(1) if block_index[head_idx, self.route_row, slot] >= Int32(0) else Int32(0)
        return count

    @cute.jit
    def get_n_block_idx(
        self,
        block_index: cute.Tensor,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        i: Int32,
        max_i: Optional[Int32] = None,
    ) -> Int32:
        if const_expr(self.implicit_fixed_support):
            raw_count = self.get_implicit_fixed_count(m_block)
            if const_expr(self.route_blocks):
                # max_i clamps a phantom index to the last real (fixed or routed) block.
                i = cutlass.min(i, max_i)
            idx = cutlass.min(i, raw_count - 1)
            min_valid = self.get_implicit_min_valid_blocks(m_block)
            max_valid = self.get_implicit_max_valid_blocks(m_block)
            prefix_count = cutlass.min(max_valid, Int32(8))
            local_begin = cutlass.max(min_valid - 8, Int32(0))
            fixed_block = (
                idx
                if local_begin <= prefix_count or idx < prefix_count
                else local_begin + idx - prefix_count
            )
            if const_expr(self.route_blocks):
                slot = cutlass.min(cutlass.max(i - raw_count, Int32(0)), Int32(self.route_blocks - 1))
                routed_block = block_index[head_idx, self.route_row, slot]
                return routed_block if i >= raw_count else fixed_block
            return fixed_block
        idx = cutlass.min(i, max_i) if const_expr(max_i is not None) else i
        return block_index[batch_idx, head_idx, m_block, idx]
