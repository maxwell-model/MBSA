# Modified for MBSA: package integration; see root NOTICE.
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
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
            idx = cutlass.min(i, raw_count - 1)
            min_valid = self.get_implicit_min_valid_blocks(m_block)
            max_valid = self.get_implicit_max_valid_blocks(m_block)
            prefix_count = cutlass.min(max_valid, Int32(8))
            local_begin = cutlass.max(min_valid - 8, Int32(0))
            return (
                idx
                if local_begin <= prefix_count or idx < prefix_count
                else local_begin + idx - prefix_count
            )
        idx = cutlass.min(i, max_i) if const_expr(max_i is not None) else i
        return block_index[batch_idx, head_idx, m_block, idx]
