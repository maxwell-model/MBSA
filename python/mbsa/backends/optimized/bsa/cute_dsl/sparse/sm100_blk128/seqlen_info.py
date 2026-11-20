# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

import cutlass.cute as cute
from cutlass import Int32, const_expr


@dataclass(frozen=True)
class SeqlenInfoQK:
    seqlen_q: Int32
    seqlen_k: Int32
    offset_q: Int32
    offset_k: Int32
    m_block: Int32
    batch_idx: Int32
    tensor_batch_idx: Int32

    @staticmethod
    @cute.jit
    def create(
        m_block: Int32,
        batch_idx: Int32,
        seqlen_q_static: Int32,
        seqlen_k_static: Int32,
        packed_tiles: Optional[cute.Tensor] = None,
    ):
        if const_expr(packed_tiles is not None):
            # Each scheduler row identifies one request-local Q tile.
            request = packed_tiles[m_block, 0]
            local_block = packed_tiles[m_block, 1]
            q_begin = packed_tiles[m_block, 2]
            q_end = packed_tiles[m_block, 3]
            k_begin = packed_tiles[m_block, 4]
            k_end = packed_tiles[m_block, 5]
            return SeqlenInfoQK(q_end - q_begin, k_end - k_begin,
                               q_begin, k_begin, local_block, request, Int32(0))
        else:
            return SeqlenInfoQK(seqlen_q_static, seqlen_k_static,
                               Int32(0), Int32(0), m_block, batch_idx, batch_idx)
