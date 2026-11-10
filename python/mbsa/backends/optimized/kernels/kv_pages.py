# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert only the pages a prepared step reads from a paged FP8 K/V cache.

The attention kernels compute in BF16. Converting the whole physical cache
costs time and memory proportional to the cache, not to the step. A step's
requests read only their own logical pages, so those pages are gathered into
a compact BF16 copy and the attention kernels use a static compact page table.
FP8 E4M3FN to BF16 is exact for every non-NaN code, so the attention inputs
are bit-identical.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

_PAGE_ELEMENTS = 4 * 128 * 128
_BLOCK = 2048


# The page table is read one entry per program, and a serving block-table slice need not be 16-byte
# aligned: not specializing on its alignment keeps the kernel warmup compiles the only one.
@triton.jit(do_not_specialize_on_alignment=["table"])
def _gather_bf16(k, v, table, slots, ko, vo, ks0, ks1, ks2, vs0, vs1, vs2,
                 BLOCK: tl.constexpr):
    page = tl.program_id(0)
    # 64-bit offsets: a head-major cache view's head stride can exceed 2^31 / 4 elements.
    x = (tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    head, token, dim = x // 16384, x // 128 % 128, x % 128
    physical = tl.load(table + tl.load(slots + page).to(tl.int64)).to(tl.int64)
    a = tl.load(k + physical * ks0 + head * ks1 + token * ks2 + dim)
    b = tl.load(v + physical * vs0 + head * vs1 + token * vs2 + dim)
    offset = page.to(tl.int64) * 65536 + x
    tl.store(ko + offset, a.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16))
    tl.store(vo + offset, b.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16))


@dataclass
class PagedKVConversion:
    """Compact BF16 copy of the logical pages of every request in a step.

    ``table`` maps request ``i``, logical page ``j`` to compact page
    ``start[i] + j``; it depends only on the page counts. ``slots`` holds the
    flattened source page-table position of every compact page, so a refresh
    reads the currently bound page table and K/V cache.
    """

    k: torch.Tensor
    v: torch.Tensor
    table: torch.Tensor
    slots: torch.Tensor
    source_k: torch.Tensor
    source_v: torch.Tensor
    source_table: torch.Tensor

    @classmethod
    def prepare(cls, k, v, page_table, counts):
        from ..index_base import stage_int32

        # Built on the device from the page counts: a long request in a large batch would otherwise make
        # B x max(counts) host integers per step.
        total, longest = sum(counts), max(counts)
        (count,) = stage_int32(k.device, (counts, (len(counts),)))
        start = torch.cumsum(count, 0, dtype=torch.int32) - count
        page = torch.arange(longest, dtype=torch.int32, device=k.device)
        table = torch.where(page < count[:, None], start[:, None] + page, 0).to(torch.int32)
        row = torch.arange(len(counts), dtype=torch.int32, device=k.device)
        slot = torch.arange(total, dtype=torch.int32, device=k.device)
        slots = (torch.repeat_interleave(row * page_table.shape[1] - start, count.long(), output_size=total)
                 + slot).to(torch.int32)
        shape = (total, 4, 128, 128)
        conversion = cls(torch.empty(shape, dtype=torch.bfloat16, device=k.device),
                         torch.empty(shape, dtype=torch.bfloat16, device=k.device),
                         table, slots, k, v, page_table)
        conversion.refresh()
        return conversion

    def rebind(self, k, v, page_table):
        return PagedKVConversion(self.k, self.v, self.table, self.slots, k, v, page_table)

    def refresh(self):
        k, v = (t.view(torch.uint8) for t in (self.source_k, self.source_v))
        _gather_bf16[(self.slots.numel(), _PAGE_ELEMENTS // _BLOCK)](
            k, v, self.source_table, self.slots, self.k, self.v,
            *k.stride()[:3], *v.stride()[:3], _BLOCK)
