# SPDX-License-Identifier: Apache-2.0
"""Stage only Fixed's referenced pages when flat FA4 cannot read the cache."""

import triton
import triton.language as tl
import torch


@triton.jit(do_not_specialize=["length"])
def _gather(k, v, table, logical_pages, ko, vo,
            ks0: tl.constexpr, ks1: tl.constexpr, ks2: tl.constexpr,
            vs0: tl.constexpr, vs1: tl.constexpr, vs2: tl.constexpr,
            length, HEADS: tl.constexpr, BLOCK: tl.constexpr):
    page = tl.program_id(0)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    token, head, dim = x // (HEADS * 128), x // 128 % HEADS, x % 128
    logical = tl.load(logical_pages + page).to(tl.int64)
    physical = tl.load(table + logical).to(tl.int64)
    valid = (token < 128) & (logical * 128 + token < length)
    a = tl.load(k + physical * ks0 + head * ks1 + token * ks2 + dim,
                valid, other=0)
    b = tl.load(v + physical * vs0 + head * vs1 + token * vs2 + dim,
                valid, other=0)
    offset = page.to(tl.int64) * 128 * HEADS * 128 + x
    tl.store(ko + offset, a, token < 128)
    tl.store(vo + offset, b, token < 128)


def gather_pages(k, v, table, logical_pages, ko, vo, length):
    _gather[(logical_pages.numel(), triton.cdiv(128 * k.shape[1] * 128, 512))](
        k, v, table, logical_pages, ko, vo,
        *k.stride()[:3], *v.stride()[:3], length, k.shape[1], 512)


class ContiguousCache:
    """Pad packed contiguous requests into separate pages; paged input stays direct."""

    def __init__(self, k, v, lengths):
        self.source_k, self.source_v = k, v
        self.lengths = lengths
        pages = [(n + 127) // 128 for n in lengths]
        shape = (sum(pages), 128, k.shape[1], 128)
        self.k = torch.zeros(shape, dtype=k.dtype, device=k.device).permute(0, 2, 1, 3)
        self.v = torch.zeros_like(self.k.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        rows, offset = [], 0
        for n in pages:
            rows.append(list(range(offset, offset + n)) + [-1] * (max(pages) - n))
            offset += n
        self.table = torch.tensor(rows, dtype=torch.int32, device=k.device)
        self.refresh()

    def refresh(self):
        offset = page = 0
        for n in self.lengths:
            pages = (n + 127) // 128
            self.k[page:page + pages].permute(0, 2, 1, 3).reshape(-1, 4, 128)[:n].copy_(self.source_k[offset:offset + n])
            self.v[page:page + pages].permute(0, 2, 1, 3).reshape(-1, 4, 128)[:n].copy_(self.source_v[offset:offset + n])
            offset += n
            page += pages

    def rebind(self, k, v):
        self.source_k, self.source_v = k, v
