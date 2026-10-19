# SPDX-License-Identifier: Apache-2.0
"""Refresh FA4 physical block IDs and the inverse page map in one launch."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["pages", "masked_count", "full_count"])
def _refresh(
    table,
    masked,
    full,
    physical_masked,
    physical_full,
    inverse,
    pages,
    masked_count,
    full_count,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    physical = tl.load(table + row, row < pages, other=0)
    tl.store(inverse + physical, row, row < pages)
    logical_masked = tl.load(masked + row, row < masked_count, other=0)
    logical_full = tl.load(full + row, row < full_count, other=0)
    mapped_masked = tl.load(table + logical_masked)
    mapped_full = tl.load(table + logical_full)
    tl.store(physical_masked + row, mapped_masked, row < masked_count)
    tl.store(physical_full + row, mapped_full, row < full_count)


def refresh_maps(table, mapping):
    masked, full, physical_masked, physical_full, inverse, _, positions = mapping
    count = max(positions.numel(), masked.numel(), full.numel())
    _refresh[(triton.cdiv(count, 256),)](
        table,
        masked,
        full,
        physical_masked,
        physical_full,
        inverse,
        positions.numel(),
        masked.numel(),
        full.numel(),
        256,
    )
