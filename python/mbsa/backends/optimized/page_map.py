# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct Index page-map views; no map staging or refresh copies."""


def page_operand_view(rows, alignment):
    """Validate adjacent active rows and return a no-copy page-map view.

    Args:
        rows: Nonempty sequence of contiguous active tensor rows in request order. Every row must
            share one backing storage and immediately follow its predecessor without gaps.
        alignment: Required byte alignment of the flattened operand's base, normally 16 for
            unpacked/MX or 4 for packed floating Index. Internal row offsets are scalar indices.

    Returns:
        1-D contiguous view spanning all active row elements, aliasing caller storage. Raises
        ValueError before publication for gaps, strides, distinct allocations or unaligned base.
    """
    if not rows:
        raise ValueError("direct Index page map requires at least one active row")
    first = rows[0]
    if first.data_ptr() % alignment:
        raise ValueError(f"direct Index page map requires a {alignment}-byte aligned base")
    offset = 0
    for row in rows:
        if (not row.is_contiguous()
                or row.untyped_storage().data_ptr() != first.untyped_storage().data_ptr()
                or row.data_ptr() != first.data_ptr() + offset * first.element_size()):
            raise ValueError(
                f"direct Index page map requires adjacent contiguous active rows "
                f"in one storage with a {alignment}-byte aligned base; "
                "pass compact active page IDs without gaps or strides"
            )
        offset += row.numel()
    return first.as_strided((offset,), (1,))


def bind_page_operands(plan, rows):
    """Validate a replacement map before changing a staged Index binding.

    Args:
        plan: Staged Index plan with either unpacked or packed map storage. Plans without an active
            map are unchanged.
        rows: Active request rows satisfying page_operand_view; their storage remains caller-owned
            and strongly referenced by the staged state.

    Returns:
        None; the selected map attribute is replaced by a direct view only after validation.
    """
    name = ('cutedsl_index_pages' if plan.cutedsl_index_pages is not None
            else 'cutedsl_packed_pages' if plan.cutedsl_packed_pages is not None else None)
    if name is None:
        return
    alignment = 16 if name == 'cutedsl_index_pages' or plan.mxfp6_router is not None else 4
    view = page_operand_view(rows, alignment)
    setattr(plan, name, view)
