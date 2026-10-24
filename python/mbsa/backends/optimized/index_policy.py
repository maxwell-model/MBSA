# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Index tuning decisions shared by preparation and the launch wrapper."""
import torch


def _validate_index_token_extent(logical_k_tokens, partition_blocks):
    """Reject token counts that overflow the signed 32-bit partition ABI.

    Args:
        logical_k_tokens: Positive logical KV-token count for one request.
        partition_blocks: Supported number of 128-token blocks per partition.

    Returns:
        None. Raises ValueError before device work if partition rounding cannot
        fit in signed 32-bit arithmetic. Tensor allocation and compilation may
        impose smaller practical limits.
    """
    if partition_blocks not in (1024, 2048, 4096, 8192):
        raise ValueError("partition_blocks must be 1024, 2048, 4096, or 8192")
    limit = (1 << 31) - 128 * partition_blocks
    if not 0 < logical_k_tokens <= limit:
        raise ValueError(
            f"logical Index KV extent must be in [1, {limit}] tokens for "
            f"PB{partition_blocks}; partition rounding uses signed int32"
        )


def select_partition_blocks(index_q, index_k, *, batch, contiguous_unpacked,
                            kv_tokens, identity_pages, capability):
    """Select the number of 128-token candidate blocks in an Index partition.

    Args:
        index_q: Index queries [Sq,Hindex,D]; dtype and shape inform selection.
        index_k: Index keys, either dense [Skv,1,D] or contiguous physical-page
            storage [pages,1,128,D]; dtype, extent and layout inform selection.
        batch: Number of requests.
        contiguous_unpacked: Whether one request uses contiguous unpacked operands.
        kv_tokens: Logical KV-token count, including cached context.
        identity_pages: Whether logical and physical Index page IDs are known identical;
            None means that identity is not established.
        capability: GPU compute capability as a (major, minor) pair.

    Returns:
        Candidate blocks per partition: 1024 or 2048. Inputs are unchanged.

    Notes:
        The current 2048-block tuning policy selects matching E4M3FN Index operands
        with Q [8192,4,128], contiguous K [kv_tokens,1,128] or its equivalent
        [kv_tokens/128,1,128,128] view, and KV lengths of at least 8192 in
        8192-token increments. It requires one contiguous unpacked request,
        known identity pages and SM103. Other inputs select 1024 blocks. The
        exact query shape and KV granularity are performance-selection limits,
        not general Index correctness or hardware/ABI requirements. Preparation
        validates storage and request legality; the launch wrapper separately
        checks signed 32-bit partition arithmetic. The public attention API's
        per-request 1M KV limit is unchanged.
    """
    if (batch == 1 and contiguous_unpacked
            and index_q.dtype == index_k.dtype == torch.float8_e4m3fn
            and tuple(index_q.shape) == (8192, 4, 128)
            and (tuple(index_k.shape) == (kv_tokens, 1, 128)
                 or tuple(index_k.shape) == (kv_tokens // 128, 1, 128, 128))
            and index_k.is_contiguous()
            and kv_tokens >= 8192 and kv_tokens % 8192 == 0
            and identity_pages is True and capability == (10, 3)):
        return 2048
    return 1024


def lean192_index_eligible(q, k, *, partition_blocks, logical_k_tokens,
                           use_page_table):
    """Check the current selection policy for the BF16 192-thread Index specialization.

    Args:
        q: Index queries [Sq,Hindex,D].
        k: Canonical dense Index keys [Skv,1,D]. Preparation may supply the
            zero-copy view of contiguous [pages,1,128,D] storage.
        partition_blocks: Number of 128-token candidate blocks per Index partition.
        logical_k_tokens: Logical KV-token count, including cached context.
        use_page_table: Whether the launch resolves logical-to-physical Index page IDs
            through a page table rather than using direct addressing.

    Returns:
        Whether these operands satisfy this helper's current selection checks.
        This is not a complete input validator and performs no compilation or device work.

    Notes:
        The current performance-selection policy uses BF16 Index Q with Sq=8192,
        Hindex=4 and D=128, and KV lengths of at least 8192 tokens in 8192-token
        increments. There is no upper-context tuning ceiling. The exact Sq,
        lower KV bound and granularity are selection limits, not general Index
        correctness requirements. The launch wrapper separately rejects extents
        that overflow signed 32-bit partition arithmetic. Allocations and
        compilation must also fit the device; removing a tuning ceiling does
        not guarantee resource feasibility at every size.

        This helper also checks matching BF16 Q/K, K geometry [Skv,1,128],
        partition_blocks=1024, logical_k_tokens equal to the physical K extent,
        and direct addressing (use_page_table=False).

        Production preparation separately requires SM103, a single nonpacked
        contiguous request that needs Dynamic attention, identity Index pages, no
        packed request metadata,
        contiguous Index/attention operands on one device, dense rank-3 attention
        K/V, and BF16 attention Q/K/V and partials. A contiguous rank-4 Index K
        may use its canonical dense view only when physical and logical extents
        agree. Those dispatch prerequisites are checked by prepare_index,
        not by this helper. Preparation also owns input alignment and allocation
        lifetime validation. The public optimized attention backend retains its
        separate per-request 1M KV limit; the standalone Index wrapper can accept
        larger operands within its arithmetic and resource bounds.
    """
    return bool(
        q.dtype == k.dtype == torch.bfloat16
        and tuple(q.shape) == (8192, 4, 128)
        and tuple(k.shape[1:]) == (1, 128)
        and partition_blocks == 1024
        and logical_k_tokens >= 8192
        and logical_k_tokens % 8192 == 0
        and logical_k_tokens == int(k.shape[0]) and not use_page_table
    )
