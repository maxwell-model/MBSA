# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unified aligned-plan ABI for provisional Top-32 block-sparse GQA.

The prefill dispatch consumes this plan for contiguous and paged KV,
including partial final Q and KV blocks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch


class KvLayout(str, Enum):
    CONTIGUOUS_NHD = "contiguous_nhd"
    PAGED_PHTD = "paged_phtd"


class PrefillPolicy(str, Enum):
    ALIGNED_CONTIGUOUS = "aligned_contiguous"
    ALIGNED_PAGED = "aligned_paged"


@dataclass(frozen=True)
class SparseAttentionSpec:
    """Immutable dimensions, precision and sparse-page budget for attention.

    Args:
        block_size: Tokens per sparse page, default 128; validate() accepts 128 only.
        prefix_blocks: Prefix page count, default 8; validate() accepts 8 only.
        dynamic_topk: Selected Dynamic page count, default 16; validate() accepts 16 only.
        local_swa_blocks: Causal local-window page count, default 8; validate() accepts 8 only.
        q_heads: Query heads, default 32; validate() accepts 32 only.
        kv_heads: KV heads, default 4; validate() accepts 4 only.
        head_dim: Elements per head, default 128; validate() accepts 128 only.
        q_dtype: Attention Q dtype, default BF16; validate() accepts BF16 only.
        kv_dtype: Attention K/V dtype, default BF16; validate() accepts BF16 only.
    """
    block_size: int = 128
    prefix_blocks: int = 8
    dynamic_topk: int = 16
    local_swa_blocks: int = 8
    q_heads: int = 32
    kv_heads: int = 4
    head_dim: int = 128
    q_dtype: torch.dtype = torch.bfloat16
    kv_dtype: torch.dtype = torch.bfloat16

    @property
    def group_size(self) -> int:
        """Compute the query-head count sharing one KV head.

        Returns:
            Integer q_heads // kv_heads; validate() enforces the supported head geometry.
        """
        return self.q_heads // self.kv_heads

    @property
    def topk(self) -> int:
        """Compute the total Fixed and Dynamic page budget.

        Returns:
            prefix_blocks + dynamic_topk + local_swa_blocks (32 for the supported specification).
        """
        return self.prefix_blocks + self.dynamic_topk + self.local_swa_blocks

    def validate(self) -> None:
        """Reject unsupported block/head/dtype/prefix/dynamic/local configuration.

        Returns:
            None; raises ValueError for a specification outside the supported BF16
            Hq32/Hkv4/D128/block128 policy. No device work.
        """
        if self.block_size != 128:
            raise ValueError("unified backend is specialized to block size 128")
        if (self.prefix_blocks, self.dynamic_topk, self.local_swa_blocks) != (
            8,
            16,
            8,
        ):
            raise ValueError("unified backend is specialized to prefix8/dynamic16/SWA8")
        if (self.q_heads, self.kv_heads, self.head_dim) != (32, 4, 128):
            raise ValueError("unified backend is specialized to Hq/Hkv/D=32/4/128")
        if self.q_heads % self.kv_heads:
            raise ValueError("q_heads must be divisible by kv_heads")
        if self.q_dtype != torch.bfloat16 or self.kv_dtype != torch.bfloat16:
            raise ValueError("unified backend is BF16-only")


SPEC = SparseAttentionSpec()
SPEC.validate()


def dynamic_partial_capacity(skv: int) -> int:
    """Choose a power-of-two partial-output capacity for the logical KV extent.

    Args:
        skv: Positive logical KV length in tokens, including cached context.

    Returns:
        4,8 or 16 partial slots based on the maximum live Dynamic pages beyond prefix/local support.
        No allocation.
    """
    if skv <= 0:
        raise ValueError("Skv must be positive")
    blocks = (skv + SPEC.block_size - 1) // SPEC.block_size
    live = min(SPEC.dynamic_topk, max(0, blocks - SPEC.prefix_blocks - SPEC.local_swa_blocks))
    return 1 << (max(4, live) - 1).bit_length()


def use_unified_aligned_prefill(*, sq: int, skv: int) -> bool:
    """Check one-request query/KV token bounds, including partial tails.

    Args:
        sq: Query-token count.
        skv: Logical KV-token count, including context.

    Returns:
        True for 1<=sq<=8192 and sq<=skv<=1048576; False otherwise. No device access.
    """
    return (
        1 <= sq <= 8192
        and 1 <= skv <= 1048576
        and sq <= skv
    )


@dataclass
class FwdTilePlan:
    """Retained route and page metadata for the attention host interface.

    Dynamic routes are request-local logical block IDs. Prefix/local support is generated by Fixed
    attention. Arrays are live storage; callers must order mutations and retain allocations through
    queued work.

    Args:
        dynamic_route: Contiguous CUDA int32 [4,total_q,16] Dynamic block IDs, with -1 padding for
            unused slots.
        page_table: CUDA int32 logical-to-physical attention pages [B,max_pages], or empty [B,0]
            for contiguous K/V.
        valid_kv_tokens: CUDA int32 [B] effective logical KV lengths, including cached context.
        q_offset_tokens: KV-minus-Q token offset for one request, or request-ordered tuple of
            offsets for packed requests.
        policy: PrefillPolicy.ALIGNED_CONTIGUOUS or ALIGNED_PAGED; selects dense versus
            physical-page operands.
        dynamic_csr: Optional (row_ptr,q_indices,schedule) storage produced by the CSR builder;
            None until built. Buffer contents are overwritten during routing.
    """

    dynamic_route: torch.Tensor
    page_table: torch.Tensor
    valid_kv_tokens: torch.Tensor
    q_offset_tokens: int | tuple[int, ...]
    policy: PrefillPolicy
    dynamic_csr: Optional[tuple[torch.Tensor, torch.Tensor, object]] = None

    def validate(self, *, sq: int, skv: int, device: torch.device) -> None:
        """Validate batch-one tensor metadata and right-aligned causal offset.

        Args:
            sq: Query-token count of this request.
            skv: Logical KV-token count including context.
            device: Expected device for route/map/length tensors.

        Returns:
            None; raises ValueError for inconsistent lengths, route shape/dtype/layout/device, map
            dtype/device or offset. Tensor values are unchanged.
        """
        if sq <= 0 or skv <= 0 or sq > skv:
            raise ValueError("require 0 < Sq <= Skv")
        if self.q_offset_tokens != skv - sq:
            raise ValueError("q_offset_tokens must equal Skv - Sq")
        expected = (SPEC.kv_heads, sq, SPEC.dynamic_topk)
        if tuple(self.dynamic_route.shape) != expected:
            raise ValueError(f"dynamic_route must have shape {expected}")
        if self.dynamic_route.dtype != torch.int32:
            raise ValueError("dynamic_route must be int32")
        if self.dynamic_route.device != device or not self.dynamic_route.is_contiguous():
            raise ValueError("dynamic_route must be contiguous on the Q device")
        if self.page_table.dtype != torch.int32 or self.page_table.device != device:
            raise ValueError("page_table must be int32 on the Q device")
        if self.valid_kv_tokens.dtype != torch.int32 or self.valid_kv_tokens.device != device:
            raise ValueError("valid_kv_tokens must be int32 on the Q device")

    def validate_packed(
        self,
        *,
        q_lens: Sequence[int],
        kv_lens: Sequence[int],
        device: torch.device,
    ) -> None:
        """Validate packed request geometry without changing route IDs.

        Args:
            q_lens: Host positive query-token lengths in request order.
            kv_lens: Host positive logical KV-token lengths, one per query request, each at least
                its Q length.
            device: Expected device for contiguous int32 [4,sum(q_lens),16] routes and int32
                page/used-length metadata.

        Returns:
            None; raises ValueError on metadata inconsistency. Dynamic routes remain request-local
            KV block IDs.
        """
        if not q_lens or len(q_lens) != len(kv_lens):
            raise ValueError("q_lens and kv_lens must describe the same non-empty batch")
        if any(q <= 0 or k < q for q, k in zip(q_lens, kv_lens)):
            raise ValueError("each request must satisfy 0 < Sq <= Skv")
        offsets = tuple(k - q for q, k in zip(q_lens, kv_lens))
        declared_offsets = (
            (self.q_offset_tokens,)
            if isinstance(self.q_offset_tokens, int)
            else tuple(self.q_offset_tokens)
        )
        if declared_offsets != offsets:
            raise ValueError("q_offset_tokens must equal per-request Skv - Sq")
        expected = (SPEC.kv_heads, sum(q_lens), SPEC.dynamic_topk)
        if tuple(self.dynamic_route.shape) != expected:
            raise ValueError(f"dynamic_route must have shape {expected}")
        if self.dynamic_route.dtype != torch.int32:
            raise ValueError("dynamic_route must be int32")
        if self.dynamic_route.device != device or not self.dynamic_route.is_contiguous():
            raise ValueError("dynamic_route must be contiguous on the Q device")
        if self.page_table.dtype != torch.int32 or self.page_table.device != device:
            raise ValueError("page_table must be int32 on the Q device")
        if self.policy == PrefillPolicy.ALIGNED_PAGED:
            max_pages = (max(kv_lens) + SPEC.block_size - 1) // SPEC.block_size
            if self.page_table.ndim != 2 or self.page_table.shape[0] != len(q_lens):
                raise ValueError("page_table must have one row per request")
            if self.page_table.shape[1] < max_pages:
                raise ValueError("page_table does not cover the longest KV sequence")
        if self.valid_kv_tokens.dtype != torch.int32 or self.valid_kv_tokens.device != device:
            raise ValueError("valid_kv_tokens must be int32 on the Q device")
        if tuple(self.valid_kv_tokens.shape) != (len(q_lens),):
            raise ValueError("valid_kv_tokens must have one entry per request")


def valid_block_count(q_global: int, *, skv: int, block_size: int = 128) -> int:
    """Count logical pages intersecting the causal prefix through one query.

    Args:
        q_global: Zero-based query position in the logical KV sequence;0<=q_global<skv.
        skv: Logical KV sequence length in tokens.
        block_size: Tokens per logical block, default 128; caller must supply a positive size.

    Returns:
        Integer causal page count, capped by ceil(skv/block_size). Includes the final partially
        visible page.
    """
    if q_global < 0 or q_global >= skv:
        raise ValueError("query global position is outside the KV sequence")
    return min((q_global + block_size) // block_size, (skv + block_size - 1) // block_size)


def fixed_block_ids(valid_blocks: int, *, spec: SparseAttentionSpec = SPEC) -> list[int]:
    """Build the sorted unique prefix/local union for a causal page extent.

    Args:
        valid_blocks: Nonnegative number of logical pages visible to this query, including a
            partially visible boundary page.
        spec: SparseAttentionSpec supplying prefix/local page counts; defaults to SPEC.

    Returns:
        New host list of logical block IDs. Prefix/local overlap appears once; token-level boundary
        masking remains the attention consumer's responsibility.
    """
    if valid_blocks < 0:
        raise ValueError("valid_blocks must be non-negative")
    prefix = range(min(spec.prefix_blocks, valid_blocks))
    local = range(max(0, valid_blocks - spec.local_swa_blocks), valid_blocks)
    return sorted(set(prefix).union(local))


__all__ = [
    "FwdTilePlan",
    "KvLayout",
    "PrefillPolicy",
    "SPEC",
    "SparseAttentionSpec",
    "fixed_block_ids",
    "use_unified_aligned_prefill",
    "valid_block_count",
]
