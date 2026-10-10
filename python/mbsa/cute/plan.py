# Copyright 2026 MBSA contributors.
# SPDX-License-Identifier: Apache-2.0
"""Host metadata contracts for independent Fixed and Dynamic attention.

This implementation derives from the MBSA API, tensor layouts and supported
geometry. It does not include an upstream planning implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import torch




class PrefillPolicy(str, Enum):
    ALIGNED_CONTIGUOUS = "aligned_contiguous"
    ALIGNED_PAGED = "aligned_paged"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class SparseAttentionSpec:
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
        return self.q_heads // self.kv_heads

    @property
    def topk(self) -> int:
        return sum((self.prefix_blocks, self.dynamic_topk, self.local_swa_blocks))

    def validate(self) -> None:
        contracts = (
            (self.block_size == 128, "SM100 requires 128-token blocks"),
            ((self.prefix_blocks, self.dynamic_topk, self.local_swa_blocks) == (8, 16, 8),
             "SM100 requires Prefix8, Dynamic16 and Local8"),
            ((self.q_heads, self.kv_heads, self.head_dim) == (32, 4, 128),
             "SM100 requires 32 query heads, 4 KV heads and dimension 128"),
        )
        for condition, message in contracts:
            _require(condition, message)
        _require(self.q_heads % self.kv_heads == 0, "Query heads must divide into KV groups")
        _require(self.kv_dtype in (torch.bfloat16, torch.float8_e4m3fn),
                 "This attention specification requires BF16 or E4M3FN KV")
        _require(self.q_dtype in ((torch.bfloat16, torch.float8_e4m3fn)
                                  if self.kv_dtype == torch.float8_e4m3fn else (torch.bfloat16,)),
                 "BF16 KV requires BF16 Q; E4M3FN KV accepts BF16 or E4M3FN Q")


SPEC = SparseAttentionSpec()
SPEC.validate()




def _int32_on(tensor: torch.Tensor, device: torch.device, name: str) -> None:
    _require(tensor.dtype == torch.int32 and tensor.device == device,
             f"{name} requires int32 storage on the query device")


@dataclass
class FwdTilePlan:
    dynamic_route: torch.Tensor
    page_table: torch.Tensor
    valid_kv_tokens: torch.Tensor
    q_offset_tokens: int | tuple[int, ...]
    policy: PrefillPolicy

    def _route_contract(self, queries: int, device: torch.device) -> None:
        shape = (SPEC.kv_heads, queries, SPEC.dynamic_topk)
        _require(tuple(self.dynamic_route.shape) == shape,
                 f"Dynamic routes require shape {shape}")
        _require(self.dynamic_route.dtype == torch.int32, "Dynamic routes require int32 entries")
        _require(self.dynamic_route.device == device and self.dynamic_route.is_contiguous(),
                 "Dynamic routes require contiguous storage on the query device")

    def validate(self, *, sq: int, skv: int, device: torch.device) -> None:
        _require(0 < sq <= skv, "Query and KV lengths must satisfy 0 < Q <= KV")
        _require(self.q_offset_tokens == skv - sq, "Query offset must equal KV length minus Q length")
        self._route_contract(sq, device)
        _int32_on(self.page_table, device, "Page table")
        _int32_on(self.valid_kv_tokens, device, "Valid KV lengths")

    def validate_packed(
        self, *, q_lens: Sequence[int], kv_lens: Sequence[int], device: torch.device,
    ) -> None:
        _require(bool(q_lens) and len(q_lens) == len(kv_lens),
                 "Packed query and KV lengths require equal nonzero request counts")
        _require(all(0 < q <= k for q, k in zip(q_lens, kv_lens)),
                 "Each packed request must satisfy 0 < Q <= KV")
        offsets = tuple(k - q for q, k in zip(q_lens, kv_lens))
        actual_offsets = ((self.q_offset_tokens,) if isinstance(self.q_offset_tokens, int)
                          else tuple(self.q_offset_tokens))
        _require(actual_offsets == offsets, "Packed offsets must equal each request's KV minus Q length")
        self._route_contract(sum(q_lens), device)
        _int32_on(self.page_table, device, "Page table")
        if self.policy == PrefillPolicy.ALIGNED_PAGED:
            _require(self.page_table.ndim == 2 and self.page_table.shape[0] == len(q_lens),
                     "Paged metadata requires a two-dimensional table with one row per request")
            pages = (max(kv_lens) + SPEC.block_size - 1) // SPEC.block_size
            _require(self.page_table.shape[1] >= pages, "Page table is too narrow for the KV lengths")
        _int32_on(self.valid_kv_tokens, device, "Valid KV lengths")
        _require(tuple(self.valid_kv_tokens.shape) == (len(q_lens),),
                 "Valid KV lengths require one entry per request")
