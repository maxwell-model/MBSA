"""Streaming block score and exact hierarchical Top-K selection.

This module is deliberately self-contained.  A partition score tensor is
discarded immediately after its local Top-K has been merged into the running
candidate set, so the complete ``[H, Q, KV-block]`` score matrix is never
materialized.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RouterConfig:
    block_size: int = 128
    dynamic_topk: int = 16
    prefix_blocks: int = 8
    local_blocks: int = 8
    partition_blocks: int = 64
    query_chunk: int = 16
    score_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "block_size",
            "dynamic_topk",
            "partition_blocks",
            "query_chunk",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.prefix_blocks < 0 or self.local_blocks < 0:
            raise ValueError("fixed-support block counts must be non-negative")


@dataclass
class RouterResult:
    indices: torch.Tensor
    scores: torch.Tensor
    local_candidate_elements: int


def _normalize_index_inputs(
    index_q: torch.Tensor, index_k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if index_q.ndim != 3 or index_k.ndim != 3:
        raise ValueError("index_q/index_k must be [tokens, heads, dim]")
    if index_q.shape[-1] != index_k.shape[-1]:
        raise ValueError("index_q and index_k dimensions must match")
    if index_k.shape[1] not in (1, index_q.shape[1]):
        raise ValueError("index_k heads must be one or match index_q heads")
    if index_q.device != index_k.device:
        raise ValueError("index_q and index_k must share a device")
    if not index_q.is_floating_point() or not index_k.is_floating_point():
        raise TypeError("index_q and index_k must be floating point")
    q = index_q.transpose(0, 1).contiguous().float()
    k = index_k.transpose(0, 1).contiguous().float()
    if k.shape[0] == 1 and q.shape[0] != 1:
        k = k.expand(q.shape[0], -1, -1)
    return q, k


def dynamic_candidate_mask(
    query_positions: torch.Tensor,
    block_ids: torch.Tensor,
    config: RouterConfig,
) -> torch.Tensor:
    """Return ``[Q, B]`` validity outside the fixed prefix/window union."""
    q_block = torch.div(
        query_positions.to(torch.long), config.block_size, rounding_mode="floor"
    )
    blocks = block_ids.to(torch.long).view(1, -1)
    causal = blocks <= q_block.view(-1, 1)
    prefix = blocks < config.prefix_blocks
    if config.local_blocks:
        local_begin = torch.clamp(
            q_block - config.local_blocks + 1,
            min=config.prefix_blocks,
        )
        local = (blocks >= local_begin.view(-1, 1)) & (
            blocks <= q_block.view(-1, 1)
        )
    else:
        local = torch.zeros_like(causal)
    return causal & ~prefix & ~local


def _merge_topk(
    lhs_scores: torch.Tensor,
    lhs_indices: torch.Tensor,
    rhs_scores: torch.Tensor,
    rhs_indices: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.cat((lhs_scores, rhs_scores), dim=-1)
    indices = torch.cat((lhs_indices, rhs_indices), dim=-1)
    take = min(k, scores.shape[-1])
    values, positions = torch.topk(scores, take, dim=-1, largest=True, sorted=True)
    winners = torch.gather(indices, -1, positions)
    winners = torch.where(torch.isfinite(values), winners, torch.full_like(winners, -1))
    if take != k:
        pad_shape = (*values.shape[:-1], k - take)
        values = torch.cat(
            (values, torch.full(pad_shape, -torch.inf, device=values.device)), -1
        )
        winners = torch.cat(
            (
                winners,
                torch.full(
                    pad_shape,
                    -1,
                    dtype=winners.dtype,
                    device=winners.device,
                ),
            ),
            -1,
        )
    return values, winners


def hierarchical_exact_topk(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    *,
    query_positions: torch.Tensor | None = None,
    config: RouterConfig = RouterConfig(),
) -> RouterResult:
    """Compute exact dynamic block routes without a dense score matrix.

    The score of a query/block pair is the maximum dot product over tokens in
    that KV block.  Each disjoint KV partition emits a local Top-K.  A running
    exact merge retains Top-K over the union of all processed partitions.
    """
    q, k_tokens = _normalize_index_inputs(index_q, index_k)
    heads, queries, dim = q.shape
    kv_tokens = k_tokens.shape[1]
    num_blocks = (kv_tokens + config.block_size - 1) // config.block_size
    if query_positions is None:
        offset = max(kv_tokens - queries, 0)
        query_positions = torch.arange(
            offset, offset + queries, device=q.device, dtype=torch.long
        )
    if query_positions.shape != (queries,):
        raise ValueError("query_positions must have shape [query_tokens]")

    running_scores = torch.full(
        (heads, queries, config.dynamic_topk),
        -torch.inf,
        device=q.device,
        dtype=torch.float32,
    )
    running_indices = torch.full(
        (heads, queries, config.dynamic_topk),
        -1,
        device=q.device,
        dtype=torch.long,
    )
    emitted = 0
    for partition_begin in range(0, num_blocks, config.partition_blocks):
        partition_end = min(partition_begin + config.partition_blocks, num_blocks)
        block_ids = torch.arange(
            partition_begin, partition_end, device=q.device, dtype=torch.long
        )
        block_count = partition_end - partition_begin
        token_begin = partition_begin * config.block_size
        token_end = min(partition_end * config.block_size, kv_tokens)
        partition = k_tokens[:, token_begin:token_end]
        padded_tokens = block_count * config.block_size
        if partition.shape[1] != padded_tokens:
            padding = torch.zeros(
                (heads, padded_tokens - partition.shape[1], dim),
                device=q.device,
                dtype=partition.dtype,
            )
            partition = torch.cat((partition, padding), dim=1)
        partition = partition.view(heads, block_count, config.block_size, dim)
        valid_tokens = torch.arange(
            token_begin,
            token_begin + padded_tokens,
            device=q.device,
        ) < kv_tokens
        valid_tokens = valid_tokens.view(block_count, config.block_size)

        for query_begin in range(0, queries, config.query_chunk):
            query_end = min(query_begin + config.query_chunk, queries)
            q_chunk = q[:, query_begin:query_end]
            token_scores = torch.einsum("hqd,hbtd->hqbt", q_chunk, partition)
            token_scores = token_scores * float(config.score_scale)
            token_scores = token_scores.masked_fill(
                ~valid_tokens.view(1, 1, block_count, config.block_size),
                -torch.inf,
            )
            block_scores = token_scores.amax(dim=-1)
            allowed = dynamic_candidate_mask(
                query_positions[query_begin:query_end], block_ids, config
            )
            block_scores = block_scores.masked_fill(~allowed.unsqueeze(0), -torch.inf)
            local_k = min(config.dynamic_topk, block_count)
            local_scores, local_positions = torch.topk(
                block_scores,
                local_k,
                dim=-1,
                largest=True,
                sorted=True,
            )
            local_indices = block_ids[local_positions]
            local_indices = torch.where(
                torch.isfinite(local_scores),
                local_indices,
                torch.full_like(local_indices, -1),
            )
            merged_scores, merged_indices = _merge_topk(
                running_scores[:, query_begin:query_end],
                running_indices[:, query_begin:query_end],
                local_scores,
                local_indices,
                config.dynamic_topk,
            )
            running_scores[:, query_begin:query_end] = merged_scores
            running_indices[:, query_begin:query_end] = merged_indices
            emitted += heads * (query_end - query_begin) * local_k

    return RouterResult(
        indices=running_indices.to(torch.int32),
        scores=running_scores,
        local_candidate_elements=emitted,
    )

