# Copyright 2024 CompassLLM and the Shopee Search team. All rights reserved.
# Modified for MBSA: standalone adapter and optional LSE; see NOTICE.

"""Maxwell Block Sparse Attention Python accuracy oracle.

Adapted from the pinned HF attention reference.
Revision: d97a6c52967a5ed332e2196a5b985f0b4a475549
Source methods: lines 492-851 (query positions, allowed keys, route selection,
fixed support and sparse attention). Their computation is preserved, including
fp32 logits and the probability cast to the value dtype before the PV product.
The model constructor/projections/cache/Transformers dependency are excluded.
``_sparse_attention`` only adds an optional natural-log LSE return.

The flat adapter accepts post-RoPE tensors. The index key is shared across all
GQA groups. Routing uses max token dot product / sqrt(index_dim), without ReLU.
Ties retain the upstream ``torch.topk`` behavior; kernels may choose a different
block with an equal score. This eager implementation is an accuracy oracle.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F


class MaxwellBlockSparseAttentionReference:
    """Standalone upstream methods, operating on batched HF tensor layouts.

    Index Q is [B,G,Q,Di], index K is [B,K,Di], main Q is [B,Hq,Q,D],
    main K/V are [B,G,K,D], and complete routes are [B,Q,G,blocks].
    """

    def __init__(
        self, *, index_head_dim: int, head_dim: int, block_size: int = 128,
        dynamic_topk: int = 16, prefix_blocks: int = 8, local_blocks: int = 8,
        query_chunk_size: int = 16,
    ) -> None:
        if min(index_head_dim, head_dim, block_size, dynamic_topk, query_chunk_size) <= 0:
            raise ValueError("dimensions, block_size, dynamic_topk and query_chunk_size must be positive")
        if prefix_blocks < 0 or local_blocks < 0:
            raise ValueError("prefix_blocks and local_blocks must be non-negative")
        self.block_size = block_size
        self.num_remote_blocks = dynamic_topk
        self.num_prefix_blocks = prefix_blocks
        self.num_local_blocks = local_blocks
        self.prefix_window_size = prefix_blocks * block_size
        self.local_window_size = local_blocks * block_size
        self.query_chunk_size = query_chunk_size
        self.indexer = SimpleNamespace(index_softmax_scale=index_head_dim ** -0.5)
        self.softmax_scale = head_dim ** -0.5

    @staticmethod
    def _query_positions(
        *, batch_size, query_len, key_len, device, cache_position
    ):
        if cache_position is None:
            positions = torch.arange(
                key_len - query_len,
                key_len,
                device=device,
                dtype=torch.long,
            )
        else:
            positions = cache_position.to(
                device=device, dtype=torch.long
            ).reshape(-1)
            if positions.numel() != query_len:
                raise ValueError(
                    f"cache_position has {positions.numel()} entries for "
                    f"query_len={query_len}"
                )
        return positions.unsqueeze(0).expand(batch_size, -1)

    @staticmethod
    def _allowed_keys(*, query_positions, key_len, attention_mask):
        key_positions = torch.arange(key_len, device=query_positions.device)
        allowed = (
            key_positions.view(1, 1, key_len)
            <= query_positions.unsqueeze(-1)
        )
        if attention_mask is None:
            return allowed
        if attention_mask.dim() != 4 or attention_mask.dtype == torch.bool:
            raise ValueError(
                "block_sparse_gqa attention expects the model-level 4D "
                "additive causal mask"
            )
        mask = attention_mask[
            :, 0, -query_positions.shape[-1]:, :key_len
        ]
        return allowed & (mask >= 0)

    def _select_blocks(self, index_query, index_key, query_positions, allowed):
        """Select an independent block-max remote route for every GQA group."""
        batch_size, num_groups, query_len, _ = index_query.shape
        key_len = index_key.shape[1]
        num_blocks = (key_len + self.block_size - 1) // self.block_size
        selected_count = min(max(self.num_remote_blocks, 1), num_blocks)
        block_starts = torch.arange(
            num_blocks,
            device=index_query.device,
        ) * self.block_size
        block_ends = block_starts + self.block_size
        selected_chunks = []

        for query_start in range(0, query_len, self.query_chunk_size):
            query_end = min(
                query_start + self.query_chunk_size,
                query_len,
            )
            chunk_len = query_end - query_start
            query_chunk = index_query[
                :, :, query_start:query_end, :
            ]
            per_group_block_scores = torch.full(
                (
                    batch_size,
                    num_groups,
                    chunk_len,
                    num_blocks,
                ),
                float("-inf"),
                dtype=torch.float32,
                device=index_query.device,
            )
            # Match the Megatron max-route oracle while bounding the eager
            # token-score temporary for 256K contexts.
            key_blocks_per_chunk = max(
                1,
                8192 // self.block_size,
            )
            for block_start in range(
                0,
                num_blocks,
                key_blocks_per_chunk,
            ):
                block_end = min(
                    block_start + key_blocks_per_chunk,
                    num_blocks,
                )
                key_start = block_start * self.block_size
                key_end = min(
                    block_end * self.block_size,
                    key_len,
                )
                token_scores = torch.einsum(
                    "bgqd,bkd->bgqk",
                    query_chunk.float(),
                    index_key[:, key_start:key_end].float(),
                )
                token_scores.mul_(self.indexer.index_softmax_scale)
                token_scores.masked_fill_(
                    ~allowed[
                        :,
                        None,
                        query_start:query_end,
                        key_start:key_end,
                    ],
                    float("-inf"),
                )
                block_count = block_end - block_start
                padding = (
                    block_count * self.block_size
                    - (key_end - key_start)
                )
                if padding:
                    token_scores = F.pad(
                        token_scores,
                        (0, padding),
                        value=float("-inf"),
                    )
                per_group_block_scores[
                    ...,
                    block_start:block_end,
                ] = token_scores.view(
                    batch_size,
                    num_groups,
                    chunk_len,
                    block_count,
                    self.block_size,
                ).amax(dim=-1)
            route_scores = per_group_block_scores.permute(
                0, 2, 1, 3
            ).contiguous()
            current_block = torch.div(
                query_positions[:, query_start:query_end],
                self.block_size,
                rounding_mode="floor",
            )
            local_start_block = (
                current_block - self.num_local_blocks + 1
            ).clamp(min=0)
            local_start = local_start_block * self.block_size
            eligible = (
                block_starts.view(1, 1, -1)
                >= self.prefix_window_size
            ) & (
                block_ends.view(1, 1, -1)
                <= local_start.unsqueeze(-1)
            )
            route_scores.masked_fill_(
                ~eligible.unsqueeze(2),
                float("-inf"),
            )
            values, indices = route_scores.topk(
                selected_count,
                dim=-1,
            )
            indices = torch.where(
                torch.isfinite(values),
                indices,
                -1,
            )
            if selected_count < self.num_remote_blocks:
                indices = F.pad(
                    indices,
                    (0, self.num_remote_blocks - selected_count),
                    value=-1,
                )
            selected_chunks.append(indices)

        return torch.cat(selected_chunks, dim=1).contiguous()

    def _build_block_aligned_route(self, remote_blocks, query_positions):
        """Add the same fixed prefix/local blocks to every group route."""
        if remote_blocks.ndim != 4:
            raise ValueError(
                "per-group remote blocks must be [batch, query, groups, remote]"
            )
        batch_size, query_len, num_groups, _ = remote_blocks.shape
        current_block = torch.div(
            query_positions,
            self.block_size,
            rounding_mode="floor",
        ).unsqueeze(-1).unsqueeze(-1)

        prefix = torch.arange(
            self.num_prefix_blocks,
            device=remote_blocks.device,
            dtype=remote_blocks.dtype,
        ).view(1, 1, 1, -1)
        prefix = prefix.expand(
            batch_size, query_len, num_groups, -1
        )
        prefix = torch.where(prefix <= current_block, prefix, -1)

        local_offsets = torch.arange(
            1 - self.num_local_blocks,
            1,
            device=remote_blocks.device,
            dtype=remote_blocks.dtype,
        ).view(1, 1, 1, -1)
        local = current_block.to(remote_blocks.dtype) + local_offsets
        local = local.expand(
            batch_size, query_len, num_groups, -1
        )
        local = torch.where(local >= self.num_prefix_blocks, local, -1)
        return torch.cat(
            (prefix, remote_blocks, local),
            dim=-1,
        ).contiguous()

    def _sparse_attention(
        self,
        query_states,
        key_states,
        value_states,
        selected_blocks,
        allowed,
        *,
        return_lse=False,
    ):
        batch_size, num_query_heads, query_len, head_dim = (
            query_states.shape
        )
        num_groups = key_states.shape[1]
        key_len = key_states.shape[2]
        if selected_blocks.ndim != 4 or selected_blocks.shape[:3] != (
            batch_size,
            query_len,
            num_groups,
        ):
            raise ValueError(
                "selected blocks must be [batch, query, groups, blocks]"
            )
        heads_per_group = num_query_heads // num_groups
        selected_tokens = selected_blocks.shape[-1] * self.block_size
        block_offsets = torch.arange(
            self.block_size,
            device=query_states.device,
        )
        output = torch.empty(
            batch_size,
            num_query_heads,
            query_len,
            head_dim,
            dtype=value_states.dtype,
            device=value_states.device,
        )

        if return_lse:
            lse = torch.empty(
                batch_size, num_query_heads, query_len,
                dtype=torch.float32, device=query_states.device,
            )

        for query_start in range(0, query_len, self.query_chunk_size):
            query_end = min(
                query_start + self.query_chunk_size,
                query_len,
            )
            chunk_len = query_end - query_start
            selected_chunk = selected_blocks[
                :, query_start:query_end
            ].permute(0, 2, 1, 3)
            token_positions = (
                selected_chunk.unsqueeze(-1)
                * self.block_size
                + block_offsets
            ).reshape(
                batch_size,
                num_groups,
                chunk_len,
                selected_tokens,
            )
            in_range = (
                (token_positions >= 0)
                & (token_positions < key_len)
            )
            gather_positions = token_positions.clamp(
                min=0,
                max=key_len - 1
            )
            allowed_chunk = allowed[
                :, None, query_start:query_end
            ].expand(
                batch_size,
                num_groups,
                chunk_len,
                key_len,
            )
            valid = in_range & torch.gather(
                allowed_chunk,
                3,
                gather_positions,
            )
            gather_index = gather_positions[..., None].expand(
                batch_size,
                num_groups,
                chunk_len,
                selected_tokens,
                head_dim,
            )
            selected_key = torch.gather(
                key_states.unsqueeze(2).expand(
                    batch_size,
                    num_groups,
                    chunk_len,
                    key_len,
                    head_dim,
                ),
                3,
                gather_index,
            )
            selected_value = torch.gather(
                value_states.unsqueeze(2).expand(
                    batch_size,
                    num_groups,
                    chunk_len,
                    key_len,
                    head_dim,
                ),
                3,
                gather_index,
            )
            grouped_query = query_states[
                :, :, query_start:query_end
            ].reshape(
                batch_size,
                num_groups,
                heads_per_group,
                chunk_len,
                head_dim,
            )
            # Accumulate scores in fp32 like the Triton kernels; a bf16 einsum
            # output quantizes logits of ~1e3 to steps of several units.
            scores = torch.einsum(
                "bghqd,bgqkd->bghqk",
                grouped_query.float(),
                selected_key.float(),
            )
            scores.mul_(self.softmax_scale)
            scores.masked_fill_(
                ~valid[:, :, None],
                float("-inf"),
            )
            if return_lse:
                lse[:, :, query_start:query_end] = torch.logsumexp(
                    scores, dim=-1
                ).reshape(batch_size, num_query_heads, chunk_len)
            probabilities = torch.softmax(
                scores,
                dim=-1,
                dtype=torch.float32,
            )
            probabilities = torch.nan_to_num(
                probabilities,
                nan=0.0,
            ).to(selected_value.dtype)
            context = torch.einsum(
                "bghqk,bgqkd->bghqd",
                probabilities,
                selected_value,
            )
            output[
                :, :, query_start:query_end
            ] = context.reshape(
                batch_size,
                num_query_heads,
                chunk_len,
                head_dim,
            )

        return (output, lse) if return_lse else output


def reference_pipeline(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_size: int = 128,
    dynamic_topk: int = 16,
    prefix_blocks: int = 8,
    local_blocks: int = 8,
    query_offset: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return output [Q,Hq,D], natural-log LSE [Q,Hq], remote route [G,Q,R].

    Main K/V are [K,G,D], index Q is [Q,G,Di], and shared index K is
    [K,1,Di] or [K,Di]. Query positions start at K-Q unless query_offset
    is supplied, covering full prefill and contiguous cached query chunks.
    Prefix/remote/local support is assembled with the upstream HF methods.
    """
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("q must be [Q,Hq,D] and k/v must be [K,G,D]")
    queries, heads, dim = q.shape
    tokens, groups, key_dim = k.shape
    if min(queries, tokens, heads, groups, dim) <= 0:
        raise ValueError("input dimensions must be non-empty")
    if dim != key_dim or heads % groups:
        raise ValueError("main head dimensions must match and Hq must be divisible by G")
    if index_q.ndim != 3 or index_q.shape[:2] != (queries, groups):
        raise ValueError("index_q must be [Q,G,Di], with one index query per KV group")
    if index_k.ndim == 3:
        if index_k.shape[1] != 1:
            raise ValueError("Maxwell Block Sparse Attention uses one shared index key head")
        index_k = index_k[:, 0]
    if index_k.ndim != 2 or index_k.shape != (tokens, index_q.shape[-1]):
        raise ValueError("index_k must be shared [K,Di] or [K,1,Di]")
    if len({tensor.device for tensor in (index_q, index_k, q, k, v)}) != 1:
        raise ValueError("all tensors must share a device")
    if not all(tensor.is_floating_point() for tensor in (index_q, index_k, q, k, v)):
        raise TypeError("all tensors must be floating point")
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise TypeError("main q/k/v must share a dtype")
    offset = tokens - queries if query_offset is None else query_offset
    if not isinstance(offset, int) or offset < 0 or offset + queries > tokens:
        raise ValueError("query_offset must place every query within the KV sequence")
    oracle = MaxwellBlockSparseAttentionReference(
        index_head_dim=index_q.shape[-1], head_dim=dim, block_size=block_size,
        dynamic_topk=dynamic_topk, prefix_blocks=prefix_blocks, local_blocks=local_blocks,
    )
    positions = oracle._query_positions(
        batch_size=1, query_len=queries, key_len=tokens, device=q.device,
        cache_position=torch.arange(offset, offset + queries, device=q.device),
    )
    allowed = oracle._allowed_keys(
        query_positions=positions, key_len=tokens, attention_mask=None,
    )
    remote = oracle._select_blocks(
        index_q.transpose(0, 1).unsqueeze(0), index_k.unsqueeze(0), positions, allowed,
    )
    complete = oracle._build_block_aligned_route(remote, positions)
    output, lse = oracle._sparse_attention(
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0), complete, allowed, return_lse=True,
    )
    return (
        output[0].transpose(0, 1).contiguous(),
        lse[0].transpose(0, 1).contiguous(),
        remote[0].permute(1, 0, 2).to(torch.int32).contiguous(),
    )
