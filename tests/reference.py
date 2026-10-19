# Copyright 2026 Maxwell Team.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch specification of causal block routing and grouped-query attention.

This MBSA implementation uses standard dot products, block maxima, stable
sorting and softmax. It requires no model code, accelerator kernel or cache
manager. Quantized index inputs must be decoded before this interface is called.
"""

from __future__ import annotations

import torch


class MaxwellBlockSparseAttentionReference:
    """Reference helpers using batched token-major [B,T,H,D] tensors.

    Index keys are shared by all index groups: [B,S,Di] or [B,S,1,Di].
    Routes are [B,T,G,R] block IDs; -1 means unused. Attention uses FP32
    scores/softmax/LSE and casts probabilities to the value dtype before PV.
    """

    def __init__(self, *, index_head_dim: int, head_dim: int,
                 block_size: int = 128, dynamic_topk: int = 16,
                 prefix_blocks: int = 8, local_blocks: int = 8,
                 query_chunk_size: int = 16) -> None:
        positive = (index_head_dim, head_dim, block_size, query_chunk_size)
        nonnegative = (dynamic_topk, prefix_blocks, local_blocks)
        if any(not isinstance(n, int) or n < 1 for n in positive):
            raise ValueError("dimensions, block_size and query_chunk_size must be positive integers")
        if any(not isinstance(n, int) or n < 0 for n in nonnegative):
            raise ValueError("route counts must be nonnegative integers")
        self.index_head_dim, self.head_dim = index_head_dim, head_dim
        self.block_size, self.dynamic_topk = block_size, dynamic_topk
        self.prefix_blocks, self.local_blocks = prefix_blocks, local_blocks
        self.query_chunk_size = query_chunk_size

    @staticmethod
    def _query_positions(*, batch_size, query_len, key_len, device, cache_position):
        """Absolute query positions; absent positions describe the cache suffix."""
        if cache_position is None:
            if query_len > key_len:
                raise ValueError("query length cannot exceed key length")
            return torch.arange(key_len - query_len, key_len, device=device).expand(batch_size, -1)
        positions = torch.as_tensor(cache_position, device=device, dtype=torch.long)
        if positions.shape == (query_len,):
            positions = positions.expand(batch_size, -1)
        if positions.shape != (batch_size, query_len):
            raise ValueError("cache_position must be [T] or [B,T]")
        if bool(((positions < 0) | (positions >= key_len)).any()):
            raise ValueError("query positions must be inside the key cache")
        return positions

    @staticmethod
    def _allowed_keys(*, query_positions, key_len, attention_mask):
        """Causal visibility with optional [B,S] keep or [B,1,T,S] masks.

        Boolean masks use True for visible keys. A 2D numeric mask uses one
        for visible keys; a 4D numeric mask uses zero for visible keys.
        """
        tokens = torch.arange(key_len, device=query_positions.device)
        allowed = tokens <= query_positions[..., None]
        if attention_mask is not None:
            mask = attention_mask.to(query_positions.device)
            if mask.ndim == 2:
                keep = mask.bool()[:, None, :]
            elif mask.ndim == 4 and mask.shape[1] == 1:
                keep = mask[:, 0] if mask.dtype == torch.bool else mask[:, 0] == 0
            else:
                raise ValueError("attention_mask must be [B,S] or [B,1,T,S]")
            allowed = allowed & keep
        return allowed

    def _select_blocks(self, index_query, index_key, query_positions, allowed):
        """Group-local raw-dot block maxima, score-desc/ID-asc remote Top-K."""
        if index_key.ndim == 4:
            if index_key.shape[2] != 1:
                raise ValueError("routing requires a shared index key")
            index_key = index_key[:, :, 0]
        if (index_query.ndim != 4 or index_key.ndim != 3 or
                index_query.shape[-1] != self.index_head_dim or
                index_key.shape[-1] != self.index_head_dim):
            raise ValueError("batched index operands must match index_head_dim")
        batch, queries, groups, _ = index_query.shape
        key_len = index_key.shape[1]
        blocks = (key_len + self.block_size - 1) // self.block_size
        result = torch.full((batch, queries, groups, self.dynamic_topk), -1,
                            dtype=torch.long, device=index_query.device)
        if self.dynamic_topk == 0:
            return result
        block_ids = torch.arange(blocks, device=index_query.device)
        block_end = (block_ids + 1) * self.block_size
        for begin in range(0, queries, self.query_chunk_size):
            end = min(begin + self.query_chunk_size, queries)
            scores = torch.einsum("btgd,bsd->btgs", index_query[:, begin:end].float(), index_key.float())
            scores = scores.masked_fill(~allowed[:, begin:end, None, :], -torch.inf)
            scores = torch.nn.functional.pad(scores, (0, blocks * self.block_size - key_len), value=-torch.inf)
            scores = scores.unflatten(-1, (blocks, self.block_size)).amax(-1)
            positions = query_positions[:, begin:end]
            local_begin = positions // self.block_size - self.local_blocks + 1
            eligible = ((block_ids >= self.prefix_blocks) &
                        (block_ids < local_begin[..., None]) &
                        (block_end <= positions[..., None] + 1) &
                        (block_end <= key_len))
            scores = scores.masked_fill(~eligible[:, :, None, :], -torch.inf)
            count = min(blocks, self.dynamic_topk)
            chosen = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :count]
            chosen = chosen.masked_fill(scores.gather(-1, chosen) == -torch.inf, -1)
            result[:, begin:end, :, :count] = chosen
        return result

    def _build_block_aligned_route(self, remote_blocks, query_positions):
        """Combine unique causal prefix/local blocks with remote IDs."""
        current = query_positions // self.block_size
        prefix = torch.arange(self.prefix_blocks, device=remote_blocks.device)
        prefix = prefix.expand(*current.shape, -1)
        prefix = prefix.masked_fill(prefix > current[..., None], -1)
        offsets = torch.arange(self.local_blocks, device=remote_blocks.device)
        local = current[..., None] - self.local_blocks + 1 + offsets
        local = local.masked_fill(local < self.prefix_blocks, -1)
        fixed = torch.cat((prefix, local), dim=-1)
        fixed = fixed[:, :, None, :].expand(*remote_blocks.shape[:-1], -1)
        route = torch.cat((fixed, remote_blocks.long()), dim=-1).sort(-1).values
        # Duplicate IDs can also be supplied by a direct helper caller.
        duplicates = route[..., 1:] == route[..., :-1]
        route[..., 1:] = route[..., 1:].masked_fill(duplicates, -1)
        return route

    def _sparse_attention(self, query_states, key_states, value_states,
                          selected_blocks, allowed, *, return_lse=False):
        """Evaluate selected causal support, with zero O/-inf LSE if empty."""
        batch, queries, heads, dim = query_states.shape
        key_len, groups = key_states.shape[1:3]
        if dim != self.head_dim or heads % groups:
            raise ValueError("attention head_dim must match and Q heads must be divisible by KV heads")
        group_size = heads // groups
        route_width = selected_blocks.shape[-1]
        if route_width == 0:
            output = torch.zeros_like(query_states)
            lse = torch.full((batch, queries, heads), -torch.inf,
                             dtype=torch.float32, device=query_states.device)
            return (output, lse) if return_lse else output
        output_parts, lse_parts = [], []
        token_offsets = torch.arange(self.block_size, device=query_states.device)
        batch_ids = torch.arange(batch, device=query_states.device)[:, None, None, None]
        group_ids = torch.arange(groups, device=query_states.device)[None, None, :, None]
        for begin in range(0, queries, self.query_chunk_size):
            end = min(begin + self.query_chunk_size, queries)
            routes = selected_blocks[:, begin:end]
            if key_len <= route_width * self.block_size:
                # Small caches can use dense token order without padding slots.
                tokens = torch.arange(key_len, device=query_states.device)
                supported = (routes[..., None] == tokens // self.block_size).any(-2)
                token_ids = tokens.expand(batch, end - begin, groups, -1)
            else:
                routes = routes.sort(-1).values
                unique = torch.ones_like(routes, dtype=torch.bool)
                unique[..., 1:] = routes[..., 1:] != routes[..., :-1]
                token_ids = (routes[..., None] * self.block_size + token_offsets).flatten(-2)
                supported = (routes.ge(0) & unique)[..., None].expand(*routes.shape, self.block_size).flatten(-2)
                supported = supported & (token_ids < key_len)
            safe_ids = token_ids.clamp(0, key_len - 1)
            visible = allowed[:, begin:end].gather(-1, safe_ids.flatten(-2)).view_as(safe_ids)
            supported = supported & visible
            keys = key_states[batch_ids, safe_ids, group_ids]
            values = value_states[batch_ids, safe_ids, group_ids]
            query = query_states[:, begin:end].reshape(batch, end - begin, groups, group_size, dim)
            scores = torch.einsum("btgnd,btgwd->btgnw", query.float(), keys.float()) * dim**-0.5
            scores = scores.masked_fill(~supported[..., None, :], -torch.inf)
            lse = scores.logsumexp(-1)
            probabilities = scores.softmax(-1)
            probabilities = torch.where(supported[..., None, :], probabilities, 0).to(values.dtype)
            # Mask values too: zero probabilities must not consume poisoned padding.
            values = values.masked_fill(~supported[..., None], 0)
            output = torch.einsum("btgnw,btgwd->btgnd", probabilities, values)
            output_parts.append(output.reshape(batch, end - begin, heads, dim).to(query_states.dtype))
            lse_parts.append(lse.reshape(batch, end - begin, heads))
        output, lse = torch.cat(output_parts, dim=1), torch.cat(lse_parts, dim=1)
        return (output, lse) if return_lse else output


def reference_pipeline(index_q: torch.Tensor, index_k: torch.Tensor,
                       q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
                       block_size: int = 128, dynamic_topk: int = 16,
                       prefix_blocks: int = 8, local_blocks: int = 8,
                       query_offset: int | None = None
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return O[T,H,D], natural-log LSE[T,H], remote IDs[G,T,TopK].

    Inputs are token-major: Q[T,H,D], K/V[S,G,D], index Q[T,G,Di]
    and shared index K[S,Di] or [S,1,Di]. Index values are floating point;
    use index_formats decoders for packed MX formats. The default query
    offset is S-T. This diagnostic implementation is not a timed backend.
    """
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("Q and matching K/V must be token-major rank-3 tensors")
    queries, heads, dim = q.shape
    key_len, groups, key_dim = k.shape
    if min(queries, heads, dim, key_len, groups) < 1 or key_dim != dim or heads % groups:
        raise ValueError("positive Q/KV dimensions and integral GQA are required")
    if index_k.ndim == 3:
        if index_k.shape[1] != 1:
            raise ValueError("routing requires a shared index key")
        index_k = index_k[:, 0]
    if (index_q.ndim != 3 or index_q.shape[:2] != (queries, groups) or
            index_k.ndim != 2 or index_k.shape != (key_len, index_q.shape[-1])):
        raise ValueError("index Q must be [T,G,Di] and shared index K [S,Di] or [S,1,Di]")
    if any(t.device != q.device for t in (index_q, index_k, k, v)):
        raise ValueError("all operands must be on the same device")
    if any(not t.is_floating_point() for t in (index_q, index_k, q, k, v)):
        raise ValueError("operands must be floating point; decode packed index formats first")
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError("Q/K/V must share a floating-point dtype")
    offset = key_len - queries if query_offset is None else query_offset
    if not isinstance(offset, int) or offset < 0 or offset + queries > key_len:
        raise ValueError("query_offset must place every query inside the key cache")
    oracle = MaxwellBlockSparseAttentionReference(index_head_dim=index_q.shape[-1], head_dim=dim,
        block_size=block_size, dynamic_topk=dynamic_topk, prefix_blocks=prefix_blocks,
        local_blocks=local_blocks)
    output_parts, lse_parts, route_parts = [], [], []
    for begin in range(0, queries, oracle.query_chunk_size):
        end = min(begin + oracle.query_chunk_size, queries)
        positions = torch.arange(offset + begin, offset + end, device=q.device)[None]
        allowed = oracle._allowed_keys(query_positions=positions, key_len=key_len, attention_mask=None)
        remote = oracle._select_blocks(index_q[None, begin:end], index_k[None], positions, allowed)
        route = oracle._build_block_aligned_route(remote, positions)
        output, lse = oracle._sparse_attention(q[None, begin:end], k[None], v[None], route,
                                             allowed, return_lse=True)
        output_parts.append(output[0])
        lse_parts.append(lse[0])
        route_parts.append(remote[0].transpose(0, 1))
    return (torch.cat(output_parts), torch.cat(lse_parts),
            torch.cat(route_parts, dim=1).to(torch.int32))
