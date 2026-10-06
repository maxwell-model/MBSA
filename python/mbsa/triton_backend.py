# SPDX-License-Identifier: Apache-2.0
"""Triton precision baseline: exact routing and split sparse attention.

Routing retains disjoint sixteen-block partitions and an exact global Top-16
merge. Fixed attention reuses K/V over query tiles; dynamic attention groups
route entries by KV block before merging their online-softmax states.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError as error:  # pragma: no cover - GPU-only dependency
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = error
else:
    _TRITON_IMPORT_ERROR = None


if triton is not None:

    @triton.jit
    def _partition_score_kernel(
        q_ptr,
        k_ptr,
        candidate_score_ptr,
        candidate_index_ptr,
        queries,
        kv_tokens,
        q_offset,
        scale,
        stride_qt: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_kt: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_ch: tl.constexpr,
        stride_cq: tl.constexpr,
        stride_cp: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        PREFIX_BLOCKS: tl.constexpr,
        LOCAL_BLOCKS: tl.constexpr,
        PARTITION_BLOCKS: tl.constexpr,
        BLOCK_Q: tl.constexpr,
    ):
        partition_id = tl.program_id(0)
        query_tile = tl.program_id(1)
        head = tl.program_id(2)
        q_ids = query_tile * BLOCK_Q + tl.arange(0, BLOCK_Q)
        d_ids = tl.arange(0, HEAD_DIM)
        q = tl.load(
            q_ptr + q_ids[:, None] * stride_qt + head * stride_qh + d_ids[None, :],
            mask=q_ids[:, None] < queries,
            other=0.0,
        )
        query_positions = q_offset + q_ids
        query_blocks = query_positions // BLOCK_SIZE
        local_begin = tl.maximum(PREFIX_BLOCKS, query_blocks - LOCAL_BLOCKS + 1)
        for slot in tl.static_range(0, PARTITION_BLOCKS):
            block = partition_id * PARTITION_BLOCKS + slot
            token_ids = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            key = tl.load(
                k_ptr
                + token_ids[:, None] * stride_kt
                + head * stride_kh
                + d_ids[None, :],
                mask=token_ids[:, None] < kv_tokens,
                other=0.0,
            )
            logits = tl.dot(q, tl.trans(key), out_dtype=tl.float32) * scale
            logits = tl.where(token_ids[None, :] < kv_tokens, logits, -float("inf"))
            score = tl.max(logits, axis=1)
            is_prefix = block < PREFIX_BLOCKS
            is_local = (block >= local_begin) & (block <= query_blocks)
            allowed = (
                (q_ids < queries)
                & (block <= query_blocks)
                & ~is_prefix
                & ~is_local
                & (block * BLOCK_SIZE < kv_tokens)
            )
            score = tl.where(allowed, score, -float("inf"))
            output = (
                head * stride_ch
                + q_ids * stride_cq
                + partition_id * stride_cp
                + slot
            )
            tl.store(candidate_score_ptr + output, score, mask=q_ids < queries)
            tl.store(
                candidate_index_ptr + output,
                tl.where(allowed, block, -1),
                mask=q_ids < queries,
            )


    @triton.jit
    def _global_topk_kernel(
        candidate_score_ptr,
        candidate_index_ptr,
        route_ptr,
        route_score_ptr,
        queries,
        candidates,
        stride_ch: tl.constexpr,
        stride_cq: tl.constexpr,
        stride_rh: tl.constexpr,
        stride_rq: tl.constexpr,
        TOPK: tl.constexpr,
        BLOCK_CANDIDATES: tl.constexpr,
    ):
        query = tl.program_id(0)
        head = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_CANDIDATES)
        source = head * stride_ch + query * stride_cq + offsets
        scores = tl.load(
            candidate_score_ptr + source,
            mask=offsets < candidates,
            other=-float("inf"),
        )
        indices = tl.load(
            candidate_index_ptr + source,
            mask=offsets < candidates,
            other=-1,
        )
        for rank in tl.static_range(0, TOPK):
            best_score = tl.max(scores, axis=0)
            best_offset = tl.argmax(scores, axis=0)
            best_index = tl.sum(
                tl.where(offsets == best_offset, indices, 0), axis=0
            )
            valid = best_score != -float("inf")
            destination = head * stride_rh + query * stride_rq + rank
            tl.store(route_ptr + destination, tl.where(valid, best_index, -1))
            tl.store(
                route_score_ptr + destination,
                tl.where(valid, best_score, -float("inf")),
            )
            scores = tl.where(offsets == best_offset, -float("inf"), scores)


    @triton.jit
    def _fixed_attention_tiled_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        lse_ptr,
        queries,
        kv_tokens,
        q_offset,
        scale,
        stride_qt: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_kt: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_vt: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_ot: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_lt: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        Q_HEADS_PER_KV: tl.constexpr,
        PREFIX_BLOCKS: tl.constexpr,
        LOCAL_BLOCKS: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        query_tile = tl.program_id(0)
        q_head = tl.program_id(1)
        q_ids = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        d_ids = tl.arange(0, HEAD_DIM)
        active = q_ids < queries
        q = tl.load(
            q_ptr + q_ids[:, None] * stride_qt + q_head * stride_qh + d_ids[None, :],
            mask=active[:, None],
            other=0.0,
        )
        kv_head = q_head // Q_HEADS_PER_KV
        positions = q_offset + q_ids
        query_block = (q_offset + query_tile * BLOCK_M) // BLOCK_SIZE
        local_begin = tl.maximum(PREFIX_BLOCKS, query_block - LOCAL_BLOCKS + 1)
        local_count = tl.maximum(0, query_block - local_begin + 1)
        accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
        running_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        log2e = 1.4426950408889634

        for support_slot in tl.static_range(0, PREFIX_BLOCKS + LOCAL_BLOCKS):
            if support_slot < PREFIX_BLOCKS:
                block = support_slot
                valid_block = block <= query_block
            else:
                local_slot = support_slot - PREFIX_BLOCKS
                block = local_begin + local_slot
                valid_block = local_slot < local_count
            token_ids = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            token_in_kv = valid_block & (token_ids >= 0) & (token_ids < kv_tokens)
            key = tl.load(
                k_ptr
                + token_ids[:, None] * stride_kt
                + kv_head * stride_kh
                + d_ids[None, :],
                mask=token_in_kv[:, None],
                other=0.0,
            )
            value = tl.load(
                v_ptr
                + token_ids[:, None] * stride_vt
                + kv_head * stride_vh
                + d_ids[None, :],
                mask=token_in_kv[:, None],
                other=0.0,
            )
            token_valid = (
                active[:, None]
                & token_in_kv[None, :]
                & (token_ids[None, :] <= positions[:, None])
            )
            score = tl.dot(q, tl.trans(key), out_dtype=tl.float32) * scale
            score = tl.where(token_valid, score, -float("inf"))
            block_max = tl.max(score, axis=1)
            has_tokens = tl.sum(token_valid.to(tl.int32), axis=1) > 0
            updated_max = tl.maximum(running_max, block_max)
            alpha = tl.where(
                has_tokens,
                tl.exp2((running_max - updated_max) * log2e),
                1.0,
            )
            probability = tl.where(
                has_tokens[:, None],
                tl.exp2((score - updated_max[:, None]) * log2e),
                0.0,
            )
            running_sum = running_sum * alpha + tl.sum(probability, axis=1)
            accumulator = accumulator * alpha[:, None] + tl.dot(
                probability.to(tl.bfloat16), value, out_dtype=tl.float32
            )
            running_max = tl.where(has_tokens, updated_max, running_max)

        normalized = accumulator / running_sum[:, None]
        lse = running_max + tl.log(running_sum)
        tl.store(
            output_ptr
            + q_ids[:, None] * stride_ot
            + q_head * stride_oh
            + d_ids[None, :],
            normalized,
            mask=active[:, None],
        )
        tl.store(lse_ptr + q_ids * stride_lt + q_head, lse, mask=active)


    @triton.jit
    def _dynamic_partial_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        route_ptr,
        order_ptr,
        partial_output_ptr,
        partial_lse_ptr,
        queries,
        kv_tokens,
        q_offset,
        scale,
        stride_qt: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_kt: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_vt: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_po_rel: tl.constexpr,
        stride_po_head: tl.constexpr,
        stride_pl_rel: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        Q_HEADS_PER_KV: tl.constexpr,
        TOPK: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        work = tl.program_id(0)
        relation = tl.load(order_ptr + work)
        query_and_head = relation // TOPK
        query = query_and_head % queries
        kv_head = query_and_head // queries
        block = tl.load(route_ptr + relation)
        row = tl.arange(0, BLOCK_M)
        d_ids = tl.arange(0, HEAD_DIM)
        active_row = row < Q_HEADS_PER_KV
        q_head = kv_head * Q_HEADS_PER_KV + row
        q = tl.load(
            q_ptr + query * stride_qt + q_head[:, None] * stride_qh + d_ids[None, :],
            mask=active_row[:, None],
            other=0.0,
        )
        token_ids = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        position = q_offset + query
        token_valid = (
            (block >= 0)
            & (token_ids >= 0)
            & (token_ids < kv_tokens)
            & (token_ids <= position)
        )
        key = tl.load(
            k_ptr
            + token_ids[:, None] * stride_kt
            + kv_head * stride_kh
            + d_ids[None, :],
            mask=token_valid[:, None],
            other=0.0,
        )
        value = tl.load(
            v_ptr
            + token_ids[:, None] * stride_vt
            + kv_head * stride_vh
            + d_ids[None, :],
            mask=token_valid[:, None],
            other=0.0,
        )
        score = tl.dot(q, tl.trans(key), out_dtype=tl.float32) * scale
        score = tl.where(active_row[:, None] & token_valid[None, :], score, -float("inf"))
        maximum = tl.max(score, axis=1)
        valid = (block >= 0) & (tl.sum(token_valid.to(tl.int32), axis=0) > 0)
        probability = tl.where(
            valid,
            tl.exp2((score - maximum[:, None]) * 1.4426950408889634),
            0.0,
        )
        denominator = tl.sum(probability, axis=1)
        output = tl.dot(probability.to(tl.bfloat16), value, out_dtype=tl.float32)
        output = output / denominator[:, None]
        lse = maximum + tl.log(denominator)
        output = tl.where(valid, output, 0.0)
        lse = tl.where(valid & active_row, lse, -float("inf"))
        tl.store(
            partial_output_ptr
            + relation * stride_po_rel
            + row[:, None] * stride_po_head
            + d_ids[None, :],
            output,
            mask=active_row[:, None],
        )
        tl.store(
            partial_lse_ptr + relation * stride_pl_rel + row,
            lse,
            mask=active_row,
        )


    @triton.jit
    def _dynamic_grouped_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        route_ptr,
        order_ptr,
        partial_output_ptr,
        partial_lse_ptr,
        queries,
        kv_tokens,
        q_offset,
        scale,
        stride_qt: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_kt: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_vt: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_po_rel: tl.constexpr,
        stride_po_head: tl.constexpr,
        stride_pl_rel: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        Q_HEADS_PER_KV: tl.constexpr,
        TOPK: tl.constexpr,
        GROUP_QUERIES: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        chunk = tl.program_id(0)
        row = tl.arange(0, BLOCK_M)
        d_ids = tl.arange(0, HEAD_DIM)
        query_slot = row // Q_HEADS_PER_KV
        group_row = row % Q_HEADS_PER_KV
        relation = tl.load(order_ptr + chunk * GROUP_QUERIES + query_slot)
        query_and_head = relation // TOPK
        query = query_and_head % queries
        kv_head = query_and_head // queries
        block = tl.load(route_ptr + relation)
        first_relation = tl.load(order_ptr + chunk * GROUP_QUERIES)
        first_query_and_head = first_relation // TOPK
        first_kv_head = first_query_and_head // queries
        first_block = tl.load(route_ptr + first_relation)
        same_chunk = tl.sum(
            ((kv_head == first_kv_head) & (block == first_block)).to(tl.int32),
            axis=0,
        ) == BLOCK_M
        q_head = kv_head * Q_HEADS_PER_KV + group_row
        q = tl.load(
            q_ptr + query[:, None] * stride_qt + q_head[:, None] * stride_qh + d_ids[None, :]
        )
        token_ids = first_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        token_valid = (
            same_chunk
            & (first_block >= 0)
            & (token_ids >= 0)
            & (token_ids < kv_tokens)
        )
        key = tl.load(
            k_ptr
            + token_ids[:, None] * stride_kt
            + first_kv_head * stride_kh
            + d_ids[None, :],
            mask=token_valid[:, None],
            other=0.0,
        )
        value = tl.load(
            v_ptr
            + token_ids[:, None] * stride_vt
            + first_kv_head * stride_vh
            + d_ids[None, :],
            mask=token_valid[:, None],
            other=0.0,
        )
        position = q_offset + query
        score_valid = token_valid[None, :] & (token_ids[None, :] <= position[:, None])
        score = tl.dot(q, tl.trans(key), out_dtype=tl.float32) * scale
        score = tl.where(score_valid, score, -float("inf"))
        maximum = tl.max(score, axis=1)
        valid = same_chunk & (first_block >= 0)
        probability = tl.where(
            valid,
            tl.exp2((score - maximum[:, None]) * 1.4426950408889634),
            0.0,
        )
        denominator = tl.sum(probability, axis=1)
        output = tl.dot(probability.to(tl.bfloat16), value, out_dtype=tl.float32)
        output = output / denominator[:, None]
        lse = maximum + tl.log(denominator)
        output = tl.where(valid, output, 0.0)
        lse = tl.where(valid, lse, -float("inf"))
        tl.store(
            partial_output_ptr
            + relation[:, None] * stride_po_rel
            + group_row[:, None] * stride_po_head
            + d_ids[None, :],
            output,
        )
        tl.store(
            partial_lse_ptr + relation * stride_pl_rel + group_row,
            lse,
        )


    @triton.jit
    def _merge_fixed_dynamic_kernel(
        fixed_output_ptr,
        fixed_lse_ptr,
        partial_output_ptr,
        partial_lse_ptr,
        output_ptr,
        lse_ptr,
        queries,
        stride_fo_t: tl.constexpr,
        stride_fo_h: tl.constexpr,
        stride_fl_t: tl.constexpr,
        stride_po_rel: tl.constexpr,
        stride_po_head: tl.constexpr,
        stride_pl_rel: tl.constexpr,
        stride_o_t: tl.constexpr,
        stride_o_h: tl.constexpr,
        stride_l_t: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        Q_HEADS_PER_KV: tl.constexpr,
        TOPK: tl.constexpr,
    ):
        query = tl.program_id(0)
        q_head = tl.program_id(1)
        d_ids = tl.arange(0, HEAD_DIM)
        kv_head = q_head // Q_HEADS_PER_KV
        group_row = q_head % Q_HEADS_PER_KV
        accumulator = tl.load(
            fixed_output_ptr
            + query * stride_fo_t
            + q_head * stride_fo_h
            + d_ids
        ).to(tl.float32)
        running_max = tl.load(fixed_lse_ptr + query * stride_fl_t + q_head)
        running_sum = 1.0
        log2e = 1.4426950408889634
        for slot in tl.static_range(0, TOPK):
            relation = (kv_head * queries + query) * TOPK + slot
            candidate_lse = tl.load(
                partial_lse_ptr + relation * stride_pl_rel + group_row
            )
            candidate = tl.load(
                partial_output_ptr
                + relation * stride_po_rel
                + group_row * stride_po_head
                + d_ids
            ).to(tl.float32)
            valid = candidate_lse != -float("inf")
            updated_max = tl.maximum(running_max, candidate_lse)
            alpha = tl.exp2((running_max - updated_max) * log2e)
            beta = tl.where(
                valid,
                tl.exp2((candidate_lse - updated_max) * log2e),
                0.0,
            )
            accumulator = accumulator * alpha + candidate * beta
            running_sum = running_sum * alpha + beta
            running_max = updated_max
        output = accumulator / running_sum
        merged_lse = running_max + tl.log(running_sum)
        tl.store(
            output_ptr + query * stride_o_t + q_head * stride_o_h + d_ids,
            output,
        )
        tl.store(lse_ptr + query * stride_l_t + q_head, merged_lse)


def route_topk(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    *,
    dynamic_topk: int = 16,
    prefix_blocks: int = 8,
    local_blocks: int = 8,
    block_size: int = 128,
    score_scale: float | None = None,
    query_offset: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact dynamic block routes and their scaled scores."""
    _require_triton()
    if not index_q.is_cuda or not index_k.is_cuda:
        raise ValueError("GPU router requires CUDA tensors")
    if index_q.ndim != 3 or index_k.ndim != 3:
        raise ValueError("index_q/index_k must be [tokens, heads, dim]")
    if index_k.shape[1] not in (1, index_q.shape[1]) or index_q.shape[-1] != 128 or index_k.shape[-1] != 128:
        raise ValueError("Triton router requires shared or matching key heads and D=128")
    if dynamic_topk != 16 or block_size != 128:
        raise ValueError("Triton router requires block_size=128 and dynamic_topk=16")
    if index_q.dtype not in (torch.float16, torch.bfloat16) or index_k.dtype != index_q.dtype:
        raise TypeError("index_q/index_k must share FP16 or BF16 dtype")
    queries, heads, head_dim = index_q.shape
    score_scale = head_dim ** -0.5 if score_scale is None else score_scale
    kv_tokens = index_k.shape[0]
    query_offset = max(kv_tokens - queries, 0) if query_offset is None else query_offset
    partition_blocks = 16
    num_blocks = (kv_tokens + block_size - 1) // block_size
    partitions = (num_blocks + partition_blocks - 1) // partition_blocks
    candidate_shape = (heads, queries, partitions, partition_blocks)
    candidate_scores = torch.empty(candidate_shape, device=index_q.device, dtype=torch.float32)
    candidate_indices = torch.empty(candidate_shape, device=index_q.device, dtype=torch.int32)
    block_q = 64
    _partition_score_kernel[
        (partitions, triton.cdiv(queries, block_q), heads)
    ](
        index_q,
        index_k,
        candidate_scores,
        candidate_indices,
        queries,
        kv_tokens,
        query_offset,
        float(score_scale),
        index_q.stride(0),
        index_q.stride(1),
        index_k.stride(0),
        0 if index_k.shape[1] == 1 else index_k.stride(1),
        candidate_scores.stride(0),
        candidate_scores.stride(1),
        candidate_scores.stride(2),
        BLOCK_SIZE=block_size,
        HEAD_DIM=128,
        PREFIX_BLOCKS=prefix_blocks,
        LOCAL_BLOCKS=local_blocks,
        PARTITION_BLOCKS=partition_blocks,
        BLOCK_Q=block_q,
        num_warps=4,
        num_stages=2,
    )
    route = torch.empty((heads, queries, dynamic_topk), device=index_q.device, dtype=torch.int32)
    route_scores = torch.empty((heads, queries, dynamic_topk), device=index_q.device, dtype=torch.float32)
    candidates = partitions * partition_blocks
    block_candidates = triton.next_power_of_2(candidates)
    _global_topk_kernel[(queries, heads)](
        candidate_scores,
        candidate_indices,
        route,
        route_scores,
        queries,
        candidates,
        candidate_scores.stride(0),
        candidate_scores.stride(1),
        route.stride(0),
        route.stride(1),
        TOPK=dynamic_topk,
        BLOCK_CANDIDATES=block_candidates,
        num_warps=4,
    )
    return route, route_scores


def _require_triton() -> None:
    if triton is None:
        raise ImportError("Triton is required for the precision baseline") from _TRITON_IMPORT_ERROR


def _validate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    route: torch.Tensor | None = None,
):
    _require_triton()
    tensors = (q, k, v) if route is None else (q, k, v, route)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("Triton attention requires CUDA tensors")
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("q=[Q,Hq,D], k/v=[K,Hkv,D] are required")
    queries, q_heads, dim = q.shape
    kv_tokens, kv_heads, kv_dim = k.shape
    if min(queries, kv_tokens, q_heads, kv_heads) <= 0 or queries > kv_tokens:
        raise ValueError("Triton attention requires nonempty heads and 0 < Q <= KV")
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("Triton attention tensors must be on the same CUDA device")
    if any(tensor.stride(-1) != 1 for tensor in (q, k, v)):
        raise ValueError("Triton attention requires contiguous head dimensions")
    if dim != 128 or kv_dim != 128 or q_heads % kv_heads:
        raise ValueError("Triton attention requires D=128 and integral GQA groups")
    group = q_heads // kv_heads
    if group > 16:
        raise ValueError("Triton attention requires a GQA group no larger than 16")
    if route is not None and route.shape != (kv_heads, queries, 16):
        raise ValueError("Triton attention requires route [Hkv,Q,16]")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("Triton attention currently requires BF16 q/k/v")
    return queries, q_heads, kv_tokens, kv_heads, group


def fixed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    prefix_blocks: int = 8,
    local_blocks: int = 8,
    block_size: int = 128,
    query_offset: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the fixed path with K/V reuse across a 128-query tile."""
    queries, q_heads, kv_tokens, _, group = _validate(q, k, v)
    query_offset = max(kv_tokens - queries, 0) if query_offset is None else query_offset
    if query_offset % block_size:
        raise ValueError("Triton attention fixed tiling requires a block-aligned query offset")
    output = torch.empty_like(q)
    lse = torch.empty((queries, q_heads), device=q.device, dtype=torch.float32)
    block_m = 128
    _fixed_attention_tiled_kernel[(triton.cdiv(queries, block_m), q_heads)](
        q, k, v, output, lse, queries, kv_tokens, query_offset, float(128 ** -0.5),
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        output.stride(0), output.stride(1), lse.stride(0),
        BLOCK_SIZE=block_size, HEAD_DIM=128, Q_HEADS_PER_KV=group,
        PREFIX_BLOCKS=prefix_blocks, LOCAL_BLOCKS=local_blocks, BLOCK_M=block_m,
        num_warps=8, num_stages=2,
    )
    return output, lse


def _build_kv_major_order(route: torch.Tensor, kv_tokens: int, block_size: int = 128):
    """Return route-entry indices sorted by KV head and KV block."""
    kv_heads, queries, topk = route.shape
    num_blocks = (kv_tokens + block_size - 1) // block_size
    head = torch.arange(kv_heads, device=route.device, dtype=torch.int64).view(-1, 1, 1)
    keys = head * num_blocks + route.to(torch.int64).clamp_min(0)
    invalid_key = kv_heads * num_blocks
    keys = torch.where(route >= 0, keys, invalid_key)
    return torch.argsort(keys.reshape(-1), stable=False)


def build_kv_grouped_schedule(
    route: torch.Tensor, kv_tokens: int, block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build sixteen-query KV-major groups plus a small boundary fallback list."""
    order = _build_kv_major_order(route, kv_tokens, block_size)
    if order.numel() % 16:
        raise ValueError("route entry count must be divisible by sixteen")
    chunks = order.view(-1, 16)
    flat_route = route.reshape(-1)
    blocks = flat_route[chunks]
    queries = route.shape[1]
    heads = chunks // (queries * route.shape[2])
    same = (blocks == blocks[:, :1]).all(dim=1) & (heads == heads[:, :1]).all(dim=1)
    boundary = chunks[~same].reshape(-1).contiguous()
    return order, boundary


def dynamic_partials(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    route: torch.Tensor,
    schedule: tuple[torch.Tensor, torch.Tensor],
    *,
    block_size: int = 128,
    query_offset: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate sixteen same-block query groups per CTA, with exact boundaries."""
    queries, _, kv_tokens, _, group = _validate(q, k, v, route)
    if group != 8:
        raise ValueError("Triton attention grouped dynamic path currently requires GQA group 8")
    order, boundary = schedule
    relations = route.numel()
    query_offset = max(kv_tokens - queries, 0) if query_offset is None else query_offset
    partial_output = torch.empty(
        (relations, group, 128), device=q.device, dtype=torch.bfloat16
    )
    partial_lse = torch.empty((relations, group), device=q.device, dtype=torch.float32)
    _dynamic_grouped_kernel[(relations // 16,)](
        q, k, v, route, order, partial_output, partial_lse,
        queries, kv_tokens, query_offset, float(128 ** -0.5),
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        partial_output.stride(0), partial_output.stride(1), partial_lse.stride(0),
        BLOCK_SIZE=block_size, HEAD_DIM=128, Q_HEADS_PER_KV=group, TOPK=16,
        GROUP_QUERIES=16, BLOCK_M=128, num_warps=8, num_stages=2,
    )
    if boundary.numel():
        _dynamic_partial_kernel[(boundary.numel(),)](
            q, k, v, route, boundary, partial_output, partial_lse,
            queries, kv_tokens, query_offset, float(128 ** -0.5),
            q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
            partial_output.stride(0), partial_output.stride(1), partial_lse.stride(0),
            BLOCK_SIZE=block_size, HEAD_DIM=128, Q_HEADS_PER_KV=group, TOPK=16,
            BLOCK_M=16, num_warps=8, num_stages=2,
        )
    return partial_output, partial_lse


def merge_states(
    fixed_output: torch.Tensor,
    fixed_lse: torch.Tensor,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    queries, q_heads, _ = fixed_output.shape
    group = q_heads // kv_heads
    output = torch.empty_like(fixed_output)
    lse = torch.empty_like(fixed_lse)
    _merge_fixed_dynamic_kernel[(queries, q_heads)](
        fixed_output, fixed_lse, partial_output, partial_lse, output, lse, queries,
        fixed_output.stride(0), fixed_output.stride(1), fixed_lse.stride(0),
        partial_output.stride(0), partial_output.stride(1), partial_lse.stride(0),
        output.stride(0), output.stride(1), lse.stride(0),
        HEAD_DIM=128, Q_HEADS_PER_KV=group, TOPK=16, num_warps=4,
    )
    return output, lse


def sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    route: torch.Tensor,
    *,
    schedule: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute tiled fixed attention and KV-major dynamic attention."""
    fixed_output, fixed_lse = fixed_attention(q, k, v)
    if schedule is None:
        schedule = build_kv_grouped_schedule(route, k.shape[0])
    partial_output, partial_lse = dynamic_partials(
        q, k, v, route, schedule
    )
    return merge_states(
        fixed_output, fixed_lse, partial_output, partial_lse, k.shape[1]
    )


def triton_pipeline(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    fixed_stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Overlap the fixed path with exact route generation on separate streams."""
    queries, q_heads, kv_tokens, kv_heads, group = _validate(q, k, v)
    if (q_heads, kv_heads, group) != (32, 4, 8):
        raise ValueError("Maxwell Block Sparse Attention Triton prefill requires Hq/Hkv/GQA=32/4/8")
    if index_q.shape != (queries, 4, 128) or index_k.shape != (kv_tokens, 1, 128):
        raise ValueError("index_q=[Q,4,128] and shared index_k=[KV,1,128] are required")
    if any(tensor.device != q.device for tensor in (index_q, index_k)):
        raise ValueError("Indexer and main tensors must share a CUDA device")
    if any(tensor.dtype != q.dtype or tensor.stride(-1) != 1 for tensor in (index_q, index_k)):
        raise ValueError("Indexer tensors must be BF16 with contiguous head dimensions")
    if (kv_tokens - queries) % 128:
        raise ValueError("Triton attention fixed tiling requires a block-aligned query offset")
    current = torch.cuda.current_stream(q.device)
    fixed_stream = fixed_stream or torch.cuda.Stream(device=q.device)
    inputs_ready = torch.cuda.Event()
    fixed_done = torch.cuda.Event()
    inputs_ready.record(current)
    with torch.cuda.stream(fixed_stream):
        fixed_stream.wait_event(inputs_ready)
        fixed_output, fixed_lse = fixed_attention(q, k, v)
        fixed_done.record(fixed_stream)
    route, _ = route_topk(index_q, index_k)
    schedule = build_kv_grouped_schedule(route, k.shape[0])
    partial_output, partial_lse = dynamic_partials(
        q, k, v, route, schedule
    )
    current.wait_event(fixed_done)
    output, lse = merge_states(
        fixed_output, fixed_lse, partial_output, partial_lse, k.shape[1]
    )
    return output, lse, route
