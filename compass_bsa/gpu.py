"""Independent Triton GPU kernels for Opt4 revision 1.

The router uses disjoint 16-block score partitions.  Each score program emits
the exact local Top-16 (all members of a 16-block partition), and a second
kernel performs the exact global Top-16 merge.  The attention kernel is
Q-centric and evaluates fixed and dynamic supports in one physical launch.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError as error:  # pragma: no cover - exercised on GPU hosts
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
    def _dual_path_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        route_ptr,
        output_ptr,
        lse_ptr,
        queries,
        kv_tokens,
        q_offset,
        softmax_scale,
        stride_qt: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_kt: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_rh: tl.constexpr,
        stride_rq: tl.constexpr,
        stride_ot: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_lt: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        Q_HEADS_PER_KV: tl.constexpr,
        PREFIX_BLOCKS: tl.constexpr,
        LOCAL_BLOCKS: tl.constexpr,
        DYNAMIC_TOPK: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        query = tl.program_id(0)
        kv_head = tl.program_id(1)
        row = tl.arange(0, BLOCK_M)
        d_ids = tl.arange(0, HEAD_DIM)
        q_head = kv_head * Q_HEADS_PER_KV + row
        active_row = row < Q_HEADS_PER_KV
        q = tl.load(
            q_ptr + query * stride_qt + q_head[:, None] * stride_qh + d_ids[None, :],
            mask=active_row[:, None] & (query < queries),
            other=0.0,
        )
        accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
        running_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        query_position = q_offset + query
        query_block = query_position // BLOCK_SIZE
        local_begin = tl.maximum(PREFIX_BLOCKS, query_block - LOCAL_BLOCKS + 1)
        local_count = tl.maximum(0, query_block - local_begin + 1)
        log2e = 1.4426950408889634

        for support_slot in tl.static_range(
            0, PREFIX_BLOCKS + LOCAL_BLOCKS + DYNAMIC_TOPK
        ):
            if support_slot < PREFIX_BLOCKS:
                block = support_slot
                valid_block = block <= query_block
            elif support_slot < PREFIX_BLOCKS + LOCAL_BLOCKS:
                local_slot = support_slot - PREFIX_BLOCKS
                block = local_begin + local_slot
                valid_block = local_slot < local_count
            else:
                dynamic_slot = support_slot - PREFIX_BLOCKS - LOCAL_BLOCKS
                block = tl.load(
                    route_ptr
                    + kv_head * stride_rh
                    + query * stride_rq
                    + dynamic_slot
                )
                valid_block = block >= 0

            token_ids = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            token_valid = (
                valid_block
                & (token_ids >= 0)
                & (token_ids < kv_tokens)
                & (token_ids <= query_position)
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
                + token_ids[:, None] * stride_kt
                + kv_head * stride_kh
                + d_ids[None, :],
                mask=token_valid[:, None],
                other=0.0,
            )
            score = tl.dot(q, tl.trans(key), out_dtype=tl.float32) * softmax_scale
            score = tl.where(
                active_row[:, None] & token_valid[None, :],
                score,
                -float("inf"),
            )
            block_max = tl.max(score, axis=1)
            has_tokens = valid_block & (tl.sum(token_valid, axis=0) > 0)
            updated_max = tl.maximum(running_max, block_max)
            alpha = tl.where(
                has_tokens,
                tl.exp2((running_max - updated_max) * log2e),
                1.0,
            )
            probability = tl.where(
                has_tokens,
                tl.exp2((score - updated_max[:, None]) * log2e),
                0.0,
            )
            running_sum = running_sum * alpha + tl.sum(probability, axis=1)
            accumulator = accumulator * alpha[:, None] + tl.dot(
                probability.to(value.dtype), value, out_dtype=tl.float32
            )
            running_max = tl.where(has_tokens, updated_max, running_max)

        output = accumulator / running_sum[:, None]
        lse = running_max + tl.log(running_sum)
        tl.store(
            output_ptr
            + query * stride_ot
            + q_head[:, None] * stride_oh
            + d_ids[None, :],
            output,
            mask=active_row[:, None] & (query < queries),
        )
        tl.store(
            lse_ptr + query * stride_lt + q_head,
            lse,
            mask=active_row & (query < queries),
        )


def _require_triton() -> None:
    if triton is None:
        raise ImportError("Triton is required for Opt4 GPU kernels") from _TRITON_IMPORT_ERROR


def gpu_hierarchical_exact_topk(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    *,
    dynamic_topk: int = 16,
    prefix_blocks: int = 8,
    local_blocks: int = 8,
    block_size: int = 128,
    score_scale: float = 1.0,
    query_offset: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the independent two-kernel GPU router."""
    _require_triton()
    if not index_q.is_cuda or not index_k.is_cuda:
        raise ValueError("GPU router requires CUDA tensors")
    if index_q.ndim != 3 or index_k.ndim != 3:
        raise ValueError("index_q/index_k must be [tokens, heads, dim]")
    if index_k.shape[1] not in (1, index_q.shape[1]) or index_q.shape[-1] != 128 or index_k.shape[-1] != 128:
        raise ValueError("revision 1 requires shared or matching key heads and D=128")
    if dynamic_topk != 16 or block_size != 128:
        raise ValueError("revision 1 requires block_size=128 and dynamic_topk=16")
    if index_q.dtype not in (torch.float16, torch.bfloat16) or index_k.dtype != index_q.dtype:
        raise TypeError("index_q/index_k must share FP16 or BF16 dtype")
    queries, heads, _ = index_q.shape
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


def gpu_dual_path_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dynamic_route: torch.Tensor,
    *,
    prefix_blocks: int = 8,
    local_blocks: int = 8,
    block_size: int = 128,
    query_offset: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fixed and dynamic sparse supports in one independent GPU kernel."""
    _require_triton()
    if not q.is_cuda or not k.is_cuda or not v.is_cuda or not dynamic_route.is_cuda:
        raise ValueError("GPU attention requires CUDA tensors")
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("q=[Q,Hq,D], k/v=[K,Hkv,D] are required")
    queries, q_heads, head_dim = q.shape
    kv_tokens, kv_heads, kv_dim = k.shape
    if head_dim != 128 or kv_dim != 128 or q_heads % kv_heads:
        raise ValueError("revision 1 requires D=128 and integral GQA groups")
    group = q_heads // kv_heads
    if group > 16 or dynamic_route.shape != (kv_heads, queries, 16):
        raise ValueError("revision 1 requires group<=16 and route [Hkv,Q,16]")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q/k/v must share FP16 or BF16 dtype")
    query_offset = max(kv_tokens - queries, 0) if query_offset is None else query_offset
    output = torch.empty_like(q)
    lse = torch.empty((queries, q_heads), device=q.device, dtype=torch.float32)
    _dual_path_attention_kernel[(queries, kv_heads)](
        q,
        k,
        v,
        dynamic_route,
        output,
        lse,
        queries,
        kv_tokens,
        query_offset,
        float(head_dim ** -0.5),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        dynamic_route.stride(0),
        dynamic_route.stride(1),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        Q_HEADS_PER_KV=group,
        PREFIX_BLOCKS=prefix_blocks,
        LOCAL_BLOCKS=local_blocks,
        DYNAMIC_TOPK=16,
        BLOCK_M=16,
        num_warps=8,
        num_stages=2,
    )
    return output, lse
