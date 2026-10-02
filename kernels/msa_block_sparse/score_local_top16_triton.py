# SPDX-License-Identifier: Apache-2.0
"""C032/C038 finalizer for Score-owned exact partition Top-16 records.

The experimental MSA Score variant stores, for each 1024-block partition and
query row, the high and low 32-bit words of 16 packed signed-64 total-order
keys in the first planes of the existing ``[Hkv, Ktiles, Q]`` Score allocation.
C038 instead stores the ordered-score word and logical block id in those
planes. This kernel normalizes either representation to the same packed key,
merges the exact local lists, and assembles the unchanged Top-32 route.
"""

from __future__ import annotations

import torch

from msa_block_sparse._vllm_compat import tl, triton


SPARSE_BLOCK_SIZE = 128
PARTITION_BLOCKS = 1024
LOCAL_CAPACITY = 16
PLANES_PER_PARTITION = 32


@triton.jit(do_not_specialize_on_alignment=["prefix_lens"])
def _merge_score_local_top16_kernel(
    storage_ptr,
    route_ptr,
    cu_seqlens,
    prefix_lens,
    sample_interval: tl.constexpr,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    num_partitions,
    stride_s_h,
    stride_s_k,
    stride_s_q,
    stride_r_h,
    stride_r_q,
    stride_r_t,
    PARTITION_BLOCKS_T: tl.constexpr,
    PLANES_PER_PARTITION_T: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    LOCAL_CAPACITY_T: tl.constexpr,
    PACKED_KEYS_T: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    seq_start = tl.load(cu_seqlens + pid_b)
    q_count = tl.load(cu_seqlens + pid_b + 1) - seq_start
    if pid_q >= q_count:
        return

    prefix_len = tl.load(prefix_lens + pid_b)
    valid_blocks = (prefix_len + pid_q * sample_interval + block_size) // block_size
    prefix_count = min(init_blocks, valid_blocks)
    local_start = max(0, valid_blocks - local_blocks)
    nonoverlap_local_start = max(local_start, prefix_count)
    local_count = valid_blocks - nonoverlap_local_start
    target_count = min(topk, valid_blocks)
    fixed_count = prefix_count + local_count
    dynamic_count = target_count - fixed_count
    valid_partitions = tl.cdiv(valid_blocks, PARTITION_BLOCKS_T)

    off_k = tl.arange(0, BLOCK_SIZE_K)
    best_keys = tl.full((LOCAL_CAPACITY_T,), -9223372036854775807, tl.int64)
    candidate_count = num_partitions * LOCAL_CAPACITY_T
    row_base = (
        storage_ptr
        + pid_h * stride_s_h
        + (seq_start + pid_q) * stride_s_q
    )
    for flat_base in tl.range(0, candidate_count, BLOCK_SIZE_K):
        flat = flat_base + off_k
        partition = flat // LOCAL_CAPACITY_T
        slot = flat - partition * LOCAL_CAPACITY_T
        valid = (flat < candidate_count) & (partition < valid_partitions)
        plane = partition * PLANES_PER_PARTITION_T + slot
        first = tl.load(
            row_base + plane * stride_s_k,
            mask=valid,
            other=0.0,
        ).to(tl.int32, bitcast=True)
        second = tl.load(
            row_base + (plane + LOCAL_CAPACITY_T) * stride_s_k,
            mask=valid,
            other=0.0,
        ).to(tl.int32, bitcast=True)
        if PACKED_KEYS_T:
            packed = (first.to(tl.int64) << 32) | (second.to(tl.int64) & 0xFFFFFFFF)
        else:
            tie_key = 0xFFFFFFFF - second.to(tl.int64)
            packed = (first.to(tl.int64) << 32) | (tie_key & 0xFFFFFFFF)
        packed = tl.where(valid, packed, -9223372036854775807)
        chunk_keys = tl.topk(packed, LOCAL_CAPACITY_T)
        best_keys = tl.topk(
            tl.cat(best_keys, chunk_keys, can_reorder=True), LOCAL_CAPACITY_T
        )

    selected_ids = (4294967295 - (best_keys & 0xFFFFFFFF)).to(tl.int32)
    off_t = tl.arange(0, LOCAL_CAPACITY_T)
    route_base = (
        route_ptr
        + pid_h * stride_r_h
        + (seq_start + pid_q) * stride_r_q
    )
    tl.store(
        route_base + off_t * stride_r_t,
        off_t.to(route_base.dtype.element_ty),
        mask=off_t < prefix_count,
    )
    tl.store(
        route_base + (prefix_count + off_t) * stride_r_t,
        (nonoverlap_local_start + off_t).to(route_base.dtype.element_ty),
        mask=off_t < local_count,
    )
    tl.store(
        route_base + (fixed_count + off_t) * stride_r_t,
        selected_ids.to(route_base.dtype.element_ty),
        mask=off_t < dynamic_count,
    )
    tl.store(
        route_base + off_t * stride_r_t,
        -1,
        mask=(off_t >= target_count) & (off_t < topk),
    )
    tl.store(
        route_base + (LOCAL_CAPACITY_T + off_t) * stride_r_t,
        -1,
        mask=(LOCAL_CAPACITY_T + off_t >= target_count)
        & (LOCAL_CAPACITY_T + off_t < topk),
    )


def candidate_partitions(
    storage: torch.Tensor, partition_blocks: int = PARTITION_BLOCKS
) -> int:
    if storage.ndim != 3:
        raise ValueError("storage must be [Hkv,Ktiles,total_q]")
    if partition_blocks not in (1024, 2048, 4096, 8192):
        raise ValueError("partition_blocks must be 1024, 2048, 4096, or 8192")
    partitions = triton.cdiv(int(storage.shape[1]), partition_blocks)
    if partitions * PLANES_PER_PARTITION > int(storage.shape[1]):
        raise ValueError("Score allocation is too small for compact candidate planes")
    return partitions


@torch.no_grad()
def merge_score_local_top16_route(
    storage: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    out: torch.Tensor,
    topk: int = 32,
    prefix_blocks: int = 8,
    local_blocks: int = 8,
    packed_keys: bool = True,
    partition_blocks: int = PARTITION_BLOCKS,
) -> torch.Tensor:
    if (
        storage.ndim != 3
        or storage.dtype != torch.float32
        or not storage.is_cuda
        or int(storage.shape[0]) != 4
    ):
        raise ValueError("storage must be CUDA FP32 [4,Ktiles,total_q]")
    if topk != 32 or prefix_blocks != 8 or local_blocks != 8:
        raise ValueError("C032 is specialized to provisional 8+8+16 Top-32")
    expected = (4, int(storage.shape[2]), 32)
    if (
        tuple(out.shape) != expected
        or out.dtype != torch.int32
        or out.device != storage.device
        or not out.is_contiguous()
    ):
        raise ValueError(f"out must be contiguous CUDA int32 {expected}")
    partitions = candidate_partitions(storage, partition_blocks)
    batch = int(cu_seqlens_q.numel()) - 1
    grid = (max_query_len, batch, 4)
    block_size_k = triton.next_power_of_2(min(64, partitions * LOCAL_CAPACITY))
    _merge_score_local_top16_kernel[grid](
        storage,
        out,
        cu_seqlens_q,
        prefix_lens,
        1,
        SPARSE_BLOCK_SIZE,
        topk,
        prefix_blocks,
        local_blocks,
        partitions,
        storage.stride(0),
        storage.stride(1),
        storage.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        PARTITION_BLOCKS_T=partition_blocks,
        PLANES_PER_PARTITION_T=PLANES_PER_PARTITION,
        BLOCK_SIZE_K=block_size_k,
        LOCAL_CAPACITY_T=LOCAL_CAPACITY,
        PACKED_KEYS_T=packed_keys,
        num_warps=1,
        num_stages=2,
    )
    return out
