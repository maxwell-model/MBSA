# Copyright 2026 Maxwell Team.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reuse optimized MBSA Index kernels with private preparation buffers."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import cuda.bindings.driver as cuda
from cutlass.base_dsl import BaseDSL
from cutlass.base_dsl.arch import Arch
from .index_base import IndexPipelineState, stage_int32
from .src.sm100.router import indexer_local_top16, _local_top16_arguments
from .kernels.plan import SPEC, PrefillPolicy
from .page_map import page_operand_view
from .index_policy import select_partition_blocks, lean192_index_eligible
from .._index_options import resolve_lean192, validate_lean192


@dataclass
class OptimizedIndexPlan(IndexPipelineState):
    # Wrappers refer only to this binding state's tensors. Retired states keep
    # their own bundle until queued work completes; there is no global cache.
    _local_top16_args: tuple | None = None
    _local_top16_compiled: object | None = None
    _local_top16_key: tuple | None = None
    lean192_mode: str = "auto"
    lean192_bound: bool = False

    def run_score(self):
        """Launch the bound local selection ABI on the current CUDA stream.

        Falls back to the general Index stage when no prepared ABI exists. The stream is resolved
        at invocation, not cached at preparation.

        Returns:
            None; writes retained candidate/partition buffers.
        """
        if self._local_top16_args is None:
            return super().run_score()
        stream = cuda.CUstream(torch.cuda.current_stream(self.index_q.device).cuda_stream)
        self._local_top16_compiled(*self._local_top16_args, stream)


def lean192_compiler_supported():
    """Match the Indexer's actual compiler-target check during preparation."""
    return BaseDSL._get_dsl().get_arch_enum().is_family_of(Arch.sm_103f)


def bind_score_operands(plan):
    """Bind immutable launch metadata and select the single-partition direct-ID epilogue.

    Args:
        plan: Mutable OptimizedIndexPlan with validated current Index operands, map and candidate
            storage. Called only during preparation or staged rebind, outside capture.

    Returns:
        None; replaces only the staged plan's callable/key/argument bundle. Operand tensors remain
        strongly referenced by that state.
    """
    plan._local_top16_args = None
    plan._local_top16_compiled = None
    plan._local_top16_key = None
    plan.lean192_bound = False
    if (plan.attention.fixed_only or plan.mxfp6_router is not None
            or plan.cutedsl_packed_tile_metadata is not None):
        return
    blocks = (plan.cutedsl_logical_k_tokens + SPEC.block_size - 1) // SPEC.block_size
    partitions = (blocks + plan.candidate_partition_blocks - 1) // plan.candidate_partition_blocks
    if plan.cutedsl_partition_scores is None:
        shape = (plan.index_q.shape[1], partitions, plan.index_q.shape[0], SPEC.dynamic_topk)
        # Preserve the binding-owned ABI for both the direct single-partition
        # output and the ordinary partition/merge path.
        plan.cutedsl_partition_scores = torch.empty(shape, device=plan.index_q.device, dtype=torch.float32)
        plan.cutedsl_partition_ids = torch.empty(shape, device=plan.index_q.device, dtype=torch.int32)
    plan._local_top16_args = _local_top16_arguments(
        plan.index_q, plan.cutedsl_index_k, plan.cutedsl_index_pages,
        plan.cutedsl_partition_scores, plan.cutedsl_partition_ids,
        plan.candidate_storage.view(torch.int32), plan.cutedsl_logical_k_tokens, 2)
    plan._local_top16_compiled, plan._local_top16_key = indexer_local_top16(
        plan.index_q, plan.cutedsl_index_k,
        partition_blocks=plan.candidate_partition_blocks,
        page_indices=plan.cutedsl_index_pages,
        logical_k_tokens=plan.cutedsl_logical_k_tokens,
        use_page_table=plan.cutedsl_use_page_table,
        _lean192_early=plan._lean192_early_router,
        out_scores=plan.cutedsl_partition_scores,
        out_ids=plan.cutedsl_partition_ids,
        out_compact_ids=plan.candidate_storage.view(torch.int32),
        _prepared_args=plan._local_top16_args,
        _prepare_only=True,
        _fuse_single_partition_compact=partitions == 1,
    )
    plan.lean192_bound = bool(plan._lean192_early_router and lean192_compiler_supported())

def prepare_index(index_q, index_k, *, index_q_scales, index_page_indices,
                  candidate_storage, prefix_lens, attention, q_lens, kv_lens,
                  index_pages_identity, check_device_inputs, original_q,
                  original_k, original_v, contiguous_unpacked, lean192="auto"):
    """Allocate selection workspaces and bind the Index stage for a prepared attention plan.

    Args:
        index_q: CUDA Index queries [T,4,128], matching BF16/FP16/E4M3 Index K, or MXFP8 uint8/E4M3
            codes with index_q_scales. T is the packed query-token count.
        index_k: CUDA Index keys [Skv,1,128] or [pages,1,128,128] for floating formats; MXFP6 uses
            packed uint8 [pages,1,128,100] with page-separated codes/scales. Physical pages contain
            128 tokens.
        index_q_scales: Optional CUDA int32 [T,>=4] MXFP8 scale words, four little-endian UE8M0
            bytes per head. Required for MXFP6 Index K; otherwise None.
        index_page_indices: Optional CUDA int32 logical-to-physical Index page IDs, compact 1-D
            active IDs or request rows [B,>=ceil(KV/128)]. Optimized binds active rows directly:
            one adjacent contiguous storage span with a 16-byte base alignment for unpacked/MX
            or 4-byte base alignment for packed floating Index. Individual compact request rows
            need not be 16-byte aligned. Gaps/strides are rejected. None creates the backend
            default mapping.
        candidate_storage: Writable contiguous FP32 candidate backing [4,32,T], with packed ID bits
            consumed through int32 views by the selection/CSR ABI. Retained by the returned plan.
        prefix_lens: CUDA int32 per-query valid-page-prefix bounds used for dynamic selection.
        attention: Prepared attention plan supplying request prefixes, tile policy and partial
            dtype.
        q_lens: Host query-token lengths in packed request order.
        kv_lens: Host logical KV-token lengths, one per query request.
        index_pages_identity: True for known identity Index page IDs, False for mapped pages, or
            None to validate/detect when device checks are enabled.
        check_device_inputs: Whether to synchronize for Index page-value range/identity checks.
            False requires valid caller metadata.
        original_q: CUDA attention queries [T,32,128]. BF16 for BF16 K/V; the default Blackwell FP8
            path also accepts BF16 or E4M3FN Q with E4M3FN K/V. Optimized converts FP8 operands to
            owned BF16 buffers per run.
        original_k: CUDA attention keys [Skv,4,128] or paged [pages,4,128,128]. Must match v in
            dtype/layout/device; BF16, or E4M3FN on Blackwell with unit scales.
        original_v: CUDA attention values with the same shape, dtype and device as k. Read-only
            until the queued invocation completes.
        contiguous_unpacked: True for one unpacked request; False selects packed request metadata
            and map handling.
        lean192: Requested preparation-only auto/on/off policy. Forced ON rejects
            ineligible inputs before a bound Index kernel can be executed.

    Returns:
        OptimizedIndexPlan retaining buffers, direct map views and selected launch metadata.
        Preparation may allocate/compile and must run outside capture.
    """
    validate_lean192(lean192)
    q, k, v = original_q, original_k, original_v
    total_q = q.shape[0]
    batch = len(q_lens)
    skv = max(kv_lens)
    tile_plan = attention.tile_plan
    partial_dtype = attention.partial_dtype
    cu_seqlens_q = None if contiguous_unpacked else attention.cu_q
    cu_seqlens_k = None if contiguous_unpacked else attention.cu_k
    q_prefix = [0]
    for length in q_lens:
        q_prefix.append(q_prefix[-1] + length)
    is_mx = index_k.dtype == torch.uint8
    blackwell_native = torch.cuda.get_device_capability(q.device) in ((10, 0), (10, 3))
    partition_blocks = select_partition_blocks(
        index_q, index_k, batch=batch, contiguous_unpacked=contiguous_unpacked,
        kv_tokens=skv, identity_pages=index_pages_identity,
        capability=torch.cuda.get_device_capability(q.device),
    )
    cutedsl_index_k = None
    cutedsl_index_pages = None
    cutedsl_logical_k_tokens = None
    cutedsl_use_page_table = False
    cutedsl_partition_scores = None
    cutedsl_partition_ids = None
    cutedsl_packed_partitions: int = 1
    cutedsl_packed_pages = None
    cutedsl_packed_tile_metadata = None
    mxfp6_router = None
    page_operand = None
    if not attention.fixed_only:
        if not blackwell_native:
            raise ValueError("the CuTeDSL router requires SM100 or SM103")
        if is_mx:
            cutedsl_index_k = index_k
        elif index_k.ndim == 4:
            if tuple(index_k.shape[1:]) != (
                1,
                SPEC.block_size,
                SPEC.head_dim,
            ):
                raise ValueError("CuTeDSL index_k pages must be [pages,1,128,128]")
            cutedsl_index_k = index_k.view(-1, 1, SPEC.head_dim)
        elif index_k.ndim == 3:
            if batch != 1:
                raise ValueError("packed CuTeDSL routing requires paged Index K")
            if tuple(index_k.shape) != (skv, 1, SPEC.head_dim):
                raise ValueError("CuTeDSL index_k must be [Skv,1,128] when unpaged")
            cutedsl_index_k = index_k
            if skv % SPEC.block_size:
                raise ValueError(
                    "a partial logical K tail requires full paged Index K storage"
                )
        else:
            raise ValueError("unsupported CuTeDSL index_k layout")
        if not is_mx and int(cutedsl_index_k.shape[0]) % SPEC.block_size:
            raise ValueError("CuTeDSL physical Index K storage must be page aligned")
        physical_pages = (
            int(cutedsl_index_k.shape[0])
            if is_mx
            else int(cutedsl_index_k.shape[0]) // SPEC.block_size
        )
        request_blocks = tuple(
            (length + SPEC.block_size - 1) // SPEC.block_size for length in kv_lens
        )
        if index_page_indices.ndim == 2:
            if index_page_indices.shape[0] != batch or any(
                index_page_indices.shape[1] < count for count in request_blocks
            ):
                raise ValueError(
                    "2-D index_page_indices must be [B,>=max logical pages]"
                )
            request_pages = tuple(
                index_page_indices[index, :count]
                for index, count in enumerate(request_blocks)
            )
        elif index_page_indices.ndim == 1:
            active_pages = sum(request_blocks)
            if index_page_indices.numel() == active_pages:
                page_offsets = [0]
                for count in request_blocks:
                    page_offsets.append(page_offsets[-1] + count)
                request_pages = tuple(
                    index_page_indices[
                        page_offsets[index] : page_offsets[index + 1]
                    ]
                    for index in range(batch)
                )
            elif index_page_indices.numel() % batch == 0:
                row_width = index_page_indices.numel() // batch
                if row_width < max(request_blocks):
                    raise ValueError("flattened page-table rows are too short")
                page_rows = index_page_indices.view(batch, row_width)
                request_pages = tuple(
                    page_rows[index, :count]
                    for index, count in enumerate(request_blocks)
                )
            else:
                raise ValueError(
                    "1-D index_page_indices must contain active pages or padded rows"
                )
        else:
            raise ValueError("index_page_indices must be rank 1 or 2")
        if check_device_inputs and any(
            bool(torch.any(pages < 0)) or bool(torch.any(pages >= physical_pages))
            for pages in request_pages
        ):
            raise ValueError("index_page_indices references Index K out of range")

        page_operand = page_operand_view(request_pages, 16 if batch == 1 or is_mx else 4)
        if batch == 1 and not is_mx:
            cutedsl_index_pages = page_operand
            cutedsl_logical_k_tokens = kv_lens[0]
            if index_pages_identity is None and check_device_inputs:
                identity_pages = torch.arange(
                    request_blocks[0],
                    dtype=torch.int32,
                    device=index_page_indices.device,
                )
                index_pages_identity = torch.equal(
                    cutedsl_index_pages, identity_pages
                )
            cutedsl_use_page_table = not index_pages_identity
            cutedsl_partitions = (
                request_blocks[0] + partition_blocks - 1
            ) // partition_blocks
            if cutedsl_partitions > 1:
                partition_shape = (
                    SPEC.kv_heads,
                    cutedsl_partitions,
                    total_q,
                    SPEC.dynamic_topk,
                )
                cutedsl_partition_scores = torch.empty(
                    partition_shape,
                    dtype=torch.float32,
                    device=q.device,
                )
                cutedsl_partition_ids = torch.empty(
                    partition_shape,
                    dtype=torch.int32,
                    device=q.device,
                )
        else:
            request_partitions = tuple(
                (kv_blocks + partition_blocks - 1) // partition_blocks
                for kv_blocks in request_blocks
            )
            # Emit all requests and KV partitions in one grid. Unused
            # partitions publish (-inf, -1), so the global merge has the same
            # request-local Top-16 set without per-request kernel launches.
            cutedsl_packed_partitions = max(request_partitions)
            cutedsl_packed_pages = page_operand
            metadata: list[int] = []
            page_begin = 0
            for request_index, (q_len, kv_len, kv_blocks) in enumerate(
                zip(q_lens, kv_lens, request_blocks)
            ):
                for local_tile in range(
                    (q_len + SPEC.block_size - 1) // SPEC.block_size
                ):
                    metadata.extend(
                        (
                            q_prefix[request_index],
                            q_prefix[request_index + 1],
                            local_tile,
                            page_begin,
                            kv_len,
                            request_partitions[request_index],
                        )
                    )
                page_begin += kv_blocks
            (cutedsl_packed_tile_metadata,) = stage_int32(
                q.device, (metadata, (len(metadata) // 6, 6))
            )
            if cutedsl_packed_partitions > 1:
                partition_shape = (
                    SPEC.kv_heads,
                    cutedsl_packed_partitions,
                    total_q,
                    SPEC.dynamic_topk,
                )
                cutedsl_partition_scores = torch.empty(
                    partition_shape,
                    dtype=torch.float32,
                    device=q.device,
                )
                cutedsl_partition_ids = torch.empty(
                    partition_shape,
                    dtype=torch.int32,
                    device=q.device,
                )
    if is_mx and not attention.fixed_only:
        from .src.sm100.mx_router import prepare_mxfp6_router

        mxfp6_router = prepare_mxfp6_router(
            index_q,
            index_q_scales,
            index_k,
            cutedsl_packed_pages,
            cutedsl_packed_tile_metadata,
            max_blocks=max(request_blocks),
            route=candidate_storage.view(torch.int32),
            partition_blocks=256,
        )
    lean192_early_router = bool(
        contiguous_unpacked and not attention.fixed_only and batch == 1
        and cu_seqlens_q is None and cu_seqlens_k is None
        and tile_plan.policy == PrefillPolicy.ALIGNED_CONTIGUOUS
        and index_pages_identity is True and not cutedsl_use_page_table
        and cutedsl_packed_tile_metadata is None and mxfp6_router is None
        and q.dtype == k.dtype == v.dtype == partial_dtype == torch.bfloat16
        and index_k.ndim in (3, 4) and k.ndim == v.ndim == 3
        and index_q.device == index_k.device == q.device
        and all(t.is_contiguous() for t in (index_q, index_k, q, k, v))
        and lean192_index_eligible(
            # Paged-shaped contiguous Index storage already has this zero-copy
            # dense view. Check its exact physical extent against logical KV.
            index_q, cutedsl_index_k, partition_blocks=partition_blocks,
            logical_k_tokens=skv, use_page_table=cutedsl_use_page_table)
        and torch.cuda.get_device_capability(q.device) == (10, 3)
    )
    plan = OptimizedIndexPlan(
        index_q=index_q, index_k=index_k, index_q_scales=index_q_scales,
        mxfp6_router=mxfp6_router, index_page_indices=index_page_indices,
        candidate_storage=candidate_storage, prefix_lens=prefix_lens,
        attention=attention, candidate_partition_blocks=partition_blocks,
        cutedsl_index_k=cutedsl_index_k, cutedsl_index_pages=cutedsl_index_pages,
        cutedsl_logical_k_tokens=cutedsl_logical_k_tokens,
        cutedsl_use_page_table=cutedsl_use_page_table,
        cutedsl_partition_scores=cutedsl_partition_scores,
        cutedsl_partition_ids=cutedsl_partition_ids,
        cutedsl_packed_partitions=cutedsl_packed_partitions,
        cutedsl_packed_pages=cutedsl_packed_pages,
        cutedsl_packed_tile_metadata=cutedsl_packed_tile_metadata,
        _lean192_early_router=resolve_lean192(lean192, lean192_early_router),
        lean192_mode=lean192,
    )
    bind_score_operands(plan)
    return plan
