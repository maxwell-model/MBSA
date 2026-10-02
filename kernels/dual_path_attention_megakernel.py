"""Experimental physical Dual-Path Attention MegaKernel for SM103.

The first milestone deliberately targets the narrow production-critical
geometry: batch one, contiguous BF16 Q/K/V, D128, GQA8, block-128, static
four-WG scheduling.  Fixed Prefix/SWA CTAs retain Anthony's Q-centric
multi-block dataflow; Dynamic-16 CTAs retain the MSA KV-centric dataflow.
Both device bodies execute under one physical CUDA launch.  Batched K2 remains
outside this kernel and consumes Fixed state as its initial state.
"""

from __future__ import annotations

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack
from quack.cute_dsl_utils import ParamsBase

from src.common import copy_utils as common_copy_utils
from src.common.cute_dsl_utils import to_cute_tensor as to_cute_tensor_kvouter
from src.sm100.fwd.atten_fwd import SparseAttentionForwardSm100
from src.sm100.fwd.config import (
    MSA_4WG_STATIC,
    MSA_5WG_STATIC_UTMALDG_SCATTERED,
)
from third_party.anthony_bsa_backend.flashinfer.cute_dsl.sparse.bsa_utils.cute_tensor_utils import (
    to_cute_tensor as to_cute_tensor_fixed,
)
from third_party.anthony_bsa_backend.flashinfer.cute_dsl.sparse.sm100_blk128.flash_fwd_sm100 import (
    FlashAttentionForwardSm100,
)


class DualPathAttentionSm100:
    """Compose the existing Fixed and Dynamic device bodies into one launch."""

    def __init__(
        self,
        *,
        async_pool_ctas: int | None = None,
        compact_batch1_schedule: bool = False,
        use_five_wg_dynamic: bool = False,
        dynamic_static_stride: bool = False,
        dynamic_chunk_size: int = 1,
    ) -> None:
        if async_pool_ctas is not None and async_pool_ctas <= 0:
            raise ValueError("async_pool_ctas must be positive")
        self.async_pool_ctas = async_pool_ctas
        self.use_five_wg_dynamic = bool(use_five_wg_dynamic)
        self.dynamic_static_stride = bool(dynamic_static_stride)
        if dynamic_chunk_size <= 0:
            raise ValueError("dynamic_chunk_size must be positive")
        self.dynamic_chunk_size = int(dynamic_chunk_size)
        self.fixed = FlashAttentionForwardSm100(
            head_dim=128,
            head_dim_v=128,
            qhead_per_kvhead=8,
            pack_gqa=True,
            m_block_size=128,
            n_block_size=128,
            is_persistent=True,
            use_2cta_instrs=False,
            use_clc_scheduler=False,
            allow_empty_block_nums=False,
            has_block_sizes=False,
            is_causal=True,
            implicit_fixed_support=True,
            is_paged_kv=False,
            split_p_arrive=None,
            sync_first_p_release=False,
            implicit_fixed_load_reduce=False,
            grouped_output=False,
            is_varlen=True,
            defer_launch=True,
            physical_num_wgs=5 if use_five_wg_dynamic else 4,
        )
        self.dynamic = SparseAttentionForwardSm100(
            cfg=(
                MSA_5WG_STATIC_UTMALDG_SCATTERED
                if use_five_wg_dynamic
                else MSA_4WG_STATIC
            ),
            head_dim=128,
            qheadperkv=8,
            m_block_size=128,
            n_block_size=128,
            paged_kv=False,
            page_size=None,
            has_seqused_k=False,
            causal=False,
            use_prepare_scheduler=True,
            qk_dtype=cutlass.BFloat16,
            pv_dtype=cutlass.BFloat16,
            safe_fwd_sync=True,
            shared_work_valid=True,
            safe_partial_v_tail=True,
            compact_batch1_schedule=compact_batch1_schedule,
            last_cta_finalize=False,
            defer_launch=True,
            registers_preconfigured=use_five_wg_dynamic,
        )

    @cute.jit
    def __call__(
        self,
        mQ_fixed,
        mK_fixed,
        mV_fixed,
        mO_fixed,
        mLSE_fixed,
        mFixedRoute,
        mFixedPackedTiles,
        mK_dynamic,
        mV_dynamic,
        mK2qIndices,
        mK2qQSplitIndices,
        mK2qCounts,
        mSchedulerMetadata,
        mWorkCount,
        mO_partial,
        mLSE_partial,
        mQ_flat,
        mCuSeqlensQ,
        mCuSeqlensK,
        softmax_scale: Float32,
        num_kv_blocks: Int32,
        num_heads_kv: Int32,
        seq_len_q: Int32,
        dynamic_work_capacity: Int32,
        logical_seqlen_k: Int32,
        mRouteReady,
        mDynamicCounter,
        stream,
    ):
        fixed_args, fixed_grid = self.fixed(
            mQ_fixed,
            mK_fixed,
            mV_fixed,
            mO_fixed,
            mLSE_fixed,
            softmax_scale,
            mFixedRoute,
            None,
            Int32(16),
            None,
            None,
            logical_seqlen_k,
            mFixedPackedTiles,
            stream,
        )
        dynamic_args, dynamic_ctas, dynamic_smem = self.dynamic(
            mK_dynamic,
            mV_dynamic,
            mK2qIndices,
            mK2qQSplitIndices,
            mK2qCounts,
            mSchedulerMetadata,
            mWorkCount,
            None,
            mO_partial,
            mLSE_partial,
            None,
            None,
            None,
            None,
            None,
            None,
            mQ_flat,
            None,
            None,
            None,
            None,
            mCuSeqlensQ,
            mCuSeqlensK,
            softmax_scale,
            Float32(1.0),
            num_kv_blocks,
            num_heads_kv,
            seq_len_q,
            dynamic_work_capacity,
            stream,
        )
        fixed_ctas = fixed_grid[0]
        smem_bytes = max(
            self.fixed.shared_storage.size_in_bytes(),
            dynamic_smem,
            49152,
        )
        logical_fixed_ctas = (
            fixed_ctas
            if cutlass.const_expr(self.async_pool_ctas is None)
            else Int32(self.async_pool_ctas)
        )
        launch_ctas = (
            fixed_ctas + dynamic_ctas
            if cutlass.const_expr(self.async_pool_ctas is None)
            else logical_fixed_ctas
        )
        self.kernel(
            *fixed_args,
            *dynamic_args,
            mRouteReady,
            mDynamicCounter,
            Int32(logical_fixed_ctas),
        ).launch(
            grid=(launch_ctas,),
            block=[self.dynamic.threads_per_cta, 1, 1],
            smem=smem_bytes,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        # Fixed Q-centric BSA launch arguments.
        f_mQ: cute.Tensor,
        f_mK: cute.Tensor,
        f_mV: cute.Tensor,
        f_mO: cute.Tensor,
        f_mLSE: Optional[cute.Tensor],
        f_tma_atom_Q: cute.CopyAtom,
        f_tma_atom_K: cute.CopyAtom,
        f_tma_atom_V: cute.CopyAtom,
        f_tma_atom_O: Optional[cute.CopyAtom],
        f_softmax_scale_log2: Float32,
        f_softmax_scale: Float32 | None,
        f_sQ_layout: cute.ComposedLayout,
        f_sK_layout: cute.ComposedLayout,
        f_tP_layout: cute.ComposedLayout,
        f_sV_layout: cute.ComposedLayout,
        f_sO_layout: cute.ComposedLayout,
        f_gmem_tiled_copy_O: Optional[cute.TiledCopy],
        f_tiled_mma_qk: cute.TiledMma,
        f_tiled_mma_pv: cute.TiledMma,
        f_tile_sched_params: ParamsBase,
        f_num_splits: Int32,
        f_mBlockIndex: cute.Tensor,
        f_mBlockSizes: cute.Tensor,
        f_block_sparse_num: Int32,
        f_mBlockNums: Optional[cute.Tensor],
        f_mPageTable: Optional[cute.Tensor],
        f_logical_seqlen_k: Int32,
        f_mPackedTiles: Optional[cute.Tensor],
        f_o_ragged_strides: tuple,
        # Dynamic KV-centric BSA launch arguments.
        d_tma_K: cute.Tensor,
        d_tma_V: cute.Tensor,
        d_mK2qIndices: cute.Tensor,
        d_mK2qQSplitIndices: cute.Tensor,
        d_mK2qCounts: cute.Tensor,
        d_mSchedulerMetadata: Optional[cute.Tensor],
        d_mWorkCount: Optional[cute.Tensor],
        d_mWorkTileOffsets: Optional[cute.Tensor],
        d_mO_partial: cute.Tensor,
        d_mLSE_partial: cute.Tensor,
        d_mLSE_temperature_partial: Optional[cute.Tensor],
        d_mSplitCountdown: Optional[cute.Tensor],
        d_mOFinal: Optional[cute.Tensor],
        d_mLSEFinal: Optional[cute.Tensor],
        d_mOInitial: Optional[cute.Tensor],
        d_mLSEInitial: Optional[cute.Tensor],
        d_mQ_2d: cute.Tensor,
        d_mQ_gather4_desc: Optional[cute.Tensor],
        d_mO_ragged_tma_desc: Optional[cute.Tensor],
        d_mPageTable,
        d_mSeqUsedK,
        d_mCuSeqlensQ,
        d_mCuSeqlensK,
        d_softmax_scale_log2: Float32,
        d_lse_temperature_scale_log2: Float32,
        d_lse_temperature_scale: Float32,
        d_sQ_layout: cute.ComposedLayout,
        d_sQ_load_layout: cute.ComposedLayout,
        d_sK_layout: cute.ComposedLayout,
        d_sV_layout: cute.ComposedLayout,
        d_sK_fp8_layout: cute.Layout,
        d_sV_fp8_layout: cute.Layout,
        d_tP_layout: cute.ComposedLayout,
        d_tma_atom_K: cute.CopyAtom,
        d_tma_atom_V: cute.CopyAtom,
        d_tma_atom_Q: cute.CopyAtom,
        d_tma_atom_O: cute.CopyAtom,
        d_tma_O: cute.Tensor,
        d_tiled_mma_qk: cute.TiledMma,
        d_tiled_mma_pv: cute.TiledMma,
        d_kv_tma_bytes: cutlass.Constexpr[int],
        d_q_tma_bytes: cutlass.Constexpr[int],
        d_q_subtile_bytes: cutlass.Constexpr[int],
        d_num_kv_blocks: Int32,
        d_num_heads_kv: Int32,
        d_seq_len_q: Int32,
        d_work_capacity: Int32,
        mRouteReady: Optional[cute.Tensor],
        mDynamicCounter: Optional[cute.Tensor],
        fixed_work_count: Int32,
    ):
        """Dispatch homogeneous CTAs to either child device body.

        The launch arguments are intentionally flattened.  CuTe DSL treats
        layout/copy/MMA objects inside a nested tuple as staged values at the
        kernel boundary, which prevents pipeline construction from consuming
        them as compile-time metadata.
        """
        bidx, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        if cutlass.const_expr(
            self.dynamic.shared_storage.size_in_bytes()
            >= self.fixed.shared_storage.size_in_bytes()
        ):
            overlay_storage = smem.allocate(self.dynamic.shared_storage)
        else:
            overlay_storage = smem.allocate(self.fixed.shared_storage)
        shared_storage_base = overlay_storage.base
        if cutlass.const_expr(self.async_pool_ctas is not None):
            if cutlass.const_expr(
                mRouteReady is None or mDynamicCounter is None
            ):
                raise ValueError(
                    "async MegaKernel requires route-ready and work-counter tensors"
                )
            # Every resident CTA first participates in the Q-centric Fixed
            # phase.  The logical stride is the bounded pool size rather than
            # the physical grid of the original standalone kernel.
            self.fixed._kernel_body(
                f_mQ,
                f_mK,
                f_mV,
                f_mO,
                f_mLSE,
                f_tma_atom_Q,
                f_tma_atom_K,
                f_tma_atom_V,
                f_tma_atom_O,
                f_softmax_scale_log2,
                f_softmax_scale,
                f_sQ_layout,
                f_sK_layout,
                f_tP_layout,
                f_sV_layout,
                f_sO_layout,
                f_gmem_tiled_copy_O,
                f_tiled_mma_qk,
                f_tiled_mma_pv,
                f_tile_sched_params,
                f_num_splits,
                f_mBlockIndex,
                f_mBlockSizes,
                f_block_sparse_num,
                f_mBlockNums,
                f_mPageTable,
                f_logical_seqlen_k,
                f_mPackedTiles,
                f_o_ragged_strides,
                shared_storage_base,
                fixed_work_count,
            )
            cute.arch.sync_threads()

            # Only one thread polls the acquire flag; the CTA barrier then
            # publishes readiness to all roles without generating 512 global
            # polling streams per CTA.
            tidx = cute.arch.thread_idx()[0]
            if tidx == Int32(0):
                ready = Int32(0)
                while ready == Int32(0):
                    ready = common_copy_utils.load_acquire_i32(
                        mRouteReady.iterator
                    )
            cute.arch.sync_threads()

            if cutlass.const_expr(self.dynamic_static_stride):
                # Long-context native schedules expose enough independent
                # records to distribute them by an interleaved grid stride.
                # This removes the global atomic and one CTA rendezvous from
                # every record while dispersing adjacent CSR rows across SMs.
                work_idx = bidx
            else:
                # Dynamic atomic claiming absorbs residual runtime variation
                # that static schedule metadata cannot model exactly.
                sWorkIdx = cute.make_tensor(
                    cute.recast_ptr(shared_storage_base, dtype=Int32),
                    cute.make_layout((1,)),
                )
                if tidx == Int32(0):
                    sWorkIdx[0] = cute.arch.atomic_add(
                        mDynamicCounter.iterator.llvm_ptr,
                        Int32(self.dynamic_chunk_size),
                        sem="relaxed",
                        scope="gpu",
                    )
                cute.arch.sync_threads()
                work_idx = sWorkIdx[0]
                chunk_offset = Int32(0)
            while work_idx < d_mWorkCount[Int32(0)]:
                self.dynamic._kernel_body(
                    d_tma_K,
                    d_tma_V,
                    d_mK2qIndices,
                    d_mK2qQSplitIndices,
                    d_mK2qCounts,
                    d_mSchedulerMetadata,
                    d_mWorkCount,
                    d_mWorkTileOffsets,
                    d_mO_partial,
                    d_mLSE_partial,
                    d_mLSE_temperature_partial,
                    d_mSplitCountdown,
                    d_mOFinal,
                    d_mLSEFinal,
                    d_mOInitial,
                    d_mLSEInitial,
                    d_mQ_2d,
                    d_mQ_gather4_desc,
                    d_mO_ragged_tma_desc,
                    d_mPageTable,
                    d_mSeqUsedK,
                    d_mCuSeqlensQ,
                    d_mCuSeqlensK,
                    d_softmax_scale_log2,
                    d_lse_temperature_scale_log2,
                    d_lse_temperature_scale,
                    d_sQ_layout,
                    d_sQ_load_layout,
                    d_sK_layout,
                    d_sV_layout,
                    d_sK_fp8_layout,
                    d_sV_fp8_layout,
                    d_tP_layout,
                    d_tma_atom_K,
                    d_tma_atom_V,
                    d_tma_atom_Q,
                    d_tma_atom_O,
                    d_tma_O,
                    d_tiled_mma_qk,
                    d_tiled_mma_pv,
                    d_kv_tma_bytes,
                    d_q_tma_bytes,
                    d_q_subtile_bytes,
                    d_num_kv_blocks,
                    d_num_heads_kv,
                    d_seq_len_q,
                    d_work_capacity,
                    Int32(0),
                    shared_storage_base,
                    work_idx,
                )
                cute.arch.sync_threads()
                if cutlass.const_expr(self.dynamic_static_stride):
                    work_idx += fixed_work_count
                else:
                    chunk_offset += Int32(1)
                    if (
                        chunk_offset < Int32(self.dynamic_chunk_size)
                        and work_idx + Int32(1) < d_mWorkCount[Int32(0)]
                    ):
                        work_idx += Int32(1)
                    else:
                        if tidx == Int32(0):
                            sWorkIdx[0] = cute.arch.atomic_add(
                                mDynamicCounter.iterator.llvm_ptr,
                                Int32(self.dynamic_chunk_size),
                                sem="relaxed",
                                scope="gpu",
                            )
                        cute.arch.sync_threads()
                        work_idx = sWorkIdx[0]
                        chunk_offset = Int32(0)
            return

        if bidx < fixed_work_count:
            self.fixed._kernel_body(
                f_mQ,
                f_mK,
                f_mV,
                f_mO,
                f_mLSE,
                f_tma_atom_Q,
                f_tma_atom_K,
                f_tma_atom_V,
                f_tma_atom_O,
                f_softmax_scale_log2,
                f_softmax_scale,
                f_sQ_layout,
                f_sK_layout,
                f_tP_layout,
                f_sV_layout,
                f_sO_layout,
                f_gmem_tiled_copy_O,
                f_tiled_mma_qk,
                f_tiled_mma_pv,
                f_tile_sched_params,
                f_num_splits,
                f_mBlockIndex,
                f_mBlockSizes,
                f_block_sparse_num,
                f_mBlockNums,
                f_mPageTable,
                f_logical_seqlen_k,
                f_mPackedTiles,
                f_o_ragged_strides,
                shared_storage_base,
                fixed_work_count,
            )
        else:
            self.dynamic._kernel_body(
                d_tma_K,
                d_tma_V,
                d_mK2qIndices,
                d_mK2qQSplitIndices,
                d_mK2qCounts,
                d_mSchedulerMetadata,
                d_mWorkCount,
                d_mWorkTileOffsets,
                d_mO_partial,
                d_mLSE_partial,
                d_mLSE_temperature_partial,
                d_mSplitCountdown,
                d_mOFinal,
                d_mLSEFinal,
                d_mOInitial,
                d_mLSEInitial,
                d_mQ_2d,
                d_mQ_gather4_desc,
                d_mO_ragged_tma_desc,
                d_mPageTable,
                d_mSeqUsedK,
                d_mCuSeqlensQ,
                d_mCuSeqlensK,
                d_softmax_scale_log2,
                d_lse_temperature_scale_log2,
                d_lse_temperature_scale,
                d_sQ_layout,
                d_sQ_load_layout,
                d_sK_layout,
                d_sV_layout,
                d_sK_fp8_layout,
                d_sV_fp8_layout,
                d_tP_layout,
                d_tma_atom_K,
                d_tma_atom_V,
                d_tma_atom_Q,
                d_tma_atom_O,
                d_tma_O,
                d_tiled_mma_qk,
                d_tiled_mma_pv,
                d_kv_tma_bytes,
                d_q_tma_bytes,
                d_q_subtile_bytes,
                d_num_kv_blocks,
                d_num_heads_kv,
                d_seq_len_q,
                d_work_capacity,
                fixed_work_count,
                shared_storage_base,
                None,
            )

class RouteReadyPublisher:
    """Publish completed route descriptors with GPU-scope release semantics."""

    @cute.jit
    def __call__(self, mRouteReady, stream):
        self.kernel(mRouteReady).launch(
            grid=(1,),
            block=[32, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mRouteReady: cute.Tensor):
        if cute.arch.thread_idx()[0] == Int32(0):
            common_copy_utils.store_release_i32(
                mRouteReady.iterator,
                Int32(1),
            )


_compile_cache: dict[tuple, object] = {}
_publish_cache: dict[tuple, object] = {}


def publish_dynamic_route_ready(route_ready: torch.Tensor) -> None:
    """Release-publish route metadata on the current PyTorch CUDA stream."""
    if (
        route_ready.dtype != torch.int32
        or route_ready.numel() != 1
        or not route_ready.is_cuda
        or not route_ready.is_contiguous()
    ):
        raise ValueError("route_ready must be a contiguous one-element CUDA int32 tensor")
    key = (route_ready.device.type, route_ready.device.index)
    if key not in _publish_cache:
        _publish_cache[key] = cute.compile(
            RouteReadyPublisher(),
            to_cute_tensor_kvouter(route_ready),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _publish_cache[key](route_ready)


def run_dual_path_attention_megakernel(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    fixed_out: torch.Tensor,
    fixed_lse: torch.Tensor,
    fixed_route_sentinel: torch.Tensor,
    fixed_packed_tiles: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    k2q_qsplit_indices: torch.Tensor,
    scheduler_metadata: torch.Tensor,
    work_count: torch.Tensor,
    work_capacity: int,
    partial_o: torch.Tensor,
    partial_lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    route_ready: torch.Tensor | None = None,
    dynamic_counter: torch.Tensor | None = None,
    async_pool_ctas: int | None = None,
    dynamic_static_stride: bool = False,
    dynamic_chunk_size: int = 1,
) -> None:
    """Launch the narrow batch-one physical MegaKernel prototype."""
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("Dual-Path prototype requires BF16 Q/K/V")
    if q.ndim != 3 or tuple(q.shape[1:]) != (32, 128):
        raise ValueError("Q must be [Sq,32,128]")
    if k.shape != v.shape or k.ndim != 3 or tuple(k.shape[1:]) != (4, 128):
        raise ValueError("K/V must be [Skv,4,128]")
    if q.shape[0] > k.shape[0] or q.shape[0] % 128 or k.shape[0] % 128:
        raise ValueError("prototype requires aligned 0 < Sq <= Skv")
    if fixed_out.shape != q.shape or fixed_lse.shape != (1, 32, q.shape[0]):
        raise ValueError("fixed output state has an unexpected shape")
    if partial_o.shape[0] != 16 or partial_o.dtype != torch.bfloat16:
        raise ValueError("prototype preserves the BF16 Dynamic-16 workspace")
    if partial_lse.shape != partial_o.shape[:-1] or partial_lse.dtype != torch.float32:
        raise ValueError("partial LSE must match Dynamic-16 partial rows")
    if cu_seqlens_q.numel() != 2 or cu_seqlens_k.numel() != 2:
        raise ValueError("prototype supports batch one only")
    if async_pool_ctas is not None:
        if not 1 <= async_pool_ctas <= 140:
            raise ValueError("async pool must reserve capacity and use 1..140 CTAs")
        for name, tensor in (
            ("route_ready", route_ready),
            ("dynamic_counter", dynamic_counter),
        ):
            if (
                tensor is None
                or tensor.dtype != torch.int32
                or tensor.numel() != 1
                or not tensor.is_cuda
            ):
                raise ValueError(f"{name} must be a one-element CUDA int32 tensor")
    elif route_ready is not None or dynamic_counter is not None:
        raise ValueError("route publication tensors require async_pool_ctas")
    if any(not tensor.is_cuda for tensor in (
        q, k, v, fixed_out, fixed_lse, fixed_route_sentinel,
        fixed_packed_tiles, k2q_row_ptr, k2q_q_indices,
        k2q_qsplit_indices, scheduler_metadata, work_count,
        partial_o, partial_lse, cu_seqlens_q, cu_seqlens_k,
    )):
        raise ValueError("all operands must be CUDA tensors")

    q_flat = q.reshape(-1, 128).contiguous()
    partial_o_flat = partial_o.reshape(-1, 128).contiguous()
    softmax_scale = float(128**-0.5)
    metadata_columns = int(scheduler_metadata.shape[1])
    if metadata_columns not in (2, 6):
        raise ValueError("scheduler metadata must use compact2 or legacy6 records")
    compact_batch1_schedule = metadata_columns == 2
    key = (
        int(q.shape[0]),
        int(k.shape[0]),
        int(work_capacity),
        tuple(scheduler_metadata.shape),
        async_pool_ctas,
        bool(dynamic_static_stride),
        int(dynamic_chunk_size),
    )
    if key not in _compile_cache:
        kernel = DualPathAttentionSm100(
            async_pool_ctas=async_pool_ctas,
            compact_batch1_schedule=compact_batch1_schedule,
            # Cross-body five-WG register redistribution is still an
            # experimental specialization.  Keep the validated four-WG
            # physical path as the default until that gate passes.
            use_five_wg_dynamic=False,
            dynamic_static_stride=dynamic_static_stride,
            dynamic_chunk_size=dynamic_chunk_size,
        )
        packed_tiles_cute = from_dlpack(
            fixed_packed_tiles.detach(), assumed_align=4, enable_tvm_ffi=True
        ).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        _compile_cache[key] = cute.compile(
            kernel,
            to_cute_tensor_fixed(q),
            to_cute_tensor_fixed(k),
            to_cute_tensor_fixed(v),
            to_cute_tensor_fixed(fixed_out),
            to_cute_tensor_fixed(fixed_lse[0], assumed_align=4),
            to_cute_tensor_fixed(fixed_route_sentinel),
            packed_tiles_cute,
            to_cute_tensor_kvouter(k),
            to_cute_tensor_kvouter(v),
            to_cute_tensor_kvouter(k2q_q_indices),
            to_cute_tensor_kvouter(k2q_qsplit_indices),
            to_cute_tensor_kvouter(k2q_row_ptr),
            to_cute_tensor_kvouter(scheduler_metadata),
            to_cute_tensor_kvouter(work_count),
            to_cute_tensor_kvouter(partial_o_flat),
            to_cute_tensor_kvouter(partial_lse),
            to_cute_tensor_kvouter(q_flat),
            to_cute_tensor_kvouter(cu_seqlens_q),
            to_cute_tensor_kvouter(cu_seqlens_k),
            Float32(softmax_scale),
            Int32(k.shape[0] // 128),
            Int32(4),
            Int32(q.shape[0]),
            Int32(work_capacity),
            Int32(k.shape[0]),
            None
            if route_ready is None
            else to_cute_tensor_kvouter(route_ready),
            None
            if dynamic_counter is None
            else to_cute_tensor_kvouter(dynamic_counter),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    _compile_cache[key](
        q,
        k,
        v,
        fixed_out,
        fixed_lse[0],
        fixed_route_sentinel,
        fixed_packed_tiles,
        k,
        v,
        k2q_q_indices,
        k2q_qsplit_indices,
        k2q_row_ptr,
        scheduler_metadata,
        work_count,
        partial_o_flat,
        partial_lse,
        q_flat,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_scale,
        k.shape[0] // 128,
        4,
        q.shape[0],
        work_capacity,
        k.shape[0],
        route_ready,
        dynamic_counter,
    )


__all__ = [
    "publish_dynamic_route_ready",
    "run_dual_path_attention_megakernel",
]
