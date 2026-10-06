# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
# Modified for MBSA: package integration; see root NOTICE.

"""Sparse forward combine kernel and public launcher.

This keeps the local fake-layout -> real-layout epilogue needed by the lean
sparse forward path.
"""

# Modified Step 7: O_out write with SMEM fake->real column permutation.
# O_partial dim is in STG.128 fake layout; O_out dim is real layout.
import math
from typing import Type, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass import Float32, Int32, Int64, Boolean, const_expr
from cutlass.cutlass_dsl import BaseDSL

from mbsa.cute.src.common import utils
from mbsa.cute.src.common.cute_dsl_utils import assume_tensor_aligned, torch2cute_dtype_map
from mbsa.cute.src.common.seqlen_info import SeqlenInfo
from cutlass.cute import FastDivmodDivisor

from mbsa.cute.src.common.pack_gqa import PackGQAComb
from mbsa.cute.src.common.tma_utils import stg128_fake_col_to_real_col, stg128_fp8_fake_col_to_real_col, stg128_half_fake_col_to_real_col
from mbsa.cute.src.sm100.fwd.config import MSA_OUTPUT_TILE_ROWS


_tma_tile_combine_compile_cache: dict = {}


@cute.jit
def _aligned_bf16x8(tensor: cute.Tensor, offset: Int64):
    """Return one known-aligned 128-bit BF16 vector at ``offset``."""
    ptr = tensor.iterator + offset
    ptr = cute.make_ptr(
        tensor.element_type,
        ptr.toint(),
        mem_space=ptr.memspace,
        assumed_align=16,
    )
    return cute.make_tensor(ptr, cute.make_layout((8,)))


class SparseAttentionShuffledCombine:
    """Gather shuffled MSA states and merge them into output-major rows."""

    def __init__(
        self,
        *,
        topk: int,
        head_dim: int = 128,
        qhead_per_kv: int = 8,
        num_threads: int = 256,
    ):
        if head_dim != 128 or qhead_per_kv != 8:
            raise ValueError("tile combine currently requires D128 GQA8")
        if not 1 <= topk <= 16:
            raise ValueError("tile combine supports one to sixteen splits")
        self.topk = topk
        self.head_dim = head_dim
        self.qhead_per_kv = qhead_per_kv
        self.num_threads = num_threads
        self.threads_per_row = 16
        self.rows_per_cta = num_threads // self.threads_per_row

    @cute.jit
    def __call__(
        self,
        mOTileRows: cute.Tensor,
        mLSETileRows: cute.Tensor,
        mPartialLocations: cute.Tensor,
        mSplitCounts: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mOInitial: Optional[cute.Tensor] = None,
        mLSEInitial: Optional[cute.Tensor] = None,
        stream: cuda.CUstream = None,
    ):
        if const_expr(mOTileRows.element_type is not cutlass.BFloat16):
            raise TypeError("shuffled O must be BF16")
        if const_expr(mLSETileRows.element_type is not Float32):
            raise TypeError("shuffled LSE must be FP32")
        if const_expr(mPartialLocations.element_type is not Int32):
            raise TypeError("partial locations must be Int32")
        if const_expr(mSplitCounts.element_type is not Int32):
            raise TypeError("split counts must be Int32")
        if const_expr(mO.element_type is not cutlass.BFloat16):
            raise TypeError("tile combine output must be BF16")
        if const_expr(mLSE.element_type is not Float32):
            raise TypeError("tile combine LSE output must be FP32")
        if const_expr((mOInitial is None) != (mLSEInitial is None)):
            raise ValueError("initial O/LSE must be provided together")

        tensors = [
            mOTileRows,
            mLSETileRows,
            mPartialLocations,
            mSplitCounts,
            mO,
            mLSE,
        ]
        tensors = [assume_tensor_aligned(t) for t in tensors]
        (
            mOTileRows,
            mLSETileRows,
            mPartialLocations,
            mSplitCounts,
            mO,
            mLSE,
        ) = tensors
        if const_expr(mOInitial is not None):
            mOInitial = assume_tensor_aligned(mOInitial)
            mLSEInitial = assume_tensor_aligned(mLSEInitial)
        total_rows = mO.shape[0] * mO.shape[1]
        self.kernel(
            mOTileRows,
            mLSETileRows,
            mPartialLocations,
            mSplitCounts,
            mO,
            mLSE,
            mOInitial,
            mLSEInitial,
        ).launch(
            grid=(cute.ceil_div(total_rows, self.rows_per_cta),),
            block=[self.num_threads, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mOTileRows: cute.Tensor,
        mLSETileRows: cute.Tensor,
        mPartialLocations: cute.Tensor,
        mSplitCounts: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mOInitial: Optional[cute.Tensor],
        mLSEInitial: Optional[cute.Tensor],
    ):
        cute.arch.griddepcontrol_wait()
        tidx = cute.arch.thread_idx()[0]
        row_linear = (
            cute.arch.block_idx()[0] * Int32(self.rows_per_cta)
            + tidx // Int32(self.threads_per_row)
        )
        col_group = tidx % Int32(self.threads_per_row)
        vector_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.BFloat16,
            num_bits_per_copy=128,
        )
        total_q = mO.shape[0]
        head_q = mO.shape[1]
        if row_linear < total_q * head_q:
            q_abs = row_linear // head_q
            q_head = row_linear - q_abs * head_q
            head_kv = q_head // Int32(self.qhead_per_kv)
            head_local = q_head - head_kv * Int32(self.qhead_per_kv)
            split_lane = col_group
            # Early causal rows may have fewer dynamic states than topk;
            # unmapped inverse slots must never be read.
            split_count = mSplitCounts[q_abs, head_kv]
            split_valid = split_lane < split_count
            split_location = Int32(0)
            split_lse = -Float32.inf
            if split_valid:
                split_location = mPartialLocations[
                    q_abs, head_kv, split_lane
                ]
                split_lse = mLSETileRows[split_location + head_local]

            initial_lse = -Float32.inf
            if const_expr(mLSEInitial is not None):
                initial_lse = mLSEInitial[Int32(0), q_head, q_abs]
            lse_max = cute.arch.warp_reduction_max(
                max(split_lse, initial_lse),
                threads_in_group=self.threads_per_row,
            )
            finite_max = Float32(0.0) if lse_max == -Float32.inf else lse_max
            log2e = Float32(math.log2(math.e))
            split_scale = (
                cute.math.exp2(
                    (split_lse - finite_max) * log2e,
                    fastmath=True,
                )
                if split_valid
                else Float32(0.0)
            )
            initial_scale = Float32(0.0)
            if const_expr(mLSEInitial is not None):
                if col_group == Int32(0):
                    initial_scale = cute.math.exp2(
                        (initial_lse - finite_max) * log2e,
                        fastmath=True,
                    )
            scale_sum = cute.arch.warp_reduction_sum(
                split_scale + initial_scale,
                threads_in_group=self.threads_per_row,
            )
            valid_state = (
                lse_max != -Float32.inf
                and scale_sum != Float32(0.0)
                and scale_sum == scale_sum
            )
            inv_sum = cute.arch.rcp_approx(
                scale_sum if valid_state else Float32(1.0)
            )
            split_scale *= inv_sum
            initial_scale = utils.shuffle_sync(
                initial_scale * inv_sum,
                Int32(0),
                width=self.threads_per_row,
            )

            values_per_thread = self.head_dim // self.threads_per_row
            vector_width = min(8, values_per_thread)
            vectors_per_thread = values_per_thread // vector_width
            col_base = col_group * Int32(values_per_thread)
            acc = cute.make_rmem_tensor((values_per_thread,), Float32)
            for i in cutlass.range_constexpr(values_per_thread):
                acc[i] = Float32(0.0)
            if const_expr(mOInitial is not None):
                initial_coord = (
                    (head_kv, q_abs, head_local, 0)
                    if const_expr(len(mOInitial.shape) == 4)
                    else (q_abs, q_head, 0)
                )
                initial_row = Int64(cute.crd2idx(initial_coord, mOInitial.layout))
                for vec in cutlass.range_constexpr(vectors_per_thread):
                    initial = cute.make_rmem_tensor(
                        (vector_width,), cutlass.BFloat16
                    )
                    initial_gmem = _aligned_bf16x8(
                        mOInitial,
                        initial_row
                        + Int64(col_base + Int32(vec * vector_width)),
                    )
                    cute.copy(vector_copy, initial_gmem, initial)
                    for i in cutlass.range_constexpr(vector_width):
                        acc[vec * vector_width + i] += (
                            initial[i].to(Float32) * initial_scale
                        )

            partial_pipe = cute.make_rmem_tensor(
                cute.make_ordered_layout((2, vector_width), order=(1, 0)),
                cutlass.BFloat16,
            )
            if split_count > Int32(0):
                first_location = utils.shuffle_sync(
                    split_location,
                    Int32(0),
                    width=self.threads_per_row,
                )
                # Every inactive split lane reuses the first valid row.  Its
                # scale is zero, so the value cannot affect the merge.  This
                # keeps all vector loads address-valid without emitting one
                # divergent branch for every unrolled split.
                split_location = (
                    split_location if split_valid else first_location
                )
                first_row = Int64(first_location + head_local)
                first_gmem = _aligned_bf16x8(
                    mOTileRows,
                    first_row * Int64(self.head_dim) + Int64(col_base),
                )
                cute.copy(vector_copy, first_gmem, partial_pipe[0, None])
                for split in cutlass.range_constexpr(self.topk):
                    if const_expr(split + 1 < self.topk):
                        next_location = utils.shuffle_sync(
                            split_location,
                            Int32(split + 1),
                            width=self.threads_per_row,
                        )
                        next_row = Int64(next_location + head_local)
                        next_gmem = _aligned_bf16x8(
                            mOTileRows,
                            next_row * Int64(self.head_dim) + Int64(col_base),
                        )
                        cute.copy(
                            vector_copy,
                            next_gmem,
                            partial_pipe[(split + 1) % 2, None],
                        )
                    scale = utils.shuffle_sync(
                        split_scale,
                        Int32(split),
                        width=self.threads_per_row,
                    )
                    for i in cutlass.range_constexpr(vector_width):
                        acc[i] += partial_pipe[split % 2, i].to(Float32) * scale

            out_row = Int64(row_linear) * Int64(self.head_dim)
            for vec in cutlass.range_constexpr(vectors_per_thread):
                out = cute.make_rmem_tensor(
                    (vector_width,), cutlass.BFloat16
                )
                for i in cutlass.range_constexpr(vector_width):
                    out[i] = acc[vec * vector_width + i].to(
                        cutlass.BFloat16
                    )
                out_gmem = _aligned_bf16x8(
                    mO,
                    out_row
                    + Int64(col_base + Int32(vec * vector_width)),
                )
                cute.copy(vector_copy, out, out_gmem)
            if col_group == Int32(0):
                mLSE[q_abs, q_head] = (
                    cute.math.log(scale_sum, fastmath=True) + lse_max
                    if valid_state
                    else -Float32.inf
                )


class SparseAttentionForwardCombine:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        dtype_partial: Type[cutlass.Numeric],
        head_dim: int,
        tile_m: int = 8,
        k_block_size: int = 64,
        topk: int = 16,
        num_threads: int = 256,
        stages: int = 4,
        use_pdl: bool = False,
        min_blocks_per_mp: int = 0,
        use_tma_partial: bool = False,
        use_vector_initial: bool = False,
    ):
        """
        Forward combine kernel for split attention computation.

        :param dtype: output data type
        :param dtype_partial: partial accumulation data type
        :param head_dim: head dimension
        :param tile_m: m block size
        :param k_block_size: k block size
        :param topk: exact number of split partials
        :param num_threads: number of threads
        :param varlen: whether using variable length sequences
        :param stages: number of pipeline stages
        """
        self.dtype = dtype
        self.dtype_partial = dtype_partial
        self.head_dim = head_dim
        self.tile_m = tile_m
        self.k_block_size = k_block_size
        self.topk = topk
        self.num_threads = num_threads
        self.is_even_k = head_dim % k_block_size == 0
        self.stages = stages
        self.use_pdl = use_pdl
        self.min_blocks_per_mp = min_blocks_per_mp
        self.use_tma_partial = use_tma_partial
        self.use_vector_initial = use_vector_initial
        self.use_stg128_half_layout = dtype_partial in (cutlass.BFloat16, cutlass.Float16)
        self.use_stg128_fp8_layout = dtype_partial is cutlass.Float8E4M3FN

    @staticmethod
    def can_implement(
        dtype,
        dtype_partial,
        head_dim,
        tile_m,
        k_block_size,
        topk,
        num_threads,
    ) -> bool:
        """Check if the kernel can be implemented with the given parameters."""
        if dtype not in [cutlass.Float16, cutlass.BFloat16, cutlass.Float32]:
            return False
        if dtype_partial not in [
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            Float32,
        ]:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if tile_m % 8 != 0:
            return False
        if topk > 256:
            return False
        if (tile_m * topk) % num_threads != 0:
            return False
        return True

    def _setup_attributes(self):
        # GMEM copy setup for O partial
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype_partial.width
        assert self.k_block_size % async_copy_elems == 0

        k_block_gmem = (
            128 if self.k_block_size % 128 == 0 else (64 if self.k_block_size % 64 == 0 else 32)
        )
        gmem_threads_per_row = k_block_gmem // async_copy_elems
        assert self.num_threads % gmem_threads_per_row == 0

        # Async copy atom for O partial load
        atom_async_copy_partial = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype_partial,
            num_bits_per_copy=universal_copy_bits,
        )
        tOpartial_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row, gmem_threads_per_row),
            order=(1, 0),
        )
        vOpartial_layout = cute.make_layout((1, async_copy_elems))
        self.gmem_tiled_copy_O_partial = cute.make_tiled_copy_tv(
            atom_async_copy_partial, tOpartial_layout, vOpartial_layout
        )

        # GMEM copy setup for final O (use universal copy for store).
        # Keep this independent from O_partial: fp8 partial uses 16 elements
        # per 128b transaction, while bf16/fp16 O stores must remain 8-wide.
        output_copy_elems = universal_copy_bits // self.dtype.width
        assert self.k_block_size % output_copy_elems == 0
        gmem_threads_per_row_o = k_block_gmem // output_copy_elems
        assert self.num_threads % gmem_threads_per_row_o == 0
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tO_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row_o, gmem_threads_per_row_o),
            order=(1, 0),
        )
        vO_layout = cute.make_layout((1, output_copy_elems))
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy,
            tO_layout,
            vO_layout,
        )
        # LSE copy setup with async copy (alignment = 1)
        lse_copy_bits = Float32.width  # 1 element per copy, width is in bits
        m_block_smem = (
            128
            if self.tile_m % 128 == 0
            else (
                64
                if self.tile_m % 64 == 0
                else (32 if self.tile_m % 32 == 0 else (16 if self.tile_m % 16 == 0 else 8))
            )
        )
        gmem_threads_per_row_lse = m_block_smem
        assert self.num_threads % gmem_threads_per_row_lse == 0

        # Async copy atom for LSE load
        atom_async_copy_lse = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            Float32,
            num_bits_per_copy=lse_copy_bits,
        )
        tLSE_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row_lse, gmem_threads_per_row_lse),
            order=(1, 0),
        )
        vLSE_layout = cute.make_layout(1)
        self.gmem_tiled_copy_LSE = cute.make_tiled_copy_tv(
            atom_async_copy_lse, tLSE_layout, vLSE_layout
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory
        # ///////////////////////////////////////////////////////////////////////////////

        # Shared memory to register copy for LSE
        self.smem_threads_per_col_lse = self.num_threads // m_block_smem
        assert 32 % self.smem_threads_per_col_lse == 0  # Must divide warp size

        s2r_layout_atom_lse = cute.make_ordered_layout(
            (self.smem_threads_per_col_lse, self.num_threads // self.smem_threads_per_col_lse),
            order=(0, 1),
        )
        self.s2r_tiled_copy_LSE = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32),
            s2r_layout_atom_lse,
            cute.make_layout(1),
        )

        # LSE shared memory layout with swizzling to avoid bank conflicts
        # This works for kBlockMSmem = 8, 16, 32, 64, 128, no bank conflicts
        if const_expr(m_block_smem == 8):
            smem_lse_swizzle = cute.make_swizzle(5, 0, 5)
        elif const_expr(m_block_smem == 16):
            smem_lse_swizzle = cute.make_swizzle(4, 0, 4)
        else:
            smem_lse_swizzle = cute.make_swizzle(3, 2, 3)
        lse_atom_splits = min(self.topk, 8)
        smem_layout_atom_lse = cute.make_composed_layout(
            smem_lse_swizzle,
            0,
            cute.make_ordered_layout((lse_atom_splits, m_block_smem), order=(1, 0)),
        )
        self.smem_layout_lse = cute.tile_to_shape(
            smem_layout_atom_lse, (self.topk, self.tile_m), (0, 1)
        )

        # O_partial staging layout.
        if const_expr(self.use_tma_partial):
            # Keep the TMA staging tile in the same explicit row-major
            # coordinate system as the zero-copy [flat_row, d, split] GMEM
            # view.  The STG.128 swizzle below is optimized for lane-owned
            # cp.async stores; using it as a TMA descriptor tile aliases the
            # second half of a 64-row CTA tile onto the wrong head rows.
            self.smem_layout_o = cute.make_ordered_layout(
                (self.tile_m, self.k_block_size, self.stages), order=(1, 0, 2)
            )
        elif const_expr(
            self.dtype_partial
            in [cutlass.Float16, cutlass.BFloat16, cutlass.Float8E4M3FN]
        ):
            smem_layout_atom_o = _get_cpasync_smem_layout_atom(
                self.dtype_partial, self.k_block_size
            )
            self.smem_layout_o = cute.tile_to_shape(
                smem_layout_atom_o,
                (self.tile_m, self.k_block_size, self.stages),
                (0, 1, 2),
            )
        else:
            self.smem_layout_o = cute.make_ordered_layout(
                (self.tile_m, self.k_block_size, self.stages), order=(1, 0, 2)
            )

    @cute.jit
    def __call__(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor] = None,
        mO_initial: Optional[cute.Tensor] = None,
        mLSE_initial: Optional[cute.Tensor] = None,
        mLSE_temperature_partial: Optional[cute.Tensor] = None,
        mLSE_temperature: Optional[cute.Tensor] = None,
        cu_seqlens: Optional[cute.Tensor] = None,
        seqused: Optional[cute.Tensor] = None,
        num_splits_dynamic_ptr: Optional[cute.Tensor] = None,
        varlen_batch_idx: Optional[cute.Tensor] = None,
        semaphore_to_reset: Optional[cute.Tensor] = None,
        mSplitCounts: Optional[cute.Tensor] = None,
        mOutputScale: Optional[cute.Tensor] = None,
        qhead_per_kvhead: Int32 = Int32(1),
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # Type checking
        if const_expr(not (mO_partial.element_type == self.dtype_partial)):
            raise TypeError("O partial tensor must match dtype_partial")
        if const_expr(not (mO.element_type == self.dtype)):
            raise TypeError("O tensor must match dtype")
        if const_expr(mLSE_partial.element_type not in [Float32]):
            raise TypeError("LSE partial tensor must be Float32")
        if const_expr(mLSE is not None and mLSE.element_type not in [Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr((mO_initial is None) != (mLSE_initial is None)):
            raise ValueError("initial O and LSE tensors must be provided together")
        if const_expr(mO_initial is not None and mO_initial.element_type != self.dtype):
            raise TypeError("initial O tensor must match output dtype")
        if const_expr(mLSE_initial is not None and mLSE_initial.element_type not in [Float32]):
            raise TypeError("initial LSE tensor must be Float32")
        if const_expr(
            mLSE_temperature_partial is not None
            and mLSE_temperature_partial.element_type not in [Float32]
        ):
            raise TypeError("temperature LSE partial tensor must be Float32")
        if const_expr(mLSE_temperature is not None and mLSE_temperature.element_type not in [Float32]):
            raise TypeError("temperature LSE tensor must be Float32")
        if const_expr((mLSE_temperature_partial is None) != (mLSE_temperature is None)):
            raise ValueError(
                "temperature LSE partial and output tensors must either both be provided or both be None"
            )

        # Shape validation - input tensors are in user format, need to be converted to kernel format
        if const_expr(len(mO_partial.shape) not in [4, 5]):
            raise ValueError(
                "O partial tensor must have 4 or 5 dimensions: (num_splits, batch, seqlen, nheads, headdim) or (num_splits, total_q, nheads, headdim)"
            )
        if const_expr(len(mLSE_partial.shape) not in [3, 4]):
            raise ValueError(
                "LSE partial tensor must have 3 or 4 dimensions: (num_splits, batch, seqlen, nheads) or (num_splits, total_q, nheads)"
            )
        if const_expr(len(mO.shape) not in [3, 4]):
            raise ValueError(
                "O tensor must have 3 or 4 dimensions: (batch, seqlen, nheads, headdim) or (total_q, nheads, headdim)"
            )
        if const_expr(mLSE is not None and len(mLSE.shape) not in [2, 3]):
            raise ValueError(
                "LSE tensor must have 2 or 3 dimensions: (batch, seqlen, nheads) or (total_q, nheads)"
            )
        if const_expr(mLSE_temperature_partial is not None and len(mLSE_temperature_partial.shape) not in [3, 4]):
            raise ValueError(
                "temperature LSE partial tensor must have 3 or 4 dimensions: "
                "(num_splits, batch, seqlen, nheads) or (num_splits, total_q, nheads)"
            )
        if const_expr(mLSE_temperature is not None and len(mLSE_temperature.shape) not in [2, 3]):
            raise ValueError(
                "temperature LSE tensor must have 2 or 3 dimensions: "
                "(batch, seqlen, nheads) or (total_q, nheads)"
            )
        if const_expr(mSplitCounts is not None):
            if const_expr(mSplitCounts.element_type not in [Int32]):
                raise TypeError("split_counts tensor must be Int32")
            if const_expr(cu_seqlens is not None):
                if const_expr(len(mSplitCounts.shape) != 2):
                    raise ValueError("varlen split_counts tensor must have shape (total_q, nheads_kv)")
            elif const_expr(len(mSplitCounts.shape) != 3):
                raise ValueError("batched split_counts tensor must have shape (batch, seqlen, nheads_kv)")
        if const_expr(mOutputScale is not None and mOutputScale.element_type not in [Float32]):
            raise TypeError("output_scale tensor must be Float32")
        if const_expr(self.use_tma_partial and cu_seqlens is None):
            raise NotImplementedError("TMA partial-state loads require varlen batch-1 layout")

        mO_partial, mO = [assume_tensor_aligned(t) for t in (mO_partial, mO)]
        if const_expr(mO_initial is not None):
            mO_initial = assume_tensor_aligned(mO_initial)
        mO_partial_tma = None
        if const_expr(self.use_tma_partial):
            # [split, total_q, head, d] -> [total_q * head, d, split].
            # This is a zero-copy view of each contiguous K2 64x128 tile.
            flat_rows = mO_partial.shape[1] * mO_partial.shape[2]
            mO_partial_tma = cute.make_tensor(
                mO_partial.iterator,
                cute.make_layout(
                    (flat_rows, mO_partial.shape[3], mO_partial.shape[0]),
                    stride=(
                        mO_partial.shape[3],
                        1,
                        flat_rows * mO_partial.shape[3],
                    ),
                ),
            )
        # (num_splits, b, seqlen, h, d) -> (seqlen, d, num_splits, h, b)
        # or (num_splits, total_q, h, d) -> (total_q, d, num_splits, h)
        O_partial_layout_transpose = (
            [2, 4, 0, 3, 1] if const_expr(cu_seqlens is None) else [1, 3, 0, 2]
        )
        # (b, seqlen, h, d) -> (seqlen, d, h, b) or (total_q, h, d) -> (total_q, d, h)
        mO_partial = cute.make_tensor(
            mO_partial.iterator, cute.select(mO_partial.layout, mode=O_partial_layout_transpose)
        )
        O_layout_transpose = [1, 3, 2, 0] if const_expr(cu_seqlens is None) else [0, 2, 1]
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        grouped_initial = const_expr(
            mO_initial is not None and cu_seqlens is not None and len(mO_initial.shape) == 4
        )
        if const_expr(grouped_initial):
            mO_initial = cute.make_tensor(
                mO_initial.iterator,
                cute.make_layout(
                    (mO_initial.shape[1], mO_initial.shape[3],
                     (mO_initial.shape[2], mO_initial.shape[0])),
                    stride=(mO_initial.stride[1], mO_initial.stride[3],
                            (mO_initial.stride[2], mO_initial.stride[0])),
                ),
            )
        else:
            mO_initial = (
                cute.make_tensor(
                    mO_initial.iterator,
                    cute.select(mO_initial.layout, mode=O_layout_transpose),
                )
                if const_expr(mO_initial is not None)
                else None
            )
        # (num_splits, b, h, seqlen) -> (seqlen, num_splits, h, b)
        # Input is pre-transposed: [topK, B, Hq, Sq] with Sq innermost for K2-friendly reads.
        # or (num_splits, total_q, h) -> (total_q, num_splits, h)
        LSE_partial_layout_transpose = [3, 0, 2, 1] if const_expr(cu_seqlens is None) else [1, 0, 2]
        mLSE_partial = cute.make_tensor(
            mLSE_partial.iterator,
            cute.select(mLSE_partial.layout, mode=LSE_partial_layout_transpose),
        )
        # (b, seqlen, h) -> (seqlen, h, b) or (total_q, h) -> (total_q, h)
        LSE_layout_transpose = [1, 2, 0] if const_expr(cu_seqlens is None) else [0, 1]
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
            if mLSE is not None
            else None
        )
        # The fixed BSA backend writes LSE in [B, H, Sq] order.  Consume that
        # native layout directly so this fusion does not need a transpose or
        # publication kernel.  The hybrid caller currently specializes B=1.
        mLSE_initial = (
            cute.make_tensor(
                mLSE_initial.iterator,
                cute.select(mLSE_initial.layout, mode=[2, 1, 0]),
            )
            if const_expr(mLSE_initial is not None)
            else None
        )
        mLSE_temperature_partial = (
            cute.make_tensor(
                mLSE_temperature_partial.iterator,
                cute.select(mLSE_temperature_partial.layout, mode=LSE_partial_layout_transpose),
            )
            if mLSE_temperature_partial is not None
            else None
        )
        mLSE_temperature = (
            cute.make_tensor(
                mLSE_temperature.iterator,
                cute.select(mLSE_temperature.layout, mode=LSE_layout_transpose),
            )
            if mLSE_temperature is not None
            else None
        )

        # Determine if we have variable length sequences
        varlen = const_expr(cu_seqlens is not None or seqused is not None)

        self._setup_attributes()
        tma_atom_O_partial = None
        if const_expr(self.use_tma_partial):
            o_tma_tile_shape = cute.select(
                self.smem_layout_o, mode=[0, 1]
            ).shape
            tma_atom_O_partial, mO_partial_tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mO_partial_tma,
                cute.select(self.smem_layout_o, mode=[0, 1]),
                o_tma_tile_shape,
            )

        # Output-dtype permutation buffer for Step 7 (tile_m × k_block_size).
        # Accumulation stays fp32; the final dtype conversion happens before
        # the fake→real SMEM scatter to reduce half-output SMEM pressure.
        if const_expr(self.dtype in [cutlass.Float16, cutlass.BFloat16]):
            smem_layout_perm = cute.make_layout(
                (self.tile_m, self.k_block_size),
                stride=(self.k_block_size + 16, 1),
            )
        else:
            smem_layout_perm = cute.make_ordered_layout(
                (self.tile_m, self.k_block_size), order=(1, 0)
            )

        @cute.struct
        class SharedStorage:
            sLSE: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.smem_layout_lse)], 128
            ]
            sLSETemperature: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.smem_layout_lse)], 128
            ]
            sMaxValidSplit: cute.struct.Align[cute.struct.MemRange[Int32, self.tile_m], 128]
            sInitialScale: cute.struct.Align[cute.struct.MemRange[Float32, self.tile_m], 128]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.dtype_partial, cute.cosize(self.smem_layout_o)], 128
            ]
            sO_perm: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(smem_layout_perm)], 128
            ]
            # Full/empty transaction barriers for the optional three-stage
            # TMA partial-O ring. The cp.async control keeps these 48 bytes
            # unused so both paths share one storage ABI in this experiment.
            oTmaMbar: cute.struct.Align[
                cute.struct.MemRange[Int64, self.stages * 2], 8
            ]

        smem_size = SharedStorage.size_in_bytes()

        # Grid: (ceil(seqlen/tile_m), ceil(dim/k_block), num_head * batch)
        # Head separated from seqlen → enables future TMA (contiguous Sq tiles)
        seqlen = mO_partial.shape[0]
        num_head = mO_partial.shape[3]
        batch_size = (
            mO_partial.shape[4]
            if const_expr(cu_seqlens is None)
            else Int32(cu_seqlens.shape[0] - 1)
        )

        seqlen_divmod = FastDivmodDivisor(seqlen)
        head_divmod = FastDivmodDivisor(num_head)

        grid_dim = (
            cute.ceil_div(seqlen * num_head, self.tile_m),
            cute.ceil_div(self.head_dim, self.k_block_size),
            batch_size,
        )

        self.kernel(
            mO_partial,
            mO_partial_tma,
            tma_atom_O_partial,
            mLSE_partial,
            mO,
            mLSE,
            mO_initial,
            mLSE_initial,
            mLSE_temperature_partial,
            mLSE_temperature,
            cu_seqlens,
            seqused,
            num_splits_dynamic_ptr,
            varlen_batch_idx,
            semaphore_to_reset,
            mSplitCounts,
            mOutputScale,
            qhead_per_kvhead,
            SharedStorage,
            self.smem_layout_lse,
            self.smem_layout_o,
            smem_layout_perm,
            self.gmem_tiled_copy_O_partial,
            self.gmem_tiled_copy_O,
            self.gmem_tiled_copy_LSE,
            self.s2r_tiled_copy_LSE,
            seqlen_divmod,
            head_divmod,
            self.use_pdl,
            varlen,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=smem_size,
            stream=stream,
            min_blocks_per_mp=self.min_blocks_per_mp,
            use_pdl=self.use_pdl,
        )

    @cute.jit
    def decode_flat_row_idx(
        self,
        idx: Int32,
        head_divmod: FastDivmodDivisor,
    ):
        """Decode flattened tile rows under the H_q-innermost contract."""
        q_idx_local, head_idx = divmod(idx, head_divmod)
        return q_idx_local, head_idx

    @cute.kernel
    def kernel(
        self,
        mO_partial: cute.Tensor,
        mO_partial_tma: Optional[cute.Tensor],
        tma_atom_O_partial: Optional[cute.CopyAtom],
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mO_initial: Optional[cute.Tensor],
        mLSE_initial: Optional[cute.Tensor],
        mLSE_temperature_partial: Optional[cute.Tensor],
        mLSE_temperature: Optional[cute.Tensor],
        cu_seqlens: Optional[cute.Tensor],
        seqused: Optional[cute.Tensor],
        num_splits_dynamic_ptr: Optional[cute.Tensor],
        varlen_batch_idx: Optional[cute.Tensor],
        semaphore_to_reset: Optional[cute.Tensor],
        mSplitCounts: Optional[cute.Tensor],
        mOutputScale: Optional[cute.Tensor],
        qhead_per_kvhead: Int32,
        SharedStorage: cutlass.Constexpr,
        smem_layout_lse: cute.Layout | cute.ComposedLayout,
        smem_layout_o: cute.Layout | cute.ComposedLayout,
        smem_layout_perm: cute.Layout,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        s2r_tiled_copy_LSE: cute.TiledCopy,
        seqlen_divmod: FastDivmodDivisor,
        head_divmod: FastDivmodDivisor,
        use_pdl: cutlass.Constexpr[bool],
        varlen: cutlass.Constexpr[bool],
    ):
        # Thread and block indices
        # Keep runtime-guarded locals in a separate JIT scope. CuTe DSL 4.6
        # otherwise tries to join Tensor definitions with an absent branch.
        tidx, _, _ = cute.arch.thread_idx()
        m_block, k_block, maybe_virtual_batch = cute.arch.block_idx()
        if const_expr(use_pdl and semaphore_to_reset is not None):
            if (
                tidx == 0
                and m_block == cute.arch.grid_dim()[0] - 1
                and k_block == cute.arch.grid_dim()[1] - 1
                and maybe_virtual_batch == cute.arch.grid_dim()[2] - 1
            ):
                cute.arch.griddepcontrol_wait()
                semaphore_to_reset[0] = 0
        batch_idx = (
            varlen_batch_idx[maybe_virtual_batch]
            if const_expr(varlen_batch_idx is not None)
            else maybe_virtual_batch
        )
        seqlen_info = SeqlenInfo.create(
            batch_idx=batch_idx,
            seqlen_static=mO_partial.shape[0],
            cu_seqlens=cu_seqlens,
            seqused=seqused,
        )
        max_idx = seqlen_info.seqlen * mO_partial.shape[3]
        if const_expr(not varlen) or m_block * self.tile_m < max_idx:
            self.kernel_valid_tile(
                mO_partial,
                mO_partial_tma,
                tma_atom_O_partial,
                mLSE_partial,
                mO,
                mLSE,
                mO_initial,
                mLSE_initial,
                mLSE_temperature_partial,
                mLSE_temperature,
                cu_seqlens,
                seqused,
                num_splits_dynamic_ptr,
                varlen_batch_idx,
                semaphore_to_reset,
                mSplitCounts,
                mOutputScale,
                qhead_per_kvhead,
                SharedStorage,
                smem_layout_lse,
                smem_layout_o,
                smem_layout_perm,
                gmem_tiled_copy_O_partial,
                gmem_tiled_copy_O,
                gmem_tiled_copy_LSE,
                s2r_tiled_copy_LSE,
                seqlen_divmod,
                head_divmod,
                use_pdl,
                varlen,
            )

    @cute.jit
    def kernel_valid_tile(
        self,
        mO_partial: cute.Tensor,
        mO_partial_tma: Optional[cute.Tensor],
        tma_atom_O_partial: Optional[cute.CopyAtom],
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mO_initial: Optional[cute.Tensor],
        mLSE_initial: Optional[cute.Tensor],
        mLSE_temperature_partial: Optional[cute.Tensor],
        mLSE_temperature: Optional[cute.Tensor],
        cu_seqlens: Optional[cute.Tensor],
        seqused: Optional[cute.Tensor],
        num_splits_dynamic_ptr: Optional[cute.Tensor],
        varlen_batch_idx: Optional[cute.Tensor],
        semaphore_to_reset: Optional[cute.Tensor],
        mSplitCounts: Optional[cute.Tensor],
        mOutputScale: Optional[cute.Tensor],
        qhead_per_kvhead: Int32,
        SharedStorage: cutlass.Constexpr,
        smem_layout_lse: cute.Layout | cute.ComposedLayout,
        smem_layout_o: cute.Layout | cute.ComposedLayout,
        smem_layout_perm: cute.Layout,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        s2r_tiled_copy_LSE: cute.TiledCopy,
        seqlen_divmod: FastDivmodDivisor,
        head_divmod: FastDivmodDivisor,
        use_pdl: cutlass.Constexpr[bool],
        varlen: cutlass.Constexpr[bool],
    ):
        # Thread and block indices
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        m_block, k_block, maybe_virtual_batch = cute.arch.block_idx()

        batch_idx = (
            varlen_batch_idx[maybe_virtual_batch]
            if const_expr(varlen_batch_idx is not None)
            else maybe_virtual_batch
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sLSE = storage.sLSE.get_tensor(smem_layout_lse)
        sLSE_temperature = storage.sLSETemperature.get_tensor(smem_layout_lse)
        sMaxValidSplit = storage.sMaxValidSplit.get_tensor((self.tile_m,))
        sInitialScale = storage.sInitialScale.get_tensor((self.tile_m,))
        sO = storage.sO.get_tensor(smem_layout_o)
        sO_perm_buf = storage.sO_perm.get_tensor(smem_layout_perm)
        o_tma_pipeline = None
        o_tma_producer = None
        o_tma_consumer = None
        tOsO_tma = None
        tOgO_tma = None
        if const_expr(self.use_tma_partial):
            gO_tma = cute.local_tile(
                mO_partial_tma,
                cute.select(smem_layout_o, mode=[0, 1]).shape,
                (m_block, k_block, None),
            )
            tOsO_tma, tOgO_tma = cpasync.tma_partition(
                tma_atom_O_partial,
                0,
                cute.make_layout(1),
                cute.group_modes(sO, 0, 2),
                cute.group_modes(gO_tma, 0, 2),
            )
            o_tma_bytes = cute.size_in_bytes(
                self.dtype_partial,
                cute.select(smem_layout_o, mode=[0, 1]),
            )
            o_tma_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.oTmaMbar.data_ptr(),
                num_stages=self.stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.num_threads
                ),
                tx_count=o_tma_bytes,
                defer_sync=True,
            )
            o_tma_producer, o_tma_consumer = o_tma_pipeline.make_participants()
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()

        # Handle semaphore reset — wait for dependent grids first

        if const_expr(num_splits_dynamic_ptr is not None):
            raise ValueError("K2 combine requires compile-time exact topK")
        num_splits = Int32(self.topk)
        # Handle variable length sequences using SeqlenInfo
        seqlen_info = SeqlenInfo.create(
            batch_idx=batch_idx,
            seqlen_static=mO_partial.shape[0],
            cu_seqlens=cu_seqlens,
            seqused=seqused,
            # Don't need to pass in tile size since we won't use offset_padded
        )
        seqlen, offset = seqlen_info.seqlen, seqlen_info.offset

        num_head = mO_partial.shape[3]
        max_idx = seqlen * num_head
        output_scale = Float32(1.0)
        if const_expr(mOutputScale is not None):
            output_scale = mOutputScale[0]

        # Wait for dependent grids (e.g., the main attention kernel that produces O_partial/LSE_partial)
        if const_expr(use_pdl):
            cute.arch.griddepcontrol_wait()

        # ===============================
        # Step 1: Load LSE_partial from gmem to shared memory
        # ===============================
        # `cLSE` (identity tensor for row/split coord tracking) is reused
        # later in steps 4-5, so it must be defined on both branches.
        cLSE = cute.make_identity_tensor((self.topk, self.tile_m))
        # Reshape mLSE_partial to PackGQA packed layout and delegate the
        # tile load to PackGQAComb.load_LSE. The packed form folds (H_q, Sq)
        # into one compound dim with H_q innermost (stride 1), so thread
        # rows that vary along h_pos produce one-sector coalesced reads.
        # Non-varlen path only — varlen keeps the original inline loop.
        if const_expr(not varlen):
            mLSE_partial_cur = seqlen_info.offset_batch(mLSE_partial, batch_idx, dim=3)
            # mLSE_partial_cur: (H_q, topK, Sq) — after initial transpose
            # [3,0,2,1] on [topK,B,Sq,H_q] and dropping B.
            # Reorder to (H_q, Sq, topK) then group modes 0..1 for packed dim:
            mLSE_partial_reord = cute.make_tensor(
                mLSE_partial_cur.iterator,
                cute.select(mLSE_partial_cur.layout, mode=[0, 2, 1]),
            )
            mLSE_partial_packed = cute.group_modes(mLSE_partial_reord, 0, 2)
            # shape ((H_q, Sq), topK) with H_q innermost.
            packgqa = PackGQAComb(
                m_block_size=self.tile_m,
                head_dim_padded=0,       # unused for LSE load
                check_hdim_oob=False,    # unused for LSE load
                qhead_per_kvhead=1,      # unused; num_heads_divmod is passed explicitly
            )
            packgqa.load_LSE(
                mLSE_partial_packed,
                sLSE,
                self.topk,
                gmem_tiled_copy_LSE,
                tidx,
                m_block,
                num_splits,
                seqlen,
                head_divmod,
                mSplitCounts,
                batch_idx,
                qhead_per_kvhead,
            )
            if const_expr(mLSE_temperature_partial is not None):
                mLSE_temperature_partial_cur = seqlen_info.offset_batch(
                    mLSE_temperature_partial, batch_idx, dim=3)
                mLSE_temperature_partial_reord = cute.make_tensor(
                    mLSE_temperature_partial_cur.iterator,
                    cute.select(mLSE_temperature_partial_cur.layout, mode=[0, 2, 1]),
                )
                mLSE_temperature_partial_packed = cute.group_modes(
                    mLSE_temperature_partial_reord, 0, 2)
                packgqa.load_LSE(
                    mLSE_temperature_partial_packed,
                    sLSE_temperature,
                    self.topk,
                    gmem_tiled_copy_LSE,
                    tidx,
                    m_block,
                    num_splits,
                    seqlen,
                    head_divmod,
                    mSplitCounts,
                    batch_idx,
                    qhead_per_kvhead,
                )
        else:
            # Varlen path keeps the same H_q-innermost flat-row contract:
            # after transpose [1, 0, 2], mLSE_partial_cur is
            # (q_local, split, head).
            # mSplitCounts is the authoritative valid-split count per
            # packed (q_abs, kv_head); masked splits stay at -inf and
            # therefore drop out of the final kernel LSE_out reduction.
            mLSE_partial_cur = seqlen_info.offset_batch(mLSE_partial, batch_idx, dim=3)
            mLSE_partial_copy = cute.tiled_divide(mLSE_partial_cur, (1,))
            gmem_thr_copy_LSE = gmem_tiled_copy_LSE.get_slice(tidx)
            tLSEsLSE = gmem_thr_copy_LSE.partition_D(sLSE)
            tLSEsLSE_temperature = gmem_thr_copy_LSE.partition_D(sLSE_temperature)
            tLSEcLSE = gmem_thr_copy_LSE.partition_S(cLSE)
            if const_expr(mLSE_temperature_partial is not None):
                mLSE_temperature_partial_cur = seqlen_info.offset_batch(
                    mLSE_temperature_partial, batch_idx, dim=3)
                mLSE_temperature_partial_copy = cute.tiled_divide(
                    mLSE_temperature_partial_cur, (1,))

            for m in cutlass.range(cute.size(tLSEcLSE, mode=[2]), unroll_full=True):
                mi = tLSEcLSE[0, 0, m][1]
                idx = m_block * self.tile_m + mi
                if idx < max_idx:
                    m_idx, head_idx = self.decode_flat_row_idx(idx, head_divmod)
                    row_count = (
                        mSplitCounts[offset + m_idx, head_idx // qhead_per_kvhead]
                        if const_expr(mSplitCounts is not None)
                        else num_splits
                    )
                    mLSE_partial_cur_copy = mLSE_partial_copy[None, m_idx, None, head_idx]
                    if const_expr(mLSE_temperature_partial is not None):
                        mLSE_temperature_partial_cur_copy = (
                            mLSE_temperature_partial_copy[None, m_idx, None, head_idx])
                    for s in cutlass.range(cute.size(tLSEcLSE, mode=[1]), unroll_full=True):
                        si = tLSEcLSE[0, s, 0][0]
                        if si < num_splits and si < row_count:
                            cute.copy(
                                gmem_thr_copy_LSE,
                                mLSE_partial_cur_copy[None, si],
                                tLSEsLSE[None, s, m],
                            )
                            if const_expr(mLSE_temperature_partial is not None):
                                cute.copy(
                                    gmem_thr_copy_LSE,
                                    mLSE_temperature_partial_cur_copy[None, si],
                                    tLSEsLSE_temperature[None, s, m],
                                )
                        else:
                            tLSEsLSE[None, s, m].fill(-Float32.inf)
                            if const_expr(mLSE_temperature_partial is not None):
                                tLSEsLSE_temperature[None, s, m].fill(-Float32.inf)
                else:
                    for s in cutlass.range(cute.size(tLSEcLSE, mode=[1]), unroll_full=True):
                        tLSEsLSE[None, s, m].fill(-Float32.inf)
                        if const_expr(mLSE_temperature_partial is not None):
                            tLSEsLSE_temperature[None, s, m].fill(-Float32.inf)
        cute.arch.cp_async_commit_group()

        # ===============================
        # Step 2: Load O_partial for pipeline stages
        # ===============================

        gmem_thr_copy_O_partial = gmem_tiled_copy_O_partial.get_slice(tidx)
        cO = cute.make_identity_tensor((self.tile_m, self.k_block_size))
        tOcO = gmem_thr_copy_O_partial.partition_D(cO)
        tOsO_partial = gmem_thr_copy_O_partial.partition_D(sO)
        mO_partial_cur = seqlen_info.offset_batch(mO_partial, batch_idx, dim=4)

        # Precompute per-row values for flattened (q_local, head) tiles.
        num_rows = const_expr(cute.size(tOcO, mode=[1]))
        tOmidx = cute.make_rmem_tensor(num_rows, cutlass.Int32)
        tOhidx = cute.make_rmem_tensor(num_rows, cutlass.Int32)
        tOSplitCount = cute.make_rmem_tensor(num_rows, cutlass.Int32)
        tOrOptr = cute.make_rmem_tensor(num_rows, cutlass.Int64)
        for m in cutlass.range(num_rows, unroll_full=True):
            mi = tOcO[0, m, 0][0]  # m coordinate in tile
            idx = m_block * self.tile_m + mi
            if idx >= max_idx:
                tOhidx[m] = -1
                tOmidx[m] = 0
                tOSplitCount[m] = 0
                tOrOptr[m] = cutlass.Int64(0)
            else:
                tOmidx[m], tOhidx[m] = self.decode_flat_row_idx(idx, head_divmod)
                if const_expr(mSplitCounts is None):
                    tOSplitCount[m] = num_splits
                elif const_expr(cu_seqlens is None):
                    tOSplitCount[m] = mSplitCounts[
                        batch_idx, tOmidx[m], tOhidx[m] // qhead_per_kvhead
                    ]
                else:
                    tOSplitCount[m] = mSplitCounts[
                        offset + tOmidx[m], tOhidx[m] // qhead_per_kvhead
                    ]
                tOrOptr[m] = utils.elem_pointer(
                    mO_partial_cur,
                    (tOmidx[m], k_block * self.k_block_size, 0, tOhidx[m]),
                ).toint()

        tOpO = None
        if const_expr(not self.is_even_k):
            tOpO = cute.make_rmem_tensor(cute.size(tOcO, mode=[2]), Boolean)
            for k in cutlass.range(cute.size(tOpO), unroll_full=True):
                tOpO[k] = tOcO[0, 0, k][1] < mO_partial.shape[1] - k_block * self.k_block_size
            # if cute.arch.thread_idx()[0] == 0 and k_block == 1: cute.print_tensor(tOpO)

        load_O_partial = None
        if const_expr(not self.use_tma_partial):
            load_O_partial = partial(
                self.load_O_partial,
                gmem_tiled_copy_O_partial,
                tOrOptr,
                tOsO_partial,
                tOhidx,
                tOSplitCount,
                tOpO,
                tOcO,
                mO_partial_cur.layout,
            )

        # Load first few stages of O_partial
        if const_expr(self.use_tma_partial):
            if warp_idx == Int32(0):
                for stage in cutlass.range(self.stages - 1, unroll_full=True):
                    if stage < num_splits:
                        empty = o_tma_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_O_partial,
                            tOgO_tma[(None, stage)],
                            tOsO_tma[(None, empty.index)],
                            tma_bar_ptr=empty.barrier,
                        )
                        empty.commit()
        else:
            for stage in cutlass.range(self.stages - 1, unroll_full=True):
                if stage < num_splits:
                    load_O_partial(stage, stage)
                cute.arch.cp_async_commit_group()

        # ===============================
        # Step 3: Load and transpose LSE from smem to registers
        # ===============================

        # Wait for LSE and initial O partial stages to complete
        if const_expr(self.use_tma_partial):
            cute.arch.cp_async_wait_group(0)
        else:
            cute.arch.cp_async_wait_group(self.stages - 1)
        cute.arch.sync_threads()
        # if cute.arch.thread_idx()[0] == 0:
        #     # cute.print_tensor(sLSE)
        #     for i in range(64):
        #         cute.printf("sLSE[%d, 0] = %f", i, sLSE[i, 0])
        # cute.arch.sync_threads()

        s2r_thr_copy_LSE = s2r_tiled_copy_LSE.get_slice(tidx)
        ts2rsLSE = s2r_thr_copy_LSE.partition_S(sLSE)
        ts2rrLSE = cute.make_rmem_tensor_like(ts2rsLSE)
        cute.copy(s2r_tiled_copy_LSE, ts2rsLSE, ts2rrLSE)
        if const_expr(mLSE_temperature_partial is not None):
            ts2rsLSE_temperature = s2r_thr_copy_LSE.partition_S(sLSE_temperature)
            ts2rrLSE_temperature = cute.make_rmem_tensor_like(ts2rsLSE_temperature)
            cute.copy(
                s2r_tiled_copy_LSE,
                ts2rsLSE_temperature,
                ts2rrLSE_temperature,
            )

        # ===============================
        # Step 4: Compute final LSE along split dimension
        # ===============================

        final_lse = cute.make_rmem_tensor(cute.size(ts2rrLSE, mode=[2]), Float32)
        initial_scale = cute.make_rmem_tensor(cute.size(ts2rrLSE, mode=[2]), Float32)
        ts2rcLSE = s2r_thr_copy_LSE.partition_D(cLSE)
        # We compute the max valid split for each row to short-circuit the computation later
        max_valid_split = cute.make_rmem_tensor(cute.size(ts2rrLSE, mode=[2]), Int32)
        assert cute.size(ts2rrLSE, mode=[0]) == 1
        # Compute max, scales, and final LSE for each row. Invalid splits
        # have already been filled with -inf, so Step 5 can write the
        # kernel-native LSE_out directly.
        for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
            initial_lse = -Float32.inf
            if const_expr(mLSE_initial is not None):
                mi = ts2rcLSE[0, 0, m][1]
                idx = m_block * self.tile_m + mi
                if idx < max_idx:
                    m_idx, head_idx = self.decode_flat_row_idx(idx, head_divmod)
                    initial_lse = mLSE_initial[m_idx, head_idx, batch_idx]
            # Find max LSE value across splits
            threads_per_col = const_expr(self.smem_threads_per_col_lse)
            lse_max = cute.arch.warp_reduction_max(
                ts2rrLSE[None, None, m]
                .load()
                .reduce(cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0),
                threads_in_group=threads_per_col,
            )
            if const_expr(mLSE_initial is not None):
                lse_max = max(lse_max, initial_lse)
            # if cute.arch.thread_idx()[0] == 0: cute.printf(lse_max)
            # Find max valid split index
            max_valid_idx = -1
            for s in cutlass.range(cute.size(ts2rrLSE, mode=[1]), unroll_full=True):
                if ts2rrLSE[0, s, m] != -Float32.inf:
                    max_valid_idx = ts2rcLSE[0, s, 0][0]  # Get split coordinate
            # if cute.arch.thread_idx()[0] < 32: cute.printf(max_valid_idx)
            max_valid_split[m] = cute.arch.warp_reduction_max(
                max_valid_idx, threads_in_group=threads_per_col
            )
            # Compute exp scales and sum
            lse_max_cur = (
                0.0 if lse_max == -Float32.inf else lse_max
            )  # In case all local LSEs are -inf
            LOG2_E = math.log2(math.e)
            lse_sum_cur = 0.0
            initial_weight = 0.0
            if const_expr(mLSE_initial is not None):
                initial_weight = cute.math.exp2(
                    initial_lse * LOG2_E - (lse_max_cur * LOG2_E), fastmath=True
                )
                # The split reduction group has multiple lanes for one
                # row.  Count the single initial state exactly once, on
                # the lane that owns split coordinate zero.
                if ts2rcLSE[0, 0, m][0] == 0:
                    lse_sum_cur += initial_weight
            for s in cutlass.range(cute.size(ts2rrLSE, mode=[1]), unroll_full=True):
                scale = cute.math.exp2(
                    ts2rrLSE[0, s, m] * LOG2_E - (lse_max_cur * LOG2_E), fastmath=True
                )
                lse_sum_cur += scale
                ts2rrLSE[0, s, m] = scale  # Store scale for later use
            lse_sum_cur = cute.arch.warp_reduction_sum(
                lse_sum_cur, threads_in_group=threads_per_col
            )
            # Normalize scales
            inv_sum = 0.0
            no_valid_state = max_valid_split[m] < 0
            if const_expr(mLSE_initial is not None):
                no_valid_state = no_valid_state and initial_lse == -Float32.inf
            if no_valid_state or lse_sum_cur == 0.0 or lse_sum_cur != lse_sum_cur:
                final_lse[m] = -Float32.inf
            else:
                final_lse[m] = cute.math.log(lse_sum_cur, fastmath=True) + lse_max
                inv_sum = 1.0 / lse_sum_cur
            initial_scale[m] = initial_weight * inv_sum
            ts2rrLSE[None, None, m].store(ts2rrLSE[None, None, m].load() * inv_sum)
        # Store the scales exp(lse - lse_logsum) back to smem
        cute.copy(s2r_tiled_copy_LSE, ts2rrLSE, ts2rsLSE)

        if const_expr(mLSE_temperature_partial is not None):
            final_lse_temperature = cute.make_rmem_tensor(
                cute.size(ts2rrLSE_temperature, mode=[2]), Float32)
            for m in cutlass.range(cute.size(ts2rrLSE_temperature, mode=[2]), unroll_full=True):
                threads_per_col = const_expr(self.smem_threads_per_col_lse)
                lse_temperature_max = cute.arch.warp_reduction_max(
                    ts2rrLSE_temperature[None, None, m]
                    .load()
                    .reduce(
                        cute.ReductionOp.MAX,
                        init_val=-Float32.inf,
                        reduction_profile=0,
                    ),
                    threads_in_group=threads_per_col,
                )
                lse_temperature_max_cur = (
                    0.0 if lse_temperature_max == -Float32.inf else lse_temperature_max
                )
                LOG2_E = math.log2(math.e)
                lse_temperature_sum_cur = 0.0
                for s in cutlass.range(
                    cute.size(ts2rrLSE_temperature, mode=[1]), unroll_full=True):
                    scale = cute.math.exp2(
                        ts2rrLSE_temperature[0, s, m] * LOG2_E
                        - (lse_temperature_max_cur * LOG2_E),
                        fastmath=True,
                    )
                    lse_temperature_sum_cur += scale
                lse_temperature_sum_cur = cute.arch.warp_reduction_sum(
                    lse_temperature_sum_cur, threads_in_group=threads_per_col
                )
                if (
                    max_valid_split[m] < 0
                    or lse_temperature_sum_cur == 0.0
                    or lse_temperature_sum_cur != lse_temperature_sum_cur
                ):
                    final_lse_temperature[m] = -Float32.inf
                else:
                    final_lse_temperature[m] = (
                        cute.math.log(lse_temperature_sum_cur, fastmath=True)
                        + lse_temperature_max
                    )

        # Store max valid split to smem
        for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
            if ts2rcLSE[0, 0, m][0] == 0:  # Only thread responsible for s=0 writes
                mi = ts2rcLSE[0, 0, m][1]
                if mi < self.tile_m:
                    sMaxValidSplit[mi] = max_valid_split[m]
                    sInitialScale[mi] = initial_scale[m]

        # ===============================
        # Step 5: Store final LSE to gmem
        # This writeback is the authoritative LSE_out returned by the
        # public Sparse Attention / Sparse Page Attention interface.
        # ===============================

        if const_expr(mLSE is not None):
            if const_expr(cu_seqlens is None):
                mLSE_cur = mLSE[None, None, batch_idx]
            else:
                mLSE_cur = cute.domain_offset((offset, 0), mLSE)
            if const_expr(mLSE_temperature is not None):
                if const_expr(cu_seqlens is None):
                    mLSE_temperature_cur = mLSE_temperature[None, None, batch_idx]
                else:
                    mLSE_temperature_cur = cute.domain_offset(
                        (offset, 0), mLSE_temperature)
            if k_block == 0:  # Only first k_block writes LSE when mLSE is provided
                for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                    if ts2rcLSE[0, 0, m][0] == 0:  # Only thread responsible for s=0 writes
                        mi = ts2rcLSE[0, 0, m][1]
                        idx = m_block * self.tile_m + mi
                        if idx < max_idx:
                            m_idx, head_idx = self.decode_flat_row_idx(idx, head_divmod)
                            mLSE_cur[m_idx, head_idx] = final_lse[m]
                            if const_expr(mLSE_temperature is not None):
                                mLSE_temperature_cur[m_idx, head_idx] = (
                                    final_lse_temperature[m])

        # ===============================
        # Step 6: Read O_partial and accumulate final O
        # ===============================

        cute.arch.sync_threads()

        # Get max valid split for this thread
        thr_max_valid_split = sMaxValidSplit[tOcO[0, 0, 0][0]]
        for m in cutlass.range(1, cute.size(tOcO, mode=[1]), unroll_full=True):
            thr_max_valid_split = max(thr_max_valid_split, sMaxValidSplit[tOcO[0, m, 0][0]])
        if const_expr(self.use_tma_partial):
            # TMA stages are CTA-owned. All 256 consumers must execute
            # the same wait/release count even when an individual row is
            # underfilled; its -inf LSE keeps the loaded slot inactive.
            thr_max_valid_split = Int32(self.topk - 1)

        tOrO_partial = cute.make_rmem_tensor_like(tOsO_partial[None, None, None, 0])
        tOrO = cute.make_rmem_tensor_like(tOrO_partial, Float32)
        tOrO.fill(0.0)
        if const_expr(mO_initial is not None):
            # Seed the accumulator from the fixed BSA state.  tOrO follows
            # the partial-output STG.128 fake-column order, while BSA O is
            # in natural column order, so map each owned fragment element
            # explicitly.  The caller may alias mO_initial with mO: every
            # CTA owns a disjoint (row, k_block) tile and completes these
            # loads before its final stores.
            mO_initial_cur = seqlen_info.offset_batch(
                mO_initial, batch_idx, dim=3
            )
            num_vals = const_expr(cute.size(tOcO, mode=[0]))
            if const_expr(self.use_vector_initial):
                assert self.dtype in [cutlass.BFloat16, cutlass.Float16]
                assert num_vals % 2 == 0
                g2r_initial_pair_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    self.dtype,
                    num_bits_per_copy=32,
                )
                rInitialPair = cute.make_rmem_tensor((2,), self.dtype)
            for m in cutlass.range(num_rows, unroll_full=True):
                row_local = tOcO[0, m, 0][0]
                scale_initial = sInitialScale[row_local]
                if tOhidx[m] >= 0 and scale_initial > 0.0:
                    for k in cutlass.range(
                        cute.size(tOcO, mode=[2]), unroll_full=True
                    ):
                        if const_expr(self.use_vector_initial):
                            for v_pair in cutlass.range(
                                num_vals // 2, unroll_full=True
                            ):
                                v = v_pair * 2
                                fake_col = tOcO[v, 0, k][1]
                                if const_expr(self.use_stg128_fp8_layout):
                                    real_col = stg128_fp8_fake_col_to_real_col(
                                        fake_col
                                    )
                                elif const_expr(self.use_stg128_half_layout):
                                    real_col = stg128_half_fake_col_to_real_col(
                                        fake_col
                                    )
                                else:
                                    real_col = stg128_fake_col_to_real_col(fake_col)
                                initial_pair_ptr = cute.make_ptr(
                                    dtype=self.dtype,
                                    value=utils.elem_pointer(
                                        mO_initial_cur,
                                        (tOmidx[m], real_col, tOhidx[m]),
                                    ).toint(),
                                    mem_space=mO_initial_cur.iterator.memspace,
                                    assumed_align=4,
                                )
                                initial_pair = cute.make_tensor(
                                    initial_pair_ptr,
                                    cute.make_layout((2,), stride=(1,)),
                                )
                                cute.copy(
                                    g2r_initial_pair_atom,
                                    initial_pair,
                                    rInitialPair,
                                )
                                o0, o1 = cute.arch.mul_packed_f32x2(
                                    (
                                        Float32(rInitialPair[0]),
                                        Float32(rInitialPair[1]),
                                    ),
                                    (scale_initial, scale_initial),
                                )
                                tOrO[v, m, k] = o0
                                tOrO[v + 1, m, k] = o1
                        else:
                            for v in cutlass.range(num_vals, unroll_full=True):
                                fake_col = tOcO[v, 0, k][1]
                                if const_expr(self.use_stg128_fp8_layout):
                                    real_col = stg128_fp8_fake_col_to_real_col(fake_col)
                                elif const_expr(self.use_stg128_half_layout):
                                    real_col = stg128_half_fake_col_to_real_col(fake_col)
                                else:
                                    real_col = stg128_fake_col_to_real_col(fake_col)
                                tOrO[v, m, k] = (
                                    Float32(
                                        mO_initial_cur[
                                            tOmidx[m], real_col, tOhidx[m]
                                        ]
                                    )
                                    * scale_initial
                                )

        stage_load = self.stages - 1
        stage_compute = 0

        # Main accumulation loop
        for s in cutlass.range(thr_max_valid_split + 1, unroll=4):
            # Get scales for this split
            scale = cute.make_rmem_tensor(num_rows, Float32)
            for m in cutlass.range(num_rows, unroll_full=True):
                scale[m] = sLSE[s, tOcO[0, m, 0][0]]  # Get scale from smem

            # Load next stage if needed
            split_to_load = s + self.stages - 1
            if const_expr(self.use_tma_partial):
                if split_to_load <= thr_max_valid_split and warp_idx == Int32(0):
                    empty = o_tma_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_O_partial,
                        tOgO_tma[(None, split_to_load)],
                        tOsO_tma[(None, empty.index)],
                        tma_bar_ptr=empty.barrier,
                    )
                    empty.commit()
            else:
                if split_to_load <= thr_max_valid_split:
                    load_O_partial(split_to_load, stage_load)
                cute.arch.cp_async_commit_group()
                stage_load = 0 if stage_load == self.stages - 1 else stage_load + 1

            # Wait for the current stage to be ready
            if const_expr(self.use_tma_partial):
                full = o_tma_consumer.wait_and_advance()
                cute.autovec_copy(
                    tOsO_partial[None, None, None, full.index],
                    tOrO_partial,
                )
                # Every CTA thread owns part of this 64x128 tile. Signal
                # the 256-arrival empty barrier directly after each
                # thread has finished its register copy; PipelineTmaAsync
                # normally elects one release thread for a size-1 group.
                o_tma_pipeline.sync_object_empty.arrive(full.index, None)
            else:
                cute.arch.cp_async_wait_group(self.stages - 1)
                # Each thread reads only the rows it loaded into SMEM.
                cute.autovec_copy(
                    tOsO_partial[None, None, None, stage_compute],
                    tOrO_partial,
                )
                stage_compute = 0 if stage_compute == self.stages - 1 else stage_compute + 1

            # Accumulate scaled partial results
            for m in cutlass.range(num_rows, unroll_full=True):
                if tOhidx[m] >= 0 and scale[m] > 0.0:
                    tOrO[None, m, None].store(
                        tOrO[None, m, None].load()
                        + scale[m] * tOrO_partial[None, m, None].load().to(Float32)
                    )

        # Flush any outstanding async-copy groups before the local Step-7
        # permutation buffer is read on the tail of the kernel.
        if const_expr(self.use_tma_partial):
            if warp_idx == Int32(0):
                o_tma_producer.tail()
        else:
            cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # ===============================
        # Step 7: Write final O to gmem (fake→real via SMEM)
        # ===============================

        mO_cur = seqlen_info.offset_batch(mO, batch_idx, dim=3)
        if const_expr(cu_seqlens is None):
            mO_cur = mO[None, None, None, batch_idx]
        else:
            mO_cur = cute.domain_offset((offset, 0, 0), mO)
        mO_cur = utils.domain_offset_aligned((0, k_block * self.k_block_size, 0), mO_cur)
        num_vals = const_expr(cute.size(tOcO, mode=[0]))
        if const_expr(not use_pdl):
            # Direct / standalone calls don't participate in the K1->K2
            # dependency chain. Use a simple per-element real-column store
            # path here to keep mixed-shape launches stable.
            for m in cutlass.range(num_rows, unroll_full=True):
                if tOhidx[m] >= 0:
                    for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                        if const_expr(self.is_even_k) or tOpO[k]:
                            for v in cutlass.range(num_vals, unroll_full=True):
                                fake_col = tOcO[v, 0, k][1]
                                if const_expr(self.use_stg128_fp8_layout):
                                    real_col = stg128_fp8_fake_col_to_real_col(fake_col)
                                elif const_expr(self.use_stg128_half_layout):
                                    real_col = stg128_half_fake_col_to_real_col(fake_col)
                                else:
                                    real_col = stg128_fake_col_to_real_col(fake_col)
                                o_val = tOrO[v, m, k]
                                if const_expr(mOutputScale is not None):
                                    o_val = o_val * output_scale
                                mO_cur[tOmidx[m], real_col, tOhidx[m]] = o_val.to(self.dtype)
        else:
            # 7a: fp32 accumulator -> output dtype SMEM with fake→real
            # permutation. The dedicated permutation buffer stays separate
            # from the O_partial pipeline staging buffer.
            sO_perm = sO_perm_buf

            if const_expr(self.dtype in [cutlass.BFloat16, cutlass.Float16]):
                # O_partial uses a dtype-specific STG.128 fake layout, but
                # sO_perm is in the final O dtype. For all supported fake
                # layouts, adjacent fake pairs map to adjacent real columns,
                # so write the final BF16/F16 O pair as one 32-bit SMEM store.
                assert num_vals % 2 == 0
                r2s_o_pair_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    cutlass.Int32,
                    num_bits_per_copy=32,
                )
                rO_pair_word = cute.make_rmem_tensor((1,), cutlass.Int32)
                sO_perm_i32_base = cute.make_ptr(
                    dtype=cutlass.Int32,
                    value=sO_perm.iterator.toint(),
                    mem_space=sO_perm.iterator.memspace,
                    assumed_align=4,
                )
                sO_perm_i32_row_stride = Int32((self.k_block_size + 16) // 2)
                for m in cutlass.range(num_rows, unroll_full=True):
                    row_local = tOcO[0, m, 0][0]
                    if tOhidx[m] >= 0:
                        for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                            for v_pair in cutlass.range(num_vals // 2, unroll_full=True):
                                v = v_pair * 2
                                fake_col = tOcO[v, 0, k][1]
                                if const_expr(self.use_stg128_fp8_layout):
                                    real_col = stg128_fp8_fake_col_to_real_col(fake_col)
                                elif const_expr(self.use_stg128_half_layout):
                                    real_col = stg128_half_fake_col_to_real_col(fake_col)
                                else:
                                    real_col = stg128_fake_col_to_real_col(fake_col)
                                o0 = tOrO[v, m, k]
                                o1 = tOrO[v + 1, m, k]
                                if const_expr(mOutputScale is not None):
                                    o0, o1 = cute.arch.mul_packed_f32x2(
                                        (o0, o1),
                                        (output_scale, output_scale),
                                    )
                                rO_pair_word[0] = utils.cvt_f16x2_f32(o0, o1, self.dtype)
                                smem_pair_ptr = cute.make_ptr(
                                    dtype=cutlass.Int32,
                                    value=(
                                        sO_perm_i32_base.toint()
                                        + Int64(
                                            row_local * sO_perm_i32_row_stride
                                            + real_col // Int32(2)
                                        )
                                        * Int64(4)
                                    ),
                                    mem_space=sO_perm.iterator.memspace,
                                    assumed_align=4,
                                )
                                sO_pair = cute.make_tensor(
                                    smem_pair_ptr,
                                    cute.make_layout((1,), stride=(1,)),
                                )
                                cute.copy(r2s_o_pair_atom, rO_pair_word, sO_pair)
            else:
                # 7a: iterate over ALL val elements in mode[0].
                # tOcO[v, m, k][1] gives different fake_col for each v.
                r2s_o_scalar_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    self.dtype,
                    num_bits_per_copy=self.dtype.width,
                )
                rO_scalar = cute.make_rmem_tensor((1,), self.dtype)
                for m in cutlass.range(num_rows, unroll_full=True):
                    row_local = tOcO[0, m, 0][0]
                    if tOhidx[m] >= 0:
                        for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                            for v in cutlass.range(num_vals, unroll_full=True):
                                fake_col = tOcO[v, 0, k][1]
                                if const_expr(self.use_stg128_fp8_layout):
                                    real_col = stg128_fp8_fake_col_to_real_col(fake_col)
                                elif const_expr(self.use_stg128_half_layout):
                                    real_col = stg128_half_fake_col_to_real_col(fake_col)
                                else:
                                    real_col = stg128_fake_col_to_real_col(fake_col)
                                o_val = tOrO[v, m, k]
                                if const_expr(mOutputScale is not None):
                                    o_val = o_val * output_scale
                                rO_scalar[0] = o_val.to(self.dtype)
                                smem_ptr = utils.elem_pointer(sO_perm, (row_local, real_col))
                                smem_scalar_ptr = cute.make_ptr(
                                    dtype=self.dtype,
                                    value=smem_ptr.toint(),
                                    mem_space=sO_perm.iterator.memspace,
                                    assumed_align=self.dtype.width // 8,
                                )
                                sO_scalar = cute.make_tensor(
                                    smem_scalar_ptr,
                                    cute.make_layout((1,), stride=(1,)),
                                )
                                cute.copy(r2s_o_scalar_atom, rO_scalar, sO_scalar)

            cute.arch.sync_threads()

            # 7b: SMEM (real order, output dtype) → GMEM
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOcO_store = gmem_thr_copy_O.partition_D(cO)
            tOsO_store = gmem_thr_copy_O.partition_D(sO_perm)
            rO = cute.make_rmem_tensor(tOcO_store.shape, self.dtype)
            elems_per_store = const_expr(cute.size(gmem_tiled_copy_O.layout_tv_tiled[1]))
            num_store_rows = const_expr(cute.size(tOcO_store, mode=[1]))
            num_store_vals = const_expr(cute.size(tOcO_store, mode=[0]))
            tOpO_store = None
            if const_expr(not self.is_even_k):
                tOpO_store = cute.make_rmem_tensor(cute.size(tOcO_store, mode=[2]), Boolean)
                for k in cutlass.range(cute.size(tOpO_store), unroll_full=True):
                    tOpO_store[k] = (
                        tOcO_store[0, 0, k][1]
                        < mO_partial.shape[1] - k_block * self.k_block_size
                    )

            # Read output dtype from SMEM (now in real column order).
            for m in cutlass.range(num_store_rows, unroll_full=True):
                for k in cutlass.range(cute.size(tOcO_store, mode=[2]), unroll_full=True):
                    if const_expr(self.is_even_k) or tOpO_store[k]:
                        cute.autovec_copy(tOsO_store[None, m, k], rO[None, m, k])

            # Write bf16 to GMEM using gmem_tiled_copy_O (same as original FA Step 7)
            for m in cutlass.range(num_store_rows, unroll_full=True):
                row_local = tOcO_store[0, m, 0][0]
                idx = m_block * self.tile_m + row_local
                if idx < max_idx:
                    m_idx, head_idx = self.decode_flat_row_idx(idx, head_divmod)
                    mO_cur_copy = cute.tiled_divide(
                        mO_cur[m_idx, None, head_idx], (elems_per_store,)
                    )
                    for k in cutlass.range(cute.size(tOcO_store, mode=[2]), unroll_full=True):
                        k_idx = tOcO_store[0, 0, k][1] // elems_per_store
                        if const_expr(self.is_even_k) or tOpO_store[k]:
                            cute.copy(gmem_thr_copy_O, rO[None, m, k], mO_cur_copy[None, k_idx])

    @cute.jit
    def load_O_partial(
        self,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        tOrOptr: cute.Tensor,
        tOsO_partial: cute.Tensor,
        tOhidx: cute.Tensor,
        tOSplitCount: cute.Tensor,
        tOpO: Optional[cute.Tensor],
        tOcO: cute.Tensor,
        mO_cur_partial_layout: cute.Layout,
        split: Int32,
        stage: Int32,
    ) -> None:
        elems_per_load = const_expr(cute.size(gmem_tiled_copy_O_partial.layout_tv_tiled[1]))
        tOsO_partial_cur = tOsO_partial[None, None, None, stage]
        for m in cutlass.range(cute.size(tOcO, [1]), unroll_full=True):
            if tOhidx[m] >= 0:
                o_gmem_ptr = cute.make_ptr(
                    tOsO_partial.element_type, tOrOptr[m], cute.AddressSpace.gmem, assumed_align=16
                )
                mO_partial_cur = cute.make_tensor(
                    o_gmem_ptr, cute.slice_(mO_cur_partial_layout, (0, None, None, 0))
                )
                mO_partial_cur_copy = cute.tiled_divide(mO_partial_cur, (elems_per_load,))
                for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                    k_idx = tOcO[0, 0, k][1] // elems_per_load
                    if split < tOSplitCount[m] and (const_expr(tOpO is None) or tOpO[k]):
                        cute.copy(
                            gmem_tiled_copy_O_partial,
                            mO_partial_cur_copy[None, k_idx, split],
                            tOsO_partial_cur[None, m, k],
                        )
                    else:
                        tOsO_partial_cur[None, m, k].fill(0)


def _get_cutlass_dtype(torch_dtype: torch.dtype):
    if torch_dtype not in torch2cute_dtype_map:
        raise TypeError(f"Unsupported dtype: {torch_dtype}")
    return torch2cute_dtype_map[torch_dtype]


_combine_compile_cache = {}


def _get_cpasync_smem_layout_atom(dtype: Type[cutlass.Numeric], k_dim: int) -> cute.ComposedLayout:
    dtype_byte = const_expr(dtype.width // 8)
    bytes_per_row = const_expr(k_dim * dtype_byte)
    smem_k_block_size = (
        const_expr(
            128
            if bytes_per_row % 128 == 0
            else (64 if bytes_per_row % 64 == 0 else (32 if bytes_per_row % 32 == 0 else 16))
        )
        // dtype_byte
    )
    swizzle_bits = (
        4
        if smem_k_block_size == 128
        else (3 if smem_k_block_size == 64 else (2 if smem_k_block_size == 32 else 1))
    )
    swizzle_base = 2 if dtype_byte == 4 else (3 if dtype_byte == 2 else 4)
    return cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, swizzle_base, swizzle_base),
        0,
        cute.make_ordered_layout(
            (8 if const_expr(k_dim % 32 == 0) else 16, smem_k_block_size),
            order=(1, 0),
        ),
    )


def combine(
    o_partial_fake,
    lse_partial,
    o_out,
    lse_out,
    *,
    initial_o=None,
    initial_lse=None,
    lse_temperature_partial=None,
    lse_temperature_out=None,
    cu_seqlens=None,
    seqused=None,
    split_counts=None,
    output_scale=None,
    use_pdl=False,
    use_tma_partial=False,
    use_vector_initial=False,
):
    """K2: merge sparse forward split partials into the final output.

    STG.128 fake-layout handling remains an internal implementation detail.
    When lse_out is provided, the kernel writes the final authoritative
    log-sum-exp for each query row/head directly into that tensor.

    Args:
        o_partial_fake:
            Batched: [num_splits, batch, Sq, head_q, dim]
            Varlen:  [num_splits, total_q, head_q, dim]
        lse_partial:
            Batched: [num_splits, batch, Sq, head_q]
            Varlen:  [num_splits, total_q, head_q]
        o_out:
            Batched: [batch, Sq, head_q, dim]
            Varlen:  [total_q, head_q, dim]
        lse_out:
            Batched: [batch, Sq, head_q]
            Varlen:  [total_q, head_q]
        initial_o:
            Optional initial attention output state.  For the current varlen
            hybrid path this has shape [total_q, head_q, dim] and may alias
            o_out because combine CTAs own disjoint output tiles.
        initial_lse:
            Optional fixed-BSA LSE in its native [1, head_q, total_q] layout.
            Must be supplied together with initial_o.
        lse_temperature_partial:
            Optional temperature-scaled LSE partial with the same shape as
            lse_partial.
        lse_temperature_out:
            Optional temperature-scaled final LSE with the same shape as
            lse_out.
        cu_seqlens: Optional [batch + 1] int32 for varlen-Q combine.
        seqused: Optional [batch] int32 effective lengths for combine.
        split_counts: Optional int32 rowwise valid split counts prepared from
            q2k metadata. Batched: [batch, seqlen, head_kv]. Varlen:
            [total_q, head_kv].
        output_scale: Optional fp32 tensor with at least one element. When
            provided, the final O accumulator is multiplied once before store.
        use_pdl: When True, wait on PDL dependencies from the producer K1
            kernel. When False, launch without PDL waits.
        use_tma_partial: Stage full Dynamic partial-O tiles with TMA. This
            experimental path requires batch-1 varlen input with 16 valid
            Dynamic splits per row.
        use_vector_initial: Load adjacent Fixed-state output pairs with one
            aligned 32-bit transaction before exact online-softmax merging.
    """
    D = o_partial_fake.shape[-1]
    num_splits = o_partial_fake.shape[0]
    has_initial_state = initial_o is not None
    grouped_initial = has_initial_state and initial_o.ndim == 4
    if has_initial_state != (initial_lse is not None):
        raise ValueError("initial_o and initial_lse must be provided together")
    if has_initial_state:
        if cu_seqlens is None or o_out.ndim != 3:
            raise NotImplementedError(
                "initial-state combine currently supports varlen output only"
            )
        initial_shape_valid = (
            initial_o.shape[1] == o_out.shape[0]
            and initial_o.shape[0] * initial_o.shape[2] == o_out.shape[1]
            and initial_o.shape[3] == D
            if grouped_initial else initial_o.shape == o_out.shape
        )
        if not initial_shape_valid or initial_o.dtype != o_out.dtype:
            raise ValueError("initial_o must be [Q,Hq,D] or [Hkv,Q,GQA,D] with output dtype")
        if not initial_o.is_contiguous():
            raise ValueError("initial_o must be contiguous in its declared layout")
        if initial_o.device != o_out.device:
            raise ValueError("initial_o and o_out must be on the same device")
        expected_initial_lse = (1, o_out.shape[-2], o_out.shape[0])
        if tuple(initial_lse.shape) != expected_initial_lse:
            raise ValueError(
                "initial_lse must have fixed-BSA shape "
                f"{expected_initial_lse}, got {tuple(initial_lse.shape)}"
            )
        if initial_lse.dtype != torch.float32 or initial_lse.device != o_out.device:
            raise ValueError("initial_lse must be float32 on the output device")
    return_temperature_lse = (
        lse_temperature_partial is not None or lse_temperature_out is not None
    )
    if (lse_temperature_partial is None) != (lse_temperature_out is None):
        raise ValueError(
            "lse_temperature_partial and lse_temperature_out must either both be provided or both be None"
        )
    if lse_temperature_partial is not None and lse_temperature_partial.shape != lse_partial.shape:
        raise ValueError(
            "lse_temperature_partial must have the same shape as lse_partial, "
            f"got {lse_temperature_partial.shape} vs {lse_partial.shape}"
        )
    if lse_temperature_out is not None:
        if lse_out is None:
            raise ValueError("lse_temperature_out requires lse_out")
        if lse_temperature_out.shape != lse_out.shape:
            raise ValueError(
                "lse_temperature_out must have the same shape as lse_out, "
                f"got {lse_temperature_out.shape} vs {lse_out.shape}"
            )
        if lse_temperature_out.dtype != torch.float32 or lse_temperature_partial.dtype != torch.float32:
            raise TypeError("temperature LSE tensors must be torch.float32")

    partial_dtype = _get_cutlass_dtype(o_partial_fake.dtype)
    out_dtype = _get_cutlass_dtype(o_out.dtype)
    if output_scale is not None:
        if output_scale.dtype != torch.float32:
            raise TypeError(f"output_scale must be torch.float32, got {output_scale.dtype}")
        if output_scale.numel() < 1:
            raise ValueError("output_scale must contain at least one element")
        if output_scale.device != o_out.device:
            raise ValueError("output_scale must be on the same device as o_out")
        output_scale = output_scale.contiguous()
    if split_counts is not None:
        if split_counts.dtype != torch.int32:
            raise TypeError(f"split_counts must be torch.int32, got {split_counts.dtype}")
        if o_out.ndim == 4:
            if split_counts.ndim != 3:
                raise ValueError(
                    f"batched split_counts must have shape [batch, seqlen, head_kv], got {split_counts.shape}"
                )
            if split_counts.shape[:2] != o_out.shape[:2]:
                raise ValueError(
                    f"split_counts shape {split_counts.shape} must match batch/seqlen of o_out {o_out.shape}"
                )
        else:
            if cu_seqlens is None:
                raise ValueError("split_counts with varlen output requires cu_seqlens")
            if split_counts.ndim != 2:
                raise ValueError(
                    f"varlen split_counts must have shape [total_q, head_kv], got {split_counts.shape}"
                )
            if split_counts.shape[0] != o_out.shape[0]:
                raise ValueError(
                    f"split_counts total_q ({split_counts.shape[0]}) must match o_out total_q "
                    f"({o_out.shape[0]})"
                )
        if o_out.shape[-2] % split_counts.shape[-1] != 0:
            raise ValueError(
                f"o_out heads ({o_out.shape[-2]}) must be divisible by split_counts heads ({split_counts.shape[-1]})"
            )
        qheadperkv = o_out.shape[-2] // split_counts.shape[-1]
    else:
        qheadperkv = 1
    if cu_seqlens is not None:
        if cu_seqlens.dtype != torch.int32:
            raise TypeError(f"cu_seqlens must be torch.int32, got {cu_seqlens.dtype}")
        if cu_seqlens.ndim != 1:
            raise ValueError(f"cu_seqlens must be rank-1, got {cu_seqlens.shape}")
        if not cu_seqlens.is_contiguous():
            raise ValueError("cu_seqlens must be contiguous")
    if seqused is not None:
        if seqused.dtype != torch.int32:
            raise TypeError(f"seqused must be torch.int32, got {seqused.dtype}")
        if seqused.ndim != 1:
            raise ValueError(f"seqused must be rank-1, got {seqused.shape}")
        if not seqused.is_contiguous():
            raise ValueError("seqused must be contiguous")

    k_block_size = 128 if D > 64 else 64
    tile_m = 64
    has_cu_seqlens = cu_seqlens is not None
    has_seqused = seqused is not None
    has_lse = lse_out is not None
    has_split_counts = split_counts is not None
    has_output_scale = output_scale is not None
    min_blocks_per_mp = 3 if has_output_scale and use_pdl else 0

    key = (
        "combine",
        D,
        k_block_size,
        tile_m,
        num_splits,
        partial_dtype,
        out_dtype,
        has_cu_seqlens,
        has_seqused,
        has_lse,
        has_initial_state,
        initial_o.shape[2] if grouped_initial else 0,
        bool(return_temperature_lse),
        has_split_counts,
        has_output_scale,
        use_pdl,
        min_blocks_per_mp,
        bool(use_tma_partial),
        bool(use_vector_initial),
    )
    if key not in _combine_compile_cache:
        from mbsa.cute.src.common.aot_cache import try_load_aot, save_aot

        loaded = try_load_aot(key)
        if loaded is not None:
            _combine_compile_cache[key] = loaded
        else:
            from quack.compile_utils import make_fake_tensor

            kernel = SparseAttentionForwardCombine(
                dtype=out_dtype,
                dtype_partial=partial_dtype,
                head_dim=D,
                tile_m=tile_m,
                k_block_size=k_block_size,
                topk=num_splits,
                use_pdl=use_pdl,
                min_blocks_per_mp=min_blocks_per_mp,
                use_tma_partial=use_tma_partial,
                use_vector_initial=use_vector_initial,
                # Three stages overlap the next partial-state load with the
                # current exact online-softmax merge.  The Top-16 reduction,
                # output order, workspace ABI and numerical contract are
                # unchanged.
                stages=3,
            )
            div = 128 // partial_dtype.width
            if has_cu_seqlens:
                total_q, nheads = (cute.sym_int64() for _ in range(2))
                mO_partial = make_fake_tensor(
                    partial_dtype, (num_splits, total_q, nheads, D), divisibility=div
                )
                mLSE_partial = make_fake_tensor(
                    Float32, (num_splits, total_q, nheads), divisibility=1, leading_dim=2
                )
                mO = make_fake_tensor(
                    out_dtype, (total_q, nheads, D), divisibility=128 // out_dtype.width
                )
                mLSE = (
                    make_fake_tensor(Float32, (total_q, nheads), divisibility=1, leading_dim=1)
                    if has_lse
                    else None
                )
                mO_initial = (
                    make_fake_tensor(
                        out_dtype,
                        (cute.sym_int64(), total_q, initial_o.shape[2], D)
                        if grouped_initial else (total_q, nheads, D),
                        divisibility=128 // out_dtype.width,
                    )
                    if has_initial_state
                    else None
                )
                mLSE_initial = (
                    make_fake_tensor(
                        Float32,
                        (1, nheads, total_q),
                        divisibility=1,
                        leading_dim=2,
                    )
                    if has_initial_state
                    else None
                )
                mLSE_temperature_partial = (
                    make_fake_tensor(
                        Float32, (num_splits, total_q, nheads), divisibility=1, leading_dim=2
                    )
                    if return_temperature_lse
                    else None
                )
                mLSE_temperature = (
                    make_fake_tensor(Float32, (total_q, nheads), divisibility=1, leading_dim=1)
                    if return_temperature_lse
                    else None
                )
            else:
                batch, sq, nheads = (cute.sym_int64() for _ in range(3))
                mO_partial = make_fake_tensor(
                    partial_dtype, (num_splits, batch, sq, nheads, D), divisibility=div
                )
                mLSE_partial = make_fake_tensor(
                    Float32, (num_splits, batch, sq, nheads), divisibility=1, leading_dim=3
                )
                mO = make_fake_tensor(
                    out_dtype, (batch, sq, nheads, D), divisibility=128 // out_dtype.width
                )
                mLSE = (
                    make_fake_tensor(Float32, (batch, sq, nheads), divisibility=1, leading_dim=2)
                    if has_lse
                    else None
                )
                mO_initial = None
                mLSE_initial = None
                mLSE_temperature_partial = (
                    make_fake_tensor(
                        Float32, (num_splits, batch, sq, nheads), divisibility=1, leading_dim=3
                    )
                    if return_temperature_lse
                    else None
                )
                mLSE_temperature = (
                    make_fake_tensor(Float32, (batch, sq, nheads), divisibility=1, leading_dim=2)
                    if return_temperature_lse
                    else None
                )
            if not has_split_counts:
                mSplitCounts = None
            elif has_cu_seqlens:
                total_q_ctr, nheads_kv = (cute.sym_int64() for _ in range(2))
                mSplitCounts = make_fake_tensor(
                    Int32, (total_q_ctr, nheads_kv), divisibility=1, leading_dim=1
                )
            else:
                nheads_kv = cute.sym_int64()
                mSplitCounts = make_fake_tensor(
                    Int32, (batch, sq, nheads_kv), divisibility=1, leading_dim=2
                )
            mOutputScale = (
                make_fake_tensor(Float32, (cute.sym_int64(),), divisibility=1, leading_dim=0)
                if has_output_scale
                else None
            )
            stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

            _combine_compile_cache[key] = cute.compile(
                kernel,
                mO_partial,
                mLSE_partial,
                mO,
                mLSE,
                mO_initial,
                mLSE_initial,
                mLSE_temperature_partial,
                mLSE_temperature,
                None
                if cu_seqlens is None
                else make_fake_tensor(Int32, (cute.sym_int64(),), divisibility=1, leading_dim=0),
                None
                if seqused is None
                else make_fake_tensor(Int32, (cute.sym_int64(),), divisibility=1, leading_dim=0),
                None,
                None,
                None,
                mSplitCounts,
                mOutputScale,
                Int32(qheadperkv),
                stream,
                options="--enable-tvm-ffi",
            )
            save_aot(key, _combine_compile_cache[key])

    with torch.cuda.nvtx.range("K2_Combine"):
        _combine_compile_cache[key](
            o_partial_fake,
            lse_partial,
            o_out,
            lse_out,
            initial_o,
            initial_lse,
            lse_temperature_partial,
            lse_temperature_out,
            cu_seqlens,
            seqused,
            None,
            None,
            None,
            split_counts,
            output_scale,
            qheadperkv,
        )


def combine_shuffled(
    o_tile_rows: torch.Tensor,
    lse_tile_rows: torch.Tensor,
    partial_locations: torch.Tensor,
    split_counts: torch.Tensor,
    o_out: torch.Tensor,
    lse_out: torch.Tensor,
    *,
    initial_o: Optional[torch.Tensor] = None,
    initial_lse: Optional[torch.Tensor] = None,
) -> None:
    """Merge dense 128x128 MSA tiles through an output-centric location map."""
    if o_tile_rows.ndim != 2 or o_tile_rows.shape[1] != 128:
        raise ValueError("shuffled O must have shape [tile_rows, 128]")
    if lse_tile_rows.shape != (o_tile_rows.shape[0],):
        raise ValueError("shuffled LSE must have one value per tile row")
    if o_tile_rows.shape[0] % MSA_OUTPUT_TILE_ROWS != 0:
        raise ValueError("shuffled workspace must contain complete 128-row tiles")
    if (
        o_tile_rows.dtype != torch.bfloat16
        or lse_tile_rows.dtype != torch.float32
    ):
        raise TypeError("shuffled partials must be BF16 O and FP32 LSE")
    if (
        partial_locations.dtype != torch.int32
        or split_counts.dtype != torch.int32
    ):
        raise TypeError("tile locations and split counts must be int32")
    if partial_locations.ndim != 3 or split_counts.ndim != 2:
        raise ValueError(
            "tile locations must be [total_q,Hkv,topK] and counts "
            "[total_q,Hkv]"
        )
    total_q, head_kv, topk = partial_locations.shape
    if split_counts.shape != (total_q, head_kv):
        raise ValueError("tile locations and split counts disagree")
    if o_out.shape != (total_q, head_kv * 8, 128):
        raise ValueError("tile combine requires D128 GQA8 output")
    if lse_out.shape != o_out.shape[:2]:
        raise ValueError("tile combine LSE shape must match output rows")
    if (initial_o is None) != (initial_lse is None):
        raise ValueError("initial O/LSE must be provided together")
    tensors = (
        o_tile_rows,
        lse_tile_rows,
        partial_locations,
        split_counts,
        o_out,
        lse_out,
    )
    if not o_out.is_cuda:
        raise ValueError("tile combine requires CUDA tensors")
    if any(t.device != o_out.device for t in tensors):
        raise ValueError("tile combine tensors must share one CUDA device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("tile combine tensors must be contiguous")
    if initial_o is not None:
        initial_shape = (
            (head_kv, total_q, 8, 128) if initial_o.ndim == 4 else tuple(o_out.shape)
        )
        if tuple(initial_o.shape) != initial_shape or initial_o.dtype != torch.bfloat16:
            raise ValueError("initial O must be BF16 [Q,Hq,D] or [Hkv,Q,8,D]")
        if initial_lse.shape != (1, o_out.shape[1], total_q):
            raise ValueError("initial LSE must have shape [1,Hq,total_q]")
        if initial_lse.dtype != torch.float32:
            raise ValueError("initial LSE must be FP32")
        if (
            initial_o.device != o_out.device
            or initial_lse.device != o_out.device
            or not initial_o.is_contiguous()
            or not initial_lse.is_contiguous()
        ):
            raise ValueError("initial O/LSE must be contiguous on the output device")

    has_initial = initial_o is not None
    grouped_initial = has_initial and initial_o.ndim == 4
    key = (
        "combine_shuffled_qhs",
        (
            torch.cuda.get_device_capability(o_out.device),
            str(BaseDSL._get_dsl().get_arch_enum()),
        ),
        int(topk),
        has_initial,
        grouped_initial,
    )
    if key not in _tma_tile_combine_compile_cache:
        from quack.compile_utils import make_fake_tensor

        tile_rows_sym = cute.sym_int64()
        total_q_sym = cute.sym_int64()
        head_kv_sym = cute.sym_int64()
        # CuTe SymInt does not support host-side arithmetic. The concrete
        # launcher validation above enforces Hq == 8 * Hkv.
        head_q_sym = cute.sym_int64()
        kernel = SparseAttentionShuffledCombine(
            topk=int(topk),
            num_threads=128,
        )
        _tma_tile_combine_compile_cache[key] = cute.compile(
            kernel,
            make_fake_tensor(
                cutlass.BFloat16,
                (tile_rows_sym, 128),
                divisibility=8,
            ),
            make_fake_tensor(
                Float32,
                (tile_rows_sym,),
                divisibility=1,
                leading_dim=0,
            ),
            make_fake_tensor(
                Int32,
                (total_q_sym, head_kv_sym, topk),
                divisibility=1,
            ),
            make_fake_tensor(
                Int32,
                (total_q_sym, head_kv_sym),
                divisibility=1,
                leading_dim=1,
            ),
            make_fake_tensor(
                cutlass.BFloat16,
                (total_q_sym, head_q_sym, 128),
                divisibility=8,
            ),
            make_fake_tensor(
                Float32,
                (total_q_sym, head_q_sym),
                divisibility=1,
                leading_dim=1,
            ),
            (
                make_fake_tensor(
                    cutlass.BFloat16,
                    (head_kv_sym, total_q_sym, 8, 128)
                    if grouped_initial else (total_q_sym, head_q_sym, 128),
                    divisibility=8,
                )
                if has_initial
                else None
            ),
            (
                make_fake_tensor(
                    Float32,
                    (1, head_q_sym, total_q_sym),
                    divisibility=1,
                    leading_dim=2,
                )
                if has_initial
                else None
            ),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    with torch.cuda.nvtx.range("K2_Combine_Shuffled"):
        _tma_tile_combine_compile_cache[key](
            o_tile_rows,
            lse_tile_rows,
            partial_locations,
            split_counts,
            o_out,
            lse_out,
            initial_o,
            initial_lse,
        )
