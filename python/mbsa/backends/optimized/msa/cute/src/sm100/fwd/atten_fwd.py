# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SM100 sparse attention forward kernel.

This kernel implements the delivered Sparse Attention / Sparse Page Attention
forward contract:
- CSR sparse metadata (`k2q_row_ptr`, `k2q_q_indices`)
- varlen Q metadata via `cu_seqlens_q`
- Sparse Attention with flat varlen K/V
- Sparse Page Attention with paged K/V
"""

import math
from enum import IntEnum
from functools import partial
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.pipeline as cutlass_pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cutlass_dsl import BaseDSL

from quack import copy_utils

from ...common.cute_dsl_utils import assume_tensor_aligned
from ...common import utils
from ...common import pipeline
from ...common import mma_sm100_desc as sm100_desc
from ...common import blackwell_helpers as sm100_helpers
from ...common.softmax import SoftmaxSm100
from ...common.named_barrier import NamedBarrierFwdSm100
from ...common.mask import AttentionMask
from ...common.seqlen_info import SeqlenInfoQK
from .config import (
    Cfg,
    OutputLayout,
    OutputStore,
    QLoad,
)
# Shared raw PTX helpers and layout conversions used by the lean kernel.
from ...common.paged_kv import PagedKVManager
from ...common.tma_utils import (
    tma_gather4_cached,
    tma_gather4_prefetch,
    ragged_o_tma_coords,
    prefetch_tma_desc_raw,
    TMA_CACHE_EVICT_LAST,
    make_16x256b_tensor_mn_view,
    real_col_to_stg128_fake_col,
    real_col_to_stg128_fp8_fake_col,
    real_col_to_stg128_half_fake_col,
    stg_128_cs,
    stg_128_bf16_cs,
    stg_128_f16_cs,
    stg_128_fp8_e4m3_cs,
    tma_tile_store_5d,
)


@cute.jit
def _store_bf16x8_predicated(ptr, valid, v0, v1, v2, v3, v4, v5, v6, v7):
    """Convert uniformly, then predicate one public 128-bit CuTe copy."""
    values = cute.make_rmem_tensor((8,), Float32)
    values[0], values[1], values[2], values[3] = v0, v1, v2, v3
    values[4], values[5], values[6], values[7] = v4, v5, v6, v7
    packed = cute.make_rmem_tensor((8,), cutlass.BFloat16)
    packed.store(values.load().to(cutlass.BFloat16))
    # Each epilogue vector starts at an eight-BF16 boundary in the partial ABI.
    target = cute.make_tensor(
        cute.make_ptr(cutlass.BFloat16, ptr.toint(), cute.AddressSpace.gmem,
                      assumed_align=16),
        cute.make_layout((8,)),
    )
    mask = cute.make_rmem_tensor(cute.make_layout((8,), stride=(0,)), cutlass.Boolean)
    mask[0] = valid
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyR2GOp(), cutlass.BFloat16, num_bits_per_copy=128,
        l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_FIRST,
    )
    cute.copy(atom, packed, target, pred=mask)


@cute.jit
def _transpose_quad_bf16_pairs(values: cute.Tensor) -> cute.Tensor:
    """Transpose four BF16 pairs across each four-lane warp subgroup."""
    pairs = cute.recast_tensor(values, Uint32)
    stage = cute.make_rmem_tensor((4,), Uint32)
    result = cute.make_rmem_tensor((4,), Uint32)
    lane = cute.arch.lane_idx() & Int32(3)
    quad_mask_and_clamp = (cute.arch.WARP_SIZE - 4) << 8 | (
        cute.arch.WARP_SIZE - 1
    )
    for pair in cutlass.range_constexpr(4):
        peer = cute.arch.shuffle_sync_bfly(
            pairs[pair ^ 1],
            offset=1,
            mask_and_clamp=quad_mask_and_clamp,
        )
        stage[pair] = peer if ((lane ^ Int32(pair)) & Int32(1)) else pairs[pair]
    for pair in cutlass.range_constexpr(4):
        peer = cute.arch.shuffle_sync_bfly(
            stage[pair ^ 2],
            offset=2,
            mask_and_clamp=quad_mask_and_clamp,
        )
        result[pair] = (
            peer if ((lane ^ Int32(pair)) & Int32(2)) else stage[pair]
        )
    return cute.recast_tensor(result, cutlass.BFloat16)


@cute.jit
def _store_f32_predicated(ptr, valid, value):
    """Predicate the scalar LSE copy after uniform computation."""
    values = cute.make_rmem_tensor((1,), Float32)
    values[0] = value
    mask = cute.make_rmem_tensor((1,), cutlass.Boolean)
    mask[0] = valid
    target = cute.make_tensor(ptr, cute.make_layout((1,)))
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyR2GOp(), Float32, num_bits_per_copy=32,
        l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_FIRST,
    )
    cute.copy(atom, values, target, pred=mask)


class FiveWgRole(IntEnum):
    """Compile-time roles for the three consumer warp groups."""

    SOFTMAX = 0
    EPILOGUE_EVEN = 1
    EPILOGUE_ODD = 2


class ConsumerOperation(IntEnum):
    """Work performed by a consumer warp group for one O pipeline stage."""

    FUSED = 0
    SOFTMAX_ONLY = 1
    EPILOGUE_ONLY = 2


class FiveWgClcRole(IntEnum):
    """Compile-time roles for the five-WG persistent path."""

    SOFTMAX = 0
    EPILOGUE_STAGE0 = 1
    EPILOGUE_STAGE1 = 2
    Q_LOAD = 3
    MMA = 7
    K_LOAD_AND_CLC = 8
    V_LOAD = 9
    CLC_RESPONSE = 10
    O_STORE = 11


class FourWgClcRole(IntEnum):
    """Compile-time roles for the four-WG persistent path."""

    SOFTMAX_STAGE0 = 0
    SOFTMAX_STAGE1 = 1
    Q_LOAD = 2
    MMA_AND_KV = 3
    MMA = 4
    K_LOAD_AND_CLC = 5
    V_LOAD = 6
    O_STORE = 7


class SparseAttentionForwardSm100:
    """SM100 sparse attention forward kernel."""

    k_tile = 64  # UTCMMA bf16 K-tile (matches sparse_fwd_utcmma.py)

    def __init__(
        self,
        cfg: Cfg,
        head_dim: int = 128,
        qheadperkv: int = 16,
        m_block_size: int = 128,
        n_block_size: int = 128,
        paged_kv: bool = False,
        page_size: Optional[int] = None,
        has_seqused_k: bool = False,
        causal: bool = False,
        use_prepare_scheduler: bool = True,
        qk_dtype=None,
        pv_dtype=None,
        safe_fwd_sync: bool = True,
        shared_work_valid: bool = True,
        safe_partial_v_tail: bool = True,
        compact_batch1_schedule: bool = False,
        causal_softmax_registers: int = 176,
        causal_store_registers: int = 112,
        use_tmem_load_reduce: bool = False,
    ):
        if head_dim != 128:
            raise NotImplementedError(
                f"SparseAttentionForwardSm100 currently supports only D=128, got D={head_dim}"
            )
        self.head_dim = 128
        # Full-tile geometry is legal on SM100; TMEM load-reduce requires SM103.
        self.use_tmem_load_reduce = cfg.full_kv_blocks and use_tmem_load_reduce
        self.qheadperkv = qheadperkv
        self.use_q_cpasync = cfg.q_load is QLoad.CP_ASYNC
        self.use_q_gather4 = qheadperkv in (4, 2, 1) and not self.use_q_cpasync
        if qheadperkv not in (16, 8, 4, 2, 1):
            raise ValueError(
                "SparseAttentionForwardSm100 supports qheadperkv in "
                f"{{1, 2, 4, 8, 16}}, got {qheadperkv}"
            )
        self.tokens_per_gather4 = 4 // qheadperkv if self.use_q_gather4 else 0
        self.m_block_size = m_block_size          # 128 packed Q heads
        self.n_block_size = n_block_size          # 128 KV-block width
        self.paged_kv = paged_kv
        self.page_size = page_size
        self.has_seqused_k = has_seqused_k
        self.causal = causal
        self.qk_dtype_param = qk_dtype
        self.pv_dtype_param = pv_dtype
        self.cfg = cfg
        self.use_o_tma = cfg.output_store is OutputStore.TMA
        self.shuffled_output = cfg.output_layout is OutputLayout.SHUFFLED
        if self.use_o_tma and not self.shuffled_output:
            raise NotImplementedError(
                "UTMASTG requires the shuffled partial-output layout"
            )
        self.use_clc = cfg.use_clc
        # Compile-time validation control.  ``False`` reproduces the original
        # nine-warp TMEM lifecycle and shared q/split consumer handoff used by
        # B0; production/default builds use the sanitizer-clean converged path.
        self.safe_fwd_sync = safe_fwd_sync
        # Compile-time validation control for the C009 shared row-valid marker.
        # False retains C007's long-lived per-thread CTA-valid predicate.
        self.shared_work_valid = shared_work_valid
        # Compile-time control retained for exact parent-vs-candidate AB/BA.
        # Production enables the tail repair; False reproduces the corrected
        # parent's aligned native kernel and keeps partial pages unsupported.
        self.safe_partial_v_tail = safe_partial_v_tail and (
            not cfg.use_clc or not cfg.full_kv_blocks
        )
        self.masked_clc = cfg.use_clc and not cfg.full_kv_blocks
        if self.masked_clc and not safe_partial_v_tail:
            raise ValueError("masked CLC requires partial V-tail readiness")
        self.compact_batch1_schedule = bool(compact_batch1_schedule)
        if self.shuffled_output and (
            not cfg.full_kv_blocks
            or causal
            or qheadperkv != 8
            or not safe_fwd_sync
            or m_block_size != 128
            or n_block_size != 128
        ):
            raise ValueError(
                "shuffled output requires full noncausal BF16 D128 GQA8"
            )
        if not use_prepare_scheduler:
            raise ValueError("SparseAttentionForwardSm100 requires prepare scheduler")
        self.use_prepare_scheduler = True
        if self.paged_kv:
            if page_size is None:
                raise ValueError("page_size must be provided when paged_kv=True")
            if page_size != n_block_size:
                raise ValueError(
                    f"page_size ({page_size}) must equal blk_kv ({n_block_size})"
                )
        else:
            self.page_size = n_block_size
        self.q_tokens_per_group = m_block_size // qheadperkv   # 8

        self.mma_tiler_qk = (m_block_size, n_block_size, self.head_dim)
        self.mma_tiler_pv = (m_block_size, self.head_dim, n_block_size)
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32

        # Pipeline configuration -- deeper Q prefetch ring plus 2-slot S/O rings.
        self.q_stage = cfg.q_stages
        self.s_stage = 2
        self.o_stage = 2
        self.kv_stage = cfg.kv_stages
        # Sparse q_idx metadata ring bridging load -> epilogue. Sized larger
        # than the in-flight group distance so epilogue can reuse q_idx
        # without rereading mK2qIndices.
        self.qidx_meta_stages = 16

        self.k_stages = 2
        self.q_stage_stride_bytes = m_block_size * self.head_dim * 2
        self.k_tile_stride_bytes = m_block_size * self.k_tile * 2
        self.token_stride_bytes = qheadperkv * self.k_tile * 2

        # Warp layout: two softmax WGs, one Q-load/epilogue WG, one
        # MMA issue warp, two K/V load warps, and one empty warp.
        self.warps_per_group = 4
        self.use_five_wg_schedule = cfg.num_wgs == 5
        self.softmax0_warp_base = 0
        self.softmax1_warp_base = self.softmax0_warp_base + self.warps_per_group
        self.store_warp_base = self.softmax1_warp_base + self.warps_per_group
        if self.use_five_wg_schedule:
            self.store_warp_base = 3 * self.warps_per_group
        self.mma_warp_id = self.store_warp_base + self.warps_per_group
        self.load_warp_base = self.mma_warp_id + 1
        self.q_load_warp_base = self.store_warp_base
        self.kv_load_warp_base = self.load_warp_base
        self.num_kv_load_warps = 2
        self.num_q_load_warps = self.warps_per_group
        self.total_warps = cfg.num_wgs * self.warps_per_group
        self.threads_per_cta = cute.arch.WARP_SIZE * self.total_warps

        # TMEM layout follows FA SM100:
        #   S0/S1: [0:128], [128:256]
        #   O0/O1: [256:384], [384:512] for hdim_v=128
        #   P dtype follows the PV operand policy and is packed into each S tile.
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        self.tmem_s_offset = 0
        self.tmem_stage_stride = n_block_size
        self.tmem_o_stage_stride = self.head_dim
        self.tmem_o_offset = self.s_stage * n_block_size
        self.tmem_s_to_p_offset = n_block_size // 2
        self.tmem_p_offset = self.tmem_s_offset + self.tmem_s_to_p_offset
        raw_tmem_total = self.tmem_o_offset + self.o_stage * self.tmem_o_stage_stride
        # SM100 TMEM allocation requires a power-of-two column count.  The
        # 128-wide path naturally uses 512 columns; 64-wide KV blocks use 384
        # columns and must round the allocation up while keeping the same
        # logical offsets.
        self.tmem_total = 1 << (raw_tmem_total - 1).bit_length()

        # Let PV start once the first 3/4 of P is visible in TMEM. The final
        # split is synchronized by a separate mbarrier consumed inside PV MMA.
        self.split_P_arrive = n_block_size // 4 * 3
        self.split_P_arrive = int(self.split_P_arrive / 32) * 32
        assert self.split_P_arrive % 32 == 0
        assert self.split_P_arrive < self.n_block_size

        # Noncausal here describes this MSA kernel, independently of KV tail
        # validity. Five-WG supports full_kv_blocks=False and masks partial or
        # empty KV blocks. A causal attention pipeline can use noncausal MSA
        # for selected past blocks and handle its causal boundary separately.
        if self.use_five_wg_schedule and (
            causal
        ):
            raise ValueError(
                "five-WG MSA requires noncausal BF16 blocks"
            )
        if cfg.use_clc and (
            causal
        ):
            raise ValueError(
                "CLC requires noncausal BF16 blocks"
            )
        self.num_regs_softmax = cfg.regs[0]
        if self.use_five_wg_schedule:
            if cfg.regs[1] != cfg.regs[2]:
                raise ValueError(
                    "the two epilogue warp groups require equal register targets"
                )
            self.num_regs_correction = cfg.regs[1]
            self.num_regs_store = cfg.regs[3]
            self.num_regs_other = cfg.regs[4]
        else:
            if cfg.regs[1] != self.num_regs_softmax:
                raise ValueError(
                    "the two softmax warp groups require equal register targets"
                )
            self.num_regs_correction = self.num_regs_softmax
            self.num_regs_store = cfg.regs[2]
            self.num_regs_other = cfg.regs[3]
        self.num_regs_mma = self.num_regs_other
        self.num_regs_load = self.num_regs_other
        self.num_regs_empty = self.num_regs_other
        self.store_reg_decrease = self.num_regs_store <= 128
        self.ex2_emu_freq = 16 if causal else 0
        self.ex2_emu_start_frg = 1
        self.buffer_align_bytes = 1024

        # SM100 config.
        self.use_2cta_instrs = False
        self.cta_group_size = 1
        self.cluster_shape_mn = (1, 1)
        self.cluster_shape_mnk = (1, 1, 1)

        self.arch = BaseDSL._get_dsl().get_arch_enum()

    @cute.jit
    def _batch_q_offset(
        self,
        batch_idx: Int32,
        mCuSeqlensQ,
    ) -> Int32:
        return mCuSeqlensQ[batch_idx]

    @cute.jit
    def _logical_seqlen_k(
        self,
        batch_idx: Int32,
        mPageTable,
        mSeqUsedK,
        mCuSeqlensK,
    ) -> Int32:
        if const_expr(self.has_seqused_k):
            return mSeqUsedK[batch_idx]
        if const_expr(self.paged_kv):
            return Int32(mPageTable.shape[1]) * Int32(self.page_size)
        return mCuSeqlensK[batch_idx + Int32(1)] - mCuSeqlensK[batch_idx]

    @cute.jit
    def _valid_cols_in_block(
        self,
        batch_idx: Int32,
        kv_block_idx: Int32,
        mPageTable,
        mSeqUsedK,
        mCuSeqlensK,
    ) -> Int32:
        seqlen_k = self._logical_seqlen_k(
            batch_idx, mPageTable, mSeqUsedK, mCuSeqlensK
        )
        block_start = kv_block_idx * Int32(self.n_block_size)
        remaining = seqlen_k - block_start
        remaining = cutlass.max(remaining, Int32(0))
        return cutlass.min(remaining, Int32(self.n_block_size))

    @cute.jit
    def _load_q_idx(
        self,
        mK2qIndices: cute.Tensor,
        head_kv_idx: Int32,
        row_start: Int32,
        qi: Int32,
    ) -> Int32:
        return mK2qIndices[head_kv_idx, row_start + qi]

    @cute.jit
    def _load_qsplit_idx(
        self,
        mK2qQSplitIndices: cute.Tensor,
        head_kv_idx: Int32,
        row_start: Int32,
        qi: Int32,
    ) -> Int32:
        return mK2qQSplitIndices[head_kv_idx, row_start + qi]

    @cute.jit
    def _decode_q_idx_from_qsplit(self, qsplit: Int32) -> Int32:
        return qsplit & Int32(0x00FF_FFFF)

    @cute.jit
    def _decode_split_idx_from_qsplit(self, qsplit: Int32) -> Int32:
        return (qsplit >> Int32(24)) & Int32(0xFF)

    @cute.jit
    def _lower_bound_q_idx(
        self,
        mK2qIndices: cute.Tensor,
        head_kv_idx: Int32,
        row_start: Int32,
        count: Int32,
        q_value: Int32,
    ) -> Int32:
        left = Int32(0)
        right = count
        # k2q_q_indices is sorted by q_idx within each CSR row. A fixed
        # 32-step loop covers int32-sized rows and keeps this CTA-level.
        for _ in cutlass.range(32, unroll=1):
            if left < right:
                mid = (left + right) // Int32(2)
                q_idx = self._load_q_idx(
                    mK2qIndices,
                    head_kv_idx,
                    row_start,
                    mid,
                )
                if q_idx < q_value:
                    left = mid + Int32(1)
                else:
                    right = mid
        return left

    # ------------------------------------------------------------------
    # Host-side: TMA descriptors, SMEM layout, launch
    # ------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        mK: cute.Tensor,            # Sparse Attention: [total_k, head_kv, dim] / Sparse Page Attention: prepared paged KV tensor
        mV: cute.Tensor,            # Sparse Attention: [total_k, head_kv, dim] / Sparse Page Attention: prepared paged KV tensor
        mK2qIndices: cute.Tensor,   # csr payload: [head_kv, nnz]
        mK2qQSplitIndices: cute.Tensor,  # csr payload: [head_kv, nnz] packed q_idx/split slot
        mK2qCounts: cute.Tensor,    # csr row_ptr: [head_kv, total_rows + 1]
        mSchedulerMetadata: Optional[cute.Tensor],
        mWorkCount: Optional[cute.Tensor],
        mWorkTileOffsets: Optional[cute.Tensor],
        mO_partial: cute.Tensor,    # flattened partial-output storage
        mLSE_partial: cute.Tensor,  # fp32 LSE_partial
        mLSE_temperature_partial: Optional[cute.Tensor],  # fp32 temperature-scaled LSE_partial
        mQ_flat: cute.Tensor,       # [batch*Sq*head_q, dim] bf16, pre-flattened
        mQ_gather4_desc: Optional[cute.Tensor],  # [128] uint8 tensor map for gather4 Q load
        mO_ragged_tma_desc: Optional[cute.Tensor],
        mPageTable,
        mSeqUsedK,
        mCuSeqlensQ,
        mCuSeqlensK,
        softmax_scale: Float32,
        lse_temperature_scale: Float32,
        num_kv_blocks: Int32,
        num_heads_kv: Int32,
        seq_len_q: Int32,
        work_capacity: Int32,
        stream=None,
    ):
        self.q_dtype = mQ_flat.element_type
        self.k_input_dtype = mK.element_type
        self.v_input_dtype = mV.element_type
        self.qk_dtype = (
            self.q_dtype if const_expr(self.qk_dtype_param is None) else self.qk_dtype_param
        )
        if const_expr(self.pv_dtype_param is None):
            legacy_fp8_kv_cache = (
                self.q_dtype == cutlass.BFloat16
                and self.k_input_dtype == cutlass.Float8E4M3FN
                and self.v_input_dtype == cutlass.Float8E4M3FN
            )
            self.pv_dtype = cutlass.BFloat16 if legacy_fp8_kv_cache else self.v_input_dtype
        else:
            self.pv_dtype = self.pv_dtype_param
        self.k_dtype = self.qk_dtype
        self.v_dtype = self.pv_dtype
        self.p_dtype = self.pv_dtype
        if const_expr(self.q_dtype not in [cutlass.BFloat16, cutlass.Float8E4M3FN]):
            raise TypeError(f"Unsupported Q/K/V dtype: {self.q_dtype}")
        if const_expr(self.qk_dtype not in [cutlass.BFloat16, cutlass.Float8E4M3FN]):
            raise TypeError(f"Unsupported qk_dtype: {self.qk_dtype}")
        if const_expr(self.pv_dtype not in [cutlass.BFloat16, cutlass.Float8E4M3FN]):
            raise TypeError(f"Unsupported pv_dtype: {self.pv_dtype}")
        if const_expr(self.q_dtype != self.qk_dtype):
            raise TypeError("Q storage dtype must match qk_dtype")
        if const_expr(
            self.k_input_dtype != self.k_dtype
            and not (self.k_input_dtype == cutlass.Float8E4M3FN and self.k_dtype == cutlass.BFloat16)
        ):
            raise TypeError("Only FP8 K -> BF16 QK staging is supported")
        if const_expr(
            self.v_input_dtype != self.v_dtype
            and not (self.v_input_dtype == cutlass.Float8E4M3FN and self.v_dtype == cutlass.BFloat16)
        ):
            raise TypeError("Only FP8 V -> BF16 PV staging is supported")
        self.k_fp8_to_bf16 = (
            self.k_input_dtype == cutlass.Float8E4M3FN
            and self.k_dtype == cutlass.BFloat16
        )
        self.v_fp8_to_bf16 = (
            self.v_input_dtype == cutlass.Float8E4M3FN
            and self.v_dtype == cutlass.BFloat16
        )
        self.kv_fp8_to_bf16 = self.k_fp8_to_bf16 or self.v_fp8_to_bf16
        if const_expr(
            self.kv_fp8_to_bf16 and (self.use_five_wg_schedule or self.use_clc)
        ):
            raise ValueError(
                "K/V storage conversion requires the static four-WG dequantization roles"
            )
        self.qk_mma_kind = "f8f6f4" if const_expr(self.qk_dtype.width == 8) else "f16"
        self.pv_mma_kind = "f8f6f4" if const_expr(self.pv_dtype.width == 8) else "f16"
        elem_bytes = const_expr(self.q_dtype.width // 8)
        self.q_stage_stride_bytes = self.m_block_size * self.head_dim * elem_bytes
        self.k_tile_stride_bytes = self.m_block_size * self.k_tile * elem_bytes
        self.token_stride_bytes = self.qheadperkv * self.k_tile * elem_bytes
        p_cols_as_fp32 = const_expr(self.n_block_size * self.p_dtype.width // Float32.width)
        self.tmem_s_to_p_offset = self.n_block_size - p_cols_as_fp32
        self.tmem_p_offset = self.tmem_s_offset + self.tmem_s_to_p_offset
        self.o_dtype = mO_partial.element_type
        # A separate ownership ring carries Q routing past the Q-MMA lifetime.
        # The persistent counter must not restart at a CLC work-item boundary.
        self.use_qidx_meta_pipeline = (
            self.use_five_wg_schedule
            and not self.shuffled_output and self.qheadperkv in (1, 2, 4, 8)
            and self.o_dtype == cutlass.BFloat16
        )
        if const_expr(
            self.o_dtype
            not in [Float32, cutlass.BFloat16, cutlass.Float16, cutlass.Float8E4M3FN]
        ):
            raise TypeError(f"Unsupported O_partial dtype: {self.o_dtype}")
        if const_expr(
            self.shuffled_output
            and (
                self.o_dtype != cutlass.BFloat16
                or self.q_dtype != cutlass.BFloat16
                or self.k_input_dtype != cutlass.BFloat16
                or self.v_input_dtype != cutlass.BFloat16
            )
        ):
            raise TypeError("shuffled output requires BF16 Q/K/V and partial O")
        mK, mV = [assume_tensor_aligned(t) for t in (mK, mV)]

        if const_expr(not self.paged_kv):
            # Flat varlen K/V use CUTE-managed TMA descriptors, matching FA:
            # K: [total_k, h, d] -> [total_k, d, h].
            # V: [total_k, h, d] -> [d, total_k, h] for MN-major PV.
            layout_t = [0, 2, 1]
            mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=layout_t))
            mV_kv = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=layout_t))
            mV = cute.make_tensor(mV_kv.iterator, cute.select(mV_kv.layout, mode=[1, 0, 2]))
        else:
            # Sparse Page Attention with page-sized blocks can use the blocked
            # paged TMA layout directly. Host input is [page, head, token, dim].
            layout_t = [2, 3, 1, 0]
            mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=layout_t))
            mV_kv = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=layout_t))
            # V: (s,d,h,b) -> (d,s,h,b) for MN-major
            mV = cute.make_tensor(mV_kv.iterator, cute.select(mV_kv.layout, mode=[1, 0, 2, 3]))

        # ------------------------------------------------------------------
        #  UTCMMA TiledMma: QK^T and PV
        # ------------------------------------------------------------------
        cta_group = tcgen05.CtaGroup.ONE
        tiled_mma_qk = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
            Float32, cta_group, self.mma_tiler_qk[:2])
        tiled_mma_pv = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN,
            Float32, cta_group, self.mma_tiler_pv[:2], tcgen05.OperandSource.TMEM)

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,))

        # ------------------------------------------------------------------
        #  SMEM layouts: sQ/sK/sV only. O_partial is written directly from
        #  registers to GMEM in the epilogue.
        # ------------------------------------------------------------------
        total_q_stages = self.q_stage
        sQ_layout = sm100_utils.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, total_q_stages)
        q_load_tile = (
            self.head_dim
            if const_expr(self.q_dtype == cutlass.Float8E4M3FN)
            else self.k_tile
        )
        q_load_subtiles_per_token = const_expr(self.head_dim // q_load_tile)
        num_subtiles_total = (
            total_q_stages * self.q_tokens_per_group * q_load_subtiles_per_token
        )
        sQ_load_layout = sm100_utils.make_smem_layout(
            tcgen05.OperandMajorMode.K,
            (self.qheadperkv, q_load_tile), self.q_dtype, num_subtiles_total)
        sK_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage)
        sV_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_pv, self.mma_tiler_pv, self.v_dtype, self.kv_stage)
        sK_fp8_layout = cute.append(
            cute.make_layout(
                (self.n_block_size, self.head_dim),
                stride=(self.head_dim, 1),
            ),
            cute.make_layout((1,)),
        )
        sV_fp8_layout = cute.append(
            cute.make_layout(
                (self.head_dim, self.n_block_size),
                stride=(1, self.head_dim),
            ),
            cute.make_layout((1,)),
        )
        # P SMEM layout metadata (no actual SMEM allocation -- P lives in TMEM,
        # overlaying the S region; this layout is only used to compute the PV
        # A-operand TMEM descriptor shape at the MMA issue site.)
        tP_layout = sm100_utils.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.p_dtype, self.s_stage)

        # ------------------------------------------------------------------
        #  TMA atoms
        # ------------------------------------------------------------------
        k_tma_layout = (
            cute.select(sK_fp8_layout, mode=[0, 1])
            if const_expr(self.k_fp8_to_bf16)
            else cute.select(sK_layout, mode=[0, 1, 2])
        )
        v_tma_layout = (
            cute.select(sV_fp8_layout, mode=[0, 1])
            if const_expr(self.v_fp8_to_bf16)
            else cute.select(sV_layout, mode=[0, 1, 2])
        )
        kv_tma_bytes = (
            cute.size_in_bytes(self.k_input_dtype, k_tma_layout)
            + cute.size_in_bytes(self.v_input_dtype, v_tma_layout))
        q_tma_bytes = cute.size_in_bytes(
            self.q_dtype, cute.select(sQ_layout, mode=[0, 1, 2]))
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        if const_expr(self.k_fp8_to_bf16):
            tma_atom_K, mK = cpasync.make_tiled_tma_atom(
                tma_load_op,
                mK,
                cute.select(sK_fp8_layout, mode=[0, 1]),
                (self.n_block_size, self.head_dim),
            )
        else:
            tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op, mK, cute.select(sK_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk, tiled_mma_qk, cta_layout_vmnk.shape)
        if const_expr(self.v_fp8_to_bf16):
            tma_atom_V, mV = cpasync.make_tiled_tma_atom(
                tma_load_op,
                mV,
                cute.select(sV_fp8_layout, mode=[0, 1]),
                (self.head_dim, self.n_block_size),
            )
        else:
            tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op, mV, cute.select(sV_layout, mode=[0, 1, 2]),
                self.mma_tiler_pv, tiled_mma_pv, cta_layout_vmnk.shape)

        # Q per-sub-tile TMA atom: bf16 uses two 64-element halves; fp8 uses
        # one 128-element row because 128 fp8 elements occupy the same 128B
        # swizzle span as 64 bf16 elements.
        mQ_flat = assume_tensor_aligned(mQ_flat)
        mQ_2d = cute.make_tensor(
            mQ_flat.iterator, cute.select(mQ_flat.layout, mode=[0, 1]))
        if const_expr(self.use_q_gather4 or self.use_q_cpasync):
            # Gather4 and cp.async keep Q as a plain GMEM tensor; the unused
            # atom preserves the common kernel signature.
            tma_atom_Q = tma_atom_V
        else:
            tma_atom_Q, mQ_2d = cpasync.make_tiled_tma_atom(
                tma_load_op, mQ_2d,
                cute.select(sQ_load_layout, mode=[0, 1]),
                (self.qheadperkv, q_load_tile))
        q_subtile_bytes = cute.size_in_bytes(
            self.q_dtype, cute.select(sQ_load_layout, mode=[0, 1]))

        softmax_scale_log2 = softmax_scale * Float32(math.log2(math.e))
        lse_temperature_scale_log2 = softmax_scale_log2 * lse_temperature_scale

        # ------------------------------------------------------------------
        #  SharedStorage -- lean: just the mbars and tiles we actually use.
        #
        #  Mbarriers (all storage rings stay below the 64-per-CTA limit):
        #    mbar_kv           [2 * kv_stage]  K/V producer-consumer rings
        #    mbar_q            [q_stage * 2]  Q producer/consumer ring
        #    mbar_s            [2]  QK UTCMMA -> softmax (full + empty)
        #    mbar_o            [2]  PV UTCMMA -> epilogue (full + empty)
        #    mbar_p            [s_stage * 2]  softmax early-P arrive -> PV
        #    mbar_p_lastsplit  [s_stage * 2]  softmax final-P arrive -> PV
        #                      (used only when ``self.split_P_arrive > 0``)
        #    mbar_sm_stats     [s_stage * 2]  softmax row_sum/row_max
        #                      publish -> epilogue consumer read.  In lean
        #                      1-WG topology the producer and consumer are
        #                      the same 128 WG_C threads, but we keep the
        #                      barrier for structural parity with FA so the
        #                      softmax body reads identically.
        # ------------------------------------------------------------------
        @cute.struct
        class SharedStorage:
            clc_mbar: cute.struct.MemRange[Int64, 4 if self.cfg.use_clc else 0]
            clc_response: cute.struct.Align[
                cute.struct.MemRange[Int32, 8 if self.cfg.use_clc else 0], 16
            ]
            mbar_k: cute.struct.MemRange[Int64, 2 * self.kv_stage]
            mbar_v: cute.struct.MemRange[Int64, 2 * self.kv_stage]
            if const_expr(self.k_fp8_to_bf16):
                mbar_k_tma: cute.struct.MemRange[Int64, 2]
            if const_expr(self.v_fp8_to_bf16 or self.safe_partial_v_tail):
                # FP8 uses it for dequantization; repaired BF16 partial pages
                # use it to delay V readiness until the tail is neutralized.
                mbar_v_tma: cute.struct.MemRange[
                    Int64, self.kv_stage if self.masked_clc else 2]
            mbar_q: cute.struct.MemRange[Int64, self.q_stage * 2]
            if const_expr(self.use_qidx_meta_pipeline):
                mbar_qidx_meta: cute.struct.MemRange[
                    Int64, self.qidx_meta_stages * 2]
            mbar_s: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_p: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_p_lastsplit: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_o: cute.struct.MemRange[Int64, self.o_stage * 2]
            mbar_sm_stats: cute.struct.MemRange[Int64, self.o_stage * 2]
            if const_expr(self.use_o_tma):
                mbar_o_store: cute.struct.MemRange[Int64, self.o_stage * 2]
            tmem_dealloc_mbar_ptr: Int64
            tmem_holding_buf: Int32
            # Per-row softmax stats cache (for epilogue LSE + rescale):
            #   [0 : m_block_size)            row_sum
            #   [m_block_size : 2*m_block_size) row_max
            sScale: cute.struct.MemRange[
                Float32, self.o_stage * self.m_block_size * 2]
            # Per-row temperature LSE row_sum cache. The row_max is shared with
            # sScale because lse_temperature_scale is positive.
            sScaleTemperature: cute.struct.MemRange[
                Float32, self.o_stage * self.m_block_size]
            # Per-token split_id from prepare-time per-edge metadata.
            sSplitIdx: cute.struct.MemRange[
                Int32, self.o_stage * self.q_tokens_per_group]
            # Per-token q_idx cache to avoid reloading sparse indices in epilogue.
            sQIdx: cute.struct.MemRange[
                Int32, self.o_stage * self.q_tokens_per_group]
            # Prefix length of q_idx-sorted row entries that may need causal
            # masking for this KV block. This is CTA-level metadata, not a
            # token-count cap.
            sDiagQCount: cute.struct.MemRange[Int32, 1]
            # CTA-wide row metadata, published once by tidx 0 and reused by
            # all warp-specialized roles:
            #   [0] batch_idx
            #   [1] kv_block_idx
            #   [2] row_start
            #   [3] count_raw
            #   [4] kv_valid_cols
            #   [5] q_batch_offset
            #   [6] k_batch_offset
            #   [7] causal_q_offset = seqlen_k - seqlen_q
            #   [8] seqlen_q, read dynamically from cu_seqlens_q
            sRowMeta: cute.struct.MemRange[Int32, 9]
            sPagedKvIdx: cute.struct.MemRange[Int32, 1]
            sQLoadMIdx: cute.struct.MemRange[
                Int32, self.q_stage * self.q_tokens_per_group]
            # Packed per-edge q/split metadata:
            #   low 24 bits = q_idx, high 8 bits = split slot.
            sQIdxMeta: cute.struct.MemRange[
                Int32, self.qidx_meta_stages * self.q_tokens_per_group]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(sV_layout)],
                self.buffer_align_bytes]
            if const_expr(self.k_fp8_to_bf16):
                sKFp8: cute.struct.Align[
                    cute.struct.MemRange[
                        self.k_input_dtype, cute.cosize(sK_fp8_layout)
                    ],
                    self.buffer_align_bytes]
            if const_expr(self.v_fp8_to_bf16):
                sVFp8: cute.struct.Align[
                    cute.struct.MemRange[
                        self.v_input_dtype, cute.cosize(sV_fp8_layout)
                    ],
                    self.buffer_align_bytes]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(sQ_layout)],
                self.buffer_align_bytes]
            if const_expr(self.use_o_tma):
                sOStore: cute.struct.Align[
                    cute.struct.MemRange[
                        cutlass.BFloat16,
                        self.o_stage * self.m_block_size * self.head_dim,
                    ],
                    self.buffer_align_bytes,
                ]

        self.shared_storage = SharedStorage
        num_ctas = work_capacity

        tma_atom_O, tma_O = tma_atom_Q, mQ_2d
        if const_expr(self.use_o_tma):
            # Row-major K_SW128 lets one TMA command cover 128 rows by 64
            # BF16 columns. The store warp issues both D halves explicitly so
            # source-read retirement cannot split one logical output tile.
            o_layout = cute.make_layout(
                (cute.size(mO_partial) // self.head_dim, self.head_dim),
                stride=(self.head_dim, 1),
            )
            o_tile = (self.m_block_size, self.head_dim // 2)
            o_tile_layout = cute.tile_to_shape(
                tcgen05.make_smem_layout_atom(
                    tcgen05.SmemLayoutAtomKind.K_SW128,
                    cutlass.BFloat16,
                ),
                o_tile,
                order=(0, 1),
            )
            tma_atom_O, tma_O = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                cute.make_tensor(mO_partial.iterator, o_layout),
                o_tile_layout,
                o_tile,
            )

        self.kernel(
            mK, mV, mK2qIndices, mK2qQSplitIndices, mK2qCounts,
            mSchedulerMetadata, mWorkCount, mWorkTileOffsets,
            mO_partial, mLSE_partial, mLSE_temperature_partial, mQ_2d, mQ_gather4_desc,
            mO_ragged_tma_desc,
            mPageTable, mSeqUsedK, mCuSeqlensQ, mCuSeqlensK,
            softmax_scale_log2, lse_temperature_scale_log2, lse_temperature_scale,
            sQ_layout, sQ_load_layout, sK_layout, sV_layout,
            sK_fp8_layout, sV_fp8_layout, tP_layout,
            tma_atom_K, tma_atom_V, tma_atom_Q, tma_atom_O, tma_O,
            tiled_mma_qk, tiled_mma_pv,
            kv_tma_bytes, q_tma_bytes, q_subtile_bytes,
            num_kv_blocks, num_heads_kv, seq_len_q, work_capacity,
        ).launch(
            grid=(num_ctas,),
            block=[self.threads_per_cta, 1, 1],
            smem=max(SharedStorage.size_in_bytes(), 49152),
            stream=stream,
            min_blocks_per_mp=1,
        )

    # ------------------------------------------------------------------
    # Device-side: kernel entry, dispatch by warpgroup
    # ------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        # Runtime tensors
        tma_K: cute.Tensor,
        tma_V: cute.Tensor,
        mK2qIndices: cute.Tensor,
        mK2qQSplitIndices: cute.Tensor,
        mK2qCounts: cute.Tensor,
        mSchedulerMetadata: Optional[cute.Tensor],
        mWorkCount: Optional[cute.Tensor],
        mWorkTileOffsets: Optional[cute.Tensor],
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mLSE_temperature_partial: Optional[cute.Tensor],
        mQ_2d: cute.Tensor,
        mQ_gather4_desc: Optional[cute.Tensor],
        mO_ragged_tma_desc: Optional[cute.Tensor],
        mPageTable,
        mSeqUsedK,
        mCuSeqlensQ,
        mCuSeqlensK,
        # Scalars
        softmax_scale_log2: Float32,
        lse_temperature_scale_log2: Float32,
        lse_temperature_scale: Float32,
        # Layouts
        sQ_layout: cute.ComposedLayout,
        sQ_load_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sK_fp8_layout: cute.Layout,
        sV_fp8_layout: cute.Layout,
        tP_layout: cute.ComposedLayout,
        # TMA atoms
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        tma_O: cute.Tensor,
        # MMA
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        # Transfer sizes
        kv_tma_bytes: cutlass.Constexpr[int],
        q_tma_bytes: cutlass.Constexpr[int],
        q_subtile_bytes: cutlass.Constexpr[int],
        # Iteration bounds
        num_kv_blocks: Int32,
        num_heads_kv: Int32,
        seq_len_q: Int32,
        work_capacity: Int32,
    ):
        # ------------------------------------------------------------------
        #  Thread / warp identity, CTA coordinate
        # ------------------------------------------------------------------
        bidx, _, _ = cute.arch.block_idx()
        row_linear = Int32(0)
        head_kv_idx = Int32(0)
        batch_idx = Int32(0)
        kv_block_idx = Int32(0)
        work_q_begin = Int32(0)
        work_q_count = Int32(0)
        work_row_start = Int32(0)
        cta_valid_work = True
        work_idx = bidx
        cta_valid_work = work_idx < mWorkCount[Int32(0)]
        if cta_valid_work:
            if const_expr(self.compact_batch1_schedule):
                work_row_start = mSchedulerMetadata[work_idx, Int32(0)]
                packed = mSchedulerMetadata[work_idx, Int32(1)]
                work_q_count = (packed & Int32(0xFFF)) + Int32(1)
                kv_block_idx = (packed >> Int32(12)) & Int32(0x1FFF)
                head_kv_idx = (packed >> Int32(25)) & Int32(0x3)
            else:
                head_kv_idx = mSchedulerMetadata[work_idx, Int32(0)]
                row_linear = mSchedulerMetadata[work_idx, Int32(1)]
                work_q_begin = mSchedulerMetadata[work_idx, Int32(2)]
                work_q_count = mSchedulerMetadata[work_idx, Int32(3)]
                batch_idx = mSchedulerMetadata[work_idx, Int32(4)]
                kv_block_idx = mSchedulerMetadata[work_idx, Int32(5)]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx = cute.arch.thread_idx()[0]
        head_q = num_heads_kv * Int32(self.qheadperkv)
        paged_kv_manager = (
            PagedKVManager.create(
                mPageTable,
                page_size=self.page_size,
                n_block_size=self.n_block_size,
            )
            if const_expr(self.paged_kv)
            else None
        )

        # Prefetch TMA descriptors (warp 0 once).
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            if const_expr(not self.use_q_gather4 and not self.use_q_cpasync):
                cpasync.prefetch_descriptor(tma_atom_Q)
            elif const_expr(self.use_q_gather4):
                with cute.arch.elect_one():
                    prefetch_tma_desc_raw(mQ_gather4_desc.iterator)
            if const_expr(self.use_o_tma):
                if const_expr(self.cfg.ragged_tma_store):
                    with cute.arch.elect_one():
                        prefetch_tma_desc_raw(mO_ragged_tma_desc.iterator)
                else:
                    cpasync.prefetch_descriptor(tma_atom_O)
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,))

        # ------------------------------------------------------------------
        #  SMEM allocation (all warps -- same SharedStorage type from __call__)
        # ------------------------------------------------------------------
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        if const_expr(self.k_fp8_to_bf16):
            sKFp8 = storage.sKFp8.get_tensor(sK_fp8_layout)
            mbar_k_tma_ptr = storage.mbar_k_tma.data_ptr()
        if const_expr(self.v_fp8_to_bf16):
            sVFp8 = storage.sVFp8.get_tensor(sV_fp8_layout)
        if const_expr(self.v_fp8_to_bf16 or self.safe_partial_v_tail):
            mbar_v_tma_ptr = storage.mbar_v_tma.data_ptr()
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sQ_load = storage.sQ.get_tensor(sQ_load_layout.outer, swizzle=sQ_load_layout.inner)
        sScale = storage.sScale.get_tensor(
            cute.make_layout(self.o_stage * self.m_block_size * 2))
        sScaleTemperature = storage.sScaleTemperature.get_tensor(
            cute.make_layout(self.o_stage * self.m_block_size))
        sSplitIdx = storage.sSplitIdx.get_tensor(
            cute.make_layout((self.o_stage * self.q_tokens_per_group,)))
        sQIdx = storage.sQIdx.get_tensor(
            cute.make_layout((self.o_stage * self.q_tokens_per_group,)))
        sDiagQCount = storage.sDiagQCount.get_tensor(cute.make_layout((1,)))
        sRowMeta = storage.sRowMeta.get_tensor(cute.make_layout((9,)))
        sPagedKvIdx = storage.sPagedKvIdx.get_tensor(cute.make_layout((1,)))
        sQLoadMIdx = storage.sQLoadMIdx.get_tensor(
            cute.make_layout((self.q_stage * self.q_tokens_per_group,)))
        sQIdxMeta = storage.sQIdxMeta.get_tensor(
            cute.make_layout((self.qidx_meta_stages * self.q_tokens_per_group,)))
        mbar_k_ptr = storage.mbar_k.data_ptr()
        mbar_v_ptr = storage.mbar_v.data_ptr()

        # ------------------------------------------------------------------
        #  TMEM allocator -- allocator warp 0 serves the whole CTA.
        # ------------------------------------------------------------------
        # Keep pointer broadcast converged across the whole valid CTA.  The
        # former 9-warp barrier was reached from three disjoint role branches,
        # which made its lifecycle opaque to synchronization checking.
        tmem_alloc_warps: cutlass.Constexpr[int] = (
            self.total_warps
            if self.safe_fwd_sync
            else self.warps_per_group * 2 + 1
        )
        tmem_alloc_threads = cute.arch.WARP_SIZE * tmem_alloc_warps
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
            num_threads=tmem_alloc_threads,
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # ------------------------------------------------------------------
        #  Warp-specialized pipelines.
        # ------------------------------------------------------------------
        ThreadCooperativeGroup = partial(
            cutlass_pipeline.CooperativeGroup, cutlass_pipeline.Agent.Thread)
        tma_thread = ThreadCooperativeGroup(1)
        mma_thread = ThreadCooperativeGroup(1)
        softmax_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * self.warps_per_group)
        epilogue_threads = softmax_threads

        pipeline_k = None
        pipeline_v = None
        clc_queue = None
        clc_producer = None
        clc_consumer = None
        clc_response_ptr = None
        if const_expr(self.use_clc):
            pipeline_k = pipeline.PipelineTmaUmma.create(
                barrier_storage=mbar_k_ptr,
                num_stages=self.kv_stage,
                producer_group=tma_thread,
                consumer_group=mma_thread,
                tx_count=kv_tma_bytes // 2,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
            if const_expr(self.masked_clc):
                # TMA completion is private to the V producer. Publish the
                # consumer slot only after any invalid token rows are zeroed.
                pipeline_v = pipeline.PipelineAsyncUmma.create(
                    barrier_storage=mbar_v_ptr,
                    num_stages=self.kv_stage,
                    producer_group=tma_thread,
                    consumer_group=mma_thread,
                    cta_layout_vmnk=cta_layout_vmnk,
                    defer_sync=True,
                )
                if tidx == 0:
                    for slot in cutlass.range_constexpr(self.kv_stage):
                        cute.arch.mbarrier_init(mbar_v_tma_ptr + slot, 1)
            else:
                pipeline_v = pipeline.PipelineTmaUmma.create(
                    barrier_storage=mbar_v_ptr,
                    num_stages=self.kv_stage,
                    producer_group=tma_thread,
                    consumer_group=mma_thread,
                    tx_count=kv_tma_bytes // 2,
                    cta_layout_vmnk=cta_layout_vmnk,
                    defer_sync=True,
                )
            clc_queue = cutlass_pipeline.PipelineClcFetchAsync.create(
                barrier_storage=storage.clc_mbar.data_ptr(),
                num_stages=2,
                tx_count=16,
                producer_group=ThreadCooperativeGroup(1),
                consumer_group=ThreadCooperativeGroup(self.threads_per_cta),
            )
            clc_producer = cutlass_pipeline.make_pipeline_state(
                cutlass_pipeline.PipelineUserType.Producer, 2
            )
            clc_consumer = cutlass_pipeline.make_pipeline_state(
                cutlass_pipeline.PipelineUserType.Consumer, 2
            )
            clc_response_ptr = storage.clc_response.data_ptr()

        if const_expr(self.use_q_cpasync):
            pipeline_q = pipeline.PipelineAsyncUmma.create(
                barrier_storage=storage.mbar_q.data_ptr(),
                num_stages=self.q_stage,
                producer_group=ThreadCooperativeGroup(
                    self.num_q_load_warps * cute.arch.WARP_SIZE
                ),
                consumer_group=mma_thread,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        else:
            pipeline_q = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.mbar_q.data_ptr(),
                num_stages=self.q_stage,
                producer_group=tma_thread,
                consumer_group=mma_thread,
                tx_count=q_tma_bytes,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        pipeline_s = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_s.data_ptr(),
            num_stages=self.s_stage,
            producer_group=mma_thread,
            consumer_group=softmax_threads,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_p = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_p.data_ptr(),
            num_stages=self.s_stage,
            producer_group=softmax_threads,
            consumer_group=mma_thread,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_p_lastsplit = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_p_lastsplit.data_ptr(),
            num_stages=self.s_stage,
            producer_group=softmax_threads,
            consumer_group=mma_thread,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_o = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_o.data_ptr(),
            num_stages=self.o_stage,
            producer_group=mma_thread,
            consumer_group=epilogue_threads,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_sm_stats = pipeline.PipelineAsync.create(
            barrier_storage=storage.mbar_sm_stats.data_ptr(),
            num_stages=self.o_stage,
            producer_group=softmax_threads,
            consumer_group=epilogue_threads,
            defer_sync=True,
        )
        pipeline_qidx_meta = None
        if const_expr(self.use_qidx_meta_pipeline):
            pipeline_qidx_meta = pipeline.PipelineAsync.create(
                barrier_storage=storage.mbar_qidx_meta.data_ptr(),
                num_stages=self.qidx_meta_stages,
                producer_group=ThreadCooperativeGroup(cute.arch.WARP_SIZE),
                consumer_group=epilogue_threads,
                defer_sync=True,
            )
        o_store_context = None
        if const_expr(self.use_o_tma):
            o_store_layout = cute.tile_to_shape(
                tcgen05.make_smem_layout_atom(
                    tcgen05.SmemLayoutAtomKind.K_SW128,
                    cutlass.BFloat16,
                ),
                (self.m_block_size, self.head_dim, self.o_stage),
                order=(0, 1, 2),
            )
            sOStore = storage.sOStore.get_tensor(
                o_store_layout.outer,
                swizzle=o_store_layout.inner,
            )
            pipeline_o_store = pipeline.PipelineAsync.create(
                barrier_storage=storage.mbar_o_store.data_ptr(),
                num_stages=self.o_stage,
                producer_group=epilogue_threads,
                consumer_group=ThreadCooperativeGroup(cute.arch.WARP_SIZE),
                defer_sync=True,
            )
            o_store_context = (sOStore, pipeline_o_store)
        # Cluster sync (no-op for 1CTA cluster).
        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        # ------------------------------------------------------------------
        #  Work count: how many Q tokens reference this CTA's KV block
        # ------------------------------------------------------------------
        k_tma_bytes = cute.size_in_bytes(
            self.k_input_dtype,
            cute.select(sK_fp8_layout, mode=[0, 1])
            if const_expr(self.k_fp8_to_bf16)
            else cute.select(sK_layout, mode=[0, 1, 2]))
        v_tma_bytes = cute.size_in_bytes(
            self.v_input_dtype,
            cute.select(sV_fp8_layout, mode=[0, 1])
            if const_expr(self.v_fp8_to_bf16)
            else cute.select(sV_layout, mode=[0, 1, 2]))
        if tidx == 0 and const_expr(not self.use_clc):
            row_batch_idx = batch_idx
            row_kv_block_idx = kv_block_idx
            if const_expr(self.compact_batch1_schedule):
                row_start = work_row_start
                count_raw = work_q_count
            else:
                base_row_start = mK2qCounts[head_kv_idx, row_linear]
                row_start = base_row_start + work_q_begin
                count_raw = work_q_count
            kv_valid_cols = (
                self._valid_cols_in_block(
                    row_batch_idx,
                    row_kv_block_idx,
                    mPageTable,
                    mSeqUsedK,
                    mCuSeqlensK,
                )
            )
            q_batch_offset = self._batch_q_offset(
                row_batch_idx, mCuSeqlensQ
            )
            k_batch_offset = (
                Int32(0)
                if const_expr(self.paged_kv)
                else mCuSeqlensK[row_batch_idx]
            )
            sRowMeta[0] = row_batch_idx
            sRowMeta[1] = row_kv_block_idx
            sRowMeta[2] = row_start
            sRowMeta[3] = count_raw
            sRowMeta[4] = kv_valid_cols
            sRowMeta[5] = q_batch_offset
            sRowMeta[6] = k_batch_offset
            seqlen_q = mCuSeqlensQ[row_batch_idx + Int32(1)] - q_batch_offset
            causal_q_offset = Int32(0)
            if const_expr(self.causal):
                seqlen_k = self._logical_seqlen_k(
                    row_batch_idx,
                    mPageTable,
                    mSeqUsedK,
                    mCuSeqlensK,
                )
                causal_q_offset = seqlen_k - seqlen_q
            sRowMeta[7] = causal_q_offset
            sRowMeta[8] = seqlen_q
            if const_expr(self.paged_kv):
                sPagedKvIdx[0] = paged_kv_manager.physical_block_index(
                    row_batch_idx, row_kv_block_idx
                )
            cute.arch.mbarrier_init(mbar_k_ptr, 1)
            cute.arch.mbarrier_init(mbar_v_ptr, 1)
            if const_expr(self.k_fp8_to_bf16):
                cute.arch.mbarrier_init(mbar_k_tma_ptr, 1)
                cute.arch.mbarrier_expect_tx(mbar_k_tma_ptr, k_tma_bytes)
            else:
                cute.arch.mbarrier_expect_tx(mbar_k_ptr, k_tma_bytes)
            if const_expr(self.v_fp8_to_bf16):
                cute.arch.mbarrier_init(mbar_v_tma_ptr, 1)
                cute.arch.mbarrier_expect_tx(mbar_v_tma_ptr, v_tma_bytes)
            elif const_expr(self.safe_partial_v_tail):
                cute.arch.mbarrier_init(mbar_v_tma_ptr, 1)
                if kv_valid_cols < Int32(self.n_block_size):
                    cute.arch.mbarrier_expect_tx(mbar_v_tma_ptr, v_tma_bytes)
                else:
                    cute.arch.mbarrier_expect_tx(mbar_v_ptr, v_tma_bytes)
            else:
                cute.arch.mbarrier_expect_tx(mbar_v_ptr, v_tma_bytes)
            diag_q_count = Int32(0)
            if const_expr(self.causal):
                row_has_visible_cols = (count_raw > Int32(0)) & (kv_valid_cols > Int32(0))
                if row_has_visible_cols:
                    kv_valid_end = (
                        row_kv_block_idx * Int32(self.n_block_size)
                        + kv_valid_cols
                    )
                    q_threshold = kv_valid_end - causal_q_offset
                    diag_q_count = self._lower_bound_q_idx(
                        mK2qIndices,
                        head_kv_idx,
                        row_start,
                        count_raw,
                        q_threshold,
                    )
            sDiagQCount[0] = diag_q_count
            if const_expr(self.shared_work_valid):
                if not cta_valid_work:
                    sRowMeta[0] = Int32(-1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_pv = tiled_mma_pv.get_slice(0)
        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        tStS_base = thr_mma_qk.make_fragment_C(qk_acc_shape)
        tStS = cute.make_tensor(
            tStS_base.iterator,
            cute.append(
                tStS_base.layout,
                cute.make_layout(
                    (self.s_stage,), stride=(self.tmem_stage_stride,)),
            ),
        )
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]
        tP_width_ratio = Float32.width // self.v_dtype.width
        tP_stage_stride = self.tmem_stage_stride * tP_width_ratio
        tOrP = cute.make_tensor(
            tOrP.iterator + self.tmem_p_offset * tP_width_ratio,
            cute.append(
                tOrP.layout,
                cute.make_layout((self.s_stage,), stride=(tP_stage_stride,)),
            ),
        )

        tmem_cols = self.tmem_total

        load_wg_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.LoadWG),
            num_threads=cute.arch.WARP_SIZE * self.num_q_load_warps,
        )
        if const_expr(self.kv_fp8_to_bf16):
            kv_load_barrier = pipeline.NamedBarrier(
                barrier_id=int(NamedBarrierFwdSm100.KvLoad),
                num_threads=cute.arch.WARP_SIZE * self.num_kv_load_warps,
            )
        if const_expr(self.k_fp8_to_bf16):
            kv_dequant_k_barrier = pipeline.NamedBarrier(
                barrier_id=int(NamedBarrierFwdSm100.KvDequantK),
                num_threads=cute.arch.WARP_SIZE * self.warps_per_group,
            )
        if const_expr(self.v_fp8_to_bf16):
            kv_dequant_v_barrier = pipeline.NamedBarrier(
                barrier_id=int(NamedBarrierFwdSm100.KvDequantV),
                num_threads=cute.arch.WARP_SIZE * self.warps_per_group,
            )
        sm_stats_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0),
            num_threads=cute.arch.WARP_SIZE * 2,
        )
        epilogue_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.StoreEpilogue),
            num_threads=cute.arch.WARP_SIZE * self.warps_per_group,
        )
        if (
            const_expr(not self.use_clc and not self.use_five_wg_schedule)
            and warp_idx == Int32(self.total_warps - 1)
        ):
            cute.arch.setmaxregister_decrease(self.num_regs_empty)

        # The MMA warp remains the sole allocator.  Every warp in a valid CTA
        # reaches the pointer broadcast before role dispatch; roles which do
        # not consume TMEM simply do not retrieve the pointer.
        # TmemAllocator predicates the hardware allocation to mma_warp_id.
        # Keep the API calls in the converged top-level control region so its
        # compile-time allocation accounting and the runtime barrier agree.
        if const_expr(self.safe_fwd_sync):
            tmem.allocate(tmem_cols)
            tmem.wait_for_alloc()

        q_load_thread_base = Int32(self.q_load_warp_base * cute.arch.WARP_SIZE)
        q_load_thread_end = Int32(
            (self.q_load_warp_base + self.num_q_load_warps) * cute.arch.WARP_SIZE
        )
        is_q_load_thread = tidx >= q_load_thread_base and tidx < q_load_thread_end
        if const_expr(
            not self.use_clc and not self.use_five_wg_schedule
        ) and is_q_load_thread and (
            sRowMeta[0] >= Int32(0)
            if const_expr(self.shared_work_valid)
            else cta_valid_work
        ):
            if self.store_reg_decrease:
                cute.arch.setmaxregister_decrease(self.num_regs_store)
            else:
                cute.arch.setmaxregister_increase(self.num_regs_store)
            row_start_load = sRowMeta[2]
            count_raw_load = sRowMeta[3]
            q_batch_offset_load = sRowMeta[5]
            # Do not gate on KV validity here; sparse entries past seqused_k
            # still need the all-masked path to produce neutral partials.
            has_work_load = count_raw_load > Int32(0)
            num_q_groups_load = (
                count_raw_load + Int32(self.q_tokens_per_group - 1)
            ) // Int32(self.q_tokens_per_group)
            if const_expr(self.use_q_gather4):
                self._wg_load_q_gather4(
                    mQ_2d,
                    mQ_gather4_desc,
                    mK2qQSplitIndices,
                    sQIdxMeta,
                    sQ,
                    pipeline_q,
                    load_wg_barrier,
                    num_q_groups_load,
                    count_raw_load,
                    has_work_load,
                    head_kv_idx,
                    row_start_load,
                    q_batch_offset_load,
                    num_heads_kv,
                )
            else:
                self._wg_load_q_tma(
                    tma_atom_Q,
                    mQ_2d,
                    mK2qQSplitIndices,
                    sQLoadMIdx,
                    sQIdxMeta,
                    sQ_load,
                    pipeline_q,
                    load_wg_barrier,
                    num_q_groups_load,
                    count_raw_load,
                    has_work_load,
                    head_kv_idx,
                    row_start_load,
                    q_batch_offset_load,
                    num_heads_kv,
                )

        if (
            const_expr(not self.use_clc and not self.use_five_wg_schedule)
            and warp_idx >= Int32(self.kv_load_warp_base)
            and warp_idx < Int32(self.kv_load_warp_base + self.num_kv_load_warps)
            and (
                sRowMeta[0] >= Int32(0)
                if const_expr(self.shared_work_valid)
                else cta_valid_work
            )
        ):
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            kv_block_idx_load = sRowMeta[1]
            k_batch_offset_load = sRowMeta[6]
            has_work_load = sRowMeta[3] > Int32(0)
            if const_expr(self.kv_fp8_to_bf16):
                self._wg_load_kv_maybe_cast(
                    tma_atom_K, tma_atom_V,
                    tma_K, tma_V,
                    sPagedKvIdx,
                    sK, sV,
                    sKFp8 if const_expr(self.k_fp8_to_bf16) else None,
                    sVFp8 if const_expr(self.v_fp8_to_bf16) else None,
                    tiled_mma_qk, tiled_mma_pv,
                    mbar_k_ptr, mbar_v_ptr,
                    mbar_k_tma_ptr if const_expr(self.k_fp8_to_bf16) else None,
                    mbar_v_tma_ptr
                    if const_expr(self.v_fp8_to_bf16 or self.safe_partial_v_tail)
                    else None,
                    kv_load_barrier,
                    has_work_load,
                    head_kv_idx, kv_block_idx_load,
                    k_batch_offset_load,
                    sRowMeta[4],
                )
            else:
                self._wg_load_kv(
                    tma_atom_K, tma_atom_V,
                    tma_K, tma_V,
                    sPagedKvIdx[0]
                    if const_expr(self.paged_kv)
                    else kv_block_idx_load,
                    sK, sV,
                    tiled_mma_qk, tiled_mma_pv,
                    mbar_k_ptr,
                    mbar_v_ptr,
                    mbar_v_tma_ptr if const_expr(self.safe_partial_v_tail) else None,
                    has_work_load,
                    head_kv_idx, kv_block_idx_load,
                    k_batch_offset_load,
                    sRowMeta[4],
                    pipeline_k,
                    pipeline_v,
                    Int32(0),
                )

        if const_expr(
            not self.use_clc and not self.use_five_wg_schedule
        ) and warp_idx == Int32(self.mma_warp_id) and (
            sRowMeta[0] >= Int32(0)
            if const_expr(self.shared_work_valid)
            else cta_valid_work
        ):
            cute.arch.setmaxregister_decrease(self.num_regs_mma)
            count_raw_mma = sRowMeta[3]
            has_work_mma = count_raw_mma > Int32(0)
            num_q_groups_mma = (
                count_raw_mma + Int32(self.q_tokens_per_group - 1)
            ) // Int32(self.q_tokens_per_group)
            if const_expr(not self.safe_fwd_sync):
                tmem.allocate(tmem_cols)
                tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self._wg_mma_issue(
                tiled_mma_qk, tiled_mma_pv,
                tOrP,
                sK, sV, sQ,
                pipeline_q, pipeline_s, pipeline_p, pipeline_p_lastsplit,
                pipeline_o,
                mbar_k_ptr, mbar_v_ptr,
                num_q_groups_mma, has_work_mma,
                Int32(0),
                Int32(0),
                pipeline_k,
                pipeline_v,
            )
            if const_expr(not self.safe_fwd_sync):
                tmem.relinquish_alloc_permit()
                tmem_alloc_barrier.arrive_and_wait()
                tmem.free(tmem_ptr, num_columns=tmem_cols)
                cute.arch.griddepcontrol_launch_dependents()

        if (
            const_expr(not self.use_clc)
            and const_expr(not self.use_five_wg_schedule)
            and warp_idx >= Int32(self.softmax0_warp_base)
            and warp_idx < Int32(self.softmax1_warp_base)
            and (
                sRowMeta[0] >= Int32(0)
                if const_expr(self.shared_work_valid)
                else cta_valid_work
            )
        ):
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            kv_block_idx_softmax = sRowMeta[1]
            count_raw_softmax = sRowMeta[3]
            kv_valid_cols_softmax = sRowMeta[4]
            causal_q_offset_softmax = sRowMeta[7]
            seqlen_q_softmax = sRowMeta[8]
            has_work_softmax = count_raw_softmax > Int32(0)
            num_q_groups_softmax = (
                count_raw_softmax + Int32(self.q_tokens_per_group - 1)
            ) // Int32(self.q_tokens_per_group)
            diag_q_count_softmax = sDiagQCount[0]
            if const_expr(self.k_fp8_to_bf16):
                self._wg_convert_fp8_kv_to_bf16_smem(
                    sKFp8,
                    sK,
                    mbar_k_tma_ptr,
                    mbar_k_ptr,
                    kv_dequant_k_barrier,
                    has_work_softmax,
                    self.softmax0_warp_base,
                    self.warps_per_group,
                    False,
                    Int32(self.n_block_size),
                )
            if const_expr(not self.safe_fwd_sync):
                tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self._wg_softmax(
                0,
                tiled_mma_qk,
                tiled_mma_pv,
                tStS,
                sScale,
                sScaleTemperature,
                sSplitIdx,
                sQIdx,
                sQIdxMeta,
                mK2qQSplitIndices,
                sRowMeta[2],
                pipeline_s, pipeline_p, pipeline_p_lastsplit,
                pipeline_o, pipeline_sm_stats,
                sm_stats_barrier,
                epilogue_barrier,
                mO_partial, mLSE_partial, mLSE_temperature_partial,
                softmax_scale_log2,
                lse_temperature_scale_log2,
                lse_temperature_scale,
                kv_block_idx_softmax,
                kv_valid_cols_softmax,
                diag_q_count_softmax,
                num_q_groups_softmax, count_raw_softmax, has_work_softmax,
                causal_q_offset_softmax,
                sRowMeta[0],
                head_kv_idx,
                seqlen_q_softmax,
                head_q,
                num_heads_kv,
                sRowMeta[5],
                mQ_2d,
                Int32(0),
                ConsumerOperation.FUSED.value,
                Int32(0),
                o_store_context,
                (
                    mWorkTileOffsets[work_idx]
                    if const_expr(self.shuffled_output)
                    else Int32(0)
                ),
            )
            if const_expr(not self.safe_fwd_sync):
                tmem_alloc_barrier.arrive()

        if (
            const_expr(not self.use_clc)
            and const_expr(not self.use_five_wg_schedule)
            and warp_idx >= Int32(self.softmax1_warp_base)
            and warp_idx < Int32(self.store_warp_base)
            and (
                sRowMeta[0] >= Int32(0)
                if const_expr(self.shared_work_valid)
                else cta_valid_work
            )
        ):
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            kv_block_idx_softmax = sRowMeta[1]
            count_raw_softmax = sRowMeta[3]
            kv_valid_cols_softmax = sRowMeta[4]
            causal_q_offset_softmax = sRowMeta[7]
            seqlen_q_softmax = sRowMeta[8]
            has_work_softmax = count_raw_softmax > Int32(0)
            num_q_groups_softmax = (
                count_raw_softmax + Int32(self.q_tokens_per_group - 1)
            ) // Int32(self.q_tokens_per_group)
            diag_q_count_softmax = sDiagQCount[0]
            if const_expr(self.v_fp8_to_bf16):
                self._wg_convert_fp8_kv_to_bf16_smem(
                    sVFp8,
                    sV,
                    mbar_v_tma_ptr,
                    mbar_v_ptr,
                    kv_dequant_v_barrier,
                    has_work_softmax,
                    self.softmax1_warp_base,
                    self.warps_per_group,
                    True,
                    (
                        kv_valid_cols_softmax
                        if const_expr(self.safe_partial_v_tail)
                        else Int32(self.n_block_size)
                    ),
                )
            if const_expr(not self.safe_fwd_sync):
                tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self._wg_softmax(
                1,
                tiled_mma_qk,
                tiled_mma_pv,
                tStS,
                sScale,
                sScaleTemperature,
                sSplitIdx,
                sQIdx,
                sQIdxMeta,
                mK2qQSplitIndices,
                sRowMeta[2],
                pipeline_s, pipeline_p, pipeline_p_lastsplit,
                pipeline_o, pipeline_sm_stats,
                sm_stats_barrier,
                epilogue_barrier,
                mO_partial, mLSE_partial, mLSE_temperature_partial,
                softmax_scale_log2,
                lse_temperature_scale_log2,
                lse_temperature_scale,
                kv_block_idx_softmax,
                kv_valid_cols_softmax,
                diag_q_count_softmax,
                num_q_groups_softmax, count_raw_softmax, has_work_softmax,
                causal_q_offset_softmax,
                sRowMeta[0],
                head_kv_idx,
                seqlen_q_softmax,
                head_q,
                num_heads_kv,
                sRowMeta[5],
                mQ_2d,
                Int32(0),
                ConsumerOperation.FUSED.value,
                Int32(0),
                o_store_context,
                (
                    mWorkTileOffsets[work_idx]
                    if const_expr(self.shuffled_output)
                    else Int32(0)
                ),
            )
            if const_expr(not self.safe_fwd_sync):
                tmem_alloc_barrier.arrive()

        if const_expr(not self.use_clc and not self.use_five_wg_schedule and self.use_o_tma):
            if (
                warp_idx == Int32(self.total_warps - 1)
                and (
                    sRowMeta[0] >= Int32(0)
                    if const_expr(self.shared_work_valid)
                    else cta_valid_work
                )
            ):
                self._warp_store_o(
                    tma_atom_O,
                    tma_O,
                    mO_ragged_tma_desc,
                    o_store_context,
                    sRowMeta[3],
                    Int32(0),
                    mWorkTileOffsets[work_idx],
                )
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                groups = cute.ceil_div(sRowMeta[3], self.q_tokens_per_group)
                if groups > Int32(0):
                    _, ready = o_store_context
                    ready.consumer_release_w_index((groups - Int32(1)) & Int32(1))

        if const_expr(not self.use_clc and self.use_five_wg_schedule):
            consumer_context = (
                epilogue_barrier,
                head_q,
                head_kv_idx,
                lse_temperature_scale,
                lse_temperature_scale_log2,
                mK2qQSplitIndices,
                mLSE_partial,
                mLSE_temperature_partial,
                mO_partial,
                mQ_2d,
                mWorkTileOffsets,
                num_heads_kv,
                o_store_context,
                pipeline_o,
                pipeline_p,
                pipeline_p_lastsplit,
                pipeline_s,
                pipeline_sm_stats,
                sQIdx,
                sQIdxMeta,
                sRowMeta,
                sScale,
                sScaleTemperature,
                sSplitIdx,
                sm_stats_barrier,
                softmax_scale_log2,
                tStS,
                tiled_mma_pv,
                tiled_mma_qk,
                tmem,
                work_idx,
                pipeline_qidx_meta,
            )
            q_context = (
                head_kv_idx,
                load_wg_barrier,
                mK2qQSplitIndices,
                mQ_2d,
                mQ_gather4_desc,
                sQ,
                num_heads_kv,
                pipeline_q,
                sQIdxMeta,
                sQLoadMIdx,
                sQ_load,
                sRowMeta,
                tma_atom_Q,
                pipeline_qidx_meta,
            )
            main_context = (
                head_kv_idx,
                mbar_k_ptr,
                mbar_v_ptr,
                mbar_v_tma_ptr,
                pipeline_k,
                pipeline_o,
                pipeline_p,
                pipeline_p_lastsplit,
                pipeline_q,
                pipeline_s,
                pipeline_sm_stats,
                pipeline_v,
                sK,
                sPagedKvIdx,
                sQ,
                sRowMeta,
                sV,
                tOrP,
                tStS,
                thr_mma_pv,
                thr_mma_qk,
                tiled_mma_pv,
                tiled_mma_qk,
                tma_K,
                tma_V,
                tma_atom_K,
                tma_atom_V,
                tmem,
                warp_idx,
            )
            if warp_idx < Int32(self.warps_per_group):
                self._run_five_wg_consumer(
                    FiveWgRole.SOFTMAX, consumer_context
                )
            elif warp_idx < Int32(2 * self.warps_per_group):
                self._run_five_wg_consumer(
                    FiveWgRole.EPILOGUE_EVEN, consumer_context
                )
            elif warp_idx < Int32(3 * self.warps_per_group):
                self._run_five_wg_consumer(
                    FiveWgRole.EPILOGUE_ODD, consumer_context
                )
            elif warp_idx < Int32(4 * self.warps_per_group):
                self._run_five_wg_q(q_context)
            elif const_expr(self.use_o_tma):
                if warp_idx == Int32(self.total_warps - 1):
                    # WG4 remains warp-group uniform for register allocation
                    # even though its last warp owns the asynchronous store.
                    cute.arch.setmaxregister_decrease(self.num_regs_other)
                    if sRowMeta[0] >= Int32(0):
                        self._warp_store_o(
                            tma_atom_O,
                            tma_O,
                            mO_ragged_tma_desc,
                            o_store_context,
                            sRowMeta[3],
                            Int32(0),
                            mWorkTileOffsets[work_idx],
                        )
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                        groups = cute.ceil_div(
                            sRowMeta[3], self.q_tokens_per_group
                        )
                        if groups > Int32(0):
                            _, ready = o_store_context
                            ready.consumer_release_w_index(
                                (groups - Int32(1)) & Int32(1)
                            )
                else:
                    self._run_five_wg_main(main_context)
            else:
                self._run_five_wg_main(main_context)

        if const_expr(self.use_clc):
            clc_context = (
                clc_queue,
                clc_producer,
                clc_consumer,
                clc_response_ptr,
                epilogue_barrier,
                head_q,
                load_wg_barrier,
                lse_temperature_scale,
                lse_temperature_scale_log2,
                mCuSeqlensK,
                mSeqUsedK,
                mCuSeqlensQ,
                mK2qCounts,
                mK2qQSplitIndices,
                mLSE_partial,
                mLSE_temperature_partial,
                mO_partial,
                mO_ragged_tma_desc,
                mPageTable,
                mQ_2d,
                mQ_gather4_desc,
                mSchedulerMetadata,
                mWorkCount,
                mWorkTileOffsets,
                mbar_k_ptr,
                mbar_v_ptr,
                mbar_v_tma_ptr if const_expr(self.masked_clc) else None,
                num_heads_kv,
                pipeline_k,
                pipeline_o,
                pipeline_p,
                pipeline_p_lastsplit,
                pipeline_q,
                pipeline_s,
                pipeline_sm_stats,
                pipeline_v,
                o_store_context,
                sK,
                sQ,
                sQIdx,
                sQIdxMeta,
                sQLoadMIdx,
                sQ_load,
                sScale,
                sScaleTemperature,
                sSplitIdx,
                sV,
                sm_stats_barrier,
                softmax_scale_log2,
                tOrP,
                tStS,
                thr_mma_pv,
                thr_mma_qk,
                tiled_mma_pv,
                tiled_mma_qk,
                tma_K,
                tma_V,
                tma_atom_K,
                tma_atom_Q,
                tma_atom_V,
                tma_atom_O,
                tma_O,
                tmem,
                warp_idx,
                work_idx,
                pipeline_qidx_meta,
            )
            if const_expr(self.use_five_wg_schedule):
                if warp_idx < Int32(4):
                    self._run_five_wg_clc_softmax(
                        (
                            clc_queue,
                            clc_producer,
                            clc_consumer,
                            clc_response_ptr,
                            mWorkCount,
                            mSchedulerMetadata,
                            mK2qQSplitIndices,
                            mPageTable,
                            mSeqUsedK,
                            mCuSeqlensK,
                            mLSE_temperature_partial,
                            pipeline_p,
                            pipeline_p_lastsplit,
                            pipeline_s,
                            pipeline_sm_stats,
                            sQIdxMeta,
                            sScale,
                            sScaleTemperature,
                            sm_stats_barrier,
                            softmax_scale_log2,
                            lse_temperature_scale_log2,
                            lse_temperature_scale,
                            tStS,
                            tiled_mma_qk,
                            tmem,
                            work_idx,
                        )
                    )
                elif warp_idx < Int32(8):
                    self._run_clc_specialized_role(
                        FiveWgClcRole.EPILOGUE_STAGE0.value, clc_context
                    )
                elif warp_idx < Int32(12):
                    self._run_clc_specialized_role(
                        FiveWgClcRole.EPILOGUE_STAGE1.value, clc_context
                    )
                elif warp_idx < Int32(16):
                    self._run_clc_specialized_role(
                        FiveWgClcRole.Q_LOAD.value, clc_context
                    )
                elif warp_idx == Int32(self.mma_warp_id):
                    self._run_clc_specialized_role(
                        FiveWgClcRole.MMA.value, clc_context
                    )
                elif warp_idx == Int32(self.kv_load_warp_base):
                    self._run_clc_specialized_role(
                        FiveWgClcRole.K_LOAD_AND_CLC.value, clc_context
                    )
                elif warp_idx == Int32(self.kv_load_warp_base + 1):
                    self._run_clc_specialized_role(
                        FiveWgClcRole.V_LOAD.value, clc_context
                    )
                elif const_expr(self.use_o_tma):
                    self._run_clc_specialized_role(
                        FiveWgClcRole.O_STORE.value, clc_context
                    )
                else:
                    self._run_clc_specialized_role(
                        FiveWgClcRole.CLC_RESPONSE.value, clc_context
                    )
            elif const_expr(self.use_o_tma):
                if warp_idx < Int32(4):
                    self._run_clc_specialized_role(
                        FourWgClcRole.SOFTMAX_STAGE0.value, clc_context
                    )
                elif warp_idx < Int32(8):
                    self._run_clc_specialized_role(
                        FourWgClcRole.SOFTMAX_STAGE1.value, clc_context
                    )
                elif warp_idx < Int32(12):
                    self._run_clc_specialized_role(
                        FourWgClcRole.Q_LOAD.value, clc_context
                    )
                elif warp_idx == Int32(self.mma_warp_id):
                    self._run_clc_specialized_role(
                        FourWgClcRole.MMA.value, clc_context
                    )
                elif warp_idx == Int32(self.kv_load_warp_base):
                    self._run_clc_specialized_role(
                        FourWgClcRole.K_LOAD_AND_CLC.value, clc_context
                    )
                elif warp_idx == Int32(self.kv_load_warp_base + 1):
                    self._run_clc_specialized_role(
                        FourWgClcRole.V_LOAD.value, clc_context
                    )
                else:
                    self._run_clc_specialized_role(
                        FourWgClcRole.O_STORE.value, clc_context
                    )
            else:
                if warp_idx < Int32(4):
                    self._run_clc_role(
                        FourWgClcRole.SOFTMAX_STAGE0.value, clc_context
                    )
                elif warp_idx < Int32(8):
                    self._run_clc_role(
                        FourWgClcRole.SOFTMAX_STAGE1.value, clc_context
                    )
                elif warp_idx < Int32(12):
                    self._run_clc_role(
                        FourWgClcRole.Q_LOAD.value, clc_context
                    )
                elif warp_idx == Int32(self.mma_warp_id):
                    self._run_clc_role(
                        FourWgClcRole.MMA.value, clc_context
                    )
                else:
                    self._run_clc_role(
                        FourWgClcRole.K_LOAD_AND_CLC.value, clc_context
                    )

        if const_expr(self.safe_fwd_sync):
            # All warp roles must finish before the allocator releases TMEM.
            # A converged CTA barrier makes teardown ownership explicit.
            tmem.relinquish_alloc_permit()
            cute.arch.barrier()
            tmem_ptr_free = tmem.retrieve_ptr(self.qk_acc_dtype)
            tmem.free(tmem_ptr_free, num_columns=tmem_cols)
            if const_expr(self.use_clc):
                if warp_idx == Int32(self.mma_warp_id):
                    cute.arch.griddepcontrol_launch_dependents()
            elif (
                sRowMeta[0] >= Int32(0)
                if const_expr(self.shared_work_valid)
                else cta_valid_work
            ) and warp_idx == Int32(self.mma_warp_id):
                cute.arch.griddepcontrol_launch_dependents()
    # ------------------------------------------------------------------
    # Warp-specialized helpers
    # ------------------------------------------------------------------

    @cute.jit
    def _run_five_wg_clc_softmax(self, context):
        """Run softmax without the epilogue and global output state."""
        cute.arch.setmaxregister_increase(self.num_regs_softmax)
        (
            clc_queue,
            clc_producer,
            clc_consumer,
            clc_response_ptr,
            mWorkCount,
            mSchedulerMetadata,
            mK2qQSplitIndices,
            mPageTable,
            mSeqUsedK,
            mCuSeqlensK,
            mLSE_temperature_partial,
            pipeline_p,
            pipeline_p_lastsplit,
            pipeline_s,
            pipeline_sm_stats,
            sQIdxMeta,
            sScale,
            sScaleTemperature,
            sm_stats_barrier,
            softmax_scale_log2,
            lse_temperature_scale_log2,
            lse_temperature_scale,
            tStS,
            tiled_mma_qk,
            tmem,
            work_idx,
        ) = context
        group_base = Int32(0)
        has_tile = cutlass.Boolean(True)
        while has_tile:
            (
                batch_idx,
                kv_block_idx,
                row_start,
                count_raw,
                kv_valid_cols,
                q_batch_offset,
                k_batch_offset,
                causal_q_offset,
                seq_len_q,
                head_kv_idx,
            ) = self._clc_count_work_metadata(
                work_idx,
                mWorkCount,
                mSchedulerMetadata,
                mPageTable,
                mSeqUsedK,
                mCuSeqlensK,
                need_valid_cols=True,
            )
            valid_work = batch_idx >= Int32(0)
            has_work = count_raw > Int32(0)
            num_q_groups = (
                count_raw + Int32(self.q_tokens_per_group - 1)
            ) // Int32(self.q_tokens_per_group)
            if valid_work:
                tmem.retrieve_ptr(self.qk_acc_dtype)
                for qi_group in cutlass.range(num_q_groups, unroll=1):
                    self._wg_softmax(
                        (group_base + qi_group) & Int32(1),
                        tiled_mma_qk,
                        None,
                        tStS,
                        sScale,
                        sScaleTemperature,
                        None,
                        None,
                        sQIdxMeta,
                        mK2qQSplitIndices,
                        row_start,
                        pipeline_s,
                        pipeline_p,
                        pipeline_p_lastsplit,
                        None,
                        pipeline_sm_stats,
                        sm_stats_barrier,
                        None,
                        None,
                        None,
                        mLSE_temperature_partial,
                        softmax_scale_log2,
                        lse_temperature_scale_log2,
                        lse_temperature_scale,
                        kv_block_idx,
                        kv_valid_cols,
                        Int32(0),
                        num_q_groups,
                        count_raw,
                        has_work,
                        causal_q_offset,
                        Int32(0),
                        head_kv_idx,
                        seq_len_q,
                        Int32(0),
                        Int32(0),
                        Int32(0),
                        None,
                        group_base,
                        ConsumerOperation.SOFTMAX_ONLY.value,
                        qi_group,
                    )

            group_base += cute.ceil_div(count_raw, self.q_tokens_per_group)
            clc_queue.consumer_wait(clc_consumer)
            response_ptr = clc_response_ptr + clc_consumer.index * Int32(4)
            response_ptr = cute.make_ptr(
                Int32,
                response_ptr.toint(),
                mem_space=response_ptr.memspace,
                assumed_align=16,
            )
            next_work, _, _, valid = cute.arch.clc_response(response_ptr)
            cute.arch.fence_proxy("async.shared", space="cta")
            clc_queue.consumer_release(clc_consumer)
            work_idx = next_work
            has_tile = valid != Int32(0)
            clc_producer.advance()
            clc_consumer.advance()

    @cute.jit
    def _run_clc_specialized_role(self, role: cutlass.Constexpr[int], context):
        """Run one persistent role with pipeline state carried across work items."""
        if const_expr(self.use_five_wg_schedule):
            if const_expr(role == FiveWgClcRole.SOFTMAX.value):
                cute.arch.setmaxregister_increase(self.num_regs_softmax)
            elif const_expr(
                role
                in (
                    FiveWgClcRole.EPILOGUE_STAGE0.value,
                    FiveWgClcRole.EPILOGUE_STAGE1.value,
                )
            ):
                cute.arch.setmaxregister_increase(self.num_regs_correction)
            elif const_expr(role == FiveWgClcRole.Q_LOAD.value):
                cute.arch.setmaxregister_decrease(self.num_regs_store)
            else:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
        else:
            if const_expr(
                role
                in (
                    FourWgClcRole.SOFTMAX_STAGE0.value,
                    FourWgClcRole.SOFTMAX_STAGE1.value,
                )
            ):
                cute.arch.setmaxregister_increase(self.num_regs_softmax)
            elif const_expr(role == FourWgClcRole.Q_LOAD.value):
                cute.arch.setmaxregister_decrease(self.num_regs_store)
            elif const_expr(self.num_regs_other > 128):
                cute.arch.setmaxregister_increase(self.num_regs_other)
            else:
                cute.arch.setmaxregister_decrease(self.num_regs_other)

        (
            clc_queue,
            clc_producer,
            clc_consumer,
            clc_response_ptr,
            epilogue_barrier,
            head_q,
            load_wg_barrier,
            lse_temperature_scale,
            lse_temperature_scale_log2,
            mCuSeqlensK,
            mSeqUsedK,
            mCuSeqlensQ,
            mK2qCounts,
            mK2qQSplitIndices,
            mLSE_partial,
            mLSE_temperature_partial,
            mO_partial,
            mO_ragged_tma_desc,
            mPageTable,
            mQ_2d,
            mQ_gather4_desc,
            mSchedulerMetadata,
            mWorkCount,
            mWorkTileOffsets,
            mbar_k_ptr,
            mbar_v_ptr,
            mbar_v_tma_ptr,
            num_heads_kv,
            pipeline_k,
            pipeline_o,
            pipeline_p,
            pipeline_p_lastsplit,
            pipeline_q,
            pipeline_s,
            pipeline_sm_stats,
            pipeline_v,
            o_store_context,
            sK,
            sQ,
            sQIdx,
            sQIdxMeta,
            sQLoadMIdx,
            sQ_load,
            sScale,
            sScaleTemperature,
            sSplitIdx,
            sV,
            sm_stats_barrier,
            softmax_scale_log2,
            tOrP,
            tStS,
            thr_mma_pv,
            thr_mma_qk,
            tiled_mma_pv,
            tiled_mma_qk,
            tma_K,
            tma_V,
            tma_atom_K,
            tma_atom_Q,
            tma_atom_V,
            tma_atom_O,
            tma_O,
            tmem,
            warp_idx,
            work_idx,
            pipeline_qidx_meta,
        ) = context

        group_base = Int32(0)
        work_seq = Int32(0)
        has_tile = cutlass.Boolean(True)
        while has_tile:
            if const_expr(
                self.use_five_wg_schedule
                and role
                in (FiveWgClcRole.SOFTMAX.value, FiveWgClcRole.MMA.value)
            ):
                work_meta = self._clc_count_work_metadata(
                    work_idx,
                    mWorkCount,
                    mSchedulerMetadata,
                )
            elif const_expr(
                self.use_five_wg_schedule
                and role
                in (
                    FiveWgClcRole.EPILOGUE_STAGE0.value,
                    FiveWgClcRole.EPILOGUE_STAGE1.value,
                    FiveWgClcRole.Q_LOAD.value,
                )
            ):
                work_meta = self._clc_q_work_metadata(
                    work_idx,
                    mWorkCount,
                    mSchedulerMetadata,
                    mK2qCounts,
                    mCuSeqlensQ,
                )
            elif const_expr(
                self.use_five_wg_schedule
                and role
                in (
                    FiveWgClcRole.K_LOAD_AND_CLC.value,
                    FiveWgClcRole.V_LOAD.value,
                )
            ):
                work_meta = self._clc_kv_work_metadata(
                    work_idx,
                    mWorkCount,
                    mSchedulerMetadata,
                    mCuSeqlensK,
                    mPageTable,
                    mSeqUsedK,
                )
            elif const_expr(
                self.use_five_wg_schedule
                and role == FiveWgClcRole.O_STORE.value
            ):
                work_meta = self._clc_count_work_metadata(
                    work_idx,
                    mWorkCount,
                    mSchedulerMetadata,
                )
            elif const_expr(
                self.use_five_wg_schedule
                and role == FiveWgClcRole.CLC_RESPONSE.value
            ):
                work_meta = (
                    Int32(0), Int32(0), Int32(0), Int32(0),
                    Int32(0), Int32(0), Int32(0), Int32(0),
                    Int32(0), Int32(0),
                )
            elif const_expr(
                not self.use_five_wg_schedule
                and self.use_o_tma
                and role
                in (
                    FourWgClcRole.SOFTMAX_STAGE0.value,
                    FourWgClcRole.SOFTMAX_STAGE1.value,
                    FourWgClcRole.MMA.value,
                    FourWgClcRole.O_STORE.value,
                )
            ):
                work_meta = self._clc_count_work_metadata(
                    work_idx,
                    mWorkCount,
                    mSchedulerMetadata,
                )
            elif const_expr(
                not self.use_five_wg_schedule
                and role == FourWgClcRole.Q_LOAD.value
            ):
                work_meta = self._clc_q_work_metadata(
                    work_idx,
                    mWorkCount,
                    mSchedulerMetadata,
                    mK2qCounts,
                    mCuSeqlensQ,
                )
            elif const_expr(
                not self.use_five_wg_schedule
                and role
                in (
                    FourWgClcRole.MMA_AND_KV.value,
                    FourWgClcRole.K_LOAD_AND_CLC.value,
                    FourWgClcRole.V_LOAD.value,
                )
            ):
                work_meta = self._clc_kv_work_metadata(
                    work_idx,
                    mWorkCount,
                    mSchedulerMetadata,
                    mCuSeqlensK,
                    mPageTable,
                    mSeqUsedK,
                )
            else:
                work_meta = self._clc_work_metadata(
                    work_idx,
                    mWorkCount,
                    mSchedulerMetadata,
                    mK2qCounts,
                    mCuSeqlensQ,
                    mCuSeqlensK,
                    mPageTable,
                    mSeqUsedK,
                )
            (
                batch_idx,
                kv_block_idx,
                row_start,
                count_raw,
                kv_valid_cols,
                q_batch_offset,
                k_batch_offset,
                causal_q_offset,
                seq_len_q,
                head_kv_idx,
            ) = work_meta
            valid_work = batch_idx >= Int32(0)
            has_work = count_raw > Int32(0)
            paged_kv_idx = (
                mPageTable[cutlass.max(batch_idx, Int32(0)), kv_block_idx]
                if const_expr(self.paged_kv)
                else kv_block_idx
            )
            num_q_groups = (
                count_raw + Int32(self.q_tokens_per_group - 1)
            ) // Int32(self.q_tokens_per_group)

            if const_expr(
                (
                    self.use_five_wg_schedule
                    and role == FiveWgClcRole.Q_LOAD.value
                )
                or (
                    not self.use_five_wg_schedule
                    and role == FourWgClcRole.Q_LOAD.value
                )
            ):
                if valid_work:
                    self._wg_load_q_transport(
                        tma_atom_Q,
                        mQ_2d,
                        mQ_gather4_desc,
                        sQ,
                        mK2qQSplitIndices,
                        sQLoadMIdx,
                        sQIdxMeta,
                        sQ_load,
                        pipeline_q,
                        load_wg_barrier,
                        num_q_groups,
                        count_raw,
                        has_work,
                        head_kv_idx,
                        row_start,
                        q_batch_offset,
                        num_heads_kv,
                        group_base,
                        pipeline_qidx_meta,
                    )
            elif const_expr(
                (
                    self.use_five_wg_schedule
                    and role
                    in (
                        FiveWgClcRole.MMA.value,
                        FiveWgClcRole.K_LOAD_AND_CLC.value,
                        FiveWgClcRole.V_LOAD.value,
                    )
                )
                or (
                    not self.use_five_wg_schedule
                    and role
                    in (
                        FourWgClcRole.MMA_AND_KV.value,
                        FourWgClcRole.MMA.value,
                        FourWgClcRole.K_LOAD_AND_CLC.value,
                        FourWgClcRole.V_LOAD.value,
                    )
                )
            ):
                if valid_work:
                    if (
                        const_expr(
                            (
                                self.use_five_wg_schedule
                                and role
                                in (
                                    FiveWgClcRole.K_LOAD_AND_CLC.value,
                                    FiveWgClcRole.V_LOAD.value,
                                )
                            )
                            or (
                                not self.use_five_wg_schedule
                                and role
                                in (
                                    FourWgClcRole.MMA_AND_KV.value,
                                    FourWgClcRole.K_LOAD_AND_CLC.value,
                                    FourWgClcRole.V_LOAD.value,
                                )
                            )
                        )
                        and warp_idx >= Int32(self.kv_load_warp_base)
                        and warp_idx < Int32(
                            self.kv_load_warp_base + self.num_kv_load_warps
                        )
                    ):
                        self._wg_load_kv(
                            tma_atom_K,
                            tma_atom_V,
                            tma_K,
                            tma_V,
                            paged_kv_idx,
                            sK,
                            sV,
                            tiled_mma_qk,
                            tiled_mma_pv,
                            mbar_k_ptr,
                            mbar_v_ptr,
                            mbar_v_tma_ptr,
                            has_work,
                            head_kv_idx,
                            kv_block_idx,
                            k_batch_offset,
                            kv_valid_cols,
                            pipeline_k,
                            pipeline_v,
                            work_seq,
                        )
                    if (
                        const_expr(
                            (
                                self.use_five_wg_schedule
                                and role == FiveWgClcRole.MMA.value
                            )
                            or (
                                not self.use_five_wg_schedule
                                and role
                                in (
                                    FourWgClcRole.MMA_AND_KV.value,
                                    FourWgClcRole.MMA.value,
                                )
                            )
                        )
                        and warp_idx == Int32(self.mma_warp_id)
                    ):
                        tmem.retrieve_ptr(self.qk_acc_dtype)
                        self._wg_mma_issue(
                            tiled_mma_qk,
                            tiled_mma_pv,
                            tOrP,
                            sK,
                            sV,
                            sQ,
                            pipeline_q,
                            pipeline_s,
                            pipeline_p,
                            pipeline_p_lastsplit,
                            pipeline_o,
                            mbar_k_ptr,
                            mbar_v_ptr,
                            num_q_groups,
                            has_work,
                            group_base,
                            work_seq,
                            pipeline_k,
                            pipeline_v,
                        )
            elif const_expr(
                (
                    self.use_five_wg_schedule
                    and role == FiveWgClcRole.O_STORE.value
                )
                or (
                    not self.use_five_wg_schedule
                    and role == FourWgClcRole.O_STORE.value
                )
            ):
                if valid_work:
                    self._warp_store_o(
                        tma_atom_O,
                        tma_O,
                        mO_ragged_tma_desc,
                        o_store_context,
                        count_raw,
                        group_base,
                        mWorkTileOffsets[work_idx],
                    )
            elif const_expr(
                (
                    self.use_five_wg_schedule
                    and role
                    in (
                        FiveWgClcRole.SOFTMAX.value,
                        FiveWgClcRole.EPILOGUE_STAGE0.value,
                        FiveWgClcRole.EPILOGUE_STAGE1.value,
                    )
                )
                or (
                    not self.use_five_wg_schedule
                    and role
                    in (
                        FourWgClcRole.SOFTMAX_STAGE0.value,
                        FourWgClcRole.SOFTMAX_STAGE1.value,
                    )
                )
            ):
                if valid_work:
                    tmem.retrieve_ptr(self.qk_acc_dtype)
                    if const_expr(
                        self.use_five_wg_schedule
                        and role == FiveWgClcRole.SOFTMAX.value
                    ):
                        for qi_group in cutlass.range(num_q_groups, unroll=1):
                            self._wg_softmax(
                                (group_base + qi_group) & Int32(1),
                                tiled_mma_qk,
                                None,
                                tStS,
                                sScale,
                                sScaleTemperature,
                                None,
                                None,
                                sQIdxMeta,
                                mK2qQSplitIndices,
                                row_start,
                                pipeline_s,
                                pipeline_p,
                                pipeline_p_lastsplit,
                                None,
                                pipeline_sm_stats,
                                sm_stats_barrier,
                                None,
                                None,
                                None,
                                mLSE_temperature_partial,
                                softmax_scale_log2,
                                lse_temperature_scale_log2,
                                lse_temperature_scale,
                                kv_block_idx,
                                kv_valid_cols,
                                Int32(0),
                                num_q_groups,
                                count_raw,
                                has_work,
                                causal_q_offset,
                                Int32(0),
                                head_kv_idx,
                                seq_len_q,
                                Int32(0),
                                Int32(0),
                                Int32(0),
                                None,
                                group_base,
                                ConsumerOperation.SOFTMAX_ONLY.value,
                                qi_group,
                            )
                    else:
                        softmax_stage = (
                            const_expr(
                                role - FiveWgClcRole.EPILOGUE_STAGE0.value
                            )
                            if const_expr(self.use_five_wg_schedule)
                            else role
                        )
                        self._wg_softmax(
                            Int32(softmax_stage),
                            tiled_mma_qk,
                            tiled_mma_pv,
                            tStS,
                            sScale,
                            sScaleTemperature,
                            sSplitIdx,
                            sQIdx,
                            sQIdxMeta,
                            mK2qQSplitIndices,
                            row_start,
                            pipeline_s,
                            pipeline_p,
                            pipeline_p_lastsplit,
                            pipeline_o,
                            pipeline_sm_stats,
                            sm_stats_barrier,
                            epilogue_barrier,
                            mO_partial,
                            mLSE_partial,
                            mLSE_temperature_partial,
                            softmax_scale_log2,
                            lse_temperature_scale_log2,
                            lse_temperature_scale,
                            kv_block_idx,
                            kv_valid_cols,
                            Int32(0),
                            num_q_groups,
                            count_raw,
                            has_work,
                            causal_q_offset,
                            batch_idx,
                            head_kv_idx,
                            seq_len_q,
                            head_q,
                            num_heads_kv,
                            q_batch_offset,
                            mQ_2d,
                            group_base,
                            const_expr(
                                ConsumerOperation.EPILOGUE_ONLY.value
                                if self.use_five_wg_schedule
                                else ConsumerOperation.FUSED.value
                            ),
                            Int32(0),
                            o_store_context,
                            (
                                mWorkTileOffsets[work_idx]
                                if const_expr(self.shuffled_output)
                                else Int32(0)
                            ),
                            pipeline_qidx_meta,
                        )

            group_base += cute.ceil_div(count_raw, self.q_tokens_per_group)
            if has_work:
                work_seq += Int32(1)

            # The K-load warp requests the next launch descriptor only after
            # it has issued the current item's K transfer.  Every role consumes
            # the same two-stage response, then carries its data-pipeline
            # indices and phases directly into the next work item.
            if const_expr(
                (
                    self.use_five_wg_schedule
                    and role == FiveWgClcRole.K_LOAD_AND_CLC.value
                )
                or (
                    not self.use_five_wg_schedule
                    and role
                    in (
                        FourWgClcRole.MMA_AND_KV.value,
                        FourWgClcRole.K_LOAD_AND_CLC.value,
                    )
                )
            ):
                if warp_idx == Int32(self.kv_load_warp_base):
                    self._clc_prefetch(
                        clc_queue, clc_producer, clc_response_ptr
                    )
            clc_queue.consumer_wait(clc_consumer)
            response_ptr = (
                clc_response_ptr + clc_consumer.index * Int32(4)
            )
            response_ptr = cute.make_ptr(
                Int32,
                response_ptr.toint(),
                mem_space=response_ptr.memspace,
                assumed_align=16,
            )
            next_work, _, _, valid = cute.arch.clc_response(response_ptr)
            cute.arch.fence_proxy("async.shared", space="cta")
            clc_queue.consumer_release(clc_consumer)
            work_idx = next_work
            has_tile = valid != Int32(0)
            clc_producer.advance()
            clc_consumer.advance()

        if const_expr(
            (
                self.use_five_wg_schedule
                and role
                in (
                    FiveWgClcRole.K_LOAD_AND_CLC.value,
                    FiveWgClcRole.V_LOAD.value,
                )
            )
            or (
                not self.use_five_wg_schedule
                and role
                in (
                    FourWgClcRole.MMA_AND_KV.value,
                    FourWgClcRole.K_LOAD_AND_CLC.value,
                    FourWgClcRole.V_LOAD.value,
                )
            )
        ):
            if warp_idx == Int32(self.kv_load_warp_base):
                clc_queue.producer_tail(clc_producer)
                for offset in cutlass.range_constexpr(self.kv_stage):
                    seq = work_seq + Int32(offset)
                    slot = seq % Int32(self.kv_stage)
                    phase = ((seq // Int32(self.kv_stage)) & Int32(1)) ^ Int32(1)
                    cute.arch.mbarrier_wait(
                        pipeline_k.sync_object_empty.get_barrier(slot), phase
                    )
            if warp_idx == Int32(self.kv_load_warp_base + 1):
                for offset in cutlass.range_constexpr(self.kv_stage):
                    seq = work_seq + Int32(offset)
                    slot = seq % Int32(self.kv_stage)
                    phase = ((seq // Int32(self.kv_stage)) & Int32(1)) ^ Int32(1)
                    cute.arch.mbarrier_wait(
                        pipeline_v.sync_object_empty.get_barrier(slot), phase
                    )
        elif const_expr(
            (
                self.use_five_wg_schedule
                and role == FiveWgClcRole.Q_LOAD.value
            )
            or (
                not self.use_five_wg_schedule
                and role == FourWgClcRole.Q_LOAD.value
            )
        ):
            if warp_idx == Int32(self.q_load_warp_base):
                for offset in cutlass.range_constexpr(self.q_stage):
                    group = group_base + Int32(offset)
                    slot = group % Int32(self.q_stage)
                    phase = (
                        ((group // Int32(self.q_stage)) & Int32(1)) ^ Int32(1)
                    )
                    cute.arch.mbarrier_wait(
                        pipeline_q.sync_object_empty.get_barrier(slot), phase
                    )
        if const_expr(
            (
                not self.use_five_wg_schedule
                and role == FourWgClcRole.O_STORE.value
            )
            or (
                self.use_five_wg_schedule
                and role == FiveWgClcRole.O_STORE.value
            )
        ):
            cute.arch.cp_async_bulk_wait_group(0, read=True)
            if group_base > Int32(0):
                _, ready = o_store_context
                ready.consumer_release_w_index(
                    (group_base - Int32(1)) & Int32(1)
                )

    @cute.jit
    def _clc_count_work_metadata(
        self, work_idx, work_count, schedule,
        page_table=None, used_k=None, cu_k=None,
        need_valid_cols: cutlass.Constexpr[bool] = False,
    ):
        """Load only validity and Q-group count for roles that need no indices."""
        valid = Int32(-1)
        count = Int32(0)
        kv_block = Int32(0)
        valid_cols = Int32(self.n_block_size)
        if work_idx < work_count[Int32(0)]:
            valid = Int32(0)
            if const_expr(self.compact_batch1_schedule):
                count = (schedule[work_idx, Int32(1)] & Int32(0xFFF)) + Int32(1)
            else:
                count = schedule[work_idx, Int32(3)]
            if const_expr(need_valid_cols and not self.cfg.full_kv_blocks):
                batch = Int32(0)
                if const_expr(self.compact_batch1_schedule):
                    kv_block = (
                        schedule[work_idx, Int32(1)] >> Int32(12)
                    ) & Int32(0x1FFF)
                else:
                    batch = schedule[work_idx, Int32(4)]
                    kv_block = schedule[work_idx, Int32(5)]
                valid_cols = self._valid_cols_in_block(
                    batch, kv_block, page_table, used_k, cu_k)
        return (
            valid,
            kv_block,
            Int32(0),
            count,
            valid_cols,
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
        )

    @cute.jit
    def _run_five_wg_consumer(
        self, role: cutlass.Constexpr[int], context
    ):
        """Run the softmax group or one alternating epilogue group."""
        if const_expr(role == FiveWgRole.SOFTMAX):
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
        else:
            cute.arch.setmaxregister_increase(self.num_regs_correction)

        (
            epilogue_barrier,
            head_q,
            head_kv_idx,
            lse_temperature_scale,
            lse_temperature_scale_log2,
            mK2qQSplitIndices,
            mLSE_partial,
            mLSE_temperature_partial,
            mO_partial,
            mQ_2d,
            mWorkTileOffsets,
            num_heads_kv,
            o_store_context,
            pipeline_o,
            pipeline_p,
            pipeline_p_lastsplit,
            pipeline_s,
            pipeline_sm_stats,
            sQIdx,
            sQIdxMeta,
            sRowMeta,
            sScale,
            sScaleTemperature,
            sSplitIdx,
            sm_stats_barrier,
            softmax_scale_log2,
            tStS,
            tiled_mma_pv,
            tiled_mma_qk,
            tmem,
            work_idx,
            pipeline_qidx_meta,
        ) = context

        if sRowMeta[0] >= Int32(0):
            count_raw = sRowMeta[3]
            has_work = count_raw > Int32(0)
            num_q_groups = (
                count_raw + Int32(self.q_tokens_per_group - 1)
            ) // Int32(self.q_tokens_per_group)
            tmem.retrieve_ptr(self.qk_acc_dtype)
            if const_expr(role == FiveWgRole.SOFTMAX):
                num_q_pairs = (num_q_groups + Int32(1)) // Int32(2)
                for qi_pair in cutlass.range(num_q_pairs, unroll=1):
                    for stage in cutlass.range_constexpr(2):
                        qi_group = qi_pair * Int32(2) + Int32(stage)
                        if qi_group < num_q_groups:
                            self._wg_softmax(
                                stage,
                                tiled_mma_qk,
                                tiled_mma_pv,
                                tStS,
                                sScale,
                                sScaleTemperature,
                                sSplitIdx,
                                sQIdx,
                                sQIdxMeta,
                                mK2qQSplitIndices,
                                sRowMeta[2],
                                pipeline_s,
                                pipeline_p,
                                pipeline_p_lastsplit,
                                pipeline_o,
                                pipeline_sm_stats,
                                sm_stats_barrier,
                                epilogue_barrier,
                                mO_partial,
                                mLSE_partial,
                                mLSE_temperature_partial,
                                softmax_scale_log2,
                                lse_temperature_scale_log2,
                                lse_temperature_scale,
                                sRowMeta[1],
                                sRowMeta[4],
                                Int32(0),
                                num_q_groups,
                                count_raw,
                                has_work,
                                Int32(0),
                                sRowMeta[0],
                                head_kv_idx,
                                sRowMeta[8],
                                head_q,
                                num_heads_kv,
                                sRowMeta[5],
                                mQ_2d,
                                Int32(0),
                                ConsumerOperation.SOFTMAX_ONLY.value,
                                qi_group,
                            )
            else:
                stage = const_expr(
                    0 if role == FiveWgRole.EPILOGUE_EVEN else 1
                )
                self._wg_softmax(
                    stage,
                    tiled_mma_qk,
                    tiled_mma_pv,
                    tStS,
                    sScale,
                    sScaleTemperature,
                    sSplitIdx,
                    sQIdx,
                    sQIdxMeta,
                    mK2qQSplitIndices,
                    sRowMeta[2],
                    pipeline_s,
                    pipeline_p,
                    pipeline_p_lastsplit,
                    pipeline_o,
                    pipeline_sm_stats,
                    sm_stats_barrier,
                    epilogue_barrier,
                    mO_partial,
                    mLSE_partial,
                    mLSE_temperature_partial,
                    softmax_scale_log2,
                    lse_temperature_scale_log2,
                    lse_temperature_scale,
                    sRowMeta[1],
                    sRowMeta[4],
                    Int32(0),
                    num_q_groups,
                    count_raw,
                    has_work,
                    Int32(0),
                    sRowMeta[0],
                    head_kv_idx,
                    sRowMeta[8],
                    head_q,
                    num_heads_kv,
                    sRowMeta[5],
                    mQ_2d,
                    Int32(0),
                    ConsumerOperation.EPILOGUE_ONLY.value,
                    Int32(0),
                    o_store_context,
                    (
                        mWorkTileOffsets[work_idx]
                        if const_expr(self.shuffled_output)
                        else Int32(0)
                    ),
                    pipeline_qidx_meta,
                )

    @cute.jit
    def _run_five_wg_q(self, context):
        """Load Q after the complete warp group donates registers."""
        cute.arch.setmaxregister_decrease(self.num_regs_store)
        (
            head_kv_idx,
            load_wg_barrier,
            mK2qQSplitIndices,
            mQ_2d,
            mQ_gather4_desc,
            sQ,
            num_heads_kv,
            pipeline_q,
            sQIdxMeta,
            sQLoadMIdx,
            sQ_load,
            sRowMeta,
            tma_atom_Q,
            pipeline_qidx_meta,
        ) = context
        if sRowMeta[0] >= Int32(0):
            count_raw = sRowMeta[3]
            num_q_groups = (
                count_raw + Int32(self.q_tokens_per_group - 1)
            ) // Int32(self.q_tokens_per_group)
            self._wg_load_q_transport(
                tma_atom_Q,
                mQ_2d,
                mQ_gather4_desc,
                sQ,
                mK2qQSplitIndices,
                sQLoadMIdx,
                sQIdxMeta,
                sQ_load,
                pipeline_q,
                load_wg_barrier,
                num_q_groups,
                count_raw,
                count_raw > Int32(0),
                head_kv_idx,
                sRowMeta[2],
                sRowMeta[5],
                num_heads_kv,
                Int32(0),
                pipeline_qidx_meta,
            )

    @cute.jit
    def _run_five_wg_main(self, context):
        """Run MMA and K/V loads after the complete warp group donates registers."""
        cute.arch.setmaxregister_decrease(self.num_regs_other)
        (
            head_kv_idx,
            mbar_k_ptr,
            mbar_v_ptr,
            mbar_v_tma_ptr,
            pipeline_k,
            pipeline_o,
            pipeline_p,
            pipeline_p_lastsplit,
            pipeline_q,
            pipeline_s,
            pipeline_sm_stats,
            pipeline_v,
            sK,
            sPagedKvIdx,
            sQ,
            sRowMeta,
            sV,
            tOrP,
            tStS,
            thr_mma_pv,
            thr_mma_qk,
            tiled_mma_pv,
            tiled_mma_qk,
            tma_K,
            tma_V,
            tma_atom_K,
            tma_atom_V,
            tmem,
            warp_idx,
        ) = context
        if sRowMeta[0] >= Int32(0):
            count_raw = sRowMeta[3]
            has_work = count_raw > Int32(0)
            if (
                warp_idx >= Int32(self.kv_load_warp_base)
                and warp_idx
                < Int32(self.kv_load_warp_base + self.num_kv_load_warps)
            ):
                self._wg_load_kv(
                    tma_atom_K,
                    tma_atom_V,
                    tma_K,
                    tma_V,
                    sPagedKvIdx[0]
                    if const_expr(self.paged_kv)
                    else sRowMeta[1],
                    sK,
                    sV,
                    tiled_mma_qk,
                    tiled_mma_pv,
                    mbar_k_ptr,
                    mbar_v_ptr,
                    mbar_v_tma_ptr,
                    has_work,
                    head_kv_idx,
                    sRowMeta[1],
                    sRowMeta[6],
                    sRowMeta[4],
                    pipeline_k,
                    pipeline_v,
                    Int32(0),
                )
            elif warp_idx == Int32(self.mma_warp_id):
                num_q_groups = (
                    count_raw + Int32(self.q_tokens_per_group - 1)
                ) // Int32(self.q_tokens_per_group)
                tmem.retrieve_ptr(self.qk_acc_dtype)
                self._wg_mma_issue(
                    tiled_mma_qk,
                    tiled_mma_pv,
                    tOrP,
                    sK,
                    sV,
                    sQ,
                    pipeline_q,
                    pipeline_s,
                    pipeline_p,
                    pipeline_p_lastsplit,
                    pipeline_o,
                    mbar_k_ptr,
                    mbar_v_ptr,
                    num_q_groups,
                    has_work,
                    Int32(0),
                    Int32(0),
                    pipeline_k,
                    pipeline_v,
                )

    @cute.jit
    def _run_clc_role(self, role: cutlass.Constexpr[int], context):
        """Run one persistent role with pipeline state carried across work items."""
        if const_expr(
            role
            in (
                FourWgClcRole.SOFTMAX_STAGE0.value,
                FourWgClcRole.SOFTMAX_STAGE1.value,
            )
        ):
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
        elif const_expr(role == FourWgClcRole.Q_LOAD.value):
            cute.arch.setmaxregister_decrease(self.num_regs_store)
        else:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

        (
            clc_queue,
            clc_producer,
            clc_consumer,
            clc_response_ptr,
            epilogue_barrier,
            head_q,
            load_wg_barrier,
            lse_temperature_scale,
            lse_temperature_scale_log2,
            mCuSeqlensK,
            mSeqUsedK,
            mCuSeqlensQ,
            mK2qCounts,
            mK2qQSplitIndices,
            mLSE_partial,
            mLSE_temperature_partial,
            mO_partial,
            _mO_ragged_tma_desc,
            mPageTable,
            mQ_2d,
            mQ_gather4_desc,
            mSchedulerMetadata,
            mWorkCount,
            mWorkTileOffsets,
            mbar_k_ptr,
            mbar_v_ptr,
            mbar_v_tma_ptr,
            num_heads_kv,
            pipeline_k,
            pipeline_o,
            pipeline_p,
            pipeline_p_lastsplit,
            pipeline_q,
            pipeline_s,
            pipeline_sm_stats,
            pipeline_v,
            o_store_context,
            sK,
            sQ,
            sQIdx,
            sQIdxMeta,
            sQLoadMIdx,
            sQ_load,
            sScale,
            sScaleTemperature,
            sSplitIdx,
            sV,
            sm_stats_barrier,
            softmax_scale_log2,
            tOrP,
            tStS,
            thr_mma_pv,
            thr_mma_qk,
            tiled_mma_pv,
            tiled_mma_qk,
            tma_K,
            tma_V,
            tma_atom_K,
            tma_atom_Q,
            tma_atom_V,
            tma_atom_O,
            tma_O,
            tmem,
            warp_idx,
            work_idx,
            _pipeline_qidx_meta,
        ) = context

        group_base = Int32(0)
        work_seq = Int32(0)
        has_tile = cutlass.Boolean(True)
        while has_tile:
            if const_expr(role == FourWgClcRole.Q_LOAD.value):
                work_meta = self._clc_q_work_metadata(
                    work_idx,
                    mWorkCount,
                    mSchedulerMetadata,
                    mK2qCounts,
                    mCuSeqlensQ,
                )
            elif const_expr(
                role
                in (
                    FourWgClcRole.MMA.value,
                    FourWgClcRole.K_LOAD_AND_CLC.value,
                )
            ):
                work_meta = self._clc_kv_work_metadata(
                    work_idx,
                    mWorkCount,
                    mSchedulerMetadata,
                    mCuSeqlensK,
                    mPageTable,
                    mSeqUsedK,
                )
            else:
                work_meta = self._clc_work_metadata(
                    work_idx,
                    mWorkCount,
                    mSchedulerMetadata,
                    mK2qCounts,
                    mCuSeqlensQ,
                    mCuSeqlensK,
                    mPageTable,
                    mSeqUsedK,
                )
            (
                batch_idx,
                kv_block_idx,
                row_start,
                count_raw,
                kv_valid_cols,
                q_batch_offset,
                k_batch_offset,
                causal_q_offset,
                seq_len_q,
                head_kv_idx,
            ) = work_meta
            valid_work = batch_idx >= Int32(0)
            has_work = count_raw > Int32(0)
            paged_kv_idx = (
                mPageTable[cutlass.max(batch_idx, Int32(0)), kv_block_idx]
                if const_expr(self.paged_kv)
                else kv_block_idx
            )
            num_q_groups = (
                count_raw + Int32(self.q_tokens_per_group - 1)
            ) // Int32(self.q_tokens_per_group)

            if const_expr(role == FourWgClcRole.Q_LOAD.value):
                if valid_work:
                    self._wg_load_q_transport(
                        tma_atom_Q,
                        mQ_2d,
                        mQ_gather4_desc,
                        sQ,
                        mK2qQSplitIndices,
                        sQLoadMIdx,
                        sQIdxMeta,
                        sQ_load,
                        pipeline_q,
                        load_wg_barrier,
                        num_q_groups,
                        count_raw,
                        has_work,
                        head_kv_idx,
                        row_start,
                        q_batch_offset,
                        num_heads_kv,
                        group_base,
                    )
            elif const_expr(role == FourWgClcRole.K_LOAD_AND_CLC.value):
                if valid_work:
                    if (
                        warp_idx >= Int32(self.kv_load_warp_base)
                        and warp_idx
                        < Int32(self.kv_load_warp_base + self.num_kv_load_warps)
                    ):
                        self._wg_load_kv(
                            tma_atom_K,
                            tma_atom_V,
                            tma_K,
                            tma_V,
                            paged_kv_idx,
                            sK,
                            sV,
                            tiled_mma_qk,
                            tiled_mma_pv,
                            mbar_k_ptr,
                            mbar_v_ptr,
                            mbar_v_tma_ptr,
                            has_work,
                            head_kv_idx,
                            kv_block_idx,
                            k_batch_offset,
                            kv_valid_cols,
                            pipeline_k,
                            pipeline_v,
                            work_seq,
                        )
            elif const_expr(role == FourWgClcRole.MMA.value):
                if valid_work:
                    tmem.retrieve_ptr(self.qk_acc_dtype)
                    self._wg_mma_issue(
                        tiled_mma_qk,
                        tiled_mma_pv,
                        tOrP,
                        sK,
                        sV,
                        sQ,
                        pipeline_q,
                        pipeline_s,
                        pipeline_p,
                        pipeline_p_lastsplit,
                        pipeline_o,
                        mbar_k_ptr,
                        mbar_v_ptr,
                        num_q_groups,
                        has_work,
                        group_base,
                        work_seq,
                        pipeline_k,
                        pipeline_v,
                    )
            else:
                if valid_work:
                    tmem.retrieve_ptr(self.qk_acc_dtype)
                    self._wg_softmax(
                        role,
                        tiled_mma_qk,
                        tiled_mma_pv,
                        tStS,
                        sScale,
                        sScaleTemperature,
                        sSplitIdx,
                        sQIdx,
                        sQIdxMeta,
                        mK2qQSplitIndices,
                        row_start,
                        pipeline_s,
                        pipeline_p,
                        pipeline_p_lastsplit,
                        pipeline_o,
                        pipeline_sm_stats,
                        sm_stats_barrier,
                        epilogue_barrier,
                        mO_partial,
                        mLSE_partial,
                        mLSE_temperature_partial,
                        softmax_scale_log2,
                        lse_temperature_scale_log2,
                        lse_temperature_scale,
                        kv_block_idx,
                        kv_valid_cols,
                        Int32(0),
                        num_q_groups,
                        count_raw,
                        has_work,
                        causal_q_offset,
                        batch_idx,
                        head_kv_idx,
                        seq_len_q,
                        head_q,
                        num_heads_kv,
                        q_batch_offset,
                        mQ_2d,
                        group_base,
                        ConsumerOperation.FUSED.value,
                        Int32(0),
                        o_store_context,
                        (
                            mWorkTileOffsets[work_idx]
                            if const_expr(self.shuffled_output)
                            else Int32(0)
                        ),
                    )

            group_base += cute.ceil_div(count_raw, self.q_tokens_per_group)
            if has_work:
                work_seq += Int32(1)

            # The K-load warp requests the next launch descriptor only after
            # it has issued the current item's K transfer.  Every role consumes
            # the same two-stage response, then carries its data-pipeline
            # indices and phases directly into the next work item.
            if const_expr(role == FourWgClcRole.K_LOAD_AND_CLC.value):
                if warp_idx == Int32(self.kv_load_warp_base):
                    self._clc_prefetch(
                        clc_queue, clc_producer, clc_response_ptr
                    )
            clc_queue.consumer_wait(clc_consumer)
            response_ptr = (
                clc_response_ptr + clc_consumer.index * Int32(4)
            )
            response_ptr = cute.make_ptr(
                Int32,
                response_ptr.toint(),
                mem_space=response_ptr.memspace,
                assumed_align=16,
            )
            next_work, _, _, valid = cute.arch.clc_response(response_ptr)
            cute.arch.fence_proxy("async.shared", space="cta")
            clc_queue.consumer_release(clc_consumer)
            work_idx = next_work
            has_tile = valid != Int32(0)
            clc_producer.advance()
            clc_consumer.advance()

        if const_expr(role == FourWgClcRole.K_LOAD_AND_CLC.value):
            if warp_idx == Int32(self.kv_load_warp_base):
                clc_queue.producer_tail(clc_producer)
                for offset in cutlass.range_constexpr(self.kv_stage):
                    seq = work_seq + Int32(offset)
                    slot = seq % Int32(self.kv_stage)
                    phase = ((seq // Int32(self.kv_stage)) & Int32(1)) ^ Int32(1)
                    cute.arch.mbarrier_wait(
                        pipeline_k.sync_object_empty.get_barrier(slot), phase
                    )
            if warp_idx == Int32(self.kv_load_warp_base + 1):
                for offset in cutlass.range_constexpr(self.kv_stage):
                    seq = work_seq + Int32(offset)
                    slot = seq % Int32(self.kv_stage)
                    phase = ((seq // Int32(self.kv_stage)) & Int32(1)) ^ Int32(1)
                    cute.arch.mbarrier_wait(
                        pipeline_v.sync_object_empty.get_barrier(slot), phase
                    )
        elif const_expr(role == FourWgClcRole.Q_LOAD.value):
            if warp_idx == Int32(self.q_load_warp_base):
                for offset in cutlass.range_constexpr(self.q_stage):
                    group = group_base + Int32(offset)
                    slot = group % Int32(self.q_stage)
                    phase = (
                        ((group // Int32(self.q_stage)) & Int32(1)) ^ Int32(1)
                    )
                    cute.arch.mbarrier_wait(
                        pipeline_q.sync_object_empty.get_barrier(slot), phase
                    )

    @cute.jit
    def _clc_work_metadata(
        self, work_idx, work_count, schedule, row_ptr, cu_q, cu_k,
        page_table, used_k,
    ):
        batch = Int32(-1)
        kv_block = Int32(0)
        row_start = Int32(0)
        count = Int32(0)
        q_offset = Int32(0)
        k_offset = Int32(0)
        seq_q = Int32(0)
        head = Int32(0)
        valid_cols = Int32(self.n_block_size)
        if work_idx < work_count[Int32(0)]:
            if const_expr(self.compact_batch1_schedule):
                row_start = schedule[work_idx, Int32(0)]
                packed = schedule[work_idx, Int32(1)]
                count = (packed & Int32(0xFFF)) + Int32(1)
                kv_block = (packed >> Int32(12)) & Int32(0x1FFF)
                head = (packed >> Int32(25)) & Int32(0x3)
                batch = Int32(0)
                seq_q = cu_q[Int32(1)] - cu_q[Int32(0)]
            else:
                head = schedule[work_idx, Int32(0)]
                row = schedule[work_idx, Int32(1)]
                begin = schedule[work_idx, Int32(2)]
                count = schedule[work_idx, Int32(3)]
                batch = schedule[work_idx, Int32(4)]
                kv_block = schedule[work_idx, Int32(5)]
                row_start = row_ptr[head, row] + begin
                q_offset = cu_q[batch]
                k_offset = cu_k[batch]
                seq_q = cu_q[batch + Int32(1)] - q_offset
            if const_expr(not self.cfg.full_kv_blocks):
                valid_cols = self._valid_cols_in_block(
                    batch, kv_block, page_table, used_k, cu_k)
        return (
            batch,
            kv_block,
            row_start,
            count,
            valid_cols,
            q_offset,
            k_offset,
            Int32(0),
            seq_q,
            head,
        )

    @cute.jit
    def _clc_q_work_metadata(
        self, work_idx, work_count, schedule, row_ptr, cu_q
    ):
        batch = Int32(-1)
        row_start = Int32(0)
        count = Int32(0)
        q_offset = Int32(0)
        head = Int32(0)
        if work_idx < work_count[Int32(0)]:
            if const_expr(self.compact_batch1_schedule):
                row_start = schedule[work_idx, Int32(0)]
                packed = schedule[work_idx, Int32(1)]
                count = (packed & Int32(0xFFF)) + Int32(1)
                head = (packed >> Int32(25)) & Int32(0x3)
                batch = Int32(0)
            else:
                head = schedule[work_idx, Int32(0)]
                row = schedule[work_idx, Int32(1)]
                begin = schedule[work_idx, Int32(2)]
                count = schedule[work_idx, Int32(3)]
                batch = schedule[work_idx, Int32(4)]
                row_start = row_ptr[head, row] + begin
                q_offset = cu_q[batch]
        return (
            batch,
            Int32(0),
            row_start,
            count,
            Int32(self.n_block_size),
            q_offset,
            Int32(0),
            Int32(0),
            Int32(0),
            head,
        )

    @cute.jit
    def _clc_kv_work_metadata(
        self, work_idx, work_count, schedule, cu_k, page_table, used_k
    ):
        batch = Int32(-1)
        kv_block = Int32(0)
        count = Int32(0)
        k_offset = Int32(0)
        head = Int32(0)
        valid_cols = Int32(self.n_block_size)
        if work_idx < work_count[Int32(0)]:
            if const_expr(self.compact_batch1_schedule):
                packed = schedule[work_idx, Int32(1)]
                count = (packed & Int32(0xFFF)) + Int32(1)
                kv_block = (packed >> Int32(12)) & Int32(0x1FFF)
                head = (packed >> Int32(25)) & Int32(0x3)
                batch = Int32(0)
            else:
                head = schedule[work_idx, Int32(0)]
                count = schedule[work_idx, Int32(3)]
                batch = schedule[work_idx, Int32(4)]
                kv_block = schedule[work_idx, Int32(5)]
                k_offset = cu_k[batch]
            if const_expr(not self.cfg.full_kv_blocks):
                valid_cols = self._valid_cols_in_block(
                    batch, kv_block, page_table, used_k, cu_k)
        return (
            batch,
            kv_block,
            Int32(0),
            count,
            valid_cols,
            Int32(0),
            k_offset,
            Int32(0),
            Int32(0),
            head,
        )

    @cute.jit
    def _clc_prefetch(self, queue, state, response):
        queue.producer_acquire(state)
        with cute.arch.elect_one():
            cute.arch.issue_clc_query(
                queue.producer_get_barrier(state),
                response + state.index * Int32(4),
            )

    @cute.jit
    def _warp_store_o(
        self,
        atom,
        gO,
        ragged_desc,
        context,
        count_raw: Int32,
        group_base: Int32,
        work_tile_base: Int32,
    ):
        """Drain dense 128x128 SMEM tiles through one TMA-store warp."""
        sO, ready = context
        groups = cute.ceil_div(count_raw, self.q_tokens_per_group)
        tile = (self.m_block_size, self.head_dim // 2)
        for qi_group in cutlass.range(groups, unroll=1):
            group = group_base + qi_group
            slot = group & Int32(1)
            phase = (group // Int32(2)) & Int32(1)
            ready.consumer_wait_w_index_phase(slot, phase)
            tile_idx = cute.arch.make_warp_uniform(
                work_tile_base + qi_group
            )
            src_tile = sO[None, None, slot]
            if const_expr(self.cfg.ragged_tma_store):
                q_begin = qi_group * Int32(self.q_tokens_per_group)
                valid_q = cutlass.min(
                    Int32(self.q_tokens_per_group), count_raw - q_begin
                )
                valid_rows = valid_q * Int32(self.qheadperkv)
                output_row = tile_idx * Int32(self.m_block_size)
                for d_half in cutlass.range_constexpr(2):
                    coords = ragged_o_tma_coords(
                        Int32(d_half * (self.head_dim // 2)),
                        output_row,
                        valid_rows,
                    )
                    with cute.arch.elect_one():
                        tma_tile_store_5d(
                            ragged_desc.iterator,
                            coords[0],
                            coords[1],
                            coords[2],
                            coords[3],
                            coords[4],
                            src_tile.iterator,
                            Int32(
                                d_half
                                * self.m_block_size
                                * (self.head_dim // 2)
                                * cutlass.BFloat16.width
                                // 8
                            ),
                        )
            else:
                for d_half in cutlass.range_constexpr(2):
                    src = cute.local_tile(src_tile, tile, (0, d_half))
                    dst = cute.local_tile(gO, tile, (tile_idx, d_half))
                    partition_s, partition_g = cpasync.tma_partition(
                        atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(src, 0, 2),
                        cute.group_modes(dst, 0, 2),
                    )
                    cute.copy(atom, partition_s, partition_g)
            # Close both physical half-tile commands as one logical store
            # group, then retain only the current group in flight. The prior
            # ring slot is safe to reuse after the read wait. UTMACMDFLUSH is
            # the SASS spelling of bulk-group commit; it does not drain GMEM.
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(1, read=True)
            if group > Int32(0):
                ready.consumer_release_w_index(
                    (group - Int32(1)) & Int32(1)
                )

    @cute.jit
    def _convert_fp8x16_to_bf16x16(
        self,
        src: cute.Tensor,
        dst: cute.Tensor,
    ):
        src_i32 = cute.recast_tensor(src, cutlass.Int32)
        dst_i32 = cute.recast_tensor(dst, cutlass.Int32)
        for word_idx in cutlass.range_constexpr(4):
            (
                dst_i32[word_idx * 2],
                dst_i32[word_idx * 2 + 1],
            ) = utils.cvt_fp8x4_e4m3_bf16x4(src_i32[word_idx])

    @cute.jit
    def _convert_fp8_kv_to_bf16_smem(
        self,
        sFp8: cute.Tensor,
        sBf16: cute.Tensor,
        lane: Int32,
        warp_idx_in_wg: Int32,
        num_dequant_warps: cutlass.Constexpr[int],
        is_v: cutlass.Constexpr[bool],
        kv_valid_cols: Int32,
        neutralize_v_tail: cutlass.Constexpr[bool],
    ):
        elems_per_load: cutlass.Constexpr[int] = 16
        elems_per_store: cutlass.Constexpr[int] = 8
        chunks_per_row: cutlass.Constexpr[int] = self.head_dim // elems_per_load
        r_fp8 = cute.make_rmem_tensor((elems_per_load,), cutlass.Float8E4M3FN)
        r_bf16 = cute.make_rmem_tensor((elems_per_load,), cutlass.BFloat16)
        total_tasks: cutlass.Constexpr[int] = self.n_block_size * chunks_per_row
        task_stride: cutlass.Constexpr[int] = num_dequant_warps * cute.arch.WARP_SIZE
        task = warp_idx_in_wg * Int32(cute.arch.WARP_SIZE) + lane
        for task_idx in cutlass.range(task, total_tasks, task_stride, unroll=1):
            row = task_idx // Int32(chunks_per_row)
            chunk = task_idx - row * Int32(chunks_per_row)
            col = chunk * Int32(elems_per_load)
            smem_offset = row * Int32(self.head_dim) + col
            s_fp8_ptr = cute.make_ptr(
                cutlass.Float8E4M3FN,
                sFp8.iterator.toint() + Int64(smem_offset),
                mem_space=sFp8.iterator.memspace,
                assumed_align=elems_per_load,
            )
            s_fp8_vec = cute.make_tensor(
                s_fp8_ptr,
                cute.make_layout(elems_per_load),
            )
            if const_expr(neutralize_v_tail):
                if row >= kv_valid_cols:
                    r_bf16.fill(cutlass.BFloat16(0.0))
                else:
                    cute.autovec_copy(s_fp8_vec, r_fp8)
                    self._convert_fp8x16_to_bf16x16(r_fp8, r_bf16)
            else:
                cute.autovec_copy(s_fp8_vec, r_fp8)
                self._convert_fp8x16_to_bf16x16(r_fp8, r_bf16)
            if const_expr(is_v):
                sBf16_view = sBf16[(None, row % Int32(16)), 0, row // Int32(16), 0]
                sBf16_vec = cute.local_tile(sBf16_view, (elems_per_load,), (chunk,))
            else:
                sBf16_vec = sBf16[
                    (row, None),
                    0,
                    (chunk % Int32(4), chunk // Int32(4)),
                    0,
                ]
            r_tiles = cute.logical_divide(r_bf16, cute.make_layout(elems_per_store))
            s_tiles = cute.logical_divide(sBf16_vec, cute.make_layout(elems_per_store))
            for v in cutlass.range_constexpr(elems_per_load // elems_per_store):
                cute.autovec_copy(r_tiles[None, v], s_tiles[None, v])
        cute.arch.fence_view_async_shared()

    @cute.jit
    def _wg_load_kv_maybe_cast(
        self,
        tma_atom_K,
        tma_atom_V,
        tma_K: cute.Tensor,
        tma_V: cute.Tensor,
        sPagedKvIdx: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sKFp8: Optional[cute.Tensor],
        sVFp8: Optional[cute.Tensor],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mbar_k_ptr,
        mbar_v_ptr,
        mbar_k_tma_ptr: Optional[cutlass.Pointer],
        mbar_v_tma_ptr: Optional[cutlass.Pointer],
        kv_load_barrier,
        has_work: Int32,
        head_kv_idx: Int32,
        kv_block_idx: Int32,
        k_batch_offset: Int32,
        kv_valid_cols: Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_idx_in_wg = warp_idx - Int32(self.kv_load_warp_base)

        if has_work:
            if warp_idx_in_wg == Int32(0):
                if const_expr(self.k_fp8_to_bf16):
                    if const_expr(self.paged_kv):
                        mK_cur = tma_K[None, None, head_kv_idx, sPagedKvIdx[0]]
                        gK = cute.local_tile(
                            mK_cur,
                            (self.n_block_size, self.head_dim),
                            (None, 0),
                        )
                        src_idx = Int32(0)
                    else:
                        mK_cur = cute.domain_offset(
                            (k_batch_offset, 0),
                            tma_K[None, None, head_kv_idx],
                        )
                        gK = cute.local_tile(
                            mK_cur,
                            (self.n_block_size, self.head_dim),
                            (None, 0),
                        )
                        src_idx = kv_block_idx
                    load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_K,
                        0,
                        cute.make_layout(1),
                        gK,
                        sKFp8,
                    )
                    load_K_fn(
                        src_idx=src_idx,
                        dst_idx=0,
                        tma_bar_ptr=mbar_k_tma_ptr,
                    )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(mbar_k_tma_ptr)
                else:
                    thr_mma_qk = tiled_mma_qk.get_slice(0)
                    if const_expr(self.paged_kv):
                        mK_cur = tma_K[None, None, head_kv_idx, None]
                        gK = cute.local_tile(
                            mK_cur,
                            cute.select(self.mma_tiler_qk, mode=[1, 2]),
                            (None, 0, None),
                        )
                    else:
                        mK_cur = cute.domain_offset(
                            (k_batch_offset, 0),
                            tma_K[None, None, head_kv_idx],
                        )
                        gK = cute.local_tile(
                            mK_cur,
                            cute.select(self.mma_tiler_qk, mode=[1, 2]),
                            (None, 0),
                        )
                    tSgK = thr_mma_qk.partition_B(gK)
                    tKsK, tKgK = cpasync.tma_partition(
                        tma_atom_K,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sK, 0, 3),
                        cute.group_modes(tSgK, 0, 3),
                    )
                    gmem_k_idx = (
                        sPagedKvIdx[0] if const_expr(self.paged_kv) else kv_block_idx
                    )
                    cute.copy(
                        tma_atom_K,
                        tKgK[(None, 0, gmem_k_idx)]
                        if const_expr(self.paged_kv)
                        else tKgK[(None, gmem_k_idx)],
                        tKsK[(None, 0)],
                        tma_bar_ptr=mbar_k_ptr,
                    )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(mbar_k_ptr)

            if warp_idx_in_wg == Int32(0):
                if const_expr(self.v_fp8_to_bf16):
                    if const_expr(self.paged_kv):
                        mV_cur = tma_V[None, None, head_kv_idx, sPagedKvIdx[0]]
                        gV = cute.local_tile(
                            mV_cur,
                            (self.head_dim, self.n_block_size),
                            (0, None),
                        )
                        src_idx = Int32(0)
                    else:
                        mV_cur = cute.domain_offset(
                            (0, k_batch_offset),
                            tma_V[None, None, head_kv_idx],
                        )
                        gV = cute.local_tile(
                            mV_cur,
                            (self.head_dim, self.n_block_size),
                            (0, None),
                        )
                        src_idx = kv_block_idx
                    load_V_fn, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_V,
                        0,
                        cute.make_layout(1),
                        gV,
                        sVFp8,
                    )
                    load_V_fn(
                        src_idx=src_idx,
                        dst_idx=0,
                        tma_bar_ptr=mbar_v_tma_ptr,
                    )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(mbar_v_tma_ptr)
                else:
                    thr_mma_pv = tiled_mma_pv.get_slice(0)
                    if const_expr(self.paged_kv):
                        mV_cur = tma_V[None, None, head_kv_idx, None]
                        gV = cute.local_tile(
                            mV_cur,
                            cute.select(self.mma_tiler_pv, mode=[1, 2]),
                            (0, None, None),
                        )
                    else:
                        mV_cur = cute.domain_offset(
                            (0, k_batch_offset),
                            tma_V[None, None, head_kv_idx],
                        )
                        gV = cute.local_tile(
                            mV_cur,
                            cute.select(self.mma_tiler_pv, mode=[1, 2]),
                            (0, None),
                        )
                    tOgV = thr_mma_pv.partition_B(gV)
                    tVsV, tVgV = cpasync.tma_partition(
                        tma_atom_V,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sV, 0, 3),
                        cute.group_modes(tOgV, 0, 3),
                    )
                    gmem_v_idx = (
                        sPagedKvIdx[0] if const_expr(self.paged_kv) else kv_block_idx
                    )
                    if const_expr(self.safe_partial_v_tail):
                        if kv_valid_cols < Int32(self.n_block_size):
                            cute.copy(
                                tma_atom_V,
                                tVgV[(None, 0, gmem_v_idx)]
                                if const_expr(self.paged_kv)
                                else tVgV[(None, gmem_v_idx)],
                                tVsV[(None, 0)],
                                tma_bar_ptr=mbar_v_tma_ptr,
                            )
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive(mbar_v_tma_ptr)
                            cute.arch.mbarrier_wait(mbar_v_tma_ptr, 0)
                            self._warp_zero_invalid_v_tail(sV, kv_valid_cols)
                            cute.arch.sync_warp()
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive(mbar_v_ptr)
                        else:
                            cute.copy(
                                tma_atom_V,
                                tVgV[(None, 0, gmem_v_idx)]
                                if const_expr(self.paged_kv)
                                else tVgV[(None, gmem_v_idx)],
                                tVsV[(None, 0)],
                                tma_bar_ptr=mbar_v_ptr,
                            )
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive(mbar_v_ptr)
                    else:
                        cute.copy(
                            tma_atom_V,
                            tVgV[(None, 0, gmem_v_idx)]
                            if const_expr(self.paged_kv)
                            else tVgV[(None, gmem_v_idx)],
                            tVsV[(None, 0)],
                            tma_bar_ptr=mbar_v_ptr,
                        )
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(mbar_v_ptr)

            kv_load_barrier.arrive_and_wait()

    @cute.jit
    def _wg_convert_fp8_kv_to_bf16_smem(
        self,
        sFp8: cute.Tensor,
        sBf16: cute.Tensor,
        mbar_tma_ptr,
        mbar_ready_ptr,
        dequant_barrier,
        has_work: Int32,
        dequant_warp_base: cutlass.Constexpr[int],
        num_dequant_warps: cutlass.Constexpr[int],
        is_v: cutlass.Constexpr[bool],
        kv_valid_cols: Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_idx_in_wg = warp_idx - Int32(dequant_warp_base)
        lane = cute.arch.lane_idx()
        if has_work:
            cute.arch.mbarrier_wait(mbar_tma_ptr, 0)
            if const_expr(self.safe_partial_v_tail and is_v):
                if kv_valid_cols < Int32(self.n_block_size):
                    self._convert_fp8_kv_to_bf16_smem(
                        sFp8,
                        sBf16,
                        lane,
                        warp_idx_in_wg,
                        num_dequant_warps,
                        is_v,
                        kv_valid_cols,
                        True,
                    )
                else:
                    self._convert_fp8_kv_to_bf16_smem(
                        sFp8,
                        sBf16,
                        lane,
                        warp_idx_in_wg,
                        num_dequant_warps,
                        is_v,
                        kv_valid_cols,
                        False,
                    )
            else:
                self._convert_fp8_kv_to_bf16_smem(
                    sFp8,
                    sBf16,
                    lane,
                    warp_idx_in_wg,
                    num_dequant_warps,
                    is_v,
                    kv_valid_cols,
                    False,
                )
            dequant_barrier.arrive_and_wait()
            if warp_idx_in_wg == Int32(0):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(mbar_ready_ptr)

    @cute.jit
    def _warp_zero_invalid_v_tail(
        self,
        sV: cute.Tensor,
        kv_valid_cols: Int32,
        kv_slot: Int32 = 0,
    ):
        """Cooperatively zero invalid token rows with the V-load warp."""
        lane = cute.arch.lane_idx()
        invalid_elements = (
            Int32(self.n_block_size) - kv_valid_cols
        ) * Int32(self.head_dim)
        for element in cutlass.range(
            lane,
            invalid_elements,
            Int32(cute.arch.WARP_SIZE),
            unroll=1,
        ):
            row = kv_valid_cols + element // Int32(self.head_dim)
            col = element % Int32(self.head_dim)
            sV_row = sV[
                (None, row % Int32(16)),
                0,
                row // Int32(16),
                kv_slot,
            ]
            sV_row[col] = self.v_dtype(0.0)
        cute.arch.fence_view_async_shared()

    @cute.jit
    def _wg_load_kv(
        self,
        tma_atom_K,
        tma_atom_V,
        tma_K: cute.Tensor,
        tma_V: cute.Tensor,
        paged_kv_idx: Int32,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mbar_k_ptr,
        mbar_v_ptr,
        mbar_v_tma_ptr: Optional[cutlass.Pointer],
        has_work: Int32,
        head_kv_idx: Int32,
        kv_block_idx: Int32,
        k_batch_offset: Int32,
        kv_valid_cols: Int32,
        pipeline_k=None,
        pipeline_v=None,
        work_seq: Int32 = 0,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_idx_in_wg = warp_idx - Int32(self.kv_load_warp_base)
        kv_slot = work_seq % Int32(self.kv_stage)
        kv_phase = (work_seq // Int32(self.kv_stage)) & Int32(1)

        if has_work:
            if const_expr(self.use_clc):
                if warp_idx_in_wg == Int32(0):
                    pipeline_k.producer_acquire_w_index_phase(
                        kv_slot, kv_phase ^ Int32(1)
                    )
                else:
                    pipeline_v.producer_acquire_w_index_phase(
                        kv_slot, kv_phase ^ Int32(1)
                    )
            if warp_idx_in_wg == Int32(0):
                thr_mma_qk = tiled_mma_qk.get_slice(0)
                if const_expr(self.paged_kv):
                    mK_cur = tma_K[None, None, head_kv_idx, None]
                    gK = cute.local_tile(
                        mK_cur,
                        cute.select(self.mma_tiler_qk, mode=[1, 2]),
                        (None, 0, None),
                    )
                else:
                    mK_cur = cute.domain_offset(
                        (k_batch_offset, 0),
                        tma_K[None, None, head_kv_idx],
                    )
                    gK = cute.local_tile(
                        mK_cur,
                        cute.select(self.mma_tiler_qk, mode=[1, 2]),
                        (None, 0),
                    )
                tSgK = thr_mma_qk.partition_B(gK)
                tKsK, tKgK = cpasync.tma_partition(
                    tma_atom_K,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sK, 0, 3),
                    cute.group_modes(tSgK, 0, 3),
                )
                gmem_k_idx = paged_kv_idx
                cute.copy(
                    tma_atom_K,
                    tKgK[(None, 0, gmem_k_idx)]
                    if const_expr(self.paged_kv)
                    else tKgK[(None, gmem_k_idx)],
                    tKsK[(None, kv_slot)],
                    tma_bar_ptr=mbar_k_ptr + kv_slot,
                )
                with cute.arch.elect_one():
                    if const_expr(not self.use_clc):
                        cute.arch.mbarrier_arrive(mbar_k_ptr)

            if warp_idx_in_wg == Int32(1):
                thr_mma_pv = tiled_mma_pv.get_slice(0)
                if const_expr(self.paged_kv):
                    mV_cur = tma_V[None, None, head_kv_idx, None]
                    gV = cute.local_tile(
                        mV_cur,
                        cute.select(self.mma_tiler_pv, mode=[1, 2]),
                        (0, None, None),
                    )
                else:
                    mV_cur = cute.domain_offset(
                        (0, k_batch_offset),
                        tma_V[None, None, head_kv_idx],
                    )
                    gV = cute.local_tile(
                        mV_cur,
                        cute.select(self.mma_tiler_pv, mode=[1, 2]),
                        (0, None),
                    )
                tOgV = thr_mma_pv.partition_B(gV)
                tVsV, tVgV = cpasync.tma_partition(
                    tma_atom_V,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sV, 0, 3),
                    cute.group_modes(tOgV, 0, 3),
                )
                gmem_v_idx = paged_kv_idx
                if const_expr(self.masked_clc):
                    # The empty-slot acquire above protects both TMA and the
                    # zeroing stores from the previous work item's PV reads.
                    # Every populated work item advances this private barrier,
                    # even when its logical block needs no tail repair.
                    tma_ready = mbar_v_tma_ptr + kv_slot
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_expect_tx(
                            tma_ready, self.n_block_size * self.head_dim
                            * self.v_input_dtype.width // 8)
                        cute.arch.mbarrier_arrive(tma_ready)
                    cute.copy(
                        tma_atom_V,
                        tVgV[(None, 0, gmem_v_idx)]
                        if const_expr(self.paged_kv)
                        else tVgV[(None, gmem_v_idx)],
                        tVsV[(None, kv_slot)],
                        tma_bar_ptr=tma_ready,
                    )
                    cute.arch.mbarrier_wait(tma_ready, kv_phase)
                    if kv_valid_cols < Int32(self.n_block_size):
                        self._warp_zero_invalid_v_tail(
                            sV, kv_valid_cols, kv_slot)
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        pipeline_v.producer_commit_w_index(kv_slot)
                elif const_expr(self.safe_partial_v_tail):
                    if kv_valid_cols < Int32(self.n_block_size):
                        cute.copy(
                            tma_atom_V,
                            tVgV[(None, 0, gmem_v_idx)]
                            if const_expr(self.paged_kv)
                            else tVgV[(None, gmem_v_idx)],
                            tVsV[(None, kv_slot)],
                            tma_bar_ptr=mbar_v_tma_ptr,
                        )
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(mbar_v_tma_ptr)
                        cute.arch.mbarrier_wait(mbar_v_tma_ptr, 0)
                        self._warp_zero_invalid_v_tail(sV, kv_valid_cols)
                        cute.arch.sync_warp()
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(mbar_v_ptr)
                    else:
                        cute.copy(
                            tma_atom_V,
                            tVgV[(None, 0, gmem_v_idx)]
                            if const_expr(self.paged_kv)
                            else tVgV[(None, gmem_v_idx)],
                            tVsV[(None, kv_slot)],
                            tma_bar_ptr=mbar_v_ptr + kv_slot,
                        )
                        with cute.arch.elect_one():
                            if const_expr(not self.use_clc):
                                cute.arch.mbarrier_arrive(mbar_v_ptr)
                else:
                    cute.copy(
                        tma_atom_V,
                        tVgV[(None, 0, gmem_v_idx)]
                        if const_expr(self.paged_kv)
                        else tVgV[(None, gmem_v_idx)],
                        tVsV[(None, kv_slot)],
                        tma_bar_ptr=mbar_v_ptr + kv_slot,
                    )
                    with cute.arch.elect_one():
                        if const_expr(not self.use_clc):
                            cute.arch.mbarrier_arrive(mbar_v_ptr)

    @cute.jit
    def _wg_load_q_gather4(
        self,
        mQ_2d: cute.Tensor,
        mQ_gather4_desc: cute.Tensor,
        mK2qQSplitIndices: cute.Tensor,
        sQIdxMeta: cute.Tensor,
        sQ: cute.Tensor,
        pipeline_q,
        load_wg_barrier,
        num_q_groups: Int32,
        count_raw: Int32,
        has_work: Int32,
        head_kv_idx: Int32,
        row_start: Int32,
        q_batch_offset: Int32,
        num_heads_kv: Int32,
        group_base: Int32 = 0,
        pipeline_qidx_meta=None,
    ):
        use_qidx_meta: cutlass.Constexpr[bool] = (
            self.use_qidx_meta_pipeline and pipeline_qidx_meta is not None
        )
        tidx = cute.arch.thread_idx()[0]
        q_load_thread_base = Int32(self.q_load_warp_base * cute.arch.WARP_SIZE)
        group_tidx = tidx - q_load_thread_base
        producer_warp_idx_in_wg = cute.arch.make_warp_uniform(
            group_tidx // Int32(cute.arch.WARP_SIZE)
        )
        q_oob_m_idx = mQ_2d.shape[0] // Int32(self.qheadperkv)
        gathers_per_warp: cutlass.Constexpr[int] = self.m_block_size // (
            self.num_q_load_warps * 4)

        if has_work:
            for qi_group in cutlass.range(num_q_groups, unroll=1):
                meta_group = group_base + qi_group
                slot = meta_group % Int32(self.q_stage)
                phase = (meta_group // Int32(self.q_stage)) & Int32(1)
                producer_phase = phase ^ Int32(1)
                if producer_warp_idx_in_wg == Int32(0):
                    if const_expr(use_qidx_meta):
                        pipeline_qidx_meta.producer_acquire_w_index_phase(
                            meta_group % Int32(self.qidx_meta_stages),
                            ((meta_group // Int32(self.qidx_meta_stages)) & Int32(1)) ^ Int32(1),
                        )
                    pipeline_q.producer_acquire_w_index_phase(
                        slot, producer_phase)
                load_wg_barrier.arrive_and_wait()

                group_tidx = tidx - q_load_thread_base
                warp_idx_in_wg = cute.arch.make_warp_uniform(
                    group_tidx // Int32(cute.arch.WARP_SIZE)
                )
                lane_idx = group_tidx % Int32(cute.arch.WARP_SIZE)
                mbar_ptr = pipeline_q.sync_object_full.get_barrier(slot)
                qidx_meta_slot = ((meta_group if const_expr(use_qidx_meta) else qi_group)
                                  & Int32(self.qidx_meta_stages - 1)) * Int32(
                                      self.q_tokens_per_group)

                meta_iters: cutlass.Constexpr[int] = (
                    (self.q_tokens_per_group
                     + self.num_q_load_warps * cute.arch.WARP_SIZE - 1)
                    // (self.num_q_load_warps * cute.arch.WARP_SIZE)
                )
                for meta_iter in cutlass.range_constexpr(meta_iters):
                    tok_idx_g4 = (
                        (Int32(meta_iter) * Int32(self.num_q_load_warps)
                         + warp_idx_in_wg)
                        * Int32(cute.arch.WARP_SIZE)
                        + lane_idx
                    )
                    if tok_idx_g4 < Int32(self.q_tokens_per_group):
                        qi = qi_group * Int32(self.q_tokens_per_group) + tok_idx_g4
                        if qi < count_raw:
                            sQIdxMeta[qidx_meta_slot + tok_idx_g4] = self._load_qsplit_idx(
                                mK2qQSplitIndices, head_kv_idx, row_start, qi
                            )
                        else:
                            sQIdxMeta[qidx_meta_slot + tok_idx_g4] = Int32(0)
                load_wg_barrier.arrive_and_wait()

                if const_expr(use_qidx_meta):
                    if producer_warp_idx_in_wg == Int32(0):
                        pipeline_qidx_meta.producer_commit_w_index(
                            meta_group % Int32(self.qidx_meta_stages))

                with cute.arch.elect_one():
                    q_desc_ptr = mQ_gather4_desc.iterator
                    sQ_ptr = sQ.iterator
                    for gather_slot in cutlass.range_constexpr(gathers_per_warp):
                        gather_idx = (
                            Int32(gather_slot) * Int32(self.num_q_load_warps)
                            + warp_idx_in_wg
                        )
                        tok_base = gather_idx * Int32(self.tokens_per_gather4)
                        if const_expr(self.qheadperkv == 1):
                            qi0 = qi_group * Int32(self.q_tokens_per_group) + tok_base
                            qi1 = qi0 + Int32(1)
                            qi2 = qi0 + Int32(2)
                            qi3 = qi0 + Int32(3)
                            row0 = q_oob_m_idx
                            row1 = q_oob_m_idx
                            row2 = q_oob_m_idx
                            row3 = q_oob_m_idx
                            if qi0 < count_raw:
                                q_idx0 = self._decode_q_idx_from_qsplit(
                                    sQIdxMeta[qidx_meta_slot + tok_base])
                                row0 = (q_batch_offset + q_idx0) * num_heads_kv + head_kv_idx
                            if qi1 < count_raw:
                                q_idx1 = self._decode_q_idx_from_qsplit(
                                    sQIdxMeta[qidx_meta_slot + tok_base + Int32(1)])
                                row1 = (q_batch_offset + q_idx1) * num_heads_kv + head_kv_idx
                            if qi2 < count_raw:
                                q_idx2 = self._decode_q_idx_from_qsplit(
                                    sQIdxMeta[qidx_meta_slot + tok_base + Int32(2)])
                                row2 = (q_batch_offset + q_idx2) * num_heads_kv + head_kv_idx
                            if qi3 < count_raw:
                                q_idx3 = self._decode_q_idx_from_qsplit(
                                    sQIdxMeta[qidx_meta_slot + tok_base + Int32(3)])
                                row3 = (q_batch_offset + q_idx3) * num_heads_kv + head_kv_idx
                        elif const_expr(self.qheadperkv == 2):
                            qi0 = qi_group * Int32(self.q_tokens_per_group) + tok_base
                            qi1 = qi0 + Int32(1)
                            row_base0 = q_oob_m_idx * Int32(self.qheadperkv)
                            row_base1 = q_oob_m_idx * Int32(self.qheadperkv)
                            if qi0 < count_raw:
                                q_idx0 = self._decode_q_idx_from_qsplit(
                                    sQIdxMeta[qidx_meta_slot + tok_base])
                                row_base0 = (
                                    (q_batch_offset + q_idx0) * num_heads_kv
                                    + head_kv_idx
                                ) * Int32(self.qheadperkv)
                            if qi1 < count_raw:
                                q_idx1 = self._decode_q_idx_from_qsplit(
                                    sQIdxMeta[qidx_meta_slot + tok_base + Int32(1)])
                                row_base1 = (
                                    (q_batch_offset + q_idx1) * num_heads_kv
                                    + head_kv_idx
                                ) * Int32(self.qheadperkv)
                            row0 = row_base0
                            row1 = row_base0 + Int32(1)
                            row2 = row_base1
                            row3 = row_base1 + Int32(1)
                        else:
                            qi0 = qi_group * Int32(self.q_tokens_per_group) + tok_base
                            row_base0 = q_oob_m_idx * Int32(self.qheadperkv)
                            if qi0 < count_raw:
                                q_idx0 = self._decode_q_idx_from_qsplit(
                                    sQIdxMeta[qidx_meta_slot + tok_base])
                                row_base0 = (
                                    (q_batch_offset + q_idx0) * num_heads_kv
                                    + head_kv_idx
                                ) * Int32(self.qheadperkv)
                            row0 = row_base0
                            row1 = row_base0 + Int32(1)
                            row2 = row_base0 + Int32(2)
                            row3 = row_base0 + Int32(3)
                        group_byte_off = gather_idx * Int32(
                            4 * self.k_tile * (self.q_dtype.width // 8)
                        )
                        if const_expr(self.q_dtype == cutlass.Float8E4M3FN):
                            stage_byte_off = slot * Int32(self.q_stage_stride_bytes)
                            full_group_byte_off = gather_idx * Int32(
                                4 * self.head_dim * (self.q_dtype.width // 8)
                            )
                            tma_gather4_cached(
                                sQ_ptr,
                                stage_byte_off + full_group_byte_off,
                                q_desc_ptr,
                                Int32(0),
                                row0,
                                row1,
                                row2,
                                row3,
                                mbar_ptr,
                                TMA_CACHE_EVICT_LAST,
                            )
                        else:
                            for ks_c in cutlass.range_constexpr(self.k_stages):
                                stage_idx = slot * Int32(self.k_stages) + Int32(ks_c)
                                stage_byte_off = stage_idx * Int32(self.k_tile_stride_bytes)
                                if const_expr(ks_c + 1 < self.k_stages):
                                    tma_gather4_prefetch(
                                        q_desc_ptr,
                                        Int32((ks_c + 1) * self.k_tile),
                                        row0,
                                        row1,
                                        row2,
                                        row3,
                                        TMA_CACHE_EVICT_LAST,
                                    )
                                tma_gather4_cached(
                                    sQ_ptr,
                                    stage_byte_off + group_byte_off,
                                    q_desc_ptr,
                                    Int32(ks_c * self.k_tile),
                                    row0,
                                    row1,
                                    row2,
                                    row3,
                                    mbar_ptr,
                                    TMA_CACHE_EVICT_LAST,
                                )
                load_wg_barrier.arrive_and_wait()

            if const_expr(not self.use_clc) and producer_warp_idx_in_wg == Int32(0):
                next_group = group_base + num_q_groups
                next_slot = next_group % Int32(self.q_stage)
                next_phase = (
                    (next_group // Int32(self.q_stage)) & Int32(1)
                ) ^ Int32(1)
                pipeline_q.producer_acquire_w_index_phase(next_slot, next_phase)

    @cute.jit
    def _wg_load_q_transport(
        self, tma_atom_Q, mQ_2d, mQ_gather4_desc, sQ,
        mK2qQSplitIndices, sQLoadMIdx, sQIdxMeta, sQ_load,
        pipeline_q, load_wg_barrier, num_q_groups, count_raw, has_work,
        head_kv_idx, row_start, q_batch_offset, num_heads_kv,
        group_base=0, pipeline_qidx_meta=None,
    ):
        """Dispatch the Q producer while retaining persistent slot ownership."""
        if const_expr(self.use_q_gather4):
            self._wg_load_q_gather4(
                mQ_2d, mQ_gather4_desc, mK2qQSplitIndices, sQIdxMeta,
                sQ, pipeline_q, load_wg_barrier, num_q_groups, count_raw,
                has_work, head_kv_idx, row_start, q_batch_offset,
                num_heads_kv, group_base, pipeline_qidx_meta,
            )
        else:
            self._wg_load_q_tma(
                tma_atom_Q, mQ_2d, mK2qQSplitIndices, sQLoadMIdx,
                sQIdxMeta, sQ_load, pipeline_q, load_wg_barrier,
                num_q_groups, count_raw, has_work, head_kv_idx, row_start,
                q_batch_offset, num_heads_kv, group_base, pipeline_qidx_meta,
            )

    @cute.jit
    def _wg_load_q_tma(
        self,
        tma_atom_Q,
        mQ_2d: cute.Tensor,
        mK2qQSplitIndices: cute.Tensor,
        sQLoadMIdx: cute.Tensor,
        sQIdxMeta: cute.Tensor,
        sQ_load: cute.Tensor,
        pipeline_q,
        load_wg_barrier,
        num_q_groups: Int32,
        count_raw: Int32,
        has_work: Int32,
        head_kv_idx: Int32,
        row_start: Int32,
        q_batch_offset: Int32,
        num_heads_kv: Int32,
        group_base: Int32 = 0,
        pipeline_qidx_meta=None,
    ):
        use_qidx_meta: cutlass.Constexpr[bool] = (
            self.use_qidx_meta_pipeline and pipeline_qidx_meta is not None
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        warp_idx_in_wg = warp_idx - Int32(self.q_load_warp_base)
        if const_expr(self.use_q_cpasync):
            q_tile: cutlass.Constexpr[int] = (
                self.head_dim if self.q_dtype == cutlass.Float8E4M3FN else self.k_tile
            )
            q_subtiles: cutlass.Constexpr[int] = self.head_dim // q_tile
            q_copy_threads: cutlass.Constexpr[int] = min(
                cute.arch.WARP_SIZE, self.qheadperkv * q_tile // (128 // self.q_dtype.width)
            )
            # Cotile the physical SW128 layout so vector lanes preserve the
            # swizzle; scalar coordinate offsets do not preserve that mapping.
            q_layout = sm100_utils.make_smem_layout(
                tcgen05.OperandMajorMode.K,
                (self.qheadperkv, q_tile),
                self.q_dtype,
                self.q_stage * self.q_tokens_per_group * q_subtiles,
            )
            sQ_load = cute.make_tensor(sQ_load.iterator, q_layout.outer)
            q_copy = cute.make_cotiled_copy(
                cute.make_copy_atom(
                    cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                    self.q_dtype,
                    num_bits_per_copy=128,
                ),
                cute.make_ordered_layout(
                    (q_copy_threads, 128 // self.q_dtype.width), (1, 0)
                ),
                cute.select(q_layout.outer, mode=[0, 1]),
            )
        elif const_expr(self.q_dtype == cutlass.Float8E4M3FN):
            gQ_full = cute.local_tile(
                mQ_2d, (self.qheadperkv, self.head_dim), (None, 0))
            load_Q_fn_full, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), gQ_full, sQ_load)
            load_Q_fn_k0, load_Q_fn_k1 = None, None
        else:
            gQ_k0 = cute.local_tile(mQ_2d, (self.qheadperkv, self.k_tile), (None, 0))
            gQ_k1 = cute.local_tile(mQ_2d, (self.qheadperkv, self.k_tile), (None, 1))
            load_Q_fn_k0, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), gQ_k0, sQ_load)
            load_Q_fn_k1, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), gQ_k1, sQ_load)
            load_Q_fn_full = None
        q_oob_m_idx = mQ_2d.shape[0] // Int32(self.qheadperkv)
        tokens_per_warp: cutlass.Constexpr[int] = (
            (self.q_tokens_per_group + self.num_q_load_warps - 1)
            // self.num_q_load_warps
        )

        if has_work:
            for qi_group in cutlass.range(num_q_groups, unroll=1):
                slot = (group_base + qi_group) % Int32(self.q_stage)
                phase = (
                    (group_base + qi_group) // Int32(self.q_stage)
                ) & Int32(1)
                producer_phase = phase ^ Int32(1)
                if warp_idx_in_wg == Int32(0):
                    if const_expr(use_qidx_meta):
                        meta_group = group_base + qi_group
                        pipeline_qidx_meta.producer_acquire_w_index_phase(
                            meta_group % Int32(self.qidx_meta_stages),
                            ((meta_group // Int32(self.qidx_meta_stages)) & Int32(1)) ^ Int32(1),
                        )
                    pipeline_q.producer_acquire_w_index_phase(
                        slot, producer_phase
                    )
                load_wg_barrier.arrive_and_wait()

                mbar_ptr = pipeline_q.sync_object_full.get_barrier(slot)
                q_load_subtiles_per_token = (
                    1 if const_expr(self.q_dtype == cutlass.Float8E4M3FN)
                    else self.k_stages
                )
                sub_stage_base = slot * Int32(
                    self.q_tokens_per_group * q_load_subtiles_per_token)
                load_meta_slot = slot * Int32(self.q_tokens_per_group)
                meta_group = (
                    group_base + qi_group
                    if const_expr(use_qidx_meta) else qi_group
                )
                qidx_meta_slot = (
                    meta_group & Int32(self.qidx_meta_stages - 1)
                ) * Int32(self.q_tokens_per_group)

                if warp_idx_in_wg == Int32(0) and lane_idx < Int32(self.q_tokens_per_group):
                    tok_idx = lane_idx
                    qi = qi_group * Int32(self.q_tokens_per_group) + tok_idx
                    if qi < count_raw:
                        qsplit = self._load_qsplit_idx(
                            mK2qQSplitIndices, head_kv_idx, row_start, qi
                        )
                        q_idx = self._decode_q_idx_from_qsplit(qsplit)
                        q_abs = q_batch_offset + q_idx
                        if const_expr(not self.use_clc or use_qidx_meta):
                            sQIdxMeta[qidx_meta_slot + tok_idx] = qsplit
                        sQLoadMIdx[load_meta_slot + tok_idx] = (
                            q_abs * num_heads_kv + head_kv_idx
                        )
                    else:
                        if const_expr(not self.use_clc or use_qidx_meta):
                            sQIdxMeta[qidx_meta_slot + tok_idx] = Int32(0)
                        sQLoadMIdx[load_meta_slot + tok_idx] = (
                            q_oob_m_idx + qi % count_raw
                        )
                if const_expr(not self.use_q_cpasync):
                    load_wg_barrier.arrive_and_wait()
                if const_expr(use_qidx_meta):
                    if warp_idx_in_wg == Int32(0):
                        # LDGSTS needs publication before the WG barrier to
                        # end metadata state before Q-copy address construction.
                        if const_expr(self.use_q_cpasync):
                            cute.arch.sync_warp()
                        pipeline_qidx_meta.producer_commit_w_index(
                            (group_base + qi_group) % Int32(self.qidx_meta_stages)
                        )
                if const_expr(self.use_q_cpasync):
                    load_wg_barrier.arrive_and_wait()

                for qi_slot in cutlass.range_constexpr(tokens_per_warp):
                    tok_idx = (
                        warp_idx_in_wg * Int32(tokens_per_warp)
                        + Int32(qi_slot)
                    )
                    if tok_idx < Int32(self.q_tokens_per_group):
                        m_tile_idx = cute.arch.make_warp_uniform(
                            sQLoadMIdx[load_meta_slot + tok_idx]
                        )
                        if const_expr(self.use_q_cpasync):
                            if lane_idx < Int32(q_copy_threads):
                                q_thread_copy = q_copy.get_slice(lane_idx)
                                for half in cutlass.range_constexpr(q_subtiles):
                                    gQ = cute.local_tile(
                                        mQ_2d,
                                        (self.qheadperkv, q_tile),
                                        (m_tile_idx, half),
                                    )
                                    sQ = sQ_load[
                                        None, None,
                                        sub_stage_base + Int32(half * self.q_tokens_per_group) + tok_idx,
                                    ]
                                    tQgQ = q_thread_copy.partition_S(gQ)
                                    tQsQ = q_thread_copy.partition_D(sQ)
                                    predicate = cute.make_rmem_tensor(
                                        (tQgQ.shape[0][1], tQgQ.shape[1], tQgQ.shape[2]),
                                        cutlass.Boolean,
                                    )
                                    predicate.fill(m_tile_idx < q_oob_m_idx)
                                    cute.copy(q_copy, tQgQ, tQsQ, pred=predicate)
                        elif const_expr(self.q_dtype == cutlass.Float8E4M3FN):
                            load_Q_fn_full(
                                src_idx=m_tile_idx,
                                dst_idx=sub_stage_base + tok_idx,
                                tma_bar_ptr=mbar_ptr,
                            )
                        else:
                            load_Q_fn_k0(
                                src_idx=m_tile_idx,
                                dst_idx=sub_stage_base + tok_idx,
                                tma_bar_ptr=mbar_ptr,
                            )
                            load_Q_fn_k1(
                                src_idx=m_tile_idx,
                                dst_idx=(
                                    sub_stage_base
                                    + Int32(self.q_tokens_per_group)
                                    + tok_idx
                                ),
                                tma_bar_ptr=mbar_ptr,
                            )

                if const_expr(self.use_q_cpasync):
                    cute.arch.cp_async_mbarrier_arrive_noinc(mbar_ptr)

            if const_expr(not self.use_clc) and warp_idx_in_wg == Int32(0):
                next_slot = (group_base + num_q_groups) % Int32(self.q_stage)
                next_phase = (
                    ((group_base + num_q_groups) // Int32(self.q_stage)) & Int32(1)
                ) ^ Int32(1)
                pipeline_q.producer_acquire_w_index_phase(next_slot, next_phase)

    @cute.jit
    def _wg_mma_issue(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tOrP: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sQ: cute.Tensor,
        pipeline_q,
        pipeline_s,
        pipeline_p,
        pipeline_p_lastsplit,
        pipeline_o,
        mbar_k_ptr,
        mbar_v_ptr,
        num_q_groups: Int32,
        has_work: Int32,
        group_base: Int32 = 0,
        work_seq: Int32 = 0,
        pipeline_k=None,
        pipeline_v=None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        is_mma_warp = warp_idx == Int32(self.mma_warp_id)
        kv_slot = work_seq % Int32(self.kv_stage)
        kv_phase = (work_seq // Int32(self.kv_stage)) & Int32(1)

        if is_mma_warp:
            if has_work:
                tSrQ = tiled_mma_qk.make_fragment_A(sQ)
                tSrK = tiled_mma_qk.make_fragment_B(sK)
                tSrQ0 = tSrQ[None, None, None, 0]
                tSrK0 = tSrK[None, None, None, kv_slot]
                tOrV = tiled_mma_pv.make_fragment_B(sV)
                tOrV0 = tOrV[None, None, None, kv_slot]
                sV0 = sV[None, None, None, kv_slot]
                pv_mma_op = tiled_mma_pv.op
                qk_mma_op = tiled_mma_qk.op
                q_smem_base = sm100_desc.smem_desc_base_from_tensor(
                    sQ, sm100_desc.Major.K)
                k_smem_base = sm100_desc.smem_desc_base_from_tensor(
                    sK, sm100_desc.Major.K)
                # The K stage is warp-uniform (and kv_stage is one on the
                # UTMASTG path). Promote it before descriptor construction
                # grows the live GPR set. The Q stage advances with group_base
                # and must remain lane-derived here.
                k_smem_start = Int32(cute.arch.make_warp_uniform(
                    sm100_desc.make_smem_desc_start_addr(
                        sK[None, None, None, kv_slot].iterator)))
                q_smem_start = sm100_desc.make_smem_desc_start_addr(
                    sQ[
                        None,
                        None,
                        None,
                        (group_base + Int32(self.q_stage - 1))
                        % Int32(self.q_stage),
                    ].iterator)
                sm100_helpers.declare_ptx_smem_desc(
                    q_smem_start,
                    q_smem_base,
                    tSrQ0.layout,
                    var_name_prefix="lean_q_desc",
                )
                sm100_helpers.declare_ptx_idesc(
                    qk_mma_op, var_name="lean_qk_idesc")
                sQ_stage_stride = (
                    sQ.layout.stride[-1] * sQ.element_type.width // 8) >> 4
                if const_expr(self.q_stage == 1):
                    sQ_stage_stride = 0
                q_wrap_offset = -(self.q_stage - 1) * sQ_stage_stride
                q_advance_offset = sQ_stage_stride
                gemm_qk = partial(
                    sm100_helpers.gemm_ptx_precomputed_varname,
                    smem_desc_base_b=k_smem_base,
                    tCrB_layout=tSrK0.layout,
                    smem_var_name_prefix="lean_q_desc",
                    idesc_var_name="lean_qk_idesc",
                    zero_init=True,
                    cta_group=self.cta_group_size,
                    mma_kind=self.qk_mma_kind,
                )
                gemm_pv = partial(
                    sm100_helpers.gemm_ptx_partial,
                    pv_mma_op,
                    tCrA=tOrP[None, None, None, 0],
                    sA=None,
                    split_arrive=(
                        self.split_P_arrive if self.split_P_arrive > 0 else None),
                    cta_group=self.cta_group_size,
                    mma_kind=self.pv_mma_kind,
                )

                cute.arch.mbarrier_wait(mbar_k_ptr + kv_slot, kv_phase)
                # Issue order:
                #   Q0K, Q1K, P0V, Q2K, P1V, Q3K, ...
                # This reuses each slot as soon as its previous PV drains,
                # instead of batching both PVs after both QKs of a pair.
                # The schedule is still 2-slot safe:
                #   - QK(qi) consumes slot qi&1
                #   - PV(qi-2) frees the same slot before QK(qi) reuses it
                #   - phases still toggle every 2 groups per slot

                # Prologue: issue up to the first two QK tiles. Q slots come
                # from the q_stage ring; S slots remain a 2-slot ring.
                for local_qi in cutlass.range(
                    cutlass.min(num_q_groups, Int32(2)), unroll=1
                ):
                    q_group = group_base + local_qi
                    # Every lane follows the same query-group schedule.  Tell
                    # the compiler that the runtime slot selector is uniform
                    # so the QK path does not need BSSY/BSYNC reconvergence.
                    slot = cute.arch.make_warp_uniform(q_group & Int32(1))
                    phase = (q_group // Int32(2)) & Int32(1)
                    pipeline_q.consumer_wait_w_index_phase(slot, phase)
                    pipeline_s.producer_acquire_w_index_phase(
                        slot, phase ^ Int32(1)
                    )
                    gemm_qk(
                        Int32(self.tmem_s_offset) +
                        slot * Int32(self.tmem_stage_stride),
                        smem_desc_start_b=k_smem_start,
                        smem_offset=(
                            Int32(q_wrap_offset) +
                            slot * Int32(q_advance_offset - q_wrap_offset)
                        ),
                    )
                    pipeline_s.producer_commit_w_index(slot)
                    pipeline_q.consumer_release_w_index(slot)

                cute.arch.mbarrier_wait(mbar_v_ptr + kv_slot, kv_phase)

                # Steady-state: for qi >= 2, reuse the S/P slot as a BLK128-
                # style handoff. The MMA warp waits for softmax to release
                # the slot after P is visible, issues PV, then immediately
                # reuses the same acquired slot for the next QK.
                for qi in cutlass.range(Int32(2), num_q_groups, unroll=1):
                    pv_qi = qi - Int32(2)
                    pv_slot = cute.arch.make_warp_uniform(
                        (group_base + pv_qi) & Int32(1)
                    )
                    pv_phase = (
                        (group_base + pv_qi) // Int32(2)
                    ) & Int32(1)
                    pipeline_p.consumer_wait_w_index_phase(pv_slot, pv_phase)
                    pipeline_o.producer_acquire_w_index_phase(
                        pv_slot, pv_phase ^ Int32(1))
                    gemm_pv(
                        Int32(self.tmem_o_offset) +
                        pv_slot * Int32(self.tmem_o_stage_stride),
                        tCrB=tOrV0,
                        sB=sV0,
                        mbar_ptr=(
                            pipeline_p_lastsplit.sync_object_full.get_barrier(pv_slot)
                            if self.split_P_arrive > 0 else None),
                        mbar_phase=(pv_phase if self.split_P_arrive > 0 else None),
                        tA_addr=(
                            Int32(self.tmem_p_offset) +
                            pv_slot * Int32(self.tmem_stage_stride)
                        ),
                        zero_init=True,
                    )
                    pipeline_o.producer_commit_w_index(pv_slot)
                    if cutlass.const_expr(self.split_P_arrive > 0):
                        pipeline_p_lastsplit.consumer_release_w_index(pv_slot)
                    pipeline_p.consumer_release_w_index(pv_slot)

                    q_slot = cute.arch.make_warp_uniform(
                        (group_base + qi) % Int32(self.q_stage)
                    )
                    q_phase = (
                        (group_base + qi) // Int32(self.q_stage)
                    ) & Int32(1)
                    s_slot = cute.arch.make_warp_uniform(
                        (group_base + qi) & Int32(1)
                    )
                    s_phase = ((group_base + qi) // Int32(2)) & Int32(1)
                    pipeline_q.consumer_wait_w_index_phase(q_slot, q_phase)
                    pipeline_s.producer_acquire_w_index_phase(
                        s_slot, s_phase ^ Int32(1))
                    gemm_qk(
                        Int32(self.tmem_s_offset) +
                        s_slot * Int32(self.tmem_stage_stride),
                        smem_desc_start_b=k_smem_start,
                        smem_offset=(
                            Int32(q_wrap_offset) +
                            q_slot * Int32(q_advance_offset - q_wrap_offset)
                        ),
                    )
                    pipeline_s.producer_commit_w_index(s_slot)
                    pipeline_q.consumer_release_w_index(q_slot)

                if const_expr(self.use_clc):
                    pipeline_k.consumer_release_w_index(kv_slot)

                # Drain the remaining one or two PV tiles.
                drain_begin = Int32(0) if num_q_groups == Int32(1) else num_q_groups - Int32(2)
                for pv_qi in cutlass.range(drain_begin, num_q_groups, unroll=1):
                    pv_slot = cute.arch.make_warp_uniform(
                        (group_base + pv_qi) & Int32(1)
                    )
                    pv_phase = (
                        (group_base + pv_qi) // Int32(2)
                    ) & Int32(1)
                    pipeline_p.consumer_wait_w_index_phase(pv_slot, pv_phase)
                    pipeline_o.producer_acquire_w_index_phase(
                        pv_slot, pv_phase ^ Int32(1))
                    gemm_pv(
                        Int32(self.tmem_o_offset) +
                        pv_slot * Int32(self.tmem_o_stage_stride),
                        tCrB=tOrV0,
                        sB=sV0,
                        mbar_ptr=(
                            pipeline_p_lastsplit.sync_object_full.get_barrier(pv_slot)
                            if self.split_P_arrive > 0 else None),
                        mbar_phase=(pv_phase if self.split_P_arrive > 0 else None),
                        tA_addr=(
                            Int32(self.tmem_p_offset) +
                            pv_slot * Int32(self.tmem_stage_stride)
                        ),
                        zero_init=True,
                    )
                    pipeline_o.producer_commit_w_index(pv_slot)
                    if cutlass.const_expr(self.split_P_arrive > 0):
                        pipeline_p_lastsplit.consumer_release_w_index(pv_slot)
                    pipeline_p.consumer_release_w_index(pv_slot)
                if const_expr(self.use_clc):
                    pipeline_v.consumer_release_w_index(kv_slot)

    @cute.jit
    def _softmax_step(
        self,
        slot: cutlass.Constexpr[int],
        s_consumer_phase: Int32,
        p_producer_phase: Int32,
        sm_stats_producer_phase: Int32,
        softmax: SoftmaxSm100,
        sScale: cute.Tensor,
        sScaleTemperature: cute.Tensor,
        pipeline_s,
        pipeline_p,
        pipeline_p_lastsplit,
        pipeline_sm_stats,
        sm_stats_barrier,
        stats_barrier_idx: Int32,
        thr_tmem_load,
        thr_tmem_store,
        tStS_t2r: cute.Tensor,
        tStP_r2t: cute.Tensor,
        tScS_t2r: cute.Tensor,
        tScP_shape,
        sQIdxMeta: cute.Tensor,
        qidx_meta_slot: Int32,
        mK2qQSplitIndices: cute.Tensor,
        qsplit_row_start: Int32,
        head_kv_idx: Int32,
        qi_group: Int32,
        group_tidx: Int32,
        masked_tok_count: Int32,
        kv_block_col_start: Int32,
        seq_len_q: Int32,
        causal_q_offset: Int32,
        kv_valid_cols: Int32,
        lse_temperature_scale: Float32,
        return_temperature_lse: cutlass.Constexpr[bool],
        apply_causal_mask: cutlass.Constexpr[bool] = False,
        signal_stats_barrier: cutlass.Constexpr[bool] = True,
    ):
        slot_rt = Int32(slot)

        pipeline_s.consumer_wait_w_index_phase(slot_rt, s_consumer_phase)

        tSrS_t2r = cute.make_rmem_tensor(
            tScS_t2r.shape,
            self.qk_acc_dtype)
        if const_expr(self.use_tmem_load_reduce):
            stats_atom = cute.make_copy_atom(
                tcgen05.copy.LdRed32x32bOp(
                    tcgen05.copy.Repetition(32),
                    redOp=tcgen05.TmemLoadRedOp.MAX,
                ),
                self.qk_acc_dtype,
            )
            tSrMax = cute.make_rmem_tensor(
                (1, cute.size(tStS_t2r.shape[2])), Float32
            )
            for k in cutlass.range_constexpr(cute.size(tStS_t2r.shape[2])):
                cute.copy_atom_call(
                    stats_atom,
                    tStS_t2r[None, 0, k],
                    (tSrS_t2r[None, 0, k], tSrMax[None, k]),
                )
        else:
            cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)

        # The score fragment is now register-owned. Drain every asynchronous
        # TMEM load before releasing this stage so QK MMA can overwrite it
        # while softmax continues from registers.
        cute.arch.fence_view_async_tmem_load()
        pipeline_s.consumer_release_w_index(slot_rt)

        if const_expr(not self.cfg.full_kv_blocks):
            seqlen_info = SeqlenInfoQK(
                Int32(0),
                Int32(0),
                Int32(0),
                Int32(0),
                seq_len_q,
                seq_len_q + causal_q_offset,
                False,
                False,
                False,
                False,
            )
            mask = AttentionMask(
                self.m_block_size,
                self.n_block_size,
                seqlen_info,
            )
            if const_expr(self.causal and apply_causal_mask):
                need_causal_mask = masked_tok_count > Int32(0)
                if need_causal_mask:
                    tok_idx = group_tidx // Int32(self.qheadperkv)
                    if const_expr(self.safe_fwd_sync):
                        qi = qi_group * Int32(self.q_tokens_per_group) + tok_idx
                        qsplit = self._load_qsplit_idx(
                            mK2qQSplitIndices,
                            head_kv_idx,
                            qsplit_row_start,
                            qi,
                        )
                    else:
                        qsplit = sQIdxMeta[qidx_meta_slot + tok_idx]
                    q_idx = self._decode_q_idx_from_qsplit(qsplit)
                    mask.apply_mask_sm100(
                        tSrS_t2r,
                        tScS_t2r,
                        m_block=Int32(0),
                        n_block=Int32(0),
                        mask_seqlen=True,
                        mask_causal=True,
                        row_idx=q_idx,
                        kv_valid_cols=kv_valid_cols,
                        kv_block_col_start=kv_block_col_start,
                    )
                elif kv_valid_cols < Int32(self.n_block_size):
                    mask.apply_mask_sm100(
                        tSrS_t2r,
                        tScS_t2r,
                        m_block=Int32(0),
                        n_block=Int32(0),
                        mask_seqlen=True,
                        mask_causal=False,
                        kv_valid_cols=kv_valid_cols,
                    )
            elif kv_valid_cols < Int32(self.n_block_size):
                mask.apply_mask_sm100(
                    tSrS_t2r,
                    tScS_t2r,
                    m_block=Int32(0),
                    n_block=Int32(0),
                    mask_seqlen=True,
                    mask_causal=False,
                    kv_valid_cols=kv_valid_cols,
                )

        # Each sparse CTA computes exactly one KV block for the current Q group,
        # so full-tile softmax is always the first and only online-softmax step.
        if const_expr(
            self.use_tmem_load_reduce
        ):
            row_max = tSrMax[0, 0]
            for k in cutlass.range_constexpr(1, cute.size(tSrMax.shape[1])):
                row_max = cute.arch.fmax(row_max, tSrMax[0, k])
            softmax.row_max[0] = row_max
            row_max = row_max if row_max != -Float32.inf else Float32(0.0)
        else:
            row_max, _ = softmax.update_row_max(tSrS_t2r.load(), True)
        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        if const_expr(return_temperature_lse):
            lse_temperature_row_sum = softmax.compute_scaled_exp2_row_sum(
                tSrS_t2r,
                lse_temperature_scale,
            )

        if cutlass.const_expr(self.split_P_arrive > 0):
            # This full barrier is the late-P handoff consumed inside
            # gemm_ptx_partial after its early PV k-slices are issued.
            pipeline_p_lastsplit.producer_acquire_w_index_phase(
                slot_rt, p_producer_phase)
        pipeline_p.producer_acquire_w_index_phase(slot_rt, p_producer_phase)
        tSrP_r2t_f32 = cute.make_rmem_tensor(
            thr_tmem_store.partition_S(
                cute.make_identity_tensor(tScP_shape)).shape,
            Float32)
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.p_dtype),
            tSrS_t2r.layout)
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,
            ex2_emu_freq=self.ex2_emu_freq,
            ex2_emu_start_frg=self.ex2_emu_start_frg,
        )

        for k in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
            cute.copy(
                thr_tmem_store,
                tSrP_r2t_f32[None, None, k],
                tStP_r2t[None, None, k])
            if cutlass.const_expr(self.split_P_arrive > 0):
                split_idx = (
                    cute.size(tStP_r2t.shape[2])
                    * self.split_P_arrive // self.n_block_size)
                if cutlass.const_expr(k + 1 == split_idx):
                    cute.arch.fence_view_async_tmem_store()
                    pipeline_p.producer_commit_w_index(slot_rt)
        cute.arch.fence_view_async_tmem_store()
        if cutlass.const_expr(self.split_P_arrive == 0):
            pipeline_p.producer_commit_w_index(slot_rt)
        else:
            pipeline_p_lastsplit.producer_commit_w_index(slot_rt)
        pipeline_sm_stats.producer_acquire_w_index_phase(
            slot_rt, sm_stats_producer_phase)
        softmax.update_row_sum(tSrS_t2r.load(), Float32(0.0), True)
        del tSrS_t2r
        sScale_slot = cute.make_tensor(
            sScale.iterator + slot_rt * Int32(self.m_block_size * 2),
            cute.make_layout(self.m_block_size * 2),
        )
        sScale_slot[group_tidx] = softmax.row_sum[0]
        sScale_slot[group_tidx + Int32(self.m_block_size)] = (
            softmax.row_max[0])
        if const_expr(return_temperature_lse):
            sScale_temperature_slot = cute.make_tensor(
                sScaleTemperature.iterator + slot_rt * Int32(self.m_block_size),
                cute.make_layout(self.m_block_size),
            )
            sScale_temperature_slot[group_tidx] = lse_temperature_row_sum
        cute.arch.fence_view_async_shared()

        if const_expr(signal_stats_barrier):
            sm_stats_barrier.arrive_w_index(index=stats_barrier_idx)

    @cute.jit
    def _wg_softmax(
        self,
        stage: cutlass.Constexpr[int],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tStS: cute.Tensor,
        sScale: cute.Tensor,
        sScaleTemperature: cute.Tensor,
        sSplitIdx: cute.Tensor,
        sQIdx: cute.Tensor,
        sQIdxMeta: cute.Tensor,
        mK2qQSplitIndices: cute.Tensor,
        qsplit_row_start: Int32,
        pipeline_s,
        pipeline_p,
        pipeline_p_lastsplit,
        pipeline_o,
        pipeline_sm_stats,
        sm_stats_barrier,
        epilogue_barrier,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mLSE_temperature_partial: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        lse_temperature_scale_log2: Float32,
        lse_temperature_scale: Float32,
        kv_block_idx: Int32,
        kv_valid_cols: Int32,
        diag_q_count: Int32,
        num_q_groups: Int32,
        count_raw: Int32,
        has_work: Int32,
        causal_q_offset: Int32,
        batch_idx: Int32,
        head_kv_idx: Int32,
        seq_len_q: Int32,
        head_q: Int32,
        num_heads_kv: Int32,
        q_batch_offset: Int32,
        mQ_2d: cute.Tensor,
        group_base: Int32 = 0,
        operation: cutlass.Constexpr[int] = ConsumerOperation.FUSED.value,
        single_group: Int32 = 0,
        o_store_context=None,
        work_tile_base: Int32 = 0,
        pipeline_qidx_meta=None,
    ):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_idx_in_wg = warp_idx % Int32(self.warps_per_group)
        group_tidx = (
            warp_idx_in_wg * Int32(cute.arch.WARP_SIZE)
            + tidx % Int32(cute.arch.WARP_SIZE)
        )
        stats_barrier_idx = (
            Int32(stage) * Int32(self.warps_per_group) + warp_idx_in_wg)

        thr0_qk = tiled_mma_qk.get_slice(0)
        tScS = thr0_qk.partition_C(
            cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]
        cta_qk_tiler = (
            self.mma_tiler_qk[0] // thr0_qk.thr_id.shape,
            self.mma_tiler_qk[1])
        tilePlikeFP32 = (
            self.mma_tiler_qk[1] // Float32.width * self.p_dtype.width)
        tScP_shape = (cta_qk_tiler[0], tilePlikeFP32)
        tSAcc = tStS[(None, None), 0, 0, stage]

        softmax = SoftmaxSm100.create(softmax_scale_log2)
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.qk_acc_dtype)
        thr_tmem_load = tcgen05.make_tmem_copy(
            tmem_load_atom, tSAcc).get_slice(group_tidx)
        tStS_t2r = thr_tmem_load.partition_S(tSAcc)
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        tStP_layout = cute.composition(
            tSAcc.layout,
            cute.make_layout((self.m_block_size, tilePlikeFP32)))
        tStP = cute.make_tensor(
            tSAcc.iterator + self.tmem_s_to_p_offset, tStP_layout)
        # P-store Repetition is dtype-aware: each PV MMA K-segment is
        # ``32 / (p_dtype.width / 8)`` fp8/bf16 columns wide, which equals
        # ``32 * Float32.width / p_dtype.width`` packed fp32 TMEM columns
        # ``// (p_dtype.width / 8)``. Concretely, R=16 packs two bf16 PV K
        # segments per chunk (shape[2]=4 => 3/4 publish boundary aligns),
        # while fp8 (PV K=32 fp8 => 8 fp32 cols) needs R=8 so
        # shape[2]=4 and split_idx=3 publishes exactly 24 fp32 cols
        # (= 96 fp8 cols = 3 PV K segments) at the early-arrive edge.
        store_rep = const_expr(
            8 if self.p_dtype == cutlass.Float8E4M3FN else 16
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(store_rep)),
            Float32)
        thr_tmem_store = tcgen05.make_tmem_copy(
            tmem_store_atom, tStP).get_slice(group_tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)

        if const_expr(operation != ConsumerOperation.SOFTMAX_ONLY.value):
            total_q = mQ_2d.shape[0] // head_q
            thr0_pv = tiled_mma_pv.get_slice(0)
            pv_acc_shape = thr0_pv.partition_shape_C(self.mma_tiler_pv[:2])
            tOtO_base = thr0_pv.make_fragment_C(pv_acc_shape)
            corr_tile_size = 64
            tOcO = thr0_pv.partition_C(
                cute.make_identity_tensor(self.mma_tiler_pv[:2]))
            tOcO_i = cute.logical_divide(
                tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
            o_tmem_copy_atom = cute.make_copy_atom(
                tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
                self.pv_acc_dtype)
            if const_expr(self.use_o_tma):
                o_tmem_copy_atom = cute.make_copy_atom(
                    tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
                    self.pv_acc_dtype,
                )

        if has_work:
            kv_block_col_start = Int32(0)
            if const_expr(self.causal):
                kv_block_col_start = kv_block_idx * Int32(self.n_block_size)

            local_start = (Int32(stage) - group_base) & Int32(1)
            num_stage_groups = (
                num_q_groups + Int32(1) - local_start
            ) // Int32(2)
            if const_expr(operation == ConsumerOperation.SOFTMAX_ONLY.value):
                num_stage_groups = Int32(1)
            for qi_iter in cutlass.range(num_stage_groups, unroll=1):
                qi_group = (
                    single_group
                    if const_expr(operation == ConsumerOperation.SOFTMAX_ONLY.value)
                    else qi_iter * Int32(2) + local_start
                )
                phase = ((group_base + qi_group) // Int32(2)) & Int32(1)
                producer_phase = phase ^ Int32(1)
                qidx_meta_slot = (
                    (qi_group & Int32(self.qidx_meta_stages - 1))
                    * Int32(self.q_tokens_per_group)
                )
                if const_expr(operation != ConsumerOperation.EPILOGUE_ONLY.value):
                    softmax.reset()

                if const_expr(
                    operation != ConsumerOperation.EPILOGUE_ONLY.value
                    and self.causal
                ):
                    qi_group_start = qi_group * Int32(self.q_tokens_per_group)
                    masked_tok_count = cutlass.max(
                        Int32(0),
                        cutlass.min(
                            Int32(self.q_tokens_per_group),
                            diag_q_count - qi_group_start,
                        ),
                    )
                    self._softmax_step(
                        stage,
                        phase,
                        producer_phase,
                        producer_phase,
                        softmax,
                        sScale,
                        sScaleTemperature,
                        pipeline_s,
                        pipeline_p,
                        pipeline_p_lastsplit,
                        pipeline_sm_stats,
                        sm_stats_barrier,
                        stats_barrier_idx,
                        thr_tmem_load,
                        thr_tmem_store,
                        tStS_t2r,
                        tStP_r2t,
                        tScS_t2r,
                        tScP_shape,
                        sQIdxMeta,
                        qidx_meta_slot,
                        mK2qQSplitIndices,
                        qsplit_row_start,
                        head_kv_idx,
                        qi_group,
                        group_tidx,
                        masked_tok_count,
                        kv_block_col_start,
                        seq_len_q,
                        causal_q_offset,
                        kv_valid_cols,
                        lse_temperature_scale,
                        const_expr(mLSE_temperature_partial is not None),
                        True,
                        False,
                    )
                elif const_expr(
                    operation != ConsumerOperation.EPILOGUE_ONLY.value
                ):
                    self._softmax_step(
                        stage,
                        phase,
                        producer_phase,
                        producer_phase,
                        softmax,
                        sScale,
                        sScaleTemperature,
                        pipeline_s,
                        pipeline_p,
                        pipeline_p_lastsplit,
                        pipeline_sm_stats,
                        sm_stats_barrier,
                        stats_barrier_idx,
                        thr_tmem_load,
                        thr_tmem_store,
                        tStS_t2r,
                        tStP_r2t,
                        tScS_t2r,
                        tScP_shape,
                        sQIdxMeta,
                        qidx_meta_slot,
                        mK2qQSplitIndices,
                        qsplit_row_start,
                        head_kv_idx,
                        qi_group,
                        group_tidx,
                        Int32(0),
                        kv_block_col_start,
                        seq_len_q,
                        Int32(0),
                        kv_valid_cols,
                        lse_temperature_scale,
                        const_expr(mLSE_temperature_partial is not None),
                        False,
                        False,
                    )
                if const_expr(operation == ConsumerOperation.SOFTMAX_ONLY.value):
                    pipeline_sm_stats.producer_commit_w_index(Int32(stage))
                elif const_expr(operation == ConsumerOperation.EPILOGUE_ONLY.value):
                    pipeline_sm_stats.consumer_wait_w_index_phase(
                        Int32(stage), phase
                    )
                if const_expr(operation != ConsumerOperation.SOFTMAX_ONLY.value):
                    if const_expr(operation == ConsumerOperation.FUSED.value):
                        epilogue_barrier.arrive_and_wait_w_index(
                            index=Int32(stage)
                        )
                    self._epilogue_step(
                        qi_group,
                        group_tidx,
                        warp_idx_in_wg,
                        tOtO_base,
                        tOcO_i,
                        o_tmem_copy_atom,
                        sScale,
                        sScaleTemperature,
                        sSplitIdx,
                        sQIdx,
                        sQIdxMeta,
                        mK2qQSplitIndices,
                        qsplit_row_start,
                        pipeline_o,
                        pipeline_sm_stats,
                        sm_stats_barrier,
                        epilogue_barrier,
                        mO_partial,
                        mLSE_partial,
                        mLSE_temperature_partial,
                        softmax_scale_log2,
                        lse_temperature_scale_log2,
                        count_raw,
                        batch_idx,
                        head_kv_idx,
                        seq_len_q,
                        head_q,
                        num_heads_kv,
                        q_batch_offset,
                        total_q,
                        False,
                        stage,
                        group_base,
                        o_store_context,
                        work_tile_base,
                        pipeline_qidx_meta,
                    )

    @cute.jit
    def _store_o_partial_vec4(
        self,
        ptr: cute.Pointer,
        v0: Float32,
        v1: Float32,
        v2: Float32,
        v3: Float32,
    ):
        stg_128_cs(ptr, v0, v1, v2, v3)

    @cute.jit
    def _store_o_partial_vec8_half(
        self,
        ptr: cute.Pointer,
        v0: Float32,
        v1: Float32,
        v2: Float32,
        v3: Float32,
        v4: Float32,
        v5: Float32,
        v6: Float32,
        v7: Float32,
    ):
        if cutlass.const_expr(self.o_dtype is cutlass.BFloat16):
            stg_128_bf16_cs(ptr, v0, v1, v2, v3, v4, v5, v6, v7)
        else:
            stg_128_f16_cs(ptr, v0, v1, v2, v3, v4, v5, v6, v7)

    @cute.jit
    def _store_o_partial_vec16_fp8(
        self,
        ptr: cute.Pointer,
        v0: Float32,
        v1: Float32,
        v2: Float32,
        v3: Float32,
        v4: Float32,
        v5: Float32,
        v6: Float32,
        v7: Float32,
        v8: Float32,
        v9: Float32,
        v10: Float32,
        v11: Float32,
        v12: Float32,
        v13: Float32,
        v14: Float32,
        v15: Float32,
    ):
        stg_128_fp8_e4m3_cs(
            ptr,
            v0,
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
            v7,
            v8,
            v9,
            v10,
            v11,
            v12,
            v13,
            v14,
            v15,
        )

    @cute.jit
    def _epilogue_step(
        self,
        qi_group: Int32,
        group_tidx: Int32,
        warp_idx_in_wg: Int32,
        tOtO_base: cute.Tensor,
        tOcO_i: cute.Tensor,
        o_tmem_copy_atom,
        sScale: cute.Tensor,
        sScaleTemperature: cute.Tensor,
        sSplitIdx: cute.Tensor,
        sQIdx: cute.Tensor,
        sQIdxMeta: cute.Tensor,
        mK2qQSplitIndices: cute.Tensor,
        qsplit_row_start: Int32,
        pipeline_o,
        pipeline_sm_stats,
        sm_stats_barrier,
        epilogue_barrier,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mLSE_temperature_partial: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        lse_temperature_scale_log2: Float32,
        count_raw: Int32,
        batch_idx: Int32,
        head_kv_idx: Int32,
        seq_len_q: Int32,
        head_q: Int32,
        num_heads_kv: Int32,
        q_batch_offset: Int32,
        total_q: Int32,
        use_stats_barrier: cutlass.Constexpr[bool],
        softmax_stage: cutlass.Constexpr[int],
        group_base: Int32 = 0,
        o_store_context=None,
        work_tile_base: Int32 = 0,
        pipeline_qidx_meta=None,
    ):
        use_qidx_meta: cutlass.Constexpr[bool] = (
            self.use_qidx_meta_pipeline and pipeline_qidx_meta is not None
        )
        slot = (group_base + qi_group) & Int32(1)
        phase = ((group_base + qi_group) // Int32(2)) & Int32(1)
        stage_base = slot * Int32(self.tmem_o_stage_stride)
        corr_tile_size = 64
        sScale_slot = cute.make_tensor(
            sScale.iterator + slot * Int32(self.m_block_size * 2),
            cute.make_layout(self.m_block_size * 2),
        )
        sScale_temperature_slot = cute.make_tensor(
            sScaleTemperature.iterator + slot * Int32(self.m_block_size),
            cute.make_layout(self.m_block_size),
        )
        sSplitIdx_slot = cute.make_tensor(
            sSplitIdx.iterator + slot * Int32(self.q_tokens_per_group),
            cute.make_layout((self.q_tokens_per_group,)),
        )
        sQIdx_slot = cute.make_tensor(
            sQIdx.iterator + slot * Int32(self.q_tokens_per_group),
            cute.make_layout((self.q_tokens_per_group,)),
        )
        qidx_meta_slot = (qi_group
                          & Int32(self.qidx_meta_stages - 1)) * Int32(
                              self.q_tokens_per_group)
        if const_expr(not use_qidx_meta):
            pipeline_o.consumer_wait_w_index_phase(slot, phase)
        if const_expr(self.use_o_tma):
            sOStore, pipeline_o_store = o_store_context
            pipeline_o_store.producer_acquire_w_index_phase(
                slot,
                phase ^ Int32(1),
            )
        if const_expr(use_stats_barrier):
            sm_stats_barrier.arrive_and_wait_w_index(
                index=slot * Int32(self.warps_per_group) + warp_idx_in_wg)

        if const_expr(use_qidx_meta):
            route_tmem = cute.make_tensor(
                tOtO_base.iterator + stage_base + Int32(self.tmem_o_offset),
                tOtO_base.layout)
            route_tmem = cute.logical_divide(
                route_tmem, cute.make_layout((self.m_block_size, corr_tile_size)))
            route_tmem_copy = tcgen05.make_tmem_copy(
                o_tmem_copy_atom, route_tmem[(None, None), 0])
            route_coords = route_tmem_copy.get_slice(group_tidx).partition_D(
                tOcO_i[(None, None), None])[None, None, None, 0]
            route_rows = make_16x256b_tensor_mn_view(route_coords)
            route_rows = cute.make_tensor(
                route_rows.iterator, cute.select(route_rows.layout, mode=[0, 1]))
            # Store vectors can address different rows within the same lane.
            # The LSE row is independent of the vector-store fragment.
            row_sums = cute.make_rmem_tensor(
                (cute.size(route_rows, mode=[0]), cute.size(route_rows, mode=[1]) // 8),
                Float32)
            for stats_row in cutlass.range_constexpr(cute.size(route_rows, mode=[0])):
                for stats_col in cutlass.range_constexpr(cute.size(route_rows, mode=[1]) // 8):
                    stats_store_row = route_rows[stats_row, stats_col * 8][0]
                    stats_store_tok = stats_store_row // Int32(self.qheadperkv)
                    stats_safe_tok = cutlass.min(
                        stats_store_tok, count_raw - qi_group * Int32(self.q_tokens_per_group) - Int32(1))
                    stats_safe_row = stats_safe_tok * Int32(self.qheadperkv) + stats_store_row % Int32(self.qheadperkv)
                    row_sums[stats_row, stats_col] = sScale_slot[stats_safe_row]
            stats_lse_tok = cutlass.min(
                group_tidx // Int32(self.qheadperkv),
                count_raw - qi_group * Int32(self.q_tokens_per_group) - Int32(1))
            stats_lse_row = (
                stats_lse_tok * Int32(self.qheadperkv)
                + group_tidx % Int32(self.qheadperkv))
            lse_row_sum = sScale_slot[stats_lse_row]
            lse_row_max = sScale_slot[stats_lse_row + Int32(self.m_block_size)]
            if const_expr(mLSE_temperature_partial is not None):
                lse_temperature_row_sum = sScale_temperature_slot[stats_lse_row]
            # The full consumer group releases only after every shared read.
            # Normalization and LSE arithmetic use the register copies below.
            pipeline_sm_stats.consumer_release_w_index(slot)

        if const_expr(use_qidx_meta):
            meta_group = group_base + qi_group
            meta_slot = meta_group % Int32(self.qidx_meta_stages)
            meta_phase = (meta_group // Int32(self.qidx_meta_stages)) & Int32(1)
            pipeline_qidx_meta.consumer_wait_w_index_phase(meta_slot, meta_phase)
            meta_base = meta_slot * Int32(self.q_tokens_per_group)
            # A lane's 16x256b fragment may map successive eight-value
            # vectors to different rows. Cache one route per store vector.
            route_q = cute.make_rmem_tensor(
                (cute.size(route_rows, mode=[0]), cute.size(route_rows, mode=[1]) // 8),
                Int32)
            route_split = cute.make_rmem_tensor_like(route_q)
            for meta_row in cutlass.range_constexpr(cute.size(route_rows, mode=[0])):
                for meta_col in cutlass.range_constexpr(cute.size(route_rows, mode=[1]) // 8):
                    meta_tok = route_rows[meta_row, meta_col * 8][0] // Int32(self.qheadperkv)
                    meta_safe_tok = cutlass.min(
                        meta_tok, count_raw - qi_group * Int32(self.q_tokens_per_group) - Int32(1))
                    packed = sQIdxMeta[meta_base + meta_safe_tok]
                    route_q[meta_row, meta_col] = self._decode_q_idx_from_qsplit(packed)
                    route_split[meta_row, meta_col] = self._decode_split_idx_from_qsplit(packed)
            lse_tok = cutlass.min(
                group_tidx // Int32(self.qheadperkv),
                count_raw - qi_group * Int32(self.q_tokens_per_group) - Int32(1))
            packed_lse = sQIdxMeta[meta_base + lse_tok]
            route_q_lse = self._decode_q_idx_from_qsplit(packed_lse)
            route_split_lse = self._decode_split_idx_from_qsplit(packed_lse)
            # Every epilogue thread has copied and decoded its rows. The slot
            # is free even while O normalization and global stores continue.
            pipeline_qidx_meta.consumer_release_w_index(meta_slot)
        if (
            not const_expr(use_qidx_meta)
            and not const_expr(self.shuffled_output)
            and group_tidx < Int32(self.q_tokens_per_group)
        ):
            tok = group_tidx
            qi = qi_group * Int32(self.q_tokens_per_group) + tok
            if qi < count_raw:
                if const_expr(self.safe_fwd_sync):
                    qsplit = self._load_qsplit_idx(
                        mK2qQSplitIndices,
                        head_kv_idx,
                        qsplit_row_start,
                        qi,
                    )
                else:
                    qsplit = sQIdxMeta[qidx_meta_slot + tok]
                q_idx = self._decode_q_idx_from_qsplit(qsplit)
                sQIdx_slot[tok] = q_idx
                sSplitIdx_slot[tok] = self._decode_split_idx_from_qsplit(qsplit)
        if const_expr(not use_qidx_meta):
            epilogue_barrier.arrive_and_wait_w_index(index=Int32(softmax_stage))
        if const_expr(self.shuffled_output):
            tma_row = group_tidx
            shuffled_row = (
                (work_tile_base + qi_group) * Int32(self.m_block_size)
                + tma_row
            )
            tile_row_sum = sScale_slot[tma_row]
            tile_valid_sum = (
                tile_row_sum != Float32(0.0)
                and tile_row_sum == tile_row_sum
            )
            tile_scale = cute.arch.rcp_approx(
                tile_row_sum if tile_valid_sum else Float32(1.0)
            )

        if const_expr(use_qidx_meta):
            pipeline_o.consumer_wait_w_index_phase(slot, phase)

        tOtO = cute.make_tensor(
            tOtO_base.iterator + stage_base + Int32(self.tmem_o_offset),
            tOtO_base.layout)
        for col_pass_idx in cutlass.range(Int32(2), unroll=1):
            col_pass = col_pass_idx * Int32(corr_tile_size)
            tOtO_pass_ptr = cute.make_ptr(
                self.pv_acc_dtype,
                tOtO.iterator.toint() + col_pass,
                cute.AddressSpace.tmem,
                assumed_align=8,
            )
            tOtO_pass = cute.make_tensor(tOtO_pass_ptr, tOtO.layout)
            tOtO_pass_i = cute.logical_divide(
                tOtO_pass,
                cute.make_layout((self.m_block_size, corr_tile_size)))
            tiled_tmem_load_pass = tcgen05.make_tmem_copy(
                o_tmem_copy_atom, tOtO_pass_i[(None, None), 0])
            thr_tmem_load_pass = tiled_tmem_load_pass.get_slice(group_tidx)
            tOtO_t2r_pass = thr_tmem_load_pass.partition_S(
                tOtO_pass_i[(None, None), None])
            tOcO_t2r_pass = thr_tmem_load_pass.partition_D(
                tOcO_i[(None, None), None])

            tOtO_t2r_i = tOtO_t2r_pass[None, None, None, 0]
            tOcO_t2r_i = tOcO_t2r_pass[None, None, None, 0]
            tOrO_frg = cute.make_rmem_tensor_like(
                tOcO_t2r_i, self.pv_acc_dtype)
            cute.copy(tiled_tmem_load_pass, tOtO_t2r_i, tOrO_frg)
            if const_expr(use_qidx_meta):
                if col_pass_idx == Int32(1):
                    cute.arch.fence_view_async_tmem_load()
                    pipeline_o.consumer_release_w_index(slot)

            if const_expr(self.use_o_tma):
                tOrO_mn = cute.make_tensor(
                    tOrO_frg.iterator,
                    cute.composition(
                        tOrO_frg.layout,
                        cute.make_layout((1, corr_tile_size)),
                    ),
                )
            else:
                tOrO_mn = make_16x256b_tensor_mn_view(tOrO_frg)
            tOrO_mn = cute.make_tensor(
                tOrO_mn.iterator,
                cute.select(tOrO_mn.layout, mode=[0, 1]))
            if const_expr(self.use_o_tma):
                tOcO_mn = cute.make_tensor(
                    tOcO_t2r_i.iterator,
                    cute.composition(
                        tOcO_t2r_i.layout,
                        cute.make_layout((1, corr_tile_size)),
                    ),
                )
            else:
                tOcO_mn = make_16x256b_tensor_mn_view(tOcO_t2r_i)
            tOcO_mn = cute.make_tensor(
                tOcO_mn.iterator,
                cute.select(tOcO_mn.layout, mode=[0, 1]))
            num_rows = cute.size(tOrO_mn, mode=[0])
            num_cols = cute.size(tOrO_mn, mode=[1])

            for r in cutlass.range_constexpr(num_rows):
                if const_expr(self.use_o_tma):
                    # Keep normalization explicitly paired because the vector
                    # expression scalarizes to FMUL with this fragment layout.
                    tma_scale_pair = (tile_scale, tile_scale)
                    for c8 in cutlass.range_constexpr(num_cols // 8):
                        c_base = c8 * 8
                        tma_o0 = tOrO_mn[r, c_base]
                        tma_o1 = tOrO_mn[r, c_base + 1]
                        tma_o2 = tOrO_mn[r, c_base + 2]
                        tma_o3 = tOrO_mn[r, c_base + 3]
                        tma_o4 = tOrO_mn[r, c_base + 4]
                        tma_o5 = tOrO_mn[r, c_base + 5]
                        tma_o6 = tOrO_mn[r, c_base + 6]
                        tma_o7 = tOrO_mn[r, c_base + 7]
                        tma_o0, tma_o1 = cute.arch.mul_packed_f32x2(
                            (tma_o0, tma_o1), tma_scale_pair)
                        tma_o2, tma_o3 = cute.arch.mul_packed_f32x2(
                            (tma_o2, tma_o3), tma_scale_pair)
                        tma_o4, tma_o5 = cute.arch.mul_packed_f32x2(
                            (tma_o4, tma_o5), tma_scale_pair)
                        tma_o6, tma_o7 = cute.arch.mul_packed_f32x2(
                            (tma_o6, tma_o7), tma_scale_pair)
                        out = cute.make_rmem_tensor(
                            (8,), cutlass.BFloat16
                        )
                        out[0] = cutlass.BFloat16(tma_o0)
                        out[1] = cutlass.BFloat16(tma_o1)
                        out[2] = cutlass.BFloat16(tma_o2)
                        out[3] = cutlass.BFloat16(tma_o3)
                        out[4] = cutlass.BFloat16(tma_o4)
                        out[5] = cutlass.BFloat16(tma_o5)
                        out[6] = cutlass.BFloat16(tma_o6)
                        out[7] = cutlass.BFloat16(tma_o7)
                        column = c_base + col_pass
                        offset = cute.crd2idx(
                            (tma_row, column, slot),
                            sOStore.layout,
                        )
                        # Applying the composed SW128 layout directly to this
                        # register vector makes the DSL scalarize the store.
                        # Resolve its physical address once so the copy stays
                        # one aligned 128-bit STS while the TMA descriptor sees
                        # the matching CuTe SW128 layout.
                        byte_offset = offset * 2
                        physical = byte_offset ^ (
                            (byte_offset & 896) >> 3
                        )
                        smem_o_ptr = cute.make_ptr(
                            cutlass.BFloat16,
                            sOStore.iterator.toint() + Int64(physical),
                            mem_space=sOStore.iterator.memspace,
                            assumed_align=16,
                        )
                        dst = cute.make_tensor(smem_o_ptr, cute.make_layout(8))
                        cute.autovec_copy(out, dst)
                elif const_expr(self.shuffled_output):
                    assert self.o_dtype is cutlass.BFloat16
                    assert num_cols % 8 == 0
                    for c8 in cutlass.range_constexpr(num_cols // 8):
                        c_base = Int32(c8) * Int32(8)
                        row = tOcO_mn[r, c_base][0]
                        row_sum_val = sScale_slot[row]
                        row_valid_sum = (
                            row_sum_val != Float32(0.0)
                            and row_sum_val == row_sum_val
                        )
                        row_scale = cute.arch.rcp_approx(
                            row_sum_val
                            if row_valid_sum
                            else Float32(1.0)
                        )
                        o0 = tOrO_mn[r, c_base]
                        o1 = tOrO_mn[r, c_base + Int32(1)]
                        o2 = tOrO_mn[r, c_base + Int32(2)]
                        o3 = tOrO_mn[r, c_base + Int32(3)]
                        o4 = tOrO_mn[r, c_base + Int32(4)]
                        o5 = tOrO_mn[r, c_base + Int32(5)]
                        o6 = tOrO_mn[r, c_base + Int32(6)]
                        o7 = tOrO_mn[r, c_base + Int32(7)]
                        scale_pair = (row_scale, row_scale)
                        o0, o1 = cute.arch.mul_packed_f32x2(
                            (o0, o1), scale_pair
                        )
                        o2, o3 = cute.arch.mul_packed_f32x2(
                            (o2, o3), scale_pair
                        )
                        o4, o5 = cute.arch.mul_packed_f32x2(
                            (o4, o5), scale_pair
                        )
                        o6, o7 = cute.arch.mul_packed_f32x2(
                            (o6, o7), scale_pair
                        )
                        out = cute.make_rmem_tensor((8,), cutlass.BFloat16)
                        out[0] = cutlass.BFloat16(o0)
                        out[1] = cutlass.BFloat16(o1)
                        out[2] = cutlass.BFloat16(o2)
                        out[3] = cutlass.BFloat16(o3)
                        out[4] = cutlass.BFloat16(o4)
                        out[5] = cutlass.BFloat16(o5)
                        out[6] = cutlass.BFloat16(o6)
                        out[7] = cutlass.BFloat16(o7)
                        out = _transpose_quad_bf16_pairs(out)
                        shuffled_store_row = (
                            (work_tile_base + qi_group)
                            * Int32(self.m_block_size)
                            + row
                        )
                        row_ptr = (
                            mO_partial.iterator
                            + Int64(shuffled_store_row)
                            * Int64(self.head_dim)
                        )
                        real_col = tOcO_mn[r, c_base][1] + col_pass
                        store_col = real_col_to_stg128_half_fake_col(real_col)
                        ptr = row_ptr + Int64(store_col)
                        target = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.BFloat16,
                                ptr.toint(),
                                mem_space=ptr.memspace,
                                assumed_align=16,
                            ),
                            cute.make_layout((8,)),
                        )
                        atom = cute.make_copy_atom(
                            cute.nvgpu.CopyR2GOp(),
                            cutlass.BFloat16,
                            num_bits_per_copy=128,
                            l1c_evict_priority=(
                                cute.nvgpu.CacheEvictionPriority.EVICT_FIRST
                            ),
                        )
                        cute.copy(atom, out, target)
                elif const_expr(self.o_dtype is Float32):
                    for c4 in cutlass.range_constexpr(num_cols // 4):
                        c_base = Int32(c4) * Int32(4)
                        row_col = tOcO_mn[r, c_base]
                        row = row_col[0]
                        col = row_col[1] + col_pass
                        if row < Int32(self.m_block_size):
                            tok = row // Int32(self.qheadperkv)
                            row_in_tok = row - tok * Int32(self.qheadperkv)
                            qi = (qi_group
                                  * Int32(self.q_tokens_per_group)
                                  + tok)
                            if qi < count_raw:
                                q_idx = sQIdx_slot[tok]
                                split = sSplitIdx_slot[tok]
                                q_abs = q_batch_offset + q_idx
                                flat_row = (
                                    Int64(split)
                                    * Int64(total_q) * Int64(head_q)
                                    + Int64(q_abs) * Int64(head_q)
                                    + Int64(head_kv_idx)
                                    * Int64(self.qheadperkv)
                                    + Int64(row_in_tok)
                                )
                                row_sum_val = sScale_slot[row]
                                is_zero_or_nan = (
                                    row_sum_val == Float32(0.0)
                                    or row_sum_val != row_sum_val)
                                row_scale = cute.arch.rcp_approx(
                                    row_sum_val
                                    if not is_zero_or_nan
                                    else Float32(1.0))
                                row_base_ptr = (
                                    flat_row * Int64(self.head_dim))
                                o0 = tOrO_mn[r, c_base]
                                o1 = tOrO_mn[r, c_base + Int32(1)]
                                o2 = tOrO_mn[r, c_base + Int32(2)]
                                o3 = tOrO_mn[r, c_base + Int32(3)]
                                scale_pair = (row_scale, row_scale)
                                o0, o1 = cute.arch.mul_packed_f32x2(
                                    (o0, o1), scale_pair)
                                o2, o3 = cute.arch.mul_packed_f32x2(
                                    (o2, o3), scale_pair)
                                fake_col = real_col_to_stg128_fake_col(col)
                                ptr = (mO_partial.iterator + row_base_ptr
                                       + Int64(fake_col))
                                self._store_o_partial_vec4(
                                    ptr,
                                    o0,
                                    o1,
                                    o2,
                                    o3,
                                )
                elif const_expr(self.o_dtype in [cutlass.BFloat16, cutlass.Float16]):
                    assert num_cols % 8 == 0, (
                        "half O_partial STG.128 requires the epilogue "
                        "TMEM fragment column count to be a multiple of 8"
                    )
                    for c8 in cutlass.range_constexpr(num_cols // 8):
                        c_base = Int32(c8) * Int32(8)
                        row_col = tOcO_mn[r, c_base]
                        row = row_col[0]
                        col = row_col[1] + col_pass
                        tok = row // Int32(self.qheadperkv)
                        row_in_tok = row - tok * Int32(self.qheadperkv)
                        qi = qi_group * Int32(self.q_tokens_per_group) + tok
                        row_valid = (
                            row < Int32(self.m_block_size) and qi < count_raw
                        )
                        safe_qi = cutlass.min(qi, count_raw - Int32(1))
                        safe_tok = (
                            safe_qi
                            - qi_group * Int32(self.q_tokens_per_group)
                        )
                        if const_expr(use_qidx_meta):
                            q_idx = route_q[r, c8]
                            split = route_split[r, c8]
                        else:
                            q_idx = sQIdx_slot[safe_tok]
                            split = sSplitIdx_slot[safe_tok]
                        q_abs = q_batch_offset + q_idx
                        flat_row = (
                            Int64(split) * Int64(total_q) * Int64(head_q)
                            + Int64(q_abs) * Int64(head_q)
                            + Int64(head_kv_idx) * Int64(self.qheadperkv)
                            + Int64(row_in_tok)
                        )
                        safe_stats_row = (
                            safe_tok * Int32(self.qheadperkv) + row_in_tok
                        )
                        row_sum_val = (
                            row_sums[r, c8] if const_expr(use_qidx_meta)
                            else sScale_slot[safe_stats_row])
                        is_zero_or_nan = (
                            row_sum_val == Float32(0.0)
                            or row_sum_val != row_sum_val
                        )
                        row_scale = cute.arch.rcp_approx(
                            row_sum_val
                            if not is_zero_or_nan
                            else Float32(1.0)
                        )
                        row_base_ptr = flat_row * Int64(self.head_dim)
                        o0 = tOrO_mn[r, c_base]
                        o1 = tOrO_mn[r, c_base + Int32(1)]
                        o2 = tOrO_mn[r, c_base + Int32(2)]
                        o3 = tOrO_mn[r, c_base + Int32(3)]
                        o4 = tOrO_mn[r, c_base + Int32(4)]
                        o5 = tOrO_mn[r, c_base + Int32(5)]
                        o6 = tOrO_mn[r, c_base + Int32(6)]
                        o7 = tOrO_mn[r, c_base + Int32(7)]
                        scale_pair = (row_scale, row_scale)
                        o0, o1 = cute.arch.mul_packed_f32x2(
                            (o0, o1), scale_pair
                        )
                        o2, o3 = cute.arch.mul_packed_f32x2(
                            (o2, o3), scale_pair
                        )
                        o4, o5 = cute.arch.mul_packed_f32x2(
                            (o4, o5), scale_pair
                        )
                        o6, o7 = cute.arch.mul_packed_f32x2(
                            (o6, o7), scale_pair
                        )
                        fake_col = real_col_to_stg128_half_fake_col(col)
                        ptr = (
                            mO_partial.iterator
                            + row_base_ptr
                            + Int64(fake_col)
                        )
                        if const_expr(self.o_dtype is cutlass.BFloat16):
                            _store_bf16x8_predicated(
                                ptr,
                                row_valid,
                                o0,
                                o1,
                                o2,
                                o3,
                                o4,
                                o5,
                                o6,
                                o7,
                            )
                        else:
                            if row_valid:
                                self._store_o_partial_vec8_half(
                                    ptr,
                                    o0,
                                    o1,
                                    o2,
                                    o3,
                                    o4,
                                    o5,
                                    o6,
                                    o7,
                                )
                else:
                    assert num_cols % 16 == 0, (
                        "fp8 O_partial STG.128 requires the epilogue "
                        "TMEM fragment column count to be a multiple of 16"
                    )
                    for c16 in cutlass.range_constexpr(num_cols // 16):
                        c_base_fp8 = Int32(c16) * Int32(16)
                        row_col_fp8 = tOcO_mn[r, c_base_fp8]
                        row_fp8 = row_col_fp8[0]
                        col_fp8 = row_col_fp8[1] + col_pass
                        if row_fp8 < Int32(self.m_block_size):
                            tok_fp8 = row_fp8 // Int32(self.qheadperkv)
                            row_in_tok_fp8 = (
                                row_fp8 - tok_fp8 * Int32(self.qheadperkv)
                            )
                            qi_fp8 = (
                                qi_group * Int32(self.q_tokens_per_group)
                                + tok_fp8
                            )
                            if qi_fp8 < count_raw:
                                q_idx_fp8 = sQIdx_slot[tok_fp8]
                                split_fp8 = sSplitIdx_slot[tok_fp8]
                                q_abs_fp8 = q_batch_offset + q_idx_fp8
                                flat_row_fp8 = (
                                    Int64(split_fp8)
                                    * Int64(total_q)
                                    * Int64(head_q)
                                    + Int64(q_abs_fp8) * Int64(head_q)
                                    + Int64(head_kv_idx)
                                    * Int64(self.qheadperkv)
                                    + Int64(row_in_tok_fp8)
                                )
                                row_sum_fp8 = sScale_slot[row_fp8]
                                zero_or_nan_fp8 = (
                                    row_sum_fp8 == Float32(0.0)
                                    or row_sum_fp8 != row_sum_fp8
                                )
                                row_scale_fp8 = cute.arch.rcp_approx(
                                    row_sum_fp8
                                    if not zero_or_nan_fp8
                                    else Float32(1.0)
                                )
                                row_base_fp8 = (
                                    flat_row_fp8 * Int64(self.head_dim)
                                )
                                o0_fp8 = tOrO_mn[r, c_base_fp8]
                                o1_fp8 = tOrO_mn[r, c_base_fp8 + Int32(1)]
                                o2_fp8 = tOrO_mn[r, c_base_fp8 + Int32(2)]
                                o3_fp8 = tOrO_mn[r, c_base_fp8 + Int32(3)]
                                o4_fp8 = tOrO_mn[r, c_base_fp8 + Int32(4)]
                                o5_fp8 = tOrO_mn[r, c_base_fp8 + Int32(5)]
                                o6_fp8 = tOrO_mn[r, c_base_fp8 + Int32(6)]
                                o7_fp8 = tOrO_mn[r, c_base_fp8 + Int32(7)]
                                o8_fp8 = tOrO_mn[r, c_base_fp8 + Int32(8)]
                                o9_fp8 = tOrO_mn[r, c_base_fp8 + Int32(9)]
                                o10_fp8 = tOrO_mn[r, c_base_fp8 + Int32(10)]
                                o11_fp8 = tOrO_mn[r, c_base_fp8 + Int32(11)]
                                o12_fp8 = tOrO_mn[r, c_base_fp8 + Int32(12)]
                                o13_fp8 = tOrO_mn[r, c_base_fp8 + Int32(13)]
                                o14_fp8 = tOrO_mn[r, c_base_fp8 + Int32(14)]
                                o15_fp8 = tOrO_mn[r, c_base_fp8 + Int32(15)]
                                scale_pair_fp8 = (row_scale_fp8, row_scale_fp8)
                                o0_fp8, o1_fp8 = cute.arch.mul_packed_f32x2(
                                    (o0_fp8, o1_fp8), scale_pair_fp8
                                )
                                o2_fp8, o3_fp8 = cute.arch.mul_packed_f32x2(
                                    (o2_fp8, o3_fp8), scale_pair_fp8
                                )
                                o4_fp8, o5_fp8 = cute.arch.mul_packed_f32x2(
                                    (o4_fp8, o5_fp8), scale_pair_fp8
                                )
                                o6_fp8, o7_fp8 = cute.arch.mul_packed_f32x2(
                                    (o6_fp8, o7_fp8), scale_pair_fp8
                                )
                                o8_fp8, o9_fp8 = cute.arch.mul_packed_f32x2(
                                    (o8_fp8, o9_fp8), scale_pair_fp8
                                )
                                o10_fp8, o11_fp8 = cute.arch.mul_packed_f32x2(
                                    (o10_fp8, o11_fp8), scale_pair_fp8
                                )
                                o12_fp8, o13_fp8 = cute.arch.mul_packed_f32x2(
                                    (o12_fp8, o13_fp8), scale_pair_fp8
                                )
                                o14_fp8, o15_fp8 = cute.arch.mul_packed_f32x2(
                                    (o14_fp8, o15_fp8), scale_pair_fp8
                                )
                                fake_col_fp8 = real_col_to_stg128_fp8_fake_col(
                                    col_fp8
                                )
                                ptr_fp8 = (
                                    mO_partial.iterator
                                    + row_base_fp8
                                    + Int64(fake_col_fp8)
                                )
                                self._store_o_partial_vec16_fp8(
                                    ptr_fp8,
                                    o0_fp8,
                                    o1_fp8,
                                    o2_fp8,
                                    o3_fp8,
                                    o4_fp8,
                                    o5_fp8,
                                    o6_fp8,
                                    o7_fp8,
                                    o8_fp8,
                                    o9_fp8,
                                    o10_fp8,
                                    o11_fp8,
                                    o12_fp8,
                                    o13_fp8,
                                    o14_fp8,
                                    o15_fp8,
                                )
        if const_expr(not use_qidx_meta):
            cute.arch.fence_view_async_tmem_load()

        tok_local = Int32(group_tidx) // Int32(self.qheadperkv)
        h_local = Int32(group_tidx) % Int32(self.qheadperkv)
        qi_lse = qi_group * Int32(self.q_tokens_per_group) + tok_local
        if const_expr(self.shuffled_output):
            tile_row_max = sScale_slot[
                group_tidx + Int32(self.m_block_size)
            ]
            tile_lse = (
                (
                    tile_row_max * softmax_scale_log2
                    + cute.math.log2(tile_row_sum, fastmath=True)
                )
                * Float32(math.log(2.0))
                if tile_valid_sum
                else -Float32.inf
            )
            mLSE_partial[shuffled_row] = tile_lse
        else:
            row_valid_lse = qi_lse < count_raw
            safe_qi_lse = cutlass.min(qi_lse, count_raw - Int32(1))
            safe_tok_lse = (
                safe_qi_lse - qi_group * Int32(self.q_tokens_per_group)
            )
            safe_row_lse = safe_tok_lse * Int32(self.qheadperkv) + h_local
            row_sum_val = (
                lse_row_sum if const_expr(use_qidx_meta)
                else sScale_slot[safe_row_lse])
            row_max_val = (
                lse_row_max if const_expr(use_qidx_meta)
                else sScale_slot[safe_row_lse + Int32(self.m_block_size)])
            is_zero_or_nan = (
                row_sum_val == Float32(0.0) or row_sum_val != row_sum_val
            )
            LN2 = Float32(math.log(2.0))
            lse_cur = (
                (
                    row_max_val * softmax_scale_log2
                    + cute.math.log2(row_sum_val, fastmath=True)
                )
                * LN2
                if not is_zero_or_nan
                else -Float32.inf
            )
            q_idx_lse = (
                route_q_lse if const_expr(use_qidx_meta)
                else sQIdx_slot[safe_tok_lse])
            h_abs = head_kv_idx * Int32(self.qheadperkv) + h_local
            split_lse = (
                route_split_lse if const_expr(use_qidx_meta)
                else sSplitIdx_slot[safe_tok_lse])
            q_abs_lse = q_batch_offset + q_idx_lse
            lse_offset = cute.crd2idx(
                (split_lse, q_abs_lse, h_abs), mLSE_partial.layout
            )
            _store_f32_predicated(
                mLSE_partial.iterator + lse_offset,
                row_valid_lse,
                lse_cur,
            )
            if const_expr(mLSE_temperature_partial is not None):
                row_sum_temperature_val = (
                    lse_temperature_row_sum if const_expr(use_qidx_meta)
                    else sScale_temperature_slot[safe_row_lse])
                is_temperature_zero_or_nan = (
                    row_sum_temperature_val == Float32(0.0)
                    or row_sum_temperature_val != row_sum_temperature_val
                )
                lse_temperature_cur = (
                    (
                        row_max_val * lse_temperature_scale_log2
                        + cute.math.log2(row_sum_temperature_val, fastmath=True)
                    )
                    * LN2
                    if not is_temperature_zero_or_nan
                    else -Float32.inf
                )
                lse_temperature_offset = cute.crd2idx(
                    (split_lse, q_abs_lse, h_abs),
                    mLSE_temperature_partial.layout,
                )
                _store_f32_predicated(
                    mLSE_temperature_partial.iterator + lse_temperature_offset,
                    row_valid_lse,
                    lse_temperature_cur,
                )
        epilogue_barrier.arrive_and_wait_w_index(index=Int32(softmax_stage))

        if const_expr(not use_qidx_meta):
            pipeline_sm_stats.consumer_release_w_index(slot)
            pipeline_o.consumer_release_w_index(slot)
        if const_expr(self.use_o_tma):
            cute.arch.fence_proxy("async.shared", space="cta")
            pipeline_o_store.producer_commit_w_index(slot)
