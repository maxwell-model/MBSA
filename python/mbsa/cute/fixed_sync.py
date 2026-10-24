# Copyright (c) 2022, the respective FlashAttention contributors, as shown by AUTHORS.
# Copyright 2026 Maxwell Team (synchronization adaptations).
# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
# ruff: noqa: F841
# Retain public inactive expressions and the collective TMEM retrieve.
# MBSA synchronization adaptation of the official flash-attn-4==4.0.0b20 wheel.
# See ../third_party/fa4/PROVENANCE.md, UPSTREAM.json, LICENSE and AUTHORS.
"""Public FA4 kernel override with converged TMEM lifecycle.

Only the installed public SM100 kernel method is derived here. The installed
base owns every other method. The preparation-only
source guard rejects another package version or a modified installed base.
"""
from functools import lru_cache, partial
import hashlib
from importlib.metadata import version
import inspect
from pathlib import Path
from types import FunctionType
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr, pipeline
from cutlass.cute.nvgpu import cpasync
import cutlass.pipeline as cutlass_pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import ClcDynamicPersistentTileScheduler
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100, DescaleTensors
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.named_barrier import NamedBarrierFwdSm100
import flash_attn.cute.pipeline as pipeline_custom
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.tile_scheduler import ClcState, TileSchedulerProtocol
from flash_attn.cute.utils import AuxData
from flash_attn.cute.softmax import SoftmaxSm100
from quack.cute_dsl_utils import ParamsBase


PUBLIC_FA4_VERSION = '4.0.0b20'
PUBLIC_FA4_SM100_SHA256 = '7a5a8f434e3efed7546a2da2d591a13f3644766c9e3abb3aafbb9e82dc4f0eda'


@lru_cache(maxsize=1)
def validate_public_source():
    """Check pinned public preparation inputs before any kernel is launched."""
    installed = version('flash-attn-4')
    if installed != PUBLIC_FA4_VERSION:
        raise RuntimeError(f'Fixed FA4 requires flash-attn-4=={PUBLIC_FA4_VERSION}; found {installed}')
    source_path = inspect.getsourcefile(FlashAttentionForwardSm100)
    if source_path is None:
        raise RuntimeError('Fixed FA4 requires the official b20 SM100 Python source')
    actual = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    if actual != PUBLIC_FA4_SM100_SHA256:
        raise RuntimeError('Fixed FA4 SM100 source differs from the pinned official b20 wheel')
    from flash_attn.cute.interface import _flash_attn_fwd
    if _flash_attn_fwd.__globals__.get('FlashAttentionForwardSm100') is not FlashAttentionForwardSm100:
        raise RuntimeError('Fixed FA4 public SM100 class binding has been replaced')
    return True


class SynchronizedFlashAttentionForwardSm100(FlashAttentionForwardSm100):
    """Public base plus a single overridden kernel method; role loops unchanged."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if ((self.m_block_size, self.n_block_size) != (128, 128) or
                self.q_stage not in (1, 2) or
                self.head_dim_padded != 128 or self.head_dim_v_padded != 128 or
                self.check_hdim_oob or self.check_hdim_v_oob or
                self.use_2cta_instrs or self.use_clc_scheduler or
                self.use_correction_warps_for_epi or self.is_split_kv or
                self.is_varlen_q or self.is_causal or self.is_local or
                not self.use_tma_KV or not self.use_tma_Q):
            raise ValueError('Synchronized Fixed FA4 supports D128, tile128, single-CTA '
                             'non-CLC block-sparse prefill with an independent epilogue')
        participants = (self.mma_warp_id, *self.softmax0_warp_ids,
                        *self.softmax1_warp_ids, *self.correction_warp_ids)
        if len(set(participants)) != len(participants):
            raise RuntimeError('Public FA4 TMEM roles overlap')
        inactive = (*self.load_warp_ids, *self.empty_warp_ids)
        if not self.use_correction_warps_for_epi:
            inactive += self.epilogue_warp_ids
        if set(participants) & set(inactive):
            raise RuntimeError('Public FA4 TMEM participants include an inactive role')
        self.tmem_participant_warp_ids = tuple(participants)

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b) or (total_q, d, h) if there is cu_seqlens_q
        mK: cute.Tensor,  # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there is cu_seqlens_k or (page_size, d, h_k, num_pages) if there is page_table
        mV: cute.Tensor,  # (d, s_k, h_k, b_k) or (d, total_k, h_k) if there is cu_seqlens_k or (d, page_size, h_k, num_pages) if there is page_table
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        descale_tensors: Optional[DescaleTensors],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_Q: Optional[cute.TiledCopy],
        gmem_tiled_copy_O: Optional[cute.TiledCopy],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        num_splits: Int32,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.
        """

        assert const_expr(blocksparse_tensors is not None), 'Synchronized Fixed FA4 requires block sparsity'
        assert const_expr(self.q_dtype in (cutlass.BFloat16, cutlass.Float8E4M3FN)
                          and self.k_dtype == self.q_dtype and self.v_dtype == self.q_dtype
                          and self.o_dtype == cutlass.BFloat16), 'Synchronized Fixed FA4 requires BF16/E4M3FN operands and BF16 output'
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )
        # Setup cta/thread coordinates
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(cute.size(tiled_mma_qk.thr_id.shape) == 1):
            mma_tile_coord_v = 0
        else:
            mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE * len(
                (self.mma_warp_id,
                 *self.softmax0_warp_ids,
                 *self.softmax1_warp_ids,
                 *self.correction_warp_ids)
            ),
        )
        # Tensor memory dealloc barrier init
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        mma_warp = ThreadCooperativeGroup(len([self.mma_warp_id]))
        tma_warp = ThreadCooperativeGroup(1)
        load_threads = ThreadCooperativeGroup(len(self.load_warp_ids) * cute.arch.WARP_SIZE)
        softmax_warps = ThreadCooperativeGroup(len(self.softmax0_warp_ids))
        softmax_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE * len(self.softmax0_warp_ids))
        # softmax_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE)
        correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        # correction_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE)
        softmax_correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids + self.correction_warp_ids)
        )
        epilogue_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE * len(self.epilogue_warp_ids))
        # For UMMA-bridging pipelines: the non-MMA side spans both CTAs in the cluster,
        # so the thread count must include warps from both CTAs.
        softmax_warps_cluster = ThreadCooperativeGroup(
            len(self.softmax0_warp_ids) * self.cta_group_size
        )
        correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids) * self.cta_group_size
        )
        softmax_correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids + self.correction_warp_ids) * self.cta_group_size
        )
        if const_expr(self.use_tma_Q):
            pipeline_q = pipeline_custom.PipelineTmaUmma.create(
                barrier_storage=storage.mbar_load_Q.data_ptr(),
                num_stages=self.q_stage,
                producer_group=tma_warp,
                consumer_group=mma_warp,
                tx_count=self.tma_copy_bytes["Q"],
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        else:
            pipeline_q = pipeline_custom.PipelineAsyncUmma.create(
                barrier_storage=storage.mbar_load_Q.data_ptr(),
                num_stages=self.q_stage,
                producer_group=load_threads,
                consumer_group=mma_warp,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        if const_expr(self.use_tma_KV):
            pipeline_kv = pipeline_custom.PipelineTmaUmma.create(
                barrier_storage=storage.mbar_load_KV.data_ptr(),
                num_stages=self.kv_stage,
                producer_group=tma_warp,
                consumer_group=mma_warp,
                tx_count=self.tma_copy_bytes["K"],
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        else:
            pipeline_kv = pipeline.PipelineAsyncUmma.create(
                barrier_storage=storage.mbar_load_KV.data_ptr(),
                num_stages=self.kv_stage,
                producer_group=load_threads,
                consumer_group=mma_warp,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        # This pipeline is not the typical producer-consumer pipeline. The "producer" mma warp
        # uses it to signal that S is ready, and the softmax threads wait for S to be ready.
        # When softmax threads write P to tmem and the correction threads have rescaled O, they
        # signal as "consumer". The mma warp then waits for that signal to do the P @ V gemm.
        pipeline_s_p_o = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_S_full_P_full_O_rescaled.data_ptr(),
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=softmax_correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_p_lastsplit = pipeline_custom.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_P_full_lastsplit.data_ptr(),
            num_stages=self.q_stage,
            producer_group=softmax_warps_cluster,
            consumer_group=mma_warp,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # MMA warp uses this to signal to the correction warps that O is ready.
        pipeline_o_acc = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_O_full.data_ptr(),
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_s0_s1_sequence = None
        if const_expr(self.s0_s1_barrier and self.q_stage > 1):
            # This is not a typical producer-consumer pipeline. We will directly use
            # pipeline_s0_s1_sequence.sync_object_full and will not use
            # pipeline_s0_s1_sequence.sync_object_empty.
            pipeline_s0_s1_sequence = pipeline_custom.PipelineAsync.create(
                barrier_storage=storage.mbar_s0_s1_sequence.data_ptr(),
                num_stages=2,
                producer_group=softmax_threads,
                consumer_group=softmax_threads,
                defer_sync=True,
            )
        pipeline_sm_stats = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_softmax_stats.data_ptr(),
            num_stages=self.q_stage,
            producer_group=softmax_threads,
            consumer_group=correction_threads,
            defer_sync=True,
        )
        # Should put the NamedBarrier inside the pipeline class so we'll just have pipeline_sm_stats
        sm_stats_barrier = pipeline_custom.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0), num_threads=cute.arch.WARP_SIZE * 2
        )
        pipeline_o_epi = None
        if const_expr(not self.use_correction_warps_for_epi):
            pipeline_o_epi = pipeline_custom.PipelineAsync.create(
                barrier_storage=storage.mbar_O_epi.data_ptr(),
                num_stages=self.q_stage,
                producer_group=correction_threads,
                consumer_group=epilogue_threads,
                defer_sync=True,
            )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
        if const_expr(not self.overlap_sO_sQ):
            sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        else:
            sO = cute.make_tensor(cute.recast_ptr(sQ.iterator, sO_layout.inner, self.o_dtype), sO_layout.outer)

        sScale = storage.sScale.get_tensor(cute.make_layout(self.q_stage * self.m_block_size * 2))

        thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)
        thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        # This is a fake tensor, by right we need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, self.s_stage))
        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(cute.append(pv_acc_shape, self.q_stage))
        tOtO = cute.make_tensor(tOtO.iterator + self.tmem_o_offset[0], tOtO.layout)
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]
        # Need to multiply by width ratio bc tP is in v_dtype but tmem offsets are in FP32
        tP_width_ratio = Float32.width // self.v_dtype.width
        # Need to adjust the stage stride manually since the two stages aren't contiguous in tmem
        tP_stage_stride = (self.tmem_p_offset[1] - self.tmem_p_offset[0]) * tP_width_ratio
        tOrP = cute.make_tensor(
            tOrP.iterator + self.tmem_p_offset[0] * tP_width_ratio,
            cute.append(tOrP.layout, cute.make_layout((self.s_stage,), stride=(tP_stage_stride,)))
        )

        block_info = BlockInfo(
            # This is cta_tiler, not mma_tiler_qk, since we move by block by (2 * mma_tiler[0], mma_tiler[1])
            self.cta_tiler[0],
            self.cta_tiler[1],
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0]
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            mCuTotalMBlocks=(
                blocksparse_tensors.cu_total_m_blocks if blocksparse_tensors is not None else None
            ),
            mCuBlockIdxOffsets=(
                blocksparse_tensors.cu_block_idx_offsets if blocksparse_tensors is not None else None
            ),
        )
        AttentionMaskCls = self._generate_attention_mask_cls(
            window_size_left, window_size_right
        )
        # Cluster wait before tensor memory alloc
        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        if const_expr(self.use_clc_scheduler):
            clc_response_ptr = storage.clc_response.data_ptr()
            clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()

            clc_pipeline_producer_group = cutlass_pipeline.CooperativeGroup(
                cutlass_pipeline.Agent.Thread
            )
            num_clc_consumer_warps_per_cta = self.threads_per_cta // cute.arch.WARP_SIZE
            # NB on CTA0 warp15 == scheduler on CTA1 == empty but still both consume
            num_clc_consumer_warps = num_clc_consumer_warps_per_cta * self.cta_group_size
            clc_pipeline_consumer_group = cutlass_pipeline.CooperativeGroup(
                cutlass_pipeline.Agent.Thread, cute.arch.WARP_SIZE * num_clc_consumer_warps
            )

            block_idx = cute.arch.block_idx()
            clc = ClcState.create(
                hw_scheduler=ClcDynamicPersistentTileScheduler.create(
                    self.tile_scheduler_cls.clc_problem_shape(tile_sched_params),
                    block_idx,
                    cute.arch.grid_dim(),
                    clc_response_ptr,
                ),
                pipeline=cutlass_pipeline.PipelineClcFetchAsync.create(
                    barrier_storage=clc_mbar_ptr,
                    num_stages=self.sched_stages,
                    producer_group=clc_pipeline_producer_group,
                    consumer_group=clc_pipeline_consumer_group,
                    tx_count=16,
                    cta_layout_vmnk=cta_layout_vmnk,
                ),
                consumer_state=cutlass_pipeline.make_pipeline_state(
                    cutlass_pipeline.PipelineUserType.Consumer, self.sched_stages
                ),
                producer_state=cutlass_pipeline.make_pipeline_state(
                    cutlass_pipeline.PipelineUserType.Producer, self.sched_stages
                ),
            )
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params, clc=clc)
        else:
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params)
        assert isinstance(tile_scheduler, TileSchedulerProtocol), f"tile_scheduler is not a TileSchedulerProtocol: {type(tile_scheduler)}"

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY / CLC SCHEDULER WARP
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(self.use_clc_scheduler):
            if warp_idx == self.clc_scheduler_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                if is_leader_cta:
                    self.clc_scheduler_warp(tile_scheduler)
                else:
                    self.empty_warp(tile_scheduler)
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i] and warp_idx != self.clc_scheduler_warp_id:
                    cute.arch.setmaxregister_decrease(self.num_regs_other)
                    self.empty_warp(tile_scheduler)
        else:
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i]:
                    cute.arch.setmaxregister_decrease(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.load(
                thr_mma_qk,
                thr_mma_pv,
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                mPageTable,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                gmem_tiled_copy_Q,
                pipeline_q,
                pipeline_kv,
                block_info,
                num_splits,
                SeqlenInfoCls,
                blocksparse_tensors,
                tile_scheduler=tile_scheduler,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(not self.use_correction_warps_for_epi):
            if warp_idx >= self.epilogue_warp_ids[0] and warp_idx <= self.epilogue_warp_ids[-1]:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                self.epilogue_s2g(
                    mO,
                    sO,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    pipeline_o_epi,
                    block_info,
                    num_splits,
                    SeqlenInfoCls,
                    mma_tile_coord_v,
                    blocksparse_tensors=blocksparse_tensors,
                    tile_scheduler=tile_scheduler,
                )

        # MBSA: allocation ownership is tracked outside dynamic role branches.
        # The public allocator internally issues allocation only on its MMA warp.
        tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
        # Typed null TMEM pointer emits no shared-memory read for inactive warps.
        # Only the allocator warp dereferences the release pointer in free().
        tmem_release_ptr = cute.make_ptr(self.qk_acc_dtype, 0,
                                        mem_space=cute.AddressSpace.tmem, assumed_align=16)
        tmem_participates = warp_idx == self.mma_warp_id
        for i in cutlass.range_constexpr(len(self.tmem_participant_warp_ids)):
            tmem_participates = tmem_participates | (warp_idx == self.tmem_participant_warp_ids[i])
        if tmem_participates:
            # One generated named-barrier PC for every actual TMEM participant.
            # q_stage=1 excludes inactive/reassigned softmax1 warps.
            tmem.wait_for_alloc()
            tmem_release_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            # ///////////////////////////////////////////////////////////////////////////////
            #  MMA
            # ///////////////////////////////////////////////////////////////////////////////
            if warp_idx == self.mma_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                self.mma(
                    tiled_mma_qk,
                    tiled_mma_pv,
                    sQ,
                    sK,
                    sV,
                    tStS,
                    tOtO,
                    tOrP,
                    pipeline_q,
                    pipeline_kv,
                    pipeline_s_p_o,
                    pipeline_p_lastsplit,
                    pipeline_o_acc,
                    is_leader_cta,
                    block_info,
                    num_splits,
                    SeqlenInfoCls,
                    blocksparse_tensors,
                    tile_scheduler=tile_scheduler,
                )

            # ///////////////////////////////////////////////////////////////////////////////
            #  Softmax
            # ///////////////////////////////////////////////////////////////////////////////
            if (
                (const_expr(self.q_stage == 2) and warp_idx <= self.softmax1_warp_ids[-1]) or
                (const_expr(self.q_stage == 1) and warp_idx <= self.softmax0_warp_ids[-1])
            ):
                # increase register after decreasing
                cute.arch.setmaxregister_increase(self.num_regs_softmax)
                softmax_loop = partial(
                    self.softmax_loop,
                    softmax_scale_log2=softmax_scale_log2,
                    softmax_scale=softmax_scale,
                    descale_tensors=descale_tensors,
                    thr_mma_qk=thr_mma_qk,
                    sScale=sScale,
                    mLSE=mLSE,
                    pipeline_s_p_o=pipeline_s_p_o,
                    pipeline_p_lastsplit=pipeline_p_lastsplit,
                    pipeline_sm_stats=pipeline_sm_stats,
                    sm_stats_barrier=sm_stats_barrier,
                    pipeline_s0_s1_sequence=pipeline_s0_s1_sequence,
                    learnable_sink=learnable_sink,
                    block_info=block_info,
                    num_splits=num_splits,
                    SeqlenInfoCls=SeqlenInfoCls,
                    AttentionMaskCls=AttentionMaskCls,
                    aux_data=aux_data,
                    fastdiv_mods=fastdiv_mods,
                    head_divmod=head_divmod,
                    blocksparse_tensors=blocksparse_tensors,
                    tile_scheduler=tile_scheduler,
                )

                if const_expr(not self.s0_s1_barrier):
                    stage = Int32(0 if const_expr(self.q_stage == 1) or warp_idx < self.softmax1_warp_ids[0] else 1)
                    softmax_loop(stage=stage, tStS=tStS)
                else:
                    # If there's s0_s1_barrier, it's faster to have 2 WGs having different code
                    if warp_idx < self.softmax1_warp_ids[0]:
                        softmax_loop(stage=0, tStS=tStS)
                    if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax1_warp_ids[0]:
                        softmax_loop(stage=1, tStS=tStS)


            # ///////////////////////////////////////////////////////////////////////////////
            #  Correction
            # ///////////////////////////////////////////////////////////////////////////////
            if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_correction)
                self.correction_loop(
                    thr_mma_qk,
                    thr_mma_pv,
                    tStS,
                    tOtO,
                    sScale,
                    mO,
                    mLSE,
                    sO,
                    pipeline_s_p_o,
                    pipeline_o_acc,
                    pipeline_sm_stats,
                    sm_stats_barrier,
                    pipeline_o_epi,
                    learnable_sink,
                    descale_tensors,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    softmax_scale_log2,
                    block_info,
                    num_splits,
                    SeqlenInfoCls,
                    blocksparse_tensors,
                    tile_scheduler=tile_scheduler,
                )

            # One converged retirement PC after all participant role loops.
            tmem_alloc_barrier.arrive_and_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(
            tmem_release_ptr,
            num_columns=cute.arch.get_max_tmem_alloc_cols("sm_100"),
        )
        return


class _FixedSoftmaxFactory:
    """Keep public FP8 P within its representable range before native PV MMA."""

    @staticmethod
    def create(scale_log2, rescale_threshold=0.0, softmax_scale=None, max_offset=0):
        # Public FP8 uses exp2(... + 8), so an updated row maximum gives P<=256.
        # Holding the old maximum for four log2 units can produce P up to4096;
        # E4M3FN conversion then saturates448 while FP32 row_sum remains exact.
        # Rescale on every FP8 maximum increase. Preserve BF16's public policy.
        threshold = 0.0 if max_offset == 8 else rescale_threshold
        return SoftmaxSm100.create(
            scale_log2, rescale_threshold=threshold,
            softmax_scale=softmax_scale, max_offset=max_offset,
        )


@lru_cache(maxsize=1)
def _fixed_forward_class():
    """Clone the public softmax method with a private factory; installed state unchanged."""
    method = inspect.unwrap(FlashAttentionForwardSm100.softmax_loop)
    namespace = dict(method.__globals__)
    namespace['SoftmaxSm100'] = _FixedSoftmaxFactory
    clone = FunctionType(method.__code__, namespace, method.__name__,
                         method.__defaults__, method.__closure__)
    clone.__kwdefaults__ = method.__kwdefaults__
    clone.__annotations__ = method.__annotations__.copy()
    clone.__module__, clone.__qualname__ = method.__module__, method.__qualname__
    # If another FA4 call already preprocessed the public function, its code and
    # DSL metadata must stay paired. Only the namespace is private here.
    clone.__dict__.update(method.__dict__)

    class FixedForward(SynchronizedFlashAttentionForwardSm100):
        softmax_loop = cute.jit(clone)

    return FixedForward


@lru_cache(maxsize=None)
def _synchronized_compile_cache(function):
    return {}


class _CompileCache(dict):
    """Share private kernels while retaining each clone's selected variants."""

    def __init__(self, shared):
        self.shared = shared

    def __contains__(self, key):
        return super().__contains__(key) or key in self.shared

    def __missing__(self, key):
        value = self.shared[key]
        super().__setitem__(key, value)
        return value

    def __setitem__(self, key, value):
        self.shared[key] = value
        super().__setitem__(key, value)


def clone_fixed_forward(function=None):
    """Private class selection and shared private JIT cache; installed state unchanged.

    The public compile key does not include the selected Python kernel class.
    Only clones selecting this synchronized FP8-safe class share compiled kernels;
    the installed wrapper's cache remains separate.
    The returned public code/defaults/closure are unchanged. Its private globals
    select this subclass and refer back to this clone for cache accesses.
    """
    validate_public_source()
    if function is None:
        from flash_attn.cute.interface import _flash_attn_fwd
        function = _flash_attn_fwd
    namespace = dict(function.__globals__)
    namespace['FlashAttentionForwardSm100'] = _fixed_forward_class()
    clone = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__annotations__ = function.__annotations__.copy()
    clone.__dict__.update(function.__dict__)
    clone.compile_cache = _CompileCache(_synchronized_compile_cache(function))
    namespace[function.__name__] = clone
    return clone
