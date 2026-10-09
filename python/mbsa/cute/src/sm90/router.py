# SPDX-License-Identifier: Apache-2.0
"""Native SM90 block-max Indexer and partition Top-16."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Float32, Int32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.utils import LayoutEnum
from quack import copy_utils, layout_utils, sm90_utils


TOPK = 16
PREFIX_BLOCKS = 8
LOCAL_BLOCKS = 8
BLOCK = 128
HEAD_DIM = 128


@dsl_user_op
def mma_fp8_f32(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """Native m16n8k32 E4M3 x E4M3 with at-least-FP32 accumulation."""
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Int32(x).ir_value(loc=loc, ip=ip) for x in (a0, a1, a2, a3, b0, b1)]
        + [Float32(x).ir_value(loc=loc, ip=ip) for x in (c0, c1, c2, c3)],
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip))
                 for i in range(4))


@cute.jit
def insert_top16(
    top_scores: cute.Tensor,
    top_ids: cute.Tensor,
    candidate_score: Float32,
    candidate_id: Int32,
) -> None:
    """Insert into a descending list; equal scores keep the smaller block ID."""
    if candidate_score > top_scores[TOPK - 1] or (
        candidate_score == top_scores[TOPK - 1]
        and candidate_id >= Int32(0)
        and (top_ids[TOPK - 1] < Int32(0) or candidate_id < top_ids[TOPK - 1])
    ):
        for slot in cutlass.range_constexpr(TOPK):
            old_score = top_scores[slot]
            old_id = top_ids[slot]
            take = candidate_score > old_score
            take = take or (
                candidate_score == old_score
                and candidate_id >= Int32(0)
                and (old_id < Int32(0) or candidate_id < old_id)
            )
            top_scores[slot] = candidate_score if take else old_score
            top_ids[slot] = candidate_id if take else old_id
            candidate_score = old_score if take else candidate_score
            candidate_id = old_id if take else candidate_id


class IndexScoreTop16Sm90:
    """One request-local query tile / Index head; Top-16 per partition."""

    q_tile = 256
    fp8_q_tile = 128
    k_tile = 128
    k_stages = 2
    num_mma_warp_groups = 2
    threads = 128 * (num_mma_warp_groups + 1)
    producer_regs = 24
    # BF16 needs 232 consumer registers to avoid accumulator spills;
    # two consumer groups and the 24-register producer fit the SM90 register pool.
    mma_regs = 232

    def __init__(self, *, partition_blocks: int, partitions: int, fp8_operand: bool = False) -> None:
        self.fp8_operand = bool(fp8_operand)
        self.mma_regs = 128 if self.fp8_operand else type(self).mma_regs
        self.q_tile = self.fp8_q_tile if self.fp8_operand else type(self).q_tile
        self.num_mma_warp_groups = 1 if self.fp8_operand else 2
        self.threads = 128 * (self.num_mma_warp_groups + 1)
        self.partition_blocks = int(partition_blocks)
        self.partitions = int(partitions)
        self.direct_route = self.partitions == 1

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,  # [total_q, 4, 128]
        k: cute.Tensor,  # [pages * 128, 128]
        page_indices: cute.Tensor,  # [logical pages] int32
        tile_metadata: cute.Tensor,  # [tiles, 6] int32
        candidate_scores: cute.Tensor,  # [4, P, total_q, 16] fp32
        candidate_ids: cute.Tensor,  # [4, P, total_q, 16] int32
        route: cute.Tensor,  # [4, total_q, 16] int32
        stream: cuda.CUstream,
    ) -> None:
        dtype = q.element_type
        assert (dtype.width == 8) == self.fp8_operand
        total_q = q.shape[0]
        heads = q.shape[1]
        mQ = cute.make_tensor(
            q.iterator,
            cute.make_layout(
                (total_q, HEAD_DIM, heads), stride=(heads * HEAD_DIM, 1, HEAD_DIM)
            ),
        )
        sQ_layout = sm90_utils.make_smem_layout(
            dtype, LayoutEnum.ROW_MAJOR, (self.q_tile, HEAD_DIM)
        )
        sK_layout = sm90_utils.make_smem_layout(
            dtype, LayoutEnum.ROW_MAJOR, (self.k_tile, HEAD_DIM), self.k_stages
        )
        tiled_mma = sm90_utils_basic.make_trivial_tiled_mma(
            dtype,
            dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.num_mma_warp_groups, 1, 1),
            tiler_mn=(64, self.k_tile),
        )
        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_q, mQ_tma = cpasync.make_tiled_tma_atom(
            tma_op, mQ, sQ_layout, (self.q_tile, HEAD_DIM)
        )
        tma_k, mK_tma = cpasync.make_tiled_tma_atom(
            tma_op, k, cute.select(sK_layout, mode=[0, 1]), (self.k_tile, HEAD_DIM)
        )
        self.q_bytes = cute.size_in_bytes(dtype, sQ_layout)
        self.k_bytes = cute.size_in_bytes(dtype, cute.select(sK_layout, mode=[0, 1]))

        @cute.struct
        class SharedStorage:
            mbar_q: cute.struct.MemRange[cutlass.Int64, 2]
            mbar_k: cute.struct.MemRange[cutlass.Int64, self.k_stages * 2]
            sQ: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sQ_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sK_layout)], 1024
            ]

            warp_scores: cute.struct.Align[
                cute.struct.MemRange[Float32, 4 * self.q_tile if self.fp8_operand else 4], 16
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma,
            tma_q,
            mQ_tma,
            tma_k,
            mK_tma,
            page_indices,
            tile_metadata,
            candidate_scores,
            candidate_ids,
            route,
            sQ_layout,
            sK_layout,
        ).launch(
            grid=(tile_metadata.shape[0], heads, self.partitions),
            block=[self.threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_q: cute.CopyAtom,
        mQ: cute.Tensor,
        tma_k: cute.CopyAtom,
        mK: cute.Tensor,
        mPageIndices: cute.Tensor,
        mTileMetadata: cute.Tensor,
        mCandidateScores: cute.Tensor,
        mCandidateIds: cute.Tensor,
        mRoute: cute.Tensor,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
    ) -> None:
        tile_idx, head, partition = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_q)
            cpasync.prefetch_descriptor(tma_k)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        producer = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        consumers = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_warp_groups * 4
        )
        pipeline_q = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_q.data_ptr(),
            num_stages=1,
            producer_group=producer,
            consumer_group=consumers,
            tx_count=self.q_bytes,
            defer_sync=True,
        )
        pipeline_k = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_k.data_ptr(),
            num_stages=self.k_stages,
            producer_group=producer,
            consumer_group=consumers,
            tx_count=self.k_bytes,
            defer_sync=True,
        )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)

        if cutlass.const_expr(self.fp8_operand):
            warp_scores = storage.warp_scores.get_tensor(
                cute.make_layout((4, self.q_tile), stride=(self.q_tile, 1))
            )

        q_begin = mTileMetadata[tile_idx, 0]
        q_end = mTileMetadata[tile_idx, 1]
        local_tile = mTileMetadata[tile_idx, 2]
        page_begin = mTileMetadata[tile_idx, 3]
        k_tokens = mTileMetadata[tile_idx, 4]
        request_partitions = mTileMetadata[tile_idx, 5]
        q_len = q_end - q_begin
        q_offset = k_tokens - q_len
        q_local_start = local_tile * Int32(self.q_tile)
        q_last = q_local_start + Int32(self.q_tile - 1)
        if q_last > q_len - Int32(1):
            q_last = q_len - Int32(1)
        # Remote block b is eligible for a query in block c iff 8 <= b <= c - 8.
        block_end = (q_offset + q_last) // Int32(BLOCK) - Int32(LOCAL_BLOCKS - 1)
        block_begin = partition * Int32(self.partition_blocks)
        partition_end = block_begin + Int32(self.partition_blocks)
        if block_end > partition_end:
            block_end = partition_end
        if block_begin < Int32(PREFIX_BLOCKS):
            block_begin = Int32(PREFIX_BLOCKS)
        has_work = partition < request_partitions and block_begin < block_end

        if warp_idx < 4:
            cute.arch.setmaxregister_decrease(self.producer_regs)
            if warp_idx == 0 and has_work:
                mQ_cur = cute.domain_offset((q_begin, 0), mQ[None, None, head])
                gQ = cute.local_tile(mQ_cur, (self.q_tile, HEAD_DIM), (local_tile, 0))
                load_q, _, _ = copy_utils.tma_get_copy_fn(
                    tma_q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
                )
                gK = cute.local_tile(mK, (self.k_tile, HEAD_DIM), (None, 0))
                load_k, _, _ = copy_utils.tma_get_copy_fn(
                    tma_k, 0, cute.make_layout(1), gK, sK
                )
                q_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, 1
                )
                pipeline_q.producer_acquire(q_state)
                load_q(tma_bar_ptr=pipeline_q.producer_get_barrier(q_state))
                k_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.k_stages
                )
                for block in cutlass.range(block_begin, block_end, unroll=1):
                    page = mPageIndices[page_begin + block]
                    pipeline_k.producer_acquire(k_state)
                    load_k(
                        src_idx=page,
                        dst_idx=k_state.index,
                        tma_bar_ptr=pipeline_k.producer_get_barrier(k_state),
                    )
                    k_state.advance()
        else:
            cute.arch.setmaxregister_increase(self.mma_regs)
            if cutlass.const_expr(self.fp8_operand):
                # Four consumer warps each own32 KV columns; all128 Q rows
                # retain native FP8 operands. Every consumer thread owns one
                # query's Top16 state after the warp maxima are merged.
                consumer = tidx - Int32(128)
                consumer_warp = consumer // Int32(32)
                lane = cute.arch.lane_idx()
                group = lane // Int32(4)
                word_in_group = lane % Int32(4)
                q_local = q_local_start + consumer
                row_valid = q_local < q_len
                row_limit = (q_offset + q_local) // Int32(BLOCK) - Int32(LOCAL_BLOCKS)
                # recast_tensor on a swizzled pointer drops its swizzle in CuTe4.8.
                # Keep the full composed layout when reinterpreting four codes.
                q_words = cute.make_tensor(
                    cute.recast_ptr(storage.sQ.data_ptr(), dtype=cutlass.Uint32),
                    cute.recast_layout(32, 8, sQ_layout),
                )
                k_words = cute.make_tensor(
                    cute.recast_ptr(storage.sK.data_ptr(), dtype=cutlass.Uint32),
                    cute.recast_layout(32, 8, sK_layout),
                )
                top_scores = cute.make_rmem_tensor((TOPK,), Float32)
                top_ids = cute.make_rmem_tensor((TOPK,), Int32)
                for slot in cutlass.range_constexpr(TOPK):
                    top_scores[slot] = -Float32.inf
                    top_ids[slot] = Int32(-1)
                if has_work:
                    q_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
                    pipeline_q.consumer_wait(q_state)
                    k_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.k_stages)
                    for block in cutlass.range(block_begin, block_end, unroll=1):
                        pipeline_k.consumer_wait(k_state)
                        for m in cutlass.range_constexpr(self.q_tile // 16):
                            acc = cute.make_rmem_tensor((4, 4), Float32)
                            acc.fill(0)
                            for k_step in cutlass.range_constexpr(HEAD_DIM // 32):
                                kw = Int32(k_step * 8) + word_in_group
                                qr = Int32(m * 16) + group
                                # A fragments: rowg,K0; rowg+8,K0;
                                # rowg,K16; rowg+8,K16, four FP8 codes/word.
                                a0 = q_words[qr, kw]
                                a1 = q_words[qr + 8, kw]
                                a2 = q_words[qr, kw + 4]
                                a3 = q_words[qr + 8, kw + 4]
                                for n in cutlass.range_constexpr(4):
                                    kr = consumer_warp * Int32(32) + Int32(n * 8) + group
                                    b0 = k_words[kr, kw, k_state.index]
                                    b1 = k_words[kr, kw + 4, k_state.index]
                                    d0, d1, d2, d3 = mma_fp8_f32(
                                        a0, a1, a2, a3, b0, b1,
                                        acc[n, 0], acc[n, 1], acc[n, 2], acc[n, 3],
                                    )
                                    acc[n, 0], acc[n, 1] = d0, d1
                                    acc[n, 2], acc[n, 3] = d2, d3
                            lo, hi = -Float32.inf, -Float32.inf
                            for n in cutlass.range_constexpr(4):
                                lo = cute.arch.fmax(lo, cute.arch.fmax(acc[n, 0], acc[n, 1]))
                                hi = cute.arch.fmax(hi, cute.arch.fmax(acc[n, 2], acc[n, 3]))
                            lo = cute.arch.warp_reduction_max(lo, threads_in_group=4)
                            hi = cute.arch.warp_reduction_max(hi, threads_in_group=4)
                            if word_in_group == Int32(0):
                                warp_scores[consumer_warp, Int32(m * 16) + group] = lo
                                warp_scores[consumer_warp, Int32(m * 16 + 8) + group] = hi
                        cute.arch.barrier(barrier_id=1, number_of_threads=128)
                        score = warp_scores[0, consumer]
                        for w in cutlass.range_constexpr(1, 4):
                            score = cute.arch.fmax(score, warp_scores[w, consumer])
                        score = score * Float32(128 ** -0.5)
                        if row_valid and block <= row_limit:
                            insert_top16(top_scores, top_ids, score, block)
                        # Every reader finishes before any warp writes next block.
                        cute.arch.barrier(barrier_id=1, number_of_threads=128)
                        pipeline_k.consumer_release(k_state)
                        k_state.advance()
                if row_valid:
                    q_global = q_begin + q_local
                    if cutlass.const_expr(self.direct_route):
                        for slot in cutlass.range_constexpr(TOPK):
                            mRoute[head, q_global, slot] = top_ids[slot]
                    else:
                        for slot in cutlass.range_constexpr(TOPK):
                            mCandidateScores[head, partition, q_global, slot] = top_scores[slot]
                            mCandidateIds[head, partition, q_global, slot] = top_ids[slot]
            else:
                mma_tidx = tidx - Int32(128)
                warp_group = cute.arch.make_warp_uniform(mma_tidx // Int32(128))
                thr_mma = tiled_mma.get_slice(mma_tidx)
                wg_mma = tiled_mma.get_slice(warp_group * Int32(128))
                _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
                    wg_mma, (self.q_tile, self.k_tile, HEAD_DIM), sQ, sK
                )
                coords = layout_utils.reshape_acc_to_mn(
                    thr_mma.partition_C(cute.make_identity_tensor((self.q_tile, self.k_tile)))
                )
                # Each thread holds 4 accumulator rows; quad lane i owns row i.
                rows = cute.size(coords, mode=[0])
                owner = cute.arch.lane_idx() % Int32(4)
                row = coords[0, 0][0]
                for r in cutlass.range_constexpr(1, rows):
                    if owner == Int32(r):
                        row = coords[r, 0][0]
                q_local = q_local_start + row
                row_valid = q_local < q_len
                # Last eligible block for this row (blocks <= c - 8).
                row_limit = (q_offset + q_local) // Int32(BLOCK) - Int32(LOCAL_BLOCKS)

                top_scores = cute.make_rmem_tensor((TOPK,), Float32)
                top_ids = cute.make_rmem_tensor((TOPK,), Int32)
                for slot in cutlass.range_constexpr(TOPK):
                    top_scores[slot] = -Float32.inf
                    top_ids[slot] = Int32(-1)

                if has_work:
                    q_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, 1
                    )
                    pipeline_q.consumer_wait(q_state)
                    k_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.k_stages
                    )
                    for block in cutlass.range(block_begin, block_end, unroll=1):
                        pipeline_k.consumer_wait(k_state)
                        acc = sm90_utils.gemm_zero_init(
                            tiled_mma,
                            (self.q_tile, self.k_tile),
                            tSrQ,
                            tSrK,
                            B_idx=k_state.index,
                            wg_wait=0,
                        )
                        pipeline_k.consumer_release(k_state)
                        k_state.advance()
                        acc_mn = layout_utils.reshape_acc_to_mn(acc)
                        score = -Float32.inf
                        for r in cutlass.range_constexpr(rows):
                            row_max = acc_mn[r, 0]
                            for j in cutlass.range_constexpr(1, cute.size(acc_mn, mode=[1])):
                                row_max = cute.arch.fmax(row_max, acc_mn[r, j])
                            row_max = cute.arch.warp_reduction_max(row_max, threads_in_group=4)
                            if owner == Int32(r):
                                score = row_max
                        if row_valid and block <= row_limit:
                            insert_top16(top_scores, top_ids, score, block)

                if row_valid:
                    q_global = q_begin + q_local
                    if cutlass.const_expr(self.direct_route):
                        for slot in cutlass.range_constexpr(TOPK):
                            mRoute[head, q_global, slot] = top_ids[slot]
                    else:
                        for slot in cutlass.range_constexpr(TOPK):
                            mCandidateScores[head, partition, q_global, slot] = top_scores[slot]
                            mCandidateIds[head, partition, q_global, slot] = top_ids[slot]


class MergeTop16Sm90:
    """Merge partition Top-16 lists into the exact global Top-16 route."""

    threads = 128

    @cute.jit
    def __call__(
        self,
        candidate_scores: cute.Tensor,  # [4, P, total_q, 16]
        candidate_ids: cute.Tensor,
        route: cute.Tensor,  # [4, total_q, 16]
        stream: cuda.CUstream,
    ) -> None:
        total_q = candidate_scores.shape[2]
        self.kernel(candidate_scores, candidate_ids, route).launch(
            grid=(cute.ceil_div(total_q, self.threads), candidate_scores.shape[0], 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mCandidateScores: cute.Tensor,
        mCandidateIds: cute.Tensor,
        mRoute: cute.Tensor,
    ) -> None:
        block_q, head, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        query = block_q * Int32(self.threads) + tidx
        if query < mCandidateScores.shape[2]:
            top_scores = cute.make_rmem_tensor((TOPK,), Float32)
            top_ids = cute.make_rmem_tensor((TOPK,), Int32)
            # Partition 0 is already sorted; seed the list with it.
            for slot in cutlass.range_constexpr(TOPK):
                top_scores[slot] = mCandidateScores[head, 0, query, slot]
                top_ids[slot] = mCandidateIds[head, 0, query, slot]
            for partition in cutlass.range(1, mCandidateScores.shape[1], unroll=1):
                for slot in cutlass.range_constexpr(TOPK):
                    insert_top16(
                        top_scores,
                        top_ids,
                        mCandidateScores[head, partition, query, slot],
                        mCandidateIds[head, partition, query, slot],
                    )
            for slot in cutlass.range_constexpr(TOPK):
                mRoute[head, query, slot] = top_ids[slot]
