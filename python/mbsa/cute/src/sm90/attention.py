"""BF16 SM90 Fixed / Dynamic sparse attention and state combine.

All attention tiles pack GQA: 128 MMA rows are 16 queries x 8 query heads of
one KV head, so every K/V block loaded by TMA is reused by the whole group.
Two consumer warpgroups (64 rows each) run WGMMA for S = Q K^T and the
register-sourced O = P V; one producer warp issues TMA loads.

* ``FixedAttentionSm90`` walks the implicit Prefix-8 + SWA-8 support of a
  16-query tile with online softmax and writes a normalized state.
* ``DynamicAttentionSm90`` is KV-centric: a persistent CTA loads one routed
  KV block once and streams the query chunks routed to it (CSR order),
  writing one normalized partial state per (query, route slot).
* ``CombineSm90`` folds the fixed state and the 16 dynamic partials into the
  final output and natural-log LSE.

Dynamic blocks are always entirely causal-visible and inside the logical KV
length, so the dynamic softmax needs no mask.
"""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
from quack import copy_utils, layout_utils, sm90_utils


BLOCK = 128
HEAD_DIM = 128
GROUP = 8
TOPK = 16
PREFIX_BLOCKS = 8
LOCAL_BLOCKS = 8
TILE_M = 128
TILE_QUERIES = TILE_M // GROUP
SMEM_O_STRIDE = HEAD_DIM + 8
LOG2E = math.log2(math.e)
LN2 = math.log(2.0)


def _tiled_mmas(warpgroups: int = 2):
    dtype = cutlass.BFloat16
    qk = sm90_utils_basic.make_trivial_tiled_mma(
        dtype,
        dtype,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.K,
        Float32,
        atom_layout_mnk=(warpgroups, 1, 1),
        tiler_mn=(64, BLOCK),
    )
    pv = sm90_utils_basic.make_trivial_tiled_mma(
        dtype,
        dtype,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.MN,
        Float32,
        atom_layout_mnk=(warpgroups, 1, 1),
        tiler_mn=(64, HEAD_DIM),
        a_source=warpgroup.OperandSource.RMEM,
    )
    return qk, pv


def _packed_q_view(q: cute.Tensor) -> cute.Tensor:
    """[total_q, 32, 128] -> ((8 heads, total_q), 128, 4 KV heads)."""
    total_q = q.shape[0]
    return cute.make_tensor(
        q.iterator,
        cute.make_layout(
            ((GROUP, total_q), HEAD_DIM, 4),
            stride=((HEAD_DIM, 4 * GROUP * HEAD_DIM), 1, GROUP * HEAD_DIM),
        ),
    )


def _kv_view(k: cute.Tensor, paged: bool) -> cute.Tensor:
    """Packed [total_k, 4, 128] -> (total_k, 128, 4).

    Paged [pages, 4, 128, 128] with any 16-byte-aligned strides (including
    vLLM's interleaved K/V pool) -> (128 tokens, 128, 4, pages).
    """
    if paged:
        return cute.make_tensor(
            k.iterator,
            cute.make_layout(
                (BLOCK, HEAD_DIM, 4, k.shape[0]),
                stride=(k.stride[2], 1, k.stride[1], k.stride[0]),
            ),
        )
    return cute.make_tensor(
        k.iterator,
        cute.make_layout((k.shape[0], HEAD_DIM, 4), stride=(4 * HEAD_DIM, 1, HEAD_DIM)),
    )


def _kv_blocks(mX: cute.Tensor, kv_head: Int32, token_start: Int32, paged: bool) -> cute.Tensor:
    """Per-block TMA source: indexed by physical page (paged) or block (packed)."""
    if paged:
        return cute.local_tile(mX[None, None, kv_head, None], (BLOCK, HEAD_DIM), (0, 0, None))
    return cute.local_tile(
        cute.domain_offset((token_start, 0), mX[None, None, kv_head]),
        (BLOCK, HEAD_DIM),
        (None, 0),
    )


@cute.jit
def _softmax_rows(
    acc_S: cute.Tensor,
    row_max: cute.Tensor,
    row_sum: cute.Tensor,
    row_scale: cute.Tensor,
    scale_log2: Float32,
    is_first: cutlass.Constexpr[bool],
) -> None:
    """Online softmax over one KV block; P overwrites S in place.

    ``row_sum`` keeps per-thread partial sums; the quad reduction happens once
    in ``_finalize_rows``.
    """
    acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
    for r in cutlass.range_constexpr(cute.size(row_max)):
        block_max = acc_S_mn[r, 0]
        for j in cutlass.range_constexpr(1, cute.size(acc_S_mn, mode=[1])):
            block_max = cute.arch.fmax(block_max, acc_S_mn[r, j])
        block_max = cute.arch.warp_reduction_max(block_max, threads_in_group=4)
        new_max = block_max
        if cutlass.const_expr(not is_first):
            new_max = cute.arch.fmax(row_max[r], block_max)
        safe_max = new_max
        if safe_max == -Float32.inf:
            safe_max = Float32(0.0)
        shift = safe_max * scale_log2
        partial = Float32(0.0)
        for j in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[1])):
            p = cute.math.exp2(acc_S_mn[r, j] * scale_log2 - shift, fastmath=True)
            acc_S_mn[r, j] = p
            partial = partial + p
        if cutlass.const_expr(is_first):
            row_scale[r] = Float32(1.0)
            row_sum[r] = partial
        else:
            scale = cute.math.exp2(row_max[r] * scale_log2 - shift, fastmath=True)
            if row_max[r] == -Float32.inf:
                scale = Float32(0.0)
            row_scale[r] = scale
            row_sum[r] = row_sum[r] * scale + partial
        row_max[r] = new_max


@cute.jit
def _mask_block(
    acc_S: cute.Tensor,
    s_coords: cute.Tensor,
    block: Int32,
    row_pos: cute.Tensor,
    row_window: cute.Tensor,
) -> None:
    """Mask tokens past each row's position or outside its SWA window."""
    acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
    block_start = block * Int32(BLOCK)
    for r in cutlass.range_constexpr(cute.size(row_pos)):
        limit = row_pos[r] - block_start
        if block >= Int32(PREFIX_BLOCKS) and block < row_window[r]:
            limit = Int32(-1)
        for j in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[1])):
            if s_coords[r, j][1] > limit:
                acc_S_mn[r, j] = -Float32.inf


@cute.jit
def _rescale_rows(acc_O: cute.Tensor, row_scale: cute.Tensor) -> None:
    acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
    for r in cutlass.range_constexpr(cute.size(row_scale)):
        for j in cutlass.range_constexpr(cute.size(acc_O_mn, mode=[1])):
            acc_O_mn[r, j] = acc_O_mn[r, j] * row_scale[r]


@cute.jit
def _finalize_rows(
    acc_O: cute.Tensor,
    row_max: cute.Tensor,
    row_sum: cute.Tensor,
    row_lse: cute.Tensor,
    scale: Float32,
) -> None:
    """Normalize O and produce the natural-log LSE of each owned row."""
    acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
    for r in cutlass.range_constexpr(cute.size(row_max)):
        total = cute.arch.warp_reduction_sum(row_sum[r], threads_in_group=4)
        inv = Float32(0.0)
        lse = -Float32.inf
        if total > Float32(0.0):
            inv = cute.arch.rcp_approx(total)
            lse = row_max[r] * scale + cute.math.log2(total, fastmath=True) * Float32(LN2)
        row_lse[r] = lse
        for j in cutlass.range_constexpr(cute.size(acc_O_mn, mode=[1])):
            acc_O_mn[r, j] = acc_O_mn[r, j] * inv


@cute.jit
def _stage_output(
    acc_O: cute.Tensor, sO: cute.Tensor, tiled_mma_pv: cute.TiledMma, tidx: Int32
) -> None:
    """Convert the normalized accumulator and stage it in padded row-major smem.

    BF16 uses ``stmatrix``; FP8 uses paired 16-bit stores.
    """
    dtype = sO.element_type
    converted = cute.make_rmem_tensor_like(acc_O, dtype)
    converted.store(acc_O.load().to(dtype))
    thr_copy = cute.make_tiled_copy_C(
        copy_utils.get_smem_store_atom(dtype), tiled_mma_pv
    ).get_slice(tidx)
    cute.copy(thr_copy, thr_copy.retile(converted), thr_copy.partition_D(sO))


def _consumer_barrier() -> None:
    cute.arch.barrier(barrier_id=1, number_of_threads=256)


class FixedAttentionSm90:
    """Prefix-8 + SWA-8 attention for one 16-query x 8-head tile per CTA."""

    kv_stages = 2
    threads = 384
    producer_regs = 24
    mma_regs = 240

    def __init__(self, *, paged: bool = False) -> None:
        self.paged = bool(paged)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,  # [total_q, 32, 128]
        k: cute.Tensor,  # [total_k, 4, 128]
        v: cute.Tensor,  # [total_k, 4, 128]
        out: cute.Tensor,  # [total_q, 32, 128]
        lse: cute.Tensor,  # [total_q, 32]
        tile_metadata: cute.Tensor,  # [tiles, 6]
        page_table: cute.Tensor,  # [rows, pages] int32 (unused when packed)
        softmax_scale: Float32,
        stream: cuda.CUstream,
    ) -> None:
        dtype = cutlass.BFloat16
        sQ_layout = sm90_utils.make_smem_layout(dtype, LayoutEnum.ROW_MAJOR, (TILE_M, HEAD_DIM))
        sK_layout = sm90_utils.make_smem_layout(
            dtype, LayoutEnum.ROW_MAJOR, (BLOCK, HEAD_DIM), self.kv_stages
        )
        tiled_mma_qk, tiled_mma_pv = _tiled_mmas()
        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_q, mQ = cpasync.make_tiled_tma_atom(
            tma_op, _packed_q_view(q), sQ_layout, ((GROUP, TILE_QUERIES), HEAD_DIM)
        )
        kv_box = cute.select(sK_layout, mode=[0, 1])
        tma_k, mK = cpasync.make_tiled_tma_atom(
            tma_op, _kv_view(k, self.paged), kv_box, (BLOCK, HEAD_DIM)
        )
        tma_v, mV = cpasync.make_tiled_tma_atom(
            tma_op, _kv_view(v, self.paged), kv_box, (BLOCK, HEAD_DIM)
        )
        self.q_bytes = cute.size_in_bytes(dtype, sQ_layout)
        self.kv_bytes = cute.size_in_bytes(dtype, kv_box)

        @cute.struct
        class SharedStorage:
            mbar_q: cute.struct.MemRange[cutlass.Int64, 2]
            mbar_k: cute.struct.MemRange[cutlass.Int64, self.kv_stages * 2]
            mbar_v: cute.struct.MemRange[cutlass.Int64, self.kv_stages * 2]
            sQ: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sQ_layout)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sK_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sK_layout)], 1024]
            sO: cute.struct.Align[cute.struct.MemRange[dtype, TILE_M * SMEM_O_STRIDE], 128]

        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma_qk,
            tiled_mma_pv,
            tma_q,
            mQ,
            tma_k,
            mK,
            tma_v,
            mV,
            out,
            lse,
            tile_metadata,
            page_table,
            softmax_scale,
            sQ_layout,
            sK_layout,
        ).launch(
            grid=(tile_metadata.shape[0], 4, 1),
            block=[self.threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tma_q: cute.CopyAtom,
        mQ: cute.Tensor,
        tma_k: cute.CopyAtom,
        mK: cute.Tensor,
        tma_v: cute.CopyAtom,
        mV: cute.Tensor,
        mOut: cute.Tensor,
        mLse: cute.Tensor,
        mTileMetadata: cute.Tensor,
        mPageTable: cute.Tensor,
        softmax_scale: Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
    ) -> None:
        tile_idx, kv_head, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_q)
            cpasync.prefetch_descriptor(tma_k)
            cpasync.prefetch_descriptor(tma_v)
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        producer = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        consumers = pipeline.CooperativeGroup(pipeline.Agent.Thread, 8)
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
            num_stages=self.kv_stages,
            producer_group=producer,
            consumer_group=consumers,
            tx_count=self.kv_bytes,
            defer_sync=True,
        )
        pipeline_v = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_v.data_ptr(),
            num_stages=self.kv_stages,
            producer_group=producer,
            consumer_group=consumers,
            tx_count=self.kv_bytes,
            defer_sync=True,
        )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sO = storage.sO.get_tensor(
            cute.make_layout((TILE_M, HEAD_DIM), stride=(SMEM_O_STRIDE, 1))
        )

        q_begin = mTileMetadata[tile_idx, 0]
        q_len = mTileMetadata[tile_idx, 1]
        q_tile = mTileMetadata[tile_idx, 2]
        k_begin = mTileMetadata[tile_idx, 3]
        k_tokens = mTileMetadata[tile_idx, 4]
        page_row = mTileMetadata[tile_idx, 5]
        q_offset = k_tokens - q_len
        q_last = q_tile + Int32(TILE_QUERIES - 1)
        if q_last > q_len - Int32(1):
            q_last = q_len - Int32(1)
        first_block = (q_offset + q_tile) // Int32(BLOCK)
        last_block = (q_offset + q_last) // Int32(BLOCK)
        prefix_count = last_block + Int32(1)
        if prefix_count > Int32(PREFIX_BLOCKS):
            prefix_count = Int32(PREFIX_BLOCKS)
        local_begin = first_block - Int32(LOCAL_BLOCKS - 1)
        if local_begin < Int32(PREFIX_BLOCKS):
            local_begin = Int32(PREFIX_BLOCKS)
        local_count = last_block + Int32(1) - local_begin
        if local_count < Int32(0):
            local_count = Int32(0)
        block_count = local_count + prefix_count

        if warp_idx < 4:
            cute.arch.setmaxregister_decrease(self.producer_regs)
            if warp_idx == 0:
                mQ_cur = cute.domain_offset(((0, q_begin + q_tile), 0), mQ[None, None, kv_head])
                gQ = cute.local_tile(mQ_cur, (TILE_M, HEAD_DIM), (0, 0))
                load_q, _, _ = copy_utils.tma_get_copy_fn(
                    tma_q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
                )
                gK = _kv_blocks(mK, kv_head, k_begin, self.paged)
                gV = _kv_blocks(mV, kv_head, k_begin, self.paged)
                load_k, _, _ = copy_utils.tma_get_copy_fn(tma_k, 0, cute.make_layout(1), gK, sK)
                load_v, _, _ = copy_utils.tma_get_copy_fn(tma_v, 0, cute.make_layout(1), gV, sV)
                q_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                pipeline_q.producer_acquire(q_state)
                load_q(tma_bar_ptr=pipeline_q.producer_get_barrier(q_state))
                kv_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.kv_stages
                )
                for i in cutlass.range(block_count, unroll=1):
                    # Diagonal-first: local blocks descending, then the prefix.
                    block = last_block - i
                    if i >= local_count:
                        block = i - local_count
                    source = block
                    if cutlass.const_expr(self.paged):
                        source = mPageTable[page_row, block]
                    pipeline_k.producer_acquire(kv_state)
                    load_k(
                        src_idx=source,
                        dst_idx=kv_state.index,
                        tma_bar_ptr=pipeline_k.producer_get_barrier(kv_state),
                    )
                    pipeline_v.producer_acquire(kv_state)
                    load_v(
                        src_idx=source,
                        dst_idx=kv_state.index,
                        tma_bar_ptr=pipeline_v.producer_get_barrier(kv_state),
                    )
                    kv_state.advance()
        else:
            cute.arch.setmaxregister_increase(self.mma_regs)
            mma_tidx = tidx - Int32(128)
            warp_group = cute.arch.make_warp_uniform(mma_tidx // Int32(128))
            thr_mma_qk = tiled_mma_qk.get_slice(mma_tidx)
            wg_mma_qk = tiled_mma_qk.get_slice(warp_group * Int32(128))
            wg_mma_pv = tiled_mma_pv.get_slice(warp_group * Int32(128))
            _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
                wg_mma_qk, (TILE_M, BLOCK, HEAD_DIM), sQ, sK
            )
            acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
                wg_mma_pv, (TILE_M, HEAD_DIM, BLOCK), None, layout_utils.transpose_view(sV)
            )
            acc_O.fill(0.0)
            s_coords = layout_utils.reshape_acc_to_mn(
                thr_mma_qk.partition_C(cute.make_identity_tensor((TILE_M, BLOCK)))
            )
            rows = cute.size(s_coords, mode=[0])
            row_pos = cute.make_rmem_tensor((rows,), Int32)
            row_window = cute.make_rmem_tensor((rows,), Int32)
            for r in cutlass.range_constexpr(rows):
                q_local = q_tile + s_coords[r, 0][0] // Int32(GROUP)
                position = q_offset + q_local
                if q_local >= q_len:
                    position = Int32(-1)
                row_pos[r] = position
                window = position // Int32(BLOCK) - Int32(LOCAL_BLOCKS - 1)
                row_window[r] = window
            row_max = cute.make_rmem_tensor((rows,), Float32)
            row_sum = cute.make_rmem_tensor((rows,), Float32)
            row_scale = cute.make_rmem_tensor((rows,), Float32)
            scale_log2 = softmax_scale * Float32(LOG2E)

            q_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            pipeline_q.consumer_wait(q_state)
            k_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.kv_stages
            )
            v_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.kv_stages
            )
            # The first block is the diagonal (or prefix block 0) and is masked.
            first = last_block
            if local_count == Int32(0):
                first = Int32(0)
            pipeline_k.consumer_wait(k_state)
            acc_S = sm90_utils.gemm_zero_init(
                tiled_mma_qk, (TILE_M, BLOCK), tSrQ, tSrK, B_idx=k_state.index, wg_wait=0
            )
            pipeline_k.consumer_release(k_state)
            k_state.advance()
            _mask_block(acc_S, s_coords, first, row_pos, row_window)
            _softmax_rows(acc_S, row_max, row_sum, row_scale, scale_log2, True)
            tOrP.store(layout_utils.reshape_acc_to_frgA(acc_S).load().to(cutlass.BFloat16))
            # Overlap S_i = Q K_i^T with O += P_{i-1} V_{i-1} inside each warpgroup.
            for i in cutlass.range(1, block_count, unroll=1):
                block = last_block - i
                if i >= local_count:
                    block = i - local_count
                pipeline_k.consumer_wait(k_state)
                acc_S = sm90_utils.gemm_zero_init(
                    tiled_mma_qk, (TILE_M, BLOCK), tSrQ, tSrK, B_idx=k_state.index, wg_wait=-1
                )
                pipeline_v.consumer_wait(v_state)
                sm90_utils.gemm_w_idx(
                    tiled_mma_pv, acc_O, tOrP, tOrVt, False, B_idx=v_state.index, wg_wait=-1
                )
                warpgroup.wait_group(1)
                pipeline_k.consumer_release(k_state)
                k_state.advance()
                if block >= first_block or (
                    block >= Int32(PREFIX_BLOCKS) and block < last_block - Int32(LOCAL_BLOCKS - 1)
                ):
                    _mask_block(acc_S, s_coords, block, row_pos, row_window)
                _softmax_rows(acc_S, row_max, row_sum, row_scale, scale_log2, False)
                warpgroup.wait_group(0)
                pipeline_v.consumer_release(v_state)
                v_state.advance()
                _rescale_rows(acc_O, row_scale)
                tOrP.store(layout_utils.reshape_acc_to_frgA(acc_S).load().to(cutlass.BFloat16))
            pipeline_v.consumer_wait(v_state)
            sm90_utils.gemm_w_idx(
                tiled_mma_pv, acc_O, tOrP, tOrVt, False, B_idx=v_state.index, wg_wait=0
            )
            pipeline_v.consumer_release(v_state)

            row_lse = cute.make_rmem_tensor((rows,), Float32)
            _finalize_rows(acc_O, row_max, row_sum, row_lse, softmax_scale)
            _stage_output(acc_O, sO, tiled_mma_pv, mma_tidx)
            lane = cute.arch.lane_idx()
            if lane % Int32(4) == Int32(0):
                for r in cutlass.range_constexpr(rows):
                    row = s_coords[r, 0][0]
                    q_local = q_tile + row // Int32(GROUP)
                    if q_local < q_len:
                        mLse[q_begin + q_local, kv_head * Int32(GROUP) + row % Int32(GROUP)] = (
                            row_lse[r]
                        )
            _consumer_barrier()
            # 16 threads write one 256-byte row; 256 threads cover 16 rows per pass.
            chunk = mma_tidx % Int32(16)
            for pass_idx in cutlass.range_constexpr(TILE_M // 16):
                row = mma_tidx // Int32(16) + Int32(16 * pass_idx)
                q_local = q_tile + row // Int32(GROUP)
                if q_local < q_len:
                    src = cute.local_tile(sO[row, None], (8,), (chunk,))
                    dst = cute.local_tile(
                        mOut[q_begin + q_local, kv_head * Int32(GROUP) + row % Int32(GROUP), None],
                        (8,),
                        (chunk,),
                    )
                    cute.autovec_copy(src, dst)


class DynamicAttentionSm90:
    """KV-centric routed attention producing per-route normalized partials.

    Each warpgroup owns independent 8-query chunks (64 MMA rows) with its own
    Q pipeline and output staging buffer, so one warpgroup's epilogue overlaps
    the other's WGMMA. Both share the work item's K/V stage.
    """

    kv_stages = 2
    q_stages = 2
    threads = 384
    producer_regs = 24
    mma_regs = 240
    wg_rows = 64
    chunk_queries = wg_rows // GROUP
    max_queries_per_work = 160

    def __init__(self, *, ctas: int, paged: bool = False, fp8_partial: bool = False) -> None:
        self.ctas = int(ctas)
        self.paged = bool(paged)
        self.fp8_partial = bool(fp8_partial)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,  # [total_q, 32, 128]
        k: cute.Tensor,  # [total_k, 4, 128]
        v: cute.Tensor,
        work: cute.Tensor,  # [W, 4] int32: kv_head, token start or page, list offset, count
        work_count: cute.Tensor,  # [1] int32
        relations: cute.Tensor,  # [R] int32 route-entry IDs grouped by work
        partial_out: cute.Tensor,  # [R, 8, 128] BF16 or FP8 E4M3FN
        partial_lse: cute.Tensor,  # [R, 8]
        softmax_scale: Float32,
        stream: cuda.CUstream,
    ) -> None:
        dtype = cutlass.BFloat16
        if cutlass.const_expr(self.fp8_partial):
            # Torch DLPack has no FP8: the host passes a uint8 view.
            partial_out = cute.recast_tensor(partial_out, cutlass.Float8E4M3FN)
        self.partial_dtype = partial_out.element_type
        # Pad each staged row by 16 bytes: rows stay 16-byte aligned.
        self.o_stride = HEAD_DIM + 128 // self.partial_dtype.width
        sQ_layout = sm90_utils.make_smem_layout(
            dtype, LayoutEnum.ROW_MAJOR, (self.wg_rows, HEAD_DIM), self.q_stages
        )
        sQ_row_layout = sm90_utils.make_smem_layout(dtype, LayoutEnum.ROW_MAJOR, (GROUP, HEAD_DIM))
        sK_layout = sm90_utils.make_smem_layout(
            dtype, LayoutEnum.ROW_MAJOR, (BLOCK, HEAD_DIM), self.kv_stages
        )
        tiled_mma_qk, tiled_mma_pv = _tiled_mmas(warpgroups=1)
        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_q, mQ = cpasync.make_tiled_tma_atom(
            tma_op, _packed_q_view(q), sQ_row_layout, ((GROUP, 1), HEAD_DIM)
        )
        kv_box = cute.select(sK_layout, mode=[0, 1])
        tma_k, mK = cpasync.make_tiled_tma_atom(
            tma_op, _kv_view(k, self.paged), kv_box, (BLOCK, HEAD_DIM)
        )
        tma_v, mV = cpasync.make_tiled_tma_atom(
            tma_op, _kv_view(v, self.paged), kv_box, (BLOCK, HEAD_DIM)
        )
        self.q_bytes = cute.size_in_bytes(dtype, cute.select(sQ_layout, mode=[0, 1]))
        self.kv_bytes = cute.size_in_bytes(dtype, kv_box)

        @cute.struct
        class SharedStorage:
            mbar_q: cute.struct.MemRange[cutlass.Int64, 2 * self.q_stages * 2]
            mbar_k: cute.struct.MemRange[cutlass.Int64, self.kv_stages * 2]
            mbar_v: cute.struct.MemRange[cutlass.Int64, self.kv_stages * 2]
            sQ0: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sQ_layout)], 1024]
            sQ1: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sQ_layout)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sK_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sK_layout)], 1024]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.partial_dtype, 2 * self.wg_rows * self.o_stride], 128
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma_qk,
            tiled_mma_pv,
            tma_q,
            mQ,
            tma_k,
            mK,
            tma_v,
            mV,
            work,
            work_count,
            relations,
            partial_out,
            partial_lse,
            q.shape[0],
            softmax_scale,
            sQ_layout,
            sK_layout,
        ).launch(
            grid=(self.ctas, 1, 1),
            block=[self.threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tma_q: cute.CopyAtom,
        mQ: cute.Tensor,
        tma_k: cute.CopyAtom,
        mK: cute.Tensor,
        tma_v: cute.CopyAtom,
        mV: cute.Tensor,
        mWork: cute.Tensor,
        mWorkCount: cute.Tensor,
        mRelations: cute.Tensor,
        mPartialOut: cute.Tensor,
        mPartialLse: cute.Tensor,
        total_q: Int32,
        softmax_scale: Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
    ) -> None:
        cta, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_q)
            cpasync.prefetch_descriptor(tma_k)
            cpasync.prefetch_descriptor(tma_v)
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        producer = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        wg_consumers = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
        consumers = pipeline.CooperativeGroup(pipeline.Agent.Thread, 8)
        pipeline_q0 = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_q.data_ptr(),
            num_stages=self.q_stages,
            producer_group=producer,
            consumer_group=wg_consumers,
            tx_count=self.q_bytes,
            defer_sync=True,
        )
        pipeline_q1 = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_q.data_ptr() + 2 * self.q_stages,
            num_stages=self.q_stages,
            producer_group=producer,
            consumer_group=wg_consumers,
            tx_count=self.q_bytes,
            defer_sync=True,
        )
        pipeline_k = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_k.data_ptr(),
            num_stages=self.kv_stages,
            producer_group=producer,
            consumer_group=consumers,
            tx_count=self.kv_bytes,
            defer_sync=True,
        )
        pipeline_v = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_v.data_ptr(),
            num_stages=self.kv_stages,
            producer_group=producer,
            consumer_group=consumers,
            tx_count=self.kv_bytes,
            defer_sync=True,
        )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        sQ0 = storage.sQ0.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sQ1 = storage.sQ1.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sO_all = storage.sO.get_tensor(
            cute.make_layout((self.wg_rows, HEAD_DIM, 2), stride=(self.o_stride, 1, self.wg_rows * self.o_stride))
        )
        work_count = mWorkCount[0]
        num_ctas = Int32(self.ctas)
        pair_queries = Int32(2 * self.chunk_queries)

        if warp_idx < 4:
            cute.arch.setmaxregister_decrease(self.producer_regs)
            if warp_idx == 0:
                q0_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.q_stages
                )
                q1_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.q_stages
                )
                kv_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.kv_stages
                )
                work_idx = cta
                while work_idx < work_count:
                    kv_head = mWork[work_idx, 0]
                    token_or_page = mWork[work_idx, 1]
                    list_offset = mWork[work_idx, 2]
                    count = mWork[work_idx, 3]
                    gK = _kv_blocks(mK, kv_head, token_or_page, self.paged)
                    gV = _kv_blocks(mV, kv_head, token_or_page, self.paged)
                    source = Int32(0)
                    if cutlass.const_expr(self.paged):
                        source = token_or_page
                    load_k, _, _ = copy_utils.tma_get_copy_fn(
                        tma_k, 0, cute.make_layout(1), gK, sK
                    )
                    load_v, _, _ = copy_utils.tma_get_copy_fn(
                        tma_v, 0, cute.make_layout(1), gV, sV
                    )
                    pipeline_k.producer_acquire(kv_state)
                    load_k(
                        src_idx=source,
                        dst_idx=kv_state.index,
                        tma_bar_ptr=pipeline_k.producer_get_barrier(kv_state),
                    )
                    pipeline_v.producer_acquire(kv_state)
                    load_v(
                        src_idx=source,
                        dst_idx=kv_state.index,
                        tma_bar_ptr=pipeline_v.producer_get_barrier(kv_state),
                    )
                    kv_state.advance()
                    mQ_head = mQ[None, None, kv_head]
                    pairs = (count + pair_queries - Int32(1)) // pair_queries
                    for p in cutlass.range(pairs, unroll=1):
                        self._load_chunk(
                            pipeline_q0, q0_state, sQ0, tma_q, mQ_head, mRelations,
                            list_offset, count, p * pair_queries, kv_head, total_q,
                        )
                        q0_state.advance()
                        self._load_chunk(
                            pipeline_q1, q1_state, sQ1, tma_q, mQ_head, mRelations,
                            list_offset, count,
                            p * pair_queries + Int32(self.chunk_queries), kv_head, total_q,
                        )
                        q1_state.advance()
                    work_idx += num_ctas
        else:
            cute.arch.setmaxregister_increase(self.mma_regs)
            mma_tidx = tidx - Int32(128)
            warp_group = cute.arch.make_warp_uniform(mma_tidx // Int32(128))
            wg_tidx = mma_tidx % Int32(128)
            if warp_group == Int32(0):
                self._consume(
                    tiled_mma_qk, tiled_mma_pv, pipeline_q0, pipeline_k, pipeline_v,
                    sQ0, sK, sV, sO_all[None, None, 0], mWork, mRelations, mPartialOut,
                    mPartialLse, work_count, cta, num_ctas, softmax_scale, wg_tidx, 0,
                )
            else:
                self._consume(
                    tiled_mma_qk, tiled_mma_pv, pipeline_q1, pipeline_k, pipeline_v,
                    sQ1, sK, sV, sO_all[None, None, 1], mWork, mRelations, mPartialOut,
                    mPartialLse, work_count, cta, num_ctas, softmax_scale, wg_tidx, 1,
                )

    @cute.jit
    def _load_chunk(
        self,
        pipeline_q: pipeline.PipelineAsync,
        q_state: pipeline.PipelineState,
        sQ: cute.Tensor,
        tma_q: cute.CopyAtom,
        mQ_head: cute.Tensor,
        mRelations: cute.Tensor,
        list_offset: Int32,
        count: Int32,
        chunk_begin: Int32,
        kv_head: Int32,
        total_q: Int32,
    ) -> None:
        """TMA-load the 8 query-head rows of each query in one chunk."""
        pipeline_q.producer_acquire(q_state)
        sQ_stage = sQ[None, None, q_state.index]
        for i in cutlass.range_constexpr(self.chunk_queries):
            slot = chunk_begin + Int32(i)
            if slot >= count:
                slot = count - Int32(1)
            relation = mRelations[list_offset + slot]
            query = relation // Int32(TOPK) - kv_head * total_q
            gQ = cute.local_tile(mQ_head, (GROUP, HEAD_DIM), (query, 0))
            sQ_rows = cute.local_tile(sQ_stage, (GROUP, HEAD_DIM), (i, 0))
            load_q, _, _ = copy_utils.tma_get_copy_fn(
                tma_q, 0, cute.make_layout(1), gQ, sQ_rows, single_stage=True
            )
            load_q(tma_bar_ptr=pipeline_q.producer_get_barrier(q_state))

    @cute.jit
    def _consume(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sO: cute.Tensor,
        mWork: cute.Tensor,
        mRelations: cute.Tensor,
        mPartialOut: cute.Tensor,
        mPartialLse: cute.Tensor,
        work_count: Int32,
        cta: Int32,
        num_ctas: Int32,
        softmax_scale: Float32,
        wg_tidx: Int32,
        warp_group: cutlass.Constexpr[int],
    ) -> None:
        thr_mma_qk = tiled_mma_qk.get_slice(wg_tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(wg_tidx)
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            thr_mma_qk, (self.wg_rows, BLOCK, HEAD_DIM), sQ, sK
        )
        acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
            thr_mma_pv, (self.wg_rows, HEAD_DIM, BLOCK), None, layout_utils.transpose_view(sV)
        )
        s_coords = layout_utils.reshape_acc_to_mn(
            thr_mma_qk.partition_C(cute.make_identity_tensor((self.wg_rows, BLOCK)))
        )
        rows = cute.size(s_coords, mode=[0])
        row_max = cute.make_rmem_tensor((rows,), Float32)
        row_sum = cute.make_rmem_tensor((rows,), Float32)
        row_scale = cute.make_rmem_tensor((rows,), Float32)
        row_lse = cute.make_rmem_tensor((rows,), Float32)
        scale_log2 = softmax_scale * Float32(LOG2E)
        lane = cute.arch.lane_idx()
        # 16-byte vector stores: 16 BF16 or 8 FP8 threads cover one 128-dim row.
        vector = 16 * 8 // self.partial_dtype.width
        row_threads = HEAD_DIM // vector
        chunk_col = wg_tidx % Int32(row_threads)
        barrier_id = 2 + warp_group
        pair_queries = Int32(2 * self.chunk_queries)

        q_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.q_stages)
        kv_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stages
        )
        work_idx = cta
        while work_idx < work_count:
            list_offset = mWork[work_idx, 2]
            count = mWork[work_idx, 3]
            pairs = (count + pair_queries - Int32(1)) // pair_queries
            pipeline_k.consumer_wait(kv_state)
            pipeline_v.consumer_wait(kv_state)
            for p in cutlass.range(pairs, unroll=1):
                chunk_begin = p * pair_queries + Int32(warp_group * self.chunk_queries)
                pipeline_q.consumer_wait(q_state)
                acc_S = sm90_utils.gemm_zero_init(
                    tiled_mma_qk,
                    (self.wg_rows, BLOCK),
                    tSrQ[None, None, None, q_state.index],
                    tSrK,
                    B_idx=kv_state.index,
                    wg_wait=0,
                )
                pipeline_q.consumer_release(q_state)
                q_state.advance()
                _softmax_rows(acc_S, row_max, row_sum, row_scale, scale_log2, True)
                tOrP.store(layout_utils.reshape_acc_to_frgA(acc_S).load().to(cutlass.BFloat16))
                sm90_utils.gemm_w_idx(
                    tiled_mma_pv, acc_O, tOrP, tOrVt, True, B_idx=kv_state.index, wg_wait=0
                )
                _finalize_rows(acc_O, row_max, row_sum, row_lse, softmax_scale)
                if chunk_begin < count:
                    # The previous chunk's copy-out must finish before sO is reused.
                    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
                    _stage_output(acc_O, sO, tiled_mma_pv, wg_tidx)
                    if lane % Int32(4) == Int32(0):
                        for r in cutlass.range_constexpr(rows):
                            row = s_coords[r, 0][0]
                            slot = chunk_begin + row // Int32(GROUP)
                            if slot < count:
                                relation = mRelations[list_offset + slot]
                                mPartialLse[relation, row % Int32(GROUP)] = row_lse[r]
                    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
                    rows_per_pass = 128 // row_threads
                    for pass_idx in cutlass.range_constexpr(self.wg_rows // rows_per_pass):
                        row = wg_tidx // Int32(row_threads) + Int32(rows_per_pass * pass_idx)
                        slot = chunk_begin + row // Int32(GROUP)
                        if slot < count:
                            relation = mRelations[list_offset + slot]
                            src = cute.local_tile(sO[row, None], (vector,), (chunk_col,))
                            dst = cute.local_tile(
                                mPartialOut[relation, row % Int32(GROUP), None],
                                (vector,),
                                (chunk_col,),
                            )
                            cute.autovec_copy(src, dst)
            pipeline_k.consumer_release(kv_state)
            pipeline_v.consumer_release(kv_state)
            kv_state.advance()
            work_idx += num_ctas


class CombineSm90:
    """Fold the fixed state and up to 16 dynamic partials per query head."""

    threads = 128

    def __init__(self, *, fp8_partial: bool = False) -> None:
        self.fp8_partial = bool(fp8_partial)

    @cute.jit
    def __call__(
        self,
        fixed_out: cute.Tensor,  # [Q, 32, 128]
        fixed_lse: cute.Tensor,  # [Q, 32]
        partial_out: cute.Tensor,  # [4 * Q * 16, 8, 128]
        partial_lse: cute.Tensor,  # [4 * Q * 16, 8]
        route: cute.Tensor,  # [4, Q, 16]
        out: cute.Tensor,
        lse: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        if cutlass.const_expr(self.fp8_partial):
            partial_out = cute.recast_tensor(partial_out, cutlass.Float8E4M3FN)
        self.kernel(fixed_out, fixed_lse, partial_out, partial_lse, route, out, lse).launch(
            grid=(fixed_out.shape[0], 1, 1),
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mFixedOut: cute.Tensor,
        mFixedLse: cute.Tensor,
        mPartialOut: cute.Tensor,
        mPartialLse: cute.Tensor,
        mRoute: cute.Tensor,
        mOut: cute.Tensor,
        mLse: cute.Tensor,
    ) -> None:
        query, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        total_q = mFixedOut.shape[0]
        head = tidx // Int32(4)
        part = tidx % Int32(4)
        kv_head = head // Int32(GROUP)
        group_row = head % Int32(GROUP)
        base_relation = (kv_head * total_q + query) * Int32(TOPK)

        lse_values = cute.make_rmem_tensor((TOPK,), Float32)
        running = mFixedLse[query, head]
        for slot in cutlass.range_constexpr(TOPK):
            value = -Float32.inf
            if mRoute[kv_head, query, slot] >= Int32(0):
                value = mPartialLse[base_relation + Int32(slot), group_row]
            lse_values[slot] = value
            running = cute.arch.fmax(running, value)
        weight = cute.math.exp2((mFixedLse[query, head] - running) * Float32(LOG2E), fastmath=True)
        total = weight
        acc = cute.make_rmem_tensor((32,), Float32)
        fixed_vec = cute.make_rmem_tensor((32,), cutlass.BFloat16)
        cute.autovec_copy(cute.local_tile(mFixedOut[query, head, None], (32,), (part,)), fixed_vec)
        acc.store(fixed_vec.load().to(Float32) * weight)
        partial_vec = cute.make_rmem_tensor((32,), mPartialOut.element_type)
        for slot in cutlass.range_constexpr(TOPK):
            if lse_values[slot] != -Float32.inf:
                w = cute.math.exp2((lse_values[slot] - running) * Float32(LOG2E), fastmath=True)
                total = total + w
                cute.autovec_copy(
                    cute.local_tile(
                        mPartialOut[base_relation + Int32(slot), group_row, None], (32,), (part,)
                    ),
                    partial_vec,
                )
                acc.store(acc.load() + partial_vec.load().to(Float32) * w)
        inv = cute.arch.rcp_approx(total)
        out_vec = cute.make_rmem_tensor((32,), cutlass.BFloat16)
        out_vec.store((acc.load() * inv).to(cutlass.BFloat16))
        cute.autovec_copy(out_vec, cute.local_tile(mOut[query, head, None], (32,), (part,)))
        if part == Int32(0):
            mLse[query, head] = running + cute.math.log2(total, fastmath=True) * Float32(LN2)
