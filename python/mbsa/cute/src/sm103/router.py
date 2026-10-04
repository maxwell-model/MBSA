"""BF16 SM103 block-max router with HF-compatible boundary refinement.

TMA/UMMA scores request-local 128-token blocks and keeps partition Top-16
candidates. A block-score workspace preserves every potential cutoff contender.
A warp per query merges candidates and uses conservative FP32 error intervals
from L1(Q) and Linf(K-block) to detect membership uncertainty. Only uncertain
queries recompute boundary block maxima with serial FP32 FMA in HF dimension
order. ID-only production output retains provably selected blocks directly;
standalone partition-score output recomputes all retained scores for merging.

The resulting compact IDs feed the existing CSR/LPT and attention pipeline.
"""

from __future__ import annotations

from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from ...._vendor.msa.python.fmha_sm100.cute.src.common import (
    pipeline as common_pipeline,
)


_BLOCK = 128
_HEAD_DIM = 128
_TOPK = 16
_PREFIX_BLOCKS = 8
_LOCAL_BLOCKS = 8
_COMPILE_CACHE: dict[tuple[Any, ...], Any] = {}
_MERGE_COMPILE_CACHE: dict[tuple[Any, ...], Any] = {}


@dsl_user_op
def _serial_fma(a, b, c, *, loc=None, ip=None):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                a.ir_value(loc=loc, ip=ip),
                b.ir_value(loc=loc, ip=ip),
                c.ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class _KeyBlockBounds:
    @cute.jit
    def __call__(self, k: cute.Tensor, bounds: cute.Tensor, stream: cuda.CUstream):
        self.kernel(k, bounds).launch(
            grid=(k.shape[0] // 128, k.shape[1]), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, k: cute.Tensor, bounds: cute.Tensor):
        block, head, _ = cute.arch.block_idx()
        tid = cute.arch.thread_idx()[0]
        maximum = Float32(0)
        for i in cutlass.range(64, unroll=1):
            element = tid + i * 256
            value = Float32(k[block * 128 + element // 128, head, element % 128])
            maximum = cute.arch.fmax(maximum, cute.arch.fmax(value, -value))
        maximum = cute.arch.warp_reduction_max(maximum)
        shared = utils.SmemAllocator().allocate_tensor(Float32, cute.make_layout(8))
        if tid % 32 == 0:
            shared[tid // 32] = maximum
        cute.arch.barrier()
        if tid == 0:
            maximum = Float32(0)
            for i in cutlass.range_constexpr(8):
                maximum = cute.arch.fmax(maximum, shared[i])
            bounds[block, head] = maximum


_BOUND_COMPILE_CACHE: dict[tuple[Any, ...], Any] = {}


def _key_block_bounds(k):
    bounds = torch.empty(
        (k.shape[0] // 128, k.shape[1]), device=k.device, dtype=torch.float32
    )
    args = (
        from_dlpack(k.detach(), assumed_align=16),
        from_dlpack(bounds, assumed_align=16),
    )
    stream = cuda.CUstream(torch.cuda.current_stream(k.device).cuda_stream)
    key = (k.device.index, tuple(k.shape))
    compiled = _BOUND_COMPILE_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(_KeyBlockBounds(), *args, stream)
        _BOUND_COMPILE_CACHE[key] = compiled
    compiled(*args, stream)
    return bounds


class Bf16IndexerLocalTop16Sm103:
    """UMMA partition candidates followed by HF-compatible global refinement."""

    q_tile = 128
    k_tile = 128
    head_dim = 128
    q_stages = 1
    k_stages = 3
    acc_stages = 3
    epilogue_threads = 128
    mma_warp_id = 4
    load_warp_id = 5
    threads = 256
    tmem_columns = 512

    def __init__(
        self,
        *,
        partition_blocks: int,
        direct_route: bool = False,
        compact_ids: bool = False,
        use_page_table: bool = False,
        logical_k_tail: bool = False,
        packed_requests: bool = False,
        packed_partitions: int = 1,
        lean_pipeline: bool = False,
        reduced_acc_pipeline: bool = False,
    ) -> None:
        if partition_blocks not in (1024, 2048, 4096, 8192):
            raise ValueError("partition_blocks must be 1024, 2048, 4096, or 8192")
        self.partition_blocks = int(partition_blocks)
        self.direct_route = bool(direct_route)
        self.compact_ids = bool(compact_ids)
        self.use_page_table = bool(use_page_table)
        self.logical_k_tail = bool(logical_k_tail)
        self.packed_requests = bool(packed_requests)
        self.packed_partitions = int(packed_partitions)
        self.k_stages = 2 if lean_pipeline else 3
        reduced_footprint = lean_pipeline or reduced_acc_pipeline
        self.acc_stages = 2 if reduced_footprint else 3
        self.tmem_columns = 256 if reduced_footprint else 512

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        page_indices: cute.Tensor,
        tile_metadata: cute.Tensor,
        candidate_scores: cute.Tensor,
        candidate_ids: cute.Tensor,
        route: cute.Tensor,
        key_bounds: cute.Tensor,
        full_scores: cute.Tensor,
        logical_k_tokens: Int32,
        q_offset: Int32,
        stream: cuda.CUstream,
    ) -> None:
        dtype = q.element_type
        total_q = q.shape[0]
        q_heads = q.shape[1]
        physical_k_tokens = k.shape[0]
        k_heads = k.shape[1]
        q_mma = cute.make_tensor(
            q.iterator,
            cute.make_layout(
                (total_q, self.head_dim, q_heads),
                stride=(q_heads * self.head_dim, 1, self.head_dim),
            ),
        )
        k_mma = cute.make_tensor(
            k.iterator,
            cute.make_layout(
                (physical_k_tokens, self.head_dim, k_heads),
                stride=(k_heads * self.head_dim, 1, self.head_dim),
            ),
        )
        mma_tiler = (self.q_tile, self.k_tile, self.head_dim)
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            (128, self.k_tile),
        )
        q_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler, dtype, self.q_stages
        )
        k_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler, dtype, self.k_stages
        )
        cluster = cute.make_layout((1, 1, 1, 1))
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_op,
            q_mma,
            cute.slice_(q_smem_layout, (None, None, None, 0)),
            mma_tiler,
            tiled_mma,
            cluster.shape,
        )
        tma_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_op,
            k_mma,
            cute.slice_(k_smem_layout, (None, None, None, 0)),
            mma_tiler,
            tiled_mma,
            cluster.shape,
        )
        effective_k_tokens = physical_k_tokens
        if cutlass.const_expr(self.logical_k_tail):
            effective_k_tokens = logical_k_tokens
        kv_blocks = cute.ceil_div(effective_k_tokens, self.k_tile)
        partitions = cute.ceil_div(kv_blocks, self.partition_blocks)
        q_tiles = cute.ceil_div(total_q, self.q_tile)
        if cutlass.const_expr(self.packed_requests):
            q_tiles = tile_metadata.shape[0]
            partitions = self.packed_partitions
        self.kernel(
            tiled_mma,
            tma_q,
            tma_k,
            page_indices,
            tile_metadata,
            candidate_scores,
            candidate_ids,
            route,
            full_scores,
            q_smem_layout,
            k_smem_layout,
            q_heads,
            k_heads,
            total_q,
            effective_k_tokens,
            q_offset,
        ).launch(
            grid=(q_tiles, q_heads, partitions),
            block=[self.threads, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )
        self.refine(
            q,
            k,
            page_indices,
            tile_metadata,
            candidate_scores,
            candidate_ids,
            route,
            key_bounds,
            full_scores,
            effective_k_tokens,
        ).launch(
            grid=(
                cute.ceil_div(total_q, 4),
                q_heads,
                1 if self.compact_ids or self.direct_route else partitions,
            ),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_q: cpasync.TmaInfo,
        tma_k: cpasync.TmaInfo,
        mPageIndices: cute.Tensor,
        mTileMetadata: cute.Tensor,
        mCandidateScores: cute.Tensor,
        mCandidateIds: cute.Tensor,
        mRoute: cute.Tensor,
        mFullScores: cute.Tensor,
        q_smem_layout: cute.ComposedLayout,
        k_smem_layout: cute.ComposedLayout,
        q_heads: Int32,
        k_heads: Int32,
        total_q: Int32,
        logical_k_tokens: Int32,
        q_offset: Int32,
    ) -> None:
        q_tile_idx, hq, partition_idx = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        epi_tidx = tidx % Int32(self.epilogue_threads)
        hk = hq // (q_heads // k_heads)
        q_begin = Int32(0)
        q_end = total_q
        q_local_tile = q_tile_idx
        page_begin = Int32(0)
        request_k_tokens = logical_k_tokens
        request_partitions = (
            logical_k_tokens + Int32(self.k_tile * self.partition_blocks - 1)
        ) // Int32(self.k_tile * self.partition_blocks)
        if cutlass.const_expr(self.packed_requests):
            q_begin = mTileMetadata[q_tile_idx, 0]
            q_end = mTileMetadata[q_tile_idx, 1]
            q_local_tile = mTileMetadata[q_tile_idx, 2]
            page_begin = mTileMetadata[q_tile_idx, 3]
            request_k_tokens = mTileMetadata[q_tile_idx, 4]
            request_partitions = mTileMetadata[q_tile_idx, 5]
        request_q_tokens = q_end - q_begin
        q_local_start = q_local_tile * Int32(self.q_tile)
        q_tile_start = q_begin + q_local_start
        q_tile_last = q_local_start + Int32(self.q_tile - 1)
        if q_tile_last >= request_q_tokens:
            q_tile_last = request_q_tokens - Int32(1)
        kv_blocks = (request_k_tokens + Int32(self.k_tile - 1)) // Int32(self.k_tile)
        partition_begin = partition_idx * Int32(self.partition_blocks)
        partition_end = partition_begin + Int32(self.partition_blocks)
        if partition_end > kv_blocks:
            partition_end = kv_blocks
        request_q_offset = request_k_tokens - request_q_tokens
        max_query_position = request_q_offset + q_tile_last
        max_visible_blocks = (max_query_position + Int32(self.k_tile)) // Int32(
            self.k_tile
        )
        if partition_end > max_visible_blocks:
            partition_end = max_visible_blocks

        @cute.struct
        class SharedStorage:
            acc_mbar: cute.struct.MemRange[cutlass.Int64, self.acc_stages * 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            k_mbar: cute.struct.MemRange[cutlass.Int64, self.k_stages * 2]
            tmem_holding: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        # This kernel is deliberately specialized to the production BF16 path.
        # Keep the element type static inside the device body; deriving it from
        # TmaInfo currently widens the DSL type to a shape/type union and makes
        # the shared-memory allocator reject it during compilation.
        dtype = cutlass.BFloat16
        sQ = smem.allocate_tensor(
            dtype, q_smem_layout.outer, 128, swizzle=q_smem_layout.inner
        )
        sK = smem.allocate_tensor(
            dtype, k_smem_layout.outer, 128, swizzle=k_smem_layout.inner
        )

        thr_mma = tiled_mma.get_slice(0)
        tCsQ = thr_mma.partition_A(sQ)
        tCsK = thr_mma.partition_B(sK)
        mQ = tma_q.tma_tensor
        mK = tma_k.tma_tensor
        if cutlass.const_expr(self.packed_requests):
            mQ = cute.domain_offset((q_begin, 0, 0), mQ)
        gQ = cute.local_tile(
            mQ,
            (self.q_tile, self.head_dim),
            (None, None, None),
        )
        tCgQ = thr_mma.partition_A(gQ)
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_q.atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tCgQ, 0, 3),
        )
        gK = cute.local_tile(
            mK,
            (self.k_tile, self.head_dim),
            (None, None, None),
        )
        tCgK = thr_mma.partition_B(gK)
        tKsK, tKgK = cpasync.tma_partition(
            tma_k.atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tCgK, 0, 3),
        )

        tCrQ = tiled_mma.make_fragment_A(sQ)
        tCrK = tiled_mma.make_fragment_B(sK)
        identity = cute.make_identity_tensor((self.q_tile, self.k_tile))
        tCcC = thr_mma.partition_C(identity)
        acc_shape = tiled_mma.partition_shape_C((self.q_tile, self.k_tile))
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.acc_stages))
        tmem = utils.TmemAllocator(
            storage.tmem_holding.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(
                barrier_id=1,
                num_threads=32 * (self.mma_warp_id + 1),
            ),
        )

        acc_pipeline = common_pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=self.acc_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.epilogue_threads
            ),
            defer_sync=True,
        )
        acc_producer, _ = acc_pipeline.make_participants()
        q_bytes = cute.size_in_bytes(dtype, tma_q.smem_layout)
        k_bytes = cute.size_in_bytes(dtype, tma_k.smem_layout)
        q_producer, q_consumer = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_mbar.data_ptr(),
            num_stages=self.q_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=q_bytes,
            defer_sync=True,
        ).make_participants()
        k_producer, k_consumer = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.k_mbar.data_ptr(),
            num_stages=self.k_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=k_bytes,
            defer_sync=True,
        ).make_participants()
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        has_work = (
            partition_idx < request_partitions
            and partition_begin < partition_end
            and q_local_start < request_q_tokens
        )
        if warp_idx == self.load_warp_id:
            if has_work:
                q_empty = q_producer.acquire_and_advance()
                cute.copy(
                    tma_q.atom,
                    tQgQ[(None, q_local_tile, 0, hq)],
                    tQsQ[(None, q_empty.index)],
                    tma_bar_ptr=q_empty.barrier,
                )
                q_empty.commit()
                for block in cutlass.range(partition_begin, partition_end, unroll=1):
                    # ``block`` remains the logical route ID. Only the K load
                    # is translated through the logical-to-physical page map.
                    physical_block = block
                    if cutlass.const_expr(self.use_page_table):
                        physical_block = mPageIndices[page_begin + block]
                    k_empty = k_producer.acquire_and_advance()
                    cute.copy(
                        tma_k.atom,
                        tKgK[(None, physical_block, 0, hk)],
                        tKsK[(None, k_empty.index)],
                        tma_bar_ptr=k_empty.barrier,
                    )
                    k_empty.commit()
                k_producer.tail()
                q_producer.tail()

        if warp_idx == self.mma_warp_id:
            tmem_pool = tmem.reserve(self.tmem_columns)
            tCtAcc = tmem_pool.allocate_tensor(tCtAcc_fake.layout, Float32)
            if has_work:
                q_full = q_consumer.wait_and_advance()
                q_full.release()
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                q_coord = (None, None, None, 0)
                for block in cutlass.range(partition_begin, partition_end, unroll=1):
                    k_full = k_consumer.wait_and_advance()
                    acc_empty = acc_producer.acquire_and_advance()
                    k_coord = (None, None, None, k_full.index)
                    acc_stage = tCtAcc[(None, None, None, acc_empty.index)]
                    cute.gemm(
                        tiled_mma,
                        acc_stage,
                        tCrQ[q_coord],
                        tCrK[k_coord],
                        acc_stage,
                    )
                    acc_empty.commit()
                    k_full.release()
                acc_producer.tail()

        if warp_idx < Int32(4):
            top_scores = cute.make_rmem_tensor((_TOPK,), Float32)
            top_ids = cute.make_rmem_tensor((_TOPK,), Int32)
            for slot in cutlass.range_constexpr(_TOPK):
                top_scores[slot] = -Float32.inf
                top_ids[slot] = Int32(-1)

            tmem_pool = tmem.reserve(self.tmem_columns)
            tCtAcc = tmem_pool.allocate_tensor(tCtAcc_fake.layout, Float32)
            load_atom = cute.make_copy_atom(
                tcgen05.LdRed32x32bOp(
                    tcgen05.Repetition.x128,
                    tcgen05.Pack.NONE,
                    tcgen05.TmemLoadRedOp.MAX,
                ),
                Float32,
            )
            tiled_t2r = tcgen05.make_tmem_copy(load_atom, tCtAcc[(None, None, None, 0)])
            copy_thread = tiled_t2r.get_slice(epi_tidx)
            thread_tmem = copy_thread.partition_S(tCtAcc)
            thread_coords = copy_thread.partition_D(tCcC)
            registers = cute.make_rmem_tensor(thread_coords.shape, Float32)
            row_reduced = cute.make_rmem_tensor((1,), Float32)
            q_local = q_local_start + epi_tidx
            q_global = q_begin + q_local
            visible_index = Int32(0)
            if has_work:
                for block in cutlass.range(partition_begin, partition_end, unroll=1):
                    stage = visible_index % Int32(self.acc_stages)
                    phase = (visible_index // Int32(self.acc_stages)) % Int32(2)
                    acc_pipeline.consumer_wait_w_index_phase(stage, phase)
                    cute.copy(
                        tiled_t2r,
                        thread_tmem[(None, None, None, None, stage)],
                        [registers, row_reduced],
                    )
                    row_max = row_reduced[0]
                    query_position = request_q_offset + q_local
                    valid_blocks = (query_position + Int32(self.k_tile)) // Int32(
                        self.k_tile
                    )
                    local_begin = valid_blocks - Int32(_LOCAL_BLOCKS)
                    if local_begin < Int32(_PREFIX_BLOCKS):
                        local_begin = Int32(_PREFIX_BLOCKS)
                    dynamic_candidate = (
                        q_local < request_q_tokens
                        and block < valid_blocks
                        and block >= Int32(_PREFIX_BLOCKS)
                        and block < local_begin
                    )
                    # HF ranks rounded FP32 scaled logits. Scaling after
                    # selection would miss ties introduced by that rounding.
                    # A positive scale commutes with the block maximum, so
                    # apply it to the reduced score before Top-K insertion.
                    candidate_score = (
                        row_max * Float32(_HEAD_DIM**-0.5)
                        if dynamic_candidate
                        else -Float32.inf
                    )
                    candidate_id = block if dynamic_candidate else Int32(-1)
                    if q_local < request_q_tokens:
                        mFullScores[hq, block, q_global] = candidate_score
                    for slot in cutlass.range_constexpr(_TOPK):
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
                    cute.arch.fence_view_async_tmem_load()
                    acc_pipeline.consumer_release_w_index(stage)
                    visible_index += Int32(1)

            if q_local < request_q_tokens:
                for slot in cutlass.range_constexpr(_TOPK):
                    mCandidateScores[hq, partition_idx, q_global, slot] = top_scores[
                        slot
                    ]
                    mCandidateIds[hq, partition_idx, q_global, slot] = top_ids[slot]
            cute.arch.barrier()
            tmem.free(tmem_pool.base_ptr)

    @cute.kernel
    def refine(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        pages: cute.Tensor,
        metadata: cute.Tensor,
        scores: cute.Tensor,
        ids: cute.Tensor,
        route: cute.Tensor,
        bounds: cute.Tensor,
        full: cute.Tensor,
        logical_k_tokens: Int32,
    ):
        tile_group, head, partition = cute.arch.block_idx()
        tid = cute.arch.thread_idx()[0]
        lane = tid % 32
        warp = tid // 32
        query = tile_group * 4 + warp
        q_begin = Int32(0)
        q_end = Int32(q.shape[0])
        page_begin = Int32(0)
        kv_tokens = logical_k_tokens
        if cutlass.const_expr(self.packed_requests):
            # Tiles are sorted by packed query start. Locate the request for
            # this warp without launching padded groups for one-token tails.
            low = Int32(0)
            high = Int32(metadata.shape[0])
            while low < high:
                middle = (low + high) // 2
                start = metadata[middle, 0] + metadata[middle, 2] * 128
                if start <= query:
                    low = middle + 1
                else:
                    high = middle
            tile = low - 1
            q_begin = metadata[tile, 0]
            q_end = metadata[tile, 1]
            page_begin = metadata[tile, 3]
            kv_tokens = metadata[tile, 4]
        query_local = query - q_begin
        key_head = head // (q.shape[1] // k.shape[1])
        if query < q_end:
            query_position = kv_tokens - (q_end - q_begin) + query_local
            visible = (query_position + 128) // 128
            remote_end = visible - 8
            begin = cutlass.max(partition * self.partition_blocks, Int32(8))
            end = cutlass.min((partition + 1) * self.partition_blocks, remote_end)
            candidate_id = Int32(-1)
            value = -Float32.inf
            old_score = -Float32.inf
            old_id = Int32(-1)
            take = cutlass.Boolean(False)
            approximate_scores = cute.make_rmem_tensor((16,), Float32)
            approximate_ids = cute.make_rmem_tensor((16,), Int32)
            for slot in cutlass.range_constexpr(16):
                approximate_scores[slot] = scores[head, partition, query, slot]
                approximate_ids[slot] = ids[head, partition, query, slot]
            if cutlass.const_expr(self.compact_ids or self.direct_route):
                begin = Int32(8)
                end = remote_end
                for part in cutlass.range_constexpr(1, scores.shape[1]):
                    for slot in cutlass.range_constexpr(16):
                        value = scores[head, part, query, slot]
                        candidate_id = ids[head, part, query, slot]
                        for destination in cutlass.range_constexpr(16):
                            old_score = approximate_scores[destination]
                            old_id = approximate_ids[destination]
                            take = value > old_score or (
                                value == old_score
                                and candidate_id >= 0
                                and (old_id < 0 or candidate_id < old_id)
                            )
                            approximate_scores[destination] = (
                                value if take else old_score
                            )
                            approximate_ids[destination] = (
                                candidate_id if take else old_id
                            )
                            value = old_score if take else value
                            candidate_id = old_id if take else candidate_id
            cutoff = approximate_scores[15]
            cutoff_id = approximate_ids[15]
            q_l1 = Float32(0)
            for d in cutlass.range_constexpr(4):
                value = Float32(q[query, head, lane + d * 32])
                q_l1 += cute.arch.fmax(value, -value)
            q_l1 = cute.arch.warp_reduction_sum(q_l1)
            error_factor = (
                q_l1
                * Float32(4 * 128 * 2**-24 / (1 - 128 * 2**-24))
                * Float32(128**-0.5)
            )
            cutoff_error = Float32(0)
            if cutoff_id >= 0:
                physical = cutoff_id
                if cutlass.const_expr(self.use_page_table):
                    physical = pages[page_begin + cutoff_id]
                cutoff_error = error_factor * bounds[physical, key_head]
            # When no outside block can cross the cutoff interval, ID-only
            # output needs no scalar dot products at all.
            possible_count = Int32(0)
            for block in cutlass.range(begin, end, unroll=1):
                physical = block
                if cutlass.const_expr(self.use_page_table):
                    physical = pages[page_begin + block]
                error = error_factor * bounds[physical, key_head]
                if full[head, block, query] + error >= cutoff - cutoff_error:
                    possible_count += 1
            top_scores = cute.make_rmem_tensor((16,), Float32)
            top_ids = cute.make_rmem_tensor((16,), Int32)
            for slot in cutlass.range_constexpr(16):
                top_scores[slot] = -Float32.inf
                top_ids[slot] = Int32(-1)
            if cutlass.const_expr(self.compact_ids or self.direct_route):
                if possible_count <= 16:
                    for slot in cutlass.range_constexpr(16):
                        top_ids[slot] = approximate_ids[slot]
            needs_refinement = possible_count > 16
            if cutlass.const_expr(not (self.compact_ids or self.direct_route)):
                needs_refinement = True
            if needs_refinement:
                for block in cutlass.range(begin, end, unroll=1):
                    physical = block
                    if cutlass.const_expr(self.use_page_table):
                        physical = pages[page_begin + block]
                    error = error_factor * bounds[physical, key_head]
                    approximate = full[head, block, query]
                    if approximate + error >= cutoff - cutoff_error:
                        value = Float32.inf
                        stable = False
                        if cutlass.const_expr(self.compact_ids or self.direct_route):
                            stable = approximate - error > cutoff + cutoff_error
                        if not stable:
                            value = -Float32.inf
                            # A warp evaluates all 128 tokens in this block.
                            # Each scalar dot keeps the HF dimension order.
                            for token_group in cutlass.range_constexpr(4):
                                token = physical * 128 + token_group * 32 + lane
                                serial = Float32(0)
                                for d in cutlass.range(128, unroll=1):
                                    serial = _serial_fma(
                                        Float32(q[query, head, d]),
                                        Float32(k[token, key_head, d]),
                                        serial,
                                    )
                                value = cute.arch.fmax(value, serial)
                            value = cute.arch.warp_reduction_max(value) * Float32(
                                128**-0.5
                            )
                        candidate_id = block
                        for slot in cutlass.range_constexpr(16):
                            old_score = top_scores[slot]
                            old_id = top_ids[slot]
                            take = value > old_score or (
                                value == old_score
                                and (old_id < 0 or candidate_id < old_id)
                            )
                            top_scores[slot] = value if take else old_score
                            top_ids[slot] = candidate_id if take else old_id
                            value = old_score if take else value
                            candidate_id = old_id if take else candidate_id
            if lane == 0:
                for slot in cutlass.range_constexpr(16):
                    if cutlass.const_expr(self.compact_ids):
                        route[head, 16 + slot, query] = top_ids[slot]
                    elif cutlass.const_expr(self.direct_route):
                        route[head, query, slot] = top_ids[slot]
                    else:
                        scores[head, partition, query, slot] = top_scores[slot]
                        ids[head, partition, query, slot] = top_ids[slot]


class Bf16IndexerExactRouteSm103:
    """Merge partition candidates into an exact global Top-16."""

    threads = 128

    def __init__(self, *, compact_ids: bool = False) -> None:
        self.compact_ids = bool(compact_ids)

    @cute.jit
    def __call__(
        self,
        candidate_scores: cute.Tensor,
        candidate_ids: cute.Tensor,
        route: cute.Tensor,
        q_offset: Int32,
        kv_blocks: Int32,
        stream: cuda.CUstream,
    ) -> None:
        total_q = candidate_scores.shape[2]
        heads = candidate_scores.shape[0]
        self.kernel(
            candidate_scores,
            candidate_ids,
            route,
            q_offset,
            kv_blocks,
        ).launch(
            grid=(cute.ceil_div(total_q, self.threads), heads),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mCandidateScores: cute.Tensor,
        mCandidateIds: cute.Tensor,
        mRoute: cute.Tensor,
        q_offset: Int32,
        kv_blocks: Int32,
    ) -> None:
        block_q, head, _ = cute.arch.block_idx()
        tidx = cute.arch.thread_idx()[0]
        query = block_q * Int32(self.threads) + tidx

        if query < mCandidateScores.shape[2]:
            top_scores = cute.make_rmem_tensor((_TOPK,), Float32)
            top_ids = cute.make_rmem_tensor((_TOPK,), Int32)
            for slot in cutlass.range_constexpr(_TOPK):
                top_scores[slot] = -Float32.inf
                top_ids[slot] = Int32(-1)

            if cutlass.const_expr(mCandidateScores.shape[1] == 1):
                for slot in cutlass.range_constexpr(_TOPK):
                    top_scores[slot] = mCandidateScores[head, 0, query, slot]
                    top_ids[slot] = mCandidateIds[head, 0, query, slot]
            else:
                for partition in cutlass.range_constexpr(mCandidateScores.shape[1]):
                    for source_slot in cutlass.range_constexpr(_TOPK):
                        candidate_score = mCandidateScores[
                            head, partition, query, source_slot
                        ]
                        candidate_id = mCandidateIds[
                            head, partition, query, source_slot
                        ]
                        for destination in cutlass.range_constexpr(_TOPK):
                            old_score = top_scores[destination]
                            old_id = top_ids[destination]
                            take = candidate_score > old_score
                            take = take or (
                                candidate_score == old_score
                                and candidate_id >= Int32(0)
                                and (old_id < Int32(0) or candidate_id < old_id)
                            )
                            top_scores[destination] = (
                                candidate_score if take else old_score
                            )
                            top_ids[destination] = candidate_id if take else old_id
                            candidate_score = old_score if take else candidate_score
                            candidate_id = old_id if take else candidate_id

            if cutlass.const_expr(self.compact_ids):
                for route_slot in cutlass.range_constexpr(_TOPK):
                    mRoute[head, Int32(_TOPK + route_slot), query] = top_ids[route_slot]
            else:
                for route_slot in cutlass.range_constexpr(_TOPK):
                    mRoute[head, query, route_slot] = top_ids[route_slot]


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    page_indices: torch.Tensor,
    logical_k_tokens: int,
) -> None:
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise TypeError("BF16 Indexer requires BF16 Q/K")
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("Q and K must be [tokens,heads,128]")
    if q.shape[2] != _HEAD_DIM or k.shape[2] != _HEAD_DIM:
        raise ValueError("BF16 Indexer requires head dimension 128")
    if q.shape[1] % k.shape[1]:
        raise ValueError("query heads must be divisible by key heads")
    if not q.is_cuda or not k.is_cuda:
        raise ValueError("Q and K must be CUDA tensors")
    if not q.is_contiguous() or not k.is_contiguous():
        raise ValueError("Q and K must be contiguous")
    if k.shape[0] % _BLOCK:
        raise ValueError("physical K storage must be block aligned")
    if logical_k_tokens < q.shape[0] or logical_k_tokens <= 0:
        raise ValueError("logical K length must be positive and cover Q")
    logical_blocks = (logical_k_tokens + _BLOCK - 1) // _BLOCK
    physical_blocks = k.shape[0] // _BLOCK
    if logical_blocks > physical_blocks:
        raise ValueError("physical K storage does not cover logical K length")
    if (
        page_indices.ndim != 1
        or page_indices.dtype != torch.int32
        or page_indices.device != q.device
        or not page_indices.is_contiguous()
        or page_indices.numel() < logical_blocks
    ):
        raise ValueError(
            "page_indices must be contiguous CUDA int32 [>=logical K blocks]"
        )
    # Page-value bounds are validated once by the prepared production plan.
    # Do not introduce a device-to-host synchronization in this launch-time
    # wrapper: it is called on every replay and belongs inside the timed path.


def bf16_indexer_local_top16(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    partition_blocks: int,
    page_indices: torch.Tensor | None = None,
    logical_k_tokens: int | None = None,
    use_page_table: bool | None = None,
    reduced_acc_pipeline: bool | None = None,
    out_scores: torch.Tensor | None = None,
    out_ids: torch.Tensor | None = None,
    out_route: torch.Tensor | None = None,
    out_compact_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact per-partition dynamic Top-16 scores and logical block ids."""

    if logical_k_tokens is None:
        logical_k_tokens = int(k.shape[0])
    else:
        logical_k_tokens = int(logical_k_tokens)
    kv_blocks = (logical_k_tokens + _BLOCK - 1) // _BLOCK
    logical_k_tail = logical_k_tokens != int(k.shape[0])
    page_indices_provided = page_indices is not None
    if page_indices is None:
        page_indices = torch.arange(kv_blocks, dtype=torch.int32, device=q.device)
    if use_page_table is None:
        use_page_table = page_indices_provided
    else:
        use_page_table = bool(use_page_table)
    _validate_inputs(q, k, page_indices, logical_k_tokens)
    partitions = (kv_blocks + partition_blocks - 1) // partition_blocks
    lean_pipeline = logical_k_tokens <= 8192 and partitions == 1
    if reduced_acc_pipeline is None:
        reduced_acc_pipeline = (
            logical_k_tokens == 32768 and partitions == 1 and use_page_table
        )
    else:
        reduced_acc_pipeline = bool(reduced_acc_pipeline)
    if out_route is not None and out_compact_ids is not None:
        raise ValueError("out_route and out_compact_ids are mutually exclusive")
    output_shape = (q.shape[1], partitions, q.shape[0], _TOPK)
    compact_ids = out_compact_ids is not None
    if compact_ids:
        if (
            out_compact_ids.ndim != 3
            or out_compact_ids.shape[0] != q.shape[1]
            or out_compact_ids.shape[1] < 2 * _TOPK
            or out_compact_ids.shape[2] != q.shape[0]
            or out_compact_ids.dtype != torch.int32
            or out_compact_ids.device != q.device
            or not out_compact_ids.is_contiguous()
        ):
            raise ValueError("out_compact_ids must be contiguous CUDA int32 [H,>=32,Q]")
        # These operands are compile-time dead in compact-ID mode.  Reuse the
        # caller-owned backing storage instead of allocating candidate arrays.
        ids = (
            out_ids
            if out_ids is not None
            else torch.empty(output_shape, device=q.device, dtype=torch.int32)
        )
        scores = (
            out_scores
            if out_scores is not None
            else torch.empty(output_shape, device=q.device, dtype=torch.float32)
        )
    else:
        scores = out_scores
        if scores is None:
            scores = torch.empty(
                output_shape,
                dtype=torch.float32,
                device=q.device,
            )
        elif (
            tuple(scores.shape) != output_shape
            or scores.dtype != torch.float32
            or scores.device != q.device
            or not scores.is_contiguous()
        ):
            raise ValueError(f"out_scores must be contiguous CUDA FP32 {output_shape}")
        ids = out_ids
        if ids is None:
            ids = torch.empty(
                output_shape,
                dtype=torch.int32,
                device=q.device,
            )
        elif (
            tuple(ids.shape) != output_shape
            or ids.dtype != torch.int32
            or ids.device != q.device
            or not ids.is_contiguous()
        ):
            raise ValueError(f"out_ids must be contiguous CUDA int32 {output_shape}")
    direct_route = out_route is not None
    if direct_route:
        route_shape = (q.shape[1], q.shape[0], _TOPK)
        if (
            tuple(out_route.shape) != route_shape
            or out_route.dtype != torch.int32
            or out_route.device != q.device
            or not out_route.is_contiguous()
        ):
            raise ValueError(f"out_route must be contiguous CUDA int32 {route_shape}")
        route = out_route
    elif compact_ids:
        route = out_compact_ids
    else:
        # The device branch is compile-time disabled; reuse an existing int32
        # tensor so the generic call ABI needs no throwaway allocation.
        route = ids
    q_offset = logical_k_tokens - q.shape[0]
    tile_metadata = page_indices[:1].view(1, 1)
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    key_bounds = _key_block_bounds(k)
    full_scores = torch.empty(
        (q.shape[1], kv_blocks, q.shape[0]), device=q.device, dtype=torch.float32
    )
    args = (
        from_dlpack(q.detach(), assumed_align=16),
        from_dlpack(k.detach(), assumed_align=16),
        from_dlpack(page_indices, assumed_align=16),
        from_dlpack(tile_metadata, assumed_align=4),
        from_dlpack(scores, assumed_align=16),
        from_dlpack(ids, assumed_align=16),
        from_dlpack(route, assumed_align=16),
        from_dlpack(key_bounds, assumed_align=16),
        from_dlpack(full_scores, assumed_align=16),
        Int32(logical_k_tokens),
        Int32(q_offset),
    )
    key = (
        q.device.index,
        tuple(q.shape),
        tuple(k.shape),
        tuple(page_indices.shape),
        int(partition_blocks),
        direct_route,
        compact_ids,
        use_page_table,
        logical_k_tail,
        lean_pipeline,
        reduced_acc_pipeline,
    )
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(
            Bf16IndexerLocalTop16Sm103(
                partition_blocks=partition_blocks,
                direct_route=direct_route,
                compact_ids=compact_ids,
                use_page_table=use_page_table,
                logical_k_tail=logical_k_tail,
                lean_pipeline=lean_pipeline,
                reduced_acc_pipeline=reduced_acc_pipeline,
            ),
            *args,
            stream,
        )
        _COMPILE_CACHE[key] = compiled
    compiled(*args, stream)
    return scores, ids


def bf16_indexer_packed_top16(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    partition_blocks: int,
    page_indices: torch.Tensor,
    tile_metadata: torch.Tensor,
    partitions: int,
    out_compact_ids: torch.Tensor | None = None,
    out_scores: torch.Tensor | None = None,
    out_ids: torch.Tensor | None = None,
    lean_pipeline: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one Router launch over request-local tiles in a packed Q tensor."""

    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise TypeError("packed BF16 Indexer requires BF16 Q/K")
    if (
        q.ndim != 3
        or k.ndim != 3
        or q.shape[2] != _HEAD_DIM
        or k.shape[2] != _HEAD_DIM
        or q.shape[1] % k.shape[1]
    ):
        raise ValueError("packed Q/K must be [tokens,heads,128]")
    if (
        not q.is_cuda
        or not k.is_cuda
        or q.device != k.device
        or not q.is_contiguous()
        or not k.is_contiguous()
        or k.shape[0] % _BLOCK
    ):
        raise ValueError("packed Q/K must be contiguous CUDA tensors")
    if (
        page_indices.ndim != 1
        or page_indices.dtype != torch.int32
        or page_indices.device != q.device
        or not page_indices.is_contiguous()
    ):
        raise ValueError("packed page_indices must be contiguous CUDA int32")
    if (
        tile_metadata.ndim != 2
        or tile_metadata.shape[1] != 6
        or tile_metadata.dtype != torch.int32
        or tile_metadata.device != q.device
        or not tile_metadata.is_contiguous()
        or tile_metadata.shape[0] == 0
    ):
        raise ValueError("tile_metadata must be contiguous CUDA int32 [tiles,6]")
    partitions = int(partitions)
    if partitions <= 0:
        raise ValueError("partitions must be positive")
    compact_ids = out_compact_ids is not None
    output_shape = (q.shape[1], partitions, q.shape[0], _TOPK)
    if compact_ids:
        if (
            out_compact_ids.ndim != 3
            or out_compact_ids.shape[0] != q.shape[1]
            or out_compact_ids.shape[1] < 2 * _TOPK
            or out_compact_ids.shape[2] != q.shape[0]
            or out_compact_ids.dtype != torch.int32
            or out_compact_ids.device != q.device
            or not out_compact_ids.is_contiguous()
        ):
            raise ValueError("out_compact_ids must be contiguous CUDA int32 [H,>=32,Q]")
        ids = (
            out_ids
            if out_ids is not None
            else torch.empty(output_shape, device=q.device, dtype=torch.int32)
        )
        scores = (
            out_scores
            if out_scores is not None
            else torch.empty(output_shape, device=q.device, dtype=torch.float32)
        )
        route = out_compact_ids
    else:
        scores = out_scores
        if scores is None:
            scores = torch.empty(output_shape, dtype=torch.float32, device=q.device)
        ids = out_ids
        if ids is None:
            ids = torch.empty(output_shape, dtype=torch.int32, device=q.device)
        if (
            tuple(scores.shape) != output_shape
            or scores.dtype != torch.float32
            or scores.device != q.device
            or not scores.is_contiguous()
            or tuple(ids.shape) != output_shape
            or ids.dtype != torch.int32
            or ids.device != q.device
            or not ids.is_contiguous()
        ):
            raise ValueError("packed candidate workspaces have invalid layout")
        route = ids

    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    key_bounds = _key_block_bounds(k)
    full_scores = torch.empty(
        (q.shape[1], partitions * partition_blocks, q.shape[0]),
        device=q.device,
        dtype=torch.float32,
    )
    args = (
        from_dlpack(q.detach(), assumed_align=16),
        from_dlpack(k.detach(), assumed_align=16),
        from_dlpack(page_indices, assumed_align=4),
        from_dlpack(tile_metadata, assumed_align=4),
        from_dlpack(scores, assumed_align=16),
        from_dlpack(ids, assumed_align=16),
        from_dlpack(route, assumed_align=16),
        from_dlpack(key_bounds, assumed_align=16),
        from_dlpack(full_scores, assumed_align=16),
        Int32(k.shape[0]),
        Int32(0),
    )
    key = (
        "packed",
        q.device.index,
        tuple(q.shape),
        tuple(k.shape),
        tuple(page_indices.shape),
        tuple(tile_metadata.shape),
        int(partition_blocks),
        partitions,
        compact_ids,
        bool(lean_pipeline),
    )
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(
            Bf16IndexerLocalTop16Sm103(
                partition_blocks=partition_blocks,
                direct_route=False,
                compact_ids=compact_ids,
                use_page_table=True,
                packed_requests=True,
                packed_partitions=partitions,
                lean_pipeline=lean_pipeline,
            ),
            *args,
            stream,
        )
        _COMPILE_CACHE[key] = compiled
    compiled(*args, stream)
    return scores, ids


def merge_bf16_indexer_exact_route(
    candidate_scores: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    q_offset: int,
    kv_blocks: int,
    out: torch.Tensor | None = None,
    out_compact_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Merge partition Top-16 lists into route or compact-ID storage."""

    if (
        candidate_scores.ndim != 4
        or candidate_scores.dtype != torch.float32
        or not candidate_scores.is_cuda
        or not candidate_scores.is_contiguous()
    ):
        raise ValueError("candidate_scores must be contiguous CUDA FP32 [H,P,Q,16]")
    if (
        candidate_ids.shape != candidate_scores.shape
        or candidate_ids.dtype != torch.int32
        or candidate_ids.device != candidate_scores.device
        or not candidate_ids.is_contiguous()
    ):
        raise ValueError("candidate_ids must be contiguous CUDA int32 and match scores")
    if int(candidate_scores.shape[3]) != _TOPK:
        raise ValueError("the candidate list width must be 16")
    if out is not None and out_compact_ids is not None:
        raise ValueError("out and out_compact_ids are mutually exclusive")
    compact_ids = out_compact_ids is not None
    if compact_ids:
        if (
            out_compact_ids.ndim != 3
            or out_compact_ids.shape[0] != candidate_scores.shape[0]
            or out_compact_ids.shape[1] < 2 * _TOPK
            or out_compact_ids.shape[2] != candidate_scores.shape[2]
            or out_compact_ids.dtype != torch.int32
            or out_compact_ids.device != candidate_scores.device
            or not out_compact_ids.is_contiguous()
        ):
            raise ValueError("out_compact_ids must be contiguous CUDA int32 [H,>=32,Q]")
        route = out_compact_ids
    else:
        route_shape = (
            candidate_scores.shape[0],
            candidate_scores.shape[2],
            _TOPK,
        )
        route = out
        if route is None:
            route = torch.empty(
                route_shape,
                dtype=torch.int32,
                device=candidate_scores.device,
            )
        elif (
            tuple(route.shape) != route_shape
            or route.dtype != torch.int32
            or route.device != candidate_scores.device
            or not route.is_contiguous()
        ):
            raise ValueError(f"out must be contiguous CUDA int32 {route_shape}")
    stream = cuda.CUstream(
        torch.cuda.current_stream(candidate_scores.device).cuda_stream
    )
    args = (
        from_dlpack(candidate_scores, assumed_align=16),
        from_dlpack(candidate_ids, assumed_align=16),
        from_dlpack(route, assumed_align=16),
        Int32(q_offset),
        Int32(kv_blocks),
    )
    key = (
        candidate_scores.device.index,
        tuple(candidate_scores.shape),
        compact_ids,
    )
    compiled = _MERGE_COMPILE_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(
            Bf16IndexerExactRouteSm103(compact_ids=compact_ids),
            *args,
            stream,
        )
        _MERGE_COMPILE_CACHE[key] = compiled
    compiled(*args, stream)
    return route


def bf16_indexer_exact_route(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    partition_blocks: int,
    page_indices: torch.Tensor | None = None,
    logical_k_tokens: int | None = None,
    use_page_table: bool | None = None,
) -> torch.Tensor:
    """Run the specialized Score/Top-16 pipeline and exact route merge."""

    if logical_k_tokens is None:
        logical_k_tokens = int(k.shape[0])
    kv_blocks = (int(logical_k_tokens) + _BLOCK - 1) // _BLOCK
    partitions = (kv_blocks + partition_blocks - 1) // partition_blocks
    if partitions == 1:
        route = torch.empty(
            (q.shape[1], q.shape[0], _TOPK),
            dtype=torch.int32,
            device=q.device,
        )
        bf16_indexer_local_top16(
            q,
            k,
            partition_blocks=partition_blocks,
            page_indices=page_indices,
            logical_k_tokens=logical_k_tokens,
            use_page_table=use_page_table,
            out_route=route,
        )
        return route
    scores, ids = bf16_indexer_local_top16(
        q,
        k,
        partition_blocks=partition_blocks,
        page_indices=page_indices,
        logical_k_tokens=logical_k_tokens,
        use_page_table=use_page_table,
    )
    return merge_bf16_indexer_exact_route(
        scores,
        ids,
        q_offset=int(logical_k_tokens - q.shape[0]),
        kv_blocks=kv_blocks,
    )


__all__ = [
    "bf16_indexer_exact_route",
    "bf16_indexer_local_top16",
    "bf16_indexer_packed_top16",
    "merge_bf16_indexer_exact_route",
]
