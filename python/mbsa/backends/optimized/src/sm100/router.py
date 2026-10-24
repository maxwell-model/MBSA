# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SM100/SM103 BF16/FP16/FP8 router with native TMA/UMMA and exact Top-16 merge."""

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
from cutlass.base_dsl import BaseDSL
# CuTe 4.6 exposes Arch here; 4.7 retains the compatibility import.
from cutlass.base_dsl.arch import Arch
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack

from ..common import pipeline as common_pipeline
from ...index_policy import lean192_index_eligible, _validate_index_token_extent


_BLOCK = 128
_HEAD_DIM = 128
_TOPK = 16
_PREFIX_BLOCKS = 8
_LOCAL_BLOCKS = 8
_COMPILE_CACHE: dict[tuple[Any, ...], Any] = {}
_MERGE_COMPILE_CACHE: dict[tuple[Any, ...], Any] = {}


def _row_major(t: torch.Tensor) -> torch.Tensor:
    """``t`` with the canonical row-major strides of its shape."""
    if not t.is_contiguous():
        raise ValueError("runtime-extent operands must be contiguous")
    strides, step = [], 1
    for size in reversed(t.shape):
        strides.append(step)
        step *= size
    strides = tuple(reversed(strides))
    return t if t.stride() == strides else t.as_strided(t.shape, strides)


def _dynamic(t: torch.Tensor, mode: int, align: int = 16):
    """Wrap a contiguous tensor whose ``mode`` extent is a runtime value."""
    t = _row_major(t.detach())
    return from_dlpack(t, assumed_align=align).mark_compact_shape_dynamic(
        mode=mode, stride_order=tuple(range(t.dim()))
    )


def _local_top16_arguments(q, k, pages, scores, ids, route,
                         logical_k_tokens, route_token_mode):
    """Build the local router ABI; prepared owners can retain these wrappers."""
    tile_metadata = pages[:1].view(1, 1)
    return (
        _dynamic(q, 0), _dynamic(k, 0), _dynamic(pages, 0),
        from_dlpack(tile_metadata, assumed_align=4),
        _dynamic(scores, 2), _dynamic(ids, 2),
        _dynamic(route, route_token_mode), Int32(logical_k_tokens),
        Int32(logical_k_tokens - q.shape[0]),
    )


@cute.kernel
def merge_native_top16(scores, ids, route, compact_ids: cutlass.Constexpr):
    """Merge native UMMA scores without reading or decoding Q/K."""
    head, tile, _ = cute.arch.block_idx()
    tid = cute.arch.thread_idx()[0]
    lane = tid % 32
    query = tile * 4 + tid // 32
    if query < scores.shape[2]:
        value = -Float32.inf
        index = Int32(-1)
        if lane < 16:
            value = scores[head, 0, query, lane]
            index = ids[head, 0, query, lane]
        for part in cutlass.range_constexpr(1, scores.shape[1]):
            if lane >= 16:
                value = scores[head, part, query, 31 - lane]
                index = ids[head, part, query, 31 - lane]
            for stage in cutlass.range_constexpr(5):
                distance = 16 >> stage
                other_value = cute.arch.shuffle_sync_bfly(value, distance)
                other_index = cute.arch.shuffle_sync_bfly(index, distance)
                other_better = other_value > value or (
                    other_value == value
                    and other_index >= 0
                    and (index < 0 or other_index < index)
                )
                take = other_better if (lane & distance) == 0 else not other_better
                value = other_value if take else value
                index = other_index if take else index
        if lane < 16:
            if cutlass.const_expr(compact_ids):
                route[head, 16 + lane, query] = index
            else:
                route[head, query, lane] = index


class IndexerLocalTop16Sm100:
    """UMMA partition candidates followed by a native-score global Top-16 merge."""

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
        fuse_single_partition_compact: bool = False,
        use_page_table: bool = False,
        logical_k_tail: bool = False,
        packed_requests: bool = False,
        packed_partitions: int = 1,
        lean_pipeline: bool = False,
        reduced_acc_pipeline: bool = False,
        lean192_early: bool = False,
        query_tile: int = 128,
        early_acc_release: bool = False,
        input_dtype: Any = cutlass.BFloat16,
    ) -> None:
        if partition_blocks not in (1024, 2048, 4096, 8192):
            raise ValueError("partition_blocks must be 1024, 2048, 4096, or 8192")
        self.partition_blocks = int(partition_blocks)
        self.use_load_reduce = BaseDSL._get_dsl().get_arch_enum().is_family_of(Arch.sm_103f)
        self.input_dtype = input_dtype
        self.early_acc_release = bool(early_acc_release)
        self.direct_route = bool(direct_route)
        self.compact_ids = bool(compact_ids)
        self.fuse_single_partition_compact = bool(fuse_single_partition_compact)
        if self.fuse_single_partition_compact and (not compact_ids or packed_requests):
            raise ValueError("direct compact output requires an unpacked compact-ID call")
        self.use_page_table = bool(use_page_table)
        self.logical_k_tail = bool(logical_k_tail)
        self.packed_requests = bool(packed_requests)
        self.packed_partitions = int(packed_partitions)
        self.k_stages = 2 if lean_pipeline else 3
        reduced_footprint = lean_pipeline or reduced_acc_pipeline
        self.acc_stages = 2 if reduced_footprint else 3
        self.tmem_columns = 256 if reduced_footprint else 512
        self.lean192_early = bool(lean192_early and self.use_load_reduce)
        if query_tile not in (128, 256):
            raise ValueError('query_tile must be 128 or 256')
        if query_tile == 256:
            if packed_requests or lean192_early:
                raise ValueError('query_tile256 requires the nonpacked regular pipeline')
            self.q_tile = 256
            self.epilogue_threads = 256
            self.mma_warp_id = 8
            self.load_warp_id = 9
            # Eight epilogue warps plus one MMA warp and one TMA warp.
            self.threads = 320
            # Each accumulator stage holds two M128 tiles in 256 TMEM columns.
            self.acc_stages = 2
            self.tmem_columns = 512
        if self.lean192_early:
            if packed_requests or input_dtype != cutlass.BFloat16 or partition_blocks != 1024:
                raise ValueError("lean192_early requires nonpacked BF16 PB1024")
            self.threads = 192
            self.k_stages = 2
            self.acc_stages = 2
            self.tmem_columns = 256

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
        if cutlass.const_expr(
            (self.compact_ids or self.direct_route)
            and not self.fuse_single_partition_compact
        ):
            merge_native_top16(
                candidate_scores, candidate_ids, route, self.compact_ids
            ).launch(
                grid=(q_heads, cute.ceil_div(total_q, 4), 1),
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
        epi_tidx = tidx % Int32(128)
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
        # Keep the element type static rather than deriving it from TmaInfo,
        # which widens the DSL type to a shape/type union during compilation.
        dtype = self.input_dtype
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
        tmem_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.epilogue_threads + cute.arch.WARP_SIZE,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding.ptr,
            barrier_for_retrieve=tmem_barrier,
            allocator_warp_id=0,  # Preserve the original epilogue-warp allocator.
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
        q_producer, q_consumer = common_pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.q_mbar.data_ptr(),
            num_stages=self.q_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=q_bytes,
            defer_sync=True,
        ).make_participants()
        # UMMA consumes SMEM asynchronously. A thread-side arrive can release
        # a K stage before its MMA finishes, allowing TMA to overwrite it.
        # The UMMA pipeline commits the empty barrier after tensor-core use.
        k_producer, k_consumer = common_pipeline.PipelineTmaUmma.create(
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
                # TMA itself completes the transaction barrier. The
                # PipelineTmaUmma producer commit is a no-op.
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
                k_producer.tail()
                q_producer.tail()

        # Allocate collectively so CUTLASS tracks one TMEM ownership state.
        tmem.allocate(self.tmem_columns)
        if cutlass.const_expr(self.lean192_early):
            # Release the allocation right; retain the live TMEM until final free.
            tmem.relinquish_alloc_permit()
        tmem_release_ptr = cute.make_ptr(
            Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16
        )
        tmem_participates = warp_idx <= self.mma_warp_id
        if tmem_participates:
            # MMA and epilogue warps use one generated NamedBarrier wait PC.
            tmem.wait_for_alloc()
            tmem_release_ptr = tmem.retrieve_ptr(Float32)
            tCtAcc = cute.make_tensor(tmem_release_ptr, tCtAcc_fake.layout)
            if warp_idx == self.mma_warp_id:
                if has_work:
                    q_full = q_consumer.wait_and_advance()
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
                    q_full.release()
                    acc_producer.tail()

            if warp_idx < Int32(self.epilogue_threads // 32):
                top_scores = cute.make_rmem_tensor((_TOPK,), Float32)
                top_ids = cute.make_rmem_tensor((_TOPK,), Int32)
                for slot in cutlass.range_constexpr(_TOPK):
                    top_scores[slot] = -Float32.inf
                    top_ids[slot] = Int32(-1)

                epi_acc = tCtAcc
                epi_coords = tCcC
                if cutlass.const_expr(self.q_tile == 256):
                    assert cute.size(tCtAcc.shape[0]) == 128 * self.k_tile
                    assert tCtAcc.shape[1] == 2
                    assert tCtAcc.shape[2] == 1
                    # Each epilogue WG reads one M128 slice; both release
                    # the combined accumulator slot before MMA may reuse it.
                    epi_acc = cute.make_tensor(
                        tCtAcc.iterator + (tidx // Int32(128)) * tCtAcc.stride[1],
                        cute.make_layout(
                            (tCtAcc.shape[0], 1, tCtAcc.shape[2], tCtAcc.shape[3]),
                            stride=(tCtAcc.stride[0], 0,
                                    tCtAcc.stride[2],
                                    tCtAcc.stride[3])))
                    epi_coords = thr_mma.partition_C(
                        cute.make_identity_tensor((128, self.k_tile)))
                if cutlass.const_expr(self.lean192_early):
                    # Flatten only the existing TMEM layout; preserve its physical strides.
                    # Current UMMA C fragment is ((128,128),1,1,acc_stages).
                    flat_acc = cute.flatten(epi_acc)
                    assert cute.rank(flat_acc) == 5
                    assert cute.size(flat_acc, mode=[0]) == 128
                    assert cute.size(flat_acc, mode=[1]) == self.k_tile
                    assert cute.size(flat_acc, mode=[2]) == 1
                    assert cute.size(flat_acc, mode=[3]) == 1
                    acc_chunk0 = cute.local_tile(
                        flat_acc, (128, 32), (0, 0, 0, 0, 0)
                    )
                    load_atom = cute.make_copy_atom(
                        tcgen05.LdRed32x32bOp(
                            tcgen05.Repetition.x32,
                            tcgen05.Pack.NONE,
                            tcgen05.TmemLoadRedOp.MAX,
                        ),
                        Float32,
                    )
                    tiled_t2r = tcgen05.make_tmem_copy(load_atom, acc_chunk0)
                    copy_thread = tiled_t2r.get_slice(epi_tidx)
                    chunk_identity = cute.make_identity_tensor((128, 32))
                    thread_coords = copy_thread.partition_D(chunk_identity)
                    # All chunks reuse the same 32-column scratch registers.
                    registers = cute.make_rmem_tensor(thread_coords.shape, Float32)
                    row_reduced = cute.make_rmem_tensor((1,), Float32)
                else:
                    # SM100 lacks load-reduce; load the same row and reduce in registers.
                    if cutlass.const_expr(self.use_load_reduce):
                        load_op = tcgen05.LdRed32x32bOp(
                            tcgen05.Repetition.x128, tcgen05.Pack.NONE,
                            tcgen05.TmemLoadRedOp.MAX)
                    else:
                        load_op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
                    load_atom = cute.make_copy_atom(load_op, Float32)
                    tiled_t2r = tcgen05.make_tmem_copy(load_atom, epi_acc[(None, None, None, 0)])
                    copy_thread = tiled_t2r.get_slice(epi_tidx)
                    thread_tmem = copy_thread.partition_S(epi_acc)
                    thread_coords = copy_thread.partition_D(epi_coords)
                    registers = cute.make_rmem_tensor(thread_coords.shape, Float32)
                    row_reduced = cute.make_rmem_tensor((1,), Float32)
                q_local = q_local_start + tidx
                q_global = q_begin + q_local
                visible_index = Int32(0)
                if has_work:
                    for block in cutlass.range(partition_begin, partition_end, unroll=1):
                        stage = visible_index % Int32(self.acc_stages)
                        phase = (visible_index // Int32(self.acc_stages)) % Int32(2)
                        acc_pipeline.consumer_wait_w_index_phase(stage, phase)
                        if cutlass.const_expr(self.lean192_early):
                            row_max = Float32(0.0)
                            for chunk in cutlass.range_constexpr(self.k_tile // 32):
                                acc_chunk = cute.local_tile(
                                    flat_acc,
                                    (128, 32),
                                    (0, chunk, 0, 0, stage),
                                )
                                thread_chunk = copy_thread.partition_S(acc_chunk)
                                cute.copy(tiled_t2r, thread_chunk, [registers, row_reduced])
                                if cutlass.const_expr(chunk == 0):
                                    # Do not initialize with -inf: retain all-NaN behavior.
                                    row_max = row_reduced[0]
                                else:
                                    # Same finite-selecting FP32 max as ld.red without .NaN.
                                    row_max = cute.arch.fmax(row_max, row_reduced[0])
                                # Complete each chunk before reusing its scratch registers.
                                # Stage release remains below, after every chunk has loaded.
                                cute.arch.fence_view_async_tmem_load()
                        else:
                            if cutlass.const_expr(self.use_load_reduce):
                                cute.copy(
                                    tiled_t2r,
                                    thread_tmem[(None, None, None, None, stage)],
                                    [registers, row_reduced],
                                )
                                row_max = row_reduced[0]
                            else:
                                cute.copy(tiled_t2r,
                                          thread_tmem[(None, None, None, None, stage)], registers)
                                cute.arch.fence_view_async_tmem_load()
                                values = cute.make_tensor(registers.iterator,
                                                          cute.make_layout(cute.size(registers)))
                                row_max = values[0]
                                for column in cutlass.range_constexpr(1, cute.size(values)):
                                    row_max = cute.arch.fmax(row_max, values[column])
                        if cutlass.const_expr(self.early_acc_release):
                            cute.arch.fence_view_async_tmem_load()
                            acc_pipeline.consumer_release_w_index(stage)
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
                        # Scale FP32 block scores before Top-K to preserve reference rounding ties.
                        candidate_score = (
                            row_max * Float32(_HEAD_DIM**-0.5)
                            if dynamic_candidate
                            else -Float32.inf
                        )
                        candidate_id = block if dynamic_candidate else Int32(-1)
                        # Most later blocks cannot enter the current Top-16.
                        # Reject them before the insertion network. All lanes
                        # still join the TMEM completion/release below.
                        can_insert = candidate_score > top_scores[_TOPK - 1] or (
                            candidate_score == top_scores[_TOPK - 1]
                            and candidate_id >= Int32(0)
                            and (top_ids[_TOPK - 1] < Int32(0)
                                 or candidate_id < top_ids[_TOPK - 1])
                        )
                        if can_insert:
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
                        if cutlass.const_expr(not self.early_acc_release):
                            cute.arch.fence_view_async_tmem_load()
                            acc_pipeline.consumer_release_w_index(stage)
                        visible_index += Int32(1)

                if q_local < request_q_tokens:
                    for slot in cutlass.range_constexpr(_TOPK):
                        if cutlass.const_expr(self.fuse_single_partition_compact):
                            # One partition already owns the final ordered IDs.
                            # Publish every slot, including invalid -1 padding.
                            mRoute[hq, 16 + slot, q_global] = top_ids[slot]
                        else:
                            mCandidateScores[hq, partition_idx, q_global, slot] = top_scores[
                                slot
                            ]
                            mCandidateIds[hq, partition_idx, q_global, slot] = top_ids[slot]
            # Every TMEM user finishes at the same retirement instruction.
            tmem_barrier.arrive_and_wait()
        # The loader and idle warps also participate in the final CTA barrier.
        cute.arch.barrier()
        if cutlass.const_expr(not self.lean192_early):
            tmem.relinquish_alloc_permit()
        tmem.free(tmem_release_ptr, num_columns=self.tmem_columns)




def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    page_indices: torch.Tensor,
    logical_k_tokens: int,
) -> None:
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn) or k.dtype != q.dtype:
        raise TypeError("Indexer requires matching BF16, FP16 or FP8 E4M3FN Q/K")
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("Q and K must be [tokens,heads,128]")
    if q.shape[2] != _HEAD_DIM or k.shape[2] != _HEAD_DIM:
        raise ValueError("Indexer requires head dimension 128")
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


def indexer_local_top16(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    partition_blocks: int,
    page_indices: torch.Tensor | None = None,
    logical_k_tokens: int | None = None,
    use_page_table: bool | None = None,
    reduced_acc_pipeline: bool | None = None,
    _lean192_early: bool = False,
    _query_tile: int | None = None,
    out_scores: torch.Tensor | None = None,
    out_ids: torch.Tensor | None = None,
    out_route: torch.Tensor | None = None,
    out_compact_ids: torch.Tensor | None = None,
    _prepared_args: tuple | None = None,
    _prepare_only: bool = False,
    _fuse_single_partition_compact: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[Any, tuple[Any, ...]]:
    """Return native UMMA per-partition Top-16 scores and logical block ids."""

    if logical_k_tokens is None:
        logical_k_tokens = int(k.shape[0])
    else:
        logical_k_tokens = int(logical_k_tokens)
    _validate_index_token_extent(logical_k_tokens, partition_blocks)
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
    # Ownership/hardware are resolved during preparation, never per timed launch.
    _lean192_early = bool(
        _lean192_early and lean192_index_eligible(
            q, k, partition_blocks=partition_blocks,
            logical_k_tokens=logical_k_tokens, use_page_table=use_page_table)
    )
    if _query_tile is None:
        # Reuse each K tile for two M128 query slices in the qualified raw
        # E4M3 prefill range. PB2048 uses the wider SM103 tuning policy;
        # other architectures retain its previously supported context window.
        # Other shapes retain the Q128 geometry.
        _query_tile = 256 if (
            q.dtype == torch.float8_e4m3fn
            and tuple(q.shape) == (8192, 4, 128)
            and tuple(k.shape[1:]) == (1, 128)
            and (partition_blocks == 1024 or
                 (partition_blocks == 2048 and
                  (196608 <= logical_k_tokens <= 262144
                   or torch.cuda.get_device_capability(q.device) == (10, 3))))
            and 16384 <= logical_k_tokens <= 1048576
            and logical_k_tokens % 8192 == 0
            and not logical_k_tail and not use_page_table
        ) else 128
    # The long Q256 partition loop benefits from overlapping Top16 with the
    # next MMA. Short loops retain the qualified instruction schedule.
    early_acc_release = (
        _query_tile == 256 and q.dtype == torch.float8_e4m3fn
        and partition_blocks in (1024, 2048) and kv_blocks >= 1024
    )
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
    if _fuse_single_partition_compact and (not compact_ids or partitions != 1):
        raise ValueError("direct compact output requires exactly one KV partition")
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
        # These operands are dead only for the private single-partition fused
        # path. Otherwise they connect the partition producer to the merge.
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
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    # Route and compact IDs are [H,Q,16] and [H,>=32,Q]: Q is mode 1 or 2.
    route_token_mode = 1 if direct_route else 2
    args = _prepared_args if _prepared_args is not None else _local_top16_arguments(
        q, k, page_indices, scores, ids, route, logical_k_tokens, route_token_mode)
    key = (
        q.device.index,
        q.dtype,
        tuple(q.shape[1:]),
        tuple(k.shape[1:]),
        int(scores.shape[1]),
        int(route.shape[1]) if compact_ids else None,
        int(partition_blocks),
        direct_route,
        compact_ids,
        use_page_table,
        logical_k_tail,
        lean_pipeline,
        reduced_acc_pipeline,
    )
    if _lean192_early:
        key += (_lean192_early,)
    if _query_tile != 128:
        key += ('query_tile', _query_tile)
    if early_acc_release:
        key += ('early_acc_release', True)
    if _fuse_single_partition_compact:
        key += ('fuse_single_partition_compact', True)
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(
            IndexerLocalTop16Sm100(
                input_dtype=cutlass.Float8E4M3FN
                if q.dtype == torch.float8_e4m3fn
                else (cutlass.Float16 if q.dtype == torch.float16 else cutlass.BFloat16),
                partition_blocks=partition_blocks,
                direct_route=direct_route,
                compact_ids=compact_ids,
                fuse_single_partition_compact=_fuse_single_partition_compact,
                use_page_table=use_page_table,
                logical_k_tail=logical_k_tail,
                lean_pipeline=lean_pipeline,
                reduced_acc_pipeline=reduced_acc_pipeline,
                lean192_early=_lean192_early,
                query_tile=_query_tile,
                early_acc_release=early_acc_release,
            ),
            *args,
            stream,
        )
        _COMPILE_CACHE[key] = compiled
    if _prepare_only:
        # Prepared owners resolve immutable policy and compilation before
        # capture, but select the launch stream on every invocation.
        return compiled, key
    compiled(*args, stream)
    return scores, ids


def indexer_packed_top16(
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

    if q.dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn) or k.dtype != q.dtype:
        raise TypeError("packed Indexer requires matching BF16, FP16 or FP8 E4M3FN Q/K")
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
    args = (
        _dynamic(q, 0),
        _dynamic(k, 0),
        _dynamic(page_indices, 0, align=4),
        _dynamic(tile_metadata, 0, align=4),
        _dynamic(scores, 2),
        _dynamic(ids, 2),
        _dynamic(route, 2),
        Int32(k.shape[0]),
        Int32(0),
    )
    key = (
        "packed",
        q.device.index,
        q.dtype,
        tuple(q.shape[1:]),
        tuple(k.shape[1:]),
        int(route.shape[1]) if compact_ids else None,
        int(partition_blocks),
        partitions,
        compact_ids,
        bool(lean_pipeline),
    )
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(
            IndexerLocalTop16Sm100(
                input_dtype=cutlass.Float8E4M3FN
                if q.dtype == torch.float8_e4m3fn
                else (cutlass.Float16 if q.dtype == torch.float16 else cutlass.BFloat16),
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






__all__ = [
    "indexer_local_top16",
    "indexer_packed_top16",
]
