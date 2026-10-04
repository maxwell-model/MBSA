"""KV-centric schedule for SM90 Dynamic attention (device-only, no host sync).

Three SIMT kernels turn the route [4, Q, 16] into work items:

1. ``RouteHistogramSm90``: one thread per route entry atomically counts its
   (KV head, global KV block) segment and keeps the returned in-segment rank.
2. ``ScheduleScanSm90``: one CTA scans segment counts, splits every segment
   into equal work items of at most ``work_queries`` queries (KV head, block
   token start or physical page, list offset, count) and re-zeroes the
   counters for the next run. Paged K/V reads the live page table, so
   rebinding page maps needs no host-side rebuild.
3. ``ScheduleScatterSm90``: writes each route-entry ID to its CSR position.

In-segment order follows atomic arrival. Every partial state depends only on
its own query rows and block, so outputs remain deterministic.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32
from quack.utils import elem_pointer, warp_prefix_sum


TOPK = 16
BLOCK = 128


class RouteHistogramSm90:
    threads = 256

    @cute.jit
    def __call__(
        self,
        route: cute.Tensor,  # [R] int32 flattened [4, Q, 16]
        block_base_of_q: cute.Tensor,  # [Q] int32
        counts: cute.Tensor,  # [segments] int32, zero on entry
        entry_segment: cute.Tensor,  # [R] int32
        entry_rank: cute.Tensor,  # [R] int32
        total_blocks: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(route, block_base_of_q, counts, entry_segment, entry_rank, total_blocks).launch(
            grid=(cute.ceil_div(route.shape[0], self.threads), 1, 1),
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mRoute: cute.Tensor,
        mBlockBase: cute.Tensor,
        mCounts: cute.Tensor,
        mSegment: cute.Tensor,
        mRank: cute.Tensor,
        total_blocks: Int32,
    ) -> None:
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        entry = bidx * Int32(self.threads) + tidx
        if entry < mRoute.shape[0]:
            total_q = mBlockBase.shape[0]
            query_and_head = entry // Int32(TOPK)
            head = query_and_head // total_q
            query = query_and_head - head * total_q
            block = mRoute[entry]
            segment = Int32(-1)
            rank = Int32(0)
            if block >= Int32(0):
                segment = head * total_blocks + mBlockBase[query] + block
                rank = cute.arch.atomic_add(elem_pointer(mCounts, segment), Int32(1))
            mSegment[entry] = segment
            mRank[entry] = rank


class ScheduleScanSm90:
    threads = 1024

    def __init__(self, *, work_queries: int, paged: bool = False) -> None:
        self.work_queries = int(work_queries)
        self.paged = bool(paged)

    @cute.jit
    def __call__(
        self,
        counts: cute.Tensor,  # [segments] int32
        segment_start: cute.Tensor,  # [segments] int32
        block_token_start: cute.Tensor,  # [total_blocks] int32
        block_row: cute.Tensor,  # [total_blocks] int32 page-table row
        block_local: cute.Tensor,  # [total_blocks] int32 request-local block
        page_table: cute.Tensor,  # [rows, pages] int32
        work: cute.Tensor,  # [W, 4] int32
        work_count: cute.Tensor,  # [1] int32
        total_blocks: Int32,
        stream: cuda.CUstream,
    ) -> None:
        @cute.struct
        class SharedStorage:
            warp_counts: cute.struct.MemRange[Int32, 32]
            warp_items: cute.struct.MemRange[Int32, 32]

        self.shared_storage = SharedStorage
        self.kernel(
            counts, segment_start, block_token_start, block_row, block_local, page_table,
            work, work_count, total_blocks,
        ).launch(
            grid=(1, 1, 1),
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mCounts: cute.Tensor,
        mSegmentStart: cute.Tensor,
        mBlockTokenStart: cute.Tensor,
        mBlockRow: cute.Tensor,
        mBlockLocal: cute.Tensor,
        mPageTable: cute.Tensor,
        mWork: cute.Tensor,
        mWorkCount: cute.Tensor,
        total_blocks: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        warp_counts = storage.warp_counts.get_tensor(cute.make_layout(32))
        warp_items = storage.warp_items.get_tensor(cute.make_layout(32))
        segments = mCounts.shape[0]
        per_thread = (segments + Int32(self.threads - 1)) // Int32(self.threads)
        begin = tidx * per_thread
        end = begin + per_thread
        if end > segments:
            end = segments
        local_count = Int32(0)
        local_items = Int32(0)
        for s in cutlass.range(begin, end, unroll=1):
            n = mCounts[s]
            local_count += n
            local_items += (n + Int32(self.work_queries - 1)) // Int32(self.work_queries)
        inclusive_count = warp_prefix_sum(local_count, lane)
        inclusive_items = warp_prefix_sum(local_items, lane)
        if lane == Int32(31):
            warp_counts[warp] = inclusive_count
            warp_items[warp] = inclusive_items
        cute.arch.sync_threads()
        if warp == Int32(0):
            total_count = warp_prefix_sum(warp_counts[lane], lane)
            total_items = warp_prefix_sum(warp_items[lane], lane)
            warp_counts[lane] = total_count
            warp_items[lane] = total_items
        cute.arch.sync_threads()
        count_offset = inclusive_count - local_count
        item_offset = inclusive_items - local_items
        if warp > Int32(0):
            count_offset += warp_counts[warp - Int32(1)]
            item_offset += warp_items[warp - Int32(1)]
        if tidx == Int32(self.threads - 1):
            mWorkCount[0] = item_offset + local_items
        for s in cutlass.range(begin, end, unroll=1):
            n = mCounts[s]
            mCounts[s] = Int32(0)
            mSegmentStart[s] = count_offset
            head = s // total_blocks
            block = s - head * total_blocks
            token_start = Int32(0)
            if cutlass.const_expr(self.paged):
                if n > Int32(0):
                    token_start = mPageTable[mBlockRow[block], mBlockLocal[block]]
            else:
                token_start = mBlockTokenStart[block]
            # Split a segment into equal-sized items: no tiny tail work.
            items = (n + Int32(self.work_queries - 1)) // Int32(self.work_queries)
            for sub in cutlass.range(items, unroll=1):
                first = n * sub // items
                mWork[item_offset, 0] = head
                mWork[item_offset, 1] = token_start
                mWork[item_offset, 2] = count_offset + first
                mWork[item_offset, 3] = n * (sub + Int32(1)) // items - first
                item_offset += Int32(1)
            count_offset += n


class ScheduleScatterSm90:
    threads = 256

    @cute.jit
    def __call__(
        self,
        entry_segment: cute.Tensor,
        entry_rank: cute.Tensor,
        segment_start: cute.Tensor,
        relations: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(entry_segment, entry_rank, segment_start, relations).launch(
            grid=(cute.ceil_div(entry_segment.shape[0], self.threads), 1, 1),
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mSegment: cute.Tensor,
        mRank: cute.Tensor,
        mSegmentStart: cute.Tensor,
        mRelations: cute.Tensor,
    ) -> None:
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        entry = bidx * Int32(self.threads) + tidx
        if entry < mSegment.shape[0]:
            segment = mSegment[entry]
            if segment >= Int32(0):
                mRelations[mSegmentStart[segment] + mRank[entry]] = entry
