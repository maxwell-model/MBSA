// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

// CUDA C++ q2k -> k2q CSR builder.
//
// Five-stage pipeline. q-ascending order within each CSR row is preserved
// by partitioning q across (CTA, warp_in_CTA) units; each unit owns a
// contiguous q-sub-range and reserves a contiguous slot range per row via
// a precomputed exclusive prefix scan.
//
//   M:  build_row_map      -- round-robin packing of rows across batches
//   H:  histogram + tile_counts
//   PR: row prefix         -- single block per head, row_counts -> row_ptr
//   PT: tile prefix        -- multi-block, scan tile_counts along (c, w) axis
//   S:  scatter (sorted)   -- per-warp slot range, q-sequential within warp
//
// Per-warp partitioning: each CTA has kWarps warps; warp w of CTA c owns
// q-range [c*q_per_cta + w*q_per_warp, c*q_per_cta + (w+1)*q_per_warp).
// tile_counts is shaped [G * kWarps, H, total_rows]; the "row" dimension
// of the prefix scan is the flattened (c * kWarps + w) index, scanned in
// lexicographic order so that warp-local slot ranges concatenate to the
// global q-sorted output.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#ifndef AT_CUDA_CHECK
#define AT_CUDA_CHECK C10_CUDA_CHECK
#endif
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INT(x) TORCH_CHECK((x).scalar_type() == at::kInt, #x " must be int32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_INT(x)

namespace {

constexpr int kWarpSize = 32;

__device__ __forceinline__ void advance_batch_only(
    int const* __restrict__ cu_q, int B, int q_abs, int& bi)
{
    while (bi < B && cu_q[bi + 1] <= q_abs) ++bi;
}

// Atomic increment of a 16-bit half within a 32-bit SMEM word; returns the
// OLD 16-bit value (slot). Per-warp count must stay < 32768 so the low
// half does not carry into the high half.
//   base_int32 : int32 pointer; element i holds rows 2*i (low) and 2*i+1 (high).
__device__ __forceinline__ int atomic_inc_int16_packed(
    int* base_int32, int row)
{
    int idx = row >> 1;
    int shift = (row & 1) << 4;  // 0 or 16
    int delta = 1 << shift;
    int old = atomicAdd(&base_int32[idx], delta);
    return (old >> shift) & 0xFFFF;
}

// Reserve one same-row group, then assign slots in increasing lane/query order.
// All valid lanes call the matching collectives; no atomic return order is assumed.
__device__ __forceinline__ int warp_ordered_slot(
    int* base_int32, int row, unsigned int valid_mask, int lane)
{
    unsigned int peers = __match_any_sync(valid_mask, row);
    int leader = __ffs(peers) - 1;
    int base = 0;
    if (lane == leader) {
        int shift = (row & 1) << 4;
        int old = atomicAdd(&base_int32[row >> 1], __popc(peers) << shift);
        base = (old >> shift) & 0xFFFF;
    }
    base = __shfl_sync(valid_mask, base, leader);
    unsigned int lower = lane == 0 ? 0u : ((1u << lane) - 1u);
    return base + __popc(peers & lower);
}

// Read 16-bit half from packed int32 storage.
__device__ __forceinline__ int read_int16_packed(int const* base_int32, int row) {
    int v = base_int32[row >> 1];
    int shift = (row & 1) << 4;
    return (v >> shift) & 0xFFFF;
}

// ---------------------------------------------------------------------------
// M: round-robin row map.
// ---------------------------------------------------------------------------
template <int kBlockK>
__global__ void k2q_build_row_map_kernel(
    int const* __restrict__ cu_k,
    int* __restrict__ row_map,
    int* __restrict__ row_coords,
    int B,
    int max_kv_blocks)
{
    int level = blockIdx.x;
    if (level >= max_kv_blocks) return;
    if (threadIdx.x != 0) return;
    int rows_before = 0;
    for (int b = 0; b < B; ++b) {
        int rb = (cu_k[b + 1] - cu_k[b] + kBlockK - 1) / kBlockK;
        rows_before += (rb < level ? rb : level);
    }
    int active_before = 0;
    for (int b = 0; b < B; ++b) {
        int rb = (cu_k[b + 1] - cu_k[b] + kBlockK - 1) / kBlockK;
        if (rb > level) {
            int row_linear = rows_before + active_before;
            row_map[(size_t)b * max_kv_blocks + level] = row_linear;
            if (row_coords != nullptr) {
                row_coords[(size_t)row_linear * 2] = b;
                row_coords[(size_t)row_linear * 2 + 1] = level;
            }
            ++active_before;
        } else {
            row_map[(size_t)b * max_kv_blocks + level] = -1;
        }
    }
}

// ---------------------------------------------------------------------------
// H: per-warp histogram + tile_counts.
// kWarps warps per CTA, each owns q-sub-range = q_per_cta / kWarps.
// SMEM hist[kWarps, total_rows] int32 (stored as packed int16 cursor:
// 2 entries per int32 word). Each warp counts to its own row.
// At end-of-CTA, write tile_counts[c*kWarps + w, h, r] = smem_hist[w, r]
// and atomicAdd(row_counts[h, r], sum over w of smem_hist[w, r]).
// A batch with more rows than one CTA's counters hold is split into row
// ranges of range_rows rows (grid y, kRowRanges): CTA (c, y) reads the same
// queries and counts only the rows of range y, so every (c, w, h, r) is still
// written exactly once. Without kRowRanges the kernel is the unsplit one.
// ---------------------------------------------------------------------------
template <int kTopK, int kBlockK, int kWarps, bool kFromLocalTop16 = false,
          bool kBatch1IdentityRows = false, bool kDynamicOnly = false,
          bool kCanonicalizeDynamicPadding = false, bool kRowRanges = false>
__global__ void k2q_hist_kernel(
    int* __restrict__ q2k,
    int const* __restrict__ cu_q,
    int const* __restrict__ row_map,
    int* __restrict__ row_counts,
    int* __restrict__ tile_counts,
    int const* __restrict__ local_candidates,
    int const* __restrict__ prefix_lens,
    int H, int B, int S_Q,
    int total_rows, int max_kv_blocks,
    int q_per_cta, int q_per_warp,
    int candidate_stride_h, int candidate_stride_k, int candidate_stride_q,
    int range_rows)
{
    constexpr int kThreads = kWarps * kWarpSize;
    int row_begin = 0;
    int row_count = total_rows;
    if constexpr (kRowRanges) {
        row_begin = blockIdx.y * range_rows;
        row_count = min(range_rows, total_rows - row_begin);
    }
    extern __shared__ int smem_hist_int[];
    int* smem_hist = smem_hist_int;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int c = blockIdx.x;
    int q_start_cta = c * q_per_cta;
    int q_end_cta = min(q_start_cta + q_per_cta, S_Q);
    int q_start_warp = min(q_start_cta + warp_id * q_per_warp, q_end_cta);
    int q_end_warp = min(q_start_warp + q_per_warp, q_end_cta);

    constexpr int kInt4PerToken = kTopK / 4;
    int packed_per_warp = ((kRowRanges ? range_rows : total_rows) + 1) >> 1;
    int* my_hist = smem_hist + warp_id * packed_per_warp;

    for (int h = 0; h < H; ++h) {
        for (int i = lane; i < packed_per_warp; i += kWarpSize) my_hist[i] = 0;
        __syncthreads();

        if (q_start_warp < q_end_warp) {
            int bi = 0;
            int qi = q_start_warp + lane;
            advance_batch_only(cu_q, B, qi, bi);

            int4* head_topk4 =
                reinterpret_cast<int4*>(q2k + (size_t)h * S_Q * kTopK);

            for (; qi < q_end_warp; qi += kWarpSize) {
                advance_batch_only(cu_q, B, qi, bi);
                int const* my_row_map = nullptr;
                if constexpr (!kBatch1IdentityRows) {
                    my_row_map = row_map + (size_t)bi * max_kv_blocks;
                }

                int4 buf[kInt4PerToken];
                if constexpr (kFromLocalTop16) {
                    int* buf_i = reinterpret_cast<int*>(buf);
                    if constexpr (kDynamicOnly) {
                        int dynamic_count = kTopK;
                        if constexpr (kCanonicalizeDynamicPadding) {
                            int q_local = qi - cu_q[bi];
                            int valid_blocks =
                                (prefix_lens[bi] + q_local + kBlockK) / kBlockK;
                            int prefix_count = min(8, valid_blocks);
                            int local_start = max(0, valid_blocks - 8);
                            int nonoverlap_local_start = max(local_start, prefix_count);
                            int fixed_count =
                                prefix_count + valid_blocks - nonoverlap_local_start;
                            dynamic_count = min(32, valid_blocks) - fixed_count;
                        }
                        // The second 16-slot Score plane is the exact dynamic
                        // set with prefix/SWA excluded by the frozen producer.
                        #pragma unroll
                        for (int t = 0; t < kTopK; ++t) {
                            buf_i[t] = t < dynamic_count
                                ? local_candidates[
                                    (size_t)h * candidate_stride_h +
                                    (size_t)(16 + t) * candidate_stride_k +
                                    (size_t)qi * candidate_stride_q]
                                : -1;
                        }
                    } else {
                        int q_local = qi - cu_q[bi];
                        int valid_blocks =
                            (prefix_lens[bi] + q_local + kBlockK) / kBlockK;
                        int prefix_count = min(8, valid_blocks);
                        int local_start = max(0, valid_blocks - 8);
                        int nonoverlap_local_start = max(local_start, prefix_count);
                        int local_count = valid_blocks - nonoverlap_local_start;
                        int target_count = min(kTopK, valid_blocks);
                        int fixed_count = prefix_count + local_count;
                        #pragma unroll
                        for (int t = 0; t < kTopK; ++t) {
                            int kvb = -1;
                            if (t < prefix_count) {
                                kvb = t;
                            } else if (t < fixed_count) {
                                kvb = nonoverlap_local_start + (t - prefix_count);
                            } else if (t < target_count) {
                                int dynamic_slot = t - fixed_count;
                                kvb = local_candidates[
                                    (size_t)h * candidate_stride_h +
                                    (size_t)(16 + dynamic_slot) * candidate_stride_k +
                                    (size_t)qi * candidate_stride_q];
                            }
                            buf_i[t] = kvb;
                        }
                    }
                    #pragma unroll
                    for (int v = 0; v < kInt4PerToken; ++v) {
                        head_topk4[(size_t)qi * kInt4PerToken + v] = buf[v];
                    }
                } else {
                    #pragma unroll
                    for (int v = 0; v < kInt4PerToken; ++v) {
                        buf[v] = head_topk4[(size_t)qi * kInt4PerToken + v];
                    }
                }
                #pragma unroll
                for (int t = 0; t < kTopK; ++t) {
                    int kvb_local = reinterpret_cast<int const*>(buf)[t];
                    if (kvb_local >= 0 && kvb_local < max_kv_blocks) {
                        int row;
                        if constexpr (kBatch1IdentityRows) {
                            row = kvb_local;
                        } else {
                            row = my_row_map[kvb_local];
                        }
                        if (row >= row_begin && row < row_begin + row_count) {
                            atomic_inc_int16_packed(my_hist, row - row_begin);
                        }
                    }
                }
            }
        }
        __syncthreads();

        int* head_row_counts = row_counts + (size_t)h * total_rows;
        // Each warp writes its own slice of tile_counts (full int32) by
        // unpacking int16 entries from SMEM.
        int* my_tile = tile_counts +
            ((size_t)(c * kWarps + warp_id) * H + h) * total_rows;
        for (int i = lane; i < row_count; i += kWarpSize) {
            my_tile[row_begin + i] = read_int16_packed(my_hist, i);
        }
        __syncthreads();

        // Sum across warps (int32 accumulator), atomicAdd to row_counts.
        for (int i = tid; i < row_count; i += kThreads) {
            int sum = 0;
            #pragma unroll
            for (int w = 0; w < kWarps; ++w) {
                sum += read_int16_packed(smem_hist + w * packed_per_warp, i);
            }
            if (sum > 0) atomicAdd(&head_row_counts[row_begin + i], sum);
        }
        if (h + 1 < H) __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// PR: row prefix. One block per head.
// ---------------------------------------------------------------------------
template <int kThreads, bool kBatch1IdentityRows = false,
          bool kCompactBatch1Schedule = false>
__global__ void k2q_row_prefix_kernel(
    int const* __restrict__ row_counts,
    int* __restrict__ row_ptr,
    int const* __restrict__ row_coords,
    int* __restrict__ scheduler_metadata,
    int* __restrict__ work_count,
    int total_rows,
    int target_q_per_cta,
    int work_capacity)
{
    int h = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ int scan_buf[kThreads];

    int const* head_counts = row_counts + (size_t)h * total_rows;
    int* head_rowptr = row_ptr + (size_t)h * (total_rows + 1);
    int chunk = (total_rows + kThreads - 1) / kThreads;
    int lo = tid * chunk;
    int hi = min(lo + chunk, total_rows);

    int local_sum = 0;
    for (int i = lo; i < hi; ++i) local_sum += head_counts[i];
    scan_buf[tid] = local_sum;
    __syncthreads();

    for (int off = 1; off < kThreads; off <<= 1) {
        int add = (tid >= off) ? scan_buf[tid - off] : 0;
        __syncthreads();
        scan_buf[tid] += add;
        __syncthreads();
    }
    int running = scan_buf[tid] - local_sum;
    for (int i = lo; i < hi; ++i) {
        int row_count = head_counts[i];
        int row_start = running;
        running += row_count;
        head_rowptr[i + 1] = running;
        if (scheduler_metadata != nullptr && work_count != nullptr && row_count > 0) {
            int num_chunks = (row_count + target_q_per_cta - 1) / target_q_per_cta;
            int base = atomicAdd(work_count, num_chunks);
            int batch_idx;
            int kv_block_idx;
            if constexpr (kBatch1IdentityRows) {
                batch_idx = 0;
                kv_block_idx = i;
            } else {
                batch_idx = row_coords[(size_t)i * 2];
                kv_block_idx = row_coords[(size_t)i * 2 + 1];
            }
            for (int c = 0; c < num_chunks; ++c) {
                int work_idx = base + c;
                if (work_idx < work_capacity) {
                    int q_begin = c * target_q_per_cta;
                    int q_count = min(target_q_per_cta, row_count - q_begin);
                    if constexpr (kCompactBatch1Schedule) {
                        // Batch-one identity-row specialization.  The first
                        // word carries the 17-bit absolute CSR start; the
                        // second carries {head:2, logical_kv_block:13,
                        // q_count-1:12}.
                        // The host gate bounds Sq*TopK, target_q_per_cta and
                        // total_rows so every field is lossless.
                        uint32_t word0 = (uint32_t)(row_start + q_begin);
                        uint32_t word1 = (uint32_t)(q_count - 1)
                            | ((uint32_t)kv_block_idx << 12)
                            | ((uint32_t)h << 25);
                        uint64_t record = (uint64_t)word0
                            | ((uint64_t)word1 << 32);
                        reinterpret_cast<uint64_t*>(scheduler_metadata)[work_idx] = record;
                    } else {
                        int* meta = scheduler_metadata + (size_t)work_idx * 6;
                        meta[0] = h;
                        meta[1] = i;
                        meta[2] = q_begin;
                        meta[3] = q_count;
                        meta[4] = batch_idx;
                        meta[5] = kv_block_idx;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// C305: approximate LPT ordering for sparse-FWD work records.
//
// The schedule is an internal execution order: changing it does not change the
// CSR, qsplit ownership, selected blocks, or numerical reduction order.  A
// single CTA groups records into 16 descending q-count buckets.  The fixed
// bucket layout keeps the device-side cost bounded while putting expensive
// CTAs in the first hardware wave.  Source and destination never alias.
// ---------------------------------------------------------------------------
__global__ void reorder_schedule_qcount_bucket16_kernel(
    int const* __restrict__ source,
    int* __restrict__ destination,
    int const* __restrict__ work_count,
    int work_capacity,
    int metadata_cols,
    int target_q_per_cta)
{
    __shared__ int bucket_counts[16];
    __shared__ int bucket_offsets[16];
    int tid = threadIdx.x;
    if (tid < 16) bucket_counts[tid] = 0;
    __syncthreads();

    int count = min(max(work_count[0], 0), work_capacity);
    for (int i = tid; i < count; i += blockDim.x) {
        int q_count;
        if (metadata_cols == 2) {
            uint32_t packed = (uint32_t)source[(size_t)i * 2 + 1];
            q_count = (int)(packed & 0xFFFu) + 1;
        } else {
            q_count = source[(size_t)i * 6 + 3];
        }
        int bucket = ((max(q_count, 1) - 1) * 16) / target_q_per_cta;
        bucket = min(max(bucket, 0), 15);
        atomicAdd(bucket_counts + bucket, 1);
    }
    __syncthreads();

    if (tid == 0) {
        int offset = 0;
        #pragma unroll
        for (int bucket = 15; bucket >= 0; --bucket) {
            bucket_offsets[bucket] = offset;
            offset += bucket_counts[bucket];
            bucket_counts[bucket] = 0;
        }
    }
    __syncthreads();

    for (int i = tid; i < count; i += blockDim.x) {
        int q_count;
        if (metadata_cols == 2) {
            uint32_t packed = (uint32_t)source[(size_t)i * 2 + 1];
            q_count = (int)(packed & 0xFFFu) + 1;
        } else {
            q_count = source[(size_t)i * 6 + 3];
        }
        int bucket = ((max(q_count, 1) - 1) * 16) / target_q_per_cta;
        bucket = min(max(bucket, 0), 15);
        int output_index = bucket_offsets[bucket] + atomicAdd(bucket_counts + bucket, 1);
        if (metadata_cols == 2) {
            reinterpret_cast<uint64_t*>(destination)[output_index] =
                reinterpret_cast<uint64_t const*>(source)[i];
        } else {
            #pragma unroll
            for (int column = 0; column < 6; ++column) {
                destination[(size_t)output_index * 6 + column] =
                    source[(size_t)i * 6 + column];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PT_smem: SMEM-staged tile prefix scan.
// Each block handles kRowsPerBlock rows for one head h. Cooperative load
// of tile_counts[*, h, base_r..base_r+M) into SMEM (better coalescing
// than per-warp uncoalesced stride reads), then per-warp scan in SMEM,
// then cooperative store back. Fuses row_ptr into the base.
// ---------------------------------------------------------------------------
template <int kThreads, int kRowsPerBlock>
__global__ void k2q_tile_prefix_smem_kernel(
    int* __restrict__ tile_counts,
    int const* __restrict__ row_ptr,
    int H, int total_rows, int G_total)
{
    static_assert(kRowsPerBlock > 0, "kRowsPerBlock must be positive");
    extern __shared__ int smem_tprefix[];
    // smem layout: smem[r_off][g] for r_off in [0, M), g in [0, G_total).

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    // Grid: H * blocks_per_h. Each block stays within a single head h
    // and processes kRowsPerBlock contiguous rows starting at b_in_h *
    // kRowsPerBlock. (Earlier flat-grid mapping `h = block_job /
    // total_rows; base_r = block_job - h*total_rows` skipped rows when
    // total_rows was not a multiple of kRowsPerBlock and H > 1, because
    // the last partial block of head h-1 left blocks of head h starting
    // at a non-zero row offset.)
    int blocks_per_h = (total_rows + kRowsPerBlock - 1) / kRowsPerBlock;
    int h = blockIdx.x / blocks_per_h;
    int b_in_h = blockIdx.x - h * blocks_per_h;
    if (h >= H) return;
    int base_r = b_in_h * kRowsPerBlock;
    if (base_r >= total_rows) return;
    int actual_M = min(kRowsPerBlock, total_rows - base_r);

    size_t stride_g = (size_t)H * total_rows;
    int* base_ptr = tile_counts + (size_t)h * total_rows + base_r;
    int total_elems = G_total * actual_M;

    // Cooperative load. Pattern: thread tid -> (r_off=tid%M, g=tid/M),
    // then strided. 32 lanes hit M r's x (32/M) g's, giving 32/M cache
    // lines per warp (vs 32 in the naive stride-along-g pattern).
    for (int i = tid; i < total_elems; i += kThreads) {
        int r_off = i % actual_M;
        int g = i / actual_M;
        smem_tprefix[r_off * G_total + g] = base_ptr[g * stride_g + r_off];
    }
    __syncthreads();

    // Per-warp scan: warp w scans row (base_r + w) if w < actual_M.
    if (warp_id < actual_M) {
        int abs_r = base_r + warp_id;
        int rp = row_ptr[(size_t)h * (total_rows + 1) + abs_r];
        int* my_smem = smem_tprefix + warp_id * G_total;
        int running = rp;
        for (int g0 = 0; g0 < G_total; g0 += kWarpSize) {
            int g = g0 + lane;
            int v = (g < G_total) ? my_smem[g] : 0;
            int x = v;
            #pragma unroll
            for (int off = 1; off < kWarpSize; off <<= 1) {
                int nbr = __shfl_up_sync(0xFFFFFFFF, x, off);
                if (lane >= off) x += nbr;
            }
            int excl = running + x - v;
            if (g < G_total) my_smem[g] = excl;
            int chunk_sum = __shfl_sync(0xFFFFFFFF, x, 31);
            running += chunk_sum;
        }
    }
    __syncthreads();

    // Cooperative store back.
    for (int i = tid; i < total_elems; i += kThreads) {
        int r_off = i % actual_M;
        int g = i / actual_M;
        base_ptr[g * stride_g + r_off] = smem_tprefix[r_off * G_total + g];
    }
}

// ---------------------------------------------------------------------------
// S: scatter. kWarps warps per CTA, each owns q-sub-range. Per-warp SMEM
// cursor and per-warp tile_offset slot range. Within a warp, q's are
// processed sequentially; lanes 0..kTopK-1 handle the topK slots in
// lockstep. Across distinct q's in the same warp, the lockstep ordering
// guarantees q-monotonic atomicAdd on smem_cursor[r].
// ---------------------------------------------------------------------------
// kQPerIter * kTopK lanes are active per warp iter; remaining lanes idle.
// For kTopK=16, kQPerIter=2 uses all 32 lanes; for kTopK=8, kQPerIter=4.
// Same-row lanes reserve one range and assign lane-ordered slots explicitly.
// Iterations, warp ranges and CTA prefix ranges preserve increasing query order.
//
// SMEM cursor stored as packed int16 (two cursors per int32). Per-warp
// row count must stay < 32768 (~q_per_warp * kTopK at max sink), which
// holds for all task.md sizes up to 1024K.
template <int kTopK, int kBlockK, int kWarps,
          bool kBatch1IdentityRows = false, bool kRowRanges = false>
__global__ void k2q_scatter_kernel(
    int const* __restrict__ q2k,
    int const* __restrict__ cu_q,
    int const* __restrict__ row_map,
    int const* __restrict__ abs_base,
    int* __restrict__ q_idx,
    int* __restrict__ qsplit_idx,
    int* __restrict__ split_counts,
    int H, int B, int S_Q,
    int total_rows, int max_kv_blocks,
    int q_per_cta, int q_per_warp,
    int max_seqlen_q, int range_rows)
{
    constexpr int kQPerIter = kWarpSize / kTopK > 0 ? kWarpSize / kTopK : 1;
    // Row range of this CTA (grid y, kRowRanges), as in the histogram: it scatters
    // only the edges whose row it owns; split slots still count every edge of a query.
    int row_begin = 0;
    int row_end = total_rows;
    if constexpr (kRowRanges) {
        row_begin = blockIdx.y * range_rows;
        row_end = min(row_begin + range_rows, total_rows);
    }
    extern __shared__ int smem_cursor_int[];
    int* smem_cursor = smem_cursor_int;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int c = blockIdx.x;
    int q_start_cta = c * q_per_cta;
    int q_end_cta = min(q_start_cta + q_per_cta, S_Q);
    int q_start_warp = min(q_start_cta + warp_id * q_per_warp, q_end_cta);
    int q_end_warp = min(q_start_warp + q_per_warp, q_end_cta);

    int q_in_iter = lane / kTopK;
    int slot_in_q = lane % kTopK;
    bool lane_active = (lane < kQPerIter * kTopK);

    // Per-warp packed cursor: one int16 per row of the CTA's range -> ceil(rows/2) int32.
    int packed_per_warp = ((kRowRanges ? range_rows : total_rows) + 1) >> 1;
    int* my_cursor = smem_cursor + warp_id * packed_per_warp;

    for (int h = 0; h < H; ++h) {
        for (int i = lane; i < packed_per_warp; i += kWarpSize) my_cursor[i] = 0;
        __syncwarp();

        if (q_start_warp < q_end_warp) {
            int bi = 0;
            advance_batch_only(cu_q, B, q_start_warp, bi);

            int const* head_q2k = q2k + (size_t)h * S_Q * kTopK;
            int const* my_abs_base =
                abs_base + ((size_t)(c * kWarps + warp_id) * H + h) * total_rows;
            int* head_qidx = q_idx + (size_t)h * S_Q * kTopK;

            // (Hot-row register cache experiment showed no measurable
            // benefit; relying on L1 to keep row 0 / row total_rows-1
            // hot since they're hit every iteration in sink workloads.)

            constexpr int kUnroll = 16;
            int qi_base = q_start_warp;
            for (; qi_base + kUnroll * kQPerIter <= q_end_warp;
                 qi_base += kUnroll * kQPerIter) {
                int kvb[kUnroll];
                int qloc[kUnroll];
                int batch[kUnroll];
                int const* rmap[kUnroll];

                #pragma unroll
                for (int u = 0; u < kUnroll; ++u) {
                    int qi_u = qi_base + u * kQPerIter + q_in_iter;
                    kvb[u] = -1;
                    qloc[u] = 0;
                    batch[u] = 0;
                    if (lane_active) {
                        advance_batch_only(cu_q, B, qi_u, bi);
                        qloc[u] = qi_u - cu_q[bi];
                        batch[u] = bi;
                        kvb[u] = head_q2k[(size_t)qi_u * kTopK + slot_in_q];
                    }
                    if constexpr (kBatch1IdentityRows) {
                        rmap[u] = nullptr;
                    } else {
                        rmap[u] = row_map + (size_t)bi * max_kv_blocks;
                    }
                }

                int row[kUnroll];
                #pragma unroll
                for (int u = 0; u < kUnroll; ++u) {
                    row[u] = -1;
                    if (lane_active && kvb[u] >= 0 && kvb[u] < max_kv_blocks) {
                        if constexpr (kBatch1IdentityRows) {
                            row[u] = kvb[u];
                        } else {
                            row[u] = rmap[u][kvb[u]];
                        }
                    }
                }

                // Pre-issue all kUnroll abs_base loads in parallel before
                // the atomic chain so memory pipeline runs concurrently
                // with SMEM atomic-adds.
                int abs_v[kUnroll];
                #pragma unroll
                for (int u = 0; u < kUnroll; ++u) {
                    abs_v[u] = (row[u] >= 0 && row[u] < total_rows)
                        ? my_abs_base[row[u]] : 0;
                }

                #pragma unroll
                for (int u = 0; u < kUnroll; ++u) {
                    int r = row[u];
                    bool valid_edge = r >= 0 && r < total_rows;
                    unsigned int valid_mask = __ballot_sync(0xFFFFFFFFu, valid_edge);
                    unsigned int group_mask = (kTopK == 32)
                        ? 0xFFFFFFFFu
                        : (((1u << kTopK) - 1u) << (q_in_iter * kTopK));
                    unsigned int lower_lane_mask = lane == 0 ? 0u : ((1u << lane) - 1u);
                    int split_slot = __popc(valid_mask & group_mask & lower_lane_mask);
                    int valid_count = __popc(valid_mask & group_mask);
                    if (split_counts != nullptr && slot_in_q == 0) {
                        int q_abs = cu_q[batch[u]] + qloc[u];
                        split_counts[(size_t)q_abs * H + h] = valid_count;
                    }
                    bool owned_edge = valid_edge;
                    unsigned int owned_mask = valid_mask;
                    if constexpr (kRowRanges) {
                        owned_edge = valid_edge && r >= row_begin && r < row_end;
                        owned_mask = __ballot_sync(0xFFFFFFFFu, owned_edge);
                    }
                    if (owned_edge) {
                        int slot = warp_ordered_slot(my_cursor, r - row_begin, owned_mask, lane);
                        int out_pos = abs_v[u] + slot;
                        head_qidx[out_pos] = qloc[u];
                        if (qsplit_idx != nullptr) {
                            qsplit_idx[(size_t)h * S_Q * kTopK + out_pos] =
                                qloc[u] | ((split_slot & 0xFF) << 24);
                        }
                    }
                }
            }
            // Tail: 1-3 iters left.
            for (; qi_base < q_end_warp; qi_base += kQPerIter) {
                int my_qi = qi_base + q_in_iter;
                bool valid_q = (my_qi < q_end_warp) && lane_active;
                int kvb_local = -1;
                int q_local = 0;
                int batch_local = 0;
                if (valid_q) {
                    advance_batch_only(cu_q, B, my_qi, bi);
                    batch_local = bi;
                    q_local = my_qi - cu_q[bi];
                    kvb_local = head_q2k[(size_t)my_qi * kTopK + slot_in_q];
                }
                int const* my_row_map = nullptr;
                if constexpr (!kBatch1IdentityRows) {
                    my_row_map = row_map + (size_t)bi * max_kv_blocks;
                }
                int row = -1;
                if (valid_q && kvb_local >= 0 && kvb_local < max_kv_blocks) {
                    if constexpr (kBatch1IdentityRows) {
                        row = kvb_local;
                    } else {
                        row = my_row_map[kvb_local];
                    }
                }
                bool valid_edge = row >= 0 && row < total_rows;
                unsigned int valid_mask = __ballot_sync(0xFFFFFFFFu, valid_edge);
                unsigned int group_mask = (kTopK == 32)
                    ? 0xFFFFFFFFu
                    : (((1u << kTopK) - 1u) << (q_in_iter * kTopK));
                unsigned int lower_lane_mask = lane == 0 ? 0u : ((1u << lane) - 1u);
                int split_slot = __popc(valid_mask & group_mask & lower_lane_mask);
                int valid_count = __popc(valid_mask & group_mask);
                if (split_counts != nullptr && valid_q && slot_in_q == 0) {
                    split_counts[(size_t)my_qi * H + h] = valid_count;
                }
                bool owned_edge = valid_edge;
                unsigned int owned_mask = valid_mask;
                if constexpr (kRowRanges) {
                    owned_edge = valid_edge && row >= row_begin && row < row_end;
                    owned_mask = __ballot_sync(0xFFFFFFFFu, owned_edge);
                }
                if (owned_edge) {
                    int slot = warp_ordered_slot(my_cursor, row - row_begin, owned_mask, lane);
                    int out_pos = my_abs_base[row] + slot;
                    head_qidx[out_pos] = q_local;
                    if (qsplit_idx != nullptr) {
                        qsplit_idx[(size_t)h * S_Q * kTopK + out_pos] =
                            q_local | ((split_slot & 0xFF) << 24);
                    }
                }
            }
        }
        if (h + 1 < H) __syncthreads();
    }
}

}  // anonymous namespace

// ===========================================================================
// Host orchestration
// ===========================================================================

template <int kTopK, int kBlockK>
static void launch_pipeline(
    torch::Tensor q2k,
    torch::Tensor cu_q,
    torch::Tensor cu_k,
    torch::Tensor row_ptr,
    torch::Tensor q_idx,
    int total_rows,
    int max_kv_blocks,
    torch::Tensor scheduler_metadata = torch::Tensor(),
    torch::Tensor work_count = torch::Tensor(),
    torch::Tensor qsplit_idx = torch::Tensor(),
    torch::Tensor split_counts = torch::Tensor(),
    int target_q_per_cta = 1,
    int work_capacity = 0,
    int max_seqlen_q = 0,
    int min_q_per_cta = 256,
    torch::Tensor row_counts_scratch = torch::Tensor(),
    torch::Tensor row_map_scratch = torch::Tensor(),
    torch::Tensor row_coords_scratch = torch::Tensor(),
    torch::Tensor tile_counts_scratch = torch::Tensor(),
    torch::Tensor local_candidate_storage = torch::Tensor(),
    torch::Tensor prefix_lens = torch::Tensor(),
    bool enable_batch1_identity_rows = true,
    bool local_top16_dynamic_only = false,
    bool canonicalize_dynamic_padding = false)
{
    int H = (int)q2k.size(0);
    int S_Q = (int)q2k.size(1);
    int topK = (int)q2k.size(2);
    TORCH_CHECK(topK == kTopK, "topK runtime != template kTopK");
    int B = (int)cu_q.size(0) - 1;
    auto device = q2k.device();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    bool fuse_local_top16 = local_candidate_storage.defined();
    int const* local_candidate_ptr = nullptr;
    int const* prefix_lens_ptr = nullptr;
    int candidate_stride_h = 0;
    int candidate_stride_k = 0;
    int candidate_stride_q = 0;
    if (fuse_local_top16) {
        TORCH_CHECK(prefix_lens.defined(),
                    "local Top-16 CSR fusion requires prefix_lens");
        TORCH_CHECK(local_candidate_storage.scalar_type() == at::kFloat,
                    "local candidate storage must be float32 backing");
        CHECK_CUDA(local_candidate_storage);
        CHECK_CONTIGUOUS(local_candidate_storage);
        CHECK_INPUT(prefix_lens);
        TORCH_CHECK(local_candidate_storage.device() == device
                    && prefix_lens.device() == device,
                    "local candidate inputs must share the q2k device");
        TORCH_CHECK(local_candidate_storage.dim() == 3
                    && local_candidate_storage.size(0) == H
                    && local_candidate_storage.size(1) >= 32
                    && local_candidate_storage.size(2) == S_Q,
                    "local candidate storage must be float32 [H,>=32,S_Q]");
        TORCH_CHECK(prefix_lens.dim() == 1 && prefix_lens.numel() == B,
                    "prefix_lens must be int32 [B]");
        local_candidate_ptr = reinterpret_cast<int const*>(
            local_candidate_storage.data_ptr<float>());
        prefix_lens_ptr = prefix_lens.data_ptr<int>();
        candidate_stride_h = (int)local_candidate_storage.stride(0);
        candidate_stride_k = (int)local_candidate_storage.stride(1);
        candidate_stride_q = (int)local_candidate_storage.stride(2);
    }

    AT_CUDA_CHECK(cudaMemsetAsync(
        row_ptr.data_ptr<int>(), 0,
        (size_t)H * (total_rows + 1) * sizeof(int), stream));
    AT_CUDA_CHECK(cudaMemsetAsync(
        q_idx.data_ptr<int>(), 0xFF,
        (size_t)H * S_Q * kTopK * sizeof(int), stream));

    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto row_counts = row_counts_scratch.defined()
        ? row_counts_scratch
        : torch::empty({H, total_rows}, opts);
    auto row_map = row_map_scratch.defined()
        ? row_map_scratch
        : torch::empty({B, max_kv_blocks}, opts);
    if (row_counts_scratch.defined()) {
        CHECK_INPUT(row_counts_scratch);
        TORCH_CHECK(row_counts_scratch.device() == device,
                    "row_counts scratch device mismatch");
        TORCH_CHECK(row_counts_scratch.size(0) == H
                    && row_counts_scratch.size(1) == total_rows,
                    "row_counts scratch shape mismatch");
    }
    if (row_map_scratch.defined()) {
        CHECK_INPUT(row_map_scratch);
        TORCH_CHECK(row_map_scratch.device() == device,
                    "row_map scratch device mismatch");
        TORCH_CHECK(row_map_scratch.size(0) == B
                    && row_map_scratch.size(1) == max_kv_blocks,
                    "row_map scratch shape mismatch");
    }
    AT_CUDA_CHECK(cudaMemsetAsync(
        row_counts.data_ptr<int>(), 0,
        (size_t)H * total_rows * sizeof(int), stream));
    bool emit_schedule = scheduler_metadata.defined();
    int scheduler_metadata_cols =
        emit_schedule ? (int)scheduler_metadata.size(1) : 0;
    bool compact_batch1_schedule = emit_schedule && scheduler_metadata_cols == 2;
    auto row_coords = emit_schedule
        ? (row_coords_scratch.defined()
            ? row_coords_scratch
            : torch::empty({total_rows, 2}, opts))
        : torch::Tensor();
    if (emit_schedule && row_coords_scratch.defined()) {
        CHECK_INPUT(row_coords_scratch);
        TORCH_CHECK(row_coords_scratch.device() == device,
                    "row_coords scratch device mismatch");
        TORCH_CHECK(row_coords_scratch.size(0) == total_rows
                    && row_coords_scratch.size(1) == 2,
                    "row_coords scratch shape mismatch");
    }
    int* scheduler_metadata_ptr = emit_schedule ? scheduler_metadata.data_ptr<int>() : nullptr;
    int* work_count_ptr = emit_schedule ? work_count.data_ptr<int>() : nullptr;
    int* qsplit_idx_ptr = emit_schedule ? qsplit_idx.data_ptr<int>() : nullptr;
    int* split_counts_ptr = emit_schedule ? split_counts.data_ptr<int>() : nullptr;
    int* row_coords_ptr = emit_schedule ? row_coords.data_ptr<int>() : nullptr;
    if (emit_schedule) {
        AT_CUDA_CHECK(cudaMemsetAsync(work_count_ptr, 0, sizeof(int), stream));
        AT_CUDA_CHECK(cudaMemsetAsync(
            scheduler_metadata_ptr, 0,
            (size_t)work_capacity * scheduler_metadata_cols * sizeof(int), stream));
    }

    int dev = q2k.get_device();
    int num_sms = 0;
    AT_CUDA_CHECK(cudaDeviceGetAttribute(
        &num_sms, cudaDevAttrMultiProcessorCount, dev));

    // -- Pick kWarps per CTA based on SMEM budget for cursor/hist ---------
    // SMEM per CTA = kWarps * total_rows * sizeof(int) (for both H and S).
    // Want at least 2 CTAs/SM for memory parallelism. SM100 SMEM = 228KB.
    // Pick the largest kWarps that fits two CTAs/SM, capped at 4.
    // SMEM cursor packed as int16 (2 entries per int32 word). A batch whose
    // rows exceed one CTA's shared memory (kMaxRangeRows int16 counters, the
    // 227 KB opt-in limit) is split into equal row ranges along grid y.
    constexpr int kMaxRangeRows = 116224;
    int row_ranges = std::max(1, (total_rows + kMaxRangeRows - 1) / kMaxRangeRows);
    int range_rows = (total_rows + row_ranges - 1) / row_ranges;
    int per_warp_smem = ((range_rows + 1) >> 1) * (int)sizeof(int);
    int kWarps_pick = 4;
    while (kWarps_pick > 1 && (kWarps_pick * per_warp_smem) * 2 > 228 * 1024) {
        kWarps_pick >>= 1;
    }
    if (kWarps_pick < 1) kWarps_pick = 1;
    // Row ranges are at least half of kMaxRangeRows long, which leaves one warp per CTA: only that
    // instantiation carries the row-range code.
    TORCH_CHECK(row_ranges == 1 || kWarps_pick == 1, "row ranges require one histogram warp per CTA");

    // -- Pick G (CTAs) ----------------------------------------------------
    // For each (kWarps, per_warp_smem) pair, the SMEM-bound occupancy is
    // 228KB / (kWarps*per_warp_smem) CTAs/SM. We size G as
    // num_sms * occupancy so a single resident wave covers all CTAs and
    // the memory pipeline runs at peak.
    int per_cta_smem_bytes = kWarps_pick * per_warp_smem;
    int max_ctas_per_sm = std::max(
        1, (228 * 1024) / std::max(1, per_cta_smem_bytes));
    if (max_ctas_per_sm > 8) max_ctas_per_sm = 8;
    TORCH_CHECK(min_q_per_cta > 0, "min_q_per_cta must be positive");
    // Cap target_g at num_sms * 3 -- empirically this balances
    // per-CTA work-size against parallelism. Higher caps regress
    // mid-size cases due to row_counts atomicAdd contention and
    // smaller q_per_cta. SMEM-bound configurations naturally cap
    // lower if max_ctas_per_sm < 3.
    int target_g = num_sms * std::min(max_ctas_per_sm, 3);
    int max_g_for_q = (S_Q + min_q_per_cta - 1) / min_q_per_cta;
    int G = std::min({target_g, max_g_for_q, S_Q});
    if (G < 1) G = 1;
    int q_per_cta = (S_Q + G - 1) / G;
    G = (S_Q + q_per_cta - 1) / q_per_cta;
    int q_per_warp = (q_per_cta + kWarps_pick - 1) / kWarps_pick;
    int G_total = G * kWarps_pick;

    auto tile_counts = tile_counts_scratch.defined()
        ? tile_counts_scratch
        : torch::empty({G_total, H, total_rows}, opts);
    if (tile_counts_scratch.defined()) {
        CHECK_INPUT(tile_counts_scratch);
        TORCH_CHECK(tile_counts_scratch.device() == device,
                    "tile_counts scratch device mismatch");
        TORCH_CHECK(tile_counts_scratch.size(0) == G_total
                    && tile_counts_scratch.size(1) == H
                    && tile_counts_scratch.size(2) == total_rows,
                    "tile_counts scratch shape mismatch");
    }

    // -- Compile-time switch on kWarps for the templated kernels ---------
    auto rmap_fn = k2q_build_row_map_kernel<kBlockK>;
    constexpr int kPtRowsPerBlock = 8;
    constexpr int kPtThreads = 256;
    auto tprefix_smem_fn = k2q_tile_prefix_smem_kernel<kPtThreads, kPtRowsPerBlock>;

    // C077: in the fused local-Top16 prefill path B==1 implies a strict
    // identity logical row map.  Keep every generic/multi-batch path on the
    // original producer and kernels.
    // C110 extends the already-exact batch-one identity row specialization to
    // the measured C109 local-Top16 interval. The Python adapter retains an
    // explicit threshold switch for same-session parent comparisons.
    bool batch1_identity_rows =
        fuse_local_top16 && B == 1 && enable_batch1_identity_rows;
    TORCH_CHECK(!batch1_identity_rows || total_rows == max_kv_blocks,
                "batch-one identity CSR requires total_rows == max_kv_blocks");
    if (!batch1_identity_rows && max_kv_blocks > 0) {
        rmap_fn<<<max_kv_blocks, 32, 0, stream>>>(
            cu_k.data_ptr<int>(), row_map.data_ptr<int>(), row_coords_ptr, B, max_kv_blocks);
    }

    auto launch_hist_scatter = [&](auto kWarps_const, auto batch1_identity_const,
                                   auto compact_schedule_const, auto row_ranges_const) {
        constexpr int W = decltype(kWarps_const)::value;
        constexpr bool kRowRanges = decltype(row_ranges_const)::value;
        constexpr bool kBatch1IdentityRows =
            decltype(batch1_identity_const)::value;
        constexpr bool kCompactBatch1Schedule =
            decltype(compact_schedule_const)::value;
        size_t smem_bytes = (size_t)W * per_warp_smem;
        auto scat_fn =
            k2q_scatter_kernel<kTopK, kBlockK, W, kBatch1IdentityRows, kRowRanges>;
        auto rprefix_fn = k2q_row_prefix_kernel<
            1024, kBatch1IdentityRows, kCompactBatch1Schedule>;
        AT_CUDA_CHECK(cudaFuncSetAttribute(
            scat_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

        if (fuse_local_top16) {
            if (local_top16_dynamic_only) {
                if (canonicalize_dynamic_padding) {
                    auto hist_fn = k2q_hist_kernel<kTopK, kBlockK, W, true,
                                                   kBatch1IdentityRows, true, true, kRowRanges>;
                    AT_CUDA_CHECK(cudaFuncSetAttribute(
                        hist_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                        (int)smem_bytes));
                    hist_fn<<<dim3(G, row_ranges), W * kWarpSize, smem_bytes, stream>>>(
                        q2k.data_ptr<int>(), cu_q.data_ptr<int>(), row_map.data_ptr<int>(),
                        row_counts.data_ptr<int>(), tile_counts.data_ptr<int>(),
                        local_candidate_ptr, prefix_lens_ptr,
                        H, B, S_Q, total_rows, max_kv_blocks, q_per_cta, q_per_warp,
                        candidate_stride_h, candidate_stride_k, candidate_stride_q, range_rows);
                } else {
                    auto hist_fn = k2q_hist_kernel<kTopK, kBlockK, W, true,
                                                   kBatch1IdentityRows, true, false, kRowRanges>;
                    AT_CUDA_CHECK(cudaFuncSetAttribute(
                        hist_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                        (int)smem_bytes));
                    hist_fn<<<dim3(G, row_ranges), W * kWarpSize, smem_bytes, stream>>>(
                        q2k.data_ptr<int>(), cu_q.data_ptr<int>(), row_map.data_ptr<int>(),
                        row_counts.data_ptr<int>(), tile_counts.data_ptr<int>(),
                        local_candidate_ptr, prefix_lens_ptr,
                        H, B, S_Q, total_rows, max_kv_blocks, q_per_cta, q_per_warp,
                        candidate_stride_h, candidate_stride_k, candidate_stride_q, range_rows);
                }
            } else {
                auto hist_fn = k2q_hist_kernel<kTopK, kBlockK, W, true,
                                               kBatch1IdentityRows, false, false, kRowRanges>;
                AT_CUDA_CHECK(cudaFuncSetAttribute(
                    hist_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)smem_bytes));
                hist_fn<<<dim3(G, row_ranges), W * kWarpSize, smem_bytes, stream>>>(
                    q2k.data_ptr<int>(), cu_q.data_ptr<int>(), row_map.data_ptr<int>(),
                    row_counts.data_ptr<int>(), tile_counts.data_ptr<int>(),
                    local_candidate_ptr, prefix_lens_ptr,
                    H, B, S_Q, total_rows, max_kv_blocks, q_per_cta, q_per_warp,
                    candidate_stride_h, candidate_stride_k, candidate_stride_q, range_rows);
            }
        } else {
            auto hist_fn = k2q_hist_kernel<kTopK, kBlockK, W, false,
                                           kBatch1IdentityRows, false, false, kRowRanges>;
            AT_CUDA_CHECK(cudaFuncSetAttribute(
                hist_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)smem_bytes));
            hist_fn<<<dim3(G, row_ranges), W * kWarpSize, smem_bytes, stream>>>(
                q2k.data_ptr<int>(), cu_q.data_ptr<int>(), row_map.data_ptr<int>(),
                row_counts.data_ptr<int>(), tile_counts.data_ptr<int>(),
                nullptr, nullptr,
                H, B, S_Q, total_rows, max_kv_blocks, q_per_cta, q_per_warp,
                0, 0, 0, range_rows);
        }

        rprefix_fn<<<H, 1024, 0, stream>>>(
            row_counts.data_ptr<int>(), row_ptr.data_ptr<int>(),
            emit_schedule ? row_coords.data_ptr<int>() : nullptr,
            scheduler_metadata_ptr,
            work_count_ptr,
            total_rows,
            target_q_per_cta,
            work_capacity);

        // Grid is H * blocks_per_h so each block stays within a single
        // head; flat (H*total_rows) grid would skip rows when total_rows
        // is not a multiple of kPtRowsPerBlock.
        int blocks_per_h = (total_rows + kPtRowsPerBlock - 1) / kPtRowsPerBlock;
        int pt_grid = H * blocks_per_h;
        if (pt_grid < 1) pt_grid = 1;
        size_t pt_smem = (size_t)kPtRowsPerBlock * G_total * sizeof(int);
        AT_CUDA_CHECK(cudaFuncSetAttribute(
            tprefix_smem_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)pt_smem));
        tprefix_smem_fn<<<pt_grid, kPtThreads, pt_smem, stream>>>(
            tile_counts.data_ptr<int>(), row_ptr.data_ptr<int>(),
            H, total_rows, G_total);

        scat_fn<<<dim3(G, row_ranges), W * kWarpSize, smem_bytes, stream>>>(
            q2k.data_ptr<int>(), cu_q.data_ptr<int>(), row_map.data_ptr<int>(),
            tile_counts.data_ptr<int>(), q_idx.data_ptr<int>(),
            qsplit_idx_ptr, split_counts_ptr,
            H, B, S_Q, total_rows, max_kv_blocks, q_per_cta, q_per_warp,
            max_seqlen_q, range_rows);
    };

    auto launch_by_warps = [&](auto batch1_identity_const,
                               auto compact_schedule_const) {
      if (kWarps_pick == 4) {
        launch_hist_scatter(std::integral_constant<int, 4>{},
                            batch1_identity_const, compact_schedule_const, std::false_type{});
      } else if (kWarps_pick == 2) {
        launch_hist_scatter(std::integral_constant<int, 2>{},
                            batch1_identity_const, compact_schedule_const, std::false_type{});
      } else if (row_ranges > 1) {
        // More rows than one CTA's counters hold: one warp per CTA and the row-range kernels.
        launch_hist_scatter(std::integral_constant<int, 1>{},
                            batch1_identity_const, compact_schedule_const, std::true_type{});
      } else {
        launch_hist_scatter(std::integral_constant<int, 1>{},
                            batch1_identity_const, compact_schedule_const, std::false_type{});
      }
    };

    // The dynamic-only unified path also uses the exact batch-one logical
    // identity row mapping.  Keep runtime row-map elision and the selected
    // kernel specialization consistent for both supported fused capacities.
    if constexpr ((kTopK == 32 || kTopK == 16) && kBlockK == 128) {
      if (batch1_identity_rows) {
        if (compact_batch1_schedule) {
          launch_by_warps(std::true_type{}, std::true_type{});
        } else {
          launch_by_warps(std::true_type{}, std::false_type{});
        }
      } else {
        launch_by_warps(std::false_type{}, std::false_type{});
      }
    } else {
      launch_by_warps(std::false_type{}, std::false_type{});
    }
}

// (topK, blk_kv) dispatch. `launch_pipeline` is already
// templated on both; upstream only ever instantiated blk_kv == 128. The
// instantiation set is what bounds compile time -- 32 / 64 / 128 covers the MSA
// configuration grid.
#define MSA_K2Q_CHECK_BLK_KV(blk_kv)                                          \
    TORCH_CHECK((blk_kv) == 32 || (blk_kv) == 64 || (blk_kv) == 128,          \
                "build_k2q_csr supports blk_kv in {32, 64, 128}, got ", (blk_kv))

#define MSA_K2Q_DISPATCH_BLK(TOPK, ...)                                       \
    do {                                                                      \
        if (blk_kv == 128) {                                                  \
            launch_pipeline<TOPK, 128>(__VA_ARGS__);                          \
        } else if (blk_kv == 64) {                                            \
            launch_pipeline<TOPK, 64>(__VA_ARGS__);                           \
        } else {                                                              \
            launch_pipeline<TOPK, 32>(__VA_ARGS__);                           \
        }                                                                     \
    } while (0)

#define MSA_K2Q_DISPATCH(...)                                                 \
    do {                                                                      \
        if (topk == 16) {                                                     \
            MSA_K2Q_DISPATCH_BLK(16, __VA_ARGS__);                            \
        } else if (topk == 8) {                                               \
            MSA_K2Q_DISPATCH_BLK(8, __VA_ARGS__);                             \
        } else if (topk == 32) {                                              \
            MSA_K2Q_DISPATCH_BLK(32, __VA_ARGS__);                            \
        } else if (topk == 4) {                                               \
            MSA_K2Q_DISPATCH_BLK(4, __VA_ARGS__);                             \
        } else {                                                              \
            TORCH_CHECK(false, "unsupported topK ", topk,                     \
                        " (expected 4, 8, 16, or 32)");                       \
        }                                                                     \
    } while (0)

void run_build_k2q_csr(
    torch::Tensor q2k,
    torch::Tensor cu_q,
    torch::Tensor cu_k,
    torch::Tensor row_ptr,
    torch::Tensor q_idx,
    int64_t topk,
    int64_t blk_kv,
    int64_t total_rows,
    int64_t max_kv_blocks,
    int64_t min_q_per_cta = 256,
    torch::Tensor row_counts_scratch = torch::Tensor(),
    torch::Tensor row_map_scratch = torch::Tensor(),
    torch::Tensor row_coords_scratch = torch::Tensor(),
    torch::Tensor tile_counts_scratch = torch::Tensor())
{
    CHECK_INPUT(q2k);
    CHECK_INPUT(cu_q);
    CHECK_INPUT(cu_k);
    CHECK_INPUT(row_ptr);
    CHECK_INPUT(q_idx);
    MSA_K2Q_CHECK_BLK_KV(blk_kv);
    int H = (int)q2k.size(0);
    int S_Q = (int)q2k.size(1);
    int tr = (int)total_rows;
    int mkv = (int)max_kv_blocks;
    TORCH_CHECK(tr >= 0 && mkv >= 0,
                "total_rows / max_kv_blocks must be non-negative");
    TORCH_CHECK(row_ptr.size(0) == H && row_ptr.size(1) == tr + 1,
                "row_ptr shape mismatch");
    TORCH_CHECK(q_idx.size(0) == H && q_idx.size(1) == (int64_t)S_Q * (int)topk,
                "q_idx shape mismatch");
    if (S_Q == 0 || tr == 0 || H == 0 || mkv == 0) {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
        AT_CUDA_CHECK(cudaMemsetAsync(
            row_ptr.data_ptr<int>(), 0,
            (size_t)H * (tr + 1) * sizeof(int), stream));
        AT_CUDA_CHECK(cudaMemsetAsync(
            q_idx.data_ptr<int>(), 0xFF,
            (size_t)H * S_Q * (int)topk * sizeof(int), stream));
        return;
    }

    MSA_K2Q_DISPATCH(
        q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv,
        torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
        1, 0, 0, (int)min_q_per_cta,
        row_counts_scratch, row_map_scratch, row_coords_scratch,
        tile_counts_scratch);
}

void run_build_k2q_csr_with_schedule(
    torch::Tensor q2k,
    torch::Tensor cu_q,
    torch::Tensor cu_k,
    torch::Tensor row_ptr,
    torch::Tensor q_idx,
    torch::Tensor scheduler_metadata,
    torch::Tensor work_count,
    torch::Tensor qsplit_idx,
    torch::Tensor split_counts,
    int64_t topk,
    int64_t blk_kv,
    int64_t total_rows,
    int64_t max_kv_blocks,
    int64_t target_q_per_cta,
    int64_t work_capacity,
    int64_t max_seqlen_q,
    int64_t min_q_per_cta = 256,
    torch::Tensor row_counts_scratch = torch::Tensor(),
    torch::Tensor row_map_scratch = torch::Tensor(),
    torch::Tensor row_coords_scratch = torch::Tensor(),
    torch::Tensor tile_counts_scratch = torch::Tensor())
{
    CHECK_INPUT(q2k);
    CHECK_INPUT(cu_q);
    CHECK_INPUT(cu_k);
    CHECK_INPUT(row_ptr);
    CHECK_INPUT(q_idx);
    CHECK_INPUT(scheduler_metadata);
    CHECK_INPUT(work_count);
    CHECK_INPUT(qsplit_idx);
    CHECK_INPUT(split_counts);
    MSA_K2Q_CHECK_BLK_KV(blk_kv);
    int H = (int)q2k.size(0);
    int S_Q = (int)q2k.size(1);
    int tr = (int)total_rows;
    int mkv = (int)max_kv_blocks;
    int target = (int)target_q_per_cta;
    int capacity = (int)work_capacity;
    int max_sq = (int)max_seqlen_q;
    TORCH_CHECK(tr >= 0 && mkv >= 0 && target > 0 && capacity > 0 && max_sq >= 0,
                "invalid schedule sizing arguments");
    TORCH_CHECK(row_ptr.size(0) == H && row_ptr.size(1) == tr + 1,
                "row_ptr shape mismatch");
    TORCH_CHECK(q_idx.size(0) == H && q_idx.size(1) == (int64_t)S_Q * (int)topk,
                "q_idx shape mismatch");
    TORCH_CHECK(qsplit_idx.sizes() == q_idx.sizes(), "qsplit_idx shape mismatch");
    TORCH_CHECK(scheduler_metadata.size(0) == capacity && scheduler_metadata.size(1) == 6,
                "scheduler_metadata shape mismatch");
    TORCH_CHECK(work_count.numel() == 1, "work_count must have one int32 element");
    TORCH_CHECK(split_counts.dim() == 2 && split_counts.size(0) == S_Q
                && split_counts.size(1) == H,
                "split_counts shape mismatch");
    if (S_Q == 0 || tr == 0 || H == 0 || mkv == 0) {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
        AT_CUDA_CHECK(cudaMemsetAsync(
            row_ptr.data_ptr<int>(), 0,
            (size_t)H * (tr + 1) * sizeof(int), stream));
        AT_CUDA_CHECK(cudaMemsetAsync(
            q_idx.data_ptr<int>(), 0xFF,
            (size_t)H * S_Q * (int)topk * sizeof(int), stream));
        AT_CUDA_CHECK(cudaMemsetAsync(work_count.data_ptr<int>(), 0, sizeof(int), stream));
        if (split_counts.numel() > 0) {
            AT_CUDA_CHECK(cudaMemsetAsync(
                split_counts.data_ptr<int>(), 0,
                (size_t)split_counts.numel() * sizeof(int), stream));
        }
        return;
    }

    MSA_K2Q_DISPATCH(
        q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv,
        scheduler_metadata, work_count, qsplit_idx, split_counts,
        target, capacity, max_sq, (int)min_q_per_cta,
        row_counts_scratch, row_map_scratch, row_coords_scratch,
        tile_counts_scratch);
}

void run_build_k2q_csr_with_schedule_from_local_top16(
    torch::Tensor local_candidate_storage,
    torch::Tensor prefix_lens,
    torch::Tensor q2k,
    torch::Tensor cu_q,
    torch::Tensor cu_k,
    torch::Tensor row_ptr,
    torch::Tensor q_idx,
    torch::Tensor scheduler_metadata,
    torch::Tensor work_count,
    torch::Tensor qsplit_idx,
    torch::Tensor split_counts,
    int64_t topk,
    int64_t blk_kv,
    int64_t total_rows,
    int64_t max_kv_blocks,
    int64_t candidate_partition_blocks,
    int64_t target_q_per_cta,
    int64_t work_capacity,
    int64_t max_seqlen_q,
    int64_t min_q_per_cta = 256,
    bool dynamic_only = false,
    bool canonicalize_dynamic_padding = false,
    bool enable_batch1_identity_rows = true,
    torch::Tensor row_counts_scratch = torch::Tensor(),
    torch::Tensor row_map_scratch = torch::Tensor(),
    torch::Tensor row_coords_scratch = torch::Tensor(),
    torch::Tensor tile_counts_scratch = torch::Tensor())
{
    CHECK_CUDA(local_candidate_storage);
    CHECK_CONTIGUOUS(local_candidate_storage);
    TORCH_CHECK(local_candidate_storage.scalar_type() == at::kFloat,
                "local_candidate_storage must be float32");
    CHECK_INPUT(prefix_lens);
    CHECK_INPUT(q2k);
    CHECK_INPUT(cu_q);
    CHECK_INPUT(cu_k);
    CHECK_INPUT(row_ptr);
    CHECK_INPUT(q_idx);
    CHECK_INPUT(scheduler_metadata);
    CHECK_INPUT(work_count);
    CHECK_INPUT(qsplit_idx);
    CHECK_INPUT(split_counts);
    MSA_K2Q_CHECK_BLK_KV(blk_kv);
    TORCH_CHECK((dynamic_only && topk == 16) || (!dynamic_only && topk == 32),
                "local Top-16 CSR fusion requires dynamic Top-16 or final Top-32");
    int H = (int)q2k.size(0);
    int S_Q = (int)q2k.size(1);
    int B = (int)cu_q.size(0) - 1;
    int tr = (int)total_rows;
    int mkv = (int)max_kv_blocks;
    int partition_blocks = (int)candidate_partition_blocks;
    int target = (int)target_q_per_cta;
    int capacity = (int)work_capacity;
    int max_sq = (int)max_seqlen_q;
    TORCH_CHECK(tr >= 0 && mkv >= 0 && target > 0 && capacity > 0 && max_sq >= 0,
                "invalid schedule sizing arguments");
    TORCH_CHECK(partition_blocks > 0 && mkv <= partition_blocks,
                "local Top-16 CSR fusion requires exactly one candidate partition");
    TORCH_CHECK(local_candidate_storage.dim() == 3
                && local_candidate_storage.size(0) == H
                && local_candidate_storage.size(1) >= 32
                && local_candidate_storage.size(2) == S_Q,
                "local candidate storage must be [H,>=32,S_Q]");
    TORCH_CHECK(prefix_lens.dim() == 1 && prefix_lens.numel() == B,
                "prefix_lens must be int32 [B]");
    TORCH_CHECK(q2k.dim() == 3 && q2k.size(2) == topk,
                "q2k route output must be [H,S_Q,topk]");
    TORCH_CHECK(row_ptr.size(0) == H && row_ptr.size(1) == tr + 1,
                "row_ptr shape mismatch");
    TORCH_CHECK(q_idx.size(0) == H && q_idx.size(1) == (int64_t)S_Q * topk,
                "q_idx shape mismatch");
    TORCH_CHECK(qsplit_idx.sizes() == q_idx.sizes(), "qsplit_idx shape mismatch");
    int metadata_cols = (int)scheduler_metadata.size(1);
    bool compact_batch1_schedule = metadata_cols == 2;
    TORCH_CHECK(scheduler_metadata.size(0) == capacity
                && (metadata_cols == 6 || compact_batch1_schedule),
                "scheduler_metadata shape mismatch");
    TORCH_CHECK(!compact_batch1_schedule ||
                (B == 1 && enable_batch1_identity_rows && dynamic_only
                 && H == 4 && topk == 16 && blk_kv == 128 && S_Q <= 8192
                 && (int64_t)S_Q * topk <= (1 << 17)
                 && target <= 4096 && tr == mkv && mkv <= 8192),
                "compact schedule requires batch-one identity Dynamic Top-16 "
                "with Sq<=8192, blk_kv=128 and representable fields");
    TORCH_CHECK(work_count.numel() == 1, "work_count must have one int32 element");
    TORCH_CHECK(split_counts.dim() == 2 && split_counts.size(0) == S_Q
                && split_counts.size(1) == H,
                "split_counts shape mismatch");
    TORCH_CHECK(local_candidate_storage.device() == q2k.device()
                && prefix_lens.device() == q2k.device()
                && cu_q.device() == q2k.device()
                && cu_k.device() == q2k.device(),
                "all fused CSR inputs must share a CUDA device");
    if (S_Q == 0 || tr == 0 || H == 0 || mkv == 0) {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
        AT_CUDA_CHECK(cudaMemsetAsync(
            row_ptr.data_ptr<int>(), 0,
            (size_t)H * (tr + 1) * sizeof(int), stream));
        AT_CUDA_CHECK(cudaMemsetAsync(
            q_idx.data_ptr<int>(), 0xFF,
            (size_t)H * S_Q * topk * sizeof(int), stream));
        AT_CUDA_CHECK(cudaMemsetAsync(work_count.data_ptr<int>(), 0, sizeof(int), stream));
        if (split_counts.numel() > 0) {
            AT_CUDA_CHECK(cudaMemsetAsync(
                split_counts.data_ptr<int>(), 0,
                (size_t)split_counts.numel() * sizeof(int), stream));
        }
        return;
    }

    MSA_K2Q_DISPATCH(
        q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv,
        scheduler_metadata, work_count, qsplit_idx, split_counts,
        target, capacity, max_sq, (int)min_q_per_cta,
        row_counts_scratch, row_map_scratch, row_coords_scratch,
        tile_counts_scratch, local_candidate_storage, prefix_lens,
        enable_batch1_identity_rows, dynamic_only, canonicalize_dynamic_padding);
}

void run_reorder_schedule_qcount_bucket16(
    torch::Tensor source,
    torch::Tensor destination,
    torch::Tensor work_count,
    int64_t target_q_per_cta)
{
    CHECK_INPUT(source);
    CHECK_INPUT(destination);
    CHECK_INPUT(work_count);
    TORCH_CHECK(source.dim() == 2 && destination.sizes() == source.sizes(),
                "schedule source/destination shape mismatch");
    int columns = (int)source.size(1);
    TORCH_CHECK(columns == 2 || columns == 6,
                "schedule reorder requires compact2 or legacy6 metadata");
    TORCH_CHECK(source.data_ptr<int>() != destination.data_ptr<int>(),
                "schedule reorder source and destination must not alias");
    TORCH_CHECK(work_count.numel() == 1,
                "work_count must have one int32 element");
    TORCH_CHECK(source.device() == destination.device()
                && source.device() == work_count.device(),
                "schedule reorder tensors must share a CUDA device");
    TORCH_CHECK(target_q_per_cta > 0 && target_q_per_cta <= 4096,
                "invalid target_q_per_cta for compact schedule fields");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    reorder_schedule_qcount_bucket16_kernel<<<1, 256, 0, stream>>>(
        source.data_ptr<int>(), destination.data_ptr<int>(),
        work_count.data_ptr<int>(), (int)source.size(0), columns,
        (int)target_q_per_cta);
    AT_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_build_k2q_csr", &run_build_k2q_csr,
          "q2k -> k2q CSR build (sorted within row)",
          pybind11::arg("q2k"),
          pybind11::arg("cu_q"),
          pybind11::arg("cu_k"),
          pybind11::arg("row_ptr"),
          pybind11::arg("q_idx"),
          pybind11::arg("topk"),
          pybind11::arg("blk_kv"),
          pybind11::arg("total_rows"),
          pybind11::arg("max_kv_blocks"),
          pybind11::arg("min_q_per_cta") = 256,
          pybind11::arg("row_counts_scratch") = torch::Tensor(),
          pybind11::arg("row_map_scratch") = torch::Tensor(),
          pybind11::arg("row_coords_scratch") = torch::Tensor(),
          pybind11::arg("tile_counts_scratch") = torch::Tensor());
    m.def("run_build_k2q_csr_with_schedule", &run_build_k2q_csr_with_schedule,
          "q2k -> k2q CSR build with fused attention schedule metadata",
          pybind11::arg("q2k"),
          pybind11::arg("cu_q"),
          pybind11::arg("cu_k"),
          pybind11::arg("row_ptr"),
          pybind11::arg("q_idx"),
          pybind11::arg("scheduler_metadata"),
          pybind11::arg("work_count"),
          pybind11::arg("qsplit_idx"),
          pybind11::arg("split_counts"),
          pybind11::arg("topk"),
          pybind11::arg("blk_kv"),
          pybind11::arg("total_rows"),
          pybind11::arg("max_kv_blocks"),
          pybind11::arg("target_q_per_cta"),
          pybind11::arg("work_capacity"),
          pybind11::arg("max_seqlen_q"),
          pybind11::arg("min_q_per_cta") = 256,
          pybind11::arg("row_counts_scratch") = torch::Tensor(),
          pybind11::arg("row_map_scratch") = torch::Tensor(),
          pybind11::arg("row_coords_scratch") = torch::Tensor(),
          pybind11::arg("tile_counts_scratch") = torch::Tensor());
    m.def("run_build_k2q_csr_with_schedule_from_local_top16",
          &run_build_k2q_csr_with_schedule_from_local_top16,
          "single-list local Top-16 -> q2k route + k2q CSR with schedule",
          pybind11::arg("local_candidate_storage"),
          pybind11::arg("prefix_lens"),
          pybind11::arg("q2k"),
          pybind11::arg("cu_q"),
          pybind11::arg("cu_k"),
          pybind11::arg("row_ptr"),
          pybind11::arg("q_idx"),
          pybind11::arg("scheduler_metadata"),
          pybind11::arg("work_count"),
          pybind11::arg("qsplit_idx"),
          pybind11::arg("split_counts"),
          pybind11::arg("topk"),
          pybind11::arg("blk_kv"),
          pybind11::arg("total_rows"),
          pybind11::arg("max_kv_blocks"),
          pybind11::arg("candidate_partition_blocks"),
          pybind11::arg("target_q_per_cta"),
          pybind11::arg("work_capacity"),
          pybind11::arg("max_seqlen_q"),
          pybind11::arg("min_q_per_cta") = 256,
          pybind11::arg("dynamic_only") = false,
          pybind11::arg("canonicalize_dynamic_padding") = false,
          pybind11::arg("enable_batch1_identity_rows") = true,
          pybind11::arg("row_counts_scratch") = torch::Tensor(),
          pybind11::arg("row_map_scratch") = torch::Tensor(),
          pybind11::arg("row_coords_scratch") = torch::Tensor(),
          pybind11::arg("tile_counts_scratch") = torch::Tensor());
    m.def("run_reorder_schedule_qcount_bucket16",
          &run_reorder_schedule_qcount_bucket16,
          "reorder sparse attention work records into descending q-count buckets",
          pybind11::arg("source"),
          pybind11::arg("destination"),
          pybind11::arg("work_count"),
          pybind11::arg("target_q_per_cta"));
}
