// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Modified for MBSA: package integration; see root NOTICE.

// The hand-written MXFP6 index-score kernel: prefill page scores, one CTA per
// (query token, KV chunk).
#pragma once

#include <cuda.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include <cute/arch/tmem_allocator_sm100.hpp>

#include "sm100_utils.cuh"

using namespace deep_gemm;

#define PAGE 128
#define NQ 8
#define HEADS 4
#define DIM 128
#define A_ROW_BYTES 96
#define CODES_BYTES (PAGE * A_ROW_BYTES)
#define PAGE_BYTES (CODES_BYTES + PAGE * 4)
#define A_SMEM (PAGE * DIM)
#define B_SMEM (NQ * DIM)
#define UMMA_K 32
// Multiple stages hide load latency; each stage uses 16 KiB of shared memory.
#ifndef STAGES
  #define STAGES 5
#endif
// Each consumer team has four warps. With two teams and four TMEM buffers,
// team 0 uses buffers 0/2 and team 1 uses buffers 1/3.
#ifndef CONSUMER_TEAMS
  #define CONSUMER_TEAMS 1
#endif
#define TEAM_WARPS 4
#define CONSUMER_WARPS (CONSUMER_TEAMS * TEAM_WARPS)
#define NUM_THREADS ((CONSUMER_WARPS + 2) * 32)
#define WARP_PRODUCER CONSUMER_WARPS
#define WARP_MMA (CONSUMER_WARPS + 1)
// Each page's four MMAs share one accumulator buffer. Independent pages
// use separate TMEM buffers to overlap their dependent MMA chains.
#ifndef TMEM_BUFS
  #define TMEM_BUFS 4
#endif

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void bar_init(uint64_t* b, int c) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)),
               "r"(c));
}
__device__ __forceinline__ void bar_wait(uint64_t* b, uint32_t phase) {
  asm volatile(
      "{ .reg .pred p; L: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;"
      " @!p bra L; }" ::"r"(smem_u32(b)),
      "r"(phase)
      : "memory");
}

// UTCCP scale slot for row r is (r%32)*4+r/32.
__host__ __device__ __forceinline__ uint32_t sf_slot(uint32_t row) {
  return (row % 32) * 4 + row / 32;
}

// grid = (query token, kv chunk). One CTA owns a contiguous run of pages for
// one query, which is the split the Triton kernel uses: the chunk count is a
// shape constant so the grid stays fixed inside a CUDA graph.
__global__ __launch_bounds__(NUM_THREADS) void index_decode(
    const __grid_constant__ CUtensorMap kv_map,
    const __grid_constant__ CUtensorMap q_map, const uint8_t* cache,
    const int* block_table, int bt_stride, const int* seq_lens,
    // Query-to-request row map; null means one request per query (decode).
    const int* q_to_req, const uint32_t* sf_q, int num_kv_chunks,
    int max_blocks, int init_blocks, int local_blocks, int score_sum,
    float sm_scale, float* scores, int score_stride_h, int score_stride_q) {
  const uint32_t q_idx = blockIdx.x;
  const uint32_t chunk = blockIdx.y;
  const int req = q_to_req ? q_to_req[q_idx] : (int)q_idx;
  // seq_lens holds each query token's causal bound; zero marks graph padding.
  const int kv_len = max(seq_lens[q_idx], 0);
  const int num_blocks = (kv_len + PAGE - 1) / PAGE;
  const int chunk_blocks = (num_blocks + num_kv_chunks - 1) / num_kv_chunks;
  const int blk_begin = chunk * chunk_blocks;
  const int blk_end = min(blk_begin + chunk_blocks, num_blocks);
  if (blk_begin >= blk_end) return;
  block_table += (size_t)req * bt_stride;
  scores += (size_t)q_idx * score_stride_q;

  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* smem_a[STAGES];
  for (int s = 0; s < STAGES; ++s) smem_a[s] = smem + s * A_SMEM;
  uint8_t* smem_b = smem + STAGES * A_SMEM;
  uint32_t* smem_sf_a = reinterpret_cast<uint32_t*>(smem_b + B_SMEM);
  uint32_t* smem_sf_q = smem_sf_a + STAGES * PAGE;
  __shared__ alignas(16) uint64_t full_kv[STAGES], empty_kv[STAGES];
  __shared__ alignas(16) uint64_t mma_done[TMEM_BUFS], tmem_free[TMEM_BUFS],
      bar_q;
  __shared__ alignas(16) uint32_t tmem_base;
  __shared__ float red[CONSUMER_WARPS][NQ];

  const uint32_t tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  constexpr uint32_t kSfACol = TMEM_BUFS * NQ;
  constexpr uint32_t kSfBCol = kSfACol + TMEM_BUFS * 4;
  constexpr uint32_t kNumTmemCols = 128;
  static_assert(kSfBCol + 4 <= kNumTmemCols, "tensor memory columns overflow");

  if (warp == 0) cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, &tmem_base);
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      bar_init(&full_kv[s], 1);
      bar_init(&empty_kv[s], 1);
    }
    for (int b = 0; b < TMEM_BUFS; ++b) {
      bar_init(&mma_done[b], 1);
      bar_init(&tmem_free[b], 1);
    }
    bar_init(&bar_q, 1);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  // Every barrier must be initialised before anyone arrives on it. Issuing
  // the Q copy from another thread without this sync is a race that happens
  // to win on this hardware -- the tests passed with it in place.
  __syncthreads();
  // Allocation is published and this CTA never allocates again. Relinquish
  // the allocation permit now; the pool remains owned until the final free.
  if (warp == 0) cute::TMEM::Allocator1Sm().release_allocation_lock();
  // Q and its scales, once per CTA.
  if (tid == 0) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(
            smem_u32(smem_b)),
        "l"(&q_map), "r"(0), "r"(q_idx * NQ), "r"(smem_u32(&bar_q))
        : "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                     smem_u32(&bar_q)),
                 "r"(NQ * DIM)
                 : "memory");
  }
  // Q scales arrive in row order (computed this step, not cached) so they are
  // placed here; rows past NQ still need a legal scale.
  for (uint32_t i = tid; i < PAGE; i += NUM_THREADS)
    smem_sf_q[sf_slot(i)] = i < NQ ? sf_q[q_idx * NQ + i] : 0x7f7f7f7fu;
  // Each scale writer publishes its own generic stores to the async proxy
  // before the CTA join makes them available to the UTCCP issuing thread.
  cutlass::arch::fence_view_async_shared();
  __syncthreads();

  if (warp == WARP_PRODUCER) {
    // Nothing but transfers: never waits on the MMA or the reduction.
    if (lane == 0) {
      for (int blk = blk_begin; blk < blk_end; ++blk) {
        const int i = blk - blk_begin, stage = i % STAGES;
        if (i >= STAGES) bar_wait(&empty_kv[stage], ((i / STAGES) & 1) ^ 1);
        const int page = block_table[blk];
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cluster.global"
            ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];" ::
                "r"(smem_u32(smem_a[stage])),
            "l"(&kv_map), "r"(0), "r"(0), "r"(page),
            "r"(smem_u32(&full_kv[stage]))
            : "memory");
        const uint8_t* sf_g = cache + (size_t)page * PAGE_BYTES + CODES_BYTES;
        asm volatile(
            "cp.async.bulk.shared::cluster.global"
            ".mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                smem_u32(smem_sf_a + stage * PAGE)),
            "l"(sf_g), "r"(PAGE * 4), "r"(smem_u32(&full_kv[stage]))
            : "memory");
        // Packed payload on the wire, not the padded SMEM footprint.
        asm volatile(
            "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                smem_u32(&full_kv[stage])),
            "r"(CODES_BYTES + PAGE * 4)
            : "memory");
      }
    }
  } else if (warp == WARP_MMA) {
    if (lane == 0) bar_wait(&bar_q, 0);
    __syncwarp();
    const auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
        cutlass::detail::float_e2m3_unpacksmem_t, cutlass::float_e4m3_t,
        float, cutlass::float_ue8m0_t, PAGE, NQ, cute::UMMA::Major::K,
        cute::UMMA::Major::K>();
    // Operand descriptors depend only on (stage, k), so they are built once.
    uint64_t a_desc[STAGES][DIM / UMMA_K], b_desc[DIM / UMMA_K],
        rt[DIM / UMMA_K];
    cute::UMMA::SmemDescriptor sf_desc[STAGES];
#pragma unroll
    for (uint32_t k = 0; k < DIM / UMMA_K; ++k) {
      rt[k] =
          mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k, k);
      b_desc[k] = static_cast<uint64_t>(
          mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, DIM, DIM>(
              smem_b, 0, k * UMMA_K));
      for (int s = 0; s < STAGES; ++s)
        a_desc[s][k] = static_cast<uint64_t>(
            mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, DIM, DIM>(
                smem_a[s], 0, k * UMMA_K));
    }
    for (int s = 0; s < STAGES; ++s)
      sf_desc[s] = mma::sm100::make_sf_desc(smem_sf_a + s * PAGE);

    // Q scales are constant for this CTA, so publish them to TMEM once.
    auto sf_q_desc = mma::sm100::make_sf_desc(smem_sf_q);
    cutlass::arch::fence_view_async_shared();
    if (lane == 0)
      cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_q_desc, tmem_base + kSfBCol);
    __syncwarp();
    for (int blk = blk_begin; blk < blk_end; ++blk) {
      const int i = blk - blk_begin, stage = i % STAGES, buf = i % TMEM_BUFS;
      if (lane == 0) {
        bar_wait(&full_kv[stage], (i / STAGES) & 1);
        if (i >= TMEM_BUFS)
          bar_wait(&tmem_free[buf], ((i / TMEM_BUFS) & 1) ^ 1);
      }
      __syncwarp();
      ptx::tcgen05_after_thread_sync();
      if (lane == 0) {
        auto sfd = sf_desc[stage];
        cutlass::arch::fence_view_async_shared();
        cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sfd, tmem_base + kSfACol + buf * 4);
#pragma unroll
        for (uint32_t k = 0; k < DIM / UMMA_K; ++k)
          ptx::SM100_MMA_MXF8F6F4_SS::fma(
              reinterpret_cast<const cute::UMMA::SmemDescriptor&>(
                  a_desc[stage][k]),
              reinterpret_cast<const cute::UMMA::SmemDescriptor&>(b_desc[k]),
              tmem_base + buf * NQ, k, rt[k], tmem_base + kSfACol + buf * 4,
              tmem_base + kSfBCol);
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster."
            "b64 [%0];" ::"r"(smem_u32(&mma_done[buf]))
            : "memory");
        // The UMMA itself releases the stage once it has read shared memory.
        // Lane 0 is already the single issuer. Avoid a second election
        // inside a warp-collective wrapper and commit the stage release directly.
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster."
            "b64 [%0];" ::"r"(smem_u32(&empty_kv[stage]))
            : "memory");
      }
      __syncwarp();
    }
  } else {
    // Consumer teams: 4 warps each, 32 accumulator rows per warp, syncing only
    // with themselves (a named barrier per team) so neither the producer nor
    // the other team ever waits for them.
    const int local_start = max(0, num_blocks - local_blocks);
    const uint32_t team = warp / TEAM_WARPS;
    const uint32_t warp_in_team = warp % TEAM_WARPS;
    const uint32_t row = warp_in_team * 32 + lane;
    const uint32_t team_bar = 1 + team;
    for (int blk = blk_begin + team; blk < blk_end; blk += CONSUMER_TEAMS) {
      const int i = blk - blk_begin, buf = i % TMEM_BUFS;
      // One thread polls the mbarrier and the named barrier releases the rest;
      // 128 threads spinning on one shared-memory word is pure contention.
      if (warp_in_team == 0 && lane == 0)
        bar_wait(&mma_done[buf], (i / TMEM_BUFS) & 1);
      asm volatile("barrier.sync %0, %1;" ::"r"(team_bar), "r"(TEAM_WARPS * 32)
                   : "memory");

      ptx::tcgen05_after_thread_sync();
      float acc[NQ];
      auto* r = reinterpret_cast<uint32_t*>(acc);
      cute::SM100_TMEM_LOAD_32dp32b8x::copy(
          tmem_base + buf * NQ, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
      cutlass::arch::fence_view_async_tmem_load();

      const int pos = blk * PAGE + static_cast<int>(row);
      const bool visible = pos < kv_len;
      // Only HEADS of the NQ rows are real; the rest exist because tcgen05
      // requires N % 8 == 0.
#pragma unroll
      for (int h = 0; h < HEADS; ++h)
        acc[h] = visible ? acc[h] * sm_scale : (score_sum ? 0.0f : -INFINITY);
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
#pragma unroll
        for (int h = 0; h < HEADS; ++h) {
          const float o = __shfl_xor_sync(0xffffffff, acc[h], off);
          acc[h] = score_sum ? acc[h] + o : fmaxf(acc[h], o);
        }
      if (lane == 0)
#pragma unroll
        for (int h = 0; h < HEADS; ++h) red[warp][h] = acc[h];
      ptx::tcgen05_before_thread_sync();
      // This barrier also means every warp is past its tcgen05.ld, so the
      // accumulator can be released here rather than after a third barrier.
      asm volatile("barrier.sync %0, %1;" ::"r"(team_bar), "r"(TEAM_WARPS * 32)
                   : "memory");
      if (warp_in_team == 0 && lane == 0)
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(
                         smem_u32(&tmem_free[buf]))
                     : "memory");
      if (warp_in_team == 0 && lane < HEADS) {
        float v = red[team * TEAM_WARPS][lane];
#pragma unroll
        for (int w = 1; w < TEAM_WARPS; ++w)
          v = score_sum ? v + red[team * TEAM_WARPS + w][lane]
                        : fmaxf(v, red[team * TEAM_WARPS + w][lane]);
        if (blk >= local_start)
          v = 1e29f;
        else if (blk < init_blocks)
          v = 1e30f;
        scores[lane * score_stride_h + blk] = v;
      }
    }
  }
  // All TMEM loads and MMA completion waits must retire before the allocating
  // warp frees the pool. Keep the TMA/MMA warps in the CTA until this join.
  ptx::tcgen05_before_thread_sync();
  __syncthreads();
  if (warp == 0) cute::TMEM::Allocator1Sm().free(tmem_base, kNumTmemCols);
}
