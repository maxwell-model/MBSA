// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-License-Identifier: MIT
// Modified for MBSA: MXFP6 index-K operands and a per-page index-score
// epilogue; see root NOTICE.

#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include "sm100_utils.cuh"
#include "paged_mqa_logits_scheduler.cuh"

namespace deep_gemm {

// Batch-1 schedule. Per SM: {first q, first split, pages of that q, 0}; entry
// kNumSMs is the end. Carrying the page count avoids a dependent load before
// the first task. Multiple requests use DeepGEMM's general schedule.
template <uint32_t kAlignedBatchSize, uint32_t SPLIT_KV, uint32_t kNumSMs,
          uint32_t BLOCK_KV>
__global__ __launch_bounds__(32, 1) void maxwell_decode_schedule(
    const uint32_t batch_size, const uint32_t* context_lens, uint32_t* meta) {
  cudaGridDependencySynchronize();
  __shared__ uint32_t prefix_sum[kAlignedBatchSize];
  const uint32_t lane_idx = threadIdx.x;
  uint32_t sum = 0;
  for (uint32_t k = 0; k < kAlignedBatchSize / 32; ++k) {
    const uint32_t q_idx = k * 32 + lane_idx;
    const uint32_t context_len = q_idx < batch_size ? context_lens[q_idx] : 0;
    uint32_t x = (context_len + SPLIT_KV - 1) / SPLIT_KV;
#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
      const uint32_t y = __shfl_up_sync(0xffffffff, x, offset);
      x += (lane_idx >= offset ? y : 0);
    }
    x += sum;
    prefix_sum[k * 32 + lane_idx] = x;
    sum = __shfl_sync(0xffffffff, x, 31);
  }
  __syncwarp();
  const uint32_t total = sum;
  const uint32_t q = total / kNumSMs, r = total % kNumSMs;
  const uint32_t pivot = kNumSMs - r;
  for (uint32_t sm_idx = lane_idx; sm_idx < kNumSMs; sm_idx += 32) {
    const uint32_t seg_starts =
        sm_idx * q + (sm_idx > pivot ? sm_idx - pivot : 0);
    uint32_t lo = 0, hi = batch_size;
    while (lo < hi) {
      const uint32_t mid = (lo + hi) / 2;
      const bool pred = prefix_sum[mid] <= seg_starts;
      lo = pred ? mid + 1 : lo;
      hi = pred ? hi : mid;
    }
    const uint32_t q_idx = min(lo, batch_size - 1);
    const uint32_t kv_split_idx =
        q_idx == 0 ? seg_starts : seg_starts - prefix_sum[q_idx - 1];
    meta[sm_idx * 4] = q_idx;
    meta[sm_idx * 4 + 1] = kv_split_idx;
    meta[sm_idx * 4 + 2] = (context_lens[q_idx] + BLOCK_KV - 1) / BLOCK_KV;
    meta[sm_idx * 4 + 3] = 0;
  }
  if (lane_idx == 0) {
    meta[kNumSMs * 4] = batch_size;
    meta[kNumSMs * 4 + 1] = 0;
    meta[kNumSMs * 4 + 2] = 0;
    meta[kNumSMs * 4 + 3] = 0;
  }
}

// PagedMQALogitsScheduler for decode (one token per request), reading the
// schedule above: identical task order, one fewer dependent load up front.
template <uint32_t BLOCK_KV, uint32_t kNumBlocksPerSplit>
struct MaxwellDecodeScheduler {
  const uint32_t* context_lens;
  uint32_t current_q_atom_idx, current_kv_idx, end_q_atom_idx, end_kv_idx,
      current_num_kv;

  CUTLASS_DEVICE static uint32_t atom_to_token_idx(const uint32_t& q) {
    return q;
  }
  CUTLASS_DEVICE static uint32_t atom_to_block_table_row(const uint32_t& q) {
    return q;
  }
  CUTLASS_DEVICE uint32_t get_last_advance() const { return 1; }

  CUTLASS_DEVICE explicit MaxwellDecodeScheduler(const uint32_t& sm_idx,
                                                 const uint32_t&,
                                                 const uint32_t* context_lens,
                                                 const uint32_t* meta,
                                                 const uint32_t*) {
    this->context_lens = context_lens;
    const uint4 cur = reinterpret_cast<const uint4*>(meta)[sm_idx];
    const uint4 end = reinterpret_cast<const uint4*>(meta)[sm_idx + 1];
    current_q_atom_idx = cur.x, current_kv_idx = cur.y * kNumBlocksPerSplit;
    current_num_kv = cur.z;
    end_q_atom_idx = end.x, end_kv_idx = end.y * kNumBlocksPerSplit;
  }

  CUTLASS_DEVICE bool fetch_next_task(uint32_t& q_atom_idx, uint32_t& kv_idx,
                                      uint32_t& num_kv) {
    q_atom_idx = current_q_atom_idx;
    kv_idx = current_kv_idx;
    num_kv = current_num_kv;
    if (current_q_atom_idx == end_q_atom_idx and current_kv_idx == end_kv_idx)
      return false;
    current_kv_idx += kNumBlocksPerSplit;
    if (current_kv_idx >= current_num_kv) {
      current_kv_idx = 0;
      current_q_atom_idx += 1;
      if (current_q_atom_idx < end_q_atom_idx or
          (current_q_atom_idx == end_q_atom_idx and 0 < end_kv_idx))
        current_num_kv =
            (context_lens[current_q_atom_idx] + BLOCK_KV - 1) / BLOCK_KV;
    }
    return true;
  }
};

}  // namespace deep_gemm

namespace deep_gemm {

// Adapted from DeepGEMM sm100_fp4_paged_mqa_logits.cuh for FP6 x FP8.
// TMA expands FP6 to byte-sized shared-memory operands.
// Decode epilogue writes [head,token,page] max/sum scores with fixed-page forcing.

template <uint32_t kNextN, uint32_t kNumHeads, uint32_t kNumRealHeads,
          uint32_t kHeadDim, uint32_t BLOCK_KV, bool kIsContextLens2D,
          bool kIsVarlen, uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV, uint32_t kNumSpecializedThreads,
          uint32_t kNumMathThreads, typename logits_dtype_t,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128,
          bool kSingleRequestSched = false>
CUTLASS_GLOBAL __launch_bounds__(
    kNumSpecializedThreads + kNumMathThreads,
    1) void sm100_fp6_paged_mqa_logits(const uint32_t batch_size,
                                       const uint32_t score_stride_h,
                                       const uint32_t score_stride_q,
                                       const uint32_t block_table_stride,
                                       const uint32_t* context_lens,
                                       logits_dtype_t* scores,
                                       const uint32_t init_blocks,
                                       const uint32_t local_blocks,
                                       const bool score_sum,
                                       const float sm_scale,
                                       const uint32_t* block_table,
                                       const uint32_t* indices,
                                       const uint32_t* schedule_meta,
                                       const __grid_constant__
                                           cute::TmaDescriptor tensor_map_q,
                                       const __grid_constant__
                                           cute::TmaDescriptor tensor_map_sf_q,
                                       const __grid_constant__
                                           cute::TmaDescriptor tensor_map_kv,
                                       const __grid_constant__
                                           cute::TmaDescriptor
                                               tensor_map_sf_kv) {
  using Barrier = cutlass::arch::ClusterTransactionBarrier;

  // Utils
  const auto sm_idx = blockIdx.x;
  const auto warp_idx = cutlass::canonical_warp_idx_sync();
  const auto warpgroup_idx = warp_idx / 4;
  const auto lane_idx = ptx::get_lane_idx();
  constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

  // Prefetch TMA descriptors
  if (warp_idx == kSpecWarpStart) {
    cute::prefetch_tma_descriptor(&tensor_map_q);
    cute::prefetch_tma_descriptor(&tensor_map_sf_q);
    cute::prefetch_tma_descriptor(&tensor_map_kv);
    cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
  }

  // For non-varlen odd kNextN >= 3, pad to even using TMA OOB zero-fill.
  static constexpr bool kPadOddN =
      (not kIsVarlen) and (kNextN % 2 == 1) and (kNextN >= 3);
  static constexpr uint32_t kNextNAtom = (kIsVarlen or kNextN >= 2) ? 2 : 1;
  static constexpr uint32_t kNumNextNAtoms =
      math::constexpr_ceil_div(kNextN, kNextNAtom);

  // UMMA configs
  static constexpr uint32_t kNumTmemStages = 3;
  static constexpr uint32_t kNumUTCCPAlignedElems = 128;
  static constexpr uint32_t UMMA_M = 128;
  static constexpr uint32_t UMMA_N = kNextNAtom * kNumHeads;
  // kind::mxf8f6f4 is K=32 with one SF per instruction (mxf4 is K=64).
  static constexpr uint32_t UMMA_K = 32;
  static constexpr uint32_t kNumSFQAtom =
      math::constexpr_align(kNextNAtom * kNumHeads, kNumUTCCPAlignedElems);
  static constexpr uint32_t kNumSFKV =
      math::constexpr_align(SPLIT_KV, kNumUTCCPAlignedElems);
  static constexpr uint32_t kRealNumSFQAtom = kNextNAtom * kNumHeads;
  DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0,
                   "Invalid threads");
  DG_STATIC_ASSERT(SPLIT_KV == kNumMathWarpGroups * UMMA_M and
                       SPLIT_KV % kNumUTCCPAlignedElems == 0,
                   "Invalid `SPLIT_KV`");

  // Shared memory configs
  static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
  static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE =
      kNextNAtom * kNumHeads * kHeadDim;
  static constexpr uint32_t SMEM_SF_Q_SIZE_PER_STAGE =
      kNumSFQAtom * sizeof(int);
  static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = SPLIT_KV * kHeadDim;
  // TMA transaction bytes != SMEM bytes for a sub-byte type: 6 bits per
  // element on the wire, one byte per element in shared memory.
  static constexpr uint32_t TMA_KV_BYTES_PER_STAGE =
      SPLIT_KV * kHeadDim * 6 / 8;
  static constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE = kNumSFKV * sizeof(int);
  static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE =
      kNextNAtom * kNumHeads * sizeof(float);
  // TMA requires 128B - aligned shared memory addresses, so pad the per - stage
  // stride (e.g. `kNextNAtom == 1` with 16 heads gives only 64B per stage)
  static constexpr uint32_t SMEM_WEIGHT_STRIDE_PER_STAGE =
      SMEM_WEIGHT_SIZE_PER_STAGE < 128 ? 128 : SMEM_WEIGHT_SIZE_PER_STAGE;

  // Align to swizzling alignment bytes
  extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
  DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % kSwizzleAlignment == 0,
                   "Unaligned TMA swizzling");
  DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0,
                   "Unaligned TMA swizzling");

  // Q and KV data on shared memory
  auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
    return smem_buffer + SMEM_Q_SIZE_PER_STAGE * i;
  });
  auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
    return smem_buffer + SMEM_Q_SIZE_PER_STAGE * kNumQStages +
           SMEM_KV_SIZE_PER_STAGE * i;
  });
  const auto smem_sf_ptr =
      smem_buffer + (SMEM_Q_SIZE_PER_STAGE * kNumQStages +
                     SMEM_KV_SIZE_PER_STAGE * kNumKVStages);
  auto smem_sf_q = utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<uint32_t*>(smem_sf_ptr +
                                       SMEM_SF_Q_SIZE_PER_STAGE * i);
  });
  auto smem_sf_kv = utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<uint32_t*>(smem_sf_ptr +
                                       SMEM_SF_Q_SIZE_PER_STAGE * kNumQStages +
                                       SMEM_SF_KV_SIZE_PER_STAGE * i);
  });
  auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<float*>(smem_sf_ptr +
                                    SMEM_SF_Q_SIZE_PER_STAGE * kNumQStages +
                                    SMEM_SF_KV_SIZE_PER_STAGE * kNumKVStages +
                                    SMEM_WEIGHT_STRIDE_PER_STAGE * i);
  });

  // Barriers and TMEM pointer on shared memory
  const auto barrier_ptr =
      reinterpret_cast<Barrier*>(smem_weights[kNumQStages]);
  auto full_q_barriers =
      utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
  auto empty_q_barriers = utils::PatternVisitor(
      [&](const uint32_t& i) { return barrier_ptr + kNumQStages + i; });
  auto full_kv_barriers = utils::PatternVisitor(
      [&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + i; });
  auto empty_kv_barriers = utils::PatternVisitor([&](const uint32_t& i) {
    return barrier_ptr + kNumQStages * 2 + kNumKVStages + i;
  });
  const auto tmem_barrier_ptr =
      barrier_ptr + kNumQStages * 2 + kNumKVStages * 2;
  auto full_tmem_barriers = utils::PatternVisitor(
      [&](const uint32_t& i) { return tmem_barrier_ptr + i; });
  auto empty_tmem_barriers = utils::PatternVisitor(
      [&](const uint32_t& i) { return tmem_barrier_ptr + kNumTmemStages + i; });
  auto tmem_ptr_in_smem =
      reinterpret_cast<uint32_t*>(tmem_barrier_ptr + kNumTmemStages * 2);

  // Tensor memory configs
  constexpr uint32_t kNumAccumTmemCols =
      kNextNAtom * kNumHeads * kNumTmemStages;
  constexpr uint32_t kNumTmemCols =
      utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFQAtom / 32 +
                                       kNumSFKV / 32>();
  constexpr uint32_t kTmemStartColOfSFQ = kNumAccumTmemCols;
  constexpr uint32_t kTmemStartColOfSFKV = kNumAccumTmemCols + kNumSFQAtom / 32;
  DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

  // Index-score epilogue. One page per math warpgroup keeps the page reduction
  // inside the warpgroup; a page split across warpgroups would need a
  // grid-wide combine this kernel has no place for.
  DG_STATIC_ASSERT(SPLIT_KV == BLOCK_KV * kNumMathWarpGroups,
                   "a math warpgroup must own exactly one page");
  DG_STATIC_ASSERT(
      kNumRealHeads <= kNumHeads and kNumRealHeads <= 32,
      "real heads must fit in the padded rows and in one warp's lanes");
  // Decode, with kNextN queries per request (1, or 1 + k for spec-decode
  // verify): the epilogue gives each token its own causal bound and its own
  // init/local windows. Varlen and 2-D context lengths stay unsupported.
  DG_STATIC_ASSERT(
      kNextN >= 1 and kNextN <= 4 and not kIsVarlen and not kIsContextLens2D,
      "the page epilogue takes 1..4 queries per request");
  __shared__ float smem_page_reduce[kNumMathThreads / 32][kNumRealHeads];

  // Initialize barriers
  if (warp_idx == kSpecWarpStart and cute::elect_one_sync()) {
#pragma unroll
    for (uint32_t i = 0; i < kNumQStages; ++i) {
      full_q_barriers[i]->init(1);
      empty_q_barriers[i]->init(kNumMathThreads + 32);
    }
    cutlass::arch::fence_barrier_init();
  }
  if (warp_idx == kSpecWarpStart + 1 and cute::elect_one_sync()) {
#pragma unroll
    for (uint32_t i = 0; i < kNumKVStages; ++i) {
      full_kv_barriers[i]->init(1);
      empty_kv_barriers[i]->init(1);
    }
    cutlass::arch::fence_barrier_init();
  }
  if (warp_idx == kSpecWarpStart + 2) {
    if (cute::elect_one_sync()) {
#pragma unroll
      for (uint32_t i = 0; i < kNumTmemStages; ++i) {
        full_tmem_barriers[i]->init(1);
        empty_tmem_barriers[i]->init(128);
      }
      cutlass::arch::fence_barrier_init();
    }
    // Allocate tensor memory
    cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
  }
  __syncthreads();

  // Wait for primary kernel completion
  cudaGridDependencySynchronize();

  // Scheduler
  constexpr uint32_t kNumBlocksPerSplit = SPLIT_KV / BLOCK_KV;
  using Scheduler = std::conditional_t<
      kSingleRequestSched, MaxwellDecodeScheduler<BLOCK_KV, kNumBlocksPerSplit>,
      sched::PagedMQALogitsScheduler<kNextN, kIsContextLens2D, kIsVarlen,
                                     BLOCK_KV, kNumBlocksPerSplit,
                                     kNumNextNAtoms>>;
  DG_STATIC_ASSERT(SPLIT_KV == BLOCK_KV * kNumBlocksPerSplit,
                   "Invalid `SPLIT_KV`");

  // Make Q, KV and TMEM pipeline
  auto make_pipeline = [](const uint32_t& num_stages) {
    // Return current stage and phase, and advance pipeline by steps
    return [iter_idx = 0u,
            num_stages](const uint32_t& step =
                            1) mutable -> cute::tuple<uint32_t, uint32_t> {
      uint32_t current_idx = iter_idx;
      iter_idx += step;
      return {current_idx % num_stages, (current_idx / num_stages) & 1};
    };
  };
  auto advance_q_pipeline = make_pipeline(kNumQStages);
  auto advance_kv_pipeline = make_pipeline(kNumKVStages);
  auto advance_tmem_pipeline = make_pipeline(kNumTmemStages);

  // Register reconfigurations
  constexpr uint32_t kNumSpecializedRegisters = 56;
  constexpr uint32_t kNumMathRegisters = 224;

  if (warp_idx == kSpecWarpStart) {
    // TMA warp for loading Q
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

    if (cute::elect_one_sync()) {
      auto scheduler =
          Scheduler(sm_idx, batch_size, context_lens, schedule_meta, indices);

      // Persistently schedule over blocks
      // Initialize outside valid range to indicate no previous task
      uint32_t last_q_atom_idx = batch_size * kNumNextNAtoms;
      uint32_t q_atom_idx, _, __;
      while (scheduler.fetch_next_task(q_atom_idx, _, __)) {
        // Issue TMA Q when (q_idx, atom_idx) changes
        if (q_atom_idx != last_q_atom_idx) {
          // Wait Q consumer release
          CUTE_TIE_DECL(advance_q_pipeline(), q_stage_idx, q_phase);
          empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);

          // Issue TMA Q
          const auto q_token_idx = Scheduler::atom_to_token_idx(q_atom_idx);
          cute::SM90_TMA_LOAD_2D::copy(
              &tensor_map_q,
              reinterpret_cast<uint64_t*>(full_q_barriers[q_stage_idx]),
              static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
              smem_q[q_stage_idx], 0, q_token_idx * kNumHeads);
          // The atom's scale rows, one per token, in one copy: the
          // descriptor's box is kNextNAtom rows. (One copy per row would put
          // the second at a 32-byte offset, which TMA rejects.)
          tma::copy<kNumHeads, kNextNAtom, 0>(
              &tensor_map_sf_q, full_q_barriers[q_stage_idx],
              smem_sf_q[q_stage_idx], 0, q_token_idx);
          full_q_barriers[q_stage_idx]->arrive_and_expect_tx(
              SMEM_Q_SIZE_PER_STAGE + kRealNumSFQAtom * sizeof(int));
        }
        last_q_atom_idx = q_atom_idx;
      }
    }
    __syncwarp();
  } else if (warp_idx == kSpecWarpStart + 1) {
    // TMA warp for loading KV cache
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    auto scheduler =
        Scheduler(sm_idx, batch_size, context_lens, schedule_meta, indices);

    // Persistently schedule over blocks
    uint32_t kv_block_idx_ptr = 32, kv_block_idx_storage;
    uint32_t last_q_atom_idx = batch_size * kNumNextNAtoms;
    uint32_t q_atom_idx, kv_idx, num_kv;
    while (scheduler.fetch_next_task(q_atom_idx, kv_idx, num_kv)) {
      // Reset block table cache on kv restart
      if (q_atom_idx != last_q_atom_idx) kv_block_idx_ptr = 32;
      last_q_atom_idx = q_atom_idx;

      // Coalesced load of block table
      if (kv_block_idx_ptr == 32) {
        kv_block_idx_ptr = 0;
        const auto block_table_offset =
            Scheduler::atom_to_block_table_row(q_atom_idx) *
            static_cast<uint64_t>(block_table_stride);
        kv_block_idx_storage =
            (kv_idx + lane_idx < num_kv)
                ? block_table[block_table_offset + kv_idx + lane_idx]
                : 0;
      }
      __syncwarp();

      // Broadcast KV block indices
      int kv_block_idx[kNumBlocksPerSplit];
#pragma unroll
      for (int i = 0; i < kNumBlocksPerSplit; ++i)
        kv_block_idx[i] =
            __shfl_sync(0xffffffff, kv_block_idx_storage, kv_block_idx_ptr + i);
      kv_block_idx_ptr += kNumBlocksPerSplit;
      DG_STATIC_ASSERT(32 % kNumBlocksPerSplit == 0, "Invalid `SPLIT_KV`");

      // Wait KV consumer release
      CUTE_TIE_DECL(advance_kv_pipeline(), kv_stage_idx, kv_phase);

      // Issue TMA KV
      if (cute::elect_one_sync()) {
        empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);
#pragma unroll
        for (int i = 0; i < kNumBlocksPerSplit; ++i) {
          cute::SM90_TMA_LOAD_3D::copy(
              &tensor_map_kv,
              reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
              static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
              smem_kv[kv_stage_idx] + (BLOCK_KV * kHeadDim) * i, 0, 0,
              kv_block_idx[i]);
          tma::copy<BLOCK_KV, 1, 0>(
              &tensor_map_sf_kv, full_kv_barriers[kv_stage_idx],
              smem_sf_kv[kv_stage_idx] + BLOCK_KV * i, 0, kv_block_idx[i]);
        }
        full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(
            TMA_KV_BYTES_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
      }
    }
  } else if (warp_idx == kSpecWarpStart + 2) {
    // UMMA warp
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    auto scheduler =
        Scheduler(sm_idx, batch_size, context_lens, schedule_meta, indices);
    DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

    // UTCCP transposer
    auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
      DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128,
                       "Invalid aligned elements");
      uint32_t values[4];
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i)
        values[i] = ptx::ld_shared(smem_ptr + i * 32 + lane_idx);
      __syncwarp();
      ptx::st_shared(smem_ptr + lane_idx * 4, values[0], values[1], values[2],
                     values[3]);
    };

    // Make UMMA desc
    auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
        cutlass::detail::float_e2m3_unpacksmem_t, cutlass::float_e4m3_t, float,
        cutlass::float_ue8m0_t, UMMA_M, UMMA_N, cute::UMMA::Major::K,
        cute::UMMA::Major::K>();
    auto sf_desc = mma::sm100::make_sf_desc(nullptr);

    // Persistently schedule over blocks
    uint32_t last_q_atom_idx = batch_size * kNumNextNAtoms;
    uint32_t q_atom_idx, kv_idx, _;
    while (scheduler.fetch_next_task(q_atom_idx, kv_idx, _)) {
      // Wait TMA Q arrivals
      uint32_t q_stage_idx, q_phase;
      if (q_atom_idx != last_q_atom_idx) {
        CUTE_TIE(advance_q_pipeline(), q_stage_idx, q_phase);

        // Release previous Q empty (UMMA warp must participate to prevent
        // running ahead of math warps in the Q pipeline)
        if (last_q_atom_idx != batch_size * kNumNextNAtoms)
          empty_q_barriers[(q_stage_idx + kNumQStages - 1) % kNumQStages]
              ->arrive();

        full_q_barriers[q_stage_idx]->wait(q_phase);

// Transpose and copy SF Q
#pragma unroll
        for (uint32_t i = 0; i < kNumSFQAtom / kNumUTCCPAlignedElems; ++i) {
          auto smem_ptr = smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
          utccp_required_smem_warp_transpose(smem_ptr);
          cutlass::arch::fence_view_async_shared();
          mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
          if (cute::elect_one_sync())
            cute::SM100_UTCCP_4x32dp128bit_1cta::copy(
                sf_desc, kTmemStartColOfSFQ + i * 4);
          __syncwarp();
        }
      }
      last_q_atom_idx = q_atom_idx;

      // Wait TMA KV arrivals
      CUTE_TIE_DECL(advance_kv_pipeline(), kv_stage_idx, kv_phase);
      full_kv_barriers[kv_stage_idx]->wait(kv_phase);

      // UMMA with SF
      if (cute::elect_one_sync()) {
// Copy SF KV
#pragma unroll
        for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++i) {
          auto smem_ptr = smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
          mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
          cute::SM100_UTCCP_4x32dp128bit_1cta::copy(
              sf_desc, kTmemStartColOfSFKV + i * 4);
        }

#pragma unroll
        for (uint32_t i = 0; i < kNumMathWarpGroups; ++i) {
          // Wait TMEM release
          CUTE_TIE_DECL(advance_tmem_pipeline(), tmem_stage_idx, tmem_phase);
          uint32_t tmem_addr = tmem_stage_idx * UMMA_N;

          empty_tmem_barriers[tmem_stage_idx]->wait(tmem_phase ^ 1);
          ptx::tcgen05_after_thread_sync();

// Issue UMMA with SF
#pragma unroll
          for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++k) {
            auto runtime_instr_desc =
                mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k,
                                                               k);
            DG_STATIC_ASSERT(kHeadDim == 128, "Invalid head dim");
            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0,
                                                     kHeadDim, kHeadDim>(
                smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
            auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0,
                                                     kHeadDim, kHeadDim>(
                smem_q[q_stage_idx], 0, k * UMMA_K);
            ptx::SM100_MMA_MXF8F6F4_SS::fma(
                a_desc, b_desc, tmem_addr, k, runtime_instr_desc,
                kTmemStartColOfSFKV + i * 4, kTmemStartColOfSFQ);
          }
          asm volatile(
              "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
              "cluster.b64 [%0];" ::"r"(cute::cast_smem_ptr_to_uint(
                  full_tmem_barriers[tmem_stage_idx])));
        }
      }
      cutlass::arch::umma_arrive(
          reinterpret_cast<uint64_t*>(empty_kv_barriers[kv_stage_idx]));
    }
  } else if (warp_idx == kSpecWarpStart + 3) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
  } else if (warp_idx < kSpecWarpStart) {
    // Math warpgroups for reduce
    cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
    auto scheduler =
        Scheduler(sm_idx, batch_size, context_lens, schedule_meta, indices);

    const auto math_warpgroup_idx = warpgroup_idx;
    const auto math_thread_idx = warp_idx * 32 + lane_idx;

    // Helper lambda for loading tensor memory
    auto tmem_load = [](auto num_elems_c, const uint32_t& tmem_addr,
                        float* accum) {
      constexpr int N = decltype(num_elems_c)::value;
      DG_STATIC_ASSERT(
          N == 1 or N == 2 or N == 4 or N == 8 or N == 16 or N == 32 or N == 64,
          "Unsupported TMEM load size");
      using Loader = cute::conditional_t<
          N == 1, cute::SM100_TMEM_LOAD_32dp32b1x,
          cute::conditional_t<
              N == 2, cute::SM100_TMEM_LOAD_32dp32b2x,
              cute::conditional_t<
                  N == 4, cute::SM100_TMEM_LOAD_32dp32b4x,
                  cute::conditional_t<
                      N == 8, cute::SM100_TMEM_LOAD_32dp32b8x,
                      cute::conditional_t<
                          N == 16, cute::SM100_TMEM_LOAD_32dp32b16x,
                          cute::conditional_t<
                              N == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
                              cute::SM100_TMEM_LOAD_32dp32b64x>>>>>>;
      [&]<size_t... Is>(cute::index_sequence<Is...>) {
        Loader::copy(tmem_addr, reinterpret_cast<uint32_t*>(accum)[Is]...);
      }(cute::make_index_sequence<N>{});
      cutlass::arch::fence_view_async_tmem_load();
    };

    // Math warpgroups process TMEM stages alternately
    // Advance pipeline to align with the assigned stage
    advance_tmem_pipeline(math_warpgroup_idx);

    // Local register buffers
    float accum[kNumHeads];

    // Persistently schedule over blocks
    uint32_t last_q_atom_idx = batch_size * kNumNextNAtoms;
    uint32_t q_atom_idx, kv_idx, num_kv;
    bool is_paired_atom = false;
    // The request's KV length, for masking the last page's tail, and the
    // warpgroup-local barrier the page reduction syncs on.
    int kv_len = 0;
    const uint32_t warp_in_group = warp_idx % 4;
    const uint32_t wg_barrier_id = 1 + math_warpgroup_idx;
    while (scheduler.fetch_next_task(q_atom_idx, kv_idx, num_kv)) {
      if (q_atom_idx != last_q_atom_idx) {
        kv_len = static_cast<int>(
            context_lens[Scheduler::atom_to_block_table_row(q_atom_idx)]);
        CUTE_TIE_DECL(advance_q_pipeline(), q_stage_idx, q_phase);

        // Release last Q empty
        if (last_q_atom_idx != batch_size * kNumNextNAtoms)
          empty_q_barriers[(q_stage_idx + kNumQStages - 1) % kNumQStages]
              ->arrive();

        // Wait TMA Q arrivals
        full_q_barriers[q_stage_idx]->wait(q_phase);

        // Check if this atom pairs two tokens from the same sequence
        if constexpr (kIsVarlen) {
          is_paired_atom = (scheduler.get_last_advance() == 2);
        }
      }
      last_q_atom_idx = q_atom_idx;

      // This warpgroup's page, and the KV position its thread holds.
      const uint32_t page_idx = kv_idx + math_warpgroup_idx;
      const int kv_pos =
          static_cast<int>(page_idx * BLOCK_KV + math_thread_idx -
                           math_warpgroup_idx * BLOCK_KV);
      const uint64_t score_offset =
          static_cast<uint64_t>(Scheduler::atom_to_token_idx(q_atom_idx)) *
              score_stride_q +
          page_idx;

      // Advance pipeline by `kNumMathWarpGroups` steps
      // Wait UMMA arrival
      CUTE_TIE_DECL(advance_tmem_pipeline(kNumMathWarpGroups), tmem_stage_idx,
                    tmem_phase);
      full_tmem_barriers[tmem_stage_idx]->wait(tmem_phase);
      ptx::tcgen05_after_thread_sync();

      // Adjust each query's causal bound and local window before page reduction.
      const uint32_t first_tok =
          Scheduler::atom_to_token_idx(q_atom_idx) -
          Scheduler::atom_to_block_table_row(q_atom_idx) * kNextN;
      const auto token_kv_len = [&](const uint32_t& i) -> int {
        if constexpr (kNextN == 1)
          return kv_len;
        else
          return kv_len - static_cast<int>(kNextN - 1 - (first_tok + i));
      };
      const auto token_num_kv = [&](const uint32_t& i) -> uint32_t {
        if constexpr (kNextN == 1) {
          return num_kv;
        } else {
          const int len = token_kv_len(i);
          return len > 0
                     ? (static_cast<uint32_t>(len) + BLOCK_KV - 1) / BLOCK_KV
                     : 0u;
        }
      };

      const auto reduce_and_store = [&](auto num_iters_c) {
        constexpr uint32_t kNumIters = decltype(num_iters_c)::value;

        // Release TMEM after all loads, before reduction, to unblock UMMA.
        // Each of the 128 threads must arrive at the barrier.
        float acc[kNumIters][kNumHeads];
#pragma unroll
        for (uint32_t i = 0; i < kNumIters; ++i) {
          uint32_t tmem_addr = tmem_stage_idx * UMMA_N + i * kNumHeads;
          tmem_load(cute::Int<kNumHeads / 2>{}, tmem_addr, acc[i]);
          tmem_load(cute::Int<kNumHeads / 2>{}, tmem_addr + kNumHeads / 2,
                    acc[i] + kNumHeads / 2);
        }
        ptx::tcgen05_before_thread_sync();
        empty_tmem_barriers[tmem_stage_idx]->arrive();

#pragma unroll
        for (uint32_t i = 0; i < kNumIters; ++i) {
          // The last split of an odd-length request carries a page that
          // does not exist. It still has to reach the TMEM release.
          const uint32_t tok_num_kv = token_num_kv(i);
          const bool page_exists = page_idx < tok_num_kv;
          const bool visible = kv_pos < token_kv_len(i);
          // A token past the causal bound must not win the max nor
          // move the sum. Only the first `kNumRealHeads` rows are
          // real; the rest exist because tcgen05 needs N % 8 == 0.
          float v[kNumRealHeads];
#pragma unroll
          for (uint32_t h = 0; h < kNumRealHeads; ++h)
            v[h] =
                visible ? acc[i][h] * sm_scale : (score_sum ? 0.0f : -INFINITY);

#pragma unroll
          for (uint32_t off = 16; off > 0; off >>= 1)
#pragma unroll
            for (uint32_t h = 0; h < kNumRealHeads; ++h) {
              const float o = __shfl_xor_sync(0xffffffff, v[h], off);
              v[h] = score_sum ? v[h] + o : fmaxf(v[h], o);
            }

          if (lane_idx == 0) {
#pragma unroll
            for (uint32_t h = 0; h < kNumRealHeads; ++h)
              smem_page_reduce[warp_idx][h] = v[h];
          }
          asm volatile("barrier.sync %0, %1;" ::"r"(wg_barrier_id), "r"(128)
                       : "memory");

          if (warp_in_group == 0 and lane_idx < kNumRealHeads and page_exists) {
            const uint32_t base = math_warpgroup_idx * 4;
            float s = smem_page_reduce[base][lane_idx];
#pragma unroll
            for (uint32_t w = 1; w < 4; ++w)
              s = score_sum ? s + smem_page_reduce[base + w][lane_idx]
                            : fmaxf(s, smem_page_reduce[base + w][lane_idx]);
            // Forced pages carry sentinels rather than +inf so the
            // selector can still order them against each other.
            if (page_idx + local_blocks >= tok_num_kv)
              s = 1e29f;
            else if (page_idx < init_blocks)
              s = 1e30f;
            scores[score_offset + i * score_stride_q +
                   lane_idx * static_cast<uint64_t>(score_stride_h)] = s;
          }
          // The next page -- this iteration's or the next task's --
          // overwrites what warp 0 has just read.
          asm volatile("barrier.sync %0, %1;" ::"r"(wg_barrier_id), "r"(128)
                       : "memory");
        }
      };

      if constexpr (kIsVarlen) {
        if (is_paired_atom)
          reduce_and_store(cute::Int<kNextNAtom>{});
        else
          reduce_and_store(cute::Int<1>{});
      } else if constexpr (kPadOddN) {
        if (q_atom_idx % kNumNextNAtoms == kNumNextNAtoms - 1)
          reduce_and_store(cute::Int<1>{});
        else
          reduce_and_store(cute::Int<kNextNAtom>{});
      } else {
        reduce_and_store(cute::Int<kNextNAtom>{});
      }
    }

    // Trigger PDL Top-K only after this CTA has stored all scores.
    cudaTriggerProgrammaticLaunchCompletion();
    // Free tensor memory
    cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
    if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
  }
}

}  // namespace deep_gemm
