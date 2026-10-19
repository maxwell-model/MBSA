// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 Maxwell Team
// SPDX-License-Identifier: MIT
//
// Derived from DeepGEMM's paged-MQA-logits operator and host launcher:
// csrc/jit_kernels/impls/smxx_fp8_fp4_paged_mqa_logits.hpp
// Adapted for MBSA MXFP6 index scoring; see
// third_party/deep_gemm/NOTICE.

// DeepGEMM-derived MXFP6 decode launcher for 1..4 queries per request.
// Reads packed codes/scales and writes per-head page max or sum scores.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda.h>

#include <map>
#include <mutex>
#include <tuple>

// Derived from deep_gemm's FP4 paged-MQA-logits kernel.
#include "sm100_fp6_paged_mqa_logits.cuh"
#include "paged_mqa_logits_scheduler.cuh"

namespace {

constexpr int PAGE = 128;
constexpr int DIM = 128;
constexpr int HEADS = 4;               // index heads, one per KV head
constexpr int NQ = 8;                  // padded to the legal minimum for N
constexpr int CODE_BYTES = PAGE * 96;  // 12288
constexpr int PAGE_BYTES = CODE_BYTES + PAGE * 4;  // 12800
constexpr int SPLIT_KV = 256;
constexpr int Q_STAGES = 3;
constexpr int KV_STAGES = 3;
constexpr int SPEC_THREADS = 128;
constexpr int MATH_THREADS = 256;
// Scheduler metadata and kernel launch must use the same SM count.
constexpr int NUM_SMS = 148;  // A different target may require this to change.

// Cache tensor-map descriptors per pointer, rank and shape to avoid repeated
// cuTensorMapEncodeTiled driver calls on the decode path.
struct MapKey {
  uintptr_t ptr;
  int type;
  uint64_t d0, d1, d2;
  uint32_t b1;  // the box: one buffer is read with one row or two tokens' rows
  bool operator<(const MapKey& o) const {
    return std::tie(ptr, type, d0, d1, d2, b1) <
           std::tie(o.ptr, o.type, o.d0, o.d1, o.d2, o.b1);
  }
};
std::map<MapKey, CUtensorMap> g_maps;
std::mutex g_mu;

CUtensorMap encode_uncached(void* ptr, CUtensorMapDataType type, int rank,
                            const uint64_t* gdim, const uint64_t* gstride,
                            const uint32_t* bdim, CUtensorMapSwizzle swizzle) {
  CUtensorMap map{};
  uint32_t estride[3] = {1, 1, 1};
  const auto rc = cuTensorMapEncodeTiled(
      &map, type, rank, ptr, gdim, gstride, bdim, estride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(rc == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed (rc=", (int)rc,
              "); check the 16-byte base alignment and that row strides are "
              "multiples of 32 bytes");
  return map;
}

const CUtensorMap& encode(void* ptr, CUtensorMapDataType type, int rank,
                          const uint64_t* gdim, const uint64_t* gstride,
                          const uint32_t* bdim, CUtensorMapSwizzle swizzle) {
  const MapKey key{
      reinterpret_cast<uintptr_t>(ptr), (int)type, gdim[0], gdim[1],
      rank > 2 ? gdim[2] : 0,           bdim[1]};
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_maps.find(key);
  if (it != g_maps.end()) return it->second;
  return g_maps
      .emplace(key,
               encode_uncached(ptr, type, rank, gdim, gstride, bdim, swizzle))
      .first->second;
}

}  // namespace

// Schedule: int32 [num_sms+1,4] for one request/query, [num_sms+1,2] otherwise.
// Layers sharing context lengths and next_n may reuse it.
torch::Tensor dg_schedule_metadata(const torch::Tensor& context_lens,
                                   int64_t num_sms, int64_t next_n) {
  TORCH_CHECK(num_sms == NUM_SMS,
              "this kernel's schedule metadata is compiled for ", NUM_SMS,
              " SMs; got ", num_sms);
  const int64_t batch = context_lens.numel();
  constexpr uint32_t kAligned = 256;
  TORCH_CHECK(batch <= kAligned, "batch ", batch, " exceeds ", kAligned);
  TORCH_CHECK(next_n >= 1 && next_n <= 4, "next_n ", next_n, " not in 1..4");
  if (batch == 1 && next_n == 1) {
    auto meta = torch::empty({num_sms + 1, 4},
                             context_lens.options().dtype(torch::kInt32));
    deep_gemm::maxwell_decode_schedule<kAligned, SPLIT_KV, NUM_SMS, PAGE>
        <<<1, 32, 0, c10::cuda::getCurrentCUDAStream()>>>(
            (uint32_t)batch,
            reinterpret_cast<const uint32_t*>(context_lens.data_ptr<int>()),
            reinterpret_cast<uint32_t*>(meta.data_ptr<int>()));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "metadata launch: ", cudaGetErrorString(cudaGetLastError()));
    return meta;
  }
  auto meta = torch::empty({num_sms + 1, 2},
                           context_lens.options().dtype(torch::kInt32));
  const int smem = kAligned * 4 * sizeof(uint32_t);
  auto kern =
      deep_gemm::sched::smxx_paged_mqa_logits_metadata<kAligned, SPLIT_KV,
                                                       NUM_SMS, false>;
  TORCH_CHECK(
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem) == cudaSuccess,
      "metadata smem");
  kern<<<1, 32, smem, c10::cuda::getCurrentCUDAStream()>>>(
      (uint32_t)batch, (uint32_t)next_n, false,
      /*num_next_n_atoms=*/(uint32_t)((next_n + 1) / 2),
      reinterpret_cast<const uint32_t*>(context_lens.data_ptr<int>()), nullptr,
      reinterpret_cast<uint32_t*>(meta.data_ptr<int>()));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "metadata launch: ", cudaGetErrorString(cudaGetLastError()));
  return meta;
}

// Returns [HEADS, total_q, score_width] fp32 page scores -- the same tensor
// the hand-written kernel returns, so `maxwell_index_topk_from_scores` reads
// it unchanged. `meta` is dg_schedule_metadata of these `context_lens`.
torch::Tensor dg_index_score(
    const torch::Tensor& q_bytes, const torch::Tensor& sf_q,
    const torch::Tensor& cache, const torch::Tensor& context_lens,
    const torch::Tensor& block_table, const torch::Tensor& meta,
    int64_t score_width, int64_t init_blocks, int64_t local_blocks,
    bool score_sum, double sm_scale, int64_t num_sms, int64_t next_n) {
  TORCH_CHECK(num_sms == NUM_SMS,
              "this kernel's schedule metadata is compiled for ", NUM_SMS,
              " SMs; got ", num_sms);
  // Requests and query rows differ once a request carries next_n > 1 queries:
  // the scheduler, the metadata and the kernel's batch_size count requests;
  // the scores and the Q descriptors count rows.
  const int64_t num_reqs = context_lens.numel();
  const int64_t total_q = q_bytes.size(0);
  TORCH_CHECK(next_n >= 1 && next_n <= 4, "next_n ", next_n, " not in 1..4");
  TORCH_CHECK(total_q == num_reqs * next_n, total_q, " query rows for ",
              num_reqs, " requests of ", next_n, " queries");
  const bool single = num_reqs == 1 && next_n == 1;
  TORCH_CHECK(meta.numel() == (single ? 4 : 2) * (num_sms + 1) &&
                  meta.scalar_type() == torch::kInt32,
              "meta must be dg_schedule_metadata of these context_lens");
  TORCH_CHECK(cache.scalar_type() == torch::kUInt8 && cache.is_contiguous(),
              "the packed index-K cache is a contiguous uint8 tensor");
  TORCH_CHECK(q_bytes.size(-1) == DIM && q_bytes.size(-2) == NQ,
              "index-Q is [batch, ", NQ, ", ", DIM, "] E4M3 bytes");
  const int64_t atom = next_n >= 2 ? 2 : 1;  // tokens per Q load (kNextNAtom)
  const int64_t pages = cache.numel() / PAGE_BYTES;

  // A request's page count never exceeds its block-table row, so a score
  // buffer that covers the row covers every page the kernel can write.
  TORCH_CHECK(score_width >= block_table.size(1), "score_width ", score_width,
              " is narrower than the block table's ", block_table.size(1));
  // Uninitialised on purpose, like the hand-written kernel's: pages past a
  // request's own page count are never written and never read, because the
  // selector bounds itself by `seq_lens`.
  auto scores = torch::empty({(int64_t)HEADS, total_q, score_width},
                             q_bytes.options().dtype(torch::kFloat32));

  auto stream = c10::cuda::getCurrentCUDAStream();

  // Five descriptors. The KV pair reads the two regions of each packed page.
  auto* base = static_cast<uint8_t*>(cache.data_ptr());
  uint64_t kv_g[3] = {DIM, PAGE, (uint64_t)pages};
  uint64_t kv_s[2] = {96, PAGE_BYTES};
  uint32_t kv_b[3] = {DIM, PAGE, 1};
  const auto& map_kv = encode(base, CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B, 3,
                              kv_g, kv_s, kv_b, CU_TENSOR_MAP_SWIZZLE_128B);

  uint64_t sfkv_g[2] = {PAGE, (uint64_t)pages};
  uint64_t sfkv_s[1] = {PAGE_BYTES};
  uint32_t sfkv_b[2] = {PAGE, 1};
  const auto& map_sf_kv =
      encode(base + CODE_BYTES, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, sfkv_g,
             sfkv_s, sfkv_b, CU_TENSOR_MAP_SWIZZLE_NONE);

  uint64_t q_g[2] = {DIM, (uint64_t)(total_q * NQ)};
  uint64_t q_s[1] = {DIM};
  uint32_t q_b[2] = {DIM, (uint32_t)(atom * NQ)};
  const auto& map_q = encode(q_bytes.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                             2, q_g, q_s, q_b, CU_TENSOR_MAP_SWIZZLE_128B);

  uint64_t sfq_g[2] = {NQ, (uint64_t)total_q};
  uint64_t sfq_s[1] = {NQ * 4};
  uint32_t sfq_b[2] = {NQ, (uint32_t)atom};
  const auto& map_sf_q =
      encode(sf_q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, sfq_g, sfq_s,
             sfq_b, CU_TENSOR_MAP_SWIZZLE_NONE);

  auto kernel_for = [](auto next_n_c, auto single_request) {
    return deep_gemm::sm100_fp6_paged_mqa_logits<
        /*kNextN=*/decltype(next_n_c)::value, /*kNumHeads=*/NQ,
        /*kNumRealHeads=*/HEADS,
        /*kHeadDim=*/DIM, /*BLOCK_KV=*/PAGE,
        /*kIsContextLens2D=*/false, /*kIsVarlen=*/false, Q_STAGES, KV_STAGES,
        SPLIT_KV, SPEC_THREADS, MATH_THREADS, float, MATH_THREADS / 128,
        decltype(single_request)::value>;
  };

  // Match kernel storage: Q/KV stages, padded scales, weight strides,
  // barrier pairs and the TMEM pointer.
  constexpr auto align128 = [](int x) { return (x + 127) / 128 * 128; };
  constexpr int kTmemStages = 3;
  const int smem = Q_STAGES * atom * NQ * DIM + KV_STAGES * SPLIT_KV * DIM +
                   Q_STAGES * align128(NQ) * 4 +
                   KV_STAGES * align128(SPLIT_KV) * 4 +
                   Q_STAGES * (NQ * 4 < 128 ? 128 : NQ * 4) +
                   (2 * Q_STAGES + 2 * KV_STAGES + 2 * kTmemStages) * 8 + 4;
  auto launch = [&](auto kernel) {
    TORCH_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    smem) == cudaSuccess,
                "the kernel wants ", smem, " bytes of shared memory");
    kernel<<<num_sms, SPEC_THREADS + MATH_THREADS, smem, stream>>>(
        (uint32_t)num_reqs, (uint32_t)(total_q * score_width),
        (uint32_t)score_width, (uint32_t)block_table.stride(0),
        reinterpret_cast<const uint32_t*>(context_lens.data_ptr<int>()),
        scores.data_ptr<float>(), (uint32_t)init_blocks, (uint32_t)local_blocks,
        score_sum, (float)sm_scale,
        reinterpret_cast<const uint32_t*>(block_table.data_ptr<int>()), nullptr,
        reinterpret_cast<const uint32_t*>(meta.data_ptr<int>()), map_q,
        map_sf_q, map_kv, map_sf_kv);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "kernel launch: ", cudaGetErrorString(cudaGetLastError()));
  };
  using N1 = std::integral_constant<int, 1>;
  if (single)
    launch(kernel_for(N1{}, std::true_type{}));
  else if (next_n == 1)
    launch(kernel_for(N1{}, std::false_type{}));
  else if (next_n == 2)
    launch(kernel_for(std::integral_constant<int, 2>{}, std::false_type{}));
  else if (next_n == 3)
    launch(kernel_for(std::integral_constant<int, 3>{}, std::false_type{}));
  else
    launch(kernel_for(std::integral_constant<int, 4>{}, std::false_type{}));
  return scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dg_schedule_metadata", &dg_schedule_metadata,
        "DeepGEMM paged-MQA schedule metadata for these context lengths");
  m.def("dg_index_score", &dg_index_score,
        "DeepGEMM-pipelined MXFP6 per-head page scores");
}
