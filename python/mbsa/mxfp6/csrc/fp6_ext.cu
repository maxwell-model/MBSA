// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-FileCopyrightText: Copyright (c) 2026 Maxwell Team
// SPDX-License-Identifier: Apache-2.0
// Derived from the vLLM Maxwell-V1-Flash fork; modified for MBSA.
// See third_party/vllm/PROVENANCE.md.

// Torch binding for the MXFP6 scorer; tensor maps are cached by pointer/pages.
// The cache allocation must outlive any graph capturing its descriptor.
#include <torch/extension.h>
// Deliberately NOT <ATen/cuda/CUDAContext.h>: it pulls in cusparse.h, which
// this container's CUDA install does not ship. The stream is all we need.
#include <c10/cuda/CUDAStream.h>

#include <map>
#include <optional>
#include <mutex>

#include "fp6_index_kernel.cuh"

namespace {

struct MapKey {
  uintptr_t ptr;
  int64_t rows;
  bool packed;
  bool operator<(const MapKey& o) const {
    return std::tie(ptr, rows, packed) < std::tie(o.ptr, o.rows, o.packed);
  }
};

std::map<MapKey, CUtensorMap> g_maps;
std::mutex g_mu;

const CUtensorMap& kv_tensor_map(const torch::Tensor& cache,
                                 int64_t num_pages) {
  const MapKey key{reinterpret_cast<uintptr_t>(cache.data_ptr()), num_pages,
                   true};
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_maps.find(key);
  if (it != g_maps.end()) return it->second;
  CUtensorMap map{};
  uint64_t gdim[3] = {DIM, PAGE, (uint64_t)num_pages};
  uint64_t gstride[2] = {A_ROW_BYTES, PAGE_BYTES};
  uint32_t bdim[3] = {DIM, PAGE, 1};
  uint32_t estride[3] = {1, 1, 1};
  TORCH_CHECK(
      cuTensorMapEncodeTiled(
          &map, CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B, 3, cache.data_ptr(),
          gdim, gstride, bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS,
      "could not encode the index-K tensor map; the cache base must be "
      "16-byte aligned and the page stride a multiple of 32 bytes");
  return g_maps.emplace(key, map).first->second;
}

const CUtensorMap& q_tensor_map(const torch::Tensor& q, int64_t rows) {
  const MapKey key{reinterpret_cast<uintptr_t>(q.data_ptr()), rows, false};
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_maps.find(key);
  if (it != g_maps.end()) return it->second;
  CUtensorMap map{};
  uint64_t gdim[2] = {DIM, (uint64_t)rows};
  uint64_t gstride[1] = {DIM};
  uint32_t bdim[2] = {DIM, NQ};
  uint32_t estride[2] = {1, 1};
  TORCH_CHECK(cuTensorMapEncodeTiled(
                  &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, q.data_ptr(), gdim,
                  gstride, bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                  CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS,
              "could not encode the index-Q tensor map");
  return g_maps.emplace(key, map).first->second;
}

}  // namespace

// Returns [HEADS, total_q, max_blocks] fp32 page scores, the layout the
// existing Triton top-k already consumes.
torch::Tensor index_score_mxfp6(
    const torch::Tensor& q_bytes, const torch::Tensor& sf_q,
    const torch::Tensor& cache, const torch::Tensor& block_table,
    const torch::Tensor& seq_lens, const std::optional<torch::Tensor>& q_to_req,
    int64_t max_blocks, int64_t score_width, int64_t num_chunks,
    int64_t init_blocks, int64_t local_blocks, bool score_sum,
    double sm_scale) {
  TORCH_CHECK(cache.scalar_type() == torch::kUInt8 && cache.is_contiguous(),
              "the packed index-K cache is a contiguous uint8 tensor");
  TORCH_CHECK(cache.numel() % PAGE_BYTES == 0,
              "the cache must be a whole number of ", PAGE_BYTES,
              "-byte pages");
  TORCH_CHECK(q_bytes.scalar_type() == torch::kUInt8 && q_bytes.is_contiguous(),
              "index-Q arrives as E4M3 bytes; pass .view(torch.uint8)");
  TORCH_CHECK(q_bytes.size(-1) == DIM && q_bytes.size(-2) == NQ,
              "index-Q must be [total_q, ", NQ, ", ", DIM,
              "]: 4 real heads padded to 8, because tcgen05 needs N % 8 == 0");
  TORCH_CHECK(sf_q.scalar_type() == torch::kInt32 && sf_q.is_contiguous(),
              "the Q scales are one int32 of four UE8M0 bytes per row");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32 &&
                  seq_lens.scalar_type() == torch::kInt32,
              "block table and sequence lengths are int32");
  const int64_t total_q = q_bytes.size(0);
  TORCH_CHECK(seq_lens.size(0) == total_q,
              "seq_lens is indexed by query token: one causal bound each");
  const int* q_to_req_ptr = nullptr;
  if (q_to_req.has_value()) {
    TORCH_CHECK(q_to_req->scalar_type() == torch::kInt32 &&
                    q_to_req->size(0) == total_q,
                "q_to_req is one int32 block-table row per query token");
    q_to_req_ptr = q_to_req->data_ptr<int>();
  }
  // The caller picks the width so the score tensor can match whatever the
  // selector expects (the Triton path pads it to a 16-divisible stride).
  auto scores = torch::empty({(int64_t)HEADS, total_q, score_width},
                             q_bytes.options().dtype(torch::kFloat32));
  const auto& kv_map = kv_tensor_map(cache, cache.numel() / PAGE_BYTES);
  const auto& qmap = q_tensor_map(q_bytes, total_q * NQ);

  const int smem = STAGES * A_SMEM + B_SMEM + (STAGES * PAGE + PAGE) * 4 + 1024;
  static bool attr_set = [&] {
    TORCH_CHECK(cudaFuncSetAttribute(
                    index_decode, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    smem) == cudaSuccess,
                "the kernel needs ", smem,
                " bytes of shared memory; lower STAGES if the device refuses");
    return true;
  }();
  (void)attr_set;
  dim3 grid((unsigned)total_q, (unsigned)num_chunks);
  index_decode<<<grid, NUM_THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(
      kv_map, qmap, static_cast<const uint8_t*>(cache.data_ptr()),
      block_table.data_ptr<int>(), (int)block_table.stride(0),
      seq_lens.data_ptr<int>(), q_to_req_ptr,
      // int32 carrying four UE8M0 bytes; torch has no uint32 view here.
      reinterpret_cast<const uint32_t*>(sf_q.data_ptr<int32_t>()),
      (int)num_chunks, (int)max_blocks, (int)init_blocks, (int)local_blocks,
      score_sum ? 1 : 0, (float)sm_scale, scores.data_ptr<float>(),
      (int)(total_q * score_width), (int)score_width);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "index_score_mxfp6 launch: ",
              cudaGetErrorString(cudaGetLastError()));
  return scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("index_score_mxfp6", &index_score_mxfp6,
        "MXFP6 index-K x MXFP8 index-Q page scores");
}
