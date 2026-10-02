#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kValuesPerThread = 16;
constexpr int kValuesPerBlock = kThreads * kValuesPerThread;

__device__ __forceinline__ uint4 convert_bf16x16_to_e4m3(
    uint4 lo, uint4 hi) {
  uint32_t input[8] = {
      lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w,
  };
  uint16_t output[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    __nv_bfloat162_raw pair{
        static_cast<unsigned short>(input[i] & 0xffffU),
        static_cast<unsigned short>(input[i] >> 16U),
    };
    output[i] = __nv_cvt_bfloat16raw2_to_fp8x2(
        pair, __NV_SATFINITE, __NV_E4M3);
  }
  uint4 result;
  result.x = static_cast<uint32_t>(output[0]) |
             (static_cast<uint32_t>(output[1]) << 16U);
  result.y = static_cast<uint32_t>(output[2]) |
             (static_cast<uint32_t>(output[3]) << 16U);
  result.z = static_cast<uint32_t>(output[4]) |
             (static_cast<uint32_t>(output[5]) << 16U);
  result.w = static_cast<uint32_t>(output[6]) |
             (static_cast<uint32_t>(output[7]) << 16U);
  return result;
}

__device__ __forceinline__ uint4 convert_scaled_bf16x16_to_e4m3(
    uint4 lo, uint4 hi, float inverse_scale) {
  uint32_t input[8] = {
      lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w,
  };
  uint16_t output[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    __nv_bfloat162_raw raw{
        static_cast<unsigned short>(input[i] & 0xffffU),
        static_cast<unsigned short>(input[i] >> 16U),
    };
    const __nv_bfloat162 pair = static_cast<__nv_bfloat162>(raw);
    float2 scaled = __bfloat1622float2(pair);
    scaled.x *= inverse_scale;
    scaled.y *= inverse_scale;
    output[i] = __nv_cvt_float2_to_fp8x2(
        scaled, __NV_SATFINITE, __NV_E4M3);
  }
  uint4 result;
  result.x = static_cast<uint32_t>(output[0]) |
             (static_cast<uint32_t>(output[1]) << 16U);
  result.y = static_cast<uint32_t>(output[2]) |
             (static_cast<uint32_t>(output[3]) << 16U);
  result.z = static_cast<uint32_t>(output[4]) |
             (static_cast<uint32_t>(output[5]) << 16U);
  result.w = static_cast<uint32_t>(output[6]) |
             (static_cast<uint32_t>(output[7]) << 16U);
  return result;
}

__global__ void raw_pair_quant_kernel(
    const uint4* q, uint4* q_out, int64_t q_values,
    const uint4* k, uint4* k_out, int64_t k_values,
    int q_blocks) {
  const int block = static_cast<int>(blockIdx.x);
  const bool is_q = block < q_blocks;
  const int local_block = is_q ? block : block - q_blocks;
  const int64_t base =
      (static_cast<int64_t>(local_block) * blockDim.x + threadIdx.x) *
      kValuesPerThread;
  const int64_t values = is_q ? q_values : k_values;
  if (base >= values) {
    return;
  }
  // Indexing uint4 counts 8 BF16 values on input and 16 FP8 values on output.
  const uint4* input = is_q ? q : k;
  uint4* output = is_q ? q_out : k_out;
  const int64_t input_index = base / 8;
  const int64_t output_index = base / 16;
  const uint4 lo = input[input_index];
  const uint4 hi = input[input_index + 1];
  output[output_index] = convert_bf16x16_to_e4m3(lo, hi);
}

__global__ void static_pair_quant_kernel(
    const uint4* q, uint4* q_out, int64_t q_values, float q_inverse_scale,
    const uint4* k, uint4* k_out, int64_t k_values, float k_inverse_scale,
    int q_blocks) {
  const int block = static_cast<int>(blockIdx.x);
  const bool is_q = block < q_blocks;
  const int local_block = is_q ? block : block - q_blocks;
  const int64_t base =
      (static_cast<int64_t>(local_block) * blockDim.x + threadIdx.x) *
      kValuesPerThread;
  const int64_t values = is_q ? q_values : k_values;
  if (base >= values) {
    return;
  }
  const uint4* input = is_q ? q : k;
  uint4* output = is_q ? q_out : k_out;
  const float inverse_scale = is_q ? q_inverse_scale : k_inverse_scale;
  const int64_t input_index = base / 8;
  const int64_t output_index = base / 16;
  const uint4 lo = input[input_index];
  const uint4 hi = input[input_index + 1];
  output[output_index] =
      convert_scaled_bf16x16_to_e4m3(lo, hi, inverse_scale);
}

__device__ __forceinline__ float bf16x8_abs_max(uint4 packed) {
  const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
  float result = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    __nv_bfloat162_raw raw{
        static_cast<unsigned short>(words[i] & 0xffffU),
        static_cast<unsigned short>(words[i] >> 16U),
    };
    const float2 values = __bfloat1622float2(
        static_cast<__nv_bfloat162>(raw));
    result = fmaxf(result, fabsf(values.x));
    result = fmaxf(result, fabsf(values.y));
  }
  return result;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
  }
  return value;
}

__global__ void block_pair_quant_kernel(
    const uint4* q, uint4* q_out, float* q_scales, int q_rows,
    int q_owners, const uint4* k, uint4* k_out, float* k_scales,
    int k_pages) {
  __shared__ float warp_maxima[8];
  const int owner = static_cast<int>(blockIdx.x);
  const bool is_q = owner < q_owners;
  const int local_owner = is_q ? owner : owner - q_owners;
  const int q_block = local_owner >> 2;
  const int group = local_owner & 3;
  const int row = static_cast<int>(threadIdx.x) >> 1;
  const int half = static_cast<int>(threadIdx.x) & 1;
  const int global_row = q_block * 128 + row;
  const bool valid = is_q ? global_row < q_rows : local_owner < k_pages;
  const int64_t element_base = is_q
      ? (static_cast<int64_t>(global_row) * 4 + group) * 128 + half * 64
      : static_cast<int64_t>(local_owner) * 16384 +
            static_cast<int64_t>(threadIdx.x) * 64;
  const uint4* input = is_q ? q : k;
  uint4 cached[8];
  float local_max = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    cached[i] = valid ? input[element_base / 8 + i] : make_uint4(0, 0, 0, 0);
    local_max = fmaxf(local_max, bf16x8_abs_max(cached[i]));
  }

  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  local_max = warp_max(local_max);
  if (lane == 0) {
    warp_maxima[warp] = local_max;
  }
  __syncthreads();
  if (warp == 0) {
    float block_max = lane < 8 ? warp_maxima[lane] : 0.0f;
    block_max = warp_max(block_max);
    if (lane == 0) {
      const float scale = block_max > 0.0f ? block_max / 448.0f : 1.0f;
      // Preserve the frozen codec's FP32 rounding order: publish rounded
      // scale first, then form its reciprocal after the CTA hand-off.
      warp_maxima[0] = scale;
      (is_q ? q_scales : k_scales)[local_owner] = scale;
    }
  }
  __syncthreads();
  if (!valid) {
    return;
  }
  // Triton's frozen codec lowers the reciprocal to div.full.f32.  Keep that
  // exact rounding even though the extension is otherwise built with
  // --use_fast_math, because E4M3 tie cases can land in adjacent buckets.
  const float inverse_scale = __fdiv_rn(1.0f, warp_maxima[0]);
  uint4* output = is_q ? q_out : k_out;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    output[element_base / 16 + i] = convert_scaled_bf16x16_to_e4m3(
        cached[2 * i], cached[2 * i + 1], inverse_scale);
  }
}

void check_pair(
    const torch::Tensor& input, const torch::Tensor& output,
    const char* name) {
  TORCH_CHECK(input.is_cuda(), name, " input must be CUDA");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16,
              name, " input must be BF16");
  TORCH_CHECK(input.is_contiguous(), name, " input must be contiguous");
  TORCH_CHECK(output.is_cuda() && output.device() == input.device(),
              name, " output must be on the same CUDA device");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::Float8_e4m3fn,
              name, " output must be E4M3FN");
  TORCH_CHECK(output.sizes() == input.sizes() && output.is_contiguous(),
              name, " output shape/layout must match input");
  TORCH_CHECK(input.numel() % kValuesPerThread == 0,
              name, " element count must be divisible by 16");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(input.data_ptr()) % alignof(uint4) == 0 &&
          reinterpret_cast<uintptr_t>(output.data_ptr()) % alignof(uint4) == 0,
      name, " input/output must be 16-byte aligned");
}

}  // namespace

void quantize_raw_pair_out(
    const torch::Tensor& q, const torch::Tensor& q_out,
    const torch::Tensor& k, const torch::Tensor& k_out) {
  check_pair(q, q_out, "Q");
  check_pair(k, k_out, "K");
  TORCH_CHECK(q.device() == k.device(), "Q and K must share one device");
  c10::cuda::CUDAGuard guard(q.device());
  const int q_blocks = static_cast<int>(
      (q.numel() + kValuesPerBlock - 1) / kValuesPerBlock);
  const int k_blocks = static_cast<int>(
      (k.numel() + kValuesPerBlock - 1) / kValuesPerBlock);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.device().index());
  raw_pair_quant_kernel<<<q_blocks + k_blocks, kThreads, 0, stream>>>(
      static_cast<const uint4*>(q.data_ptr()),
      static_cast<uint4*>(q_out.data_ptr()), q.numel(),
      static_cast<const uint4*>(k.data_ptr()),
      static_cast<uint4*>(k_out.data_ptr()), k.numel(), q_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void quantize_static_pair_out(
    const torch::Tensor& q, const torch::Tensor& q_out, double q_scale,
    const torch::Tensor& k, const torch::Tensor& k_out, double k_scale) {
  check_pair(q, q_out, "Q");
  check_pair(k, k_out, "K");
  TORCH_CHECK(q.device() == k.device(), "Q and K must share one device");
  TORCH_CHECK(q_scale > 0.0 && std::isfinite(q_scale),
              "Q scale must be finite and positive");
  TORCH_CHECK(k_scale > 0.0 && std::isfinite(k_scale),
              "K scale must be finite and positive");
  c10::cuda::CUDAGuard guard(q.device());
  const int q_blocks = static_cast<int>(
      (q.numel() + kValuesPerBlock - 1) / kValuesPerBlock);
  const int k_blocks = static_cast<int>(
      (k.numel() + kValuesPerBlock - 1) / kValuesPerBlock);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.device().index());
  static_pair_quant_kernel<<<q_blocks + k_blocks, kThreads, 0, stream>>>(
      static_cast<const uint4*>(q.data_ptr()),
      static_cast<uint4*>(q_out.data_ptr()), q.numel(),
      static_cast<float>(1.0 / q_scale),
      static_cast<const uint4*>(k.data_ptr()),
      static_cast<uint4*>(k_out.data_ptr()), k.numel(),
      static_cast<float>(1.0 / k_scale), q_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void quantize_block_pair_out(
    const torch::Tensor& q, const torch::Tensor& q_out,
    const torch::Tensor& q_scales, const torch::Tensor& k,
    const torch::Tensor& k_out, const torch::Tensor& k_scales) {
  check_pair(q, q_out, "Q");
  check_pair(k, k_out, "K");
  TORCH_CHECK(q.device() == k.device(), "Q and K must share one device");
  TORCH_CHECK(q.dim() == 3 && q.size(1) == 4 && q.size(2) == 128,
              "Q must be [rows,4,128]");
  TORCH_CHECK(k.dim() == 4 && k.size(1) == 1 && k.size(2) == 128 &&
                  k.size(3) == 128,
              "K must be [pages,1,128,128]");
  const int q_blocks = static_cast<int>((q.size(0) + 127) / 128);
  const int q_owners = q_blocks * 4;
  TORCH_CHECK(q_scales.is_cuda() && q_scales.device() == q.device() &&
                  q_scales.scalar_type() == at::kFloat &&
                  q_scales.is_contiguous() && q_scales.numel() == q_owners,
              "Q scales must be contiguous CUDA FP32 [ceil(rows/128),4]");
  TORCH_CHECK(k_scales.is_cuda() && k_scales.device() == k.device() &&
                  k_scales.scalar_type() == at::kFloat &&
                  k_scales.is_contiguous() && k_scales.numel() == k.size(0),
              "K scales must be contiguous CUDA FP32 [pages]");
  c10::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.device().index());
  block_pair_quant_kernel<<<q_owners + k.size(0), 256, 0, stream>>>(
      static_cast<const uint4*>(q.data_ptr()),
      static_cast<uint4*>(q_out.data_ptr()), q_scales.data_ptr<float>(),
      static_cast<int>(q.size(0)), q_owners,
      static_cast<const uint4*>(k.data_ptr()),
      static_cast<uint4*>(k_out.data_ptr()), k_scales.data_ptr<float>(),
      static_cast<int>(k.size(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("quantize_raw_pair_out", &quantize_raw_pair_out,
             "Packed BF16x2 to unit-scale E4M3 Q/K conversion");
  module.def("quantize_static_pair_out", &quantize_static_pair_out,
             "Packed scaled BF16 to E4M3 Q/K conversion");
  module.def("quantize_block_pair_out", &quantize_block_pair_out,
             "Single-read q128/group and K-page E4M3 conversion");
}
