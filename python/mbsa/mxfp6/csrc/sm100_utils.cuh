// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-License-Identifier: MIT
// Modified for MBSA: reduced to the definitions the MXFP6 scorers use; see
// root NOTICE.

// DeepGEMM-derived SM100 UMMA, TMA and PTX helpers for MXFP6 scorers.
#pragma once

#include <type_traits>
#include <utility>

#include <cuda/std/cstdint>
#include <cuda_bf16.h>

#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/copy_sm100_tma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/int_tuple.hpp>
#include <cute/numeric/math.hpp>
#include <cute/util/type_traits.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/detail/helper_macros.hpp>
#include <cutlass/float_subbyte.h>
#include <cutlass/numeric_types.h>

// common/exception.cuh
#ifndef DG_TRAP_ONLY_DEVICE_ASSERT
#define DG_TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) \
        asm("trap;"); \
} while (0)
#endif

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

// common/cute_tie.cuh, two variables
#define CUTE_TIE_CONCAT_IMPL(A, B) A##B
#define CUTE_TIE_CONCAT(A, B) CUTE_TIE_CONCAT_IMPL(A, B)

#define CUTE_TIE_DECL(TUPLE_EXPR, V1, V2) \
    auto&& CUTE_TIE_CONCAT(cute_tie__temp_tuple_, __LINE__) = (TUPLE_EXPR); \
    auto V1 = ::cute::get<0>(CUTE_TIE_CONCAT(cute_tie__temp_tuple_, __LINE__)); \
    auto V2 = ::cute::get<1>(CUTE_TIE_CONCAT(cute_tie__temp_tuple_, __LINE__));

#define CUTE_TIE(TUPLE_EXPR, V1, V2) \
    do { \
        auto&& CUTE_TIE_CONCAT(cute_tie__temp_tuple_, __LINE__) = (TUPLE_EXPR); \
        V1 = ::cute::get<0>(CUTE_TIE_CONCAT(cute_tie__temp_tuple_, __LINE__)); \
        V2 = ::cute::get<1>(CUTE_TIE_CONCAT(cute_tie__temp_tuple_, __LINE__)); \
    } while (0)

// common/math.cuh
namespace deep_gemm::math {

template <typename T>
CUTLASS_HOST_DEVICE T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_align(T a, T b) {
    return constexpr_ceil_div(a, b) * b;
}

} // namespace deep_gemm::math

// common/utils.cuh
namespace deep_gemm::utils {

template <typename FuncT>
struct PatternVisitor {
    FuncT func;

    CUTLASS_HOST_DEVICE
    explicit PatternVisitor(FuncT&& func): func(std::forward<FuncT>(func)) {}

    CUTLASS_HOST_DEVICE
    auto operator [](const uint32_t& i) const {
        return func(i);
    }
};

template <uint32_t kNumCols>
CUTLASS_DEVICE constexpr uint32_t get_num_aligned_tmem_cols() {
    DG_STATIC_ASSERT(kNumCols <= 512, "Too many tensor memory columns");
    if constexpr (kNumCols <=  32) return  32;
    if constexpr (kNumCols <=  64) return  64;
    if constexpr (kNumCols <= 128) return 128;
    if constexpr (kNumCols <= 256) return 256;
    return 512;
}

} // namespace deep_gemm::utils

namespace deep_gemm::ptx {

// ptx/utils.cuh
CUTLASS_DEVICE uint32_t get_lane_idx() {
    uint32_t lane_id;
    asm ("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id;
}

// ptx/ld_st.cuh
CUTLASS_DEVICE uint32_t ld_shared(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(ret) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

CUTLASS_DEVICE void st_shared(const void* ptr, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(__cvta_generic_to_shared(ptr)), "r"(x), "r"(y), "r"(z), "r"(w));
}

// ptx/tcgen05.cuh
struct SM100_MMA_MXF8F6F4_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc,
        uint32_t const& tmem_sfa,
        uint32_t const& tmem_sfb) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
          "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p; \n\t"
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c),
            "r"(tmem_sfa), "r"(tmem_sfb));
    }
};

CUTLASS_DEVICE void tcgen05_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}

CUTLASS_DEVICE void tcgen05_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

} // namespace deep_gemm::ptx

// common/tma_copy.cuh, unpacked element types
namespace deep_gemm::tma {

template <uint32_t BLOCK_INNER, uint32_t kSwizzleMode, typename dtype_t>
constexpr uint32_t get_inner_block_atom_size() {
    return kSwizzleMode == 0 ? BLOCK_INNER : kSwizzleMode / sizeof(dtype_t);
}

template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER,
          uint32_t kSwizzleMode, typename dtype_t,
          uint64_t kCacheHint = static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL)>
CUTLASS_DEVICE void
copy(void const* desc_ptr, cutlass::arch::ClusterTransactionBarrier* barrier_ptr,
     dtype_t* smem_ptr, const uint32_t& inner_idx, const uint32_t& outer_idx) {
    constexpr uint32_t BLOCK_INNER_ATOM = get_inner_block_atom_size<BLOCK_INNER, kSwizzleMode, dtype_t>();
    DG_STATIC_ASSERT(BLOCK_INNER % BLOCK_INNER_ATOM == 0, "TMA inner block must contain whole atoms");

    #pragma unroll
    for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
        cute::SM90_TMA_LOAD_2D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                     kCacheHint,
                                     smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                     inner_idx + i * BLOCK_INNER_ATOM, outer_idx);
    }
}

} // namespace deep_gemm::tma

// mma/sm100.cuh, K-major byte operands
namespace deep_gemm::mma::sm100 {

CUTLASS_DEVICE
cute::UMMA::SmemDescriptor make_smem_desc(cute::UMMA::LayoutType layout, void* smem_ptr,
                                          const uint32_t& stride_byte_offset, const uint32_t& leading_byte_offset) {
    cute::UMMA::SmemDescriptor desc;

    // Set the version for SM100
    desc.version_ = 1;

    // Legacy mode
    desc.lbo_mode_ = 0;

    // Layout
    desc.layout_type_ = static_cast<uint8_t>(layout);

    // Start address
    const auto uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
    desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);

    // Base offset
    desc.base_offset_ = 0;

    // SBO and LBO
    desc.stride_byte_offset_ = stride_byte_offset >> 4;
    desc.leading_byte_offset_ = leading_byte_offset >> 4;

    return desc;
}

CUTLASS_DEVICE
cute::UMMA::SmemDescriptor make_sf_desc(void* smem_ptr) {
    // UTCCP uses K-major 8x128-bit atoms; one K atom permits zero LBO.
    return make_smem_desc(cute::UMMA::LayoutType::SWIZZLE_NONE, smem_ptr, 8 * 16, 0);
}

CUTLASS_DEVICE
void replace_smem_desc_addr(cute::UMMA::SmemDescriptor& desc, const void* smem_ptr) {
    const auto uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
    desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);
}

CUTLASS_DEVICE
static uint32_t get_atom_base(const cute::UMMA::LayoutType& layout_type) {
    return layout_type == cute::UMMA::LayoutType::SWIZZLE_128B_BASE32B ? 32 : 16;
}

template <uint32_t kSwizzleMode>
constexpr static cute::UMMA::LayoutType to_umma_layout_type() {
    DG_STATIC_ASSERT(kSwizzleMode == 0 or kSwizzleMode == 16 or
                     kSwizzleMode == 32 or kSwizzleMode == 64 or
                     kSwizzleMode == 128, "Invalid swizzling mode");
    if constexpr (kSwizzleMode == 0)   return cute::UMMA::LayoutType::SWIZZLE_NONE;
    if constexpr (kSwizzleMode == 16)  return cute::UMMA::LayoutType::SWIZZLE_NONE;
    if constexpr (kSwizzleMode == 32)  return cute::UMMA::LayoutType::SWIZZLE_32B;
    if constexpr (kSwizzleMode == 64)  return cute::UMMA::LayoutType::SWIZZLE_64B;
    if constexpr (kSwizzleMode == 128) return cute::UMMA::LayoutType::SWIZZLE_128B;
}

template <cute::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode,
          typename dtype_t>
CUTLASS_DEVICE
cute::UMMA::SmemDescriptor make_umma_desc(dtype_t* base_smem_ptr, uint32_t mn_idx, uint32_t k_idx) {
    DG_STATIC_ASSERT(kMajorMode == cute::UMMA::Major::K and sizeof(dtype_t) == 1,
                     "Only K-major byte operands are kept");
    const auto layout_type = to_umma_layout_type<kSwizzleMode>();
    const auto num_non_contiguous = 128 / get_atom_base(layout_type);
    // NOTES: for K-major layout, the swizzle must be the same as `BLOCK_K` elements in bytes
    // also, atom index must be 0, so that each block has exactly one swizzle atom on the K axis
    DG_STATIC_ASSERT(kSwizzleMode == BLOCK_K * sizeof(dtype_t), "Unexpected value");

    // Atom size: 8 x `kSwizzleMode` (in bytes, on K)
    // {SBO, LBO} means the byte stride between atoms on {MN, K}
    // NOTES: on K, there is only 1 atom as asserted previously, so LBO can be 0
    const uint32_t stride_byte_offset = num_non_contiguous * BLOCK_K * sizeof(dtype_t);
    const uint32_t leading_byte_offset = 0;
    const auto byte_ptr = reinterpret_cast<uint8_t*>(base_smem_ptr) +
                          (mn_idx * BLOCK_K + k_idx) * sizeof(dtype_t);
    return make_smem_desc(layout_type, byte_ptr, stride_byte_offset, leading_byte_offset);
}

CUTLASS_DEVICE uint64_t make_runtime_instr_desc_with_sf_id(
    cute::UMMA::InstrDescriptorBlockScaled desc, const uint32_t& sfa_id, const uint32_t& sfb_id) {
    desc.a_sf_id_ = sfa_id, desc.b_sf_id_ = sfb_id;
    return static_cast<uint64_t>(static_cast<uint32_t>(desc)) << 32;
}

} // namespace deep_gemm::mma::sm100
