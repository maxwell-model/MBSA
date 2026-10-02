# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Allocation-free GPU FP8 quantizers for the experimental Indexer path.

These kernels write into caller-owned buffers, so quantization can be included
inside an eager or CUDA-Graph complete-chain timing window without allocator or
host-readback artifacts.  Per-block domains exactly match the frozen q128
Indexer tile and physical 128-token K page.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


FP8_MAX = 448.0
_CHUNK = 1024
_VALUES_PER_BLOCK = 128 * 128
_CHUNKS_PER_BLOCK = _VALUES_PER_BLOCK // _CHUNK


@triton.jit
def _static_quant_kernel(x, y, n_elements: tl.constexpr, inv_scale: tl.constexpr,
                         BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    value = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    limit = tl.full((BLOCK,), 448.0, tl.float32)
    value = (value * inv_scale).to(tl.float32)
    value = tl.maximum(-limit, tl.minimum(limit, value))
    tl.store(y + offsets, value, mask=mask)


@triton.jit
def _static_quant_pair_kernel(
    q,
    q_out,
    k,
    k_out,
    q_elements: tl.constexpr,
    k_elements: tl.constexpr,
    q_programs: tl.constexpr,
    q_inv_scale: tl.constexpr,
    k_inv_scale: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Quantize current Q and the appended K tranche in one launch.

    The program-domain branch is CTA-uniform.  It changes neither codec nor
    ownership; it only removes the fixed cost of a second launch.
    """
    program = tl.program_id(0)
    is_q = program < q_programs
    local_program = program - tl.where(is_q, 0, q_programs)
    offsets = local_program * BLOCK + tl.arange(0, BLOCK)
    mask = tl.where(is_q, offsets < q_elements, offsets < k_elements)
    x_ptr = tl.where(is_q, q + offsets, k + offsets)
    y_ptr = tl.where(is_q, q_out + offsets, k_out + offsets)
    inv_scale = tl.where(is_q, q_inv_scale, k_inv_scale)
    value = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
    limit = tl.full((BLOCK,), 448.0, tl.float32)
    value = (value * inv_scale).to(tl.float32)
    value = tl.maximum(-limit, tl.minimum(limit, value))
    tl.store(y_ptr, value, mask=mask)


@triton.jit
def _k_page_quant_kernel(x, y, scales, n_pages: tl.constexpr):
    page = tl.program_id(0)
    lane = tl.arange(0, 1024)
    base = page * 16384
    amax = 0.0
    for chunk in range(16):
        value = tl.load(x + base + chunk * 1024 + lane).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(value)))
    scale = tl.where(amax > 0.0, amax / 448.0, amax + 1.0).to(tl.float32)
    tl.store(scales + page, scale)
    inv_scale = (1.0 / scale).to(tl.float32)
    limit = tl.full((1024,), 448.0, tl.float32)
    for chunk in range(16):
        offsets = base + chunk * 1024 + lane
        value = tl.load(x + offsets).to(tl.float32) * inv_scale
        value = tl.maximum(-limit, tl.minimum(limit, value))
        tl.store(y + offsets, value)


@triton.jit
def _q128_group_quant_kernel(x, y, scales, n_rows: tl.constexpr):
    owner = tl.program_id(0)
    q_block = owner // 4
    group = owner % 4
    lane = tl.arange(0, 1024)
    amax = 0.0
    for chunk in range(16):
        logical = chunk * 1024 + lane
        q_row = logical // 128
        channel = logical % 128
        global_q_row = q_block * 128 + q_row
        offsets = (global_q_row * 4 + group) * 128 + channel
        valid = global_q_row < n_rows
        value = tl.load(x + offsets, mask=valid, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(value)))
    scale = tl.where(amax > 0.0, amax / 448.0, amax + 1.0).to(tl.float32)
    tl.store(scales + owner, scale)
    inv_scale = (1.0 / scale).to(tl.float32)
    limit = tl.full((1024,), 448.0, tl.float32)
    for chunk in range(16):
        logical = chunk * 1024 + lane
        q_row = logical // 128
        channel = logical % 128
        global_q_row = q_block * 128 + q_row
        offsets = (global_q_row * 4 + group) * 128 + channel
        valid = global_q_row < n_rows
        value = tl.load(x + offsets, mask=valid, other=0.0).to(tl.float32) * inv_scale
        value = tl.maximum(-limit, tl.minimum(limit, value))
        tl.store(y + offsets, value, mask=valid)


@triton.jit
def _q128_group_k_page_quant_pair_kernel(
    q,
    q_out,
    q_scales,
    k,
    k_out,
    k_scales,
    q_rows: tl.constexpr,
    q_owners: tl.constexpr,
    k_pages: tl.constexpr,
):
    """Fuse q128/group and physical-page K quantization launch domains."""
    owner = tl.program_id(0)
    is_q = owner < q_owners
    q_owner = owner
    page = owner - q_owners
    q_block = q_owner // 4
    group = q_owner % 4
    lane = tl.arange(0, 1024)
    amax = 0.0
    for chunk in range(16):
        logical = chunk * 1024 + lane
        q_row = logical // 128
        channel = logical % 128
        global_q_row = q_block * 128 + q_row
        q_offset = (global_q_row * 4 + group) * 128 + channel
        k_offset = page * 16384 + logical
        offset = tl.where(is_q, q_offset, k_offset)
        valid = tl.where(is_q, global_q_row < q_rows, page < k_pages)
        x_ptr = tl.where(is_q, q + offset, k + offset)
        value = tl.load(x_ptr, mask=valid, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(value)))
    scale = tl.where(amax > 0.0, amax / 448.0, amax + 1.0).to(tl.float32)
    scale_ptr = tl.where(is_q, q_scales + q_owner, k_scales + page)
    tl.store(scale_ptr, scale)
    inv_scale = (1.0 / scale).to(tl.float32)
    limit = tl.full((1024,), 448.0, tl.float32)
    for chunk in range(16):
        logical = chunk * 1024 + lane
        q_row = logical // 128
        channel = logical % 128
        global_q_row = q_block * 128 + q_row
        q_offset = (global_q_row * 4 + group) * 128 + channel
        k_offset = page * 16384 + logical
        offset = tl.where(is_q, q_offset, k_offset)
        valid = tl.where(is_q, global_q_row < q_rows, page < k_pages)
        x_ptr = tl.where(is_q, q + offset, k + offset)
        y_ptr = tl.where(is_q, q_out + offset, k_out + offset)
        value = tl.load(x_ptr, mask=valid, other=0.0).to(tl.float32) * inv_scale
        value = tl.maximum(-limit, tl.minimum(limit, value))
        tl.store(y_ptr, value, mask=valid)


def _validate_pair(x: torch.Tensor, out: torch.Tensor, name: str) -> None:
    if not x.is_cuda or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError(f"{name} input must be contiguous CUDA BF16")
    if (
        out.device != x.device
        or out.dtype != torch.float8_e4m3fn
        or out.shape != x.shape
        or not out.is_contiguous()
    ):
        raise ValueError(f"{name} output must be matching contiguous CUDA E4M3")


def quantize_static_out(x: torch.Tensor, out: torch.Tensor, scale: float) -> None:
    """Quantize with a frozen positive scalar into caller-owned storage."""
    _validate_pair(x, out, "static quant")
    scale = float(scale)
    if not (scale > 0.0 and scale < float("inf")):
        raise ValueError("scale must be finite and positive")
    block = 4096
    _static_quant_kernel[(triton.cdiv(x.numel(), block),)](
        x, out, x.numel(), 1.0 / scale, BLOCK=block, num_warps=8
    )


def quantize_raw_unit_out(x: torch.Tensor, out: torch.Tensor) -> None:
    """Direct satfinite E4M3 conversion with no scale metadata.

    This intentionally shares the allocation-free vectorized implementation
    with static-scale quantization while fixing the compile-time inverse scale
    to one.  The separate entry point prevents benchmarks and production
    adapters from accidentally treating unit-scale data as calibrated data.
    """
    _validate_pair(x, out, "raw unit quant")
    block = 4096
    _static_quant_kernel[(triton.cdiv(x.numel(), block),)](
        x, out, x.numel(), 1.0, BLOCK=block, num_warps=8
    )


def quantize_static_pair_out(
    q: torch.Tensor,
    q_out: torch.Tensor,
    q_scale: float,
    k: torch.Tensor,
    k_out: torch.Tensor,
    k_scale: float,
) -> None:
    """Quantize current Q and appended K with frozen positive scales."""
    _quantize_static_pair_out_config(
        q,
        q_out,
        q_scale,
        k,
        k_out,
        k_scale,
        block=4096,
        num_warps=8,
    )


def _quantize_static_pair_out_config(
    q: torch.Tensor,
    q_out: torch.Tensor,
    q_scale: float,
    k: torch.Tensor,
    k_out: torch.Tensor,
    k_scale: float,
    *,
    block: int,
    num_warps: int,
) -> None:
    """Diagnostic entry point for fused-conversion CTA geometry."""
    _validate_pair(q, q_out, "static Q pair quant")
    _validate_pair(k, k_out, "static K pair quant")
    q_scale = float(q_scale)
    k_scale = float(k_scale)
    if not (q_scale > 0.0 and q_scale < float("inf")):
        raise ValueError("Q scale must be finite and positive")
    if not (k_scale > 0.0 and k_scale < float("inf")):
        raise ValueError("K scale must be finite and positive")
    q_programs = triton.cdiv(q.numel(), block)
    k_programs = triton.cdiv(k.numel(), block)
    _static_quant_pair_kernel[(q_programs + k_programs,)](
        q,
        q_out,
        k,
        k_out,
        q.numel(),
        k.numel(),
        q_programs,
        1.0 / q_scale,
        1.0 / k_scale,
        BLOCK=block,
        num_warps=num_warps,
    )


def quantize_raw_unit_pair_out(
    q: torch.Tensor,
    q_out: torch.Tensor,
    k: torch.Tensor,
    k_out: torch.Tensor,
) -> None:
    """Quantize current Q and appended K to unit-scale E4M3 in one launch."""
    quantize_static_pair_out(q, q_out, 1.0, k, k_out, 1.0)


def quantize_k_per_page_out(
    k: torch.Tensor,
    out: torch.Tensor,
    scales: torch.Tensor,
) -> None:
    """Quantize physical K pages; ``scales`` is FP32 ``[num_pages]``."""
    _validate_pair(k, out, "K page quant")
    if k.ndim != 4 or k.shape[1:] != (1, 128, 128):
        raise ValueError("k must be [pages,1,128,128]")
    if (
        scales.device != k.device
        or scales.dtype != torch.float32
        or scales.shape != (k.shape[0],)
        or not scales.is_contiguous()
    ):
        raise ValueError("K scales must be contiguous CUDA FP32 [pages]")
    _k_page_quant_kernel[(k.shape[0],)](
        k, out, scales, k.shape[0], num_warps=8
    )


def quantize_q_per_block_out(
    q: torch.Tensor,
    out: torch.Tensor,
    scales: torch.Tensor,
) -> None:
    """Quantize Q by q128/index-group into caller-owned storage."""
    _validate_pair(q, out, "Q block quant")
    if q.ndim != 3 or q.shape[1:] != (4, 128):
        raise ValueError("q must be [Sq,4,128]")
    n_q_blocks = triton.cdiv(q.shape[0], 128)
    if (
        scales.device != q.device
        or scales.dtype != torch.float32
        or scales.shape != (n_q_blocks, 4)
        or not scales.is_contiguous()
    ):
        raise ValueError("Q scales must be contiguous CUDA FP32 [Sq/128,4]")
    _q128_group_quant_kernel[(n_q_blocks * 4,)](
        q, out, scales, q.shape[0], num_warps=8
    )


def quantize_qk_per_block_pair_out(
    q: torch.Tensor,
    q_out: torch.Tensor,
    q_scales: torch.Tensor,
    k: torch.Tensor,
    k_out: torch.Tensor,
    k_scales: torch.Tensor,
) -> None:
    """Quantize q128/group Q and appended physical K pages in one launch."""
    _validate_pair(q, q_out, "Q block pair quant")
    _validate_pair(k, k_out, "K page pair quant")
    if q.ndim != 3 or q.shape[1:] != (4, 128):
        raise ValueError("q must be [Sq,4,128]")
    if k.ndim != 4 or k.shape[1:] != (1, 128, 128):
        raise ValueError("k must be [pages,1,128,128]")
    n_q_blocks = triton.cdiv(q.shape[0], 128)
    if (
        q_scales.device != q.device
        or q_scales.dtype != torch.float32
        or q_scales.shape != (n_q_blocks, 4)
        or not q_scales.is_contiguous()
    ):
        raise ValueError("Q scales must be contiguous CUDA FP32 [Sq/128,4]")
    if (
        k_scales.device != k.device
        or k_scales.dtype != torch.float32
        or k_scales.shape != (k.shape[0],)
        or not k_scales.is_contiguous()
    ):
        raise ValueError("K scales must be contiguous CUDA FP32 [pages]")
    q_owners = n_q_blocks * 4
    _q128_group_k_page_quant_pair_kernel[(q_owners + k.shape[0],)](
        q,
        q_out,
        q_scales,
        k,
        k_out,
        k_scales,
        q.shape[0],
        q_owners,
        k.shape[0],
        num_warps=8,
    )


__all__ = [
    "quantize_k_per_page_out",
    "quantize_qk_per_block_pair_out",
    "quantize_q_per_block_out",
    "quantize_raw_unit_pair_out",
    "quantize_raw_unit_out",
    "quantize_static_pair_out",
    "quantize_static_out",
]
