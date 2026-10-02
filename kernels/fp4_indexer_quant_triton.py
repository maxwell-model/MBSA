# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Allocation-free SM103 FP4 Indexer operand preparation.

The kernels in this module write both packed E2M1 payloads and block scales
directly in the storage order consumed by the vendored SM100/SM103 Tensor-Core
Indexer.  They are experimental and are not wired into the production BF16
dispatch.

NVFP4 uses an immutable per-tensor calibration range plus one E4M3 scale per
16 adjacent channels.  This makes an append-only K cache possible: historical
pages never need requantization.  MXFP4 uses one UE8M0 scale per 32 adjacent
channels and has no tensor-global scale.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import triton
import triton.language as tl


_HEAD_DIM = 128
_PACKED_HEAD_DIM = 64


@triton.jit
def _e2m1_code(value):
    """Nearest E2M1 code; exact midpoints choose the lower magnitude."""

    magnitude = tl.minimum(tl.abs(value), 6.0)
    # The small upward guard preserves the frozen lower-magnitude midpoint
    # policy even when compiler reassociation turns an exact rational quotient
    # (for example 3.5) into the next FP32 value.
    index = (
        (magnitude > 0.250001).to(tl.int32)
        + (magnitude > 0.750001).to(tl.int32)
        + (magnitude > 1.250001).to(tl.int32)
        + (magnitude > 1.750001).to(tl.int32)
        + (magnitude > 2.500001).to(tl.int32)
        + (magnitude > 3.500001).to(tl.int32)
        + (magnitude > 5.000001).to(tl.int32)
    )
    negative = (value < 0.0).to(tl.int32) * (magnitude != 0.0).to(tl.int32)
    return index | (negative << 3)


@triton.jit
def _scale_storage_offset(m, group, l, REST_M: tl.constexpr, REST_G: tl.constexpr):
    """Offset for contiguous [L,restM,restG,32,4,4] MMA scale storage."""

    return (
        l * REST_M * REST_G * 512
        + (m // 128) * REST_G * 512
        + (group // 4) * 512
        + (m % 32) * 16
        + ((m % 128) // 32) * 4
        + (group % 4)
    )


@triton.jit
def _nvfp4_quantize_preordered_kernel(
    x,
    payload,
    scale_storage,
    global_scale,
    MN: tl.constexpr,
    L: tl.constexpr,
    INPUT_M_MAJOR: tl.constexpr,
    REST_M: tl.constexpr,
    REST_G: tl.constexpr,
):
    pid = tl.program_id(0)
    if INPUT_M_MAJOR:
        m = pid // L
        l = pid % L
    else:
        l = pid // MN
        m = pid % MN

    channel = tl.arange(0, 128)
    value = tl.load(x + pid * 128 + channel).to(tl.float32)
    block = tl.reshape(value, (8, 16))
    amax = tl.max(tl.abs(block), axis=1)
    raw_scale = tl.minimum(amax * global_scale / 6.0, 448.0)
    scale_f8 = raw_scale.to(tl.float8e4nv)
    scale = scale_f8.to(tl.float32)
    normalized = tl.where(
        scale[:, None] > 0.0,
        block * global_scale / scale[:, None],
        0.0,
    )
    codes = tl.reshape(_e2m1_code(normalized), (128,))
    low, high = tl.split(tl.reshape(codes, (64, 2)))
    packed = low | (high << 4)
    tl.store(payload + pid * 64 + tl.arange(0, 64), packed)

    group = tl.arange(0, 8)
    scale_offset = _scale_storage_offset(m, group, l, REST_M, REST_G)
    tl.store(scale_storage + scale_offset, scale_f8)


@triton.jit
def _mxfp4_quantize_preordered_kernel(
    x,
    payload,
    scale_storage,
    MN: tl.constexpr,
    L: tl.constexpr,
    INPUT_M_MAJOR: tl.constexpr,
    REST_M: tl.constexpr,
    REST_G: tl.constexpr,
):
    pid = tl.program_id(0)
    if INPUT_M_MAJOR:
        m = pid // L
        l = pid % L
    else:
        l = pid // MN
        m = pid % MN

    channel = tl.arange(0, 128)
    value = tl.load(x + pid * 128 + channel).to(tl.float32)
    block = tl.reshape(value, (4, 32))
    amax = tl.max(tl.abs(block), axis=1)
    raw_scale = amax / 6.0
    # UE8M0 is a positive power of two rounded toward +infinity.  Byte zero
    # remains reserved for an all-zero block; byte 255 is NaN and is excluded.
    exponent = tl.ceil(tl.log2(tl.maximum(raw_scale, 2.0 ** -126)))
    exponent = tl.maximum(-126.0, tl.minimum(127.0, exponent))
    decoded_scale = tl.exp2(exponent)
    encoded_scale = tl.where(amax > 0.0, exponent.to(tl.int32) + 127, 0)
    normalized = tl.where(
        amax[:, None] > 0.0,
        block / decoded_scale[:, None],
        0.0,
    )
    codes = tl.reshape(_e2m1_code(normalized), (128,))
    low, high = tl.split(tl.reshape(codes, (64, 2)))
    packed = low | (high << 4)
    tl.store(payload + pid * 64 + tl.arange(0, 64), packed)

    group = tl.arange(0, 4)
    scale_offset = _scale_storage_offset(m, group, l, REST_M, REST_G)
    tl.store(scale_storage + scale_offset, encoded_scale.to(tl.uint8))


def _storage_shape(mn: int, l: int, groups: int) -> tuple[int, int, int, int, int, int]:
    return (int(l), triton.cdiv(int(mn), 128), triton.cdiv(groups, 4), 32, 4, 4)


@dataclass(frozen=True)
class NativeFp4QuantizedBuffers:
    """Caller-owned payload and preordered scale storage."""

    q: torch.Tensor
    k: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    fp4_format: str
    q_inverse_global_scale: float
    k_inverse_global_scale: float


def _validate_format_and_globals(buffers: NativeFp4QuantizedBuffers) -> tuple[int, torch.dtype]:
    if buffers.fp4_format == "nvfp4":
        groups, scale_dtype = 8, torch.float8_e4m3fn
    elif buffers.fp4_format == "mxfp4":
        groups, scale_dtype = 4, torch.float8_e8m0fnu
    else:
        raise ValueError("buffers.fp4_format must be 'nvfp4' or 'mxfp4'")
    for name, value in (
        ("q_inverse_global_scale", buffers.q_inverse_global_scale),
        ("k_inverse_global_scale", buffers.k_inverse_global_scale),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"buffers.{name} must be finite and positive")
    return groups, scale_dtype


def _validate_output_tensor(
    value: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if (
        tuple(value.shape) != shape
        or value.dtype != dtype
        or value.device != device
        or not value.is_contiguous()
    ):
        raise ValueError(
            f"buffers.{name} must be contiguous {dtype} on {device} with shape {shape}"
        )


def _validate_q_call(
    index_q: torch.Tensor, buffers: NativeFp4QuantizedBuffers
) -> None:
    groups, scale_dtype = _validate_format_and_globals(buffers)
    if (
        not index_q.is_cuda
        or index_q.dtype != torch.bfloat16
        or not index_q.is_contiguous()
        or index_q.ndim != 3
        or tuple(index_q.shape[1:]) != (4, _HEAD_DIM)
    ):
        raise ValueError("index_q must be contiguous CUDA BF16 [Sq,4,128]")
    sq, heads, _ = index_q.shape
    _validate_output_tensor(
        buffers.q,
        name="q",
        shape=(sq, heads, _PACKED_HEAD_DIM),
        dtype=torch.uint8,
        device=index_q.device,
    )
    _validate_output_tensor(
        buffers.q_scale,
        name="q_scale",
        shape=_storage_shape(sq, heads, groups),
        dtype=scale_dtype,
        device=index_q.device,
    )


def _validate_k_call(
    index_k: torch.Tensor, buffers: NativeFp4QuantizedBuffers
) -> None:
    groups, scale_dtype = _validate_format_and_globals(buffers)
    if (
        not index_k.is_cuda
        or index_k.dtype != torch.bfloat16
        or not index_k.is_contiguous()
        or index_k.ndim != 4
        or tuple(index_k.shape[1:]) != (1, 128, _HEAD_DIM)
    ):
        raise ValueError(
            "index_k must be contiguous CUDA BF16 [pages,1,128,128]"
        )
    pages, heads, tokens, _ = index_k.shape
    _validate_output_tensor(
        buffers.k,
        name="k",
        shape=(pages, heads, tokens, _PACKED_HEAD_DIM),
        dtype=torch.uint8,
        device=index_k.device,
    )
    _validate_output_tensor(
        buffers.k_scale,
        name="k_scale",
        shape=_storage_shape(tokens, pages * heads, groups),
        dtype=scale_dtype,
        device=index_k.device,
    )


def allocate_native_fp4_buffers(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    *,
    fp4_format: str,
    calibration_amax_q: float = 6.0,
    calibration_amax_k: float = 6.0,
) -> NativeFp4QuantizedBuffers:
    """Allocate output buffers outside a formal timing/capture window."""

    if (
        not index_q.is_cuda
        or index_q.dtype != torch.bfloat16
        or not index_q.is_contiguous()
        or tuple(index_q.shape[1:]) != (4, _HEAD_DIM)
    ):
        raise ValueError("index_q must be contiguous CUDA BF16 [Sq,4,128]")
    if (
        index_k.device != index_q.device
        or index_k.dtype != torch.bfloat16
        or not index_k.is_contiguous()
        or tuple(index_k.shape[1:]) != (1, 128, _HEAD_DIM)
    ):
        raise ValueError("index_k must be matching contiguous CUDA BF16 [pages,1,128,128]")
    if fp4_format not in ("nvfp4", "mxfp4"):
        raise ValueError("fp4_format must be 'nvfp4' or 'mxfp4'")
    for name, value in (("calibration_amax_q", calibration_amax_q), ("calibration_amax_k", calibration_amax_k)):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    groups = 8 if fp4_format == "nvfp4" else 4
    scale_dtype = (
        torch.float8_e4m3fn
        if fp4_format == "nvfp4"
        else torch.float8_e8m0fnu
    )
    q = torch.empty((*index_q.shape[:-1], _PACKED_HEAD_DIM), dtype=torch.uint8, device=index_q.device)
    k = torch.empty((*index_k.shape[:-1], _PACKED_HEAD_DIM), dtype=torch.uint8, device=index_k.device)
    q_scale = torch.empty(_storage_shape(index_q.shape[0], index_q.shape[1], groups), dtype=scale_dtype, device=index_q.device)
    k_scale = torch.empty(_storage_shape(index_k.shape[2], index_k.shape[0] * index_k.shape[1], groups), dtype=scale_dtype, device=index_q.device)
    q_inv = float(calibration_amax_q) / (448.0 * 6.0) if fp4_format == "nvfp4" else 1.0
    k_inv = float(calibration_amax_k) / (448.0 * 6.0) if fp4_format == "nvfp4" else 1.0
    return NativeFp4QuantizedBuffers(q, k, q_scale, k_scale, fp4_format, q_inv, k_inv)


def quantize_q_native_out(index_q: torch.Tensor, buffers: NativeFp4QuantizedBuffers) -> None:
    """Quantize the current Q chunk directly into native scorer storage."""

    _validate_q_call(index_q, buffers)
    sq, heads, _ = index_q.shape
    rest_m = triton.cdiv(sq, 128)
    if buffers.fp4_format == "nvfp4":
        global_scale = 1.0 / buffers.q_inverse_global_scale
        _nvfp4_quantize_preordered_kernel[(sq * heads,)](
            index_q,
            buffers.q,
            buffers.q_scale,
            global_scale,
            MN=sq,
            L=heads,
            INPUT_M_MAJOR=True,
            REST_M=rest_m,
            REST_G=2,
            num_warps=4,
        )
    else:
        _mxfp4_quantize_preordered_kernel[(sq * heads,)](
            index_q,
            buffers.q,
            buffers.q_scale.view(torch.uint8),
            MN=sq,
            L=heads,
            INPUT_M_MAJOR=True,
            REST_M=rest_m,
            REST_G=1,
            num_warps=4,
        )


def quantize_k_pages_native_out(
    index_k: torch.Tensor,
    buffers: NativeFp4QuantizedBuffers,
    *,
    page_start: int = 0,
    page_count: int | None = None,
) -> None:
    """Quantize only newly appended physical K pages.

    K payload and scale storage are both page-major in the native interface,
    so a page slice is a closed append domain.  Existing cache pages are not
    read or rewritten.
    """

    _validate_k_call(index_k, buffers)
    pages, heads, tokens, _ = index_k.shape
    page_start = int(page_start)
    if page_count is None:
        page_count = pages - page_start
    page_count = int(page_count)
    if page_start < 0 or page_count < 0 or page_start + page_count > pages:
        raise ValueError("K page append range is outside the cache")
    if page_count == 0:
        return
    x = index_k[page_start : page_start + page_count]
    payload = buffers.k[page_start : page_start + page_count]
    scale = buffers.k_scale[page_start * heads : (page_start + page_count) * heads]
    l = page_count * heads
    if buffers.fp4_format == "nvfp4":
        global_scale = 1.0 / buffers.k_inverse_global_scale
        _nvfp4_quantize_preordered_kernel[(l * tokens,)](
            x,
            payload,
            scale,
            global_scale,
            MN=tokens,
            L=l,
            INPUT_M_MAJOR=False,
            REST_M=1,
            REST_G=2,
            num_warps=4,
        )
    else:
        _mxfp4_quantize_preordered_kernel[(l * tokens,)](
            x,
            payload,
            scale.view(torch.uint8),
            MN=tokens,
            L=l,
            INPUT_M_MAJOR=False,
            REST_M=1,
            REST_G=1,
            num_warps=4,
        )


__all__ = [
    "NativeFp4QuantizedBuffers",
    "allocate_native_fp4_buffers",
    "quantize_k_pages_native_out",
    "quantize_q_native_out",
]
