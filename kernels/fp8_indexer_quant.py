# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Explicit FP8 E4M3 quantization contracts for the block-score Indexer.

The helpers in this module are reference/prototype implementations.  They do
not hide scale generation or cache mutation, which lets the benchmark report
pre-quantized ceilings separately from production chains that include Q/K
quantization.  Production callers should pass calibrated scales for the
per-tensor contract; recomputing a global K scale as an append-only cache grows
would invalidate all previously stored pages.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)


def _validate_source(x: torch.Tensor, name: str) -> None:
    if not x.is_cuda or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous CUDA BF16")


def _validate_scalar_scale(scale: float, name: str) -> float:
    value = float(scale)
    if not (value > 0.0 and value < float("inf")):
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass(frozen=True)
class PerTensorFp8:
    q: torch.Tensor
    k: torch.Tensor
    q_scale: float
    k_scale: float


@dataclass(frozen=True)
class RawUnitFp8:
    """Unscaled E4M3 payloads matching the MiniMax-M3 index-cache ABI.

    The source values are already-prepared Shopee Index Q/K activations.  The
    delivery repository does not contain the upstream projection,
    RMS-normalization or RoPE producer, so this object deliberately describes
    only the production-isomorphic boundary available here: current Q rows and
    newly appended K pages are rounded directly to satfinite E4M3 and no scale
    metadata is produced.
    """

    q: torch.Tensor
    k: torch.Tensor


@dataclass(frozen=True)
class PerBlockFp8:
    q: torch.Tensor
    k: torch.Tensor
    # Q scale is one positive FP32 value per [128 queries, index/KV group].
    q_scale: torch.Tensor
    # K scale is one positive FP32 value per physical 128-token page.
    k_scale: torch.Tensor


def calibrated_scalar_scale(x: torch.Tensor, *, percentile: float = 1.0) -> float:
    """Return a diagnostic calibration scale; never call this in a timed path."""
    _validate_source(x, "x")
    if percentile != 1.0:
        if not 0.0 < percentile < 1.0:
            raise ValueError("percentile must be in (0,1]")
        amax = torch.quantile(x.float().abs().flatten(), percentile)
    else:
        amax = x.float().abs().amax()
    value = max(float(amax.item()) / FP8_MAX, torch.finfo(torch.float32).tiny)
    return value


def quantize_per_tensor(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    q_scale: float,
    k_scale: float,
) -> PerTensorFp8:
    """Quantize with immutable model/layer calibration scales.

    ``k`` is the append-only paged cache ``[pages,1,128,128]``.  The caller is
    responsible for using the same ``k_scale`` for every page for the lifetime
    of that cache.
    """
    _validate_source(q, "q")
    _validate_source(k, "k")
    qs = _validate_scalar_scale(q_scale, "q_scale")
    ks = _validate_scalar_scale(k_scale, "k_scale")
    q_fp8 = (q.float() / qs).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    k_fp8 = (k.float() / ks).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return PerTensorFp8(q_fp8, k_fp8, qs, ks)


def quantize_raw_unit(q: torch.Tensor, k: torch.Tensor) -> RawUnitFp8:
    """Round prepared BF16 Index Q/K directly to satfinite E4M3.

    PyTorch's ``float8_e4m3fn`` conversion implements finite E4M3 conversion;
    the explicit clamp records the intended satfinite contract and makes the
    independent oracle unambiguous.  There is no scale tensor or scalar in the
    returned ABI.  Score consumers use compile/launch constants equal to one.
    """
    _validate_source(q, "q")
    _validate_source(k, "k")
    q_fp8 = q.float().clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    k_fp8 = k.float().clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return RawUnitFp8(q_fp8, k_fp8)


def quantize_q_per_block(
    q: torch.Tensor,
    *,
    q_block_rows: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize Q by ``[q128, index-group]`` without a Python tile loop."""
    _validate_source(q, "q")
    if q.ndim != 3 or q.shape[1:] != (4, 128):
        raise ValueError("q must be [Sq,4,128]")
    if q_block_rows != 128:
        raise ValueError("the production per-block contract uses q128 tiles")
    rows = q.shape[0]
    padded_rows = ((rows + q_block_rows - 1) // q_block_rows) * q_block_rows
    q_float_flat = q.float()
    if padded_rows != rows:
        padded = torch.zeros(
            (padded_rows, 4, 128), dtype=torch.float32, device=q.device
        )
        padded[:rows].copy_(q_float_flat)
        q_float_flat = padded
    q_float = q_float_flat.view(-1, q_block_rows, 4, 128)
    tiny = torch.finfo(torch.float32).tiny
    q_scale = (q_float.abs().amax(dim=(1, 3)) / FP8_MAX).clamp_min(tiny)
    q_fp8 = (
        q_float / q_scale.view(-1, 1, 4, 1)
    ).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).view(
        padded_rows, 4, 128
    )[:rows].contiguous()
    return q_fp8, q_scale.contiguous()


def quantize_k_per_page(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each physical 128-token K page with one FP32 scale."""
    _validate_source(k, "k")
    if k.ndim != 4 or k.shape[1:] != (1, 128, 128):
        raise ValueError("k must be [pages,1,128,128]")
    k_float = k.float()
    tiny = torch.finfo(torch.float32).tiny
    k_scale = (
        k_float.abs().amax(dim=(1, 2, 3)) / FP8_MAX
    ).clamp_min(tiny).contiguous()
    k_fp8 = (
        k_float / k_scale.view(-1, 1, 1, 1)
    ).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).contiguous()
    return k_fp8, k_scale


def quantize_per_block(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    q_block_rows: int = 128,
) -> PerBlockFp8:
    """Quantize Q by q128/group and K by physical 128-token page.

    Q has shape ``[Sq,4,128]`` and K ``[pages,1,128,128]``.  Q scales vary by
    q128 tile and index/KV group but remain common across every candidate block
    in that query row.  K scales vary by page and therefore *must* be applied
    to each block score before block-max ranking and Exact Top-16 selection.
    """
    _validate_source(q, "q")
    _validate_source(k, "k")
    if q.ndim != 3 or q.shape[1:] != (4, 128):
        raise ValueError("q must be [Sq,4,128]")
    if k.ndim != 4 or k.shape[1:] != (1, 128, 128):
        raise ValueError("k must be [pages,1,128,128]")
    if q_block_rows != 128:
        raise ValueError("the production per-block contract uses q128 tiles")

    q_fp8, q_scale = quantize_q_per_block(q, q_block_rows=q_block_rows)
    k_fp8, k_scale = quantize_k_per_page(k)
    return PerBlockFp8(q_fp8, k_fp8, q_scale, k_scale)


def dequantize_per_block(value: PerBlockFp8) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference dequantization used only by accuracy/oracle tests."""
    rows = torch.arange(value.q.shape[0], device=value.q.device) // 128
    q = value.q.float() * value.q_scale[rows].unsqueeze(-1)
    k = value.k.float() * value.k_scale.view(-1, 1, 1, 1)
    return q, k


__all__ = [
    "FP8_DTYPE",
    "FP8_MAX",
    "PerBlockFp8",
    "PerTensorFp8",
    "RawUnitFp8",
    "calibrated_scalar_scale",
    "dequantize_per_block",
    "quantize_per_block",
    "quantize_q_per_block",
    "quantize_k_per_page",
    "quantize_per_tensor",
    "quantize_raw_unit",
]
