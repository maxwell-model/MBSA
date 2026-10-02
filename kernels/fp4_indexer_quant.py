# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Reference FP4 contracts for the experimental block-score Indexer.

This module deliberately separates the *numerical* contract from the eventual
SM103 tensor-core storage layout.  ``payload`` is packed E2M1 (two nibbles per
byte), while ``scale`` is kept in logical row-major order so an independent
oracle can be read and tested without depending on a CuTe layout helper.

NVFP4 follows the FlashInfer proxy-score contract frozen in the campaign:
one E4M3 scale per 16 adjacent channels and one inverse global scale per
tensor.  MXFP4 is an additional Pareto point with one UE8M0 (power-of-two)
scale per 32 channels.  Neither path changes the production BF16 dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


E2M1_MAX = 6.0
E4M3_MAX = 448.0
_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@dataclass(frozen=True)
class Fp4Tensor:
    """Packed block-scaled FP4 tensor plus its exact decode metadata."""

    payload: torch.Tensor
    scale: torch.Tensor
    inverse_global_scale: float
    original_shape: tuple[int, ...]
    scale_vec_size: int
    recipe: str


def _validate_source(x: torch.Tensor, name: str) -> None:
    if not x.is_cuda or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous CUDA BF16")
    if x.ndim < 2 or x.shape[-1] != 128:
        raise ValueError(f"{name} must have a contiguous head dimension of 128")


def _e2m1_quantize(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unpacked E2M1 codes and their FP32 decoded values.

    ``torch.argmin`` makes midpoint ties deterministic by choosing the lower
    magnitude.  Production/random inputs almost never lie exactly on a
    midpoint; explicit midpoint tests pin this oracle policy so a hardware
    implementation must either match it or document a tie-aware equivalence.
    """
    levels = torch.tensor(_E2M1_LEVELS, dtype=torch.float32, device=values.device)
    magnitude = values.float().abs().clamp_max(E2M1_MAX)
    index = (magnitude.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    negative = torch.signbit(values) & (magnitude != 0)
    codes = index.to(torch.uint8) | (negative.to(torch.uint8) << 3)
    decoded = levels[index] * torch.where(negative, -1.0, 1.0)
    return codes, decoded


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    if codes.shape[-1] % 2:
        raise ValueError("E2M1 packing requires an even final dimension")
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def _unpack_codes(payload: torch.Tensor) -> torch.Tensor:
    lo = payload & 0x0F
    hi = payload >> 4
    return torch.stack((lo, hi), dim=-1).flatten(-2)


def _decode_codes(codes: torch.Tensor) -> torch.Tensor:
    levels = torch.tensor(_E2M1_LEVELS, dtype=torch.float32, device=codes.device)
    magnitude = levels[(codes & 0x7).to(torch.long)]
    return torch.where((codes & 0x8) != 0, -magnitude, magnitude)


def _e4m3_round_scale(scale: torch.Tensor, *, maximum: float) -> torch.Tensor:
    # Conversion to float8_e4m3fn is the independent satfinite scale oracle.
    # Keep the physical byte dtype: SM103 consumes this scale directly.
    return scale.float().clamp(0.0, maximum).to(torch.float8_e4m3fn)


def _nvfp4_candidate(
    blocks: torch.Tensor,
    scaled_amax: torch.Tensor,
    global_scale: float,
    divisor: float,
    e4m3_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale_hp = scaled_amax / divisor
    scale_fp8 = _e4m3_round_scale(scale_hp, maximum=e4m3_max)
    scale = scale_fp8.float()
    normalized = torch.where(
        scale.unsqueeze(-1) > 0,
        blocks.float() * float(global_scale) / scale.unsqueeze(-1),
        torch.zeros_like(blocks, dtype=torch.float32),
    )
    codes, decoded_payload = _e2m1_quantize(normalized)
    reconstructed = decoded_payload * scale.unsqueeze(-1) / float(global_scale)
    return codes, scale_fp8, reconstructed


def quantize_nvfp4(
    x: torch.Tensor,
    *,
    calibration_amax: float | None = None,
    recipe: str = "standard",
    error_metric: str = "mae",
    e4m3_max: float = E4M3_MAX,
) -> Fp4Tensor:
    """Quantize BF16 rows to NVFP4 with per-16 E4M3 scales.

    ``calibration_amax`` makes the per-tensor global scale immutable, which is
    required for an append-only K cache.  Omitting it computes an oracle
    ceiling from the complete tensor and is therefore not production-isomorphic
    for a growing cache.  ``recipe='4over6'`` tests both amax/6 and amax/4
    local scales and retains the lower-error reconstruction per 16 values.
    """
    _validate_source(x, "x")
    if recipe not in ("standard", "4over6"):
        raise ValueError("recipe must be 'standard' or '4over6'")
    if error_metric not in ("mae", "mse"):
        raise ValueError("error_metric must be 'mae' or 'mse'")
    if e4m3_max not in (256.0, E4M3_MAX):
        raise ValueError("e4m3_max must be 256 or 448")

    if calibration_amax is None:
        amax = float(x.float().abs().amax().item())
    else:
        amax = float(calibration_amax)
    if not math.isfinite(amax) or amax < 0:
        raise ValueError("calibration_amax must be finite and non-negative")
    global_scale = (e4m3_max * E2M1_MAX / amax) if amax > 0 else 1.0

    rows = x.float().reshape(-1, 128)
    blocks = rows.view(rows.shape[0], 128 // 16, 16)
    scaled_amax = blocks.abs().amax(dim=-1) * global_scale
    codes6, scale6, recon6 = _nvfp4_candidate(
        blocks, scaled_amax, global_scale, 6.0, e4m3_max
    )
    if recipe == "standard":
        codes, scale = codes6, scale6
    else:
        codes4, scale4, recon4 = _nvfp4_candidate(
            blocks, scaled_amax, global_scale, 4.0, e4m3_max
        )
        if error_metric == "mae":
            err4 = (recon4 - blocks).abs().sum(dim=-1)
            err6 = (recon6 - blocks).abs().sum(dim=-1)
        else:
            err4 = (recon4 - blocks).square().sum(dim=-1)
            err6 = (recon6 - blocks).square().sum(dim=-1)
        choose4 = err4 < err6
        codes = torch.where(choose4.unsqueeze(-1), codes4, codes6)
        scale = torch.where(choose4, scale4, scale6)

    return Fp4Tensor(
        payload=_pack_codes(codes.view(rows.shape[0], 128)).view(
            *x.shape[:-1], 64
        ),
        scale=scale.view(*x.shape[:-1], 8).contiguous(),
        inverse_global_scale=1.0 / global_scale,
        original_shape=tuple(x.shape),
        scale_vec_size=16,
        recipe=f"nvfp4_{recipe}_{error_metric}_e4m3_{int(e4m3_max)}",
    )


def quantize_mxfp4(x: torch.Tensor) -> Fp4Tensor:
    """Quantize BF16 rows to MXFP4 with one UE8M0 scale per 32 values."""
    _validate_source(x, "x")
    rows = x.float().reshape(-1, 128)
    blocks = rows.view(rows.shape[0], 128 // 32, 32)
    amax = blocks.abs().amax(dim=-1)
    raw_scale = (amax / E2M1_MAX).contiguous()
    # Match CUDA's positive-infinity UE8M0 conversion exactly: preserve an
    # exact power of two, otherwise increment the FP32 exponent, and reserve
    # byte zero for a zero block.  Byte 255 is NaN in UE8M0, so saturate at
    # 254 as the device intrinsic does.
    bits = raw_scale.view(torch.int32)
    rounded_bits = torch.where(
        (bits & 0x007FFFFF) != 0,
        (bits + 0x00800000) & 0x7F800000,
        bits & 0x7F800000,
    )
    ue8m0 = ((rounded_bits >> 23) & 0xFF).clamp(max=254).to(torch.uint8)
    ue8m0 = torch.where(amax > 0, ue8m0, torch.zeros_like(ue8m0))
    decoded_scale = torch.where(
        ue8m0 > 0,
        (ue8m0.to(torch.int32) << 23).view(torch.float32),
        torch.zeros_like(raw_scale),
    )
    normalized = torch.where(
        amax.unsqueeze(-1) > 0,
        blocks / decoded_scale.unsqueeze(-1),
        torch.zeros_like(blocks),
    )
    codes, _ = _e2m1_quantize(normalized)
    # UE8M0 stores the biased exponent.  Keep it uint8 for the SM103 scale MMA.
    return Fp4Tensor(
        payload=_pack_codes(codes.view(rows.shape[0], 128)).view(
            *x.shape[:-1], 64
        ),
        scale=ue8m0.view(torch.float8_e8m0fnu)
        .view(*x.shape[:-1], 4)
        .contiguous(),
        inverse_global_scale=1.0,
        original_shape=tuple(x.shape),
        scale_vec_size=32,
        recipe="mxfp4_ue8m0_per32",
    )


def dequantize_fp4(value: Fp4Tensor) -> torch.Tensor:
    """Decode an :class:`Fp4Tensor` to FP32 in logical row-major order."""
    codes = _unpack_codes(value.payload.reshape(-1, 64))
    decoded = _decode_codes(codes).view(-1, 128 // value.scale_vec_size, value.scale_vec_size)
    if value.recipe.startswith("mxfp4"):
        encoded = value.scale.view(torch.uint8).reshape(-1, 128 // 32)
        scale = torch.where(
            encoded > 0,
            (encoded.to(torch.int32) << 23).view(torch.float32),
            torch.zeros_like(encoded, dtype=torch.float32),
        )
    else:
        scale = value.scale.reshape(-1, 128 // 16).float()
    out = decoded * scale.unsqueeze(-1) * float(value.inverse_global_scale)
    return out.reshape(value.original_shape)


__all__ = [
    "E2M1_MAX",
    "E4M3_MAX",
    "Fp4Tensor",
    "dequantize_fp4",
    "quantize_mxfp4",
    "quantize_nvfp4",
]
