"""Lazy-built SM103 packed raw-E4M3 quantizer.

The extension is experimental and is never selected by production BF16 or the
default FP8 adapter.  Formal callers must prebuild it outside the timed window.
"""

from __future__ import annotations

import functools
import pathlib

import torch
from torch.utils.cpp_extension import load


@functools.lru_cache(maxsize=1)
def _extension():
    source = pathlib.Path(__file__).with_name("csrc") / "fp8_raw_quant_sm103.cu"
    return load(
        name="fp8_raw_quant_sm103",
        sources=[str(source)],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-gencode=arch=compute_103a,code=sm_103a",
        ],
        extra_cflags=["-O3"],
        verbose=False,
    )


def quantize_raw_pair_out(
    q: torch.Tensor,
    q_out: torch.Tensor,
    k: torch.Tensor,
    k_out: torch.Tensor,
) -> None:
    """Convert contiguous BF16 Q/K to caller-owned raw E4M3 buffers."""
    _extension().quantize_raw_pair_out(q, q_out, k, k_out)


def quantize_static_pair_out(
    q: torch.Tensor,
    q_out: torch.Tensor,
    q_scale: float,
    k: torch.Tensor,
    k_out: torch.Tensor,
    k_scale: float,
) -> None:
    """Convert BF16 Q/K to scaled E4M3 without a staging buffer."""
    _extension().quantize_static_pair_out(
        q, q_out, float(q_scale), k, k_out, float(k_scale)
    )


def quantize_block_pair_out(
    q: torch.Tensor,
    q_out: torch.Tensor,
    q_scales: torch.Tensor,
    k: torch.Tensor,
    k_out: torch.Tensor,
    k_scales: torch.Tensor,
) -> None:
    """Quantize q128/group Q and physical K pages after one global read."""
    _extension().quantize_block_pair_out(
        q, q_out, q_scales, k, k_out, k_scales
    )
