# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Maxwell Team
# SPDX-License-Identifier: Apache-2.0
# Derived from the vLLM Maxwell-V1-Flash fork; modified for MBSA.
# See third_party/vllm/NOTICE.

"""Triton encoders for the packed MXFP6 index K and the MXFP8 index Q."""

from functools import cache

import torch
import triton
import triton.language as tl

from .. import index_formats

#: Rows per token of the MXFP8 index Q: tcgen05 needs ``N % 8 == 0``, so the
#: four index heads are padded to eight.
Q_ROWS = 8
_BLOCK = 32


@cache
def page_layout() -> dict[str, int]:
    """The packed page layout as the encoders' constexpr arguments.

    Cached: the fused launch asks once per layer per step.
    """
    return {
        "PAGE_BYTES": index_formats.PAGE_BYTES,
        "CODE_BYTES": index_formats.CODE_BYTES_PER_TOKEN,
        "CODE_REGION": index_formats.PAGE_CODE_BYTES,
        "TOKENS": index_formats.PAGE_TOKENS,
    }


@triton.jit
def _fwht_stage(x, STRIDE: tl.constexpr, DIM: tl.constexpr):
    """One butterfly stage: pair element ``i`` with ``i ^ STRIDE``."""
    g = tl.trans(tl.reshape(x, (DIM // (2 * STRIDE), 2, STRIDE)), 0, 2, 1)
    a, b = tl.split(g)
    return tl.reshape(tl.trans(tl.join(a + b, a - b), 0, 2, 1), (DIM,))


@triton.jit
def _hadamard_128(x, DIM: tl.constexpr):
    # Ascending stages, normalized once: the order is the numeric contract.
    x = _fwht_stage(x, 1, DIM)
    x = _fwht_stage(x, 2, DIM)
    x = _fwht_stage(x, 4, DIM)
    x = _fwht_stage(x, 8, DIM)
    x = _fwht_stage(x, 16, DIM)
    x = _fwht_stage(x, 32, DIM)
    x = _fwht_stage(x, 64, DIM)
    return x * (DIM**-0.5)


@triton.jit
def mxfp6_encode_store(
    x,
    slot,
    cache_ptr,
    ROTATE: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    CODE_BYTES: tl.constexpr,
    CODE_REGION: tl.constexpr,
    TOKENS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Rotate, block-quantize and pack one token's fp32 index K into its slot."""
    page = slot // TOKENS
    off = slot % TOKENS
    if ROTATE:
        x = _hadamard_128(x, DIM)

    # Quark's even_round; uint32 because the 0xFF800000 mask overflows int32.
    xb = tl.reshape(x, (DIM // BLOCK, BLOCK))
    amax = tl.max(tl.abs(xb), axis=1)
    bits = amax.to(tl.uint32, bitcast=True)
    bits = (bits + (1 << 19)) & 0xFF800000
    rounded = bits.to(tl.float32, bitcast=True)
    rounded = tl.where(amax == 0, 2.0**-126, rounded)
    exp = tl.floor(tl.log2(rounded)) - 2.0
    exp = tl.maximum(tl.minimum(exp, 127.0), -127.0)
    scale = tl.exp2(exp)

    # Round to the E2M3 grid, then saturate.
    v = xb / scale[:, None]
    e = tl.maximum(tl.floor(tl.log2(tl.maximum(tl.abs(v), 1e-30))), 0.0)
    step = tl.exp2(e - 3.0)
    g = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(v / step) * step, -7.5), 7.5)

    a = tl.abs(g)
    normal = a >= 1.0
    vexp = tl.where(normal, tl.floor(tl.log2(tl.maximum(a, 1.0))) + 1.0, 0.0)
    unit = tl.where(normal, tl.exp2(vexp - 1.0), 1.0)
    mant = tl.extra.cuda.libdevice.rint(
        tl.where(normal, (a / unit - 1.0) * 8.0, a * 8.0)
    )
    code = tl.where(g < 0, 32, 0) + vexp.to(tl.int32) * 8 + mant.to(tl.int32)
    code = tl.where(a == 0, 0, code)

    # Element i*4 + a*2 + b of a (DIM/4, 2, 2) view splits out as (b, a).
    quad = tl.reshape(code, (DIM // 4, 2, 2))
    even, odd = tl.split(quad)
    c0, c2 = tl.split(even)
    c1, c3 = tl.split(odd)
    b0 = (c0 | ((c1 & 0x3) << 6)) & 0xFF
    b1 = ((c1 >> 2) | ((c2 & 0xF) << 4)) & 0xFF
    b2 = ((c2 >> 4) | (c3 << 2)) & 0xFF

    base = cache_ptr + page * PAGE_BYTES + off * CODE_BYTES
    g3 = tl.arange(0, DIM // 4) * 3
    tl.store(base + g3, b0.to(tl.uint8))
    tl.store(base + g3 + 1, b1.to(tl.uint8))
    tl.store(base + g3 + 2, b2.to(tl.uint8))

    # Scale word, in the UTCCP order of index_formats.scale_slot.
    sf_slot = (off % 32) * 4 + off // 32
    sf_base = cache_ptr + page * PAGE_BYTES + CODE_REGION + sf_slot * 4
    tl.store(sf_base + tl.arange(0, DIM // BLOCK), (exp + 127.0).to(tl.uint8))


@triton.jit
def mxfp8_q_pad_store(out_row_ptr, sf_ptr, DIM: tl.constexpr):
    """A padding row: zero codes under a 2^0 scale in every block."""
    tl.store(out_row_ptr + tl.arange(0, DIM), tl.zeros((DIM,), dtype=tl.uint8))
    tl.store(sf_ptr, 0x7F7F7F7F)


@triton.jit
def mxfp8_q_encode_store(
    x, out_row_ptr, sf_ptr, DIM: tl.constexpr, BLOCK: tl.constexpr
):
    """Rotate, block-scale and cast one fp32 index-Q row to MXFP8."""
    d = tl.arange(0, DIM)
    x = _hadamard_128(x, DIM)

    xb = tl.reshape(x, (DIM // BLOCK, BLOCK))
    amax = tl.max(tl.abs(xb), axis=1)
    bits = amax.to(tl.uint32, bitcast=True)
    bits = (bits + (1 << 19)) & 0xFF800000
    rounded = bits.to(tl.float32, bitcast=True)
    rounded = tl.where(amax == 0, 2.0**-126, rounded)
    # `rounded` has no mantissa, so its exponent field is log2 exactly and the
    # inverse scale is a power of two built from bits.
    e_field = ((rounded.to(tl.uint32, bitcast=True) >> 23) & 0xFF).to(tl.int32)
    exp_i = tl.maximum(tl.minimum(e_field - 135, 127), -127)
    exp = exp_i.to(tl.float32)
    inv_scale = ((127 - exp_i) << 23).to(tl.float32, bitcast=True)
    # Clamp first: the hardware cast turns values past 448 into NaN.
    v = tl.minimum(tl.maximum(xb * inv_scale[:, None], -448.0), 448.0)
    tl.store(
        out_row_ptr + d,
        tl.reshape(v, (DIM,)).to(tl.float8e4nv).to(tl.uint8, bitcast=True),
    )
    word = tl.sum((exp + 127.0).to(tl.int32) << (8 * tl.arange(0, DIM // BLOCK)))
    tl.store(sf_ptr, word)


@triton.jit
def _write_index_k_kernel(
    k_ptr,
    slots_ptr,
    cache_ptr,
    stride_k_t,
    ROTATE: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    CODE_BYTES: tl.constexpr,
    CODE_REGION: tl.constexpr,
    TOKENS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    slot = tl.load(slots_ptr + t)
    if slot < 0:
        return
    x = tl.load(k_ptr + t * stride_k_t + tl.arange(0, DIM)).to(tl.float32)
    mxfp6_encode_store(
        x, slot, cache_ptr, ROTATE, PAGE_BYTES, CODE_BYTES, CODE_REGION,
        TOKENS, DIM, BLOCK,
    )  # fmt: skip


def write_index_k(
    key: torch.Tensor, slots: torch.Tensor, cache: torch.Tensor, rotate: bool = True
) -> None:
    """Quantize post-RoPE K [T,128] into a flat uint8 MXFP6 cache.
    slots [T] gives global token positions; negative slots are skipped."""
    if key.shape[-1] != 128:
        raise ValueError(f"index-K head dim must be 128, got {key.shape[-1]}")
    if cache.dtype is not torch.uint8:
        raise ValueError(f"the packed cache is uint8, got {cache.dtype}")
    _write_index_k_kernel[(key.shape[0],)](
        key,
        slots,
        cache,
        key.stride(0),
        ROTATE=rotate,
        DIM=128,
        BLOCK=_BLOCK,
        **page_layout(),
    )


@triton.jit
def _quantize_q_mxfp8_kernel(
    q_ptr,
    out_ptr,
    sf_ptr,
    stride_q_t,
    stride_q_h,
    HEADS: tl.constexpr,
    PAD: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    if h >= HEADS:
        mxfp8_q_pad_store(out_ptr + (t * PAD + h) * DIM, sf_ptr + t * PAD + h, DIM)
        return
    x = tl.load(q_ptr + t * stride_q_t + h * stride_q_h + tl.arange(0, DIM))
    mxfp8_q_encode_store(
        x.to(tl.float32), out_ptr + (t * PAD + h) * DIM, sf_ptr + t * PAD + h,
        DIM, BLOCK,
    )  # fmt: skip


def quantize_q_mxfp8_fused(
    query: torch.Tensor, pad_to: int = Q_ROWS
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode post-RoPE Q [T,H,128] as uint8 E4M3 [T,pad_to,128]
    and int32 scale words [T,pad_to]; pad_to must cover H and be divisible by eight."""
    tokens, heads, dim = query.shape
    if heads > pad_to:
        raise ValueError(f"{heads} index heads do not fit in {pad_to} rows")
    out = torch.empty(tokens, pad_to, dim, dtype=torch.uint8, device=query.device)
    sf = torch.empty((tokens, pad_to), dtype=torch.int32, device=query.device)
    _quantize_q_mxfp8_kernel[(tokens, pad_to)](
        query,
        out,
        sf,
        query.stride(0),
        query.stride(1),
        HEADS=heads,
        PAD=pad_to,
        DIM=dim,
        BLOCK=_BLOCK,
    )
    return out, sf
