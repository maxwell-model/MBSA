# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MX index operand codecs compatible with the vLLM Maxwell cache layout.

Adapted from MiniMax/Shopee vLLM branch ``aip/maxwell-v1-flash``, revision
950d80710610ebffc41128dad89845346262af48, file
vllm/model_executor/layers/maxwell_moe_index_mxfp6.py (Apache-2.0).

The dequantization functions are precision oracles. Performance kernels read
packed operands and block scales directly; they must not use a global BF16
cache expansion. Hadamard butterfly order and UE8M0 rounding are part of the
format contract.
"""

import functools

import torch

#: (exponent bits, mantissa bits, emax, maxval); emax matches Quark's
#: get_dtype_params, i.e. 2**(ebits-1) for these two formats.
_FMT = {"e4m3": (4, 3, 8, 448.0), "e2m3": (2, 3, 2, 7.5)}
_BLOCK = 32


@functools.lru_cache(maxsize=8)
def _hadamard(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Normalized Sylvester H_n. Symmetric and orthogonal, so (Hq).(Hk) = q.k."""
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / n**0.5).to(device=device, dtype=dtype)


def _even_round(amax: torch.Tensor, mbits: int, emax: int) -> torch.Tensor:
    """Quark's UE8M0 scale rule.

    Adding ``2**(23-mbits-1)`` to the float32 bit pattern before masking the
    mantissa carries into the exponent once the mantissa reaches 15/16, so the
    binade boundary is ``2**n * 1.9375``. Kernels must match this exactly.
    """
    x = amax.to(torch.float32).clone()
    zero, nan = x == 0, torch.isnan(x)
    i = x.view(torch.int32)
    i.add_(1 << (23 - mbits - 1))
    i.bitwise_and_(((1 << 9) - 1) << 23)
    x = i.view(torch.float32)
    x.masked_fill_(zero, 2.0**-126)
    x.masked_fill_(nan, float("nan"))
    return x.log2_().floor_().sub_(emax).clamp_(-127, 127).exp2_()


def _round_to_grid(
    x: torch.Tensor, mbits: int, min_exp: int, maxval: float
) -> torch.Tensor:
    e = torch.floor(torch.log2(x.abs().clamp(min=1e-30))).clamp(min=min_exp)
    step = torch.exp2(e - mbits)
    return (torch.round(x / step) * step).clamp(-maxval, maxval)


def mx_quant_dequant(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """Block-scaled quantize/dequantize along the last dim, returned in fp32."""
    ebits, mbits, emax, maxval = _FMT[fmt]
    min_exp = 1 - (2 ** (ebits - 1) - 1)
    xs = x.float().unflatten(-1, (-1, _BLOCK))
    scale = _even_round(xs.abs().amax(-1), mbits, emax).unsqueeze(-1)
    return (_round_to_grid(xs / scale, mbits, min_exp, maxval) * scale).flatten(-2)


def rotate_dense(x: torch.Tensor) -> torch.Tensor:
    """The Hadamard as a dense matmul, for cross-checking :func:`rotate`."""
    h = _hadamard(x.shape[-1], x.device, torch.float32)
    return x.float() @ h


def rotate(x: torch.Tensor) -> torch.Tensor:
    """Apply the Hadamard. Must be applied to both operands or scores change.

    Butterfly stages in ascending stride, normalized once at the end. This
    order is the definition: a dense matmul differs in floating point and can
    flip values on a quantization boundary, so kernels must match it.
    """
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(f"the Hadamard needs a power-of-two dim, got {n}")
    y = x.float()
    stride = 1
    while stride < n:
        y = y.unflatten(-1, (-1, 2, stride))
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack([a + b, a - b], dim=-2).flatten(-3)
        stride *= 2
    return y * n**-0.5


def fake_quant_k(key: torch.Tensor, rotate_first: bool = True) -> torch.Tensor:
    """index-K as MXFP6-E2M3 would hold it."""
    return mx_quant_dequant(rotate(key) if rotate_first else key, "e2m3")


def fake_quant_q(query: torch.Tensor, rotate_first: bool = True) -> torch.Tensor:
    """index-Q as MXFP8-E4M3 would hold it."""
    return mx_quant_dequant(rotate(query) if rotate_first else query, "e4m3")


BLOCK = 32
#: Four 6-bit codes fill three bytes exactly, little-endian in bit order. This
#: is the cache layout; kernels may expand it in shared memory.
CODES_PER_WORD = 4
BYTES_PER_WORD = 3


def encode_e2m3(v: torch.Tensor) -> torch.Tensor:
    """Scale-normalized E2M3 values to their 6-bit codes: sign, 2 exp, 3 mantissa.

    ``v`` must already sit on the E2M3 grid, which it does coming out of
    ``mx_quant_dequant``; the rounding here is exact, not a second quantization.
    """
    a = v.abs()
    normal = a >= 1.0
    exp = torch.where(
        normal, torch.floor(torch.log2(a.clamp(min=1.0))) + 1, torch.zeros_like(a)
    )
    unit = torch.where(normal, torch.exp2(exp - 1), torch.ones_like(a))
    mant = torch.where(normal, (a / unit - 1.0) * 8.0, a * 8.0).round()
    code = (v < 0).to(torch.int32) * 32 + exp.to(torch.int32) * 8 + mant.to(torch.int32)
    return torch.where(a == 0, torch.zeros_like(code), code).to(torch.uint8)


def decode_e2m3(code: torch.Tensor) -> torch.Tensor:
    c = code.to(torch.int32)
    exp, mant = (c >> 3) & 0x3, (c & 0x7).float()
    normal = exp > 0
    a = torch.where(
        normal, torch.exp2(exp.float() - 1) * (1.0 + mant / 8.0), mant / 8.0
    )
    return torch.where((c & 32) > 0, -a, a)


def pack_codes(code: torch.Tensor) -> torch.Tensor:
    c = code.to(torch.int32).unflatten(-1, (-1, CODES_PER_WORD))
    c0, c1, c2, c3 = (c[..., i] for i in range(CODES_PER_WORD))
    packed = torch.stack(
        [c0 | (c1 & 0x3) << 6, (c1 >> 2) | (c2 & 0xF) << 4, (c2 >> 4) | c3 << 2], -1
    )
    return (packed & 0xFF).to(torch.uint8).flatten(-2)


def unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    b = packed.to(torch.int32).unflatten(-1, (-1, BYTES_PER_WORD))
    b0, b1, b2 = (b[..., i] for i in range(BYTES_PER_WORD))
    return (
        torch.stack(
            [
                b0 & 0x3F,
                (b0 >> 6) | (b1 & 0xF) << 2,
                (b1 >> 4) | (b2 & 0x3) << 4,
                b2 >> 2,
            ],
            -1,
        )
        .to(torch.uint8)
        .flatten(-2)
    )


def pack_mxfp6(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to MXFP6-E2M3 and pack: 6.25 bits per element.

    Args:
        x: values whose last dimension is a whole number of 32-element blocks.

    Returns:
        ``(codes, scales)``: packed 6-bit codes, and one UE8M0 exponent byte per
        block.

    Raises:
        ValueError: if the last dimension is not a multiple of 32.
    """
    if x.shape[-1] % BLOCK:
        raise ValueError(
            f"last dim must be a whole number of {BLOCK}-element blocks, "
            f"got {x.shape[-1]}"
        )
    xs = x.float().unflatten(-1, (-1, BLOCK))
    scale = _even_round(xs.abs().amax(-1), 3, 2).unsqueeze(-1)
    codes = encode_e2m3(_round_to_grid(xs / scale, 3, 0, 7.5)).flatten(-2)
    exponent = (torch.log2(scale.squeeze(-1)) + 127).to(torch.uint8)
    return pack_codes(codes), exponent


def unpack_mxfp6(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_mxfp6`, exact on the values it was given."""
    v = decode_e2m3(unpack_codes(codes)).unflatten(-1, (-1, BLOCK))
    return (v * torch.exp2(scales.float() - 127).unsqueeze(-1)).flatten(-2)


# A page holds 128 tokens as [codes 128 x 96 B][scales 128 x 4 B] = 12800 B.
# Keeping the regions apart lets one TMA descriptor address the codes: the row
# pitch stays 96 bytes (a multiple of 32) and the page pitch 16-byte aligned.

PAGE_TOKENS = 128
CODE_BYTES_PER_TOKEN = 96  # 128 codes x 6 bit
SCALE_BYTES_PER_TOKEN = 4  # one UE8M0 per 32-element block
PAGE_CODE_BYTES = PAGE_TOKENS * CODE_BYTES_PER_TOKEN
PAGE_BYTES = PAGE_CODE_BYTES + PAGE_TOKENS * SCALE_BYTES_PER_TOKEN
#: Stored bytes per token, i.e. the `head_size` of a uint8 cache spec.
BYTES_PER_TOKEN = PAGE_BYTES // PAGE_TOKENS


def scale_slot(token: torch.Tensor | int) -> torch.Tensor | int:
    """Where a token's scale word lives inside the page's scale region.

    The MMA reads block scales out of tensor memory, and UTCCP fills that from
    shared memory as 32 datapaths of 128 bits. The four rows one datapath
    carries are strided by 32, not consecutive, so token ``t`` belongs at
    ``(t % 32) * 4 + t // 32``. Token order is a silent bug that only shows
    once a page's tokens have different scales. Permuting on write lets the
    kernel stream the region straight into shared memory.
    """
    if isinstance(token, int):
        return (token % 32) * 4 + token // 32
    return (token % 32) * 4 + torch.div(token, 32, rounding_mode="floor")


def build_page(key: torch.Tensor) -> torch.Tensor:
    """Lay out one page of index-K exactly as the kernel reads it.

    Args:
        key: ``[128, 128]`` index-K for one page, already rotated.

    Returns:
        ``[12800]`` uint8: the packed codes, then the permuted scale words.

    Raises:
        ValueError: if the page is not 128 tokens of 128 dims.
    """
    if key.shape != (PAGE_TOKENS, 128):
        raise ValueError(
            f"a page is {PAGE_TOKENS} tokens of 128 dims, got {tuple(key.shape)}"
        )
    codes, scales = pack_mxfp6(key)
    page = torch.empty(PAGE_BYTES, dtype=torch.uint8, device=key.device)
    page[:PAGE_CODE_BYTES] = codes.reshape(-1)
    slots = scale_slot(torch.arange(PAGE_TOKENS, device=key.device))
    tail = page[PAGE_CODE_BYTES:].view(PAGE_TOKENS, SCALE_BYTES_PER_TOKEN)
    tail[slots] = scales.to(torch.uint8)
    return page


def read_page(page: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`build_page`, for tests and for the fake-quant path."""
    codes = page[:PAGE_CODE_BYTES].view(PAGE_TOKENS, CODE_BYTES_PER_TOKEN)
    slots = scale_slot(torch.arange(PAGE_TOKENS, device=page.device))
    tail = page[PAGE_CODE_BYTES:].view(PAGE_TOKENS, SCALE_BYTES_PER_TOKEN)
    return unpack_mxfp6(codes, tail[slots])


def quantize_q_mxfp8(
    query: torch.Tensor, rotate_first: bool = True, pad_to: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """index-Q as the packed kernel wants it: E4M3 bytes plus UE8M0 words.

    Args:
        query: ``[T, H, 128]`` index-Q, post-RoPE.
        rotate_first: Apply the Hadamard. It must match the key side, since
            ``(Hq).(Hk) = q.k`` only holds when both are rotated.
        pad_to: Rows per token in the output. tcgen05 needs ``N % 8 == 0``, so
            4 real heads are padded to 8; the padding is zeros and its scale
            is 2^0, which keeps it a legal operand producing exactly zero.

    Returns:
        ``(bytes, scales)``: ``[T, pad_to, 128]`` uint8 E4M3 and
        ``[T, pad_to]`` int32, four UE8M0 bytes per row.

    Values are clamped before the cast: the UE8M0 scale rule can leave a block
    maximum above E4M3's 448, and the cast turns that into NaN rather than
    saturating, poisoning the page score.
    """
    q = rotate(query) if rotate_first else query.float()
    blocks = q.unflatten(-1, (-1, BLOCK))
    scale = _even_round(blocks.abs().amax(-1), 3, 8)
    normalised = (blocks / scale.unsqueeze(-1)).clamp(-448.0, 448.0)
    codes = normalised.flatten(-2).to(torch.float8_e4m3fn).view(torch.uint8)
    exponent = (torch.log2(scale) + 127).to(torch.int32)

    heads = query.shape[-2]
    if heads > pad_to:
        raise ValueError(f"{heads} index heads do not fit in {pad_to} rows")
    out = torch.zeros(
        *query.shape[:-2],
        pad_to,
        query.shape[-1],
        dtype=torch.uint8,
        device=query.device,
    )
    out[..., :heads, :] = codes
    words = torch.full(
        (*query.shape[:-2], pad_to), 0x7F7F7F7F, dtype=torch.int32, device=query.device
    )
    packed_exp = sum(exponent[..., b] << (8 * b) for b in range(exponent.shape[-1]))
    words[..., :heads] = packed_exp
    return out, words


def quantize_index_k_mxfp6(
    key: torch.Tensor, *, rotate_first: bool = True
) -> torch.Tensor:
    """Return native cache view ``[pages,1,128,100]`` from BF16 index K.

    Input is ``[pages,1,128,128]``. The view's last dimension is a storage
    convention: a page contains 12288 code bytes then 512 permuted scale bytes,
    rather than 100 interleaved bytes for each token.
    """
    if key.ndim != 4 or tuple(key.shape[1:]) != (1, PAGE_TOKENS, 128):
        raise ValueError("index K must have shape [pages,1,128,128]")
    values = rotate(key[:, 0]) if rotate_first else key[:, 0].float()
    codes, scales = pack_mxfp6(values)
    page = torch.empty((key.shape[0], PAGE_BYTES), dtype=torch.uint8, device=key.device)
    page[:, :PAGE_CODE_BYTES] = codes.flatten(1)
    slots = scale_slot(torch.arange(PAGE_TOKENS, device=key.device))
    page[:, PAGE_CODE_BYTES:].view(-1, PAGE_TOKENS, SCALE_BYTES_PER_TOKEN)[:, slots] = (
        scales
    )
    return page.view(-1, 1, PAGE_TOKENS, BYTES_PER_TOKEN)


def dequantize_index_k_mxfp6(cache: torch.Tensor) -> torch.Tensor:
    """FP32 precision oracle for the native packed page view."""
    if (
        cache.dtype != torch.uint8
        or cache.ndim != 4
        or tuple(cache.shape[1:]) != (1, PAGE_TOKENS, BYTES_PER_TOKEN)
        or not cache.is_contiguous()
    ):
        raise ValueError("MXFP6 cache must be contiguous uint8 [pages,1,128,100]")
    pages = cache.view(-1, PAGE_BYTES)
    codes = pages[:, :PAGE_CODE_BYTES].reshape(-1, PAGE_TOKENS, CODE_BYTES_PER_TOKEN)
    slots = scale_slot(torch.arange(PAGE_TOKENS, device=cache.device))
    scales = pages[:, PAGE_CODE_BYTES:].reshape(-1, PAGE_TOKENS, SCALE_BYTES_PER_TOKEN)[
        :, slots
    ]
    return unpack_mxfp6(codes, scales).unsqueeze(1)


def dequantize_index_q_mxfp8(query: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """FP32 oracle accepting padded native Q or its noncontiguous head slice.

    Q is E4M3 or raw uint8 bytes ``[T,H,128]``; scales are int32 words
    ``[T,>=H]`` with four little-endian UE8M0 exponent bytes per head.
    """
    if (
        query.ndim != 3
        or query.shape[-1] != 128
        or query.dtype not in (torch.uint8, torch.float8_e4m3fn)
    ):
        raise ValueError("MXFP8 Q must be E4M3 or uint8 [T,H,128]")
    if (
        scales.dtype != torch.int32
        or scales.ndim != 2
        or scales.shape[0] != query.shape[0]
        or scales.shape[1] < query.shape[1]
        or scales.device != query.device
    ):
        raise ValueError("MXFP8 Q scales must be int32 [T,>=H] on the Q device")
    words = scales[:, : query.shape[1]]
    exponents = torch.stack([(words >> (8 * b)) & 255 for b in range(4)], dim=-1)
    values = query.view(torch.float8_e4m3fn).float().unflatten(-1, (4, BLOCK))
    return (values * torch.exp2(exponents.float() - 127).unsqueeze(-1)).flatten(-2)
