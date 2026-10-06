# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Adapted from Block-Sparse-Attention/bsa_attn_interface.py for FlashInfer integration.

import os
import math
from functools import lru_cache
from enum import Enum
from typing import Optional, Tuple

import torch

from mbsa.cute.src.sm100.fixed.api_logging import flashinfer_api

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute

from mbsa.cute.src.sm100.fixed.bsa_utils.cache_utils import get_jit_cache
from mbsa.cute.src.sm100.fixed.bsa_utils.testing import is_fake_mode
from mbsa.cute.src.sm100.fixed.bsa_utils import fa_logging
from mbsa.cute.src.sm100.fixed.bsa_utils.cute_tensor_utils import to_cute_tensor
from mbsa.cute.src.sm100.fixed.flash_fwd_sm100 import FlashAttentionForwardSm100

_bsa_clc_enabled: bool = os.environ.get("BSA_CLC", "1") == "1"


class BsaOutputLayout(str, Enum):
    """Physical O layout; packed LSE is [Hq,T] in both layouts."""

    SCATTERED = "scattered"
    NON_SCATTERED = "non_scattered"


def _get_use_clc_scheduler() -> bool:
    return _bsa_clc_enabled


def _parse_arch_str(arch_str):
    import re

    match = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_str)
    if not match:
        raise ValueError(f"Invalid arch format: {arch_str}")
    major, minor, _ = match.groups()
    return int(major) * 10 + int(minor)


@lru_cache(maxsize=None)
def _get_device_arch(device_index: int):
    # device_index is unused when FLASH_ATTENTION_ARCH is set, but keeping it
    # in the signature keeps the lru_cache keyed per-device below -- a
    # zero-arg cache would pin the first-seen device's arch for the rest of
    # the process on a mixed-architecture host.
    arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    major, minor = torch.cuda.get_device_capability(device_index)
    return major * 10 + int(minor)


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


_sm100_blk128_compile_cache = get_jit_cache("bsa_fwd")


@flashinfer_api
def bsa_attn_sm100_blk128_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    allow_empty_block_nums: bool = True,
    softmax_scale: Optional[float] = None,
    pack_gqa: Optional[bool] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    causal: bool = False,
    implicit_fixed_support: bool = False,
    page_table: Optional[torch.Tensor] = None,
    use_clc_scheduler: Optional[bool] = None,
    split_p_arrive: Optional[int] = None,
    sync_first_p_release: bool = False,
    implicit_fixed_load_reduce: bool = False,
    output_layout: BsaOutputLayout = BsaOutputLayout.SCATTERED,
    logical_seqlen_k: Optional[int] = None,
    packed_tiles: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for BSA block-sparse attention using the blk128 CuTe-DSL kernel.

    Supports SM100 and SM103.

    Args:
        q: Packed query tensor [T,H,D] with packed_tiles, or legacy [B,S,H,D]
        k: Packed dense key tensor ``[Tkv,Hkv,D]`` (legacy ``[B,S,Hkv,D]``), or paged key
            tensor ``[physical_pages,Hkv,128,D]`` when ``page_table`` is set.
        v: Value tensor with the same dense/paged layout as ``k``.
        q2k_block_index: (batch, num_heads, num_q_blocks, max_kv_blocks) int32
        block_sparse_num: Number of KV blocks per Q block (even, >= 2).
            Ignored when q2k_block_nums is provided.
        block_sizes: Actual token count per KV block (num_kv_blocks,) int32.
            Pass None to skip block-size masking (assumes full blocks).
        q2k_block_nums: Per-(batch,head,q_block) number of KV blocks,
            (batch, num_heads, num_q_blocks) int32.  Optional.
        allow_empty_block_nums: Allow q2k_block_nums to contain 0.
        softmax_scale: Softmax scale (default: 1/sqrt(head_dim)).
        pack_gqa: Whether to pack GQA heads.
        return_lse: Whether to return log-sum-exp.
        out: Packed output [T,Hq,D] for scattered or [Hkv,T,GQA,D]
            for non-scattered output. Legacy inputs retain the leading B axis.
        lse: Pre-allocated LSE tensor.
        causal: Apply token-level causal masking within selected block-128 pages.
            When ``seqlen_q < seqlen_k``, queries are aligned to the end of the
            KV sequence (bottom-right causal masking), including partial tiles.
        implicit_fixed_support: Generate the prefix-8 plus local-8 union inside
            the kernel. The route tensor is then a one-element ABI sentinel.
        page_table: Optional ``[batch,logical_pages]`` int32 mapping.  In paged
            mode logical block IDs continue to drive causal masking while the
            load warp resolves physical page IDs immediately before TMA.
        logical_seqlen_k: Valid KV length for paged input. Defaults to full
            pages. Uniform batches share this length; packed tiles carry
            their own per-request KV bounds.
        packed_tiles: Optional request-local tile descriptors for packed fixed
            attention. Each row is [request, local_m_tile, q_begin, q_end,
            k_begin, k_end]; tile width is 128 packed GQA rows. Build once
            during plan preparation; all requests execute in one kernel.
        use_clc_scheduler: Optional compile-time scheduler policy. ``None``
            preserves the process default; callers may select the static
            scheduler for a measured fixed-shape dispatch region.
    """
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    if not isinstance(output_layout, BsaOutputLayout):
        raise TypeError("output_layout must be BsaOutputLayout")
    grouped_output = output_layout is BsaOutputLayout.NON_SCATTERED
    is_varlen = packed_tiles is not None
    if is_varlen:
        if q.ndim != 3:
            raise ValueError("packed Q must be [T,H,D]")
        seqlen_q, num_head, head_dim = q.shape
        batch_size = 1  # Scheduler has one flat request-tile list.
    else:
        batch_size, seqlen_q, num_head, head_dim = q.shape
    if is_varlen:
        if not implicit_fixed_support:
            raise ValueError("packed tiles require implicit fixed support")
        if (packed_tiles.ndim != 2 or packed_tiles.shape[1] != 6
                or packed_tiles.shape[0] == 0 or packed_tiles.dtype != torch.int32
                or packed_tiles.device != q.device or not packed_tiles.is_contiguous()):
            raise ValueError("packed_tiles must be contiguous [tiles,6] int32 on the Q device")
    is_paged_kv = page_table is not None
    if is_paged_kv:
        page_table = maybe_contiguous(page_table)
        if page_table.ndim != 2 or (not is_varlen and page_table.shape[0] != batch_size):
            raise ValueError("page_table must be [batch,logical_pages]")
        if page_table.dtype != torch.int32 or page_table.device != q.device:
            raise ValueError("page_table must be CUDA int32 on the Q device")
        if (
            k.ndim != 4
            or v.ndim != 4
            or tuple(k.shape[:3]) != tuple(v.shape[:3])
            or k.shape[2] != 128
        ):
            raise ValueError("paged K/V must be [physical_pages,Hkv,128,D]")
        num_head_kv = k.shape[1]
        padded_seqlen_k = page_table.shape[1] * 128
        seqlen_k = padded_seqlen_k if logical_seqlen_k is None else logical_seqlen_k
        if not 0 < seqlen_k <= padded_seqlen_k:
            raise ValueError("logical_seqlen_k must fit the page table")
    else:
        expected_rank = 3 if is_varlen else 4
        if (k.ndim != expected_rank or v.ndim != expected_rank
                or tuple(k.shape[:-1]) != tuple(v.shape[:-1])
                or (not is_varlen and k.shape[0] != batch_size)):
            raise ValueError("dense K/V must be [T,Hkv,D] (legacy: [B,S,Hkv,D])")
        num_head_kv = k.shape[-2]
        seqlen_k = k.shape[-3]
        if logical_seqlen_k is not None and logical_seqlen_k != seqlen_k:
            raise ValueError("dense logical_seqlen_k must match K/V length")
    head_dim_v = v.shape[-1]

    assert q.dtype in [torch.float16, torch.bfloat16], (
        "inputs must be float16 or bfloat16"
    )
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"

    if not is_fake_mode():
        assert all(t.is_cuda for t in (q, k, v)), "inputs must be on CUDA device"

    arch = _get_device_arch(q.device.index)
    assert arch in (100, 103), (
        f"bsa_attn_sm100_blk128_fwd only supports SM100/SM103, got SM{arch}"
    )
    assert num_head % num_head_kv == 0
    if causal and not is_varlen and seqlen_k < seqlen_q:
        raise ValueError("causal BSA requires seqlen_k >= seqlen_q")

    assert q2k_block_index.dtype == torch.int32, "q2k_block_index must be int32"
    if implicit_fixed_support:
        if not causal or q2k_block_nums is not None:
            raise ValueError(
                "implicit fixed support requires causal=True and no block counts"
            )
        if q2k_block_index.numel() != 1:
            raise ValueError("implicit fixed support expects one int32 route sentinel")
        block_sparse_num = 16
    has_block_sizes = block_sizes is not None
    if has_block_sizes:
        assert block_sizes.dtype == torch.int32, "block_sizes must be int32"
    if q2k_block_nums is not None:
        q2k_block_nums = maybe_contiguous(q2k_block_nums)
        assert q2k_block_nums.dtype == torch.int32, "q2k_block_nums must be int32"
    else:
        assert block_sparse_num >= 2 and block_sparse_num % 2 == 0, (
            f"block_sparse_num={block_sparse_num} must be even and >= 2"
        )

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    qhead_per_kvhead = num_head // num_head_kv

    tile_n = 128
    use_2cta_instrs = False
    tile_m = 128

    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1
    if pack_gqa and (tile_m % qhead_per_kvhead != 0):
        pack_gqa = False
    if grouped_output and not pack_gqa:
        raise ValueError("non-scattered output requires packed GQA tiles")

    lse_shape = (num_head, seqlen_q) if is_varlen else (batch_size, num_head, seqlen_q)
    requires_grad = q.requires_grad or k.requires_grad or v.requires_grad

    output_shape = (
        (batch_size, num_head_kv, seqlen_q, qhead_per_kvhead, head_dim_v)
        if grouped_output else (batch_size, seqlen_q, num_head, head_dim_v)
    )
    if is_varlen:
        output_shape = output_shape[1:]
    if out is None:
        out = torch.empty(output_shape, dtype=q.dtype, device=q.device)
    if tuple(out.shape) != output_shape or out.dtype != q.dtype or out.device != q.device:
        raise ValueError(f"out must have shape {output_shape} and Q dtype/device")
    if out.stride(-1) != 1:
        raise ValueError("output columns must be contiguous")
    if is_varlen and any(stride % head_dim_v for stride in out.stride()[:-1]):
        raise ValueError("packed output strides must be multiples of a full D row")

    if lse is None:
        lse = (
            torch.empty(lse_shape, dtype=torch.float32, device=q.device)
            if requires_grad or return_lse
            else None
        )

    dtype = torch2cute_dtype_map[q.dtype]
    if use_clc_scheduler is None:
        use_clc_scheduler = _get_use_clc_scheduler()
    if split_p_arrive is not None and split_p_arrive not in (32, 64, 96):
        raise ValueError("split_p_arrive must be one of 32, 64, 96, or None")
    if sync_first_p_release and split_p_arrive is None:
        raise ValueError("sync_first_p_release requires an explicit split boundary")

    current_stream = (
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        if is_fake_mode()
        else cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    )

    has_variable_block_nums = q2k_block_nums is not None

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        lse is None,
        tile_m,
        tile_n,
        pack_gqa,
        arch,
        use_2cta_instrs,
        use_clc_scheduler,
        fa_logging.get_fa_log_level(),
        has_variable_block_nums,
        allow_empty_block_nums and has_variable_block_nums,
        has_block_sizes,
        causal,
        implicit_fixed_support,
        is_paged_kv,
        split_p_arrive,
        sync_first_p_release,
        implicit_fixed_load_reduce,
        output_layout,
        is_varlen,
    )

    if compile_key not in _sm100_blk128_compile_cache:
        q_tensor, k_tensor, v_tensor, o_tensor = [
            to_cute_tensor(t) for t in (q, k, v, out)
        ]
        lse_tensor = to_cute_tensor(lse, assumed_align=4) if lse is not None else None
        block_index_tensor = to_cute_tensor(q2k_block_index)
        block_sizes_tensor = to_cute_tensor(block_sizes) if has_block_sizes else None
        block_nums_tensor = (
            to_cute_tensor(q2k_block_nums) if has_variable_block_nums else None
        )
        # A request row can start at any int32 offset in a packed page table.
        page_table_tensor = to_cute_tensor(page_table, assumed_align=4) if is_paged_kv else None

        # The descriptor has six contiguous fields for every request tile.
        # Keep that stride static while allowing the number of tiles to vary.
        packed_tiles_tensor = (
            cute.runtime.from_dlpack(
                packed_tiles.detach(), assumed_align=4, enable_tvm_ffi=True
            ).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
            if is_varlen else None
        )

        fa_fwd = FlashAttentionForwardSm100(
            head_dim,
            head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            pack_gqa=pack_gqa,
            m_block_size=tile_m,
            n_block_size=tile_n,
            is_persistent=True,
            use_2cta_instrs=use_2cta_instrs,
            use_clc_scheduler=use_clc_scheduler,
            allow_empty_block_nums=allow_empty_block_nums and has_variable_block_nums,
            has_block_sizes=has_block_sizes,
            is_causal=causal,
            implicit_fixed_support=implicit_fixed_support,
            is_paged_kv=is_paged_kv,
            split_p_arrive=split_p_arrive,
            sync_first_p_release=sync_first_p_release,
            implicit_fixed_load_reduce=implicit_fixed_load_reduce,
            grouped_output=grouped_output,
            is_varlen=is_varlen,
        )

        _sm100_blk128_compile_cache[compile_key] = cute.compile(
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            block_index_tensor,
            block_sizes_tensor,
            block_sparse_num,
            block_nums_tensor,
            page_table_tensor,
            seqlen_k,
            packed_tiles_tensor,
            current_stream,
            options="--enable-tvm-ffi",
        )

    if not is_fake_mode():
        with torch.cuda.nvtx.range("bsa_attn_sm100_blk128_fwd_kernel"):
            _sm100_blk128_compile_cache[compile_key](
                q.detach(),
                k.detach(),
                v.detach(),
                out.detach(),
                lse,
                softmax_scale,
                q2k_block_index.detach(),
                block_sizes.detach() if has_block_sizes else None,
                block_sparse_num,
                q2k_block_nums.detach() if has_variable_block_nums else None,
                page_table.detach() if is_paged_kv else None,
                seqlen_k,
                packed_tiles,
                current_stream,
            )

    return out, lse


def bsa_attn_fwd(*args, **kwargs):
    """Deprecated alias for bsa_attn_sm100_blk128_fwd.

    .. deprecated:: 0.6.18
        Use :func:`bsa_attn_sm100_blk128_fwd` instead.
    """
    import warnings

    warnings.warn(
        "bsa_attn_fwd is deprecated and will be removed in a future release. "
        "Use bsa_attn_sm100_blk128_fwd instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return bsa_attn_sm100_blk128_fwd(*args, **kwargs)
