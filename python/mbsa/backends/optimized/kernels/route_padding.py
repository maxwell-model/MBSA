# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Canonical padding for underfilled dynamic routes.

Preserve the source notices; this utility does not load an Index scorer.
See UPSTREAM.json for original/intermediate source identities and limitations.
"""
import torch
from .msa_block_sparse._vllm_compat import tl, triton

SPARSE_BLOCK_SIZE = 128

@triton.jit
def _canonicalize_dynamic_top16_padding_kernel(
    route_ptr,
    prefix_lens,
    total_q,
    stride_h,
    stride_q,
    stride_t,
    block_size: tl.constexpr,
):
    """Rewrite only underfilled Dynamic-16 tail slots to canonical ``-1``."""
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_q >= total_q:
        return
    valid_blocks = (tl.load(prefix_lens) + pid_q + block_size) // block_size
    prefix_count = min(8, valid_blocks)
    local_start = max(0, valid_blocks - 8)
    nonoverlap_local_start = max(local_start, prefix_count)
    fixed_count = prefix_count + valid_blocks - nonoverlap_local_start
    dynamic_count = min(32, valid_blocks) - fixed_count
    off_t = tl.arange(0, 16)
    base = route_ptr + pid_h * stride_h + pid_q * stride_q
    tl.store(base + off_t * stride_t, -1, mask=off_t >= dynamic_count)


@torch.no_grad()
def canonicalize_dynamic_top16_padding(
    route: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> None:
    """Rewrite unused Dynamic16 tail slots to canonical -1 in place.

    Args:
        route: Writable contiguous CUDA int32 [4,T,16] Dynamic routes; valid entries are preserved.
        prefix_lens: CUDA int32 [1] cached-prefix token count on route.device; this helper handles
            one request.

    Returns:
        None; enqueues a current-stream padding kernel. Retain route/prefix storage and order
        subsequent consumers.
    """
    if (
        route.ndim != 3
        or route.shape[0] != 4
        or route.shape[2] != 16
        or route.dtype != torch.int32
        or not route.is_cuda
        or not route.is_contiguous()
    ):
        raise ValueError("route must be contiguous CUDA int32 [4,Sq,16]")
    if (
        prefix_lens.shape != (1,)
        or prefix_lens.dtype != torch.int32
        or prefix_lens.device != route.device
    ):
        raise ValueError("canonical padding requires CUDA int32 prefix_lens [1]")
    total_q = int(route.shape[1])
    _canonicalize_dynamic_top16_padding_kernel[(total_q, 4)](
        route,
        prefix_lens,
        total_q,
        route.stride(0),
        route.stride(1),
        route.stride(2),
        SPARSE_BLOCK_SIZE,
        num_warps=1,
    )
