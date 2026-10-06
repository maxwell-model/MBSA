# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
# Modified for MBSA: package integration; see root NOTICE.

"""CUDA C++ extension for the q2k -> k2q CSR builder.

On import this module loads the prebuilt ``mbsa_build_k2q_csr_ext`` next to
it, or compiles ``build_k2q_csr.cu`` via ``torch.utils.cpp_extension.load``
(cached in ``~/.cache/torch_extensions/``), and exposes ``run_build_k2q_csr``.

The kernel pipeline is tuned and verified for SM100; other
architectures are not supported.
"""

from __future__ import annotations

import os

import torch
from torch.utils.cpp_extension import load

from ....._cuda_jit import (
    cuda_arch_flag,
    load_extension,
    nvidia_wheel_include_dirs,
    template_stub_flags,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_THIS_DIR, "build_k2q_csr.cu")


_extra_cflags = ["-O3"]
_extra_cuda_cflags = [
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    cuda_arch_flag(),
    "--ptxas-options=-v",
    "--expt-relaxed-constexpr",
    "-I/usr/local/cuda/include/cccl",
    *template_stub_flags(),
]

_ext = load_extension(
    __package__,
    "mbsa_build_k2q_csr_ext",
    lambda: load(
        name="mbsa_build_k2q_csr_ext",
        sources=[_SRC],
        extra_cflags=_extra_cflags,
        extra_cuda_cflags=_extra_cuda_cflags,
        extra_ldflags=["-lcuda"],
        extra_include_paths=nvidia_wheel_include_dirs(),
        verbose=False,
    ),
)


def run_build_k2q_csr(
    q2k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    row_ptr: torch.Tensor,
    q_idx: torch.Tensor,
    topk: int,
    blk_kv: int,
    total_rows: int,
    max_kv_blocks: int,
    min_q_per_cta: int = 256,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """In-place fill of ``row_ptr`` and ``q_idx``.

    Args:
      q2k:           int32 [H, total_q, topK] contiguous (CUDA).
      cu_seqlens_q:  int32 [B+1] contiguous (CUDA).
      cu_seqlens_k:  int32 [B+1] contiguous (CUDA).
      row_ptr:       int32 [H, total_rows + 1] CUDA, written in place.
      q_idx:         int32 [H, total_q * topK] CUDA, written in place
                     (trailing slots set to -1).
      topk:          must be in {4, 8, 16, 32}.
      blk_kv:        must be one of {32, 64, 128}.
      total_rows:    sum over batches of ceil(seqlen_k / blk_kv).
      max_kv_blocks: max over batches of ceil(seqlen_k / blk_kv); upper bound
                     used to size the row_map workspace and clamp valid kv ids.
    """
    scratch_args = () if scratch is None else tuple(scratch)
    if len(scratch_args) not in (0, 4):
        raise ValueError("CSR scratch must contain four tensors")
    _ext.run_build_k2q_csr(
        q2k,
        cu_seqlens_q,
        cu_seqlens_k,
        row_ptr,
        q_idx,
        int(topk),
        int(blk_kv),
        int(total_rows),
        int(max_kv_blocks),
        int(min_q_per_cta),
        *scratch_args,
    )


def run_build_k2q_csr_with_schedule(
    q2k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    row_ptr: torch.Tensor,
    q_idx: torch.Tensor,
    scheduler_metadata: torch.Tensor,
    work_count: torch.Tensor,
    qsplit_idx: torch.Tensor,
    split_counts: torch.Tensor,
    topk: int,
    blk_kv: int,
    total_rows: int,
    max_kv_blocks: int,
    target_q_per_cta: int,
    work_capacity: int,
    max_seqlen_q: int,
    min_q_per_cta: int = 256,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """In-place fill of CSR plus fused sparse attention schedule metadata."""
    scratch_args = () if scratch is None else tuple(scratch)
    if len(scratch_args) not in (0, 4):
        raise ValueError("CSR scratch must contain four tensors")
    _ext.run_build_k2q_csr_with_schedule(
        q2k,
        cu_seqlens_q,
        cu_seqlens_k,
        row_ptr,
        q_idx,
        scheduler_metadata,
        work_count,
        qsplit_idx,
        split_counts,
        int(topk),
        int(blk_kv),
        int(total_rows),
        int(max_kv_blocks),
        int(target_q_per_cta),
        int(work_capacity),
        int(max_seqlen_q),
        int(min_q_per_cta),
        *scratch_args,
    )


def run_build_k2q_csr_with_schedule_from_local_top16(
    local_candidate_storage: torch.Tensor,
    prefix_lens: torch.Tensor,
    q2k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    row_ptr: torch.Tensor,
    q_idx: torch.Tensor,
    scheduler_metadata: torch.Tensor,
    work_count: torch.Tensor,
    qsplit_idx: torch.Tensor,
    split_counts: torch.Tensor,
    topk: int,
    blk_kv: int,
    total_rows: int,
    max_kv_blocks: int,
    candidate_partition_blocks: int,
    target_q_per_cta: int,
    work_capacity: int,
    max_seqlen_q: int,
    min_q_per_cta: int = 256,
    dynamic_only: bool = False,
    canonicalize_dynamic_padding: bool = False,
    batch1_identity_rows: bool = True,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Build route and CSR directly from one exact local Top-16 list."""
    scratch_args = () if scratch is None else tuple(scratch)
    if len(scratch_args) not in (0, 4):
        raise ValueError("CSR scratch must contain four tensors")
    _ext.run_build_k2q_csr_with_schedule_from_local_top16(
        local_candidate_storage,
        prefix_lens,
        q2k,
        cu_seqlens_q,
        cu_seqlens_k,
        row_ptr,
        q_idx,
        scheduler_metadata,
        work_count,
        qsplit_idx,
        split_counts,
        int(topk),
        int(blk_kv),
        int(total_rows),
        int(max_kv_blocks),
        int(candidate_partition_blocks),
        int(target_q_per_cta),
        int(work_capacity),
        int(max_seqlen_q),
        int(min_q_per_cta),
        bool(dynamic_only),
        bool(canonicalize_dynamic_padding),
        bool(batch1_identity_rows),
        *scratch_args,
    )


def run_reorder_schedule_qcount_bucket16(
    source: torch.Tensor,
    destination: torch.Tensor,
    work_count: torch.Tensor,
    target_q_per_cta: int,
    route_ready: torch.Tensor | None = None,
) -> None:
    """Reorder work records and optionally release-publish route readiness."""
    args = (
        source,
        destination,
        work_count,
        int(target_q_per_cta),
    )
    if route_ready is None:
        _ext.run_reorder_schedule_qcount_bucket16(*args)
    else:
        _ext.run_reorder_schedule_qcount_bucket16(*args, route_ready)


def run_publish_route_ready(route_ready: torch.Tensor) -> None:
    """Release-publish route metadata on the current CUDA stream."""
    _ext.run_publish_route_ready(route_ready)


def is_supported(topk: int, blk_kv: int) -> bool:
    return int(topk) in (4, 8, 16, 32) and int(blk_kv) in (32, 64, 128)


__all__ = [
    "run_build_k2q_csr",
    "run_build_k2q_csr_with_schedule",
    "run_build_k2q_csr_with_schedule_from_local_top16",
    "run_reorder_schedule_qcount_bucket16",
    "run_publish_route_ready",
    "is_supported",
]
