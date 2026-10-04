# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Host validation and GPU q2k -> k2q CSR construction for sparse forward.

Input: [Hkv, total_q, topK] batch-local KV block selections, padded with -1.
Output: int32 k2q row pointers/query indices and optional LPT schedule.
"""

from typing import Optional

import torch

from .src.sm100.prepare_k2q_csr import SparseK2qCsrBuilderSm100


def _validate_cu_seqlens(cu_seqlens: torch.Tensor, *, name: str) -> None:
    if cu_seqlens.dtype != torch.int32:
        raise TypeError(f"{name} must be torch.int32, got {cu_seqlens.dtype}")
    if cu_seqlens.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got shape {tuple(cu_seqlens.shape)}")
    if cu_seqlens.numel() < 1:
        raise ValueError(f"{name} must have at least one element")
    if not cu_seqlens.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


_K2Q_CSR_BUILDER = SparseK2qCsrBuilderSm100()


def build_k2q_csr(
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    kv_block_size: int,
    *,
    total_k: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    total_rows: Optional[int] = None,
    qhead_per_kv: int = 1,
    min_q_per_cta: int = 256,
    return_schedule: bool = False,
    buffers=None,
    scratch=None,
    local_top16_storage: torch.Tensor | None = None,
    prefix_lens: torch.Tensor | None = None,
    candidate_partition_blocks: int | None = None,
    local_top16_dynamic_only: bool = False,
    canonicalize_dynamic_padding: bool = False,
    batch1_identity_rows: bool = True,
    compact_batch1_schedule: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, object]:
    """Build the public k2q CSR reverse index on GPU.

    Runtime construction does not read device-side ``cu_seqlens`` on the host,
    so callers must provide size hints such as ``total_k`` from already-known
    tensor shapes.

    Parameters
    ----------
    q2k_indices : torch.Tensor
        Shape ``[head_kv, total_q, topK]``, dtype int32, contiguous.  Values are
        batch-local KV block indices with trailing ``-1`` padding.
    cu_seqlens_q : torch.Tensor
        Shape ``[batch_size + 1]``, dtype int32.  Prefix sums of Q lengths.
    cu_seqlens_k : torch.Tensor
        Shape ``[batch_size + 1]``, dtype int32.  Prefix sums of KV lengths.
    kv_block_size : int
        Number of KV tokens per sparse block.
    total_k : int
        Total KV token count.  Required; normally ``k.shape[0]`` for dense KV
        or ``sum(kv_segment_lens)`` for paged KV.
    max_seqlen_k : int, optional
        Maximum KV sequence length.  Passing this avoids recomputing a bound.
    max_seqlen_q : int, optional
        Maximum Q sequence length.
    total_rows : int, optional
        Total number of packed KV-block rows across the batch.  If omitted,
        the builder derives it from ``cu_seqlens_k`` and ``kv_block_size``.
    qhead_per_kv : int, optional
        Number of Q heads per KV head under GQA.
    return_schedule : bool, optional
        If True, also return the sparse forward schedule object produced by the
        SM100 builder.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor] or tuple[torch.Tensor, torch.Tensor, object]
        ``(k2q_row_ptr, k2q_q_indices)`` or
        ``(k2q_row_ptr, k2q_q_indices, schedule)``.  CSR tensors are int32 on
        the same CUDA device as ``q2k_indices``.
    """
    if total_k is None:
        raise ValueError("build_k2q_csr requires total_k from k.shape[0]")
    if kv_block_size <= 0:
        raise ValueError(f"kv_block_size must be > 0, got {kv_block_size}")
    if q2k_indices.dtype != torch.int32:
        raise TypeError(f"q2k_indices must be torch.int32, got {q2k_indices.dtype}")
    if q2k_indices.ndim != 3:
        raise ValueError(f"q2k_indices must be rank-3, got shape {tuple(q2k_indices.shape)}")
    if not q2k_indices.is_contiguous():
        raise ValueError("q2k_indices must be contiguous with layout [head_kv, total_q, topK]")
    _validate_cu_seqlens(cu_seqlens_q, name="cu_seqlens_q")
    _validate_cu_seqlens(cu_seqlens_k, name="cu_seqlens_k")
    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same shape [B + 1]")
    if q2k_indices.device != cu_seqlens_q.device or q2k_indices.device != cu_seqlens_k.device:
        raise ValueError("q2k_indices, cu_seqlens_q, and cu_seqlens_k must be on the same device")
    return _K2Q_CSR_BUILDER(
        q2k_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        total_k=int(total_k),
        blk_kv=int(kv_block_size),
        max_seqlen_k=max_seqlen_k,
        max_seqlen_q=max_seqlen_q,
        total_rows=total_rows,
        qhead_per_kv=qhead_per_kv,
        min_q_per_cta=min_q_per_cta,
        return_schedule=return_schedule,
        buffers=buffers,
        scratch=scratch,
        local_top16_storage=local_top16_storage,
        prefix_lens=prefix_lens,
        candidate_partition_blocks=candidate_partition_blocks,
        local_top16_dynamic_only=local_top16_dynamic_only,
        canonicalize_dynamic_padding=canonicalize_dynamic_padding,
        batch1_identity_rows=batch1_identity_rows,
        compact_batch1_schedule=compact_batch1_schedule,
    )
