# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Modified for MBSA: package integration; see root NOTICE.
"""BF16 Dynamic MSA and native KV-centric CSR adapters."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from . import sm100_sparse as fmha_sm100


def allocate_csr_scratch(**kwargs):
    """Allocate exact-shape native CSR scratch before a timed window.

    The dynamic-route adapter uses 128-query CSR partitions by default.
    Keep the vendored MSA default unchanged so callers outside this adapter
    retain their upstream-compatible 256-query policy.
    """
    kwargs.setdefault("min_q_per_cta", 128)
    return fmha_sm100.allocate_k2q_csr_scratch(**kwargs)


def build_csr(
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_size: int,
    *,
    total_k: int,
    max_seqlen_k: int,
    max_seqlen_q: int,
    total_rows: int,
    num_kv_heads: int,
    qhead_per_kv: int,
    min_q_per_cta: int = 128,
    buffers: Any = None,
    scratch: Any = None,
    local_top16_storage: Optional[torch.Tensor] = None,
    prefix_lens: Optional[torch.Tensor] = None,
    candidate_partition_blocks: Optional[int] = None,
    local_top16_dynamic_only: bool = False,
    canonicalize_dynamic_padding: bool = False,
    batch1_identity_rows: bool = True,
    compact_batch1_schedule: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    """Build the key-to-query CSR schedule. Not part of the timed attention kernel.

    Args:
        q2k_indices: Hkv-major ``[Hkv, total_q, topk]`` int32 page selection
            for dynamic attention.
        cu_seqlens_q, cu_seqlens_k: ``[batch + 1]`` int32 prefix sums.
        block_size: KV page size (128 in this project).
        total_k: ``sum(kv_lens)`` -- passed explicitly so the builder never
            reads ``cu_seqlens_k`` back on the host.
        total_rows: ``sum(ceil(kv_len / block_size))``.
        num_kv_heads: Number of physical KV heads (4 for Maxwell Block Sparse Attention).
        qhead_per_kv: GQA group size (``num_qo_heads // num_kv_heads``).

    Returns:
        ``(k2q_row_ptr, k2q_q_indices, schedule)``.
    """
    if q2k_indices.ndim != 3 or q2k_indices.shape[0] != num_kv_heads:
        raise ValueError(
            "q2k_indices must have one row per KV group, got "
            f"{tuple(q2k_indices.shape)}"
        )
    if q2k_indices.dtype != torch.int32 or not q2k_indices.is_cuda:
        raise ValueError("q2k_indices must be CUDA int32")
    return fmha_sm100.build_k2q_csr(
        q2k_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        block_size,
        total_k=total_k,
        max_seqlen_k=max_seqlen_k,
        max_seqlen_q=max_seqlen_q,
        total_rows=total_rows,
        qhead_per_kv=qhead_per_kv,
        min_q_per_cta=min_q_per_cta,
        return_schedule=True,
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


def sparse_gqa_stage2_attn_fwd_msa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    topk: int,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size: int = 128,
    causal: bool = True,
    softmax_scale: float,
    schedule: Any,
    out: Optional[torch.Tensor] = None,
    partial_dtype: torch.dtype = torch.bfloat16,
    return_softmax_lse: bool = False,
    page_table: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    workspace: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    usable_sm_count: int = -1,
    initial_out: Optional[torch.Tensor] = None,
    initial_lse: Optional[torch.Tensor] = None,
    safe_fwd_sync: bool = True,
    shared_work_valid: bool = True,
    safe_partial_v_tail: bool = True,
    full_score_tiles: Optional[bool] = None,
    kernel_cfg: Any = None,
    phase: Optional[str] = None,
    last_cta_finalize: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Block-sparse attention over the paged/dense KV, given a built CSR schedule.

    Args:
        q: ``[total_q, Hq, 128]`` BF16.
        k, v: dense ``[total_k, Hkv, 128]`` or paged
            ``[num_pages, Hkv, block_size, 128]`` BF16. Paged mode requires
            ``page_table`` and ``seqused_k`` and supports arbitrary physical
            page mappings.
        k2q_row_ptr, k2q_q_indices, schedule: return of :func:`build_csr`.
        topk: pages selected per query (16 for this project).
        block_size: KV page size, 128 in this project.
        softmax_scale: ``1/sqrt(128)``.
        out: optional preallocated ``[total_q, Hq, 128]`` BF16 output buffer.
        kernel_cfg: optional compile-time MSA configuration used to force a
            benchmark variant. Production dispatch selects it when omitted.
        phase: optional ``"msa"`` or ``"combine"`` benchmark launch phase.
            Production leaves this unset and launches both kernels.

    Returns:
        ``[total_q, Hq, 128]`` BF16 attention output.
    """
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
    ):
        raise ValueError("MSA prefill sparse attention in this project is BF16-only.")
    phase_args = {}
    if phase is not None:
        phase_args["phase"] = fmha_sm100.SparseForwardPhase(phase)
    return fmha_sm100.sparse_atten_func(
        q,
        k,
        v,
        k2q_row_ptr,
        k2q_q_indices,
        topk,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        blk_kv=block_size,
        causal=causal,
        softmax_scale=softmax_scale,
        schedule=schedule,
        out=out,
        partial_dtype=partial_dtype,
        return_softmax_lse=return_softmax_lse,
        initial_out=initial_out,
        initial_lse=initial_lse,
        page_table=page_table,
        seqused_k=seqused_k,
        workspace=workspace,
        usable_SM_count=usable_sm_count,
        safe_fwd_sync=safe_fwd_sync,
        shared_work_valid=shared_work_valid,
        safe_partial_v_tail=safe_partial_v_tail,
        full_score_tiles=full_score_tiles,
        kernel_cfg=kernel_cfg,
        last_cta_finalize=last_cta_finalize,
        **phase_args,
    )
