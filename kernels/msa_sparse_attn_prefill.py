# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""③ block-sparse attention, PREFILL — MSA CuTe-DSL CSR kernel.

Given a page selection (from ``msa_topk_select.select_prefill``), this is the
same code path Compass's production serving takes: ``build_k2q_csr`` inverts
the per-query page selection into a key-to-query CSR schedule (one selected
128-token page == one physical vLLM KV block, read in place, nothing
gathered), then ``sparse_atten_func`` runs the CSR-scheduled block-sparse
attention.

Both functions are provided by the MSA CuTe-DSL submodule at
``third_party/msa`` (package ``fmha_sm100``); there is no Triton alternative
for prefill attention in this MSA checkout.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from msa_block_sparse.minimax_sparse_attn_triton import (
    minimax_m3_sparse_attn_prefill,
)

try:
    import fmha_sm100

    _MSA_OK = True
except (ImportError, RuntimeError):
    # See kernels/msa_indexer_score.py: a bare `import fmha_sm100` allocates
    # pinned CUDA host memory at module load time, so it raises RuntimeError
    # (not ImportError) on any machine without a live NVIDIA driver.
    fmha_sm100 = None
    _MSA_OK = False


def is_msa_available() -> bool:
    return _MSA_OK


def allocate_csr_scratch(**kwargs):
    """Allocate exact-shape native CSR scratch before a timed window.

    The provisional Top-32 adapter uses 128-query CSR partitions by default.
    Keep the vendored MSA default unchanged so callers outside this adapter
    retain their upstream-compatible 256-query policy.
    """
    if not _MSA_OK:
        raise ImportError("the MSA submodule (fmha_sm100) is required.")
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
            for provisional Top-32, or one shared row for C0 compatibility.
        cu_seqlens_q, cu_seqlens_k: ``[batch + 1]`` int32 prefix sums.
        block_size: KV page size (128 in this project).
        total_k: ``sum(kv_lens)`` -- passed explicitly so the builder never
            reads ``cu_seqlens_k`` back on the host.
        total_rows: ``sum(ceil(kv_len / block_size))``.
        num_kv_heads: Number of physical KV heads (4 for Compass-v4).
        qhead_per_kv: GQA group size (``num_qo_heads // num_kv_heads``).

    Returns:
        ``(k2q_row_ptr, k2q_q_indices, schedule)``.
    """
    if not _MSA_OK:
        raise ImportError("the MSA submodule (fmha_sm100) is required.")
    if q2k_indices.ndim != 3 or q2k_indices.shape[0] not in (1, num_kv_heads):
        raise ValueError(
            "q2k_indices must have one shared or one per-KV-group row, got "
            f"{tuple(q2k_indices.shape)}"
        )
    if q2k_indices.dtype != torch.int32 or not q2k_indices.is_cuda:
        raise ValueError("q2k_indices must be CUDA int32")
    # The public MSA builder requires contiguous Hkv-major input. B0 already
    # produces this layout. C0's single shared row is materialized only in the
    # historical compatibility path.
    q2k_for_csr = (
        q2k_indices
        if q2k_indices.shape[0] == num_kv_heads
        else q2k_indices.expand(num_kv_heads, -1, -1).contiguous()
    )
    return fmha_sm100.build_k2q_csr(
        q2k_for_csr,
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
    q2k_indices: Optional[torch.Tensor] = None,
    prefix_lens: Optional[torch.Tensor] = None,
    lse_out: Optional[torch.Tensor] = None,
    initial_out: Optional[torch.Tensor] = None,
    initial_lse: Optional[torch.Tensor] = None,
    direct_route_partial_page: bool = False,
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
    if not _MSA_OK:
        raise ImportError("the MSA submodule (fmha_sm100) is required.")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("MSA prefill sparse attention in this project is BF16-only.")
    if direct_route_partial_page:
        if page_table is None or seqused_k is None:
            raise ValueError("partial-page direct route requires page_table and seqused_k")
        if q2k_indices is None or prefix_lens is None:
            raise ValueError("partial-page direct route requires q2k_indices and prefix_lens")
        if not return_softmax_lse:
            raise ValueError("partial-page B0 dispatch always returns output and LSE")
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError("partial-page direct route requires paged [P,Hkv,128,D] K/V")
        if out is None:
            out = torch.empty_like(q)
        if lse_out is None:
            lse_out = torch.empty(
                (q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device
            )
        minimax_m3_sparse_attn_prefill(
            q,
            k,
            v,
            q2k_indices,
            page_table,
            cu_seqlens_q,
            seqused_k,
            prefix_lens,
            max_seqlen_q,
            k.shape[1],
            softmax_scale,
            out,
            lse_out,
        )
        return out, lse_out
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
