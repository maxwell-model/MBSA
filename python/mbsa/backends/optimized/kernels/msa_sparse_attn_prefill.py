# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""(3) block-sparse attention, PREFILL -- MSA CuTe-DSL CSR kernel.

Given a page selection, ``build_k2q_csr`` inverts
the per-query page selection into a key-to-query CSR schedule (one selected
128-token page == one physical vLLM KV block, read in place, nothing
gathered), then ``sparse_atten_func`` runs the CSR-scheduled block-sparse
attention.

Both functions are provided by the private MSA CuTe-DSL implementation.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

try:
    from .. import msa as fmha_sm100
    _MSA_OK = True
except (ImportError, RuntimeError):
    # See kernels/msa_indexer_score.py: a bare `import fmha_sm100` allocates
    # pinned CUDA host memory at module load time, so it raises RuntimeError
    # (not ImportError) on any machine without a live NVIDIA driver.
    fmha_sm100 = None
    _MSA_OK = False


def is_msa_available() -> bool:
    """Check whether the private sparse attention import succeeded.

    Returns:
        Boolean import availability; this does not compile a kernel or validate a particular
        GPU/shape.
    """
    return _MSA_OK


def allocate_csr_scratch(**kwargs):
    """Allocate caller-owned CSR scratch before capture or timed execution.

    Args:
        kwargs: Keyword arguments to allocate_k2q_csr_scratch:
            head_kv,total_q,total_rows,batch,max_kv_blocks,device and optional positive
            min_q_per_cta. This adapter defaults min_q_per_cta to 128.

    Returns:
        Four CUDA int32 scratch tensors sized for the requested geometry; caller retains them
        through queued use.
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
        max_seqlen_k: Maximum logical KV-token count across requests.
        max_seqlen_q: Maximum query-token count across requests.
        total_rows: ``sum(ceil(kv_len / block_size))``.
        num_kv_heads: Number of physical KV heads.
        qhead_per_kv: GQA group size (``num_qo_heads // num_kv_heads``).
        min_q_per_cta: Minimum query-token target per histogram CTA; default 128 in this adapter.
            Must be positive.
        buffers: Optional reusable CSR/schedule output bundle accepted by SparseK2qCsrBuilderSm100;
            overwritten in place. None allocates it.
        scratch: Optional four-tensor int32 scratch tuple from allocate_k2q_csr_scratch, matching
            current geometry/device. None lets the builder allocate scratch.
        local_top16_storage: Optional local selection storage used by fused merge/CSR construction;
            must match prefix_lens and candidate_partition_blocks. None consumes the final
            q2k_indices directly.
        prefix_lens: Optional CUDA int32 prefix/context lengths used to exclude prefix/local Fixed
            pages from dynamic candidates.
        candidate_partition_blocks: Optional KV-block span of each local candidate partition,
            required with local_top16_storage.
        local_top16_dynamic_only: Whether local candidates already exclude Fixed pages; default
            False.
        canonicalize_dynamic_padding: Whether to rewrite unused dynamic slots to canonical -1;
            default False.
        batch1_identity_rows: Enable direct logical-row mapping for a single request (default
            True).
        compact_batch1_schedule: Request compact batch-one work metadata (default False).

    Returns:
        ``(k2q_row_ptr, k2q_q_indices, schedule)``.

    The returned CSR buffers/schedule are live CUDA storage. Supplied buffers and scratch may be
    overwritten; order reuse on the current stream.
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
    initial_out: Optional[torch.Tensor] = None,
    initial_lse: Optional[torch.Tensor] = None,
    safe_fwd_sync: bool = True,
    shared_work_valid: bool = True,
    safe_partial_v_tail: bool = True,
    full_score_tiles: Optional[bool] = None,
    kernel_cfg: Any = None,
    phase: Optional[str] = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Block-sparse attention over the paged/dense KV, given a built CSR schedule.

    Args:
        q: ``[total_q, Hq, 128]`` BF16.
        k, v: dense ``[total_k, Hkv, 128]`` or paged
            ``[num_pages, Hkv, block_size, 128]`` BF16. Paged mode requires
            ``page_table`` and ``seqused_k`` and supports arbitrary physical
            page mappings.
        k2q_row_ptr, k2q_q_indices: CSR tensors returned by :func:`build_csr`.
        topk: pages selected per query (16 for this project).
        cu_seqlens_q: Contiguous CUDA int32 [B+1] packed query-token prefix sums.
        cu_seqlens_k: Contiguous CUDA int32 [B+1] logical KV-token prefix sums.
        max_seqlen_q: Maximum query-token length in the batch.
        max_seqlen_k: Maximum logical KV-token length in the batch.
        block_size: KV page size, 128 in this project.
        causal: Apply right-aligned token-level causal masking; default True.
        softmax_scale: ``1/sqrt(128)``.
        schedule: Sparse forward schedule from build_csr, matching the current routes and geometry.
        out: optional preallocated ``[total_q, Hq, 128]`` BF16 output buffer.
        partial_dtype: Per-block partial output dtype, BF16 by default; forwarded to the MSA
            interface.
        return_softmax_lse: Return (output,FP32 natural-log LSE) when True, otherwise output only;
            default False.
        page_table: Optional CUDA int32 [B,max_pages] logical-to-physical page IDs for paged K/V.
        seqused_k: Optional CUDA int32 [B] effective logical KV lengths for paged causal attention.
        workspace: Optional writable (O_partial,LSE_partial,LSE_out) CUDA buffers. Standard shapes
            are [topK,T,Hq,128], [topK,T,Hq], [T,Hq]; shuffled schedules use O
            [tile_capacity*128,128] and partial LSE [tile_capacity*128]. O uses partial_dtype; LSE
            buffers use FP32. None allocates storage. Split phases require persistent workspace and
            matching schedule.
        usable_sm_count: Maximum usable SM count; -1 lets the scheduler use all SMs.
        initial_out: Optional writable/readable BF16 Fixed attention state matching out, consumed
            by combine; may alias out in this low-level API. Must be paired with initial_lse.
        initial_lse: Optional FP32 Fixed natural-log LSE [1,Hq,T], supplied with initial_out.
            Inputs must be refreshed before reusing an aliased state.
        safe_fwd_sync: Compile-time safe forward barrier protocol (default True). Keep enabled for
            the supported synchronization path; disabling selects the alternate low-level protocol.
        shared_work_valid: Compile-time shared work-valid publication (default True), consumed by
            producer/consumer roles. Changes the internal control path, not route values.
        safe_partial_v_tail: Enable the safe partial-V tail path (default True); the kernel
            disables this option when its CLC variant supplies the corresponding protocol.
        full_score_tiles: Optional full-score-tile specialization override. None infers
            eligibility; True requires noncausal batch-one block-aligned geometry. False keeps
            general masking.
        kernel_cfg: optional compile-time MSA configuration used to force a
            benchmark variant. Production dispatch selects it when omitted.
        phase: optional ``"msa"`` or ``"combine"`` benchmark launch phase.
            Production leaves this unset and launches both kernels.

    Returns:
        ``[total_q, Hq, 128]`` BF16 attention output.

    Q/K/V storage in this adapter must be BF16. All launches use the current CUDA stream; retain
    supplied output/workspace storage through completion. Separate producer/combine phases require
    a persistent matching schedule/workspace and valid refreshed Fixed state.
    """
    if not _MSA_OK:
        raise ImportError("the MSA submodule (fmha_sm100) is required.")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
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
        **phase_args,
    )
