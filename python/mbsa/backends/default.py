# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original prefill backend factory and binding implementation."""

from __future__ import annotations
from collections.abc import Sequence
import torch
from ._lifecycle import PreparedOwnerBase


class PreparedOwner(PreparedOwnerBase):
    backend = "default"

    def _bind_paged(self, plan, index_q, index_k, q, k, v, *,
                    index_page_indices, page_table, out, index_q_scales=None):
        p = plan
        mx_router = getattr(p, "mxfp6_router", None)
        is_mx = getattr(p, "index_q_scales", None) is not None
        if is_mx:
            if index_q_scales is None:
                raise ValueError("MX rebind requires index_q_scales")
            if (
                index_q_scales.shape,
                index_q_scales.stride(),
                index_q_scales.dtype,
                index_q_scales.device,
            ) != (
                p.index_q_scales.shape,
                p.index_q_scales.stride(),
                p.index_q_scales.dtype,
                p.index_q_scales.device,
            ):
                raise ValueError(
                    "index_q_scales must retain prepared shape/stride/dtype/device"
                )
        elif index_q_scales is not None:
            raise ValueError("index_q_scales is only valid for an MX plan")
        if hasattr(p, "rebind"):
            p.rebind(
                index_q,
                index_k,
                q,
                k,
                v,
                index_page_indices=index_page_indices,
                page_table=page_table,
                out=out,
            )
            return
        a = p.attention
        if p.mega_graph_captured:
            raise RuntimeError("cannot rebind a captured mega graph")
        if k.ndim != 4:
            raise ValueError("rebind requires physical paged K/V")
        pairs = (
            ("index_q", index_q, p.index_q),
            ("index_k", index_k, p.index_k),
            ("q", q, a.q),
            ("k", k, a.k),
            ("v", v, a.v),
            ("index_page_indices", index_page_indices, p.index_page_indices),
            ("page_table", page_table, a.tile_plan.page_table),
            ("out", out, a.out),
        )
        for name, new, old in pairs:
            if (new.shape, new.stride(), new.dtype, new.device) != (
                old.shape,
                old.stride(),
                old.dtype,
                old.device,
            ):
                raise ValueError(
                    f"{name} must retain prepared shape/stride/dtype/device"
                )
        counts = tuple(
            (ke - kb + 127) // 128 for _, _, kb, ke in a.fixed_request_ranges
        )
        if index_page_indices.ndim == 2:
            rows = tuple(index_page_indices[i, :n] for i, n in enumerate(counts))
        elif index_page_indices.numel() == sum(counts):
            rows = index_page_indices.split(counts)
        else:
            padded = index_page_indices.view(a.batch, -1)
            rows = tuple(padded[i, :n] for i, n in enumerate(counts))
        if p.cutedsl_index_pages is not None:
            p.cutedsl_index_pages = rows[0]
            p.cutedsl_use_page_table = True
        if p.cutedsl_packed_pages is not None:
            if index_page_indices.ndim == 1 and index_page_indices.numel() == sum(
                counts
            ):
                p.cutedsl_packed_pages = index_page_indices
            else:
                offset = 0
                for row in rows:
                    p.cutedsl_packed_pages[offset : offset + row.numel()].copy_(row)
                    offset += row.numel()
        p.index_q, p.index_k, p.index_page_indices = (
            index_q,
            index_k,
            index_page_indices,
        )
        if is_mx:
            p.index_q_scales = index_q_scales
        if mx_router is not None:
            p.cutedsl_index_k = index_k
            mx_router.rebind(index_q, index_q_scales, index_k, p.cutedsl_packed_pages)
        elif p.cutedsl_index_k is not None:
            p.cutedsl_index_k = index_k.view(-1, 1, 128)
        a.q, a.k, a.v = q, k, v
        a.tile_plan.page_table = page_table
        if getattr(a, "dynamic_kvouter", None) is not None:
            a.dynamic_kvouter.rebind(a.compute_q, k, v, page_table)
        if getattr(a, "fixed_attention", None) is not None:
            a.fixed_attention.rebind(a.compute_q, k, v, page_table)
        if a.fixed_out is a.out:
            a.fixed_out = out
        if a.dynamic_out is a.out:
            a.dynamic_out = out
        a.out = out
        p._record_fixed_stream_tensors(inputs_only=True)


def prepare_sm100(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    index_q_scales: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    request_lens: Sequence[tuple[int, int]] | None = None,
    index_page_indices: torch.Tensor | None = None,
    page_table: torch.Tensor | None = None,
    logical_k_tokens: int | None = None,
    overlap_fixed_score: bool | None = None,
    partial_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
    check_device_inputs: bool = True,
    fixed_stream: torch.cuda.Stream | None = None,
):
    """Prepare the original Blackwell backend, retaining operand and workspace ownership.

    Preparation allocates storage and may compile kernels; call it outside CUDA Graph capture.
    Plans are single-in-flight: order all invocations and their CUDA work, or use independent
    plans. Input contents must remain valid until CUDA completes. Optimized direct operands/output
    require 16-byte base alignment, and MXFP6 Index K requires 128 bytes; conversion/staging
    buffers satisfy alignment where used. Read-only operands may alias; output may not.

    Args:
        index_q: CUDA Index queries [T,4,128], matching BF16/FP16/E4M3 Index K, or MXFP8 uint8/E4M3
            codes with index_q_scales. T is the packed query-token count.
        index_k: CUDA Index keys [Skv,1,128] or [pages,1,128,128] for floating formats; MXFP6 uses
            packed uint8 [pages,1,128,100] with page-separated codes/scales. Physical pages contain
            128 tokens.
        q: CUDA attention queries [T,32,128]. BF16 for BF16 K/V; the default Blackwell FP8 path
            also accepts BF16 or E4M3FN Q with E4M3FN K/V. Optimized converts FP8 operands to owned
            BF16 buffers per run.
        k: CUDA attention keys [Skv,4,128] or paged [pages,4,128,128]. Must match v in
            dtype/layout/device; BF16, or E4M3FN on Blackwell with unit scales.
        v: CUDA attention values with the same shape, dtype and device as k. Read-only until the
            queued invocation completes.
        index_q_scales: Optional CUDA int32 [T,>=4] MXFP8 scale words, four little-endian UE8M0
            bytes per head. Required for MXFP6 Index K; otherwise None.
        cu_seqlens_q: Optional contiguous CUDA int32 [B+1] query prefix sums, starting at zero and
            ending at T. Supply together with cu_seqlens_k for packed requests.
        cu_seqlens_k: Optional contiguous CUDA int32 [B+1] logical KV prefix sums for packed
            requests. Lengths include cached context; supply with cu_seqlens_q.
        request_lens: Optional host sequence of B (query_tokens, kv_tokens) pairs matching the
            prefix sums. Omission with packed prefixes can read device lengths back and
            synchronize. Optimized requires 0<Q<=8192 and Q<=KV<=1048576 per request.
        index_page_indices: Optional CUDA int32 logical-to-physical Index page IDs, compact 1-D
            active IDs or request rows [B,>=ceil(KV/128)]. Optimized binds active rows directly:
            one adjacent contiguous storage span with a 16-byte base alignment for unpacked/MX
            or 4-byte base alignment for packed floating Index. Individual compact request rows
            need not be 16-byte aligned. Gaps/strides are rejected. None creates the backend
            default mapping.
        page_table: Optional contiguous CUDA int32 [B,>=ceil(KV/128)] attention logical-to-physical
            page table, required for paged K/V. Each active ID addresses a physical K/V page.
        logical_k_tokens: Optional logical KV token count for one unpacked request, excluding
            physical page padding. Cannot be supplied with packed prefix sums; None infers
            contiguous K/V extent; optimized paged K/V requires an explicit logical count or packed
            prefix sums.
        overlap_fixed_score: Whether Fixed attention runs beside Index on a side stream, joined
            before Dynamic attention. None enables overlap for batch one on default; optimized also
            enables it for packed batches with maximum KV<=8192.
        partial_dtype: BF16 or torch.float8_e4m3fn partial-output storage. None reads
            MBSA_FP8_PARTIAL_OUT (0=BF16, the default; 1=E4M3FN). Final output remains BF16.
        out: Optional writable CUDA BF16 [T,32,128] output. Its byte span must be disjoint from all
            inputs. None allocates plan-owned output; supplied storage stays caller-owned.
        check_device_inputs: If True (default), validate device metadata values; these checks may
            synchronize. False requires the caller to provide valid in-range metadata. Host
            shape/layout validation remains active.
        fixed_stream: Optional CUDA stream on the operand device for overlapping Fixed attention.
            None lets preparation create/select a stream when overlap is enabled.

    Returns:
        CutedslPipeline with caller-owned out when supplied and plan-owned LSE/routes.
    """
    from ..api import CutedslPipeline, _resolve_partial_dtype
    partial_dtype = _resolve_partial_dtype(partial_dtype)
    if cu_seqlens_q is None:
        if k.ndim == 4 and logical_k_tokens is None:
            raise ValueError("paged KV requires logical_k_tokens")
        kv_len = logical_k_tokens if logical_k_tokens is not None else k.shape[0]
        q_lens, kv_lens = (q.shape[0],), (kv_len,)
    else:
        if request_lens is None:
            raise ValueError("packed preparation requires host request_lens")
        q_lens = tuple(int(qn) for qn, _ in request_lens)
        kv_lens = tuple(int(kn) for _, kn in request_lens)
    if not q_lens or len(q_lens) != len(kv_lens):
        raise ValueError("Q/KV prefix sums must describe the same non-empty batch")
    if any(
        not 0 < qn <= min(kn, 8192) or kn > 1048576 for qn, kn in zip(q_lens, kv_lens)
    ):
        raise ValueError("each request requires 0 < Q <= 8192 and Q <= KV <= 1M")
    from ..cute.prefill import prepare_pipeline, stage_int32
    from ..cute.plan import FwdTilePlan, PrefillPolicy

    request_blocks = tuple((n + 127) // 128 for n in kv_lens)
    index_pages_identity = None
    if index_page_indices is None:
        index_page_indices = torch.arange(
            sum(request_blocks), dtype=torch.int32, device=q.device
        )
        index_pages_identity = len(q_lens) == 1
    if index_k.ndim == 3 and index_k.shape[0] % 128:
        if len(q_lens) != 1:
            raise ValueError("packed Index-K requires physical pages")
        padded = torch.zeros(
            (request_blocks[0] * 128, 1, 128),
            dtype=index_k.dtype,
            device=index_k.device,
        )
        padded[: index_k.shape[0]].copy_(index_k)
        index_k = padded.view(-1, 1, 128, 128)
    policy = (
        PrefillPolicy.ALIGNED_PAGED if k.ndim == 4 else PrefillPolicy.ALIGNED_CONTIGUOUS
    )
    if policy is PrefillPolicy.ALIGNED_PAGED and page_table is None:
        raise ValueError("paged KV requires page_table")
    if policy is PrefillPolicy.ALIGNED_CONTIGUOUS and page_table is not None:
        raise ValueError("page_table is only valid for physical KV pages")
    if page_table is None:
        page_table = torch.empty((len(q_lens), 0), dtype=torch.int32, device=q.device)
    offsets = tuple(kn - qn for qn, kn in zip(q_lens, kv_lens))
    valid_kv_tokens, prefix_lens = stage_int32(
        q.device, (kv_lens, (len(kv_lens),)), (offsets, (len(offsets),))
    )
    tile_plan = FwdTilePlan(
        dynamic_route=torch.full(
            (4, q.shape[0], 16), -1, dtype=torch.int32, device=q.device
        ),
        page_table=page_table,
        valid_kv_tokens=valid_kv_tokens,
        q_offset_tokens=offsets[0] if len(offsets) == 1 else offsets,
        policy=policy,
    )
    if overlap_fixed_score is None and len(q_lens) == 1:
        overlap_fixed_score = True
    plan = prepare_pipeline(
        index_q=index_q,
        index_k=index_k,
        index_q_scales=index_q_scales,
        index_page_indices=index_page_indices,
        candidate_storage=torch.empty(
            (4, 32, q.shape[0]), dtype=torch.float32, device=q.device
        ),
        prefix_lens=prefix_lens,
        q=q,
        k=k,
        v=v,
        tile_plan=tile_plan,
        out=torch.empty_like(q, dtype=torch.bfloat16) if out is None else out,
        partial_dtype=partial_dtype,
        overlap_fixed_score=overlap_fixed_score,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        request_lens=None if cu_seqlens_q is None else tuple(zip(q_lens, kv_lens)),
        check_device_inputs=check_device_inputs,
        index_pages_identity=index_pages_identity,
        fixed_stream=fixed_stream,
        _static_identity_mask=(cu_seqlens_q is None and
                               policy == PrefillPolicy.ALIGNED_CONTIGUOUS),
        _lean192_early=(cu_seqlens_q is None and index_pages_identity is True
                        and policy == PrefillPolicy.ALIGNED_CONTIGUOUS
                        and index_k.ndim == 3),
    )
    return CutedslPipeline(plan, PreparedOwner(plan, tuple(zip(q_lens, kv_lens))))


def prepare_sm90(
    index_q,
    index_k,
    q,
    k,
    v,
    *,
    cu_seqlens_q,
    cu_seqlens_k,
    index_page_indices,
    page_table,
    logical_k_tokens,
    overlap_fixed_score,
    partial_dtype,
    out,
    request_lens,
):
    """Prepare the original Hopper BF16 attention backend outside capture.

    Args:
        index_q: CUDA Index queries [T,4,128], matching BF16/FP16/E4M3 Index K, or MXFP8 uint8/E4M3
            codes with index_q_scales. T is the packed query-token count.
        index_k: CUDA Index keys [Skv,1,128] or [pages,1,128,128] for floating formats; MXFP6 uses
            packed uint8 [pages,1,128,100] with page-separated codes/scales. Physical pages contain
            128 tokens.
        q: CUDA BF16 attention Q [T,32,128]; Hopper does not support FP8 attention.
        k: CUDA BF16 dense or paged attention K with 4 heads and head dimension 128.
        v: CUDA BF16 V matching K geometry and device.
        cu_seqlens_q: Optional contiguous CUDA int32 [B+1] query prefix sums, starting at zero and
            ending at T. Supply together with cu_seqlens_k for packed requests.
        cu_seqlens_k: Optional contiguous CUDA int32 [B+1] logical KV prefix sums for packed
            requests. Lengths include cached context; supply with cu_seqlens_q.
        index_page_indices: Optional CUDA int32 logical-to-physical Index page IDs, compact 1-D
            active IDs or request rows [B,>=ceil(KV/128)]. Optimized binds active rows directly:
            one adjacent contiguous storage span with a 16-byte base alignment for unpacked/MX
            or 4-byte base alignment for packed floating Index. Individual compact request rows
            need not be 16-byte aligned. Gaps/strides are rejected. None creates the backend
            default mapping.
        page_table: Optional contiguous CUDA int32 [B,>=ceil(KV/128)] attention logical-to-physical
            page table, required for paged K/V. Each active ID addresses a physical K/V page.
        logical_k_tokens: Optional logical KV token count for one unpacked request, excluding
            physical page padding. Cannot be supplied with packed prefix sums; None infers
            contiguous K/V extent; optimized paged K/V requires an explicit logical count or packed
            prefix sums.
        overlap_fixed_score: Whether Fixed attention runs beside Index on a side stream, joined
            before Dynamic attention. None enables overlap for batch one on default; optimized also
            enables it for packed batches with maximum KV<=8192.
        partial_dtype: BF16 or torch.float8_e4m3fn partial-output storage. None reads
            MBSA_FP8_PARTIAL_OUT (0=BF16, the default; 1=E4M3FN). Final output remains BF16.
        out: Optional writable CUDA BF16 [T,32,128] output. Its byte span must be disjoint from all
            inputs. None allocates plan-owned output; supplied storage stays caller-owned.
        request_lens: Optional host sequence of B (query_tokens, kv_tokens) pairs matching the
            prefix sums. Omission with packed prefixes can read device lengths back and
            synchronize. Optimized requires 0<Q<=8192 and Q<=KV<=1048576 per request.

    Returns:
        CutedslPipeline owning workspace and live LSE/routes, with optional caller-owned output.
    """
    from ..api import CutedslPipeline, _host_request_lens, _resolve_partial_dtype
    if (cu_seqlens_q is None) != (cu_seqlens_k is None):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be provided together")
    if cu_seqlens_q is None:
        if k.ndim == 4 and logical_k_tokens is None:
            raise ValueError("paged KV requires logical_k_tokens")
        kv_len = logical_k_tokens if logical_k_tokens is not None else k.shape[0]
        q_lens, kv_lens = (q.shape[0],), (kv_len,)
    else:
        if logical_k_tokens is not None:
            raise ValueError("packed lengths are owned by cu_seqlens_k")
        if request_lens is None:
            request_lens = _host_request_lens(cu_seqlens_q, cu_seqlens_k)
        q_lens = tuple(int(qn) for qn, _ in request_lens)
        kv_lens = tuple(int(kn) for _, kn in request_lens)
    if not q_lens or len(q_lens) != len(kv_lens):
        raise ValueError("Q/KV prefix sums must describe the same non-empty batch")
    if any(
        not 0 < qn <= min(kn, 8192) or kn > 1048576 for qn, kn in zip(q_lens, kv_lens)
    ):
        raise ValueError("each request requires 0 < Q <= 8192 and Q <= KV <= 1M")
    if (k.ndim == 4) != (page_table is not None):
        raise ValueError(
            "page_table is required for, and only valid with, physical KV pages"
        )
    request_blocks = tuple((n + 127) // 128 for n in kv_lens)
    if index_page_indices is None:
        index_page_indices = torch.arange(
            sum(request_blocks), dtype=torch.int32, device=q.device
        )
    if index_k.ndim == 3 and index_k.shape[0] % 128:
        if len(q_lens) != 1:
            raise ValueError("packed Index-K requires physical pages")
        padded = torch.zeros(
            (request_blocks[0] * 128, 1, 128),
            dtype=index_k.dtype,
            device=index_k.device,
        )
        padded[: index_k.shape[0]].copy_(index_k)
        index_k = padded
    from ..cute.sm90 import prepare_sm90_pipeline

    plan = prepare_sm90_pipeline(
        index_q,
        index_k,
        q,
        k,
        v,
        q_lens=q_lens,
        kv_lens=kv_lens,
        index_page_indices=index_page_indices,
        page_table=page_table,
        partial_dtype=_resolve_partial_dtype(partial_dtype),
        overlap_fixed_score=overlap_fixed_score,
        out=out,
    )
    return CutedslPipeline(plan, PreparedOwner(plan, tuple(zip(q_lens, kv_lens))))
