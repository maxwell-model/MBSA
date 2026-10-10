# SPDX-License-Identifier: Apache-2.0
"""Public preparation API for Hopper and Blackwell sparse prefill."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os

import torch


def _resolve_partial_dtype(partial_dtype: torch.dtype | None) -> torch.dtype:
    if partial_dtype is None:
        mode = os.environ.get("MBSA_FP8_PARTIAL_OUT", "0")
        if mode not in ("0", "1"):
            raise ValueError("MBSA_FP8_PARTIAL_OUT must be 0 or 1")
        partial_dtype = torch.float8_e4m3fn if mode == "1" else torch.bfloat16
    if partial_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("partial_dtype must be BF16 or FP8 E4M3FN")
    return partial_dtype


def _host_request_lens(
    cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor
) -> tuple[tuple[int, int], ...]:
    """Read (q_len,kv_len) from device prefix sums; synchronizes with the host."""
    q_prefix = cu_seqlens_q.tolist()
    k_prefix = cu_seqlens_k.tolist()
    return tuple(
        (q_end - q_begin, k_end - k_begin)
        for q_begin, q_end, k_begin, k_end in zip(
            q_prefix, q_prefix[1:], k_prefix, k_prefix[1:]
        )
    )


@dataclass
class CutedslPipeline:
    """Reusable plan; run returns output, natural-log LSE, dynamic block IDs.

    Prepared request lengths and offsets stay fixed. Rebind changes operand
    addresses with the same geometry; changing lengths requires preparation.
    """

    plan: object

    def run(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output, lse = self.plan.run()
        return output, lse, self.plan.attention.tile_plan.dynamic_route

    def rebind(
        self,
        index_q,
        index_k,
        q,
        k,
        v,
        *,
        index_page_indices,
        page_table,
        out,
        index_q_scales=None,
    ) -> None:
        """Rebind paged operands with identical shapes, strides and lengths without host sync.
        Page IDs may change; rebinding a captured graph is forbidden."""
        p = self.plan
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


def _prepare_sm90(
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
) -> CutedslPipeline:
    """H100 path: packed or paged main K/V, paged Index-K, BF16/FP8 partials."""
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
    from .cute.sm90 import prepare_sm90_pipeline

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
    return CutedslPipeline(plan)


def prepare_cutedsl_pipeline(
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
) -> CutedslPipeline:
    """Prepare BF16 attention, or native FP8 attention on SM100/SM103 for repeated plan.run() calls.
    Q/Index Q are NHD; K/V are NHD or PHTD pages with logical page tables.
    Index operands are matching BF16/FP16/E4M3, or E4M3 Q plus int32 scales and
    uint8 MXFP6 K [pages,1,128,100].
    Packed calls supply CUDA int32 prefix sums and matching host request_lens.
    Blackwell uses FA4 Fixed and Fireworks Dynamic. Matching E4M3FN K/V
    accepts BF16 Q (converted once per run) or E4M3FN Q, with unit scales.
    QK and PV use native FP8 MMA; output remains BF16. BF16 partials are the default."""
    if not q.is_cuda:
        raise ValueError("CuTe DSL requires CUDA tensors on SM90, SM100 or SM103")
    capability = torch.cuda.get_device_capability(q.device)
    if capability == (9, 0):
        if index_q_scales is not None or index_k.dtype == torch.uint8:
            raise ValueError("MXFP8/MXFP6 Index requires SM100 or SM103; SM90 has no native MX MMA")
        return _prepare_sm90(
            index_q,
            index_k,
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            index_page_indices=index_page_indices,
            page_table=page_table,
            logical_k_tokens=logical_k_tokens,
            overlap_fixed_score=overlap_fixed_score,
            partial_dtype=partial_dtype,
            out=out,
            request_lens=request_lens,
        )
    if capability not in ((10, 0), (10, 3)):
        raise ValueError("the fast prefill path requires SM100 or SM103")
    if (cu_seqlens_q is None) != (cu_seqlens_k is None):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be provided together")
    if cu_seqlens_q is not None:
        if logical_k_tokens is not None:
            raise ValueError("packed lengths are owned by cu_seqlens_k")
        if request_lens is None:
            request_lens = _host_request_lens(cu_seqlens_q, cu_seqlens_k)
    return prepare_sm100(
        index_q,
        index_k,
        q,
        k,
        v,
        index_q_scales=index_q_scales,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        request_lens=request_lens,
        index_page_indices=index_page_indices,
        page_table=page_table,
        logical_k_tokens=logical_k_tokens,
        overlap_fixed_score=overlap_fixed_score,
        partial_dtype=partial_dtype,
        out=out,
    )


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
) -> CutedslPipeline:
    """Prepare SM100/SM103 plans from host lengths.
    check_device_inputs validates device metadata and synchronizes when enabled.
    overlap_fixed_score runs Fixed beside Router/index preparation on fixed_stream;
    Dynamic starts after both streams join. Batch-one calls enable it by default."""
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
    from .cute.prefill import prepare_pipeline, stage_int32
    from .cute.plan import FwdTilePlan, PrefillPolicy

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
    return CutedslPipeline(plan)
