# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Public preparation API for Hopper and Blackwell sparse prefill."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import os

import torch


def _validate_backend(backend: str) -> None:
    if backend not in ("default", "optimized"):
        raise ValueError("backend must be default or optimized")


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
    """Reusable single-in-flight sparse-prefill pipeline.

    Prefer prepare_cutedsl_pipeline() to direct construction. Order all calls and queued work;
    independent plans are required for concurrency. Read-only operands may alias one another, but
    output must be disjoint. run()/run_mega() return live buffers overwritten on the next
    invocation. Captured addresses are immutable and cannot be rebound.

    Args:
        plan: Prepared backend plan retaining tensors, workspace and execution entry points; its
            geometry must already be validated.
        _owner: Optional backend lifecycle owner associated with plan. None wraps plan in the
            default owner for compatibility; callers normally leave this private field to
            preparation.
    """

    plan: object
    _owner: object = field(default=None, repr=False)

    def __post_init__(self):
        if self._owner is None:
            from .backends.default import PreparedOwner
            self._owner = PreparedOwner(self.plan)

    @property
    def backend(self) -> str:
        """Read the backend chosen during preparation.

        Returns:
            "default" or "optimized"; immutable for this pipeline's lifetime.
        """
        return self._owner.backend

    @property
    def lean192(self) -> str:
        """Requested preparation policy; immutable for this plan's lifetime."""
        if self.backend == "optimized":
            return self.plan._state.options["lean192"]
        return "auto"

    @property
    def lean192_enabled(self) -> bool:
        """Whether this plan currently binds the actual 192-thread Indexer."""
        return bool(self.backend == "optimized"
                    and self.plan._state.index.lean192_bound)

    def run(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Submit one invocation on the current CUDA stream.

        Keep calls and queued work on this plan ordered; use independent plans for concurrency.
        Inputs must remain valid until completion. The call enqueues work without a host completion
        guarantee.

        Returns:
            Live (output, lse, routes): BF16 [T,32,128], FP32 natural-log [T,32], int32 [4,T,16]
            with -1 padding. Output uses caller out when supplied; LSE/routes and otherwise output
            belong to the plan and are overwritten on the next run.
        """
        output, lse = self._owner.run()
        return output, lse, self._owner.dynamic_route

    def capture_mega_graph(self, *, warmups: int = 3,
                           stream: torch.cuda.Stream | None = None):
        """Warm up and capture the full chain into a backend-owned CUDA Graph.

        Capture freezes bindings; later rebind is forbidden. Call outside an active capture, after
        preparation, with input storage valid. Calls on this plan remain single-in-flight.

        Args:
            warmups: Number of eager warmup invocations before capture; default 3. Warmup/capture
                overwrite the live output buffers.
            stream: Optional CUDA capture stream on the plan device; None selects the backend
                capture stream.

        Returns:
            This pipeline, with its graph ready for run_mega().
        """
        self._owner.capture_mega_graph(warmups=warmups, stream=stream)
        return self

    def run_mega(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replay the captured full chain on the current stream.

        Requires capture_mega_graph() first. Addresses stay fixed; in-place operand updates must be
        ordered before replay. Replay enqueues asynchronous CUDA work and overwrites live outputs.

        Returns:
            Live (output, lse, routes): BF16 [T,32,128], FP32 natural-log [T,32], int32 [4,T,16]
            with -1 padding. Output uses caller out when supplied; LSE/routes and otherwise output
            belong to the plan and are overwritten on the next run.
        """
        output, lse = self._owner.run_mega()
        return output, lse, self._owner.dynamic_route

    def rebind(self, index_q, index_k, q, k, v, *, index_page_indices,
               page_table, out, index_q_scales=None) -> None:
        """Validate and publish replacement paged operands atomically.

        All tensor metadata and request geometry must match preparation; map values may change.
        Validation may synchronize. Rejection leaves the previous binding usable, but cannot undo
        caller writes to tensor contents. Rebinding after any plan capture or during active capture
        is forbidden. Old bindings remain retained until ordered CUDA work completes.

        Args:
            index_q: CUDA Index queries [T,4,128], matching BF16/FP16/E4M3 Index K, or MXFP8
                uint8/E4M3 codes with index_q_scales. T is the packed query-token count.
            index_k: CUDA Index keys [Skv,1,128] or [pages,1,128,128] for floating formats; MXFP6
                uses packed uint8 [pages,1,128,100] with page-separated codes/scales. Physical
                pages contain 128 tokens.
            q: CUDA attention queries [T,32,128]. BF16 for BF16 K/V; the default Blackwell FP8 path
                also accepts BF16 or E4M3FN Q with E4M3FN K/V. Optimized converts FP8 operands to
                owned BF16 buffers per run.
            k: CUDA attention keys [Skv,4,128] or paged [pages,4,128,128]. Must match v in
                dtype/layout/device; BF16, or E4M3FN on Blackwell with unit scales.
            v: CUDA attention values with the same shape, dtype and device as k. Read-only until
                the queued invocation completes.
            index_page_indices: Replacement CUDA int32 logical-to-physical Index page IDs, compact
                1-D active IDs or request rows [B,>=ceil(KV/128)]. Optimized binds active rows
                directly: one adjacent contiguous span with a 16-byte base alignment for
                unpacked/MX or 4-byte base alignment for packed floating Index. Individual
                compact request rows need not be 16-byte aligned. Gaps/strides are rejected.
                Supply an explicit replacement map matching the prepared metadata; None does
                not regenerate a map during rebind.
            page_table: Replacement paged-attention map with the prepared shape/stride/dtype/device
                and valid physical page IDs.
            out: Replacement writable BF16 output with prepared shape/stride/device and disjoint
                input span. The caller owns this storage.
            index_q_scales: Optional CUDA int32 [T,>=4] MXFP8 scale words, four little-endian UE8M0
                bytes per head. Required for MXFP6 Index K; otherwise None.

        Returns:
            None. Subsequent run() calls use the new bindings and return the usual live buffers.
        """
        self._owner.rebind(index_q, index_k, q, k, v,
                           index_page_indices=index_page_indices,
                           page_table=page_table, out=out,
                           index_q_scales=index_q_scales)



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
    from .backends.default import prepare_sm90
    return prepare_sm90(index_q, index_k, q, k, v, cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k, index_page_indices=index_page_indices,
                        page_table=page_table, logical_k_tokens=logical_k_tokens,
                        overlap_fixed_score=overlap_fixed_score, partial_dtype=partial_dtype,
                        out=out, request_lens=request_lens)


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
    backend: str = "default",
    lean192: str = "auto",
) -> CutedslPipeline:
    """Prepare a reusable sparse-prefill pipeline on Hopper or Blackwell.

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
        backend: Implementation selector: "default" (SM90/SM100/SM103) or "optimized"
            (SM100/SM103). Selection is fixed for the plan lifetime.
        lean192: Optimized Index policy: "auto" leaves the choice to the Index dispatch (on SM103
            the 2-SM Index kernel replaces Lean192 above 12K KV tokens), "off" disables only
            Lean192, and "on" requires its existing eligible SM103 BF16 dense single-request Q8192
            geometry. Unsupported forced ON raises
            ValueError; it never falls back silently. Fixed for the plan lifetime.
            Non-auto values require backend="optimized". Inspect lean192_enabled
            on the returned pipeline for the actual resolved specialization.
            Forced ON rejects supplied mutable Index maps and checked rebind;
            prepare a new dense plan to replace its operand addresses.
            The compiler target must also support SM103 Lean192.

    Returns:
        CutedslPipeline retaining operand/workspace references. run() returns live BF16 O
        [T,32,128], FP32 natural-log LSE [T,32] and int32 routes [4,T,16] (-1 padding). Subsequent
        runs overwrite these buffers.
    """
    _validate_backend(backend)
    from .backends._index_options import validate_lean192
    validate_lean192(lean192, backend=backend)
    from .backends._binding import validate_output_aliases
    validate_output_aliases(out, index_q, index_k, q, k, v, index_q_scales,
                           index_page_indices, page_table, cu_seqlens_q, cu_seqlens_k)
    if not q.is_cuda:
        raise ValueError("CuTe DSL requires CUDA tensors on SM90, SM100 or SM103")
    capability = torch.cuda.get_device_capability(q.device)
    if capability == (9, 0):
        if backend == "optimized":
            raise ValueError(backend + " backend requires SM100 or SM103")
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
        backend=backend,
        lean192=lean192,
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
    backend: str = "default",
    lean192: str = "auto",
) -> CutedslPipeline:
    """Prepare a Blackwell pipeline with explicit validation and stream controls.

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
        backend: Implementation selector: "default" or "optimized", both on SM100/SM103.
            Selection is fixed for the plan lifetime.
        lean192: Preparation-only "auto"/"on"/"off" Index policy. Auto leaves the choice
            to the Index dispatch; off changes only the optimized Lean192 selection.
            On requires the existing eligible SM103 BF16 dense B1/Q8192 policy
            and raises instead of falling back. Non-auto requires optimized.
            The returned lean192_enabled reports the actual bound specialization.
            Forced ON does not accept supplied mutable Index maps or checked
            rebind. Use a new dense plan or auto/off for replacement bindings.
            An incompatible compiler target also rejects forced ON.

    Returns:
        CutedslPipeline with the same live O/LSE/routes contract as prepare_cutedsl_pipeline.
    """
    _validate_backend(backend)
    from .backends._index_options import validate_lean192
    validate_lean192(lean192, backend=backend)
    if not q.is_cuda or torch.cuda.get_device_capability(q.device) not in ((10, 0), (10, 3)):
        raise ValueError("prepare_sm100 requires SM100 or SM103 CUDA tensors")
    from .backends._binding import validate_output_aliases
    validate_output_aliases(out, index_q, index_k, q, k, v, index_q_scales,
                           index_page_indices, page_table, cu_seqlens_q, cu_seqlens_k)
    if (cu_seqlens_q is None) != (cu_seqlens_k is None):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be provided together")
    if cu_seqlens_q is not None and logical_k_tokens is not None:
        raise ValueError("packed lengths are owned by cu_seqlens_k")
    if backend == "optimized":
        from .backends.optimized.runtime import prepare_sm100 as prepare
        index_options = {"lean192": lean192}
    else:
        from .backends.default import prepare_sm100 as prepare
        index_options = {}
    return prepare(index_q, index_k, q, k, v, index_q_scales=index_q_scales,
                   cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                   request_lens=request_lens, index_page_indices=index_page_indices,
                   page_table=page_table, logical_k_tokens=logical_k_tokens,
                   overlap_fixed_score=overlap_fixed_score, partial_dtype=partial_dtype,
                   out=out, check_device_inputs=check_device_inputs, fixed_stream=fixed_stream,
                   **index_options)
