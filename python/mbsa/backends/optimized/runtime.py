# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend-owned sparse attention with the optimized MBSA Index scorer."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import copy
from itertools import accumulate
import torch

from .._binding import validate_output_aliases
from .._lifecycle import PreparedOwnerBase
from .._index_options import validate_lean192


def stage_int32(device, *arrays):
    """Stage host int32 arrays into shared aligned CUDA storage.

    Args:
        device: Destination CUDA device.
        arrays: Positional (host_values, shape) pairs. Values flatten to int32; returned shaped
            views share the staged device buffer.

    Returns:
        Tuple of device views with 16-byte aligned offsets; the copy is enqueued on the current
        stream.
    """
    from .index_base import stage_int32 as stage
    return stage(device, *arrays)


def _metadata(tensor):
    return tensor.shape, tensor.stride(), tensor.dtype, tensor.device


_OWNED_FIELDS: dict[type, tuple[str, ...]] = {}


def _owned_fields(cls):
    """Dataclass fields that may own tensors; host-only fields (metadata ownership=False) hold no tensor that
    other fields do not hold."""
    names = _OWNED_FIELDS.get(cls)
    if names is None:
        names = _OWNED_FIELDS[cls] = tuple(f.name for f in fields(cls) if f.metadata.get("ownership", True))
    return names


def _backing_tensors(*roots):
    """Snapshot the prepared tensor ownership graph without merging aliases."""
    pending, seen, tensors = list(roots), set(), []
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
        elif is_dataclass(value) and not isinstance(value, type):
            pending.extend(getattr(value, name) for name in _owned_fields(type(value)))
    return tuple(tensors)


def _page_views(pages, counts):
    batch = len(counts)
    if pages.ndim == 2:
        if pages.shape[0] != batch or pages.shape[1] < max(counts):
            raise ValueError("index_page_indices must cover each request")
        return tuple(pages[i, :n] for i, n in enumerate(counts))
    if pages.ndim != 1:
        raise ValueError("index_page_indices must be rank 1 or 2")
    if pages.numel() == sum(counts):
        return pages.split(counts)
    if pages.numel() % batch or pages.numel() // batch < max(counts):
        raise ValueError("index_page_indices must contain active pages or padded rows")
    return tuple(pages.view(batch, -1)[i, :n] for i, n in enumerate(counts))


def _validate_inputs(index_q, index_k, q, k, v, index_q_scales, pages,
                     page_table, out, lens, check_device_inputs):
    device = q.device
    total_q = sum(qn for qn, _ in lens)
    if tuple(q.shape) != (total_q, 32, 128):
        raise ValueError("Q must be [total_q,32,128]")
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("Q must be BF16 or FP8 E4M3FN")
    if not q.is_contiguous():
        raise ValueError("Q must be contiguous")
    if k.dtype not in (torch.bfloat16, torch.float8_e4m3fn) or v.dtype != k.dtype:
        raise ValueError("K/V must use matching BF16 or FP8 E4M3FN")
    if q.dtype == torch.float8_e4m3fn and k.dtype != q.dtype:
        raise ValueError("FP8 Q requires FP8 K/V")
    if k.shape != v.shape or k.ndim not in (3, 4):
        raise ValueError("K/V must have matching contiguous or physical-page shapes")
    if k.ndim == 3:
        if tuple(k.shape) != (sum(kn for _, kn in lens), 4, 128):
            raise ValueError("contiguous K/V must be [total_k,4,128]")
        if page_table is not None:
            raise ValueError("page_table is only valid for physical K/V pages")
    elif tuple(k.shape[1:]) != (4, 128, 128):
        raise ValueError("paged K/V must be [pages,4,128,128]")
    if k.ndim == 4 and any(t.stride(-1) != 1 or any(s <= 0 or s % 128 for s in t.stride()[:-1]) for t in (k, v)):
        raise ValueError("paged K/V require positive strides aligned to D rows")
    if k.ndim == 3 and any(not t.is_contiguous() for t in (k, v)):
        raise ValueError("contiguous K/V must be contiguous")
    counts = tuple((kn + 127) // 128 for _, kn in lens)
    if k.ndim == 4:
        if (page_table is None or page_table.dtype != torch.int32
                or page_table.device != device or not page_table.is_contiguous()
                or page_table.ndim != 2 or page_table.shape[0] != len(lens)
                or page_table.shape[1] < max(counts)):
            raise ValueError("page_table must be contiguous int32 [B,>=logical pages]")
    if any(t.device != device for t in (q, k, v, index_k)) or not index_k.is_contiguous():
        raise ValueError("all operands must be on one CUDA device with contiguous Index K")
    if index_q.device != device or tuple(index_q.shape) != (total_q, 4, 128):
        raise ValueError("Index Q must be [total_q,4,128] on the Q device")
    mx = index_k.dtype == torch.uint8
    if mx:
        if index_q.dtype not in (torch.uint8, torch.float8_e4m3fn):
            raise ValueError("MXFP6 Index K requires MXFP8 Index Q")
        if index_q.stride(2) != 1 or index_q.stride(1) != 128:
            raise ValueError("MXFP8 Index Q requires contiguous head rows")
        if tuple(index_k.shape[1:]) != (1, 128, 100) or index_k.ndim != 4:
            raise ValueError("MXFP6 Index K must be [pages,1,128,100]")
        if (index_q_scales is None or index_q_scales.dtype != torch.int32
                or index_q_scales.ndim != 2 or index_q_scales.shape[0] != total_q
                or index_q_scales.shape[1] < 4 or not index_q_scales.is_contiguous()
                or index_q_scales.device != device):
            raise ValueError("MXFP8 Index Q requires contiguous int32 [T,>=4] scales")
    else:
        if (index_q.dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn)
                or index_k.dtype != index_q.dtype or not index_q.is_contiguous()):
            raise ValueError("Index Q/K must use matching contiguous BF16, FP16 or FP8 E4M3FN")
        if index_q_scales is not None:
            raise ValueError("Index Q scales are only valid with MXFP6 Index K")
        if index_k.ndim == 4:
            if tuple(index_k.shape[1:]) != (1, 128, 128):
                raise ValueError("paged Index K must be [pages,1,128,128]")
        elif index_k.ndim != 3 or len(lens) != 1 or tuple(index_k.shape) != (lens[0][1], 1, 128):
            raise ValueError("unpaged Index K must be batch-one [logical_k,1,128]")
    if pages.dtype != torch.int32 or pages.device != device:
        raise ValueError("index_page_indices must be int32 on the Q device")
    rows = _page_views(pages, counts)
    capacity = index_k.shape[0] if index_k.ndim == 4 else (index_k.shape[0] + 127) // 128
    if check_device_inputs:
        for i, (row, count) in enumerate(zip(rows, counts)):
            if bool(torch.any((row < 0) | (row >= capacity))):
                raise ValueError("index_page_indices references Index K out of range")
            if k.ndim == 4 and bool(torch.any((page_table[i, :count] < 0) | (page_table[i, :count] >= k.shape[0]))):
                raise ValueError("page_table references K/V out of range")
    if out is not None and (out.shape != q.shape or out.dtype != torch.bfloat16
                            or out.device != device or not out.is_contiguous()):
        raise ValueError("out must be contiguous BF16 [T,32,128] on the Q device")
    # Validate the effective bases consumed by TMA/vectorized kernels before
    # allocation or rebind publication. Strides do not prove base alignment.
    # FP8 attention and partial unpaged Index K are copied to owned buffers;
    # their public source addresses need not satisfy the destination contract.
    direct = [(name, tensor, 16) for name, tensor in
              (("Q", q), ("K", k), ("V", v))
              if tensor.dtype == torch.bfloat16]
    if out is not None:
        direct.append(("out", out, 16))
    if not mx or index_q.is_contiguous():
        direct.append(("Index Q", index_q, 16))
    if mx:
        direct.extend((("MXFP6 Index K", index_k, 128),
                       ("Index Q scales", index_q_scales, 16)))
    elif index_k.ndim == 4 or index_k.shape[0] % 128 == 0:
        direct.append(("Index K", index_k, 16))
    for name, tensor, alignment in direct:
        if tensor.data_ptr() % alignment:
            raise ValueError(f"{name} base address must be {alignment}-byte aligned")
    validate_output_aliases(out, index_q, index_k, q, k, v, index_q_scales, pages, page_table)
    return rows


@dataclass
class _State:
    attention: object
    index: object
    bindings: dict
    lens: tuple
    page_views: tuple
    conversions: tuple
    options: dict
    fixed_stream: object = None
    inputs_ready: object = None
    fixed_done: object = None
    backing_tensors: tuple = ()

    def _record_eager_storage(self, stream):
        for tensor in self.backing_tensors:
            tensor.record_stream(stream)

    def _refresh(self):
        for conversion in self.conversions:
            if isinstance(conversion, tuple):
                destination, source = conversion
                destination.copy_(source)
            else:
                conversion.refresh()

    def run_plan(self):
        a, p = self.attention, self.index
        if a.query_outer:
            # The query-outer pass reads the route itself. An Index launch that merges into the compact
            # candidate plane ([H,16:32,T], best first, -1 padding) publishes it here.
            if not p.route_published:
                a.tile_plan.dynamic_route.copy_(
                    p.candidate_storage.view(torch.int32)[:, 16:32].transpose(1, 2))
            return
        if not a.fixed_only:
            # The MBSA scorer merges all partitions before publishing these
            # IDs. The attention consumer receives one globally selected candidate
            # set, independently of the scorer's partition tile width.
            merged_span = max(p.candidate_partition_blocks,
                              max((kn + 127) // 128 for _, kn in self.lens))
            a.build_dynamic_from_candidates(
                p.candidate_storage, p.prefix_lens,
                candidate_partition_blocks=merged_span,
                canonicalize_dynamic_padding=a.max_seqlen_k < 32768,
            )

    def run(self):
        # CuTe launches bypass PyTorch's dispatcher. Protect every backing
        # allocation on the actual eager launch stream, including plan-owned
        # CSR/partials when a caller drops a plan with work still queued.
        a, p = self.attention, self.index
        current = torch.cuda.current_stream(a.q.device)
        if not torch.cuda.is_current_stream_capturing():
            self._record_eager_storage(current)
        self._refresh()
        if a.fixed_only:
            return a.run()
        if self.fixed_stream is not None and current != self.fixed_stream:
            self.inputs_ready.record(current)
            # Preserve the score-first enqueue policy; both streams join
            # before Dynamic consumes routes or Fixed output.
            p.run_score()
            self.fixed_stream.wait_event(self.inputs_ready)
            try:
                with torch.cuda.stream(self.fixed_stream):
                    try:
                        a.run_fixed()
                    finally:
                        self.fixed_done.record(self.fixed_stream)
                self.run_plan()
            finally:
                current.wait_event(self.fixed_done)
            a.run_dynamic()
            return a.run_merge()
        p.run_score()
        self.run_plan()
        return a.run()


class Plan:
    """One single-in-flight private state; captured bindings are immutable."""
    def __init__(self, state):
        """Own the optimized prepared state and its capture resources.

        Args:
            state: Prepared private state containing retained input bindings, Index/attention
                buffers, conversions and stream/event dependencies.

        Returns:
            None. Calls using this plan must remain single-in-flight.
        """
        self._state = state
        self._mega_graph = None
        self._mega_capture_stream = None
        self._binding_external_capture = False

    def __getattr__(self, name):
        if name == "attention":
            return self._state.attention
        if name in self._state.bindings:
            return self._state.bindings[name]
        return getattr(self._state.index, name)

    @property
    def mega_graph_captured(self):
        """Check whether this plan owns a captured full-chain graph.

        Returns:
            True after capture_mega_graph(), otherwise False.
        """
        return self._mega_graph is not None

    def run(self):
        """Enqueue the complete chain, including conversion refresh and allocation lifetime records.

        Order calls and their CUDA work; inputs must remain valid through completion.

        Returns:
            Live (output, natural-log LSE) buffers; subsequent calls overwrite them.
        """
        return self._state.run()

    def run_score(self):
        """Refresh conversion buffers and submit only the Index selection stage.

        Order calls and their CUDA work; inputs must remain valid through completion.

        Returns:
            None; writes the plan-owned candidate/partition buffers.
        """
        self._state._refresh()
        return self._state.index.run_score()

    def run_plan(self):
        """Publish dynamic selections and rebuild CSR/schedule metadata from current candidates.

        Order calls and their CUDA work; inputs must remain valid through completion.

        Returns:
            None; updates the attention plan's route and schedule storage.
        """
        return self._state.run_plan()

    def capture_mega_graph(self, *, warmups=3, stream=None):
        """Capture the full chain with fixed addresses and reusable output buffers.

        Rejects capture inside an active capture or after any graph was captured. Inputs and
        previous work are ordered onto the capture stream.

        Args:
            warmups: Requested warmup count (default 3); at least one eager warmup is executed.
            stream: Optional CUDA capture stream on the attention device; None creates one.

        Returns:
            This plan; subsequent rebind is forbidden.
        """
        if self.mega_graph_captured or self._binding_external_capture:
            raise RuntimeError("plan is already captured")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cannot capture a plan from an existing capture")
        stream = stream or torch.cuda.Stream(device=self.attention.q.device)
        current = torch.cuda.current_stream(self.attention.q.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(max(1, warmups)):
                self.run()
        current.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            self.run()
        self._mega_graph, self._mega_capture_stream = graph, stream
        return self

    def run_mega(self):
        """Replay the owned graph; capture_mega_graph() must have completed first.

        Order calls and their CUDA work; inputs must remain valid through completion.

        Returns:
            Live (output, natural-log LSE), overwritten by the replay.
        """
        if self._mega_graph is None:
            raise RuntimeError("capture_mega_graph must run before run_mega")
        self._mega_graph.replay()
        return self.attention.out, self.attention.out_lse


class PreparedOwner(PreparedOwnerBase):
    backend = "optimized"

    def rebind(self, index_q, index_k, q, k, v, *, index_page_indices,
               page_table, out, index_q_scales=None):
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
                also accepts BF16 or E4M3FN Q with E4M3FN K/V. Optimized converts FP8 Q and, for paged
                K/V, only the logical pages of the prepared requests to owned BF16 buffers per run.
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
            None. Failed validation leaves all published bindings usable.
        """
        return self._stage_binding(index_q, index_k, q, k, v,
            index_page_indices=index_page_indices, page_table=page_table, out=out,
            index_q_scales=index_q_scales, check_pages=True)

    def _stage_binding(self, index_q, index_k, q, k, v, *, index_page_indices,
                       page_table, out, index_q_scales, check_pages):
        self._check_mutable()
        old = self.plan._state
        if check_pages and old.index.lean192_mode == "on":
            raise ValueError(
                'lean192="on" cannot rebind a mutable Index page map; '
                'prepare a new dense plan or use lean192="auto"/"off"'
            )
        if old.bindings['index_q_scales'] is not None and index_q_scales is None:
            raise ValueError("MX Index requires index_q_scales")
        replacements = dict(index_q=index_q, index_k=index_k, q=q, k=k, v=v,
                            index_page_indices=index_page_indices, page_table=page_table,
                            out=out, index_q_scales=index_q_scales)
        for name, new in replacements.items():
            previous = old.bindings[name]
            if (new is None) != (previous is None) or new is not None and _metadata(new) != _metadata(previous):
                raise ValueError(name + " must retain prepared shape/stride/dtype/device")
        rows = _validate_inputs(index_q, index_k, q, k, v, index_q_scales,
                               index_page_indices, page_table, out, old.lens, check_pages)
        # Stage host bindings only: serving layer reuse keeps all prepared
        # workspaces, schedules and compile identities. No kernel runs before
        # publication; only a changed Index dispatch choice stages new partition
        # buffers (host-to-device copies on the current stream). Rejected
        # staging preserves old bindings.
        staged = copy.copy(old)
        a = staged.attention = copy.copy(old.attention)
        a.tile_plan = copy.copy(old.attention.tile_plan)
        p = staged.index = copy.copy(old.index)
        p.attention = a
        staged.bindings = replacements
        staged.page_views = rows
        staged.options = dict(old.options, **replacements)
        staged.conversions = tuple(
            (conversion[0], next((replacements[name] for name, previous in old.bindings.items()
                                  if conversion[1] is previous), conversion[1]))
            if isinstance(conversion, tuple) else conversion.rebind(k, v, page_table)
            for conversion in old.conversions)
        for name in ('q', 'k', 'v'):
            previous = old.bindings[name]
            operand = getattr(old.attention, name)
            setattr(a, name, replacements[name] if operand is previous else operand)
        if all(isinstance(conversion, tuple) for conversion in old.conversions):
            # A compact FP8 page copy keeps its static table; refresh reads page_table.
            a.tile_plan.page_table = page_table if page_table is not None else old.attention.tile_plan.page_table
        a.out = out
        if old.attention.dynamic_out is old.attention.out:
            a.dynamic_out = out
        p.index_q, p.index_q_scales = index_q, index_q_scales
        p.index_page_indices = index_page_indices
        padded_index = old.index.index_k is not old.bindings['index_k']
        p.index_k = old.index.index_k if padded_index else index_k
        if p.cutedsl_index_k is not None:
            p.cutedsl_index_k = p.index_k if index_k.dtype == torch.uint8 else p.index_k.view(-1, 1, 128)
        if check_pages and p.cutedsl_index_pages is not None:
            p.cutedsl_use_page_table = True
            p._lean192_early_router = False
        from .index import bind_score_operands, require_kv_lengths_launch
        from .page_map import bind_page_operands
        bind_page_operands(p, rows)
        if p.mxfp6_router is not None:
            p.mxfp6_router = copy.copy(p.mxfp6_router)
            p.mxfp6_router.rebind(index_q, index_q_scales, index_k, p.cutedsl_packed_pages)
        bind_score_operands(p)
        require_kv_lengths_launch(p)
        # Build the replacement ownership snapshot before publishing bindings.
        # The retired state keeps its own snapshot until its queued work ends.
        staged.backing_tensors = _backing_tensors(
            staged.bindings, staged.conversions, a, p)
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(q.device))
        self._retired_bindings = [entry for entry in self._retired_bindings if not entry[0].query()]
        self.plan._state = staged
        self._retired_bindings.append((done, old))
        _record_fixed_storage(staged)

    def bind_layer(self, index_q, index_k, q, k, v, out, *, index_q_scales=None):
        """Transactionally bind a new serving layer with the existing request/maps.

        Call outside capture and before any graph has been captured. Tensor metadata must match
        preparation; rejection preserves the prior binding. Order old work before publication on
        the current stream.

        Args:
            index_q: CUDA Index queries [T,4,128], matching BF16/FP16/E4M3 Index K, or MXFP8
                uint8/E4M3 codes with index_q_scales. T is the packed query-token count.
            index_k: CUDA Index keys [Skv,1,128] or [pages,1,128,128] for floating formats; MXFP6
                uses packed uint8 [pages,1,128,100] with page-separated codes/scales. Physical
                pages contain 128 tokens.
            q: CUDA attention queries [T,32,128]. BF16 for BF16 K/V; the default Blackwell FP8 path
                also accepts BF16 or E4M3FN Q with E4M3FN K/V. Optimized converts FP8 Q and, for paged
                K/V, only the logical pages of the prepared requests to owned BF16 buffers per run.
            k: CUDA attention keys [Skv,4,128] or paged [pages,4,128,128]. Must match v in
                dtype/layout/device; BF16, or E4M3FN on Blackwell with unit scales.
            v: CUDA attention values with the same shape, dtype and device as k. Read-only until
                the queued invocation completes.
            out: Replacement writable BF16 output with prepared shape/stride/device and disjoint
                input span. The caller owns this storage.
            index_q_scales: Optional CUDA int32 [T,>=4] MXFP8 scale words, four little-endian UE8M0
                bytes per head. Required for MXFP6 Index K; otherwise None.

        Returns:
            None; publishes the validated layer operands and retains retired backing storage.
        """
        state = self.plan._state
        return self._stage_binding(index_q, index_k, q, k, v,
                           index_page_indices=state.bindings['index_page_indices'],
                           page_table=state.bindings['page_table'], out=out,
                           index_q_scales=index_q_scales, check_pages=False)


def _record_fixed_storage(state):
    if state.fixed_stream is None:
        return
    a = state.attention
    for tensor in (a.q, a.k, a.v, a.cu_q, a.cu_k, a.tile_plan.page_table,
                   a.tile_plan.valid_kv_tokens, a.fixed_out, a.fixed_lse,
                   a.fixed_packed_tiles, a.route_sentinel):
        tensor.record_stream(state.fixed_stream)


def _prepare_state(index_q, index_k, q, k, v, *, index_q_scales=None,
                   cu_seqlens_q=None, cu_seqlens_k=None, request_lens=None,
                   index_page_indices=None, page_table=None, logical_k_tokens=None,
                   overlap_fixed_score=None, partial_dtype=None, out=None,
                   check_device_inputs=True, fixed_stream=None, lean192="auto", kv_lengths=None):
    validate_lean192(lean192)
    if lean192 == "on":
        from .index import lean192_compiler_supported
        if not lean192_compiler_supported():
            raise ValueError('lean192="on" requires an SM103 compiler target')
    from ...api import _resolve_partial_dtype
    from .kernels.attention import prepare_unified_aligned_prefill
    from .kernels.plan import FwdTilePlan, PrefillPolicy
    from .kernels.attention import BsaOutputLayout
    from .index import prepare_index
    partial_dtype = _resolve_partial_dtype(partial_dtype)
    if (cu_seqlens_q is None) != (cu_seqlens_k is None):
        raise ValueError("both Q/K prefix sums are required")
    unpacked = cu_seqlens_q is None
    if unpacked:
        if k.ndim == 4 and logical_k_tokens is None:
            raise ValueError("paged KV requires logical_k_tokens")
        lens = ((q.shape[0], logical_k_tokens if logical_k_tokens is not None else k.shape[0]),)
    else:
        if request_lens is None or logical_k_tokens is not None:
            raise ValueError("packed preparation requires host request_lens; lengths belong to prefix sums")
        lens = tuple((int(qn), int(kn)) for qn, kn in request_lens)
    if not lens or any(not 0 < qn <= min(kn, 8192) or kn > 1048576 for qn, kn in lens):
        raise ValueError("each request requires 0 < Q <= 8192 and Q <= KV <= 1M")
    q_lens, kv_lens = tuple(qn for qn, _ in lens), tuple(kn for _, kn in lens)
    if kv_lengths is not None:
        # The step runs at KV lengths published after preparation: request_lens are its capacities.
        if (unpacked or max(q_lens) != 1 or k.ndim != 4 or index_k.dtype == torch.uint8
                or any(t.dtype != torch.bfloat16 for t in (q, k, v))):
            raise ValueError("kv_lengths requires packed one-token requests (cu_seqlens and request_lens), "
                             "paged BF16 Q/K/V and a BF16, FP16 or E4M3 Index")
        if kv_lengths.device != q.device:
            raise ValueError("kv_lengths must be on the Q device")
        kv_lengths._bind(kv_lens)     # prepare_sm100 undoes a first binding if the preparation fails
    counts = tuple((kn + 127) // 128 for kn in kv_lens)
    pages_supplied = index_page_indices is not None
    if index_page_indices is None:
        index_page_indices = torch.arange(sum(counts), dtype=torch.int32, device=q.device)
    rows = _validate_inputs(index_q, index_k, q, k, v, index_q_scales,
                            index_page_indices, page_table, out, lens, check_device_inputs)
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    bindings = dict(index_q=index_q, index_k=index_k, q=q, k=k, v=v,
                    index_q_scales=index_q_scales, index_page_indices=index_page_indices,
                    page_table=page_table, out=out)
    conversions = []
    compute = []
    # Paged FP8 K/V: convert only the logical pages of this step's requests,
    # not the whole physical cache (see kernels/kv_pages.py). Requests sharing
    # pages can own more logical pages than the cache holds; then the whole
    # cache is converted, which is never larger.
    paged_fp8 = k.ndim == 4 and k.dtype == torch.float8_e4m3fn and sum(counts) < k.shape[0]
    for operand in (q,) if paged_fp8 else (q, k, v):
        if operand.dtype == torch.bfloat16:
            compute.append(operand)
        else:
            destination = torch.empty(operand.shape, dtype=torch.bfloat16, device=operand.device)
            destination.copy_(operand)
            conversions.append((destination, operand))
            compute.append(destination)
    if paged_fp8:
        from .kernels.kv_pages import PagedKVConversion
        paged = PagedKVConversion.prepare(k, v, page_table, counts)
        conversions.append(paged)
        compute += [paged.k, paged.v]
    # Pad only the physical Index tail, preserving the public logical tensor.
    effective_index_k = index_k
    if index_k.ndim == 3 and index_k.shape[0] % 128:
        padded = torch.zeros((counts[0]*128, 1, 128), dtype=index_k.dtype, device=q.device)
        conversions.append((padded[:index_k.shape[0]], index_k))
        padded[:index_k.shape[0]].copy_(index_k)
        effective_index_k = padded.view(-1, 1, 128, 128)
    valid, offsets = stage_int32(q.device, (kv_lens, (len(lens),)),
                                 (tuple(kn-qn for qn, kn in lens), (len(lens),)))
    if unpacked:
        cu_seqlens_q, cu_seqlens_k = stage_int32(q.device,
            ((0, q.shape[0]), (2,)), ((0, kv_lens[0]), (2,)))
    elif check_device_inputs:
        if tuple(cu_seqlens_q.tolist()) != tuple(accumulate(q_lens, initial=0)) or tuple(cu_seqlens_k.tolist()) != tuple(accumulate(kv_lens, initial=0)):
            raise ValueError("device prefix sums disagree with request_lens")
    policy = PrefillPolicy.ALIGNED_PAGED if k.ndim == 4 else PrefillPolicy.ALIGNED_CONTIGUOUS
    table = (paged.table if paged_fp8 else page_table if page_table is not None
             else torch.empty((len(lens), 0), dtype=torch.int32, device=q.device))
    tile_plan = FwdTilePlan(
        dynamic_route=torch.full((4, q.shape[0], 16), -1, dtype=torch.int32, device=q.device),
        page_table=table, valid_kv_tokens=valid,
        q_offset_tokens=kv_lens[0]-q_lens[0] if len(lens)==1 else tuple(kn-qn for qn,kn in lens),
        policy=policy,
    )
    # A decode-like step (one query token per request) runs query-outer: the Index stage, then one BSA
    # pass over each token's prefix, local and routed blocks. Its Index and attention buffers are not sized
    # by the KV blocks of the whole batch (the FP8 K/V page conversion above still is), and there is no
    # separate Fixed pass to overlap.
    query_outer = max(q_lens) == 1
    if overlap_fixed_score is None:
        overlap_fixed_score = len(lens)==1 or max(kv_lens)<=8192
    # One static persistent policy for all supported OG prefill requests.
    # Scheduling does not depend on KV length, packing or score overlap.
    fixed_use_clc_scheduler = False
    fully_visible = index_k.dtype != torch.uint8
    attention = prepare_unified_aligned_prefill(
        *compute, tile_plan, out=out, partial_dtype=partial_dtype,
        fuse_fixed_state_in_dynamic_combine=len(lens)==1,
        fixed_use_clc_scheduler=fixed_use_clc_scheduler,
        fixed_output_layout=BsaOutputLayout.SCATTERED,
        fixed_split_p_arrive=64, fixed_sync_first_p_release=True,
        compact_batch1_schedule=len(lens)==1 and max(kv_lens)>=32768,
        dynamic_route_fully_visible=fully_visible,
        dynamic_full_score_tiles=fully_visible,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, query_outer=query_outer,
        kv_lengths=kv_lengths,
    )
    if attention.query_outer:
        overlap_fixed_score = False
    index = prepare_index(index_q, effective_index_k, index_q_scales=index_q_scales,
        index_page_indices=index_page_indices, candidate_storage=torch.empty((4,32,q.shape[0]), dtype=torch.float32, device=q.device),
        prefix_lens=offsets, attention=attention, q_lens=q_lens, kv_lens=kv_lens,
        index_pages_identity=not pages_supplied and len(lens)==1,
        check_device_inputs=check_device_inputs, original_q=q, original_k=k,
        original_v=v, contiguous_unpacked=unpacked and k.ndim==3, lean192=lean192,
        overlap_fixed_score=bool(overlap_fixed_score) and not attention.fixed_only, kv_lengths=kv_lengths)
    if not overlap_fixed_score or attention.fixed_only:
        fixed_stream = None
    elif fixed_stream is None:
        fixed_stream = torch.cuda.Stream(device=q.device)
    elif fixed_stream.device != q.device:
        raise ValueError("fixed_stream must be on the Q device")
    options = dict(bindings, cu_seqlens_q=None if unpacked else cu_seqlens_q,
                   cu_seqlens_k=None if unpacked else cu_seqlens_k,
                   request_lens=None if unpacked else lens, logical_k_tokens=logical_k_tokens,
                   overlap_fixed_score=overlap_fixed_score, partial_dtype=partial_dtype,
                   check_device_inputs=check_device_inputs, fixed_stream=fixed_stream,
                   lean192=lean192, kv_lengths=kv_lengths)
    state = _State(attention, index, bindings, lens, rows, tuple(conversions), options,
                    fixed_stream, torch.cuda.Event() if fixed_stream else None,
                    torch.cuda.Event() if fixed_stream else None)
    state.backing_tensors = _backing_tensors(bindings, tuple(conversions), attention, index)
    _record_fixed_storage(state)
    return state


def prepare_sm100(*args, **kwargs):
    """Prepare optimized Index, private attention buffers and backend-owned lifecycle.

    Use the public mbsa.api.prepare_sm100 signature and contracts; preparation allocates/compiles
    outside capture.

    Args:
        args: Positional index_q, index_k, q, k, v CUDA operands as documented by
            mbsa.api.prepare_sm100.
        kwargs: Keyword preparation arguments accepted by mbsa.api.prepare_sm100, except backend
            (already selected). Includes lengths/maps, output, partial dtype, validation and Fixed
            stream controls.

    Returns:
        CutedslPipeline with retained bindings/workspaces and live O/LSE/routes.
    """
    from ...api import CutedslPipeline
    lengths = kwargs.get("kv_lengths")
    first = lengths is not None and lengths.capacities is None
    try:
        state = _prepare_state(*args, **kwargs)
    except BaseException:
        if first:
            lengths._unbind()     # a failed first preparation leaves the object as it was created
        raise
    plan = Plan(state)
    return CutedslPipeline(plan, PreparedOwner(plan, state.lens))
