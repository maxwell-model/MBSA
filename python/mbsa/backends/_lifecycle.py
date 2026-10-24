# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordered lifecycle policy for backend-owned prepared plans."""

import copy
from types import MethodType
import torch
from ._binding import stage_plan, publish_plan, validate_output_aliases


class PreparedOwnerBase:
    """Own one single-in-flight plan and retain bindings until CUDA finishes."""


    def __init__(self, plan, request_lens=None):
        """Retain a prepared plan and track its graph/rebind lifecycle.

        Args:
            plan: Prepared backend plan with run/capture methods and attention buffers. Its
                compatibility run method is wrapped to observe external capture.
            request_lens: Optional host request (Q,KV) token counts used for page-ID validation;
                None uses the plan's fixed request ranges.

        Returns:
            None. The owner retains old bindings until ordered CUDA work completes.
        """
        self.plan = plan
        self.request_lens = request_lens
        # Preserve the compatibility plan.run entry point while observing
        # external capture on that entry point as well as on the public API.
        plan._binding_external_capture = False
        original_run = type(plan).run

        def tracked_run(bound_plan):
            capturing = torch.cuda.is_current_stream_capturing()
            result = original_run(bound_plan)
            if capturing:
                bound_plan._binding_external_capture = True
            return result

        plan.run = MethodType(tracked_run, plan)
        self._retired_bindings = []

    def __getattr__(self, name):
        return getattr(self.plan, name)

    def run(self):
        """Submit the owned plan on the current stream.

        Returns:
            Live (BF16 output, FP32 natural-log LSE); next execution overwrites them.
        """
        return self.plan.run()

    @property
    def dynamic_route(self):
        """Read the plan-owned current dynamic page selections.

        Returns:
            Live int32 [Hkv,T,16] routes with -1 padding, overwritten by subsequent execution.
        """
        return self.plan.attention.tile_plan.dynamic_route

    def capture_mega_graph(self, *, warmups=3, stream=None):
        """Warm up and capture the owned full-chain plan; future rebind is forbidden.

        Args:
            warmups: Eager warmup count before capture, default 3; live outputs are overwritten.
            stream: Optional CUDA capture stream on the plan device; None delegates stream
                selection to the backend.

        Returns:
            This owner. Capture/replay remain single-in-flight.
        """
        self.plan.capture_mega_graph(warmups=warmups, stream=stream)
        return self

    def run_mega(self):
        """Replay the previously captured graph on the current stream.

        Returns:
            Live (output, LSE) buffers, overwritten by replay.
        """
        return self.plan.run_mega()

    def _check_mutable(self):
        if self.plan._binding_external_capture or self.plan.mega_graph_captured:
            raise RuntimeError("cannot rebind a captured mega graph or externally captured plan")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cannot rebind during CUDA graph capture")

    def _publish(self, staged):
        old = copy.copy(self.plan)
        old.attention = copy.copy(self.plan.attention)
        self._retired_bindings = [
            entry for entry in self._retired_bindings if not entry[0].query()
        ]
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(staged.attention.out.device))
        publish_plan(self.plan, staged)
        self._retired_bindings.append((done, old))

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
                also accepts BF16 or E4M3FN Q with E4M3FN K/V. Optimized converts FP8 operands to
                owned BF16 buffers per run.
            k: CUDA attention keys [Skv,4,128] or paged [pages,4,128,128]. Must match v in
                dtype/layout/device; BF16, or E4M3FN on Blackwell with unit scales.
            v: CUDA attention values with the same shape, dtype and device as k. Read-only until
                the queued invocation completes.
            index_page_indices: Replacement CUDA int32 logical-to-physical Index page IDs, compact
                1-D active IDs or request rows [B,>=ceil(KV/128)]. Optimized binds active rows
                directly: one adjacent contiguous storage span, each row aligned to 16 bytes for
                unpacked/MX or 4 bytes for packed floating Index. Gaps/strides are rejected;
                compact maps avoid row padding. Supply an explicit replacement map matching the
                prepared metadata; None does not regenerate a map during rebind.
            page_table: Replacement paged-attention map with the prepared shape/stride/dtype/device
                and valid physical page IDs.
            out: Replacement writable BF16 output with prepared shape/stride/device and disjoint
                input span. The caller owns this storage.
            index_q_scales: Optional CUDA int32 [T,>=4] MXFP8 scale words, four little-endian UE8M0
                bytes per head. Required for MXFP6 Index K; otherwise None.

        Returns:
            None; publication follows validation and preserves old bindings on rejection.
        """
        self._check_mutable()
        validate_output_aliases(out, index_q, index_k, q, k, v, index_q_scales,
                                index_page_indices, page_table)
        staged = stage_plan(self.plan, page_table=page_table)
        self._bind_paged(staged, index_q, index_k, q, k, v,
                         index_page_indices=index_page_indices,
                         page_table=page_table, out=out,
                         index_q_scales=index_q_scales)
        self._validate_pages(index_k, k, index_page_indices, page_table)
        fixed = getattr(staged.attention, "fixed_attention", None)
        if fixed is not None and page_table.data_ptr() == self.plan.attention.fixed_attention.block_tables.data_ptr():
            fixed.refresh_table()
        self._publish(staged)

    def _validate_pages(self, index_k, k, index_pages, page_table):
        lens = self.request_lens
        if lens is None:
            ranges = getattr(self.plan.attention, "fixed_request_ranges", ())
            lens = tuple((qe - qb, ke - kb) for qb, qe, kb, ke in ranges)
        counts = tuple((kn + 127) // 128 for _, kn in lens)
        if not counts:
            return
        if index_pages.ndim == 2:
            rows = tuple(index_pages[i, :count] for i, count in enumerate(counts))
        elif index_pages.numel() == sum(counts):
            rows = index_pages.split(counts)
        else:
            padded = index_pages.view(len(counts), -1)
            rows = tuple(padded[i, :count] for i, count in enumerate(counts))
        index_capacity = index_k.shape[0] if index_k.ndim == 4 else index_k.shape[0] // 128
        for i, (row, count) in enumerate(zip(rows, counts)):
            if any(n < 0 or n >= index_capacity for n in row.tolist()):
                raise ValueError("index_page_indices references Index K out of range")
            if k.ndim == 4 and any(n < 0 or n >= k.shape[0] for n in page_table[i, :count].tolist()):
                raise ValueError("page_table references K/V out of range")

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
                also accepts BF16 or E4M3FN Q with E4M3FN K/V. Optimized converts FP8 operands to
                owned BF16 buffers per run.
            k: CUDA attention keys [Skv,4,128] or paged [pages,4,128,128]. Must match v in
                dtype/layout/device; BF16, or E4M3FN on Blackwell with unit scales.
            v: CUDA attention values with the same shape, dtype and device as k. Read-only until
                the queued invocation completes.
            out: Replacement writable BF16 output with prepared shape/stride/device and disjoint
                input span. The caller owns this storage.
            index_q_scales: Optional CUDA int32 [T,>=4] MXFP8 scale words, four little-endian UE8M0
                bytes per head. Required for MXFP6 Index K; otherwise None.

        Returns:
            None; queued old work retains its backing allocations until completion.
        """
        self._check_mutable()
        validate_output_aliases(out, index_q, index_k, q, k, v, index_q_scales)
        staged = stage_plan(self.plan)
        staged.bind_layer(index_q, index_k, q, k, v, out,
                          index_q_scales=index_q_scales)
        self._publish(staged)
