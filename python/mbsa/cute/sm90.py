# SPDX-License-Identifier: Apache-2.0
"""BF16 SM90 prefill: router, schedule, fixed/dynamic attention and combine.
K/V use packed [total_k,4,128] or paged [pages,4,128,128] layouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Float8E4M3FN
from cutlass.cute.runtime import from_dlpack

from .src.sm90.attention import (
    CombineSm90,
    DynamicAttentionSm90,
    FixedAttentionSm90,
)
from .src.sm90.router import IndexScoreTop16Sm90, MergeTop16Sm90
from .src.sm90.schedule import RouteHistogramSm90, ScheduleScanSm90, ScheduleScatterSm90


BLOCK = 128
TOPK = 16
PREFIX_BLOCKS = 8
LOCAL_BLOCKS = 8
TILE_QUERIES = 16
WORK_QUERIES = DynamicAttentionSm90.max_queries_per_work
SOFTMAX_SCALE = 128 ** -0.5
_COMPILE_CACHE: dict[tuple, object] = {}


def _cute(t: torch.Tensor, align: int = 16):
    # PyTorch cu126 cannot export FP8 via DLPack. Keep its bytes and strides
    # unchanged, then describe the original format to CuTe/WGMMA.
    if t.dtype == torch.float8_e4m3fn:
        tensor = from_dlpack(t.view(torch.uint8), assumed_align=align)
        tensor.element_type = Float8E4M3FN
        return tensor
    return from_dlpack(t, assumed_align=align)


def _signature(*tensors: torch.Tensor) -> tuple:
    return tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in tensors)


def _compile(key: tuple, kernel_factory, args: tuple):
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(kernel_factory(), *args, _stream())
        _COMPILE_CACHE[key] = compiled
    return compiled


def _stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _router_partitions(q_tiles: int, kv_blocks: int, sms: int) -> tuple[int, int]:
    """Split KV only when the query tiles alone cannot fill the device."""
    ctas = q_tiles * 4
    partitions = 1
    while ctas * partitions < sms and kv_blocks // (partitions * 2) >= 64:
        partitions *= 2
    partition_blocks = -(-kv_blocks // partitions)
    return partition_blocks, -(-kv_blocks // partition_blocks)


@dataclass
class Sm90Attention:
    """Attention stages; mirrors the SM103 ``AttentionPlan`` stage hooks."""

    out: torch.Tensor
    out_lse: torch.Tensor
    tile_plan: SimpleNamespace
    fixed_only: bool
    dynamic_workspace: tuple = ()
    launches: dict = field(default_factory=dict)

    def _launch(self, name: str) -> None:
        kernel, args = self.launches[name]
        kernel(*args, _stream())

    def run_fixed(self) -> None:
        self._launch("fixed")

    def run_dynamic(self) -> None:
        if not self.fixed_only:
            self._launch("dynamic")

    def run_merge(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.fixed_only:
            self._launch("combine")
        return self.out, self.out_lse

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.run_fixed()
        self.run_dynamic()
        return self.run_merge()


@dataclass
class Sm90PipelinePlan:
    """Prepared H100 router, KV-centric schedule and attention chain."""

    attention: Sm90Attention
    route: torch.Tensor
    geometry: SimpleNamespace
    workspace: SimpleNamespace
    bound: tuple = ()
    overlap_fixed_score: bool = False
    fixed_stream: torch.cuda.Stream | None = None
    inputs_ready: torch.cuda.Event | None = None
    fixed_done: torch.cuda.Event | None = None
    _mega_graph: torch.cuda.CUDAGraph | None = field(default=None, init=False, repr=False)
    _mega_capture_stream: torch.cuda.Stream | None = field(
        default=None, init=False, repr=False
    )

    @property
    def mega_graph_captured(self) -> bool:
        """Whether this prepared, pointer-stable plan owns a captured graph."""
        return self._mega_graph is not None

    def capture_mega_graph(
        self, *, warmups: int = 3, stream: torch.cuda.Stream | None = None
    ) -> "Sm90PipelinePlan":
        """Capture the complete chain (including the fixed-stream fork) once.

        Tensor addresses and shapes must stay stable while the graph lives.
        """
        if warmups < 1:
            raise ValueError("mega-graph capture requires at least one warmup")
        if self._mega_graph is not None:
            return self
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cannot prepare a mega graph during another capture")
        capture_stream = stream or torch.cuda.Stream(device=self.route.device)
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(warmups):
                self.run()
        capture_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            self.run()
        self._mega_graph = graph
        self._mega_capture_stream = capture_stream
        return self

    def run_mega(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay a previously captured execution graph."""
        if self._mega_graph is None:
            raise RuntimeError("call capture_mega_graph() before run_mega()")
        self._mega_graph.replay()
        return self.attention.out, self.attention.out_lse

    def _launch(self, name: str) -> None:
        launch = self.attention.launches.get(name)
        if launch is not None:
            launch[0](*launch[1], _stream())

    def run_score(self) -> None:
        if not self.attention.fixed_only:
            self._launch("router")
            self._launch("merge")

    def run_plan(self) -> None:
        """Counting-sort route entries into KV-centric work items on device."""
        if not self.attention.fixed_only:
            for name in ("histogram", "scan", "scatter"):
                self._launch(name)

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.attention
        if a.fixed_only:
            a.run_fixed()
            return a.out, a.out_lse
        if self.overlap_fixed_score:
            current = torch.cuda.current_stream()
            self.inputs_ready.record(current)
            self.fixed_stream.wait_event(self.inputs_ready)
            with torch.cuda.stream(self.fixed_stream):
                a.run_fixed()
                self.fixed_done.record(self.fixed_stream)
            self.run_score()
            self.run_plan()
            a.run_dynamic()
            current.wait_event(self.fixed_done)
            return a.run_merge()
        self.run_score()
        self.run_plan()
        return a.run()

    def bind(self, index_q, index_k, q, k, v, index_pages, page_table, out) -> None:
        """Compile (cached by geometry) and bind kernel arguments to tensors."""
        g, w, a = self.geometry, self.workspace, self.attention
        paged = g.paged
        index_k_rows = index_k.reshape(-1, 128)
        pages = index_pages.reshape(-1)
        table = page_table if paged else w.empty_table
        fixed_out = out if a.fixed_only else w.fixed_out
        fixed_lse = a.out_lse if a.fixed_only else w.fixed_lse
        softmax_scale = Float32(SOFTMAX_SCALE)
        fp8 = g.partial_dtype == torch.float8_e4m3fn
        launches = {}

        fixed_args = tuple(_cute(t) for t in (q, k, v, fixed_out, fixed_lse)) + (
            _cute(g.fixed_meta, 4),
            _cute(table, 4),
            softmax_scale,
        )
        launches["fixed"] = (
            _compile(
                ("fixed", paged) + _signature(q, k, v, fixed_out, g.fixed_meta, table),
                lambda: FixedAttentionSm90(paged=paged),
                fixed_args,
            ),
            fixed_args,
        )
        if not a.fixed_only:
            router_args = (
                _cute(index_q),
                _cute(index_k_rows),
                _cute(pages, 4),
                _cute(g.router_meta, 4),
                _cute(w.scores),
                _cute(w.ids),
                _cute(self.route),
            )
            launches["router"] = (
                _compile(
                    ("router", g.partition_blocks, g.partitions)
                    + _signature(index_q, index_k_rows, pages, g.router_meta, w.scores),
                    lambda: IndexScoreTop16Sm90(
                        partition_blocks=g.partition_blocks, partitions=g.partitions,
                        fp8_operand=index_q.dtype == torch.float8_e4m3fn
                    ),
                    router_args,
                ),
                router_args,
            )
            if g.partitions > 1:
                merge_args = (_cute(w.scores), _cute(w.ids), _cute(self.route))
                launches["merge"] = (
                    _compile(("merge",) + _signature(w.scores), MergeTop16Sm90, merge_args),
                    merge_args,
                )
            blocks = Int32(g.total_blocks)
            histogram_args = (
                _cute(self.route.view(-1)),
                _cute(g.block_base_of_q, 4),
                _cute(w.counts, 4),
                _cute(w.entry_segment, 4),
                _cute(w.entry_rank, 4),
                blocks,
            )
            scan_args = (
                _cute(w.counts, 4),
                _cute(w.segment_start, 4),
                _cute(g.block_token_start, 4),
                _cute(g.block_row, 4),
                _cute(g.block_local, 4),
                _cute(table, 4),
                _cute(w.work, 4),
                _cute(w.work_count, 4),
                blocks,
            )
            scatter_args = (
                _cute(w.entry_segment, 4),
                _cute(w.entry_rank, 4),
                _cute(w.segment_start, 4),
                _cute(w.relations, 4),
            )
            launches["histogram"] = (
                _compile(
                    ("histogram",) + _signature(self.route, g.block_base_of_q, w.counts),
                    RouteHistogramSm90,
                    histogram_args,
                ),
                histogram_args,
            )
            launches["scan"] = (
                _compile(
                    ("scan", paged) + _signature(w.counts, w.work, g.block_row, table),
                    lambda: ScheduleScanSm90(work_queries=WORK_QUERIES, paged=paged),
                    scan_args,
                ),
                scan_args,
            )
            launches["scatter"] = (
                _compile(
                    ("scatter",) + _signature(w.entry_segment, w.counts),
                    ScheduleScatterSm90,
                    scatter_args,
                ),
                scatter_args,
            )
            dynamic_args = tuple(_cute(t) for t in (q, k, v)) + (
                _cute(w.work, 4),
                _cute(w.work_count, 4),
                _cute(w.relations, 4),
                _cute(w.partial_out.view(torch.uint8) if fp8 else w.partial_out),
                _cute(w.partial_lse),
                _cute(w.work_count, 4),  # unused query limit: remote routes are unmasked
                softmax_scale,
            )
            launches["dynamic"] = (
                _compile(
                    ("dynamic", g.sms, paged)
                    + _signature(q, k, v, w.work, w.relations, w.partial_out),
                    lambda: DynamicAttentionSm90(ctas=g.sms, paged=paged, fp8_partial=fp8),
                    dynamic_args,
                ),
                dynamic_args,
            )
            combine_args = tuple(
                _cute(t)
                for t in (
                    w.fixed_out.view(q.shape[0], -1, 8, 128),
                    w.fixed_lse,
                    w.partial_out.view(torch.uint8) if fp8 else w.partial_out,
                    w.partial_lse,
                    self.route,
                    out,
                    a.out_lse,
                )
            )
            launches["combine"] = (
                _compile(
                    ("combine",) + _signature(q, w.partial_out, out),
                    lambda: CombineSm90(fp8_partial=fp8),
                    combine_args,
                ),
                combine_args,
            )
        a.launches = launches
        a.out = out
        a.tile_plan.page_table = page_table
        self.bound = (index_q, index_k, q, k, v, index_pages, page_table, out)

    def rebind(self, index_q, index_k, q, k, v, *, index_page_indices, page_table, out) -> None:
        """Rebind tensors with the prepared shapes, strides and dtypes.

        Logical lengths stay owned by preparation; page maps may change.
        """
        if self._mega_graph is not None:
            raise RuntimeError("cannot rebind a captured mega graph")
        new = (index_q, index_k, q, k, v, index_page_indices, page_table, out)
        names = ("index_q", "index_k", "q", "k", "v", "index_page_indices", "page_table", "out")
        for name, fresh, old in zip(names, new, self.bound):
            if (fresh is None) != (old is None):
                raise ValueError(f"{name} must keep its prepared presence")
            if fresh is not None and (
                fresh.shape, fresh.stride(), fresh.dtype, fresh.device
            ) != (old.shape, old.stride(), old.dtype, old.device):
                raise ValueError(f"{name} must retain prepared shape/stride/dtype/device")
        self.bind(*new)

    def bind_layer(self, index_q, index_k, q, k, v, out, *, index_q_scales=None):
        if index_q_scales is not None:
            raise ValueError("MXFP6 Index requires native Blackwell MX MMA")
        self.rebind(index_q, index_k, q, k, v,
                    index_page_indices=self.bound[5], page_table=self.bound[6], out=out)


def prepare_sm90_pipeline(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_lens: tuple[int, ...],
    kv_lens: tuple[int, ...],
    index_page_indices: torch.Tensor,
    page_table: torch.Tensor | None,
    partial_dtype: torch.dtype,
    overlap_fixed_score: bool | None,
    out: torch.Tensor | None,
) -> Sm90PipelinePlan:
    """Validate, allocate and compile the SM90 plan for packed BF16 prefill."""
    device = q.device
    paged = k.ndim == 4
    batch = len(q_lens)
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("the SM90 path requires BF16 Q/K/V")
    if index_q.dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn) or index_k.dtype != index_q.dtype:
        raise TypeError("the SM90 path requires matching BF16, FP16 or FP8 E4M3FN Index-Q/K")
    if q.shape[1:] != (32, 128) or v.shape != k.shape:
        raise ValueError("the SM90 path requires Q=[T,32,128] and matching K/V")
    if index_q.shape != (q.shape[0], 4, 128):
        raise ValueError("index_q must be [total_q,4,128]")
    if not q.is_contiguous() or not index_q.is_contiguous():
        raise ValueError("the SM90 path requires contiguous Q and Index-Q")
    request_blocks = [(n + BLOCK - 1) // BLOCK for n in kv_lens]
    if paged:
        if k.shape[1:] != (4, 128, 128) or k.stride(-1) != 1 or v.stride(-1) != 1:
            raise ValueError("paged K/V must be [pages,4,128,128] with contiguous D")
        if k.stride() != v.stride() or any(s % 8 for s in k.stride()[:-1]):
            raise ValueError("paged K/V strides must match and be 16-byte aligned")
        if page_table is None or page_table.ndim != 2 or page_table.dtype != torch.int32:
            raise ValueError("paged K/V requires an int32 [requests,pages] page_table")
        if page_table.shape[0] < batch or page_table.shape[1] < max(request_blocks):
            raise ValueError("page_table must cover every request block")
        if page_table.stride(1) != 1:
            raise ValueError("page_table rows must be contiguous")
    else:
        if k.shape[1:] != (4, 128) or not k.is_contiguous() or not v.is_contiguous():
            raise ValueError("packed K/V must be contiguous [total_k,4,128]")
        if k.shape[0] != sum(kv_lens):
            raise ValueError("packed K/V rows must equal the summed KV lengths")
    if index_k.reshape(-1, 128).shape[0] % BLOCK or not index_k.is_contiguous():
        raise ValueError("Index-K must be contiguous 128-token pages")
    if partial_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("partial_dtype must be BF16 or FP8 E4M3FN")
    total_q = q.shape[0]
    total_blocks = sum(request_blocks)
    sms = torch.cuda.get_device_properties(device).multi_processor_count

    # Index-K page rows: 2D tables or padded rows keep a fixed row stride.
    if index_page_indices.ndim == 2:
        page_bases = [r * index_page_indices.shape[1] for r in range(batch)]
    elif index_page_indices.numel() == total_blocks:
        page_bases = [sum(request_blocks[:r]) for r in range(batch)]
    else:
        row = index_page_indices.numel() // batch
        page_bases = [r * row for r in range(batch)]
    if any(
        base + nb > index_page_indices.numel()
        for base, nb in zip(page_bases, request_blocks)
    ):
        raise ValueError("index_page_indices must cover every logical KV block")

    q_begins, k_begins, block_bases = [0], [0], [0]
    for qn, kn, nb in zip(q_lens, kv_lens, request_blocks):
        q_begins.append(q_begins[-1] + qn)
        k_begins.append(k_begins[-1] + kn)
        block_bases.append(block_bases[-1] + nb)

    router_q_tile = (IndexScoreTop16Sm90.fp8_q_tile
                     if index_q.dtype == torch.float8_e4m3fn
                     else IndexScoreTop16Sm90.q_tile)
    router_tiles = sum((qn + router_q_tile - 1) // router_q_tile for qn in q_lens)
    partition_blocks, partitions = _router_partitions(router_tiles, max(request_blocks), sms)
    router_meta = []
    for r, (qn, kn) in enumerate(zip(q_lens, kv_lens)):
        parts = -(-request_blocks[r] // partition_blocks)
        for tile in range((qn + router_q_tile - 1) // router_q_tile):
            router_meta.append([q_begins[r], q_begins[r] + qn, tile, page_bases[r], kn, parts])
    fixed_meta = []
    for r, (qn, kn) in enumerate(zip(q_lens, kv_lens)):
        for tile in range(0, qn, TILE_QUERIES):
            fixed_meta.append([q_begins[r], qn, tile, k_begins[r], kn, r])
    request_of_q = torch.repeat_interleave(
        torch.arange(batch, device=device), torch.tensor(q_lens, device=device)
    )
    block_of = [(r, b) for r, nb in enumerate(request_blocks) for b in range(nb)]
    geometry = SimpleNamespace(
        paged=paged,
        partial_dtype=partial_dtype,
        sms=sms,
        partitions=partitions,
        partition_blocks=partition_blocks,
        total_blocks=total_blocks,
        router_meta=torch.tensor(router_meta, dtype=torch.int32, device=device),
        fixed_meta=torch.tensor(fixed_meta, dtype=torch.int32, device=device),
        block_base_of_q=torch.tensor(block_bases[:-1], dtype=torch.int32, device=device)[
            request_of_q
        ].contiguous(),
        block_token_start=torch.tensor(
            [k_begins[r] + b * BLOCK for r, b in block_of], dtype=torch.int32, device=device
        ),
        block_row=torch.tensor([r for r, _ in block_of], dtype=torch.int32, device=device),
        block_local=torch.tensor([b for _, b in block_of], dtype=torch.int32, device=device),
    )

    route = torch.full((4, total_q, TOPK), -1, dtype=torch.int32, device=device)
    out = torch.empty_like(q) if out is None else out
    out_lse = torch.empty((total_q, 32), dtype=torch.float32, device=device)
    fixed_only = all(kn <= (PREFIX_BLOCKS + LOCAL_BLOCKS) * BLOCK for kn in kv_lens)
    relations_total = 4 * total_q * TOPK
    segments = 4 * total_blocks
    max_work = -(-relations_total // WORK_QUERIES) + segments
    workspace = SimpleNamespace(
        empty_table=torch.zeros((1, 1), dtype=torch.int32, device=device)
    )
    if not fixed_only:
        workspace.__dict__.update(
            fixed_out=torch.empty_like(q),
            fixed_lse=torch.empty_like(out_lse),
            scores=torch.empty(
                (4, partitions, total_q, TOPK), dtype=torch.float32, device=device
            ),
            ids=torch.empty((4, partitions, total_q, TOPK), dtype=torch.int32, device=device),
            counts=torch.zeros(segments, dtype=torch.int32, device=device),
            segment_start=torch.empty(segments, dtype=torch.int32, device=device),
            entry_segment=torch.empty(relations_total, dtype=torch.int32, device=device),
            entry_rank=torch.empty(relations_total, dtype=torch.int32, device=device),
            relations=torch.zeros(relations_total, dtype=torch.int32, device=device),
            work=torch.zeros((max_work, 5), dtype=torch.int32, device=device),
            work_count=torch.zeros(1, dtype=torch.int32, device=device),
            partial_out=torch.empty(
                (relations_total, 8, 128), dtype=partial_dtype, device=device
            ),
            partial_lse=torch.empty(
                (relations_total, 8, 2) if partial_dtype == torch.float8_e4m3fn
                else (relations_total, 8), dtype=torch.float32, device=device),
        )
    attention = Sm90Attention(
        out=out,
        out_lse=out_lse,
        tile_plan=SimpleNamespace(dynamic_route=route, page_table=page_table),
        fixed_only=fixed_only,
        dynamic_workspace=()
        if fixed_only
        else (workspace.partial_out, workspace.partial_lse),
    )
    plan = Sm90PipelinePlan(
        attention=attention, route=route, geometry=geometry, workspace=workspace
    )
    plan.bind(index_q, index_k, q, k, v, index_page_indices, page_table, out)
    if overlap_fixed_score is None:
        overlap_fixed_score = batch == 1
    if overlap_fixed_score and not fixed_only:
        plan.overlap_fixed_score = True
        plan.fixed_stream = torch.cuda.Stream(device=device)
        plan.inputs_ready = torch.cuda.Event()
        plan.fixed_done = torch.cuda.Event()
    return plan
