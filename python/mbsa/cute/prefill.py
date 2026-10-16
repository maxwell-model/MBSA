# Copyright 2026 Maxwell Team.
# SPDX-License-Identifier: Apache-2.0
"""Maxwell Team host orchestration for FA4 Fixed and Fireworks Dynamic."""
from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from itertools import accumulate
import math
from typing import Any
import torch
from .src.sm100.router import indexer_local_top16, indexer_packed_top16
from .src.sm100.merge_vectorized import _merge_two_attention_states_vectorized
from .plan import FwdTilePlan, PrefillPolicy, SPEC

def stage_int32(
    device: torch.device, *arrays: tuple[Sequence[int], tuple[int, ...]]
) -> tuple[torch.Tensor, ...]:
    """Stage (values,shape) host int32 arrays in one asynchronous copy.
    Returned views share a device buffer with 16-byte-aligned offsets."""
    flat: list[int] = []
    spans = []
    for values, shape in arrays:
        flat.extend([0] * ((-len(flat)) % 4))
        spans.append((len(flat), shape))
        flat.extend(values)
    host = torch.tensor(flat or [0], dtype=torch.int32, pin_memory=True)
    staged = host.to(device, non_blocking=True)
    return tuple(
        staged[offset : offset + math.prod(shape)].view(shape)
        for offset, shape in spans
    )


def _packed_lengths(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    request_lens: Sequence[tuple[int, int]] | None,
    *,
    total_q: int,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Validate host request_lens against packed sizes without device reads.
    Callers must keep device prefix sums consistent with these lengths."""
    if cu_seqlens_q.shape != cu_seqlens_k.shape or cu_seqlens_q.numel() < 2:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must share shape [B + 1]")
    if any(t.dtype != torch.int32 or t.ndim != 1 for t in (cu_seqlens_q, cu_seqlens_k)):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be rank-1 int32")
    if any(
        not t.is_cuda or not t.is_contiguous() for t in (cu_seqlens_q, cu_seqlens_k)
    ):
        raise ValueError(
            "cu_seqlens_q and cu_seqlens_k must be contiguous CUDA tensors"
        )
    if request_lens is None:
        raise ValueError("packed preparation requires host request_lens")
    q_lens = tuple(int(q_len) for q_len, _ in request_lens)
    k_lens = tuple(int(kv_len) for _, kv_len in request_lens)
    if len(q_lens) != cu_seqlens_q.numel() - 1:
        raise ValueError("request_lens must have one entry per request")
    if any(length <= 0 for length in q_lens + k_lens):
        raise ValueError("request lengths must be positive")
    q_prefix = tuple(accumulate(q_lens, initial=0))
    k_prefix = tuple(accumulate(k_lens, initial=0))
    if q_prefix[-1] != total_q:
        raise ValueError("request_lens do not cover the packed query tokens")
    return q_prefix, k_prefix, q_lens, k_lens


@dataclass
class AttentionPlan:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    tile_plan: FwdTilePlan
    fixed_out: torch.Tensor
    fixed_lse: torch.Tensor
    dynamic_out: torch.Tensor
    dynamic_workspace: tuple
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    fixed_request_ranges: tuple
    out: torch.Tensor
    out_lse: torch.Tensor
    max_seqlen_k: int
    batch: int
    fixed_attention: Any = None
    dynamic_kvouter: Any = None
    contiguous_cache: Any = None
    q_fp8: torch.Tensor | None = None

    @property
    def compute_q(self):
        return self.q if self.q_fp8 is None else self.q_fp8

    @property
    def fixed_backend(self):
        return "fa4"

    @property
    def dynamic_backend(self):
        return "fireworks" if self.dynamic_kvouter is not None else None

    @property
    def fixed_only(self):
        return self.dynamic_kvouter is None

    def build_dynamic_from_candidates(self, candidates):
        ids = candidates.view(torch.int32)[:, 16:32, :].transpose(1, 2)
        self.tile_plan.dynamic_route.copy_(ids)
        self.dynamic_kvouter.run_index()

    def run_fixed(self):
        # Refresh live BF16 Q exactly once before either FP8 consumer. The
        # Dynamic launch joins the Fixed stream before reading this buffer.
        if self.q_fp8 is not None:
            self.q_fp8.copy_(self.q)
        self.fixed_attention.run()

    def run_dynamic(self):
        if self.dynamic_kvouter is not None:
            self.dynamic_kvouter.run_forward()
            self.dynamic_kvouter.run_combine()

    def run_merge(self):
        if self.fixed_only:
            self.out.copy_(self.fixed_out)
            self.out_lse.copy_(self.fixed_lse[0].transpose(0, 1))
        else:
            _merge_two_attention_states_vectorized(
                self.fixed_out, self.fixed_lse, self.dynamic_out,
                self.dynamic_workspace[2], self.out, self.out_lse)
        return self.out, self.out_lse

    def run(self):
        self.run_fixed()
        self.run_dynamic()
        return self.run_merge()

@dataclass
class CutedslPipelinePlan:
    """Prepared native Index routing feeding fixed/dynamic attention.

    Calls sharing this plan must be ordered: streams, events and workspaces are reused.
    """

    index_q: torch.Tensor
    index_k: torch.Tensor
    index_page_indices: torch.Tensor
    candidate_storage: torch.Tensor
    prefix_lens: torch.Tensor
    attention: AttentionPlan
    candidate_partition_blocks: int
    overlap_fixed_score: bool = False
    fixed_stream: torch.cuda.Stream | None = None
    inputs_ready: torch.cuda.Event | None = None
    fixed_done: torch.cuda.Event | None = None
    index_q_scales: torch.Tensor | None = None
    mxfp6_router: Any = None
    cutedsl_index_k: torch.Tensor | None = None
    cutedsl_index_pages: torch.Tensor | None = None
    cutedsl_logical_k_tokens: int | None = None
    cutedsl_use_page_table: bool = False
    cutedsl_partition_scores: torch.Tensor | None = None
    cutedsl_partition_ids: torch.Tensor | None = None
    cutedsl_packed_partitions: int = 1
    cutedsl_packed_pages: torch.Tensor | None = None
    cutedsl_packed_tile_metadata: torch.Tensor | None = None
    _lean192_early_router: bool = False

    _mega_graph: torch.cuda.CUDAGraph | None = field(
        default=None, init=False, repr=False
    )
    _mega_capture_stream: torch.cuda.Stream | None = field(
        default=None, init=False, repr=False
    )

    def run_score(self) -> None:
        if self.attention.fixed_only:
            return
        if self.mxfp6_router is not None:
            self.mxfp6_router.run()
            return
        if self.cutedsl_index_k is None:
            raise RuntimeError("CuTeDSL router workspace was not prepared")
        compact_ids = self.candidate_storage.view(torch.int32)
        if self.cutedsl_packed_tile_metadata is not None:
            if self.cutedsl_packed_pages is None:
                raise RuntimeError("packed CuTeDSL page table is missing")
            indexer_packed_top16(
                self.index_q,
                self.cutedsl_index_k,
                partition_blocks=self.candidate_partition_blocks,
                page_indices=self.cutedsl_packed_pages,
                tile_metadata=self.cutedsl_packed_tile_metadata,
                partitions=self.cutedsl_packed_partitions,
                out_compact_ids=compact_ids,
                out_scores=self.cutedsl_partition_scores,
                out_ids=self.cutedsl_partition_ids,
                lean_pipeline=False,
            )
            return
        if self.cutedsl_index_pages is None or self.cutedsl_logical_k_tokens is None:
            raise RuntimeError("batch-one CuTeDSL router metadata is missing")
        if self.cutedsl_partition_scores is None:
            indexer_local_top16(
                self.index_q,
                self.cutedsl_index_k,
                partition_blocks=self.candidate_partition_blocks,
                page_indices=self.cutedsl_index_pages,
                logical_k_tokens=self.cutedsl_logical_k_tokens,
                use_page_table=self.cutedsl_use_page_table,
                _lean192_early=self._lean192_early_router,
                out_compact_ids=compact_ids,
            )
        else:
            if self.cutedsl_partition_ids is None:
                raise RuntimeError("CuTeDSL partition IDs were not prepared")
            indexer_local_top16(
                self.index_q,
                self.cutedsl_index_k,
                partition_blocks=self.candidate_partition_blocks,
                page_indices=self.cutedsl_index_pages,
                logical_k_tokens=self.cutedsl_logical_k_tokens,
                use_page_table=self.cutedsl_use_page_table,
                _lean192_early=self._lean192_early_router,
                out_scores=self.cutedsl_partition_scores,
                out_ids=self.cutedsl_partition_ids,
                out_compact_ids=compact_ids,
            )
        return

    def run_plan(self) -> None:
        if not self.attention.fixed_only:
            self.attention.build_dynamic_from_candidates(self.candidate_storage)

    def _record_fixed_stream_tensors(self, *, inputs_only=False) -> None:
        """Keep Fixed storage valid across asynchronous runs and rebinding.

        Register at preparation/rebind, outside capture. The allocator waits
        for the side stream's queued work when a tensor's last owner releases it.
        """
        if not self.overlap_fixed_score or self.fixed_stream is None:
            return
        a = self.attention
        pending = [a.q, a.k, a.v, a.tile_plan.page_table] if inputs_only else [
            a.q, a.k, a.v, a.cu_q, a.cu_k,
            a.fixed_out, a.fixed_lse,
            a.tile_plan.page_table, a.tile_plan.valid_kv_tokens,
        ]
        if not inputs_only and a.fixed_attention is not None:
            pending.extend(vars(a.fixed_attention).values())
        seen = set()
        while pending:
            value = pending.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            if isinstance(value, torch.Tensor):
                value.record_stream(self.fixed_stream)
            elif isinstance(value, (tuple, list)):
                pending.extend(value)
            elif isinstance(value, dict):
                pending.extend(value.values())
            elif is_dataclass(value) and not isinstance(value, type):
                pending.extend(getattr(value, f.name) for f in fields(value))

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.attention.contiguous_cache is not None:
            self.attention.contiguous_cache.refresh()
        if self.attention.fixed_only:
            return self.attention.run()
        if self.overlap_fixed_score:
            if (self.fixed_stream is None or self.inputs_ready is None
                    or self.fixed_done is None):
                raise RuntimeError("overlapped fixed attention is not prepared")
            current = torch.cuda.current_stream(self.index_q.device)
            # A caller may choose the supplied Fixed stream for eager work or
            # capture. The same-stream case follows the ordinary serial path.
            if current != self.fixed_stream:
                self.inputs_ready.record(current)
                self.fixed_stream.wait_event(self.inputs_ready)
                try:
                    with torch.cuda.stream(self.fixed_stream):
                        try:
                            self.attention.run_fixed()
                        finally:
                            self.fixed_done.record(self.fixed_stream)
                    self.run_score()
                    self.run_plan()
                finally:
                    # Rejoin before Dynamic and before capture ends, including
                    # a failed stage launch that leaves Fixed in flight.
                    current.wait_event(self.fixed_done)
                self.attention.run_dynamic()
                return self.attention.run_merge()
        self.run_score()
        self.run_plan()
        return self.attention.run()

    def bind_layer(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        *,
        index_q_scales: torch.Tensor | None = None,
    ) -> None:
        """Point this prepared plan at another layer's tensors."""
        if self._mega_graph is not None:
            raise RuntimeError("cannot rebind a captured mega graph")
        a = self.attention
        operands = [
            ("index_q", index_q, self.index_q),
            ("index_k", index_k, self.index_k),
            ("q", q, a.q),
            ("k", k, a.k),
            ("v", v, a.v),
            ("out", out, a.out),
        ]
        if self.index_q_scales is not None or index_q_scales is not None:
            operands.append(("index_q_scales", index_q_scales, self.index_q_scales))
        for name, new, old in operands:
            if new is None or old is None or (
                new.shape,
                new.stride(),
                new.dtype,
                new.device,
            ) != (old.shape, old.stride(), old.dtype, old.device):
                raise ValueError(
                    f"{name} must keep the prepared shape/stride/dtype/device"
                )
        self.index_q, self.index_k = index_q, index_k
        if self.mxfp6_router is not None:
            self.index_q_scales = index_q_scales
            self.cutedsl_index_k = index_k
            self.mxfp6_router.rebind(
                index_q, index_q_scales, index_k, self.cutedsl_packed_pages
            )
        elif self.cutedsl_index_k is not None:
            self.cutedsl_index_k = (
                index_k.view(-1, 1, SPEC.head_dim) if index_k.ndim == 4 else index_k
            )
        a.q, a.k, a.v = q, k, v
        if a.dynamic_kvouter is not None:
            if a.tile_plan.policy == PrefillPolicy.ALIGNED_PAGED:
                cache_k, cache_v, table = k, v, a.tile_plan.page_table
            else:
                if a.contiguous_cache is not None:
                    a.contiguous_cache.rebind(k, v)
                    cache_k, cache_v = a.contiguous_cache.k, a.contiguous_cache.v
                else:
                    cache_k = k.view(-1, 128, 4, 128).permute(0, 2, 1, 3)
                    cache_v = v.view(-1, 128, 4, 128).permute(0, 2, 1, 3)
                table = a.dynamic_kvouter.block_tables
            a.dynamic_kvouter.rebind(a.compute_q, cache_k, cache_v, table)
        if a.fixed_attention is not None:
            if a.tile_plan.policy == PrefillPolicy.ALIGNED_PAGED:
                cache_k, cache_v, table = k, v, a.tile_plan.page_table
            else:
                if a.contiguous_cache is not None:
                    a.contiguous_cache.rebind(k, v)
                    cache_k, cache_v = a.contiguous_cache.k, a.contiguous_cache.v
                else:
                    cache_k = k.view(-1, 128, 4, 128).permute(0, 2, 1, 3)
                    cache_v = v.view(-1, 128, 4, 128).permute(0, 2, 1, 3)
                table = a.fixed_attention.block_tables
            a.fixed_attention.rebind(a.compute_q, cache_k, cache_v, table)
        a.out = out
        self._record_fixed_stream_tensors(inputs_only=True)

    @property
    def mega_graph_captured(self) -> bool:
        """Whether this prepared, pointer-stable plan owns a captured graph."""
        return self._mega_graph is not None

    def capture_mega_graph(
        self,
        *,
        warmups: int = 3,
        stream: torch.cuda.Stream | None = None,
    ) -> "CutedslPipelinePlan":
        """Capture the complete router, Fixed, Dynamic and merge sequence.
        Fixed overlap forks from and rejoins the capture stream inside the graph.
        Tensor addresses and prepared geometry must remain stable for the graph lifetime."""
        if warmups < 1:
            raise ValueError("mega-graph capture requires at least one warmup")
        if self._mega_graph is not None:
            return self
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cannot prepare a mega graph during another capture")

        capture_stream = stream
        if capture_stream is None:
            capture_stream = torch.cuda.Stream(device=self.index_q.device)
        # Preparation and a preceding eager run may still be queued on the
        # caller's stream. Warmup reuses the CSR and partial buffers, so it
        # must wait before touching them from the capture stream.
        capture_stream.wait_stream(torch.cuda.current_stream(self.index_q.device))
        for _ in range(warmups):
            with torch.cuda.stream(capture_stream):
                self.run()
        capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            self.run()
        self._mega_graph = graph
        self._mega_capture_stream = capture_stream
        return self

    def run_mega(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay a previously captured dual-path execution graph."""
        if self._mega_graph is None:
            raise RuntimeError("call capture_mega_graph() before run_mega()")
        self._mega_graph.replay()
        return self.attention.out, self.attention.out_lse


def prepare_attention(q, k, v, tile_plan, *, out, partial_dtype=torch.bfloat16,
                      cu_seqlens_q=None,
                      cu_seqlens_k=None, request_lens=None, check_device_inputs=True,
                      _static_identity_mask=False):
    total_q = int(q.shape[0])
    if (cu_seqlens_q is None) != (cu_seqlens_k is None):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be provided together")
    if cu_seqlens_q is None:
        if not isinstance(tile_plan.q_offset_tokens, int):
            raise ValueError(
                "batch-one preparation requires an integer q_offset_tokens"
            )
        total_k = tile_plan.q_offset_tokens + total_q
        cu_q, cu_k = stage_int32(q.device, ((0, total_q), (2,)), ((0, total_k), (2,)))
        q_prefix, k_prefix = (0, total_q), (0, total_k)
        q_lens, kv_lens = (total_q,), (total_k,)
    else:
        cu_q, cu_k = cu_seqlens_q, cu_seqlens_k
        q_prefix, k_prefix, q_lens, kv_lens = _packed_lengths(
            cu_q, cu_k, request_lens, total_q=total_q
        )
        total_k = k_prefix[-1]
        if cu_q.device != q.device or cu_k.device != q.device:
            raise ValueError("cu_seqlens_q and cu_seqlens_k must be on the Q device")
    batch = len(q_lens)
    max(q_lens)
    skv = max(kv_lens)
    sum(
        (length + SPEC.block_size - 1) // SPEC.block_size for length in kv_lens
    )
    max_kv_blocks = (skv + SPEC.block_size - 1) // SPEC.block_size
    if batch == 1 and q_lens[0] % SPEC.block_size == 0:
        tile_plan.validate(sq=q_lens[0], skv=kv_lens[0], device=q.device)
    else:
        tile_plan.validate_packed(q_lens=q_lens, kv_lens=kv_lens, device=q.device)
    if tile_plan.policy not in (
        PrefillPolicy.ALIGNED_CONTIGUOUS,
        PrefillPolicy.ALIGNED_PAGED,
    ):
        raise ValueError("unsupported SM100 prefill policy")
    if tuple(tile_plan.valid_kv_tokens.shape) != (batch,):
        raise ValueError("valid_kv_tokens must have one entry per request")
    if tuple(q.shape) != (total_q, SPEC.q_heads, SPEC.head_dim):
        raise ValueError("q must be [total_q,32,128]")
    if tile_plan.policy == PrefillPolicy.ALIGNED_CONTIGUOUS:
        if (
            tuple(k.shape) != (total_k, SPEC.kv_heads, SPEC.head_dim)
            or v.shape != k.shape
        ):
            raise ValueError("contiguous k/v must be packed [total_k,4,128]")
    else:
        if (
            tile_plan.page_table.ndim != 2
            or tile_plan.page_table.shape[0] != batch
            or tile_plan.page_table.shape[1] < max_kv_blocks
            or not tile_plan.page_table.is_contiguous()
        ):
            raise ValueError("paged page_table must be contiguous [B,>=max_kv_blocks]")
        if (
            k.ndim != 4
            or tuple(k.shape[1:])
            != (
                SPEC.kv_heads,
                SPEC.block_size,
                SPEC.head_dim,
            )
            or v.shape != k.shape
        ):
            raise ValueError("paged k/v must be [physical_pages,4,128,128]")
        for batch_idx, kv_len in enumerate(kv_lens if check_device_inputs else ()):
            logical_pages = (kv_len + SPEC.block_size - 1) // SPEC.block_size
            request_pages = tile_plan.page_table[batch_idx, :logical_pages]
            if bool(torch.any(request_pages < 0)) or bool(
                torch.any(request_pages >= k.shape[0])
            ):
                raise ValueError("page_table contains a physical page outside k/v")
    if k.dtype != v.dtype or k.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("K/V must have matching BF16 or FP8 E4M3FN dtype")
    if q.dtype not in ((torch.bfloat16, torch.float8_e4m3fn)
                       if k.dtype == torch.float8_e4m3fn else (torch.bfloat16,)):
        raise ValueError("BF16 KV requires BF16 Q; FP8 KV accepts BF16 or E4M3FN Q")
    if not q.is_contiguous() or any(t.device != q.device for t in (k, v)):
        raise ValueError("q must be contiguous and q/k/v must share one CUDA device")
    if tile_plan.policy == PrefillPolicy.ALIGNED_PAGED:
        # vLLM splits an interleaved page pool into strided K/V views. Both
        # consumers carry those strides in their descriptors; copying the pool
        # would turn every attention call into a full-cache transfer.
        if any(
            t.stride(-1) != 1
            or any(s <= 0 or s % SPEC.head_dim for s in t.stride()[:-1])
            for t in (k, v)
        ):
            raise ValueError("paged k/v require positive strides aligned to D rows")
    elif any(not t.is_contiguous() for t in (k, v)):
        raise ValueError("contiguous k/v must be contiguous")
    if out.shape != q.shape or out.dtype != torch.bfloat16 or out.device != q.device:
        raise ValueError("out must be BF16 with Q shape and device")
    if check_device_inputs and tuple(tile_plan.valid_kv_tokens.tolist()) != kv_lens:
        raise ValueError("valid_kv_tokens does not match per-request KV lengths")
    if partial_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("partials must be BF16 or FP8 E4M3FN")
    from .fixed_attention import PreparedFixedAttention
    from .kvouter.prepared import PreparedKVOuter

    attention = AttentionPlan(
        q=q, k=k, v=v, tile_plan=tile_plan,
        fixed_out=torch.empty_like(q, dtype=torch.bfloat16),
        fixed_lse=torch.empty(1, 32, total_q, dtype=torch.float32, device=q.device),
        dynamic_out=torch.empty_like(q, dtype=torch.bfloat16), dynamic_workspace=(), cu_q=cu_q, cu_k=cu_k,
        fixed_request_ranges=tuple((q_prefix[i], q_prefix[i+1], k_prefix[i], k_prefix[i+1]) for i in range(batch)),
        out=out, out_lse=torch.empty(total_q, 32, dtype=torch.float32, device=q.device),
        max_seqlen_k=skv, batch=batch)
    if k.dtype == torch.float8_e4m3fn and q.dtype == torch.bfloat16:
        attention.q_fp8 = q.to(torch.float8_e4m3fn)
    compute_q = attention.compute_q
    paged = tile_plan.policy == PrefillPolicy.ALIGNED_PAGED
    if paged:
        cache_k, cache_v, table = k, v, tile_plan.page_table
    elif any(n % 128 for n in kv_lens):
        from .cache_pages import ContiguousCache
        attention.contiguous_cache = ContiguousCache(k, v, kv_lens)
        cache_k, cache_v, table = (attention.contiguous_cache.k,
                                  attention.contiguous_cache.v,
                                  attention.contiguous_cache.table)
    else:
        cache_k = k.view(-1, 128, 4, 128).permute(0, 2, 1, 3)
        cache_v = v.view(-1, 128, 4, 128).permute(0, 2, 1, 3)
        table_rows = []
        for begin, end in zip(k_prefix[:-1], k_prefix[1:]):
            row = list(range(begin // 128, end // 128))
            table_rows.extend(row + [-1] * (max_kv_blocks - len(row)))
        table = stage_int32(q.device, (table_rows, (batch, max_kv_blocks)))[0]
    identity = not paged and batch == 1 and attention.contiguous_cache is None
    attention.fixed_attention = PreparedFixedAttention(
        compute_q, cache_k, cache_v, table, tile_plan.valid_kv_tokens,
        tuple(zip(q_lens, kv_lens)), out=attention.fixed_out,
        lse=attention.fixed_lse[0], pack_gqa=True, immutable_table=not paged,
        identity_table=identity, vectorized_paged_mask=not identity,
        prepared_launch=True,
        static_identity_mask=bool(identity and _static_identity_mask and
                                  cu_seqlens_q is None and total_q % 128 == 0))
    if skv > 2048:
        prepared = PreparedKVOuter(
            compute_q, cache_k, cache_v, tile_plan.dynamic_route.permute(1, 0, 2),
            table, cu_q.to(torch.int64), tile_plan.valid_kv_tokens,
            out=attention.dynamic_out, partial_dtype=partial_dtype,
            q_load_stage=(3 if compute_q.dtype == torch.bfloat16 and partial_dtype == torch.bfloat16 and torch.cuda.get_device_capability(q.device) == (10, 3) else None))
        attention.dynamic_kvouter = prepared
        attention.dynamic_workspace = (prepared.buffers["o_flat"],
                                      prepared.buffers["m_partial"],
                                      prepared.lse.transpose(0, 1))
    return attention


def prepare_pipeline(
    *,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_q_scales: torch.Tensor | None = None,
    index_page_indices: torch.Tensor,
    candidate_storage: torch.Tensor,
    prefix_lens: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tile_plan: FwdTilePlan,
    out: torch.Tensor,
    partial_dtype: torch.dtype = torch.bfloat16,
    overlap_fixed_score: bool | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    request_lens: Sequence[tuple[int, int]] | None = None,
    check_device_inputs: bool = True,
    index_pages_identity: bool | None = None,
    fixed_stream: torch.cuda.Stream | None = None,
    _static_identity_mask: bool = False,
    _lean192_early: bool = False,
) -> CutedslPipelinePlan:
    """Prepare routing and attention buffers before run().
    Packed calls supply matching host request_lens; device validation synchronizes.
    index_pages_identity declares the batch-one Index-K map. overlap_fixed_score
    overlaps Fixed with Router/index preparation on fixed_stream, then joins
    before Dynamic and the independent-state merge. Fixed-only calls run serially.
    _static_identity_mask is an internal opt-in for API-owned fixed lengths;
    direct plans and caller-supplied packed prefix sums retain scalar masks."""
    total_q = int(q.shape[0])
    if (cu_seqlens_q is None) != (cu_seqlens_k is None):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be provided together")
    if cu_seqlens_q is None:
        if not isinstance(tile_plan.q_offset_tokens, int):
            raise ValueError("batch-one chain requires an integer q_offset_tokens")
        q_prefix = (0, total_q)
        kv_lens = (tile_plan.q_offset_tokens + total_q,)
        k_prefix = (0, kv_lens[0])
        q_lens = (total_q,)
    else:
        q_prefix, k_prefix, q_lens, kv_lens = _packed_lengths(
            cu_seqlens_q, cu_seqlens_k, request_lens, total_q=total_q
        )
    batch = len(q_lens)
    skv = max(kv_lens)
    if overlap_fixed_score is None:
        # Preserve the direct prepared-plan default for packed short contexts;
        # the public batch-one API explicitly enables overlap by default.
        overlap_fixed_score = bool(batch > 1 and skv <= 8192)
    if tuple(index_q.shape) != (total_q, SPEC.kv_heads, SPEC.head_dim):
        raise ValueError("index_q must be [total_q,4,128]")
    is_mx = index_k.dtype == torch.uint8
    if is_mx:
        if index_q.dtype not in (torch.uint8, torch.float8_e4m3fn):
            raise ValueError("MXFP6 Index K requires MXFP8 Index Q")
        if index_q_scales is None:
            raise ValueError("MXFP8 Index Q requires index_q_scales")
        if index_q.stride(2) != 1 or index_q.stride(1) != 128:
            raise ValueError("MXFP8 Index Q requires contiguous head rows")
        if (
            index_q_scales.dtype != torch.int32
            or index_q_scales.ndim != 2
            or index_q_scales.shape[0] != total_q
            or index_q_scales.shape[1] < 4
            or not index_q_scales.is_contiguous()
            or index_q_scales.device != q.device
        ):
            raise ValueError(
                "index_q_scales must be contiguous int32 [T,>=4] on the Q device"
            )
        if index_k.ndim != 4 or tuple(index_k.shape[1:]) != (1, 128, 100):
            raise ValueError("MXFP6 Index K must be [pages,1,128,100]")
    else:
        if index_q_scales is not None:
            raise ValueError("index_q_scales is only valid with MXFP6 Index K")
        if (
            index_q.dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn)
            or index_k.dtype != index_q.dtype
        ):
            raise ValueError("Index Q/K must use matching BF16, FP16 or FP8 E4M3FN dtypes")
        if not index_q.is_contiguous():
            raise ValueError("Index Q/K must be contiguous on the Q device")
    if (
        any(t.device != q.device for t in (index_q, index_k))
        or not index_k.is_contiguous()
    ):
        raise ValueError("Index Q/K must be on the Q device with contiguous Index K")
    if index_page_indices.dtype != torch.int32 or index_page_indices.device != q.device:
        raise ValueError("index_page_indices must be CUDA int32 on the Q device")
    if prefix_lens.shape != (batch,) or prefix_lens.dtype != torch.int32:
        raise ValueError("prefix_lens must be int32 [B]")
    if prefix_lens.device != q.device:
        raise ValueError("prefix_lens must be on the Q device")
    expected_prefix_lens = tuple(k - q_len for q_len, k in zip(q_lens, kv_lens))
    if check_device_inputs and tuple(prefix_lens.tolist()) != expected_prefix_lens:
        raise ValueError("prefix_lens must equal per-request Skv - Sq")
    expected_storage = (SPEC.kv_heads, candidate_storage.shape[1], total_q)
    if (
        tuple(candidate_storage.shape) != expected_storage
        or candidate_storage.shape[1] < 32
        or candidate_storage.dtype != torch.float32
        or candidate_storage.device != q.device
        or not candidate_storage.is_contiguous()
    ):
        raise ValueError(
            "candidate_storage must be contiguous CUDA FP32 [4,>=32,total_q]"
        )

    blocks = (skv + SPEC.block_size - 1) // SPEC.block_size
    if blocks > 8192:
        raise ValueError("SM100 backend supports total KV up to 1M")
    partition_blocks = 1024
    blackwell_native = torch.cuda.get_device_capability(q.device) in ((10, 0), (10, 3))
    attention = prepare_attention(
        q,
        k,
        v,
        tile_plan,
        out=out,
        partial_dtype=partial_dtype,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        request_lens=request_lens,
        check_device_inputs=check_device_inputs,
        _static_identity_mask=_static_identity_mask,
    )
    cutedsl_index_k = None
    cutedsl_index_pages = None
    cutedsl_logical_k_tokens = None
    cutedsl_use_page_table = False
    cutedsl_partition_scores = None
    cutedsl_partition_ids = None
    cutedsl_packed_partitions: int = 1
    cutedsl_packed_pages = None
    cutedsl_packed_tile_metadata = None
    mxfp6_router = None
    if not attention.fixed_only:
        if not blackwell_native:
            raise ValueError("the CuTeDSL router requires SM100 or SM103")
        if is_mx:
            cutedsl_index_k = index_k
        elif index_k.ndim == 4:
            if tuple(index_k.shape[1:]) != (
                1,
                SPEC.block_size,
                SPEC.head_dim,
            ):
                raise ValueError("CuTeDSL index_k pages must be [pages,1,128,128]")
            cutedsl_index_k = index_k.view(-1, 1, SPEC.head_dim)
        elif index_k.ndim == 3:
            if batch != 1:
                raise ValueError("packed CuTeDSL routing requires paged Index K")
            if tuple(index_k.shape) != (skv, 1, SPEC.head_dim):
                raise ValueError("CuTeDSL index_k must be [Skv,1,128] when unpaged")
            cutedsl_index_k = index_k
            if skv % SPEC.block_size:
                raise ValueError(
                    "a partial logical K tail requires full paged Index K storage"
                )
        else:
            raise ValueError("unsupported CuTeDSL index_k layout")
        if not is_mx and int(cutedsl_index_k.shape[0]) % SPEC.block_size:
            raise ValueError("CuTeDSL physical Index K storage must be page aligned")
        physical_pages = (
            int(cutedsl_index_k.shape[0])
            if is_mx
            else int(cutedsl_index_k.shape[0]) // SPEC.block_size
        )
        request_blocks = tuple(
            (length + SPEC.block_size - 1) // SPEC.block_size for length in kv_lens
        )
        if index_page_indices.ndim == 2:
            if index_page_indices.shape[0] != batch or any(
                index_page_indices.shape[1] < count for count in request_blocks
            ):
                raise ValueError(
                    "2-D index_page_indices must be [B,>=max logical pages]"
                )
            request_pages = tuple(
                index_page_indices[index, :count].clone()
                for index, count in enumerate(request_blocks)
            )
        elif index_page_indices.ndim == 1:
            active_pages = sum(request_blocks)
            if index_page_indices.numel() == active_pages:
                page_offsets = [0]
                for count in request_blocks:
                    page_offsets.append(page_offsets[-1] + count)
                request_pages = tuple(
                    index_page_indices[
                        page_offsets[index] : page_offsets[index + 1]
                    ].clone()
                    for index in range(batch)
                )
            elif index_page_indices.numel() % batch == 0:
                row_width = index_page_indices.numel() // batch
                if row_width < max(request_blocks):
                    raise ValueError("flattened page-table rows are too short")
                page_rows = index_page_indices.view(batch, row_width)
                request_pages = tuple(
                    page_rows[index, :count].clone()
                    for index, count in enumerate(request_blocks)
                )
            else:
                raise ValueError(
                    "1-D index_page_indices must contain active pages or padded rows"
                )
        else:
            raise ValueError("index_page_indices must be rank 1 or 2")
        if check_device_inputs and any(
            bool(torch.any(pages < 0)) or bool(torch.any(pages >= physical_pages))
            for pages in request_pages
        ):
            raise ValueError("index_page_indices references Index K out of range")

        if batch == 1 and not is_mx:
            cutedsl_index_pages = request_pages[0]
            cutedsl_logical_k_tokens = kv_lens[0]
            if index_pages_identity is None and check_device_inputs:
                identity_pages = torch.arange(
                    request_blocks[0],
                    dtype=torch.int32,
                    device=index_page_indices.device,
                )
                index_pages_identity = torch.equal(
                    cutedsl_index_pages, identity_pages
                )
            cutedsl_use_page_table = not index_pages_identity
            cutedsl_partitions = (
                request_blocks[0] + partition_blocks - 1
            ) // partition_blocks
            if cutedsl_partitions > 1:
                partition_shape = (
                    SPEC.kv_heads,
                    cutedsl_partitions,
                    total_q,
                    SPEC.dynamic_topk,
                )
                cutedsl_partition_scores = torch.empty(
                    partition_shape,
                    dtype=torch.float32,
                    device=q.device,
                )
                cutedsl_partition_ids = torch.empty(
                    partition_shape,
                    dtype=torch.int32,
                    device=q.device,
                )
        else:
            request_partitions = tuple(
                (kv_blocks + partition_blocks - 1) // partition_blocks
                for kv_blocks in request_blocks
            )
            # Emit all requests and KV partitions in one grid. Unused
            # partitions publish (-inf, -1), so the global merge has the same
            # request-local Top-16 set without per-request kernel launches.
            cutedsl_packed_partitions = max(request_partitions)
            cutedsl_packed_pages = torch.cat(request_pages).contiguous()
            metadata: list[int] = []
            page_begin = 0
            for request_index, (q_len, kv_len, kv_blocks) in enumerate(
                zip(q_lens, kv_lens, request_blocks)
            ):
                for local_tile in range(
                    (q_len + SPEC.block_size - 1) // SPEC.block_size
                ):
                    metadata.extend(
                        (
                            q_prefix[request_index],
                            q_prefix[request_index + 1],
                            local_tile,
                            page_begin,
                            kv_len,
                            request_partitions[request_index],
                        )
                    )
                page_begin += kv_blocks
            (cutedsl_packed_tile_metadata,) = stage_int32(
                q.device, (metadata, (len(metadata) // 6, 6))
            )
            if cutedsl_packed_partitions > 1:
                partition_shape = (
                    SPEC.kv_heads,
                    cutedsl_packed_partitions,
                    total_q,
                    SPEC.dynamic_topk,
                )
                cutedsl_partition_scores = torch.empty(
                    partition_shape,
                    dtype=torch.float32,
                    device=q.device,
                )
                cutedsl_partition_ids = torch.empty(
                    partition_shape,
                    dtype=torch.int32,
                    device=q.device,
                )
    if is_mx and not attention.fixed_only:
        from .src.sm100.mx_router import prepare_mxfp6_router

        mxfp6_router = prepare_mxfp6_router(
            index_q,
            index_q_scales,
            index_k,
            cutedsl_packed_pages,
            cutedsl_packed_tile_metadata,
            max_blocks=max(request_blocks),
            route=candidate_storage.view(torch.int32),
            partition_blocks=256,
        )
    lean192_early_router = bool(
        _lean192_early and not attention.fixed_only and batch == 1
        and cu_seqlens_q is None and cu_seqlens_k is None
        and tile_plan.policy == PrefillPolicy.ALIGNED_CONTIGUOUS
        and index_pages_identity is True and not cutedsl_use_page_table
        and cutedsl_packed_tile_metadata is None and mxfp6_router is None
        and index_q.dtype == index_k.dtype == torch.bfloat16
        and q.dtype == k.dtype == v.dtype == partial_dtype == torch.bfloat16
        and tuple(index_q.shape) == (8192, 4, 128)
        and index_k.ndim == k.ndim == v.ndim == 3
        and tuple(index_k.shape) == (skv, 1, 128)
        and index_q.device == index_k.device == q.device
        and all(t.is_contiguous() for t in (index_q, index_k, q, k, v))
        and partition_blocks == 1024 and 8192 <= skv <= 262144 and skv % 8192 == 0
        and torch.cuda.get_device_capability(q.device) == (10, 3)
    )
    overlap_fixed_score = bool(overlap_fixed_score and not attention.fixed_only)
    if not overlap_fixed_score:
        fixed_stream = None
    elif fixed_stream is None:
        fixed_stream = torch.cuda.Stream(device=q.device)
    elif fixed_stream.device != q.device:
        raise ValueError("fixed_stream must be on the Q device")
    plan = CutedslPipelinePlan(
        index_q=index_q,
        index_k=index_k,
        index_q_scales=index_q_scales,
        mxfp6_router=mxfp6_router,
        index_page_indices=index_page_indices,
        candidate_storage=candidate_storage,
        prefix_lens=prefix_lens,
        attention=attention,
        candidate_partition_blocks=partition_blocks,
        overlap_fixed_score=overlap_fixed_score,
        fixed_stream=fixed_stream,
        inputs_ready=torch.cuda.Event(enable_timing=False) if overlap_fixed_score else None,
        fixed_done=torch.cuda.Event(enable_timing=False) if overlap_fixed_score else None,
        cutedsl_index_k=cutedsl_index_k,
        cutedsl_index_pages=cutedsl_index_pages,
        cutedsl_logical_k_tokens=cutedsl_logical_k_tokens,
        cutedsl_use_page_table=cutedsl_use_page_table,
        cutedsl_partition_scores=cutedsl_partition_scores,
        cutedsl_partition_ids=cutedsl_partition_ids,
        cutedsl_packed_partitions=cutedsl_packed_partitions,
        cutedsl_packed_pages=cutedsl_packed_pages,
        cutedsl_packed_tile_metadata=cutedsl_packed_tile_metadata,
        _lean192_early_router=lean192_early_router,
    )
    plan._record_fixed_stream_tensors()
    return plan
