# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Modified for MBSA: package integration; see root NOTICE.
"""BF16 CuTe DSL router, fixed/dynamic attention, and state reduction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from mbsa.cute.src.sm100.router import bf16_indexer_local_top16, bf16_indexer_packed_top16
from mbsa.cute.src.sm100.merge import _merge_two_attention_states_cutedsl
from mbsa.cute.msa import allocate_csr_scratch, build_csr, sparse_gqa_stage2_attn_fwd_msa_prefill
from mbsa.cute.plan import FwdTilePlan, PrefillPolicy, SPEC, dynamic_partial_capacity
from mbsa.cute.src.sm100.fixed.interface import BsaOutputLayout, bsa_attn_sm100_blk128_fwd


def _packed_lengths(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    total_q: int,
    total_k: int,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Read preparation-time packed lengths and reject malformed prefix sums."""
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
    q_prefix = tuple(int(x) for x in cu_seqlens_q.cpu().tolist())
    k_prefix = tuple(int(x) for x in cu_seqlens_k.cpu().tolist())
    if q_prefix[0] != 0 or k_prefix[0] != 0:
        raise ValueError("cu_seqlens prefix sums must start at zero")
    if q_prefix[-1] != total_q or k_prefix[-1] != total_k:
        raise ValueError("cu_seqlens endpoints do not match packed tensor sizes")
    q_lens = tuple(end - begin for begin, end in zip(q_prefix, q_prefix[1:]))
    k_lens = tuple(end - begin for begin, end in zip(k_prefix, k_prefix[1:]))
    if any(length <= 0 for length in q_lens + k_lens):
        raise ValueError("cu_seqlens must be strictly increasing")
    return q_prefix, k_prefix, q_lens, k_lens


@dataclass
class AttentionPlan:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    tile_plan: FwdTilePlan
    route_sentinel: torch.Tensor
    dynamic_state: tuple[torch.Tensor, torch.Tensor, object]
    dynamic_csr_scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    fixed_out: torch.Tensor
    fixed_lse: torch.Tensor
    dynamic_out: torch.Tensor
    dynamic_workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    fixed_request_ranges: tuple[tuple[int, int, int, int], ...]
    fixed_packed_tiles: torch.Tensor
    out: torch.Tensor
    out_lse: torch.Tensor
    partial_dtype: torch.dtype
    max_seqlen_k: int
    total_k: int
    total_rows: int
    max_seqlen_q: int
    batch: int
    fuse_fixed_state_in_dynamic_combine: bool = False
    dynamic_route_fully_visible: bool = False
    dynamic_full_score_tiles: bool | None = None
    fixed_use_clc_scheduler: bool | None = None
    fixed_output_layout: BsaOutputLayout = BsaOutputLayout.SCATTERED
    compact_batch1_schedule: bool = False
    fixed_split_p_arrive: int | None = None
    fixed_sync_first_p_release: bool = False
    fixed_tmem_load_reduce: bool = False
    native_scheduler_metadata: torch.Tensor | None = None
    lpt_scheduler_metadata: torch.Tensor | None = None
    use_lpt_schedule: bool = False
    dynamic_kernel_cfg: Any = None

    @property
    def fixed_only(self) -> bool:
        """Prefix8 and local8 cover every causal block through KV2048."""
        return (
            self.max_seqlen_k
            <= (SPEC.prefix_blocks + SPEC.local_swa_blocks) * SPEC.block_size
        )

    @property
    def partial_capacity(self) -> int:
        return dynamic_partial_capacity(self.max_seqlen_k)

    def _refresh_dynamic_output_map(self) -> None:
        if self.dynamic_kernel_cfg is None:
            return
        from mbsa.cute.src.sm100.fwd.config import OutputLayout  # noqa: PLC0415

        if self.dynamic_kernel_cfg.output_layout is not OutputLayout.SHUFFLED:
            return
        from mbsa.cute.src.sm100.prepare_scheduler import prepare_sparse_fwd_tma_tile_map

        row_ptr, _, schedule = self.dynamic_state
        prepare_sparse_fwd_tma_tile_map(
            schedule=schedule,
            k2q_row_ptr=row_ptr,
            cu_seqlens_q=self.cu_q,
            total_q=self.q.shape[0],
            topk=self.partial_capacity,
            head_kv=SPEC.kv_heads,
            qhead_per_kv=SPEC.group_size,
        )

    def build_dynamic_from_candidates(
        self,
        candidate_storage: torch.Tensor | None,
        prefix_lens: torch.Tensor,
        *,
        candidate_partition_blocks: int,
        canonicalize_dynamic_padding: bool = False,
        route_ready: torch.Tensor | None = None,
    ) -> None:
        """Publish dynamic16 and rebuild its native CSR from C123 candidates."""
        schedule = self.dynamic_state[2]
        if self.native_scheduler_metadata is not None:
            schedule.scheduler_metadata = self.native_scheduler_metadata
        self.dynamic_state = build_csr(
            self.tile_plan.dynamic_route,
            self.cu_q,
            self.cu_k,
            SPEC.block_size,
            total_k=self.total_k,
            max_seqlen_k=self.max_seqlen_k,
            max_seqlen_q=self.max_seqlen_q,
            total_rows=self.total_rows,
            num_kv_heads=SPEC.kv_heads,
            qhead_per_kv=SPEC.group_size,
            buffers=self.dynamic_state,
            scratch=self.dynamic_csr_scratch,
            local_top16_storage=candidate_storage,
            prefix_lens=prefix_lens if candidate_storage is not None else None,
            candidate_partition_blocks=candidate_partition_blocks,
            local_top16_dynamic_only=candidate_storage is not None,
            canonicalize_dynamic_padding=canonicalize_dynamic_padding,
            batch1_identity_rows=self.batch == 1,
            compact_batch1_schedule=self.compact_batch1_schedule,
        )
        schedule = self.dynamic_state[2]
        if self.use_lpt_schedule:
            if (
                self.native_scheduler_metadata is None
                or self.lpt_scheduler_metadata is None
            ):
                raise RuntimeError("LPT schedule workspace was not prepared")
            from mbsa.cute.src.sm100.build_k2q_csr import run_reorder_schedule_qcount_bucket16

            run_reorder_schedule_qcount_bucket16(
                self.native_scheduler_metadata,
                self.lpt_scheduler_metadata,
                schedule.work_count,
                schedule.target_q_per_cta,
                route_ready,
            )
            schedule.scheduler_metadata = self.lpt_scheduler_metadata
        elif route_ready is not None:
            from mbsa.cute.src.sm100.build_k2q_csr import run_publish_route_ready

            run_publish_route_ready(route_ready)
        self._refresh_dynamic_output_map()
        self.tile_plan.dynamic_csr = self.dynamic_state

    def run_fixed(self) -> None:
        paged = self.tile_plan.policy == PrefillPolicy.ALIGNED_PAGED
        bsa_attn_sm100_blk128_fwd(
            self.q,
            self.k,
            self.v,
            q2k_block_index=self.route_sentinel,
            block_sparse_num=16,
            block_sizes=None,
            out=self.fixed_out,
            lse=self.fixed_lse[0],
            return_lse=True,
            causal=True,
            implicit_fixed_support=True,
            logical_seqlen_k=self.max_seqlen_k if paged else self.total_k,
            page_table=self.tile_plan.page_table if paged else None,
            packed_tiles=self.fixed_packed_tiles,
            use_clc_scheduler=self.fixed_use_clc_scheduler,
            output_layout=self.fixed_output_layout,
            split_p_arrive=self.fixed_split_p_arrive,
            sync_first_p_release=self.fixed_sync_first_p_release,
            implicit_fixed_load_reduce=self.fixed_tmem_load_reduce,
        )

    def _run_dynamic(self, phase: str | None = None) -> None:
        if self.fixed_only:
            return
        row_ptr, q_indices, schedule = self.dynamic_state
        paged = self.tile_plan.policy == PrefillPolicy.ALIGNED_PAGED
        sparse_gqa_stage2_attn_fwd_msa_prefill(
            self.q,
            self.k,
            self.v,
            row_ptr,
            q_indices,
            self.partial_capacity,
            cu_seqlens_q=self.cu_q,
            cu_seqlens_k=self.cu_k,
            max_seqlen_q=self.max_seqlen_q,
            max_seqlen_k=self.max_seqlen_k,
            block_size=SPEC.block_size,
            # Only the C123 exact Dynamic-16 producer proves that every
            # selected block is outside the local causal window.  Keep the
            # general aligned-plan API causal unless that ownership proof is
            # recorded explicitly in the prepared plan.
            causal=(not self.dynamic_route_fully_visible),
            softmax_scale=SPEC.head_dim**-0.5,
            schedule=schedule,
            out=self.dynamic_out,
            partial_dtype=self.partial_dtype,
            return_softmax_lse=True,
            workspace=self.dynamic_workspace,
            page_table=self.tile_plan.page_table if paged else None,
            seqused_k=self.tile_plan.valid_kv_tokens if paged else None,
            initial_out=self.fixed_out
            if self.fuse_fixed_state_in_dynamic_combine
            else None,
            initial_lse=self.fixed_lse
            if self.fuse_fixed_state_in_dynamic_combine
            else None,
            full_score_tiles=self.dynamic_full_score_tiles,
            kernel_cfg=self.dynamic_kernel_cfg,
            phase=phase,
        )

    def run_dynamic(self) -> None:
        """Launch the production MSA producer and K2 combine sequence."""
        self._run_dynamic()

    def run_dynamic_msa(self) -> None:
        """Launch only the MSA producer into the persistent partial workspace."""
        self._run_dynamic("msa")

    def run_dynamic_combine(self) -> None:
        """Launch only K2 combine from the persistent partial workspace."""
        self._run_dynamic("combine")

    def run_merge(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fixed_only:
            if self.fixed_output_layout is BsaOutputLayout.NON_SCATTERED:
                self.out.view(-1, SPEC.kv_heads, SPEC.group_size, SPEC.head_dim).copy_(
                    self.fixed_out.permute(1, 0, 2, 3)
                )
            else:
                self.out.copy_(self.fixed_out)
            self.out_lse.copy_(self.fixed_lse[0].transpose(0, 1))
            return self.out, self.out_lse
        if self.fuse_fixed_state_in_dynamic_combine:
            return self.out, self.out_lse
        _merge_two_attention_states_cutedsl(
            self.fixed_out,
            self.fixed_lse,
            self.dynamic_out,
            self.dynamic_workspace[2],
            self.out,
            self.out_lse,
            fixed_grouped=self.fixed_output_layout is BsaOutputLayout.NON_SCATTERED,
        )
        return self.out, self.out_lse

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.run_fixed()
        self.run_dynamic()
        return self.run_merge()


@dataclass
class CutedslPipelinePlan:
    """Prepared BF16 router feeding fixed/dynamic attention."""

    index_q: torch.Tensor
    index_k: torch.Tensor
    index_page_indices: torch.Tensor
    candidate_storage: torch.Tensor
    prefix_lens: torch.Tensor
    attention: AttentionPlan
    candidate_partition_blocks: int
    csr_candidate_partition_blocks: int
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

    fuse_dynamic_padding_in_csr: bool = True
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
            bf16_indexer_packed_top16(
                self.index_q,
                self.cutedsl_index_k,
                partition_blocks=self.candidate_partition_blocks,
                page_indices=self.cutedsl_packed_pages,
                tile_metadata=self.cutedsl_packed_tile_metadata,
                partitions=self.cutedsl_packed_partitions,
                out_compact_ids=compact_ids,
                out_scores=self.cutedsl_partition_scores,
                out_ids=self.cutedsl_partition_ids,
                lean_pipeline=self.attention.max_seqlen_k <= 8192,
            )
            return
        if self.cutedsl_index_pages is None or self.cutedsl_logical_k_tokens is None:
            raise RuntimeError("batch-one CuTeDSL router metadata is missing")
        if self.cutedsl_partition_scores is None:
            bf16_indexer_local_top16(
                self.index_q,
                self.cutedsl_index_k,
                partition_blocks=self.candidate_partition_blocks,
                page_indices=self.cutedsl_index_pages,
                logical_k_tokens=self.cutedsl_logical_k_tokens,
                use_page_table=self.cutedsl_use_page_table,
                out_compact_ids=compact_ids,
            )
        else:
            if self.cutedsl_partition_ids is None:
                raise RuntimeError("CuTeDSL partition IDs were not prepared")
            bf16_indexer_local_top16(
                self.index_q,
                self.cutedsl_index_k,
                partition_blocks=self.candidate_partition_blocks,
                page_indices=self.cutedsl_index_pages,
                logical_k_tokens=self.cutedsl_logical_k_tokens,
                use_page_table=self.cutedsl_use_page_table,
                out_scores=self.cutedsl_partition_scores,
                out_ids=self.cutedsl_partition_ids,
                out_compact_ids=compact_ids,
            )
        return

    def run_plan(self, *, route_ready: torch.Tensor | None = None) -> None:
        if self.attention.fixed_only:
            return
        self.attention.build_dynamic_from_candidates(
            self.candidate_storage,
            self.prefix_lens,
            candidate_partition_blocks=self.csr_candidate_partition_blocks,
            canonicalize_dynamic_padding=self.fuse_dynamic_padding_in_csr,
            route_ready=route_ready,
        )

    @property
    def fixed_first_score_overlap(self) -> bool:
        """Whether Fixed BSA should be published before the Router launch."""
        max_seqlen_k = self.attention.max_seqlen_k
        score_first = max_seqlen_k == 32768 or 131072 <= max_seqlen_k <= 262144
        return not score_first

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.attention.fixed_only:
            return self.attention.run()
        if self.overlap_fixed_score:
            if (
                self.fixed_stream is None
                or self.inputs_ready is None
                or self.fixed_done is None
            ):
                raise RuntimeError("overlapped fixed attention is not prepared")
            current = torch.cuda.current_stream(self.index_q.device)
            self.inputs_ready.record(current)
            if self.fixed_first_score_overlap:
                # The specialized CuTeDSL Score kernel is short but launches
                # enough CTAs to transiently occupy the device.  Fixed-first
                # remains preferable at 8K--64K and at 512K+, while the
                # measured 128K--256K regime benefits from Score-first.
                self.fixed_stream.wait_event(self.inputs_ready)
                with torch.cuda.stream(self.fixed_stream):
                    self.attention.run_fixed()
                    self.fixed_done.record(self.fixed_stream)
                self.run_score()
            else:
                # Preserve the measured Score-first order for the
                # CuTeDSL 128K--256K regime, where the Router critical
                # path is long enough that Fixed-first scheduling increases
                # resource contention without hiding more work.
                self.run_score()
                self.fixed_stream.wait_event(self.inputs_ready)
                with torch.cuda.stream(self.fixed_stream):
                    self.attention.run_fixed()
                    self.fixed_done.record(self.fixed_stream)
            self.run_plan()
            current.wait_event(self.fixed_done)
            self.attention.run_dynamic()
            return self.attention.run_merge()
        self.run_score()
        self.run_plan()
        return self.attention.run()

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
        """Capture the complete dual-path chain as one replayable device unit.

        The graph preserves the Score/fixed-BSA cross-stream overlap, exact
        route construction, KV-centric dynamic BSA, and fused initial-state
        combine.  Tensor addresses and the prepared shape must remain stable
        for the lifetime of the capture.  This launch-fusion backend does not
        alter the specialized physical kernels used by either attention path.
        """
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


def prepare_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tile_plan: FwdTilePlan,
    *,
    out: torch.Tensor,
    partial_dtype: torch.dtype = torch.bfloat16,
    fuse_fixed_state_in_dynamic_combine: bool = False,
    fixed_use_clc_scheduler: bool | None = None,
    fixed_output_layout: BsaOutputLayout = BsaOutputLayout.SCATTERED,
    compact_batch1_schedule: bool = False,
    fixed_split_p_arrive: int | None = None,
    fixed_sync_first_p_release: bool = False,
    prepare_lpt_schedule: bool | None = None,
    dynamic_route_fully_visible: bool = False,
    dynamic_full_score_tiles: bool | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    dynamic_kernel_cfg: Any = None,
) -> AttentionPlan:
    """Prepare fixed BSA, dynamic MSA, and combine for packed KV.

    Omitting ``cu_seqlens_q`` and ``cu_seqlens_k`` preserves the batch-one
    path. Providing both enables packed variable-sequence Q/KV: lengths
    may end on partial tiles, and fixed BSA
    processes request-local tiles in one packed kernel launch.
    """
    if dynamic_full_score_tiles and not dynamic_route_fully_visible:
        raise ValueError(
            "full Dynamic-MSA score tiles require a fully-visible route proof"
        )
    total_q = int(q.shape[0])
    if (cu_seqlens_q is None) != (cu_seqlens_k is None):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be provided together")
    if cu_seqlens_q is None:
        if not isinstance(tile_plan.q_offset_tokens, int):
            raise ValueError(
                "batch-one preparation requires an integer q_offset_tokens"
            )
        total_k = tile_plan.q_offset_tokens + total_q
        cu_q = torch.tensor([0, total_q], dtype=torch.int32, device=q.device)
        cu_k = torch.tensor([0, total_k], dtype=torch.int32, device=q.device)
        q_prefix, k_prefix = (0, total_q), (0, total_k)
        q_lens, kv_lens = (total_q,), (total_k,)
    else:
        cu_q, cu_k = cu_seqlens_q, cu_seqlens_k
        total_k = int(cu_k[-1].item())
        q_prefix, k_prefix, q_lens, kv_lens = _packed_lengths(
            cu_q,
            cu_k,
            total_q=total_q,
            total_k=total_k,
        )
        if cu_q.device != q.device or cu_k.device != q.device:
            raise ValueError("cu_seqlens_q and cu_seqlens_k must be on the Q device")
    batch = len(q_lens)
    max_seqlen_q = max(q_lens)
    skv = max(kv_lens)
    if batch != 1 and fuse_fixed_state_in_dynamic_combine:
        raise ValueError(
            "packed prefill requires separate Fixed/Dynamic states and K2 merge"
        )
    if compact_batch1_schedule and batch != 1:
        raise ValueError("compact_batch1_schedule requires batch size one")
    # Remote routes exclude the local window, so complete logical KV pages
    # need no token mask even when queries are packed across requests.
    if any(k % SPEC.block_size for k in kv_lens) and dynamic_full_score_tiles:
        dynamic_full_score_tiles = False
    total_rows = sum(
        (length + SPEC.block_size - 1) // SPEC.block_size for length in kv_lens
    )
    max_kv_blocks = (skv + SPEC.block_size - 1) // SPEC.block_size
    if prepare_lpt_schedule is None:
        # Production uses LPT only for the short-KV static schedule. The
        # persistent five-WG schedules consume native descriptor order.
        prepare_lpt_schedule = skv < 32 * 1024
    if batch == 1 and q_lens[0] % SPEC.block_size == 0:
        tile_plan.validate(sq=q_lens[0], skv=kv_lens[0], device=q.device)
    else:
        tile_plan.validate_packed(q_lens=q_lens, kv_lens=kv_lens, device=q.device)
    if tile_plan.policy not in (
        PrefillPolicy.ALIGNED_CONTIGUOUS,
        PrefillPolicy.ALIGNED_PAGED,
    ):
        raise ValueError("unsupported unified prefill policy")
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
        for batch_idx, kv_len in enumerate(kv_lens):
            logical_pages = (kv_len + SPEC.block_size - 1) // SPEC.block_size
            request_pages = tile_plan.page_table[batch_idx, :logical_pages]
            if bool(torch.any(request_pages < 0)) or bool(
                torch.any(request_pages >= k.shape[0])
            ):
                raise ValueError("page_table contains a physical page outside k/v")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise ValueError("unified aligned prefill is BF16-only")
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
    if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
        raise ValueError("out must match q")
    valid_kv_tokens = tuple(int(x) for x in tile_plan.valid_kv_tokens.cpu().tolist())
    if valid_kv_tokens != kv_lens:
        raise ValueError("valid_kv_tokens does not match per-request KV lengths")

    if dynamic_kernel_cfg is None:
        from mbsa.cute.src.sm100.fwd.config import select_msa_cfg  # noqa: PLC0415

        inferred_full_score_tiles = (
            bool(dynamic_full_score_tiles)
            if dynamic_full_score_tiles is not None
            else (
                dynamic_route_fully_visible
                and batch == 1
                and skv % SPEC.block_size == 0
            )
        )
        dynamic_kernel_cfg = select_msa_cfg(
            causal=False,
            full_kv_blocks=inferred_full_score_tiles,
            sm=torch.cuda.get_device_capability(q.device),
            max_seqlen_k=skv,
            paged_kv=tile_plan.policy is PrefillPolicy.ALIGNED_PAGED,
        )

    # All host-side preparation, allocation, and native CSR planning happens
    # before the execution window.  ``dynamic_route`` is already Hkv-major,
    # which is the native input layout of the frozen C123 CSR builder.
    dynamic_csr_scratch = allocate_csr_scratch(
        head_kv=SPEC.kv_heads,
        total_q=total_q,
        total_rows=total_rows,
        batch=batch,
        max_kv_blocks=max_kv_blocks,
        device=q.device,
    )
    dynamic_state = build_csr(
        tile_plan.dynamic_route,
        cu_q,
        cu_k,
        SPEC.block_size,
        total_k=total_k,
        max_seqlen_k=skv,
        max_seqlen_q=max_seqlen_q,
        total_rows=total_rows,
        num_kv_heads=SPEC.kv_heads,
        qhead_per_kv=SPEC.group_size,
        scratch=dynamic_csr_scratch,
        batch1_identity_rows=batch == 1,
    )
    if compact_batch1_schedule:
        schedule = dynamic_state[2]
        schedule.scheduler_metadata = torch.empty(
            (schedule.work_capacity, 2), dtype=torch.int32, device=q.device
        )
        schedule.metadata_format = "compact_batch1_v1"
    schedule = dynamic_state[2]
    if 16384 <= skv <= 65536:
        # C305's high-cost-first ordering makes smaller qsplit owners useful:
        # more uniform CTA costs reduce the short-context tail.  Resize the
        # schedule outside execution so the device builder remains fail-safe.
        target_q_per_cta = 768
        work_capacity = max(
            1,
            total_rows * SPEC.kv_heads
            + (total_q * SPEC.dynamic_topk * SPEC.kv_heads + target_q_per_cta - 1)
            // target_q_per_cta,
        )
        schedule.scheduler_metadata = torch.empty(
            (work_capacity, schedule.scheduler_metadata.shape[1]),
            dtype=torch.int32,
            device=q.device,
        )
        schedule.target_q_per_cta = target_q_per_cta
    native_scheduler_metadata = schedule.scheduler_metadata
    lpt_scheduler_metadata = (
        torch.empty_like(native_scheduler_metadata) if prepare_lpt_schedule else None
    )
    shuffled_output = False
    partial_capacity = dynamic_partial_capacity(skv)
    if dynamic_kernel_cfg is not None:
        from mbsa.cute.src.sm100.fwd.config import OutputLayout  # noqa: PLC0415

        shuffled_output = dynamic_kernel_cfg.output_layout is OutputLayout.SHUFFLED
    if shuffled_output:
        if batch != 1:
            raise ValueError("shuffled MSA output currently requires batch size one")
        from mbsa.cute.src.sm100.prepare_scheduler import prepare_sparse_fwd_tma_tile_map

        prepare_sparse_fwd_tma_tile_map(
            schedule=schedule,
            k2q_row_ptr=dynamic_state[0],
            cu_seqlens_q=cu_q,
            total_q=total_q,
            topk=partial_capacity,
            head_kv=SPEC.kv_heads,
            qhead_per_kv=SPEC.group_size,
        )
    tile_plan.dynamic_csr = dynamic_state

    # K2 reads both initial-state operands even when LSE assigns zero weight.
    if not isinstance(fixed_output_layout, BsaOutputLayout):
        raise TypeError("fixed_output_layout must be BsaOutputLayout")
    if fixed_output_layout is BsaOutputLayout.NON_SCATTERED:
        fixed_out = torch.zeros(
            (SPEC.kv_heads, total_q, SPEC.q_heads // SPEC.kv_heads, SPEC.head_dim),
            dtype=q.dtype,
            device=q.device,
        )
        fixed_lse = torch.full(
            (1, SPEC.q_heads, total_q), -torch.inf, dtype=torch.float32, device=q.device
        )
    elif fuse_fixed_state_in_dynamic_combine:
        fixed_out = out.zero_()
        fixed_lse = torch.full(
            (1, SPEC.q_heads, total_q), -torch.inf, dtype=torch.float32, device=q.device
        )
    else:
        fixed_out = torch.empty_like(q)
        fixed_lse = torch.empty(
            (1, SPEC.q_heads, total_q), dtype=torch.float32, device=q.device
        )
    dynamic_out = out if fuse_fixed_state_in_dynamic_combine else torch.empty_like(q)
    out_lse = torch.empty((total_q, SPEC.q_heads), dtype=torch.float32, device=q.device)
    if shuffled_output:
        from mbsa.cute.src.sm100.fwd.config import MSA_OUTPUT_TILE_ROWS  # noqa: PLC0415

        tile_rows = schedule.tma_tile_capacity * MSA_OUTPUT_TILE_ROWS
        partial_o = torch.empty(
            tile_rows,
            SPEC.head_dim,
            dtype=partial_dtype,
            device=q.device,
        )
        partial_lse = torch.empty(
            tile_rows,
            dtype=torch.float32,
            device=q.device,
        )
    else:
        partial_o = torch.empty(
            partial_capacity,
            total_q,
            SPEC.q_heads,
            SPEC.head_dim,
            dtype=partial_dtype,
            device=q.device,
        )
        partial_lse = torch.empty(
            partial_capacity,
            total_q,
            SPEC.q_heads,
            dtype=torch.float32,
            device=q.device,
        )
    dynamic_workspace = (
        partial_o,
        partial_lse,
        out_lse
        if fuse_fixed_state_in_dynamic_combine
        else torch.empty(total_q, SPEC.q_heads, dtype=torch.float32, device=q.device),
    )
    # The implicit-fixed backend does not dereference route storage.  Keeping
    # a one-element device sentinel satisfies the compiled call ABI without
    # publishing prefix/SWA route records.
    route_sentinel = torch.zeros((1,), dtype=torch.int32, device=q.device)
    fixed_request_ranges = tuple(
        (q_prefix[i], q_prefix[i + 1], k_prefix[i], k_prefix[i + 1])
        for i in range(batch)
    )
    # One descriptor per 128 packed GQA rows; the scheduler never creates a
    # tile spanning two requests, including partial final Q tiles.
    fixed_packed_tiles = torch.tensor(
        [
            (i, tile, qb, qe, kb, ke)
            for i, (qb, qe, kb, ke) in enumerate(fixed_request_ranges)
            for tile in range(((qe - qb) * SPEC.group_size + 127) // 128)
        ],
        dtype=torch.int32,
        device=q.device,
    )
    return AttentionPlan(
        q=q,
        k=k,
        v=v,
        tile_plan=tile_plan,
        route_sentinel=route_sentinel,
        dynamic_state=dynamic_state,
        dynamic_csr_scratch=dynamic_csr_scratch,
        fixed_out=fixed_out,
        fixed_lse=fixed_lse,
        dynamic_out=dynamic_out,
        dynamic_workspace=dynamic_workspace,
        cu_q=cu_q,
        cu_k=cu_k,
        fixed_request_ranges=fixed_request_ranges,
        fixed_packed_tiles=fixed_packed_tiles,
        out=out,
        out_lse=out_lse,
        partial_dtype=partial_dtype,
        max_seqlen_k=skv,
        total_k=total_k,
        total_rows=total_rows,
        max_seqlen_q=max_seqlen_q,
        batch=batch,
        fuse_fixed_state_in_dynamic_combine=fuse_fixed_state_in_dynamic_combine,
        dynamic_route_fully_visible=dynamic_route_fully_visible,
        dynamic_full_score_tiles=dynamic_full_score_tiles,
        fixed_use_clc_scheduler=fixed_use_clc_scheduler,
        fixed_output_layout=fixed_output_layout,
        compact_batch1_schedule=compact_batch1_schedule,
        fixed_split_p_arrive=(fixed_split_p_arrive if batch == 1 else None),
        fixed_sync_first_p_release=(
            fixed_sync_first_p_release if batch == 1 else False
        ),
        # Split barriers and the reduced-TMEM schedule are qualified for the
        # uniform batch-one grid only.  Packed request-local launches use the
        # conservative pipeline until their independent tuning gate exists.
        # LdRed requires complete query tiles and KV blocks. Query or KV
        # tails can leave participating lanes waiting indefinitely.
        # Keep tails on the ordinary TMEM load and software reduction.
        fixed_tmem_load_reduce=(
            batch == 1
            and total_q * SPEC.group_size % 128 == 0
            and skv % SPEC.block_size == 0
            and 16384 <= skv <= 262144
        ),
        native_scheduler_metadata=native_scheduler_metadata,
        lpt_scheduler_metadata=lpt_scheduler_metadata,
        use_lpt_schedule=bool(prepare_lpt_schedule),
        dynamic_kernel_cfg=dynamic_kernel_cfg,
    )


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
    fuse_fixed_state_in_dynamic_combine: bool = False,
    overlap_fixed_score: bool | None = None,
    fixed_output_layout: BsaOutputLayout = BsaOutputLayout.SCATTERED,
    fixed_use_clc_scheduler: bool | None = None,
    compact_batch1_schedule: bool | None = None,
    fixed_split_p_arrive: int | None = 64,
    fixed_sync_first_p_release: bool = True,
    prepare_lpt_schedule: bool | None = None,
    dynamic_route_fully_visible: bool = True,
    dynamic_full_score_tiles: bool | None = True,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    dynamic_kernel_cfg: Any = None,
) -> CutedslPipelinePlan:
    """Prepare the CuTe DSL router and attention chain.

    Short KV with no dynamic support skips routing and CSR execution.
    Packed prefix sums preserve request-local route IDs.
    Consumer buffers and native CSR scratch are prepared before ``run``.
    """
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
        total_k = int(cu_seqlens_k[-1].item())
        q_prefix, k_prefix, q_lens, kv_lens = _packed_lengths(
            cu_seqlens_q,
            cu_seqlens_k,
            total_q=total_q,
            total_k=total_k,
        )
    batch = len(q_lens)
    max_seqlen_q = max(q_lens)
    skv = max(kv_lens)
    if overlap_fixed_score is None:
        # The low-resource 8K Router overlaps profitably with packed Fixed BSA.
        # Above 8K, measurements show resource contention outweighs the saved
        # launch gap, so packed callers default to sequential execution.
        overlap_fixed_score = bool(batch > 1 and skv <= 8192)
    # The two-word schedule is a measured win from 32K onward.  Retain the
    # inherited six-word schedule below that boundary, where fixed launch and
    # setup costs dominate and the compact consumer is not consistently faster.
    if compact_batch1_schedule is None:
        compact_batch1_schedule = batch == 1 and skv >= 32768
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
            index_q.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
            or index_k.dtype != index_q.dtype
        ):
            raise ValueError("Index Q/K must use matching BF16 or FP8 E4M3FN dtypes")
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
    if tuple(int(x) for x in prefix_lens.cpu().tolist()) != expected_prefix_lens:
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
        raise ValueError("unified backend supports total KV up to 1M")
    if blocks <= 1024:
        csr_partition_blocks = 1024
    elif blocks <= 2048:
        csr_partition_blocks = 2048
    elif blocks <= 4096:
        csr_partition_blocks = 4096
    else:
        csr_partition_blocks = 8192
    # Keep 1024-block partitions for the exact hierarchical router.
    partition_blocks = 1024
    blackwell_native = torch.cuda.get_device_capability(q.device) in ((10, 0), (10, 3))
    # CLC wins in isolation and from the 56K boundary onward.  In the measured
    # 24K--32K region and at the 40K/44K/48K full-query grid points, the
    # fixed branch overlaps the Score producer and static scheduling reduces
    # cross-stream resource contention.  Keep residual chunks and every other
    # shape on the inherited CLC path.
    static_fixed_region = 24576 <= skv <= 32768 or (
        40960 <= skv <= 49152 and skv % 4096 == 0
    )
    if fixed_use_clc_scheduler is None:
        if batch != 1:
            # Packed requests are described by a flat, request-local tile
            # list.  Keep that descriptor stream on the static scheduler;
            # the inherited CLC path assumes one uniform logical Q/K extent
            # and can leave warp roles consuming different request records.
            fixed_use_clc_scheduler = False
        else:
            fixed_use_clc_scheduler = not (
                overlap_fixed_score and max_seqlen_q == 8192 and static_fixed_region
            )
    attention = prepare_attention(
        q,
        k,
        v,
        tile_plan,
        out=out,
        partial_dtype=partial_dtype,
        fuse_fixed_state_in_dynamic_combine=fuse_fixed_state_in_dynamic_combine,
        fixed_use_clc_scheduler=fixed_use_clc_scheduler,
        fixed_output_layout=fixed_output_layout,
        compact_batch1_schedule=compact_batch1_schedule,
        fixed_split_p_arrive=fixed_split_p_arrive,
        fixed_sync_first_p_release=fixed_sync_first_p_release,
        prepare_lpt_schedule=prepare_lpt_schedule,
        dynamic_route_fully_visible=dynamic_route_fully_visible,
        dynamic_full_score_tiles=dynamic_full_score_tiles,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        dynamic_kernel_cfg=dynamic_kernel_cfg,
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
        if any(
            bool(torch.any(pages < 0)) or bool(torch.any(pages >= physical_pages))
            for pages in request_pages
        ):
            raise ValueError("index_page_indices references Index K out of range")

        if batch == 1 and not is_mx:
            cutedsl_index_pages = request_pages[0]
            cutedsl_logical_k_tokens = kv_lens[0]
            identity_pages = torch.arange(
                request_blocks[0],
                dtype=torch.int32,
                device=index_page_indices.device,
            )
            cutedsl_use_page_table = not torch.equal(
                cutedsl_index_pages, identity_pages
            )
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
            metadata = []
            page_begin = 0
            for request_index, (q_len, kv_len, kv_blocks) in enumerate(
                zip(q_lens, kv_lens, request_blocks)
            ):
                for local_tile in range(
                    (q_len + SPEC.block_size - 1) // SPEC.block_size
                ):
                    metadata.append(
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
            cutedsl_packed_tile_metadata = torch.tensor(
                metadata,
                dtype=torch.int32,
                device=q.device,
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
        from mbsa.cute.src.sm100.mx_router import prepare_mxfp6_router

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
    fixed_stream = torch.cuda.Stream(device=q.device) if overlap_fixed_score else None
    inputs_ready = (
        torch.cuda.Event(enable_timing=False) if overlap_fixed_score else None
    )
    fixed_done = torch.cuda.Event(enable_timing=False) if overlap_fixed_score else None
    return CutedslPipelinePlan(
        index_q=index_q,
        index_k=index_k,
        index_q_scales=index_q_scales,
        mxfp6_router=mxfp6_router,
        index_page_indices=index_page_indices,
        candidate_storage=candidate_storage,
        prefix_lens=prefix_lens,
        attention=attention,
        candidate_partition_blocks=partition_blocks,
        csr_candidate_partition_blocks=csr_partition_blocks,
        overlap_fixed_score=overlap_fixed_score,
        fixed_stream=fixed_stream,
        inputs_ready=inputs_ready,
        fixed_done=fixed_done,
        cutedsl_index_k=cutedsl_index_k,
        cutedsl_index_pages=cutedsl_index_pages,
        cutedsl_logical_k_tokens=cutedsl_logical_k_tokens,
        cutedsl_use_page_table=cutedsl_use_page_table,
        cutedsl_partition_scores=cutedsl_partition_scores,
        cutedsl_partition_ids=cutedsl_partition_ids,
        cutedsl_packed_partitions=cutedsl_packed_partitions,
        cutedsl_packed_pages=cutedsl_packed_pages,
        cutedsl_packed_tile_metadata=cutedsl_packed_tile_metadata,
        fuse_dynamic_padding_in_csr=skv < 32768,
    )


__all__ = [
    "AttentionPlan",
    "CutedslPipelinePlan",
    "prepare_attention",
    "prepare_pipeline",
]
