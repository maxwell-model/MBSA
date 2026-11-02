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
"""Consumer-native aligned prefill backend for provisional Top-32 GQA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .route_padding import canonicalize_dynamic_top16_padding

from .merge import (
    _merge_two_attention_states_cutedsl,
)
from .msa_sparse_attn_prefill import (
    allocate_csr_scratch,
    build_csr,
    sparse_gqa_stage2_attn_fwd_msa_prefill,
)
from .plan import FwdTilePlan, PrefillPolicy, SPEC, dynamic_partial_capacity
from ..bsa.cute_dsl.sparse.bsa_attn_sm100_blk128 import (
    BsaOutputLayout,
    bsa_attn_sm100_blk128_fwd,
)


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
    if any(
        t.dtype != torch.int32 or t.ndim != 1
        for t in (cu_seqlens_q, cu_seqlens_k)
    ):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be rank-1 int32")
    if any(not t.is_cuda or not t.is_contiguous() for t in (cu_seqlens_q, cu_seqlens_k)):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be contiguous CUDA tensors")
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
class UnifiedAlignedPrefillPlan:
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
    # Benchmark-only override. Production leaves this unset and uses the MSA
    # selector so dispatch policy stays separate from kernel mechanics.
    dynamic_kernel_cfg: Any = None

    @property
    def fixed_only(self) -> bool:
        """Check whether prefix/local pages cover the maximum causal KV extent.

        Returns:
            True when max_seqlen_k<=2048; no Dynamic attention is needed.
        """
        return self.max_seqlen_k <= (SPEC.prefix_blocks + SPEC.local_swa_blocks) * SPEC.block_size

    @property
    def partial_capacity(self) -> int:
        """Read the Dynamic partial capacity selected from the maximum KV length.

        Returns:
            Integer 4, 8 or 16 from dynamic_partial_capacity(max_seqlen_k).
        """
        return dynamic_partial_capacity(self.max_seqlen_k)

    def _refresh_dynamic_output_map(self) -> None:
        if self.dynamic_kernel_cfg is None:
            return
        from ..msa.cute.src.sm100.fwd.config import OutputLayout  # noqa: PLC0415

        if self.dynamic_kernel_cfg.output_layout is not OutputLayout.SHUFFLED:
            return
        from ..msa.cute.src.sm100.prepare_scheduler import (  # noqa: PLC0415
            prepare_sparse_fwd_tma_tile_map,
        )

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
    ) -> None:
        """Publish Dynamic routes and rebuild retained CSR/schedule buffers.

        Args:
            candidate_storage: Optional local/global candidate selection storage consumed by the
                CSR builder. None consumes existing tile_plan.dynamic_route directly.
            prefix_lens: CUDA int32 prefix/context lengths used with candidate storage to exclude
                Fixed pages.
            candidate_partition_blocks: Logical KV-block span represented by a candidate partition;
                must match the supplied selection storage.
            canonicalize_dynamic_padding: Rewrite unused Dynamic slots to canonical -1 when True
                (default False).

        Returns:
            None; overwrites Dynamic route/CSR/schedule storage and optional output tile maps on
            the current stream. Retain storage and order reuse through completion.
        """
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
            from ..msa.cute.src.sm100.build_k2q_csr import (  # noqa: PLC0415
                run_reorder_schedule_qcount_bucket16,
            )

            run_reorder_schedule_qcount_bucket16(
                self.native_scheduler_metadata,
                self.lpt_scheduler_metadata,
                schedule.work_count,
                schedule.target_q_per_cta,
            )
            schedule.scheduler_metadata = self.lpt_scheduler_metadata
        self._refresh_dynamic_output_map()
        self.tile_plan.dynamic_csr = self.dynamic_state

    def run_fixed(self) -> None:
        """Enqueue one packed Fixed BSA kernel into the private Fixed O/LSE buffers.

        Uses bound inputs on the current CUDA stream; keep buffers valid until completion and order
        producer/consumer reuse. Fixed-only plans skip Dynamic stages.

        Returns:
            None; modifies the stage's retained output/workspace buffers.
        """
        paged = self.tile_plan.policy == PrefillPolicy.ALIGNED_PAGED
        bsa_attn_sm100_blk128_fwd(
            self.q,
            self.k,
            self.v,
            q2k_block_index=self.route_sentinel,
            block_sparse_num=16,
            block_sizes=None,
            softmax_scale=SPEC.head_dim**-0.5,
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
            causal=not self.dynamic_route_fully_visible,
            softmax_scale=SPEC.head_dim**-0.5,
            schedule=schedule,
            out=self.dynamic_out,
            partial_dtype=self.partial_dtype,
            return_softmax_lse=True,
            workspace=self.dynamic_workspace,
            page_table=self.tile_plan.page_table if paged else None,
            seqused_k=self.tile_plan.valid_kv_tokens if paged else None,
            initial_out=self.fixed_out if self.fuse_fixed_state_in_dynamic_combine else None,
            initial_lse=self.fixed_lse if self.fuse_fixed_state_in_dynamic_combine else None,
            full_score_tiles=self.dynamic_full_score_tiles,
            kernel_cfg=self.dynamic_kernel_cfg,
            phase=phase,
        )

    def run_dynamic(self) -> None:
        """Enqueue Dynamic MSA and its split combine into retained output storage.

        Uses bound inputs on the current CUDA stream; keep buffers valid until completion and order
        producer/consumer reuse. Fixed-only plans skip Dynamic stages.

        Returns:
            None; modifies the stage's retained output/workspace buffers.
        """
        self._run_dynamic()

    def run_dynamic_msa(self) -> None:
        """Enqueue only the Dynamic MSA producer into persistent partial storage.

        Uses bound inputs on the current CUDA stream; keep buffers valid until completion and order
        producer/consumer reuse. Fixed-only plans skip Dynamic stages.

        Returns:
            None; modifies the stage's retained output/workspace buffers.
        """
        self._run_dynamic("msa")

    def run_dynamic_combine(self) -> None:
        """Enqueue only Dynamic combine from valid producer partials and refreshed Fixed state.

        Uses bound inputs on the current CUDA stream; keep buffers valid until completion and order
        producer/consumer reuse. Fixed-only plans skip Dynamic stages.

        Returns:
            None; modifies the stage's retained output/workspace buffers.
        """
        self._run_dynamic("combine")

    def run_merge(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge current Fixed/Dynamic states, or return the fused result.

        The plan is single-in-flight; input and workspace storage must remain valid through
        completion.

        Returns:
            Live (BF16 out [T,32,128], FP32 natural-log out_lse [T,32]), overwritten by subsequent
            runs.
        """
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
        """Submit Fixed, Dynamic and state merge in order on the current stream.

        The plan is single-in-flight; input and workspace storage must remain valid through
        completion.

        Returns:
            Live (BF16 out [T,32,128], FP32 natural-log out_lse [T,32]), overwritten by subsequent
            runs.
        """
        self.run_fixed()
        self.run_dynamic()
        return self.run_merge()






def prepare_unified_aligned_prefill(
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
) -> UnifiedAlignedPrefillPlan:
    """Allocate and prepare private Fixed/Dynamic attention stages outside capture.

    Args:
        q: CUDA BF16 [T,32,128] attention queries.
        k: CUDA BF16 dense [total_k,4,128] or paged [pages,4,128,128] keys, selected by
            tile_plan.policy.
        v: CUDA BF16 values matching K shape/layout/device.
        tile_plan: Validated FwdTilePlan holding request-local int32 routes/maps, used lengths and
            causal offsets.
        out: Caller-provided writable CUDA BF16 [T,32,128] final output, retained by the plan.
        partial_dtype: MSA partial-output storage dtype, BF16 by default; passed to the MSA
            interface.
        fuse_fixed_state_in_dynamic_combine: Consume Fixed O/LSE in Dynamic combine instead of a
            separate merge (default False); requires batch one.
        fixed_use_clc_scheduler: Optional Fixed BSA scheduler override; None uses the BSA process
            default, False static, True CLC.
        fixed_output_layout: BsaOutputLayout.SCATTERED (default) for contiguous [T,Hq,D], or
            NON_SCATTERED for grouped private Fixed storage. Final out remains token-major.
        compact_batch1_schedule: Whether to build compact batch-one CSR work metadata (default
            False).
        fixed_split_p_arrive: Optional Fixed probability arrival threshold 32,64 or 96 columns; None
            uses the BSA default.
        fixed_sync_first_p_release: Enable synchronization on the first Fixed probability release
            (default False); requires an explicit split threshold.
        prepare_lpt_schedule: Optional longest-work-first schedule preparation; None enables it
            only when maximum KV length is below 32768 tokens.
        dynamic_route_fully_visible: Caller proof that all Dynamic-selected blocks precede the
            local causal window; True permits noncausal Dynamic dispatch, default False.
        dynamic_full_score_tiles: Optional full-score-tile specialization; True requires
            dynamic_route_fully_visible and compatible geometry; None uses MSA inference.
        cu_seqlens_q: Optional contiguous CUDA int32 [B+1] Q prefix sums; pair with cu_seqlens_k.
            Omission selects batch one.
        cu_seqlens_k: Optional contiguous CUDA int32 [B+1] logical KV prefix sums; tails are
            allowed.
        dynamic_kernel_cfg: Optional explicit MSA Cfg override; None preserves the selected MSA
            heuristic.

    Returns:
        UnifiedAlignedPrefillPlan owning stage outputs, scratch and schedule resources while
        retaining input/output references. Preparation can read lengths to the host and compile
        kernels; calls and their queued work must be ordered.
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
            raise ValueError("batch-one preparation requires an integer q_offset_tokens")
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
    if compact_batch1_schedule and batch != 1:
        raise ValueError("compact_batch1_schedule requires batch size one")
    # The MSA fast tile contract uses batch-one address arithmetic. Ragged
    # routes remain fully visible, but use the general noncausal load path.
    if (batch != 1 or any(k % SPEC.block_size for k in kv_lens)) and dynamic_full_score_tiles:
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
        if k.ndim != 4 or tuple(k.shape[1:]) != (
            SPEC.kv_heads,
            SPEC.block_size,
            SPEC.head_dim,
        ) or v.shape != k.shape:
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
    if tile_plan.policy == PrefillPolicy.ALIGNED_CONTIGUOUS:
        if any(not t.is_contiguous() for t in (k, v)):
            raise ValueError("contiguous k/v must be contiguous")
    elif any(t.stride(-1) != 1 or any(s <= 0 or s % SPEC.head_dim for s in t.stride()[:-1]) for t in (k, v)):
        # Both native consumers carry these public interleaved-pool strides
        # directly in their descriptors; no full-cache copy is required.
        raise ValueError("paged k/v require positive strides aligned to D rows")
    if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
        raise ValueError("out must match q")
    valid_kv_tokens = tuple(int(x) for x in tile_plan.valid_kv_tokens.cpu().tolist())
    if valid_kv_tokens != kv_lens:
        raise ValueError("valid_kv_tokens does not match per-request KV lengths")

    if dynamic_kernel_cfg is None:
        from ..msa.cute.src.sm100.fwd.config import select_msa_cfg  # noqa: PLC0415

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
            causal=not dynamic_route_fully_visible,
            full_kv_blocks=inferred_full_score_tiles,
            sm=torch.cuda.get_device_capability(q.device),
            max_seqlen_k=skv,
            paged_kv=tile_plan.policy is PrefillPolicy.ALIGNED_PAGED,
            qhead_per_kv=q.shape[1] // SPEC.kv_heads,
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
        torch.empty_like(native_scheduler_metadata)
        if prepare_lpt_schedule
        else None
    )
    shuffled_output = False
    partial_capacity = dynamic_partial_capacity(skv)
    if dynamic_kernel_cfg is not None:
        from ..msa.cute.src.sm100.fwd.config import OutputLayout  # noqa: PLC0415

        shuffled_output = (
            dynamic_kernel_cfg.output_layout is OutputLayout.SHUFFLED
        )
    if shuffled_output:
        if batch != 1:
            raise ValueError("shuffled MSA output currently requires batch size one")
        from ..msa.cute.src.sm100.prepare_scheduler import (  # noqa: PLC0415
            prepare_sparse_fwd_tma_tile_map,
        )

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
            dtype=q.dtype, device=q.device,
        )
        fixed_lse = torch.full(
            (1, SPEC.q_heads, total_q), -torch.inf, dtype=torch.float32, device=q.device
        )
    elif fuse_fixed_state_in_dynamic_combine:
        # Fixed state is an input to combine, independent of caller final O.
        # Its rank selects the contiguous [Q,Hq,D] initial-state addressing.
        fixed_out = torch.zeros_like(q)
        fixed_lse = torch.full(
            (1, SPEC.q_heads, total_q), -torch.inf, dtype=torch.float32, device=q.device
        )
    else:
        fixed_out = torch.empty_like(q)
        fixed_lse = torch.empty(
            (1, SPEC.q_heads, total_q), dtype=torch.float32, device=q.device
        )
    dynamic_out = out if fuse_fixed_state_in_dynamic_combine else torch.empty_like(q)
    out_lse = torch.empty(
        (total_q, SPEC.q_heads), dtype=torch.float32, device=q.device
    )
    if shuffled_output:
        from ..msa.cute.src.sm100.fwd.config import MSA_OUTPUT_TILE_ROWS  # noqa: PLC0415

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
        [(i, tile, qb, qe, kb, ke)
         for i, (qb, qe, kb, ke) in enumerate(fixed_request_ranges)
         for tile in range(((qe - qb) * SPEC.group_size + 127) // 128)],
        dtype=torch.int32, device=q.device,
    )
    return UnifiedAlignedPrefillPlan(
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
        fixed_split_p_arrive=fixed_split_p_arrive,
        fixed_sync_first_p_release=fixed_sync_first_p_release,
        # The 8K boundary has no profitable fully-visible fixed tile.  At
        # 512K+ the Fixed-BSA share makes the measured effect smaller than the
        # promotion/noise gate, so keep the inherited cubin outside the
        # contiguous 16K--256K material-gain interval.
        # tcgen05.ld.red is SM103-specific for the supported MBSA targets.
        # SM100 uses the original load/register-reduction branch.
        # The reduced-load branch selects a warp-collective TMEM copy from
        # each row's causal page predicate. Keep its documented aligned query
        # offset: otherwise valid/padded rows can disagree within one warp.
        # Unaligned requests retain the general masked load path.
        fixed_tmem_load_reduce=(torch.cuda.get_device_capability(q.device) == (10, 3)
                                and 16384 <= skv <= 262144
                                and all(((ke - kb) - (qe - qb)) % SPEC.block_size == 0
                                        for qb, qe, kb, ke in fixed_request_ranges)),
        native_scheduler_metadata=native_scheduler_metadata,
        lpt_scheduler_metadata=lpt_scheduler_metadata,
        use_lpt_schedule=bool(prepare_lpt_schedule),
        dynamic_kernel_cfg=dynamic_kernel_cfg,
    )
