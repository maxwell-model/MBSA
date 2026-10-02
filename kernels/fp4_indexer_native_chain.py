# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Experimental native-FP4 Indexer chain for Pareto measurement.

This module does not alter the production BF16 dispatch.  It connects the
vendored MiniMax SM100/SM103 block-scaled score kernel to the frozen exact
Top-32 selector, CSR builder, and BF16 sparse-attention backend.  The native
kernel currently publishes dense FP32 block scores; consequently this path is
an execution ceiling, not a production candidate at long context.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kernels.fp4_indexer_quant import quantize_mxfp4, quantize_nvfp4
from kernels.fp4_indexer_quant_triton import NativeFp4QuantizedBuffers
from kernels.msa_block_sparse.minimax_index_topk_triton import minimax_m3_index_topk
from kernels.msa_sparse_attn_prefill import build_csr
from kernels.unified_block_sparse_plan import SPEC
from kernels.unified_block_sparse_prefill import UnifiedAlignedPrefillPlan


def _native_fp4_symbols():
    # Importing the sparse shim registers the historical top-level ``cute``
    # source directory used by the vendored package. Keep this lazy so CPU-only
    # source and manifest tools do not initialize CUDA or CuTe DSL.
    from fmha_sm100.sparse import fp4_indexer_block_scores  # noqa: PLC0415
    from fp4_indexer_interface import (  # noqa: PLC0415
        fp4_indexer_reorder_scales_for_mma_cute,
    )

    return fp4_indexer_block_scores, fp4_indexer_reorder_scales_for_mma_cute


def _mma_scale_storage(scale_view: torch.Tensor) -> torch.Tensor:
    """Convert the native semantic scale view to contiguous storage order."""

    storage = scale_view.permute(5, 2, 4, 0, 1, 3)
    if not storage.is_contiguous():
        raise RuntimeError("native FP4 MMA scale storage is not contiguous")
    return storage


@dataclass(frozen=True)
class NativeFp4Operands:
    q: torch.Tensor
    k: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    q_inverse_global_scale: float
    k_inverse_global_scale: float
    fp4_format: str
    recipe: str
    ready_event: torch.cuda.Event

    @property
    def score_inverse_global_scale(self) -> float:
        return self.q_inverse_global_scale * self.k_inverse_global_scale

    @property
    def raw_score_to_contract_scale(self) -> float:
        """Positive scalar mapping native dot products to frozen Score values.

        The selector does not need this multiplication because a positive,
        tensor-global factor preserves exact membership and block-id ties. A
        numerical Score comparison must apply it before comparing values.
        """

        return self.score_inverse_global_scale * SPEC.head_dim**-0.5


def native_fp4_operands_from_buffers(
    buffers: NativeFp4QuantizedBuffers,
) -> NativeFp4Operands:
    """Bind allocation-free quantizer outputs to the native scorer.

    The returned event is initially recorded on the current stream.  Callers
    that refresh Q or append K on another stream must record ``ready_event``
    again after the final quantization launch and before invoking the chain.
    """

    ready_event = torch.cuda.Event(enable_timing=False)
    ready_event.record(torch.cuda.current_stream(buffers.q.device))
    return NativeFp4Operands(
        q=buffers.q,
        k=buffers.k,
        q_scale=buffers.q_scale,
        k_scale=buffers.k_scale,
        q_inverse_global_scale=float(buffers.q_inverse_global_scale),
        k_inverse_global_scale=float(buffers.k_inverse_global_scale),
        fp4_format=buffers.fp4_format,
        recipe=f"{buffers.fp4_format}_triton_preordered",
        ready_event=ready_event,
    )


def prepare_native_fp4_operands(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    *,
    fp4_format: str,
    calibration_amax_q: float = 6.0,
    calibration_amax_k: float = 6.0,
    nvfp4_recipe: str = "standard",
    nvfp4_error_metric: str = "mae",
    nvfp4_e4m3_max: float = 448.0,
) -> NativeFp4Operands:
    """Prepare packed operands and native MMA scale storage outside timing.

    NVFP4 uses immutable Q/K calibration factors so historical K pages never
    need requantization. MXFP4 has no global scalar. The preparation currently
    uses the independent PyTorch oracle implementation and is therefore a
    prequantized ceiling; production quantization is measured separately.
    """

    if fp4_format == "nvfp4":
        q = quantize_nvfp4(
            index_q,
            calibration_amax=calibration_amax_q,
            recipe=nvfp4_recipe,
            error_metric=nvfp4_error_metric,
            e4m3_max=nvfp4_e4m3_max,
        )
        k = quantize_nvfp4(
            index_k,
            calibration_amax=calibration_amax_k,
            recipe=nvfp4_recipe,
            error_metric=nvfp4_error_metric,
            e4m3_max=nvfp4_e4m3_max,
        )
    elif fp4_format == "mxfp4":
        q = quantize_mxfp4(index_q)
        k = quantize_mxfp4(index_k)
    else:
        raise ValueError("fp4_format must be 'nvfp4' or 'mxfp4'")

    _, reorder_scales = _native_fp4_symbols()
    q_mma, k_mma = reorder_scales(q.scale, k.scale, fp4_format=fp4_format)
    ready_event = torch.cuda.Event(enable_timing=False)
    ready_event.record(torch.cuda.current_stream(index_q.device))
    return NativeFp4Operands(
        q=q.payload,
        k=k.payload,
        q_scale=_mma_scale_storage(q_mma),
        k_scale=_mma_scale_storage(k_mma),
        q_inverse_global_scale=float(q.inverse_global_scale),
        k_inverse_global_scale=float(k.inverse_global_scale),
        fp4_format=fp4_format,
        recipe=q.recipe,
        ready_event=ready_event,
    )


@dataclass
class NativeFp4DenseAlignedChainPlan:
    """Dense-score FP4 producer feeding the frozen aligned attention plan."""

    operands: NativeFp4Operands
    attention: UnifiedAlignedPrefillPlan
    kv_indices: torch.Tensor
    prefix_lens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    cu_page_offsets: torch.Tensor
    qo_offset: torch.Tensor
    scores: torch.Tensor
    final_route: torch.Tensor
    plan_ready: torch.cuda.Event
    fixed_stream: torch.cuda.Stream
    inputs_ready: torch.cuda.Event
    fixed_done: torch.cuda.Event

    def run_score(self) -> None:
        fp4_indexer_block_scores, _ = _native_fp4_symbols()
        fp4_indexer_block_scores(
            self.operands.q,
            self.operands.k,
            self.operands.q_scale,
            self.operands.k_scale,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
            self.cu_page_offsets,
            max_seqlen_q=int(self.operands.q.shape[0]),
            max_seqlen_k=self.attention.skv,
            kv_indices=self.kv_indices,
            fp4_format=self.operands.fp4_format,
            causal=True,
            qo_offset=self.qo_offset,
            scale_layout="preordered_mma",
            out=self.scores,
        )

    def _rebuild_dynamic_csr(self) -> None:
        schedule = self.attention.dynamic_state[2]
        if self.attention.native_scheduler_metadata is not None:
            schedule.scheduler_metadata = self.attention.native_scheduler_metadata
        self.attention.dynamic_state = build_csr(
            self.attention.tile_plan.dynamic_route,
            self.attention.cu_q,
            self.attention.cu_k,
            SPEC.block_size,
            total_k=self.attention.skv,
            max_seqlen_k=self.attention.skv,
            max_seqlen_q=self.attention.q.shape[0],
            total_rows=(self.attention.skv + SPEC.block_size - 1) // SPEC.block_size,
            num_kv_heads=SPEC.kv_heads,
            qhead_per_kv=SPEC.group_size,
            buffers=self.attention.dynamic_state,
            scratch=self.attention.dynamic_csr_scratch,
            compact_batch1_schedule=self.attention.compact_batch1_schedule,
        )
        schedule = self.attention.dynamic_state[2]
        if self.attention.use_lpt_schedule:
            if (
                self.attention.native_scheduler_metadata is None
                or self.attention.lpt_scheduler_metadata is None
            ):
                raise RuntimeError("LPT schedule workspace was not prepared")
            from src.sm100.build_k2q_csr import (  # noqa: PLC0415
                run_reorder_schedule_qcount_bucket16,
            )

            run_reorder_schedule_qcount_bucket16(
                self.attention.native_scheduler_metadata,
                self.attention.lpt_scheduler_metadata,
                schedule.work_count,
                schedule.target_q_per_cta,
            )
            schedule.scheduler_metadata = self.attention.lpt_scheduler_metadata
        self.attention.tile_plan.dynamic_csr = self.attention.dynamic_state

    def run_plan(self) -> None:
        # For an 8K query chunk at total KV >= 16K, Prefix-8 and Local-8 are
        # disjoint for every row. The exact selector therefore writes the
        # Dynamic-16 in slots [16:32]. The 8K square boundary remains on the
        # production BF16 path because this ownership statement is false for
        # its earliest rows.
        if self.attention.skv < 16384 or self.attention.q.shape[0] != 8192:
            raise ValueError("native FP4 dense chain currently supports Sq=8K, Skv>=16K")
        minimax_m3_index_topk(
            self.scores.transpose(1, 2),
            self.cu_seqlens_q,
            self.prefix_lens,
            int(self.attention.q.shape[0]),
            32,
            8,
            8,
            self.final_route,
            _packed_exact=True,
            _packed_width=32,
        )
        self.attention.tile_plan.dynamic_route.copy_(self.final_route[:, :, 16:32])
        self._rebuild_dynamic_csr()

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        current = torch.cuda.current_stream(self.operands.q.device)
        current.wait_event(self.operands.ready_event)
        current.wait_event(self.plan_ready)
        self.inputs_ready.record(current)
        self.run_score()
        self.fixed_stream.wait_event(self.inputs_ready)
        with torch.cuda.stream(self.fixed_stream):
            self.attention.run_fixed()
            self.fixed_done.record(self.fixed_stream)
        self.run_plan()
        current.wait_event(self.fixed_done)
        self.attention.run_dynamic()
        return self.attention.run_merge()


def prepare_native_fp4_dense_chain(
    *,
    operands: NativeFp4Operands,
    attention: UnifiedAlignedPrefillPlan,
    kv_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> NativeFp4DenseAlignedChainPlan:
    """Allocate the diagnostic dense-score chain outside its timed window."""

    sq = int(attention.q.shape[0])
    skv = int(attention.skv)
    if sq != 8192 or skv < 16384:
        raise ValueError("native FP4 dense chain currently supports Sq=8K, Skv>=16K")
    if tuple(operands.q.shape) != (sq, SPEC.kv_heads, SPEC.head_dim // 2):
        raise ValueError("packed FP4 Q must be [Sq,4,64]")
    pages = (skv + SPEC.block_size - 1) // SPEC.block_size
    if tuple(operands.k.shape[1:]) != (1, SPEC.block_size, SPEC.head_dim // 2):
        raise ValueError("packed FP4 K must be [pages,1,128,64]")
    if operands.k.shape[0] < pages:
        raise ValueError("packed FP4 K cache does not cover logical KV")
    device = operands.q.device
    if operands.k.device != device or attention.q.device != device:
        raise ValueError("FP4 operands and attention tensors must share one device")
    if (
        kv_indices.shape != (pages,)
        or kv_indices.dtype != torch.int32
        or kv_indices.device != device
        or not kv_indices.is_contiguous()
    ):
        raise ValueError("kv_indices must be contiguous CUDA int32 [logical_pages]")
    if bool(torch.any(kv_indices < 0)) or bool(
        torch.any(kv_indices >= operands.k.shape[0])
    ):
        raise ValueError("kv_indices contains a physical page outside packed K")
    if (
        prefix_lens.shape != (1,)
        or prefix_lens.dtype != torch.int32
        or prefix_lens.device != device
        or not prefix_lens.is_contiguous()
    ):
        raise ValueError("prefix_lens must be contiguous CUDA int32 [1]")
    if int(prefix_lens.item()) != skv - sq:
        raise ValueError("prefix_lens must encode bottom-right rectangular causal offset")
    # The public-route CSR builder currently accepts the legacy six-column
    # scheduler record.  Production BF16 can use the compact two-column path
    # because it fuses CSR construction directly from local candidate storage.
    # Keep this diagnostic adapter correct by restoring the six-column buffer;
    # report its overhead separately rather than silently calling an
    # incompatible compact builder.  A consumer-native FP4 epilogue would be
    # required to remove this adapter cost.
    if attention.compact_batch1_schedule:
        schedule = attention.dynamic_state[2]
        legacy = torch.empty(
            (schedule.work_capacity, 6), dtype=torch.int32, device=device
        )
        schedule.scheduler_metadata = legacy
        schedule.metadata_format = "legacy6"
        attention.native_scheduler_metadata = legacy
        attention.lpt_scheduler_metadata = (
            torch.empty_like(legacy) if attention.use_lpt_schedule else None
        )
        attention.compact_batch1_schedule = False
    cu_q = torch.tensor([0, sq], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, skv], dtype=torch.int32, device=device)
    cu_pages = torch.tensor([0, pages], dtype=torch.int32, device=device)
    qo_offset = torch.tensor([skv - sq], dtype=torch.int32, device=device)
    scores = torch.empty(
        (SPEC.kv_heads, pages, sq), dtype=torch.float32, device=device
    )
    final_route = torch.empty(
        (SPEC.kv_heads, sq, 32), dtype=torch.int32, device=device
    )
    plan_ready = torch.cuda.Event(enable_timing=False)
    plan_ready.record(torch.cuda.current_stream(device))
    return NativeFp4DenseAlignedChainPlan(
        operands=operands,
        attention=attention,
        kv_indices=kv_indices,
        prefix_lens=prefix_lens,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        cu_page_offsets=cu_pages,
        qo_offset=qo_offset,
        scores=scores,
        final_route=final_route,
        plan_ready=plan_ready,
        fixed_stream=torch.cuda.Stream(device=device),
        inputs_ready=torch.cuda.Event(enable_timing=False),
        fixed_done=torch.cuda.Event(enable_timing=False),
    )


__all__ = [
    "NativeFp4DenseAlignedChainPlan",
    "NativeFp4Operands",
    "native_fp4_operands_from_buffers",
    "prepare_native_fp4_dense_chain",
    "prepare_native_fp4_operands",
]
