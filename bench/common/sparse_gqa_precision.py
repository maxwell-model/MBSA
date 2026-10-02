"""Benchmark-only low-precision Indexer adapters for the V3 Prefill chain.

The adapters reuse V3's Score/CSR/BSA/MSA/combine objects and change only the
prepared Index Q/K operands.  They provide two explicit timing scopes:
``prepared`` starts after Q and the visible K cache are E4M3, while ``e2e``
includes current-Q plus newly appended K conversion on every 8K iteration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kernels.fp8_indexer_quant_triton import (
    quantize_k_per_page_out,
    quantize_q_per_block_out,
    quantize_raw_unit_out,
    quantize_static_out,
)
from kernels.fp8_raw_quant_native import (
    quantize_block_pair_out,
    quantize_raw_pair_out,
    quantize_static_pair_out,
)
from kernels.indexer_precision import IndexerPrecision, parse_indexer_precision


_DEFAULT_STATIC_SCALE = 6.0 / 448.0


@dataclass
class Fp8PreparedOperands:
    mode: IndexerPrecision
    source_q: torch.Tensor
    source_k: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    q_scale: float | None
    k_scale: float | None
    q_block_scale: torch.Tensor | None
    k_block_scale: torch.Tensor | None

    @classmethod
    def allocate(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        mode: str | IndexerPrecision,
        *,
        q_scale: float = _DEFAULT_STATIC_SCALE,
        k_scale: float = _DEFAULT_STATIC_SCALE,
    ) -> "Fp8PreparedOperands":
        precision = parse_indexer_precision(mode)
        if not precision.is_fp8:
            raise ValueError("FP8 prepared operands require an E4M3 precision mode")
        if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
            raise ValueError("FP8 preparation sources must be BF16")
        q_out = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        k_out = torch.empty_like(k, dtype=torch.float8_e4m3fn)
        q_block = None
        k_block = None
        scalar_q = None
        scalar_k = None
        if precision is IndexerPrecision.PER_TENSOR_E4M3:
            scalar_q = float(q_scale)
            scalar_k = float(k_scale)
        elif precision is IndexerPrecision.PER_BLOCK_E4M3:
            q_block = torch.empty(
                ((q.shape[0] + 127) // 128, 4),
                dtype=torch.float32,
                device=q.device,
            )
            k_block = torch.empty((k.shape[0],), dtype=torch.float32, device=k.device)
        result = cls(
            precision,
            q,
            k,
            q_out,
            k_out,
            scalar_q,
            scalar_k,
            q_block,
            k_block,
        )
        result.prepare_all()
        return result

    def prepare_all(self) -> None:
        """Populate every buffer outside a timed window and prebuild kernels."""
        if self.mode is IndexerPrecision.RAW_E4M3:
            quantize_raw_unit_out(self.source_q, self.q)
            quantize_raw_unit_out(self.source_k, self.k)
        elif self.mode is IndexerPrecision.PER_TENSOR_E4M3:
            quantize_static_out(self.source_q, self.q, self.q_scale)
            quantize_static_out(self.source_k, self.k, self.k_scale)
        else:
            quantize_q_per_block_out(self.source_q, self.q, self.q_block_scale)
            quantize_k_per_page_out(self.source_k, self.k, self.k_block_scale)

    def prepare_increment(
        self,
        *,
        q_begin: int,
        q_end: int,
        k_page_begin: int,
        k_page_end: int,
    ) -> None:
        """Quantize current Q and only newly appended physical K pages."""
        q_src = self.source_q[q_begin:q_end]
        q_dst = self.q[q_begin:q_end]
        k_src = self.source_k[k_page_begin:k_page_end]
        k_dst = self.k[k_page_begin:k_page_end]
        if self.mode is IndexerPrecision.RAW_E4M3:
            quantize_raw_pair_out(q_src, q_dst, k_src, k_dst)
        elif self.mode is IndexerPrecision.PER_TENSOR_E4M3:
            quantize_static_pair_out(
                q_src, q_dst, self.q_scale, k_src, k_dst, self.k_scale
            )
        else:
            q_block_begin = q_begin // 128
            q_block_end = (q_end + 127) // 128
            quantize_block_pair_out(
                q_src,
                q_dst,
                self.q_block_scale[q_block_begin:q_block_end],
                k_src,
                k_dst,
                self.k_block_scale[k_page_begin:k_page_end],
            )

    def configure_plan(self, plan, *, q_begin: int, q_end: int, pages: int):
        logical_k_scale = None
        if self.k_block_scale is not None:
            max_k_tiles = int(plan.index_plan[3]["max_k_tiles"])
            logical_k_scale = torch.ones(
                (max_k_tiles,), dtype=torch.float32, device=self.k_block_scale.device
            )
            physical_pages = plan.index_page_indices[:pages].to(torch.long)
            logical_k_scale[:pages] = self.k_block_scale[physical_pages]
        return plan.with_indexer_precision(
            index_q=self.q[q_begin:q_end],
            index_k=self.k[:pages],
            indexer_precision=self.mode,
            index_q_scale=self.q_scale,
            index_k_scale=self.k_scale,
            index_k_block_scale=(
                logical_k_scale
            ),
        )


class PrecisionPlanAdapter:
    """Delegate V3 phases while optionally charging incremental preparation."""

    def __init__(self, plan, operands: Fp8PreparedOperands, *, include_preparation: bool):
        self.plan = plan
        self.operands = operands
        self.include_preparation = bool(include_preparation)
        self.attention = plan.attention

    def _prepare(self) -> None:
        if not self.include_preparation:
            return
        pages = self.operands.k.shape[0]
        appended = min(64, pages)
        self.operands.prepare_increment(
            q_begin=0,
            q_end=self.operands.q.shape[0],
            k_page_begin=pages - appended,
            k_page_end=pages,
        )

    def run(self):
        self._prepare()
        return self.plan.run()

    def run_score(self):
        self._prepare()
        return self.plan.run_score()

    def run_plan(self):
        return self.plan.run_plan()


def adapt_step_plan(
    plan,
    *,
    source_q: torch.Tensor,
    source_k: torch.Tensor,
    precision: str | IndexerPrecision,
    timing_scope: str,
) -> PrecisionPlanAdapter:
    if timing_scope not in ("prepared", "e2e"):
        raise ValueError("FP8 timing scope must be prepared or e2e")
    operands = Fp8PreparedOperands.allocate(source_q, source_k, precision)
    configured = operands.configure_plan(
        plan, q_begin=0, q_end=source_q.shape[0], pages=source_k.shape[0]
    )
    return PrecisionPlanAdapter(
        configured, operands, include_preparation=timing_scope == "e2e"
    )


class PrecisionOverlapFullPrefill:
    """Composition wrapper retaining the exact V3 paged/full-prefill chain."""

    def __init__(
        self,
        base,
        precision: str | IndexerPrecision,
        *,
        timing_scope: str,
    ) -> None:
        if timing_scope not in ("prepared", "e2e"):
            raise ValueError("FP8 timing scope must be prepared or e2e")
        self.base = base
        self.mode = parse_indexer_precision(precision)
        if not self.mode.is_fp8:
            raise ValueError("precision full-prefill wrapper requires E4M3")
        self.timing_scope = timing_scope
        self.operands = Fp8PreparedOperands.allocate(
            base.index_q_full, base.index_k_full, self.mode
        )
        self.plans = []
        for chunk, plan in zip(base.chunks, base.unified_plans):
            self.plans.append(
                None
                if plan is None
                else self.operands.configure_plan(
                    plan,
                    q_begin=chunk.start,
                    q_end=chunk.end,
                    pages=chunk.pages,
                )
            )
        self.stream = base.stream
        self.chunks = base.chunks
        self.validated_chunks = 0
        self._check_replay()

    def _run_once(self) -> None:
        previous_pages = 0
        for chunk, plan in zip(self.chunks, self.plans):
            if self.timing_scope == "e2e":
                self.operands.prepare_increment(
                    q_begin=chunk.start,
                    q_end=chunk.end,
                    k_page_begin=previous_pages,
                    k_page_end=chunk.pages,
                )
            if plan is None:
                self.base._run_parent_chunk(chunk)
            else:
                plan.run()
            previous_pages = chunk.pages

    def _check_replay(self) -> None:
        with torch.cuda.stream(self.stream):
            self._run_once()
            expected_out = self.chunks[-1].output.clone()
            final = self.plans[-1]
            expected_lse = (
                self.chunks[-1].large_workspace[2].clone()
                if final is None
                else final.attention.out_lse.clone()
            )
            self._run_once()
        self.stream.synchronize()
        final = self.plans[-1]
        actual_lse = (
            self.chunks[-1].large_workspace[2]
            if final is None
            else final.attention.out_lse
        )
        torch.testing.assert_close(self.chunks[-1].output, expected_out, rtol=0, atol=0)
        torch.testing.assert_close(actual_lse, expected_lse, rtol=0, atol=0)
        self.validated_chunks = len(self.chunks)

    def run_mode(self, overlap: bool) -> None:
        for plan in self.plans:
            if plan is not None:
                plan.overlap_fixed_score = overlap
        with torch.cuda.stream(self.stream):
            self._run_once()

    def measure_mode(self, overlap: bool) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.stream):
            start.record(self.stream)
            self.run_mode(overlap)
            end.record(self.stream)
        end.synchronize()
        return float(start.elapsed_time(end))


__all__ = [
    "Fp8PreparedOperands",
    "PrecisionOverlapFullPrefill",
    "PrecisionPlanAdapter",
    "adapt_step_plan",
]
