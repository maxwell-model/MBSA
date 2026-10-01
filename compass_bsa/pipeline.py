"""Independent Opt4 revision-1 orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .attention import AttentionResult, dual_path_attention
from .csr import CsrSchedule, build_kv_centric_csr
from .router import RouterConfig, RouterResult, hierarchical_exact_topk


@dataclass
class PipelineResult:
    router: RouterResult
    csr: CsrSchedule
    attention: AttentionResult


class IndependentBsaV1:
    """Correctness-first implementation with no legacy source dependency."""

    def __init__(
        self,
        config: RouterConfig = RouterConfig(),
        *,
        queries_per_record: int = 128,
    ) -> None:
        self.config = config
        self.queries_per_record = queries_per_record

    def __call__(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        query_positions: torch.Tensor | None = None,
    ) -> PipelineResult:
        router = hierarchical_exact_topk(
            index_q,
            index_k,
            query_positions=query_positions,
            config=self.config,
        )
        num_kv_blocks = (k.shape[0] + self.config.block_size - 1) // self.config.block_size
        csr = build_kv_centric_csr(
            router.indices,
            num_kv_blocks=num_kv_blocks,
            queries_per_record=self.queries_per_record,
        )
        attention = dual_path_attention(
            q,
            k,
            v,
            router.indices,
            query_positions=query_positions,
            config=self.config,
        )
        return PipelineResult(router=router, csr=csr, attention=attention)

