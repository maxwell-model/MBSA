"""Reusable sparse-GQA benchmark workloads for the delivery harness."""

from __future__ import annotations

import torch

from kernels.minimax_top32 import (
    allocate_prefill_fwd_workspace,
    prefill_top32_chain,
    safe_prefill_workspace_tier,
)
from kernels.msa_indexer_score import build_prefill_plan
from kernels.msa_sparse_attn_prefill import allocate_csr_scratch, build_csr
from kernels.unified_block_sparse_plan import (
    FwdTilePlan,
    PrefillPolicy,
    split_c123_route_reference,
)
from kernels.unified_block_sparse_prefill import prepare_unified_c123_aligned_chain


BLOCK = 128
HQ = 32
HKV = 4
D = 128


class SparseGqaStepCase:
    """Prepared parent and unified sparse-GQA single-chunk workloads."""

    def __init__(
        self,
        skv: int,
        seed: int,
        *,
        fused_state: bool = False,
        partial_dtype: torch.dtype | None = None,
    ) -> None:
        if partial_dtype not in (None, torch.bfloat16, torch.float8_e4m3fn):
            raise ValueError("partial_dtype must be BF16 or FP8 E4M3")
        self.skv = int(skv)
        self.sq = min(8192, self.skv)
        self.pages = (self.skv + BLOCK - 1) // BLOCK
        self.seed = int(seed)
        self.stream = torch.cuda.Stream()
        with torch.cuda.stream(self.stream):
            torch.manual_seed(seed)
            self.index_q = torch.randn(
                (self.sq, HKV, D), dtype=torch.bfloat16, device="cuda"
            )
            self.index_k = torch.randn(
                (self.pages, 1, BLOCK, D), dtype=torch.bfloat16, device="cuda"
            )
            self.q = torch.randn(
                (self.sq, HQ, D), dtype=torch.bfloat16, device="cuda"
            )
            self.k = torch.randn(
                (self.skv, HKV, D), dtype=torch.bfloat16, device="cuda"
            )
            self.v = torch.randn_like(self.k)
            self.page_ids = torch.arange(
                self.pages, dtype=torch.int32, device="cuda"
            )
            self.cu_q = torch.tensor(
                [0, self.sq], dtype=torch.int32, device="cuda"
            )
            self.cu_k = torch.tensor(
                [0, self.skv], dtype=torch.int32, device="cuda"
            )
            self.prefix_lens = torch.tensor(
                [self.skv - self.sq], dtype=torch.int32, device="cuda"
            )
            self.seqused_k = torch.tensor(
                [self.skv], dtype=torch.int32, device="cuda"
            )
        self.stream.synchronize()
        self.index_plan = build_prefill_plan(
            torch.tensor([self.sq], dtype=torch.int32),
            torch.tensor([self.skv], dtype=torch.int32),
            qo_offset=torch.tensor([self.skv - self.sq], dtype=torch.int32),
            num_idx_heads=HKV,
            num_idx_kv_heads=1,
            page_size=BLOCK,
        )
        with torch.cuda.stream(self.stream):
            self.candidates = torch.empty(
                (HKV, int(self.index_plan[3]["max_k_tiles"]), self.sq),
                dtype=torch.float32,
                device="cuda",
            )
            self.parent_route = torch.empty(
                (HKV, self.sq, 32), dtype=torch.int32, device="cuda"
            )
            self.parent_out = torch.empty_like(self.q)
            bootstrap = self._run_parent(buffers=None, scratch=None, workspace=None)
        self.stream.synchronize()
        dynamic_cpu = split_c123_route_reference(
            bootstrap.route.cpu(), skv=self.skv, q_offset=self.skv - self.sq
        )
        with torch.cuda.stream(self.stream):
            self.parent_scratch = allocate_csr_scratch(
                head_kv=HKV,
                total_q=self.sq,
                total_rows=self.pages,
                batch=1,
                max_kv_blocks=self.pages,
                device="cuda",
            )
            self.parent_buffers = build_csr(
                self.parent_route,
                self.cu_q,
                self.cu_k,
                BLOCK,
                total_k=self.skv,
                max_seqlen_k=self.skv,
                max_seqlen_q=self.sq,
                total_rows=self.pages,
                num_kv_heads=HKV,
                qhead_per_kv=HQ // HKV,
                scratch=self.parent_scratch,
            )
            tier = safe_prefill_workspace_tier(max_seq_len=self.skv)
            if partial_dtype is not None:
                tier = "large" if partial_dtype == torch.bfloat16 else "medium"
            self.parent_workspace, self.partial_dtype = allocate_prefill_fwd_workspace(
                self.sq, device="cuda", tier=tier
            )
            tile_plan = FwdTilePlan(
                dynamic_route=dynamic_cpu.to("cuda"),
                page_table=self.page_ids.view(1, -1),
                valid_kv_tokens=self.seqused_k,
                q_offset_tokens=self.skv - self.sq,
                policy=PrefillPolicy.ALIGNED_CONTIGUOUS,
            )
            self.unified_out = torch.empty_like(self.q)
            self.unified = prepare_unified_c123_aligned_chain(
                index_q=self.index_q,
                index_k=self.index_k,
                index_plan=self.index_plan,
                index_page_indices=self.page_ids,
                candidate_storage=self.candidates,
                prefix_lens=self.prefix_lens,
                q=self.q,
                k=self.k,
                v=self.v,
                tile_plan=tile_plan,
                out=self.unified_out,
                partial_dtype=self.partial_dtype,
                fuse_fixed_state_in_dynamic_combine=fused_state,
            )
        self.stream.synchronize()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def _run_parent(self, *, buffers, scratch, workspace):
        non_sm103_score_overrides = {}
        if torch.cuda.get_device_capability(self.q.device) != (10, 3):
            non_sm103_score_overrides = {
                "score_single_correction_warp": False,
                "onlyscore_no_o_barrier": False,
                "early_c_commit": False,
            }
        return prefill_top32_chain(
            index_q=self.index_q,
            index_k=self.index_k,
            main_q=self.q,
            main_k=self.k,
            main_v=self.v,
            index_plan=self.index_plan,
            index_page_indices=self.page_ids,
            main_page_table=None,
            cu_seqlens_q=self.cu_q,
            cu_seqlens_k=self.cu_k,
            prefix_lens=self.prefix_lens,
            seqused_k=None,
            max_query_len=self.sq,
            max_seq_len=self.skv,
            total_k=self.skv,
            total_rows=self.pages,
            softmax_scale=D**-0.5,
            score_out=self.candidates,
            route_out=self.parent_route,
            csr_buffers=buffers,
            csr_scratch=scratch,
            output_out=self.parent_out,
            fwd_workspace=workspace,
            partial_dtype=(
                self.partial_dtype if hasattr(self, "partial_dtype") else torch.bfloat16
            ),
            **non_sm103_score_overrides,
        )

    def parent(self):
        return self._run_parent(
            buffers=self.parent_buffers,
            scratch=self.parent_scratch,
            workspace=self.parent_workspace,
        )

    def candidate(self):
        return self.unified.run()

    def measure(self, fn) -> float:
        with torch.cuda.stream(self.stream):
            self.start.record(self.stream)
            fn()
            self.end.record(self.stream)
        self.end.synchronize()
        return float(self.start.elapsed_time(self.end))

    def check(self) -> None:
        with torch.cuda.stream(self.stream):
            parent = self.parent()
            candidate, candidate_lse = self.candidate()
        self.stream.synchronize()
        torch.testing.assert_close(candidate, parent.output, atol=4e-2, rtol=4e-2)
        torch.testing.assert_close(candidate_lse, parent.lse, atol=4e-2, rtol=4e-2)


def build_overlap(
    base: SparseGqaStepCase,
    *,
    prepare_lpt_schedule: bool | None = None,
    dynamic_kernel_cfg=None,
    bsa_variant=None,
    use_cutedsl_router: bool = False,
):
    """Build the fixed-BSA/Score overlap candidate for a prepared case."""
    source = base.unified.attention.tile_plan
    tile_plan = FwdTilePlan(
        dynamic_route=torch.empty_like(source.dynamic_route),
        page_table=source.page_table,
        valid_kv_tokens=source.valid_kv_tokens,
        q_offset_tokens=source.q_offset_tokens,
        policy=source.policy,
    )
    optional_schedule = {}
    if bsa_variant is not None:
        from third_party.anthony_bsa_backend.flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk128 import BsaOutputLayout

        if set(bsa_variant) != {"output_layout", "scheduler"}:
            raise ValueError("BSA variant requires output_layout and scheduler")
        if bsa_variant["scheduler"] not in ("static", "clc"):
            raise ValueError("BSA scheduler must be static or clc")
        optional_schedule["fixed_output_layout"] = BsaOutputLayout(bsa_variant["output_layout"])
        optional_schedule["fixed_use_clc_scheduler"] = bsa_variant["scheduler"] == "clc"
    if prepare_lpt_schedule is not None:
        optional_schedule["prepare_lpt_schedule"] = prepare_lpt_schedule
    if dynamic_kernel_cfg is not None:
        optional_schedule["dynamic_kernel_cfg"] = dynamic_kernel_cfg
        from src.sm100.fwd.config import OutputLayout  # noqa: PLC0415

        partial_dtype = (
            torch.bfloat16
            if dynamic_kernel_cfg.output_layout is OutputLayout.SHUFFLED
            else base.partial_dtype
        )
    else:
        partial_dtype = base.partial_dtype
    plan = prepare_unified_c123_aligned_chain(
        index_q=base.index_q,
        index_k=base.index_k,
        index_plan=base.index_plan,
        index_page_indices=base.page_ids,
        candidate_storage=base.candidates,
        prefix_lens=base.prefix_lens,
        q=base.q,
        k=base.k,
        v=base.v,
        tile_plan=tile_plan,
        out=torch.empty_like(base.q),
        partial_dtype=partial_dtype,
        fuse_fixed_state_in_dynamic_combine=True,
        overlap_fixed_score=True,
        use_cutedsl_router=use_cutedsl_router,
        **optional_schedule,
    )
    return plan


# Historical benchmark and test imports retain this name during cleanup.
Case = SparseGqaStepCase
