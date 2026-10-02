"""Prepared sequential sparse-GQA prefill workloads shared by benchmark drivers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kernels.minimax_top32 import (
    allocate_prefill_fwd_workspace,
    prefill_top32_chain,
    safe_prefill_workspace_tier,
)
from kernels.msa_indexer_score import build_prefill_plan, score_prefill
from kernels.msa_sparse_attn_prefill import allocate_csr_scratch, build_csr
from kernels.msa_topk_select import select_prefill_exact_per_group
from kernels.unified_block_sparse_plan import (
    FwdTilePlan,
    PrefillPolicy,
)
from kernels import unified_block_sparse_plan as _prefill_plan
from kernels.unified_block_sparse_prefill import prepare_unified_c123_aligned_chain

PAGE = 128
HIDX = 4
H_Q = 32
H_KV = 4
DIM = 128
TOPK = 32
SCALE = DIM**-0.5
DTYPE = torch.bfloat16


def _use_unified_aligned_prefill(*, sq: int, skv: int) -> bool:
    selector = getattr(_prefill_plan, "use_unified_aligned_prefill", None)
    if selector is not None:
        return selector(sq=sq, skv=skv)
    # The historical full-prefill baseline predates the shared selector.
    return skv >= 32768 and sq >= 129


@dataclass
class Chunk:
    start: int
    end: int
    sq: int
    skv: int
    pages: int
    plan: object
    index_q: torch.Tensor
    main_q: torch.Tensor
    index_k: torch.Tensor
    main_k: torch.Tensor
    main_v: torch.Tensor
    index_pages: torch.Tensor
    page_table: torch.Tensor
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    prefix_lens: torch.Tensor
    seqused_k: torch.Tensor
    score: torch.Tensor
    route: torch.Tensor
    csr_scratch: object
    csr_buffers: object
    output: torch.Tensor
    large_workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    medium_workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class FullPrefill:
    def __init__(
        self,
        total_kv: int,
        seed: int,
        chunk_size: int = 8192,
        *,
        validate_pair: bool = True,
        variants: tuple[str, str] = ("B0_LEGACY_BF16", "BEST_SAFE_DISPATCH"),
    ) -> None:
        if total_kv <= 0:
            raise ValueError("formal full-prefill total_kv must be positive")
        if chunk_size <= 0 or chunk_size % PAGE:
            raise ValueError("chunk_size must be positive and page aligned")
        self.total_kv = int(total_kv)
        self.seed = int(seed)
        self.chunk_size = int(chunk_size)
        if len(variants) != 2:
            raise ValueError("formal interleaving requires exactly two labels")
        self.variants = tuple(variants)
        self.stream = torch.cuda.Stream()

        descriptors: list[tuple[int, int, int, object, int]] = []
        max_score_elements = 0
        for start in range(0, self.total_kv, self.chunk_size):
            end = min(start + self.chunk_size, self.total_kv)
            sq = end - start
            plan = build_prefill_plan(
                torch.tensor([sq], dtype=torch.int32),
                torch.tensor([end], dtype=torch.int32),
                qo_offset=torch.tensor([start], dtype=torch.int32),
                num_idx_heads=HIDX,
                num_idx_kv_heads=1,
                page_size=PAGE,
            )
            if bool(plan[0]):
                raise RuntimeError("formal full-prefill benchmark does not allow a mixed plan")
            max_k_tiles = int(plan[3]["max_k_tiles"])
            descriptors.append((start, end, sq, plan, max_k_tiles))
            max_score_elements = max(max_score_elements, HIDX * max_k_tiles * sq)

        max_sq = max(item[2] for item in descriptors)
        total_pages = (self.total_kv + PAGE - 1) // PAGE
        with torch.cuda.stream(self.stream):
            torch.manual_seed(self.seed)
            self.index_q_full = torch.randn(
                (self.total_kv, HIDX, DIM), device="cuda", dtype=DTYPE
            )
            self.main_q_full = torch.randn(
                (self.total_kv, H_Q, DIM), device="cuda", dtype=DTYPE
            )
            self.index_k_full = torch.randn(
                (total_pages, 1, PAGE, DIM), device="cuda", dtype=DTYPE
            )
            self.main_k_full = torch.randn(
                (total_pages, H_KV, PAGE, DIM), device="cuda", dtype=DTYPE
            )
            self.main_v_full = torch.randn(
                (total_pages, H_KV, PAGE, DIM), device="cuda", dtype=DTYPE
            )
            self.index_pages_full = torch.arange(
                total_pages, device="cuda", dtype=torch.int32
            )
            self.score_storage = torch.empty(
                max_score_elements, device="cuda", dtype=torch.float32
            )
            self.route_storage = torch.empty(
                H_KV * max_sq * TOPK, device="cuda", dtype=torch.int32
            )
            self.output_storage = torch.empty(
                max_sq * H_Q * DIM, device="cuda", dtype=DTYPE
            )
            large = (
                torch.empty(TOPK * max_sq * H_Q * DIM, device="cuda", dtype=DTYPE),
                torch.empty(TOPK * max_sq * H_Q, device="cuda", dtype=torch.float32),
                torch.empty(max_sq * H_Q, device="cuda", dtype=torch.float32),
            )
            medium_exact, medium_dtype = allocate_prefill_fwd_workspace(
                max_sq, device="cuda", tier="medium"
            )
            if medium_dtype != torch.float8_e4m3fn:
                raise RuntimeError(f"unexpected medium partial dtype: {medium_dtype}")
            medium = tuple(t.reshape(-1) for t in medium_exact)
        self.stream.synchronize()

        self.chunks: list[Chunk] = []
        for start, end, sq, plan, max_k_tiles in descriptors:
            pages = (end + PAGE - 1) // PAGE
            score = self.score_storage[: HIDX * max_k_tiles * sq].view(
                HIDX, max_k_tiles, sq
            )
            route = self.route_storage[: H_KV * sq * TOPK].view(H_KV, sq, TOPK)
            output = self.output_storage[: sq * H_Q * DIM].view(sq, H_Q, DIM)
            large_workspace = (
                large[0][: TOPK * sq * H_Q * DIM].view(TOPK, sq, H_Q, DIM),
                large[1][: TOPK * sq * H_Q].view(TOPK, sq, H_Q),
                large[2][: sq * H_Q].view(sq, H_Q),
            )
            medium_workspace = (
                medium[0][: TOPK * sq * H_Q * DIM].view(TOPK, sq, H_Q, DIM),
                medium[1][: TOPK * sq * H_Q].view(TOPK, sq, H_Q),
                medium[2][: sq * H_Q].view(sq, H_Q),
            )
            with torch.cuda.stream(self.stream):
                cu_q = torch.tensor([0, sq], device="cuda", dtype=torch.int32)
                cu_k = torch.tensor([0, end], device="cuda", dtype=torch.int32)
                prefix_lens = torch.tensor([start], device="cuda", dtype=torch.int32)
                seqused_k = torch.tensor([end], device="cuda", dtype=torch.int32)
            csr_scratch = allocate_csr_scratch(
                head_kv=H_KV,
                total_q=sq,
                total_rows=pages,
                batch=1,
                max_kv_blocks=pages,
                device="cuda",
            )
            chunk = Chunk(
                start=start,
                end=end,
                sq=sq,
                skv=end,
                pages=pages,
                plan=plan,
                index_q=self.index_q_full[start:end],
                main_q=self.main_q_full[start:end],
                index_k=self.index_k_full[:pages],
                main_k=self.main_k_full[:pages],
                main_v=self.main_v_full[:pages],
                index_pages=self.index_pages_full[:pages],
                page_table=self.index_pages_full[:pages].view(1, pages),
                cu_q=cu_q,
                cu_k=cu_k,
                prefix_lens=prefix_lens,
                seqused_k=seqused_k,
                score=score,
                route=route,
                csr_scratch=csr_scratch,
                csr_buffers=None,
                output=output,
                large_workspace=large_workspace,
                medium_workspace=medium_workspace,
            )
            with torch.cuda.stream(self.stream):
                score_prefill(
                    chunk.index_q,
                    chunk.index_k,
                    chunk.index_k,
                    plan=chunk.plan,
                    kv_indices=chunk.index_pages,
                    sm_scale=SCALE,
                    out=chunk.score,
                )
                select_prefill_exact_per_group(
                    chunk.score,
                    cu_seqlens_q=chunk.cu_q,
                    prefix_lens=chunk.prefix_lens,
                    max_query_len=chunk.sq,
                    topk=TOPK,
                    prefix_blocks=8,
                    local_blocks=8,
                    out=chunk.route,
                )
                chunk.csr_buffers = build_csr(
                    chunk.route,
                    chunk.cu_q,
                    chunk.cu_k,
                    PAGE,
                    total_k=chunk.skv,
                    max_seqlen_k=chunk.skv,
                    max_seqlen_q=chunk.sq,
                    total_rows=chunk.pages,
                    num_kv_heads=H_KV,
                    qhead_per_kv=H_Q // H_KV,
                    scratch=chunk.csr_scratch,
                )
            self.chunks.append(chunk)
        self.stream.synchronize()

        # Compile both variants and validate the final chunk outside formal timing.
        self.final_output_max_abs_diff = 0.0
        self.final_lse_max_abs_diff = 0.0
        if validate_pair:
            first, second = self.variants
            self.run_pass(first)
            self.stream.synchronize()
            with torch.cuda.stream(self.stream):
                first_output = self.chunks[-1].output.clone()
                first_tier, _, _, _ = self._variant_config(first, self.chunks[-1])
                first_lse = (
                    self.chunks[-1].large_workspace[2]
                    if first_tier == "large"
                    else self.chunks[-1].medium_workspace[2]
                ).clone()
            self.stream.synchronize()
            self.run_pass(second)
            self.stream.synchronize()
            final_tier, _, _, _ = self._variant_config(second, self.chunks[-1])
            final_lse = (
                self.chunks[-1].large_workspace[2]
                if final_tier == "large"
                else self.chunks[-1].medium_workspace[2]
            )
            self.final_output_max_abs_diff = float(
                (first_output.float() - self.chunks[-1].output.float()).abs().max().item()
            )
            self.final_lse_max_abs_diff = float((first_lse - final_lse).abs().max().item())
            torch.testing.assert_close(
                first_output, self.chunks[-1].output, atol=4e-2, rtol=4e-2
            )
            torch.testing.assert_close(first_lse, final_lse, atol=3e-3, rtol=3e-3)
        else:
            self.run_pass("BEST_SAFE_DISPATCH")
            self.stream.synchronize()
        self.start_events = {
            variant: torch.cuda.Event(enable_timing=True)
            for variant in self.variants
        }
        self.end_events = {
            variant: torch.cuda.Event(enable_timing=True)
            for variant in self.variants
        }

    @staticmethod
    def _variant_config(variant: str, chunk: Chunk) -> tuple[str, bool, bool, bool]:
        if variant == "B0_LEGACY_BF16":
            return "large", False, False, False
        if variant == "C014_DIRECT_PARTIAL":
            return safe_prefill_workspace_tier(max_seq_len=chunk.skv), True, True, False
        if variant in (
            "BEST_SAFE_DISPATCH",
            "SAFE_BASELINE_A",
            "SAFE_BASELINE_B",
            "C014_NATIVE_PARTIAL",
        ):
            return safe_prefill_workspace_tier(max_seq_len=chunk.skv), True, True, True
        raise ValueError(variant)

    def run_pass(self, variant: str) -> None:
        with torch.cuda.stream(self.stream):
            for chunk in self.chunks:
                tier, safe_fwd_sync, shared_work_valid, safe_partial_v_tail = self._variant_config(
                    variant, chunk
                )
                partial_dtype = DTYPE if tier == "large" else torch.float8_e4m3fn
                prefill_top32_chain(
                    index_q=chunk.index_q,
                    index_k=chunk.index_k,
                    main_q=chunk.main_q,
                    main_k=chunk.main_k,
                    main_v=chunk.main_v,
                    index_plan=chunk.plan,
                    index_page_indices=chunk.index_pages,
                    main_page_table=chunk.page_table,
                    cu_seqlens_q=chunk.cu_q,
                    cu_seqlens_k=chunk.cu_k,
                    prefix_lens=chunk.prefix_lens,
                    seqused_k=chunk.seqused_k,
                    max_query_len=chunk.sq,
                    max_seq_len=chunk.skv,
                    total_k=chunk.skv,
                    total_rows=chunk.pages,
                    softmax_scale=SCALE,
                    score_out=chunk.score,
                    route_out=chunk.route,
                    csr_buffers=chunk.csr_buffers,
                    csr_scratch=chunk.csr_scratch,
                    output_out=chunk.output,
                    fwd_workspace=(
                        chunk.large_workspace
                        if tier == "large"
                        else chunk.medium_workspace
                    ),
                    partial_dtype=partial_dtype,
                    safe_fwd_sync=safe_fwd_sync,
                    # Actual Sq=8192 synccheck requires the CTA-shared work
                    # marker.  The register-valid specialization was only
                    # sanitizer-qualified at Sq=33 in the earlier campaign.
                    shared_work_valid=shared_work_valid,
                    direct_route_partial_page=(variant == "C014_DIRECT_PARTIAL"),
                    safe_partial_v_tail=safe_partial_v_tail,
                )

    def measure(self, variant: str) -> float:
        with torch.cuda.stream(self.stream):
            self.start_events[variant].record(self.stream)
            self.run_pass(variant)
            self.end_events[variant].record(self.stream)
        self.end_events[variant].synchronize()
        return float(self.start_events[variant].elapsed_time(self.end_events[variant]))


class UnifiedFullPrefill(FullPrefill):
    def __init__(
        self,
        total_kv: int,
        seed: int,
        *,
        fused_state: bool = False,
        compare_fusion: bool = False,
        prepare_overlap: bool = False,
        bsa_variant: dict | None = None,
    ) -> None:
        self._initializing = True
        fixed_options = {}
        if bsa_variant is not None:
            from third_party.anthony_bsa_backend.flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk128 import (
                BsaOutputLayout,
            )

            if set(bsa_variant) != {"output_layout", "scheduler"}:
                raise ValueError("BSA variant requires output_layout and scheduler")
            if bsa_variant["scheduler"] not in ("static", "clc"):
                raise ValueError("BSA scheduler must be static or clc")
            fixed_options = {
                "fixed_output_layout": BsaOutputLayout(bsa_variant["output_layout"]),
                "fixed_use_clc_scheduler": bsa_variant["scheduler"] == "clc",
            }
        super().__init__(
            total_kv,
            seed,
            validate_pair=False,
            variants=(
                ("UNIFIED", "FUSED")
                if compare_fusion
                else ("C123", "UNIFIED")
            ),
        )
        self.unified_plans = []
        self.fused_plans = [] if compare_fusion else None
        shared_by_variant_policy = {}
        for chunk in self.chunks:
            if not _use_unified_aligned_prefill(sq=chunk.sq, skv=chunk.skv):
                self.unified_plans.append(None)
                if self.fused_plans is not None:
                    self.fused_plans.append(None)
                continue
            policy = PrefillPolicy.ALIGNED_PAGED
            def prepare_variant(name: str, use_fusion: bool):
                tile_plan = FwdTilePlan(
                    dynamic_route=torch.full(
                        (H_KV, chunk.sq, 16),
                        -1,
                        dtype=torch.int32,
                        device="cuda",
                    ),
                    page_table=chunk.page_table,
                    valid_kv_tokens=chunk.seqused_k,
                    q_offset_tokens=chunk.start,
                    policy=policy,
                )
                plan = prepare_unified_c123_aligned_chain(
                    index_q=chunk.index_q,
                    index_k=chunk.index_k,
                    index_plan=chunk.plan,
                    index_page_indices=chunk.index_pages,
                    candidate_storage=chunk.score,
                    prefix_lens=chunk.prefix_lens,
                    q=chunk.main_q,
                    k=chunk.main_k,
                    v=chunk.main_v,
                    tile_plan=tile_plan,
                    out=chunk.output,
                    partial_dtype=torch.bfloat16,
                    fuse_fixed_state_in_dynamic_combine=use_fusion,
                    overlap_fixed_score=prepare_overlap,
                    **fixed_options,
                )
                attention = plan.attention
                # With 8192-token chunks the first chunk has the largest Q
                # and partial capacity. Residual chunks use contiguous views
                # into that backing storage, including reduced split counts.
                shared_key = (name, policy)
                shared = shared_by_variant_policy.get(shared_key)
                if shared is None:
                    shared_by_variant_policy[shared_key] = (
                        attention.route_sentinel,
                        attention.fixed_out,
                        attention.fixed_lse,
                        attention.dynamic_out,
                        attention.dynamic_workspace,
                        attention.out_lse,
                    )
                else:
                    def subset(backing, target):
                        if target.numel() > backing.numel():
                            raise ValueError("prefill chunk exceeds shared workspace")
                        return backing.view(-1)[:target.numel()].view(target.shape)

                    attention.route_sentinel = shared[0]
                    attention.fixed_out = subset(shared[1], attention.fixed_out)
                    attention.fixed_lse = subset(shared[2], attention.fixed_lse)
                    attention.dynamic_out = subset(shared[3], attention.dynamic_out)
                    attention.dynamic_workspace = tuple(
                        subset(backing, target)
                        for backing, target in zip(shared[4], attention.dynamic_workspace)
                    )
                    attention.out_lse = subset(shared[5], attention.out_lse)
                return plan

            self.unified_plans.append(prepare_variant("UNIFIED", fused_state))
            if self.fused_plans is not None:
                self.fused_plans.append(prepare_variant("FUSED", True))
        torch.cuda.empty_cache()
        self._initializing = False
        self.stream.wait_stream(torch.cuda.current_stream())

        # Historical short-prompt dispatch may use the parent for every chunk.
        self.validated_chunks = 0
        # Chunks share output and partial workspaces, so compare each result
        # before the next chunk overwrites it.
        for index, chunk in enumerate(self.chunks):
            with torch.cuda.stream(self.stream):
                self._run_parent_chunk(chunk)
                reference_out = chunk.output.clone()
                reference_lse = chunk.large_workspace[2].clone()
                plans = [self.unified_plans[index]]
                if self.fused_plans is not None:
                    plans.append(self.fused_plans[index])
                for plan in plans:
                    if plan is None:
                        self._run_parent_chunk(chunk)
                        candidate_lse = chunk.large_workspace[2]
                    else:
                        plan.run()
                        candidate_lse = plan.attention.out_lse
                    self.stream.synchronize()
                    torch.testing.assert_close(
                        chunk.output, reference_out, atol=4e-2, rtol=4e-2
                    )
                    torch.testing.assert_close(
                        candidate_lse, reference_lse, atol=4e-2, rtol=4e-2
                    )
            self.validated_chunks += 1

    def _run_parent_chunk(self, chunk) -> None:
        workspace = chunk.large_workspace
        partial_dtype = torch.bfloat16
        # C118/C123's O-barrier and early-C policies are SM103-only.  The
        # frozen production wrapper derives the policy from length before it
        # applies the architecture guard to the single-correction-warp flag,
        # so a B200 development run would otherwise reject the combination at
        # Skv >= 256K.  Keep production behavior byte-for-byte unchanged on
        # B300 and explicitly select the supported generic Score schedule only
        # in this development harness on non-SM103 devices.
        non_sm103_score_overrides = {}
        if torch.cuda.get_device_capability(chunk.main_q.device) != (10, 3):
            non_sm103_score_overrides = {
                "score_single_correction_warp": False,
                "onlyscore_no_o_barrier": False,
                "early_c_commit": False,
            }
        prefill_top32_chain(
            index_q=chunk.index_q,
            index_k=chunk.index_k,
            main_q=chunk.main_q,
            main_k=chunk.main_k,
            main_v=chunk.main_v,
            index_plan=chunk.plan,
            index_page_indices=chunk.index_pages,
            main_page_table=chunk.page_table,
            cu_seqlens_q=chunk.cu_q,
            cu_seqlens_k=chunk.cu_k,
            prefix_lens=chunk.prefix_lens,
            seqused_k=chunk.seqused_k,
            max_query_len=chunk.sq,
            max_seq_len=chunk.skv,
            total_k=chunk.skv,
            total_rows=chunk.pages,
            softmax_scale=SCALE,
            score_out=chunk.score,
            route_out=chunk.route,
            csr_buffers=chunk.csr_buffers,
            csr_scratch=chunk.csr_scratch,
            output_out=chunk.output,
            fwd_workspace=workspace,
            partial_dtype=partial_dtype,
            safe_fwd_sync=True,
            shared_work_valid=True,
            safe_partial_v_tail=True,
            **non_sm103_score_overrides,
        )

    def run_pass(self, variant: str) -> None:
        if getattr(self, "_initializing", False):
            with torch.cuda.stream(self.stream):
                for chunk in self.chunks:
                    self._run_parent_chunk(chunk)
            return
        if variant not in ("C123", "UNIFIED", "FUSED"):
            raise ValueError(variant)
        plans = self.fused_plans if variant == "FUSED" else self.unified_plans
        with torch.cuda.stream(self.stream):
            for chunk, unified in zip(self.chunks, plans):
                if variant == "C123" or unified is None:
                    self._run_parent_chunk(chunk)
                else:
                    unified.run()


class OverlapFullPrefill(UnifiedFullPrefill):
    def __init__(self, total_kv: int, seed: int, *, bsa_variant: dict | None = None) -> None:
        super().__init__(
            total_kv, seed, fused_state=True, prepare_overlap=True, bsa_variant=bsa_variant
        )
        for plan in self.unified_plans:
            if plan is None:
                continue
            plan.fixed_stream = torch.cuda.Stream(device=plan.index_q.device)
            plan.inputs_ready = torch.cuda.Event(enable_timing=False)
            plan.fixed_done = torch.cuda.Event(enable_timing=False)

    def run_mode(self, overlap: bool) -> None:
        for plan in self.unified_plans:
            if plan is None:
                continue
            plan.overlap_fixed_score = overlap
        self.run_pass("UNIFIED")

    def measure_mode(self, overlap: bool) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.stream):
            start.record(self.stream)
            self.run_mode(overlap)
            end.record(self.stream)
        end.synchronize()
        return float(start.elapsed_time(end))
