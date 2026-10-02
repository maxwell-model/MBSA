# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Provisional MiniMax-style per-group Top-32 block-sparse GQA chain.

This module is intentionally a thin, explicit orchestration layer. It keeps
the customer Top-16 compatibility entry points untouched while freezing B0's
logical route as ``[B, Hkv=4, Sq, 32]`` (physically Hkv-major for the current
batch-flattened ABI). Plans and workspaces are caller-owned so formal timing
can exclude JIT/autotune and allocation without timing invented component sums.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from msa_indexer_score import score_prefill
from msa_block_sparse.score_local_top16_triton import merge_score_local_top16_route
from msa_sparse_attn_decode import PreparedSparseGqaDecode, sparse_gqa_stage2_attn_fwd_msa_decode
from msa_sparse_attn_prefill import build_csr, sparse_gqa_stage2_attn_fwd_msa_prefill
from msa_topk_select import (
    select_decode_top32_per_group,
    select_prefill_exact_per_group,
)


@dataclass(frozen=True)
class Top32Contract:
    hq: int = 32
    hkv: int = 4
    group_size: int = 8
    head_dim: int = 128
    index_heads: int = 4
    index_dim: int = 128
    block_size: int = 128
    topk: int = 32
    prefix_blocks: int = 8
    local_blocks: int = 8

    def validate(self) -> None:
        if self.hq != self.hkv * self.group_size:
            raise ValueError("Hq must equal Hkv * group_size")
        if self.index_heads != self.hkv:
            raise ValueError("provisional contract maps one index head to each KV group")
        if self.prefix_blocks + self.local_blocks > self.topk:
            raise ValueError("fixed quotas cannot exceed TopK")
        if self.block_size != 128 or self.head_dim != 128 or self.index_dim != 128:
            raise ValueError("B0 supports block/head/index dimensions of 128 only")


CONTRACT = Top32Contract()
CONTRACT.validate()

# C038 is materially positive for both standard chunk and true full-prefill
# from 128K onward.  Its 32K/64K full-prefill effect is below noise, and direct
# irregular 32,769/36,987 replay has negative seed directions, so those shapes
# remain on the qualified parent.  The equal-score shortcut additionally
# requires the TileQ=256 descending Q owner, so residual chunks of at most 128
# queries remain on the dense exact path.  Both decisions are explicit shape
# dispatches; neither reads a device flag or silently falls back after launch.
SCORE_LOCAL_TOP16_MIN_KV = 32768
SCORE_LOCAL_TOP16_MIN_Q = 129
SCORE_LOCAL_TOP16_PARTITION2048_MIN_KV = 131073
SCORE_LOCAL_TOP16_PARTITION4096_MIN_KV = 262145
SCORE_LOCAL_TOP16_PARTITION8192_MIN_KV = 524289
SCORE_EARLY_RELEASE_MIN_KV = 32768
SCORE_EIGHT_WAY_REDUCTION_MIN_KV = 32768
SCORE_ROLLING_REDUCTION_MIN_KV = 32768
SCORE_ROLLING_PREFETCH3_MIN_KV = 32768
SCORE_TMEM_LOAD_REDUCE_MIN_KV = 32768
SCORE_TMEM_LOAD_REDUCE_X64_MIN_KV = 32768
SCORE_SINGLE_CORRECTION_WARP_MIN_KV = 131072
# C118 removes the dead MMA<->correction O-buffer token only from the fused
# OnlyScore path. Direct paired measurements show stable Score/chain benefit
# from 256K onward; 128K regresses in Score and therefore retains C113.
SCORE_ONLYSCORE_NO_O_BARRIER_MIN_KV = 262144
# C123 advances the unchanged per-tile PipelineC token ahead of the fused
# OnlyScore TMEM load/reduce/candidate work.  Formal paired chunk and true
# full-prefill measurements are positive from 256K onward; shorter contexts
# keep C118 so this dispatch does not manufacture a small-shape regression.
SCORE_EARLY_C_COMMIT_MIN_KV = 262144
CSR_BATCH1_IDENTITY_MIN_KV_BLOCKS = 768


def _supports_score_tmem_load_reduce(device: torch.device) -> bool:
    """Return whether the active device can execute native TMEM load-reduce.

    ``tcgen05.ld.red`` is encoded only in the SM103 module.  The generic C089
    variant remains the explicit fallback on SM100 and on contexts shorter
    than the measured 128K dispatch boundary.
    """
    return (
        device.type == "cuda"
        and torch.cuda.get_device_capability(device) == (10, 3)
    )


_PREFILL_WORKSPACE_DTYPES = {
    "large": torch.bfloat16,
    "medium": torch.float8_e4m3fn,
}


def safe_prefill_workspace_tier(*, max_seq_len: int) -> str:
    """Return the measured-safe workspace tier for one prefill chunk.

    The medium FP8 partial path is not within the frozen output tolerance at
    4K/8K context for all formal seeds.  Longer-context chunks retain C007;
    short contexts use the B0 BF16 partial path explicitly.
    """
    if int(max_seq_len) <= 0:
        raise ValueError("max_seq_len must be positive")
    return "large" if int(max_seq_len) <= 8192 else "medium"


def prefill_workspace_dtype(tier: str) -> torch.dtype:
    """Return the explicit Top-32 prefill partial dtype for a workspace tier.

    ``large`` is the frozen B0 BF16-partial path. ``medium`` is the C005 FP8
    partial path. There is deliberately no implicit fallback and no claimed
    zero-workspace native-CSR path.
    """
    try:
        return _PREFILL_WORKSPACE_DTYPES[tier]
    except KeyError as exc:
        supported = ", ".join(sorted(_PREFILL_WORKSPACE_DTYPES))
        raise ValueError(f"unsupported prefill workspace tier {tier!r}; use {supported}") from exc


def allocate_prefill_fwd_workspace(
    total_q: int,
    *,
    device: torch.device | str,
    tier: str = "large",
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.dtype]:
    """Allocate a reusable native-CSR FWD workspace outside timed windows."""
    total_q = int(total_q)
    if total_q <= 0:
        raise ValueError("total_q must be positive")
    partial_dtype = prefill_workspace_dtype(tier)
    workspace = (
        torch.empty(
            (CONTRACT.topk, total_q, CONTRACT.hq, CONTRACT.head_dim),
            dtype=partial_dtype,
            device=device,
        ),
        torch.empty(
            (CONTRACT.topk, total_q, CONTRACT.hq),
            dtype=torch.float32,
            device=device,
        ),
        torch.empty(
            (total_q, CONTRACT.hq),
            dtype=torch.float32,
            device=device,
        ),
    )
    return workspace, partial_dtype


@dataclass
class PrefillResult:
    score: torch.Tensor
    route: torch.Tensor
    k2q_row_ptr: torch.Tensor
    k2q_q_indices: torch.Tensor
    schedule: Any
    output: torch.Tensor
    lse: torch.Tensor


@dataclass
class DecodeResult:
    route: torch.Tensor
    output: torch.Tensor
    lse: torch.Tensor


def _validate_common_q(q: torch.Tensor) -> None:
    if q.ndim != 3 or tuple(q.shape[1:]) != (CONTRACT.hq, CONTRACT.head_dim):
        raise ValueError(f"q must have shape [total_q,{CONTRACT.hq},{CONTRACT.head_dim}]")
    if q.dtype != torch.bfloat16 or not q.is_cuda:
        raise ValueError("B0 q must be CUDA BF16")


@torch.no_grad()
def prefill_top32_chain(
    *,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    main_q: torch.Tensor,
    main_k: torch.Tensor,
    main_v: torch.Tensor,
    index_plan: Any,
    index_page_indices: torch.Tensor,
    main_page_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    prefix_lens: torch.Tensor,
    seqused_k: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    total_k: int,
    total_rows: int,
    softmax_scale: float,
    score_out: Optional[torch.Tensor] = None,
    route_out: Optional[torch.Tensor] = None,
    csr_buffers: Any = None,
    csr_scratch: Any = None,
    output_out: Optional[torch.Tensor] = None,
    fwd_workspace: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    partial_dtype: torch.dtype = torch.bfloat16,
    safe_fwd_sync: bool = True,
    shared_work_valid: bool = True,
    direct_route_partial_page: bool = False,
    safe_partial_v_tail: bool = True,
    score_local_top16: Optional[bool] = None,
    score_local_top16_int32: Optional[bool] = None,
    score_local_top16_partition2048: Optional[bool] = None,
    score_local_top16_partition4096: Optional[bool] = None,
    score_local_top16_partition8192: Optional[bool] = None,
    score_single_correction_warp: Optional[bool] = None,
    onlyscore_no_o_barrier: Optional[bool] = None,
    early_c_commit: Optional[bool] = None,
    csr_min_q_per_cta: int = 128,
    fuse_local_top16_csr: Optional[bool] = None,
) -> PrefillResult:
    """Run one real chunk step as a same-stream complete sparse chain.

    ``index_plan`` is prebuilt outside the formal window. All tensors and all
    four stages run on the current stream; this function performs no host
    readback or synchronization.
    """
    _validate_common_q(main_q)
    if index_q.shape != (main_q.shape[0], CONTRACT.index_heads, CONTRACT.index_dim):
        raise ValueError("index_q must be [total_q,4,128]")
    if int(cu_seqlens_q.numel()) != int(cu_seqlens_k.numel()):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must describe the same batch")
    if int(prefix_lens.numel()) + 1 != int(cu_seqlens_q.numel()):
        raise ValueError("prefix_lens must have one entry per request")
    if partial_dtype == torch.float8_e4m3fn and safe_prefill_workspace_tier(
        max_seq_len=max_seq_len
    ) != "medium":
        raise ValueError(
            "medium FP8 prefill partials are outside the validated domain at "
            "Skv <= 8192; use the large BF16 workspace tier"
        )

    use_score_local_top16 = (
        int(max_seq_len) >= SCORE_LOCAL_TOP16_MIN_KV
        and int(max_query_len) >= SCORE_LOCAL_TOP16_MIN_Q
        if score_local_top16 is None
        else bool(score_local_top16)
    )
    use_score_local_top16_int32 = (
        use_score_local_top16
        if score_local_top16_int32 is None
        else bool(score_local_top16_int32)
    )
    if use_score_local_top16_int32 and not use_score_local_top16:
        raise ValueError("int32 local Top-16 requires local-candidate Score")
    if use_score_local_top16_int32 and int(max_query_len) < SCORE_LOCAL_TOP16_MIN_Q:
        raise ValueError("int32 local Top-16 requires the TileQ=256 query owner")
    use_partition2048 = (
        use_score_local_top16_int32
        and int(max_seq_len) >= SCORE_LOCAL_TOP16_PARTITION2048_MIN_KV
        if score_local_top16_partition2048 is None
        else bool(score_local_top16_partition2048)
    )
    if use_partition2048 and not use_score_local_top16_int32:
        raise ValueError("2,048-block partitions require int32 local Top-16")
    use_partition4096 = (
        use_score_local_top16_int32
        and int(max_seq_len) >= SCORE_LOCAL_TOP16_PARTITION4096_MIN_KV
        if score_local_top16_partition4096 is None
        else bool(score_local_top16_partition4096)
    )
    if use_partition4096 and not use_score_local_top16_int32:
        raise ValueError("4,096-block partitions require int32 local Top-16")
    use_partition8192 = (
        use_score_local_top16_int32
        and int(max_seq_len) >= SCORE_LOCAL_TOP16_PARTITION8192_MIN_KV
        if score_local_top16_partition8192 is None
        else bool(score_local_top16_partition8192)
    )
    if use_partition8192 and not use_score_local_top16_int32:
        raise ValueError("8,192-block partitions require int32 local Top-16")
    if use_partition8192:
        use_partition4096 = False
        use_partition2048 = False
    if use_partition4096:
        use_partition2048 = False
    use_single_correction_warp = (
        use_score_local_top16_int32
        and int(max_seq_len) >= SCORE_SINGLE_CORRECTION_WARP_MIN_KV
        if score_single_correction_warp is None
        else bool(score_single_correction_warp)
    )
    if use_single_correction_warp and not use_score_local_top16_int32:
        raise ValueError("single correction warp requires int32 local Top-16")
    use_onlyscore_no_o_barrier = (
        use_single_correction_warp
        and int(max_seq_len) >= SCORE_ONLYSCORE_NO_O_BARRIER_MIN_KV
        if onlyscore_no_o_barrier is None
        else bool(onlyscore_no_o_barrier)
    )
    if use_onlyscore_no_o_barrier and not use_single_correction_warp:
        raise ValueError("OnlyScore O-barrier elision requires the single correction warp path")
    use_early_c_commit = (
        use_onlyscore_no_o_barrier
        and int(max_seq_len) >= SCORE_EARLY_C_COMMIT_MIN_KV
        if early_c_commit is None
        else bool(early_c_commit)
    )
    if use_early_c_commit and not use_onlyscore_no_o_barrier:
        raise ValueError("early C commit requires O-barrier elision")
    candidate_partition_blocks = (
        8192 if use_partition8192 else
        4096 if use_partition4096 else
        2048 if use_partition2048 else 1024
    )
    candidate_domain_blocks = (
        int(max_seq_len) + CONTRACT.block_size - 1
    ) // CONTRACT.block_size
    use_fused_local_top16_csr = (
        use_score_local_top16_int32
        and candidate_domain_blocks <= candidate_partition_blocks
        if fuse_local_top16_csr is None
        else bool(fuse_local_top16_csr)
    )
    if use_fused_local_top16_csr and not use_score_local_top16_int32:
        raise ValueError("fused local Top-16 CSR requires int32 candidate Score")
    if (
        use_fused_local_top16_csr
        and candidate_domain_blocks > candidate_partition_blocks
    ):
        raise ValueError("fused local Top-16 CSR requires exactly one candidate partition")
    score = score_prefill(
        index_q,
        index_k,
        index_k,
        plan=index_plan,
        kv_indices=index_page_indices,
        sm_scale=softmax_scale,
        out=score_out,
        local_top16=use_score_local_top16,
        local_top16_int32=use_score_local_top16_int32,
        local_top16_partition2048=use_partition2048,
        local_top16_partition4096=use_partition4096,
        local_top16_partition8192=use_partition8192,
        early_s_release=(
            use_score_local_top16_int32
            and int(max_seq_len) >= SCORE_EARLY_RELEASE_MIN_KV
        ),
        eight_way_reduction=(
            use_score_local_top16_int32
            and int(max_seq_len)
            >= min(SCORE_EIGHT_WAY_REDUCTION_MIN_KV,
                   SCORE_ROLLING_REDUCTION_MIN_KV)
        ),
        rolling_reduction=(
            use_score_local_top16_int32
            and int(max_seq_len) >= SCORE_ROLLING_REDUCTION_MIN_KV
        ),
        rolling_prefetch3=(
            use_score_local_top16_int32
            and int(max_seq_len) >= SCORE_ROLLING_PREFETCH3_MIN_KV
        ),
        tmem_load_reduce=(
            use_score_local_top16_int32
            and int(max_seq_len) >= SCORE_TMEM_LOAD_REDUCE_MIN_KV
            and _supports_score_tmem_load_reduce(index_q.device)
        ),
        tmem_load_reduce_x64=(
            use_score_local_top16_int32
            and int(max_seq_len) >= SCORE_TMEM_LOAD_REDUCE_X64_MIN_KV
            and _supports_score_tmem_load_reduce(index_q.device)
        ),
        single_correction_warp=(
            use_single_correction_warp
            and use_score_local_top16_int32
            and int(max_seq_len) >= SCORE_SINGLE_CORRECTION_WARP_MIN_KV
            and _supports_score_tmem_load_reduce(index_q.device)
        ),
        onlyscore_no_o_barrier=use_onlyscore_no_o_barrier,
        early_c_commit=use_early_c_commit,
    )
    if use_score_local_top16:
        if route_out is None:
            route_out = torch.empty(
                (CONTRACT.hkv, int(index_q.shape[0]), CONTRACT.topk),
                device=index_q.device,
                dtype=torch.int32,
            )
        if use_fused_local_top16_csr:
            route = route_out
        else:
            route = merge_score_local_top16_route(
                score,
                cu_seqlens_q=cu_seqlens_q,
                prefix_lens=prefix_lens,
                max_query_len=max_query_len,
                out=route_out,
                topk=CONTRACT.topk,
                prefix_blocks=CONTRACT.prefix_blocks,
                local_blocks=CONTRACT.local_blocks,
                packed_keys=not use_score_local_top16_int32,
                partition_blocks=candidate_partition_blocks,
            )
    else:
        route = select_prefill_exact_per_group(
            score,
            cu_seqlens_q=cu_seqlens_q,
            prefix_lens=prefix_lens,
            max_query_len=max_query_len,
            topk=CONTRACT.topk,
            prefix_blocks=CONTRACT.prefix_blocks,
            local_blocks=CONTRACT.local_blocks,
            out=route_out,
        )
    k2q_row_ptr, k2q_q_indices, schedule = build_csr(
        route,
        cu_seqlens_q,
        cu_seqlens_k,
        CONTRACT.block_size,
        total_k=total_k,
        max_seqlen_k=max_seq_len,
        max_seqlen_q=max_query_len,
        total_rows=total_rows,
        num_kv_heads=CONTRACT.hkv,
        qhead_per_kv=CONTRACT.group_size,
        min_q_per_cta=csr_min_q_per_cta,
        buffers=csr_buffers,
        scratch=csr_scratch,
        local_top16_storage=score if use_fused_local_top16_csr else None,
        prefix_lens=prefix_lens if use_fused_local_top16_csr else None,
        candidate_partition_blocks=(
            candidate_partition_blocks if use_fused_local_top16_csr else None
        ),
        batch1_identity_rows=(
            candidate_domain_blocks >= CSR_BATCH1_IDENTITY_MIN_KV_BLOCKS
        ),
    )
    fwd_result = sparse_gqa_stage2_attn_fwd_msa_prefill(
        main_q,
        main_k,
        main_v,
        k2q_row_ptr,
        k2q_q_indices,
        CONTRACT.topk,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_seq_len,
        block_size=CONTRACT.block_size,
        causal=True,
        softmax_scale=softmax_scale,
        schedule=schedule,
        out=output_out,
        partial_dtype=partial_dtype,
        return_softmax_lse=True,
        page_table=main_page_table,
        seqused_k=seqused_k,
        workspace=fwd_workspace,
        q2k_indices=route,
        prefix_lens=prefix_lens,
        lse_out=fwd_workspace[2] if fwd_workspace is not None else None,
        direct_route_partial_page=(
            direct_route_partial_page and bool(total_k % CONTRACT.block_size)
        ),
        safe_fwd_sync=safe_fwd_sync,
        shared_work_valid=shared_work_valid,
        # Preserve the corrected parent's exact aligned hot path.  The repair
        # specialization is needed only when the physical final page is
        # partial; this also keeps aligned AB/BA compile keys identical.
        safe_partial_v_tail=(
            safe_partial_v_tail and bool(total_k % CONTRACT.block_size)
        ),
    )
    output, lse = fwd_result
    return PrefillResult(score, route, k2q_row_ptr, k2q_q_indices, schedule, output, lse)


@torch.no_grad()
def decode_top32_chain(
    *,
    index_q: torch.Tensor,
    index_k_cache: torch.Tensor,
    main_q: torch.Tensor,
    main_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    softmax_scale: float,
    decode_query_len: int = 1,
    score_out: Optional[torch.Tensor] = None,
    route_out: Optional[torch.Tensor] = None,
    selection_workspace: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    output_out: Optional[torch.Tensor] = None,
    lse_out: Optional[torch.Tensor] = None,
    fwd_workspace: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    prepared_fwd: Optional[PreparedSparseGqaDecode] = None,
) -> DecodeResult:
    """Run decode score/selection and sparse FWD on the current stream."""
    _validate_common_q(main_q)
    if index_q.shape != (main_q.shape[0], CONTRACT.index_heads, CONTRACT.index_dim):
        raise ValueError("index_q must be [total_q,4,128]")
    if prepared_fwd is not None:
        if fwd_workspace is not None:
            raise ValueError("prepared_fwd owns its workspace; omit fwd_workspace")
        bindings = (main_q, main_kv_cache, route_out, block_table, seq_lens)
        if route_out is None or any(a is not b for a, b in zip(bindings, prepared_fwd.inputs)):
            raise ValueError("prepared_fwd must bind the chain's exact Q/KV/route_out/table/length tensors")
        if ((output_out is not None and output_out is not prepared_fwd.out)
                or (lse_out is not None and lse_out is not prepared_fwd.lse)):
            raise ValueError("chain output/LSE must match prepared_fwd buffers")
        if (prepared_fwd.num_kv_heads != CONTRACT.hkv
                or prepared_fwd.decode_query_len != decode_query_len
                or prepared_fwd.softmax_scale != softmax_scale):
            raise ValueError("chain head count, query mapping, and scale must match prepared_fwd")
    route = select_decode_top32_per_group(
        index_q,
        index_k_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        sm_scale=softmax_scale,
        num_kv_heads=CONTRACT.hkv,
        decode_query_len=decode_query_len,
        score_out=score_out,
        topk_out=route_out,
        workspace=selection_workspace,
    )
    if prepared_fwd is not None:
        fwd_result = prepared_fwd.run()
    else:
        fwd_result = sparse_gqa_stage2_attn_fwd_msa_decode(
            main_q,
            main_kv_cache,
            route,
            block_table,
            seq_lens,
            num_kv_heads=CONTRACT.hkv,
            softmax_scale=softmax_scale,
            decode_query_len=decode_query_len,
            out=output_out,
            lse_out=lse_out,
            workspace=fwd_workspace,
            return_lse=True,
        )
    output, lse = fwd_result
    return DecodeResult(route, output, lse)
