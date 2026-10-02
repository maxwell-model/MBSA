# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""① block-level indexer score — MSA CuTe-DSL max-score FMHA proxy pass (PREFILL).

Compass's block-sparse (MSA) path scores each 128-token KV page with a dense
FMHA proxy pass: ``fmha_sm100(..., output_o=False, output_maxscore=True)``
returns the largest raw QK logit any query sees in that page (no softmax, no
output, no LSE), which ``msa_topk_select.select_prefill`` then top-k's over.

This is the prefill-only score stage. Decode fuses score+select into one
Triton call (``minimax_m3_index_decode``, see ``msa_topk_select.py``), so there
is no separate decode score function here -- physically nothing to time in
isolation on that side.

Plan building (``build_prefill_plan``) is metadata construction, not the timed
kernel: it should sit outside the measured region, same as
``bench/common/indexer_topk_core.py::_prepare_score`` does for the token-level
DeepGEMM indexer.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

try:
    import fmha_sm100

    _MSA_OK = True
except (ImportError, RuntimeError):
    # ImportError: the MSA submodule isn't installed in this venv.
    # RuntimeError: it IS installed, but fmha_sm100/api.py allocates pinned
    # CUDA host memory (`torch.empty(..., pin_memory=True)`) as top-level
    # module state -- so a bare `import fmha_sm100` itself fails on any
    # machine without a live NVIDIA driver (e.g. during `docker build`,
    # which has no GPU attached). Confirmed on real hardware: the exact
    # failure is `RuntimeError: Found no NVIDIA driver on your system.`
    fmha_sm100 = None
    _MSA_OK = False


def is_msa_available() -> bool:
    return _MSA_OK


def build_prefill_plan(
    qo_lens: torch.Tensor,
    kv_lens: torch.Tensor,
    *,
    qo_offset: torch.Tensor,
    num_idx_heads: int,
    num_idx_kv_heads: int = 1,
    page_size: int = 128,
) -> Any:
    """Build the proxy-pass plan for one query chunk. Not part of the timed kernel.

    ``num_idx_kv_heads`` is the proxy's own internal KV-head count (MQA-style,
    defaults to 1 dense proxy KV head), independent of Compass's real
    ``num_kv_heads`` -- the proxy scores every real KV head's worth of queries
    against one shared proxy K.
    """
    if not _MSA_OK:
        raise ImportError("the MSA submodule (fmha_sm100) is required.")
    return fmha_sm100.fmha_sm100_plan(
        qo_lens,
        kv_lens,
        num_idx_heads,
        qo_offset=qo_offset,
        causal=True,
        page_size=page_size,
        num_kv_heads=num_idx_kv_heads,
        output_maxscore=True,
    )


def score_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    plan: Any,
    kv_indices: torch.Tensor,
    sm_scale: float,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    k_block_scale: Optional[torch.Tensor] = None,
    scale_policy: str = "bf16",
    out: Optional[torch.Tensor] = None,
    local_top16: bool = False,
    local_top16_int32: bool = False,
    local_top16_partition2048: bool = False,
    local_top16_partition4096: bool = False,
    local_top16_partition8192: bool = False,
    early_s_release: bool = False,
    eight_way_reduction: bool = False,
    rolling_reduction: bool = False,
    rolling_prefetch3: bool = False,
    tmem_load_reduce: bool = False,
    tmem_load_reduce_x64: bool = False,
    single_correction_warp: bool = False,
    onlyscore_no_o_barrier: bool = False,
    early_c_commit: bool = False,
) -> torch.Tensor:
    """Run the max-score proxy pass. Returns ``max_score`` only (no attention output).

    Args:
        q: ``[chunk_q, num_idx_heads, D]`` proxy queries, BF16 or FP8 E4M3.
        k, v: ``[num_pages, num_idx_kv_heads, page_size, D]`` paged proxy KV.
            The dtype must match Q. V is never read when ``output_o=False``;
            callers may alias ``k``.
        plan: return of :func:`build_prefill_plan`.
        kv_indices: ``[num_pages]`` int32 flattened physical page table.
        sm_scale: softmax scale of the proxy pass (``1/sqrt(D)``).
        q_scale, k_scale: Optional positive scalar dequantization scales for
            FP8 E4M3 Q/K.  These are forwarded to the SM100 FMHA primitive.
            A single pair is valid only when its product is common to every
            block compared by one Exact Top-K row.  Block-varying K scales
            must use ``k_block_scale`` instead of folding into this scalar.
        k_block_scale: Optional logical-block-ordered FP32 K scales.  This is
            the experimental per-block FP8 route path; the Score epilogue
            applies each value before Exact candidate comparison.
        scale_policy: Explicit cache/JIT identity for BF16, raw E4M3,
            per-tensor E4M3, or per-block E4M3.
        out: optional preallocated ``max_score`` buffer.

    Returns:
        ``max_score``: ``[num_idx_heads, max_k_tiles, chunk_q]`` fp32, ``-inf``
        on pages a query cannot see. ``max_k_tiles`` is rounded up to a
        multiple of 128 by the kernel.
    """
    if not _MSA_OK:
        raise ImportError("the MSA submodule (fmha_sm100) is required.")
    _, max_score = fmha_sm100.fmha_sm100(
        q,
        k,
        v,
        plan_info=plan,
        kv_indices=kv_indices,
        sm_scale=sm_scale,
        q_scale=q_scale,
        k_scale=k_scale,
        score_k_block_scale=k_block_scale,
        score_scale_policy=scale_policy,
        output_o=False,
        output_maxscore=True,
        max_score=out,
        score_local_top16=local_top16,
        score_local_top16_int32=local_top16_int32,
        score_local_top16_partition2048=local_top16_partition2048,
        score_local_top16_partition4096=local_top16_partition4096,
        score_local_top16_partition8192=local_top16_partition8192,
        score_early_s_release=early_s_release,
        score_eight_way_reduction=eight_way_reduction,
        score_rolling_reduction=rolling_reduction,
        score_rolling_prefetch3=rolling_prefetch3,
        score_tmem_load_reduce=tmem_load_reduce,
        score_tmem_load_reduce_x64=tmem_load_reduce_x64,
        score_single_correction_warp=single_correction_warp,
        score_onlyscore_no_o_barrier=onlyscore_no_o_barrier,
        score_early_c_commit=early_c_commit,
    )
    if max_score is None:
        raise RuntimeError(
            "fmha_sm100 returned no max_score (num_idx_heads * max_k_tiles * "
            "chunk_q exceeded 2**31 -- shrink the query chunk)."
        )
    return max_score
