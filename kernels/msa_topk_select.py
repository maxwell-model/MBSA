# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""② block top-k select, with prefix/local window pinning.

Compass-v4 block-sparse config: ``topk=16`` pages = 1 global prefix page
(``force_begin_blocks=1``) + 14 pages chosen by score + 1 page ending at the
query's own causal boundary (``force_end_blocks=1``). Pinned pages are counted
*inside* the 16-page budget, not on top of it. The 4 KV-group score rows are
max-reduced to one row before top-k. The selector therefore runs exactly once
and returns one physical route shared by every Q head.

The original ``select_prefill`` and default ``select_decode`` calls preserve
the customer Top-16 shared-route behavior. Provisional Top-32 entry points use
the exact Triton selector and keep all four KV/index groups independent.

Backends:

  * Prefill: ``select_prefill`` wraps MSA CuTe-DSL's ``sparse_topk_select``
    over the ``msa_indexer_score.score_prefill`` output.
  * Decode: ``select_decode`` wraps the vendored Triton
    ``minimax_m3_index_decode``, which fuses block scoring and top-k selection
    into one call (three internal kernel launches) -- there is no separate
    decode score/topk split to measure, unlike the prefill side.
"""

from __future__ import annotations

from typing import Optional

import torch

from msa_block_sparse.minimax_index_topk_triton import (  # noqa: F401
    canonicalize_dynamic_top16_padding,
    minimax_m3_index_decode,
    minimax_m3_index_topk,
)

try:
    import fmha_sm100

    _MSA_OK = True
except (ImportError, RuntimeError):
    # See kernels/msa_indexer_score.py: a bare `import fmha_sm100` allocates
    # pinned CUDA host memory at module load time, so it raises RuntimeError
    # (not ImportError) on any machine without a live NVIDIA driver.
    fmha_sm100 = None
    _MSA_OK = False


def is_msa_available() -> bool:
    return _MSA_OK


def select_prefill(
    max_score: torch.Tensor,
    topk: int,
    *,
    num_valid_pages: int,
    force_begin_blocks: int = 1,
    force_end_blocks: int = 1,
    prescale: float = 1.0,
    per_query_valid: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Top-k pages from the prefill proxy-score tensor.

    Args:
        max_score: ``[num_idx_heads, max_k_tiles, chunk_q]`` fp32, output of
            ``msa_indexer_score.score_prefill``.
        topk: pages to select per query (16 for this project).
        num_valid_pages: causal page bound shared by every query in this call
            (omitting this is an MSA-documented correctness risk: OOB page
            accesses downstream).
        force_begin_blocks: leading pages pinned globally (1 = sink page).
        force_end_blocks: trailing pages pinned at each query's own causal
            boundary (1 = the query's own page, since the proxy pass cannot
            score the diagonal block itself).
        prescale: multiplicative rescale applied to real (non-forced) scores
            before ranking; only needed to keep an FP8-derived score
            distribution inside the selector's fp16 histogram range. The
            delivered BF16 and FP8 benchmark paths both leave it at 1.0.
        per_query_valid: optional ``[chunk_q]`` int32, per-row causal bound
            (use instead of a single ``num_valid_pages`` for ragged rows).
        out: optional preallocated output buffer.

    Returns:
        ``[chunk_q, 1, topk]`` int32, ascending, ``-1`` padded. There is no
        per-group selector mode in this project.
    """
    if not _MSA_OK:
        raise ImportError("the MSA submodule (fmha_sm100) is required.")
    return fmha_sm100.sparse_topk_select(
        max_score.contiguous(),
        topk,
        num_valid_pages=num_valid_pages,
        output=out,
        force_begin_blocks=force_begin_blocks,
        force_end_blocks=force_end_blocks,
        per_query_valid=per_query_valid,
        prescale=prescale,
        group_size=int(max_score.shape[0]),
        group_reduce="max",
    )


def select_prefill_exact_per_group(
    max_score: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    topk: int = 32,
    prefix_blocks: int = 8,
    local_blocks: int = 8,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Select the exact provisional MiniMax route for prefill.

    ``max_score`` uses MSA's native ``[Hkv, Ktiles, total_q]`` layout. The
    returned logical route is Hkv-major ``[Hkv, total_q, topk]``. Prefix and
    local blocks are a unique fixed union inside the budget; remaining slots
    contain exact dynamic winners, followed by ``-1`` for short rows.
    """
    if max_score.ndim != 3 or max_score.dtype != torch.float32:
        raise ValueError("max_score must be FP32 [Hkv,Ktiles,total_q]")
    if max_score.shape[0] != 4:
        raise ValueError(
            "provisional contract requires Hkv/index heads=4, got "
            f"{max_score.shape[0]}"
        )
    if cu_seqlens_q.ndim != 1 or prefix_lens.ndim != 1:
        raise ValueError("cu_seqlens_q and prefix_lens must be rank-1")
    if cu_seqlens_q.numel() != prefix_lens.numel() + 1:
        raise ValueError("cu_seqlens_q must have batch+1 entries")
    # The Triton selector accepts arbitrary strides. This transpose is a view,
    # so the chain does not publish or copy a second score tensor.
    return minimax_m3_index_topk(
        max_score.transpose(1, 2),
        cu_seqlens_q,
        prefix_lens,
        max_query_len,
        topk,
        prefix_blocks,
        local_blocks,
        out,
        _packed_exact=True,
        _packed_width=16,
    )


def select_decode(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    topk: int,
    init_blocks: int = 1,
    local_blocks: int = 1,
    num_kv_heads: int,
    sm_scale: float,
    decode_query_len: int = 1,
    per_group: bool = False,
    score_out: Optional[torch.Tensor] = None,
    topk_out: Optional[torch.Tensor] = None,
    workspace: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Decode block score + top-k, fused (Triton, cudagraph-safe).

    Thin pass-through to :func:`minimax_m3_index_decode` -- see that
    function's docstring for the exact kernel contract. Kept as a named entry
    point here (rather than importing the Triton module directly from bench/
    verify scripts) so ``docs/block_sparse/KERNELS.md`` has one place to point
    at per stage.

    Returns:
        ``[Hkv, total_q, topk]`` when ``per_group=True``; otherwise the C0
        compatibility shape ``[1, total_q, topk]``. Entries are 0-indexed page
        ids with ``-1`` padding.
    """
    return minimax_m3_index_decode(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        max_seq_len,
        topk,
        init_blocks,
        local_blocks,
        num_kv_heads,
        sm_scale,
        decode_query_len,
        per_group=per_group,
        score_out=score_out,
        topk_out=topk_out,
        workspace=workspace,
    )


def select_decode_top32_per_group(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    sm_scale: float,
    num_kv_heads: int = 4,
    decode_query_len: int = 1,
    score_out: Optional[torch.Tensor] = None,
    topk_out: Optional[torch.Tensor] = None,
    workspace: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Provisional per-group ``8 prefix + 8 local + 16 dynamic`` route."""
    return select_decode(
        idx_q,
        index_kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        topk=32,
        init_blocks=8,
        local_blocks=8,
        num_kv_heads=num_kv_heads,
        sm_scale=sm_scale,
        decode_query_len=decode_query_len,
        per_group=True,
        score_out=score_out,
        topk_out=topk_out,
        workspace=workspace,
    )
