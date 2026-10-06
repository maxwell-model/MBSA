# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for MBSA: package integration; see root NOTICE.

"""MXFP6 page scorers, JIT-built from ``csrc/`` on SM100.

Decode scores with a DeepGEMM-pipelined paged-MQA kernel (persistent CTAs on
a per-SM schedule, 1..4 queries per request). Prefill scores with the
hand-written kernel, one CTA per (query token, KV chunk), given each token's
own causal bound. Both read the same packed cache and return
``[4, total_q, width]`` fp32 page scores, bit-identical to each other, in the
layout a page top-k consumes.

A packaging build compiles the two extensions ahead of time as
``mbsa_mxfp6_prefill`` and ``mbsa_mxfp6_decode`` in this package. Without
them they are built with ``torch.utils.cpp_extension`` from ``csrc/`` alone
(the DeepGEMM-derived helpers are in ``csrc/sm100_utils.cuh``) plus the
CUTLASS headers :func:`mbsa._cuda_jit.cutlass_include_dir` locates. A cold
build takes minutes; :func:`compile_extensions` loads or builds both eagerly
so that a serving engine pays it before the first request.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import torch

from .._cuda_jit import (
    cutlass_include_dir,
    load_extension,
    nvidia_wheel_include_dirs,
)

_CSRC = Path(__file__).resolve().parent / "csrc"

#: Index heads the scorers are built for (``HEADS`` in csrc/).
INDEX_HEADS = 4
#: SMs the decode schedule is compiled for (``NUM_SMS`` in fp6_dg_ext.cu).
NUM_SMS = 148
#: Decode requests one schedule can hold (``kAligned`` in fp6_dg_ext.cu).
MAX_DECODE_REQS = 256
#: Queries per request the decode kernel is instantiated for.
MAX_DECODE_QUERY_LEN = 4


def _arch() -> str:
    """The arch-specific target, e.g. ``103a``: tcgen05 needs the ``a`` form."""
    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    if cap is None or cap[0] != 10:
        raise RuntimeError(
            f"the MXFP6 index scorers need an SM100-family GPU, got {cap}"
        )
    return f"{cap[0]}{cap[1]}a"


def _load(source: str, name: str) -> Any:
    return load_extension(__package__, name, lambda: _jit_build(source, name))


def _jit_build(source: str, name: str) -> Any:
    src = _CSRC / source
    if not src.exists():
        raise FileNotFoundError(
            f"the MXFP6 kernel source {src} is missing; this installation is "
            "incomplete."
        )
    include_paths = [str(_CSRC), cutlass_include_dir()]
    arch = _arch()
    from torch.utils.cpp_extension import load

    return load(
        name=f"{name}_sm{arch}",
        sources=[str(src)],
        extra_include_paths=include_paths + nvidia_wheel_include_dirs(),
        extra_cuda_cflags=[
            "-std=c++20",
            "--expt-relaxed-constexpr",
            # An explicit -gencode keeps torch from adding the non-``a``
            # targets, whose ptxas rejects tcgen05.
            "-gencode",
            f"arch=compute_{arch},code=sm_{arch}",
        ],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )


@cache
def _prefill_extension() -> Any:
    return _load("fp6_ext.cu", "mbsa_mxfp6_prefill")


@cache
def _decode_extension() -> Any:
    return _load("fp6_dg_ext.cu", "mbsa_mxfp6_decode")


def compile_extensions() -> None:
    """Load both scorer extensions, building any that is not prebuilt.

    Raises:
        RuntimeError: when building, on a non-SM100 device or if no CUTLASS
            headers are found; with ``MBSA_EXTENSIONS=prebuilt``, if a
            prebuilt module is missing.
    """
    _prefill_extension()
    _decode_extension()


def schedule_metadata(seq_lens: torch.Tensor, next_n: int = 1) -> torch.Tensor:
    """The decode kernel's per-SM work split for ``seq_lens``.

    It depends on the lengths alone, so every layer of a step can share one.
    """
    return _decode_extension().dg_schedule_metadata(seq_lens, NUM_SMS, next_n)


def shared_schedule(
    seq_lens: torch.Tensor, next_n: int, owner: object | None = None
) -> torch.Tensor:
    """:func:`schedule_metadata`, computed once per ``owner``.

    ``owner`` is any object that lives exactly as long as one set of lengths,
    such as a serving engine's per-step context; the schedule is cached on it.
    The lengths buffer may be persistent, so a later owner can hand over the
    same tensor with new contents. Without an owner nothing is cached.
    """
    if owner is None:
        return schedule_metadata(seq_lens, next_n)
    key = (seq_lens.data_ptr(), seq_lens.numel(), next_n)
    cached = getattr(owner, "_mbsa_mxfp6_schedule", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    schedule = schedule_metadata(seq_lens, next_n)
    owner._mbsa_mxfp6_schedule = (key, schedule)
    return schedule


def decode_score(
    q_bytes: torch.Tensor,
    sf_q: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    init_blocks: int,
    local_blocks: int,
    score_sum: bool,
    sm_scale: float,
    next_n: int = 1,
    schedule: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode page scores, ``[4, total_q, block_table.shape[1]]`` fp32.

    Forced init/local pages carry the ``1e30``/``1e29`` sentinels; columns
    past a request's own page count are left unwritten.

    Args:
        q_bytes: ``[total_q, 8, 128]`` uint8 E4M3 index Q, request-major.
        sf_q: ``[total_q, 8]`` int32, four UE8M0 bytes per row.
        cache: Flat uint8 packed index-K cache.
        block_table: ``[num_reqs, max_blocks]`` int32.
        seq_lens: ``[num_reqs]`` int32 KV lengths, counting all ``next_n``
            queries. Zero-length rows produce no work.
        init_blocks: Leading pages forced into the selection.
        local_blocks: Trailing pages forced in; wins over ``init_blocks``.
        score_sum: Sum a page's tokens instead of taking their max.
        sm_scale: Multiplier applied to every Q.K.
        next_n: Queries per request.
        schedule: :func:`schedule_metadata` of ``seq_lens``; computed here
            when omitted.

    Raises:
        ValueError: if the query rows do not match ``seq_lens * next_n``.

    """
    if seq_lens.numel() * next_n != q_bytes.shape[0]:
        raise ValueError(
            f"{q_bytes.shape[0]} query rows for {seq_lens.numel()} requests "
            f"of {next_n} queries"
        )
    if schedule is None:
        schedule = schedule_metadata(seq_lens, next_n)
    return _decode_extension().dg_index_score(
        q_bytes,
        sf_q,
        cache,
        seq_lens,
        block_table,
        schedule,
        block_table.shape[1],
        init_blocks,
        local_blocks,
        score_sum,
        sm_scale,
        NUM_SMS,
        next_n,
    )


def prefill_score(
    q_bytes: torch.Tensor,
    sf_q: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    max_seq_len: int,
    score_sum: bool,
    sm_scale: float,
    init_blocks: int = 0,
    local_blocks: int = 0,
) -> torch.Tensor:
    """Prefill page scores, ``[4, total_q, round_up(max_blocks, 16)]`` fp32.

    Token ``t`` of a request scores the pages up to its own position. All
    shapes are given up front, so nothing reads back to the host.

    Args:
        q_bytes: ``[total_q, 8, 128]`` uint8 E4M3 index Q.
        sf_q: ``[total_q, 8]`` int32, four UE8M0 bytes per row.
        cache: Flat uint8 packed index-K cache.
        block_table: ``[num_reqs, max_blocks]`` int32.
        cu_seqlens_q: ``[num_reqs + 1]`` int32 query offsets, from 0.
        context_lens: ``[num_reqs]`` int32 tokens already cached.
        max_seq_len: Longest KV length in the batch.
        score_sum: Sum a page's tokens instead of taking their max.
        sm_scale: Multiplier applied to every Q.K.
        init_blocks: Leading pages forced, as sentinels.
        local_blocks: Trailing pages forced, as sentinels.

    """
    total_q = q_bytes.shape[0]
    num_reqs = cu_seqlens_q.shape[0] - 1
    device = q_bytes.device
    req_ids = torch.repeat_interleave(
        torch.arange(num_reqs, device=device, dtype=torch.int32),
        cu_seqlens_q[1:] - cu_seqlens_q[:-1],
        output_size=total_q,
    )
    rows = req_ids.long()
    pos = torch.arange(total_q, device=device, dtype=torch.int32) - cu_seqlens_q[
        rows
    ].to(torch.int32)
    kv_lens = context_lens[rows].to(torch.int32) + pos + 1

    max_blocks = block_table.shape[1]
    # ~1280 CTAs fill the machine without splitting pages so thinly that the
    # pipeline fill dominates; shape-derived, so the grid is static.
    num_chunks = max(1, min(1280 // max(1, total_q), max(1, max_blocks // 8)))
    # The prefill top-k reads a 16-aligned stride.
    width = (-(-max_seq_len // 128) + 15) // 16 * 16
    return _prefill_extension().index_score_mxfp6(
        q_bytes,
        sf_q,
        cache,
        block_table,
        kv_lens,
        req_ids,
        max_blocks,
        width,
        num_chunks,
        init_blocks,
        local_blocks,
        score_sum,
        sm_scale,
    )
