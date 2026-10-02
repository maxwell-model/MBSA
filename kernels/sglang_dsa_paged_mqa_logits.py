# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""① decode: SGLang DSA wrapper around DeepGEMM's fp8_paged_mqa_logits.

Pure tensor reshaping — the actual kernel is DeepGEMM's (pip package `deep_gemm`,
NOT vendored: it is a runtime-JIT library that needs its `_C` extension plus
cutlass headers at run time, so it cannot be reduced to a couple of files).

This repository includes only the tensor-layout wrapper, so the benchmark does
not require a full SGLang install. `bench/common/indexer_topk_core.py` prefers
the installed `sglang.jit_kernel.dsa` and falls back to this copy.
"""
from __future__ import annotations

from collections.abc import Callable

import torch


def deepgemm_paged_mqa_logits_split(
    fp8_paged_mqa_logits_fn: Callable[..., torch.Tensor],
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    ctx_lens_2d: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_seq_len: int,
    *,
    q_offset: int,
) -> torch.Tensor:
    q_fp8 = q_fp8.unsqueeze(1)
    return fp8_paged_mqa_logits_fn(
        q_fp8[:q_offset],
        kv_cache_fp8,
        weights[:q_offset],
        ctx_lens_2d,
        block_tables,
        schedule_metadata,
        max_seq_len,
        clean_logits=False,
    )
