# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Single fail-closed precision contract for the sparse-GQA Indexer.

The production default is BF16.  E4M3 modes are explicit opt-in paths and FP4
modes are research identities only until their native SM103 synchronization
gate is closed.  Keeping this policy in one module prevents benchmark, plan,
and cache code from independently interpreting environment variables.
"""

from __future__ import annotations

from enum import Enum


class IndexerPrecision(str, Enum):
    BF16 = "bf16"
    RAW_E4M3 = "raw_e4m3"
    PER_TENSOR_E4M3 = "per_tensor_e4m3"
    PER_BLOCK_E4M3 = "per_block_e4m3"
    NVFP4 = "nvfp4"
    MXFP4 = "mxfp4"

    @property
    def is_fp8(self) -> bool:
        return self in {
            IndexerPrecision.RAW_E4M3,
            IndexerPrecision.PER_TENSOR_E4M3,
            IndexerPrecision.PER_BLOCK_E4M3,
        }

    @property
    def is_fp4_research(self) -> bool:
        return self in {IndexerPrecision.NVFP4, IndexerPrecision.MXFP4}

    @property
    def sparse_score_supported(self) -> bool:
        return self is IndexerPrecision.BF16 or self.is_fp8


def parse_indexer_precision(value: str | IndexerPrecision) -> IndexerPrecision:
    if isinstance(value, IndexerPrecision):
        return value
    try:
        return IndexerPrecision(value)
    except ValueError as error:
        legal = ", ".join(mode.value for mode in IndexerPrecision)
        raise ValueError(f"unsupported Indexer precision {value!r}; choose {legal}") from error


def require_sparse_score_support(value: str | IndexerPrecision) -> IndexerPrecision:
    mode = parse_indexer_precision(value)
    if mode.is_fp4_research:
        raise RuntimeError(
            f"{mode.value} is a research codec/ceiling only: the native SM103 "
            "Score path has not passed the delivery synchronization gate"
        )
    return mode


__all__ = [
    "IndexerPrecision",
    "parse_indexer_precision",
    "require_sparse_score_support",
]
