"""MiniMax MSA-compatible sparse attention entry points for Maxwell geometry.

``sparse_atten_func``, ``build_k2q_csr`` and ``SparseForwardPhase`` keep the
signatures of the bundled MSA runtime (``fmha_sm100.sparse``). Calls on
SM100-family GPUs forward to that runtime unchanged; calls on SM90 (H100)
run the CuTe DSL Hopper implementation in ``mbsa.cute.sm90_sparse``.
"""

from __future__ import annotations

from enum import Enum
from importlib import import_module

import torch


class SparseForwardPhase(str, Enum):
    """Host launch phase used for component attribution (MSA values)."""

    FULL = "full"
    MSA = "msa"
    COMBINE = "combine"


def _is_sm90(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and torch.cuda.get_device_capability(tensor.device) == (9, 0)


def _msa():
    return import_module(".cute.sm100_sparse", __package__)


def build_k2q_csr(q2k_indices, cu_seqlens_q, cu_seqlens_k, kv_block_size, **kwargs):
    """Build the k2q CSR reverse index (see MSA ``build_k2q_csr``)."""
    if q2k_indices.ndim != 3 or q2k_indices.shape[0] != 4:
        raise NotImplementedError("Maxwell CSR requires 4 KV heads")
    kwargs.setdefault("qhead_per_kv", 8)
    if kwargs["qhead_per_kv"] != 8:
        raise NotImplementedError("Maxwell supports GQA=8 only")
    if _is_sm90(q2k_indices):
        from .cute.sm90_sparse import build_k2q_csr as build

        return build(q2k_indices, cu_seqlens_q, cu_seqlens_k, kv_block_size, **kwargs)
    return _msa().build_k2q_csr(
        q2k_indices, cu_seqlens_q, cu_seqlens_k, kv_block_size, **kwargs
    )


def sparse_atten_func(q, k, v, k2q_row_ptr, k2q_q_indices, topK, *,
                      phase=SparseForwardPhase.FULL, **kwargs):
    """CSR block-sparse varlen attention (see MSA ``sparse_atten_func``)."""
    kv_heads = k.shape[1] if k.ndim in (3, 4) else None
    if (q.ndim != 3 or q.shape[1:] != (32, 128) or kv_heads != 4
            or k.shape[-1] != 128):
        raise NotImplementedError(
            "Maxwell attention requires Q heads=32, KV heads=4, GQA=8, D=128"
        )
    if _is_sm90(q):
        from .cute.sm90_sparse import sparse_atten_func as run

        return run(q, k, v, k2q_row_ptr, k2q_q_indices, topK, phase=phase, **kwargs)
    msa = _msa()
    return msa.sparse_atten_func(
        q, k, v, k2q_row_ptr, k2q_q_indices, topK,
        phase=msa.SparseForwardPhase(SparseForwardPhase(phase).value),
        **kwargs,
    )


__all__ = ["SparseForwardPhase", "build_k2q_csr", "sparse_atten_func"]
