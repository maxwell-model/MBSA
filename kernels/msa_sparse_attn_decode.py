# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Paged sparse decode with compatible Triton and prepared HMMA runtimes."""

from __future__ import annotations

from typing import Optional

import torch

from msa_block_sparse.minimax_sparse_attn_triton import (
    minimax_m3_sparse_attn_decode,
)


DecodeWorkspace = tuple[torch.Tensor, torch.Tensor] | torch.Tensor


def _resolve_backend(q, kv_cache, topk_idx, block_table, *, num_kv_heads,
                     decode_query_len, workspace, backend, split_k):
    if backend not in ("auto", "triton", "hmma"):
        raise ValueError("backend must be auto, triton, or hmma")
    if workspace is not None and not isinstance(workspace, (tuple, torch.Tensor)):
        raise ValueError("workspace must be a Triton tuple or HMMA FP32 tensor")
    geometry = (q.is_cuda and q.dtype == torch.bfloat16 and q.ndim == 3
                and tuple(q.shape[1:]) == (32, 128) and num_kv_heads == 4
                and kv_cache.dtype == torch.bfloat16 and kv_cache.ndim == 5
                and tuple(kv_cache.shape[1:]) == (2, 128, 4, 128))
    hardware = geometry and torch.cuda.get_device_capability(q.device) == (10, 3)
    if backend == "auto":
        if isinstance(workspace, tuple):
            backend = "triton"
        elif isinstance(workspace, torch.Tensor) or split_k is not None:
            backend = "hmma"
        else:
            # Default only the measured Top32 single-token, aligned shapes.
            # Other shapes retain Triton; HMMA remains explicitly selectable.
            measured = (hardware and torch.cuda.get_device_properties(q.device).multi_processor_count in (72, 148)
                        and decode_query_len == 1 and q.shape[0] in (1, 32, 64)
                        and topk_idx.ndim == 3 and topk_idx.shape[-1] == 32
                        and block_table.ndim == 2 and 64 <= block_table.shape[1] <= 8192
                        and all(t.stride(-1) == 1 and t.data_ptr() % 16 == 0
                                and all(s % 8 == 0 for s in t.stride()[:-1])
                                for t in (q, kv_cache)))
            backend = "hmma" if measured else "triton"
    if backend == "hmma":
        if not hardware:
            raise ValueError("HMMA decode requires SM103 BF16 Q[total_q,32,128] and KV[pages,2,128,4,128]")
        if isinstance(workspace, tuple):
            raise ValueError("HMMA workspace is an FP32 tensor; legacy partial tuples require Triton")
    else:
        if isinstance(workspace, torch.Tensor):
            raise ValueError("Triton workspace must be (o_partial, lse_partial)")
        if split_k is not None:
            raise ValueError("split_k is selectable only for HMMA")
    return backend


class PreparedSparseGqaDecode:
    """Bound inputs and buffers, prepared before capture, launched on the current stream.

    Contents may change in place. Prepare another plan to bind new addresses;
    use distinct plans and workspaces for overlapping invocations.
    """

    def __init__(self, q, kv_cache, topk_idx, block_table, seq_lens, *,
                 num_kv_heads, softmax_scale, decode_query_len=1,
                 out=None, lse_out=None, workspace=None, backend="auto",
                 split_k=None, return_lse=False):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare_sparse_gqa_decode must run before CUDA Graph capture")
        self.backend = _resolve_backend(
            q, kv_cache, topk_idx, block_table, num_kv_heads=num_kv_heads,
            decode_query_len=decode_query_len, workspace=workspace,
            backend=backend, split_k=split_k)
        self.inputs = (q, kv_cache, topk_idx, block_table, seq_lens)
        self.num_kv_heads, self.softmax_scale = num_kv_heads, softmax_scale
        self.decode_query_len, self.return_lse = decode_query_len, return_lse
        if self.backend == "hmma":
            from kernels.msa_hmma_decode import PreparedHmmaDecode
            self._hmma = PreparedHmmaDecode(
                *self.inputs, num_kv_heads=num_kv_heads, softmax_scale=softmax_scale,
                decode_query_len=decode_query_len, split_k=split_k,
                out=out, lse_out=lse_out, workspace=workspace)
            self.out, self.lse, self.workspace = self._hmma.out, self._hmma.lse, self._hmma.workspace
            self.split_k = self._hmma.split_k
        else:
            if q.dtype != torch.bfloat16:
                raise ValueError("MSA decode sparse attention in this project is BF16-only.")
            self.out = torch.empty_like(q) if out is None else out
            self.lse = (torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
                        if lse_out is None else lse_out)
            if workspace is None:
                target = max(1, min(topk_idx.shape[-1], 256 // max(1, q.shape[0]*num_kv_heads)))
                chunks = 1 << (target.bit_length()-1)
                workspace = (torch.empty((chunks, *q.shape), dtype=q.dtype, device=q.device),
                             torch.empty((chunks, *q.shape[:2]), dtype=torch.float32, device=q.device))
            self.workspace = workspace
            # Warm compilation and validate the legacy workspace outside capture.
            self.run()
            self.split_k = self.workspace[0].shape[0]

    def run(self):
        """Always return O/LSE; the public call respects return_lse."""
        if self.backend == "hmma":
            return self._hmma()
        lse = minimax_m3_sparse_attn_decode(
            *self.inputs, self.num_kv_heads, self.softmax_scale, self.out,
            self.decode_query_len, lse_out=self.lse, workspace=self.workspace)
        return self.out, lse

    def __call__(self):
        out, lse = self.run()
        return (out, lse) if self.return_lse else out


def prepare_sparse_gqa_decode(*args, **kwargs) -> PreparedSparseGqaDecode:
    """Prepare allocation-free sparse decode; see PreparedSparseGqaDecode."""
    return PreparedSparseGqaDecode(*args, **kwargs)


def sparse_gqa_stage2_attn_fwd_msa_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    num_kv_heads: int,
    softmax_scale: float,
    decode_query_len: int = 1,
    out: Optional[torch.Tensor] = None,
    lse_out: Optional[torch.Tensor] = None,
    workspace: Optional[DecodeWorkspace] = None,
    return_lse: bool = False,
    backend: str = "auto",
    split_k: Optional[int] = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """One decode step of block-sparse GQA attention.

    Args:
        q: ``[batch * decode_query_len, Hq, D]`` BF16 queries.
        kv_cache: ``[num_pages, 2, page_size, Hkv, D]`` interleaved K/V
            (``kv_cache[:, 0]`` is K, ``kv_cache[:, 1]`` is V), BF16 in this project. FP8 caches also
            work (auto-detected by the kernel) but this project is BF16-only.
        topk_idx: Hkv-major ``[Hkv, batch * decode_query_len, topk]`` int32
            logical page ids for provisional Top-32, or ``[1, ..., topk]`` for
            C0 compatibility. ``-1`` entries and positive OOB entries are
            masked on device; canonical internal routes contain only a valid
            prefix followed by ``-1``.
        block_table: ``[batch, max_pages]`` int32 logical->physical page ids.
        seq_lens: ``[batch]`` int32 KV length per request. Causal / partial
            last-page masking is derived from this inside the kernel; unlike
            MSA's native decode path, no pre-zeroed padding is required.
        num_kv_heads: 4 in this project.
        softmax_scale: ``1/sqrt(D)``.
        out: optional preallocated ``[batch * decode_query_len, Hq, D]``
            output buffer (BF16). Allocated if omitted.
        workspace: legacy Triton partial tuple, or HMMA FP32 tensor. Auto
            dispatch preserves the legacy tuple ABI.
        backend: auto selects measured aligned SM103 Top32 shapes; explicit
            triton/hmma select the corresponding implementation.
        split_k: optional HMMA split override, one of1/2/4/8/16/32.
        return_lse: also return FP32 natural-log LSE when true.

    Returns:
        ``out``, or ``(out, lse)`` when requested, written in place.
    """
    if q.dtype != torch.bfloat16:
        raise ValueError("MSA decode sparse attention in this project is BF16-only.")
    if topk_idx.ndim != 3 or topk_idx.shape[0] not in (1, num_kv_heads):
        raise ValueError(
            "topk_idx must contain one shared or one per-KV-group route row, got "
            f"{tuple(topk_idx.shape)}"
        )
    if decode_query_len <= 0 or q.shape[0] != seq_lens.numel() * decode_query_len:
        raise ValueError("total_q must equal batch * positive decode_query_len")
    backend = _resolve_backend(
        q, kv_cache, topk_idx, block_table, num_kv_heads=num_kv_heads,
        decode_query_len=decode_query_len, workspace=workspace,
        backend=backend, split_k=split_k)
    if backend == "hmma":
        from kernels.msa_hmma_decode import PreparedHmmaDecode
        prepared = PreparedHmmaDecode(
            q, kv_cache, topk_idx, block_table, seq_lens,
            num_kv_heads=num_kv_heads, softmax_scale=softmax_scale,
            decode_query_len=decode_query_len, split_k=split_k,
            out=out, lse_out=lse_out, workspace=workspace)
        result = prepared()
        return result if return_lse else result[0]
    if out is None:
        out = torch.empty_like(q)
    lse = minimax_m3_sparse_attn_decode(
        q,
        kv_cache,
        topk_idx,
        block_table,
        seq_lens,
        num_kv_heads,
        softmax_scale,
        out,
        decode_query_len,
        lse_out=lse_out,
        workspace=workspace,
    )
    return (out, lse) if return_lse else out
