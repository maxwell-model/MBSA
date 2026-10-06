# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Sparse attention interface.

Current delivery scope:
    - head dimension is supported only for D=128

Public API:
    sparse_atten_func(...)

Internal forward core:
    _sparse_atten_csr_varlen_forward(...)

Preprocessing (external, done once):
    q2k_indices [head_kv, total_q, topK]  ->  sparse_index_utils.build_k2q_csr()
        -> k2q_row_ptr   [head_kv, total_rows + 1]  int32
        -> k2q_q_indices [head_kv, total_q * topK]  int32
"""

import math
import weakref
import os
from enum import Enum
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack

from mbsa.cute.src.sm100.fwd.combine import combine, combine_shuffled
from mbsa.cute.src.sm100.fwd.atten_fwd import SparseAttentionForwardSm100
from mbsa.cute.src.sm100.fwd.config import Cfg, MSA_OUTPUT_TILE_ROWS, OutputLayout, select_msa_cfg
from mbsa.cute.src.sm100.prepare_scheduler import SparseAttentionSchedule, prepare_sparse_fwd_schedule_and_split, prepare_sparse_fwd_tma_tile_map
from mbsa.cute.src.common.cute_dsl_utils import to_cute_tensor as to_cute_tensor_kvouter
from mbsa.cute.src.common.tma_utils import create_q_gather4_tma_desc, create_ragged_o_tma_desc

_compile_cache: dict = {}
_ragged_o_tma_desc_cache: dict = {}
_SPARSE_FWD_ABI_VERSION = 7
_TEMPERATURE_LSE_FAST_PATH_ABS_TOL = 1e-12
_SUPPORTED_SPARSE_TOPK = (4, 8, 16, 32)
_SUPPORTED_FWD_DTYPES = (torch.bfloat16,)
_SUPPORTED_FWD_MMA_DTYPES = (torch.bfloat16,)


def _get_ragged_o_tma_desc(
    owner: torch.Tensor, tensor: torch.Tensor
) -> torch.Tensor:
    """Reuse the descriptor while its backing partial-output tensor is live."""
    key = (tensor.device.index, tensor.data_ptr(), tuple(tensor.shape), tensor.dtype)
    cached = _ragged_o_tma_desc_cache.get(key)
    if cached is not None and cached[0]() is not None:
        return cached[1]

    desc = create_ragged_o_tma_desc(tensor)

    def discard(_):
        _ragged_o_tma_desc_cache.pop(key, None)

    _ragged_o_tma_desc_cache[key] = (weakref.ref(owner, discard), desc)
    return desc


class SparseForwardPhase(str, Enum):
    """Host launch phase used for component attribution."""

    FULL = "full"
    MSA = "msa"
    COMBINE = "combine"


def _normalize_partial_dtype(partial_dtype: torch.dtype) -> torch.dtype:
    supported = {torch.float32, torch.bfloat16, torch.float16, torch.float8_e4m3fn}
    if partial_dtype not in supported:
        raise TypeError(
            "partial_dtype must be one of torch.float32 / torch.bfloat16 / "
            "torch.float16 / torch.float8_e4m3fn, "
            f"got {partial_dtype}"
        )
    return partial_dtype


def _normalize_forward_mma_dtype(dtype: Optional[torch.dtype], fallback: torch.dtype, name: str) -> torch.dtype:
    dtype = fallback if dtype is None else dtype
    if dtype not in _SUPPORTED_FWD_MMA_DTYPES:
        raise TypeError(
            f"{name} must be torch.bfloat16, got {dtype}"
        )
    return dtype


def _resolve_forward_mma_dtypes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qk_dtype: Optional[torch.dtype],
    pv_dtype: Optional[torch.dtype],
) -> tuple[torch.dtype, torch.dtype]:
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise TypeError("sparse forward requires BF16 Q/K/V")
    qk_dtype = _normalize_forward_mma_dtype(qk_dtype, q.dtype, "qk_dtype")
    pv_dtype = _normalize_forward_mma_dtype(pv_dtype, v.dtype, "pv_dtype")
    return qk_dtype, pv_dtype


def _to_cute_tensor_meta(t: torch.Tensor, assumed_align: int = 4):
    tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=True)
    return tensor.mark_layout_dynamic(leading_dim=0)


def _torch_dtype_to_cutlass_dtype(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    raise TypeError(
        f"Only torch.bfloat16 and torch.float16 supported, got {dtype}"
    )


def _prepare_paged_kv_for_tma(k, v, blk_kv: int):
    page_size = int(k.shape[2])
    if page_size != blk_kv:
        raise ValueError(f"Sparse Page Attention requires page_size == blk_kv, got {page_size} vs {blk_kv}")
    return k, v


def _validate_cu_seqlens(
    cu_seqlens: torch.Tensor,
    *,
    name: str,
    device: torch.device,
) -> None:
    if cu_seqlens.device != device:
        raise ValueError(f"{name} must be on the same device as q")
    if cu_seqlens.dtype != torch.int32:
        raise TypeError(f"{name} must be torch.int32")
    if cu_seqlens.ndim != 1:
        raise ValueError(f"{name} must have shape [B + 1]")
    if cu_seqlens.shape[0] < 1:
        raise ValueError(f"{name} must have at least one element")
    if not cu_seqlens.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _csr_row_capacity(k2q_row_ptr: torch.Tensor) -> int:
    return int(k2q_row_ptr.shape[1] - 1)


def _validate_csr_varlen_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    topK: int,
    blk_kv: int,
    page_table: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_k: Optional[torch.Tensor],
) -> tuple[int, int]:
    if q.ndim != 3:
        raise ValueError("CSR sparse forward requires q to have shape [total_q, Hq, D]")
    if q.dtype not in _SUPPORTED_FWD_DTYPES:
        raise TypeError(
            f"CSR sparse forward requires BF16 Q/K/V, got {q.dtype}"
        )
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, v must all have BF16 dtype")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError("q, k, v must have the same head dimension")
    dim = q.shape[-1]
    if dim != 128:
        raise NotImplementedError(
            f"CSR sparse forward currently supports only D=128, got D={dim}"
        )
    if page_table is None:
        if k.shape[-2] != v.shape[-2] or k.shape[-1] != v.shape[-1]:
            raise ValueError("k and v must have the same [Hkv, D] tail dimensions")
        head_kv = k.shape[-2]
    else:
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                "Sparse Page Attention requires k and v to have shape "
                "[num_pages, Hkv, page_size, D]"
            )
        if k.shape[1] != v.shape[1] or k.shape[-1] != v.shape[-1]:
            raise ValueError(
                "Sparse Page Attention k and v must have the same Hkv and D"
            )
        head_kv = k.shape[1]
    if (
        q.device != k2q_row_ptr.device
        or q.device != k2q_q_indices.device
    ):
        raise ValueError("CSR metadata must be on the same device as q")
    if (
        k2q_row_ptr.dtype != torch.int32
        or k2q_q_indices.dtype != torch.int32
    ):
        raise TypeError("CSR metadata tensors must be torch.int32")
    if k2q_row_ptr.ndim != 2 or k2q_q_indices.ndim != 2:
        raise ValueError("k2q_row_ptr and k2q_q_indices must be rank-2")

    _validate_cu_seqlens(cu_seqlens_q, name="cu_seqlens_q", device=q.device)
    _validate_cu_seqlens(cu_seqlens_k, name="cu_seqlens_k", device=q.device)
    if cu_seqlens_k.shape != cu_seqlens_q.shape:
        raise ValueError("cu_seqlens_k must have shape [B + 1] matching cu_seqlens_q")
    batch = int(cu_seqlens_q.shape[0] - 1)
    total_q = q.shape[0]

    head_q = q.shape[1]
    if head_q % head_kv != 0:
        raise ValueError("q.shape[1] must be divisible by Hkv")
    qhead_per_kv = head_q // head_kv
    if qhead_per_kv not in (1, 2, 4, 8, 16):
        raise NotImplementedError(
            "CSR forward is currently supported only for qhead_per_kv in {1, 2, 4, 8, 16}"
        )
    if k2q_row_ptr.shape[0] != head_kv or k2q_q_indices.shape[0] != head_kv:
        raise ValueError("CSR metadata head dimension must match KV head count")
    if k2q_q_indices.shape[1] < total_q * topK:
        raise ValueError(
            f"k2q_q_indices.shape[1] ({k2q_q_indices.shape[1]}) must be >= total_q * topK ({total_q * topK})"
        )
    if k2q_row_ptr.shape[1] < 1:
        raise ValueError("k2q_row_ptr must contain at least one row pointer column")

    if page_table is None:
        if seqused_k is not None:
            raise ValueError("seqused_k is only supported together with page_table")
        total_k = k.shape[0]
        if k.ndim != 3 or v.ndim != 3:
            raise ValueError("Sparse Attention requires k and v to have shape [total_k, Hkv, D]")
        if k.shape != (total_k, head_kv, q.shape[-1]) or v.shape != (total_k, head_kv, q.shape[-1]):
            raise ValueError("Sparse Attention k and v must match [total_k, Hkv, D]")
    else:
        if page_table.device != q.device:
            raise ValueError("page_table must be on the same device as q")
        if page_table.dtype != torch.int32:
            raise TypeError("page_table must be torch.int32")
        if page_table.ndim != 2 or page_table.shape[0] != batch:
            raise ValueError("page_table must have shape [B, max_num_pages_per_seq]")
        if page_table.stride(-1) != 1:
            raise ValueError("page_table must be contiguous in the last dimension")
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                "Sparse Page Attention requires k and v to have shape "
                "[num_pages, Hkv, page_size, D]"
            )
        if k.shape != v.shape:
            raise ValueError(f"k and v must have the same shape, got {k.shape} and {v.shape}")
        if k.shape[1] != head_kv or k.shape[3] != q.shape[-1]:
            raise ValueError(
                "Sparse Page Attention k and v must match "
                "[num_pages, Hkv, page_size, D]"
            )
        page_size = int(k.shape[2])
        if page_size != blk_kv:
            raise ValueError(
                f"Unsupported Sparse Page Attention page_size={page_size} for blk_kv={blk_kv}; "
                "require page_size == blk_kv"
            )
        if seqused_k is not None:
            if seqused_k.device != q.device:
                raise ValueError("seqused_k must be on the same device as q")
            if seqused_k.dtype != torch.int32:
                raise TypeError("seqused_k must be torch.int32")
            if seqused_k.shape != (batch,):
                raise ValueError("seqused_k must have shape [B]")
            if not seqused_k.is_contiguous():
                raise ValueError("seqused_k must be contiguous")
    if topK not in _SUPPORTED_SPARSE_TOPK:
        raise ValueError(
            f"CSR sparse forward supports topK in {_SUPPORTED_SPARSE_TOPK}, got {topK}"
        )
    return batch, head_kv


def _validate_schedule_common(
    schedule: SparseAttentionSchedule,
    *,
    device: torch.device,
) -> None:
    if schedule.scheduler_metadata is None:
        raise ValueError("schedule.scheduler_metadata is required")
    if schedule.work_count is None:
        raise ValueError("schedule.work_count is required")
    metadata = schedule.scheduler_metadata
    work_count = schedule.work_count
    if metadata.device != device or work_count.device != device:
        raise ValueError("schedule tensors must be on the same device as q")
    if metadata.dtype != torch.int32 or work_count.dtype != torch.int32:
        raise TypeError("schedule.scheduler_metadata and schedule.work_count must be torch.int32")
    expected_cols = 2 if schedule.metadata_format == "compact_batch1_v1" else 6
    if schedule.metadata_format not in ("legacy6", "compact_batch1_v1"):
        raise ValueError(f"unsupported schedule metadata format {schedule.metadata_format!r}")
    if metadata.ndim != 2 or metadata.shape[1] != expected_cols:
        raise ValueError(
            f"schedule.scheduler_metadata must have shape [capacity, {expected_cols}]"
        )
    if work_count.shape != (1,):
        raise ValueError("schedule.work_count must have shape [1]")
    if not metadata.is_contiguous() or not work_count.is_contiguous():
        raise ValueError("schedule.scheduler_metadata and schedule.work_count must be contiguous")


def _validate_fwd_schedule(
    schedule: SparseAttentionSchedule,
    *,
    q: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    head_kv: int,
) -> None:
    _validate_schedule_common(schedule, device=q.device)
    if schedule.qsplit_indices is None:
        raise ValueError("schedule.qsplit_indices is required for forward")
    if schedule.split_counts is None:
        raise ValueError("schedule.split_counts is required for forward")
    qsplit = schedule.qsplit_indices
    split_counts = schedule.split_counts
    if qsplit.device != q.device or split_counts.device != q.device:
        raise ValueError("forward schedule tensors must be on the same device as q")
    if qsplit.dtype != torch.int32 or split_counts.dtype != torch.int32:
        raise TypeError("schedule.qsplit_indices and schedule.split_counts must be torch.int32")
    if qsplit.shape != k2q_q_indices.shape:
        raise ValueError("schedule.qsplit_indices shape must match k2q_q_indices")
    total_q = q.shape[0]
    if split_counts.shape != (total_q, head_kv):
        raise ValueError(
            "schedule.split_counts must have shape "
            f"({total_q}, {head_kv}), got {tuple(split_counts.shape)}"
        )
    if not qsplit.is_contiguous() or not split_counts.is_contiguous():
        raise ValueError("schedule.qsplit_indices and schedule.split_counts must be contiguous")


def sparse_atten_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    topK: int,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    blk_kv: int = 128,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    lse_temperature_scale: float = 1.0,
    return_temperature_lse: bool = False,
    partial_dtype: torch.dtype = torch.bfloat16,
    return_softmax_lse: bool = False,
    page_table: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    schedule: Optional[SparseAttentionSchedule] = None,
    usable_SM_count: int = -1,
    qk_dtype: Optional[torch.dtype] = None,
    pv_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    initial_out: Optional[torch.Tensor] = None,
    initial_lse: Optional[torch.Tensor] = None,
    safe_fwd_sync: bool = True,
    shared_work_valid: bool = True,
    safe_partial_v_tail: bool = True,
    full_score_tiles: Optional[bool] = None,
    kernel_cfg: Optional[Cfg] = None,
    phase: SparseForwardPhase = SparseForwardPhase.FULL,
    last_cta_finalize: bool = False,
):
    """Run SM100 CSR block-sparse varlen attention.

    This is the public forward-only sparse attention API.  It consumes
    query-to-key block selections converted to CSR metadata by
    ``build_k2q_csr`` and supports both dense KV layout and paged KV layout.

    Parameters
    ----------
    q : torch.Tensor
        Shape ``[total_q, Hq, 128]`` on CUDA.  The dtype is BF16.
    k : torch.Tensor
        Dense layout ``[total_k, Hkv, 128]`` or paged layout
        ``[num_pages, Hkv, blk_kv, 128]``, with BF16 dtype.
    v : torch.Tensor
        Same layout and head count as ``k``.
    k2q_row_ptr : torch.Tensor
        CSR row pointers with shape ``[Hkv, total_rows + 1]`` and dtype int32.
    k2q_q_indices : torch.Tensor
        CSR query indices with shape ``[Hkv, >= total_q * topK]`` and dtype
        int32.
    topK : int
        Number of selected KV blocks per query.  Supported values are
        ``4, 8, 16, 32``.
    cu_seqlens_q : torch.Tensor
        Shape ``[batch_size + 1]``, dtype int32.  Prefix sums of Q lengths.
    cu_seqlens_k : torch.Tensor
        Shape ``[batch_size + 1]``, dtype int32.  Prefix sums of KV lengths.
    max_seqlen_q : int
        Maximum Q sequence length in the batch.
    max_seqlen_k : int
        Maximum KV sequence length in the batch.
    blk_kv : int, optional
        KV block size.  Paged KV requires ``k.shape[2] == blk_kv``.
    causal : bool, optional
        Whether to apply causal masking.
    softmax_scale : float, optional
        Softmax scale.  Defaults to ``1 / sqrt(128)``.
    lse_temperature_scale : float, optional
        Extra divisor used only for temperature-scaled LSE output.
    return_temperature_lse : bool, optional
        If True, also return LSE computed with logits scaled by
        ``softmax_scale / lse_temperature_scale``.  Requires
        ``return_softmax_lse=True``.
    partial_dtype : torch.dtype, optional
        Accumulation dtype for per-block partial O.  Supported values are
        FP32, BF16, and FP16.
    return_softmax_lse : bool, optional
        If True, return ``(out, softmax_lse)`` or
        ``(out, softmax_lse, temperature_lse)``.
    page_table : torch.Tensor, optional
        Paged-KV physical page table with shape
        ``[batch_size, max_num_pages_per_seq]`` and dtype int32.
    seqused_k : torch.Tensor, optional
        Shape ``[batch_size]``, dtype int32.  Effective KV length per request
        for paged causal attention.
    schedule : SparseAttentionSchedule, optional
        Prebuilt sparse forward schedule.  If omitted, the schedule is built
        during the call.
    usable_SM_count : int, optional
        Maximum number of SMs used by the scheduler.  ``-1`` uses all SMs.
    qk_dtype : torch.dtype, optional
        Compile-time MMA operand dtype for QK.  Defaults to BF16.
    pv_dtype : torch.dtype, optional
        Compile-time MMA operand dtype for PV.  Defaults to BF16.
    initial_out, initial_lse : torch.Tensor, optional
        Optional batch-1 fixed-attention state consumed directly by the final
        split combine.  ``initial_out`` matches ``out`` and may alias it;
        ``initial_lse`` uses fixed-BSA layout ``[1, Hq, total_q]``.  Both must
        be provided together.
    kernel_cfg : Cfg, optional
        Explicit compile-time MSA configuration for variant measurement.
        Production callers should leave this unset so ``select_msa_cfg`` owns
        dispatch policy.
    phase : SparseForwardPhase, optional
        ``FULL`` launches the production MSA producer followed by K2 combine.
        ``MSA`` and ``COMBINE`` expose those same launches separately for
        component attribution with a caller-owned persistent workspace.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        Output shape ``[total_q, Hq, 128]`` with BF16 dtype.  Optional LSE
        outputs have shape ``[total_q, Hq]`` and dtype float32.

    Notes
    -----
    ``Hq / Hkv`` must be one of ``1, 2, 4, 8, 16``.  Current kernels support
    head dimension 128 only.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    if not isinstance(phase, SparseForwardPhase):
        raise TypeError(f"phase must be SparseForwardPhase, got {phase!r}")
    if phase is not SparseForwardPhase.FULL:
        if workspace is None or schedule is None:
            raise ValueError(
                "split MSA/combine phases require caller-owned workspace and schedule"
            )
        if return_temperature_lse:
            raise NotImplementedError(
                "split MSA/combine phases do not retain temperature-LSE partials"
            )
    lse_temperature_scale = float(lse_temperature_scale)
    if not math.isfinite(lse_temperature_scale) or lse_temperature_scale <= 0.0:
        raise ValueError(
            f"lse_temperature_scale must be finite and > 0, got {lse_temperature_scale}"
        )
    return_temperature_lse = bool(return_temperature_lse)
    if return_temperature_lse and not return_softmax_lse:
        raise ValueError("return_temperature_lse=True requires return_softmax_lse=True")
    partial_dtype = _normalize_partial_dtype(partial_dtype)
    qk_dtype, pv_dtype = _resolve_forward_mma_dtypes(q, k, v, qk_dtype, pv_dtype)

    if cu_seqlens_q is None or cu_seqlens_k is None:
        raise ValueError(
            "sparse_atten_func requires CSR varlen metadata: pass cu_seqlens_q and cu_seqlens_k"
        )
    batch, head_kv = _validate_csr_varlen_inputs(
        q,
        k,
        v,
        k2q_row_ptr,
        k2q_q_indices,
        topK,
        blk_kv,
        page_table,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
    )
    max_seqlen_q = int(max_seqlen_q)
    max_seqlen_k = int(max_seqlen_k)

    # Paged KV: _prepare_paged_kv_for_tma() passes k/v straight through to the
    # CuTe tensor wrapper untouched (no contiguity requirement -- CuTe's TMA
    # descriptors carry arbitrary strides natively), so materializing them
    # here just copies the entire per-layer KV cache pool for nothing. A
    # caller's paged view is typically a permuted/split slice of that pool
    # (e.g. K/V packed in one cache tensor, sliced by half), never actually
    # contiguous, so this was a full-pool copy on every call.
    k_in = k if page_table is not None else k.contiguous()
    v_in = v if page_table is not None else v.contiguous()

    return _sparse_atten_csr_varlen_forward(
        q.contiguous(),
        k_in,
        v_in,
        k2q_row_ptr.contiguous(),
        k2q_q_indices.contiguous(),
        int(topK),
        int(blk_kv),
        bool(causal),
        float(softmax_scale),
        lse_temperature_scale,
        return_temperature_lse,
        partial_dtype,
        bool(return_softmax_lse),
        cu_seqlens_q.contiguous(),
        cu_seqlens_k.contiguous(),
        None if page_table is None else page_table.contiguous(),
        None if seqused_k is None else seqused_k.contiguous(),
        schedule,
        int(usable_SM_count),
        int(batch),
        int(head_kv),
        int(max_seqlen_q),
        int(max_seqlen_k),
        qk_dtype,
        pv_dtype,
        out,
        workspace,
        initial_out,
        initial_lse,
        safe_fwd_sync,
        shared_work_valid,
        safe_partial_v_tail,
        full_score_tiles,
        kernel_cfg,
        phase,
        last_cta_finalize,
    )


def _sparse_atten_csr_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    topK: int,
    blk_kv: int,
    causal: bool,
    softmax_scale: float,
    lse_temperature_scale: float,
    return_temperature_lse: bool,
    partial_dtype: torch.dtype,
    return_softmax_lse: bool,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    schedule: Optional[SparseAttentionSchedule],
    usable_SM_count: int,
    batch: int,
    head_kv: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    qk_dtype: torch.dtype,
    pv_dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    initial_out: Optional[torch.Tensor] = None,
    initial_lse: Optional[torch.Tensor] = None,
    safe_fwd_sync: bool = True,
    shared_work_valid: bool = True,
    safe_partial_v_tail: bool = True,
    full_score_tiles: Optional[bool] = None,
    kernel_cfg: Optional[Cfg] = None,
    phase: SparseForwardPhase = SparseForwardPhase.FULL,
    last_cta_finalize: bool = False,
):
    total_q, head_q, dim = q.shape
    if (initial_out is None) != (initial_lse is None):
        raise ValueError("initial_out and initial_lse must be provided together")
    if initial_out is not None:
        if batch != 1:
            raise NotImplementedError("initial attention state currently supports batch=1")
        expected_initial = (
            (head_kv, total_q, head_q // head_kv, dim)
            if initial_out.ndim == 4 else (total_q, head_q, dim)
        )
        if tuple(initial_out.shape) != expected_initial:
            raise ValueError("initial_out must be [Q,Hq,D] or [Hkv,Q,GQA,D]")
        if initial_out.dtype != torch.bfloat16 or initial_out.device != q.device:
            raise ValueError("initial_out must be BF16 on the q device")
        if tuple(initial_lse.shape) != (1, head_q, total_q):
            raise ValueError("initial_lse must have shape [1, Hq, total_q]")
        if initial_lse.dtype != torch.float32 or initial_lse.device != q.device:
            raise ValueError("initial_lse must be FP32 on the q device")
    last_cta_finalize = bool(last_cta_finalize)
    if last_cta_finalize and (
        phase is not SparseForwardPhase.FULL
        or initial_out is None
        or out is None
        or return_temperature_lse
        or partial_dtype != torch.bfloat16
        or topK != 16
        or batch != 1
        or causal
        or head_q // head_kv != 8
        or dim != 128
    ):
        raise ValueError(
            "last_cta_finalize requires FULL-phase batch-1 noncausal BF16 "
            "D128 GQA8 Dynamic-16 with caller-owned output and fixed state"
        )
    if head_q % head_kv != 0:
        raise ValueError("q.shape[1] must be divisible by head_kv")
    max_num_kv_blocks = _csr_row_capacity(k2q_row_ptr)
    qhead_per_kv = head_q // head_kv
    sm = torch.cuda.get_device_capability(q.device)
    full_kv_blocks = (
        not causal
        and cu_seqlens_k.numel() == 2
        and max_seqlen_k % blk_kv == 0
        and (
            (page_table is None and seqused_k is None and max_seqlen_k == k.shape[0])
            or (page_table is not None and seqused_k is not None)
        )
    )
    cfg = kernel_cfg
    if cfg is None:
        cfg = select_msa_cfg(
            causal=bool(causal),
            full_kv_blocks=full_kv_blocks,
            sm=sm,
            max_seqlen_k=max_seqlen_k,
            paged_kv=page_table is not None,
        )
    shuffled_output = cfg.output_layout is OutputLayout.SHUFFLED
    shuffled_output_eligible = (
        full_kv_blocks
        and total_q > 0
        and max_num_kv_blocks > 0
        and partial_dtype == torch.bfloat16
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and dim == 128
        and qhead_per_kv == 8
        and topK <= 16
        and not return_temperature_lse
        and sm == (10, 3)
        and (workspace is None or workspace[0].ndim == 2)
    )
    if shuffled_output and not shuffled_output_eligible:
        failed = [
            name
            for name, passed in (
                ("full_kv_blocks", full_kv_blocks),
                ("total_q", total_q > 0),
                ("csr_rows", max_num_kv_blocks > 0),
                ("partial_bf16", partial_dtype == torch.bfloat16),
                ("q_bf16", q.dtype == torch.bfloat16),
                ("k_bf16", k.dtype == torch.bfloat16),
                ("v_bf16", v.dtype == torch.bfloat16),
                ("head_dim_128", dim == 128),
                ("gqa8", qhead_per_kv == 8),
                ("topk_le_16", topK <= 16),
                ("no_temperature_lse", not return_temperature_lse),
                ("sm103", sm == (10, 3)),
                ("shuffled_workspace", workspace is None or workspace[0].ndim == 2),
            )
            if not passed
        ]
        raise ValueError(
            "shuffled output requires full-block BF16 D128 GQA8 TopK<=16; "
            f"failed: {', '.join(failed)}"
        )
    if shuffled_output:
        if schedule is None:
            if phase is SparseForwardPhase.COMBINE:
                raise ValueError("combine-only shuffled phase requires a prepared schedule")
            k2q_qsplit_indices = torch.empty_like(k2q_q_indices)
            split_counts = torch.zeros(
                (total_q, head_kv),
                dtype=torch.int32,
                device=q.device,
            )
            schedule = prepare_sparse_fwd_schedule_and_split(
                k2q_row_ptr=k2q_row_ptr,
                k2q_q_indices=k2q_q_indices,
                k2q_qsplit_indices=k2q_qsplit_indices,
                split_counts=split_counts,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                total_q=total_q,
                max_seqlen_q=max_seqlen_q,
                topk=topK,
                head_kv=head_kv,
                qhead_per_kv=qhead_per_kv,
                blk_kv=blk_kv,
                device=q.device,
                enabled=max_num_kv_blocks > 0,
                usable_SM_count=usable_SM_count,
            )
        else:
            _validate_fwd_schedule(
                schedule,
                q=q,
                k2q_q_indices=k2q_q_indices,
                head_kv=head_kv,
            )
            k2q_qsplit_indices = schedule.qsplit_indices
            split_counts = schedule.split_counts
        if phase is SparseForwardPhase.COMBINE:
            if any(
                value is None
                for value in (
                    schedule.tma_tile_offsets,
                    schedule.tma_partial_locations,
                    schedule.tma_tile_count,
                )
            ):
                raise ValueError(
                    "combine-only shuffled phase requires a prepared tile map"
                )
        else:
            schedule = prepare_sparse_fwd_tma_tile_map(
                schedule=schedule,
                k2q_row_ptr=k2q_row_ptr,
                cu_seqlens_q=cu_seqlens_q,
                total_q=total_q,
                topk=topK,
                head_kv=head_kv,
                qhead_per_kv=qhead_per_kv,
            )
    temperature_lse_fast_path = (
        return_temperature_lse
        and math.isclose(
            float(lse_temperature_scale),
            1.0,
            rel_tol=0.0,
            abs_tol=_TEMPERATURE_LSE_FAST_PATH_ABS_TOL,
        )
    )
    kernel_return_temperature_lse = (
        return_temperature_lse and not temperature_lse_fast_path
    )

    if workspace is None:
        if shuffled_output:
            tile_rows = schedule.tma_tile_capacity * MSA_OUTPUT_TILE_ROWS
            O_partial = torch.empty(
                tile_rows,
                dim,
                dtype=partial_dtype,
                device=q.device,
            )
            LSE_partial = torch.empty(
                tile_rows,
                dtype=torch.float32,
                device=q.device,
            )
        else:
            O_partial = torch.empty(
                topK, total_q, head_q, dim, dtype=partial_dtype, device=q.device
            )
            LSE_partial = torch.empty(
                topK, total_q, head_q, dtype=torch.float32, device=q.device
            )
        LSE_out = torch.empty(total_q, head_q, dtype=torch.float32, device=q.device)
    else:
        O_partial, LSE_partial, LSE_out = workspace
        if shuffled_output:
            tile_rows = schedule.tma_tile_capacity * MSA_OUTPUT_TILE_ROWS
            if O_partial.shape != (tile_rows, dim):
                raise ValueError(
                    "shuffled O_partial workspace shape does not match schedule"
                )
            if LSE_partial.shape != (tile_rows,):
                raise ValueError(
                    "shuffled LSE_partial workspace shape does not match schedule"
                )
        else:
            if O_partial.shape != (topK, total_q, head_q, dim):
                raise ValueError("CSR O_partial workspace shape does not match q")
            if LSE_partial.shape != (topK, total_q, head_q):
                raise ValueError("CSR LSE_partial workspace shape does not match q")
        if LSE_out.shape != (total_q, head_q):
            raise ValueError("CSR LSE workspace shape does not match q")
        if (
            O_partial.dtype != partial_dtype
            or LSE_partial.dtype != torch.float32
            or LSE_out.dtype != torch.float32
        ):
            raise ValueError("CSR workspace dtypes do not match the attention contract")
    LSE_temperature_partial = (
        torch.empty(topK, total_q, head_q, dtype=torch.float32, device=q.device)
        if kernel_return_temperature_lse
        else None
    )
    if out is not None:
        expected = (total_q, head_q, dim)
        if tuple(out.shape) != expected:
            raise ValueError(f"out must be {expected}, got {tuple(out.shape)}")
        if out.dtype != torch.bfloat16 or not out.is_contiguous():
            raise ValueError(
                f"out must be contiguous bfloat16, got {out.dtype} "
                f"contiguous={out.is_contiguous()}"
            )
        O_out = out
    else:
        O_out = torch.empty(
            total_q, head_q, dim, dtype=torch.bfloat16, device=q.device
        )
    LSE_temperature_out = (
        torch.empty_like(LSE_out) if kernel_return_temperature_lse else None
    )
    if schedule is None:
        if phase is SparseForwardPhase.COMBINE:
            raise ValueError("combine-only phase requires a prepared schedule")
        k2q_qsplit_indices = torch.empty_like(k2q_q_indices)
        split_counts = torch.zeros(
            (total_q, head_kv),
            dtype=torch.int32,
            device=q.device,
        )
    else:
        _validate_fwd_schedule(
            schedule,
            q=q,
            k2q_q_indices=k2q_q_indices,
            head_kv=head_kv,
        )
        k2q_qsplit_indices = schedule.qsplit_indices
        split_counts = schedule.split_counts
    if phase is not SparseForwardPhase.COMBINE:
        schedule = _call_sparse_forward_sm100_csr_varlen(
            q,
            k,
            v,
            k2q_row_ptr,
            k2q_q_indices,
            k2q_qsplit_indices,
            split_counts,
            cu_seqlens_q,
            cu_seqlens_k,
            page_table,
            seqused_k,
            O_partial,
            LSE_partial,
            LSE_temperature_partial,
            softmax_scale,
            lse_temperature_scale,
            kernel_return_temperature_lse,
            max_num_kv_blocks,
            blk_kv,
            head_kv,
            max_seqlen_q,
            max_seqlen_k,
            usable_SM_count,
            causal=causal,
            schedule=schedule,
            qk_dtype=qk_dtype,
            pv_dtype=pv_dtype,
            safe_fwd_sync=bool(safe_fwd_sync),
            shared_work_valid=bool(shared_work_valid),
            safe_partial_v_tail=bool(safe_partial_v_tail),
            full_score_tiles=full_score_tiles,
            kernel_cfg=cfg,
            work_tile_offsets=(
                schedule.tma_tile_offsets if shuffled_output else None
            ),
            O_out=O_out if last_cta_finalize else None,
            LSE_out=initial_lse if last_cta_finalize else None,
            initial_out=initial_out if last_cta_finalize else None,
            initial_lse=initial_lse if last_cta_finalize else None,
            last_cta_finalize=last_cta_finalize,
        )
    if last_cta_finalize:
        # The fixed BSA owns a contiguous [1,H,Q] state.  Dynamic finalizers
        # update that layout in place; expose the ordinary [Q,H] API as a view.
        LSE_out = initial_lse[0].transpose(0, 1)
    if phase is not SparseForwardPhase.MSA and not last_cta_finalize:
        # Sparse Attention and Sparse Page Attention both use the varlen-Q
        # combine path; the kernel-written LSE_out is the final contract.
        if shuffled_output:
            combine_shuffled(
                O_partial,
                LSE_partial,
                schedule.tma_partial_locations,
                split_counts,
                O_out,
                LSE_out,
                initial_o=initial_out,
                initial_lse=initial_lse,
            )
        else:
            combine(
                O_partial,
                LSE_partial,
                O_out,
                LSE_out,
                initial_o=initial_out,
                initial_lse=initial_lse,
                lse_temperature_partial=LSE_temperature_partial,
                lse_temperature_out=LSE_temperature_out,
                cu_seqlens=cu_seqlens_q,
                split_counts=split_counts,
                use_pdl=True,
                use_tma_partial=(
                    os.environ.get("FLASHINFER_K2_TMA_PARTIAL", "1") == "1"
                    # FP8 TMA combine is not qualified across consecutive
                    # shapes (graph/eager and strict HF gates can disagree).
                    # Use the established cp.async path for FP8 partials.
                    and partial_dtype == torch.bfloat16
                    and batch == 1
                    and topK == 16
                    and max_seqlen_k > max_seqlen_q
                    and max_seqlen_k <= 524288
                ),
                use_vector_initial=(
                    os.environ.get("FLASHINFER_K2_VECTOR_INITIAL", "1") == "1"
                    and batch == 1
                    and topK == 16
                    and max_seqlen_k >= 16384
                    and max_seqlen_k <= 131072
                ),
            )
    if temperature_lse_fast_path:
        LSE_temperature_out = LSE_out

    if return_softmax_lse:
        if return_temperature_lse:
            return O_out, LSE_out, LSE_temperature_out
        return O_out, LSE_out
    return O_out


def _call_sparse_forward_sm100_csr_varlen(
    q,
    k,
    v,
    k2q_row_ptr,
    k2q_q_indices,
    k2q_qsplit_indices,
    split_counts,
    cu_seqlens_q,
    cu_seqlens_k,
    page_table,
    seqused_k,
    O_partial,
    LSE_partial,
    LSE_temperature_partial,
    softmax_scale,
    lse_temperature_scale,
    return_temperature_lse,
    max_num_kv_blocks,
    blk_kv,
    head_kv,
    max_seqlen_q,
    max_seqlen_k,
    usable_SM_count=-1,
    *,
    causal=False,
    use_prepare_scheduler=True,
    schedule: Optional[SparseAttentionSchedule] = None,
    qk_dtype: torch.dtype,
    pv_dtype: torch.dtype,
    safe_fwd_sync: bool = True,
    shared_work_valid: bool = True,
    safe_partial_v_tail: bool = True,
    full_score_tiles: Optional[bool] = None,
    kernel_cfg: Optional[Cfg] = None,
    work_tile_offsets: Optional[torch.Tensor] = None,
    O_out: Optional[torch.Tensor] = None,
    LSE_out: Optional[torch.Tensor] = None,
    initial_out: Optional[torch.Tensor] = None,
    initial_lse: Optional[torch.Tensor] = None,
    last_cta_finalize: bool = False,
):
    """Compile and launch the SM100 sparse forward K1 kernel on CSR metadata."""
    head_dim = q.shape[-1]
    dtype = q.dtype
    qk_dtype = _normalize_forward_mma_dtype(qk_dtype, q.dtype, "qk_dtype")
    pv_dtype = _normalize_forward_mma_dtype(pv_dtype, v.dtype, "pv_dtype")
    partial_dtype = O_partial.dtype
    safe_fwd_sync = bool(safe_fwd_sync)
    shared_work_valid = bool(shared_work_valid)
    safe_partial_v_tail = bool(safe_partial_v_tail)
    # The 192/80 split wins once the causal inverse-CSR domain reaches 128 KV
    # blocks (16K).  Preserve C308's 176/112 policy for the 64-block 8K path,
    # where the additional cubin regresses all three paired seeds.
    causal_softmax_registers, causal_store_registers = (
        (192, 80)
        if bool(causal) and int(max_num_kv_blocks) >= 128
        else (176, 112)
    )
    return_temperature_lse = bool(return_temperature_lse)
    if return_temperature_lse != (LSE_temperature_partial is not None):
        raise ValueError(
            "return_temperature_lse must match LSE_temperature_partial presence"
        )
    lse_temperature_scale = float(lse_temperature_scale)
    if not math.isfinite(lse_temperature_scale) or lse_temperature_scale <= 0.0:
        raise ValueError(
            f"lse_temperature_scale must be finite and > 0, got {lse_temperature_scale}"
        )
    lse_temperature_inv_scale = 1.0 / lse_temperature_scale
    n_block_size = int(blk_kv)
    head_q = q.shape[1]
    qhead_per_kv = head_q // head_kv
    paged_kv = page_table is not None
    if not bool(use_prepare_scheduler):
        raise RuntimeError("sparse forward requires prepare scheduler")
    schedule_enabled = k2q_row_ptr.shape[1] > 1
    page_size = int(k.shape[2]) if paged_kv else None
    if paged_kv:
        k_kernel, v_kernel = _prepare_paged_kv_for_tma(k, v, n_block_size)
    else:
        k_kernel = k
        v_kernel = v
    O_partial_flat = O_partial.reshape(-1, head_dim).contiguous()
    Q_flat = q.reshape(-1, head_dim).contiguous()
    Q_gather4_desc = (
        create_q_gather4_tma_desc(
            Q_flat,
            box_x=64,
        )
        if qhead_per_kv in (1, 2, 4)
        else None
    )
    if schedule is None:
        schedule = prepare_sparse_fwd_schedule_and_split(
            k2q_row_ptr=k2q_row_ptr,
            k2q_q_indices=k2q_q_indices,
            k2q_qsplit_indices=k2q_qsplit_indices,
            split_counts=split_counts,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            total_q=int(q.shape[0]),
            max_seqlen_q=max_seqlen_q,
            topk=int(O_partial.shape[0]),
            head_kv=head_kv,
            qhead_per_kv=qhead_per_kv,
            blk_kv=n_block_size,
            device=q.device,
            enabled=schedule_enabled,
        )
    use_prepare_scheduler = schedule.enabled
    scheduler_metadata = schedule.scheduler_metadata
    work_count = schedule.work_count
    work_capacity = schedule.work_capacity
    if not use_prepare_scheduler or scheduler_metadata is None or work_count is None or work_capacity <= 0:
        raise RuntimeError("sparse forward requires a non-empty prepared schedule")
    compact_batch1_schedule = schedule.metadata_format == "compact_batch1_v1"
    last_cta_finalize = bool(last_cta_finalize)
    inferred_full_score_tiles = (
        not causal
        and cu_seqlens_k.numel() == 2
        and max_seqlen_k % n_block_size == 0
        and (
            (
                not paged_kv
                and seqused_k is None
                and max_seqlen_k == k.shape[0]
            )
            or (paged_kv and seqused_k is not None)
        )
    )
    if full_score_tiles is None:
        full_kv_blocks = inferred_full_score_tiles
    else:
        full_kv_blocks = bool(full_score_tiles)
        # full_kv_blocks states that every selected KV block is physically
        # complete, which is a property of the plan geometry, not of masking:
        # the diagonal causal mask is applied to the score fragment after the
        # QK MMA, so a block-aligned batch-one plan stays full under causal.
        # Explicit packed noncausal callers may also prove that all selected
        # request-local pages are complete. Their prepared plan owns the
        # per-request extent proof; this does not infer full pages from max-K.
        if full_kv_blocks and not (
            max_seqlen_k % n_block_size == 0
            and (
                cu_seqlens_k.numel() == 2
                or (not causal and paged_kv and seqused_k is not None)
            )
        ):
            raise ValueError(
                "full_score_tiles requires a block-aligned plan with complete selected pages"
            )
    cfg = kernel_cfg
    if cfg is None:
        cfg = select_msa_cfg(
            causal=bool(causal),
            full_kv_blocks=full_kv_blocks,
            sm=torch.cuda.get_device_capability(q.device),
            max_seqlen_k=max_seqlen_k,
            paged_kv=paged_kv,
            causal_registers=(
                causal_softmax_registers,
                causal_store_registers,
            ),
        )
    shuffled_output = cfg.output_layout is OutputLayout.SHUFFLED
    O_ragged_tma_desc = (
        _get_ragged_o_tma_desc(O_partial, O_partial_flat)
        if cfg.ragged_tma_store
        else None
    )
    if compact_batch1_schedule and not (
        cu_seqlens_q.numel() == 2
        and int(q.shape[0]) <= 8192
        and (shuffled_output or int(O_partial.shape[0]) == 16)
        and n_block_size == 128
        and head_kv == 4
        and int(k2q_row_ptr.shape[1] - 1) <= 8192
        and int(schedule.target_q_per_cta) <= 4096
    ):
        raise ValueError(
            "compact schedule is outside the batch-one Dynamic Top-16 contract"
        )
    if shuffled_output:
        if work_tile_offsets is None:
            raise ValueError("shuffled output requires per-work tile offsets")
        if (
            work_tile_offsets.dtype != torch.int32
            or work_tile_offsets.device != q.device
            or work_tile_offsets.ndim != 1
            or not work_tile_offsets.is_contiguous()
        ):
            raise ValueError("work tile offsets must be contiguous CUDA int32")
    elif work_tile_offsets is not None:
        raise ValueError("scattered output does not consume tile offsets")
    if last_cta_finalize and (
        cfg.use_clc
        or cfg.num_wgs != 5
        or shuffled_output
        or cfg.output_store.value != "stg"
        or bool(causal)
        or qhead_per_kv != 8
        or partial_dtype != torch.bfloat16
        or O_partial.shape[0] != 16
        or O_out is None
        or LSE_out is None
        or initial_out is None
        or initial_lse is None
    ):
        raise ValueError(
            "last_cta_finalize requires static 5-WG scattered-STG BF16 "
            "noncausal GQA8 Top-16 with fixed initial state"
        )

    key = (
        f"sparse_forward_sm100_csr_varlen_{cfg.symbol_suffix}",
        _SPARSE_FWD_ABI_VERSION,
        head_dim,
        n_block_size,
        qhead_per_kv,
        dtype,
        k.dtype,
        v.dtype,
        qk_dtype,
        pv_dtype,
        partial_dtype,
        bool(causal),
        bool(paged_kv),
        bool(use_prepare_scheduler),
        page_size,
        bool(seqused_k is not None),
        bool(return_temperature_lse),
        safe_fwd_sync,
        shared_work_valid,
        safe_partial_v_tail,
        compact_batch1_schedule,
        last_cta_finalize,
        "causal_register_policy",
        causal_softmax_registers,
        causal_store_registers,
    )
    if key not in _compile_cache:
        from mbsa.cute.src.common.aot_cache import try_load_aot, save_aot

        loaded = try_load_aot(key)
        if loaded is not None:
            _compile_cache[key] = loaded
        else:
            kernel = SparseAttentionForwardSm100(
                head_dim=head_dim,
                qheadperkv=qhead_per_kv,
                n_block_size=n_block_size,
                paged_kv=paged_kv,
                page_size=page_size,
                has_seqused_k=seqused_k is not None,
                causal=bool(causal),
                use_prepare_scheduler=use_prepare_scheduler,
                qk_dtype=_torch_dtype_to_cutlass_dtype(qk_dtype),
                pv_dtype=_torch_dtype_to_cutlass_dtype(pv_dtype),
                cfg=cfg,
                safe_fwd_sync=safe_fwd_sync,
                shared_work_valid=shared_work_valid,
                safe_partial_v_tail=safe_partial_v_tail,
                compact_batch1_schedule=compact_batch1_schedule,
                last_cta_finalize=last_cta_finalize,
                causal_softmax_registers=causal_softmax_registers,
                causal_store_registers=causal_store_registers,
            )
            _compile_cache[key] = cute.compile(
                kernel,
                to_cute_tensor_kvouter(k_kernel),
                to_cute_tensor_kvouter(v_kernel),
                to_cute_tensor_kvouter(k2q_q_indices),
                to_cute_tensor_kvouter(k2q_qsplit_indices),
                to_cute_tensor_kvouter(k2q_row_ptr),
                None if scheduler_metadata is None else to_cute_tensor_kvouter(scheduler_metadata),
                None if work_count is None else to_cute_tensor_kvouter(work_count),
                None if work_tile_offsets is None else to_cute_tensor_kvouter(work_tile_offsets),
                to_cute_tensor_kvouter(O_partial_flat),
                to_cute_tensor_kvouter(LSE_partial),
                None
                if LSE_temperature_partial is None
                else to_cute_tensor_kvouter(LSE_temperature_partial),
                None if not last_cta_finalize else to_cute_tensor_kvouter(split_counts),
                None if O_out is None else to_cute_tensor_kvouter(O_out),
                None if LSE_out is None else to_cute_tensor_kvouter(LSE_out),
                None if initial_out is None else to_cute_tensor_kvouter(initial_out),
                None if initial_lse is None else to_cute_tensor_kvouter(initial_lse),
                to_cute_tensor_kvouter(Q_flat),
                None if Q_gather4_desc is None else to_cute_tensor_kvouter(Q_gather4_desc),
                None
                if O_ragged_tma_desc is None
                else to_cute_tensor_kvouter(O_ragged_tma_desc),
                None if page_table is None else to_cute_tensor_kvouter(page_table),
                None if seqused_k is None else to_cute_tensor_kvouter(seqused_k),
                to_cute_tensor_kvouter(cu_seqlens_q),
                to_cute_tensor_kvouter(cu_seqlens_k),
                Float32(softmax_scale),
                Float32(lse_temperature_inv_scale),
                Int32(max_num_kv_blocks),
                Int32(head_kv),
                Int32(max_seqlen_q),
                Int32(work_capacity),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
            save_aot(key, _compile_cache[key])

    with torch.cuda.nvtx.range("Fwd_SparseAttn_Sm100_CsrVarlen"):
        _compile_cache[key](
            k_kernel,
            v_kernel,
            k2q_q_indices,
            k2q_qsplit_indices,
            k2q_row_ptr,
            scheduler_metadata,
            work_count,
            work_tile_offsets,
            O_partial_flat,
            LSE_partial,
            LSE_temperature_partial,
            split_counts if last_cta_finalize else None,
            O_out,
            LSE_out,
            initial_out,
            initial_lse,
            Q_flat,
            Q_gather4_desc,
            O_ragged_tma_desc,
            page_table,
            seqused_k,
            cu_seqlens_q,
            cu_seqlens_k,
            softmax_scale,
            lse_temperature_inv_scale,
            max_num_kv_blocks,
            head_kv,
            max_seqlen_q,
            work_capacity,
        )
    return schedule
