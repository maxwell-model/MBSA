# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sparse k2q CSR builder for SM100.

Thin dispatcher that calls the CUDA C++ kernel pipeline in
``src.sm100.build_k2q_csr``. Supports ``topK in {4, 8, 16, 32}`` and
``blk_kv in {32, 64, 128}`` -- other shapes raise ``ValueError`` rather than
silently falling back to a torch-reference path.
"""

from __future__ import annotations

from typing import Optional

import torch

from mbsa.backends.optimized.msa.cute.src.sm100.prepare_scheduler import SparseAttentionSchedule, SPARSE_SCHEDULE_MODEL


_SUPPORTED_TOPK = (4, 8, 16, 32)
# The CUDA pipeline templates on blk_kv and only uses it for
# per-batch row-count arithmetic, so 32/64 are exact -- not an approximation.
_SUPPORTED_BLK_KV = (32, 64, 128)
#: The histogram stage keeps one packed int16 counter per CSR row per warp in
#: shared memory (2 bytes per row). Halving blk_kv doubles the row count, so a
#: config that fits at 128 can fail at 64 -- and it fails as a bare
#: cudaErrorInvalidValue. Catch it here, where the numbers can be explained.
_CSR_SMEM_BYTES_PER_ROW = 2


def _check_csr_smem_budget(total_rows: int, device) -> None:
    if total_rows <= 0:
        return
    try:
        limit = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    except (AttributeError, RuntimeError):
        limit = 227 * 1024
    needed = total_rows * _CSR_SMEM_BYTES_PER_ROW
    if needed <= limit:
        return
    raise ValueError(
        f"build_k2q_csr needs {needed / 1024:.0f} KB of shared memory for its row "
        f"histogram ({total_rows} CSR rows x {_CSR_SMEM_BYTES_PER_ROW} B), above this "
        f"device's {limit / 1024:.0f} KB per-CTA limit. Use a larger blk_kv.")


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _scratch_geometry(
    total_q: int, total_rows: int, device, min_q_per_cta: int = 256
) -> tuple[int, int]:
    """Mirror the native launch geometry used to size caller-owned scratch."""
    per_warp_smem = ((total_rows + 1) >> 1) * 4
    warps = 4
    while warps > 1 and (warps * per_warp_smem) * 2 > 228 * 1024:
        warps >>= 1
    per_cta_smem = max(1, warps * per_warp_smem)
    max_ctas_per_sm = min(8, max(1, (228 * 1024) // per_cta_smem))
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    target_g = num_sms * min(max_ctas_per_sm, 3)
    if min_q_per_cta <= 0:
        raise ValueError("min_q_per_cta must be positive")
    max_g_for_q = _ceil_div(total_q, min_q_per_cta)
    groups = max(1, min(target_g, max_g_for_q, total_q))
    q_per_cta = _ceil_div(total_q, groups)
    groups = _ceil_div(total_q, q_per_cta)
    return warps, groups * warps


def allocate_k2q_csr_scratch(
    *,
    head_kv: int,
    total_q: int,
    total_rows: int,
    batch: int,
    max_kv_blocks: int,
    device,
    min_q_per_cta: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate caller-owned scratch matching native CSR histogram geometry.

    Args:
        head_kv: Positive KV-head count.
        total_q: Positive total packed query-token count.
        total_rows: Positive sum of per-request ceil(KV/blk_kv) sparse row counts.
        batch: Positive number of requests.
        max_kv_blocks: Positive maximum logical KV-block count per request.
        device: CUDA allocation device; its SM count participates in histogram scratch sizing.
        min_q_per_cta: Positive query-token target per histogram CTA, default 256; use the same
            value when building CSR.

    Returns:
        Four uninitialized CUDA int32 tensors: [Hkv,total_rows], [B,max_kv_blocks], [total_rows,2],
        and [histogram_warps,Hkv,total_rows]. The caller owns and retains them through queued CSR
        work.
    """
    if min(head_kv, total_q, total_rows, batch, max_kv_blocks) <= 0:
        raise ValueError("non-empty CSR scratch geometry must be positive")
    _, total_warp_groups = _scratch_geometry(
        total_q, total_rows, device, min_q_per_cta
    )
    return (
        torch.empty((head_kv, total_rows), dtype=torch.int32, device=device),
        torch.empty((batch, max_kv_blocks), dtype=torch.int32, device=device),
        torch.empty((total_rows, 2), dtype=torch.int32, device=device),
        torch.empty(
            (total_warp_groups, head_kv, total_rows),
            dtype=torch.int32,
            device=device,
        ),
    )


class SparseK2qCsrBuilderSm100:
    """Build the k2q CSR reverse index for sparse attention on SM100.

    The public API matches the historical CUTE DSL builder so callers
    (``sparse_index_utils.build_k2q_csr``, attention kernels) need no
    changes. Internally the kernel pipeline runs five CUDA C++ kernels:
    ``build_row_map`` -> ``hist`` -> ``row_prefix`` -> ``tile_prefix_smem``
    -> ``scatter`` (5 kernels + 2 ``cudaMemsetAsync``).
    """

    def __init__(self) -> None:
        # No persistent state -- the JIT-compiled extension is loaded
        # lazily by ``src.sm100.build_k2q_csr`` on first call.
        self._run = None
        self._run_with_schedule = None
        self._run_with_schedule_from_local_top16 = None

    def _ensure_loaded(self) -> None:
        if self._run is None:
            from mbsa.backends.optimized.msa.cute.src.sm100.build_k2q_csr import (
                run_build_k2q_csr,
                run_build_k2q_csr_with_schedule,
                run_build_k2q_csr_with_schedule_from_local_top16,
            )
            self._run = run_build_k2q_csr
            self._run_with_schedule = run_build_k2q_csr_with_schedule
            self._run_with_schedule_from_local_top16 = (
                run_build_k2q_csr_with_schedule_from_local_top16
            )

    def __call__(
        self,
        q2k_indices: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        *,
        total_k: int,
        blk_kv: int = 128,
        max_seqlen_k: Optional[int] = None,
        max_seqlen_q: Optional[int] = None,
        total_rows: Optional[int] = None,
        qhead_per_kv: int = 1,
        min_q_per_cta: int = 256,
        return_schedule: bool = False,
        buffers: tuple[
            torch.Tensor, torch.Tensor, SparseAttentionSchedule
        ] | None = None,
        scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        local_top16_storage: torch.Tensor | None = None,
        prefix_lens: torch.Tensor | None = None,
        candidate_partition_blocks: int | None = None,
        local_top16_dynamic_only: bool = False,
        canonicalize_dynamic_padding: bool = False,
        batch1_identity_rows: bool = True,
        compact_batch1_schedule: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, SparseAttentionSchedule]
    ):
        # ---- Validation ----------------------------------------------------
        if blk_kv not in _SUPPORTED_BLK_KV:
            raise ValueError(
                f"SparseK2qCsrBuilderSm100 supports blk_kv in "
                f"{_SUPPORTED_BLK_KV}, got {blk_kv}"
            )
        if q2k_indices.dtype != torch.int32:
            raise TypeError(
                f"q2k_indices must be torch.int32, got {q2k_indices.dtype}"
            )
        if q2k_indices.ndim != 3:
            raise ValueError(
                f"q2k_indices must be rank-3 [head_kv, total_q, topK], "
                f"got shape {tuple(q2k_indices.shape)}"
            )
        if not q2k_indices.is_contiguous():
            raise ValueError("q2k_indices must be contiguous")
        if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
            raise TypeError("cu_seqlens_q and cu_seqlens_k must be torch.int32")
        if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
            raise ValueError("cu_seqlens_q and cu_seqlens_k must be rank-1")
        if cu_seqlens_q.shape != cu_seqlens_k.shape:
            raise ValueError(
                "cu_seqlens_q and cu_seqlens_k must share shape [B + 1]"
            )
        if not (q2k_indices.is_cuda and cu_seqlens_q.is_cuda and cu_seqlens_k.is_cuda):
            raise ValueError("all inputs must be CUDA tensors")
        if (
            q2k_indices.device != cu_seqlens_q.device
            or q2k_indices.device != cu_seqlens_k.device
        ):
            raise ValueError("all inputs must share a device")
        if not cu_seqlens_q.is_contiguous() or not cu_seqlens_k.is_contiguous():
            raise ValueError("cu_seqlens_q and cu_seqlens_k must be contiguous")

        total_k = int(total_k)
        if total_k < 0:
            raise ValueError(f"total_k must be non-negative, got {total_k}")

        head_kv, total_q, topk = (int(v) for v in q2k_indices.shape)
        if topk not in _SUPPORTED_TOPK:
            raise ValueError(
                f"SparseK2qCsrBuilderSm100 only supports topK in "
                f"{_SUPPORTED_TOPK}, got {topk}"
            )
        if local_top16_storage is not None:
            if not return_schedule:
                raise ValueError("local Top-16 CSR fusion requires return_schedule=True")
            expected_topk = 16 if local_top16_dynamic_only else 32
            if topk != expected_topk:
                raise ValueError(
                    "local Top-16 CSR fusion requires dynamic Top-16 or final Top-32"
                )
            if prefix_lens is None or candidate_partition_blocks is None:
                raise ValueError(
                    "local Top-16 CSR fusion requires prefix_lens and "
                    "candidate_partition_blocks"
                )
            if (
                local_top16_storage.dtype != torch.float32
                or not local_top16_storage.is_cuda
                or not local_top16_storage.is_contiguous()
                or local_top16_storage.ndim != 3
                or tuple(local_top16_storage.shape[:1]) != (head_kv,)
                or int(local_top16_storage.shape[1]) < 32
                or int(local_top16_storage.shape[2]) != total_q
            ):
                raise ValueError(
                    "local_top16_storage must be contiguous CUDA FP32 "
                    "[head_kv,>=32,total_q]"
                )
            if (
                prefix_lens.dtype != torch.int32
                or not prefix_lens.is_cuda
                or not prefix_lens.is_contiguous()
                or prefix_lens.ndim != 1
            ):
                raise ValueError("prefix_lens must be contiguous CUDA int32 [B]")

        batch = int(cu_seqlens_q.shape[0] - 1)
        if batch < 0:
            raise ValueError("cu_seqlens tensors must have shape [B + 1]")
        if return_schedule and max_seqlen_k is None:
            raise ValueError("build_k2q_csr requires max_seqlen_k when return_schedule=True")
        max_k_tokens = int(max_seqlen_k) if max_seqlen_k is not None else total_k
        max_kv_blocks = _ceil_div(max(max_k_tokens, blk_kv), blk_kv)
        if total_rows is not None:
            total_rows = int(total_rows)
        elif batch == 1 and total_k % blk_kv == 0:
            total_rows = total_k // blk_kv
        else:
            # Request-local tails need separate rows even when their combined
            # token count is block-aligned. Use the existing allocation bound
            # unless a single request or explicit row count proves otherwise.
            total_rows = _ceil_div(total_k + batch * (blk_kv - 1), blk_kv)
        if total_rows < 0:
            raise ValueError(f"total_rows must be non-negative, got {total_rows}")
        total_rows = max(total_rows, 0)
        _check_csr_smem_budget(total_rows, q2k_indices.device)
        nnz_upper_bound = total_q * topk
        qhead_per_kv = int(qhead_per_kv)
        if qhead_per_kv <= 0:
            raise ValueError(f"qhead_per_kv must be positive, got {qhead_per_kv}")
        min_q_per_cta = int(min_q_per_cta)
        if min_q_per_cta <= 0:
            raise ValueError("min_q_per_cta must be positive")
        if return_schedule:
            if max_seqlen_q is None:
                raise ValueError("build_k2q_csr requires max_seqlen_q when return_schedule=True")
            max_seqlen_q = int(max_seqlen_q)
        device = q2k_indices.device
        if scratch is None and total_q > 0 and total_rows > 0 and head_kv > 0:
            # Backward-compatible convenience path. Formal callers provide
            # scratch explicitly so this allocation remains outside timing.
            scratch = allocate_k2q_csr_scratch(
                head_kv=head_kv,
                total_q=total_q,
                total_rows=total_rows,
                batch=batch,
                max_kv_blocks=max_kv_blocks,
                device=device,
                min_q_per_cta=min_q_per_cta,
            )
        if scratch is not None:
            if len(scratch) != 4:
                raise ValueError("CSR scratch must contain four tensors")
            _, total_warp_groups = _scratch_geometry(
                total_q, total_rows, device, min_q_per_cta
            )
            expected_shapes = (
                (head_kv, total_rows),
                (batch, max_kv_blocks),
                (total_rows, 2),
                (total_warp_groups, head_kv, total_rows),
            )
            for tensor, expected in zip(scratch, expected_shapes):
                if (
                    tuple(tensor.shape) != expected
                    or tensor.dtype != torch.int32
                    or tensor.device != device
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(
                        f"CSR scratch must be contiguous CUDA int32 {expected}"
                    )

        # ---- Output tensors ------------------------------------------------
        schedule = None
        if buffers is not None:
            if not return_schedule:
                raise ValueError("CSR buffers require return_schedule=True")
            k2q_row_ptr, k2q_q_indices, schedule = buffers
            if k2q_row_ptr.shape != (head_kv, total_rows + 1):
                raise ValueError(
                    "CSR row-pointer buffer shape does not match graph plan: "
                    f"got {tuple(k2q_row_ptr.shape)}, expected "
                    f"{(head_kv, total_rows + 1)}"
                )
            if k2q_q_indices.shape != (head_kv, nnz_upper_bound):
                raise ValueError(
                    "CSR query-index buffer shape does not match graph plan: "
                    f"got {tuple(k2q_q_indices.shape)}, expected "
                    f"{(head_kv, nnz_upper_bound)}"
                )
            if (
                k2q_row_ptr.device != device
                or k2q_q_indices.device != device
                or k2q_row_ptr.dtype != torch.int32
                or k2q_q_indices.dtype != torch.int32
            ):
                raise ValueError(
                    "CSR graph buffers must be int32 tensors on the input device"
                )
            if not k2q_row_ptr.is_contiguous() or not k2q_q_indices.is_contiguous():
                raise ValueError("CSR graph buffers must be contiguous")
            if (
                not schedule.enabled
                or schedule.scheduler_metadata is None
                or schedule.work_count is None
                or schedule.qsplit_indices is None
                or schedule.split_counts is None
            ):
                raise ValueError("CSR graph buffers require a complete schedule")
            expected_metadata_cols = (
                2 if schedule.metadata_format == "compact_batch1_v1" else 6
            )
            if schedule.metadata_format not in ("legacy6", "compact_batch1_v1"):
                raise ValueError(
                    f"unsupported CSR schedule metadata format {schedule.metadata_format!r}"
                )
            if schedule.scheduler_metadata.shape != (
                schedule.work_capacity,
                expected_metadata_cols,
            ):
                raise ValueError("CSR graph scheduler metadata shape does not match format")
            if bool(compact_batch1_schedule) != (
                schedule.metadata_format == "compact_batch1_v1"
            ):
                raise ValueError("CSR compact schedule option does not match graph buffers")
            if (
                schedule.qsplit_indices.shape != k2q_q_indices.shape
                or schedule.split_counts.shape != (total_q, head_kv)
            ):
                raise ValueError("CSR graph schedule shapes do not match the input")
            schedule_tensors = (
                schedule.scheduler_metadata,
                schedule.work_count,
                schedule.qsplit_indices,
                schedule.split_counts,
            )
            if any(t.device != device or t.dtype != torch.int32 for t in schedule_tensors):
                raise ValueError(
                    "CSR graph schedule tensors must be int32 tensors on the input device"
                )
            if any(not t.is_contiguous() for t in schedule_tensors):
                raise ValueError("CSR graph schedule tensors must be contiguous")
        else:
            k2q_row_ptr = torch.empty(
                (head_kv, total_rows + 1), dtype=torch.int32, device=device,
            )
            k2q_q_indices = torch.empty(
                (head_kv, nnz_upper_bound), dtype=torch.int32, device=device,
            )
        if return_schedule and buffers is None:
            compact_batch1_schedule = bool(compact_batch1_schedule)
            if compact_batch1_schedule and not (
                batch == 1
                and batch1_identity_rows
                and local_top16_storage is not None
                and local_top16_dynamic_only
                and topk == 16
                and blk_kv == 128
                and total_q <= 8192
                and total_q * topk <= (1 << 17)
                and total_rows == max_kv_blocks
                and max_kv_blocks <= 8192
                and head_kv == 4
            ):
                raise ValueError(
                    "compact schedule requires batch-one identity Dynamic Top-16 "
                    "with Sq<=8192 and at most 8192 KV blocks"
                )
            target_q_per_cta = SPARSE_SCHEDULE_MODEL.balanced_target_q_per_cta(
                total_q=total_q,
                topk=topk,
                blk_kv=blk_kv,
                head_kv=head_kv,
                qhead_per_kv=qhead_per_kv,
                device=device,
            )
            scheduler_metadata_capacity = SPARSE_SCHEDULE_MODEL.flat_schedule_capacity(
                total_rows=total_rows,
                total_q=total_q,
                topk=topk,
                head_kv=head_kv,
                target_q_per_cta=target_q_per_cta,
            )
            scheduler_metadata = torch.empty(
                (scheduler_metadata_capacity, 2 if compact_batch1_schedule else 6),
                dtype=torch.int32,
                device=device,
            )
            work_count = torch.empty((1,), dtype=torch.int32, device=device)
            qsplit_indices = torch.empty_like(k2q_q_indices)
            split_counts = torch.empty(
                (total_q, head_kv), dtype=torch.int32, device=device
            )
            schedule = SparseAttentionSchedule(
                enabled=True,
                scheduler_metadata=scheduler_metadata,
                work_count=work_count,
                qsplit_indices=qsplit_indices,
                split_counts=split_counts,
                target_q_per_cta=target_q_per_cta,
                metadata_format=(
                    "compact_batch1_v1" if compact_batch1_schedule else "legacy6"
                ),
            )

        # Empty workload short-circuit (the CUDA path also handles this,
        # but doing it here saves a JIT load for trivial calls).
        if total_rows == 0 or total_q == 0 or head_kv == 0 or topk == 0:
            k2q_row_ptr.zero_()
            k2q_q_indices.fill_(-1)
            if schedule is not None:
                schedule.work_count.zero_()
                schedule.split_counts.zero_()
                return k2q_row_ptr, k2q_q_indices, schedule
            return k2q_row_ptr, k2q_q_indices

        self._ensure_loaded()
        with torch.cuda.nvtx.range("SparseK2qCsr_Pipeline"):
            if schedule is None:
                self._run(
                    q2k_indices,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    k2q_row_ptr,
                    k2q_q_indices,
                    topk,
                    blk_kv,
                    total_rows,
                    max_kv_blocks,
                    min_q_per_cta,
                    scratch=scratch,
                )
            elif local_top16_storage is not None:
                self._run_with_schedule_from_local_top16(
                    local_top16_storage,
                    prefix_lens,
                    q2k_indices,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    k2q_row_ptr,
                    k2q_q_indices,
                    schedule.scheduler_metadata,
                    schedule.work_count,
                    schedule.qsplit_indices,
                    schedule.split_counts,
                    topk,
                    blk_kv,
                    total_rows,
                    max_kv_blocks,
                    int(candidate_partition_blocks),
                    schedule.target_q_per_cta,
                    schedule.work_capacity,
                    max_seqlen_q,
                    min_q_per_cta,
                    local_top16_dynamic_only,
                    canonicalize_dynamic_padding,
                    batch1_identity_rows,
                    scratch=scratch,
                )
            else:
                self._run_with_schedule(
                    q2k_indices,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    k2q_row_ptr,
                    k2q_q_indices,
                    schedule.scheduler_metadata,
                    schedule.work_count,
                    schedule.qsplit_indices,
                    schedule.split_counts,
                    topk,
                    blk_kv,
                    total_rows,
                    max_kv_blocks,
                    schedule.target_q_per_cta,
                    schedule.work_capacity,
                    max_seqlen_q,
                    min_q_per_cta,
                    scratch=scratch,
                )
        if schedule is not None:
            return k2q_row_ptr, k2q_q_indices, schedule
        return k2q_row_ptr, k2q_q_indices
