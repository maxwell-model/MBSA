# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CuTe DSL implementation of Compass token-sparse GQA attention forward.

The production kernel uses four warp-level MMAs and 128-bit ``cp.async`` K/V
gathers.  It returns attention output and LSE and can optionally emit the
teacher-probability side output used by the stage-2 KL loss.  Head dimension
128 and GQA group sizes up to 16 are supported.
"""

from __future__ import annotations

import math
from typing import Any

import torch

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cute.nvgpu import cpasync, warp
    from cutlass.cute.runtime import from_dlpack

    _CUTEDSL_OK = True
except ImportError:  # pragma: no cover - CPU-only development environments
    cuda = None
    cutlass = None
    cute = None
    utils = None
    from_dlpack = None
    _CUTEDSL_OK = False


_COMPILED_KERNELS: dict[tuple[Any, ...], Any] = {}


def is_cutedsl_available() -> bool:
    """Return whether CUDA Python and NVIDIA CUTLASS DSL are importable."""

    return _CUTEDSL_OK


if _CUTEDSL_OK:

    class _SparseGqaFwd:
        """TileLang-equivalent four-warp MMA implementation.

        The four warps split the 64 selected keys along N.  The MMA tile is
        16x64 with inactive GQA rows zero-padded, matching the TileLang
        ``FullRow`` lowering observed in its generated CUDA source.
        """

        threads = 128
        group_block = 16
        tile_k = 64
        head_dim = 128

        def __init__(
            self, softmax_scale: float, group_size: int, write_teacher: bool = False
        ) -> None:
            self.softmax_scale = softmax_scale
            self.group_size = group_size
            self.write_teacher = write_teacher

        @cute.jit
        def __call__(
            self,
            q: cute.Tensor,
            k: cute.Tensor,
            v: cute.Tensor,
            topk: cute.Tensor,
            q_global: cute.Tensor,
            out: cute.Tensor,
            lse: cute.Tensor,
            teacher: cute.Tensor,
            stream: cuda.CUstream,
        ) -> None:
            dtype = q.element_type
            layout_atom = cute.make_composed_layout(
                cute.make_swizzle(3, 3, 3), 0, cute.make_layout((8, 64), stride=(64, 1))
            )
            q_layout = cute.tile_to_shape(layout_atom, (self.group_block, self.head_dim), (0, 1))
            kv_layout = cute.tile_to_shape(layout_atom, (self.tile_k, self.head_dim), (0, 1))
            p_layout = cute.tile_to_shape(layout_atom, (self.group_block, self.tile_k), (0, 1))

            @cute.struct
            class SharedStorage:
                q: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(q_layout)], 1024]
                k: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(kv_layout)], 1024]
                v: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(kv_layout)], 1024]
                p: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(p_layout)], 1024]
                indices: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, self.tile_k], 16]
                valid: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, self.tile_k], 16]
                reduce_max: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, 4 * self.group_block], 16
                ]
                reduce_sum: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, 4 * self.group_block], 16
                ]

            copy_128b = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=128
            )
            copy_g2s_128b = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                dtype,
                num_bits_per_copy=128,
            )
            tiled_mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(dtype, cutlass.Float32, (16, 8, 16)),
                (1, 4, 1),
                permutation_mnk=(16, 64, 16),
            )
            self.kernel(
                q,
                k,
                v,
                topk,
                q_global,
                out,
                lse,
                teacher,
                q_layout,
                kv_layout,
                p_layout,
                copy_128b,
                copy_g2s_128b,
                tiled_mma,
                SharedStorage,
            ).launch(
                grid=(q.layout.shape[0], q.layout.shape[1], k.layout.shape[2]),
                block=(self.threads, 1, 1),
                smem=SharedStorage.size_in_bytes(),
                min_blocks_per_mp=5,
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mTopk: cute.Tensor,
            mQGlobal: cute.Tensor,
            mOut: cute.Tensor,
            mLse: cute.Tensor,
            mTeacher: cute.Tensor,
            q_layout: cute.ComposedLayout,
            kv_layout: cute.ComposedLayout,
            p_layout: cute.ComposedLayout,
            copy_128b: cute.CopyAtom,
            copy_g2s_128b: cute.CopyAtom,
            tiled_mma: cute.TiledMma,
            SharedStorage: cutlass.Constexpr,
        ) -> None:
            tidx, _, _ = cute.arch.thread_idx()
            q_idx, batch_idx, kv_head = cute.arch.block_idx()
            lane_in_row = tidx % 16
            row_in_wave = tidx // 16

            smem = utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sQ = storage.q.get_tensor(q_layout)
            sK = storage.k.get_tensor(kv_layout)
            sV = storage.v.get_tensor(kv_layout)
            sP = storage.p.get_tensor(p_layout)
            sIndices = storage.indices.get_tensor(cute.make_layout((self.tile_k,), stride=(1,)))
            sValid = storage.valid.get_tensor(cute.make_layout((self.tile_k,), stride=(1,)))
            sReduceMax = storage.reduce_max.get_tensor(
                cute.make_layout((4, self.group_block), stride=(self.group_block, 1))
            )
            sReduceSum = storage.reduce_sum.get_tensor(
                cute.make_layout((4, self.group_block), stride=(self.group_block, 1))
            )

            # Each thread transfers one aligned 16-byte vector.  Two waves
            # cover the padded 16x128 Q tile.
            for wave in cutlass.range_constexpr(2):
                row = wave * 8 + row_in_wave
                sQVec = cute.local_tile(sQ[row, None], (8,), (lane_in_row,))
                if row < self.group_size:
                    gQRow = mQ[q_idx, batch_idx, kv_head * self.group_size + row, None]
                    gQVec = cute.local_tile(gQRow, (8,), (lane_in_row,))
                    cute.copy(copy_128b, gQVec, sQVec)
                else:
                    sQVec.fill(mQ.element_type(0.0))
            cute.arch.barrier()

            thr_mma = tiled_mma.get_slice(tidx)
            tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
            tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
            sVt = cute.composition(
                sV, cute.make_layout((self.head_dim, self.tile_k), stride=(self.tile_k, 1))
            )
            tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
            tPrP = thr_mma.make_fragment_A(thr_mma.partition_A(sP))

            smem_copy_q = cute.make_tiled_copy_A(
                cute.make_copy_atom(
                    warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), mQ.element_type
                ),
                tiled_mma,
            )
            smem_copy_k = cute.make_tiled_copy_B(
                cute.make_copy_atom(
                    warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), mK.element_type
                ),
                tiled_mma,
            )
            smem_copy_v = cute.make_tiled_copy_B(
                cute.make_copy_atom(
                    warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), mV.element_type
                ),
                tiled_mma,
            )
            smem_copy_p = cute.make_tiled_copy_A(
                cute.make_copy_atom(
                    warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), mQ.element_type
                ),
                tiled_mma,
            )
            q_copy_thr = smem_copy_q.get_slice(tidx)
            k_copy_thr = smem_copy_k.get_slice(tidx)
            v_copy_thr = smem_copy_v.get_slice(tidx)
            p_copy_thr = smem_copy_p.get_slice(tidx)
            tSsQ = q_copy_thr.partition_S(sQ)
            tSrQCopy = q_copy_thr.retile(tSrQ)
            tSsK = k_copy_thr.partition_S(sK)
            tSrKCopy = k_copy_thr.retile(tSrK)
            tOsVt = v_copy_thr.partition_S(sVt)
            tOrVtCopy = v_copy_thr.retile(tOrVt)
            tPsP = p_copy_thr.partition_S(sP)
            tPrPCopy = p_copy_thr.retile(tPrP)

            for phase in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
                cute.copy(smem_copy_q, tSsQ[None, None, phase], tSrQCopy[None, None, phase])

            acc_o = cute.make_fragment(
                thr_mma.partition_shape_C((self.group_block, self.head_dim)), cutlass.Float32
            )
            acc_o.fill(0.0)
            row_max = cute.make_fragment(
                (acc_o.shape[0][0] * acc_o.shape[1]), cutlass.Float32
            )
            row_sum = cute.make_fragment_like(row_max, cutlass.Float32)
            row_max.fill(-cutlass.Float32.inf)
            row_sum.fill(0.0)

            score_coords = cute.make_identity_tensor((self.group_block, self.tile_k))
            tScoreCoords = thr_mma.partition_C(score_coords)
            tScoreCoordsMn = self._make_acc_tensor_mn_view(tScoreCoords)
            q_global = mQGlobal[q_idx]
            num_tiles = mTopk.layout.shape[2] // self.tile_k

            for tile_idx in cutlass.range_constexpr(num_tiles):
                lane_in_warp = tidx % 32
                warp_idx = tidx // 32
                warp_index = cutlass.Int32(0)
                if lane_in_warp < 16:
                    index_row = (
                        (lane_in_warp // 2) * 8
                        + warp_idx * 2
                        + lane_in_warp % 2
                    )
                    warp_index = mTopk[
                        batch_idx, q_idx, tile_idx * self.tile_k + index_row
                    ]

                for wave in cutlass.range_constexpr(8):
                    row = wave * 8 + row_in_wave
                    source_lane = wave * 2 + row_in_wave % 2
                    key_index = cute.arch.shuffle_sync(
                        warp_index, offset=source_lane, mask=-1, mask_and_clamp=31
                    )
                    sKVec = cute.local_tile(sK[row, None], (8,), (lane_in_row,))
                    sVVec = cute.local_tile(sV[row, None], (8,), (lane_in_row,))
                    gKRow = mK[key_index, batch_idx, kv_head, None]
                    gVRow = mV[key_index, batch_idx, kv_head, None]
                    gKVec = cute.local_tile(gKRow, (8,), (lane_in_row,))
                    gVVec = cute.local_tile(gVRow, (8,), (lane_in_row,))
                    cute.copy(copy_g2s_128b, gKVec, sKVec)
                    cute.copy(copy_g2s_128b, gVVec, sVVec)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()

                acc_s = cute.make_fragment(
                    thr_mma.partition_shape_C((self.group_block, self.tile_k)), cutlass.Float32
                )
                acc_s.fill(0.0)
                for phase in cutlass.range_constexpr(cute.size(tSsK.shape[2])):
                    cute.copy(smem_copy_k, tSsK[None, None, phase], tSrKCopy[None, None, phase])
                    cute.gemm(
                        tiled_mma, acc_s, tSrQ[None, None, phase], tSrK[None, None, phase], acc_s
                    )

                self._softmax_rescale(
                    acc_s,
                    acc_o,
                    row_max,
                    row_sum,
                    tScoreCoordsMn,
                    sReduceMax,
                    sReduceSum,
                    is_first=(tile_idx == 0),
                )

                accSMn = self._make_acc_tensor_mn_view(acc_s)
                for row in cutlass.range_constexpr(cute.size(accSMn.shape[0])):
                    for col in cutlass.range_constexpr(cute.size(accSMn.shape[1])):
                        coord = tScoreCoordsMn[row, col]
                        if coord[0] < self.group_size:
                            sP[coord[0], coord[1]] = accSMn[row, col].to(mQ.element_type)
                cute.arch.barrier()
                for phase in cutlass.range_constexpr(cute.size(tPsP.shape[2])):
                    cute.copy(smem_copy_p, tPsP[None, None, phase], tPrPCopy[None, None, phase])
                    cute.copy(smem_copy_v, tOsVt[None, None, phase], tOrVtCopy[None, None, phase])
                cute.arch.barrier()
                for phase in cutlass.range_constexpr(cute.size(tPsP.shape[2])):
                    cute.gemm(
                        tiled_mma, acc_o, tPrP[None, None, phase], tOrVt[None, None, phase], acc_o
                    )

            # ============================================================
            # Teacher-reuse second pass (QK only).  ``row_max`` / ``row_sum``
            # now hold the final online-softmax (m, l), so the normalized
            # probability of every selected slot can be recomputed and
            # atomic-added into the per-slot teacher accumulator.  Gated on a
            # compile-time flag so the O/LSE fast path is byte-for-byte
            # unchanged when teacher output is not requested.
            # ============================================================
            if cutlass.const_expr(self.write_teacher):
                for tile_idx in cutlass.range_constexpr(num_tiles):
                    if tidx < self.tile_k:
                        raw_index = mTopk[batch_idx, q_idx, tile_idx * self.tile_k + tidx]
                        is_valid = (
                            (raw_index >= 0)
                            & (raw_index < mK.layout.shape[0])
                            & (raw_index <= q_global)
                        )
                        sIndices[tidx] = cutlass.Int32(raw_index) if is_valid else 0
                        sValid[tidx] = cutlass.Int32(1) if is_valid else 0
                    cute.arch.barrier()

                    for wave in cutlass.range_constexpr(8):
                        row = wave * 8 + row_in_wave
                        key_index = sIndices[row]
                        sKVec = cute.local_tile(sK[row, None], (8,), (lane_in_row,))
                        if sValid[row] != 0:
                            gKRow = mK[key_index, batch_idx, kv_head, None]
                            gKVec = cute.local_tile(gKRow, (8,), (lane_in_row,))
                            cute.copy(copy_g2s_128b, gKVec, sKVec)
                        else:
                            sKVec.fill(mK.element_type(0.0))
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.barrier()

                    acc_s_t = cute.make_fragment(
                        thr_mma.partition_shape_C((self.group_block, self.tile_k)), cutlass.Float32
                    )
                    acc_s_t.fill(0.0)
                    for phase in cutlass.range_constexpr(cute.size(tSsK.shape[2])):
                        cute.copy(smem_copy_k, tSsK[None, None, phase], tSrKCopy[None, None, phase])
                        cute.gemm(
                            tiled_mma,
                            acc_s_t,
                            tSrQ[None, None, phase],
                            tSrK[None, None, phase],
                            acc_s_t,
                        )
                    self._write_teacher(
                        acc_s_t,
                        row_max,
                        row_sum,
                        tScoreCoordsMn,
                        sValid,
                        mTeacher,
                        batch_idx,
                        q_idx,
                        tile_idx * self.tile_k,
                    )
                    cute.arch.barrier()

            self._normalize(acc_o, row_sum)
            acc_o_mn = self._make_acc_tensor_mn_view(acc_o)
            output_coords = cute.make_identity_tensor((self.group_block, self.head_dim))
            tOutputCoords = thr_mma.partition_C(output_coords)
            tOutputCoordsMn = self._make_acc_tensor_mn_view(tOutputCoords)
            for row in cutlass.range_constexpr(cute.size(acc_o_mn.shape[0])):
                for col in cutlass.range_constexpr(cute.size(acc_o_mn.shape[1])):
                    coord = tOutputCoordsMn[row, col]
                    if coord[0] < self.group_size:
                        mOut[q_idx, batch_idx, kv_head * self.group_size + coord[0], coord[1]] = (
                            acc_o_mn[row, col].to(mOut.element_type)
                        )
                    if (coord[0] < self.group_size) & (coord[1] == 0):
                        mLse[batch_idx, q_idx, kv_head * self.group_size + coord[0]] = row_max[
                            row
                        ] * cutlass.Float32(self.softmax_scale) + cute.math.log(
                            row_sum[row], fastmath=True
                        )

        @cute.jit
        def _softmax_rescale(
            self,
            acc_s: cute.Tensor,
            acc_o: cute.Tensor,
            row_max: cute.Tensor,
            row_sum: cute.Tensor,
            score_coords_mn: cute.Tensor,
            reduce_max: cute.Tensor,
            reduce_sum: cute.Tensor,
            is_first: cutlass.Constexpr,
        ) -> None:
            acc_s_mn = self._make_acc_tensor_mn_view(acc_s)
            acc_o_mn = self._make_acc_tensor_mn_view(acc_o)
            tidx, _, _ = cute.arch.thread_idx()
            warp_idx = tidx // 32
            quad_lane = tidx % 4
            scale_log2 = cutlass.Float32(self.softmax_scale * 1.4426950408889634)

            # Stage 1: publish every row's per-warp maximum before a single
            # cross-warp synchronization.
            for row in cutlass.range_constexpr(cute.size(row_max)):
                global_row = score_coords_mn[row, 0][0]
                if global_row < self.group_size:
                    values = acc_s_mn[row, None].load()
                    current_max = values.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
                    current_max = self._quad_reduce_max(current_max)
                    if quad_lane == 0:
                        reduce_max[warp_idx, global_row] = current_max
            cute.arch.barrier()

            # Stage 2: finish all max reductions, form probabilities and
            # publish every row's partial sum. Apply the old-output online
            # softmax correction before the second shared-memory barrier.
            for row in cutlass.range_constexpr(cute.size(row_max)):
                global_row = score_coords_mn[row, 0][0]
                if global_row < self.group_size:
                    values = acc_s_mn[row, None].load()
                    current_max = reduce_max[0, global_row]
                    for reduce_warp in cutlass.range_constexpr(1, 4):
                        current_max = cute.arch.fmax(
                            current_max, reduce_max[reduce_warp, global_row]
                        )
                    previous_max = row_max[row]
                    if cutlass.const_expr(not is_first):
                        current_max = cute.arch.fmax(previous_max, current_max)
                    probabilities = cute.TensorSSA(
                        self._exp2(values * scale_log2 - current_max * scale_log2),
                        tuple(values.shape),
                        cutlass.Float32,
                    )
                    current_sum = probabilities.reduce(
                        cute.ReductionOp.ADD, cutlass.Float32.zero, 0
                    )
                    current_sum = self._quad_reduce_sum(current_sum)
                    if quad_lane == 0:
                        reduce_sum[warp_idx, global_row] = current_sum
                    if cutlass.const_expr(not is_first):
                        correction = self._exp2(
                            previous_max * scale_log2 - current_max * scale_log2
                        )
                        row_sum[row] *= correction
                        acc_o_mn[row, None] = acc_o_mn[row, None].load() * correction
                    row_max[row] = current_max
                    acc_s_mn[row, None] = probabilities
            cute.arch.barrier()

            # Stage 3: consume all cross-warp sums. row_sum already contains
            # the corrected contribution from preceding key tiles.
            for row in cutlass.range_constexpr(cute.size(row_max)):
                global_row = score_coords_mn[row, 0][0]
                if global_row < self.group_size:
                    current_sum = reduce_sum[0, global_row]
                    for reduce_warp in cutlass.range_constexpr(1, 4):
                        current_sum += reduce_sum[reduce_warp, global_row]
                    if cutlass.const_expr(not is_first):
                        current_sum += row_sum[row]
                    row_sum[row] = current_sum

        @cute.jit
        def _normalize(self, acc_o: cute.Tensor, row_sum: cute.Tensor) -> None:
            acc_o_mn = self._make_acc_tensor_mn_view(acc_o)
            for row in cutlass.range_constexpr(cute.size(row_sum)):
                scale = cute.arch.rcp_approx(row_sum[row])
                acc_o_mn[row, None] = acc_o_mn[row, None].load() * scale

        @cute.jit
        def _write_teacher(
            self,
            acc_s: cute.Tensor,
            row_max: cute.Tensor,
            row_sum: cute.Tensor,
            score_coords_mn: cute.Tensor,
            valid: cute.Tensor,
            teacher: cute.Tensor,
            batch_idx: cutlass.Int32,
            q_idx: cutlass.Int32,
            slot_base: cutlass.Constexpr,
        ) -> None:
            """Atomic-add ``sum_h softmax(scores)`` for this tile's slots.

            ``acc_s`` holds freshly recomputed raw QK scores; ``row_max`` and
            ``row_sum`` are the final online-softmax state, so the normalized
            probability equals ``exp2((s - m) * scale_log2) / l``.  Each C
            element is owned by exactly one thread, so the atomic accumulates
            once per (head, slot) across the group and across KV-head CTAs,
            matching the TileLang ``WRITE_TEACHER`` contract.
            """
            acc_s_mn = self._make_acc_tensor_mn_view(acc_s)
            scale_log2 = cutlass.Float32(self.softmax_scale * 1.4426950408889634)
            # 1D view over the contiguous slot dimension for this (batch,
            # query) row.  ``iterator + slot`` yields the global element
            # pointer consumed by the atomic accumulate.
            teacher_row = teacher[batch_idx, q_idx, None]
            for row in cutlass.range_constexpr(cute.size(row_max)):
                global_row = score_coords_mn[row, 0][0]
                if global_row < self.group_size:
                    inv_sum = cutlass.Float32(1.0) / cute.arch.fmax(
                        row_sum[row], cutlass.Float32(1.0e-20)
                    )
                    m_scaled = row_max[row] * scale_log2
                    for col in cutlass.range_constexpr(cute.size(acc_s_mn.shape[1])):
                        key = score_coords_mn[row, col][1]
                        if valid[key] != 0:
                            prob = (
                                cute.arch.exp2(acc_s_mn[row, col] * scale_log2 - m_scaled) * inv_sum
                            )
                            cute.arch.atomic_add(
                                teacher_row.iterator + (slot_base + key),
                                prob,
                                sem="relaxed",
                                scope="gpu",
                            )

        def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
            column_major = cute.make_layout(acc.layout.shape)
            mn_layout = cute.make_layout(
                (
                    (column_major.shape[0][1], column_major.shape[1]),
                    (column_major.shape[0][0], column_major.shape[2]),
                ),
                stride=(
                    (column_major.stride[0][1], column_major.stride[1]),
                    (column_major.stride[0][0], column_major.stride[2]),
                ),
            )
            return cute.make_tensor(acc.iterator, cute.composition(acc.layout, mn_layout))

        def _quad_reduce_max(self, value: cutlass.Float32) -> cutlass.Float32:
            value = cute.arch.fmax(
                value, cute.arch.shuffle_sync_bfly(value, offset=2, mask=-1, mask_and_clamp=31)
            )
            return cute.arch.fmax(
                value, cute.arch.shuffle_sync_bfly(value, offset=1, mask=-1, mask_and_clamp=31)
            )

        def _quad_reduce_sum(self, value: cutlass.Float32) -> cutlass.Float32:
            value += cute.arch.shuffle_sync_bfly(value, offset=2, mask=-1, mask_and_clamp=31)
            return value + cute.arch.shuffle_sync_bfly(value, offset=1, mask=-1, mask_and_clamp=31)

        def _exp2(self, value):
            if isinstance(value, cute.TensorSSA):
                result = cute.make_fragment(value.shape, cutlass.Float32)
                result.store(value)
                for idx in range(cute.size(value.shape)):
                    result[idx] = cute.arch.exp2(result[idx])
                return result.load()
            return cute.arch.exp2(value)


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_indices: torch.Tensor,
    q_global: torch.Tensor,
) -> None:
    if not _CUTEDSL_OK:
        raise ImportError("NVIDIA CUTLASS DSL and CUDA Python are required.")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("CuTe DSL sparse GQA forward currently requires bf16 Q/K/V.")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q, K, and V must have layout [sequence, batch, heads, dim].")
    if q.shape[1] != k.shape[1] or k.shape[:3] != v.shape[:3]:
        raise ValueError("Q/K/V batch and K/V sequence/head dimensions must match.")
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError("The number of Q heads must be divisible by the number of KV heads.")
    group_size = q.shape[2] // k.shape[2]
    if not 1 <= group_size <= 16 or q.shape[3] != 128 or k.shape[3] != 128 or v.shape[3] != 128:
        raise ValueError("CuTe DSL sparse GQA requires group_size <= 16 and QK/V dim 128.")
    if topk_indices.shape[:2] != (q.shape[1], q.shape[0]):
        raise ValueError("TopK must have shape [batch, query sequence, topk].")
    if topk_indices.shape[2] % 64 != 0:
        raise ValueError("TopK must be divisible by the 64-wide key tile.")
    if q_global.shape != (q.shape[0],):
        raise ValueError("q_global must have shape [query sequence].")
    if not all(t.is_cuda for t in (q, k, v, topk_indices, q_global)):
        raise ValueError("All inputs must be CUDA tensors.")


def sparse_gqa_stage2_attn_fwd_cutedsl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_indices: torch.Tensor,
    q_global: torch.Tensor,
    softmax_scale: float | None = None,
    *,
    teacher_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute token-sparse GQA attention and return ``(output, LSE)``.

    Inputs follow the existing stage-2 operator layout.  The output remains
    four-dimensional ``[Sq, B, Hq, Dv]`` so the prototype can be compared
    directly with the TileLang kernel without dispatcher-side reshaping.

    When ``teacher_probs`` is provided (``[B, S_local, TOPK]`` fp32), a second
    QK-only pass atomic-adds ``sum_h softmax(scores)`` into it, matching the
    TileLang ``WRITE_TEACHER`` side-output.  The buffer must be pre-zeroed by
    the caller because the kernel accumulates with ``atomic_add``.
    """

    _validate_inputs(q, k, v, topk_indices, q_global)
    group_size = q.shape[2] // k.shape[2]
    scale = float(softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(128))

    write_teacher = teacher_probs is not None
    # PyTorch disallows exporting a requires-grad tensor through DLPack.  The
    # surrounding custom autograd Function owns gradient propagation, so a
    # detached view is the correct CuTe DSL runtime boundary.
    q_c = q.detach().contiguous()
    k_c = k.detach().contiguous()
    v_c = v.detach().contiguous()
    topk_i32 = topk_indices.to(torch.int32).contiguous()
    qg_i32 = q_global.to(torch.int32).contiguous()
    out = torch.empty_like(q_c)
    lse = torch.empty((q.shape[1], q.shape[0], q.shape[2]), dtype=torch.float32, device=q.device)

    teacher_shape = (q.shape[1], q.shape[0], topk_i32.shape[2])
    if write_teacher:
        if tuple(teacher_probs.shape) != teacher_shape:
            raise ValueError(
                f"teacher_probs shape {tuple(teacher_probs.shape)} != {teacher_shape}."
            )
        if teacher_probs.dtype != torch.float32 or not teacher_probs.is_contiguous():
            raise ValueError("teacher_probs must be contiguous fp32.")
        teacher = teacher_probs
    else:
        # Kernel signature requires a teacher tensor; a dummy buffer keeps the
        # rank/shape compatible while WRITE_TEACHER=False compiles out writes.
        teacher = torch.empty(teacher_shape, dtype=torch.float32, device=q.device)

    base_args = (
        from_dlpack(q_c, assumed_align=16),
        from_dlpack(k_c, assumed_align=16),
        from_dlpack(v_c, assumed_align=16),
        from_dlpack(topk_i32, assumed_align=16),
        from_dlpack(qg_i32, assumed_align=16),
        from_dlpack(out, assumed_align=16),
        from_dlpack(lse, assumed_align=16),
        from_dlpack(teacher, assumed_align=16),
    )
    cute_args = base_args
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = (
        q.device.index,
        tuple(q.shape),
        tuple(k.shape),
        tuple(topk_i32.shape),
        scale,
        write_teacher,
    )
    compiled = _COMPILED_KERNELS.get(key)
    if compiled is None:
        op = _SparseGqaFwd(softmax_scale=scale, group_size=group_size, write_teacher=write_teacher)
        compiled = cute.compile(op, *cute_args, stream)
        _COMPILED_KERNELS[key] = compiled
    compiled(*cute_args, stream)
    return out, lse
