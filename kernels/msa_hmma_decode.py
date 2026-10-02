# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Paged Top32 GQA decode with warp HMMA and tiered split-K reduction.

Math/layout donor: v4 anchengc_persistent.py at 427e76a716d5.
One query token and eight Q heads share each selected page load. Cluster
splits reduce through a transaction-counted DSMEM mailbox; larger splits
publish FP32 normalized O/log2-LSE states for a subsequent combine kernel.
"""
from __future__ import annotations

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack


class HmmaDecode:
    threads_per_group = 128
    experimental_groups = 6
    group_size = 8
    group_block = 16
    tile_k = 64
    head_dim = 128
    validate_topk = True

    def __init__(self, scale: float, split_k: int, query_len: int,
                 vectorized: bool = True, internal_groups: int = 1):
        self.softmax_scale = scale
        self.split_k = split_k
        self.query_len = query_len
        self.vectorized = vectorized
        self.internal_groups = internal_groups
        self.threads = self.threads_per_group * internal_groups
        self.cluster_size = split_k if split_k <= 8 else 1

    @cute.jit
    def __call__(self, q: cute.Tensor, kv: cute.Tensor, route: cute.Tensor,
                 table: cute.Tensor, lengths: cute.Tensor, out: cute.Tensor,
                 lse: cute.Tensor, partial: cute.Tensor, stream: cuda.CUstream):
        atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3), 0, cute.make_layout((8, 64), stride=(64, 1))
        )
        q_layout = cute.tile_to_shape(atom, (8, 128), (0, 1))
        kv_tile = cute.tile_to_shape(atom, (64, 128), (0, 1))
        p_tile = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3), 0,
            cute.make_layout((64, 8), stride=(1, 64)),
        )
        if cutlass.const_expr(self.internal_groups == 1):
            kv_layout = kv_tile
            p_layout = p_tile
        else:
            kv_layout = cute.tile_to_shape(
                atom, (64, 128, self.internal_groups), (0, 1, 2)
            )
            p_layout = cute.tile_to_shape(
                p_tile, (64, 8, self.internal_groups), (0, 1, 2)
            )
        dtype = q.element_type
        mailbox_size = self.split_k * 1032 if 1 < self.split_k <= 8 else 1

        @cute.struct
        class Storage:
            q: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(q_layout)], 1024]
            k: cute.struct.Align[cute.struct.MemRange[
                dtype, cute.cosize(kv_layout)
            ], 1024]
            v: cute.struct.Align[cute.struct.MemRange[
                dtype, cute.cosize(kv_layout)
            ], 1024]
            p: cute.struct.Align[cute.struct.MemRange[
                dtype, cute.cosize(p_layout)
            ], 1024]
            valid: cute.struct.MemRange[cutlass.Int32, 64 * self.internal_groups]
            reduce_max: cute.struct.MemRange[cutlass.Float32, 64 * self.internal_groups]
            reduce_sum: cute.struct.MemRange[cutlass.Float32, 64 * self.internal_groups]
            mailbox: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, mailbox_size], 16]
            full: cute.struct.MemRange[cutlass.Int64, 1]

        if cutlass.const_expr(self.vectorized):
            copy = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), dtype,
                num_bits_per_copy=128,
            )
        else:
            copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype,
                                       num_bits_per_copy=16)
        mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(dtype, cutlass.Float32, (16, 8, 16)),
            (4, 1, 1), permutation_mnk=(64, 8, 16),
        )
        self.kernel(q, kv, route, table, lengths, out, lse, partial,
                    q_layout, kv_layout, p_layout, copy, mma, Storage).launch(
            grid=(self.split_k, q.shape[0], kv.shape[3]),
            block=(self.threads, 1, 1), cluster=(self.cluster_size, 1, 1),
            smem=Storage.size_in_bytes(), stream=stream, use_pdl=True,
        )

    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mKV: cute.Tensor, mRoute: cute.Tensor,
               mTable: cute.Tensor, mLengths: cute.Tensor, mOut: cute.Tensor,
               mLse: cute.Tensor, mPartial: cute.Tensor,
               q_layout: cute.ComposedLayout, kv_layout: cute.ComposedLayout,
               p_layout: cute.ComposedLayout, copy_g2s_128b: cute.CopyAtom,
               tiled_mma: cute.TiledMma, Storage: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        split, q_idx, kv_head = cute.arch.block_idx()
        group = tidx // self.threads_per_group
        group_tid = tidx % self.threads_per_group
        cute.arch.griddepcontrol_wait()
        req = q_idx // self.query_len
        query_pos = mLengths[req] - self.query_len + q_idx % self.query_len
        smem = utils.SmemAllocator()
        storage = smem.allocate(Storage)
        sQ = storage.q.get_tensor(q_layout)
        if cutlass.const_expr(self.internal_groups == 1):
            sK = storage.k.get_tensor(kv_layout)
            sV = storage.v.get_tensor(kv_layout)
            sP = storage.p.get_tensor(p_layout)
        else:
            sK = storage.k.get_tensor(kv_layout)[None, None, group]
            sV = storage.v.get_tensor(kv_layout)[None, None, group]
            sP = storage.p.get_tensor(p_layout)[None, None, group]
        sValidAll = storage.valid.get_tensor(cute.make_layout(
            (self.internal_groups, 64), stride=(64, 1)
        ))
        sReduceMaxAll = storage.reduce_max.get_tensor(cute.make_layout(
            (self.internal_groups, 4, 16), stride=(64, 16, 1)
        ))
        sReduceSumAll = storage.reduce_sum.get_tensor(cute.make_layout(
            (self.internal_groups, 16, 4), stride=(64, 4, 1)
        ))
        sValid = sValidAll[group, None]
        sReduceMax = sReduceMaxAll[group, None, None]
        sReduceSum = sReduceSumAll[group, None, None]
        sState = cute.make_tensor(
            cute.recast_ptr(storage.k.data_ptr(), dtype=cutlass.Float32),
            cute.make_layout(
                (self.internal_groups, 8, 129),
                stride=(
                    cute.cosize(kv_layout) // self.internal_groups // 2,
                    129,
                    1,
                ),
            ),
        )
        mailbox = storage.mailbox.data_ptr()
        full = storage.full.data_ptr()
        if cutlass.const_expr(1 < self.split_k <= 8):
            if tidx == 0:
                cute.arch.mbarrier_init(full, 1)
            cute.arch.mbarrier_init_fence()
            if (split == 0) & (tidx == 0):
                cute.arch.mbarrier_arrive_and_expect_tx(full, self.split_k * 1032 * 4)
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

        lane_in_row = group_tid % 16
        row_in_wave = group_tid // 16
        gQVec = cute.local_tile(mQ[q_idx, kv_head * 8 + row_in_wave, None], (8,), (lane_in_row,))
        sQVec = cute.local_tile(sQ[row_in_wave, None], (8,), (lane_in_row,))
        if group == 0:
            cute.copy(copy_g2s_128b, gQVec, sQVec)
        cute.arch.cp_async_commit_group()
        thr_mma = tiled_mma.get_slice(group_tid)
        sQActive = cute.local_tile(
            sQ,
            (self.group_size, self.head_dim),
            (0, 0),
        )
        sK0 = sK
        sV0 = sV
        sVt0 = cute.composition(
            sV0,
            cute.make_layout(
                (self.head_dim, self.tile_k),
                stride=(self.tile_k, 1),
            ),
        )
        tSrK = thr_mma.make_fragment_A(thr_mma.partition_A(sK0))
        tSrQ = thr_mma.make_fragment_B(thr_mma.partition_B(sQActive))
        tOrV = thr_mma.make_fragment_A(thr_mma.partition_A(sVt0))
        # ``sP`` is the column-major C[key,head] store view. ``sPLoad``
        # aliases those bytes as row-major operand B[head,key].
        sPLoad = cute.composition(
            sP,
            cute.make_layout(
                (self.group_size, self.tile_k),
                stride=(self.tile_k, 1),
            ),
        )
        tPrP = thr_mma.make_fragment_B(
            thr_mma.partition_B(sPLoad)
        )
        score_coords = cute.make_identity_tensor(
            (self.tile_k, self.group_size)
        )
        tScoreCoords = thr_mma.partition_C(score_coords)

        ldmatrix = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(
                transpose=False, num_matrices=4
            ),
            mQ.element_type,
        )
        smem_copy_k = cute.make_tiled_copy_A(ldmatrix, tiled_mma)
        smem_copy_q = cute.make_tiled_copy_B(ldmatrix, tiled_mma)
        smem_copy_v = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(
                    transpose=True, num_matrices=4
                ),
                mQ.element_type,
            ),
            tiled_mma,
        )
        p_store_atom = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(
                transpose=True,
                num_matrices=2,
            ),
            mQ.element_type,
        )
        p_store_copy_c = cute.make_tiled_copy_C_atom(
            p_store_atom, tiled_mma
        )
        p_store_copy = cute.make_tiled_copy_S(
            p_store_atom, p_store_copy_c
        )
        # ``sPLoad`` is the exact [head,key] operand-B alias. The native
        # B mapping performs the reload directly into every M-partitioned
        # warp's PV fragment, eliminating the register shuffle transpose.
        smem_copy_p = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(
                    transpose=False, num_matrices=2
                ),
                mQ.element_type,
            ),
            tiled_mma,
        )

        k_copy_thr = smem_copy_k.get_slice(group_tid)
        q_copy_thr = smem_copy_q.get_slice(group_tid)
        v_copy_thr = smem_copy_v.get_slice(group_tid)
        p_copy_thr = smem_copy_p.get_slice(group_tid)
        tSsK = k_copy_thr.partition_S(sK0)
        tSrKCopy = k_copy_thr.retile(tSrK)
        tSsQ = q_copy_thr.partition_S(sQActive)
        tSrQCopy = q_copy_thr.retile(tSrQ)
        tOsV = v_copy_thr.partition_S(sVt0)
        tOrVCopy = v_copy_thr.retile(tOrV)
        tPsP = p_copy_thr.partition_S(sPLoad)
        tPrPCopy = p_copy_thr.retile(tPrP)
        p_store_thr = p_store_copy.get_slice(group_tid)
        tPsPStore = p_store_thr.partition_D(sP)
        rPStoreShape = cute.shape(p_store_thr.partition_S(sP))
        rPStoreLayout = cute.make_layout(rPStoreShape[:3])
        acc_o = cute.make_fragment(
            thr_mma.partition_shape_C(
                (self.head_dim, self.group_size)
            ),
            cutlass.Float32,
        )
        acc_o.fill(0.0)
        row_max = cute.make_fragment((2,), cutlass.Float32)
        row_sum = cute.make_fragment_like(row_max, cutlass.Float32)
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)
        if cutlass.const_expr(self.internal_groups == 1):
            chunk_blocks = (mRoute.shape[2] + self.split_k - 1) // self.split_k
            num_tiles = chunk_blocks * 2
        else:
            chunk_blocks = (mRoute.shape[2] + self.internal_groups - 1 - group) // self.internal_groups
            num_tiles = chunk_blocks * 2
        route_head = kv_head if mRoute.shape[0] != 1 else 0
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        for phase in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
            cute.copy(smem_copy_q, tSsQ[None, None, phase], tSrQCopy[None, None, phase])

        for tile_idx in cutlass.range(num_tiles, unroll=1):
            if cutlass.const_expr(self.internal_groups == 1):
                slot = split * chunk_blocks + tile_idx // 2
            else:
                slot = group + self.internal_groups * (tile_idx // 2)
            logical = cutlass.Int32(-1)
            if slot < mRoute.shape[2]:
                logical = mRoute[route_head, q_idx, slot]
            block_valid = (logical >= 0) & (logical < mTable.shape[1]) & (logical * 128 <= query_pos)
            page = cutlass.Int32(0)
            if block_valid:
                page = mTable[req, logical]
            block_valid = block_valid & (page >= 0) & (page < mKV.shape[0])
            for wave_idx in cutlass.range_constexpr(8):
                row = wave_idx * 8 + row_in_wave
                offset = (tile_idx % 2) * 64 + row
                valid = block_valid & (logical * 128 + offset <= query_pos)
                if lane_in_row == 0:
                    sValid[row] = cutlass.Int32(valid)
                sKVec = cute.local_tile(sK[row, None], (8,), (lane_in_row,))
                sVVec = cute.local_tile(sV[row, None], (8,), (lane_in_row,))
                if valid:
                    krow = mKV[cutlass.Int64(page), 0, offset, kv_head, None]
                    vrow = mKV[cutlass.Int64(page), 1, offset, kv_head, None]
                    cute.copy(copy_g2s_128b, cute.local_tile(krow, (8,), (lane_in_row,)), sKVec)
                    cute.copy(copy_g2s_128b, cute.local_tile(vrow, (8,), (lane_in_row,)), sVVec)
                else:
                    sKVec.fill(mQ.element_type(0.0))
                    sVVec.fill(mQ.element_type(0.0))
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            self._group_sync(group)
            acc_s = cute.make_fragment(
                thr_mma.partition_shape_C(
                    (self.tile_k, self.group_size)
                ),
                cutlass.Float32,
            )
            acc_s.fill(0.0)
            for phase in cutlass.range_constexpr(
                cute.size(tSsK.shape[2])
            ):
                cute.copy(
                    smem_copy_k,
                    tSsK[None, None, phase],
                    tSrKCopy[None, None, phase],
                )
                cute.gemm(
                    tiled_mma,
                    acc_s,
                    tSrK[None, None, phase],
                    tSrQ[None, None, phase],
                    acc_s,
                )

            if cutlass.const_expr(self.validate_topk):
                for head_slot in cutlass.range_constexpr(2):
                    for key_slot in cutlass.range_constexpr(2):
                        index = ((head_slot, key_slot), 0, 0)
                        key = tScoreCoords[index][0]
                        if sValid[key] == 0:
                            acc_s[index] = -cutlass.Float32.inf

            self._softmax_rescale_swapab(
                acc_s,
                acc_o,
                row_max,
                row_sum,
                sReduceMax,
                sReduceSum,
                is_first=(tile_idx == 0),
                is_last=(tile_idx + 1 == num_tiles),
            )

            # ``acc_s`` is C[key, head].  STSM publishes it through the
            # aliased sPt[key, head] view; PV reloads the same bytes as
            # sP[head, key] for V^T @ P^T.
            tPrPStore = p_store_copy.retile(acc_s)
            rPStore = cute.make_rmem_tensor(
                rPStoreLayout.shape,
                mQ.element_type,
            )
            rPStore.store(tPrPStore.load().to(mQ.element_type))
            cute.copy(p_store_copy, rPStore, tPsPStore)
            self._group_sync(group)

            if tile_idx + 1 == num_tiles:
                self._finish_softmax_sum_swapab(
                    row_sum,
                    sReduceSum,
                )

            for phase in cutlass.range_constexpr(
                cute.size(tPsP.shape[2])
            ):
                cute.copy(
                    smem_copy_p,
                    tPsP[None, None, phase],
                    tPrPCopy[None, None, phase],
                )
                cute.copy(
                    smem_copy_v,
                    tOsV[None, None, phase],
                    tOrVCopy[None, None, phase],
                )
            # All P/V shared reads are complete.  The next iteration may
            # overwrite those allocations while PV consumes registers.
            self._group_sync(group)
            for phase in cutlass.range_constexpr(
                cute.size(tPsP.shape[2])
            ):
                cute.gemm(
                    tiled_mma,
                    acc_o,
                    tOrV[None, None, phase],
                    tPrP[None, None, phase],
                    acc_o,
                )


        cute.arch.griddepcontrol_launch_dependents()
        self._normalize_swapab(acc_o, row_sum)
        output_coords = cute.make_identity_tensor((128, 8))
        coords = thr_mma.partition_C(output_coords)
        for hs in cutlass.range_constexpr(2):
            log_z = -cutlass.Float32.inf
            if row_sum[hs] > 0.0:
                log_z = row_max[hs] * cutlass.Float32(self.softmax_scale * 1.4426950408889634)
                log_z += cute.math.log2(row_sum[hs], fastmath=True)
            for m8 in cutlass.range_constexpr(2):
                for m64 in cutlass.range_constexpr(2):
                    idx = ((hs, m8), m64, 0)
                    dim, head = coords[idx]
                    value = acc_o[idx]
                    if cutlass.const_expr(self.internal_groups > 1):
                        sState[group, head, dim] = value
                        if dim == 0:
                            sState[group, head, 128] = log_z
                    elif cutlass.const_expr(self.split_k == 1):
                        mOut[q_idx, kv_head * 8 + head, dim] = value.to(mOut.element_type)
                        if dim == 0:
                            mLse[q_idx, kv_head * 8 + head] = log_z * cutlass.Float32(0.6931471805599453)
                    elif cutlass.const_expr(self.split_k <= 8):
                        # Every rank, including rank zero, publishes exactly
                        # 8*(128+1)*4 bytes to the leader's FULL mailbox.
                        addr = split * 1032 + head * 129 + dim
                        self._store_mailbox(mailbox + addr, value, full)
                        if dim == 0:
                            self._store_mailbox(mailbox + split * 1032 + head * 129 + 128,
                                                log_z, full)
                    else:
                        mPartial[split, q_idx, kv_head, head, dim] = value
                        if dim == 0:
                            mPartial[split, q_idx, kv_head, head, 128] = log_z

        if cutlass.const_expr(self.internal_groups > 1):
            cute.arch.barrier()
            if (group == 0) | (group == 2) | (group == 4):
                self._merge_group_states(
                    sState, sReduceMaxAll, group, group + 1, group_tid
                )
            cute.arch.barrier()
            if group == 0:
                self._merge_group_states(
                    sState, sReduceMaxAll, 0, 2, group_tid
                )
            cute.arch.barrier()
            if group == 0:
                self._merge_group_states(
                    sState, sReduceMaxAll, 0, 4, group_tid
                )
            cute.arch.barrier()

            warp_idx = tidx // 32
            lane = tidx % 32
            for pass_idx in cutlass.range_constexpr(2):
                chunk = warp_idx + pass_idx * 24
                if chunk < 32:
                    element = chunk * 32 + lane
                    head = element // 128
                    dim = element % 128
                    mOut[q_idx, kv_head * 8 + head, dim] = sState[0, head, dim].to(
                        mOut.element_type
                    )
            if tidx < 8:
                mLse[q_idx, kv_head * 8 + tidx] = (
                    sState[0, tidx, 128]
                    * cutlass.Float32(0.6931471805599453)
                )

        if cutlass.const_expr(self.internal_groups == 1 and 1 < self.split_k <= 8):
            if split == 0:
                cute.arch.mbarrier_wait(full, 0)
                for head in cutlass.range_constexpr(8):
                    maximum = -cutlass.Float32.inf
                    for peer in cutlass.range_constexpr(self.split_k):
                        maximum = cute.arch.fmax(maximum, (mailbox + peer * 1032 + head * 129 + 128).load())
                    numerator = cutlass.Float32(0.0)
                    denominator = cutlass.Float32(0.0)
                    if maximum != -cutlass.Float32.inf:
                        for peer in cutlass.range_constexpr(self.split_k):
                            weight = cute.arch.exp2((mailbox + peer * 1032 + head * 129 + 128).load() - maximum)
                            denominator += weight
                            numerator += weight * (mailbox + peer * 1032 + head * 129 + tidx).load()
                    value = cutlass.Float32(0.0)
                    log_z = -cutlass.Float32.inf
                    if denominator > 0.0:
                        value = numerator / denominator
                        log_z = (maximum + cute.math.log2(denominator, fastmath=True)) * cutlass.Float32(0.6931471805599453)
                    mOut[q_idx, kv_head * 8 + head, tidx] = value.to(mOut.element_type)
                    if tidx == 0:
                        mLse[q_idx, kv_head * 8 + head] = log_z
            # Leader's DSMEM stays alive until all remote stores and reads
            # have completed. Every rank, including empty splits, arrives.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

    @cute.jit
    def _merge_group_states(
        self,
        states: cute.Tensor,
        merge_meta: cute.Tensor,
        left: cutlass.Int32,
        right: cutlass.Int32,
        group_tid: cutlass.Int32,
    ) -> None:
        if group_tid < 8:
            head = group_tid
            lse_left = states[left, head, 128]
            lse_right = states[right, head, 128]
            maximum = cute.arch.fmax(lse_left, lse_right)
            alpha = cutlass.Float32(0.0)
            beta = cutlass.Float32(0.0)
            merged_lse = -cutlass.Float32.inf
            if maximum != -cutlass.Float32.inf:
                weight_left = cute.arch.exp2(lse_left - maximum)
                weight_right = cute.arch.exp2(lse_right - maximum)
                denominator = weight_left + weight_right
                inverse = cute.arch.rcp_approx(denominator)
                alpha = weight_left * inverse
                beta = weight_right * inverse
                merged_lse = maximum + cute.math.log2(
                    denominator, fastmath=True
                )
            merge_meta[left, 0, head] = alpha
            merge_meta[left, 1, head] = beta
            states[left, head, 128] = merged_lse
        self._group_sync(left)
        for head in cutlass.range_constexpr(8):
            states[left, head, group_tid] = (
                merge_meta[left, 0, head] * states[left, head, group_tid]
                + merge_meta[left, 1, head] * states[right, head, group_tid]
            )

    @cute.jit
    def _store_mailbox(self, ptr: cute.Pointer, value: cutlass.Float32,
                       full: cute.Pointer):
        atom = cute.make_copy_atom(cpasync.CopyDsmemStoreOp(), cutlass.Float32)
        src = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Float32)
        src[0] = value
        dst = cute.make_tensor(cute.arch.map_dsmem_ptr(ptr, 0), cute.make_layout(1))
        cute.copy(atom, src, dst, mbar_ptr=cute.arch.map_dsmem_ptr(full, 0))

    @cute.jit
    def _group_sync(self, group: cutlass.Int32) -> None:
        if cutlass.const_expr(self.internal_groups == 1):
            cute.arch.barrier()
        else:
            cute.arch.barrier(
                barrier_id=group + 1,
                number_of_threads=self.threads_per_group,
            )

    @cute.jit
    def _octet_reduce_max(self, value: cutlass.Float32) -> cutlass.Float32:
        value = cute.arch.fmax(
            value,
            cute.arch.shuffle_sync_bfly(
                value, offset=4, mask=-1, mask_and_clamp=31
            ),
        )
        value = cute.arch.fmax(
            value,
            cute.arch.shuffle_sync_bfly(
                value, offset=8, mask=-1, mask_and_clamp=31
            ),
        )
        value = cute.arch.fmax(
            value,
            cute.arch.shuffle_sync_bfly(
                value, offset=16, mask=-1, mask_and_clamp=31
            ),
        )
        return value


    @cute.jit
    def _octet_reduce_sum(self, value: cutlass.Float32) -> cutlass.Float32:
        value += cute.arch.shuffle_sync_bfly(
            value, offset=4, mask=-1, mask_and_clamp=31
        )
        value += cute.arch.shuffle_sync_bfly(
            value, offset=8, mask=-1, mask_and_clamp=31
        )
        value += cute.arch.shuffle_sync_bfly(
            value, offset=16, mask=-1, mask_and_clamp=31
        )
        return value



    @cute.jit
    def _normalize_swapab(
        self,
        acc_o: cute.Tensor,
        row_sum: cute.Tensor,
    ) -> None:
        for head_slot in cutlass.range_constexpr(2):
            if cutlass.const_expr(self.validate_topk):
                scale = cutlass.Float32(0.0)
                if row_sum[head_slot] > 0.0:
                    scale = cute.arch.rcp_approx(row_sum[head_slot])
            else:
                scale = cute.arch.rcp_approx(row_sum[head_slot])
            for m8 in cutlass.range_constexpr(2):
                for m64 in cutlass.range_constexpr(2):
                    index = ((head_slot, m8), m64, 0)
                    acc_o[index] *= scale


    @cute.jit
    def _softmax_rescale_swapab(
        self,
        acc_s: cute.Tensor,
        acc_o: cute.Tensor,
        row_max: cute.Tensor,
        row_sum: cute.Tensor,
        reduce_max: cute.Tensor,
        reduce_sum: cute.Tensor,
        is_first: cutlass.Boolean,
        is_last: cutlass.Boolean,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        group = tidx // 128
        group_tid = tidx % 128
        lane_in_warp = group_tid % 32
        warp_idx = group_tid // 32
        head_pair = lane_in_warp % 4
        scale_log2 = cutlass.Float32(
            self.softmax_scale * 1.4426950408889634
        )

        for head_slot in cutlass.range_constexpr(2):
            value0 = acc_s[((head_slot, 0), 0, 0)]
            value1 = acc_s[((head_slot, 1), 0, 0)]
            current_max = self._octet_reduce_max(
                cute.arch.fmax(value0, value1)
            )
            if lane_in_warp < 4:
                reduce_max[
                    warp_idx, head_pair * 2 + head_slot
                ] = current_max
        self._group_sync(group)

        current_maxes = cute.make_fragment((2,), cutlass.Float32)
        corrections = cute.make_fragment((2,), cutlass.Float32)
        corrections.fill(1.0)
        for head_slot in cutlass.range_constexpr(2):
            head = head_pair * 2 + head_slot
            current_max = reduce_max[0, head]
            for reduce_warp in cutlass.range_constexpr(1, 4):
                current_max = cute.arch.fmax(
                    current_max, reduce_max[reduce_warp, head]
                )
            previous_max = row_max[head_slot]
            if not is_first:
                current_max = cute.arch.fmax(
                    previous_max, current_max
                )
            current_maxes[head_slot] = current_max
            if cutlass.const_expr(self.validate_topk):
                probability0 = cutlass.Float32(0.0)
                probability1 = cutlass.Float32(0.0)
                # An entirely invalid row has max=-inf. Avoid the undefined
                # -inf-(-inf) subtraction and define its probability as zero.
                if current_max != -cutlass.Float32.inf:
                    probability0 = cute.arch.exp2(
                        (acc_s[((head_slot, 0), 0, 0)] - current_max)
                        * scale_log2
                    )
                    probability1 = cute.arch.exp2(
                        (acc_s[((head_slot, 1), 0, 0)] - current_max)
                        * scale_log2
                    )
            else:
                probability0 = cute.arch.exp2(
                    (acc_s[((head_slot, 0), 0, 0)] - current_max)
                    * scale_log2
                )
                probability1 = cute.arch.exp2(
                    (acc_s[((head_slot, 1), 0, 0)] - current_max)
                    * scale_log2
                )
            acc_s[((head_slot, 0), 0, 0)] = probability0
            acc_s[((head_slot, 1), 0, 0)] = probability1
            if not is_first:
                if cutlass.const_expr(self.validate_topk):
                    if current_max != -cutlass.Float32.inf:
                        corrections[head_slot] = cute.arch.exp2(
                            (previous_max - current_max) * scale_log2
                        )
                else:
                    corrections[head_slot] = cute.arch.exp2(
                        (previous_max - current_max) * scale_log2
                    )

        for head_slot in cutlass.range_constexpr(2):
            head = head_pair * 2 + head_slot
            current_sum = self._octet_reduce_sum(
                acc_s[((head_slot, 0), 0, 0)]
                + acc_s[((head_slot, 1), 0, 0)]
            )
            if not is_first:
                correction = corrections[head_slot]
                row_sum[head_slot] *= correction
                for m8 in cutlass.range_constexpr(2):
                    for m64 in cutlass.range_constexpr(2):
                        index = ((head_slot, m8), m64, 0)
                        acc_o[index] *= correction
            row_sum[head_slot] += current_sum
            if is_last:
                if lane_in_warp < 4:
                    reduce_sum[head, warp_idx] = row_sum[head_slot]
            row_max[head_slot] = current_maxes[head_slot]


    @cute.jit
    def _finish_softmax_sum_swapab(
        self,
        row_sum: cute.Tensor,
        reduce_sum: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        group_tid = tidx % 128
        lane_in_warp = group_tid % 32
        head_pair = lane_in_warp % 4
        reduce_copy_128b = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float32,
            num_bits_per_copy=128,
        )
        reduce_fragment = cute.make_rmem_tensor((4,), cutlass.Float32)
        for head_slot in cutlass.range_constexpr(2):
            head = head_pair * 2 + head_slot
            leader_lane = head_pair + 4 * head_slot
            current_sum = cutlass.Float32(0.0)
            if lane_in_warp == leader_lane:
                cute.copy(
                    reduce_copy_128b,
                    reduce_sum[head, None],
                    reduce_fragment,
                )
                for reduce_warp in cutlass.range_constexpr(4):
                    current_sum += reduce_fragment[reduce_warp]
            current_sum = cute.arch.shuffle_sync(
                current_sum,
                offset=leader_lane,
                mask=-1,
                mask_and_clamp=31,
            )
            row_sum[head_slot] = current_sum


@cute.kernel
def _combine(partial: cute.Tensor, out: cute.Tensor, lse: cute.Tensor,
             splits: cutlass.Constexpr):
    cute.arch.griddepcontrol_wait()
    cute.arch.griddepcontrol_launch_dependents()
    tid, _, _ = cute.arch.thread_idx()
    qi, qh, dim_group = cute.arch.block_idx()
    kh, head = qh // 8, qh % 8
    lane = tid % 32
    shard = lane // 8
    if cutlass.const_expr(out.shape[0] <= 2):
        dim_base = dim_group * 32 + lane % 8
    else:
        dim_base = (tid // 32) * 32 + lane % 8
    maxima = cute.make_rmem_tensor(cute.make_layout(splits // 4), cutlass.Float32)
    maximum = -cutlass.Float32.inf
    for i in cutlass.range_constexpr(splits // 4):
        maxima[i] = partial[i * 4 + shard, qi, kh, head, 128]
        maximum = cute.arch.fmax(maximum, maxima[i])
    for delta in cutlass.range_constexpr(2):
        maximum = cute.arch.fmax(maximum, cute.arch.shuffle_sync_bfly(
            maximum, offset=8 << delta, mask=-1, mask_and_clamp=31))
    numerator = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
    numerator.fill(0.0)
    denominator = cutlass.Float32(0.0)
    if maximum != -cutlass.Float32.inf:
        for i in cutlass.range_constexpr(splits // 4):
            weight = cute.arch.exp2(maxima[i] - maximum)
            denominator += weight
            for col in cutlass.range_constexpr(4):
                numerator[col] += weight * partial[i * 4 + shard, qi, kh, head, dim_base + col * 8]
    for delta in cutlass.range_constexpr(2):
        denominator += cute.arch.shuffle_sync_bfly(
            denominator, offset=8 << delta, mask=-1, mask_and_clamp=31)
        for col in cutlass.range_constexpr(4):
            numerator[col] += cute.arch.shuffle_sync_bfly(
                numerator[col], offset=8 << delta, mask=-1, mask_and_clamp=31)
    if shard == 0:
        for col in cutlass.range_constexpr(4):
            value = cutlass.Float32(0.0)
            if denominator > 0.0:
                value = numerator[col] / denominator
            out[qi, qh, dim_base + col * 8] = value.to(out.element_type)
        if (tid == 0) & (dim_group == 0):
            log_z = -cutlass.Float32.inf
            if denominator > 0.0:
                log_z = (maximum + cute.math.log2(denominator, fastmath=True)) * cutlass.Float32(0.6931471805599453)
            lse[qi, qh] = log_z



@cute.jit
def _launch_combine(partial: cute.Tensor, out: cute.Tensor, lse: cute.Tensor,
                    stream: cuda.CUstream):
    _combine(partial, out, lse, partial.shape[0]).launch(
        grid=(out.shape[0], 32, 4 if out.shape[0] <= 2 else 1),
        block=(32 if out.shape[0] <= 2 else 128, 1, 1), stream=stream, use_pdl=True,
    )


_COMPILED: dict = {}


def default_split_k(total_q: int, topk: int = 32, *, sm_count: int = 72) -> int:
    # B300A has 72 SMs; B300 SXM6 has 148. Keep the policy graph-stable
    # and avoid overpartitioning the smaller device into extra cluster waves.
    target_grid = 512 if sm_count >= 128 else 256
    target = max(1, min(topk, target_grid // max(1, total_q * 4)))
    return 1 << (target.bit_length() - 1)


class PreparedHmmaDecode:
    """Caller-owned tensors and precompiled current-stream decode launches.

    Route/table/length contents may change in place between graph replays.
    Prepare a new instance to bind different tensor addresses. No route value
    is read on the CPU and device validity masking is always enabled.
    """

    def __init__(self, q, kv_cache, topk_idx, block_table, seq_lens, *,
                 num_kv_heads=4, softmax_scale=128**-0.5, decode_query_len=1,
                 split_k=None, out=None, lse_out=None, workspace=None,
                 internal_groups=1):
        if q.ndim != 3 or tuple(q.shape[1:]) != (32, 128) or q.dtype != torch.bfloat16:
            raise ValueError("q must be BF16 [total_q,32,128]")
        if num_kv_heads != 4 or kv_cache.ndim != 5 or tuple(kv_cache.shape[1:]) != (2, 128, 4, 128):
            raise ValueError("kv_cache must be [pages,2,128,4,128]")
        if kv_cache.dtype != q.dtype:
            raise ValueError("kv_cache must be BF16")
        if (topk_idx.ndim != 3 or topk_idx.shape[0] not in (1, 4)
                or topk_idx.shape[1] != q.shape[0] or not 1 <= topk_idx.shape[2] <= 63):
            raise ValueError("route must be [1|4,total_q,topk], topk=1..63")
        if (decode_query_len < 1 or seq_lens.ndim != 1
                or q.shape[0] != seq_lens.numel() * decode_query_len or q.shape[0] == 0):
            raise ValueError("total_q must equal nonzero batch * decode_query_len")
        if block_table.ndim != 2 or block_table.shape[0] != seq_lens.numel():
            raise ValueError("block_table must be [batch,max_pages]")
        inputs = (q, kv_cache, topk_idx, block_table, seq_lens)
        if not q.is_cuda or any(t.device != q.device for t in inputs):
            raise ValueError("all tensors must share a CUDA device")
        if any(t.dtype != torch.int32 for t in (topk_idx, block_table, seq_lens)):
            raise ValueError("route, table, lengths must be int32")
        # Preserve strided inputs without copying whole tensors. Aligned rows
        # use cp.async; other layouts use synchronous scalar loads into SMEM.
        vectorized = all(t.stride(-1) == 1 and t.data_ptr() % 16 == 0
                         and all(s % 8 == 0 for s in t.stride()[:-1])
                         for t in (q, kv_cache))
        self.split_k = (default_split_k(q.shape[0], topk_idx.shape[2],
                                      sm_count=torch.cuda.get_device_properties(q.device).multi_processor_count)
                        if split_k is None else int(split_k))
        if self.split_k not in (1, 2, 4, 8, 16, 32):
            raise ValueError("split_k must be 1,2,4,8,16,32")
        self.internal_groups = int(internal_groups)
        if self.internal_groups not in (1, HmmaDecode.experimental_groups):
            raise ValueError("internal_groups must be 1 or experimental 6")
        if self.internal_groups > 1 and self.split_k != 1:
            raise ValueError("six internal groups require split_k=1")
        if out is None:
            out = torch.empty_like(q)
        if lse_out is None:
            lse_out = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
        if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
            raise ValueError("out must match q shape/dtype/device")
        if tuple(lse_out.shape) != tuple(q.shape[:2]) or lse_out.dtype != torch.float32 or lse_out.device != q.device:
            raise ValueError("lse_out must be FP32 [total_q,32] on q.device")
        shape = (self.split_k, q.shape[0], 4, 8, 129) if self.split_k > 8 else (1, 1, 1, 1, 1)
        if workspace is None:
            workspace = torch.empty(shape, dtype=torch.float32, device=q.device)
        if tuple(workspace.shape) != shape or workspace.dtype != torch.float32 or workspace.device != q.device:
            raise ValueError(f"workspace must be FP32 {shape} on q.device")
        self.out, self.lse, self.workspace = out, lse_out, workspace
        self.tensors = (*inputs, out, lse_out, workspace)
        self.args = tuple(from_dlpack(t.detach(), assumed_align=16 if i < 2 and vectorized else t.element_size())
                          for i, t in enumerate(self.tensors))
        key = (q.device.index, tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in self.tensors),
               float(softmax_scale), self.split_k, int(decode_query_len), vectorized,
               self.internal_groups)
        stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
        if key not in _COMPILED:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare HMMA decode before CUDA Graph capture")
            op = HmmaDecode(
                float(softmax_scale), self.split_k, int(decode_query_len),
                vectorized, self.internal_groups,
            )
            fwd = cute.compile(op, *self.args, stream)
            merge = cute.compile(_launch_combine, self.args[-1], self.args[-3], self.args[-2], stream) if self.split_k > 8 else None
            _COMPILED[key] = fwd, merge
        self.compiled, self.combine = _COMPILED[key]

    def attention(self):
        stream = cuda.CUstream(torch.cuda.current_stream(self.out.device).cuda_stream)
        self.compiled(*self.args, stream)

    def reduce(self):
        if self.combine is not None:
            stream = cuda.CUstream(torch.cuda.current_stream(self.out.device).cuda_stream)
            self.combine(self.args[-1], self.args[-3], self.args[-2], stream)

    def __call__(self):
        self.attention()
        self.reduce()
        return self.out, self.lse
