# SPDX-License-Identifier: Apache-2.0
"""SM100 MXFP8 Q x packed MXFP6 K router with native UMMA and partition Top-16."""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as helpers
import cutlass.utils.blockscaled_layout as bs
from cutlass import Float32, Int32
import torch
import cuda.bindings.driver as cuda
from .router import _dynamic, merge_native_top16
from cutlass.cute.nvgpu import tcgen05, cpasync
from ..common import pipeline as common_pipeline


class Mxfp6IndexerLocalTop16Sm103:
    def __init__(self, partition_blocks=256):
        if partition_blocks not in (64, 128, 256, 512, 1024, 2048, 4096, 8192):
            raise ValueError("invalid MXFP6 partition capacity")
        self.partition_blocks = int(partition_blocks)

    @cute.jit
    def __call__(self, q, sq, k, pages, metadata, scores, ids, route, stream):
        mma = helpers.make_blockscaled_trivial_tiled_mma(
            cutlass.Float8E4M3FN,
            cutlass.Float6E2M3FN,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float8E8M0FNU,
            32,
            tcgen05.CtaGroup.ONE,
            (128, 128),
        )
        qa = helpers.make_smem_layout_a(mma, (128, 128, 128), cutlass.Float8E4M3FN, 1)
        kb = helpers.make_smem_layout_b(mma, (128, 128, 128), cutlass.Uint8, 3)
        sa = bs.make_smem_layout_sfa(mma, (128, 128, 128), 32, 1)
        sb = bs.make_smem_layout_sfb(mma, (128, 128, 128), 32, 3)
        ta = bs.make_tmem_layout_sfa(
            mma, (128, 128, 128), 32, cute.slice_(sa, (None, None, None, 0))
        )
        tb = bs.make_tmem_layout_sfb(
            mma, (128, 128, 128), 32, cute.slice_(sb, (None, None, None, 0))
        )
        q_tensor = cute.make_tensor(
            q.iterator,
            cute.make_layout(
                (q.shape[0], 128, q.shape[1]),
                stride=(q.stride[0], q.stride[2], q.stride[1]),
            ),
        )
        q_tma = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            q_tensor,
            cute.slice_(qa, (None, None, None, 0)),
            (128, 128, 128),
            mma,
            (1, 1, 1, 1),
        )
        # Three residues express the native 12800-byte page stride exactly
        # in FP6 element units: each descriptor advances three pages (38400B).
        descriptors = []
        for residue in cutlass.range_constexpr(3):
            ptr = cute.make_ptr(
                cutlass.Float6E2M3FN,
                k.iterator.toint() + residue * 12800,
                mem_space=cute.AddressSpace.gmem,
                assumed_align=128,
            )
            tensor = cute.make_tensor(
                ptr,
                cute.make_layout(
                    (128, 128, cute.ceil_div(k.shape[0] - residue, 3)),
                    stride=(128, 1, 51200),
                ),
            )
            descriptor = cute.nvgpu.make_tiled_tma_atom_B(
                cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
                tensor,
                cute.slice_(kb, (None, None, None, 0)),
                (128, 128, 128),
                mma,
                (1, 1, 1, 1),
                internal_type=cutlass.Uint8,
            )
            descriptors.append(descriptor)
        # Copy the 512 scale bytes as bytes, preserving UE8M0 encodings.
        # The same K-stage barrier covers 12288 payload bytes plus scales.
        scale_ptr = cute.make_ptr(
            cutlass.Uint8,
            k.iterator.toint() + 12288,
            mem_space=cute.AddressSpace.gmem,
            assumed_align=128,
        )
        scale_tensor = cute.make_tensor(
            scale_ptr, cute.make_layout((512, k.shape[0]), stride=(1, 12800))
        )
        scale_atom, scale_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            scale_tensor,
            cute.make_layout((512, 1), stride=(1, 512)),
            (512, 1),
        )
        self.score_kernel(
            q,
            sq,
            k,
            pages,
            metadata,
            scores,
            ids,
            mma,
            qa,
            kb,
            sa,
            sb,
            ta,
            tb,
            tuple(descriptors),
            scale_atom,
            scale_tma,
            q_tma,
        ).launch(
            grid=(q.shape[1], metadata.shape[0], scores.shape[1]),
            block=(192, 1, 1),
            stream=stream,
        )
        merge_native_top16(scores, ids, route, True).launch(
            grid=(scores.shape[0], cute.ceil_div(scores.shape[2], 4), 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def score_kernel(
        self,
        q,
        sq,
        k,
        pages,
        metadata,
        scores,
        ids,
        mma,
        qa,
        kb,
        sa,
        sb,
        ta,
        tb,
        descriptors,
        scale_atom,
        scale_tma,
        q_tma,
    ):
        head, tile, partition = cute.arch.block_idx()
        q_begin = metadata[tile, 0]
        q_end = metadata[tile, 1]
        q_start = q_begin + metadata[tile, 2] * 128
        page_begin = metadata[tile, 3]
        kv_tokens = metadata[tile, 4]
        last_query = cutlass.min(q_start + 127, q_end - 1)
        max_visible = (
            kv_tokens - (q_end - q_begin) + last_query - q_begin + 128
        ) // 128
        begin = cutlass.max(Int32(8), partition * self.partition_blocks)
        end = cutlass.min((partition + 1) * self.partition_blocks, max_visible - 8)
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        smem = utils.SmemAllocator()
        sQ = smem.allocate_tensor(cutlass.Float8E4M3FN, qa.outer, 128, swizzle=qa.inner)
        sK = smem.allocate_tensor(cutlass.Uint8, kb.outer, 128, swizzle=kb.inner)
        sA = smem.allocate_tensor(cutlass.Float8E8M0FNU, sa, 128)
        sB = smem.allocate_tensor(cutlass.Float8E8M0FNU, sb, 128)
        rawA = cute.make_tensor(
            cute.make_ptr(
                cutlass.Uint8,
                sA.iterator.toint(),
                mem_space=cute.AddressSpace.smem,
                assumed_align=16,
            ),
            cute.make_layout(512),
        )
        scale_shared = cute.make_tensor(
            sB.iterator, cute.make_layout((512, 3), stride=(1, 512))
        )
        scale_global = cute.local_tile(scale_tma, (512, 1), (None, None))
        sfS, sfG = cpasync.tma_partition(
            scale_atom,
            0,
            cute.make_layout(1),
            scale_shared,
            cute.group_modes(scale_global, 0, 2),
        )
        # Request tail rows may read subsequent queries, but their scores
        # are never emitted; tensor-end TMA reads receive hardware OOB fill.
        q_tma_global = cute.domain_offset((q_start, 0, 0), q_tma.tma_tensor)
        q_tile = cute.local_tile(q_tma_global, (128, 128), (None, None, None))
        q_partition = mma.get_slice(0).partition_A(q_tile)
        qS, qG = cpasync.tma_partition(
            q_tma.atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(q_partition, 0, 3),
        )
        q_barriers = smem.allocate_tensor(cutlass.Int64, cute.make_layout(2), 8)
        q_producer, q_consumer = common_pipeline.PipelineTmaUmma.create(
            barrier_storage=q_barriers.iterator,
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=16384,
            defer_sync=True,
        ).make_participants()
        hold = smem.allocate_tensor(Int32, cute.make_layout(1), 4)
        acc_barriers = smem.allocate_tensor(cutlass.Int64, cute.make_layout(4), 8)
        k_barriers = smem.allocate_tensor(cutlass.Int64, cute.make_layout(6), 8)
        acc_pipe = common_pipeline.PipelineUmmaAsync.create(
            barrier_storage=acc_barriers.iterator,
            num_stages=2,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            defer_sync=True,
        )
        acc_producer, _ = acc_pipe.make_participants()
        k_producer, k_consumer = common_pipeline.PipelineTmaUmma.create(
            barrier_storage=k_barriers.iterator,
            num_stages=3,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=12800,
            defer_sync=True,
        ).make_participants()
        cute.arch.mbarrier_init_fence()
        for i in cutlass.range(tid, 512, 192):
            slot = i // 4
            block = i % 4
            row = slot // 4 + (slot % 4) * 32
            word = Int32(0x7F7F7F7F)
            if q_start + row < q_end:
                word = sq[q_start + row, head]
            rawA[i] = cutlass.Uint8((word >> (block * 8)) & 255)
        cute.arch.barrier()
        cute.arch.fence_view_async_shared()
        tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=160)
        allocator = utils.TmemAllocator(
            hold.iterator,
            barrier_for_retrieve=tmem_barrier,
            allocator_warp_id=0,
        )
        fake = mma.make_fragment_C(cute.append(mma.partition_shape_C((128, 128)), 2))
        fa = mma.make_fragment_A(sQ)
        fb = mma.make_fragment_B(sK)
        if warp == 5:
            q_empty = q_producer.acquire_and_advance()
            cute.copy(
                q_tma.atom,
                qG[(None, 0, 0, head)],
                qS[(None, q_empty.index)],
                tma_bar_ptr=q_empty.barrier,
            )
            for item in cutlass.range(begin, end, unroll=1):
                block = item
                physical = pages[page_begin + block]
                empty = k_producer.acquire_and_advance()
                cute.copy(
                    scale_atom,
                    sfG[(None, 0, physical)],
                    sfS[(None, empty.index)],
                    tma_bar_ptr=empty.barrier,
                )
                for residue in cutlass.range_constexpr(3):
                    if physical % 3 == residue:
                        tma = descriptors[residue]
                        gK = cute.local_tile(
                            tma.tma_tensor, (128, 128), (None, None, None)
                        )
                        tCgK = mma.get_slice(0).partition_B(gK)
                        tKsK, tKgK = cpasync.tma_partition(
                            tma.atom,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sK, 0, 3),
                            cute.group_modes(tCgK, 0, 3),
                        )
                        cute.copy(
                            tma.atom,
                            tKgK[(None, 0, 0, physical // 3)],
                            tKsK[(None, empty.index)],
                            tma_bar_ptr=empty.barrier,
                        )
            k_producer.tail()
            q_producer.tail()
        if warp < 5:
            # NamedBarrier is aligned: every TMEM participant must execute
            # reserve at this one instruction, before splitting by warp role.
            pool = allocator.reserve(512)
            acc = pool.allocate_tensor(fake.layout, Float32)
            tA = pool.allocate_tensor(ta, cutlass.Float8E8M0FNU)
            tb_staged = cute.make_layout((tb.shape, 3), stride=(tb.stride, 16))
            tB_staged = pool.allocate_tensor(tb_staged, cutlass.Float8E8M0FNU)
            if warp == 4:
                q_full = q_consumer.wait_and_advance()
                atom = cute.make_copy_atom(
                    tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), cutlass.Float8E8M0FNU
                )
                ca = tcgen05.make_s2t_copy(atom, cute.filter_zeros(tA))
                thra = ca.get_slice(0)
                cute.copy(
                    ca,
                    tcgen05.get_s2t_smem_desc_tensor(
                        ca, thra.partition_S(cute.filter_zeros(sA[(None, None, None, 0)]))
                    ),
                    thra.partition_D(cute.filter_zeros(tA)),
                )
                for item in cutlass.range(begin, end, unroll=1):
                    kfull = k_consumer.wait_and_advance()
                    empty = acc_producer.acquire_and_advance()
                    tB = cute.make_tensor(tB_staged.iterator + kfull.index * 16, tb)
                    b_atom = cute.make_copy_atom(
                        tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), cutlass.Float8E8M0FNU
                    )
                    cb = tcgen05.make_s2t_copy(b_atom, cute.filter_zeros(tB))
                    thrb = cb.get_slice(0)
                    cute.copy(
                        cb,
                        tcgen05.get_s2t_smem_desc_tensor(
                            cb,
                            thrb.partition_S(
                                cute.filter_zeros(sB[(None, None, None, kfull.index)])
                            ),
                        ),
                        thrb.partition_D(cute.filter_zeros(tB)),
                    )
                    accum = acc[(None, None, None, empty.index)]
                    block_mma = cute.make_mma_atom(mma.op)
                    block_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for chunk in cutlass.range_constexpr(4):
                        cute.gemm(
                            block_mma,
                            accum,
                            [fa[(None, None, chunk, 0)], tA[(None, None, chunk)]],
                            [fb[(None, None, chunk, kfull.index)], tB[(None, None, chunk)]],
                            accum,
                        )
                        block_mma.set(tcgen05.Field.ACCUMULATE, True)
                    empty.commit()
                    # Release follows completion of asynchronous UMMA use of K.
                    kfull.release()
                acc_producer.tail()
                q_full.release()
            if warp < 4:
                identity = cute.make_identity_tensor((128, 128))
                coords = mma.get_slice(0).partition_C(identity)
                atom = cute.make_copy_atom(
                    tcgen05.LdRed32x32bOp(
                        tcgen05.Repetition.x128,
                        tcgen05.Pack.NONE,
                        tcgen05.TmemLoadRedOp.MAX,
                    ),
                    Float32,
                )
                cp = tcgen05.make_tmem_copy(atom, acc[(None, None, None, 0)])
                thr = cp.get_slice(tid)
                thread_acc = thr.partition_S(acc)
                regs = cute.make_rmem_tensor(thr.partition_D(coords).shape, Float32)
                reduced = cute.make_rmem_tensor((1,), Float32)
                top_scores = cute.make_rmem_tensor((16,), Float32)
                top_ids = cute.make_rmem_tensor((16,), Int32)
                for slot in cutlass.range_constexpr(16):
                    top_scores[slot] = -Float32.inf
                    top_ids[slot] = Int32(-1)
                count = Int32(0)
                for block in cutlass.range(begin, end, unroll=1):
                    stage = count % 2
                    phase = (count // 2) % 2
                    acc_pipe.consumer_wait_w_index_phase(stage, phase)
                    cute.copy(
                        cp, thread_acc[(None, None, None, None, stage)], [regs, reduced]
                    )
                    page_maximum = reduced[0]
                    cute.arch.fence_view_async_tmem_load()
                    acc_pipe.consumer_release_w_index(stage)
                    count += 1
                    q_global = q_start + tid
                    position = kv_tokens - (q_end - q_begin) + q_global - q_begin
                    visible = (position + 128) // 128
                    valid = q_global < q_end and block < visible - 8
                    candidate_score = (
                        page_maximum * Float32(128**-0.5) if valid else -Float32.inf
                    )
                    candidate_id = block if valid else Int32(-1)
                    enters = candidate_id >= 0 and (
                        candidate_score > top_scores[15]
                        or (
                            candidate_score == top_scores[15]
                            and (top_ids[15] < 0 or candidate_id < top_ids[15])
                        )
                    )
                    if enters:
                        for slot in cutlass.range_constexpr(16):
                            old_score = top_scores[slot]
                            old_id = top_ids[slot]
                            take = candidate_score > old_score or (
                                candidate_score == old_score
                                and candidate_id >= 0
                                and (old_id < 0 or candidate_id < old_id)
                            )
                            top_scores[slot] = candidate_score if take else old_score
                            top_ids[slot] = candidate_id if take else old_id
                            candidate_score = old_score if take else candidate_score
                            candidate_id = old_id if take else candidate_id
                if q_start + tid < q_end:
                    for slot in cutlass.range_constexpr(16):
                        scores[head, partition, q_start + tid, slot] = top_scores[slot]
                        ids[head, partition, q_start + tid, slot] = top_ids[slot]
            # MMA tail and every epilogue TMEM read retire at the same PC.
            tmem_barrier.arrive_and_wait()
            # Keep reserve/free in this common scope so the public allocator's
            # DSL ownership state retains the original 512-column allocation.
            # The TMA loader only touches SMEM and has no TMEM allocation.
            allocator.relinquish_alloc_permit()
            allocator.free(pool.base_ptr, num_columns=512)
        # The independent TMA loader also drains K and Q before CTA retirement.
        cute.arch.barrier()



#: Mode of each operand whose extent counts query tokens, KV pages or tiles:
#: Q, Q scales, K pages, page map, tile metadata, scores, IDs and route.
_DYNAMIC_MODES = (0, 0, 0, 0, 0, 2, 2, 2)


def _operands(tensors):
    """Wrap operands with their token, page and tile extents as runtime values."""
    return tuple(_dynamic(t, mode) for t, mode in zip(tensors, _DYNAMIC_MODES))


def _compact_q(q):
    """E4M3 view of Q with compact rows."""
    if q.dtype == torch.uint8:
        q = q.view(torch.float8_e4m3fn)
    return q.contiguous()


class PreparedMxfp6Router:
    """Own MX routing buffers and launch compiled kernels on the current stream."""

    def __init__(self, tensors, score_kernel, *, source_q):
        self.tensors = tensors
        self.operands = _operands(tensors)
        self.score_kernel = score_kernel
        # Native pad8 Q views have a larger token stride than the compact
        # kernel ABI. Retain their live source instead of snapshotting Q once.
        self.source_q = source_q

    def rebind(self, q, scales, k, pages):
        source_q = q.view(torch.float8_e4m3fn) if q.dtype == torch.uint8 else q
        if (source_q.shape, source_q.stride(), source_q.dtype, source_q.device) != (
            self.source_q.shape, self.source_q.stride(),
            self.source_q.dtype, self.source_q.device,
        ):
            raise ValueError("MX source Q rebind must preserve shape, stride, dtype and device")
        # Reuse owned compact storage for padded sources; contiguous sources
        # continue to alias the caller's Q without a copy or allocation.
        compact_q = source_q if source_q.is_contiguous() else self.tensors[0]
        inputs = (compact_q, scales, k.view(-1, 1, 12800), pages)
        for new, old in zip(inputs, self.tensors[:4]):
            if (new.shape, new.stride(), new.dtype, new.device) != (
                old.shape,
                old.stride(),
                old.dtype,
                old.device,
            ):
                raise ValueError(
                    "MX rebind must preserve shape, stride, dtype and device"
                )
        self.source_q = source_q
        self.tensors = inputs + self.tensors[4:]
        self.operands = _operands(self.tensors)

    def run(self):
        current = torch.cuda.current_stream(self.tensors[0].device)
        self.source_q.record_stream(current)
        # CuTe launches bypass PyTorch's dispatcher. Protect every operand
        # against allocator reuse if the caller rebinds or drops this router
        # before work on a non-default stream completes.
        for tensor in self.tensors:
            tensor.record_stream(current)
        if not self.source_q.is_contiguous():
            # Keep this on the launch stream and inside graph capture so every
            # eager call/replay observes live values in the original pad8 Q.
            self.tensors[0].copy_(self.source_q)
        stream = cuda.CUstream(current.cuda_stream)
        self.score_kernel(*self.operands, stream)


_MX_COMPILE_CACHE = {}


def prepare_mxfp6_router(
    q, scales, k, pages, metadata, *, max_blocks, route, partition_blocks=256
):
    """Prepare native MX operands with request-local page metadata."""
    if q.dtype == torch.uint8:
        q = q.view(torch.float8_e4m3fn)
    if (
        q.dtype != torch.float8_e4m3fn
        or q.ndim != 3
        or tuple(q.shape[1:]) != (4, 128)
        or q.stride(2) != 1
        or q.stride(1) != 128
    ):
        raise ValueError("MXFP8 Q must be E4M3 [T,4,128] with contiguous head rows")
    source_q = q
    q = _compact_q(q)
    if (
        scales.dtype != torch.int32
        or scales.ndim != 2
        or scales.shape[0] != q.shape[0]
        or scales.shape[1] < 4
        or not scales.is_contiguous()
    ):
        raise ValueError("MXFP8 Q scales must be contiguous int32 [T,>=4]")
    if (
        k.dtype != torch.uint8
        or k.ndim != 4
        or tuple(k.shape[1:]) != (1, 128, 100)
        or not k.is_contiguous()
    ):
        raise ValueError("MXFP6 K must be contiguous uint8 [pages,1,128,100]")
    if pages.dtype != torch.int32 or pages.ndim != 1 or not pages.is_contiguous():
        raise ValueError("MXFP6 pages must be contiguous int32 [logical pages]")
    if (
        metadata.dtype != torch.int32
        or metadata.ndim != 2
        or metadata.shape[1] < 5
        or not metadata.is_contiguous()
    ):
        raise ValueError("MXFP6 tile metadata must be contiguous int32 [tiles,>=5]")
    if (
        route.dtype != torch.int32
        or route.ndim != 3
        or route.shape[0] != 4
        or route.shape[1] < 32
        or route.shape[2] != q.shape[0]
        or not route.is_contiguous()
    ):
        raise ValueError("MXFP6 compact route must be contiguous int32 [4,>=32,T]")
    if not q.is_cuda or any(
        t.device != q.device for t in (scales, k, pages, metadata, route)
    ):
        raise ValueError("MXFP6 operands must share a CUDA device")
    if torch.cuda.get_device_capability(q.device) not in ((10, 0), (10, 3)):
        raise ValueError("MXFP6 Router requires SM100 or SM103; SM90 has no native MX MMA")
    if max_blocks <= 0 or partition_blocks not in (
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
    ):
        raise ValueError("invalid MXFP6 partition capacity")
    partitions = (max_blocks + partition_blocks - 1) // partition_blocks
    shape = (4, partitions, q.shape[0], 16)
    scores = torch.empty(shape, device=q.device, dtype=torch.float32)
    ids = torch.empty(shape, device=q.device, dtype=torch.int32)
    tensors = (
        q,
        scales,
        k.view(-1, 1, 12800),
        pages,
        metadata,
        scores,
        ids,
        route,
    )
    operands = _operands(tensors)
    # Token, page and tile extents are runtime values; the partition count is
    # a compile-time loop bound of the merge.
    key = (
        partition_blocks,
        q.device.index,
        partitions,
        int(scales.shape[1]),
        int(route.shape[1]),
    )
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    if key not in _MX_COMPILE_CACHE:
        _MX_COMPILE_CACHE[key] = (
            cute.compile(
                Mxfp6IndexerLocalTop16Sm103(partition_blocks), *operands, stream
            ),
        )
    return PreparedMxfp6Router(tensors, *_MX_COMPILE_CACHE[key], source_q=source_q)
