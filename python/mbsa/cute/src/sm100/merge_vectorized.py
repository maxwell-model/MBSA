# Copyright 2026 MBSA contributors.
# SPDX-License-Identifier: Apache-2.0
"""Original warp-owned reduction of two attention states.

One lane computes each row's log-sum-exp weights, then broadcasts them. Each
lane consumes four contiguous channels before any matching output store. Empty
or zero-weight states are never loaded, including their poisoned output data.
"""

from __future__ import annotations

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import nvvm
from cutlass import Float32, Int64, Constexpr, const_expr
from cutlass.cute.runtime import from_dlpack

_CACHE = {}


class MergeStatesVectorized:
    def __init__(self, vectorized=True):
        self.rows_per_warp = 4
        self.vectorized = vectorized

    @cute.jit
    def __call__(self, a, la, b, lb, out, lout, head_major: Constexpr, stream):
        self.kernel(a, la, b, lb, out, lout, head_major).launch(
            grid=(cute.ceil_div(out.shape[0], 8 * self.rows_per_warp),),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, a, la, b, lb, out, lout, head_major: Constexpr):
        tid = cute.arch.thread_idx()[0]
        lane, warp = tid % 32, tid // 32
        block_base = Int64(cute.arch.block_idx()[0]) * (8 * self.rows_per_warp)
        ra = cute.make_rmem_tensor((4,), a.element_type)
        rb = cute.make_rmem_tensor((4,), b.element_type)
        ro = cute.make_rmem_tensor((4,), out.element_type)
        if const_expr(self.vectorized):
            copy_a = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                a.element_type,
                num_bits_per_copy=4 * a.element_type.width,
            )
            copy_b = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                b.element_type,
                num_bits_per_copy=4 * b.element_type.width,
            )
            copy_o = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                out.element_type,
                num_bits_per_copy=4 * out.element_type.width,
            )
        for iteration in cutlass.range_constexpr(self.rows_per_warp):
            row = block_base + Int64(iteration * 8 + warp)
            if row < out.shape[0]:
                wa, wb, result = Float32(0), Float32(0), -Float32.inf
                if lane == 0:
                    token, head = row // 32, row % 32
                    x = Float32(la[head * (out.shape[0] // 32) + token])
                    if const_expr(head_major):
                        y = Float32(lb[head * (out.shape[0] // 32) + token])
                    else:
                        y = Float32(lb[row])
                    z = cute.arch.fmax(x, y)
                    if z != -Float32.inf:
                        wa = cute.math.exp2((x - z) * 1.4426950408889634, fastmath=True)
                        wb = cute.math.exp2((y - z) * 1.4426950408889634, fastmath=True)
                    total = wa + wb
                    if total > 0:
                        result = z + cute.math.log(total, fastmath=True)
                        wa, wb = wa / total, wb / total
                wa = cute.arch.shuffle_sync(wa, offset=0, kind=nvvm.ShflKind.idx)
                wb = cute.arch.shuffle_sync(wb, offset=0, kind=nvvm.ShflKind.idx)
                # Four private channels are fully consumed before stores. No
                # other lane or warp consumes these channels, even for O alias.
                ra.fill(0)
                rb.fill(0)
                channel_base = Int64(lane * 4)
                if const_expr(self.vectorized):
                    element_offset = row * Int64(128) + channel_base
                    ga = cute.make_tensor(
                        cute.make_ptr(
                            a.element_type,
                            a.iterator.toint()
                            + element_offset * (a.element_type.width // 8),
                            cute.AddressSpace.gmem,
                            assumed_align=4 * a.element_type.width // 8,
                        ),
                        cute.make_layout((4,), stride=(1,)),
                    )
                    gb = cute.make_tensor(
                        cute.make_ptr(
                            b.element_type,
                            b.iterator.toint()
                            + element_offset * (b.element_type.width // 8),
                            cute.AddressSpace.gmem,
                            assumed_align=4 * b.element_type.width // 8,
                        ),
                        cute.make_layout((4,), stride=(1,)),
                    )
                    go = cute.make_tensor(
                        cute.make_ptr(
                            out.element_type,
                            out.iterator.toint()
                            + element_offset * (out.element_type.width // 8),
                            cute.AddressSpace.gmem,
                            assumed_align=4 * out.element_type.width // 8,
                        ),
                        cute.make_layout((4,), stride=(1,)),
                    )
                    if wa > 0:
                        cute.copy(copy_a, ga, ra)
                    if wb > 0:
                        cute.copy(copy_b, gb, rb)
                else:
                    if wa > 0:
                        for component in cutlass.range_constexpr(4):
                            ra[component] = a[row, channel_base + component]
                    if wb > 0:
                        for component in cutlass.range_constexpr(4):
                            rb[component] = b[row, channel_base + component]
                for component in cutlass.range_constexpr(4):
                    value = wa * Float32(ra[component]) + wb * Float32(rb[component])
                    ro[component] = value.to(out.element_type)
                if const_expr(self.vectorized):
                    cute.copy(copy_o, ro, go)
                else:
                    for component in cutlass.range_constexpr(4):
                        out[row, channel_base + component] = ro[component]
                if lane == 0:
                    lout[row] = result


def _dynamic64(tensor):
    # Contiguous flattening can retain a nonzero storage offset. Do not promise
    # stronger alignment than the actual pointer, especially in scalar fallback.
    alignment = min(16, tensor.element_size())
    while alignment < 16 and tensor.data_ptr() % (alignment * 2) == 0:
        alignment *= 2
    return from_dlpack(
        tensor.detach(), assumed_align=alignment, use_32bit_stride=False
    ).mark_compact_shape_dynamic(mode=0, stride_order=tuple(range(tensor.dim())))


def _merge_two_attention_states_vectorized(
    fixed_o,
    fixed_lse,
    dynamic_o,
    dynamic_lse,
    out,
    out_lse,
):
    """Merge into contiguous O and row-major FP32 LSE; exact O alias is safe.

    LSE state values must be finite or -inf, matching the existing attention
    state contract. Dynamic LSE accepts row-major [Q,32] or the exact transpose of a contiguous
    head-major [32,Q]. All dynamic descriptor strides use Int64. Each CTA
    processes 32 rows; pointer alignment selects vector copies or scalar fallback.
    """
    if out.ndim != 3 or out.shape[1:] != (32, 128) or out.shape[0] < 1:
        raise ValueError("state outputs must have positive [Q,32,128] shape")
    if any(
        t.shape != out.shape or not t.is_contiguous() for t in (fixed_o, dynamic_o, out)
    ):
        raise ValueError("state outputs must share contiguous [Q,32,128] shape")
    if (
        fixed_lse.shape != (1, 32, out.shape[0])
        or dynamic_lse.shape != (out.shape[0], 32)
        or out_lse.shape != dynamic_lse.shape
    ):
        raise ValueError("state LSE layouts must be [1,32,Q] and [Q,32]")
    if not fixed_lse.is_contiguous() or not out_lse.is_contiguous():
        raise ValueError("fixed and output LSE must be contiguous")
    if not dynamic_lse.is_contiguous() and dynamic_lse.stride() != (1, out.shape[0]):
        raise ValueError("dynamic LSE must be contiguous or a head-major transpose")
    if any(
        t.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        for t in (fixed_o, dynamic_o, out)
    ):
        raise ValueError("state outputs require BF16/FP16/FP32")
    if any(t.dtype != torch.float32 for t in (fixed_lse, dynamic_lse, out_lse)):
        raise ValueError("state LSE values must be float32")
    if out.device.type != "cuda" or any(
        t.device != out.device
        for t in (fixed_o, fixed_lse, dynamic_o, dynamic_lse, out_lse)
    ):
        raise ValueError("all states must be on the same CUDA device")
    aligned = all(
        t.data_ptr() % (4 * t.element_size()) == 0 for t in (fixed_o, dynamic_o, out)
    )
    vectorized = aligned
    rows = out.numel() // 128
    head_major = dynamic_lse.stride() == (1, out.shape[0])
    lse_flat = (dynamic_lse.transpose(0, 1) if head_major else dynamic_lse).reshape(
        rows
    )
    args = (
        _dynamic64(fixed_o.view(rows, 128)),
        _dynamic64(fixed_lse.reshape(rows)),
        _dynamic64(dynamic_o.view(rows, 128)),
        _dynamic64(lse_flat),
        _dynamic64(out.view(rows, 128)),
        _dynamic64(out_lse.reshape(rows)),
    )
    stream = cuda.CUstream(torch.cuda.current_stream(out.device).cuda_stream)
    alignments = tuple(
        min(16, t.data_ptr() & -t.data_ptr())
        for t in (fixed_o, fixed_lse, dynamic_o, dynamic_lse, out, out_lse)
    )
    key = (
        fixed_o.dtype,
        dynamic_o.dtype,
        out.dtype,
        out.device.index,
        head_major,
        bool(vectorized),
        alignments,
    )
    if key not in _CACHE:
        _CACHE[key] = cute.compile(
            MergeStatesVectorized(bool(vectorized)),
            *args,
            head_major,
            stream,
        )
    # Constexpr is intentionally absent from the compiled runtime ABI.
    _CACHE[key](*args, stream)
