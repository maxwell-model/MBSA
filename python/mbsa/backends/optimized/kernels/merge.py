# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optimized MSA TopK32: causal BSA fixed16 + CSR dynamic16 + LSE merge."""

from __future__ import annotations


import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int64
from cutlass.cute.runtime import from_dlpack



PAGE = 128
H_Q = 32
H_KV = 4
HEADS_PER_KV = 8
D = 128


# Keyed by dtypes, device and layout only: the row count is a dynamic shape,
# so mixed batches with new total query counts reuse one compiled kernel.
_MERGE_KERNELS: dict[tuple[torch.dtype, torch.dtype, torch.dtype, int | None, bool], object] = {}


class _MergeTwoAttentionStates:
    """CuTe DSL exact merge for two natural-log LSE attention states."""

    threads = 256
    warps = threads // 32

    @cute.jit
    def __call__(
        self,
        mFixedO: cute.Tensor,
        mFixedLSE: cute.Tensor,
        mDynamicO: cute.Tensor,
        mDynamicLSE: cute.Tensor,
        mOut: cute.Tensor,
        mOutLSE: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(len(mFixedO.shape) == 4):
            mFixedO = cute.make_tensor(
                mFixedO.iterator, cute.select(mFixedO.layout, mode=[2, 0, 1, 3])
            )
            mFixedO = cute.group_modes(mFixedO, 0, 3)
        self.kernel(
            mFixedO,
            mFixedLSE,
            mDynamicO,
            mDynamicLSE,
            mOut,
            mOutLSE,
        ).launch(
            grid=(cute.ceil_div(mOut.shape[0], self.warps),),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mFixedO: cute.Tensor,
        mFixedLSE: cute.Tensor,
        mDynamicO: cute.Tensor,
        mDynamicLSE: cute.Tensor,
        mOut: cute.Tensor,
        mOutLSE: cute.Tensor,
    ):
        tidx = cute.arch.thread_idx()[0]
        lane = tidx % 32
        warp = tidx // 32
        row = Int64(cute.arch.block_idx()[0]) * self.warps + Int64(warp)
        if row < mOut.shape[0]:
            token = row // H_Q
            head = row - token * H_Q
            fixed_lse = Float32(
                mFixedLSE[head * (mOut.shape[0] // H_Q) + token]
            )
            dynamic_lse = Float32(mDynamicLSE[row])
            max_lse = cute.arch.fmax(fixed_lse, dynamic_lse)
            # The two inputs expose natural-log LSE. CuTe provides exp2, so
            # convert the exponent using log2(e).
            fixed_weight = cute.math.exp2(
                (fixed_lse - max_lse) * 1.4426950408889634, fastmath=True
            )
            dynamic_weight = cute.math.exp2(
                (dynamic_lse - max_lse) * 1.4426950408889634, fastmath=True
            )
            denominator = Float32(fixed_weight + dynamic_weight)
            inv_denominator = Float32(1.0) / denominator
            if lane == 0:
                mOutLSE[row] = cute.math.log(denominator, fastmath=True) + max_lse
            for column_group in cutlass.range_constexpr(D // 32):
                column = lane + column_group * 32
                fixed_value = Float32(mFixedO[row, column])
                dynamic_value = Float32(mDynamicO[row, column])
                merged = (
                    fixed_weight * fixed_value + dynamic_weight * dynamic_value
                ) * inv_denominator
                mOut[row, column] = merged.to(mOut.element_type)


def _merge_two_attention_states_cutedsl(
    fixed_o: torch.Tensor,
    fixed_lse: torch.Tensor,
    dynamic_o: torch.Tensor,
    dynamic_lse: torch.Tensor,
    out: torch.Tensor,
    out_lse: torch.Tensor,
    *,
    fixed_grouped: bool = False,
    inputs: tuple | None = None,
) -> tuple:
    """Merge Fixed and Dynamic attention states into out / out_lse on the current stream.

    inputs: wrappers of (fixed_o, fixed_lse, dynamic_o, dynamic_lse, out_lse) returned by an earlier call with
    the same tensors; a plan's layers share them, so only out is wrapped again. Returns those wrappers.
    """
    rows = fixed_o.numel() // D
    if out.numel() != rows * D or not out.is_contiguous():
        raise ValueError("attention outputs must be contiguous [rows,128] tensors")

    def dynamic(tensor, mode=0):
        # Contiguous (checked): the row-major stride order.
        return from_dlpack(tensor, assumed_align=16).mark_compact_shape_dynamic(
            mode=mode, stride_order=tuple(range(tensor.dim()))
        )

    if inputs is None:
        if fixed_grouped and tuple(fixed_o.shape) != (H_KV, rows // H_Q, HEADS_PER_KV, D):
            raise ValueError("grouped fixed O must be [Hkv,Q,GQA,D]")
        if any(t.numel() != rows * D or not t.is_contiguous() for t in (fixed_o, dynamic_o)):
            raise ValueError("attention outputs must be contiguous [rows,128] tensors")
        if any(
            t.numel() != rows or not t.is_contiguous()
            for t in (fixed_lse, dynamic_lse, out_lse)
        ):
            raise ValueError("LSE inputs must be contiguous and match the output rows")
        inputs = (
            dynamic(fixed_o.detach(), 1)
            if fixed_grouped
            else dynamic(fixed_o.detach().reshape(rows, D)),
            dynamic(fixed_lse.detach().reshape(rows)),
            dynamic(dynamic_o.detach().reshape(rows, D)),
            dynamic(dynamic_lse.detach().reshape(rows)),
            dynamic(out_lse.detach().reshape(rows)),
        )
    cute_args = (*inputs[:4], dynamic(out.detach().reshape(rows, D)), inputs[4])
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = (fixed_o.dtype, dynamic_o.dtype, out.dtype, out.device.index, fixed_grouped)
    compiled = _MERGE_KERNELS.get(key)
    if compiled is None:
        compiled = cute.compile(_MergeTwoAttentionStates(), *cute_args, stream)
        _MERGE_KERNELS[key] = compiled
    compiled(*cute_args, stream)
    return inputs
