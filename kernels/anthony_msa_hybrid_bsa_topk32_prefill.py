# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Optimized MSA TopK32: causal BSA fixed16 + CSR dynamic16 + LSE merge."""

from __future__ import annotations

from dataclasses import dataclass

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int64
from cutlass.cute.runtime import from_dlpack

from third_party.anthony_bsa_backend.flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk128 import (
    bsa_attn_sm100_blk128_fwd,
)
from msa_sparse_attn_prefill import build_csr, sparse_gqa_stage2_attn_fwd_msa_prefill


PAGE = 128
H_Q = 32
H_KV = 4
HEADS_PER_KV = 8
D = 128


_MERGE_KERNELS: dict[tuple[torch.dtype, int | None, int, bool], object] = {}


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
) -> None:
    rows = fixed_o.numel() // D
    if fixed_grouped and tuple(fixed_o.shape) != (H_KV, rows // H_Q, HEADS_PER_KV, D):
        raise ValueError("grouped fixed O must be [Hkv,Q,GQA,D]")
    tensors = (fixed_o, dynamic_o, out)
    if any(t.numel() != rows * D or not t.is_contiguous() for t in tensors):
        raise ValueError("attention outputs must be contiguous [rows,128] tensors")
    if any(
        t.numel() != rows or not t.is_contiguous()
        for t in (fixed_lse, dynamic_lse, out_lse)
    ):
        raise ValueError("LSE inputs must be contiguous and match the output rows")
    cute_args = (
        from_dlpack(
            fixed_o.detach() if fixed_grouped else fixed_o.detach().reshape(rows, D),
            assumed_align=16,
        ),
        from_dlpack(fixed_lse.detach().reshape(rows), assumed_align=16),
        from_dlpack(dynamic_o.detach().reshape(rows, D), assumed_align=16),
        from_dlpack(dynamic_lse.detach().reshape(rows), assumed_align=16),
        from_dlpack(out.detach().reshape(rows, D), assumed_align=16),
        from_dlpack(out_lse.detach().reshape(rows), assumed_align=16),
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = (out.dtype, out.device.index, rows, fixed_grouped)
    compiled = _MERGE_KERNELS.get(key)
    if compiled is None:
        compiled = cute.compile(_MergeTwoAttentionStates(), *cute_args, stream)
        _MERGE_KERNELS[key] = compiled
    compiled(*cute_args, stream)


@dataclass
class MSAHybridBSATopK32PrefillPlan:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    fixed_routes: torch.Tensor
    fixed_route_counts: torch.Tensor
    fixed_out: torch.Tensor
    fixed_lse: torch.Tensor
    dynamic_routes: torch.Tensor
    dynamic_state: tuple[torch.Tensor, torch.Tensor, object]
    dynamic_out: torch.Tensor
    dynamic_workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    out: torch.Tensor
    out_lse: torch.Tensor
    partial_dtype: torch.dtype
    fuse_fixed_state_in_dynamic_combine: bool = False

    def run_csr(self) -> None:
        q_len = int(self.q.shape[0])
        kv_len = int(self.k.shape[0])
        self.dynamic_state = build_csr(
            self.dynamic_routes.permute(1, 0, 2).contiguous(),
            self.cu_seqlens_q,
            self.cu_seqlens_k,
            PAGE,
            total_k=kv_len,
            max_seqlen_k=kv_len,
            max_seqlen_q=q_len,
            total_rows=kv_len // PAGE,
            num_kv_heads=H_KV,
            qhead_per_kv=HEADS_PER_KV,
        )

    def run_fixed(self) -> None:
        bsa_attn_sm100_blk128_fwd(
            self.q.unsqueeze(0),
            self.k.unsqueeze(0),
            self.v.unsqueeze(0),
            q2k_block_index=self.fixed_routes,
            block_sparse_num=1,
            q2k_block_nums=self.fixed_route_counts,
            softmax_scale=D**-0.5,
            out=self.fixed_out.unsqueeze(0),
            lse=self.fixed_lse,
            return_lse=True,
            causal=True,
        )

    def run_dynamic(self) -> None:
        row_ptr, q_indices, schedule = self.dynamic_state
        sparse_gqa_stage2_attn_fwd_msa_prefill(
            self.q,
            self.k,
            self.v,
            row_ptr,
            q_indices,
            16,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_k=self.cu_seqlens_k,
            max_seqlen_q=self.q.shape[0],
            max_seqlen_k=self.k.shape[0],
            block_size=PAGE,
            causal=True,
            softmax_scale=D**-0.5,
            schedule=schedule,
            out=self.dynamic_out,
            partial_dtype=self.partial_dtype,
            return_softmax_lse=True,
            workspace=self.dynamic_workspace,
            initial_out=(
                self.fixed_out if self.fuse_fixed_state_in_dynamic_combine else None
            ),
            initial_lse=(
                self.fixed_lse if self.fuse_fixed_state_in_dynamic_combine else None
            ),
        )

    def run_merge(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fuse_fixed_state_in_dynamic_combine:
            return self.out, self.out_lse
        _merge_two_attention_states_cutedsl(
            self.fixed_out,
            self.fixed_lse,
            self.dynamic_out,
            self.dynamic_workspace[2],
            self.out,
            self.out_lse,
        )
        return self.out, self.out_lse

    def run_attention(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.run_fixed()
        self.run_dynamic()
        return self.run_merge()

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.run_csr()
        return self.run_attention()


def prepare_msa_hybrid_bsa_topk32_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dynamic_routes: torch.Tensor,
    *,
    out: torch.Tensor,
    partial_dtype: torch.dtype = torch.bfloat16,
    fuse_fixed_state_in_dynamic_combine: bool = False,
) -> MSAHybridBSATopK32PrefillPlan:
    """Prepare the production 8-prefix + TopK16 + 8-SWA prefill path."""

    q_len = int(q.shape[0])
    kv_len = int(k.shape[0])
    if q_len % PAGE or tuple(q.shape) != (q_len, H_Q, D):
        raise ValueError("q must be [S,32,128] with S divisible by 128")
    if (
        kv_len < q_len
        or kv_len % PAGE
        or tuple(k.shape) != (kv_len, H_KV, D)
        or v.shape != k.shape
    ):
        raise ValueError(
            "k/v must be [K,4,128] with K >= Q and K divisible by 128"
        )
    if dynamic_routes.shape != (q_len, H_KV, 16):
        raise ValueError("dynamic_routes must be [Q,4,16]")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q/k/v must use bfloat16")
    if dynamic_routes.dtype != torch.int32:
        raise ValueError("dynamic_routes must use int32")
    if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
        raise ValueError("out must match q shape, dtype, and device")
    if any(t.device != q.device for t in (k, v, dynamic_routes)):
        raise ValueError("q/k/v/dynamic_routes must be on the same device")

    num_q_blocks = q_len // PAGE
    num_kv_blocks = kv_len // PAGE
    q_block_offset = num_kv_blocks - num_q_blocks
    fixed_rows: list[list[int]] = []
    for local_q_block in range(num_q_blocks):
        q_block = q_block_offset + local_q_block
        fixed = sorted(
            set(range(min(8, q_block + 1)))
            | set(range(max(0, q_block - 7), q_block + 1))
        )
        fixed_rows.append(fixed)
    fixed_route_counts_cpu = torch.tensor(
        [len(row) for row in fixed_rows], dtype=torch.int32
    )
    max_fixed = max(2, (int(fixed_route_counts_cpu.max()) + 1) & ~1)
    fixed_routes_cpu = torch.full(
        (num_q_blocks, max_fixed), -1, dtype=torch.int32
    )
    for q_block, row in enumerate(fixed_rows):
        fixed_routes_cpu[q_block, : len(row)] = torch.tensor(row, dtype=torch.int32)
    fixed_routes = (
        fixed_routes_cpu.repeat_interleave(HEADS_PER_KV, dim=0)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(1, H_KV, -1, -1)
        .contiguous()
        .to(q.device)
    )
    fixed_route_counts = (
        fixed_route_counts_cpu.repeat_interleave(HEADS_PER_KV)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(1, H_KV, -1)
        .contiguous()
        .to(q.device)
    )
    # K2 reads both initial-state operands even when LSE assigns zero weight.
    if fuse_fixed_state_in_dynamic_combine:
        fixed_out = out.zero_()
        fixed_lse = torch.full(
            (1, H_Q, q_len), -torch.inf, dtype=torch.float32, device=q.device
        )
    else:
        fixed_out = torch.empty_like(q)
        fixed_lse = torch.empty(
            (1, H_Q, q_len), dtype=torch.float32, device=q.device
        )
    out_lse = torch.empty(q_len, H_Q, dtype=torch.float32, device=q.device)

    cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=q.device)
    cu_k = torch.tensor([0, kv_len], dtype=torch.int32, device=q.device)
    dynamic_state = build_csr(
        dynamic_routes.permute(1, 0, 2).contiguous(),
        cu_q,
        cu_k,
        PAGE,
        total_k=kv_len,
        max_seqlen_k=kv_len,
        max_seqlen_q=q_len,
        total_rows=num_kv_blocks,
        num_kv_heads=H_KV,
        qhead_per_kv=HEADS_PER_KV,
    )
    dynamic_out = out if fuse_fixed_state_in_dynamic_combine else torch.empty_like(q)
    dynamic_workspace = (
        torch.empty(16, q_len, H_Q, D, dtype=partial_dtype, device=q.device),
        torch.empty(16, q_len, H_Q, dtype=torch.float32, device=q.device),
        torch.empty(q_len, H_Q, dtype=torch.float32, device=q.device),
    )
    return MSAHybridBSATopK32PrefillPlan(
        q=q,
        k=k,
        v=v,
        fixed_routes=fixed_routes,
        fixed_route_counts=fixed_route_counts,
        fixed_out=fixed_out,
        fixed_lse=fixed_lse,
        dynamic_routes=dynamic_routes,
        dynamic_state=dynamic_state,
        dynamic_out=dynamic_out,
        dynamic_workspace=dynamic_workspace,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        out=out,
        out_lse=out_lse,
        partial_dtype=partial_dtype,
        fuse_fixed_state_in_dynamic_combine=fuse_fixed_state_in_dynamic_combine,
    )
