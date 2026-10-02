# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for MiniMax M3 block-sparse GQA attention.

Vendored from vLLM's own MiniMax M3 support (PR #45381,
``vllm/models/minimax_m3/common/ops/sparse_attn.py``) so Compass's MSA backend
can use the same decode kernel upstream uses. Upstream runs MSA for prefill and
this Triton split-K kernel for decode; MSA's paged decode plans on the host and
reads its cost model back off the device, which cannot happen inside a cudagraph
capture. This kernel takes only device tensors and a shape-constant chunk count,
so the decode forward becomes capturable.

Kept byte-for-byte apart from this note, so it can be re-synced with upstream.

The main heads attend only to the blocks selected by the lightning indexer (see
``index_topk``). Adapted to vLLM's paged KV cache: the KV page size is forced to
equal the sparse block size (128), so one selected block maps to exactly one
page.

Main K/V cache layout (vLLM):
  ``(num_blocks, 2, 128, num_kv_heads, head_dim)``  K=[:,0] V=[:,1]

Only the paths MiniMax M3 uses are implemented: no attention sink, base-2
(exp2/log2) softmax. The decode kernels use split-K (flash-decoding) over the
selected blocks with a separate merge step, since one query token per request
leaves the prefill kernels (which parallelize over the query dim) idle.

Vendored into this repository from
``.tmp/vllm/vllm/model_executor/layers/minimax_sparse_attn_triton.py`` (the
``aip/compassv4-msa-score-reduce-topk32`` branch). The only change from that
copy is the import line below: ``vllm.platforms``/``vllm.triton_utils`` are
replaced with the local ``_vllm_compat`` shim so this file needs only
torch+triton, not a vLLM install.
"""

import torch

from msa_block_sparse._vllm_compat import current_platform, tl, triton

# One sparse block == one KV page.
SPARSE_BLOCK_SIZE = 128


_SPARSE_ATTN_NUM_STAGES_KWARG: dict | None = None


def _sparse_attn_num_stages_kwarg() -> dict:
    """Triton ``num_stages`` override for the sparse-attn GEMM kernels.

    Forced only where required: CDNA3 (gfx942) caps LDS at
    64 KB, and the default 2-stage pipeline double-buffers the 128x128 K/V tiles
    to ~66 KB ("out of resource: shared memory"), so pin gfx942 to a single
    stage (~32 KB, which fits). Everywhere else (NVIDIA, CDNA4 gfx950) return an
    empty kwarg and let Triton keep its own default -- don't second-guess it.
    Cached: the arch is fixed per process.
    """
    global _SPARSE_ATTN_NUM_STAGES_KWARG
    if _SPARSE_ATTN_NUM_STAGES_KWARG is None:
        kwarg: dict = {}
        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx942

            if on_gfx942():
                kwarg = {"num_stages": 1}
        _SPARSE_ATTN_NUM_STAGES_KWARG = kwarg
    return _SPARSE_ATTN_NUM_STAGES_KWARG


# ---------------------------------------------------------------------------
# GQA block-sparse attention (paged). Main heads attend only to the selected
# blocks. BLOCK_SIZE_K == 128 so each selected block is one page.
# ---------------------------------------------------------------------------
# since prefill metadata is sliced from mixed batch metadata, seq_lens and prefix_lens
# might lose pointer alignment, which trigger Triton recompiles. we don't actually
# need pointer alignment for those tensors anyway because we do scalar load.
@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
        "BLOCK_SIZE_QH": lambda args: args["BLOCK_SIZE_Q"]
        * triton.next_power_of_2(args["gqa_group_size"]),
    }
)
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    k_ptr,  # paged K: [num_blocks, num_kv_heads, 128, head_dim]
    v_ptr,  # paged V: [num_blocks, num_kv_heads, 128, head_dim]
    t_ptr,  # topk_idx: [Hkv|1, total_q, topk]
    o_ptr,  # [total_q, num_heads, head_dim]
    lse_ptr,  # natural-log LSE: [total_q, num_heads]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens_q,
    cu_seqblocks_q,
    seq_lens,
    prefix_lens,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    max_topk,
    num_q_loop,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_k_blk,
    stride_k_h,
    stride_k_pos,
    stride_k_d,
    stride_v_blk,
    stride_v_h,
    stride_v_pos,
    stride_v_d,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_lse_n,
    stride_lse_h,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = pid_kh * gqa_group_size
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block_start = tl.load(cu_seqblocks_q + pid_b)
    q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q * num_q_loop >= q_block_len:
        return
    real_q_loop = min(num_q_loop, q_block_len - pid_q * num_q_loop)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    for j in range(real_q_loop):
        pid_q_j = pid_q * num_q_loop + j
        t_ptr_j = t_ptr + (q_block_start + pid_q_j) * stride_tn + pid_kh * stride_th
        off_t = tl.arange(0, BLOCK_SIZE_T)
        topk_idx = tl.load(t_ptr_j + off_t * stride_tk, mask=off_t < max_topk, other=-1)
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_qn, stride_qh, stride_qd),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
        off_q = (
            tl.arange(0, BLOCK_SIZE_Q)[:, None]
            + pid_q_j * BLOCK_SIZE_Q
            + prefix_len
            - tl.arange(0, BLOCK_SIZE_K)[None, :]
        )
        m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
        lse_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
        acc_o = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), dtype=tl.float32)
        q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)
        num_blocks = (seq_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
        for _ in range(max_topk):
            blk = tl.load(t_ptr_j).to(tl.int32)
            t_ptr_j = t_ptr_j + stride_tk
            blk_valid = (blk >= 0) & (blk < num_blocks)
            c = blk * BLOCK_SIZE_K
            page = tl.load(bt_row + blk, mask=blk_valid, other=0).to(tl.int64)
            pos = c + off_n
            pos_mask = blk_valid & (pos < seq_len)
            k = tl.load(
                k_ptr
                + page * stride_k_blk
                + pid_kh * stride_k_h
                + off_n[None, :] * stride_k_pos
                + off_d[:, None] * stride_k_d,
                mask=d_mask[:, None] & pos_mask[None, :],
                other=0.0,
            )
            if USE_FP8:
                k = k.to(q.dtype)
            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            # causal: q_abs_pos - k_off >= block_start (c)
            qk += tl.where(off_q[:, None, :] >= c, 0, float("-inf"))
            qk = tl.reshape(qk, BLOCK_SIZE_QH, BLOCK_SIZE_K)
            qk += tl.dot(q, k) * sm_scale_log2e
            qk += tl.where(pos_mask[None, :], 0, float("-inf"))
            block_has_tokens = blk_valid & (
                c <= prefix_len + pid_q_j * BLOCK_SIZE_Q
            )
            m_candidate = tl.maximum(m_i, tl.max(qk, axis=1))
            m_ij = tl.where(block_has_tokens, m_candidate, m_i)
            p = tl.where(
                block_has_tokens & pos_mask[None, :],
                tl.exp2(qk - m_ij[:, None]),
                0.0,
            )
            l_ij = tl.sum(p, axis=1)
            alpha = tl.where(block_has_tokens, tl.exp2(m_i - m_ij), 1.0)
            acc_o = acc_o * alpha[:, None]
            v = tl.load(
                v_ptr
                + page * stride_v_blk
                + pid_kh * stride_v_h
                + off_n[:, None] * stride_v_pos
                + off_d[None, :] * stride_v_d,
                mask=pos_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            if USE_FP8:
                v = v.to(q.dtype)
            acc_o += tl.dot(p.to(v.dtype), v)
            m_i = m_ij
            lse_candidate = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
            lse_i = tl.where(block_has_tokens, lse_candidate, lse_i)
        scale = tl.where(
            lse_i > float("-inf"), tl.exp2(m_i - lse_i), tl.zeros_like(lse_i)
        )
        acc_o = acc_o * scale[:, None]
        acc_o = tl.reshape(acc_o, BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D)
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + q_start * stride_on + pid_h * stride_oh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_on, stride_oh, stride_od),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))
        lse_out = tl.reshape(
            lse_i * 0.6931471805599453, BLOCK_SIZE_Q, BLOCK_SIZE_H
        )
        lse_ptrs = tl.make_block_ptr(
            base=lse_ptr + q_start * stride_lse_n + pid_h * stride_lse_h,
            shape=(q_len, gqa_group_size),
            strides=(stride_lse_n, stride_lse_h),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H),
            order=(1, 0),
        )
        tl.store(lse_ptrs, lse_out, boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# Decode kernels (split-K). Decode batches are flattened request-major, with a
# runtime query length used to map each query token back to its request metadata.
# This parallelizes over the selected top-k blocks, producing partials that the
# merge kernel combines (flash-decoding). All chunk counts depend only on shape
# constants so the grid is fixed within a cuda graph. Base-2 (exp2/log2)
# softmax matches the prefill kernel.
# ---------------------------------------------------------------------------
@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
    }
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _gqa_sparse_decode_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # main cache: [num_blocks, 2, 128, num_kv_heads, head_dim]
    t_ptr,  # topk_idx: [Hkv|1, total_q, topk]
    o_ptr,  # partial out: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partial lse (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    total_q,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    decode_query_len,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_kv,
    stride_kv_pos,
    stride_kv_h,
    stride_kv_d,
    stride_th,
    stride_tn,
    stride_tk,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
    USE_PDL: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    # split-K over the topk dimension: pid(0) folds (query-token, chunk).
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % total_q
    pid_c = pid_bc // total_q
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len
    pid_h = pid_kh * gqa_group_size
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_compiletime = chunk_start_topk + chunk_size_topk

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)

    num_blocks = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    idx_base = t_ptr + pid_kh * stride_th + pid_b * stride_tn
    chunk_end_topk = tl.minimum(chunk_end_compiletime, max_topk)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    bt_row = block_table_ptr + req_id * stride_bt_b

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qn + pid_h * stride_qh,
        shape=(gqa_group_size, head_dim),
        strides=(stride_qh, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")

    cur_idx_ptr = idx_base + chunk_start_topk * stride_tk
    for _ in tl.range(chunk_start_topk, chunk_end_topk):
        blk = tl.load(cur_idx_ptr).to(tl.int32)
        cur_idx_ptr = cur_idx_ptr + stride_tk
        blk_valid = (blk >= 0) & (blk < num_blocks)
        c = blk * BLOCK_SIZE_K
        page = tl.load(bt_row + blk, mask=blk_valid, other=0).to(tl.int64)
        pos = c + off_n
        pos_mask = blk_valid & (pos < kv_len)
        k = tl.load(
            kv_cache_ptr
            + page * stride_kv_blk
            + 0 * stride_kv_kv
            + off_n[None, :] * stride_kv_pos
            + pid_kh * stride_kv_h
            + off_d[:, None] * stride_kv_d,
            mask=d_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        if USE_FP8:
            k = k.to(q.dtype)
        qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * sm_scale_log2e
        block_has_tokens = blk_valid & (c < kv_len)
        m_candidate = tl.maximum(m_i, tl.max(qk, axis=1))
        m_ij = tl.where(block_has_tokens, m_candidate, m_i)
        p = tl.where(
            block_has_tokens & pos_mask[None, :],
            tl.exp2(qk - m_ij[:, None]),
            0.0,
        )
        l_ij = tl.sum(p, axis=1)
        alpha = tl.where(block_has_tokens, tl.exp2(m_i - m_ij), 1.0)
        acc_o = acc_o * alpha[:, None]
        v = tl.load(
            kv_cache_ptr
            + page * stride_kv_blk
            + 1 * stride_kv_kv
            + off_n[:, None] * stride_kv_pos
            + pid_kh * stride_kv_h
            + off_d[None, :] * stride_kv_d,
            mask=pos_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        if USE_FP8:
            v = v.to(q.dtype)
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_candidate = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
        lse_i = tl.where(block_has_tokens, lse_candidate, lse_i)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    # Empty chunks for active rows must store zero output; otherwise the merge
    # can hit 0 * NaN. All-empty padded rows may still produce NaNs in merge.
    scale = tl.where(lse_i > float("-inf"), tl.exp2(m_i - lse_i), tl.zeros_like(lse_i))
    acc_o = acc_o * scale[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_o_h, stride_o_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
        shape=(gqa_group_size,),
        strides=(stride_l_h,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_H,),
        order=(0,),
    )
    tl.store(lse_ptrs, lse_i.to(lse_ptr.dtype.element_ty), boundary_check=(0,))


@triton.heuristics(
    {"BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"])}
)
@triton.jit
def _merge_topk_attn_out_kernel(
    o_ptr,  # partials: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partials (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    out_ptr,  # merged out: [total_q, num_heads, head_dim]
    lse_out_ptr,  # merged natural-log LSE: [total_q, num_heads]
    head_dim,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_out_n,
    stride_out_h,
    stride_out_d,
    stride_lse_out_n,
    stride_lse_out_h,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)

    # NOTE: assume seq_lens is safe to load before gdc_wait()
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    off_c = tl.arange(0, NUM_TOPK_CHUNKS)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(NUM_TOPK_CHUNKS, head_dim),
        strides=(stride_o_c, stride_o_d),
        offsets=(0, 0),
        block_shape=(NUM_TOPK_CHUNKS, BLOCK_SIZE_D),
        order=(1, 0),
    )
    lse_ptrs = lse_ptr + pid_b * stride_l_b + pid_h * stride_l_h + off_c * stride_l_c
    o = tl.load(o_ptrs, boundary_check=(0, 1), padding_option="zero")
    lse = tl.load(lse_ptrs)
    valid = lse > float("-inf")
    lse_max_raw = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max_raw > float("-inf"), lse_max_raw, 0.0)
    weights = tl.where(valid, tl.exp2(lse - lse_max), 0.0)
    denom = tl.sum(weights, axis=0)
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    weights = weights * inv_denom
    o_merged = tl.sum(o * weights[:, None], axis=0)
    out_ptrs = (
        out_ptr + pid_b * stride_out_n + pid_h * stride_out_h + off_d * stride_out_d
    )
    tl.store(out_ptrs, o_merged.to(out_ptr.dtype.element_ty), mask=off_d < head_dim)
    lse_out = tl.where(
        denom > 0.0,
        (lse_max + tl.log2(denom)) * 0.6931471805599453,
        float("-inf"),
    )
    tl.store(
        lse_out_ptr + pid_b * stride_lse_out_n + pid_h * stride_lse_out_h,
        lse_out,
    )


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
@torch.no_grad()
def minimax_m3_sparse_attn_prefill(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    k: torch.Tensor,  # [num_blocks, num_kv_heads, 128, head_dim]
    v: torch.Tensor,  # [num_blocks, num_kv_heads, 128, head_dim]
    topk_idx: torch.Tensor,  # [Hkv|1, total_q, topk]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
    lse_output: torch.Tensor,  # [total_q, num_heads] FP32
) -> None:
    """GQA block-sparse attention over the selected blocks. block_size_q == 1."""
    total_q, num_heads, head_dim = q.shape
    if topk_idx.ndim != 3 or topk_idx.shape[0] not in (1, num_kv_heads):
        raise ValueError(
            "topk_idx must have one shared or one per-KV-group route row; "
            f"got {tuple(topk_idx.shape)}"
        )
    batch = cu_seqlens_q.shape[0] - 1
    topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    if k.shape != v.shape or k.ndim != 4:
        raise ValueError("k and v must have matching [pages,Hkv,128,D] shapes")
    if tuple(k.shape[1:]) != (num_kv_heads, SPARSE_BLOCK_SIZE, head_dim):
        raise ValueError("paged k/v geometry does not match Hkv/block/head_dim")
    if tuple(lse_output.shape) != (total_q, num_heads) or lse_output.dtype != torch.float32:
        raise ValueError("lse_output must be FP32 [total_q,num_heads]")
    use_fp8 = k.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    if v.dtype != k.dtype:
        raise ValueError("k and v dtypes must match")
    grid = (max_query_len, num_kv_heads, batch)
    _gqa_sparse_fwd_kernel[grid](
        q,
        k,
        v,
        topk_idx,
        output,
        lse_output,
        block_table,
        cu_seqlens_q,
        cu_seqlens_q,  # cu_seqblocks_q == cu_seqlens_q when block_size_q == 1
        seq_lens,
        prefix_lens,
        num_kv_heads,
        gqa_group_size,
        head_dim,
        topk,
        1,  # num_q_loop
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        0 if topk_idx.shape[0] == 1 else topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse_output.stride(0),
        lse_output.stride(1),
        block_table.stride(0),
        BLOCK_SIZE_Q=1,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        USE_FP8=use_fp8,
        **_sparse_attn_num_stages_kwarg(),
    )


@torch.no_grad()
def minimax_m3_sparse_attn_decode(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, 2, 128, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [Hkv|1, total_q, topk]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
    decode_query_len: int,
    *,
    lse_out: torch.Tensor | None = None,
    workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """GQA block-sparse attention for decode (split-K over the top-k blocks)."""
    total_q, num_heads, head_dim = q.shape
    if topk_idx.ndim != 3 or topk_idx.shape[0] not in (1, num_kv_heads):
        raise ValueError(
            "topk_idx must have one shared or one per-KV-group route row; "
            f"got {tuple(topk_idx.shape)}"
        )
    if topk_idx.dtype != torch.int32 or not topk_idx.is_cuda:
        raise ValueError("topk_idx must be CUDA int32")
    if topk_idx.shape[1] != total_q:
        raise ValueError("topk_idx total_q dimension must match q")
    if num_heads != num_kv_heads * 8 or head_dim != 128:
        raise ValueError("provisional contract requires Hq=32, Hkv=4, D=128")
    assert total_q == seq_lens.shape[0] * decode_query_len
    max_topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    use_fp8 = kv_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    use_pdl = current_platform.is_arch_support_pdl()
    # `launch_pdl` is a Triton runtime kwarg only some backends accept (CUDA
    # SM9+); this ROCm Triton rejects it even when False ("Keyword argument
    # launch_pdl was specified but unrecognised"). Only pass it when PDL is
    # actually supported -- on ROCm use_pdl is always False, so it's omitted.
    pdl_launch = {"launch_pdl": True} if use_pdl else {}
    # split-K over the selected blocks; chunk count is shape-constant (cuda graph).
    TARGET_GRID = 256
    target = max(1, min(max_topk, TARGET_GRID // max(1, total_q * num_kv_heads)))
    num_topk_chunks = 1 << (target.bit_length() - 1)
    expected_o = (num_topk_chunks, total_q, num_heads, head_dim)
    expected_lse = (num_topk_chunks, total_q, num_heads)
    if workspace is None:
        o_partial = torch.empty(expected_o, dtype=q.dtype, device=q.device)
        lse_partial = torch.empty(expected_lse, dtype=torch.float32, device=q.device)
    else:
        if len(workspace) != 2:
            raise ValueError("decode workspace must be (o_partial, lse_partial)")
        o_partial, lse_partial = workspace
        if tuple(o_partial.shape) != expected_o or o_partial.dtype != q.dtype:
            raise ValueError(
                f"o_partial must be {expected_o} with dtype {q.dtype}, got "
                f"{tuple(o_partial.shape)} {o_partial.dtype}"
            )
        if tuple(lse_partial.shape) != expected_lse or lse_partial.dtype != torch.float32:
            raise ValueError(
                f"lse_partial must be {expected_lse} FP32, got "
                f"{tuple(lse_partial.shape)} {lse_partial.dtype}"
            )
        if o_partial.device != q.device or lse_partial.device != q.device:
            raise ValueError("decode workspace must be on q.device")
    if lse_out is None:
        lse_out = torch.empty((total_q, num_heads), dtype=torch.float32, device=q.device)
    elif (
        tuple(lse_out.shape) != (total_q, num_heads)
        or lse_out.dtype != torch.float32
        or lse_out.device != q.device
    ):
        raise ValueError("lse_out must be CUDA FP32 [total_q,num_heads] on q.device")
    grid = (total_q * num_topk_chunks, num_kv_heads)
    _gqa_sparse_decode_kernel[grid](
        q,
        kv_cache,
        topk_idx,
        o_partial,
        lse_partial,
        block_table,
        seq_lens,
        total_q,
        gqa_group_size,
        head_dim,
        max_topk,
        sm_scale,
        decode_query_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        kv_cache.stride(4),
        0 if topk_idx.shape[0] == 1 else topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_FP8=use_fp8,
        USE_PDL=use_pdl,
        **_sparse_attn_num_stages_kwarg(),
        **pdl_launch,
    )
    merge_grid = (total_q, num_heads)
    _merge_topk_attn_out_kernel[merge_grid](
        o_partial,
        lse_partial,
        output,
        lse_out,
        head_dim,
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse_out.stride(0),
        lse_out.stride(1),
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_PDL=use_pdl,
        **pdl_launch,
    )
    return lse_out
