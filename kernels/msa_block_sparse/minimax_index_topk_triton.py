# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for MiniMax M3 lightning-indexer block scoring + top-k.

Vendored from vLLM's own MiniMax M3 support (PR #45381,
``vllm/models/minimax_m3/common/ops/index_topk.py``). Upstream scores and
selects blocks with these rather than with MSA's max-score FMHA proxy, and that
is what lets its decode path be captured: MSA plans on the host and reads the
plan cost back off the device, while these are split-K with a shape-constant
grid. ``minimax_m3_index_decode`` retains the customer shared-route mode and
adds a provisional MiniMax-style mode that selects one route per KV/index
group.

Index queries score each 128-token block of index keys (max over the block),
then the top-k blocks (plus forced init/local blocks) are selected per query
token. Adapted to vLLM's paged KV cache: the KV page size is forced to equal the
sparse block size (128), so one sparse block maps to exactly one page.

Index-K cache layout (vLLM): ``(num_blocks, 128, idx_head_dim)`` (single head).

Only score_type="max" with the score-only indexer is implemented. The selected
block ids feed the block-sparse attention kernels in ``sparse_attn``.

Vendored into this repository from
``.tmp/vllm/vllm/model_executor/layers/minimax_index_topk_triton.py`` (the
``aip/compassv4-msa-score-reduce-topk32`` branch). The only change from that
copy is the import line below: ``vllm.platforms``/``vllm.triton_utils``/
``vllm.utils.math_utils`` are replaced with the local ``_vllm_compat`` shim so
this file needs only torch+triton, not a vLLM install.
"""

import torch

from msa_block_sparse._vllm_compat import current_platform, round_up, tl, triton

# One sparse block == one KV page.
SPARSE_BLOCK_SIZE = 128


# ---------------------------------------------------------------------------
# Bitonic top-k helpers (layout-agnostic).
# ---------------------------------------------------------------------------
@triton.jit
def _compare_and_swap(x, ids, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)
    mask = tl.arange(0, 2)[None, :, None]
    # Do not use y * 0 to select a half: fixed/invalid priority classes use
    # +/-inf, and IEEE inf * 0 is NaN. ``where`` preserves those classes.
    left = tl.broadcast_to(
        tl.sum(tl.where(mask == 0, y, 0.0), 1)[:, None, :], shape
    ).to(y.dtype)
    right = tl.broadcast_to(
        tl.sum(tl.where(mask != 0, y, 0.0), 1)[:, None, :], shape
    ).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(
        tl.sum(tl.where(mask == 0, y_idx, 0), 1)[:, None, :], shape
    )
    right_idx = tl.broadcast_to(
        tl.sum(tl.where(mask != 0, y_idx, 0), 1)[:, None, :], shape
    )
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)
    equal = left == right
    left_better = (left > right) | (equal & (left_idx < right_idx))
    # Preserve the original network's direction convention exactly while
    # extending its total order with the deterministic id tie-break.
    cond = left_better != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(
    x, ids, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    if order == 2:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(
            tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


# ---------------------------------------------------------------------------
# Index block-score kernel (paged). score[h, token, block] = max over the
# 128-token block of (idx_q . index_k), causal-masked. BLOCK_SIZE_K == 128 so
# each K-tile is exactly one page (BLOCKS_PER_K_BLOCK == 1).
# ---------------------------------------------------------------------------
# since prefill metadata is sliced from mixed batch metadata, seq_lens and prefix_lens
# might lose pointer alignment, which trigger Triton recompiles. we don't actually
# need pointer alignment for those tensors anyway because we do scalar load.
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _index_block_score_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    score_ptr,  # [num_idx_heads, total_q, max_block]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens,  # [batch+1] query start offsets
    seq_lens,  # [batch] total K length
    prefix_lens,  # [batch] context length before this chunk's queries
    num_idx_heads,
    head_dim: tl.constexpr,
    sm_scale,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // num_idx_heads
    pid_h = pid_bh % num_idx_heads

    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if BLOCK_SIZE_Q * pid_q >= q_len:
        return

    q_ptrs = tl.make_block_ptr(
        base=q_ptr + seq_start * stride_q_n + pid_h * stride_q_h,
        shape=(q_len, head_dim),
        strides=(stride_q_n, stride_q_d),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, head_dim),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0,), padding_option="zero")
    q_start = prefix_len + pid_q * BLOCK_SIZE_Q

    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q + prefix_len
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, head_dim)
    # Block table row for this request.
    bt_row = block_table_ptr + pid_b * stride_bt_b
    # Causal window: only blocks up to the last query token's position.
    hi = min(seq_len, prefix_len + (pid_q + 1) * BLOCK_SIZE_Q)
    for i in tl.range(0, hi, BLOCK_SIZE_K):
        blk = i // BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = i + off_k
        # index-K for this page: [BLOCK_SIZE_D, BLOCK_SIZE_K] (transposed)
        # we don't need masked load for K, because KV cache ensures
        # allocation is multiple of BLOCK_SIZE_K.
        # for tokens beyond seqlen, they will be masked in qk later.
        k = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_k[None, :] * stride_ik_pos
            + off_d[:, None] * stride_ik_d,
        )
        qk = tl.dot(q, k) * sm_scale_log2e
        # apply causal mask as needed
        if q_start < i + BLOCK_SIZE_K:
            qk = tl.where(off_q[:, None] >= pos[None, :], qk, float("-inf"))
        # one sparse block per K-tile -> max over the 128 positions
        score = tl.max(qk, axis=1)  # [BLOCK_SIZE_Q]
        s_ptrs = (
            score_ptr
            + pid_h * stride_s_h
            + (seq_start + pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q))
            * stride_s_n
            + blk * stride_s_k
        )
        q_store_mask = (pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) < q_len
        tl.store(s_ptrs, score, mask=q_store_mask)


# ---------------------------------------------------------------------------
# Top-k selection over per-token block scores (layout-agnostic). block_size_q
# is 1 for M3, so top-k is computed per query token.
# ---------------------------------------------------------------------------
# since prefill metadata is sliced from mixed batch metadata, prefix_lens
# might lose pointer alignment, which trigger Triton recompiles. we don't actually
# need pointer alignment for those tensors anyway because we do scalar load.
@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_K": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 64}, num_warps=2, num_stages=2),
    ],
    key=["BLOCK_SIZE_T"],
)
@triton.jit(do_not_specialize_on_alignment=["prefix_lens"])
def _topk_index_kernel(
    s_ptr,  # [num_heads, total_q, max_block]
    ti_ptr,  # [num_heads, total_q, topk]
    sample_interval: tl.constexpr,  # block_size_q (1 for M3)
    block_size: tl.constexpr,  # sparse block size (128)
    cu_seqlens,
    cu_seqblocks_q,
    prefix_lens,
    topk,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_ti_h,
    stride_ti_n,
    stride_ti_t,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    MASK_INIT: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_K > BLOCK_SIZE_T)
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    seq_start = tl.load(cu_seqlens + pid_b)
    block_start = tl.load(cu_seqblocks_q + pid_b)
    block_num = tl.load(cu_seqblocks_q + pid_b + 1) - block_start
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q >= block_num:
        return
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    s_ptrs = (
        s_ptr
        + (seq_start + pid_q * sample_interval) * stride_s_n
        + pid_h * stride_s_h
        + off_k * stride_s_k
    )
    topk_score = tl.full((BLOCK_SIZE_K,), float("-inf"), dtype=tl.float32)
    topk_idx = tl.full((BLOCK_SIZE_K,), 0, dtype=tl.int32)
    left_half_mask = tl.arange(0, BLOCK_SIZE_K) < BLOCK_SIZE_K // 2
    valid_blocks = (prefix_len + pid_q * sample_interval + block_size) // block_size
    for i in tl.range(0, valid_blocks, BLOCK_SIZE_K):
        causal_mask = i + off_k < valid_blocks
        local_mask = i + off_k >= max(0, valid_blocks - local_blocks)
        init_mask = i + off_k < init_blocks
        score = tl.load(s_ptrs, mask=causal_mask, other=float("-inf")).to(tl.float32)
        # Reserve +/-inf for fixed and invalid classes. Dynamic +inf maps to
        # the greatest finite FP32 value; NaN/-inf map to the least finite
        # value. Signed zero remains equal and is resolved by block id.
        max_finite: tl.constexpr = 3.4028234663852886e38
        score = tl.where(score != score, -max_finite, score)
        score = tl.maximum(tl.minimum(score, max_finite), -max_finite)
        # Invalid lanes must remain below a valid dynamic -inf/NaN. Otherwise
        # sentinel block id zero wins the deterministic lower-id tie.
        score = tl.where(causal_mask, score, float("-inf"))
        s_ptrs = s_ptrs + stride_s_k * BLOCK_SIZE_K
        if MASK_INIT:
            score = tl.where(causal_mask & init_mask, float("-inf"), score)
        else:
            score = tl.where(causal_mask & init_mask, float("inf"), score)
        if MASK_LOCAL:
            score = tl.where(causal_mask & local_mask, float("-inf"), score)
        else:
            score = tl.where(causal_mask & local_mask, float("inf"), score)
        topk_score, last_topk_score = score, topk_score
        topk_idx, last_topk_idx = (tl.where(causal_mask, i + off_k + 1, 0), topk_idx)
        n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
        for j in tl.static_range(1, n_dims):
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), j, 2, n_dims
            )
        if i != 0:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, False, n_dims
            )
            topk_score_new = tl.where(
                left_half_mask, last_topk_score, topk_score
            )
            topk_idx_new = tl.where(left_half_mask, last_topk_idx, topk_idx)
            topk_score, topk_idx = _bitonic_merge(
                topk_score_new, topk_idx_new.to(tl.int32), n_dims, True, n_dims
            )
        else:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, True, n_dims
            )
    topk_mask = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    topk_idx = tl.sum(
        tl.where(
            topk_mask[:, None],
            tl.reshape(topk_idx - 1, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
            0,
        ),
        axis=0,
    )
    ti_ptrs = (
        ti_ptr
        + (block_start + pid_q) * stride_ti_n
        + pid_h * stride_ti_h
        + off_t * stride_ti_t
    )
    store_mask = off_t < topk
    valid_mask = off_t < valid_blocks
    topk_idx = tl.where(store_mask & valid_mask, topk_idx, -1)
    tl.store(ti_ptrs, topk_idx.to(ti_ptrs.dtype.element_ty), mask=store_mask)


@triton.heuristics(
    {
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["packed_width"])
    }
)
@triton.jit(do_not_specialize_on_alignment=["prefix_lens"])
def _topk_packed_exact_kernel(
    s_ptr,
    ti_ptr,
    sample_interval: tl.constexpr,
    block_size: tl.constexpr,
    cu_seqlens,
    cu_seqblocks_q,
    prefix_lens,
    topk,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    packed_width,
    k_bucket,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_ti_h,
    stride_ti_n,
    stride_ti_t,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    seq_start = tl.load(cu_seqlens + pid_b)
    block_start = tl.load(cu_seqblocks_q + pid_b)
    block_num = tl.load(cu_seqblocks_q + pid_b + 1) - block_start
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q >= block_num:
        return

    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    valid_blocks = (prefix_len + pid_q * sample_interval + block_size) // block_size
    prefix_count = min(init_blocks, valid_blocks)
    local_start = max(0, valid_blocks - local_blocks)
    nonoverlap_local_start = max(local_start, prefix_count)
    local_count = valid_blocks - nonoverlap_local_start
    target_count = min(topk, valid_blocks)
    fixed_count = prefix_count + local_count
    dynamic_count = target_count - fixed_count

    s_ptrs = (
        s_ptr
        + (seq_start + pid_q * sample_interval) * stride_s_n
        + pid_h * stride_s_h
        + off_k * stride_s_k
    )
    best_keys = tl.full((BLOCK_SIZE_T,), -9223372036854775807, tl.int64)
    for i in tl.range(0, valid_blocks, BLOCK_SIZE_K):
        block_ids = i + off_k
        causal_mask = block_ids < valid_blocks
        fixed_mask = (block_ids < prefix_count) | (block_ids >= local_start)
        candidate_mask = causal_mask & ~fixed_mask
        score = tl.load(s_ptrs, mask=causal_mask, other=float("-inf")).to(tl.float32)
        max_finite: tl.constexpr = 3.4028234663852886e38
        score = tl.where(score != score, -max_finite, score)
        score = tl.maximum(tl.minimum(score, max_finite), -max_finite)
        # The frozen provisional contract treats both signed zeros as one
        # score, with block id as the only tie breaker.
        score = tl.where(score == 0.0, 0.0, score)
        score = tl.where(candidate_mask, score, float("-inf"))

        # This signed transform preserves FP32 order.  Keeping it in the high
        # 32 bits of a signed int64 and inverting the id in the low 32 bits
        # gives the frozen higher-score/lower-id total order.
        score_bits = score.to(tl.int32, bitcast=True)
        ordered_score = score_bits ^ ((score_bits >> 31) & 0x7FFFFFFF)
        tie_key = 4294967295 - block_ids.to(tl.int64)
        packed = (ordered_score.to(tl.int64) << 32) | tie_key
        chunk_keys = tl.topk(packed, BLOCK_SIZE_T)
        best_keys = tl.topk(
            tl.cat(best_keys, chunk_keys, can_reorder=True), BLOCK_SIZE_T
        )
        s_ptrs = s_ptrs + stride_s_k * BLOCK_SIZE_K

    selected_ids = (4294967295 - (best_keys & 0xFFFFFFFF)).to(tl.int32)
    ti_base = (
        ti_ptr
        + (block_start + pid_q) * stride_ti_n
        + pid_h * stride_ti_h
    )
    tl.store(
        ti_base + off_t * stride_ti_t,
        off_t.to(ti_base.dtype.element_ty),
        mask=off_t < prefix_count,
    )
    tl.store(
        ti_base + (prefix_count + off_t) * stride_ti_t,
        (nonoverlap_local_start + off_t).to(ti_base.dtype.element_ty),
        mask=off_t < local_count,
    )
    tl.store(
        ti_base + (fixed_count + off_t) * stride_ti_t,
        selected_ids.to(ti_base.dtype.element_ty),
        mask=off_t < dynamic_count,
    )
    tl.store(
        ti_base + off_t * stride_ti_t,
        -1,
        mask=(off_t >= target_count) & (off_t < topk),
    )
    tl.store(
        ti_base + (BLOCK_SIZE_T + off_t) * stride_ti_t,
        -1,
        mask=(BLOCK_SIZE_T + off_t >= target_count)
        & (BLOCK_SIZE_T + off_t < topk),
    )


@triton.jit
def _canonicalize_dynamic_top16_padding_kernel(
    route_ptr,
    prefix_lens,
    total_q,
    stride_h,
    stride_q,
    stride_t,
    block_size: tl.constexpr,
):
    """Rewrite only underfilled Dynamic-16 tail slots to canonical ``-1``."""
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_q >= total_q:
        return
    valid_blocks = (tl.load(prefix_lens) + pid_q + block_size) // block_size
    prefix_count = min(8, valid_blocks)
    local_start = max(0, valid_blocks - 8)
    nonoverlap_local_start = max(local_start, prefix_count)
    fixed_count = prefix_count + valid_blocks - nonoverlap_local_start
    dynamic_count = min(32, valid_blocks) - fixed_count
    off_t = tl.arange(0, 16)
    base = route_ptr + pid_h * stride_h + pid_q * stride_q
    tl.store(base + off_t * stride_t, -1, mask=off_t >= dynamic_count)


# ---------------------------------------------------------------------------
# Decode index-score kernel (split-K over seq blocks). Decode batches are
# flattened request-major, with a runtime query length used to map each query
# token back to its request metadata. Chunk counts depend only on shape
# constants so the grid is fixed within a cuda graph. Base-2 (exp2/log2)
# softmax matches prefill.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["num_kv_chunks", "decode_query_len"])
def _decode_index_score_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    score_ptr,  # [num_idx_heads, total_q, max_block]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    init_blocks,
    local_blocks,
    sm_scale,
    decode_query_len,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    num_kv_chunks,
    USE_PDL: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_b = tl.program_id(0)  # flattened query-token id
    pid_c = tl.program_id(1)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    # block-aligned fixed-count split: grid independent of seq_len (cuda graph).
    chunk_size_blocks = (num_blocks + num_kv_chunks - 1) // num_kv_chunks
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)
    if chunk_start_block >= chunk_end_block:
        # Nothing to produce, so the dependents are free to start now.
        if USE_PDL:
            tl.extra.cuda.gdc_launch_dependents()
        return
    off_k = tl.arange(0, BLOCK_SIZE_K)  # positions within a 128-block
    off_d = tl.arange(0, head_dim)
    bt_row = block_table_ptr + req_id * stride_bt_b
    # Force-select init (1e30) and local (1e29, higher priority) blocks.
    local_start = tl.maximum(0, num_blocks - local_blocks)
    # query vectors across all heads
    q = tl.load(
        q_ptr
        + pid_b * stride_q_n
        + tl.arange(0, num_idx_heads) * stride_q_h
        + off_d[:, None] * stride_q_d,
    )  # [D,H]
    for blk in tl.range(chunk_start_block, chunk_end_block):
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = blk * BLOCK_SIZE_K + off_k
        pos_mask = pos < kv_len
        # we don't need masked load for K, because KV cache ensures
        # allocation is multiple of BLOCK_SIZE_K.
        # for tokens beyond seqlen, they will be masked in qk later.
        k = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_k[:, None] * stride_ik_pos
            + off_d * stride_ik_d,
        )  # [N,D]
        kq = tl.dot(k, q) * sm_scale_log2e  # [N,H]
        kq = tl.where(pos_mask[:, None], kq, float("-inf"))
        score = tl.max(kq, axis=0)  # [H]
        max_finite: tl.constexpr = 3.4028234663852886e38
        score = tl.where(score != score, -max_finite, score)
        score = tl.maximum(tl.minimum(score, max_finite), -max_finite)
        is_init = blk < init_blocks
        is_local = (blk >= local_start) & (blk < num_blocks)
        # Fixed membership is a separate +inf priority class. Dynamic +inf was
        # normalized to max_finite above, so it cannot displace a fixed block.
        score = tl.where(is_init | is_local, float("inf"), score)
        tl.store(
            score_ptr
            + tl.arange(0, num_idx_heads) * stride_s_h
            + pid_b * stride_s_n
            + blk * stride_s_k,
            score,
        )

    # Release the dependents only once the scores this CTA owns are stored.
    # Signalling in the preamble -- as this kernel used to, and as the merge
    # kernel still can because nothing downstream reads its output through the
    # PDL chain -- lets `_topk_index_partial_kernel` start reading `score`
    # before it has been written. The sibling partial kernel already places its
    # signal after its work for the same reason.
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


# ---------------------------------------------------------------------------
# Decode top-k (split-K): per-chunk partial top-k + merge. Forced init/local
# blocks are already encoded in the scores.
# ---------------------------------------------------------------------------
@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE_K": 64}, num_warps=2, num_stages=2),
    ],
    key=["topk"],
)
@triton.jit(do_not_specialize=["chunk_blocks", "decode_query_len"])
def _topk_index_partial_kernel(
    s_ptr,  # score: [num_idx_heads, total_q, max_block]
    ts_partial_ptr,  # partial scores out: [NUM_TOPK_CHUNKS, num_idx_heads, total_q, T]
    ti_partial_ptr,  # partial idx out (1-indexed global, 0=invalid): same shape
    seq_lens,  # [num_reqs]
    block_size: tl.constexpr,  # sparse block size (128)
    topk: tl.constexpr,
    chunk_blocks,  # how many score-blocks each chunk owns
    decode_query_len,
    stride_s_h,
    stride_s_b,
    stride_s_k,
    stride_ts_c,
    stride_ts_h,
    stride_ts_b,
    stride_ts_t,
    stride_ti_c,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    NUM_SCORE_HEADS: tl.constexpr,
    PER_GROUP: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    tl.static_assert(topk < BLOCK_SIZE_K)
    pid_b = tl.program_id(0)  # flattened query-token id
    pid_h = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + block_size - 1) // block_size

    # Slice this chunk owns within [0, num_blocks).
    chunk_start = pid_chunk * chunk_blocks
    chunk_end = tl.minimum(chunk_start + chunk_blocks, num_blocks)
    chunk_actual = tl.maximum(chunk_end - chunk_start, 0)

    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)

    s_ptrs = (
        s_ptr
        + pid_b * stride_s_b
        + (chunk_start + off_k) * stride_s_k
    )

    topk_score = tl.full((BLOCK_SIZE_K,), float("-inf"), dtype=tl.float32)
    topk_idx = tl.full((BLOCK_SIZE_K,), 0, dtype=tl.int32)
    left_half_mask = tl.arange(0, BLOCK_SIZE_K) < BLOCK_SIZE_K // 2

    # Streaming top-K within this chunk. tl.range(0, 0) is a no-op so empty
    # chunks (chunk_actual == 0) skip the body and store sentinel -1e30 / 0.
    for i in tl.range(0, chunk_actual, BLOCK_SIZE_K):
        mask = off_k < chunk_actual - i
        if PER_GROUP:
            score = tl.load(
                s_ptrs + pid_h * stride_s_h,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)
            score = tl.where(score != score, -3.4028234663852886e38, score)
        else:
            # C0 compatibility: fold score rows inside the first partial pass.
            score = tl.full((BLOCK_SIZE_K,), float("-inf"), dtype=tl.float32)
            for h in tl.static_range(0, NUM_SCORE_HEADS):
                head_score = tl.load(
                    s_ptrs + h * stride_s_h,
                    mask=mask,
                    other=float("-inf"),
                ).to(tl.float32)
                head_score = tl.where(
                    head_score != head_score, -3.4028234663852886e38, head_score
                )
                score = tl.maximum(score, head_score)
        s_ptrs = s_ptrs + stride_s_k * BLOCK_SIZE_K
        topk_score, last_topk_score = score, topk_score
        topk_idx, last_topk_idx = (
            tl.where(mask, chunk_start + i + off_k + 1, 0),  # 1-indexed global
            topk_idx,
        )
        n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
        for j in tl.static_range(1, n_dims):
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), j, 2, n_dims
            )
        if i != 0:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, False, n_dims
            )
            topk_score_new = tl.where(
                left_half_mask, last_topk_score, topk_score
            )
            topk_idx_new = tl.where(left_half_mask, last_topk_idx, topk_idx)
            topk_score, topk_idx = _bitonic_merge(
                topk_score_new, topk_idx_new.to(tl.int32), n_dims, True, n_dims
            )
        else:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, True, n_dims
            )

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    # Extract first BLOCK_SIZE_T entries (top-K of this chunk after the sort).
    topk_mask_extract = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    final_score = tl.sum(
        tl.where(
            topk_mask_extract[:, None],
            tl.reshape(topk_score, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
            0.0,
        ),
        axis=0,
    )
    final_idx = tl.sum(
        tl.where(
            topk_mask_extract[:, None],
            tl.reshape(topk_idx, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
            0,
        ),
        axis=0,
    )

    # Always write all BLOCK_SIZE_T slots — invalid slots carry -inf / 0
    # sentinels and lose to real scores in the merge stage.
    ts_ptrs = (
        ts_partial_ptr
        + pid_chunk * stride_ts_c
        + pid_b * stride_ts_b
        + pid_h * stride_ts_h
        + off_t * stride_ts_t
    )
    ti_ptrs = (
        ti_partial_ptr
        + pid_chunk * stride_ti_c
        + pid_b * stride_ti_b
        + pid_h * stride_ti_h
        + off_t * stride_ti_t
    )
    tl.store(ts_ptrs, final_score)
    tl.store(ti_ptrs, final_idx)


@triton.heuristics(
    {
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"]),
        "BLOCK_SIZE_K": lambda args: triton.next_power_of_2(
            args["num_topk_chunks"] * triton.next_power_of_2(args["topk"])
        ),
    }
)
@triton.jit(do_not_specialize=["num_topk_chunks", "decode_query_len"])
def _topk_index_merge_kernel(
    ts_partial_ptr,  # partial scores: [NUM_TOPK_CHUNKS, num_idx_heads, total_q, T]
    ti_partial_ptr,  # partial idx (1-indexed global, 0=invalid): same shape
    ti_final_ptr,  # final idx (0-indexed, -1=invalid): [num_idx_heads, total_q, topk]
    seq_lens,  # [num_reqs]
    block_size: tl.constexpr,  # sparse block size (128)
    topk: tl.constexpr,
    decode_query_len,
    stride_ts_c,
    stride_ts_h,
    stride_ts_b,
    stride_ts_t,
    stride_ti_c,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    stride_tif_h,
    stride_tif_b,
    stride_tif_t,
    num_topk_chunks,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid_b = tl.program_id(0)  # flattened query-token id
    pid_h = tl.program_id(1)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + block_size - 1) // block_size

    # Load NUM_TOPK_CHUNKS * BLOCK_SIZE_T candidates, padded to BLOCK_SIZE_K.
    # Candidate at flat position p comes from chunk = p // BLOCK_SIZE_T,
    # in_chunk = p % BLOCK_SIZE_T.
    off = tl.arange(0, BLOCK_SIZE_K)
    chunk_idx = off // BLOCK_SIZE_T
    in_chunk_idx = off % BLOCK_SIZE_T
    valid = chunk_idx < num_topk_chunks

    score_offset = (
        chunk_idx * stride_ts_c
        + pid_h * stride_ts_h
        + pid_b * stride_ts_b
        + in_chunk_idx * stride_ts_t
    )
    idx_offset = (
        chunk_idx * stride_ti_c
        + pid_h * stride_ti_h
        + pid_b * stride_ti_b
        + in_chunk_idx * stride_ti_t
    )

    score = tl.load(ts_partial_ptr + score_offset, mask=valid, other=float("-inf")).to(
        tl.float32
    )
    score = tl.where(score != score, -3.4028234663852886e38, score)
    idx = tl.load(ti_partial_ptr + idx_offset, mask=valid, other=0).to(tl.int32)

    # Full bitonic descending sort of BLOCK_SIZE_K items.
    n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
    for j in tl.static_range(1, n_dims):
        score, idx = _bitonic_merge(score, idx.to(tl.int32), j, 2, n_dims)
    score, idx = _bitonic_merge(score, idx.to(tl.int32), n_dims, True, n_dims)

    # Extract first BLOCK_SIZE_T positions — these are the global top-K.
    extract_mask = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    topk_idx_final = tl.sum(
        tl.where(
            extract_mask[:, None],
            tl.reshape(idx - 1, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
            0,
        ),
        axis=0,
    )

    off_t = tl.arange(0, BLOCK_SIZE_T)
    tif_ptrs = (
        ti_final_ptr
        + pid_h * stride_tif_h
        + pid_b * stride_tif_b
        + off_t * stride_tif_t
    )
    store_mask = off_t < topk
    topk_idx_final = tl.where(off_t < tl.minimum(topk, num_blocks), topk_idx_final, -1)
    tl.store(
        tif_ptrs, topk_idx_final.to(ti_final_ptr.dtype.element_ty), mask=store_mask
    )

    # As in the score kernel: signal only once the selection is stored. This
    # kernel is not the end of the PDL chain -- the block-sparse attention
    # kernel is launched next and reads exactly this buffer, so releasing it in
    # the preamble lets it gather from page ids that have not been written.
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
@torch.no_grad()
def minimax_m3_index_score(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    max_seq_len: int,
    num_kv_heads: int,
    sm_scale: float,
) -> torch.Tensor:
    """Compute per-token index scores for each visible sparse block.

    Returns score [num_kv_heads, total_q, max_block], where each score is the
    max over a 128-token index-K block. M3 has num_idx_heads == num_kv_heads.
    """
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert num_idx_heads == num_kv_heads, (
        "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    )
    batch = cu_seqlens_q.shape[0] - 1
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)

    # Keep score strides 16-divisible to avoid Triton recompiles.
    score_block_stride = round_up(max_block, 16)
    score = torch.empty(
        (num_idx_heads, total_q, score_block_stride),
        dtype=torch.float32,
        device=idx_q.device,
    )
    BLOCK_SIZE_Q = 64
    grid_score = (triton.cdiv(max_query_len, BLOCK_SIZE_Q), batch * num_idx_heads)
    _index_block_score_kernel[grid_score](
        idx_q,
        index_kv_cache,
        score,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        num_idx_heads,
        head_dim,
        sm_scale,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        index_kv_cache.stride(0),
        index_kv_cache.stride(1),
        index_kv_cache.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
    )
    return score


@torch.no_grad()
def minimax_m3_index_topk(
    score: torch.Tensor,  # [num_idx_heads, total_q, max_block]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    out: torch.Tensor | None = None,
    *,
    _packed_exact: bool = False,
    _packed_width: int = 32,
) -> torch.Tensor:
    """Select exact per-score-head top-k with deterministic block-id ties."""
    if topk <= 0 or topk >= 64:
        raise ValueError(f"topk must be in [1, 63], got {topk}")
    if init_blocks < 0 or local_blocks < 0:
        raise ValueError("init_blocks and local_blocks must be non-negative")
    if init_blocks + local_blocks > topk:
        raise ValueError(
            f"fixed quota {init_blocks}+{local_blocks} exceeds topk={topk}"
        )
    if _packed_exact and _packed_width not in (16, 32):
        raise ValueError("packed exact selector width must be 16 or 32")
    if score.ndim != 3 or not score.is_cuda or score.dtype != torch.float32:
        raise ValueError("score must be CUDA FP32 [num_idx_heads,total_q,max_block]")
    num_idx_heads = score.shape[0]
    batch = cu_seqlens_q.shape[0] - 1
    total_q = score.shape[1]
    expected = (num_idx_heads, total_q, topk)
    if out is None:
        topk_idx = torch.empty(expected, dtype=torch.int32, device=score.device)
    else:
        if (
            tuple(out.shape) != expected
            or out.dtype != torch.int32
            or out.device != score.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                f"out must be contiguous CUDA int32 {expected} on score.device"
            )
        topk_idx = out
    # block_size_q == 1 -> query blocks coincide with query tokens.
    grid_topk = (max_query_len, batch, num_idx_heads)
    kernel = _topk_packed_exact_kernel if _packed_exact else _topk_index_kernel
    k_bucket = 1 << (int(score.shape[2]) - 1).bit_length()
    if _packed_exact:
        # C023 established 32/1 through the aligned 32K domain. C025/C026
        # established 64/1 above that boundary and through 1M. Select from
        # the actual score width so the production path is deterministic.
        packed_block_size = 64 if int(score.shape[2]) > 256 else 32
        extra_args = (_packed_width, k_bucket)
        launch_meta = {
            "BLOCK_SIZE_K": packed_block_size,
            "num_warps": 1,
            "num_stages": 2,
        }
    else:
        extra_args = ()
        launch_meta = {"MASK_INIT": False, "MASK_LOCAL": False}
    kernel[grid_topk](
        score,
        topk_idx,
        1,  # sample_interval (block_size_q)
        SPARSE_BLOCK_SIZE,
        cu_seqlens_q,
        cu_seqlens_q,  # cu_seqblocks_q == cu_seqlens_q when block_size_q == 1
        prefix_lens,
        topk,
        init_blocks,
        local_blocks,
        *extra_args,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        **launch_meta,
    )
    return topk_idx


@torch.no_grad()
def canonicalize_dynamic_top16_padding(
    route: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> None:
    """Canonicalize ``INT_MAX`` empty slots emitted by fused Score Top-16."""
    if (
        route.ndim != 3
        or route.shape[0] != 4
        or route.shape[2] != 16
        or route.dtype != torch.int32
        or not route.is_cuda
        or not route.is_contiguous()
    ):
        raise ValueError("route must be contiguous CUDA int32 [4,Sq,16]")
    if (
        prefix_lens.shape != (1,)
        or prefix_lens.dtype != torch.int32
        or prefix_lens.device != route.device
    ):
        raise ValueError("canonical padding requires CUDA int32 prefix_lens [1]")
    total_q = int(route.shape[1])
    _canonicalize_dynamic_top16_padding_kernel[(total_q, 4)](
        route,
        prefix_lens,
        total_q,
        route.stride(0),
        route.stride(1),
        route.stride(2),
        SPARSE_BLOCK_SIZE,
        num_warps=1,
    )


@torch.no_grad()
def minimax_m3_index_decode(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    sm_scale: float,
    decode_query_len: int,
    *,
    per_group: bool = False,
    score_out: torch.Tensor | None = None,
    topk_out: torch.Tensor | None = None,
    workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
    _score_grid_counts_heads: bool = False,
) -> torch.Tensor:
    """Decode index block-score + top-k, both split-K (cudagraph-safe).

    Returns topk_idx ``[Hkv,total_q,topk]`` when ``per_group=True`` or the C0
    compatibility shape ``[1,total_q,topk]`` otherwise.

    Token scores within each 128-token block are reduced with max. With
    ``per_group=True`` the four index/KV groups are selected independently;
    with ``per_group=False`` their score rows are max-reduced for exact C0
    compatibility. There is no sum-reduction mode.
    """
    if topk <= 0 or topk >= 64:
        raise ValueError(f"topk must be in [1, 63], got {topk}")
    if init_blocks < 0 or local_blocks < 0:
        raise ValueError("init_blocks and local_blocks must be non-negative")
    if init_blocks + local_blocks > topk:
        raise ValueError(
            f"fixed quota {init_blocks}+{local_blocks} exceeds topk={topk}"
        )
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert num_idx_heads == num_kv_heads, (
        "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    )
    assert total_q == seq_lens.shape[0] * decode_query_len
    batch = total_q
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    use_pdl = current_platform.is_arch_support_pdl()
    # `launch_pdl` is a Triton runtime kwarg only some backends accept (CUDA
    # SM9+); this ROCm Triton rejects it even when False ("Keyword argument
    # launch_pdl was specified but unrecognised"). Only pass it when PDL is
    # actually supported -- on ROCm use_pdl is always False, so it's omitted.
    pdl_launch = {"launch_pdl": True} if use_pdl else {}

    # Keep score strides 16-divisible to avoid Triton recompiles.
    score_block_stride = round_up(max_block, 16)
    score_shape = (num_idx_heads, total_q, score_block_stride)
    if score_out is None:
        score = torch.empty(score_shape, dtype=torch.float32, device=idx_q.device)
    else:
        if (
            tuple(score_out.shape) != score_shape
            or score_out.dtype != torch.float32
            or score_out.device != idx_q.device
            or not score_out.is_contiguous()
        ):
            raise ValueError(
                f"score_out must be contiguous CUDA FP32 {score_shape}"
            )
        score = score_out
    # split-K over seq blocks; chunk count depends only on shape constants so
    # the grid is fixed within a cuda graph.
    TARGET_GRID = 4096
    MAX_NUM_KV_CHUNKS = 256
    # One score CTA already computes every index/KV-group head. Counting heads
    # again underfills the long-context grid (1024 CTAs and 1.38 waves/SM at
    # BS64), but the additional empty chunks regress short KV. Keep the
    # measured C003 win behind a strict target-shape whitelist and retain B0
    # everywhere else. The private switch is only for same-process evidence.
    use_c003_score_grid = batch in (32, 64) and max_seq_len >= 131_072
    count_heads = _score_grid_counts_heads or not use_c003_score_grid
    score_grid_owners = batch * (num_idx_heads if count_heads else 1)
    target = max(1, min(MAX_NUM_KV_CHUNKS, TARGET_GRID // max(1, score_grid_owners)))
    num_kv_chunks = 1 << (target.bit_length() - 1)
    grid_score = (batch, num_kv_chunks)
    _decode_index_score_kernel[grid_score](
        idx_q,
        index_kv_cache,
        score,
        block_table,
        seq_lens,
        num_idx_heads,
        head_dim,
        init_blocks,
        local_blocks,
        sm_scale,
        decode_query_len,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        index_kv_cache.stride(0),
        index_kv_cache.stride(1),
        index_kv_cache.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        num_kv_chunks=num_kv_chunks,
        USE_PDL=use_pdl,
        **pdl_launch,
    )

    sel_heads = num_idx_heads if per_group else 1

    topk_shape = (sel_heads, total_q, topk)
    if topk_out is None:
        topk_idx = torch.empty(topk_shape, dtype=torch.int32, device=idx_q.device)
    else:
        if (
            tuple(topk_out.shape) != topk_shape
            or topk_out.dtype != torch.int32
            or topk_out.device != idx_q.device
            or not topk_out.is_contiguous()
        ):
            raise ValueError(
                f"topk_out must be contiguous CUDA int32 {topk_shape}"
            )
        topk_idx = topk_out
    # Chunk count is shape-constant (cudagraph-safe), capped so the merge sorts
    # pow2(num_topk_chunks * pow2(topk)) candidates.
    TOPK_TARGET_GRID = 64
    MAX_NUM_TOPK_CHUNKS = 16
    topk_target = max(
        1, min(MAX_NUM_TOPK_CHUNKS, TOPK_TARGET_GRID // max(1, batch * sel_heads))
    )
    num_topk_chunks = 1 << (topk_target.bit_length() - 1)
    block_size_t = triton.next_power_of_2(topk)
    chunk_blocks = (max_block + num_topk_chunks - 1) // num_topk_chunks
    partial_shape = (num_topk_chunks, sel_heads, batch, block_size_t)
    if workspace is None:
        topk_score_partial = torch.empty(
            partial_shape, dtype=torch.float32, device=idx_q.device
        )
        topk_idx_partial = torch.empty(
            partial_shape, dtype=torch.int32, device=idx_q.device
        )
    else:
        if len(workspace) != 2:
            raise ValueError("top-k workspace must be (score_partial, idx_partial)")
        topk_score_partial, topk_idx_partial = workspace
        if (
            tuple(topk_score_partial.shape) != partial_shape
            or topk_score_partial.dtype != torch.float32
            or topk_score_partial.device != idx_q.device
        ):
            raise ValueError(
                f"score_partial must be CUDA FP32 {partial_shape} on idx_q.device"
            )
        if (
            tuple(topk_idx_partial.shape) != partial_shape
            or topk_idx_partial.dtype != torch.int32
            or topk_idx_partial.device != idx_q.device
        ):
            raise ValueError(
                f"idx_partial must be CUDA int32 {partial_shape} on idx_q.device"
            )
    _topk_index_partial_kernel[(batch, sel_heads, num_topk_chunks)](
        score,
        topk_score_partial,
        topk_idx_partial,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        topk,
        chunk_blocks,
        decode_query_len,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        topk_score_partial.stride(0),
        topk_score_partial.stride(1),
        topk_score_partial.stride(2),
        topk_score_partial.stride(3),
        topk_idx_partial.stride(0),
        topk_idx_partial.stride(1),
        topk_idx_partial.stride(2),
        topk_idx_partial.stride(3),
        NUM_SCORE_HEADS=num_idx_heads,
        PER_GROUP=per_group,
        USE_PDL=use_pdl,
        **pdl_launch,
    )
    _topk_index_merge_kernel[(batch, sel_heads)](
        topk_score_partial,
        topk_idx_partial,
        topk_idx,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        topk,
        decode_query_len,
        topk_score_partial.stride(0),
        topk_score_partial.stride(1),
        topk_score_partial.stride(2),
        topk_score_partial.stride(3),
        topk_idx_partial.stride(0),
        topk_idx_partial.stride(1),
        topk_idx_partial.stride(2),
        topk_idx_partial.stride(3),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        num_topk_chunks=num_topk_chunks,
        USE_PDL=use_pdl,
        **pdl_launch,
    )
    return topk_idx
