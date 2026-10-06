"""MiniMax MSA-compatible sparse attention on SM90 (H100).

Implements ``build_k2q_csr`` and ``sparse_atten_func`` with the signatures
and numerical contract of the bundled SM100 MSA runtime
(``mbsa.cute.sm100_sparse``), on the SM90 CuTe DSL kernels:

* CSR: ``k2q_row_ptr [Hkv, rows + 1]`` and ``k2q_q_indices [Hkv, >= Q*topK]``
  hold request-local queries in ascending order per row. Rows enumerate the
  packed KV blocks level-major: block ``l`` of every request with more than
  ``l`` blocks, in request order, before block ``l + 1``.
* Causal masking is bottom-right aligned (``seqlen_k - seqlen_q``); the KV
  tail (``seqused_k`` for paged KV) is always masked.
* ``initial_out``/``initial_lse`` fold a fixed-attention state into the
  result, as the SM100 K2 combine does (any batch size here).

The CSR is converted on device into the same KV-centric work items used by
the MBSA Dynamic path; no step synchronizes with the host except deriving
``total_rows`` when ``build_k2q_csr`` is called without it.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import math

import torch
from cutlass import Float32, Int32

from .sm90 import _compile, _cute as _wrap, _signature, _stream
from .src.sm90.attention import CombineSm90, DynamicAttentionSm90
from .src.sm90.schedule import CsrCountsSm90, CsrRelationsSm90, ScheduleScanSm90


BLOCK = 128
SUPPORTED_BLOCKS = (32, 64, 128)
HEAD_DIM = 128
SUPPORTED_TOPK = (4, 8, 16, 32)
SUPPORTED_GROUPS = (8,)
PARTIAL_DTYPES = (torch.float32, torch.bfloat16, torch.float16, torch.float8_e4m3fn)
_WORKSPACES: dict[tuple, dict] = {}
_WRAPPERS: OrderedDict = OrderedDict()
_MAX_WRAPPERS = 512


def _cute(t: torch.Tensor, align: int = 16):
    """DLPack wrappers encode only pointer and layout; reuse them across calls."""
    key = (t.data_ptr(), tuple(t.shape), t.stride(), t.dtype, t.device, align)
    wrapper = _WRAPPERS.pop(key, None)
    if wrapper is None:
        wrapper = _wrap(t, align)
        if len(_WRAPPERS) >= _MAX_WRAPPERS:
            _WRAPPERS.popitem(last=False)
    _WRAPPERS[key] = wrapper
    return wrapper


def _row_coordinates(cu_seqlens_k: torch.Tensor, max_blocks: int, rows: int, block_size: int = BLOCK):
    """Request and block of every level-major CSR row (device only)."""
    device = cu_seqlens_k.device
    batch = cu_seqlens_k.numel() - 1
    blocks = (cu_seqlens_k[1:] - cu_seqlens_k[:-1] + block_size - 1) // block_size
    levels = torch.arange(max_blocks, device=device)
    active = (levels[:, None] < blocks[None, :]).reshape(-1)  # [L, B] level-major
    row_id = torch.cumsum(active.to(torch.int64), 0) - 1
    target = torch.where(active, row_id, torch.full_like(row_id, rows))
    request = torch.zeros(rows + 1, dtype=torch.int32, device=device)
    block = torch.zeros(rows + 1, dtype=torch.int32, device=device)
    request.scatter_(0, target, torch.arange(batch, device=device, dtype=torch.int32).repeat(max_blocks))
    block.scatter_(0, target, levels.to(torch.int32).repeat_interleave(batch))
    row_map = torch.where(active, row_id, -1).view(max_blocks, batch).t()
    return request[:rows].contiguous(), block[:rows].contiguous(), row_map


def build_k2q_csr(
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    kv_block_size: int,
    *,
    total_k: int | None = None,
    max_seqlen_k: int | None = None,
    max_seqlen_q: int | None = None,
    total_rows: int | None = None,
    qhead_per_kv: int = 8,
    return_schedule: bool = False,
    **_unused,
):
    """SM90 replacement of MSA ``build_k2q_csr`` with an optional owned schedule.

    SM100-only builder options (LPT scheduling, fused local Top-16, scratch
    buffers) have no SM90 meaning and are accepted but ignored.
    """
    if total_k is None:
        raise ValueError("build_k2q_csr requires total_k from k.shape[0]")
    if kv_block_size not in SUPPORTED_BLOCKS:
        raise ValueError(f"kv_block_size must be in {SUPPORTED_BLOCKS}")
    if q2k_indices.dtype != torch.int32 or q2k_indices.ndim != 3:
        raise ValueError("q2k_indices must be int32 [head_kv, total_q, topK]")
    if not q2k_indices.is_contiguous():
        raise ValueError("q2k_indices must be contiguous")
    for name, t in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k)):
        if t.dtype != torch.int32 or t.ndim != 1 or not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous int32 [B + 1]")
    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same shape [B + 1]")
    heads, total_q, topk = q2k_indices.shape
    device = q2k_indices.device
    max_blocks = -(-(max_seqlen_k if max_seqlen_k is not None else total_k) // kv_block_size)
    if total_rows is None:
        lengths = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        total_rows = int(((lengths + kv_block_size - 1) // kv_block_size).sum())
    _, _, row_map = _row_coordinates(cu_seqlens_k, max_blocks, total_rows, kv_block_size)
    positions = torch.arange(total_q, device=device, dtype=torch.int32)
    request = torch.searchsorted(cu_seqlens_q[1:], positions, right=True)
    q_local = positions - cu_seqlens_q[request]
    blocks = q2k_indices.long()
    valid = (blocks >= 0) & (blocks < max_blocks)
    rows = row_map[request[None, :, None].expand_as(blocks), blocks.clamp(0, max_blocks - 1)]
    rows = torch.where(valid & (rows >= 0), rows, total_rows).view(heads, -1)
    entries = total_q * topk
    order = torch.sort(rows * entries + torch.arange(entries, device=device), dim=1).indices
    counts = torch.zeros((heads, total_rows + 1), dtype=torch.int64, device=device)
    counts.scatter_add_(1, rows, torch.ones_like(rows))
    row_ptr = torch.zeros((heads, total_rows + 1), dtype=torch.int32, device=device)
    row_ptr[:, 1:] = torch.cumsum(counts[:, :total_rows], 1)
    q_indices = q_local[order // topk].to(torch.int32).contiguous()
    if return_schedule:
        return row_ptr, q_indices, SparseAttentionScheduleSm90(
            heads, total_q, topk, total_rows, qhead_per_kv, kv_block_size, device
        )
    return row_ptr, q_indices


@dataclass
class SparseAttentionScheduleSm90:
    """Caller-owned Hopper schedule and split-phase producer state.

    ``workspace`` uses the MSA public [topK, Q, Hq, D] layout. The schedule
    retains the native KV-centric relation ordering between MSA and COMBINE.
    """
    heads: int
    total_q: int
    topk: int
    rows: int
    group: int
    block_size: int
    device: torch.device
    enabled: bool = True
    state: dict = field(default_factory=dict, repr=False)


def _workspace(device, heads, total_q, topk, group, rows, partial_dtype, block_size=BLOCK, owner=None):
    key = (device, heads, total_q, topk, group, rows, partial_dtype, block_size)
    cache = _WORKSPACES if owner is None else owner.state
    ws = cache.get(key)
    if ws is None:
        relations = heads * total_q * topk
        segments = heads * rows
        ws = dict(
            counts=torch.zeros(segments, dtype=torch.int32, device=device),
            segment_start=torch.empty(segments, dtype=torch.int32, device=device),
            slot_counts=torch.zeros(heads * total_q, dtype=torch.int32, device=device),
            relations=torch.zeros(relations, dtype=torch.int32, device=device),
            route=torch.full((heads, total_q, topk), -1, dtype=torch.int32, device=device),
            work=torch.zeros(
                (-(-relations // DynamicAttentionSm90.max_queries_per_work) + segments, 5),
                dtype=torch.int32,
                device=device,
            ),
            work_count=torch.zeros(1, dtype=torch.int32, device=device),
            partial_out=torch.empty((relations, group, HEAD_DIM), dtype=partial_dtype, device=device),
            partial_lse=torch.empty((relations, group), dtype=torch.float32, device=device),
            empty_table=torch.zeros((1, 1), dtype=torch.int32, device=device),
        )
        cache[key] = ws
    return ws


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
    softmax_scale: float | None = None,
    lse_temperature_scale: float = 1.0,
    return_temperature_lse: bool = False,
    partial_dtype: torch.dtype = torch.bfloat16,
    return_softmax_lse: bool = False,
    page_table: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    schedule=None,
    usable_SM_count: int = -1,
    qk_dtype: torch.dtype | None = None,
    pv_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
    workspace=None,
    initial_out: torch.Tensor | None = None,
    initial_lse: torch.Tensor | None = None,
    safe_fwd_sync: bool = True,
    shared_work_valid: bool = True,
    safe_partial_v_tail: bool = True,
    full_score_tiles: bool | None = None,
    kernel_cfg=None,
    phase=None,
    last_cta_finalize: bool = False,
):
    """SM90 ``sparse_atten_func``: CSR block-sparse varlen attention.

    BF16 Q/K/V with GQA, packed/paged tails, caller workspace, split phases,
    initial state and temperature LSE. Hardware tuning hints retain Hopper's
    own schedule. Temperature LSE uses a second native attention pass when
    temperature differs from one; the standard output is preserved.
    """
    phase_value = getattr(phase, "value", phase) if phase is not None else "full"
    if phase_value not in ("full", "msa", "combine"):
        raise TypeError(f"invalid SparseForwardPhase {phase!r}")
    if phase_value != "full" and (workspace is None or schedule is None):
        raise ValueError("split MSA/combine phases require caller-owned workspace and schedule")
    if phase_value != "full" and return_temperature_lse:
        raise NotImplementedError("split MSA/combine phases do not retain temperature-LSE partials")
    temperature = float(lse_temperature_scale)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("lse_temperature_scale must be finite and > 0")
    if return_temperature_lse and not return_softmax_lse:
        raise ValueError("return_temperature_lse=True requires return_softmax_lse=True")
    if any(dtype not in (None, torch.bfloat16) for dtype in (qk_dtype, pv_dtype)):
        raise NotImplementedError("MSA sparse forward uses BF16 MMA operands")
    if blk_kv not in SUPPORTED_BLOCKS:
        raise ValueError(f"blk_kv must be in {SUPPORTED_BLOCKS}")
    if partial_dtype not in PARTIAL_DTYPES:
        raise TypeError(f"unsupported partial_dtype {partial_dtype}")
    if topK not in SUPPORTED_TOPK:
        raise ValueError(f"CSR sparse forward supports topK in {SUPPORTED_TOPK}, got {topK}")
    if q.ndim != 3 or q.dtype != torch.bfloat16 or q.shape[-1] != HEAD_DIM:
        raise ValueError("q must be BF16 [total_q, Hq, 128]")
    if k.dtype != q.dtype or v.dtype != q.dtype or k.shape != v.shape:
        raise ValueError("k and v must be BF16 tensors of the same shape")
    paged = page_table is not None
    if paged:
        if k.ndim != 4 or k.shape[2] != blk_kv or k.stride(-1) != 1 or k.stride() != v.stride():
            raise ValueError("paged k/v must be [pages, Hkv, blk_kv, 128] with matching strides")
        if page_table.dtype != torch.int32 or page_table.ndim != 2 or page_table.stride(-1) != 1:
            raise ValueError("page_table must be int32 [B, pages] with contiguous rows")
        heads = k.shape[1]
    else:
        if seqused_k is not None:
            raise ValueError("seqused_k is only supported together with page_table")
        if k.ndim != 3 or k.shape[1:] != (k.shape[1], HEAD_DIM):
            raise ValueError("k and v must be [total_k, Hkv, 128]")
        k, v = k.contiguous(), v.contiguous()
        heads = k.shape[1]
    q = q.contiguous()
    total_q, q_heads, _ = q.shape
    if q_heads % heads:
        raise ValueError("q.shape[1] must be divisible by Hkv")
    group = q_heads // heads
    if group not in SUPPORTED_GROUPS:
        raise NotImplementedError(
            f"the SM90 runtime supports qhead_per_kv in {SUPPORTED_GROUPS}, got {group}"
        )
    if (
        k2q_row_ptr.dtype != torch.int32
        or k2q_q_indices.dtype != torch.int32
        or k2q_row_ptr.ndim != 2
        or k2q_q_indices.ndim != 2
        or k2q_row_ptr.shape[0] != heads
        or k2q_q_indices.shape[0] != heads
    ):
        raise ValueError("CSR metadata must be int32 [Hkv, ...]")
    if k2q_q_indices.shape[1] < total_q * topK:
        raise ValueError("k2q_q_indices.shape[1] must be >= total_q * topK")
    if (initial_out is None) != (initial_lse is None):
        raise ValueError("initial_out and initial_lse must be provided together")
    if initial_out is not None:
        valid_shapes = ((total_q, q_heads, HEAD_DIM), (heads, total_q, group, HEAD_DIM))
        if initial_out.shape not in valid_shapes or initial_out.dtype != torch.bfloat16:
            raise ValueError("initial_out must be BF16 [Q,Hq,128] or [Hkv,Q,group,128]")
        if initial_lse.shape != (1, q_heads, total_q) or initial_lse.dtype != torch.float32:
            raise ValueError("initial_lse must be FP32 [1,Hq,Q]")
        if any(t.device != q.device or not t.is_contiguous() for t in (initial_out, initial_lse)):
            raise ValueError("initial state must be contiguous on the q device")
    if softmax_scale is None:
        softmax_scale = HEAD_DIM ** -0.5
    device = q.device
    row_ptr, q_indices = k2q_row_ptr.contiguous(), k2q_q_indices.contiguous()
    rows = row_ptr.shape[1] - 1
    batch = cu_seqlens_q.numel() - 1

    # Device-side request metadata: row coordinates, block starts, query limits.
    request, block, _ = _row_coordinates(cu_seqlens_k, -(-int(max_seqlen_k) // blk_kv), rows, blk_kv)
    block_token_start = (cu_seqlens_k[request] + block * blk_kv).to(torch.int32)
    q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    kv_lens = seqused_k if seqused_k is not None else cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    positions = torch.arange(total_q, device=device, dtype=torch.int32)
    query_request = torch.searchsorted(cu_seqlens_q[1:], positions, right=True).clamp_(max=batch - 1)
    q_local = positions - cu_seqlens_q[query_request]
    last_token = kv_lens[query_request] - 1
    limit = last_token
    if causal:
        limit = torch.minimum(last_token, q_local + kv_lens[query_request] - q_lens[query_request])
    query_limit = limit.to(torch.int32).contiguous()

    # A split schedule owns native state; unrelated calls cannot overwrite it.
    if schedule is not None:
        if not isinstance(schedule, SparseAttentionScheduleSm90):
            raise TypeError("SM90 requires SparseAttentionScheduleSm90")
        expected = (heads, total_q, topK, rows, group, blk_kv, device)
        actual = (schedule.heads, schedule.total_q, schedule.topk, schedule.rows,
                  schedule.group, schedule.block_size, schedule.device)
        if actual != expected:
            raise ValueError("schedule dimensions do not match sparse attention")
    ws = _workspace(device, heads, total_q, topK, group, rows, partial_dtype, blk_kv, schedule)
    table = page_table if paged else ws["empty_table"]
    output = torch.empty_like(q) if out is None else out
    lse = torch.empty((total_q, q_heads), dtype=torch.float32, device=device)
    if workspace is not None:
        if len(workspace) != 3:
            raise ValueError("workspace must contain O_partial, LSE_partial, LSE_out")
        po, pl, lse = workspace
        if (po.shape != (topK, total_q, q_heads, HEAD_DIM)
                or pl.shape != (topK, total_q, q_heads) or lse.shape != (total_q, q_heads)):
            raise ValueError("CSR workspace shapes do not match q")
        if po.dtype != partial_dtype or pl.dtype != torch.float32 or lse.dtype != torch.float32:
            raise ValueError("CSR workspace dtypes do not match the attention contract")
        if any(t.device != device or not t.is_contiguous() for t in workspace):
            raise ValueError("CSR workspace must be contiguous and on the q device")
    if (output.shape != q.shape or output.dtype != q.dtype or output.device != device
            or not output.is_contiguous()):
        raise ValueError("out must be contiguous and match q shape, dtype and device")
    if initial_out is not None:
        initial_view = (
            initial_out.permute(1, 0, 2, 3)
            if initial_out.ndim == 4
            else initial_out.view(total_q, heads, group, HEAD_DIM)
        )
        initial_lse_view = initial_lse[0].t()
    else:
        initial_view, initial_lse_view = output.view(total_q, heads, group, HEAD_DIM), lse
    fp8 = partial_dtype == torch.float8_e4m3fn
    partial = ws["partial_out"].view(torch.uint8) if fp8 else ws["partial_out"]
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    ctas = sms if usable_SM_count <= 0 else min(sms, int(usable_SM_count))
    blocks = Int32(rows)

    counts_args = (_cute(row_ptr, 4), _cute(ws["counts"], 4))
    scan_args = (
        _cute(ws["counts"], 4),
        _cute(ws["segment_start"], 4),
        _cute(block_token_start, 4),
        _cute(request, 4),
        _cute(block, 4),
        _cute(table, 4),
        _cute(ws["work"], 4),
        _cute(ws["work_count"], 4),
        blocks,
    )
    relation_args = (
        _cute(row_ptr, 4),
        _cute(q_indices, 4),
        _cute(request, 4),
        _cute(block, 4),
        _cute(cu_seqlens_q, 4),
        _cute(ws["segment_start"], 4),
        _cute(ws["slot_counts"], 4),
        _cute(ws["relations"], 4),
        _cute(ws["route"], 4),
    )
    dynamic_args = (
        _cute(q),
        _cute(k),
        _cute(v),
        _cute(ws["work"], 4),
        _cute(ws["work_count"], 4),
        _cute(ws["relations"], 4),
        _cute(partial),
        _cute(ws["partial_lse"]),
        _cute(query_limit, 4),
        Float32(softmax_scale),
    )
    combine_args = (
        _cute(initial_view),
        _cute(initial_lse_view, 4),
        _cute(partial),
        _cute(ws["partial_lse"]),
        _cute(ws["route"], 4),
        _cute(output),
        _cute(lse),
    )
    has_initial = initial_out is not None
    launches = (
        (("msa-csr-counts",) + _signature(row_ptr, ws["counts"]), CsrCountsSm90, counts_args),
        (
            ("scan", paged, blk_kv) + _signature(ws["counts"], ws["work"], request, table),
            lambda: ScheduleScanSm90(
                work_queries=DynamicAttentionSm90.max_queries_per_work, paged=paged, block_size=blk_kv
            ),
            scan_args,
        ),
        (
            ("msa-csr-relations", topK) + _signature(row_ptr, q_indices, ws["route"], cu_seqlens_q),
            lambda: CsrRelationsSm90(topk=topK),
            relation_args,
        ),
        (
            ("msa-dynamic", ctas, paged, group, topK, partial_dtype, blk_kv)
            + _signature(q, k, v, ws["work"], ws["relations"], partial),
            lambda: DynamicAttentionSm90(
                ctas=ctas,
                paged=paged,
                fp8_partial=fp8,
                group=group,
                topk=topK,
                masked=True,
                q_stages=1 if partial_dtype == torch.float32 else 2,
                block_size=blk_kv,
            ),
            dynamic_args,
        ),
        (
            ("msa-combine", group, topK, has_initial, partial_dtype)
            + _signature(initial_view, initial_lse_view, partial, output),
            lambda: CombineSm90(fp8_partial=fp8, group=group, topk=topK, has_initial=has_initial),
            combine_args,
        ),
    )
    stream = _stream()
    state_key = ("producer", partial_dtype, blk_kv)
    if phase_value == "combine":
        if schedule.state.get(state_key) != (row_ptr.data_ptr(), q_indices.data_ptr(), workspace[0].data_ptr()):
            raise ValueError("COMBINE requires a matching completed MSA phase")
        ws["partial_out"].copy_(workspace[0].view(topK, total_q, heads, group, HEAD_DIM).permute(2, 1, 0, 3, 4).reshape_as(ws["partial_out"]))
        ws["partial_lse"].copy_(workspace[1].view(topK, total_q, heads, group).permute(2, 1, 0, 3).reshape_as(ws["partial_lse"]))
    else:
        ws["route"].fill_(-1)
        ws["slot_counts"].zero_()
        ws["partial_out"].zero_()
        ws["partial_lse"].fill_(float("-inf"))
        for key, factory, args in launches[:-1]:
            _compile(key, factory, args)(*args, stream)
        if workspace is not None:
            # Recreate the views after producer launches; reshape may allocate.
            workspace[0].copy_(ws["partial_out"].view(heads, total_q, topK, group, HEAD_DIM).permute(2, 1, 0, 3, 4).reshape(topK, total_q, q_heads, HEAD_DIM))
            workspace[1].copy_(ws["partial_lse"].view(heads, total_q, topK, group).permute(2, 1, 0, 3).reshape(topK, total_q, q_heads))
        if schedule is not None:
            schedule.state[state_key] = (row_ptr.data_ptr(), q_indices.data_ptr(), workspace[0].data_ptr() if workspace is not None else None)
    if phase_value != "msa":
        key, factory, args = launches[-1]
        _compile(key, factory, args)(*args, stream)
    if return_temperature_lse:
        temperature_lse = lse
        if temperature != 1.0:
            temperature_lse = torch.empty_like(lse)
            scratch_out = torch.empty_like(output)
            key, factory, args = launches[-2]
            temp_args = args[:-1] + (Float32(softmax_scale / temperature),)
            _compile(key, factory, temp_args)(*temp_args, stream)
            # Upstream temperature LSE reduces sparse partials only; its
            # temperature=1 fast path aliases the standard merged LSE.
            _, _, args = launches[-1]
            temp_combine = args[:-2] + (_cute(scratch_out), _cute(temperature_lse))
            key = ("msa-temperature-combine", group, topK, partial_dtype) + _signature(partial, scratch_out)
            factory = lambda: CombineSm90(fp8_partial=fp8, group=group, topk=topK, has_initial=False)
            _compile(key, factory, temp_combine)(*temp_combine, stream)
        return output, lse, temperature_lse
    if return_softmax_lse:
        return output, lse
    return output


__all__ = ["build_k2q_csr", "sparse_atten_func", "SparseAttentionScheduleSm90"]
