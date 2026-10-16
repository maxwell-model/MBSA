# Copyright 2026 Maxwell Team.
# SPDX-License-Identifier: Apache-2.0
"""Prepared, separately executable public Fireworks KV-outer stages.

Kernels retain their public source and licenses; this module binds their launch
arguments once, outside graph capture. Route contents may change in place before
``run_index`` / graph replay. Fixed attention remains a separate MBSA branch.
"""
from __future__ import annotations

import math

import torch


class PreparedKVOuter:
    """Bind reusable index, schedule, raw partial and combine buffers.

    Q is ``[Tq,Hq,128]``. Cache is ``[pages,Hkv,page_size,128]``; selected
    blocks are ``[Tq,Hkv,topK]`` int32 with -1 padding. ``lse`` is head-major
    ``[Hq,Tq]`` float32. Upstream supports BF16/FP16/FP32 partials; FP8
    partials use normalized, scaled outputs with FP32 mass and scale metadata.

    In FP8 mode the FP32 stats carry log2 mass and output scale; other partial
    types retain the upstream raw numerator, row maximum and row sum ABI.

    Preparation executes each stage once to compile and bind launchers. Supply
    initialized tensors; no host reads of query/route contents are performed.
    """

    def __init__(
        self, q, k_cache, v_cache, selected, block_tables, cu_seqlens_q,
        used_kv_lens, *, causal=True, softmax_scale=None, out=None, lse=None,
        partial_dtype=torch.bfloat16, num_splits=None, q_load_stage=None,
        warmup=True,
    ):
        tq, hq, dim = q.shape
        pages, hkv, page_size, kv_dim = k_cache.shape
        if dim != 128 or kv_dim != 128 or page_size not in (64, 128):
            raise ValueError("public Fireworks KV-outer requires D=128 and page_size=64/128")
        if k_cache.shape != v_cache.shape or q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
            raise ValueError("Q/K/V must have matching dtype and K/V cache shapes")
        if q.dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2):
            raise ValueError("public Fireworks KV-outer supports BF16/FP16/FP8 Q/K/V")
        if partial_dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float8_e4m3fn):
            raise ValueError("partial storage supports BF16/FP16/FP32/FP8 E4M3FN")
        if selected.ndim != 3 or selected.shape[:2] != (tq, hkv) or selected.dtype != torch.int32:
            raise ValueError("selected must be [Tq,Hkv,topK] int32")
        if tq < 1 or hkv < 1 or hq % hkv != 0 or selected.shape[-1] < 1:
            raise ValueError("Tq/Hkv/topK must be positive and Hq divisible by Hkv")
        if block_tables.ndim != 2 or block_tables.dtype != torch.int32 or not block_tables.is_contiguous():
            raise ValueError("block_tables must be contiguous [B,max_pages] int32")
        if cu_seqlens_q.dtype != torch.int64 or not cu_seqlens_q.is_contiguous():
            raise ValueError("cu_seqlens_q must be contiguous int64")
        if used_kv_lens.dtype != torch.int32 or not used_kv_lens.is_contiguous():
            raise ValueError("used_kv_lens must be contiguous int32")
        if not q.is_contiguous():
            raise ValueError("q must be contiguous")
        if selected.stride(-1) != 1 or any(stride <= 0 for stride in selected.stride()):
            raise ValueError("selected requires contiguous rank and positive outer strides")
        for cache in (k_cache, v_cache):
            if cache.stride(-1) != 1 or any(
                stride <= 0 or (size > 1 and stride * cache.element_size() % 16)
                for size, stride in zip(cache.shape[:-1], cache.stride()[:-1])
            ):
                raise ValueError("cache requires contiguous D and positive 16-byte-aligned outer strides")
        n_batches = cu_seqlens_q.numel() - 1
        if n_batches < 1 or block_tables.shape[1] < 1 or n_batches != block_tables.shape[0] or used_kv_lens.shape != (n_batches,):
            raise ValueError("batch metadata shapes do not match")
        if q.device.type != "cuda" or any(t.device != q.device for t in (
            k_cache, v_cache, selected, block_tables, cu_seqlens_q, used_kv_lens,
        )):
            raise ValueError("all input tensors must be on the same CUDA device")
        from . import build_kvouter_index as index
        from .sparse_fwd_kvouter import _cached_sm_count

        self.q = q
        self.k_cache, self.v_cache = k_cache, v_cache
        self.selected, self.block_tables = selected, block_tables
        self.cu_seqlens_q, self.used_kv_lens = cu_seqlens_q, used_kv_lens
        self.tq, self.hq, self.hkv, self.topk = tq, hq, hkv, selected.shape[-1]
        self.page_size, self.block_size = page_size, 128
        self.ratio = 128 // page_size
        self.msb = max(1, (block_tables.shape[1] + self.ratio - 1) // self.ratio)
        self.num_block_slots = n_batches * self.msb
        self.n_batches = n_batches
        self.causal = bool(causal)
        self.softmax_scale = float(softmax_scale if softmax_scale is not None else 128 ** -0.5)
        self.partial_dtype = partial_dtype
        self.q_load_stage = q_load_stage
        self.num_splits = int(num_splits if num_splits is not None else _cached_sm_count(q.device.index))
        if self.num_splits < 1:
            raise ValueError("num_splits must be positive")
        self.replicas = index._adaptive_replicas(tq * hkv * self.topk, hkv * self.num_block_slots)
        self.out = out if out is not None else torch.empty_like(q, dtype=torch.bfloat16)
        self.lse = lse if lse is not None else torch.empty(hq, tq, dtype=torch.float32, device=q.device)
        if self.out.shape != q.shape or self.out.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("out must be [Tq,Hq,D] BF16/FP16/FP32")
        if not self.out.is_contiguous() or self.out.device != q.device:
            raise ValueError("out must be contiguous and on Q's device")
        if self.lse.shape != (hq, tq) or self.lse.dtype != torch.float32:
            raise ValueError("lse must be [Hq,Tq] float32")
        if not self.lse.is_contiguous() or self.lse.device != q.device:
            raise ValueError("lse must be contiguous and on Q's device")
        nbs, r, topk = self.num_block_slots, self.replicas, self.topk
        shapes = {
            "count": ((hkv, nbs, r), torch.int32),
            "count_total": ((hkv, nbs), torch.int32),
            "edge_local": ((tq * hkv * topk,), torch.int32),
            "topk_slot_ids": ((hkv, nbs * self.ratio), torch.int64),
            "kv_to_q_offsets": ((hkv, nbs + 1), torch.int32),
            "sel_slots": ((hkv, nbs), torch.int32),
            "sel_offsets": ((hkv, nbs + 1), torch.int32),
            "num_sel": ((hkv,), torch.int32),
            # Retain the public builder's capacity contract per head.
            "kv_to_q_indices_and_ranks": ((hkv, tq * hkv * topk, 2), torch.int32),
            "inv": ((hkv, tq, topk), torch.int32),
            "o_flat": ((tq * hq * topk, dim), partial_dtype),
            "m_partial": ((hkv, tq * topk * (hq // hkv)), torch.float32),
            "l_partial": ((hkv, tq * topk * (hq // hkv)), torch.float32),
        }
        self.buffers = {
            name: torch.empty(shape, dtype=dtype, device=q.device)
            for name, (shape, dtype) in shapes.items()
        }
        self.work_start = torch.empty(self.num_splits, 3, dtype=torch.int32, device=q.device)
        self.work_end = torch.empty_like(self.work_start)
        self.nqps = max(1, (hkv * tq * topk + self.num_splits - 1) // self.num_splits)
        self.schedule = (self.work_start, self.work_end, self.num_splits, self.nqps)
        self.index_launches = self.forward_launches = self.combine_launches = None
        self._ready = False
        if warmup:
            self.warmup()

    def warmup(self):
        """Compile and bind every stage. Call before graph capture."""
        from . import build_kvouter_index as index
        from . import sparse_fwd_kvouter as fwd
        from . import sparse_fwd_kvouter_combine as combine
        from . import sparse_fwd_kvouter_load_balance_schedule as schedule

        b = self.buffers
        index._build_kvouter_index_cute(
            self.selected, hkv=self.hkv, topk=self.topk,
            num_block_slots=self.num_block_slots, block_size=128, page_size=self.page_size,
            cu_seqlens_q=self.cu_seqlens_q, causal=self.causal,
            used_kv_lens=self.used_kv_lens, block_tables=self.block_tables,
            msb=self.msb, buffers=b,
        )
        schedule.build_load_balanced_schedule(
            b["sel_offsets"], total_q=self.tq, num_splits=self.num_splits,
            topk=self.topk, buffers=(self.work_start, self.work_end),
        )
        fwd.indexed_block_partials(
            self.q, self.k_cache, self.v_cache, b["topk_slot_ids"],
            b["kv_to_q_offsets"], b["kv_to_q_indices_and_ranks"], topk=self.topk,
            block_size=128, page_size=self.page_size, softmax_scale=self.softmax_scale,
            causal=self.causal, cu_seqlens_q=self.cu_seqlens_q,
            used_kv_lens=self.used_kv_lens, partial_dtype=self.partial_dtype,
            inv=b["inv"], num_splits=self.num_splits, q_load_stage=self.q_load_stage,
            sel_slots=b["sel_slots"], sel_offsets=b["sel_offsets"],
            num_sel=b["num_sel"], buffers=b, schedule=self.schedule,
        )
        combine.merge_kv_partials(
            b["o_flat"], b["m_partial"], b["l_partial"], b["inv"],
            out_dtype=self.out.dtype, return_lse=True, out=self.out, lse=self.lse,
        )
        self._bind_launches(index, fwd, combine, schedule)
        self._ready = True
        return self

    def _bind_launches(self, index, fwd, combine, schedule):
        from cutlass import Int32, Int64
        from flash_attn.cute.cute_dsl_utils import to_cute_tensor

        b = self.buffers
        hkv, nbs, topk, r = self.hkv, self.num_block_slots, self.topk, self.replicas
        cap, nb, msb = self.tq * hkv * topk, self.n_batches, self.msb
        key = (hkv, topk, 128, self.causal, True, self.ratio, r)
        parallel = nbs > index._COUNT_TO_OFFSETS_PARALLEL_THRESHOLD
        off_args = (b["count_total"], b["kv_to_q_offsets"], b["sel_slots"],
                    b["sel_offsets"], b["num_sel"], nbs)
        if parallel:
            off_args += ((nbs + 1 + 255) // 256,)
        self.index_launches = [
            (index._init_slots_compile_cache[(hkv, self.ratio, self.page_size, True, r)],
             (self.selected, self.block_tables, b["count"], b["topk_slot_ids"],
              nbs, msb, self.block_tables.shape[1])),
            (index._count_edges_compile_cache[key],
             (self.selected, self.cu_seqlens_q, self.used_kv_lens, b["topk_slot_ids"],
              b["count"], b["edge_local"], cap, msb, nb)),
            (index._reduce_replicas_compile_cache[(r,)],
             (b["count"], b["count_total"], hkv * nbs, nbs)),
            (index._offsets_compile_cache[(hkv, parallel)], off_args),
            (index._scatter_compile_cache[key],
             (b["edge_local"], self.selected, self.cu_seqlens_q, self.used_kv_lens,
              b["topk_slot_ids"], b["kv_to_q_offsets"], b["count"],
              b["kv_to_q_indices_and_ranks"], b["inv"], cap, msb, nb)),
            (schedule._compile_cache[(hkv, self.num_splits)],
             (b["sel_offsets"], self.work_start, self.work_end, Int32(nbs), Int64(self.nqps))),
        ]
        qls, obuf = fwd._arch_defaults(self.q.dtype, self.partial_dtype)
        qls = self.q_load_stage if self.q_load_stage is not None else qls
        q_dtype = to_cute_tensor(self.q, leading_dim=2).element_type
        o_dtype = to_cute_tensor(b["o_flat"], leading_dim=1).element_type
        forward_key = (self.hq // hkv, hkv, self.page_size, self.causal,
                       qls, obuf, q_dtype, o_dtype)
        self._q_is_fp8 = self.q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        k, v = self.k_cache.permute(0, 2, 1, 3), self.v_cache.permute(0, 2, 1, 3)
        q = self.q
        if self._q_is_fp8:
            q, k, v = q.view(torch.uint8), k.view(torch.uint8), v.view(torch.uint8)
        self.forward_launches = [(fwd._compile_cache[forward_key], (
            q, k, v, b["o_flat"].view(torch.uint8) if self.partial_dtype == torch.float8_e4m3fn else b["o_flat"], b["m_partial"], b["l_partial"],
            b["topk_slot_ids"], b["sel_offsets"], b["kv_to_q_indices_and_ranks"],
            self.work_start, self.work_end, b["sel_slots"], b["num_sel"],
            self.num_splits, self.cu_seqlens_q, self.used_kv_lens, nb,
            self.softmax_scale, b["o_flat"].view(torch.uint8) if self.partial_dtype == torch.float8_e4m3fn else b["o_flat"],
        ))]
        combine_key = (self.out.dtype, self.partial_dtype, 128,
                       max(math.ceil(math.log2(max(topk, 2))), 5), True, True, True)
        self.combine_launches = [(combine._compile_cache[combine_key], (
            b["o_flat"].view(torch.uint8) if self.partial_dtype == torch.float8_e4m3fn else b["o_flat"], b["m_partial"].reshape(-1), self.out.unsqueeze(0),
            b["l_partial"].reshape(-1), b["inv"], self.lse.unsqueeze(0),
            None, None, None, None, None,
        ))]

    def _launch(self, launches):
        if not self._ready:
            raise RuntimeError("call warmup before running prepared Fireworks stages")
        for fn, args in launches:
            fn(*args)

    def run_index(self):
        """Rebuild CSR, inverse map and schedule from current route contents."""
        self._launch(self.index_launches)

    def rebind(self, q, k_cache, v_cache, block_tables):
        """Bind live inputs with identical metadata, without copies or compilation.

        Existing CUDA graphs retain the addresses captured earlier; recapture a
        graph after rebinding. Eager prepared launches use the new addresses.
        """
        if not self._ready:
            raise RuntimeError("call warmup before rebinding prepared Fireworks stages")
        for name, old, new in (
            ("Q", self.q, q), ("K", self.k_cache, k_cache),
            ("V", self.v_cache, v_cache), ("block_tables", self.block_tables, block_tables),
        ):
            if (new.shape != old.shape or new.stride() != old.stride()
                    or new.dtype != old.dtype or new.device != old.device):
                raise ValueError(f"{name} must retain prepared shape, strides, dtype and device")
        self.q, self.k_cache, self.v_cache, self.block_tables = q, k_cache, v_cache, block_tables
        fn, args = self.index_launches[0]
        self.index_launches[0] = (fn, (args[0], block_tables, *args[2:]))
        k, v = k_cache.permute(0, 2, 1, 3), v_cache.permute(0, 2, 1, 3)
        q_rt = q
        if self._q_is_fp8:
            q_rt, k, v = q.view(torch.uint8), k.view(torch.uint8), v.view(torch.uint8)
        fn, args = self.forward_launches[0]
        self.forward_launches[0] = (fn, (q_rt, k, v, *args[3:]))
        return self

    def run_forward(self, q=None):
        """Write raw partials; never rebuild routes or run combine."""
        if q is None or q is self.q:
            return self._launch(self.forward_launches)
        if not self._ready:
            raise RuntimeError("call warmup before running prepared Fireworks stages")
        if q.shape != self.q.shape or q.dtype != self.q.dtype or q.device != self.q.device or not q.is_contiguous():
            raise ValueError("replacement Q must match prepared shape, dtype, device and strides")
        fn, args = self.forward_launches[0]
        q_rt = q.view(torch.uint8) if self._q_is_fp8 else q
        fn(q_rt, *args[1:])

    def run_combine(self):
        """Consume existing partials, write caller-owned output and natural-log LSE."""
        self._launch(self.combine_launches)
        return self.out, self.lse

    def run(self, q=None):
        self.run_index()
        self.run_forward(q)
        return self.run_combine()
