# Copyright 2026 MBSA contributors.
# SPDX-License-Identifier: Apache-2.0
"""Fixed Prefix8 union causal Local8 using the public FA4 Q-outer kernel.

This adapter calls flash-attn-4==4.0.0b20; it contains no vendored kernel body.
The separately attributed public FA4 kernel override fixes TMEM synchronization
and retains its upstream BSD-3-Clause license. Each query tile performs online
softmax across its selected blocks and writes one output, avoiding a 16-rank
raw partial tensor. Fixed and Dynamic outputs remain independent.

Preparation creates compact immutable block metadata on the host, then warms
the public kernel. Run accepts live Q/K/V addresses via ``rebind``. Batch
requests execute separately; this first backend prioritizes long prefill.
"""
from __future__ import annotations

from itertools import accumulate
from importlib.util import find_spec

import torch


def fixed_tile_metadata(q_len, kv_len, *, query_tile=256):
    """Return masked/full block rows for the exact Prefix8 union Local8 mask.

    Metadata refers to logical KV blocks. Full blocks are visible to every real
    query in the tile. Boundary blocks use a token mask in FA4. Ascending unique
    IDs avoid double counting the overlap between prefix and local windows.
    """
    if q_len <= 0 or kv_len < q_len or query_tile <= 0:
        raise ValueError("require 0 < Q <= KV and positive query_tile")
    masked, full = [], []
    for begin in range(0, q_len, query_tile):
        first = kv_len - q_len + begin
        last = kv_len - q_len + min(begin + query_tile, q_len) - 1
        first_block, last_block = first // 128, last // 128
        candidates = sorted(set(range(min(8, last_block + 1))) |
                            set(range(max(8, first_block - 7), last_block + 1)))
        full_row, masked_row = [], []
        for block in candidates:
            fully_visible = ((block + 1) * 128 - 1 <= first and
                             (block < 8 or block >= last_block - 7))
            (full_row if fully_visible else masked_row).append(block)
        masked.append(masked_row)
        full.append(full_row)
    return masked, full


_compiled_masks = {}
_static_identity_masks = {}
_paged_vector_mask = None


def _make_cute_mask(identity_table=False):
    """FA4 scalar mask callbacks use length-one TensorSSA coordinates.

    Convert those coordinates with the public FA4 helpers before indexing a
    CuTe Tensor, then return its expected length-one Boolean TensorSSA. No
    vectorization attribute is set, so FA4 selects its scalar callback path.
    """
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute.utils import scalar_to_ssa, ssa_to_scalar

    @cute.jit
    def mask(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        query = ssa_to_scalar(q_idx)
        key = ssa_to_scalar(kv_idx)
        logical_key = key
        if cutlass.const_expr(not identity_table):
            logical_key = aux_tensors[0][key // 128] * 128 + key % 128
        logical_length = aux_tensors[1][0]
        absolute_query = query + logical_length - seqlen_info.seqlen_q
        keep = ((logical_key <= absolute_query) & (logical_key < logical_length) &
                ((logical_key // 128 < 8) | (logical_key // 128 >= absolute_query // 128 - 7)))
        return scalar_to_ssa(keep, cutlass.Boolean)

    return mask


def _make_static_identity_mask(query_len, kv_len):
    """Original callback for public FA4 b20's packed Uint32 mask interface.

    The public callback broadcasts one logical Q index to 32 KV lanes. Bit i
    keeps lane i. Geometry is immutable; no inverse or length tensor is read.
    """
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute.utils import scalar_to_ssa

    offset = kv_len - query_len

    @cute.jit
    def mask(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        absolute = q_idx[0] + offset
        local_begin = absolute // 128 - 7
        word = cutlass.Uint32(0)
        for i in cutlass.range_constexpr(32):
            key = kv_idx[i]
            keep = ((key <= absolute) & (key < kv_len) &
                    ((key // 128 < 8) | (key // 128 >= local_begin)))
            word = word | ((cutlass.Uint32(1) << i) if keep else cutlass.Uint32(0))
        return scalar_to_ssa(word, cutlass.Uint32)

    mask.__vec_size__ = 32
    return mask


def _make_paged_vector_mask():
    """Mask 32 physical KV lanes using the live inverse page map."""
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute.utils import scalar_to_ssa

    @cute.jit
    def mask(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        logical_length = aux_tensors[1][0]
        absolute = q_idx[0] + logical_length - seqlen_info.seqlen_q
        local_begin = absolute // 128 - 7
        # A sparse KV tile is one aligned 128-token physical page.
        logical_page = aux_tensors[0][kv_idx[0] // 128]
        word = cutlass.Uint32(0)
        for i in cutlass.range_constexpr(32):
            physical_key = kv_idx[i]
            logical_key = logical_page * 128 + physical_key % 128
            keep = ((logical_key <= absolute) & (logical_key < logical_length) &
                    ((logical_key // 128 < 8) | (logical_key // 128 >= local_begin)))
            word = word | ((cutlass.Uint32(1) << i) if keep else cutlass.Uint32(0))
        return scalar_to_ssa(word, cutlass.Uint32)

    mask.__vec_size__ = 32
    return mask


class PreparedFixedAttention:
    """Public FA4 BF16/FP8 Fixed branch over packed Q and a paged cache.

    Q/out: [Tq,Hq,D], K/V: [pages,Hkv,128,D], LSE: [Hq,Tq].
    ``request_lens`` must describe the immutable prepared query/cache lengths.
    Tables and cache values may change in place; page strides are preserved.
    Partial storage dtype has no effect because FA4 writes the final BF16 O.
    Tail pages are staged with invalid rows zeroed before FA4 reads them.
    Aligned caches with unique pages and foldable page/token axes are read
    directly. Other layouts stage only the pages selected by Fixed. Public FA4 disallows
    block-sparse plus paged mode, so this adapter supplies physical-flat K/V,
    physical block IDs, and an inverse page map for boundary token masks.
    ``immutable_table=True`` prepares those maps once and runs only forward.
    Use it only for a table whose values remain fixed, such as an internal
    identity table. Call ``refresh_table`` after explicit in-place changes;
    rebinding a different table address refreshes its maps automatically.
    ``identity_table=True`` requires immutable batch-one full identity pages
    and uses an arithmetic token mask without inverse-page memory reads.
    ``static_identity_mask=True`` additionally requires complete Q/KV blocks
    and fixed used-KV lengths matching ``request_lens`` for the instance's
    lifetime. It uses a 32-lane arithmetic mask without auxiliary tensors.
    Direct adapters default to the scalar mask with live metadata inputs.
    """

    def __init__(self, q, k_cache, v_cache, block_tables, used_kv_lens,
                 request_lens, *, out=None, lse=None, warmup=True, pack_gqa=False,
                 immutable_table=False, identity_table=False,
                 static_identity_mask=False, vectorized_paged_mask=False,
                 prepared_launch=False):
        from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
        from flash_attn.cute.interface import _flash_attn_fwd
        from .fixed_sync import clone_fixed_forward
        identity_table = bool(identity_table)
        if identity_table not in _compiled_masks:
            _compiled_masks[identity_table] = _make_cute_mask(identity_table)
        self._mask_mod = _compiled_masks[identity_table]
        lengths = tuple((int(nq), int(nk)) for nq, nk in request_lens)
        if (q.ndim != 3 or q.shape[-1] != 128 or q.dtype not in (torch.bfloat16, torch.float8_e4m3fn) or
                not q.is_cuda or not q.is_contiguous()):
            raise ValueError("Fixed FA4 requires contiguous CUDA BF16/E4M3FN Q [Tq,Hq,128]")
        if (k_cache.ndim != 4 or k_cache.shape != v_cache.shape or
                k_cache.shape[2:] != (128, 128) or q.shape[1] % k_cache.shape[1]):
            raise ValueError("Fixed FA4 requires K/V [pages,Hkv,128,128] and integral GQA")
        if any(t.device != q.device or t.dtype != q.dtype or t.stride(-1) != 1
               for t in (k_cache, v_cache)):
            raise ValueError("K/V must match Q device/dtype with contiguous D")
        foldable = all(t.shape[0] <= 1 or t.stride(0) == 128 * t.stride(2)
                       for t in (k_cache, v_cache))
        if (not lengths or sum(nq for nq, _ in lengths) != q.shape[0] or
                block_tables.shape[0] != len(lengths) or block_tables.ndim != 2 or
                block_tables.dtype != torch.int32 or not block_tables.is_contiguous() or
                used_kv_lens.shape != (len(lengths),) or used_kv_lens.dtype != torch.int32 or
                not used_kv_lens.is_contiguous() or
                block_tables.device != q.device or used_kv_lens.device != q.device):
            raise ValueError("request lengths and CUDA int32 page metadata must match packed Q")
        if any(nk > block_tables.shape[1] * 128 for _, nk in lengths):
            raise ValueError("page table is too short for request_lens")
        if static_identity_mask:
            if (not identity_table or not immutable_table or len(lengths) != 1 or
                    not 0 < lengths[0][0] <= lengths[0][1] or
                    lengths[0][0] % 128 or lengths[0][1] != k_cache.shape[0] * 128):
                raise ValueError("static_identity_mask requires immutable batch-one identity full Q/KV blocks")
            if tuple(used_kv_lens.tolist()) != (lengths[0][1],):
                raise ValueError("static_identity_mask requires fixed used-KV lengths matching request_lens")
        self.q, self.k_cache, self.v_cache = q, k_cache, v_cache
        self.block_tables, self.used_kv_lens = block_tables, used_kv_lens
        self.out = out if out is not None else torch.empty_like(q, dtype=torch.bfloat16)
        self.lse = lse if lse is not None else torch.empty(q.shape[1], q.shape[0],
                                                         dtype=torch.float32, device=q.device)
        if (self.out.shape != q.shape or self.out.dtype != torch.bfloat16 or self.out.device != q.device or
                not self.out.is_contiguous() or self.lse.shape != (q.shape[1], q.shape[0]) or
                self.lse.dtype != torch.float32 or self.lse.device != q.device or
                not self.lse.is_contiguous()):
            raise ValueError("out/LSE must match prepared BF16/FP32 output contract")
        self.request_lens = lengths
        self.pack_gqa = bool(pack_gqa)
        self.immutable_table = bool(immutable_table)
        self.identity_table = identity_table
        self.static_identity_mask = bool(static_identity_mask)
        self.prepared_launch = bool(prepared_launch)
        self._prepared_forward = None
        self._prepared_args = None
        self.vectorized_paged_mask = bool(vectorized_paged_mask)
        self._refresh_maps = None
        if self.vectorized_paged_mask:
            if identity_table or static_identity_mask:
                raise ValueError("vectorized_paged_mask requires a physical page map")
            global _paged_vector_mask
            if _paged_vector_mask is None:
                _paged_vector_mask = _make_paged_vector_mask()
            self._mask_mod = _paged_vector_mask
            if find_spec("triton") is not None:
                from .paged_fixed_maps import refresh_maps
                self._refresh_maps = refresh_maps
        if identity_table and (not immutable_table or len(lengths) != 1 or
                               block_tables.shape[1] != k_cache.shape[0] or
                               k_cache.shape[0] != (lengths[0][1] + 127) // 128):
            raise ValueError("identity_table requires immutable batch-one full identity pages")
        self._mapping_ready = False
        self.q_prefix = tuple(accumulate((nq for nq, _ in lengths), initial=0))
        self._forward = clone_fixed_forward(_flash_attn_fwd)
        self.metadata = []
        self.mappings = []
        self.staging = []
        for i, (nq, nk) in enumerate(lengths):
            n_pages = (nk + 127) // 128
            host_pages = block_tables[i, :n_pages].tolist()
            if any(page < 0 or page >= k_cache.shape[0] for page in host_pages):
                raise ValueError("Fixed page IDs must be in range")
            if identity_table and host_pages != list(range(n_pages)):
                raise ValueError("identity_table requires identity page IDs")
            q_stage = 2 if nq * (q.shape[1] // k_cache.shape[1] if pack_gqa else 1) > 128 else 1
            query_tile = 128 * q_stage
            masked, full = fixed_tile_metadata(nq, nk, query_tile=query_tile)
            staged = nk % 128 != 0 or not foldable or len(set(host_pages)) != len(host_pages)
            if staged:
                selected = sorted({block for row in masked + full for block in row})
                lookup = {block: j for j, block in enumerate(selected)}
                logical_pages = torch.tensor(selected, dtype=torch.int32, device=q.device)
                sk = torch.empty(len(selected), 128, k_cache.shape[1], 128,
                                 dtype=q.dtype, device=q.device).permute(0, 2, 1, 3)
                sv = torch.empty_like(sk.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
                self.staging.append((logical_pages, sk, sv))
                masked = [[lookup[x] for x in row] for row in masked]
                full = [[lookup[x] for x in row] for row in full]
            else:
                self.staging.append(None)
            width = max(1, max(map(len, masked)), max(map(len, full)))
            def stage(rows):
                counts = torch.tensor([len(row) for row in rows], dtype=torch.int32, device=q.device)
                ids = torch.tensor([row + [0] * (width - len(row)) for row in rows],
                                   dtype=torch.int32, device=q.device)
                return counts[None, None], ids[None, None]
            masked_count, masked_idx = stage(masked)
            full_count, full_idx = stage(full)
            physical_masked, physical_full = torch.empty_like(masked_idx), torch.empty_like(full_idx)
            self.metadata.append(BlockSparseTensorsTorch(
                masked_count, physical_masked, full_count, physical_full, block_size=(query_tile, 128)))
            self.mappings.append((
                masked_idx.reshape(-1).long(), full_idx.reshape(-1).long(),
                physical_masked.view(-1), physical_full.view(-1),
                torch.empty(k_cache.shape[0], dtype=torch.int32, device=q.device),
                torch.empty(n_pages, dtype=torch.int64, device=q.device),
                torch.arange(n_pages, dtype=torch.int32, device=q.device)))
            if staged:
                physical_masked.copy_(masked_idx)
                physical_full.copy_(full_idx)
                inverse = logical_pages.clone()
                self.mappings[-1] = (*self.mappings[-1][:4], inverse, *self.mappings[-1][5:])
        if self.static_identity_mask:
            geometry = lengths[0]
            if geometry not in _static_identity_masks:
                _static_identity_masks[geometry] = _make_static_identity_mask(*geometry)
            self._mask_mod = _static_identity_masks[geometry]
        self._ready = False
        if warmup:
            self.warmup()

    def _validate_table(self, table):
        for i, (_, nk) in enumerate(self.request_lens):
            pages = table[i, :(nk + 127) // 128].tolist()
            if (any(page < 0 or page >= self.k_cache.shape[0] for page in pages) or
                    (self.staging[i] is None and len(set(pages)) != len(pages))):
                raise ValueError("Fixed page IDs must be in range and direct pages must be unique")
            if self.identity_table and pages != list(range(len(pages))):
                raise ValueError("identity_table requires identity page IDs")

    def _refresh_mappings(self):
        for i, mapping in enumerate(self.mappings):
            if self.staging[i] is not None:
                continue
            if self._refresh_maps is not None:
                self._refresh_maps(self.block_tables[i], mapping)
                continue
            logical_masked, logical_full, physical_masked, physical_full, inverse, table64, positions = mapping
            table_row = self.block_tables[i]
            table64.copy_(table_row[:table64.numel()])
            inverse.scatter_(0, table64, positions)
            torch.index_select(table_row, 0, logical_masked, out=physical_masked)
            torch.index_select(table_row, 0, logical_full, out=physical_full)
        self._mapping_ready = True

    def refresh_table(self):
        """Validate and refresh mappings outside graph capture after table edits.

        Host validation is a preparation action. Previously captured immutable
        graphs read these same mapping buffers on their next replay.
        """
        self._validate_table(self.block_tables)
        self._refresh_mappings()
        return self

    def _run(self):
        for i, staging in enumerate(self.staging):
            if staging is not None:
                from .cache_pages import gather_pages
                logical, sk, sv = staging
                gather_pages(self.k_cache, self.v_cache, self.block_tables[i],
                             logical, sk, sv, self.request_lens[i][1])
        if not self.immutable_table or not self._mapping_ready:
            self._refresh_mappings()
        if self._prepared_forward is not None:
            self._prepared_forward(*self._prepared_args)
            return self.out, self.lse
        for i, ((nq, nk), metadata) in enumerate(zip(self.request_lens, self.metadata)):
            kc, vc = ((self.k_cache, self.v_cache) if self.staging[i] is None else self.staging[i][1:])
            k = kc.permute(0, 2, 1, 3).view(1, -1, kc.shape[1], 128)
            v = vc.permute(0, 2, 1, 3).view(1, -1, vc.shape[1], 128)
            begin, end = self.q_prefix[i:i + 2]
            inverse = self.mappings[i][4]
            self._forward(
                self.q[begin:end].unsqueeze(0), k, v,
                max_seqlen_q=nq, max_seqlen_k=k.shape[1],
                softmax_scale=128 ** -0.5, causal=False, num_splits=1, pack_gqa=self.pack_gqa,
                tile_mn=(128, 128), mask_mod=self._mask_mod, block_sparse_tensors=metadata,
                aux_tensors=(None if self.static_identity_mask else
                             [inverse, self.used_kv_lens[i:i + 1]]),
                return_lse=True, out=self.out[begin:end].unsqueeze(0),
                lse=self.lse[:, begin:end].unsqueeze(0))
        return self.out, self.lse

    def warmup(self):
        self._run()
        if self.prepared_launch and len(self.request_lens) == 1:
            self._bind_prepared_forward()
        self._ready = True
        return self

    def _bind_prepared_forward(self):
        """Bind the public compiled callable after its wrapper compiles the kernel."""
        from flash_attn.cute.block_sparsity import normalize_block_sparse_config
        from flash_attn.cute.utils import AuxData

        if len(self._forward.compile_cache) != 1:
            raise RuntimeError("prepared Fixed launch requires one compiled FA4 variant")
        nq, _ = self.request_lens[0]
        kc, vc = ((self.k_cache, self.v_cache) if self.staging[0] is None else self.staging[0][1:])
        k = kc.permute(0, 2, 1, 3).view(1, -1, kc.shape[1], 128)
        v = vc.permute(0, 2, 1, 3).view(1, -1, vc.shape[1], 128)
        metadata, _, _ = normalize_block_sparse_config(
            self.metadata[0], batch_size=1, num_head=self.q.shape[1],
            seqlen_q=nq, seqlen_k=k.shape[1], block_size=(128, 128),
            q_stage=(2 if nq * (self.q.shape[1] // kc.shape[1] if self.pack_gqa else 1) > 128 else 1),
        )
        sparse = tuple(getattr(metadata, name) for name in (
            "mask_block_cnt", "mask_block_idx", "full_block_cnt", "full_block_idx",
            "cu_total_m_blocks", "cu_block_idx_offsets", "dq_write_order", "dq_write_order_full",
        ))
        aux = None if self.static_identity_mask else [self.mappings[0][4], self.used_kv_lens]
        self._prepared_forward = next(iter(self._forward.compile_cache.values()))
        q = self.q.unsqueeze(0)
        if self.q.dtype == torch.float8_e4m3fn:
            q, k, v = (t.view(torch.uint8) for t in (q, k, v))
        self._prepared_args = (
            q, k, v, self.out.unsqueeze(0), self.lse.unsqueeze(0),
            128 ** -0.5, *([None] * 9), sparse, AuxData(aux, None),
        )

    def run(self):
        if not self._ready:
            raise RuntimeError("call warmup before executing Fixed FA4")
        return self._run()

    def rebind(self, q, k_cache, v_cache, block_tables):
        for name, old, new in (("Q", self.q, q), ("K", self.k_cache, k_cache),
                               ("V", self.v_cache, v_cache), ("table", self.block_tables, block_tables)):
            if (new.shape != old.shape or new.stride() != old.stride() or
                    new.dtype != old.dtype or new.device != old.device):
                raise ValueError(f"{name} must retain prepared shape/strides/dtype/device")
        table_changed = block_tables.data_ptr() != self.block_tables.data_ptr()
        if table_changed:
            self._validate_table(block_tables)
        self.q, self.k_cache, self.v_cache, self.block_tables = q, k_cache, v_cache, block_tables
        if self._prepared_forward is not None:
            k_cache, v_cache = ((k_cache, v_cache) if self.staging[0] is None else self.staging[0][1:])
            q_view = q.unsqueeze(0)
            k_view = k_cache.permute(0, 2, 1, 3).view(1, -1, k_cache.shape[1], 128)
            v_view = v_cache.permute(0, 2, 1, 3).view(1, -1, v_cache.shape[1], 128)
            if q.dtype == torch.float8_e4m3fn:
                q_view, k_view, v_view = (t.view(torch.uint8) for t in (q_view, k_view, v_view))
            self._prepared_args = (q_view, k_view, v_view, *self._prepared_args[3:])
        if table_changed:
            self._refresh_mappings()
        return self
