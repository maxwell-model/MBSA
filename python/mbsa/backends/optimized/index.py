# Copyright 2026 Maxwell Team.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reuse optimized MBSA Index kernels with private preparation buffers."""
from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
import torch
import cuda.bindings.driver as cuda
from cutlass.base_dsl import BaseDSL
from cutlass.base_dsl.arch import Arch
from .index_base import (IndexPipelineState, TWO_SM_ROUTER_OPTIONS, FP8_Q256_ROUTER_OPTIONS,
                         FP8_TWO_SM_ROUTER_OPTIONS, HOST_ONLY, stage_int32)
from .src.sm100.router import (indexer_local_top16, indexer_packed_top16, _local_top16_arguments,
                               _packed_top16_arguments)
from .kernels.plan import SPEC, PrefillPolicy
from .page_map import page_operand_view
from .index_policy import (select_partition_blocks, lean192_index_eligible, two_sm_index_preferred,
                           packed_two_sm_index_preferred, skip_fixed_blocks_preferred, index_sm_count,
                           index_launch_shape, TWO_SM_MIN_KV_TOKENS, fp8_q256_index_preferred,
                           fp8_two_sm_index_preferred, resolve_index_dispatch, index_dispatch_request,
                           BASELINE_INDEX_DISPATCH, FINE_PARTITION_BLOCKS, FINE_PARTITION_MAX_PARTITIONS,
                           SKIP_FIXED_MAX_KV_TOKENS, FP8_TWO_SM_MIN_KV_TOKENS, FP8_TWO_SM_MIN_PARTITION_BLOCKS,
                           SERIAL_MERGE_MAX_PARTITIONS, HEADS_IN_ROWS_PARTITION_BLOCKS,
                           heads_in_rows_partition_blocks)
from .._binding import explicit_device
from .._index_options import resolve_lean192, validate_lean192


@dataclass
class OptimizedIndexPlan(IndexPipelineState):
    # Wrappers refer only to this binding state's tensors. Retired states keep
    # their own bundle until queued work completes; there is no global cache.
    _local_top16_args: tuple | None = field(default=None, metadata=HOST_ONLY)
    _local_top16_compiled: object | None = field(default=None, metadata=HOST_ONLY)
    _local_top16_key: tuple | None = field(default=None, metadata=HOST_ONLY)
    lean192_mode: str = "auto"
    lean192_bound: bool = False
    # KV blocks per router partition; may be finer than candidate_partition_blocks, which the attention
    # consumer keeps (the router merges all partitions before publishing candidate IDs).
    router_partition_blocks: int | None = None
    # E4M3 spill-free chunked epilogue bound (fp8_q256_bound is on the base plan).
    fp8_epilogue_bound: bool = False
    # (policy inputs, choice) of the last binding: serving rebinds every layer of a step with the same
    # requests, so later layers reuse the dispatch choice instead of re-running the launch policies.
    _launch_choice: tuple | None = field(default=None, metadata=HOST_ONLY)
    # (index_dispatch_request(), IndexDispatch) of the last binding.
    _dispatch: tuple | None = field(default=None, metadata=HOST_ONLY)
    # (plan-owned router operands, their wrappers): serving rebinds every layer with the same output buffers
    # and tile rows, so their wrappers are reused across layer rebinds.
    _output_arguments: tuple | None = field(default=None, metadata=HOST_ONLY)
    # Packed router launch (compiled, arguments), prepared by the first run_score after a binding.
    _packed_launch: tuple | None = field(default=None, metadata=HOST_ONLY)

    def run_score(self):
        """Launch the bound local selection ABI on the current CUDA stream.

        Falls back to the general Index stage when no prepared ABI exists. The stream is resolved
        at invocation, not cached at preparation.

        Returns:
            None; writes retained candidate/partition buffers.
        """
        if (self._local_top16_args is None and self.cutedsl_packed_tile_metadata is not None
                and not self.attention.fixed_only and self.mxfp6_router is None and self._packed_launch is None):
            _prepare_packed_launch(self)
        stream = cuda.CUstream(torch.cuda.current_stream(self.index_q.device).cuda_stream)
        if self._packed_launch is not None:
            self._packed_launch[0](*self._packed_launch[1], stream)
            return
        if self._local_top16_args is None:
            return super().run_score()
        self._local_top16_compiled(*self._local_top16_args, stream)


def _baseline_selection(dispatch):
    """The baseline variant selection with `dispatch`'s launch forms (formats the variants are not measured for)."""
    return dataclasses.replace(BASELINE_INDEX_DISPATCH, runtime_partitions=dispatch.runtime_partitions,
                               heads_in_rows=dispatch.heads_in_rows)


def lean192_compiler_supported():
    """Match the Indexer's actual compiler-target check during preparation."""
    return BaseDSL._get_dsl().get_arch_enum().is_family_of(Arch.sm_103f)


def _stage_packed_tile_metadata(device, requests, partition_blocks, query_tile):
    """Stage packed tile rows (q_begin, q_end, local_tile, page_begin, kv_len, partitions).

    One row per query tile of each request, in packed order. Query tile 256 rounds every request up
    to whole 2-SM CTA pairs; a peer tile past q_end joins its pair's pipeline and stores no rows.
    """
    metadata: list[int] = []
    q_begin = page_begin = 0
    for q_len, kv_len in requests:
        kv_blocks = (kv_len + SPEC.block_size - 1) // SPEC.block_size
        tiles = (q_len + query_tile - 1) // query_tile
        tiles += tiles % 2 if query_tile == 256 else 0
        partitions = (kv_blocks + partition_blocks - 1) // partition_blocks
        for local_tile in range(tiles):
            metadata.extend((q_begin, q_begin + q_len, local_tile, page_begin, kv_len, partitions))
        q_begin += q_len
        page_begin += kv_blocks
    return stage_int32(device, (metadata, (len(metadata) // 6, 6)))


def bind_score_operands(plan):
    """Bind the Index launch of the current operands: the dispatch variant, its compiled router and arguments.

    Args:
        plan: Mutable OptimizedIndexPlan with validated current Index operands, map and candidate
            storage. Called only during preparation or staged rebind, outside capture.

    Returns:
        None; replaces the staged plan's callable/key/argument bundle and, when the bound launch shape
        changes, its partition buffers and staged packed tile metadata. Operand tensors remain strongly
        referenced by that state.
    """
    plan._local_top16_args = None
    plan._local_top16_compiled = None
    plan._local_top16_key = None
    plan._packed_launch = None
    plan.lean192_bound = False
    plan.two_sm_bound = False
    plan.skip_fixed_blocks_bound = False
    plan.router_partition_blocks = None
    plan.fp8_q256_bound = False
    plan.fp8_epilogue_bound = False
    plan.parallel_insert_bound = False
    plan.queued_insert_bound = False
    plan.fp8_two_sm_bound = False
    plan.runtime_partitions_bound = False
    plan.heads_in_rows_bound = False
    plan.route_published = False
    if plan.attention.fixed_only or plan.mxfp6_router is not None:
        return
    # Every variant below is exact; the dispatch only selects which one is bound (index_policy):
    # - skip_fixed: skip the UMMA of blocks that cannot be dynamic candidates for any row of a tile (prefix
    #   blocks and the last row's local window);
    # - two_sm: BF16 Q256 + 2-SM UMMA + early accumulator release (a forced lean192="on" keeps Lean192);
    # - partitions: finer KV partitions for launches whose PB1024 grid leaves SMs idle (beside Fixed attention
    #   only where they save more Index time than it takes: a PB1024 launch leaves those SMs to Fixed attention);
    # - fp8_q256: E4M3 Q256 + early release outside the dense T=8192 geometry (the spill-free chunked
    #   epilogue applies only with parallel and queued insertion off, which the default dispatch never binds);
    # - parallel_insert: one parallel compare-and-shift per block instead of the serial Top-16 bubble. Only
    #   full-row epilogues: the BF16 2-SM router and Lean192 read 32-column chunks, where the parallel network
    #   compiles to 64 registers and spills (the E4M3 2-SM router reads full rows and runs the network inside
    #   its queued insertion); BF16 packed launches keep the serial insertion;
    # - queued_insert: queue each row's admitted candidates and run the insertion network at most once per KV
    #   block (E4M3 only; BF16 hides the epilogue behind its longer MMA). Q256 adds the setmaxnreg register split;
    # - fp8_two_sm: the E4M3 Q256 + queued-insertion rows as 2-SM CTA pairs (each SM receives half of a K tile).
    # A plan keeps its device: rebinding with the same override and environment reuses the resolution.
    request = index_dispatch_request()
    if plan._dispatch is None or plan._dispatch[0] != request:
        plan._dispatch = (request, resolve_index_dispatch(plan.index_q.device, request))
    dispatch = plan._dispatch[1]
    if plan.index_q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        # The variants are measured for BF16 and E4M3 Index only: other formats bind the baseline selection,
        # with the dispatch's launch forms (runtime partition counts, heads in rows).
        dispatch = _baseline_selection(dispatch)
    skip_fixed_mode = dispatch.skip_fixed
    two_sm_mode = dispatch.two_sm
    partition_mode = dispatch.partitions
    fp8_mode = dispatch.fp8_q256
    parallel_insert = dispatch.parallel_insert
    queued_insert = dispatch.queued_insert
    fp8_two_sm_mode = dispatch.fp8_two_sm
    plan.cutedsl_packed_router_tile_metadata = None
    plan.cutedsl_packed_router_partitions = None
    if plan.cutedsl_packed_tile_metadata is not None:
        # Packed requests share one launch: Q128 regular, or 2-SM over request-local CTA pairs.
        requests = plan.cutedsl_packed_requests
        bf16 = plan.index_q.dtype == plan.cutedsl_index_k.dtype == torch.bfloat16
        fp8 = plan.index_q.dtype == plan.cutedsl_index_k.dtype == torch.float8_e4m3fn
        sm_count = lambda: index_sm_count(plan.index_q.device)   # noqa: E731 (device work only when needed)
        heads = int(plan.index_q.shape[1])
        rows = heads * int(plan.index_q.shape[0])
        if (plan.attention.query_outer and dispatch.heads_in_rows and plan.index_q.is_contiguous()
                and rows <= 65535):
            # One-token requests (the query-outer attention pass reads the route): all heads of a token share
            # a CTA, so each request's Index K is read once. The launch is memory-bound: Q128 rows, no
            # fixed-block skip (it would save 16 block loads of a request and double the kernels to warm up),
            # and its own partition width, which keeps the SMs streaming whatever the partitions mode (one
            # CTA per request would leave most SMs idle). The merge publishes the route itself.
            key = (dispatch, plan.index_q.dtype, plan.cutedsl_index_k.dtype, requests, "heads_in_rows")
            if plan._launch_choice is None or plan._launch_choice[0] != key:
                plan._launch_choice = (key, heads_in_rows_partition_blocks(requests, sm_count=sm_count()))
            partition_blocks = plan._launch_choice[1]
            cache = plan.cutedsl_packed_fine_tile_metadata
            if cache is None:
                cache = plan.cutedsl_packed_fine_tile_metadata = {}
            if (partition_blocks, "rows") not in cache:
                (cache[partition_blocks, "rows"],) = _stage_packed_tile_metadata(
                    plan.index_q.device, tuple((heads * q_len, kv_len) for q_len, kv_len in requests),
                    partition_blocks, 128)
            partitions = max((kv + SPEC.block_size * partition_blocks - 1) // (SPEC.block_size * partition_blocks)
                             for _, kv in requests)
            shape = (1, partitions, rows, SPEC.dynamic_topk)
            scores = plan.cutedsl_partition_scores
            if not isinstance(scores, torch.Tensor) or tuple(scores.shape) != shape:
                plan.cutedsl_partition_scores = torch.empty(shape, device=plan.index_q.device, dtype=torch.float32)
                plan.cutedsl_partition_ids = torch.empty(shape, device=plan.index_q.device, dtype=torch.int32)
            plan.cutedsl_packed_router_tile_metadata = cache[partition_blocks, "rows"]
            plan.cutedsl_packed_router_partitions = partitions
            plan.router_partition_blocks = partition_blocks
            plan.runtime_partitions_bound = dispatch.runtime_partitions
            plan.parallel_insert_bound = parallel_insert and fp8
            plan.queued_insert_bound = queued_insert and fp8
            plan.heads_in_rows_bound = plan.route_published = True
            return

        def choose():
            partition_blocks = plan.candidate_partition_blocks
            fp8_q256 = False
            if partition_mode == "auto" and (bf16 or fp8) and partition_blocks == 1024:
                # Joint 2-SM (E4M3: single-SM Q256) / partition-width choice on the shared launch; the packed
                # Q128 baseline has no Lean192 kernel, so there is no KV floor here.
                if fp8:
                    two_sm = False
                    fp8_q256, partition_blocks = index_launch_shape(
                        requests, two_sm="auto" if fp8_mode == "auto" else "0", heads=int(plan.index_q.shape[1]),
                        sm_count=sm_count(), pairs=False, beside_fixed=plan.overlap_fixed_score)
                else:
                    two_sm, partition_blocks = index_launch_shape(
                        requests, two_sm=two_sm_mode, heads=int(plan.index_q.shape[1]), sm_count=sm_count(),
                        beside_fixed=plan.overlap_fixed_score)
            else:
                two_sm = (
                    two_sm_mode != "0" and bf16
                    and (two_sm_mode == "1" or packed_two_sm_index_preferred(
                        requests, partition_blocks=plan.candidate_partition_blocks, sm_count=sm_count()))
                )
                fp8_q256 = fp8 and fp8_mode == "auto" and fp8_q256_index_preferred(
                    requests, heads=int(plan.index_q.shape[1]), partition_blocks=partition_blocks,
                    sm_count=sm_count())
            # "auto" skip keys on the longest request.
            skip_fixed = skip_fixed_mode == "1" or (skip_fixed_mode == "auto" and skip_fixed_blocks_preferred(
                two_sm=two_sm, lean192=False, logical_k_tokens=max(kv for _, kv in requests)))
            # E4M3 Q256 + queue as 2-SM pairs: the 256-row tile rows come in whole CTA pairs.
            fp8_two_sm = bool(
                fp8_q256 and queued_insert and fp8 and not two_sm and fp8_two_sm_mode == "auto"
                and fp8_two_sm_index_preferred(requests, heads=int(plan.index_q.shape[1]),
                                               partition_blocks=partition_blocks, sm_count=sm_count()))
            return two_sm, fp8_q256, partition_blocks, skip_fixed, fp8_two_sm

        key = (dispatch, plan.index_q.dtype, plan.cutedsl_index_k.dtype, requests, plan.candidate_partition_blocks,
               plan.overlap_fixed_score)
        if plan._launch_choice is None or plan._launch_choice[0] != key:
            plan._launch_choice = (key, choose())
        two_sm, fp8_q256, partition_blocks, skip_fixed, fp8_two_sm = plan._launch_choice[1]
        if (two_sm or fp8_q256) and plan.cutedsl_packed_two_sm_tile_metadata is None:
            # 256-row tile rows (whole CTA pairs; a single-SM Q256 launch leaves the padding CTA idle).
            (plan.cutedsl_packed_two_sm_tile_metadata,) = _stage_packed_tile_metadata(
                plan.index_q.device, requests, plan.candidate_partition_blocks, 256)
        if fp8_q256 and partition_blocks == plan.candidate_partition_blocks:
            plan.cutedsl_packed_router_tile_metadata = plan.cutedsl_packed_two_sm_tile_metadata
        if partition_blocks != plan.candidate_partition_blocks:
            # Finer router partitions: request-local partition counts and the merge buffers follow the
            # router width; candidate_partition_blocks (the attention consumer's) is unchanged.
            cache = plan.cutedsl_packed_fine_tile_metadata
            if cache is None:
                cache = plan.cutedsl_packed_fine_tile_metadata = {}
            tile = 256 if two_sm or fp8_q256 else 128
            if (partition_blocks, tile) not in cache:
                (cache[partition_blocks, tile],) = _stage_packed_tile_metadata(
                    plan.index_q.device, requests, partition_blocks, tile)
            partitions = max((kv + SPEC.block_size * partition_blocks - 1) // (SPEC.block_size * partition_blocks)
                             for _, kv in requests)
            shape = (plan.index_q.shape[1], partitions, plan.index_q.shape[0], SPEC.dynamic_topk)
            scores = plan.cutedsl_partition_scores
            if scores is None or (isinstance(scores, torch.Tensor) and tuple(scores.shape) != shape):
                plan.cutedsl_partition_scores = torch.empty(shape, device=plan.index_q.device, dtype=torch.float32)
                plan.cutedsl_partition_ids = torch.empty(shape, device=plan.index_q.device, dtype=torch.int32)
            plan.cutedsl_packed_router_tile_metadata = cache[partition_blocks, tile]
            plan.cutedsl_packed_router_partitions = partitions
            plan.router_partition_blocks = partition_blocks
        else:
            # The merge reads every partition of the bound buffers: restore the prepared PB1024 layout after a
            # finer binding. A single-partition launch still writes its partition buffers; they are bound here,
            # with the plan's backing tensors, not at the first run.
            default_partitions = plan.cutedsl_packed_partitions
            shape = (plan.index_q.shape[1], default_partitions, plan.index_q.shape[0], SPEC.dynamic_topk)
            scores = plan.cutedsl_partition_scores
            if not isinstance(scores, torch.Tensor) or tuple(scores.shape) != shape:
                plan.cutedsl_partition_scores = torch.empty(shape, device=plan.index_q.device, dtype=torch.float32)
                plan.cutedsl_partition_ids = torch.empty(shape, device=plan.index_q.device, dtype=torch.int32)
        plan.skip_fixed_blocks_bound = skip_fixed
        plan.two_sm_bound = two_sm
        plan.runtime_partitions_bound = dispatch.runtime_partitions
        plan.fp8_q256_bound = fp8_q256
        # BF16 packed launches keep the serial insertion: with a long request in the launch the parallel network
        # measures -0.5..+1.1 % (the BF16 MMA already hides the serial chain).
        plan.parallel_insert_bound = parallel_insert and fp8
        plan.queued_insert_bound = queued_insert and fp8 and not two_sm
        plan.fp8_two_sm_bound = fp8_two_sm
        return
    bf16 = plan.index_q.dtype == plan.cutedsl_index_k.dtype == torch.bfloat16
    fp8 = plan.index_q.dtype == plan.cutedsl_index_k.dtype == torch.float8_e4m3fn
    requests = ((int(plan.index_q.shape[0]), plan.cutedsl_logical_k_tokens),)
    sm_count = lambda: index_sm_count(plan.index_q.device)   # noqa: E731 (device work only when needed)

    def choose():
        two_sm_legal = two_sm_mode != "0" and plan.lean192_mode != "on" and bf16
        partition_blocks = plan.candidate_partition_blocks
        fp8_q256 = False
        if (partition_mode == "auto" and (bf16 or fp8) and partition_blocks == 1024
                and not plan._lean192_early_router and plan.lean192_mode != "on"):
            kv = plan.cutedsl_logical_k_tokens
            if fp8:
                # E4M3: Q128, or the single-SM Q256 router when fp8_q256 is "auto".
                two_sm = False
                fp8_q256, partition_blocks = index_launch_shape(
                    requests, two_sm="auto" if fp8_mode == "auto" else "0", heads=int(plan.index_q.shape[1]),
                    sm_count=sm_count(), pairs=False, beside_fixed=plan.overlap_fixed_score)
            else:
                mode = two_sm_mode if two_sm_legal and (two_sm_mode == "1" or kv >= TWO_SM_MIN_KV_TOKENS) else "0"
                two_sm, partition_blocks = index_launch_shape(
                    requests, two_sm=mode, heads=int(plan.index_q.shape[1]), sm_count=sm_count(),
                    beside_fixed=plan.overlap_fixed_score)
        else:
            two_sm = two_sm_legal and (two_sm_mode == "1" or two_sm_index_preferred(
                plan.index_q, plan.cutedsl_index_k,
                partition_blocks=plan.candidate_partition_blocks,
                logical_k_tokens=plan.cutedsl_logical_k_tokens))
            fp8_q256 = fp8 and fp8_mode == "auto" and fp8_q256_index_preferred(
                requests, heads=int(plan.index_q.shape[1]), partition_blocks=partition_blocks, sm_count=sm_count())
        lean192 = bool(plan._lean192_early_router and not two_sm and lean192_compiler_supported())
        skip_fixed = skip_fixed_mode == "1" or (skip_fixed_mode == "auto" and skip_fixed_blocks_preferred(
            two_sm=two_sm, lean192=lean192, logical_k_tokens=plan.cutedsl_logical_k_tokens))
        return two_sm, fp8_q256, partition_blocks, lean192, skip_fixed

    key = (dispatch, plan.index_q.dtype, plan.cutedsl_index_k.dtype, requests, plan.candidate_partition_blocks,
           plan.lean192_mode, plan._lean192_early_router, plan.overlap_fixed_score)
    if plan._launch_choice is None or plan._launch_choice[0] != key:
        plan._launch_choice = (key, choose())
    two_sm, fp8_q256, partition_blocks, lean192, skip_fixed = plan._launch_choice[1]
    fp8_options = {}
    if fp8_mode == "auto" and fp8:
        fp8_options["_fp8_epilogue"] = True
        if fp8_q256:
            fp8_options["_fp8_q256"] = True
    blocks = (plan.cutedsl_logical_k_tokens + SPEC.block_size - 1) // SPEC.block_size
    partitions = (blocks + partition_blocks - 1) // partition_blocks
    shape = (plan.index_q.shape[1], partitions, plan.index_q.shape[0], SPEC.dynamic_topk)
    scores = plan.cutedsl_partition_scores
    if scores is None or (isinstance(scores, torch.Tensor) and tuple(scores.shape) != shape):
        # Preserve the binding-owned ABI for both the direct single-partition
        # output and the ordinary partition/merge path.
        plan.cutedsl_partition_scores = torch.empty(shape, device=plan.index_q.device, dtype=torch.float32)
        plan.cutedsl_partition_ids = torch.empty(shape, device=plan.index_q.device, dtype=torch.int32)
    outputs = (plan.cutedsl_partition_scores, plan.cutedsl_partition_ids, plan.candidate_storage,
               dispatch.runtime_partitions)
    reuse = plan._output_arguments is not None and all(
        a is b for a, b in zip(plan._output_arguments[0], outputs))
    plan._local_top16_args = _local_top16_arguments(
        plan.index_q, plan.cutedsl_index_k, plan.cutedsl_index_pages,
        plan.cutedsl_partition_scores, plan.cutedsl_partition_ids,
        plan.candidate_storage.view(torch.int32), plan.cutedsl_logical_k_tokens, 2,
        dispatch.runtime_partitions, outputs=plan._output_arguments[1] if reuse else None)
    plan._output_arguments = (outputs, plan._local_top16_args[4:7])
    plan._local_top16_compiled, plan._local_top16_key = indexer_local_top16(
        plan.index_q, plan.cutedsl_index_k,
        partition_blocks=partition_blocks,
        page_indices=plan.cutedsl_index_pages,
        logical_k_tokens=plan.cutedsl_logical_k_tokens,
        use_page_table=plan.cutedsl_use_page_table,
        _lean192_early=plan._lean192_early_router and not two_sm,
        out_scores=plan.cutedsl_partition_scores,
        out_ids=plan.cutedsl_partition_ids,
        out_compact_ids=plan.candidate_storage.view(torch.int32),
        _prepared_args=plan._local_top16_args,
        _prepare_only=True,
        _fuse_single_partition_compact=partitions == 1,
        **(TWO_SM_ROUTER_OPTIONS if two_sm else {}),
        **(dict(_skip_fixed_blocks=True) if skip_fixed else {}),
        **fp8_options,
        **(dict(_parallel_insert=True) if parallel_insert and not two_sm and not lean192 else {}),
        **(dict(_queued_insert=True) if queued_insert and fp8 and not two_sm and not lean192 else {}),
        **(dict(_fp8_two_sm=True) if fp8_two_sm_mode == "auto" and queued_insert and fp8 and not lean192 else {}),
        _runtime_partitions=dispatch.runtime_partitions,
    )
    key = tuple(plan._local_top16_key or ())
    plan.fp8_q256_bound = fp8 and ("query_tile", 256) in tuple(zip(key, key[1:]))
    plan.fp8_epilogue_bound = fp8 and "chunked_epilogue" in key
    plan.parallel_insert_bound = "parallel_insert" in key
    plan.queued_insert_bound = "queued_insert" in key
    plan.fp8_two_sm_bound = fp8 and "two_sm" in key
    plan.two_sm_bound = two_sm
    plan.skip_fixed_blocks_bound = skip_fixed
    plan.lean192_bound = lean192
    plan.router_partition_blocks = partition_blocks
    plan.runtime_partitions_bound = dispatch.runtime_partitions


def _prepare_packed_launch(plan):
    """Compile the bound packed router launch and retain its arguments for later runs of this binding."""
    if plan.cutedsl_index_k is None:
        raise RuntimeError("CuTeDSL router workspace was not prepared")
    if plan.cutedsl_packed_pages is None:
        raise RuntimeError("packed CuTeDSL page table is missing")
    partitions = plan.cutedsl_packed_router_partitions or plan.cutedsl_packed_partitions
    scores, ids = plan.cutedsl_partition_scores, plan.cutedsl_partition_ids
    tile_metadata = (plan.cutedsl_packed_router_tile_metadata
                     if plan.cutedsl_packed_router_tile_metadata is not None
                     else plan.cutedsl_packed_two_sm_tile_metadata if plan.two_sm_bound
                     else plan.cutedsl_packed_tile_metadata)
    if plan.heads_in_rows_bound:
        # Rows are (token, head) pairs of the [T*H,1,128] query view; the merge writes the route [H,T,16].
        q_rows = plan.index_q.view(-1, 1, SPEC.head_dim)
        route = plan.attention.tile_plan.dynamic_route
        owned = (tile_metadata, scores, ids, route, plan.runtime_partitions_bound)
        reuse = plan._output_arguments is not None and all(a is b for a, b in zip(plan._output_arguments[0], owned))
        args = _packed_top16_arguments(q_rows, plan.cutedsl_index_k, plan.cutedsl_packed_pages, tile_metadata,
                                       scores, ids, route, plan.runtime_partitions_bound,
                                       outputs=plan._output_arguments[1] if reuse else None, route_token_mode=1)
        plan._output_arguments = (owned, args[3:7])
        compiled, _ = indexer_packed_top16(
            q_rows, plan.cutedsl_index_k, partition_blocks=plan.router_partition_blocks,
            page_indices=plan.cutedsl_packed_pages, tile_metadata=tile_metadata, partitions=partitions,
            out_route=route, out_scores=scores, out_ids=ids, lean_pipeline=False,
            **(dict(_parallel_insert=True) if plan.parallel_insert_bound else {}),
            **(dict(_queued_insert=True) if plan.queued_insert_bound else {}),
            _runtime_partitions=plan.runtime_partitions_bound, _row_heads=int(plan.index_q.shape[1]),
            _prepared_args=args, _prepare_only=True)
        plan._packed_launch = (compiled, args)
        return
    owned = (tile_metadata, scores, ids, plan.candidate_storage, plan.runtime_partitions_bound)
    reuse = plan._output_arguments is not None and all(a is b for a, b in zip(plan._output_arguments[0], owned))
    args = _packed_top16_arguments(plan.index_q, plan.cutedsl_index_k, plan.cutedsl_packed_pages, tile_metadata,
                                   scores, ids, plan.candidate_storage.view(torch.int32),
                                   plan.runtime_partitions_bound,
                                   outputs=plan._output_arguments[1] if reuse else None)
    plan._output_arguments = (owned, args[3:7])
    compiled, _ = indexer_packed_top16(
        plan.index_q, plan.cutedsl_index_k,
        partition_blocks=plan.router_partition_blocks or plan.candidate_partition_blocks,
        page_indices=plan.cutedsl_packed_pages, tile_metadata=tile_metadata, partitions=partitions,
        out_compact_ids=plan.candidate_storage.view(torch.int32), out_scores=scores, out_ids=ids,
        lean_pipeline=False, _skip_fixed_blocks=plan.skip_fixed_blocks_bound,
        **(TWO_SM_ROUTER_OPTIONS if plan.two_sm_bound
           else FP8_TWO_SM_ROUTER_OPTIONS if plan.fp8_two_sm_bound
           else FP8_Q256_ROUTER_OPTIONS if plan.fp8_q256_bound else {}),
        **(dict(_parallel_insert=True) if plan.parallel_insert_bound else {}),
        **(dict(_queued_insert=True) if plan.queued_insert_bound else {}),
        _runtime_partitions=plan.runtime_partitions_bound, _prepared_args=args, _prepare_only=True)
    plan._packed_launch = (compiled, args)


def _serving_router_variants(dispatch, fp8, max_kv_tokens):
    """Enumerate the router kernels `dispatch` can bind for paged serving steps (page table, logical K tail).

    Yields (packed, partition_blocks, partitions, logical_k_tokens, options), one per compile key: query-tile
    variant, partition width, merge form (single-partition fused, serial or tree) and fixed-block skip, pruned
    with the policy's KV and width thresholds, then the heads-in-rows kernels of one-token requests (options
    with _row_heads). With static partition counts every count is its own kernel.
    """
    block = SPEC.block_size
    max_blocks = (max_kv_tokens + block - 1) // block
    skips = {"0": (False,), "1": (True,), "auto": (True, False)}[dispatch.skip_fixed]
    widths = (1024,) + (FINE_PARTITION_BLOCKS if dispatch.partitions == "auto" else ())
    for packed in (False, True):
        # BF16 packed launches keep the serial insertion (bind_score_operands).
        inserts = dict(_parallel_insert=True) if dispatch.parallel_insert and (fp8 or not packed) else {}
        if fp8 and dispatch.queued_insert:
            inserts["_queued_insert"] = True
        # (options, KV floor, partition-width floor, skip modes)
        tiles = [(inserts, 0, 0, skips)]
        if fp8 and dispatch.fp8_q256 == "auto":
            tiles.append((dict(FP8_Q256_ROUTER_OPTIONS, **inserts), 0, 0, skips))
            if dispatch.fp8_two_sm == "auto" and dispatch.queued_insert:
                tiles.append((dict(FP8_TWO_SM_ROUTER_OPTIONS, **inserts), FP8_TWO_SM_MIN_KV_TOKENS,
                              FP8_TWO_SM_MIN_PARTITION_BLOCKS, skips))
        if not fp8 and dispatch.two_sm != "0":
            # "auto" always skips on the 2-SM router (skip_fixed_blocks_preferred).
            tiles.append((TWO_SM_ROUTER_OPTIONS, 0, 0, (False,) if dispatch.skip_fixed == "0" else (True,)))
        for width in widths:
            cap = max_blocks if width == 1024 else min(max_blocks, FINE_PARTITION_MAX_PARTITIONS * width)
            # (lowest, highest) KV blocks of each merge form; a finer width only binds when it splits the range.
            forms = [] if packed or width != 1024 else [("fused", 1, min(width, cap))]
            forms += [("serial", 1 if packed else width + 1, min(SERIAL_MERGE_MAX_PARTITIONS * width, cap)),
                      ("tree", SERIAL_MERGE_MAX_PARTITIONS * width + 1, cap)]
            for form, low, high in forms:
                if low > high:
                    continue
                for options, kv_floor, width_floor, skip_modes in tiles:
                    if width < width_floor:
                        continue
                    two_sm_floor = TWO_SM_MIN_KV_TOKENS if options is TWO_SM_ROUTER_OPTIONS and not packed else 0
                    lowest = max(low, (max(kv_floor, two_sm_floor) + block - 1) // block)
                    if lowest > high:
                        continue
                    for skip in skip_modes:
                        if dispatch.skip_fixed == "auto" and options is not TWO_SM_ROUTER_OPTIONS:
                            # "auto" skips up to SKIP_FIXED_MAX_KV_TOKENS.
                            limit = SKIP_FIXED_MAX_KV_TOKENS // block
                            if (lowest > limit) if skip else (high <= limit):
                                continue
                        blocks = lowest if skip or dispatch.skip_fixed != "auto" else max(
                            lowest, SKIP_FIXED_MAX_KV_TOKENS // block + 1)
                        kvs = [blocks * block]
                        if form == "fused":
                            # The lean (<= 8K) and reduced-accumulator (32K) pipelines are separate kernels; any
                            # other extent in range stands for the plain fused kernel.
                            plain = high * block - (block if high * block == 32768 else 0)
                            kvs = [kv for kv in (8192, 32768, plain) if low * block <= kv <= high * block
                                   and kv >= blocks * block]
                        for kv in kvs:
                            partitions = (kv + block * width - 1) // (block * width)
                            yield (packed, width, partitions, kv,
                                   dict(options, **(dict(_skip_fixed_blocks=True) if skip else {})))
    if dispatch.heads_in_rows:
        # Packed steps of one-token requests: one kernel per partition width and merge form.
        inserts = dict(_row_heads=SPEC.kv_heads, **(dict(_parallel_insert=True) if fp8 and dispatch.parallel_insert
                                                     else {}))
        if fp8 and dispatch.queued_insert:
            inserts["_queued_insert"] = True
        for width in HEADS_IN_ROWS_PARTITION_BLOCKS:
            cap = min(max_blocks, FINE_PARTITION_MAX_PARTITIONS * width)
            for low, high in ((1, min(SERIAL_MERGE_MAX_PARTITIONS * width, cap)),
                              (SERIAL_MERGE_MAX_PARTITIONS * width + 1, cap)):
                if low <= high:
                    yield True, width, (low + width - 1) // width, low * block, inserts


_COMPILED_SERVING_VARIANTS = set()


def compile_serving_index_variants(device, index_dtype, *, max_kv_tokens):
    """Compile every Index router kernel the device's dispatch can bind for paged serving steps.

    Args:
        device: CUDA device of the serving runner.
        index_dtype: Index Q/K dtype: BF16 or E4M3 (the dispatch variants), or FP16 (always the baseline
            selection). MX Index formats use their own scorer.
        max_kv_tokens: Largest per-request KV extent of the steps.

    Returns:
        Number of router variants prepared (compiled unless already cached); 0 for other Index dtypes, a
        dispatch with static partition counts (one kernel per count, not enumerated) or a repeated call
        for the same GPU. Allocates temporary dummy operands and compiles without launching.
    """
    device = explicit_device(device)
    dispatch = resolve_index_dispatch(device)
    if index_dtype == torch.float16:
        dispatch = _baseline_selection(dispatch)   # as bind_score_operands binds FP16 Index
    fp8 = index_dtype == torch.float8_e4m3fn
    done = (device.index, index_dtype, dispatch, max_kv_tokens)
    if (not dispatch.runtime_partitions or done in _COMPILED_SERVING_VARIANTS
            or index_dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn)):
        return 0
    prepared = 0
    heads, dim, block, rows = SPEC.kv_heads, 128, SPEC.block_size, 256
    variants = list(_serving_router_variants(dispatch, fp8, max_kv_tokens))
    if not variants:
        _COMPILED_SERVING_VARIANTS.add(done)
        return 0
    # Compilation reads only shapes: one uninitialized K buffer (the largest variant's) serves every variant.
    k_pages = max((kv + block - 1) // block * (2 if packed else 1) + 1 for packed, _, _, kv, _ in variants)
    k_all = torch.empty((k_pages * block, 1, dim), device=device, dtype=index_dtype)
    q = torch.empty((rows, heads, dim), device=device, dtype=index_dtype)
    for packed, width, partitions, kv, options in variants:
        blocks = (kv + block - 1) // block
        requests = ((rows // 2, kv), (rows // 2, kv)) if packed else ((rows, kv),)
        pages = torch.arange(blocks * len(requests), dtype=torch.int32, device=device)
        k = k_all[:(pages.numel() + 1) * block]
        topk = SPEC.dynamic_topk
        if "_row_heads" in options:
            # Two one-token requests; rows are (token, head) pairs and the merge writes the route.
            shape = (1, partitions, 2 * heads, topk)
            outputs = dict(out_compact_ids=None,
                           out_route=torch.empty((heads, 2, topk), device=device, dtype=torch.int32))
        else:
            shape = (heads, partitions, rows, topk)
            outputs = dict(out_compact_ids=torch.empty((heads, 2 * topk, rows), device=device, dtype=torch.int32))
        common = dict(partition_blocks=width, page_indices=pages,
                      out_scores=torch.empty(shape, device=device, dtype=torch.float32),
                      out_ids=torch.empty(shape, device=device, dtype=torch.int32), **outputs,
                      _runtime_partitions=dispatch.runtime_partitions, _prepare_only=True)
        if "_row_heads" in options:
            (metadata,) = _stage_packed_tile_metadata(device, ((heads, kv), (heads, kv)), width, 128)
            indexer_packed_top16(q[:2].view(-1, 1, dim), k, tile_metadata=metadata, partitions=partitions,
                                 **common, **options)
        elif packed:
            tile = options.get("_query_tile", 128)
            (metadata,) = _stage_packed_tile_metadata(device, requests, width, tile)
            indexer_packed_top16(q, k, tile_metadata=metadata, partitions=partitions, **common, **options)
        else:
            indexer_local_top16(q, k, logical_k_tokens=kv, use_page_table=True,
                                _fuse_single_partition_compact=partitions == 1, **common, **options)
        prepared += 1
    _COMPILED_SERVING_VARIANTS.add(done)
    return prepared


def prepare_index(index_q, index_k, *, index_q_scales, index_page_indices,
                  candidate_storage, prefix_lens, attention, q_lens, kv_lens,
                  index_pages_identity, check_device_inputs, original_q,
                  original_k, original_v, contiguous_unpacked, lean192="auto", overlap_fixed_score=False):
    """Allocate selection workspaces and bind the Index stage for a prepared attention plan.

    Args:
        index_q: CUDA Index queries [T,4,128], matching BF16/FP16/E4M3 Index K, or MXFP8 uint8/E4M3
            codes with index_q_scales. T is the packed query-token count.
        index_k: CUDA Index keys [Skv,1,128] or [pages,1,128,128] for floating formats; MXFP6 uses
            packed uint8 [pages,1,128,100] with page-separated codes/scales. Physical pages contain
            128 tokens.
        index_q_scales: Optional CUDA int32 [T,>=4] MXFP8 scale words, four little-endian UE8M0
            bytes per head. Required for MXFP6 Index K; otherwise None.
        index_page_indices: Optional CUDA int32 logical-to-physical Index page IDs, compact 1-D
            active IDs or request rows [B,>=ceil(KV/128)]. Optimized binds active rows directly:
            one adjacent contiguous storage span with a 16-byte base alignment for unpacked/MX
            or 4-byte base alignment for packed floating Index. Individual compact request rows
            need not be 16-byte aligned. Gaps/strides are rejected. None creates the backend
            default mapping.
        candidate_storage: Writable contiguous FP32 candidate backing [4,32,T], with packed ID bits
            consumed through int32 views by the selection/CSR ABI. Retained by the returned plan.
        prefix_lens: CUDA int32 per-query valid-page-prefix bounds used for dynamic selection.
        attention: Prepared attention plan supplying request prefixes, tile policy and partial
            dtype.
        q_lens: Host query-token lengths in packed request order.
        kv_lens: Host logical KV-token lengths, one per query request.
        index_pages_identity: True for known identity Index page IDs, False for mapped pages, or
            None to validate/detect when device checks are enabled.
        check_device_inputs: Whether to synchronize for Index page-value range/identity checks.
            False requires valid caller metadata.
        original_q: CUDA attention queries [T,32,128]. BF16 for BF16 K/V; the default Blackwell FP8
            path also accepts BF16 or E4M3FN Q with E4M3FN K/V. Optimized converts FP8 operands to
            owned BF16 buffers per run.
        original_k: CUDA attention keys [Skv,4,128] or paged [pages,4,128,128]. Must match v in
            dtype/layout/device; BF16, or E4M3FN on Blackwell with unit scales.
        original_v: CUDA attention values with the same shape, dtype and device as k. Read-only
            until the queued invocation completes.
        contiguous_unpacked: True for one unpacked request; False selects packed request metadata
            and map handling.
        lean192: Requested preparation-only auto/on/off policy. Forced ON rejects
            ineligible inputs before a bound Index kernel can be executed.
        overlap_fixed_score: Whether Fixed attention runs beside the Index on a side stream; the
            Index dispatch then keeps the SMs a PB1024 launch leaves idle for it where that is faster.

    Returns:
        OptimizedIndexPlan retaining buffers, direct map views and selected launch metadata.
        Preparation may allocate/compile and must run outside capture.
    """
    validate_lean192(lean192)
    q, k, v = original_q, original_k, original_v
    total_q = q.shape[0]
    batch = len(q_lens)
    skv = max(kv_lens)
    tile_plan = attention.tile_plan
    partial_dtype = attention.partial_dtype
    cu_seqlens_q = None if contiguous_unpacked else attention.cu_q
    cu_seqlens_k = None if contiguous_unpacked else attention.cu_k
    is_mx = index_k.dtype == torch.uint8
    blackwell_native = torch.cuda.get_device_capability(q.device) in ((10, 0), (10, 3))
    partition_blocks = select_partition_blocks(
        index_q, index_k, batch=batch, contiguous_unpacked=contiguous_unpacked,
        kv_tokens=skv, identity_pages=index_pages_identity,
        capability=torch.cuda.get_device_capability(q.device),
    )
    cutedsl_index_k = None
    cutedsl_index_pages = None
    cutedsl_logical_k_tokens = None
    cutedsl_use_page_table = False
    cutedsl_partition_scores = None
    cutedsl_partition_ids = None
    cutedsl_packed_partitions: int = 1
    cutedsl_packed_pages = None
    cutedsl_packed_tile_metadata = None
    cutedsl_packed_requests = None
    mxfp6_router = None
    page_operand = None
    if not attention.fixed_only:
        if not blackwell_native:
            raise ValueError("the CuTeDSL router requires SM100 or SM103")
        if is_mx:
            cutedsl_index_k = index_k
        elif index_k.ndim == 4:
            if tuple(index_k.shape[1:]) != (
                1,
                SPEC.block_size,
                SPEC.head_dim,
            ):
                raise ValueError("CuTeDSL index_k pages must be [pages,1,128,128]")
            cutedsl_index_k = index_k.view(-1, 1, SPEC.head_dim)
        elif index_k.ndim == 3:
            if batch != 1:
                raise ValueError("packed CuTeDSL routing requires paged Index K")
            if tuple(index_k.shape) != (skv, 1, SPEC.head_dim):
                raise ValueError("CuTeDSL index_k must be [Skv,1,128] when unpaged")
            cutedsl_index_k = index_k
            if skv % SPEC.block_size:
                raise ValueError(
                    "a partial logical K tail requires full paged Index K storage"
                )
        else:
            raise ValueError("unsupported CuTeDSL index_k layout")
        if not is_mx and int(cutedsl_index_k.shape[0]) % SPEC.block_size:
            raise ValueError("CuTeDSL physical Index K storage must be page aligned")
        physical_pages = (
            int(cutedsl_index_k.shape[0])
            if is_mx
            else int(cutedsl_index_k.shape[0]) // SPEC.block_size
        )
        request_blocks = tuple(
            (length + SPEC.block_size - 1) // SPEC.block_size for length in kv_lens
        )
        if index_page_indices.ndim == 2:
            if index_page_indices.shape[0] != batch or any(
                index_page_indices.shape[1] < count for count in request_blocks
            ):
                raise ValueError(
                    "2-D index_page_indices must be [B,>=max logical pages]"
                )
            request_pages = tuple(
                index_page_indices[index, :count]
                for index, count in enumerate(request_blocks)
            )
        elif index_page_indices.ndim == 1:
            active_pages = sum(request_blocks)
            if index_page_indices.numel() == active_pages:
                page_offsets = [0]
                for count in request_blocks:
                    page_offsets.append(page_offsets[-1] + count)
                request_pages = tuple(
                    index_page_indices[
                        page_offsets[index] : page_offsets[index + 1]
                    ]
                    for index in range(batch)
                )
            elif index_page_indices.numel() % batch == 0:
                row_width = index_page_indices.numel() // batch
                if row_width < max(request_blocks):
                    raise ValueError("flattened page-table rows are too short")
                page_rows = index_page_indices.view(batch, row_width)
                request_pages = tuple(
                    page_rows[index, :count]
                    for index, count in enumerate(request_blocks)
                )
            else:
                raise ValueError(
                    "1-D index_page_indices must contain active pages or padded rows"
                )
        else:
            raise ValueError("index_page_indices must be rank 1 or 2")
        if check_device_inputs and any(
            bool(torch.any(pages < 0)) or bool(torch.any(pages >= physical_pages))
            for pages in request_pages
        ):
            raise ValueError("index_page_indices references Index K out of range")

        page_operand = page_operand_view(request_pages, 16 if batch == 1 or is_mx else 4)
        if batch == 1 and not is_mx:
            cutedsl_index_pages = page_operand
            cutedsl_logical_k_tokens = kv_lens[0]
            if index_pages_identity is None and check_device_inputs:
                identity_pages = torch.arange(
                    request_blocks[0],
                    dtype=torch.int32,
                    device=index_page_indices.device,
                )
                index_pages_identity = torch.equal(
                    cutedsl_index_pages, identity_pages
                )
            cutedsl_use_page_table = not index_pages_identity
            cutedsl_partitions = (
                request_blocks[0] + partition_blocks - 1
            ) // partition_blocks
            if cutedsl_partitions > 1:
                partition_shape = (
                    SPEC.kv_heads,
                    cutedsl_partitions,
                    total_q,
                    SPEC.dynamic_topk,
                )
                cutedsl_partition_scores = torch.empty(
                    partition_shape,
                    dtype=torch.float32,
                    device=q.device,
                )
                cutedsl_partition_ids = torch.empty(
                    partition_shape,
                    dtype=torch.int32,
                    device=q.device,
                )
        else:
            request_partitions = tuple(
                (kv_blocks + partition_blocks - 1) // partition_blocks
                for kv_blocks in request_blocks
            )
            # Emit all requests and KV partitions in one grid. Unused
            # partitions publish (-inf, -1), so the global merge has the same
            # request-local Top-16 set without per-request kernel launches.
            cutedsl_packed_partitions = max(request_partitions)
            cutedsl_packed_pages = page_operand
            cutedsl_packed_requests = tuple(zip(q_lens, kv_lens))
            (cutedsl_packed_tile_metadata,) = _stage_packed_tile_metadata(
                q.device, cutedsl_packed_requests, partition_blocks, 128)
            if cutedsl_packed_partitions > 1:
                partition_shape = (
                    SPEC.kv_heads,
                    cutedsl_packed_partitions,
                    total_q,
                    SPEC.dynamic_topk,
                )
                cutedsl_partition_scores = torch.empty(
                    partition_shape,
                    dtype=torch.float32,
                    device=q.device,
                )
                cutedsl_partition_ids = torch.empty(
                    partition_shape,
                    dtype=torch.int32,
                    device=q.device,
                )
    if is_mx and not attention.fixed_only:
        from .src.sm100.mx_router import prepare_mxfp6_router

        mxfp6_router = prepare_mxfp6_router(
            index_q,
            index_q_scales,
            index_k,
            cutedsl_packed_pages,
            cutedsl_packed_tile_metadata,
            max_blocks=max(request_blocks),
            route=candidate_storage.view(torch.int32),
            partition_blocks=256,
        )
    lean192_early_router = bool(
        contiguous_unpacked and not attention.fixed_only and batch == 1
        and cu_seqlens_q is None and cu_seqlens_k is None
        and tile_plan.policy == PrefillPolicy.ALIGNED_CONTIGUOUS
        and index_pages_identity is True and not cutedsl_use_page_table
        and cutedsl_packed_tile_metadata is None and mxfp6_router is None
        and q.dtype == k.dtype == v.dtype == partial_dtype == torch.bfloat16
        and index_k.ndim in (3, 4) and k.ndim == v.ndim == 3
        and index_q.device == index_k.device == q.device
        and all(t.is_contiguous() for t in (index_q, index_k, q, k, v))
        and lean192_index_eligible(
            # Paged-shaped contiguous Index storage already has this zero-copy
            # dense view. Check its exact physical extent against logical KV.
            index_q, cutedsl_index_k, partition_blocks=partition_blocks,
            logical_k_tokens=skv, use_page_table=cutedsl_use_page_table)
        and torch.cuda.get_device_capability(q.device) == (10, 3)
    )
    plan = OptimizedIndexPlan(
        index_q=index_q, index_k=index_k, index_q_scales=index_q_scales,
        mxfp6_router=mxfp6_router, index_page_indices=index_page_indices,
        candidate_storage=candidate_storage, prefix_lens=prefix_lens,
        attention=attention, candidate_partition_blocks=partition_blocks,
        cutedsl_index_k=cutedsl_index_k, cutedsl_index_pages=cutedsl_index_pages,
        cutedsl_logical_k_tokens=cutedsl_logical_k_tokens,
        cutedsl_use_page_table=cutedsl_use_page_table,
        cutedsl_partition_scores=cutedsl_partition_scores,
        cutedsl_partition_ids=cutedsl_partition_ids,
        cutedsl_packed_partitions=cutedsl_packed_partitions,
        cutedsl_packed_pages=cutedsl_packed_pages,
        cutedsl_packed_tile_metadata=cutedsl_packed_tile_metadata,
        cutedsl_packed_requests=cutedsl_packed_requests,
        _lean192_early_router=resolve_lean192(lean192, lean192_early_router),
        lean192_mode=lean192, overlap_fixed_score=overlap_fixed_score,
    )
    bind_score_operands(plan)
    return plan
