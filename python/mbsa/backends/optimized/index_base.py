# Copyright 2026 Maxwell Team.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared Index state and submission shared by the optimized runtime."""
from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
import math
import torch
from .src.sm100.router import indexer_local_top16, indexer_packed_top16

# Field metadata of host-only plan state (host values, launch wrappers and caches): runtime._backing_tensors
# does not walk them, as every tensor they refer to is also held by another field.
HOST_ONLY = {"ownership": False}
# Router options of the BF16 2-SM Index specialization: Q256 CTAs paired by 2-SM UMMA, early
# accumulator release and the 32-column epilogue.
TWO_SM_ROUTER_OPTIONS = dict(_query_tile=256, _two_sm=True, _force_early_acc_release=True,
                             _chunked_epilogue=True)
# E4M3 single-SM Q256 rows with early accumulator release (IndexDispatch.fp8_q256).
FP8_Q256_ROUTER_OPTIONS = dict(_query_tile=256, _force_early_acc_release=True)
# The same E4M3 Q256 rows as 2-SM CTA pairs, full-row epilogue (IndexDispatch.fp8_two_sm, queued insertion).
FP8_TWO_SM_ROUTER_OPTIONS = dict(_query_tile=256, _two_sm=True, _force_early_acc_release=True)


def stage_int32(
    device: torch.device, *arrays: tuple[Sequence[int], tuple[int, ...]]
) -> tuple[torch.Tensor, ...]:
    """Stage host int32 arrays into shared aligned CUDA storage.

    Args:
        device: Destination CUDA device.
        arrays: Positional (host_values, shape) pairs. Values flatten to int32; returned shaped
            views share the staged device buffer.

    Returns:
        Tuple of device views with 16-byte aligned offsets; the copy is enqueued on the current
        stream.
    """
    flat: list[int] = []
    spans = []
    for values, shape in arrays:
        flat.extend([0] * ((-len(flat)) % 4))
        spans.append((len(flat), shape))
        flat.extend(values)
    host = torch.tensor(flat or [0], dtype=torch.int32, pin_memory=True)
    staged = host.to(device, non_blocking=True)
    return tuple(
        staged[offset : offset + math.prod(shape)].view(shape)
        for offset, shape in spans
    )



@dataclass
class IndexPipelineState:
    """Prepared native Index routing feeding fixed/dynamic attention.

    Calls sharing this plan must be ordered: streams, events and workspaces are reused.
    """

    index_q: torch.Tensor
    index_k: torch.Tensor
    index_page_indices: torch.Tensor
    candidate_storage: torch.Tensor
    prefix_lens: torch.Tensor
    attention: Any
    candidate_partition_blocks: int
    overlap_fixed_score: bool = False
    fixed_stream: torch.cuda.Stream | None = None
    inputs_ready: torch.cuda.Event | None = None
    fixed_done: torch.cuda.Event | None = None
    index_q_scales: torch.Tensor | None = None
    mxfp6_router: Any = None
    cutedsl_index_k: torch.Tensor | None = None
    cutedsl_index_pages: torch.Tensor | None = None
    cutedsl_logical_k_tokens: int | None = None
    cutedsl_use_page_table: bool = False
    cutedsl_partition_scores: torch.Tensor | None = None
    cutedsl_partition_ids: torch.Tensor | None = None
    cutedsl_packed_partitions: int = 1
    cutedsl_packed_pages: torch.Tensor | None = None
    cutedsl_packed_tile_metadata: torch.Tensor | None = None
    # Host (q_len, kv_len) per packed request; the 2-SM rows are staged on first bind.
    cutedsl_packed_requests: tuple[tuple[int, int], ...] | None = field(default=None, metadata=HOST_ONLY)
    cutedsl_packed_two_sm_tile_metadata: torch.Tensor | None = None
    # Packed router launch chosen at bind time for finer KV partitions or Q256 rows (else None: the
    # prepared PB1024 metadata and partition count above are used).
    cutedsl_packed_router_tile_metadata: torch.Tensor | None = None
    cutedsl_packed_router_partitions: int | None = None
    cutedsl_packed_fine_tile_metadata: dict | None = None
    # E4M3 single-SM Q256 + early-release router bound (IndexDispatch.fp8_q256).
    fp8_q256_bound: bool = False
    # Parallel Top-16 insertion bound (IndexDispatch.parallel_insert).
    parallel_insert_bound: bool = False
    # Queued Top-16 insertion (+ setmaxnreg register split on Q256) bound (IndexDispatch.queued_insert).
    queued_insert_bound: bool = False
    # E4M3 Q256 rows as 2-SM CTA pairs bound (IndexDispatch.fp8_two_sm).
    fp8_two_sm_bound: bool = False
    # Router compiled with a runtime KV partition count (IndexDispatch.runtime_partitions).
    runtime_partitions_bound: bool = False
    _lean192_early_router: bool = False
    skip_fixed_blocks_bound: bool = False
    two_sm_bound: bool = False

    _mega_graph: torch.cuda.CUDAGraph | None = field(
        default=None, init=False, repr=False
    )
    _mega_capture_stream: torch.cuda.Stream | None = field(
        default=None, init=False, repr=False
    )

    def run_score(self) -> None:
        """Submit Index scoring and local/global selection on the current CUDA stream.

        Uses this state's bound tensors, retained workspaces and selected format. Fixed-only plans
        skip scoring. Caller must preserve binding contents until queued work completes.

        Returns:
            None; candidate IDs and any partition scores/IDs are overwritten.
        """
        if self.attention.fixed_only:
            return
        if self.mxfp6_router is not None:
            self.mxfp6_router.run()
            return
        if self.cutedsl_index_k is None:
            raise RuntimeError("CuTeDSL router workspace was not prepared")
        compact_ids = self.candidate_storage.view(torch.int32)
        if self.cutedsl_packed_tile_metadata is not None:
            if self.cutedsl_packed_pages is None:
                raise RuntimeError("packed CuTeDSL page table is missing")
            indexer_packed_top16(
                self.index_q,
                self.cutedsl_index_k,
                partition_blocks=self.candidate_partition_blocks,
                page_indices=self.cutedsl_packed_pages,
                tile_metadata=self.cutedsl_packed_tile_metadata,
                partitions=self.cutedsl_packed_partitions,
                out_compact_ids=compact_ids,
                out_scores=self.cutedsl_partition_scores,
                out_ids=self.cutedsl_partition_ids,
                lean_pipeline=False,
            )
            return
        if self.cutedsl_index_pages is None or self.cutedsl_logical_k_tokens is None:
            raise RuntimeError("batch-one CuTeDSL router metadata is missing")
        if self.cutedsl_partition_scores is None:
            indexer_local_top16(
                self.index_q,
                self.cutedsl_index_k,
                partition_blocks=self.candidate_partition_blocks,
                page_indices=self.cutedsl_index_pages,
                logical_k_tokens=self.cutedsl_logical_k_tokens,
                use_page_table=self.cutedsl_use_page_table,
                _lean192_early=self._lean192_early_router,
                out_compact_ids=compact_ids,
            )
        else:
            if self.cutedsl_partition_ids is None:
                raise RuntimeError("CuTeDSL partition IDs were not prepared")
            indexer_local_top16(
                self.index_q,
                self.cutedsl_index_k,
                partition_blocks=self.candidate_partition_blocks,
                page_indices=self.cutedsl_index_pages,
                logical_k_tokens=self.cutedsl_logical_k_tokens,
                use_page_table=self.cutedsl_use_page_table,
                _lean192_early=self._lean192_early_router,
                out_scores=self.cutedsl_partition_scores,
                out_ids=self.cutedsl_partition_ids,
                out_compact_ids=compact_ids,
            )
        return
