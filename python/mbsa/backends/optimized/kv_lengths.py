# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Run-time KV lengths of decode-like steps prepared at a KV capacity."""
from __future__ import annotations

from itertools import accumulate

import torch

from .._binding import explicit_device
from .index_policy import heads_in_rows_partition_counts, heads_in_rows_work_rows, index_sm_count
from .kernels.plan import SPEC

_BLOCK = SPEC.block_size
_INDEX_HEADS = SPEC.kv_heads


def index_work_rows(row_groups, capacities, kv_lens, counts, rows):
    """Flat work rows of an even-split heads-in-rows Index launch (see IndexerLocalTop16Sm100.even_split).

    Args:
        row_groups: Query rows (query tokens x Index heads) of each request, in packed order.
        capacities: KV tokens each request's pages were laid out for (its page span in the Index page map).
        kv_lens: Current KV tokens of each request.
        counts: Partitions (rows) of each request, at least one (heads_in_rows_partition_counts).
        rows: Rows of the launch; the rows no request uses stay zero.

    Returns:
        rows * 7 integers: (q_begin, q_end, 0, page_begin, kv_len, partitions, partition) per row. Row r of the
        first len(kv_lens) rows is the first partition of request r.
    """
    q_begin = list(accumulate(row_groups, initial=0))
    page_begin = list(accumulate(((n + _BLOCK - 1) // _BLOCK for n in capacities), initial=0))
    flat = []
    for partition in (0, 1):     # first partitions in request order, then the others
        for request, count in enumerate(counts):
            for index in range(partition, count if partition else 1):
                flat += (q_begin[request], q_begin[request + 1], 0, page_begin[request], kv_lens[request],
                         count, index)
    flat += [0] * (rows * 7 - len(flat))
    return flat


def attention_tiles(capacities, kv_lens):
    """Flat packed attention tiles of one-token requests: (request, 0, q, q + 1, k_begin, k_begin + kv_len)."""
    k_begin = list(accumulate(capacities, initial=0))
    return [value for request, kv_len in enumerate(kv_lens)
            for value in (request, 0, request, request + 1, k_begin[request], k_begin[request] + kv_len)]


class KvLengths:
    """Current KV lengths of the requests of a decode-like step whose plans were prepared at a KV capacity.

    Pass one object as kv_lengths to prepare_cutedsl_pipeline for every plan of the step (one per layer); the
    plans' request_lens then give each request's KV capacity. The plans share two small device tables, which
    set() rewrites: call it once per step, before the plans run or their captured graphs replay, on the
    stream that runs them. Lengths are host integers; the step starts at its capacities.
    """

    def __init__(self, device) -> None:
        """Create the lengths of one step on `device` (a bare "cuda" names the current device)."""
        self.device = explicit_device(device)
        self._capacities: tuple[int, ...] | None = None
        self._lengths: tuple[int, ...] | None = None
        self._rows = 0
        self._host: torch.Tensor | None = None        # int32 [rows * 7 + requests * 6]: both tables
        self._tables: torch.Tensor | None = None      # its device copy
        self._blocks: tuple[int, ...] | None = None   # KV blocks of the published lengths
        self._counts: list[int] | None = None         # partitions per request the Index rows are laid out for
        self._row_request: torch.Tensor | None = None  # request of every used Index row
        self._request_rows: torch.Tensor | None = None  # [requests,4]: the row columns a request never changes
        self._k_begin: torch.Tensor | None = None

    @property
    def capacities(self) -> tuple[int, ...] | None:
        """KV capacity of each request, fixed by the first plan prepared with this object."""
        return self._capacities

    @property
    def lengths(self) -> tuple[int, ...] | None:
        """KV tokens of each request that the plans read on their next run."""
        return self._lengths

    def _bind(self, capacities) -> None:
        """Fix the step's requests (first plan) or check that a further plan has the same ones."""
        capacities = tuple(int(n) for n in capacities)
        if self._capacities is not None:
            if capacities != self._capacities:
                raise ValueError("plans sharing kv_lengths must be prepared for the same requests and "
                                 "KV capacities")
            return
        requests = len(capacities)
        rows = heads_in_rows_work_rows(requests, sm_count=index_sm_count(self.device))
        host = torch.zeros((rows * 7 + requests * 6,), dtype=torch.int32)
        tables = torch.empty_like(host, device=self.device)
        k_begin = torch.tensor(list(accumulate(capacities, initial=0))[:-1], dtype=torch.int32)
        self._rows, self._host, self._tables, self._k_begin = rows, host, tables, k_begin
        # What set() never changes: a request's (q_begin, q_end, 0, page_begin) and its tile but for the end.
        first = index_work_rows((_INDEX_HEADS,) * requests, capacities, capacities, (1,) * requests, requests)
        self._request_rows = torch.tensor(first, dtype=torch.int32).view(requests, 7)[:, :4].clone()
        host[rows * 7:].copy_(torch.tensor(attention_tiles(capacities, capacities), dtype=torch.int32))
        # One tensor object per table: a plan recognises its operands by identity.
        self._index_view = tables[:rows * 7].view(rows, 7)
        self._tiles_view = tables[rows * 7:].view(requests, 6)
        self._capacities = capacities
        self._publish(capacities)

    def _unbind(self) -> None:
        """Forget the requests: the preparation that fixed them failed."""
        self._capacities = self._lengths = self._blocks = self._counts = None
        self._host = self._tables = self._row_request = self._request_rows = self._k_begin = None
        self._rows = 0

    def _index_rows(self, row_groups, rows) -> torch.Tensor:
        """The shared Index work rows [rows,7]; every plan must ask for this launch."""
        if tuple(row_groups) != (_INDEX_HEADS,) * len(self._capacities) or rows != self._rows:
            raise ValueError("plans sharing kv_lengths must bind the same Index launch")
        return self._index_view

    def _attention_tiles(self) -> torch.Tensor:
        """The shared attention tiles [requests,6]."""
        return self._tiles_view

    def _lay_out(self, index, counts) -> None:
        """Write the rows index_work_rows gives these partition counts, but for the lengths (column 4)."""
        requests = len(counts)
        others = torch.tensor(counts, dtype=torch.int64) - 1
        ids = torch.arange(requests)
        request = torch.cat((ids, torch.repeat_interleave(ids, others)))    # first partitions, then the others
        used = request.numel()
        index[used:] = 0
        index[:used, :4] = self._request_rows[request]
        index[:used, 5] = (others + 1)[request]
        index[:requests, 6] = 0
        # A request's other partitions are consecutive rows: 1, 2, ... after those of the earlier requests.
        index[requests:used, 6] = (torch.arange(1, used - requests + 1)
                                   - torch.repeat_interleave(torch.cumsum(others, 0) - others, others))
        self._row_request, self._counts = request, counts

    def _publish(self, lengths) -> None:
        requests, rows = len(lengths), self._rows
        index = self._host[:rows * 7].view(rows, 7)
        tiles = self._host[rows * 7:].view(requests, 6)
        blocks = tuple((n + _BLOCK - 1) // _BLOCK for n in lengths)
        if blocks != self._blocks:
            # The share of rows per request follows the KV blocks; the rows are laid out again only when a
            # share changes (a token more per request seldom moves one).
            counts = heads_in_rows_partition_counts(lengths, rows)
            if counts != self._counts:
                self._lay_out(index, counts)
            self._blocks = blocks
        values = torch.tensor(lengths, dtype=torch.int32)
        index[:self._row_request.numel(), 4] = values[self._row_request]
        tiles[:, 5] = self._k_begin + values
        self._lengths = tuple(lengths)
        self._tables.copy_(self._host)

    def set(self, kv_lens) -> None:
        """Publish the requests' current KV tokens to the plans prepared with this object.

        Args:
            kv_lens: One integer per request, in packed order, 1 <= kv_len <= the request's capacity. The
                page maps the plans bound must already map the pages of these lengths.

        Returns:
            None. The write is a host-to-device copy on the current CUDA stream: it returns once the work
            already queued on that stream has finished, and runs and replays submitted after it on that
            stream read the new lengths.
        """
        if self._capacities is None:
            raise RuntimeError("prepare a plan with this kv_lengths before set()")
        lengths = tuple(int(n) for n in kv_lens)
        if len(lengths) != len(self._capacities) or any(
                not 0 < n <= capacity for n, capacity in zip(lengths, self._capacities)):
            raise ValueError("kv_lens must hold one length per request, each between 1 and its capacity")
        self._publish(lengths)
