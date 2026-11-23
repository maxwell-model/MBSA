# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Paged prefill: Top-16 routing, fixed attention and dynamic attention.
Stage host lengths once per step; rebind operands for each layer.
Requests are split into chunks of at most 8192 queries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate
import math

import torch

Q_HEADS, KV_HEADS, HEAD_DIM, PAGE_SIZE = 32, 4, 128, 128
PREFIX_PAGES, DYNAMIC_PAGES, LOCAL_PAGES = 8, 16, 8
TOPK_PAGES = PREFIX_PAGES + DYNAMIC_PAGES + LOCAL_PAGES
MAX_Q_PER_REQUEST = 8192
MAX_KV_PER_REQUEST = 1 << 20
SUPPORTED_CAPABILITIES = ((9, 0), (10, 0), (10, 3))
INDEX_FORMATS = ("bf16", "fp16", "fp8", "mxfp6")

#: Largest number of dynamic partial states kept per query head.
_PARTIAL_SLOTS = 16
_ROUTER_PARTITION_PAGES = 1024
_MX_ROUTER_PARTITION_PAGES = 256


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def split_requests(
    request_lens: Sequence[tuple[int, int]],
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    """Split requests into at most MAX_Q_PER_REQUEST queries per piece.
    Return (pieces, request_rows) in query order; each piece retains its causal KV bound."""
    pieces: list[tuple[int, int]] = []
    rows: list[int] = []
    for row, (q_len, kv_len) in enumerate(request_lens):
        context = kv_len - q_len
        for start in range(0, q_len, MAX_Q_PER_REQUEST):
            end = min(q_len, start + MAX_Q_PER_REQUEST)
            pieces.append((end - start, context + end))
            rows.append(row)
    return tuple(pieces), tuple(rows)


def unsupported_reason(
    *,
    capability: tuple[int, int],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    topk_pages: int,
    prefix_pages: int,
    local_pages: int,
    kv_dtype: torch.dtype,
    index_format: str,
    softmax_scale: float,
    page_score: str = "max",
    shared_route: bool = False,
) -> str | None:
    """Return None for supported token-max, per-KV-head routing; otherwise explain why."""
    if tuple(capability) not in SUPPORTED_CAPABILITIES:
        return "prefill needs an SM90, SM100 or SM103 GPU"
    if tuple(capability) == (9, 0) and index_format == "mxfp6":
        return "MXFP6 Index requires native Blackwell MX MMA"
    if (num_q_heads, num_kv_heads, head_dim) != (Q_HEADS, KV_HEADS, HEAD_DIM):
        return "prefill needs 32 query and 4 KV heads of dimension 128 per rank"
    if page_size != PAGE_SIZE:
        return "prefill needs 128-token pages"
    if (topk_pages, prefix_pages, local_pages) != (
        TOPK_PAGES,
        PREFIX_PAGES,
        LOCAL_PAGES,
    ):
        return "prefill needs 32 pages a query: 8 prefix, 16 dynamic, 8 local"
    if kv_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        return "prefill needs a BF16 or FP8 E4M3FN KV cache"
    if kv_dtype == torch.float8_e4m3fn and tuple(capability) == (9, 0):
        return "FP8 KV requires native Blackwell FP8 attention"
    if index_format not in INDEX_FORMATS:
        return "prefill needs a BF16, FP16, FP8 or MXFP6 index cache"
    if not math.isclose(softmax_scale, HEAD_DIM**-0.5, rel_tol=1e-6):
        return "prefill needs a 1/sqrt(128) softmax scale"
    if page_score != "max" or shared_route:
        return "prefill routes each KV head by its own page maximum"
    return None


@dataclass(frozen=True)
class Limits:
    """Workspace limits: query tokens per step, KV tokens per request,
    request count and total KV tokens before splitting."""

    max_tokens: int
    max_kv_tokens: int
    max_requests: int
    max_total_kv_tokens: int

    def batch_reason(self, request_lens: Sequence[tuple[int, int]]) -> str | None:
        """Why a step's ``(q_len, kv_len)`` list cannot run; None when it can."""
        if not request_lens:
            return "the step has no prefill request"
        if len(request_lens) > self.max_requests:
            return "the step has more prefill requests than the workspace allows"
        total_q = total_kv = 0
        for q_len, kv_len in request_lens:
            if not 0 < q_len <= kv_len:
                return "a prefill request has no query tokens or exceeds its KV"
            if kv_len > min(self.max_kv_tokens, MAX_KV_PER_REQUEST):
                return "a request is longer than the workspace allows"
            total_q += q_len
            total_kv += kv_len
        if total_q > self.max_tokens:
            return "the step has more prefill tokens than the workspace allows"
        if total_kv > self.max_total_kv_tokens:
            return "the step's summed KV length exceeds the workspace"
        return None

    def workspace_bytes(
        self,
        *,
        partial_dtype: torch.dtype,
        index_format: str,
        capability: tuple[int, int] | None = None,
        backend: str = "default",
        kv_dtype: torch.dtype = torch.bfloat16,
    ) -> int:
        """Upper bound of the device memory one step allocates.

        Args:
            partial_dtype: Dynamic partial-output dtype, BF16 or E4M3FN.
            index_format: Index cache format: bf16, fp16, fp8 or mxfp6.
            capability: Optional GPU compute capability; None reads the current device.
            backend: "default" or "optimized"; selects the backend's private buffers.
            kv_dtype: Attention K/V cache dtype. With E4M3FN the optimized backend
                converts Q and the step's logical K/V pages to BF16.

        Returns:
            Bytes to reserve for one prepared step, including all of its layers.
        """
        from .api import _validate_backend
        _validate_backend(backend)
        if kv_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise ValueError("kv_dtype must be BF16 or FP8 E4M3FN")
        tokens = self.max_tokens
        max_kv = min(self.max_kv_tokens, MAX_KV_PER_REQUEST)
        pages = _ceil_div(max_kv, PAGE_SIZE)
        # Splitting adds at most one piece per 8192 queries, each with up to
        # one request's KV.
        extra = tokens // MAX_Q_PER_REQUEST
        requests = self.max_requests + extra
        rows = _ceil_div(self.max_total_kv_tokens + extra * max_kv, PAGE_SIZE)
        rows += requests
        if capability is None and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
        partition_pages = (
            64
            if capability in (None, (9, 0))
            else _MX_ROUTER_PARTITION_PAGES
            if index_format == "mxfp6"
            else _ROUTER_PARTITION_PAGES
        )
        partitions = _ceil_div(pages, partition_pages)
        slots = requests * _ceil_div(self.max_kv_tokens, PAGE_SIZE)
        bins = KV_HEADS * slots
        edges = tokens * KV_HEADS * DYNAMIC_PAGES
        # Adaptive replicas are at least 16, at most 128, and below twice
        # edges/(4*bins). Include padded page-table slots, not just used pages.
        counters = min(128 * bins, 16 * bins + _ceil_div(edges, 2)) * 4
        partial_bytes = torch.empty((), dtype=partial_dtype).element_size()
        per_token = (
            # Dynamic partial outputs and their LSE.
            _PARTIAL_SLOTS * Q_HEADS * (HEAD_DIM * partial_bytes + 8)
            # BF16 fixed and dynamic states of a packed step, and three LSEs.
            + 2 * Q_HEADS * HEAD_DIM * 2
            + 3 * Q_HEADS * 4
            # Route, edge ranks, per-head indices/ranks and inverse edges.
            + KV_HEADS * DYNAMIC_PAGES * (3 + 2 * KV_HEADS) * 4
            # Router partition scores and IDs, including a one-partition copy.
            + 2 * KV_HEADS * (partitions + 1) * DYNAMIC_PAGES * 4
            # Compact MXFP8 Index-Q and optional shared FP8 attention Q.
            + (KV_HEADS + Q_HEADS) * HEAD_DIM
            # Scheduler records: native, resized and reordered copies.
            + 3 * 4 * 6 * 4
        )
        per_row = KV_HEADS * 4 * 2 + 2 * 4 + 3 * 4 + 3 * 4 * 6 * 4
        total = (
            tokens * per_token
            + rows * per_row
            + slots * KV_HEADS * 24
            + counters
            + (tokens + requests * 2304) * KV_HEADS * HEAD_DIM * 4
            + (64 << 20)
        )
        if backend == "optimized":
            total += _optimized_extra_bytes(self, tokens, rows, partitions, kv_dtype)
        return total


# Bounds of the decode-like path: a request owns at most index_policy.HEADS_IN_ROWS_REQUEST_ROWS rows of a
# heads-in-rows launch; attention.query_outer_splits gives at most SMs / 4 states (devices of up to 300 SMs).
DECODE_INDEX_ROWS = 256
DECODE_SPLIT_STATES = 256


def _optimized_extra_bytes(limits: Limits, tokens: int, rows: int, partitions: int,
                           kv_dtype: torch.dtype) -> int:
    """Optimized-backend buffers beyond the default estimate.

    ``rows`` bounds the step's logical KV pages, pieces included."""
    from .backends.optimized.index_policy import FINE_PARTITION_MAX_PARTITIONS

    decode_requests = min(tokens, limits.max_requests)
    extra = (
        # Index partition scores and IDs beyond the prepared count (finer dispatch widths, MX routers).
        2 * KV_HEADS * max(0, FINE_PARTITION_MAX_PARTITIONS - partitions) * DYNAMIC_PAGES * 4 * tokens
        # Index candidate storage and Dynamic routes.
        + tokens * (KV_HEADS * Q_HEADS * 4 + KV_HEADS * DYNAMIC_PAGES * 4)
        # CSR histogram scratch: at most 4 warps per 128-query CTA group.
        + 4 * KV_HEADS * 4 * _ceil_div(tokens, 128) * rows
        # Decode-like steps: the partition buffers of a heads-in-rows Index launch (no request owns more than
        # DECODE_INDEX_ROWS partitions) and the split attention states (at most DECODE_SPLIT_STATES per step).
        + 2 * KV_HEADS * DYNAMIC_PAGES * 4 * (decode_requests * DECODE_INDEX_ROWS + decode_requests)
        + DECODE_SPLIT_STATES * Q_HEADS * (HEAD_DIM * 2 + 4)
    )
    if kv_dtype == torch.float8_e4m3fn:
        # BF16 copies of FP8 Q and of every piece's logical K/V pages. A request with
        # q queries adds ceil(q/8192)-1 pieces, at most (tokens-1)//8192 in a step.
        pieces = (tokens - 1) // MAX_Q_PER_REQUEST
        max_kv = min(limits.max_kv_tokens, MAX_KV_PER_REQUEST)
        pages = (_ceil_div(limits.max_total_kv_tokens + pieces * max_kv, PAGE_SIZE)
                 + limits.max_requests + pieces)
        extra += tokens * Q_HEADS * HEAD_DIM * 2
        extra += pages * 2 * KV_HEADS * PAGE_SIZE * HEAD_DIM * 2
    return extra


class PrefillStep:
    """One scheduler step's prefill geometry, prepared on its first layer."""

    def __init__(
        self,
        owner: Prefill,
        request_lens: tuple[tuple[int, int], ...],
        page_table: torch.Tensor,
        index_page_table: torch.Tensor,
    ) -> None:
        """Retain a serving step's lengths and page maps; preparation is lazy.

        Args:
            owner: Prefill runner providing device, partial dtype and backend.
            request_lens: Tuple of (Q,KV) token counts in packed query order; KV includes cached
                context.
            page_table: CUDA int32 attention page table [>=B,>=ceil(max_KV/128)], mapping logical
                pages to physical K/V cache pages. Storage is retained; order mutations before
                queued use.
            index_page_table: CUDA int32 Index page table with one row per request, or compact 1-D
                active IDs for optimized. Optimized requires adjacent contiguous active rows
                (16-byte base alignment for unpacked/MX, 4-byte for packed floating Index).
                Requests with Q>8192 are split into pieces; their map rows are copied once per
                step when the step is prepared, as for the default backend.

        Returns:
            None. Page-table views share caller storage; layer operands are bound on run().
        """
        self._owner = owner
        self.request_lens = request_lens
        self.total_q = sum(q_len for q_len, _ in request_lens)
        self.pieces, self._rows = split_requests(request_lens)
        self._page_table = page_table[: len(request_lens)]
        if owner.backend == "optimized" and index_page_table.ndim == 1:
            self._index_page_table = index_page_table
        else:
            self._index_page_table = index_page_table[: len(request_lens)]
        self.plan = None

    def run(
        self,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        index_q: torch.Tensor,
        index_k_cache: torch.Tensor,
        out: torch.Tensor,
        index_q_scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write one layer's BF16 output [total_q,32,128].
        Q: [total_q,32,128]; K/V: [pages,4,128,128], matching BF16 or E4M3FN.
        FP8 K/V uses FP8 QK/PV MMA with unit scales; BF16 Q is converted once
        each run into a shared FP8 buffer, or supply prequantized E4M3FN Q.
        Index Q: [total_q,4,128] BF16/FP16/E4M3, or MXFP8 bytes with int32 scales.
        Index K: [pages,128,128] BF16/FP16/E4M3, or [pages,128,100] uint8 MXFP6."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("the serving prefill runs eagerly, not under capture")
        if q.shape[0] != self.total_q:
            raise ValueError("q must hold exactly the staged prefill tokens")
        width = 100 if index_k_cache.dtype == torch.uint8 else HEAD_DIM
        index_k = index_k_cache.view(-1, 1, PAGE_SIZE, width)
        if self.plan is None:
            self.plan = self._prepare(
                index_q, index_k, q, key_cache, value_cache, out, index_q_scales
            )
        else:
            self.plan.bind_layer(
                index_q,
                index_k,
                q,
                key_cache,
                value_cache,
                out,
                index_q_scales=index_q_scales,
            )
        self.plan.run()
        return out

    def _prepare(self, index_q, index_k, q, k, v, out, index_q_scales):
        from .api import prepare_sm100, prepare_cutedsl_pipeline
        capability = torch.cuda.get_device_capability(q.device)
        if self._owner.backend == "optimized" and capability not in ((10, 0), (10, 3)):
            raise ValueError(self._owner.backend + " backend requires SM100 or SM103")
        if self._owner.backend == "optimized":
            from .backends.optimized.runtime import stage_int32
        else:
            from .cute.prefill import stage_int32

        lens = self.pieces
        batch = len(lens)
        page_table, index_page_table = self._page_table, self._index_page_table
        if batch != len(self.request_lens):
            # Pieces of one request share its page-table rows.
            (rows,) = stage_int32(q.device, (self._rows, (batch,)))
            page_table = page_table.index_select(0, rows)
            if self._owner.backend == "optimized":
                # Optimized binds adjacent active rows: gather a compact map.
                index_page_table = _piece_active_pages(
                    index_page_table, self.request_lens, lens, self._rows, stage_int32)
            else:
                index_page_table = index_page_table.index_select(0, rows)
        packed = {}
        if batch == 1:
            packed["logical_k_tokens"] = lens[0][1]
        else:
            q_prefix = list(accumulate((q_len for q_len, _ in lens), initial=0))
            k_prefix = list(accumulate((kv_len for _, kv_len in lens), initial=0))
            packed["cu_seqlens_q"], packed["cu_seqlens_k"] = stage_int32(
                q.device, (q_prefix, (batch + 1,)), (k_prefix, (batch + 1,))
            )
            packed["request_lens"] = lens
        if capability == (9, 0):
            return prepare_cutedsl_pipeline(
                index_q,
                index_k,
                q,
                k,
                v,
                index_page_indices=index_page_table,
                page_table=page_table,
                partial_dtype=self._owner.partial_dtype,
                out=out,
                backend=self._owner.backend,
                lean192=self._owner.lean192,
                **packed,
            )._owner
        return prepare_sm100(
            index_q,
            index_k,
            q,
            k,
            v,
            index_q_scales=index_q_scales,
            index_page_indices=index_page_table,
            page_table=page_table,
            partial_dtype=self._owner.partial_dtype,
            out=out,
            check_device_inputs=False,
            fixed_stream=None,
            backend=self._owner.backend,
            lean192=self._owner.lean192,
            **packed,
        )._owner


def _piece_active_pages(pages, request_lens, pieces, piece_rows, stage_int32):
    """Gather a compact 1-D active Index map for split requests.

    Request ``r`` owns one contiguous span of ``pages``: a 2-D row, exact 1-D
    active IDs, or a flattened padded row. A piece reads the first pages of its
    request's span.
    """
    counts = [_ceil_div(kv_len, PAGE_SIZE) for _, kv_len in request_lens]
    if pages.ndim == 2:
        width = pages.shape[1]
        if width < max(counts):
            raise ValueError("index_page_table rows must cover every request's pages")
        pages = pages.contiguous().view(-1)
        starts = [row * width for row in range(len(counts))]
    elif pages.numel() == sum(counts):
        starts = [0, *accumulate(counts)][:-1]
    elif pages.numel() % len(counts) == 0 and pages.numel() // len(counts) >= max(counts):
        width = pages.numel() // len(counts)
        starts = [row * width for row in range(len(counts))]
    else:
        raise ValueError("index_page_table must contain active pages or padded rows")
    positions = [starts[row] + page for row, (_, kv_len) in zip(piece_rows, pieces)
                 for page in range(_ceil_div(kv_len, PAGE_SIZE))]
    (positions,) = stage_int32(pages.device, (positions, (len(positions),)))
    return pages.index_select(0, positions)


class Prefill:
    """Run prefill layers serially on one device, with fixed attention on a side stream."""

    def __init__(
        self, device: torch.device, *, partial_dtype: torch.dtype | None = None,
        backend: str = "default",
        lean192: str = "auto",
    ) -> None:
        """Create a serial serving runner for one device and backend.

        Args:
            device: CUDA device accepted by torch.device; used for stage allocations and streams. A bare
                "cuda" is resolved once, here, to the current device.
            partial_dtype: BF16 or torch.float8_e4m3fn partial-output storage. None reads
                MBSA_FP8_PARTIAL_OUT (0=BF16, the default; 1=E4M3FN). Final output remains BF16.
            backend: Implementation selector: "default" (SM90/SM100/SM103) or "optimized"
                (SM100/SM103). Selection is fixed for the plan lifetime.
            lean192: Immutable auto/on/off policy for the optimized Indexer. Auto
                leaves the choice to the Index dispatch; off disables only Lean192. Forced
                on rejects requests outside its existing dense SM103 BF16 policy;
                in particular it does not enable Lean192 for paged serving.

        Returns:
            None. Device buffers and compilation are deferred until a step first runs.
        """
        from .api import _resolve_partial_dtype, _validate_backend
        from .backends._binding import explicit_device
        from .backends._index_options import validate_lean192

        _validate_backend(backend)
        validate_lean192(lean192, backend=backend)
        partial_dtype = _resolve_partial_dtype(partial_dtype)
        self.device = explicit_device(device)
        self.partial_dtype = partial_dtype
        self.backend = backend
        self._lean192 = lean192
        self._fixed_stream: torch.cuda.Stream | None = None

    @property
    def lean192(self) -> str:
        """Preparation policy fixed when this serial runner is constructed."""
        return self._lean192

    def fixed_stream(self) -> torch.cuda.Stream:
        if self._fixed_stream is None:
            self._fixed_stream = torch.cuda.Stream(device=self.device)
        return self._fixed_stream

    def stage(
        self,
        request_lens: Sequence[tuple[int, int]],
        page_table: torch.Tensor,
        index_page_table: torch.Tensor,
    ) -> PrefillStep:
        """Stage host (q_len,kv_len) pairs in packed query order.
        Page tables are int32 [>=B,>=pages] logical-to-physical mappings.
        Optimized also accepts compact 1-D active Index page IDs."""
        lens = tuple((int(q_len), int(kv_len)) for q_len, kv_len in request_lens)
        return PrefillStep(self, lens, page_table, index_page_table)


_RUNNERS: dict[tuple[torch.device, torch.dtype, str, str], Prefill] = {}


def runner(device: torch.device, partial_dtype: torch.dtype | None = None, *,
           backend: str = "default", lean192: str = "auto") -> Prefill:
    """Get the process-cached serial runner for a device, dtype and backend.

    Args:
        device: CUDA device accepted by torch.device; part of the cache key. A bare "cuda" names the
            current device at each call, so each GPU gets its own runner.
        partial_dtype: BF16 or torch.float8_e4m3fn partial-output storage. None reads
            MBSA_FP8_PARTIAL_OUT (0=BF16, the default; 1=E4M3FN). Final output remains BF16.
        backend: Implementation selector: "default" (SM90/SM100/SM103) or "optimized"
            (SM100/SM103). Selection is fixed for the plan lifetime.
        lean192: Requested optimized Index policy, also part of the runner cache
            identity. Non-auto values require backend="optimized".

    Returns:
        Shared Prefill instance for the resolved key. Use a separately constructed Prefill when
        independent runner state is needed.
    """
    from .api import _resolve_partial_dtype, _validate_backend
    from .backends._binding import explicit_device
    from .backends._index_options import validate_lean192

    _validate_backend(backend)
    validate_lean192(lean192, backend=backend)
    partial_dtype = _resolve_partial_dtype(partial_dtype)
    key = (explicit_device(device), partial_dtype, backend, lean192)
    if key not in _RUNNERS:
        _RUNNERS[key] = Prefill(key[0], partial_dtype=partial_dtype, backend=backend,
                               lean192=lean192)
    return _RUNNERS[key]


def warmup_request_sets(
    limits: Limits, *, index_format: str
) -> list[tuple[tuple[int, int], ...]]:
    """Return in-limit request geometries covering kernel variants and alignment boundaries."""
    # The combine kernel keeps 4 dynamic partials through 2560 tokens, 8
    # through 3072 and 16 above.
    aligned = [2048, 2560, 3072, 4096, 12288, 16384, 24576, 32768, 36864, 40960, 65536]
    aligned += [131072 * p for p in (1, 2)]
    # Each further 1024 pages adds a router partition.
    aligned += [131072 * p + PAGE_SIZE for p in range(1, 8)]
    lengths = sorted(set(aligned) | {n - 64 for n in aligned})
    sets: list[tuple[tuple[int, int], ...]] = []
    for kv_len in lengths:
        for q_len in (16, 17):
            sets.append(((q_len, kv_len),))
        sets.append(((16, kv_len), (17, kv_len)))
        if kv_len <= MAX_Q_PER_REQUEST:
            # Without cached context, a lone request combines without TMA.
            sets.append(((kv_len, kv_len),))
    for kv_len in (24576, 32768, 40960):
        sets.append(((MAX_Q_PER_REQUEST, kv_len),))
    # Steps above 8192 queries split one request into packed pieces.
    for q_len in (MAX_Q_PER_REQUEST + 16, 2 * MAX_Q_PER_REQUEST):
        sets.append(((q_len, max(q_len, 40960)),))
    # The default KV-outer index kernels compile per atomic replica count, a power of two from 16
    # to 128 that grows with queries per KV page: 160 and 320 queries over 2560 tokens select 32
    # and 64, the sets above only 16 and 128.
    sets += [((q_len, 2560),) for q_len in (160, 320)]
    # One-token requests (decode-like steps): the optimized backend runs them query-outer, in one
    # attention pass over fixed and routed blocks; a packed step binds its own Index launch.
    # A batch that leaves SMs without an attention tile splits its tiles, into a power of two that halves
    # as the batch doubles: the powers of two and the largest batch reach every split count (and its
    # combine kernel) and, when the limits allow such a batch, the unsplit pass.
    largest = min(limits.max_requests, 64)
    sets += [((1, 4096),) * batch for batch in sorted({1, 2, 4, 8, 16, 32, 64, largest}) if batch <= largest]
    if index_format == "mxfp6":
        # The MXFP6 router partitions every 256 pages, for any batch.
        sets += [((16, 32768 * p + PAGE_SIZE),) for p in range(1, 32)]
    seen = set()
    kept = []
    for lens in sets:
        if lens not in seen and limits.batch_reason(lens) is None:
            seen.add(lens)
            kept.append(lens)
    return kept


def warmup(
    prefill: Prefill,
    request_lens: Sequence[tuple[int, int]],
    *,
    index_format: str,
    kv_dtype: torch.dtype = torch.bfloat16,
    q_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Compile and execute a serving geometry using initialized dummy caches.

    Allocates temporary storage on prefill.device and submits CUDA work; use outside graph capture.
    It does not bind caller model tensors. With the optimized backend, the first call for a GPU and a
    BF16, FP16 or E4M3 Index format also compiles every Index router kernel the Index dispatch can bind for
    steps within the per-request KV limit (the default dispatch on SM103, the baseline selection on SM100
    or with MBSA_INDEX_DISPATCH=baseline), and an E4M3FN kv_dtype compiles the step's page conversion, so
    that steps of other geometries compile no Index router or page-conversion kernel at run time. The
    MXFP6 scorer compiles one kernel per partition count; warmup_request_sets lists a step for each.

    Args:
        prefill: Prefill runner whose backend, device and partial dtype select the programs.
        request_lens: Host sequence of (Q,KV) token lengths to prepare and execute.
        index_format: Index cache format: bf16, fp16, fp8 or mxfp6.
        kv_dtype: Attention K/V dtype, BF16 by default or E4M3FN on Blackwell.
        q_dtype: Attention Q dtype, BF16 by default; E4M3FN is allowed only with E4M3FN K/V.

    Returns:
        None. Compiled entries remain in the process caches; queued dummy execution is ordered on
        the current stream.
    """
    if kv_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("warmup KV dtype must be BF16 or FP8 E4M3FN")
    if q_dtype not in ((torch.bfloat16, torch.float8_e4m3fn)
                       if kv_dtype == torch.float8_e4m3fn else (torch.bfloat16,)):
        raise ValueError("warmup Q dtype must match the KV precision contract")
    device = prefill.device
    lens = tuple(request_lens)
    total_q = sum(q_len for q_len, _ in lens)
    width = _ceil_div(max(kv_len for _, kv_len in lens), PAGE_SIZE)
    pages = width
    index_pages = width + 1
    logical = torch.arange(width, dtype=torch.int32, device=device)
    table = logical.expand(len(lens), width).contiguous()
    index_table = logical.expand(len(lens), width).contiguous()
    if prefill.backend == "optimized" and len(lens) > 1:
        # Optimized binds a compact active Index map: padded rows of shorter requests leave gaps.
        index_table = torch.cat([logical[: _ceil_div(kv_len, PAGE_SIZE)] for _, kv_len in lens])
    blackwell = torch.cuda.get_device_capability(device)[0] >= 10
    shape = ((pages, KV_HEADS, PAGE_SIZE, 2 * HEAD_DIM) if blackwell else
             (pages, PAGE_SIZE, KV_HEADS, 2 * HEAD_DIM))
    kv = torch.zeros(
        shape, dtype=kv_dtype, device=device
    )
    k, v = kv.split(HEAD_DIM, dim=-1)
    if not blackwell:
        k, v = (t.permute(0, 2, 1, 3) for t in (k, v))
    q = torch.zeros(total_q, Q_HEADS, HEAD_DIM, dtype=q_dtype, device=device)
    scales = None
    if index_format == "mxfp6":
        index_k = torch.zeros(
            index_pages, PAGE_SIZE, 100, dtype=torch.uint8, device=device
        )
        index_q = torch.zeros(
            total_q, 2 * KV_HEADS, HEAD_DIM, dtype=torch.uint8, device=device
        )[:, :KV_HEADS]
        scales = torch.zeros(total_q, 2 * KV_HEADS, dtype=torch.int32, device=device)
    else:
        dtype = {
            "fp8": torch.float8_e4m3fn,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[index_format]
        index_k = torch.zeros(
            index_pages, PAGE_SIZE, HEAD_DIM, dtype=dtype, device=device
        )
        index_q = torch.zeros(total_q, KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    prefill.stage(lens, table, index_table).run(
        q,
        k,
        v,
        index_q=index_q,
        index_k_cache=index_k,
        out=torch.empty_like(q, dtype=torch.bfloat16),
        index_q_scales=scales,
    )
    if prefill.backend == "optimized" and index_format in ("bf16", "fp16", "fp8"):
        from .backends.optimized.index import compile_serving_index_variants

        compile_serving_index_variants(device, index_q.dtype, max_kv_tokens=MAX_KV_PER_REQUEST)
    if prefill.backend == "optimized" and kv_dtype == torch.float8_e4m3fn:
        # The dummy cache above holds exactly the step's pages, so the step converted it whole. A serving
        # cache is larger and converts the step's pages: run that conversion once on a two-page cache.
        from .backends.optimized.kernels.kv_pages import PagedKVConversion

        dummy_k, dummy_v = torch.zeros((2,) + tuple(kv.shape[1:]), dtype=kv_dtype, device=device).split(
            HEAD_DIM, dim=-1)
        PagedKVConversion.prepare(dummy_k, dummy_v, torch.zeros((1, 1), dtype=torch.int32, device=device), [1])
    if blackwell and prefill.backend != "optimized":
        from .cute.paged_fixed_maps import refresh_maps

        indices = logical.long()
        refresh_maps(table[0], (
            indices, indices, torch.empty_like(logical), torch.empty_like(logical),
            torch.empty_like(logical), None, logical,
        ))


def compiled_kernel_count() -> int:
    """Count entries currently held by the loaded CuTe launcher caches.

    Returns:
        Integer cache-entry count across supported loaded modules. This excludes Triton/native
        extension compilation and is not a count of GPU launches.
    """
    import sys

    modules = (
        "cute.sm90",
        "cute.src.sm100.router",
        "cute.src.sm100.mx_router",
        "cute.src.sm100.merge_vectorized",
        "cute.kvouter.build_kvouter_index",
        "cute.kvouter.sparse_fwd_kvouter",
        "cute.kvouter.sparse_fwd_kvouter_combine",
        "cute.kvouter.sparse_fwd_kvouter_load_balance_schedule",
    )
    modules += (
        "backends.optimized.src.sm100.router",
        "backends.optimized.src.sm100.mx_router",
        "backends.optimized.bsa.cute_dsl.sparse.bsa_attn_sm100_blk128",
        "backends.optimized.msa.cute.interface",
        "backends.optimized.msa.cute.src.sm100.fwd.combine",
        "backends.optimized.kernels.merge",
    )
    return sum(
        len(value)
        for name in modules
        if (module := sys.modules.get(f"{__package__}.{name}")) is not None
        for key, value in vars(module).items()
        if isinstance(value, dict) and (
            key.endswith("CACHE") or key.endswith("cache") or key == "_MERGE_KERNELS"
        )
    )


__all__ = [
    "KvLengths",
    "INDEX_FORMATS",
    "Limits",
    "PrefillStep",
    "Prefill",
    "compiled_kernel_count",
    "runner",
    "split_requests",
    "unsupported_reason",
    "warmup",
    "warmup_request_sets",
]


def __getattr__(name: str):
    """KvLengths: the run-time KV lengths of decode-like steps on the optimized backend, loaded on first use."""
    if name == "KvLengths":
        from .backends.optimized.kv_lengths import KvLengths
        return KvLengths
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
