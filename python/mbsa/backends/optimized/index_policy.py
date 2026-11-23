# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Index tuning decisions shared by preparation and the launch wrapper."""
import bisect
import contextlib
import contextvars
import dataclasses
import functools
import os

import torch


INDEX_DISPATCH_ENV = "MBSA_INDEX_DISPATCH"


@dataclasses.dataclass(frozen=True)
class IndexDispatch:
    """Index variant selection modes. Every variant is exact: routes, output and LSE do not change.

    Attributes:
        two_sm: BF16 256-row query tile with 2-SM UMMA: "0", "1" (wherever legal) or "auto".
        skip_fixed: Skip the Index MMA of prefix and local-window blocks: "0", "1" or "auto".
        partitions: Finer KV partitions for launches that leave SMs idle: "0" or "auto".
        fp8_q256: E4M3 256-row query tile outside the dense T=8192 geometry: "0" or "auto". Its
            spill-free chunked epilogue applies only when parallel and queued insertion are off.
        parallel_insert: Parallel Top-16 insertion on full-row epilogues.
        queued_insert: Queued Top-16 insertion on E4M3 routers.
        fp8_two_sm: E4M3 256-row tiles with the queued insertion as 2-SM CTA pairs: "0" or "auto".
        runtime_partitions: Compile the router with the KV partition count as a runtime extent, so the
            kernel depends on the partition width and merge form, not the count: serving warmup can then
            compile every kernel of the dispatch. False (one kernel per count) is for comparisons only.
        heads_in_rows: Packed steps of one-token requests (decode-like) score all Index heads of a token in
            one CTA, so a request's Index K is read once instead of once per head; the launch is a list of
            work rows shared among the requests (heads_in_rows_partition_counts), whatever `partitions`
            says. False (one CTA per head) is for comparisons only.
    """

    two_sm: str = "auto"
    skip_fixed: str = "auto"
    partitions: str = "auto"
    fp8_q256: str = "auto"
    parallel_insert: bool = True
    queued_insert: bool = True
    fp8_two_sm: str = "auto"
    runtime_partitions: bool = True
    heads_in_rows: bool = True

    def __post_init__(self):
        for name, allowed in (("two_sm", ("0", "1", "auto")), ("skip_fixed", ("0", "1", "auto")),
                              ("partitions", ("0", "auto")), ("fp8_q256", ("0", "auto")),
                              ("fp8_two_sm", ("0", "auto"))):
            if getattr(self, name) not in allowed:
                raise ValueError(f"IndexDispatch.{name} must be one of {allowed}")
        for name in ("parallel_insert", "queued_insert", "runtime_partitions", "heads_in_rows"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"IndexDispatch.{name} must be a bool")


# B300 (SM103) default: each variant inside its measured faster domain (see the policies below).
AUTO_INDEX_DISPATCH = IndexDispatch()
# Baseline selection: Q128 with the qualified dense E4M3 Q256 rules, Lean192, PB1024 / PB2048. Like the default
# it takes the partition count at run time: SM100 and MBSA_INDEX_DISPATCH=baseline serve from warmed-up kernels too.
BASELINE_INDEX_DISPATCH = IndexDispatch(two_sm="0", skip_fixed="0", partitions="0", fp8_q256="0",
                                        parallel_insert=False, queued_insert=False, fp8_two_sm="0")
# Per thread / task: an override in one serving thread never reaches plans bound by another.
_dispatch_override = contextvars.ContextVar("mbsa_index_dispatch_override", default=None)


@contextlib.contextmanager
def index_dispatch_override(dispatch):
    """Bind Index plans prepared inside the block, in this context, with `dispatch` on any device (tests and
    measurements)."""
    if not isinstance(dispatch, IndexDispatch):
        raise TypeError("dispatch must be an IndexDispatch")
    token = _dispatch_override.set(dispatch)
    try:
        yield dispatch
    finally:
        _dispatch_override.reset(token)


def index_dispatch_request():
    """The active override and MBSA_INDEX_DISPATCH value: resolve_index_dispatch's inputs besides the device."""
    return _dispatch_override.get(), os.environ.get(INDEX_DISPATCH_ENV, "auto")


def resolve_index_dispatch(device, request=None):
    """Select the Index variant modes for a plan bound on `device`.

    Args:
        device: Device of the Index operands.
        request: index_dispatch_request() result; None reads it.

    Returns:
        IndexDispatch. MBSA_INDEX_DISPATCH="auto" (default) selects AUTO_INDEX_DISPATCH on SM103 and
        BASELINE_INDEX_DISPATCH on other CUDA devices, whose power and SM budgets were not measured;
        "baseline" always selects BASELINE_INDEX_DISPATCH. index_dispatch_override takes precedence.
        Reads the environment when a plan is bound; performs no device work.
    """
    override, mode = index_dispatch_request() if request is None else request
    if mode not in ("auto", "baseline"):
        raise ValueError(f'{INDEX_DISPATCH_ENV} must be "auto" or "baseline"')
    if override is not None:
        return override
    device = torch.device(device)
    if mode == "baseline" or (device.type == "cuda" and torch.cuda.get_device_capability(device) != (10, 3)):
        return BASELINE_INDEX_DISPATCH
    return AUTO_INDEX_DISPATCH


def _memoized(policy):
    """Memoize a pure policy of (requests, **options): serving rebinds every layer of a step with the same requests."""
    cached = functools.lru_cache(maxsize=4096)(lambda requests, options: policy(requests, **dict(options)))

    @functools.wraps(policy)
    def memoized(requests, **options):
        return cached(tuple((int(q_len), int(kv_len)) for q_len, kv_len in requests), tuple(sorted(options.items())))
    memoized.cache_clear = cached.cache_clear
    memoized.cache_info = cached.cache_info
    return memoized


def _validate_index_token_extent(logical_k_tokens, partition_blocks):
    """Reject token counts that overflow the signed 32-bit partition ABI.

    Args:
        logical_k_tokens: Positive logical KV-token count for one request.
        partition_blocks: Supported number of 128-token blocks per partition.

    Returns:
        None. Raises ValueError before device work if partition rounding cannot
        fit in signed 32-bit arithmetic. Tensor allocation and compilation may
        impose smaller practical limits.
    """
    if partition_blocks not in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
        raise ValueError("partition_blocks must be 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, or 8192")
    limit = (1 << 31) - 128 * partition_blocks
    if not 0 < logical_k_tokens <= limit:
        raise ValueError(
            f"logical Index KV extent must be in [1, {limit}] tokens for "
            f"PB{partition_blocks}; partition rounding uses signed int32"
        )


def select_partition_blocks(index_q, index_k, *, batch, contiguous_unpacked,
                            kv_tokens, identity_pages, capability):
    """Select the number of 128-token candidate blocks in an Index partition.

    Args:
        index_q: Index queries [Sq,Hindex,D]; dtype and shape inform selection.
        index_k: Index keys, either dense [Skv,1,D] or contiguous physical-page
            storage [pages,1,128,D]; dtype, extent and layout inform selection.
        batch: Number of requests.
        contiguous_unpacked: Whether one request uses contiguous unpacked operands.
        kv_tokens: Logical KV-token count, including cached context.
        identity_pages: Whether logical and physical Index page IDs are known identical;
            None means that identity is not established.
        capability: GPU compute capability as a (major, minor) pair.

    Returns:
        Candidate blocks per partition: 1024 or 2048. Inputs are unchanged.

    Notes:
        The current 2048-block tuning policy selects matching E4M3FN Index operands
        with Q [8192,4,128], contiguous K [kv_tokens,1,128] or its equivalent
        [kv_tokens/128,1,128,128] view, and KV lengths of at least 8192 in
        8192-token increments. It requires one contiguous unpacked request,
        known identity pages and SM103. Other inputs select 1024 blocks. The
        exact query shape and KV granularity are performance-selection limits,
        not general Index correctness or hardware/ABI requirements. Preparation
        validates storage and request legality; the launch wrapper separately
        checks signed 32-bit partition arithmetic. The public attention API's
        per-request 1M KV limit is unchanged.
    """
    if (batch == 1 and contiguous_unpacked
            and index_q.dtype == index_k.dtype == torch.float8_e4m3fn
            and tuple(index_q.shape) == (8192, 4, 128)
            and (tuple(index_k.shape) == (kv_tokens, 1, 128)
                 or tuple(index_k.shape) == (kv_tokens // 128, 1, 128, 128))
            and index_k.is_contiguous()
            and kv_tokens >= 8192 and kv_tokens % 8192 == 0
            and identity_pages is True and capability == (10, 3)):
        return 2048
    return 1024


def lean192_index_eligible(q, k, *, partition_blocks, logical_k_tokens,
                           use_page_table):
    """Check the current selection policy for the BF16 192-thread Index specialization.

    Args:
        q: Index queries [Sq,Hindex,D].
        k: Canonical dense Index keys [Skv,1,D]. Preparation may supply the
            zero-copy view of contiguous [pages,1,128,D] storage.
        partition_blocks: Number of 128-token candidate blocks per Index partition.
        logical_k_tokens: Logical KV-token count, including cached context.
        use_page_table: Whether the launch resolves logical-to-physical Index page IDs
            through a page table rather than using direct addressing.

    Returns:
        Whether these operands satisfy this helper's current selection checks.
        This is not a complete input validator and performs no compilation or device work.

    Notes:
        The current performance-selection policy uses BF16 Index Q with Sq=8192,
        Hindex=4 and D=128, and KV lengths of at least 8192 tokens in 8192-token
        increments. There is no upper-context tuning ceiling. The exact Sq,
        lower KV bound and granularity are selection limits, not general Index
        correctness requirements. The launch wrapper separately rejects extents
        that overflow signed 32-bit partition arithmetic. Allocations and
        compilation must also fit the device; removing a tuning ceiling does
        not guarantee resource feasibility at every size.

        This helper also checks matching BF16 Q/K, K geometry [Skv,1,128],
        partition_blocks=1024, logical_k_tokens equal to the physical K extent,
        and direct addressing (use_page_table=False).

        Production preparation separately requires SM103, a single nonpacked
        contiguous request that needs Dynamic attention, identity Index pages, no
        packed request metadata,
        contiguous Index/attention operands on one device, dense rank-3 attention
        K/V, and BF16 attention Q/K/V and partials. A contiguous rank-4 Index K
        may use its canonical dense view only when physical and logical extents
        agree. Those dispatch prerequisites are checked by prepare_index,
        not by this helper. Preparation also owns input alignment and allocation
        lifetime validation. The public optimized attention backend retains its
        separate per-request 1M KV limit; the standalone Index wrapper can accept
        larger operands within its arithmetic and resource bounds.
    """
    return bool(
        q.dtype == k.dtype == torch.bfloat16
        and tuple(q.shape) == (8192, 4, 128)
        and tuple(k.shape[1:]) == (1, 128)
        and partition_blocks == 1024
        and logical_k_tokens >= 8192
        and logical_k_tokens % 8192 == 0
        and logical_k_tokens == int(k.shape[0]) and not use_page_table
    )


# Measured selection domain of the BF16 Q256 2-SM Index specialization (B300, 148 SMs, PB1024).
# The launch grid (query tiles x heads x KV partitions) is list-scheduled in grid order: Q128 CTAs
# one per SM, 2-SM CTA pairs on SM pairs, each costing its query rows x causal K-loop blocks. Under
# the power cap 2-SM UMMA moves that work 1.15..1.35x faster per SM, so it wins while its makespan
# stays within TWO_SM_MAX_MAKESPAN_RATIO of the Q128 one: every measured faster shape is at <= 1.18,
# every slower or tied one at >= 1.33 (one Q128 wave against a 2-SM wave of twice the rows, or a
# Q128 tail wave that the 2-SM launch cannot absorb). Below TWO_SM_MIN_KV_TOKENS the batch-one
# Lean192 kernel wins.
TWO_SM_MIN_KV_TOKENS = 12288
TWO_SM_MAX_MAKESPAN_RATIO = 1.25


# Every row attends the first _PREFIX_BLOCKS and its last _LOCAL_BLOCKS K blocks anyway; the makespan model counts
# only the blocks that can become dynamic candidates.
_PREFIX_BLOCKS = 8
_LOCAL_BLOCKS = 8


def _index_makespan(requests, *, query_tile, heads, partition_blocks, sm_count, pairs=True):
    """Greedy list-scheduling makespan of one Index launch, in query-row x K-block units.

    CTAs dispatch in grid order (query tile fastest, then head, then KV partition) to the earliest
    free slot. A CTA costs query_tile rows x its K-loop blocks: the tile's causal visible blocks inside
    its partition, without prefix and local-window blocks. With query_tile=256 a slot is an SM pair
    whose K loop follows the pair's later tile; requests are padded to whole pairs. pairs=False models
    single-SM Q256 CTAs (the E4M3 Q256 router) on sm_count slots. Memoized: serving rebinds every layer
    of a step with the same requests, and the schedule costs milliseconds of host time.
    """
    return _list_schedule(tuple((int(q_len), int(kv_len)) for q_len, kv_len in requests), int(query_tile),
                          int(heads), int(partition_blocks), int(sm_count), bool(pairs))


@functools.lru_cache(maxsize=4096)
def _list_schedule(requests, query_tile, heads, partition_blocks, sm_count, pairs):
    group = 2 if query_tile == 256 and pairs else 1
    visible, partitions = [], 1
    for q_len, kv_len in requests:
        tiles = -(-q_len // (query_tile * group)) * group
        # Causal K blocks of the request's CTAs (CTA pairs), ascending with the tile.
        visible.append([(kv_len - q_len + min(q_len, (tile + 1) * query_tile) - 1) // 128 + 1
                        for tile in range(group - 1, tiles, group)])
        partitions = max(partitions, ((kv_len + 127) // 128 + partition_blocks - 1) // partition_blocks)

    def partition_runs(partition):
        # One head's CTAs of a partition in grid order as (cost, count) runs: per request, the CTAs without blocks
        # in the partition, then the ones whose K loop ends inside it, then the ones covering it.
        begin, end = max(partition * partition_blocks, _PREFIX_BLOCKS), (partition + 1) * partition_blocks
        runs = []
        for blocks in visible:
            first = bisect.bisect_right(blocks, begin + _LOCAL_BLOCKS)
            full = bisect.bisect_left(blocks, end + _LOCAL_BLOCKS, first)
            for cost, count in [((b - _LOCAL_BLOCKS - begin) * query_tile, 1) for b in blocks[first:full]] + (
                    [((end - begin) * query_tile, len(blocks) - full)] if full < len(blocks) else []):
                if runs and runs[-1][0] == cost:
                    runs[-1] = (cost, runs[-1][1] + count)
                else:
                    runs.append((cost, count))
        return runs

    groups = []   # consecutive partitions with the same runs: [runs, partitions]
    for partition in range(partitions):
        runs = partition_runs(partition)
        if groups and groups[-1][0] == runs:
            groups[-1][1] += 1
        else:
            groups.append([runs, 1])
    slots = sm_count // group
    loads = [0] * slots   # ascending
    for runs, repeat in groups:
        # Every head of every partition in the group repeats the runs.
        runs = [(runs[0][0], runs[0][1] * heads * repeat)] if len(runs) == 1 else runs * (heads * repeat)
        for cost, count in runs:
            if count == 1 and loads[-1] - loads[0] <= cost:
                loads.append(loads.pop(0) + cost)   # the least loaded slot then carries the largest load
                continue
            while count:
                if loads[-1] - loads[0] <= cost:
                    # A CTA on the least loaded slot then reaches every other load: whole rounds add one CTA
                    # to every slot, the rest go to the least loaded slots in order.
                    rounds, rest = divmod(count, slots)
                    if rounds:
                        loads = [load + rounds * cost for load in loads]
                    if rest:
                        loads = loads[rest:] + [load + cost for load in loads[:rest]]
                    break
                bisect.insort(loads, loads.pop(0) + cost)
                count -= 1
    return loads[-1]


@_memoized
def _index_load_floor(requests, *, query_tile, heads, sm_count, pairs=True):
    """Lower bound of every partition width's makespan: the launch's work spread evenly over the slots.

    A CTA (pair) whose causal K loop covers b blocks works on b - _PREFIX_BLOCKS - _LOCAL_BLOCKS of them across all
    partitions (prefix and local window excluded), whatever the partition width.
    """
    group = 2 if query_tile == 256 and pairs else 1
    work = 0
    for q_len, kv_len in requests:
        tiles = -(-q_len // (query_tile * group)) * group
        for tile in range(group - 1, tiles, group):
            work += max(0, (kv_len - q_len + min(q_len, (tile + 1) * query_tile) - 1) // 128 + 1
                        - _PREFIX_BLOCKS - _LOCAL_BLOCKS)
    return -(-work * query_tile * heads // (sm_count // group))


@_memoized
def two_sm_index_faster(requests, *, heads=4, partition_blocks=1024, sm_count=148):
    """Check whether one 2-SM launch over these requests is inside the measured faster domain.

    Args:
        requests: Host (q_len, kv_len) pairs in launch order.
        heads: Index query heads.
        partition_blocks: Number of 128-token candidate blocks per Index partition.
        sm_count: Streaming multiprocessors of the device.

    Returns:
        Whether the 2-SM makespan is within TWO_SM_MAX_MAKESPAN_RATIO of the Q128 regular one. Pure
        host arithmetic; correctness does not depend on this choice.
    """
    if partition_blocks != 1024:
        return False
    args = dict(heads=heads, partition_blocks=partition_blocks, sm_count=sm_count)
    regular = _index_makespan(requests, query_tile=128, **args)
    return regular > 0 and (_index_makespan(requests, query_tile=256, **args)
                            <= TWO_SM_MAX_MAKESPAN_RATIO * regular)


# Finer KV partitions for grids that leave SMs idle (B300). A partition is an independent K range; the merge
# keeps each row's Top-16 by (score, smaller id), so any split selects the same blocks. Short query extents
# with long KV launch fewer PB1024 CTAs than SMs; a finer width is taken only when it shortens the estimated
# makespan to FINE_PARTITION_MAX_MAKESPAN of the PB1024 one, a margin that covers the extra CTA prologues and
# the partition merge, which the model leaves out. Widths 128 and 64 serve tiny query extents. Merging more
# than FINE_PARTITION_MAX_PARTITIONS partitions per row costs more than the finer split gains.
FINE_PARTITION_BLOCKS = (512, 256, 128, 64)
FINE_PARTITION_MAX_MAKESPAN = 0.85
FINE_PARTITION_MAX_WAVES = 4
FINE_PARTITION_MAX_PARTITIONS = 32
# A packed launch runs every request's tiles over the longest request's partitions (shorter requests' extra CTAs
# exit at once) and merges that many partitions for every row; the makespan ignores both. A finer width is taken
# only while at most FINE_PARTITION_MAX_EMPTY_SHARE of its CTAs are empty: (8, 1M) + (8184, 8K) at PB256 launches
# 8032 empty of 8320 CTAs and is 3.7..5.5 % slower than PB1024; (3000, 32K) + (5192, 200K) at PB256 (288 of 952)
# is 6..15 % faster.
FINE_PARTITION_MAX_EMPTY_SHARE = 0.5
# Merges of up to SERIAL_MERGE_MAX_PARTITIONS partitions walk them serially; more take the tree merge.
SERIAL_MERGE_MAX_PARTITIONS = 16
# Heads-in-rows launches (IndexDispatch.heads_in_rows) are a list of work rows, one CTA each: a row is one
# partition of one request, and the router divides the request's own KV blocks equally among its partitions at
# run time (IndexerLocalTop16Sm100.even_split). Their time is the Index K read, not the UMMA: the launch has
# the fewest rows that give every SM this many CTAs, so that it reads K at the device's memory bandwidth. B300
# (148 SMs), 32 requests: 128 CTAs already do; 64 CTAs read 256K tokens per request 1.4x slower; 256 or 512
# are no faster and only lengthen the merge. The rows a request gets follow the KV lengths of the step, not
# the lengths the launch was prepared for, so one captured launch serves every length up to its capacity.
HEADS_IN_ROWS_CTAS_PER_SM = 0.85
# The compiled partition width is unused by an even split; one value keeps one kernel.
EVEN_SPLIT_PARTITION_BLOCKS = 16
# A small batch leaves a request more rows than it has use for: every partition of a row is one more list
# for the merge to fold (0.43 us each on B300), and a CTA scans a block in 0.36 us after 3 us of start-up. One
# request, B300: 8K KV tokens in 64 partitions 10.1 us (router 3.6 + merge 6.5), in 32 about 8.6; 16K and 32K
# in 126 partitions 14 and 14.9 us, in 32 about 9.3 and 10.8. Long KV keeps every row (1M: 65 blocks each).
HEADS_IN_ROWS_MERGE_PARTITIONS = 32
HEADS_IN_ROWS_PARTITION_MIN_BLOCKS = 8


def heads_in_rows_work_rows(batch, *, sm_count=148):
    """Work rows (CTAs) of a heads-in-rows launch of `batch` requests.

    A whole number of rows per request (equal KV lengths then get equal shares): the fewest that put
    HEADS_IN_ROWS_CTAS_PER_SM CTAs on every SM, but never more rows than SMs while a request can keep a row
    of its own. A row beyond the SM count waits for an SM, which costs more than the SMs a smaller whole
    number leaves idle (B300, 256K KV per request: 25 requests take 404 us with 150 rows and 295 us with
    125; 50 requests 801 us with 150 and 621 us with 100). Pure host arithmetic.
    """
    budget = -int(-HEADS_IN_ROWS_CTAS_PER_SM * sm_count // 1)
    per_request = max(1, -(-budget // batch))
    if batch * per_request > sm_count:
        per_request = max(1, sm_count // batch)
    return batch * per_request


def heads_in_rows_partition_counts(kv_lens, rows):
    """Partitions (work rows) per request for the given KV lengths.

    One each; the other rows in proportion to a request's KV blocks beyond its first (largest remainders first,
    ties to the earlier request), which equalises the blocks per partition. A request takes at most one
    partition per KV block, and beyond HEADS_IN_ROWS_MERGE_PARTITIONS only partitions of at least
    HEADS_IN_ROWS_PARTITION_MIN_BLOCKS blocks; the rows it leaves go to the other requests in the same
    proportion. Partitions are equal multiples of a block, so a request keeps only as many as hold one;
    some rows may stay unused. Pure host arithmetic.
    """
    blocks = [(int(n) + 127) // 128 for n in kv_lens]
    limit = [min(count, max(HEADS_IN_ROWS_MERGE_PARTITIONS, -(-count // HEADS_IN_ROWS_PARTITION_MIN_BLOCKS))) - 1
             for count in blocks]
    extra = [0] * len(blocks)
    open_ = [i for i, value in enumerate(limit) if value]
    spare = rows - len(blocks)
    while open_ and spare > 0:
        total = sum(blocks[i] - 1 for i in open_)
        full = [i for i in open_ if (blocks[i] - 1) * spare >= limit[i] * total]
        if not full:
            for i in open_:
                extra[i] = (blocks[i] - 1) * spare // total
            order = sorted(open_, key=lambda i: (-((blocks[i] - 1) * spare % total), i))
            for i in order[:spare - sum(extra[i] for i in open_)]:
                extra[i] += 1
            break
        # These requests reach their limit: fix them and share the rest among the others.
        for i in full:
            extra[i] = limit[i]
            spare -= limit[i]
        open_ = [i for i in open_ if i not in full]
    # The router gives every partition ceil(blocks / partitions) blocks: drop the ones that would get none.
    return [-(-count // -(-count // (1 + value))) for count, value in zip(blocks, extra)]


# Batch-one plans, and packed plans whose KV fits 8192 tokens, run Fixed attention beside the Index on a side
# stream. A PB1024 launch that leaves SMs idle then hands them to Fixed attention, and a finer width that fills
# them delays it: beside Fixed attention a finer width is kept only when it shortens the makespan by at least
# the Fixed span, FIXED_ATTENTION_ROW_COST makespan units per query row spread over the SMs and at least
# FIXED_ATTENTION_MIN_SPAN (Fixed attention alone: 22.5 us at T=17, 37 / 78 / 140 / 285 us at
# T=1024 / 2048 / 4096 / 8192; a PB1024 makespan unit is 2.6..3.6 ns). Measured: (2048, 16K) BF16 at PB64 is
# 33 % faster alone but its layer 8.9 % slower (the PB1024 Index, 49 us, hides behind 78 us of Fixed
# attention); (17, 256K) BF16 at PB64 is 89 % faster alone and 73 % per layer.
FIXED_ATTENTION_ROW_COST = 1600
FIXED_ATTENTION_MIN_SPAN = 6500


@_memoized
def index_launch_shape(requests, *, two_sm, heads=4, sm_count=148, pairs=True, beside_fixed=False):
    """Choose the 2-SM kernel and the KV partition width for one Index launch.

    Args:
        requests: Host (q_len, kv_len) pairs; one pair for the batch-one router.
        two_sm: "0" Q128 only, "1" 2-SM only, "auto" whichever two_sm_index_faster prefers.
        heads: Index query heads.
        sm_count: Streaming multiprocessors of the device.
        pairs: True for the BF16 2-SM kernel; False when the 256-row option is the single-SM E4M3 Q256
            router, compared with FP8_Q256_MAX_MAKESPAN_RATIO.
        beside_fixed: Whether Fixed attention over the same query rows runs beside the launch.

    Returns:
        (two_sm, partition_blocks): each kernel takes its finest width in FINE_PARTITION_BLOCKS that
        reaches FINE_PARTITION_MAX_MAKESPAN of its PB1024 makespan with at most
        FINE_PARTITION_MAX_EMPTY_SHARE empty CTAs, else 1024; "auto" then picks 2-SM when its makespan is
        within TWO_SM_MAX_MAKESPAN_RATIO of the Q128 one. Beside Fixed attention a finer width that does not
        shorten its kernel's PB1024 makespan by the Fixed span falls back to the PB1024 choice. Pure host
        arithmetic.
    """
    def best(query_tile, fine):
        """(width, makespan, PB1024 makespan) of one kernel."""
        args = dict(query_tile=query_tile, heads=heads, sm_count=sm_count, pairs=pairs)
        width, span = 1024, _index_makespan(requests, partition_blocks=1024, **args)
        whole = span
        # Grids with at least FINE_PARTITION_MAX_WAVES waves of PB1024 CTAs keep PB1024: a finer split only
        # trims their tail, which the model overestimates there (and the estimate grows with the grid).
        ctas = heads * sum(-(-q_len // query_tile) * -(-((kv_len + 127) // 128) // 1024)
                           for q_len, kv_len in requests)
        if not fine or span == 0 or ctas >= FINE_PARTITION_MAX_WAVES * sm_count:
            return width, span, whole
        if _index_load_floor(requests, **args) > FINE_PARTITION_MAX_MAKESPAN * span:
            return width, span, whole   # no width reaches the threshold: skip the schedules
        group = 2 if query_tile == 256 and pairs else 1
        tiles = [-(-q_len // (query_tile * group)) * group for q_len, _ in requests]
        for blocks in FINE_PARTITION_BLOCKS:
            own = [-(-((kv_len + 127) // 128) // blocks) for _, kv_len in requests]
            if max(own) > FINE_PARTITION_MAX_PARTITIONS:
                break
            if max(own) == 1:
                continue   # no split: the PB1024 schedule
            if sum(tiles) * max(own) - sum(t * p for t, p in zip(tiles, own)) > (
                    FINE_PARTITION_MAX_EMPTY_SHARE * sum(tiles) * max(own)):
                break   # finer widths only leave more CTAs empty
            fine_span = _index_makespan(requests, partition_blocks=blocks, **args)
            if fine_span <= FINE_PARTITION_MAX_MAKESPAN * span:
                width, span = blocks, fine_span
        return width, span, whole

    def choose(fine):
        regular = best(128, fine) if two_sm != "1" else None
        ratio = TWO_SM_MAX_MAKESPAN_RATIO if pairs else FP8_Q256_MAX_MAKESPAN_RATIO
        if two_sm == "auto" and regular[1] > 0 and _index_load_floor(
                requests, query_tile=256, heads=heads, sm_count=sm_count, pairs=pairs) > ratio * regular[1]:
            return False, regular   # no 256-row width reaches the ratio: skip its schedules
        paired = best(256, fine) if two_sm != "0" else None
        if regular is None or (paired is not None and regular[1] > 0 and paired[1] <= ratio * regular[1]):
            return True, paired
        return False, regular

    wide, (width, span, whole) = choose(True)
    if beside_fixed and width != 1024 and whole - span < max(
            FIXED_ATTENTION_MIN_SPAN, FIXED_ATTENTION_ROW_COST * sum(q for q, _ in requests) // sm_count):
        wide, (width, span, whole) = choose(False)
    return wide, width


def two_sm_index_preferred(q, k, *, partition_blocks, logical_k_tokens):
    """Check the current performance-selection policy for the 2-SM Index specialization.

    Args:
        q: Index queries [Sq,Hindex,D].
        k: Canonical dense or page-flattened Index keys [Skv,1,D].
        partition_blocks: Number of 128-token candidate blocks per Index partition.
        logical_k_tokens: Logical KV-token count, including cached context.

    Returns:
        Whether BF16 operands fall inside the measured faster domain: KV of at least
        TWO_SM_MIN_KV_TOKENS and two_sm_index_faster on the device's SM count (148 for non-CUDA
        tensors). Correctness does not depend on this choice; page tables and logical tails are
        supported. Performs no device work.
    """
    if not (q.dtype == k.dtype == torch.bfloat16
            and tuple(q.shape[1:]) == (4, 128)
            and tuple(k.shape[1:]) == (1, 128)
            and partition_blocks == 1024
            and logical_k_tokens >= TWO_SM_MIN_KV_TOKENS):
        return False
    return two_sm_index_faster(((int(q.shape[0]), logical_k_tokens),), heads=int(q.shape[1]),
                               partition_blocks=partition_blocks, sm_count=index_sm_count(q.device))


def index_sm_count(device):
    """SM count of a CUDA device; 148 (B300) for other devices such as meta tensors."""
    device = torch.device(device)
    if device.type != 'cuda':
        return 148
    return _cuda_sm_count(torch.cuda.current_device() if device.index is None else device.index)


@functools.lru_cache(maxsize=None)
def _cuda_sm_count(index):
    return torch.cuda.get_device_properties(index).multi_processor_count


# Measured domain of the fixed-block skip (B300, PB1024 BF16 / PB2048 E4M3). The 2-SM and Lean192
# kernels gain at every measured KV. The Q128 regular and E4M3 Q256 kernels gain up to
# SKIP_FIXED_MAX_KV_TOKENS; beyond it the skipped prefix/local UMMA is below 0.1 % of the work and
# the changed loop bounds move register spills, costing 0.2..1.2 % at 384K..1M.
SKIP_FIXED_MAX_KV_TOKENS = 262144


def skip_fixed_blocks_preferred(*, two_sm, lean192, logical_k_tokens):
    """Check the performance-selection policy for the exact fixed-block skip.

    Args:
        two_sm: Whether the 2-SM Index specialization is being bound.
        lean192: Whether the Lean192 early router is being bound.
        logical_k_tokens: Logical KV-token count, including cached context.

    Returns:
        Whether the skip is inside its measured faster domain. Correctness does not depend on this
        choice. Performs no device work.
    """
    return two_sm or lean192 or logical_k_tokens <= SKIP_FIXED_MAX_KV_TOKENS


def packed_two_sm_index_preferred(requests, *, partition_blocks, sm_count=148):
    """Check the performance-selection policy for the 2-SM specialization of packed requests.

    Args:
        requests: Host (q_len, kv_len) pairs in packed order; BF16 Index operands are checked by the
            caller. Each request is padded to whole 512-row CTA pairs.
        partition_blocks: Number of 128-token candidate blocks per Index partition.
        sm_count: Streaming multiprocessors of the device.

    Returns:
        two_sm_index_faster for the shared launch. The packed Q128 baseline has no Lean192 kernel,
        so there is no KV floor. Correctness does not depend on this choice. Performs no device work.
    """
    return two_sm_index_faster(tuple(requests), partition_blocks=partition_blocks, sm_count=sm_count)


# E4M3 Index Q256 geometry outside the dense T=8192 range (B300, 1100 W). The baseline router binds
# its Q256 + early-release kernel only for dense identity T=8192 operands; page tables, logical tails and
# other query extents run Q128. Q256 halves the K bytes delivered per query row and is 1.2..1.35x faster
# per SM under the power cap (paged T=8192 128K/1M -25 %, T=4096 1M -18 %); like the 2-SM kernel it halves
# the CTA count, so it is taken when its single-SM makespan is within FP8_Q256_MAX_MAKESPAN_RATIO of Q128.
FP8_Q256_MAX_MAKESPAN_RATIO = 1.25


@_memoized
def fp8_q256_index_preferred(requests, *, heads=4, partition_blocks=1024, sm_count=148):
    """Check the performance-selection policy for the E4M3 Q256 router outside its dense geometry.

    Args:
        requests: Host (q_len, kv_len) pairs; one pair for the batch-one router.
        heads: Index query heads.
        partition_blocks: Number of 128-token candidate blocks per Index partition.
        sm_count: Streaming multiprocessors of the device.

    Returns:
        Whether the Q256 makespan is within FP8_Q256_MAX_MAKESPAN_RATIO of the Q128 one. Exact either way.
    """
    args = dict(heads=heads, partition_blocks=partition_blocks, sm_count=sm_count)
    regular = _index_makespan(requests, query_tile=128, **args)
    return regular > 0 and (_index_makespan(requests, query_tile=256, pairs=False, **args)
                            <= FP8_Q256_MAX_MAKESPAN_RATIO * regular)


# E4M3 Q256 rows as 2-SM CTA pairs (B300), with the queued insertion only. The pair splits every K tile between
# its two SMs, so each SM receives half the K bytes of the single-SM Q256 router for the same rows; under the
# power cap the saved delivery energy is time (2.6..5 % from 32K KV and 256-block partitions, not below 16K). Each
# partition starts with blocks that every row inserts (the worst case for the pair), so 2-SM stops at
# FP8_TWO_SM_MIN_PARTITION_BLOCKS. The pair
# launch pads a request to whole 512-row pairs and streams the pair's later causal tile, so it is taken while that
# schedule stays within FP8_TWO_SM_MAX_MAKESPAN_RATIO of the single-SM Q256 one, for KV from FP8_TWO_SM_MIN_KV_TOKENS.
FP8_TWO_SM_MIN_KV_TOKENS = 32768
FP8_TWO_SM_MIN_PARTITION_BLOCKS = 256
FP8_TWO_SM_MAX_MAKESPAN_RATIO = 1.02


@_memoized
def fp8_two_sm_index_preferred(requests, *, heads=4, partition_blocks=1024, sm_count=148):
    """Check the performance-selection policy for the E4M3 2-SM router (Q256 rows in CTA pairs).

    Args:
        requests: Host (q_len, kv_len) pairs in launch order; one pair for the batch-one router.
        heads: Index query heads.
        partition_blocks: Number of 128-token candidate blocks per Index partition.
        sm_count: Streaming multiprocessors of the device.

    Returns:
        Whether the partitions hold at least FP8_TWO_SM_MIN_PARTITION_BLOCKS blocks, the longest request reaches
        FP8_TWO_SM_MIN_KV_TOKENS and the paired makespan is within FP8_TWO_SM_MAX_MAKESPAN_RATIO of the single-SM
        Q256 one. Exact either way; performs no device work.
    """
    if (partition_blocks < FP8_TWO_SM_MIN_PARTITION_BLOCKS
            or max(kv_len for _, kv_len in requests) < FP8_TWO_SM_MIN_KV_TOKENS):
        return False
    args = dict(query_tile=256, heads=heads, partition_blocks=partition_blocks, sm_count=sm_count)
    single = _index_makespan(requests, pairs=False, **args)
    return single > 0 and _index_makespan(requests, pairs=True, **args) <= FP8_TWO_SM_MAX_MAKESPAN_RATIO * single


# E4M3 Index chunked epilogue (B300). The E4M3 Q256 router reads each accumulator row in one x128 load-reduce and
# spills at its 168-register cap. The 32-column chunked epilogue removes the spills but adds three fences per
# block: more cycles, less energy. With serial insertion it pays where the accumulator is released early and the
# query extent is the full 8192-row chunk, from 16K to 256K KV. The parallel and queued insertions remove the
# spill pressure themselves, so the default dispatch (which binds them) does not take it.
FP8_CHUNKED_EPILOGUE_MIN_QUERY_ROWS = 8192
FP8_CHUNKED_EPILOGUE_MIN_KV_TOKENS = 16384
FP8_CHUNKED_EPILOGUE_MAX_KV_TOKENS = 262144


def fp8_chunked_epilogue_preferred(*, query_tile, logical_k_tokens, partition_blocks=1024,
                                   early_acc_release=True, query_rows=8192, parallel_insert=False):
    """Check the performance-selection policy for the E4M3 chunked epilogue.

    Args:
        query_tile: Query rows per CTA the router selected (the policy covers the Q256 geometry).
        logical_k_tokens: Logical KV-token count, including cached context.
        partition_blocks: KV blocks per router partition (measured with PB1024 and PB2048).
        early_acc_release: Whether the router releases the accumulator before the Top-16 insertion.
        query_rows: Index query rows of the launch.
        parallel_insert: Whether the router inserts with the parallel Top-16 network. The full-row epilogue
            with parallel insertion beats both chunked variants (B300: 1M 3.40 vs 3.59 / 4.17 ms, 128K 0.430
            vs 0.479 / 0.584 ms), so the chunked epilogue is never preferred with it. The router passes True
            for the queued insertion as well, which keeps the full-row load.

    Returns:
        Whether the chunked epilogue is inside its measured faster domain. Exact either way.
    """
    return (not parallel_insert and query_tile == 256 and early_acc_release and partition_blocks >= 1024
            and query_rows >= FP8_CHUNKED_EPILOGUE_MIN_QUERY_ROWS
            and FP8_CHUNKED_EPILOGUE_MIN_KV_TOKENS <= logical_k_tokens <= FP8_CHUNKED_EPILOGUE_MAX_KV_TOKENS)
