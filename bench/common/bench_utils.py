#!/usr/bin/env python3
"""Shared timing harness for the sparse_attention benchmarks under scripts/.

One place that defines *how* we time, so every model's numbers are comparable.
See docs/token_sparse/BENCHMARK.md and docs/block_sparse/BENCHMARK.md for the
full protocols. In short:

  * Reduction is **median primary, p10/p90 reported** (never best-of-min).
  * Every single-kernel sample uses one CUDA event pair around one invocation.
    INNER defaults to 1 for every stage so repeated launches cannot change cache
    residency or clock state inside a sample.
  * Chunked prefill uses `bench_pass`: one *pass* = the summed per-chunk kernel
    time (chunks timed with events, a single sync at pass end); PREFILL_REP passes
    -> median/p10/p90 across passes.

Every knob is env-overridable, but the canonical full-table runner passes the
same values to all three venvs in the delivery image.

    WARMUP / REP / INNER              -> decode & cheap single kernels
    PREFILL_WARMUP / PREFILL_REP      -> chunked prefill passes AND ms-scale
                                         single-call prefill (with inner=1)
"""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, List, Union

import torch

# --- standardized knobs (override via env) -------------------------------
WARMUP = int(os.environ.get("WARMUP", "20"))
REP = int(os.environ.get("REP", "30"))
INNER = int(os.environ.get("INNER", "1"))
# FA4's JIT ms-scale prefill kernel needs enough untimed calls for compilation
# and GPU clocks to settle. The canonical runner uses 20 for every stage.
PREFILL_WARMUP = int(os.environ.get("PREFILL_WARMUP", "20"))
PREFILL_REP = int(os.environ.get("PREFILL_REP", "30"))


def allocation_counter() -> int:
    """Return CUDA's cumulative allocation count, or -1 when unavailable."""
    return int(torch.cuda.memory_stats().get("allocation.all.allocated", -1))


@dataclass
class Stats:
    """median / p10 / p90 of a set of per-call latency samples (ms)."""

    median: float
    p10: float
    p90: float

    def fmt(self, w: int = 10, prec: int = 4) -> str:
        return (
            f"{self.median:>{w}.{prec}f} | {self.p10:>{w}.{prec}f} | "
            f"{self.p90:>{w}.{prec}f}"
        )

    @staticmethod
    def header(w: int = 10) -> str:
        return f"{'med(ms)':>{w}} | {'p10(ms)':>{w}} | {'p90(ms)':>{w}}"


def _percentile(sorted_vals: List[float], q: float) -> float:
    """Linear-interpolated percentile (numpy-compatible)."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def summarize(vals: List[float]) -> Stats:
    s = sorted(vals)
    return Stats(
        median=statistics.median(s),
        p10=_percentile(s, 0.10),
        p90=_percentile(s, 0.90),
    )


def bench_call(fn: Callable[[], object], warmup: int = WARMUP, rep: int = REP,
               inner: int = INNER) -> Stats:
    """Amortized per-call latency.

    `warmup` untimed calls, then `rep` samples where each sample wraps `inner`
    back-to-back calls in one event pair and divides by `inner`.  The canonical
    full-table protocol fixes inner=1 for every stage.
    """
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    vals: List[float] = []
    for _ in range(rep):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(inner):
            fn()
        e.record()
        torch.cuda.synchronize()
        vals.append(s.elapsed_time(e) / inner)
    return summarize(vals)


def bench_pass(pass_fn: Callable[[], Union[float, Dict[str, float]]],
               warmup: int = PREFILL_WARMUP, rep: int = PREFILL_REP
               ) -> Union[Stats, Dict[str, Stats]]:
    """For chunked prefill. `pass_fn()` runs one full pass and returns either a
    single summed-ms float, or a dict {section_name: summed_ms} (e.g. score/topk).
    Returns Stats or {name: Stats} accordingly, over `rep` passes.
    """
    for _ in range(warmup):
        pass_fn()
    torch.cuda.synchronize()
    samples = [pass_fn() for _ in range(rep)]
    if isinstance(samples[0], dict):
        return {k: summarize([s[k] for s in samples]) for k in samples[0]}
    return summarize(samples)  # type: ignore[arg-type]


class _Section:
    __slots__ = ("_start", "_end")

    def __init__(self, start: torch.cuda.Event, end: torch.cuda.Event):
        self._start = start
        self._end = end

    def __enter__(self) -> "_Section":
        self._start.record()
        return self

    def __exit__(self, *exc) -> bool:
        self._end.record()
        return False


class PassTimer:
    """Accumulate per-kernel CUDA event timings across a chunked pass, summed
    after a *single* sync (no per-chunk sync overhead).

        pt = PassTimer()
        for chunk in chunks:
            with pt.section("score"):
                score_kernel(...)
            with pt.section("topk"):
                topk_kernel(...)
        pt.finish()                 # one sync for the whole pass
        pt.totals()                 # -> {"score": ms, "topk": ms}
    """

    def __init__(self) -> None:
        self._pairs: Dict[str, List] = {}

    def section(self, name: str = "_") -> _Section:
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        self._pairs.setdefault(name, []).append((s, e))
        return _Section(s, e)

    def finish(self) -> None:
        torch.cuda.synchronize()

    def total(self, name: str = "_") -> float:
        return sum(s.elapsed_time(e) for s, e in self._pairs[name])

    def totals(self) -> Dict[str, float]:
        return {n: self.total(n) for n in self._pairs}

    def grand_total(self) -> float:
        return sum(self.total(n) for n in self._pairs)


def print_config(extra: str = "") -> None:
    """Emit the standardized timing-config banner every script shares."""
    print(
        f"# timing: median+p10/p90 | warmup={WARMUP} rep={REP} inner={INNER} "
        f"| prefill_warmup={PREFILL_WARMUP} prefill_rep={PREFILL_REP}"
        + (f" | {extra}" if extra else ""),
        flush=True,
    )
