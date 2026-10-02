"""Zero-overhead local substitute for FlashInfer's optional API logger.

The unified backend intentionally does not import FlashInfer's process-wide
logging and CUDA Graph monkey-patching infrastructure.  Production profiling
uses explicit NVTX ranges in the caller.
"""

from __future__ import annotations

from typing import Callable, TypeVar


F = TypeVar("F", bound=Callable)


def flashinfer_api(function: F) -> F:
    return function


__all__ = ["flashinfer_api"]

