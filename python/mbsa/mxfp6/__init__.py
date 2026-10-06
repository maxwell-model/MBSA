# SPDX-License-Identifier: Apache-2.0
"""Kernels for the packed MXFP6 index-K cache (SM100).

Index K is Hadamard-rotated and stored as MXFP6-E2M3 codes with one UE8M0
scale per 32 elements, 100 bytes a token; index Q is rotated with the same
butterfly and quantized to MXFP8-E4M3. Pages are scored with ``tcgen05``
block-scaled MMAs. The format and its page layout are defined by
:mod:`mbsa.index_formats`; this package holds the kernels that produce and
read it:

* :mod:`.write`: Triton encoders, as standalone launchers and as
  ``@triton.jit`` device functions a fused kernel can call.
* :mod:`.scoring`: the JIT-built CUDA page scorers for decode and prefill.

Importing the package loads neither Triton nor a compiler.
"""

from .scoring import (
    INDEX_HEADS,
    MAX_DECODE_QUERY_LEN,
    MAX_DECODE_REQS,
    NUM_SMS,
    compile_extensions,
    decode_score,
    prefill_score,
    schedule_metadata,
    shared_schedule,
)

__all__ = [
    "INDEX_HEADS",
    "MAX_DECODE_QUERY_LEN",
    "MAX_DECODE_REQS",
    "NUM_SMS",
    "compile_extensions",
    "decode_score",
    "prefill_score",
    "schedule_metadata",
    "shared_schedule",
]
