# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
# Modified for MBSA: package integration; see root NOTICE.

"""Import surface for the retained SM100 BF16 forward and CSR runtime.

Relative imports keep the bundled runtime inside its owning package without
registering generic ``src`` or ``interface`` modules in the application.
"""

from mbsa.cute.interface import SparseForwardPhase, sparse_atten_func
from mbsa.cute.sparse_index_utils import build_k2q_csr
from mbsa.cute.src.sm100.prepare_k2q_csr import SparseK2qCsrBuilderSm100, allocate_k2q_csr_scratch

__all__ = [
    "sparse_atten_func", "SparseForwardPhase", "build_k2q_csr",
    "SparseK2qCsrBuilderSm100", "allocate_k2q_csr_scratch",
]
