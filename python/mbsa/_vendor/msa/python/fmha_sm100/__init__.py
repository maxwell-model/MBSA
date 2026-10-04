# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Lazy public exports for the retained BF16 CuTe-DSL sparse runtime."""

_SPARSE_LAZY_EXPORTS = frozenset({
    "sparse_atten_func", "SparseForwardPhase", "build_k2q_csr",
    "SparseK2qCsrBuilderSm100", "allocate_k2q_csr_scratch",
})
__all__ = sorted(_SPARSE_LAZY_EXPORTS)


def __getattr__(name):
    if name in _SPARSE_LAZY_EXPORTS:
        from . import sparse
        return getattr(sparse, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *_SPARSE_LAZY_EXPORTS})
