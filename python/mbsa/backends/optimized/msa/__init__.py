# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private lazy MSA prefill exports; no native Index scorer or FFI registry."""

def __getattr__(name):
    if name in {"SparseForwardPhase", "sparse_atten_func"}:
        from .cute import interface
        return getattr(interface, name)
    if name == "build_k2q_csr":
        from .cute.sparse_index_utils import build_k2q_csr
        return build_k2q_csr
    if name == "allocate_k2q_csr_scratch":
        from .cute.src.sm100.prepare_k2q_csr import allocate_k2q_csr_scratch
        return allocate_k2q_csr_scratch
    raise AttributeError(name)
