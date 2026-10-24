# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""JIT-loaded CUDA C++ extension for the q2k -> k2q CSR builder.

This module compiles ``build_k2q_csr.cu`` on first import via
``torch.utils.cpp_extension.load`` and exposes ``run_build_k2q_csr``.
The extension is cached in ``~/.cache/torch_extensions/`` so subsequent
imports are cheap.

The kernel pipeline is tuned and verified for SM100; other
architectures are not supported.
"""

from __future__ import annotations

import glob
import hashlib
import os
from importlib import metadata
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_THIS_DIR, "build_k2q_csr.cu")


def _cuda_home_include():
    for var in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(var)
        if root and os.path.isdir(os.path.join(root, "include")):
            return os.path.join(root, "include")
    return "/usr/local/cuda/include" if os.path.isdir("/usr/local/cuda/include") else None


def _nvidia_wheel_include_dirs():
    """A shim include dir for CUDA headers that live in pip wheels, not CUDA_HOME.

    Containers built on the pip CUDA stack (``nvidia-cu13``, ``nvidia-cusparse``,
    ...) keep some headers under ``site-packages/nvidia/**/include`` while
    ``/usr/local/cuda/include`` carries only the toolkit's own.  ``ATen/cuda/
    CUDAContext.h`` pulls in ``cusparse.h``, so without help the extension fails
    to compile with a bare "No such file or directory".

    Adding the wheel dirs wholesale does not work: they also ship
    ``crt/host_runtime.h`` and friends, and ``torch.utils.cpp_extension`` places
    ``extra_include_paths`` *ahead* of ``-isystem $CUDA_HOME/include``.  A wheel
    whose patch level differs from the installed ``nvcc`` then shadows the
    matching runtime headers and cudafe's generated stubs stop compiling
    (``macro "__cudaLaunch" passed 2 arguments``).

    So symlink only the headers CUDA_HOME is actually missing into one shim
    directory and expose that.
    """
    try:
        import nvidia  # noqa: PLC0415 - optional, only present on pip CUDA stacks
    except ImportError:
        return []

    cuda_include = _cuda_home_include()
    if cuda_include is None:
        return []

    wheel_dirs = []
    for root in getattr(nvidia, "__path__", []):
        # Both the flat (nvidia/cu13/include) and per-component
        # (nvidia/cusparse/include) wheel layouts are in the wild.
        for pattern in ("*/include", "*/*/include"):
            wheel_dirs.extend(sorted(glob.glob(os.path.join(root, pattern))))
    wheel_dirs = [d for d in dict.fromkeys(wheel_dirs) if os.path.isdir(d)]
    if not wheel_dirs:
        return []

    import tempfile

    # One deterministic owner per missing header. Include the actual toolkit
    # and wheel sources in the identity; a shared HOME must not retain another
    # runtime's headers. Contents also distinguish in-place wheel updates.
    headers = {}
    for wheel_dir in wheel_dirs:
        for header in sorted(glob.glob(os.path.join(wheel_dir, "*.h"))):
            name = os.path.basename(header)
            if not os.path.exists(os.path.join(cuda_include, name)):
                headers.setdefault(name, os.path.realpath(header))
    if not headers:
        return []
    identity = hashlib.sha256(os.path.realpath(cuda_include).encode())
    toolkit_header = Path(cuda_include) / "cuda.h"
    if toolkit_header.exists():
        identity.update(toolkit_header.read_bytes())
    for name, header in sorted(headers.items()):
        identity.update((name + "\0" + header + "\0").encode())
        identity.update(Path(header).read_bytes())
    cache_root = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    shim = os.path.join(cache_root, "mbsa", "optimized",
                        "cuda_wheel_include_shim", identity.hexdigest())
    os.makedirs(shim, exist_ok=True)
    for name, header in headers.items():
        link = os.path.join(shim, name)
        if os.path.islink(link) and os.path.exists(link) and os.path.realpath(link) == header:
            continue
        # Publish atomically and repair stale/broken links. Concurrent callers
        # for this same identity publish the same target. Never follow a stale
        # link to write into another installation.
        fd, temporary = tempfile.mkstemp(prefix=".header-", dir=shim)
        os.close(fd)
        try:
            os.unlink(temporary)
            os.symlink(header, temporary)
            os.replace(temporary, link)
        finally:
            if os.path.lexists(temporary):
                os.unlink(temporary)
    return [shim]


def _cuda_arch_flag():
    """``-arch`` for the running device, falling back to the SM100 baseline.

    These kernels are plain CUDA C++ (no tcgen05), so the family target is
    enough -- no ``a`` suffix needed -- and matching the device avoids a PTX JIT
    at every process start on SM103 parts.
    """
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:  # noqa: BLE001 - no device visible at import time
        return "-arch=sm_100"
    if major < 10:
        return "-arch=sm_100"
    return f"-arch=sm_{major}{minor}"


def _template_stub_flags():
    """Keep template ``__global__`` stubs externally linked.

    CUDA 12.8 flipped ``-static-global-template-stub`` to ``true`` by default,
    which breaks the generated launch stubs for template kernels declared in an
    anonymous namespace (``__cudaLaunch was not declared in this scope``).  The
    repo's csrc JIT already pins this flag; the CSR builder needs it for the
    same reason.  The option does not exist before 12.8, so it is gated.
    """
    raw = torch.version.cuda or ""
    try:
        major, minor = (int(p) for p in raw.split(".")[:2])
    except ValueError:
        return []
    return ["-static-global-template-stub=false"] if (major, minor) >= (12, 8) else []


_extra_cflags = ["-O3"]
_extra_cuda_cflags = [
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    _cuda_arch_flag(),
    "--ptxas-options=-v",
    "--expt-relaxed-constexpr",
    "-I/usr/local/cuda/include/cccl",
    *_template_stub_flags(),
]

def _extension_identity():
    """Separate this family's source/compiler ABI from native MSA caches."""
    digest = hashlib.sha256(Path(_SRC).read_bytes())
    for name in ("torch", "cuda-bindings"):
        digest.update((name + "=" + metadata.version(name)).encode())
    digest.update(_cuda_arch_flag().encode())
    return digest.hexdigest()[:16]


_ext = load(
    name="mbsa_optimized_csr_" + _extension_identity(),
    sources=[_SRC],
    extra_cflags=_extra_cflags,
    extra_cuda_cflags=_extra_cuda_cflags,
    extra_include_paths=_nvidia_wheel_include_dirs(),
    verbose=False,
)


def run_build_k2q_csr(
    q2k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    row_ptr: torch.Tensor,
    q_idx: torch.Tensor,
    topk: int,
    blk_kv: int,
    total_rows: int,
    max_kv_blocks: int,
    min_q_per_cta: int = 256,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """In-place fill of ``row_ptr`` and ``q_idx``.

    Args:
      q2k:           int32 [H, total_q, topK] contiguous (CUDA).
      cu_seqlens_q:  int32 [B+1] contiguous (CUDA).
      cu_seqlens_k:  int32 [B+1] contiguous (CUDA).
      row_ptr:       int32 [H, total_rows + 1] CUDA, written in place.
      q_idx:         int32 [H, total_q * topK] CUDA, written in place
                     (trailing slots set to -1).
      topk:          must be in {4, 8, 16, 32}.
      blk_kv:        must be one of {32, 64, 128}.
      total_rows:    sum over batches of ceil(seqlen_k / blk_kv).
      max_kv_blocks: max over batches of ceil(seqlen_k / blk_kv); upper bound
                     used to size the row_map workspace and clamp valid kv ids.
    """
    scratch_args = () if scratch is None else tuple(scratch)
    if len(scratch_args) not in (0, 4):
        raise ValueError("CSR scratch must contain four tensors")
    _ext.run_build_k2q_csr(
        q2k,
        cu_seqlens_q,
        cu_seqlens_k,
        row_ptr,
        q_idx,
        int(topk),
        int(blk_kv),
        int(total_rows),
        int(max_kv_blocks),
        int(min_q_per_cta),
        *scratch_args,
    )


def run_build_k2q_csr_with_schedule(
    q2k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    row_ptr: torch.Tensor,
    q_idx: torch.Tensor,
    scheduler_metadata: torch.Tensor,
    work_count: torch.Tensor,
    qsplit_idx: torch.Tensor,
    split_counts: torch.Tensor,
    topk: int,
    blk_kv: int,
    total_rows: int,
    max_kv_blocks: int,
    target_q_per_cta: int,
    work_capacity: int,
    max_seqlen_q: int,
    min_q_per_cta: int = 256,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """In-place fill of CSR plus fused sparse attention schedule metadata."""
    scratch_args = () if scratch is None else tuple(scratch)
    if len(scratch_args) not in (0, 4):
        raise ValueError("CSR scratch must contain four tensors")
    _ext.run_build_k2q_csr_with_schedule(
        q2k,
        cu_seqlens_q,
        cu_seqlens_k,
        row_ptr,
        q_idx,
        scheduler_metadata,
        work_count,
        qsplit_idx,
        split_counts,
        int(topk),
        int(blk_kv),
        int(total_rows),
        int(max_kv_blocks),
        int(target_q_per_cta),
        int(work_capacity),
        int(max_seqlen_q),
        int(min_q_per_cta),
        *scratch_args,
    )


def run_build_k2q_csr_with_schedule_from_local_top16(
    local_candidate_storage: torch.Tensor,
    prefix_lens: torch.Tensor,
    q2k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    row_ptr: torch.Tensor,
    q_idx: torch.Tensor,
    scheduler_metadata: torch.Tensor,
    work_count: torch.Tensor,
    qsplit_idx: torch.Tensor,
    split_counts: torch.Tensor,
    topk: int,
    blk_kv: int,
    total_rows: int,
    max_kv_blocks: int,
    candidate_partition_blocks: int,
    target_q_per_cta: int,
    work_capacity: int,
    max_seqlen_q: int,
    min_q_per_cta: int = 256,
    dynamic_only: bool = False,
    canonicalize_dynamic_padding: bool = False,
    batch1_identity_rows: bool = True,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Build route and CSR directly from one exact local Top-16 list."""
    scratch_args = () if scratch is None else tuple(scratch)
    if len(scratch_args) not in (0, 4):
        raise ValueError("CSR scratch must contain four tensors")
    _ext.run_build_k2q_csr_with_schedule_from_local_top16(
        local_candidate_storage,
        prefix_lens,
        q2k,
        cu_seqlens_q,
        cu_seqlens_k,
        row_ptr,
        q_idx,
        scheduler_metadata,
        work_count,
        qsplit_idx,
        split_counts,
        int(topk),
        int(blk_kv),
        int(total_rows),
        int(max_kv_blocks),
        int(candidate_partition_blocks),
        int(target_q_per_cta),
        int(work_capacity),
        int(max_seqlen_q),
        int(min_q_per_cta),
        bool(dynamic_only),
        bool(canonicalize_dynamic_padding),
        bool(batch1_identity_rows),
        *scratch_args,
    )


def run_reorder_schedule_qcount_bucket16(
    source: torch.Tensor,
    destination: torch.Tensor,
    work_count: torch.Tensor,
    target_q_per_cta: int,
) -> None:
    """Reorder every valid work record into descending q-count buckets."""
    _ext.run_reorder_schedule_qcount_bucket16(
        source,
        destination,
        work_count,
        int(target_q_per_cta),
    )


def is_supported(topk: int, blk_kv: int) -> bool:
    return int(topk) in (4, 8, 16, 32) and int(blk_kv) in (32, 64, 128)


__all__ = [
    "run_build_k2q_csr",
    "run_build_k2q_csr_with_schedule",
    "run_build_k2q_csr_with_schedule_from_local_top16",
    "run_reorder_schedule_qcount_bucket16",
    "is_supported",
]
