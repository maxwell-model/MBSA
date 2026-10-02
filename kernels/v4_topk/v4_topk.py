"""② top-k: DeepSeek-V4 `deepseek_v4_topk_transform_512` loader.

The kernel is SGLang's `sgl-kernel/csrc/elementwise/deepseek_v4_topk.cu` — a single
fused kernel that does select + paged-physical-slot transform in one launch
(`kMaxTopK=1024`, `kBlockSize=512`, `grid = B`, one block per row).

Both `deepseek_v4_topk.cu` and `v4_wrap.cpp` are included in this repository.
The `.cu` is SGLang (Apache-2.0, license header intact); the wrapper matches its signature
(`const at::Tensor&` / `at::Tensor&`, NOT by-value — a by-value declaration will
not link, different mangled names).

`load()` resolves in this order:

  1. `torch.ops.sgl_kernel.deepseek_v4_topk_transform_512` — only when another
     embedding application has already registered the exact native op.
  2. JIT-build from `deepseek_v4_topk.cu` + `v4_wrap.cpp` in this directory via
     `torch.utils.cpp_extension.load` (~17s). This is the path we actually use.

The file is self-contained (only ATen / c10 / CUDA headers — no cutlass, no
sgl-internal headers), which is why the standalone JIT build works.

Build environment for B300 (CUDA 13 CCCL include + sm_103 via PTX):

    CCCL=/usr/local/cuda-13.1/targets/x86_64-linux/include/cccl
    export CPATH=$CCCL NVCC_PREPEND_FLAGS="-I$CCCL" TORCH_CUDA_ARCH_LIST="10.0+PTX"

Authoritative operator test = `tests/verify_topk_v4_512.py`; performance is measured
by `bench/bench_indexer_topk.py`. `topk_transform_512()` below is a thin
typed wrapper over the same JIT module; the shapes/dtypes it asserts are the
contract verified by `tests/verify_topk_v4_512.py` (set-equal + page==raw, PASS).
"""

from __future__ import annotations

import os
import pathlib
from typing import Optional, Tuple

import torch

_HERE = pathlib.Path(__file__).resolve().parent
_CU = _HERE / "deepseek_v4_topk.cu"
_WRAP = _HERE / "v4_wrap.cpp"

_impl = None
_source = ""


def _try_native():
    op = getattr(torch.ops.sgl_kernel, "deepseek_v4_topk_transform_512", None)
    if op is None:
        raise RuntimeError("sgl_kernel does not register deepseek_v4_topk_transform_512")
    return op


def _try_jit():
    if not _CU.exists():
        raise FileNotFoundError(
            f"{_CU} not found — the delivery source is incomplete"
        )
    from torch.utils.cpp_extension import load

    return load(
        name="deepseek_v4_topk_standalone",
        sources=[str(_WRAP), str(_CU)],
        extra_cuda_cflags=["-O3"],   # Validated flags; intentionally no fast_math.
        verbose=bool(int(os.environ.get("V4_TOPK_VERBOSE", "0"))),
    ).run


def load() -> Tuple[object, str]:
    """Return (callable, source_label). Cached after the first call."""
    global _impl, _source
    if _impl is not None:
        return _impl, _source
    errors = []
    for label, fn in (("sgl_kernel-native", _try_native), ("jit", _try_jit)):
        try:
            _impl, _source = fn(), label
            return _impl, _source
        except Exception as exc:
            errors.append(f"  {label}: {type(exc).__name__}: {exc}")
    raise ImportError(
        "deepseek_v4_topk_transform_512 unavailable:\n" + "\n".join(errors)
    )


def topk_transform_512(
    scores: torch.Tensor,        # [B, L] fp32, contiguous in dim 1
    seq_lens: torch.Tensor,      # [B] int32, per-row valid causal prefix length
    page_table: torch.Tensor,    # [B, num_pages] int32, logical -> physical page
    page_indices: torch.Tensor,  # [B, topk] int32 OUT, physical slot ids
    page_size: int,              # power of two (Compass bench uses 64)
    raw_indices: Optional[torch.Tensor] = None,  # [B, topk] int32 OUT, logical positions
):
    """Thin typed wrapper. `topk` is taken from `page_indices.size(1)` (<= 1024)."""
    assert scores.dtype == torch.float32 and scores.dim() == 2
    assert scores.stride(1) == 1, "scores must be contiguous along dim 1"
    assert seq_lens.dtype == torch.int32 and seq_lens.shape[0] == scores.shape[0]
    assert page_table.dtype == torch.int32
    assert page_indices.dtype == torch.int32
    assert page_indices.shape[1] <= 1024, "kMaxTopK = 1024"
    assert page_size & (page_size - 1) == 0, "page_size must be a power of two"
    fn, _ = load()
    return fn(scores, seq_lens, page_table, page_indices, page_size, raw_indices)


def source_label() -> str:
    """Which backend `load()` resolved to — print this in a bench banner."""
    load()
    return _source
