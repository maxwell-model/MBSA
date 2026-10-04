# Retained MiniMax MSA runtime

This directory contains the runtime subset used by Maxwell Block Sparse Attention, adapted from [MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA).
The repository root contains the Maxwell Block Sparse Attention API, correctness checks, and benchmarks.

Retained components:

- BF16 `SparseAttentionForwardSm100` CuTe-DSL sparse prefill producer.
- K2 partial-output/LSE combine, including the initial fixed-attention state.
- q2k → k2q CSR construction, split metadata, and LPT scheduling.
- Required TMA, pipeline, softmax, and layout helpers.

The standalone dense C++ attention/JIT layer, FP4 indexer, NVFP4 prefill,
FP8 decode, and unrelated utilities have been removed. The public sparse
interface accepts BF16 Q/K/V. Shared forward/helper code still contains dtype
specializations inherited from upstream; these do not expose alternate backends.
The CSR metadata builder remains a small CUDA C++ extension compiled lazily
because the performance path depends on it.

## Public API

```python
from fmha_sm100.sparse import (
    SparseForwardPhase,
    sparse_atten_func,
    build_k2q_csr,
    SparseK2qCsrBuilderSm100,
    allocate_k2q_csr_scratch,
)
```

A bare `import fmha_sm100` is lazy and does not import Torch/CUDA, allocate
CUDA buffers, or register dense TVM entry points.

Within Maxwell Block Sparse Attention, this snapshot is loaded as
`mbsa._vendor.msa.python.fmha_sm100`. Its imports are package-relative,
so it does not register global `src`, `interface` or `fmha_sm100` modules.

## Runtime requirements

Use Linux with an NVIDIA SM100 GPU, the CUDA toolkit, and dependencies listed
in `requirements.txt`. CuTe-DSL compiles the attention producer and combine;
Torch compiles the CSR extension on its first use. No vendored CUTLASS C++
submodule is needed. Install with `pip install -e .` from the
repository root, or use the root project's editable install.

## License

Upstream copyright and license headers are retained. See [LICENSE](LICENSE)
and [NOTICE](NOTICE) for attribution and third-party notices.
