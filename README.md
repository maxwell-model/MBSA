# Maxwell Block Sparse Attention (MBSA)

MBSA is a GPU kernel library for long-context block-sparse prefill attention on
NVIDIA Hopper and Blackwell. It combines prefix, sliding-window and dynamically
selected blocks to reduce attention work while keeping causal masking.

- Paged and contiguous KV caches, packed requests and partial query/KV blocks.
- BF16 attention, plus native FP8 KV attention on Blackwell.
- BF16, FP16 or FP8 Index routing; native MXFP6 indexing on Blackwell.
- Reusable prepared plans, CUDA Graph replay and Fixed/Index routing overlap.
- Optional FP8 partial outputs and a serving interface for model integration.

Current geometry: 32 query heads, 4 KV heads, head dimension 128 and 128-token
blocks. Each request chunk contains at most 8192 query tokens.

## Architecture

Fixed attention covers prefix and local blocks; Dynamic attention selects remote
blocks. The two branches produce independent output and LSE states, then merge.
Hopper uses native SM90 kernels; Blackwell uses FA4 Fixed and Fireworks KV-outer Dynamic.

![MBSA architecture](docs/architecture.svg)

## Installation

Requires Linux, a supported NVIDIA GPU (SM90, SM100 or SM103), and a compatible
CUDA/PyTorch stack. The commands below install the CUDA 13 accelerator dependencies.

```bash
uv venv --python 3.12
uv pip install -e '.[fireworks,dev]'
uv pip install --no-deps 'flash-attn-4==4.0.0b20'
```

FA4 b20 is installed separately to use the pinned stable CUTLASS DSL 4.6.0 stack.
Its upstream metadata requests a development version, so `pip check` reports that
version mismatch; see [FA4 provenance](python/mbsa/third_party/fa4/PROVENANCE.md).
Native MXFP6 scorers additionally require `nvcc` and CUTLASS headers; set
`MBSA_CUTLASS_INCLUDE` to the headers' `include` directory when needed.

## Quick start

A single request with contiguous BF16 operands:

```python
import torch
from mbsa import prepare_cutedsl_pipeline

q_len, kv_len = 128, 4096

def rand(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

q = rand(q_len, 32, 128)
k, v = rand(kv_len, 4, 128), rand(kv_len, 4, 128)
index_q, index_k = rand(q_len, 4, 128), rand(kv_len, 1, 128)

pipeline = prepare_cutedsl_pipeline(index_q, index_k, q, k, v)
output, lse, dynamic_blocks = pipeline.run()
```

Prepare once and reuse the plan for the same geometry. Paged inputs provide
`page_table` and `index_page_indices`; MX indexing also supplies `index_q_scales`.
Address changes require rebinding before graph capture.

On Blackwell, matching `torch.float8_e4m3fn` K/V caches use native FP8 QK
and PV Tensor Core computation in both branches, with BF16 output. Supply BF16
Q to convert it once per run into a shared FP8 buffer, or supply prequantized
E4M3FN Q. FP8 tensors contain the represented operand values with unit scales;
additional Q/K/V descale factors are not supported. Hopper attention remains BF16.

BF16 partial outputs are the API default. Pass
`partial_dtype=torch.float8_e4m3fn` to enable FP8 partials.

For serving, `mbsa.serving.Prefill` stages requests and binds layer operands.

## Performance

Historical benchmark results are retained below. They describe the recorded
version and configuration, rather than a fresh measurement of the current tree.

![MBSA historical performance](docs/performance.svg)

[PNG version](docs/performance.png)

Run the prefill benchmark on Blackwell:

```bash
.venv/bin/python benchmarks/bench_prefill.py \
  --index-dtype fp8 --lengths 8192,16384,32768,65536,131072,262144 \
  --no-plot --output /tmp/mbsa-prefill.json
```

The benchmark compares MBSA with dense GQA. Sparse totals sum 8192-token chunk
medians and exclude preparation, compilation, projections and cache writes;
sparse and dense attention use different masks.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
uv build
.venv/bin/python -m tests.smoke.test_packaging --wheel dist/mbsa-*.whl
```

Core checks cover the FP32 reference, imports, packaging and source attribution.
GPU regressions cover precision, paging, tails, Fixed/Dynamic merging, rebinding
and CUDA Graph replay. GPU tests require supported hardware and dependencies.

## License and sources

Original MBSA code is licensed under [Apache-2.0](LICENSE). Third-party components
retain their original licenses; see [NOTICE](NOTICE).
Public source pins and adaptations are recorded for
[MSA](python/mbsa/third_party/msa/PROVENANCE.md),
[Fireworks](python/mbsa/third_party/fireworks/PROVENANCE.md) and
[FlashAttention-4](python/mbsa/third_party/fa4/PROVENANCE.md).
