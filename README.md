# Maxwell Block Sparse Attention (MBSA)

MBSA is a GPU kernel library for long-context block-sparse prefill attention on
NVIDIA Hopper and Blackwell. It combines prefix, sliding-window and dynamically
selected blocks to reduce attention work while keeping causal masking.

- Paged and contiguous KV caches, packed requests and partial query/KV blocks.
- BF16 attention, plus backend-specific FP8 KV support on Blackwell.
- BF16, FP16 or FP8 Index routing; native MXFP6 indexing on Blackwell.
- Reusable prepared plans, CUDA Graph replay and Fixed/Index routing overlap.
- Optional FP8 partial outputs and a serving interface for model integration.

Current geometry: 32 query heads, 4 KV heads, head dimension 128 and 128-token
blocks. Each request chunk contains at most 8192 query tokens.

## Architecture

Fixed attention covers prefix and local blocks; Dynamic attention selects remote
blocks. The two branches produce independent output and LSE states, then merge.
The default backend uses native SM90 kernels on Hopper and FA4 Fixed plus
Fireworks KV-outer Dynamic on Blackwell. The explicitly selected optimized
Blackwell backend uses its separately attributed BSA/MSA implementation.

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
Its upstream metadata requests CUTLASS DSL 4.6.0.dev0, which conflicts with this
validated stable stack: CUTLASS DSL 4.6.0, QuACK 0.6.3 and TVM FFI >=0.1.12,<0.2.
The separate `--no-deps` step intentionally overrides that dependency pin without
modifying the official wheel; `pip check` still reports the CUTLASS version
mismatch. Fixed preparation requires the unmodified official FA4 b20 source
and checks its version, source content and class binding before launching.
Preparation and kernel warmup failures propagate instead of silently falling back.
See the [FA4 backend details](python/mbsa/third_party/fa4/PROVENANCE.md) for
source attribution and integration notes.
One tested B300 combination is official PyTorch `2.10.0+cu130`, Triton `3.6.0`,
CUTLASS DSL `4.6.0`, QuACK `0.6.3`, TVM FFI `0.1.12` and FA4 `4.0.0b20`.
Select PyTorch's CUDA 13.0 wheel before installing the extras, and pin Triton
to `3.6.0` when reproducing that combination. This is not a claim that every
version allowed by the dependency minima has been validated. In particular,
older PyTorch/CUDA combinations need separate B300 FP8/DLPack validation.

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

On Blackwell, the default backend uses matching `torch.float8_e4m3fn` K/V
caches for native FP8 QK and PV Tensor Core computation, with BF16 output.
Supply BF16 Q to convert it once per run into a shared FP8 buffer, or supply
prequantized E4M3FN Q. The optimized backend instead converts FP8 attention
operands to BF16 in owned buffers on each run and performs BF16 attention.
FP8 tensors contain represented operand values with unit scales; additional
Q/K/V descale factors are not supported. Hopper attention remains BF16.
FP8 Index routing is independent of attention storage and compute precision.

BF16 partial outputs are the API default. Pass
`partial_dtype=torch.float8_e4m3fn` to enable FP8 partials.

For serving, `mbsa.serving.Prefill` stages requests and binds layer operands.

Preparation and serving accept `backend="default"` and `backend="optimized"`. The optimized backend combines the MBSA Index scorer with BSA, MSA and combine kernels on SM100 and SM103 with CuTe DSL 4.6.0. Pass `backend="optimized"` to `prepare_cutedsl_pipeline` or `Prefill(device, backend="optimized")`. The default backend also supports SM90.

The optimized Index stage supports BF16, FP16, E4M3 and MX formats. FP8 attention operands are converted to BF16 in owned buffers on each run. BSA writes contiguous `[T,Hq,D]` output and uses static persistent scheduling across KV lengths. Mixed requests use a separate attention-state merge; batch one uses fused initial-state combine. Original source notices, license texts and adaptation records ship under `mbsa/backends/optimized/`.

Optimized Index page maps bind directly to the supplied tensor. Active request rows must form one adjacent contiguous storage span, with 16-byte row alignment for unpacked or MX Index and 4-byte alignment for packed non-MX Index. Preparation and rebind reject gapped, strided or unaligned maps. Pass compact active page IDs for variable-length requests. Optimized serving accepts a 1-D compact Index map and requests with Q <= 8192; split longer requests and their maps before staging to avoid map duplication. In-place map updates are visible during eager execution and CUDA Graph replay.

Direct Q/K/V, Index Q and floating Index K, MX Q scales and caller output
require 16-byte base alignment; MXFP6 Index K requires 128-byte alignment.
These checks apply at preparation
and rebind. FP8 attention inputs, padded MX queries and partial unpaged Index K
are copied into owned aligned buffers. Aligned storage-offset views are supported.

Each prepared backend owns its workspace, stage connections and graph resources.
A plan is single-in-flight: order its calls and CUDA work, and use independent
plans for concurrent execution. `run()` returns live BF16 output, FP32 natural-log
LSE and int32 routes; the next run overwrites these buffers. A supplied output
remains caller-owned; otherwise output, LSE and routes remain plan-owned.
Read-only inputs may share storage; output must have a disjoint byte span.
Keep operand contents valid until their CUDA work finishes.

Public `rebind()` validates matching paged geometry before publishing bindings.
Rejection preserves prior bindings, but cannot undo caller writes into inputs.
Use `capture_mega_graph()` and `run_mega()` for backend-owned capture and replay.
Once captured, a plan cannot be rebound. The existing `.plan` inspection and
execution entry points remain available for compatibility.

### Compatibility and validation boundaries

Backend selection defaults to `default`; adding `optimized` does not select it
implicitly. Both retain the customer selection contract. Prepared owners now
reject output/input byte-span overlap and stage rebind updates transactionally.
Compared with the earlier default-only API, `rebind()` also validates page IDs
and may synchronize with the host. Do not treat it as a no-host-sync hot-path
operation. This does not change `run()` into a synchronous operation.

When Fixed/Index overlap is enabled, a plan creates its own side stream if no
stream is supplied. Serving no longer supplies a runner-wide shared stream;
this is a change in stream ownership, not an unconditional disabling of overlap.
Use independent plans for simultaneous requests and preserve the documented
single-in-flight ordering for each plan.

These are implementation contracts, not validation of every dtype, architecture
or private primitive. See the [validation scope](docs/validation.md) and the
historical [default FP8 report](docs/kvfp8-validation.md) for their limits.

### Experimental NVFP4 KV attention

The optimized namespace retains the low-level `sparse_atten_nvfp4_kv_func`
interface and optional Transformer Engine quantization helpers under
`mbsa.backends.optimized.msa.cute`. This path consumes packed E2M1 K/V,
E4M3 per-16-element block scales in 128x4 tiled storage, and optional FP32
tensor scales. It is not an FP4 Indexer and is not selected by the prepared
pipeline or serving API; default attention precision is unchanged.

Treat this low-level path as experimental. Its numerical quality and safety
require separate validation; BF16/FP8 validation does not qualify NVFP4.
Transformer Engine is needed only when using its quantization helper.

## Performance

Historical benchmark results are retained below. They describe the recorded
version and configuration, rather than a fresh measurement of the current tree.

![MBSA historical performance](docs/performance.svg)

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

Core checks cover the FP32 reference, imports, packaging and bundled license files.
GPU regressions cover precision, paging, tails, Fixed/Dynamic merging, rebinding
and CUDA Graph replay. GPU tests require supported hardware and dependencies.

## Contributing

Issues and pull requests are welcome. For substantial API or kernel changes,
please open an issue first to discuss the design. Run the relevant validation
checks before submitting, and preserve applicable copyright, license and source
attribution for any third-party code.

## License and sources

Original MBSA code is licensed under [Apache-2.0](LICENSE). Third-party components
retain their original licenses; see [NOTICE](NOTICE).
Source attribution and MBSA modification summaries are retained in that notice
and the component license directories under `python/mbsa/third_party/`.
The optimized backend has a separate [provenance record](python/mbsa/backends/optimized/PROVENANCE.md),
including historical-source availability limits and unresolved earlier origins.
The existing [MSA](python/mbsa/third_party/msa/PROVENANCE.md),
[Fireworks](python/mbsa/third_party/fireworks/PROVENANCE.md) and
[FlashAttention-4](python/mbsa/third_party/fa4/PROVENANCE.md) records retain
their detailed source identities and adaptations.
