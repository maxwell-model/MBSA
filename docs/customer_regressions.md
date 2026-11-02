# Integration regression coverage

The follow-up fixes keep the explicit `backend="optimized"` selection and the
default BF16 behavior. They do not add a NaN scan or output reconstruction.

- Prepared plans track external capture through a cached class method rather
  than an instance-owned bound method. Plan ownership no longer requires cyclic
  garbage collection after compilation.
- Dynamic combine completes generic shared-memory reads before releasing a
  stage for asynchronous TMA reuse.
  Coverage includes FP8 intermediate partials from the real producer, compared
  exactly against a mathematically checked no-TMA control with unused slots poisoned.
- Fixed attention executes collective TMEM loads uniformly across the warp.
  Empty rows still produce the existing empty state; legally consumed NaNs are
  not converted into a zero output.
- CSR scatter reserves a range per equal-row warp group and uses lane rank
  within that range, rather than assuming an ordering for atomic return values.
- Optimized self-imports are relative, allowing a vendored package prefix.
- Compact MX Index page maps require an aligned flat base and adjacent active
  rows, not 16-byte alignment of every request row. Gaps and strides remain
  unsupported.
- Packed subtraction uses the public CuTe primitive instead of a private QuACK
  helper. This does not change the pinned dependency stack.

Logical nonfinite inputs do not impose masked-input NaN isolation. Physical
padding and unused partial slots must remain excluded. Tests distinguish these
cases rather than requiring every poisoned request to produce finite output.

## Tests

On a supported Blackwell GPU with the documented dependencies:

```bash
python -m unittest -v tests.regression.test_customer_review
python -m unittest -v tests.regression.test_backends tests.regression.test_fixed_fa4
python -m unittest -v tests.smoke.test_direct_page_map tests.smoke.test_imports
```

Serving warmup must cover the request geometries used at runtime. In particular,
single-request fused combine does not exercise the separate mixed-request merge
kernel. The regression suite checks mixed-request merge compilation and reuse
from an initially empty merge cache.
`compiled_kernel_count()` also includes that separate merge cache, rather than
counting only dictionaries whose names end in `cache`/`CACHE`.

## Measuring prefill

`benchmarks/bench_prefill.py --backend optimized` selects the backend explicitly.
Its historical summary is a **sum of independently measured chunk medians**,
not a direct full-prefill latency. Cold-cache flush work is drained before the
start event so it cannot hide CPU enqueue gaps.

For direct eager timing of a real ordered prompt:

```bash
python benchmarks/bench_true_prefill.py --backend optimized --lengths 65536,131072 \
  --scope full --index-dtype bf16 --output full-bf16.json
python benchmarks/bench_true_prefill.py --backend optimized --lengths 65536,131072 \
  --scope full --index-dtype fp8 --include-preparation --output full-index-fp8.json
```

Each chunk sees only its legal KV prefix. The enclosing CUDA events measure the
whole chain directly, including its stream dependencies and enqueue gaps.
The FP8 preparation option includes casting the current Index Q and inserting
the newly appended Index K slice; it does not include Projection, RMSNorm,
RoPE, or attention-cache writes. Attention operands remain BF16.
Any internal layout conversion or cache transport performed by the selected
backend remains inside the timed run.

Clock, power and temperature telemetry should accompany sustained measurements.
A lower observed clock alone does not identify a power or thermal limit.
