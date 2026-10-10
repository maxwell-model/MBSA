# KV FP8 validation — 2026-10-10

Validation ran on NVIDIA B300 at `10.251.209.5` with Python 3.12.14,
PyTorch 2.11.0+cu130, Triton 3.6.0 and CUTLASS DSL 4.6.0.
E4M3FN K/V use unit-scale represented values. BF16 Q is converted into a
persistent E4M3FN buffer on every execution and shared by Fixed and Dynamic;
prequantized E4M3FN Q is also accepted. Output is BF16. External KV descales
are not part of this API.

All three branch checkouts passed six numerical regression cases:
fixed-only; Dynamic with a KV tail and BF16 Index; FP8 partial output;
MXFP6 Index; packed requests with live Q updates, layer rebinding and graph
replay; and a 32K KV context with live BF16 Q updates and Fixed/Router overlap.
Garena's 32K case explicitly checked the selected 5-WG CLC configuration.
The oracle computes FP32 attention on the represented FP8 values, applying
actual selected Dynamic pages, Prefix8, Local8 and token-level causal masks.
It does not compare routes against an independent router implementation.

Public additionally passed a seventh adversarial Fixed regression. A later
page raises the online maximum by less than the upstream FP8 rescale threshold,
which previously saturated the E4M3FN probability numerator while preserving
the FP32 denominator. The private FA4 softmax factory now rescales on every
FP8 maximum increase, keeping probabilities at or below 256. BF16 keeps the
upstream policy; the test also checks the installed upstream factory identity.
This guard changes no installed dependency or global cache.

PTX inspection of Fixed and Dynamic attention specializations found 32 and
16 `kind::f8f6f4` MMA instructions respectively, with zero `kind::f16` MMA
instructions in either attention kernel. The 4K numerical probe's maximum
output error was 0.000787 on public and 0.000705 on both Garena checkouts;
maximum LSE error was 0.00000191. These observations verify native FP8 QK/PV,
not a performance claim or a bound for every possible input value.

`benchmarks/bench_prefill.py --kv-dtype fp8` uses true FP8 K/V and includes
BF16-to-FP8 Q conversion in the measured MBSA execution. Its dense/MSA
baselines use the same represented Q/K/V decoded to BF16 before timing;
they must not be reported as native-FP8 baseline kernels.

The public FP8 KV benchmark completed at 8K and 32K using seed 17, chunk
8192, FP8 Index, cold L2, two warmups and five sparse timing samples. Its
existing output tolerance (0.03 absolute / 0.03 relative), LSE and route
checks passed without relaxation. Summed MBSA chunk medians were 1.041504 ms
and 4.704832 ms respectively; BF16 Q conversion is included. These short
runs are validation evidence, not a native-FP8 MSA performance comparison.
Three representative BF16 Fixed/overlap/merge regressions also passed after
the FP8 softmax guard. The local CPU suite passed 54 tests (39 skipped for
CUDA or optional dependencies).

Final GPU logs, benchmark JSON and compiled-runtime evidence are retained
on the B300 host under `/data/mbsa-kvfp8-20261010`.
