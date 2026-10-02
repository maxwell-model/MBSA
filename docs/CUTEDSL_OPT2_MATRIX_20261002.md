# CuTeDSL Opt2 Qualification Matrix (2026-10-02)

This report qualifies commit `c9fbf75` against the opt3 oracle on one B300
GPU. Every timed case first compares Dynamic Top-16, KV-centric CSR, output,
and LSE. Performance uses CUDA events, 10 warmups, 21 alternating opt3/CuTeDSL
pairs, and a 256 MiB L2 flush before every sample. The flush is ordered before
the start event and is excluded from reported latency.

## Correctness

All 28 primary matrix points passed:

- uniform batch-one prefill at 8K, 32K, 64K, 128K, 256K, 512K, and 1M KV;
- packed variable-length prefill at the same maximum request KV lengths;
- identity/contiguous and deterministically permuted Index-K page tables;
- single-partition and hierarchical multi-partition Exact Top-K paths.

For every point, Dynamic Top-16 and CSR were exactly equal to opt3, while O
and LSE were bitwise equal (`max_abs_o_diff = max_abs_lse_diff = 0`).

## Uniform batch-one performance

The query chunk is fixed at 8,192 tokens.

| KV | Page table | opt3 (ms) | CuTeDSL opt2 (ms) | Delta |
|---:|---|---:|---:|---:|
| 8K | identity | 0.6931 | 0.8089 | +16.71% |
| 8K | permuted | 0.7106 | 0.8295 | +16.72% |
| 32K | identity | 0.8970 | 0.9636 | +7.43% |
| 32K | permuted | 0.8981 | 0.9431 | +5.01% |
| 64K | identity | 1.1016 | 1.1223 | +1.88% |
| 64K | permuted | 1.0760 | 1.1356 | +5.54% |
| 128K | identity | 1.4008 | 1.4776 | +5.48% |
| 128K | permuted | 1.3957 | 1.4981 | +7.34% |
| 256K | identity | 2.1757 | 2.3183 | +6.55% |
| 256K | permuted | 2.1504 | 2.2907 | +6.52% |
| 512K | identity | 3.9055 | 3.6916 | -5.48% |
| 512K | permuted | 3.9393 | 3.7620 | -4.50% |
| 1M | identity | 7.2869 | 6.7389 | -7.52% |
| 1M | permuted | 7.2479 | 6.7686 | -6.61% |

An independent hot-machine repeat of the 8K identity case measured 0.7035 ms
for opt3 and 0.8212 ms for CuTeDSL (+16.74%), confirming that the short-context
gap is not a startup artifact.

## Packed variable-length performance

Packed Q lengths are `[4095, 4097]`, preserving an 8K total query chunk while
forcing partial final Q tiles. For each row below, request KV lengths are
`[KV, KV / 2 + 1]`; the second request therefore also exercises a partial KV
tail and a distinct causal offset.

| Maximum KV | Page table | opt3 (ms) | CuTeDSL opt2 (ms) | Delta |
|---:|---|---:|---:|---:|
| 8K | contiguous | 1.0262 | 1.1139 | +8.55% |
| 8K | permuted | 0.8354 | 0.8180 | -2.08% |
| 32K | contiguous | 1.1990 | 1.1778 | -1.77% |
| 32K | permuted | 1.2042 | 1.1694 | -2.89% |
| 64K | contiguous | 1.4307 | 1.3929 | -2.65% |
| 64K | permuted | 1.4325 | 1.4387 | +0.43% |
| 128K | contiguous | 1.8972 | 1.8381 | -3.12% |
| 128K | permuted | 1.8984 | 1.8851 | -0.70% |
| 256K | contiguous | 2.7760 | 2.4720 | -10.95% |
| 256K | permuted | 2.7811 | 2.5436 | -8.54% |
| 512K | contiguous | 4.5793 | 4.0990 | -10.49% |
| 512K | permuted | 4.5947 | 4.1062 | -10.63% |
| 1M | contiguous | 8.2094 | 6.9437 | -15.42% |
| 1M | permuted | 8.2359 | 6.9457 | -15.67% |

An independent repeat of packed 8K contiguous measured 1.0035 ms for opt3 and
1.0988 ms for CuTeDSL (+9.49%). This confirms a small but repeatable violation
of the 8% target at the shortest contiguous packed shape.

## Qualification conclusion

Functional production coverage passes across all tested lengths, page-table
layouts, partial tails, and Router partition counts. The 8% performance target
passes for every uniform point from 32K through 1M and for every packed point
from 32K through 1M. It does not yet pass for uniform 8K or packed 8K
contiguous. Short-context launch and fixed-cost optimization is therefore the
remaining performance gate; long-context Router/Exact-Top-K behavior is
qualified.

## Commands

```bash
# Uniform; repeat for each KV and add --shuffle-index-pages for the second row.
CUDA_VISIBLE_DEVICES=7 python3 bench/bench_bf16_cutedsl_router_e2e.py \
  --kv 8192 --seed 17 --warmups 10 --pairs 21 --l2-flush-mib 256

# Packed; repeat the KV pair per table row and add --shuffle-index-pages.
CUDA_VISIBLE_DEVICES=7 python3 bench/bench_bf16_cutedsl_packed_e2e.py \
  --full-chain --q-lens 4095 4097 --kv-lens 8192 4097 \
  --seed 17 --warmups 10 --pairs 21 --l2-flush-mib 256
```
