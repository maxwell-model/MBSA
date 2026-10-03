# CuTeDSL Opt2 Midrange Scheduling Follow-up (2026-10-03)

This follow-up reduces the remaining uniform-prefill gap in the 32K--256K
range by adapting the cross-stream launch order of Fixed BSA and the CuTeDSL
Router. It does not change Router arithmetic, Exact Top-K, KV-centric CSR,
Dynamic BSA, or the online-softmax merge.

## Attribution

Cold-L2 stage measurements showed that Router and CSR latency alone did not
explain the end-to-end regression. Under the original Fixed-first policy, the
overlapped Fixed/Router prefix was consistently more expensive than the opt3
Score-first prefix:

| KV | Score + CSR delta | Overlapped prefix delta | Full-chain delta |
|---:|---:|---:|---:|
| 32K | -8.73% | +15.08% | +5.53% |
| 64K | -1.16% | +7.52% | +1.89% |
| 128K | -2.07% | +8.92% | +4.13% |
| 256K | +1.22% | +8.93% | +7.61% |

The result identifies cross-stream GPU resource contention, rather than the
standalone Router/CSR implementation, as the dominant midrange issue.

## Scheduling policy

The qualified policy is length- and layout-specialized:

- 8K and 64K retain Fixed-first publication;
- both 32K layouts use Score-first publication;
- the 32K identity layout retains the full three-stage accumulator pipeline
  and 512 TMEM columns;
- a non-identity 32K page table uses a two-stage accumulator pipeline and 256
  TMEM columns; K loading retains its three-stage pipeline;
- 128K--256K uses Score-first publication, reducing harmful interference on
  the longer Router critical path;
- 512K and 1M retain their previously qualified Fixed-first path.

Both orders preserve the same dependency graph. Fixed BSA and route
generation remain on separate CUDA streams; Dynamic BSA starts only after
both the fixed-state and route/CSR dependencies are satisfied.

## Performance and correctness

The new 32K, 128K, and 256K measurements use one B300 GPU, 10 warmups, 21
alternating opt3/CuTeDSL samples, CUDA events, and a 256 MiB L2 flush before
every sample. The flush is excluded from the reported latency. The 64K path
is unchanged, so its value below is retained from the existing full
qualification matrix rather than remeasured.

| KV | CuTeDSL order | opt3 (ms) | CuTeDSL opt2 (ms) | Delta | Evidence |
|---:|---|---:|---:|---:|---|
| 32K identity | Score-first | 0.8950 | 0.8745 | -2.29% | new 21-pair run |
| 32K permuted | Score-first | 0.8970 | 0.8769 | -2.24% | new 21-pair repeat |
| 64K | Fixed-first | 1.1016 | 1.1223 | +1.88% | retained 21-pair matrix |
| 128K | Score-first | 1.4061 | 1.4551 | +3.49% | new 21-pair run |
| 256K | Score-first | 2.1987 | 2.1984 | -0.01% | new 21-pair run |

For the new 32K, 128K, and 256K runs, Dynamic Top-16 and KV-centric CSR are
exactly equal to opt3, and output and LSE are bitwise equal. Independent 32K
repeats measured 0.8756 ms versus 0.8999 ms for identity (-2.71%) and 0.8735
ms versus 0.8980 ms for permuted pages (-2.73%), confirming both results. The
implementation therefore remains within the 8% production target throughout
32K--256K.

## Rejected variant

The 8K low-resource Router configuration was experimentally extended through
64K. It increased Router latency by 1.08% at 32K and 5.65% at 64K, producing
full-chain regressions of 8.34% and 8.08%, respectively. The production
implementation consequently keeps this configuration restricted to 8K.

Applying the reduced-accumulator variant to the 32K identity layout was also
rejected. Although one run measured a 1.95% speedup, two independent runs
regressed by 9.22% and 7.98%, revealing a cross-stream scheduling bifurcation.
A controlled same-process matrix isolated the cause: Identity benefits from
Score-first publication while retaining the full accumulator/TMEM footprint.
The resulting full-footprint configuration produced repeatable speedups in
two independent 21-pair runs.

## Command

```bash
# Controlled 32K layout/resource/order matrix.
CUDA_VISIBLE_DEVICES=7 python3 bench/bench_bf16_cutedsl_32k_layout_matrix.py \
  --seed 17 --warmups 10 --pairs 21 --l2-flush-mib 256

# Repeat with --kv 262144 for the second new measurement.
CUDA_VISIBLE_DEVICES=7 python3 bench/bench_bf16_cutedsl_router_stages.py \
  --kv 131072 --seed 17 --warmups 10 --pairs 21 --l2-flush-mib 256
```
