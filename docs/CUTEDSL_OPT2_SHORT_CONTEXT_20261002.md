# CuTeDSL Opt2 Short-Context Follow-up (2026-10-02)

This follow-up closes the 8K performance exceptions identified by the full
qualification matrix. It keeps the 32K--1M path unchanged.

## Optimization

The 8K Router uses a compile-time lean pipeline with two K stages, two
accumulator stages, and 256 TMEM columns. Longer contexts retain the original
three-stage pipeline and 512-column TMEM allocation. The smaller 8K footprint
reduces destructive SM103 resource interference with concurrently executing
Fixed BSA CTAs.

Packed 8K additionally permits Fixed BSA to overlap route generation even
when Fixed and Dynamic states use the independent online-softmax merge. The
Dynamic BSA launch still waits for both the route/CSR dependency and the
Fixed-state dependency. Packed contexts above 8K retain sequential Fixed and
Router execution because a 32K A/B showed that overlap was counterproductive.

## Qualification method

Measurements use one B300 GPU, 10 warmups, 21 alternating opt3/CuTeDSL
samples, CUDA events, and a 256 MiB L2 flush before every sample. The flush is
excluded from latency. Packed comparisons keep the original opt3 path
sequential and enable overlap only for the CuTeDSL candidate.

| Workload | Page table | opt3 (ms) | CuTeDSL opt2 (ms) | Delta |
|---|---|---:|---:|---:|
| Uniform 8K | identity | 0.6247 | 0.6534 | +4.61% |
| Uniform 8K | permuted | 0.6278 | 0.6492 | +3.40% |
| Packed 8K | contiguous | 0.8358 | 0.7997 | -4.31% |
| Packed 8K | permuted | 0.8343 | 0.7937 | -4.87% |

Dynamic Top-16 and KV-centric CSR are exactly equal to opt3 for every row.
Output and LSE are bitwise equal, with zero maximum absolute difference.

A 32K guard measurement kept the original three-stage, non-overlapped packed
policy: uniform identity measured +6.28% and packed contiguous measured
+7.50%, both within the 8% target. Enabling packed overlap at 32K regressed
the candidate by 10.75%, so the production policy does not enable it there.

## Rejected variants

- Reducing the Router launch from 256 to 192 threads changed end-to-end 8K
  latency by less than measurement noise.
- Reversing the cross-stream launch order and assigning a high-priority Fixed
  stream both increased latency.
- Fully serializing Fixed BSA and Router reduced the original 8K regression
  but reached only +11.92%, short of the target.

## Commands

```bash
CUDA_VISIBLE_DEVICES=7 python3 bench/bench_bf16_cutedsl_router_e2e.py \
  --kv 8192 --seed 17 --warmups 10 --pairs 21 --l2-flush-mib 256

CUDA_VISIBLE_DEVICES=7 python3 bench/bench_bf16_cutedsl_packed_e2e.py \
  --full-chain \
  --q-lens 4095 4097 --kv-lens 8192 4097 \
  --seed 17 --warmups 10 --pairs 21 --l2-flush-mib 256
```
