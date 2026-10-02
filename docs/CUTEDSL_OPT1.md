# CuTeDSL Opt1

`cutedsl-opt1` is the first qualified CuTeDSL router integrated with the
Flash BSA dual-path prefill runtime. It targets BF16, batch-one, block-aligned
single-chunk prefill on NVIDIA B300 (SM103).

## Execution flow

```text
Index Score (CuTeDSL TMA + UMMA)
    -> per-partition register Top-16
    -> second-stage exact Top-16 merge
    -> compact dynamic block-ID planes
    -> KV-centric CSR + LPT
    -> Dynamic MSA + K2 online-softmax combine
```

The first-level partition contains 1024 KV blocks. Contexts through 128K use
one partition; 256K, 512K, and 1M contexts use two, four, and eight partitions
respectively. The second stage merges every partition candidate list exactly,
using block ID as the deterministic tie-breaker, and writes the global Top-16
directly into the compact layout consumed by the existing CSR builder.

The router does not materialize a dense score matrix, compute `Score x V`, or
produce Indexer O/LSE tensors.

## Validated contract

| Parameter | Value |
|---|---:|
| GPU | NVIDIA B300 / SM103 |
| Query chunk | 8192 tokens |
| Q / KV heads | 32 / 4 |
| Index Q / K heads | 4 / 1 |
| Head dimension | 128 |
| KV block | 128 tokens |
| Fixed support | Prefix-8 + Sliding-Window-8 |
| Dynamic support | Exact Top-16 |
| Dtype | BF16 |

Across 8K--1M KV length, the optimized path matches the qualified opt3 oracle
exactly for the dynamic route and CSR. Output and LSE are bitwise identical
with maximum absolute differences of zero.

## B300 cold-L2 performance

Each sample performs a same-stream 256 MiB read-modify-write L2 flush before
the start event. The flush is excluded from timing. Results are medians of 21
alternating paired samples after 10 warmups.

| KV | Partitions | opt3 | cutedsl-opt1 | Delta |
|---:|---:|---:|---:|---:|
| 8K | 1 | 0.696 ms | 0.765 ms | +9.84% |
| 32K | 1 | 0.890 ms | 0.948 ms | +6.56% |
| 64K | 1 | 1.090 ms | 1.117 ms | +2.54% |
| 128K | 1 | 1.398 ms | 1.401 ms | +0.22% |
| 256K | 2 | 2.151 ms | 2.307 ms | +7.23% |
| 512K | 4 | 3.860 ms | 3.648 ms | -5.49% |
| 1M | 8 | 7.357 ms | 6.804 ms | -7.50% |

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=0 python3 bench/bench_bf16_cutedsl_router_e2e.py \
  --kv 262144 --seed 17 --warmups 10 --pairs 21 --l2-flush-mib 256
```

Repeat with KV lengths `8192`, `32768`, `65536`, `131072`, `262144`,
`524288`, and `1048576` for the complete sweep.

## Source boundary

The new hierarchical router is implemented in
`kernels/bf16_indexer_cutedsl.py`. Fixed BSA, Dynamic MSA, K2 combine,
KV-centric CSR/LPT, and selected CuTeDSL helpers remain attributed runtime
dependencies. Their original licenses, notices, and copyright headers are
preserved; upstream non-runtime documents and benchmark suites are excluded.
