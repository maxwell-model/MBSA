# CuTeDSL Opt2: Production Coverage

`cutedsl-opt2` extends the qualified CuTeDSL Router beyond the block-aligned,
identity-page-table contract of `cutedsl-opt1`. The work is delivered in
independently qualified checkpoints so incomplete packed-request support is
not exposed as production-ready.

## Checkpoint 1: paged Index K and partial KV tails

The Router now separates three quantities that were previously conflated:

- logical KV block IDs, which remain the IDs emitted by Exact Top-16 and
  consumed by KV-centric CSR;
- physical Index-K page IDs, resolved through `index_page_indices` only when
  loading K;
- logical KV token length, used for causal visibility and final-page bounds
  independently of the padded physical page extent.

Identity and non-identity page tables use compile-time specialized direct and
indirect load variants. Block-aligned inputs retain the static K-extent path;
only partial-tail variants carry a distinct logical length. Page-table value
validation is performed once during plan preparation, not during replay, so
the timed path contains no device-to-host synchronization.

Partial logical KV tails require paged Index-K storage: the final physical
page remains 128 tokens wide while `logical_k_tokens` determines visibility.
The Router continues to emit request-logical block IDs, so CSR, LPT, Dynamic
MSA, and online-softmax state combination require no semantic changes.

## Qualification

The coverage gate compares the CuTeDSL Router with the opt3 Score/Top-16
oracle. The end-to-end gate additionally compares KV-centric CSR, output, and
LSE. All reported output comparisons are bitwise exact.

| KV | Valid tokens in final page | Index page table | Route | CSR | O / LSE |
|---:|---:|---|---|---|---|
| 8,192 | 128 | identity | exact | exact | bitwise |
| 8,192 | 128 | permuted | exact | exact | bitwise |
| 8,193 | 1 | identity | exact | exact | bitwise |
| 8,193 | 1 | permuted | exact | exact | bitwise |
| 33,007 | 111 | identity | exact | Router-only | Router-only |
| 33,007 | 111 | permuted | exact | Router-only | Router-only |

The 8,193-token full-chain smoke gate used one cold-L2 sample after one
warmup. It measured 0.825 ms versus 0.772 ms for opt3 on the identity path
(+6.87%), and 1.091 ms versus 1.006 ms with a permuted page table (+8.37%).
These smoke numbers are qualification signals, not the final performance
sweep.

```bash
CUDA_VISIBLE_DEVICES=0 python3 bench/bench_bf16_cutedsl_router_coverage.py \
  --kv 33007 --shuffle-index-pages

CUDA_VISIBLE_DEVICES=0 python3 bench/bench_bf16_cutedsl_router_e2e.py \
  --kv 8193 --shuffle-index-pages \
  --warmups 1 --pairs 1 --l2-flush-mib 64
```

## Checkpoint 2: packed variable-length requests

Packed BF16 prefill now accepts request-local Q and KV lengths through
`cu_seqlens_q` and `cu_seqlens_k`. Each request owns its logical Index-K page
table and causal offset. For the common single-partition Top-16 case, one
packed CuTeDSL Router launch consumes descriptors for all request-local Q
tiles, resolves request-local page-table ranges, and writes the global compact
ID planes directly. The unchanged packed KV-centric CSR/LPT builder then
consumes those logical block IDs.

The packed execution path has the following structure:

- the CuTeDSL Router launches once for the packed batch and supports partial
  final Q tiles;
- request-local Q starts are applied through a TMA tensor `domain_offset`, so
  Q continues through the native TMA-to-SW128-to-UMMA pipeline without an
  auxiliary staging kernel;
- requests that require multiple hierarchical Top-16 partitions retain the
  request-local fallback until packed Exact Top-K merge is implemented;
- Fixed BSA stages each request in a packed-GQA-tile-aligned Q buffer. Padding
  is split between the Q prefix and a masked KV/Q suffix when necessary, so
  bottom-right causal positions and original KV block boundaries are preserved;
- Dynamic BSA consumes the existing packed CSR representation;
- Fixed and Dynamic states use the independent K2 online-softmax merge because
  the fused initial-state combine is currently batch-one only.

The request-local Fixed path uses the conservative static pipeline. CLC,
split-barrier, and reduced-TMEM variants remain enabled only for their
qualified uniform batch-one domain. This checkpoint does not claim that the
experimental single-launch packed Fixed kernel is production-ready.

The packed full-chain gate uses `Q=[777,1153]` and
`KV=[33007,16641]`, exercising two partial Q tiles, two partial KV tails, and
different request-local causal offsets. Both physical Index-K layouts passed:

| Index page table | Dynamic Top-16 | KV-centric CSR | O | LSE |
|---|---|---|---|---|
| contiguous request pages | exact | exact | bitwise | bitwise |
| globally permuted pages | exact | exact | bitwise | bitwise |

A second full-chain boundary case, `Q=[2047,2049]` and `KV=[2047,2049]`,
forces KV-tail staging for requests with no prefix gap. It also matched the
opt3 route, CSR, O, and LSE bitwise.

A 21-pair alternating cold-L2 run on the contiguous-page case used six
warmups and a 64 MiB flush before every sample. The packed CuTeDSL path
measured 0.5990 ms versus 0.6441 ms for opt3, a 7.00% speedup. An independent
11-pair run measured 0.5991 ms versus 0.6469 ms, a 7.40% speedup. These results
include Router, unchanged packed CSR/LPT, Fixed BSA staging, Dynamic BSA, and
the independent K2 online-softmax merge.

The final 8K--1M qualification uses a stricter 256 MiB flush and covers both
uniform and packed variable-length workloads under identity/contiguous and
permuted page tables. All 28 points are bitwise exact. The initial matrix
passed the 8% performance gate from 32K through 1M and identified two 8K
exceptions. A subsequent length-specialized Router and packed-overlap policy
closed those exceptions; all four 8K page-layout cases now pass the gate. See
[`CUTEDSL_OPT2_MATRIX_20261002.md`](CUTEDSL_OPT2_MATRIX_20261002.md) for the
original full matrix and
[`CUTEDSL_OPT2_SHORT_CONTEXT_20261002.md`](CUTEDSL_OPT2_SHORT_CONTEXT_20261002.md)
for the 8K follow-up.

```bash
CUDA_VISIBLE_DEVICES=7 python3 bench/bench_bf16_cutedsl_packed_e2e.py \
  --full-chain

CUDA_VISIBLE_DEVICES=7 python3 bench/bench_bf16_cutedsl_packed_e2e.py \
  --full-chain --shuffle-index-pages

CUDA_VISIBLE_DEVICES=7 python3 bench/bench_bf16_cutedsl_packed_e2e.py \
  --full-chain --warmups 6 --pairs 21 --l2-flush-mib 64

CUDA_VISIBLE_DEVICES=7 python3 bench/bench_bf16_cutedsl_packed_e2e.py \
  --full-chain --q-lens 2047 2049 --kv-lens 2047 2049
```

## Remaining opt2 work

1. a broader production performance sweep across request-count and length
   distributions;
2. a packed hierarchical Exact Top-K merge for requests that span multiple
   Router partitions;
3. replacement of request-local Fixed launches with a tuned packed-grid
   variant where measurements justify the added complexity;
4. evaluation of dense batch greater than one and additional Indexer
   precisions after packed BF16 is qualified.

KV-centric CSR/LPT and host-side attention orchestration remain unchanged in
this checkpoint.
