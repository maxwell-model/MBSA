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
table, Router workspace, and optional hierarchical Top-16 workspace. Router
outputs remain request-local logical block IDs and are copied into the
corresponding Q slice of the global compact ID planes before the unchanged
packed KV-centric CSR/LPT builder runs.

The coverage path deliberately favors robust semantics over final packed-grid
tuning:

- the CuTeDSL Router launches once per request and supports partial final Q
  tiles;
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

A five-pair cold-L2 smoke run on the contiguous-page case measured
0.767 ms for the CuTeDSL coverage path versus 0.658 ms for the opt3 Router
(+16.49%). The approximately 0.109 ms delta includes two request-local Router
launches plus Fixed-BSA staging and copies. It is not an optimized packed-grid
result or a production performance qualification; the purpose of this
checkpoint is complete variable-length semantics without a magnitude-level
regression.

```bash
CUDA_VISIBLE_DEVICES=7 python3 bench/bench_bf16_cutedsl_packed_e2e.py \
  --full-chain

CUDA_VISIBLE_DEVICES=7 python3 bench/bench_bf16_cutedsl_packed_e2e.py \
  --full-chain --shuffle-index-pages

CUDA_VISIBLE_DEVICES=7 python3 bench/bench_bf16_cutedsl_packed_e2e.py \
  --full-chain --warmups 2 --pairs 5 --l2-flush-mib 64

CUDA_VISIBLE_DEVICES=7 python3 bench/bench_bf16_cutedsl_packed_e2e.py \
  --full-chain --q-lens 2047 2049 --kv-lens 2047 2049
```

## Remaining opt2 work

1. a production performance sweep for the new coverage shapes;
2. replacement of request-local Router/Fixed launches with tuned packed-grid
   variants where measurements justify the added complexity;
3. evaluation of dense batch greater than one and additional Indexer
   precisions after packed BF16 is qualified.

KV-centric CSR/LPT and host-side attention orchestration remain unchanged in
this checkpoint.
