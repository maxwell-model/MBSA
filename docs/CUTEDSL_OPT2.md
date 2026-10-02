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

## Remaining opt2 work

1. packed variable-length requests, including request-local Q/K offsets and
   partial Q tiles;
2. a production performance sweep for the new coverage shapes;
3. evaluation of dense batch greater than one and additional Indexer
   precisions after packed BF16 is qualified.

KV-centric CSR/LPT and host-side attention orchestration remain unchanged in
this checkpoint.
