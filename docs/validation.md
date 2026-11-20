# Validation scope and reproduction

Source/packaging, numerical correctness, sanitizer results, model quality and
performance are separate claims. A successful build or source hash check
does not qualify an attention kernel. A GPU result applies only to its exact
source, device, precision, inputs and invocation path.

## Host checks

The source/manifest checks below use only Python's standard library. They
validate local imports, notices, document links and source hashes, and reject
corrupted hashes, malformed revisions and source paths. They do not launch
GPU kernels. The customer-preimage test additionally requires a Git checkout
containing the recorded base; an archive without those objects cannot make
that verification claim.

```bash
python -m unittest -v \
  tests.smoke.test_packaging.SourcePackagingContracts.test_source_imports_licenses_and_document_links \
  tests.smoke.test_provenance
ruff check .
python -m build
```

The full wheel checker also exercises CPU runtime imports and therefore
requires PyTorch; accelerator extras must remain lazy until preparation:

```bash
python -m tests.smoke.test_packaging --wheel dist/mbsa-*.whl
```

## GPU regressions

Use the documented compatible stack, build/warm kernels before timing and
record the exact tested Git commit/tree and installed versions. The following
are the relevant full modules, not a claim that all have passed on every
architecture or on the current checkout:

```bash
python -m unittest -v \
  tests.regression.test_optimized \
  tests.regression.test_backends \
  tests.regression.test_fixed_fa4 \
  tests.regression.test_customer_review \
  tests.regression.test_serving_followup
```

The optimized Index dispatch (README) has its own modules: `test_index_dispatch`,
`test_two_sm`, `test_skip_fixed_blocks`, `test_fine_partitions`,
`test_packed_partitions`, `test_fp8_policy`, `test_parallel_insert`,
`test_queued_insert` and `test_fp8_two_sm` under `tests/regression`. Each
variant module binds its kernel through `tests.common.index_dispatch` and
compares it with the baseline selection bit for bit. On SM103 the rest of the
suite runs with the default dispatch; `MBSA_INDEX_DISPATCH=baseline` runs it with
the baseline selection.

Report PASS, FAIL and SKIP separately with their reasons. Do not call a
module qualified because unsupported or dependency-missing cases were skipped.
Native memcheck/synccheck/racecheck require both a successful application
oracle and native zero-error/zero-hazard results, not just a wrapper exit.

## Recorded evidence and limits

A maintainer validation run on B300 SXM6 AC, 148 SM, used official
PyTorch 2.10.0+cu130, Triton 3.6.0, CUTLASS DSL 4.6.0, FA4 b20 and Compute Sanitizer
2025.4.1. Its source snapshot was
`6c580edfabc4b0e9e6048d7f50d1e21b743c5e61`, before NVFP4 was restored.
Six synccheck, six racecheck and three memcheck invocations passed for six
representative BF16/Raw E4M3 Indexer cases, including reversed partial pages,
physical-tail NaNs, independent streams and Graph replay. Ordinary validation
used 1000 replays per live-input variant; instrumented validation used three.
Small cases checked all queries mathematically. Large Q8192 cases checked
all output/LSE finite state and optimized all-query CSR, with 42 stratified
query rows/all 32 attention heads for independent mathematics, not 8192-row math.

These recorded results do not qualify the later NVFP4-inclusive source tree,
the full three modules above, SM100/Hopper, every format, or every private
primitive. Fresh NVFP4 GPU/oracle/sanitizer qualification remains outstanding.
The historical default native-FP8 evidence has its own
[separate report](kvfp8-validation.md); it is not optimized BF16 arithmetic.

Logical-valid operands are expected finite. Physical unused padding must be
isolated; stronger isolation of NaNs inside logical but masked inputs is not
promised. Legally consumed nonfinite values must not be hidden by output
sanitization. Independent route oracles must use the customer runtime's
FP32-scaled comparison/tie domain rather than silently substituting raw-score
ordering, which can differ when scaling rounds close scores into a tie.

## Performance

The historical `bench_prefill.py` summary sums separately measured chunk medians
and excludes preparation/cache writes; it is not a direct true-full-prefill timer.
`bench_true_prefill.py --scope full` instead encloses all ordered legal-visible-cache
iterations in one CUDA-event scope. Include adapter, conversion and metadata work in
the declared lifetime; do not sum component medians or mix GPU models/SM counts.
No new whole-matrix or model-quality result is asserted by this document.
