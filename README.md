# CompassV4 BSA Opt4 Triton Baseline

Opt4 is an independently authored implementation of the CompassV4
block-sparse attention contract. The repository contains no copied external
attention kernel source and has no runtime dependency on another sparse
attention implementation.

The first revision contains an executable semantic baseline and independently
authored Triton GPU kernels.  It preserves the intended algorithmic
decomposition:

1. block-level Index Score computation;
2. partition-local Top-K followed by an exact global candidate merge;
3. KV-centric CSR construction and query-count-aware work ordering;
4. separate fixed and dynamic attention states; and
5. numerically stable online-softmax-state merging.

The PyTorch implementation is the golden specification.  The initial GPU
revision uses an exact two-stage router and a Q-centric physical dual-path
kernel that evaluates fixed and dynamically selected blocks in one launch.
It is a correctness and performance starting point rather than the final
B300-tuned implementation.

Run the source-independence check and unit tests with:

```bash
python tools/check_independence.py
python -m unittest discover -s tests -v
```

The current upload baseline is revision 2. Its design, constraints,
correctness results, and B300 measurements are documented in
`docs/TRITON_BASELINE.md`.

On a CUDA host, run the stable GPU gate with:

```bash
python bench/bench_triton_baseline.py \
  --revision 2 --kv 8192,32768,65536
```
