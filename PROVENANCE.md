# Source provenance

## Permitted inputs

- The mathematical definition of causal grouped-query attention.
- The CompassV4 model contract: block size, fixed prefix/window support, and
  content-dependent dynamic block count.
- Public PyTorch APIs and public hardware programming documentation.
- Independently generated numerical reference results and latency
  measurements.

## Independent baseline source boundary

- All implementation files in this repository were authored independently.
- No external attention kernel source was copied into this repository.
- The implementation depends only on public Python, PyTorch, Triton, and CUDA
  runtime interfaces.
- External implementations may be treated as black-box numerical or latency
  references, but are not build-time or runtime dependencies.

These statements apply to the `compass_bsa/` Triton baseline and its original
tests and benchmarks. They do not describe the optional `cutedsl-opt1` path.

## CuTeDSL optimized-path boundary

`cutedsl-opt1` uses attributed runtime components under `third_party/`:

- `third_party/msa/` provides Dynamic MSA, K2 combine, CuTeDSL helpers, and
  the CUDA KV-centric CSR/LPT implementation.
- `third_party/anthony_bsa_backend/` provides the Fixed BSA implementation.

The branch preserves the upstream `LICENSE`, `NOTICE`, provenance, and source
copyright headers. Non-runtime upstream papers, benchmarks, tests, and scripts
are intentionally excluded from the dependency snapshot.

The hierarchical BF16 Index Score and exact Top-K implementation in
`kernels/bf16_indexer_cutedsl.py`, together with its integration and benchmark
entry points, is the `cutedsl-opt1` contribution on top of that runtime.

## Revision 1

Revision 1 is an independently authored executable specification.  It uses
streaming partitions, so it never constructs the complete Index Score matrix.
Its performance is not representative of the final B300 kernel.

## Triton baseline (revision 2)

Revision 2 adds independently authored Triton kernels for the tiled fixed
path, KV-major grouped dynamic path, and stable state merge. PyTorch is used
for tensor/runtime orchestration and KV-major sorting. No prohibited source
tree is a build-time or runtime dependency. See `docs/TRITON_BASELINE.md` for
the precise implementation and performance scope.
