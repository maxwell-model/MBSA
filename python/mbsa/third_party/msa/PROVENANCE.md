# MiniMax MSA-derived helpers

Source: https://github.com/MiniMax-AI/MSA at
`80434d7f67877c6570ca19cac444b84bc9855dac` (MIT).

## Pipeline helpers

`cute/src/common/pipeline.py` retains the two pipeline types used by the native
Blackwell BF16/FP16/FP8 and MXFP6 routers: `PipelineTmaUmma` and
`PipelineUmmaAsync`. Unused pipeline types and the Main attention/CSR backend
were removed.

The package marker files `cute/src/__init__.py`,
`cute/src/common/__init__.py`, and `cute/src/sm100/__init__.py` also correspond
to files at the fixed upstream commit.

## CUDA JIT helper

`python/mbsa/_cuda_jit.py` evolved from the CUDA build and include-path helper
code in:

`python/fmha_sm100/cute/src/sm100/build_k2q_csr/__init__.py`

It was refactored into a shared module for MBSA's CUDA C++ extensions and
extended with prebuilt-versus-JIT extension loading, NVIDIA CUDA wheel include
discovery, architecture selection, template-stub flags, and CUTLASS path
discovery. It is therefore recorded as `derived-and-refactored`, not as an
unchanged copy of the upstream file.

The MiniMax copyright and MIT license are retained. Maxwell Team's
refactoring and extensions are also released under MIT. `UPSTREAM.json`
records each upstream path and Git blob plus the current local SHA-256;
LICENSE and NOTICE are retained.
