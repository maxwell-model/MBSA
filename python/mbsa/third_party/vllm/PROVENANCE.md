# vLLM Maxwell-V1-Flash MXFP6 provenance

Base project: <https://github.com/vllm-project/vllm> (Apache-2.0).

The five MBSA files recorded here were traced to the Maxwell-V1-Flash vLLM
development fork at commit
`950d80710610ebffc41128dad89845346262af48`. That commit contains the exact
source paths, Git blobs, and contents recorded in `UPSTREAM.json`. The source
snapshot was in a private development fork at extraction time, so its commit
is not expected to resolve in the public vLLM repository; the blob IDs and
SHA-256 values preserve an internally and mechanically verifiable record.

The MXFP6 implementation first entered that fork in commit
`a0196b22303295eefd397c8f8454dd9460cbe11d` and was later merged into the
fixed source snapshot. It was migrated into the MBSA repository in commit
`8cce3bcda04078db833f31021e9f13c067ee3d97`.

## Local files

- `python/mbsa/index_formats.py` derives from
  `vllm/model_executor/layers/maxwell_moe_index_mxfp6.py`. MBSA keeps the
  reference MXFP6/MXFP8 numerics and packed-page helpers while exposing them
  as a standalone package module.
- `python/mbsa/mxfp6/write.py` derives from
  `vllm/model_executor/layers/maxwell_moe_mxfp6_write.py`. Package imports,
  public launch helpers, and integration details were adapted for MBSA.
- `python/mbsa/mxfp6/scoring.py` consolidates the host-side loading and launch
  logic from `maxwell_moe_mxfp6_kernel.py` and
  `maxwell_moe_mxfp6_dg_kernel.py`, with prebuilt-or-JIT extension loading and
  MBSA's public scoring API.
- `python/mbsa/mxfp6/csrc/fp6_index_kernel.cuh` derives from the fork's file
  of the same name. MBSA replaces the full DeepGEMM include tree with the
  separately attributed reduced helper and includes later synchronization
  and safety changes.
- `python/mbsa/mxfp6/csrc/fp6_ext.cu` derives from the fork's file of the same
  name. It retains the Torch binding and tensor-map cache structure with MBSA
  validation and packaging adaptations.

The fixed fork snapshot is licensed under Apache-2.0. Its Python sources
carry the vLLM contributor copyright header; the C++ files are covered by the
fork's root Apache-2.0 license. MBSA now makes the same license explicit on
all five files and records Maxwell Team's adaptations without removing the
vLLM contributor attribution.
