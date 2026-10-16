# DeepGEMM-derived MXFP6 scorer

Official repository: <https://github.com/deepseek-ai/DeepGEMM>.

The paged-MQA-logits kernel, scheduler, and public host launcher are pinned to
official commit `a6b593d2826719dcf4892609af7b84ee23aaf32a`. The reduced SM100
helper set is pinned to the public
<https://github.com/maxwell-model/DeepGEMM> fork at commit
`acc39c15353f46dec5ded3e78687d6ef897eea9d`.

## Local files

- `python/mbsa/mxfp6/csrc/sm100_fp6_paged_mqa_logits.cuh` derives from
  `deep_gemm/include/deep_gemm/impls/sm100_fp4_paged_mqa_logits.cuh`. MBSA
  changes the index operand to MXFP6 and adds the per-page score epilogue.
- `python/mbsa/mxfp6/csrc/paged_mqa_logits_scheduler.cuh` derives from
  `deep_gemm/include/deep_gemm/scheduler/paged_mqa_logits.cuh`. It is
  reformatted, uses the reduced helper header, and explicitly publishes shared
  prefix sums before they are consumed.
- `python/mbsa/mxfp6/csrc/sm100_utils.cuh` reduces the listed public
  `common/`, `ptx/`, and `mma/` helpers to definitions used by the scorer. No
  complete DeepGEMM or CUTLASS header tree is vendored.
- `python/mbsa/mxfp6/csrc/fp6_dg_ext.cu` derives its launch structure from
  `csrc/jit_kernels/impls/smxx_fp8_fp4_paged_mqa_logits.hpp`. MBSA provides a
  Torch extension entry point, tensor-map descriptor cache, MXFP6 operands,
  scorer-specific validation, and page max/sum outputs.

These four files are DeepGEMM-derived MIT code, not vLLM-derived files.
DeepSeek's copyright and MIT license are retained. Maxwell Team's adaptations
are released under the same MIT license. `UPSTREAM.json` records the pinned
commits, upstream paths and blobs, upstream hashes, relationships, and current
local hashes.
