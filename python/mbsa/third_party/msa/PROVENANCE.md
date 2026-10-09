# MiniMax MSA pipeline helpers

Source: https://github.com/MiniMax-AI/MSA at
`80434d7f67877c6570ca19cac444b84bc9855dac` (MIT).

Only `cute/src/common/pipeline.py` remains. The native Blackwell BF16/FP16/FP8
and MXFP6 routers use `PipelineTmaUmma` and `PipelineUmmaAsync`. Unused pipeline
types and the Main attention/CSR backend were removed. Retained definitions
preserve their upstream copyright and MIT license. `UPSTREAM.json` records the
original source blob and current local SHA-256; LICENSE and NOTICE are retained.
