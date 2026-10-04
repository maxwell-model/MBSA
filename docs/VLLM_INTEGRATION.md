# vLLM integration and qualification

本文记录历史执行快照。旧 benchmark/diagnostic 命令的源码已归档到
[legacy-test-sources.tar.gz](validation/legacy-test-sources.tar.gz)；当前入口为
`tests/test_precision.py`、`tests/benchmark_performance.py` 和 `tests/config.json`。


The full acceptance target remains active: real serving shapes for chunked
prefill and decode, correctness and controlled performance comparisons against
the previous sparse-GQA code and the `vllm-compassv4-release-min` container,
production vLLM integration, and a BF16/FP8 partial-output switch.

The source suite (19 tests), installed-wheel GPU suite (10 tests), and retained
native decode replay (11 cases in each container) have passed. Four-layer
startup and graph serving work. Broader mixed-chunk comparisons disagree with
native unified; MBSA passes the Python oracle across both complete affected
chunks at all four layers in BF16 and FP8. Overall acceptance remains open pending broader/full-model
qualification.

## Evidence recorded on 2026-10-03

Host `10.251.209.5`, container `maxwell-refactor-serve-min`, B300 SM103.
The installed native vLLM is `0.26.0rc2.dev25+gc0a8b6c2b`.
The four-layer checkpoint loaded 3.51 GiB of real weights. Its original
geometry and RoPE parameters were retained; broken framework/tokenizer asset
links were repaired in an independent temporary model directory. The original
checkpoint and installed vLLM sources were left intact.

`tools/trace_vllm_attention.py` instruments actual class calls under eager
execution. Metadata synchronization makes this a shape trace, not a performance
measurement. The successful native run used token budget 8192, maximum length
16384, prompt lengths 129/4097/8193, four decode steps and KV memory utilization
0.03. A prior attempt at 0.07 exhausted available memory during CSR warmup.

| Actual serving call | Request `(Q, KV)` lengths | Calls across four layers |
|---|---|---|
| Unified prefill | `(129,129), (4097,4097), (3966,3966)` | 4 |
| Unified prefill, mixed decode/chunk | `(1,130), (1,4098), (4227,8193)` | 4 |
| Decode | `(1,131), (1,4099), (1,8194)` | 4 |
| Decode | `(1,132), (1,4100), (1,8195)` | 4 |
| Decode | `(1,8196)` | 4 |

Native K/V cache views have strides `(131072,1024,256,1)` in
`[pages,128,4,128]` order. MBSA consumes the permuted PHTD view directly,
including the gaps between K and V. It does not copy the full cache.

The FP8 partial implementation passed all 15 then-current tests without skips
(`154.467 s`). The corresponding wheel SHA-256 is
`d83f8672b426bf494639cdd60c9ac41a85ea85b9ddfc48ffc322b3ba3a404742`.
Subsequent individual GPU tests passed for a strided paged cache with Q=1/17,
KV=4113, and for plan rebinding from identity to permuted Index-K pages with
new input/output pointers. These later edits are not in that earlier wheel.

The packed Fixed Attention path now submits the prepared request-tile list
once, removing per-request Q/output staging and repeated kernel launches.
All 18 then-current CPU/GPU tests passed on B300 with `MBSA_FP8_PARTIAL_OUT=1`,
without skips, in 57.490 s. This includes packed single-token chunks and reuse
across changed layer tensors/page maps. Post-change replay is being measured.

Earlier synthetic replay used a page-half K/V layout. Its strided-layout
checks remain useful, but its timings do not qualify the exact native cache
layout. The layout was corrected on 2026-10-03; all 19 tests passed on B300
without skips in 35.483 s with `MBSA_FP8_PARTIAL_OUT=1`.

Prepared static-shape CUDA Graph replay passed precision gates on both
observed prefill batches. MBSA medians were 0.556/0.568 ms, versus native unified
0.553/0.624 ms. These are captured operator plans with stable pointers; they
do not establish full vLLM graph serving. The FP8-partial same-engine A/B run
matched all 12 generated token IDs across three requests. BF16's previously
recorded tie-related divergence still requires broader model evaluation.

Raw native trace, generations and FP8 test logs are under
[validation/vllm-20261003/mbsa-vllm-evidence](validation/vllm-20261003/mbsa-vllm-evidence).
The earlier [B300 qualification](VALIDATION_B300.md) records a distinct
13-test snapshot and wheel; its hash does not describe current integration work.

## Precision switch

`MBSA_FP8_PARTIAL_OUT=0` (default) stores BF16 partials; `=1` stores E4M3FN
partials. `prepare_cutedsl_pipeline(partial_dtype=...)` overrides this default.
Q/K/V and final output are BF16; LSE/state reduction remain FP32. The vLLM
adapter passes the existing `self.prefill_partial_dtype` explicitly, preserving
the production `MAXWELL_BSA_PREFILL_FP8_PARTIALS` choice.

## Adapter under qualification

`mbsa.vllm.PagedPrefill` is an eager adapter with four cached plans shared
serially across layers on each device. Keys include exact request lengths,
precision, shapes and strides. Hot calls refresh all cloned Index-K mappings,
flattened Index-K views, main inputs, page tables and fused output aliases.
New geometry prepares a new plan and evicts before allocation at the limit.
The caller owns page-ID validity and serial execution on the device.

An isolated vLLM worktree at `/Users/min.yang/gitlab/vllm-mbsa`, branch
`integration/mbsa-vllm`, adds opt-in `MBSA_VLLM_PREFILL=1` dispatch before the
old TileQ=256 guard. The trace tool's `--mbsa` runtime hook tests serving
without modifying the installed container package. That hook deliberately
uses native startup warmup by default; `--mbsa-warmup` tests MBSA startup, so its success alone cannot qualify MBSA startup
warmup or graph serving. Decode remains the native Triton split-K path.

Outer CUDA graph capture and quantized Q/K/V or index caches are rejected by
the adapter. Staged graphs and four-layer startup are now implemented and
qualified on the workloads below; broader coverage and geometry-independent
workspace capacity remain open.

`MBSA_VLLM_PREFILL_GRAPH=1` is an experimental staged graph adapter. Query,
Index-Q, page mappings and output use stable buffers; per-layer graphs bind
main and Index-K cache pointers. K/V pools are read directly. Four geometries
and up to 64 layer graphs per geometry are retained; layer graphs share the
geometry's partial/CSR workspaces. This adapter is being qualified, and still
rejects entry during an outer vLLM graph capture.

## Remaining acceptance gates

| Gate | Current state |
|---|---|
| FP8 partial storage and explicit override | GPU unit tests passed |
| Real native call chain / initial shape trace | Passed for one workload |
| Interleaved paged K/V, KV tails, Q=1 | Individual GPU test passed |
| Layer/page/output rebinding | Individual GPU test passed |
| Installed wheel and runtime origin | All ten GPU backend tests passed; every MBSA module loads from the install root |
| MBSA real-model serving A/B | FP8 eager and staged graph matched 12/12 generated tokens; BF16 longest sequence diverged at token 3 |
| Prefix cache, budget boundaries, long context, broader batches | Affected complete MBSA chunks pass Python; native Remote routing disagrees; broader evaluation remains open |
| Normal and MTP decode correctness / performance | Both containers: 11 cases passed; baseline MTP requires serial emulation |
| Previous sparse-GQA and reference-container controlled comparison | Five additional shapes pass; MBSA BF16 faster than reference; one target-FP8 regression remains |
| BF16 and FP8 real-model tokens / logprobs | Both recorded; exact logprob equivalence is not established |
| Production warmup, CUDA graphs, memory and throughput | Startup and graph-serving passed on four-layer model; broader coverage/full-model gate remains |

`benchmarks/bench_vllm_replay.py` replays observed prefill geometries with
deterministic synthetic values in interleaved paged storage
`[pages,128,4,2,128]`, with K/V views at strides `(131072,1024,256,1)`
and V offset by 128 elements. It gates timings
on correctness against native sparse GQA and compares hot eager CUDA-event
and synchronized wall times under the same warm-L2 policy. Replay performance
must not be described as end-to-end model throughput or complete shape coverage.

The first controlled BF16 replay measured MBSA's pre-optimization eager adapter
at 1.403/1.367 ms on the two observed prefill batches, versus current native
unified at 0.893/0.932 ms. The reference container measured 0.944/0.986 ms
for unified and 1.208/1.139 ms for sparse GQA on identical synthetic geometries
and precision. Each case used nine alternating warm-L2 CUDA-event samples.
These initial MBSA numbers do not satisfy the performance parity requirement.

Real-layer A/B passed `rtol=0.03, atol=0.03` for all 24 request/layer pairs,
with max error 0.03125 and max RMS error 0.001455. Same-engine generation A/B
recorded a tie between the reference's first two logits at the longest request's
third token. MBSA chose the reference's second-ranked token; each path's chosen
token belongs to the other's top-5. Some top-5 sets differ, and common-token
logprob deltas reach roughly 0.146 before divergence. This is evidence to
evaluate further, not a claim of exact model equivalence.

MBSA's CUDA CSR extension uses the private name `mbsa_build_k2q_csr_ext`.
Sharing the native vLLM extension name caused alternating source paths to
recompile the same cache entry at process startup. The distinct name keeps
native and MBSA extension modules and build locks independent.

## Exact cache-layout rerun

The corrected replay gates both output comparisons on `atol=rtol=0.04`.
All five cases passed. Preliminary target-container medians for the first
actual batch are 0.743 ms for MBSA staged graph adapter, 0.891 ms for native
unified eager, and 1.200 ms for native sparse GQA. Prepared graph medians are
0.557 ms MBSA and 0.554 ms native. The Q=8192/KV=32768 boundary passed as well:
1.005 ms MBSA staged adapter versus 1.019 ms native unified eager, and
0.895/0.962 ms for their prepared graphs. These are operator measurements,
including staging where specified, not end-to-end serving throughput.

The staged graph real-model FP8 A/B matched all 12 token IDs. Two of twelve
top-5 token sets differ; common-token logprob differences reach 0.168569.
Token equality on this workload does not imply identical logits.

`benchmarks/bench_vllm_decode.py` independently qualifies the retained native
Triton decode path, including request-major MTP verification lengths 2/4/8,
per-query causal positions, KV tails and live-input graph replay. Its oracle
is the retained Python reference. It can be run unchanged in both containers.

A reviewable [vLLM integration patch](integration/vllm-mbsa-prefill.patch)
corresponds to the isolated worktree. Unsupported quantized index operands
are excluded by the vLLM capability gate before opt-in MBSA dispatch.

Startup qualification initially stopped while loading another CUDA module.
Stage-by-stage synchronization isolated the pending kernel to Fixed Attention
at Q=1/KV=16384. The reduced TMEM (`LdRed`) path stalls on this partial packed
query tile. Disabling just that path made the probe pass with CLC and split
barriers unchanged. Its promotion now requires complete 128-row packed-GQA
tiles and full KV blocks; query/KV tails use ordinary TMEM loads/software reduction. The initial
module-load stack was a symptom of the earlier pending Fixed kernel, rather
than an independently established module-loading bug. Serial priming was
removed after isolation. All 19 CPU/GPU tests, including the added query/KV tail
cases, passed without skips in 34.396 s on B300 with FP8 partials. The subsequent
startup qualification is recorded below.

Target decode qualification passed all 11 observed/boundary cases, with exact
complete route sets and output max absolute error 0.0078125 against the Python
oracle (including live-input graph replay). Additional verification lengths are
2/4/8; batches are 1/3/8 and KV boundaries range from 128 through 32769. This
qualifies the retained native decode kernel, not a new MBSA decode backend or
end-to-end speculative decoding with an MTP model.

The exact-layout reference-container replay also passed all five precision
gates. Its unified eager medians are 0.927/0.988 ms on the two observed batches,
and 1.176 ms on Q8192/KV32768. Prepared graph medians are 0.580/0.640/1.012 ms.
MBSA's corresponding staged adapter medians are 0.743/0.739/1.005 ms and its
prepared graph medians 0.557/0.567/0.895 ms. Short Q1/Q128 cases are refused by
reference unified but pass MBSA's precision gate. Raw exact-layout results
supersede the earlier synthetic page-half layout timing qualification.

The corrected eager-engine startup dispatched 100 MBSA warmup calls and
completed initialization in 114.09 s. FP8 generation A/B matched all 12 tokens.
Five alternating, trace/logprob-disabled generation samples measured median
126.224 ms MBSA versus 128.576 ms native in that eager engine.

A separate production graph-serving run enabled vLLM PIECEWISE prefill and
FULL decode capture. Its trace recorded 16 capture calls and eight actual
serving MBSA prefill dispatches; graph replays bypass Python hooks. All 12
FP8 generation tokens matched the same-engine native run. Five hot generation
samples measured median 51.252 ms MBSA versus 53.396 ms native. This run used
native startup warmup; the separate eager startup test qualified MBSA warmup.
The four-layer model is not full production-model throughput qualification.
Graph profiling/capture retained approximately 0.04 GiB for native vLLM graphs;
this does not include MBSA's additional plan/staging allocations.

The reference container's decode API supports one query per request and has
no caller-output-buffer parameter. The decode replay detects those API
capabilities. Normal queries use its native path; MTP lengths 2/4/8 explicitly
use serial per-token emulation with concatenation and causal length updates.
All eleven reference cases passed exact routes and live-input graph replay.
These emulated MTP timings must not be presented as a native batched MTP
baseline. On the observed normal batches, target decode graph medians are
0.0207/0.0209/0.0188 ms versus reference 0.0233/0.0240/0.0214 ms.

## Broader mixed-chunk precision investigation

The FP8 four-layer graph-serving run with prefix caching, a 4096-token budget
and prompt lengths 1/127/128/129/2047/4095/8193/16257 completes operationally,
but **does not pass acceptance**. The second request in packed geometries
`(255,4095),(3836,7804)` and `(389,8193),(3701,11893)` exceeds the
`atol=rtol=0.03` native-output gate at all four layers. Maximum absolute errors
reach 1.5234375 and 1.5 respectively. The rerun using native startup reproduces
these errors, with 20 serving and four cached-serving MBSA dispatches and 64
phase-labelled comparisons.

The initial cached generation also differs on the 4095-token request.
Chunk boundaries change with prefix-cache hits, and native vLLM itself warns
that this model's routing can change with chunking; token equality across
cache phases alone therefore cannot identify an MBSA defect. Same-input
layer comparison and the retained Python oracle are the acceptance evidence.

`tools/trace_vllm_attention.py --dump-divergence DIR` saves the worst 16-query
window for each failing request/layer, logical K/V and Index-K gathered from
the actual page maps, query positions and MBSA routes. Run
`tools/check_vllm_divergence.py DUMP --output REPORT` with the source or installed
package on `PYTHONPATH` to compare both implementations with the oracle, and
to check MBSA Attention separately using its captured route. The first-window
probe did not cover the maximum errors. The subsequent eight worst-query
windows (two geometries, four layers each) all match the Python remote-route
sets exactly and pass the MBSA output gate. Their MBSA maximum absolute errors
are 0.03125–0.0625; relative tolerance is part of the gate. Native unified fails
in the same windows, with maxima 0.59765625–1.53125. This isolates the sampled
large discrepancy to native unified relative to the Python semantics. It does
not establish full-query oracle coverage. A further rerun captures native
Remote routes before MBSA executes: all eight windows have a native-route set
match fraction of zero, versus MBSA's one. Evaluating Python Attention with
the captured native routes makes native output pass in all eight windows.
The large sampled difference is therefore explained by native Remote routing,
rather than the fixed/dynamic Attention arithmetic. Some native-added blocks
have exact oracle scores of zero while omitted blocks have substantially
positive scores, so this is not explained by equal-score Top-K ordering alone.
The underlying native routing/page-plan cause and broader qualification remain
open; native-output identity is not a valid correctness conclusion here.

The same four-layer graph-serving/cache workload also completes with BF16
partials (20 serving and four cached-serving dispatches, 64 comparisons).
Its eight worst-query windows reproduce the routing diagnosis: MBSA routes
match the Python sets exactly and all MBSA outputs pass, with maxima
0.015625–0.03125. Native route-set matches are zero in these sampled windows;
native Attention passes when Python uses the native routes. Both precision
modes produce equal initial/cached token sequences on seven of eight requests,
with the known chunk-dependent difference on the 4095-token request. These
samples do not qualify exact generation/cache invariance or full-model output.

Full-query reruns then save every query of each failing request/layer using
`--dump-full-divergence`. Both BF16 and FP8 pass the Python output gate on all
3836 and 3701 queries at each of four real layers (30,148 query/layer entries
per precision). Every MBSA Remote-route set matches Python exactly throughout
these complete chunks. BF16 maximum absolute errors are 0.03125; FP8 maxima
range from 0.046875 to 0.125, with the relative/absolute gate passing for every
element. Native Attention also passes throughout when Python is forced to use
the captured native Remote routes, reinforcing the routing diagnosis beyond
the initial worst-window samples. Full-model evaluation and other geometries
remain distinct acceptance gates.

The installed-wheel GPU suite subsequently passed all ten backend tests,
without skips, failures or errors, in 139.248 seconds. Every loaded `mbsa.*`
module resolves inside `/tmp/mbsa-vllm-integration/installed-fixed-tail`;
the wheel SHA-256 is
`fb7deb10037c79878d209f7c21a0b15591e2fd2acbb2dfbac9a0667b181cb991`.
The rerun's stack samples show cold CUDA-extension and CuTe compilation before
completion. The earlier 180-second timeout alone did not establish a kernel
hang. Broader accuracy and full-model gates remain outstanding despite the
installed package, earlier three-request and decode successes.

## Broader controlled operator replay

All five new observed geometries pass the synthetic `atol=rtol=0.04` gates
for MBSA with BF16 and FP8 partials, and for the reference container with BF16
partials. Inputs use exact native interleaved cache strides and permuted pages.
The replay reserves the scheduler's rounded TileQ=256 capacity (4096 for a
4095-token live batch), rather than underallocating the native bucket.
Nineteen alternating warm-L2 samples are taken for both eager submission and
prepared graph comparison; compilation and allocation are excluded.

CUDA-event medians in milliseconds compare the staged MBSA graph adapter
(including input/map/output staging) with native unified eager submission:

| Requests `(Q,KV)` | MBSA BF16 | Target BF16 | Reference BF16 | MBSA FP8 | Target FP8 |
|---|---:|---:|---:|---:|---:|
| `(1919,2047),(1920,3840),(1,1),(127,127),(128,128)` | 0.5131 | 0.8377 | 0.9083 | 0.5005 | 0.8367 |
| `(255,4095),(3836,7804)` | 0.6902 | 0.8437 | 0.8694 | 0.6537 | 0.8018 |
| `(389,8193),(3701,11893)` | 0.7002 | 0.8411 | 0.8828 | 0.6618 | 0.8114 |
| `(4089,15982)` | 0.6976 | 0.8147 | 0.7858 | 0.5837 | 0.5576 |
| `(275,16257)` | 0.3177 | 0.4977 | 0.6132 | 0.2503 | 0.3059 |

MBSA BF16 is faster than reference unified eager on these five cases, by
approximately 11–48%. Its prepared BF16 graphs are also faster than the target
and reference prepared graphs in all five cases. FP8 is not uniformly faster
than the target: Q4089/KV15982 takes 0.5837 versus 0.5576 ms with staging
(4.7% slower), and 0.4799 versus 0.4490 ms for prepared graphs (6.9% slower).
This remaining target-FP8 regression is recorded rather than hidden by the
BF16 reference comparison. These are synthetic operator timings, not full
model throughput or proof that the real serving routing discrepancy is fixed.

A 31-sample Q4089/KV15982 diagnostic disables MBSA Fixed/Router overlap at
preparation. Its prepared graph remains 0.4802 ms versus native 0.4526 ms;
disabling overlap does not resolve this regression. The override is recorded
in `fp8-no-overlap-probe.json`; it is an experiment, not the shipped default.

Raw results are `mbsa-broad-alternating-{bf16,fp8}.json` and
`baseline-broad-alternating-bf16.json` in the evidence directory.
