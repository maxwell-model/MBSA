# Public Fireworks KV-outer provenance

This record covers `mbsa.cute.kvouter`, used by the default backend. Its public
source and separate-state descriptions do not cover the separately imported
[optimized backend](../../backends/optimized/PROVENANCE.md). No global
clean-room assertion is made for the combined repository.

Public repository: <https://github.com/MiniMax-AI/MSA/tree/fireworks-msa>.
Pinned commit: `e68899832e125f519d913a49e24571b38146d82b`.

The four KV-outer source modules under `mbsa.cute.kvouter` are copied directly
from this public commit's `python/fmha_sm100/kvouter/` directory. Their original
copyright and SPDX headers remain. [UPSTREAM.json](UPSTREAM.json) records source
Git blob IDs, source SHA-256 and resulting local SHA-256.

The index and schedule are Apache-2.0 Fireworks AI code. The forward and combine
retain `BSD-3-Clause AND Apache-2.0`: portions derive from FlashAttention-4
commit `6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15`, as recorded in their upstream
headers. Full texts are retained in [LICENSE](LICENSE),
[FlashAttention LICENSE](flash_attention/LICENSE),
[FlashAttention AUTHORS](flash_attention/AUTHORS),
[MSA LICENSE](msa/LICENSE), and the unmodified upstream [NOTICE](NOTICE).
FlashAttention-4, CUTLASS DSL, QuACK and CUDA Python are installed dependencies;
those packages retain their own distributions and licenses.

MBSA adaptations:

- A new, MBSA-owned `prepared.py` allocates buffers and compiles stages before
  graph capture. It binds the public kernel launchers with fixed tensor addresses
  and shapes. `run_index` refreshes index, inverse map and static balanced schedule;
  `run_forward` emits partials; `run_combine` consumes partials and writes reusable
  output and natural-log LSE. Route contents may change in place on replay.
- Count replicas retain the adaptive heuristic; the A/B environment override
  is removed.
- Upstream host wrappers accept optional caller-owned buffers, a prebuilt schedule,
  and output/LSE buffers. No private global compile monkeypatch is added.
- The MBSA pipeline passes the public `q_load_stage=3` parameter when the device
  capability is SM103 `(10, 3)` and partial storage is BF16. Other devices or
  partial types pass `None` and retain the upstream architecture defaults.
  This choice comes from independent B300 measurements and changes the
  preparation call only; the four public source modules are unchanged by this
  tuning.
- The installed CUTLASS 4.6 MMA helper receives separate A/B dtype arguments.
  The legacy `cute.core.ThrMma` type annotation uses `cute.ThrMma`; deprecated
  `cute.make_fragment` register tensors use `cute.make_rmem_tensor`.
  TMEM allocation, allocation-permit release and free execute outside dynamic
  role branches so the allocator can track ownership. The original participants
  use one converged allocation/retirement barrier.
- A final half-full page64 sparse block uses a harmless allocated page for a
  missing second physical page. Real `used_kv_lens` masks that half. Prepared
  block-slot capacity rounds up page-table width instead of dropping the tail.
- Strided cache views retain their live underlying storage, with contiguous D
  and 16-byte-aligned positive outer strides. Preparation does not copy K/V;
  in-place cache updates are visible to forward launches. `rebind` updates Q,
  K, V and page-table addresses for eager prepared calls with identical tensor
  metadata; CUDA graphs must be recaptured after rebinding. Rank-contiguous
  selected-route views with dynamic outer strides also avoid a per-call copy.
- Package imports are local and lazy at the entry point; compiler initialization
  happens during preparation. Existing inactive upstream DSL expressions are
  retained and explicitly excluded from unused-variable lint checks.

The forward remains public Fireworks' packed-GQA, three-warp cooperative
`cp.async` gather, resident KV block, compact selected-slot traversal and static
load-balanced persistent implementation. It is separate from Fixed attention;
no Fixed-state fusion is introduced. Raw tile-ordered partials and per-row
`m`/`l` are merged using the public gather-combine kernel.

Public Fireworks supports BF16/FP16/FP32 partial storage. MBSA adds scaled
FP8 partial storage below. Temperature-scaled LSE is not exposed by this adapter.
FP8 Q/K/V inputs retain the upstream FP8 compute path with BF16 partials.

GPU compatibility, numerical correctness and performance must be established
using the validation report; this source/provenance file does not assert
that the integrated backend has already passed those checks.

The separately attributed Fixed path uses the installed public
`flash-attn-4[cu13]==4.0.0b20` package, not these Fireworks kernel bodies.
Its MBSA metadata adapter, BSD-3-Clause wheel attribution, public kernel
synchronization override, private JIT cache and TVM FFI caller-stream contract and pinned b20 source guards are described in the [FA4 provenance](../fa4/PROVENANCE.md).

## FP8 partial storage

The BF16 raw-partial path is unchanged. Explicit E4M3FN partial storage normalizes
P before PV, so an unnormalized numerator cannot overflow the 448 FP8 limit.
The correction stage scales each output row to a maximum magnitude of 416 or
less, and stores that scale in the FP32 L plane. For this variant, M holds
log2 of the actual attention mass. Combine sums mass for its denominator and
multiplies each output weight by its recorded scale. Empty routes remain zero
output with -inf LSE. This is an MBSA semantic adaptation of the public source,
not an unmodified upstream ABI. Debug artifact switches were removed.
