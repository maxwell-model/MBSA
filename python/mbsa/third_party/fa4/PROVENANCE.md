# Public FlashAttention-4 Fixed backend

The independent [Fixed metadata adapter](../../cute/fixed_attention.py) calls
`flash-attn-4[cu13]==4.0.0b20` with the exact Prefix8 union causal Local8 mask.
Fixed and Dynamic produce separate outputs and LSE for the MBSA state merge.
No non-public implementation body is reused.

[UPSTREAM.json](UPSTREAM.json) identifies the official PyPI wheel, its SHA-256,
the original SM100 source file and the extracted `kernel` method. The
[LICENSE](LICENSE) and [AUTHORS](AUTHORS) retain that public distribution's
BSD-3-Clause attribution. Most FA4 code remains in the installed dependency.

Installation uses the unmodified official b20 wheel as a separate `--no-deps`
step after installing the `fireworks` extra. That extra pins stable CUTLASS DSL
`4.6.0` and QuACK `0.6.3`, with TVM FFI `>=0.1.12,<0.2`. The b20 wheel's
upstream metadata instead pins CUTLASS DSL `4.6.0.dev0`; ordinary dependency
resolution cannot combine that pin with the validated stable stack.
Validation used PyTorch `2.11.0+cu130`, stable CUTLASS `4.6.0`, QuACK `0.6.3`
and the official b20 source/version guard. The separate install explicitly
overrides the upstream dependency pin without modifying the wheel or its
metadata. `pip check` still reports the CUTLASS metadata mismatch. The
[README installation steps](../../../../README.md#install) include the official
wheel URL with the SHA-256 recorded in `UPSTREAM.json` for reproducibility.

The [SM100 synchronization override](../../cute/fixed_sync.py) derives only
`FlashAttentionForwardSm100.kernel`, lines 799–1320 of the pinned public source.
It inherits the installed public base methods and privately clones the public
`softmax_loop` with its implementation unchanged. MBSA's changes are limited
to TMEM lifecycle synchronization, preparation guards and an FP8 numerical guard:

- Allocation, allocation-permit release and free execute outside dynamic role
  branches, preserving CUTLASS 4.6 allocator bookkeeping.
- Actual MMA, softmax and correction participants rendezvous at one allocation
  wait/retrieve instruction and one retirement instruction. The role union
  excludes inactive or reassigned qstage1 warps. The independent epilogue and
  loader remain outside that participant branch; their role bodies are retained.
- Inactive warps carry a typed null release pointer and do not retrieve an
  unpublished shared-memory TMEM address. Only synchronized participants load
  the actual address; the public allocator internally restricts free to MMA.
- The private softmax factory rescales on every FP8 row-maximum increase.
  The public offset scales probabilities to256; retaining an older maximum
  for four log2 units can saturate E4M3FN probabilities at448 while the FP32
  denominator remains unsaturated. The guard prevents that numerator error
  and preserves the public BF16 rescale policy. Installed classes, factories
  and caches remain unchanged.
- The override requires D128 with matching BF16 or E4M3FN Q/K/V operands
  and BF16 output, tile128, block sparsity, single CTA, non-CLC, independent
  epilogue, one split and non-varlen input.
  The exact Fixed mask callback handles causality; the public base's causal and
  sliding-window modes are not selected. qstage1 and qstage2 use actual roles.

The adapter uses the installed host wrapper's unchanged Python code object,
defaults and closure with private globals, subclass selection and a private JIT
cache shared by synchronized prepared instances. Each instance keeps only its
selected variants for prepared launch binding. The public compile key does not identify the
selected subclass, so the original shared cache cannot be reused safely. No
installed function, class binding or shared cache entry is replaced. Prepared
warmup calls go through the public wrapper. Batch-one prepared plans then bind
the compiled public TVM FFI callable and its normalized sparse metadata.
Rebinding updates the Q/K/V arguments; each launch resolves the caller stream.
No launch recording or global wrapper patch is used.
The public TVM FFI wrapper resolves the current CUDA caller stream at launch.

Before launch, preparation checks the installed version, the official base
source-file SHA-256 and its host-wrapper class binding. A failed preflight
reports the reason.
Kernel warmup failures propagate and cannot trigger a silent fallback.

Physical-flat K/V views preserve live strides without copying the cache. Mutable
tables refresh their mappings on each run; immutable identity tables require
explicit refresh after in-place edits. Address rebinding preserves prepared
metadata and requires graph recapture. The paged adapter uses a 32-lane mask,
reads one inverse page ID per aligned KV tile, and refreshes both block IDs and
the inverse page map with one Triton launch. Fixed and Dynamic states remain
separate.

Numerical, caller-stream, graph, rebind and sanitizer evidence is recorded
separately. Original failing upstream synccheck runs and API-probe sanitizer
results remain part of the validation record. The source attribution itself
does not assert a complete pipeline performance gate.
