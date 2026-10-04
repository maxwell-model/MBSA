# Source provenance

The project is named Maxwell Block Sparse Attention (MBSA), distributed as `mbsa` and imported as `mbsa`. Literal upstream identifiers, URLs, historical paths and copyright holders below retain their original spelling for traceability.

## Python accuracy oracle

`python/mbsa/reference.py` extracts the block-sparse methods from `CompassMoeBlockSparseGQAAttention` in [the upstream Hugging Face model](https://git.garena.com/shopee/llm/instruction-tuning/compass-training/compassv4_hf_assets/-/blob/d97a6c52967a5ed332e2196a5b985f0b4a475549/modeling_compass_moe.py), revision `d97a6c52967a5ed332e2196a5b985f0b4a475549`, source lines 492–851. The original copyright attribution is preserved.

Query positions, allowed keys, block selection and fixed support retain their source computations. Sparse attention adds an optional natural-log LSE return for alignment. A standalone constructor and NHD adapter replace model projections, cache handling and Transformers integration. Raw index logits have no ReLU; the source probability cast before PV is preserved.

## Triton precision baseline

`python/mbsa/triton_backend.py` retains the router kernels from `compass_bsa/gpu.py` and the latest fixed/dynamic/merge kernels from `compass_bsa/gpu_v2.py` at MBSA revision `7490490`. These six kernel bodies are unchanged. Host functions and public names are consolidated; router score units default to `index_dim**-0.5`, matching the Python source.

## CuTe DSL performance path

The BF16 router, prepared prefill pipeline and state merge are retained from the latest MBSA revision `7490490`, removing legacy producer/precision dispatch and experimental entry points. Host preparation now lives under `python/mbsa/cute/`; GPU scoring and merge code live under `cute/src/sm103/`.

Attributed dependencies are packaged under `python/mbsa/_vendor/`:

- `msa/`: MiniMax MSA BF16 sparse forward, K2 combine, CuTe helpers, and native CUDA CSR/LPT metadata. Dense C++ attention, its JIT stack, quantization and decode were removed. Shared forward-kernel math and hardware helpers remain unchanged.
- `anthony_bsa_backend/`: the existing Fixed BSA snapshot; see its `PROVENANCE.md`. Only the runtime fake-mode predicate remains from the upstream testing helper.

Each dependency preserves its `LICENSE`, `NOTICE`, copyright and source attribution. The repository organization follows [MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA); it does not replace the previously qualified runtime with an unqualified new upstream checkout.

Bundled MSA imports were converted to package-relative imports to avoid global `src`, `interface` and `fmha_sm100` namespace collisions. Kernel computation bodies are retained; host validation and the public API are adapted for the installable Maxwell Block Sparse Attention package.

The October 4 B300 requalification additionally changes the SM103 Router to
score packed requests in one launch and refine uncertain Top-16 membership in
the pinned HF FP32 dimension order. It changes two Fixed BSA TMEM-reduction
predicates to enforce row-specific Prefix/SWA membership at unaligned cached
offsets. Graph warmup now waits on the caller stream, router scratch owners
remain live through launches, and MSA's host dispatch restricts TMA partial
Combine to BF16 after consecutive FP8 shapes exposed strict discrepancies.
These are explicit adaptations of the retained snapshot, rather than claims
that every original kernel body is still identical. Triton kernel bodies and
the independently retained SM90 implementation are unaffected by these fixes.

The H100 path under `cute/src/sm90/` and `cute/sm90.py` is new code written for this package; it does not reuse the bundled SM100 kernels or the CUDA CSR builder. It follows the structure of the CUTLASS CuTe DSL Hopper FMHA example and the FlashAttention CuTe DSL SM90 forward (TMA/WGMMA warp specialization, packed-GQA Q tiles), and uses `quack-kernels` helpers (`sm90_utils`, `layout_utils`, `copy_utils`) at runtime. No source from those projects is vendored.

B300 validation also restores the explicit `head_dim**-0.5` scale at the Dynamic MSA call. For CuTe DSL 4.6, K2's runtime tile guard calls a separate JIT scope to avoid joining newly defined tensors with an absent branch. The guarded arithmetic, masks, synchronization and stores match the original AST; the grid semaphore reset remains outside the tile guard. The CPU import-boundary test runs in a fresh process so prior GPU tests cannot contaminate its module state.

## Figures

The vLLM integration work allows strided PHTD views of interleaved K/V pages
in host validation, refreshes cloned router page maps on plan rebinding, and
accepts BF16 or E4M3FN partial-output storage. Q/K/V and final output remain
BF16. The bundled CUDA CSR extension is named `mbsa_build_k2q_csr_ext` to keep
its modules and build locks independent of the installed vLLM runtime.

The user-prepared `docs/images/maxwell-block-sparse-attention-structure.png` and `docs/images/maxwell-block-sparse-attention-performance.png` are preserved without regeneration. Repository documentation uses these final figures instead of the earlier drafts.

The structure figure describes Indexer routing preprocessing and two Attention paths: Fixed BSA over Prefix-8 + SWA-8 and Dynamic BSA over Remote Top-16, with shared main Q/K/V and state merge. The split maps mathematically to the pinned upstream Python source's attention over the union of selected blocks. The earlier draw.io/SVG generator is archived as `tools/generate_attention_diagram.py` in [the historical source archive](docs/validation/legacy-test-sources.tar.gz). The supplied PNG figures are preserved in `docs/images`; diagram tooling is outside the installed runtime.

The performance figure is supplied comparison data. Its reported GQA/MSA*/MBSA measurements were not rerun during this CPU-host refactor. The new benchmark entry points measure the retained workload explicitly and do not claim to regenerate every comparison series in that image.

Fixed BSA TMEM load-reduction promotion additionally requires complete
packed-GQA query tiles and full KV blocks. B300 vLLM startup qualification
exposed waits for Q=1/KV=16384 and Q=16/KV=16385; tail shapes use the existing
ordinary load/software reduction path. The later row-level Fixed mask fix
described above is a separate change.
