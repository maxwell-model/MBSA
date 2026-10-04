# Latest validation evidence

Host: `10.251.209.5`; target: `maxwell-refactor-serve-min`; reference:
`vllm-compassv4-release-min`. Model validation uses four real layers, not the
full production checkpoint. See [qualification report](../../../VLLM_INTEGRATION.md).

| Evidence | Scope |
|---|---|
| [Installed wheel](installed-fixed-tail-gpu-v2.json) | Ten GPU tests pass; every MBSA module originates in the install root |
| [BF16 full-query oracle](divergence-full-bf16) | Both affected chunks, four layers, 30,148 query/layer entries pass; exact MBSA Remote-route sets |
| [FP8 full-query oracle](divergence-full-fp8) | Same full-query coverage; all MBSA outputs pass the relative/absolute gate |
| [BF16 graph-serving trace](mbsa-prefix-cache-full-bf16.json) | Eight requests, chunk budget 4096, prefix-cache reuse, 64 same-input comparisons |
| [FP8 graph-serving trace](mbsa-prefix-cache-full-fp8.json) | Same workload; native Remote routing disagrees with Python |
| [BF16 operator replay](mbsa-broad-alternating-bf16.json) | Five observed geometries; 19 alternating eager/graph samples |
| [FP8 operator replay](mbsa-broad-alternating-fp8.json) | Same synthetic cases; one remaining target-FP8 performance regression |
| [Reference operator replay](baseline-broad-alternating-bf16.json) | Same synthetic inputs/layout and BF16 precision |

Native Attention passes the Python output gate when using its captured native
routes; the large real-layer disagreement is explained by Remote routing.
Full tensor dumps remain in the target container under
`/tmp/mbsa-vllm-integration/divergence-full-{bf16,fp8}`. Local JSON reports
record the comparisons; they do not claim exact logits, cache-phase generation
invariance, full-model acceptance or exhaustive shape coverage.
