# Default KV caches are unchanged by rad0037/38/40/41/42 (VTP44)

## Summary
With the patches applied, Flash-Next's default bf16 and fp8 caches give the same output as the stock build, to the KLD
tool's float floor. The packed layouts are additive: they do not move the caches that were already there.

## Outcome
`qwen3.8-next-flash-hcbf16`, two 7900 XTX TP2 exact wire, experts tiered, placement pinned (`--deterministic`), 64-token
chunks, no speculation, two documents (4,161 positions). The stock build (no experiment) recorded a reference for each
cache; the `kv-bf16-packed` build scored the same cache against it.

| cache | mean KLD | p99.9 | max | top-1 |
|---|---|---|---|---|
| bf16 | 6.4e-11 | 1.9e-08 | 2.0e-08 | 100% |
| fp8 | 3.3e-11 | 1.8e-08 | 2.2e-08 | 100% |

## Detail
These are not literally 0 because the two sides are different binaries and the stored reference keeps the top 64
log-probabilities plus a tail; the same value (about 5e-11 mean, 2e-8 max) is the floor the tool gives for a comparison
with no arithmetic change (the VTP43 27B bf16 repeat, RAD53's identity check). A same-binary repeat gives exactly 0.
KLD at the floor with 100% top-1 agreement means the logits agree to about 1e-5. A stricter check would compare
greedy token streams across the two builds; it was not run.

Builds: stock `radiance:stock:linux-radiance` at origin/main bdb299751; patched is the same lane with experiment
`kv-bf16-packed`. The 27B default caches were not re-run here: patch A's 27B KLD equality is separate and still open.
