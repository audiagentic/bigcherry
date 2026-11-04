---
id: RV4228
review-of: RAD46
plan: patching-radiance-rdna3
state: closed
reviewed-by: codex
reviewed-at: ''
---

# Outlier routes mapped and output-lane prototype evaluated

## Notes



## Findings

An AQL-compatible diagnostic mapped 5318 captured tail requests through 131 scheduler tickets with no missing mappings. The instrumented tail reproduced p99.9 3.71049. Position 5458 used layer-15 expert 280 and layer-40 expert 491; position 5459 used layer-11 experts 185 and 424 among five tail routes. A stage replay found one BF16 gate mismatch for expert 424 at position 5459 but no quantised intermediate change on that rank; these associations are not causal proof. An output-owned batched prototype matched grouped GEMM output hashes on both split ranks for these two requests and on three earlier full-width captures. Short traces showed mixed performance by shape.

## Conclusion

Keep pending; run production-layout quality twice and compare end-to-end decode and prefill throughput before accepting any candidate.
