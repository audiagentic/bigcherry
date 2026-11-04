---
id: RV4227
review-of: RAD46
plan: patching-radiance-rdna3
state: closed
reviewed-by: codex
reviewed-at: ''
---

# Grouped fold matches replay but needs performance work

## Notes



## Findings

An opt-in stage comparison on the 122-route layer-6 capture found 3 gate and 5 up BF16 mismatches on rank 0, 5 gate and 6 up mismatches on rank 1, with changed intermediate quantisation in 6 groups per rank. A grouped-fold rows prototype produced exactly the same final replay hashes as the alternate GEMM on both ranks and matched its reference relative L2, but three-capture kernel traces rose from 740/519 us to 4445/2721 us for gate/up and down. These are exploratory kernel traces, not end-to-end measurements.

## Conclusion

Do not promote the prototype. Preserve its numerical insight and find a lower-cost batched mapping; the full quality and throughput gates remain open.
