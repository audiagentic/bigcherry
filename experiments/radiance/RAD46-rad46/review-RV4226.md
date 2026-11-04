---
id: RV4226
review-of: RAD46
plan: patching-radiance-rdna3
state: closed
reviewed-by: codex
reviewed-at: ''
---

# Alternate grouped GEMM passes the tail quality gate

## Notes



## Findings

With the same candidate build and reference, setting R3_MOE_WORKER_ROWS=0 passed all quality limits: helper-off p99.9 2.79229, two-XTX 3.19805, three-helper tail 1.9001. The controls agree with the preceding capture. The changed rows path still fails at 3.71049, isolating the remaining quality difference to that path or its interaction with later helper computation.

## Conclusion

Do not select the slower grouped path as the final fix without meeting the separate throughput acceptance; investigate matching its numerics in the rows path.
