---
id: BCOP354
order: 354
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-21'
created-by: agent
priority: P2
work: S
---

# PRBE115: gfx1100 verify-band Q8_1 producer-fusion gate

**Finding:** RDNA fork #192 (https://github.com/stew675/llama-cpp-rdna-boosts/pull/192) admits 2-8-row Q8_1 producer fusion on gfx1100; external E2E +1.9-2.0%, below BigCherry's 3% promotion gate. Pinned b11552 lacks the fork's verify-band switch. BigCherry 1307 already merges Q8_1 cache and RMSNorm/unary/HC/gated producers with no width==1 restriction. Its graph-global quantized matmul <=8 admission can block an otherwise eligible producer; no first-party width2-5 residual is established.

**Owner:** PRBE115 technical gate, 1307 implementation, PRBE113 receipt, QFP13 ranking. No duplicate kernel/cache/dispatch. Active RAD15/16 excluded.

**Eligibility:** Branch feat/bc-gfx12-moe-gemv-default-on independent head 68bc881 (2026-10-21 05:08 UTC). PRBE115/1307 and fork #192 last independent updates predate cutoff 2026-10-20 17:13 UTC. Off-repository queues unknown.

**Validation:** 8 source assertions; five host selector unittest groups including 4,000 seeded cases; Python compile passed. No GPU build or benchmark.

**Disposition:** HOLD_EXISTING_PRODUCER_RECEIPT. Use terminal width2-5 traces, cache misses, graph gate and exposed E2E. Close ALREADY_COVERED_1307, NO_VERIFY_ROUTE, GRAPH_GLOBAL_INELIGIBLE, NO_MATCHING_PRODUCER or BELOW_CEILING. Only a proven >=2.9126% removable E2E opportunity can enter one 1307-owned exact-Q8_1 correction and paired CI95-low >=3% E2E qualification.
