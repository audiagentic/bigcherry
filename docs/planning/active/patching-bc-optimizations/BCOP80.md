---
id: BCOP80
order: 80
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-09T01:04:01+00:00'
created-by: agent
priority: P2
work: S
---

# PRBE12: reconcile RD13 fusion producer and current-pin gate

## Discovery / disposition

Patch 1206 already includes RESHAPE plus PRBE39-qualified zero-offset contiguous VIEW, memory-range protection, WARN activation and direct-ADD fallback. Its producer has emitted positive/control tg128 benchmark lanes and typed promotion evidence since 2026-09-21. PRBE12/README/SUMMARY/TESTING still described missing producers and contract binding. Correct the stale ownership/status; **do not** promote or edit implementation. Historical gfx1100 b11126 four-session gains (+0.47% to +0.66%) are not current b11474 proof; gfx1201 was inconclusive, gfx1030 performance unqualified.

## Owner and dependencies

PRBE12/1206 owns fusion, patch-local producer and performance decision. PRBE39's VIEW/overlap protection is already merged into 1206; PVPS15 owns activation-before-timing; existing contract and validation package own thresholds. No second producer, graph matcher, scheduler or telemetry framework. QFP/MTP, Radiance and engine-registry work active within 12 hours is untouched.

## Terminal gate

First run existing offline matcher/producer tests, adding only missing VIEW/alias and typed positive/control dispatcher cases. Then serial isolated gfx1100/gfx1201/gfx1030 b11474 campaigns with subject-only activation, 64-step full-vocabulary parity, graph replay and fixed-work tg128 controls. Frozen contract: >=4 independent sessions/architecture, >=10 paired rounds/session, CI95-low >0% positive, <=1% negative-control regression, no optional stopping. If no clean benefit or correctness failure, leave 1206 untested/default-off for that architecture; do not infer promotion from older evidence.

## External references

llama.cpp PR #29633 (MMVF dispatch signature/warp size), PR #27220 (Vulkan fusion gating analogue); AMD TLX fused GEMM epilogue discussion is conceptual only, not RDNA evidence. Original fork commit stew675/llama.cpp 0153d580dbc2caa8f29b55ad8ddc7088b4c457dd.
