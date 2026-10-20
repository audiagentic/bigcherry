---
id: BCOP344
order: 344
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-20'
created-by: agent
priority: P3
work: S
---

# PRBE15: gate M-RoPE Q/K RMSNorm–RoPE–FWHT fusion

**Discovery:** [RDNA fork #175](https://github.com/stew675/llama-cpp-rdna-boosts/pull/175) adds MROPE/IMROPE to [#172](https://github.com/stew675/llama-cpp-rdna-boosts/pull/172)'s fused Q/K norm–RoPE–FWHT kernel. External gfx1201 decode +1.2%; tensor-split 31.5K prefill +11.7% with single/layer prefill flat, not attributed to the kernel. Pinned b11552 lacks the FWHT fused stage and already rejects BF16 SET_ROWS. Direct fork port invalid.

**Owner:** PRBE15 owns the technical plan. Keep PRBE115/1307 Q8_1 caching, RNX08 norm/RoPE, QFP17 prefill, PNRO04 GDN, PRBE41 SSM separate. No new kernel, dispatcher, allocator, cache, telemetry, hardware queue or technical plan.

**Eligibility:** `feat/shape-harness` from independent Audumla `c6248fb45f74` (2026-10-20 11:05:37 UTC). Shape-harness, RAD14 FP8/MTP and RCS05 worker activity within 12h excluded. PRBE15 last independent edit October 9; no matching BigCherry work or queue in inspected recent repository history; external queues unverified. #175/#172 is material new source evidence since last PRBE15 audit.

**Completed:** pinned model, matcher, ROPE/FWHT dispatch, fork diffs, vLLM/SGLang comparisons; 10 source assertions, 77,568 plane, 600 bitwise butterfly, 300 exact-alias host cases passed. No GPU build, test or benchmark.

**Action:** HOLD_ROUTE_AND_CEILING until first-party executed route and >=2.9126% removable exposed E2E latency; otherwise NO_ROUTE/BELOW_CEILING. Surviving candidate requires MROPE/IMROPE exactness, alias/graph and physical-work parity, ISA/VGPR/LDS/spills and CI95-low >=3% E2E with <=1% controls. Technical detail only in PRBE15.
