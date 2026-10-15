---
id: BCOP202
order: 202
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-14T05:09:38Z'
created-by: agent
priority: P2
work: S
---

# QFP03/1294: remove dead atomic gather ahead of deterministic ordered TOP_K

**Discovery/change:** At b11552, 1294's default-on ordered radix kernel overwrites all `dst` entries using only input and radix prefix/rank, yet current control flow first runs counter reset and atomic gather. Those two launches and the gather input scan are redundant on the ordered arm, not on tie-only/disabled arms. Existing docs inaccurately call the result globally ascending; it is stable greater-first, tie-second partitioned index order. This is a source-level opportunity, not measured AMD speedup.

**Owner/consolidation:** QFP03 owns the bounded change within existing validated 1294; RNX02 retains D=256 KV attention/1300, QFP15 residual draft nondeterminism, QFP04 ordered-index consumer, QFP25 sparse attention, PNRO07/1256 conflicting alternative. No new kernel, dispatcher, cache, allocator or scheduler. Historical 1294 manifest RNX02 mapping is not a second implementation owner.

**12h/novelty:** QFP03 last independent commit 2026-10-03; 1294 source last changed 2026-10-11T23:22Z. No independent 1294/QFP03 submission, queued lane, benchmark or material planning change was identified after 2026-10-13T17:09Z. Excluded Radiance GPU mailbox/helper, protected expert maps, active RAD03 parking and engine/release work; avoided BCOP190–201 capabilities.

**Evidence:** 11/11 pinned-source checks; 6 fixed + 20,000 seeded randomized host cases matched direct ordered vs gather-then-ordered. Counterexample `[0,2,0,3], k=3` yields `[1,3,0]`, not globally ascending. Historical 1294 cross-start greedy stability at 32K/80K remains valid; 2026-10-03 speed-neutral result predates 2026-10-07 ordered addition. No patch applied, repository pytest, HIP build, GPU run or new performance measurement.

**External:** llama.cpp #28713 is upstream radix/chunk dispatch baseline, not ordered tie fix; vLLM #54521 separates unstable top-k set/order under ties, and SGLang DSA supports deterministic tie-break via FlashInfer. No external throughput result transfers to AMD.

**Unresolved action / terminal gate:** Modify only 1294's chunk-local dispatch to skip reset/gather for deterministic+ordered, preserving all other modes. First pass exact-key and chunk-boundary fixtures, full index parity and patch composition. Then paired gfx1100/gfx1201 runs with physical-route markers, GPU launch accounting, greedy/logit/KLD/MTP and multi-request parity. Promote if CI95-low >=3% exposed E2E, controls <=1% regression and all correctness gates pass. Otherwise keep validated 1294 unchanged and close QFP03/BCOP202 as NO_EXPOSED_GAIN. Do not queue active Radiance hardware.
