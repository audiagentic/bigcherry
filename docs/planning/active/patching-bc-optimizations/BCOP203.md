---
id: BCOP203
order: 203
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-14T06:09:12Z'
created-by: agent
priority: P2
work: S
---

# QFP47: gate prefill VRAM loan on named physical owner

**Discovery:** QFP47 references former standalone 1339, now part of 1340. 1329 logical tensor ranking, 1331 logical peak, 1340 physical graph-reserved per-device arenas and 1358 CPU split-state metadata are not identified idle decode-only VRAM pools. Historical ub1024 shortfall (~1.3 GiB/XTX, ~2 GiB/R9700) is demand, not supply. No cross-rank P2P aggregation or double-counted 1340 savings.

**Authoritative owner:** QFP47 Gate 0 only. QFP17 owns QSA/ubatch; MSM02/03 and 1340 own Meta arenas; QFP24/1326 own host inputs; MET12 owns expert residency. No new allocator, scheduler, cache, telemetry or implementation package; active Radiance work untouched.

**Eligibility/novelty:** Selected feat/rad03-gpu-lanes from newest independent Radiance development (selection HEAD 1275742cfcdb8b61be2937b174c9bc33e83bcbf4, refreshed before write). QFP47 last independent edit 2026-10-08 11:44 UTC; 1340 last 2026-10-12 23:29 UTC; no QFP47 PR, queued experiment or independent work found in preceding 12 hours. Excluded RAD03 helper/mailbox/mask-only, RAD04 two-half prefill and active engine/release work. Avoided BCOP190-202 slices.

**Evidence:** Pinned b11552 GGML Meta/scheduler allocation and 1329/1331/1340/1358 sources inspected. llama.cpp #29838 diff and vLLM CuMemAllocator code inspected; neither provides GGML mid-request borrowing. Source-shaped host discriminator 10/10 assertions passed after one fixture equality correction; legacy ScratchLease model accepts zero/negative borrow and hangs on empty-event restoration. No repository pytest, HIP build, patch apply, physical GPU census or benchmark. No performance improvement claimed.

**Action/disposition:** One read-only per-physical-GPU census at 245760/f16 ub512/1024/2048, first failed allocation and named decode-only owner. If none provides per-rank releasable bytes and exact restore semantics, close NO_ELIGIBLE_POOL. Otherwise host generation/graph/value fixture first, then one default-off same-binary paired hardware gate under QFP47. Promote only for safe new fit, CI95-low >=3% TTFT/prefill, <=1% decode/control regression and full correctness; else NO_NET_GAIN. No hardware lane queued.
