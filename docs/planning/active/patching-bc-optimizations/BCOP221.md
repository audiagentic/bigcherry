---
id: BCOP221
order: 221
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-15T01:08:55Z'
created-by: agent
priority: P1
work: S
---

# RNX11: fence default-off CPU-root large graph replay; retire duplicate fixed-grid patch

**Discovery:** Pinned 1291 small CPU-root uses device-derived per-rank epochs and a single block; its opt-in large path instead bakes host `++l_gen` into HIP stream memops and publishes a descriptor only on enqueue. If HIP captures/replays that sequence, the old generation may satisfy the reused reduction wait without new CPU work. Conditional source-level correctness risk, not a reproduced GPU failure. Large path is default-off and historically ~1450 -> 1060 t/s prefill (negative).

**Owners/consolidation:** RNX11 owns protocol/capture decision, existing 1291/QFP01 owns CPU-root implementation and qualification, PGC10/1244/1276 owns root3 alternatives, PGC09/0840/0860 provider routing, 1252 owns two-rank content-checked P2P, PGC14 RCCL and RNX09 cross-RDNA promotion. No patch 1311, new ring, scheduler, dispatch or telemetry. RNX07 is absent from active RNX; do not resurrect it. Current MEN10, RAD helper/mailbox/placement, RWQ and PGC15 work is protected and untouched.

**Eligibility:** selected `feat/men10-engine-cli` at independent `3d06adfbe8d7` (2026-10-15 00:59Z); RNX11 last independent planning 2026-10-03, no same-path commit, PR, test, queued hardware or evidence in preceding 12 hours. Recent BCOP219/220 cover different mechanisms.

**Unresolved action/terminal:** use existing 1291/RCCL lab to prove `NO_LARGE_ROUTE` or `CAPTURE_REJECTED_SAFE` before any patch. Only if capture succeeds, test changed-input graph replays, generation/descriptor/result publication, per-rank stream/H2D lifetime, zero-contribution masks, mixed gfx1100/gfx1201 no-P2P and CPU-double correctness. A demonstrated stale result must fail closed to RCCL or use a proven minimal device-generation gate; no speculative fixed-grid rewrite. Terminal `REPLAY_STALE_RESULT`, `STREAM_LIFETIME_UNPROVEN`, `NUMERIC_FAIL`, `REJECT_LARGE` or `QUALIFIED_OPT_IN`; promotion requires CI95-low >=3% E2E and <=1% decode regression.

**Evidence:** 12/12 source assertions; 9/9 disposable host-only generation/stream fixtures passed. No HIP build, GPU capture, repo test, new benchmark or new measured gain. Technical design and sources in RNX11.
