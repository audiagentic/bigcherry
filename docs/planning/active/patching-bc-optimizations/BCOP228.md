---
id: BCOP228
order: 228
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-15T08:10:57Z'
created-by: agent
priority: P2
work: S
---

# QFP44: close duplicate compact-grid and Stream-K work; gate residual MoE geometry

**Discovery/change:** On selected independent `main` HEAD `08441aa4` (2026-10-15 05:21 UTC), QFP44 still described 1350 as absent, though 1350 is validated and QFP37 completed. 1237 README/SUMMARY retained pre-promotion status and incorrectly implied exact-R launches. The real 1237/1265 path launches host upper-bound B, computes R on GPU, and guards padded tail. Host-only 256-expert/J128 fixtures: B=256 versus R=2 (single-hot), R=8 (8-hot), R=256 (uniform). Padded blocks are not missing MMQ work; no new gain is claimed.

**Owners/consolidation:** QFP44 is the authoritative residual receipt/decision owner. 1237/1265 own validated compact MoE MMQ; completed QFP37/1350 owns ordinary Q8_0 Stream-K. PRBE22/23/24 remain folded. No second block-map, Stream-K scheduler, J table, allocator, telemetry or 1351 package.

**12-hour/novelty:** QFP44 last independent edit 2026-10-08 06:47 UTC; 1237/1350 last patch reconciliations 2026-10-11 09:51/09:52 UTC. No related BigCherry implementation, PR, benchmark, queue or plan change in preceding 12 hours; recent BCOP219–227 address other capabilities. Excluded active MEN10, RHP/Radiance, RWQ and PGC15 paths untouched.

**Upstream/fork:** llama.cpp #30553 (open, 2026-10-14) caps Stream-K K-work, not compact MoE; #28195 (merged, 2026-10-14) is gfx115x I=64, not a gfx1100/gfx1201 result; #30477 is an unmerged duplicate-expert-ID PoC requiring base routing qualification. AMD-Ecosystem #63 and vLLM align-block-size use host bound plus device-side actual count. No direct port approved.

**Measured history:** 1237 b11126 four-session gfx1100 pp512 ~+7.32% (contract PASS), 1265 separate gfx1201/gfx1030 records around +1% on some lanes, 1350 b11474 Flash-Next ordinary Q8_0 +1.2–1.9%. No new measurement. 12/12 source assertions and 6/6 host-only tests (10,000 randomized expert distributions) passed; no HIP/build/repository test.

**Unresolved action and terminal disposition:** First reuse 1237/1265 receipts for B/R, padded launch count, map/MMQ critical-path time, exact physical I/J/LDS and ordered routing. If exposed removable E2E share <3% or no eligible route, `NO_EXPOSED_HEADROOM` and close QFP44. Duplicate IDs failing stock oracle -> `BASE_ROUTING_UNQUALIFIED` (base expert-ID owner); changed route or numerical/work mismatch -> `CORRECTNESS_FAIL`; only a proven positive MoE-specific occupancy/launch gap permits a single existing-owner opt-in, four-session/ten-pair AMD qualification. Do not queue work against active lanes.
