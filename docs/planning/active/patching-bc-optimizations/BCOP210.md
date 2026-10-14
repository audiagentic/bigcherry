---
id: BCOP210
order: 210
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-14T13:12:00Z'
created-by: agent
priority: P2
work: S
---

# PRBE54: reconcile archived Q5 GPU failure

**Discovery/change:** Archived b11126 Q5_0 `1271/evidence/validation.json` includes gfx1201 pp512 +0.0143% (CI95 [-0.2110,+0.2306]), full-vocab PASS; gfx1100 +0.7658% (CI95 [-0.0309,+2.0003]), full-vocab **FAIL** (material diff 0.342 > 0.02). One session each, both contract-ineligible. The prior plan overlooked real historical GPU receipts. Not a new benchmark or attributed defect.

**Owners/consolidation:** PRBE54/1271 owns FA-only Q5 converter. BCOP79 already owns Q5_1 half2 parity; RD64 superseded. No duplicate selector, scheduler, cache, allocator or telemetry. QFP17 and PRBE115 are distinct.

**Already acting/exclusion:** PRBE54 last independent plan edit 2026-10-09 00:15 UTC; 1271 code/evidence mechanical relocation 2026-10-09 04:20 UTC. No independent Q5 patch, PR, queued experiment or evidence in prior 12 hours. Active RAD02/RAD03/RAD04 work untouched; BCOP200-209 slices not repeated.

**Unresolved bounded action:** One gfx1100 same-process A/A and A/B Q5_0 full-vocab discriminator, actual FA/converter route, graph scratch and physical-rank proof. If correct, measure exposed converter time and close if optimistic E2E ceiling <3%. Q5_1 needs float-before-half parity or stock fallback before any hardware test. No four-session queue until gates pass.

**Terminal:** `UNSTABLE_REFERENCE`, `Q5_0_CORRECTNESS_FAIL`, `Q5_1_ROUNDING_FAIL`, `NO_EXECUTED_STAGING`, `BELOW_EXPOSED_GATE`, `NO_VALIDATED_GAIN`, `PROMOTABLE`. Present candidate NOT_ADOPTED; registry remains `untested`.

**Validation:** inspected b11552 source and archived hardware receipts, upstream #27140; host arithmetic fixture Q5_0 0/131072 and Q5_1 28727/131072 differences, Python compile passed. No new HIP, GPU, benchmark, patch apply or repository pytest.
