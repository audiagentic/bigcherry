---
id: BCOP292
order: 292
plan: patching-bc-optimizations
state: done
created-at: '2026-10-17T23:10:36Z'
created-by: agent
priority: P2
work: S
---

# Retire PNRO18 probe hardening under rejected PNRO03/1252

**Discovery:** 1252 is already rejected (2026-10-08, following 2026-10-07 Brutus fault). Its four historical validation records have no subject activation marker; six-run P2P-requested throughput cannot establish executed P2P speed. The source probe does not exercise production `dev_tmp`/events or threshold±1, but those gaps are not a proven fault cause.

**Authoritative owner:** PNRO03/1252 owns the terminal P2P disposition and any new-evidence requalification. PNRO18 is retired as a duplicate subtask. PGC17 remains separate host transport/topology feasibility; PGC14 owns RCCL SHM; existing `hipMemcpyDefault` hardware fixture is reused, not duplicated.

**12-hour exclusion:** Selected the newest independently active branch `feat/rad06-gemm-counters` at `67faa072c522656c3acbf8503cd2ccd6bc97f953` (2026-10-17 23:17:27 UTC, RAD06), disregarding prior BCOP-only activity. No PNRO03/PNRO18/1252 commits since 2026-10-17 11:00 UTC on the inspected default-branch history; checked recent repository PRs, with no conflicting submission. Off-repository hardware queues were not independently verified. Protected: current RAD06 long-row helper/GEMM/replay and MEN/Radiance qualification. These remain untouched.

**External comparison:** Current HIP documents source-attached peer streams; llama.cpp master retains host staging. Dual-gfx1201 P2P claims and separate dual-gfx1100 RCCL peer stalls do not establish Brutus P2P correctness or performance.

**Action:** No implementation or new experiment. Reopen PNRO03 only after a changed topology/driver receipt, production-path bidirectional byte verification, exact activation and all-rank correctness. Then require CI95-low >=3% end-to-end gain and no material regression; otherwise remain rejected.

**Terminal:** `RETIRED_DUPLICATE` (PNRO18), `REJECTED_P2P` (PNRO03/1252). No new performance claim.

**Validation:** 7/7 host-only admission/threshold fixtures passed; Python compilation passed. No repository pytest, HIP build, GPU run or new benchmark.

**Publication reconciliation (2026-10-18):** Original audit commit object `f00ed64359573c16959e52a09521daf45e5cb86b` was unreferenced on `main`. This disposition was reapplied using normal contents commits on `feat/rad06-gemm-counters`; the orphan object was not force-pushed or merged. The related PNRO03, PNRO18 and 1252 README edits were committed individually and must be verified from branch history.
