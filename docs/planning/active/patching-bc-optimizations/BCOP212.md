---
id: BCOP212
order: 212
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-14T15:10:00Z'
created-by: agent
priority: P2
work: S
---

# PNRO02/1250: gate residual fusion on reachable exact-F32 provider

**Discovery:** b11552 1250 fused-ADD admits only two ranks, Q8_0 wire and >=1 MiB compressed bytes (>=193 tokens at width 5120), excluding normal decode/MTP. Historical PNRO02 incorrectly describes independent exact-F32 fusion. 1250 requires rejected 1252 P2P and untested 1272 host wire; 1291 CPU-root has no fused hook. 1272 SUMMARY's conflict sentence contradicted metadata. No new GPU measurement.

**Owners/consolidation:** PNRO02 conditional future exact-F32 CPU-root epilogue; PGC11/1272 wire; QFP01/1291 provider/lifetime; QFP19 unrelated rebase items. No new provider, scheduler, codec, allocator, cache, dispatcher, telemetry or hardware queue. 1250 remains evaluated/not adopted, 1252 rejected, 1272 untested; active RAD02/RAD03/RAD04 untouched.

**12-hour/novelty:** selected `feat/rad03-gpu-lanes` from independent `e07de541d561d9e91cddfc53713fe520c6433078` (2026-10-14 14:57 UTC). PNRO02/PGC11 last independent Oct 3, QFP19 Oct 4, 1250/1272 mechanical relocation Oct 9. No relevant independent work/PR/queued experiment within 12h. BCOP202–211 do not own Q8 fused ADD.

**Unresolved gate:** prove executed Meta AR->single-use mirrored F32 ADD and >=3% optimistic exposed E2E latency. Otherwise close without code. If admitted, qualify one 1291-owned exact-F32 consume epilogue with safe fallback, numerical parity, multiple requests/ubatches, graph replay, event/rank identity and full transfer/work accounting. vLLM NVIDIA fusion is correctness reference only.

**Terminal:** `NO_MATCHED_ADD`, `BELOW_EXPOSED_GATE`, `UNQUALIFIED_Q8`, `PROVIDER_UNSUPPORTED`, `NUMERIC_OR_LIFETIME_FAIL`, `NO_VALIDATED_GAIN`, `PROMOTABLE`.

**Validation:** 12/12 repository source predicates, 10/10 host-only admission tests and Python compilation passed. No repository pytest, patch apply, HIP build, GPU run or new benchmark.
