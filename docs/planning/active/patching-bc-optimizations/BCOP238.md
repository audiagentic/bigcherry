---
id: BCOP238
order: 238
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-15T18:11:29Z'
created-by: agent
priority: P2
work: S
---

# PGC12: retire unproven phase-aware AllReduce; transfer receipt ownership to PGC09

**Discovery:** PGC12's old 1 MiB MTP +7.98% claim used BF16 host wire. Matched exact-F32 MTP at 1 MiB instead measured about -1.4%; 96 KiB about +0.3%. Current 0860/0840 production is exact-F32, 96 KiB, not 1 MiB. Historical 1 MiB plain pp1024 -4.2% and a 4-token/80 KiB tail do not prove phase-aware dispatch is profitable. 1277 logs a non-atomic, pre-execution preference, not actual provider completion. This is a documentation/evidence admission correction, not a reproduced GPU defect or a new benchmark.

**Ownership/consolidation:** PGC12 phase-policy proposal DONE/retired. PGC09/0840/0860 remains sole runtime provider and absorbs conditional 0830+1277 provider-outcome receipt. PGC13 owns offline 64/96/128 KiB scalar comparison; PGC11 wire, PGC14 RCCL, PGC15 tiled overlap, QFP01/1291 N=3 CPU-root. No new dispatcher, phase cache, scheduler, allocator, telemetry service, patch or hardware queue. MEN10/RAD/RHP/RWQ and recent BCOP236/237 paths remain untouched.

**Eligibility:** selected main at independent 08441aa4a69e34d6905e6449e19467900ae19c95 (2026-10-15 05:21 UTC), newest independent development HEAD after excluding BCOP-only branches. PGC12 last substantive plan work 2026-10-09; PGC09's last main update 2026-10-15 00:20 UTC; no PGC12/0840/1277/PGC13 capability PR, queued lane or commit in the preceding 12 hours. Excluded recent MEN10/Radiance/RHP/RWQ PR metadata updates (06:39 UTC). Novelty: unlike recent BCOP219/220/234-237, this closes the old mixed-wire phase-policy premise.

**Action/terminal:** retain validated t-0840e exact-F32 96 KiB (tg128 +3.87%..+3.98%, pp512 +0.02%..+0.04%, bit-identical). First require a same-binary, same-GGUF, same-wire, same-topology provider-completion/phase receipt and >=3% E2E exposed headroom. If absent: NO_PHASE_CONFLICT / BELOW_CEILING, no further work. Only an independently proven opposite provider optimum at overlapping message size may reopen a default-off PGC09 phase tag, with graph-generation ownership, all-rank event ordering, KLD/greedy/MTP acceptance, multi-request and no missing reductions; failure => REJECT_PHASE_POLICY. vLLM #61848 P2P release/acquire and #57042 rank-order invariance are correctness references, not portable no-P2P performance claims.

**Evidence/validation:** 16/16 repository-source assertions; 14/14 disposable host-only route tests and Python compile PASS. No repo pytest, HIP build, GPU run, new measured speedup or performance profile. Technical design and links belong in PGC12; the BCOP item is disposition only.
