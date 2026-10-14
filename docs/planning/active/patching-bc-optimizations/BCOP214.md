---
id: BCOP214
order: 214
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-14T18:13:00Z'
created-by: agent
priority: P2
work: S
---

# PHA04: distinguish requested RCCL plan from executed candidate

**Discovery:** `rccl_qualify.py` includes channels in case identity/results but does not apply `case.channels` to the RCCL Tests command or compare observed channels in `_classify`; Ring/Simple with mismatched channels can still receive algorithm/protocol `PLAN_VERIFIED`. Patch 0830 observes the BigCherry provider, not the RCCL backend. This is an evidence-integrity risk, not a demonstrated GPU failure or measured speedup.

**Authoritative owners:** PHA04 candidate-verification/admission; PHA03/1225 topology/AtomicOps guard; PHA05 promotion. RQ08/RQ10 retain campaign execution, 0830 telemetry and PGC09/PGC14 provider/performance. No new dispatcher, cache, communicator, registry or hardware queue.

**New external mechanism:** ROCm/rocm-systems develop `05c6bb90` has buffer/op/graph-aware `rcclGetCollImplInfo`, but `RCCL_EXPOSE_STATIC` is required and a query is not dispatch proof. RCCL tuner plugins may silently fall back; do not treat requested values as executed candidates.

**Unresolved action:** Host-only channel/plan-verification regression; inspect exact installed RCCL build/ABI in a crash-isolated probe; correlate one executed collective to an immutable candidate receipt. No candidate allowlist or runtime patch until PHA03 evidence, actual execution identity and a >=3% exposed E2E opportunity exist.

**12-hour and novelty:** `feat/rad03-gpu-lanes` selected from independent `50ba926430330cee746ef688c65f95c912257f48` (2026-10-14 18:06 UTC). PHA04 last substantive change 2026-09-09; PHA05 2026-09-10. No independent PHA04/05 work or related submitted/queued lane in preceding 12h. Excluded active RAD02/RAD03 helper/mailbox/routing/qualification and RAD04 prefill. Recent BCOP205–213 cover different capabilities; prior BCOP114 concerns PHC03 copy safety, not this candidate seam.

**Terminal:** `OBSERVATION_UNAVAILABLE`, `NO_ELIGIBLE_CANDIDATE`, `CANDIDATE_MISMATCH`, `SAFETY_FAIL`, `BELOW_EXPOSED_GATE`, `QUALIFIED_DEFAULT_OFF` or `PROMOTION_READY` only after PHA03/05 and actual hardware evidence. Unknown/substituted candidate fails closed.

**Validation:** 5/5 live-source static assertions and 7/7 synthetic host admission cases passed; no repository pytest, patch application, HIP build, GPU execution or new inference benchmark. Detailed algorithm and gates belong in PHA04.
