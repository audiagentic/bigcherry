---
id: QFP19
order: 0
plan: patching-qwen-flash-next
state: deprecated
created-at: '2026-10-04T09:35:19.931428+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P3
work: S
---

# Reconcile patches broken by the 0504396 pin bump (1250, 1268, 1275, 1320)

## Description

Pin bump c061df198 -> 0504396140d1 (upstream master, 58 commits: #29825, #29824, #29856, #27694 probabilistic MTP). Non-recipe patches with anchor-no-match recorded as known_broken (revision+digest bound): 1250_nro01_allreduce_q8_wire (allreduce.cu nro01-p2p-q8-outer / nro01-q8-provider-route), 1268_prbe52_adaptive_mtp_wiring (speculative.cpp begin-reset / draft-reset / depth-limit), 1275_ar_small_latency (allreduce.cu ar-small-init-once), 1320_meta_compute_timing (ggml-backend-meta.cpp meta-timing-ar / meta-timing-end, collide with a base patch-set edit of the AllReduce loop). 1315/1317 re-anchored in the bump commit.

## Steps

Re-anchor each when it is next needed; 1320 only if the meta timing is needed again (data already collected); 1268 before any adaptive-MTP-depth work; 1250/1275 only if those AllReduce paths are revisited.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes



BACKLOG TRIAGE 2026-11-04 (CLOSE; revalidated against main 6eb31a708, verdict CONFIRM): Close the obsolete 0504396 rebase checklist; keep individual patch dispositions and reopen under a specific owner only on a proven current-pin need. Evidence: docs/planning/active/patching-qwen-flash-next/QFP19.md:14-37 is conditional 0504396 anchor repair, not a live performance lane; engines/llamacpp/patches/1268_prbe52_adaptive_mtp_wiring/SUMMARY.md:1-5,30-40 rejects the adaptive path, 1320_meta_compute_timing/SUMMARY.md:1-9 is validated diagnostic, and 1250_nro01_allreduce_q8_wire/SUMMARY.md:1-9 has no viable production 3-rank route. The only active outside mentions (docs/planning/active/patching-nasone-rdna-optimizations/PNRO02.md:21; docs/planning/active/patching-bc-optimizations/BCOP212.md:16) call QFP19 historical or unrelated, not a prerequisite; re-anchor any future 1275/1320 use under its live patch owner. Still referenced by: BCOP212, PNRO02. Original content kept below for history.

## 2026-10-14 1250 disposition (BCOP212)

Do not re-anchor 1250 merely to clear historical `known_broken`: it depends on rejected 1252 P2P, and its residual hook is two-rank Q8-only, >=1 MiB copy-engine-gated, not exact-F32 or production 1291 CPU-root. PNRO02/PGC11 own the follow-up. The independent 1268/1275/1320 need-triggered rebase items remain untouched.

## Change Log

- 2026-10-04T09:35:19.931428+00:00 (created-by): Created by agent
- 2026-11-04T04:11:08.853027+00:00 (updated-by): Updated: section:notes
- 2026-11-04T04:11:12.789979+00:00 (state-transition): State: pending → deprecated
