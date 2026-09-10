---
id: PRBE56
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:22.875427+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# UP-MGPU-001: Speculative scheduler caching/reduced cross-GPU synchronization

## Description

Decompose and qualify atomic multi-GPU speculative-scheduler improvements: plan caching, redundant-sync removal, and output mirroring, with independent evidence on dual XTX/R9700.

## Steps

Recheck PR #27173 and split its compound changes; benchmark plan-cache only, synchronization removal only, and output mirroring separately on dual XTX/R9700 MTP widths 2..8; compare single-GPU and speculation-off controls; record scheduler CPU time, sync count, copies/token, effective TG, temp-0 identity and rollback; port only atomic wins and reject mirroring if VRAM/traffic trade is unfavorable.

## Detailed Solution & Technical Design

Mine the compound upstream scheduler bundle into independently reversible subchanges. Cache recurring speculative plans and remove redundant cross-GPU synchronization where dependencies permit; treat output-layer mirroring as a separate tradeoff. Preserve rollback and do not conflate results.

## Code Samples & Guidance



## Files

Speculative scheduler plan cache/sync coordination; optional output-mirroring path; multi-GPU correctness and rollback tests; dual-XTX/R9700 replay evidence.

## Validation

Temp-0 identity and rollback correctness; scheduler CPU time, GPU sync count, copies/token, effective TG; single-GPU/speculation-off controls. Acceptance: port only atomic subchanges with independent wins.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Each promoted subchange must independently preserve temp-0 identity/rollback and show repeatable scheduler or E2E benefit; output mirroring requires favorable VRAM/traffic evidence or remains unported.

## Notes

Supersedes: RD67
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd67

## Change Log

- 2026-09-09T10:57:22.875427+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:36.234929+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.380667+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.183399+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:13:12.765992+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031346_repaired-four-more-migrated-pa_4345
- 2026-09-10T03:13:46.478053+00:00 (updated-by): Updated: section:ledger-events
