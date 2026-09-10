---
id: THA06
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:48:29.428808+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Winner/tuning persistence across llama.cpp revision bumps -- retain-by-default, retune-as-overlay

## Description

Retain promoted winners across llama.cpp revision bumps using explicit per-model/profile projection and retune only affected signatures.

## Steps

Respect HI125→HI127→HI128→HI129 sequencing; add explicit --retain-from receipt; run tuning-free record-only on new build; build target through normal planner/admission; project prior measurements via replay_projection; use per-candidate identity rather than whole-manifest hash; quarantine failed re-attestation and retune only changed candidates.

## Detailed Solution & Technical Design

No global cross-model store, global revision-match disable, production misses as primary discovery, or duplicate orchestrator. Reuse existing capability and implementation-equivalence projection, with explicit operator-selected prior generation and target provenance.

## Code Samples & Guidance



## Files

workflow refresh path and CLI; replay_projection; catalog/candidate identity; retention tests and A→B evidence.

## Validation

Revision A tune/promote, unrelated revision B change, prove untouched winners replay and changed signatures retune; provenance/capability mismatch rejects.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Retain only per-candidate compatible winners from explicit prior receipt, flag affected candidates for retune, and never silently reuse incompatible or unverified history.

## Notes

Supersedes: HI131
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi131

## Change Log

- 2026-09-09T10:48:29.428808+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:04:32.923217+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.814804+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.299901+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:27:06.082906+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032737_repaired-four-major-tuning-suc_7897
- 2026-09-10T03:27:37.903243+00:00 (updated-by): Updated: section:ledger-events
