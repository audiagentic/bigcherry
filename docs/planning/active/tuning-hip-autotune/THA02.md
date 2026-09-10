---
id: THA02
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:48:13.939003+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Fused MUL_MAT(_ID)+GLU correctness-evidence harness (real two-op graph, not the single-op --test-file mapper)

## Description

Build and validate a real two-op fused MUL_MAT(_ID)+GLU correctness-evidence harness with deterministic routing and proof the fused signature executed.

## Steps

Use HI118 fusion flags/geometry and real shared activation/ids tensor identities; fix/init deterministic expert-ID seeding; construct real two mul_mat_id outputs plus terminal GLU (no post-GLU scaling); emit existing digest/metric evidence; require observed signature digest/candidate resolution matches requested; cover remaining dense sibling and behavioral flag toggles; run fresh schema-2 Brutus 4-GPU correctness/record validation and genuine HI83 evidence before promotion.

## Detailed Solution & Technical Design

The single-op mapper cannot represent fused GLU. Add a bespoke ggml C-API harness using exact gate geometry and pointer identity, supported SWIGLU/GEGLU/SWIGLU_OAI only, m==1 restriction, deterministic routing, and observation uniqueness. Preserve schema-2 fail-closed checks and do not certify numerics when dispatch fell back.

## Code Samples & Guidance



## Files

New fused-GLU evidence producer; hi80_generate_correctness_evidence.py integration; correctness_evidence.py shared hook only if needed; deterministic test-backend-ops seed coverage; schema/observation tests.

## Validation

Native/candidate output parity with max abs/NMSE; deterministic repeated routing; exact observed signature/candidate match; fresh schema-2 4-GPU hardware run; HI83-format validation record; full offline suite.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

No promotion until real two-op graph correctness, deterministic IDs, signature execution proof, schema-2 provenance, and HI83 evidence all pass; preserve native fallback and fail closed on ambiguity.

## Notes

Supersedes: HI119
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi119

## Change Log

- 2026-09-09T10:48:13.939003+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:04:15.324062+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.795562+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.270993+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:24:43.274489+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032455_repaired-two-tuning-successors_8434
- 2026-09-10T03:24:55.054433+00:00 (updated-by): Updated: section:ledger-events
