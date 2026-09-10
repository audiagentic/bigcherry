---
id: PRBE02
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:53:35.078709+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Establish flash-attention WMMA correctness barriers

## Description

Qualify the materialized flash-attention WMMA barrier/race repair independently from performance configuration. The change must remain correctness-only until repeated validation passes.

## Steps

- Extract and review the barrier/race-repair hunk from 18fc188/patch 1203 as an independent logical identity.
- Define architecture and shape guards; cover head sizes 192/256/320/512/576 where applicable.
- Run targeted correctness under graph and non-graph execution, repeated under load, against native/reference outputs.
- Run the backend corpus and architecture gates; block PRBE03/any performance claim on a failure.
- Record immutable source SHA, resolved patch identity, environment, repetitions and negative/fallback evidence.

## Detailed Solution & Technical Design

Keep barrier synchronization independent from RDNA4 configuration. Treat this as a race-safety patch: no hidden tuning or performance claims. Acceptance requires repeated agreement for affected head sizes and no graph-lifetime or backend divergence.

## Code Samples & Guidance



## Files

patches/1203_rd050607_rdna4_wmma_fa_q6k_mmq; flash-attention WMMA barrier sources; targeted graph/non-graph correctness fixtures; campaign evidence for PRBE02.

## Validation

Targeted heads 192/256/320/512/576 where applicable; graph and non-graph; repeated loaded runs; backend corpus; gfx1100/gfx1201 architecture guards; native/reference numerical comparison; fail-closed behavior.

## Effort & Risk



## Standards

Fail closed on correctness; no performance reliance before prerequisite passes; immutable patch/source identity and reproducible evidence.

## Acceptance Criteria

Barrier repair is independently identified and all targeted shapes pass repeated graph/non-graph and loaded correctness against reference within preregistered tolerances; architecture guards hold; no dependent performance arm is accepted before this gate.

## Notes

Supersedes: RD05
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd05

## Change Log

- 2026-09-09T10:53:35.078709+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:10:12.738042+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.134485+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.804684+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:30:13.076053+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_023037_three-rdna-boost-successors-no_6965
- 2026-09-10T02:30:37.067104+00:00 (updated-by): Updated: section:ledger-events
