---
id: THTR05
order: 0
plan: tuning-hip-tune-recovery
state: pending
created-at: '2026-09-09T10:51:38.359111+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P0
---

# Adaptive recovery: bisect with the shortest vector that actually reproduces the failure

## Description

Optimize recovery bisection by selecting the shortest vector that reproduces the same first acceptance divergence, with full-corpus final validation.

## Steps

Run full corpus once; identify failing vectors and first divergence; select shortest reproducer, optionally minimize n_predict; use only that oracle for bisection/alternative search; cache deterministic native trace; validate final candidate cache against immutable full corpus before publish.

Keep the existing full-corpus shortest-reproducer recovery protocol. Add the following separate bounded evidence/safety tracks consolidated from THTR01-THTR04; none changes production policy by itself:
- BehavioralFailureWitness identity: bind candidate/signature/source/build/model/profile/MTP args/hardware/runtime/corpus/assignment/failing set/verdict/divergence/schema. Exact-context failure may be skipped; same candidate+environment with different ensemble may only be downranked; different context is diagnostic-only. Invalidate witnesses on identity change and require HI143 behavioral PASS before publishing.
- Recovery escalation is recommendation-only: expose reason, provenance and suggested next campaign. It must never autonomously retune, promote, blacklist or mutate cache. Absent/ambiguous evidence remains non-actionable.
- Multiplicity correction is a study only: use archived native-twin/known-null campaign data preserving cross-signature correlation to compare BH/BY/Holm under a fixed prospective hypothesis family. Design a versioned policy only if material BH risk is empirically shown.
- Synthetic-to-E2E fidelity is a separate bounded empirical study: select high-cost/frequency signatures, compare tune winner/alternatives in fixed full-cache ensembles differing one signature, use randomized/ABBA sustained runs with p50/p95/variance, and quantify inversion frequency/effect/uncertainty. Only repeatable material inversion across independent ensembles can justify a separate follow-up.

## Detailed Solution & Technical Design

Never use a cheap vector that cannot observe the failure. Oracle selection is anchored to same first divergence, guarding against isolating a different defect.

## Code Samples & Guidance



## Files

Recovery oracle selection/bisection, native trace cache, full-corpus validator and tests.

## Validation

Known failure oracle reproduces same divergence, isolates same signature as full bisection, final cache passes full corpus.

Existing THTR05 gate: known failure oracle reproduces the same first divergence; shortest-reproducer isolation matches full bisection; final candidate cache passes the immutable full corpus. For THTR01, synthetic three-level exact-context/different-ensemble/different-context policy fixtures and publish invariant requiring HI143 PASS. For THTR02, reason classification and deterministic recommendation report with tests proving no tuning-state, launch or cache side effects. For THTR03, compare empirical FDR/power using archived data with real correlation, not independent uniform simulation. For THTR04, real sustained E2E ranking versus tune effective_us with inversion effect/frequency/uncertainty and independent repeated ensembles. No production policy or hardware claim follows without the corresponding evidence.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Search cost is reduced without changing isolated cause or evidence quality; publication always requires full immutable corpus validation.

P0 recovery remains pending until its shortest-vector search preserves the isolated cause and final immutable full-corpus validation passes. Behavioral witnesses are exact-context and cannot blacklist unrelated contexts; publication requires HI143 PASS. Escalation remains recommendation-only with no autonomous tuning or cache side effects. Production multiplicity correction remains unchanged absent material empirical BH risk; any justified future correction is versioned with fixed hypothesis-family boundaries. Production ranking remains unchanged absent repeatable material inversion across independent sustained E2E ensembles. These adjacent studies do not weaken the original P0 evidence/publication gate.

## Notes

Supersedes: HTR07
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-tune-recovery-htr07

THTR01-THTR04 were consolidated into active THTR05 on 2026-11-20 after review RV4252 as separate, bounded safety/evidence tracks. THTR05 remains P0 and pending. No automated retuning, production policy change, full-corpus validation or hardware result is claimed by this plan update.

## Change Log

- 2026-09-09T10:51:38.359111+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:08:03.783654+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.030411+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.648480+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:36:54.896836+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_033709_repaired-five-recovery-success_3695
- 2026-09-10T03:37:09.569044+00:00 (updated-by): Updated: section:ledger-events

## Reviews

- RV4252
- 2026-11-20T03:23:53.839986+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria, section:notes
