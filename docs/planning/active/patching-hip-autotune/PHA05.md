---
id: PHA05
order: 0
plan: patching-hip-autotune
state: pending
created-at: '2026-09-09T10:49:03.289342+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: S
priority: null
---

# RCCL admission gate: promote to validated and wire into default build

## Description

Promotion remains intentionally pending. It may only consume completed PHA03 level-1 topology admission and PHA04 level-2 per-call candidate binding evidence; no default-build wiring is justified from current capability-only guard, diagnostic runs, nearest RCCL source, or historical qualification artifacts.

## Steps

1. Require PHA03 completed with a topology identity, fail-closed comm-init admission, qualified/unqualified/unknown tests, and real Brutus controls.
2. Require PHA04 completed with exact RCCL compatibility/build, reduction signature, topology identity, and algorithm/protocol/channel candidate binding, including unknown and performance-excluded outcomes.
3. Require independent correctness, crash-freedom, and production-shaped hardware evidence for every candidate admitted.
4. Wire only the reviewed validated state into the default build; preserve explicit opt-out and fail-closed behavior for stale, missing, or incompatible evidence.
5. Re-run patch admission, build provenance, campaign, and default-build regression gates; record the promotion decision and rollback/disposition.

## Detailed Solution & Technical Design

Capability owner: patching.

PHA05 is a lifecycle/promotion boundary, not a place to implement RCCL selection. PHA03 owns topology viability; PHA04 owns exact candidate binding. PHA05 consumes their immutable evidence and changes only the patch/build lifecycle state. Historical HI148/RQ08/RQ10 results remain provenance on predecessors and cannot be relabeled as current validation.

Promotion contract: exact RCCL compatibility revision + build identity, reduction operation signature, portable topology identity, candidate tuple, correctness verdict, crash-free repetitions, and production-shaped evidence must all match. Missing, unknown, stale, or incompatible records fail closed. Performance exclusion is retained as a distinct disposition and never silently promoted.

## Code Samples & Guidance



## Files

patches/1225_hi85_nccl_heterogeneous_arch_guard/patch.toml
patches/0830_split_reduce_telemetry/patch.toml
tools/bigcherry/patch/patch_admission.py
tools/tests/patch/
docs/reference/testing/RCCL_HETEROGENEOUS_RUNBOOK.md

## Validation

PHA03 and PHA04 must be completed first. Validate exact compatibility/build/topology/signature/candidate identity, correctness and crash-free controls, production-shaped hardware evidence, patch applicability and provenance, default-build composition, stale/missing/unknown fail-closed behavior, and rollback. Diagnostic-only or historical evidence cannot satisfy promotion.

## Effort & Risk

Promotion is high risk: an incomplete identity or an over-broad allowlist can reintroduce the RCCL hard-crash or silently change collective semantics. Keep the item pending until both prerequisite gates and real hardware evidence are complete.

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md
docs/reference/testing/RCCL_HETEROGENEOUS_RUNBOOK.md
BigCherry patch admission and release-state contracts

## Acceptance Criteria

- PHA03 level-1 topology admission is complete and validated on real homogeneous, qualified mixed, and unsafe/unknown controls.
- PHA04 level-2 candidate binding is complete and validated for exact compatibility, signature, topology, and candidate tuple.
- Correctness, crash-freedom, production-shaped performance, and provenance evidence are current and identity-matched.
- Default-build wiring is applied only after patch admission accepts the validated state; stale, missing, unknown, or performance-excluded entries fail closed.
- Promotion/rollback and all changed tests are recorded in the ledger and pushed.

## Notes

Supersedes: HI148
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-hip-autotune-hi148

## Change Log

- 2026-09-09T10:49:03.289342+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:05:07.364077+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.852748+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T14:01:18.377277+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:standards, section:acceptance_criteria
- chg_20260909_140201_made-rccl-default-build-promot_9276
- 2026-09-09T14:02:01.352183+00:00 (updated-by): Updated: section:ledger-events
