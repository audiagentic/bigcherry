---
id: PKC04
order: 0
plan: patching-kernel-coverage
state: pending
created-at: '2026-09-09T10:51:56.429711+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: P3
---

# Decide whether Vulkan scope needs a reusable lifecycle extension

## Description

Make a conditional ownership decision about any remaining Vulkan attention or transport/collective lifecycle semantics after current provider coverage; do not implement speculative kernels or a parallel lifecycle system.

## Steps

1. Defer until TRVP12-15 and EC16/EC19 establish current Vulkan ownership and capability coverage. 2. Inventory remaining attention, transport, and collective semantics and map each to an existing owner/contract. 3. Close PKC04 when existing ownership/contracts are sufficient. 4. If a genuine lifecycle gap remains, open the smallest follow-up under the correct existing owner with explicit evidence and dependency. 5. Keep speculative transport candidates out of the critical path.

## Detailed Solution & Technical Design

This is a decision and ownership gate only. Vulkan attention is separate from Vulkan MUL_MAT only when semantics/lifecycle differ materially. K-VKC-03/04 remain conditional on transport correctness evidence. Do not create backend implementation, capability, attention, transport, collective, optimization, benchmark, or parallel candidate work here; route any proven gap to the existing owner.

## Code Samples & Guidance



## Files

Vulkan provider/runtime/recipe ownership matrix; TRVP12-15 outputs; EC16/EC19 references; Vulkan tests and evidence; follow-up plan item only if a lifecycle gap is proven.

## Validation

Produce a scope/ownership decision showing coverage, capability/driver/fallback requirements, and disposition for every remaining attention/transport semantic. Verify no speculative candidate or duplicate lifecycle system is created before transport correctness evidence. Close or create the smallest named follow-up with owner and rationale.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

A conditional ownership decision is recorded after current Vulkan coverage is reviewed. Existing contracts close the item, or any proven gap is split to an existing owner with explicit evidence. No Vulkan kernel, attention, transport, collective, optimization, or benchmark implementation is performed here; speculative transport stays off the critical path.

## Notes

Supersedes: KC04
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-kernel-coverage-kc04

Supersedes: KC04
Inherited constraints: RV108, RV113, RV124 — lifecycle scope gate only; keep speculative transport candidates out of the critical path.
Migration: capability-rebaseline-v3-2026-09

Supersedes KC04. Preserve RV108/RV113/RV124 and keep TRVP12 owner of Vulkan MUL_MAT/pipeline/shader/capability/split-reduction. Do not create a parallel Vulkan lifecycle or candidate system.

## Change Log

- 2026-09-09T10:51:56.429711+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:08:24.078150+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.048940+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.683688+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:52:50.713702+00:00 (updated-by): Updated: section:description, section:steps, section:acceptance_criteria, section:notes
- chg_20260910_005948_legacy-planning-folders-now-co_1240
- 2026-09-10T00:59:49.031122+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:47:46.421161+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_034835_repaired-the-final-six-live-su_3009
- 2026-09-10T03:48:35.200144+00:00 (updated-by): Updated: section:ledger-events
