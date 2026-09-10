---
id: PRBE72
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:58:35.919608+00:00'
breadth: ''
skill: ''
created-by: capability-rebaseline-v3
priority: null
---

# Hardware-eligibility needs device traits beyond architecture (integrated/UMA/P2P/driver/GPU-count)

## Description

Extend hardware eligibility keys with integrated/UMA/P2P/GPU-count/driver traits separate from architecture, and audit hardware-scoped contracts.

## Steps

Add device-trait fields integrated, uma, peer_access, gpu_count and driver to Experiment Contracts/autotune hardware key; audit RD22/PRBE17 and similar items for architecture-versus-trait eligibility; update contract scopes and tests so traits drive selection where they are the real condition.

## Detailed Solution & Technical Design

Architecture strings alone misclassify integrated/UMA correctness work. Preserve architecture as one dimension but include topology/device traits and driver identity in eligibility and candidate keys, with backward-compatible defaults and fail-closed unknown traits.

## Code Samples & Guidance



## Files

Experiment Contract schema and hardware-key resolution; device discovery; RD22/PRBE17 contract updates; eligibility serialization and tests.

## Validation

Unit/integration tests for trait combinations, serialization compatibility, integrated versus discrete/P2P and multi-GPU selection.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Hardware-scoped items select by proven device traits rather than architecture alone, with unknown traits falling back safely and existing contracts remaining compatible.

## Notes

Supersedes: RD92
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd92

## Change Log

- 2026-09-09T10:58:35.919608+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:46.423644+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.454152+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.291596+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:20:20.626577+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032047_repaired-five-more-active-succ_6361
- 2026-09-10T03:20:47.947706+00:00 (updated-by): Updated: section:ledger-events
