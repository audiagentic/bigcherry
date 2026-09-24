---
id: PRBE72
order: 0
plan: patching-rdna-boost-experiments
state: deprecated
created-at: '2026-09-09T10:58:35.919608+00:00'
breadth: ''
skill: ''
created-by: capability-rebaseline-v3
priority: null
---

# Hardware-eligibility needs device traits beyond architecture (integrated/UMA/P2P/driver/GPU-count)

## Description

OBSOLETE (already implemented). tools/bigcherry/experiment/contract.py's `Scope` dataclass (line ~576) already carries `integrated: bool | None`, `uma: bool | None`, `peer_access: bool | None`, `gpu_count: GpuCountConstraint | None`, and `driver: DriverVersionConstraint | None` alongside `architectures`, and `evaluate_scope_eligibility()` already fails closed on missing hardware observations for trait-bearing scopes. The dataclass's own docstring cites RD22 by name as the motivating case, already expressible as `Scope(backend="hip", architectures=("gfx1151",), integrated=True, uma=True)`.

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

2026-09-24 relevance: OBSOLETE, premise already implemented. Verified via direct read of tools/bigcherry/experiment/contract.py lines ~560-600 (Scope dataclass with integrated/uma/peer_access/gpu_count/driver fields, evaluate_scope_eligibility fail-closed helper, RD22 named in the docstring as the motivating case). The remaining piece of the item's own ask -- 'audit RD22/PRBE17 and similar items for architecture-vs-trait eligibility' -- may still be open work; PRBE16 (gfx1151 defer item) does not yet reference Scope's trait fields explicitly, so if that audit is still wanted it should be filed as a fresh, narrowly-scoped item (e.g. 'apply Scope trait fields to PRBE16/RD22's own contract declaration') rather than kept alive under this item's now-satisfied schema-design premise. GPT design request: not submitted -- disposition made directly from source inspection (schema already exists, no design question remains); no GPT request id.

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
- 2026-09-24T02:35:00.841581+00:00 (updated-by): Updated: section:description, section:notes
- 2026-09-24T02:35:15.360640+00:00 (state-transition): State: pending → deprecated
