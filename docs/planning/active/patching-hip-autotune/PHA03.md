---
id: PHA03
order: 0
plan: patching-hip-autotune
state: pending
created-at: '2026-09-09T10:48:53.551059+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# RCCL admission gate: implement level-1 comm-init topology admission

## Description

Corrected the stale HI85/1225 mechanical tests to assert the current GP02 PCIe-AtomicOps admission predicate rather than the retired raw-architecture guard. The actual topology-qualified level-1 admission implementation and Brutus validation remain open.

## Steps

1. Keep the current shared PCIe-AtomicOps predicate as the baseline safety mechanism and preserve its source/patch tests.\n2. Design the topology qualification identity and level-1 admission policy from HI142/THA07 evidence; do not reintroduce raw-architecture or ordinal checks.\n3. Add fail-closed tests for qualified, unqualified, and unknown topology records at the real comm-init seam.\n4. Validate on Brutus with homogeneous, qualified mixed, and device-3/unknown controls before any promotion.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

tools/tests/patch/test_hi85_nccl_heterogeneous_arch_guard.py; patches/1225_hi85_nccl_heterogeneous_arch_guard/patch.py; docs/reference/testing/RCCL_HETEROGENEOUS_RUNBOOK.md

## Validation

Fixed stale test expectations and ran `python -m pytest -q tools/tests/patch/test_hi85_nccl_heterogeneous_arch_guard.py tools/tests/profiling/test_rccl_qualify.py tools/tests/profiling/test_rccl_qualify_campaign.py`: 55 passed, 1 skipped. This only validates the current capability predicate and qualification harness; it is not yet evidence for topology allowlisting or production promotion.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Level-1 comm-init admission must be capability/topology-identity based, fail closed for unknown/unqualified sets, pass current qualified sets without regression, and have real Brutus evidence. The current test refresh is a prerequisite, not closure.

## Notes

Supersedes: HI146
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-hip-autotune-hi146

## Change Log

- 2026-09-09T10:48:53.551059+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:04:57.995242+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.842046+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T13:36:36.486669+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:acceptance_criteria
- chg_20260909_133646_updated-stale-rccl-admission-t_7143
- 2026-09-09T13:36:46.061920+00:00 (updated-by): Updated: section:ledger-events
