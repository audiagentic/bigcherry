---
id: PHA03
order: 0
plan: patching-hip-autotune
state: in_progress
created-at: '2026-09-09T10:48:53.551059+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# RCCL admission gate: implement level-1 comm-init topology admission

## Description

The current shared PCIe-AtomicOps predicate and stale-test refresh are correct prerequisites, but no admission-enabling allowlist is justified yet. Dev GPT review (req_a569c67f6b3e4312) confirms the real next slice is fail-closed topology/version evidence plumbing; patch 1225 must continue returning false for unknown/unqualified records until exact Brutus whitelist inputs exist.

## Steps

1. Keep the current shared PCIe-AtomicOps predicate and its direct plus 0840 secondary-communicator wiring; do not add another admission path.
2. Extend RcclTopology evidence beyond descriptive ID and architecture: canonical PCI BDF/device placement, PCIe nodes/edges/root-port relations, link/hop/NUMA properties, AtomicOps/transport capabilities, and explicit unknown state.
3. Persist exact ROCm runtime/driver and RCCL build identity with qualification artifacts; missing or unreadable fields make a record non-admissible.
4. Add no-GPU tests for canonical identity stability, placement/order differences, missing topology data, version capture, unknown/fail-closed behavior, and shared-predicate coverage.
5. Run Brutus qualification to freeze exact positive and negative topology/version rows, including managed direct and 0840 secondary communicator paths.
6. Only after that evidence exists, implement an exact conjunction allowlist; otherwise retain the current fail-closed false outcome and document the external evidence gap.

## Detailed Solution & Technical Design

Capability owner: patching.

Safe implementation slice: extend tools/bigcherry/profiling/rccl_schema.py with a portable RcclTopologyEvidence identity that is independent of HIP ordinals but includes exact placement and PCIe graph/capability facts, plus ROCm/driver provenance. Extend rccl_qualify.py and its result artifacts to carry the evidence identity. Keep the existing PCIe-AtomicOps admission seam unchanged.

Do not create rccl_probe.py merely because an old plan file names it; only add a separately justified probe responsibility. Do not encode a positive topology allowlist from descriptive topology IDs, architecture tuples, historical artifacts, or the current GPU-only AtomicOps probe. A positive row requires complete graph evidence, exact runtime/build windows, managed communicator-init and collective success, and negative controls.

## Code Samples & Guidance



## Files

tools/bigcherry/profiling/rccl_schema.py
tools/bigcherry/profiling/rccl_qualify.py
tools/bigcherry/profiling/rccl_qualify_campaign.py
tools/tests/profiling/test_rccl_qualify.py
tools/tests/profiling/test_rccl_qualify_campaign.py
tools/tests/patch/test_hi85_nccl_heterogeneous_arch_guard.py
patches/1225_hi85_nccl_heterogeneous_arch_guard/patch.py
docs/reference/testing/RCCL_HETEROGENEOUS_RUNBOOK.md
docs/evidence/pha03-rccl-topology-evidence-20260910/README.md

## Validation

Hardware-free evidence slice implemented and verified. Focused command:
$env:PYTHONPATH='tools'; uv run --no-sync python -m pytest -q tools/tests/patch/test_hi85_nccl_heterogeneous_arch_guard.py tools/tests/profiling/test_rccl_qualify.py tools/tests/profiling/test_rccl_qualify_campaign.py
Result: 58 passed, 1 skipped. Ruff check over the changed profiling and test files passes.

Brutus snapshot 2026-09-10 records ROCm/HIP 7.2.53211-97f5574fe2, RCCL 2.27.7, exact librccl SHA/build ID, GPU placement graph, and raw evidence hashes in docs/evidence/pha03-rccl-topology-evidence-20260910/README.md. This is provenance only: complete per-component AtomicOps/transport evidence and managed direct plus 0840 collective success are still missing, so no positive admission row is claimed.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

- Topology evidence identity is canonical, ordinal-independent, placement-sensitive, and fail-closed when any required field is missing or unknown.
- Qualification artifacts retain exact PCIe graph/capability and ROCm/driver/RCCL provenance.
- Existing direct and 0840 communicator paths continue to use the shared predicate; no silent post-init fallback is added.
- Positive admission remains disabled until exact Brutus whitelist inputs are captured; no invented allowlist is accepted.
- After evidence capture, only an exact topology plus AtomicOps plus version-window conjunction may be enabled and promoted through PHA04/PHA05.

## Notes

Supersedes: HI146
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-hip-autotune-hi146

Supersedes: HI146
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-hip-autotune-hi146

Dev GPT gate review req_a569c67f6b3e4312: no admission-enabling change until exact topology/version whitelist inputs exist. Current fail-closed predicate remains authoritative.

## Change Log

- 2026-09-09T10:48:53.551059+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:04:57.995242+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.842046+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T13:36:36.486669+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:acceptance_criteria
- chg_20260909_133646_updated-stale-rccl-admission-t_7143
- 2026-09-09T13:36:46.061920+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T13:39:18.720766+00:00 (state-transition): State: pending → in_progress
- 2026-09-09T14:12:16.449369+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260909_141259_added-the-safe-rccl-evidence-l_5757
- 2026-09-09T14:12:59.981478+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T14:14:24.980527+00:00 (updated-by): Updated: section:validation
- chg_20260909_141436_the-safe-rccl-evidence-plumbin_6883
- 2026-09-09T14:14:36.246107+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T14:17:44.578152+00:00 (updated-by): Updated: section:files, section:validation, section:notes
- chg_20260909_141754_added-current-brutus-topology_6647
- 2026-09-09T14:17:54.082286+00:00 (updated-by): Updated: section:ledger-events
