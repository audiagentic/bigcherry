---
id: RHA02
order: 0
plan: run-hip-autotune
state: in_progress
created-at: '2026-09-09T10:48:42.206554+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Multi-GPU validation and end-to-end graph capture verification

## Description

Offline validation and the headline dual-XTX graph lifecycle proof are complete. The remaining qualification scope is explicitly bounded to any still-required mixed-topology/long-context rows and durable raw artifact retention; this item must not be used to claim RHA04 production parity.

## Steps

1. Preserve the strict offline validator and graph-lifecycle parser.
2. Retain the real dual-XTX graph-on lifecycle evidence from the HI14 successor run: capture_begin, capture_end, instantiate and replay, with clean teardown and real graph reuse.
3. Determine whether mixed gfx1100/gfx1201 and long-context rows remain release-required; if yes, run them through the maintained server harness with identity, topology and lifecycle evidence.
4. Retain raw logs and machine-readable validator output under docs/evidence and record limitations.
5. Keep production parity/performance admission under RHA04.

## Detailed Solution & Technical Design

Capability owner: run. The strict validator remains the acceptance authority for graph lifecycle and per-device evidence. Existing real dual-XTX evidence proves the four lifecycle stages on the intended tensor-split topology, but the archived note lacks a durable raw log bundle and does not cover mixed topology or long-context qualification. Any follow-up must use the maintained server-bench runner and preserve graph lifecycle, dispatch, identity and teardown evidence.

## Code Samples & Guidance



## Files

tools/bigcherry/multi_gpu_validate.py
tools/bigcherry/graph_lifecycle_evidence.py
tools/tests/test_multi_gpu_validate.py
tools/tests/test_graph_lifecycle_evidence.py
docs/evidence/2026-08-23-hi14-graph-lifecycle/ (follow-up artifact if re-run)
docs/planning/completed/hip-autotune/HI14.md
docs/planning/completed/hip-autotune/HI90.md

## Validation

Offline validator/parser and unit tests are complete. Real Brutus dual-XTX graph-on evidence is recorded in HI14/HI90 notes and ledger events: production-shaped tensor split, graphs reused, and capture_begin/capture_end/instantiate/replay markers observed. Completion is not claimed for mixed gfx1100/gfx1201 or long-context rows, and RHA04 performance admission remains independent.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: HI14
Migration: capability-rebaseline-v3-2026-09
Successor key: run-hip-autotune-hi14

Successor of HI14; migration capability-rebaseline-v3-2026-09. Evaluated against current evidence: the core graph lifecycle gap is closed, but durable raw artifact retention and any release-required mixed/long-context rows remain. Do not duplicate RHA04's native/replay parity matrix or attestation gate.

## Change Log

- 2026-09-09T10:48:42.206554+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:04:46.034026+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.828955+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T18:18:09.476379+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:notes
- 2026-09-09T18:18:17.869715+00:00 (state-transition): State: pending → in_progress
- chg_20260909_181829_rha02-is-no-longer-stale-its_8078
- 2026-09-09T18:18:29.048613+00:00 (updated-by): Updated: section:ledger-events
