---
id: PRBE42
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:23.562520+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# AMD-GDN-001: Chunked fused GatedDeltaNet recurrence

## Description

Implement and qualify the AMD-GDN-001 chunked fused GatedDeltaNet recurrence on exact supported gfx1151 shapes, with safe fallback on gfx1100/gfx1201 and unsupported workloads.

## Steps

1. Recheck AMD PR #54 and the pinned GATED_DELTA_NET APIs. 2. Implement chunked recurrence with state retained in registers/LDS for the narrow scalar-gate, K==1, S_v==128 predicate. 3. Gate explicitly by hardware, shape, gate mode, token/ubatch, and workspace limits; keep the old kernel for all other cases. 4. Validate interaction with PRBE41 channels-major SSM_CONV and downstream PRBE43/PRBE44/RD53 work. 5. Run direct recurrent-state/output parity over long sequences and real Qwen hybrid model checks. 6. Benchmark GDN op and end-to-end prefill, reporting VGPR/LDS/workspace and decode n=1 neutrality.

## Detailed Solution & Technical Design

Replace token-by-token GDN overhead with a chunked HIP recurrence that keeps state in registers/LDS, but only under the exact supported shape predicate: scalar gate, K=1, S_v=128, target gfx1151/RDNA3.5, and eligible prefill ubatches. The eligibility gate must fail closed on gfx1100/gfx1201, decode n=1, other gate modes, S_v/K values, and unsupported model graphs. Preserve the existing dispatch path and state semantics outside the gate.

## Code Samples & Guidance

Trigger: Qwen hybrid/GDN prefill on gfx1151 with scalar gate, K==1, S_v==128 and ubatch 512..4096. Controls: other S_v/K/gate modes, decode n=1, standard attention models, gfx1100/gfx1201. Boundary: chunk size, context length, workspace, VGPR/LDS pressure.

## Files

HIP GATED_DELTA_NET chunked kernel and dispatch eligibility predicate; state/workspace tests; architecture fallback tests; Qwen hybrid replay manifests and evidence; composition tests with PRBE41 and downstream GDN successors.

## Validation

Correctness: recurrent state and output parity against old kernel over long sequences plus PPL/deterministic model checks. Fallback: real gfx1100/gfx1201 and unsupported-shape runs must use old kernel normally. Performance: report GDN op time, PP E2E, chunk size, VGPR/LDS/workspace, variance, and decode n=1. Promote only with exact predicate and repeatable benefit; source headline 1.89x/+20% is not acceptance by itself.

## Effort & Risk

L; invasive kernel and state changes are model-specific. Hardware performance remains blocked without gfx1151, so separate compile/fallback evidence from performance qualification.

## Standards

Fail-closed hardware/shape gating, preserve recurrent state semantics, no unsupported fallback regression, and retain campaign evidence/provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: RD50
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd50

Supersedes RD50. Root prerequisite for PRBE43 and the later GDN-003/GDN-004 successors. Existing compile-safety and fallback evidence does not substitute for gfx1151 performance qualification.

## Change Log

- 2026-09-09T10:56:23.562520+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:38.267576+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.319945+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.085474+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:04:32.318831+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- chg_20260910_030451_carried-forward-the-detailed-s_2071
- 2026-09-10T03:04:51.342920+00:00 (updated-by): Updated: section:ledger-events
