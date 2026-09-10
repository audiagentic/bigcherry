---
id: PRBE43
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:27.912675+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-GDN-002: DPP row-shift reduction inside chunked GDN

## Description

Implement and qualify AMD-GDN-002 RDNA DPP row-shift reduction inside the PRBE42 chunked GDN kernel, with portable shuffle/DS fallback.

## Steps

1. Confirm PRBE42's chunked kernel and supported target shapes are available. 2. Replace the reduction primitive with RDNA DPP row-shift operations for gfx1100/gfx1151/gfx1201 where the launch geometry and lane mapping are proven. 3. Retain the portable shuffle/DS reduction for non-RDNA targets or any unsupported geometry. 4. Add tensor/state parity tests across chunk sizes and long sequences. 5. Measure DS utilization, kernel time, VGPR/LDS impact, and end-to-end PP against the PRBE42 baseline; select the architecture-specific path only when the measured kernel gain exceeds 2% without regression.

## Detailed Solution & Technical Design

Optimize the reduction inside the chunked GDN kernel using RDNA DPP lane row-shifts instead of shuffle/DS traffic. This is a micro-optimization layered on PRBE42, not a new recurrence. Select it through an architecture and geometry predicate, preserve the baseline reduction as a portable fallback, and keep state/output behavior identical.

## Code Samples & Guidance

Trigger: exact PRBE42 GDN shapes and launch geometry on gfx1100/gfx1151/gfx1201. Controls: non-RDNA hardware, unsupported geometry, and PRBE42 baseline. Boundary: DPP versus shuffle only; unchanged launch geometry.

## Files

PRBE42 HIP GDN reduction primitive and architecture selector; direct tensor/state parity tests; architecture fallback tests; kernel profiling and PP replay evidence for AMD-GDN-002.

## Validation

Correctness: output and recurrent-state parity against PRBE42 baseline over long sequences and chunk boundaries. Performance: report DS utilization, kernel microseconds, VGPR/LDS, and E2E PP with variance. Acceptance: enable the DPP selector only when the repeatable kernel gain is >2% and no correctness or end-to-end regression occurs; otherwise retain fallback.

## Effort & Risk

M; lane mapping and reduction ordering can cause subtle numerical/state errors. Keep the baseline path and require architecture-specific tests.

## Standards

Preserve PRBE42 eligibility/fallback, deterministic state semantics, portable non-RDNA behavior, and evidence provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: RD51
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd51

Supersedes RD51. Depends on PRBE42 (AMD-GDN-001); this optimization must not be treated as independently valid without the chunked recurrence.

## Change Log

- 2026-09-09T10:56:27.912675+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:41.977160+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.324437+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.090930+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:04:38.837182+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- chg_20260910_030451_carried-forward-the-detailed-s_2071
- 2026-09-10T03:04:51.354035+00:00 (updated-by): Updated: section:ledger-events
