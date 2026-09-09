---
id: PHA02
order: 0
plan: patching-hip-autotune
state: completed
created-at: '2026-09-09T10:48:38.219639+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# RCCL heterogeneous per-rank kernel/code-object dispatch investigation

## Description

The exact historical RCCL source/build provenance required for a dispatch-level repair is externally unavailable. The Brutus checkout traced RCCL revision 57e58688f44c77076ad536ef1f6b68741fc6e694; exact 9fb6fbe7bfb4df87c4c9a09b5bb5239670f04ffe acquisition returned `not our ref`, additive unshallow completed without the object, and an independent canonical GitHub lookup returned no matching ref. The nearest reachable 593de54e52679b51428571c13271e2ea9f91b1b1 source is a documented non-substitute. Close as external-blocker/no-safe-change; retain patch 1225 unchanged and fail-closed.

## Steps

1. Freeze the exact RCCL source/build/runtime and code-object provenance.\n2. Trace per-rank architecture/tuning state through RCCL kernel selection and launch.\n3. Correlate the source trace with the existing crash-isolated homogeneous/mixed controls without relocating historical evidence.\n4. Decide whether a safe source fix is justified; if not, record a no-fix disposition and preserve the production guard.\n5. Any future RCCL repair must be isolated from BigCherry and pass the runbook correctness/crash/topology gates.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

docs/evidence/pha02-rccl-source-audit-20260909/README.md; docs/reference/testing/RCCL_HETEROGENEOUS_RUNBOOK.md; tools/bigcherry/profiling/rccl_qualify_campaign.py

## Validation

Completed the bounded source audit and repeated acquisition attempts: exact fetch failure, successful additive unshallow with object absence, nearest-public-source negative control, and independent canonical-public exact-SHA lookup with no matching ref. Exact registration→architecture-selection→launch trace cannot be established for the missing historical runtime. Historical 57e5868 and 593de54 evidence remains provenance only; no source repair or positive RCCL dispatch conclusion is claimed.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

- Exact 9fb6fbe7bfb4df87c4c9a09b5bb5239670f04ffe source/build provenance is either obtained or its external unavailability is documented with fetch, unshallow, and canonical-public lookup evidence.\n- The exact dispatch trace is produced only if matching source becomes available; otherwise the item closes as external-blocker/no-safe-change.\n- 57e5868 and 593de54 remain explicitly non-substitute controls.\n- No RCCL source repair is justified, and patch 1225 remains unchanged and fail-closed.\n- Reopen only on exact source or authoritative source-to-historical-binary provenance.

## Notes

Supersedes: HI137
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-hip-autotune-hi137

Supersedes: HI137
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-hip-autotune-hi137
Latest acquisition evidence: exact fetch returned `not our ref`; unshallow completed; exact object remained absent; nearest reachable 593de54e source was inspected as a non-substitute control.

Supersedes: HI137\nMigration: capability-rebaseline-v3-2026-09\nSuccessor key: patching-hip-autotune-hi137\n\nFinal disposition: external-blocker / no-safe-change. Exact historical source object is unavailable from the canonical public remote; runtime probing cannot satisfy this source-dispatch boundary. Reopen only if exact source or authoritative source-to-binary provenance becomes available.

## Change Log

- 2026-09-09T10:48:38.219639+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:04:41.718755+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.823035+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T13:33:37.333128+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:acceptance_criteria
- chg_20260909_133351_pha02-now-has-a-reproducible-r_8961
- 2026-09-09T13:33:51.713052+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T13:39:02.797891+00:00 (state-transition): State: pending → in_progress
- 2026-09-09T13:45:37.624883+00:00 (updated-by): Updated: section:description, section:validation, section:acceptance_criteria
- chg_20260909_134551_pha02-evidence-is-now-accurate_7163
- 2026-09-09T13:45:51.372943+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T13:55:42.341337+00:00 (updated-by): Updated: section:description, section:validation, section:acceptance_criteria, section:notes
- chg_20260909_135552_strengthened-pha02-provenance_7624
- 2026-09-09T13:55:52.202983+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T14:28:26.499070+00:00 (updated-by): Updated: section:description, section:validation, section:acceptance_criteria, section:notes
- 2026-09-09T14:29:03.776178+00:00 (state-transition): State: in_progress → completed
- chg_20260909_142913_closed-the-rccl-source-dispatc_5240
- 2026-09-09T14:29:13.427571+00:00 (updated-by): Updated: section:ledger-events
