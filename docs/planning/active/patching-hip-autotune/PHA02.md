---
id: PHA02
order: 0
plan: patching-hip-autotune
state: in_progress
created-at: '2026-09-09T10:48:38.219639+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# RCCL heterogeneous per-rank kernel/code-object dispatch investigation

## Description

Current audit is useful but not closure evidence: it traced RCCL revision 57e58688f44c77076ad536ef1f6b68741fc6e694 and the multi-arch bundle, while dev-GPT identified the historical failing runtime as RCCL 2.27.7-1/source 9fb6fbe7bfb4df87c4c9a09b5bb5239670f04ffe. A fresh fetch of that exact object from the available Brutus remote failed (`not our ref`). PHA02 remains open pending matching-source acquisition or a fresh blocked/no-safe-change disposition.

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

Dev-GPT review requires exact failing-runtime source/build provenance, exact RCCL kernel/code-object registration→architecture-selection→launch function, and if instrumentable a mixed-architecture pre-launch trace. Current evidence records the non-matching source audit, code-object presence, and fresh fetch failure; it does not claim closure. Same-arch control and existing 1225 guard evidence remain historical and unchanged.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Do not close on the non-matching 57e5868 audit alone. Obtain/review source matching 9fb6fbe7bfb4df87c4c9a09b5bb5239670f04ffe and produce the exact dispatch trace, or after a fresh acquisition attempt document a concrete external blocker/no-safe-change conclusion while retaining 1225 unchanged.

## Notes

Supersedes: HI137
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-hip-autotune-hi137

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
