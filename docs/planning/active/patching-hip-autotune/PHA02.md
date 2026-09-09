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

Completed a source/runtime audit against the pinned Brutus RCCL checkout. The multi-architecture bundle contains gfx1030/gfx1100/gfx1201 code objects; RCCL records per-rank architecture/tuning state but launches a generic kernel function without a BigCherry-visible per-rank code-object selector. Existing crash-isolated qualification evidence remains scoped to its tested topologies. No safe RCCL source fix was identified; the production guard and META heterogeneous path remain unchanged.

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

Brutus source checkout revision `57e58688f44c77076ad536ef1f6b68741fc6e694` was inspected. Existing source/build artifacts prove multi-arch code objects (`gfx1030`, `gfx1100`, `gfx1201`) and exact RCCL linkage. Static trace covers `src/init.cc:1355-1364,2135-2221`, `src/graph/tuning.cc:638-783,1017-1032`, and `src/enqueue.cc:896-898,1808-1832`. Existing `rq08-01/cases.jsonl` remains scoped evidence and is summarized in the new evidence record. No BigCherry production patch was changed.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Source/runtime provenance and dispatch trace are recorded; the observed limitation is bounded to the tested RCCL/runtime/topology; no unsupported BigCherry fix is introduced; the fail-closed guard and META fallback remain the production boundary. A future source repair, if pursued, must be independently qualified before integration.

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
