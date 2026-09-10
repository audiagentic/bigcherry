---
id: PPS01
order: 0
plan: patching-patch-system
state: pending
created-at: '2026-09-09T10:53:22.583220+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Local CI + docs + pilot migrations + acceptance (RS12–RS18)

## Description

Complete PA04's remaining local-CI, overlay, isolated campaign, cross-machine, pin-staleness, regression, documentation, and Definition-of-Done acceptance gates. RS12-RS18 software/package implementation and PA18/RV85 flat-patch disposition remain complete; do not resurrect the superseded flat production patch.

## Steps

1. Resolve ownership/disposition for the two TR14.LAB_UNCLASSIFIED tools and the eight overlay.vendor_sync differences without overwriting shared vendor files. 2. Rerun check --quick, check --default, and check --full; add deterministic overlay.vendor_sync tests if absent. 3. Audit/reuse valid §72/§73/§74/§79/§80 evidence. 4. Run isolated §76 PRBE11/RD13 and §77 RD08 subject/control hardware campaigns with declared correctness, smoke, trace-marker, workload, VDR, benchmark, and experiment-contract artifacts. 5. Complete applicable §79 second-machine/architecture and §80 pin-bump staleness evidence. 6. Run §82 regression, §83 documentation/link audit, and §88 Definition-of-Done at the same accepted revision. 7. Assemble immutable acceptance matrix, record exact revision/environment/command/exit/artifact/hash/disposition, and link ledger events to PPS01.

## Detailed Solution & Technical Design

Use PATCH_REFACTOR_RUNBOOK §§39-51, 56-58, 65-88 as reconciled by PA18/RV85. Keep package-only policy: historical flat/simple §75 is superseded/N/A and packaged 1002 is not an automatic replacement. Hardware validation remains explicit through patch-validate/campaign tooling; bigcherry check must not compile ROCm or launch campaigns. Shared overlay synchronization follows the owning workflow and is fail-closed.

## Code Samples & Guidance

Required gates: check --quick, check --default, check --full; isolated PRBE11/RD13 and RD08 campaign manifests; §79 cross-machine record; §80 pin-staleness record; §82 regression; §83 links/docs; §88 DoD acceptance matrix. Every record must identify MCP-proven revision, environment, command, exit code, artifact/hash, and PASS/FAIL/BLOCKED.

## Files

tools/bigcherry/check.py; tools/bigcherry/__main__.py; tools/bigcherry/paths.py; tools/tests/test_check.py; patches/_template/**; patches/1205_rd12_paired_mmvq_dual_output/**; patches/1206_rd13_mul_mat_add_view_fusion/**; patches/1204_rd08_q6k_mmvq_vdr2/**; docs/reference/patches/PATCH_SYSTEM.md; PATCH_AUTHORING.md; PATCH_VALIDATION.md; PATCH_REFACTOR_RUNBOOK.md; campaign manifests/evidence and immutable acceptance matrix.

## Validation

Software baseline: retained RS12-RS18 tests and package validation (historically 2042 passed, 1 skipped, 66 subtests; rerun at accepted revision). Overlay: vendor sync clean and check --default/full exit 0. Hardware: isolated subject/control evidence for RD12/RD13/RD08, with correctness/smoke/trace/VDR/benchmark contracts. Cross-machine and pin-staleness gates: §79/§80 as applicable. Final §82/§83/§88 all green at one revision.

## Effort & Risk

L; software is largely complete, but shared-overlay ownership and hardware/cross-machine provenance are high-risk. Never convert unavailable or disputed evidence to PASS and never stage unrelated vendor files.

## Standards

PATCH_REFACTOR_RUNBOOK.md §§39-51,56-58,65-88; package-only PA18/RV85 policy; fail-closed local CI and campaign evidence contracts; ag-ledger provenance.

## Acceptance Criteria

Complete only when quick/default/full checks are green, overlay ownership is resolved, isolated RD12/RD13/RD08 evidence and applicable cross-machine/pin-staleness gates exist, §82/§83/§88 are green, and the immutable acceptance matrix has exact provenance. Unavailable hardware or unsynchronized overlays remain BLOCKED, never PASS.

## Notes

Supersedes: PA04
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-patch-system-pa04

Supersedes PA04. Current known blockers from predecessor: TR14.LAB_UNCLASSIFIED for two lab tools, overlay.vendor_sync differences, and missing isolated §76/§77 plus applicable §79/§80 evidence. Preserve the historical flat-patch N/A disposition and keep PA05/RD19 separate.

## Change Log

- 2026-09-09T10:53:22.583220+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:09:54.316400+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.121336+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T12:09:33.788649+00:00 (updated-by): Updated: section:title
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.786694+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:14:41.509859+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:acceptance_criteria, section:notes
- chg_20260910_031453_repaired-pps01-so-the-active-p_1890
- 2026-09-10T03:14:53.220244+00:00 (updated-by): Updated: section:ledger-events
