---
id: PHA01
order: 0
plan: patching-hip-autotune
state: in_progress
created-at: '2026-09-09T10:48:04.945767+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Production patch admission gate for validated-state evidence (HI83 admission phase)

## Description

Implemented the direct `bigcherry apply --source` admission seam for validated-state evidence. Apply now fails closed before overlay or patch mutation when evidence is stale/missing, with an explicit development-only `--allow-stale-validation-evidence` escape hatch. Rebase-report/known-good apply remains independently fail-closed on its freshness proof.

## Steps

1. Resolve and TOCTOU-check the exact source composition and live upstream revision.
2. Run patch admission before audit-dependent mutation/overlay installation.
3. Emit admission failures and explicit escape-hatch warnings; never relax production build/campaign admission.
4. Validate rejection-before-mutation and parser wiring with focused release, admission, rebase, catalog, and CLI tests.
5. Record the implementation in the release ledger and push the branch.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

tools/bigcherry/__main__.py; tools/bigcherry/cli/main.py; tools/bigcherry/cli/patch.py; tools/tests/release/test_releases.py; tools/tests/core/test_cli_tooling_surface.py

## Validation

Focused validation passed: `python -m pytest -q tools/tests/release/test_releases.py tools/tests/patch/test_patch_admission.py tools/tests/patch/test_patch_catalog.py tools/tests/patch/test_patch_rebase.py` (92 passed, 5 subtests); CLI/campaign surface passed: `python -m pytest -q tools/tests/core/test_cli_tooling_surface.py tools/tests/campaign/test_campaign_build.py` (33 passed, 1 skipped). The new release test proves admission rejection occurs before `_copy_overlay`; parser coverage proves the escape hatch is explicit.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Direct source apply invokes patch admission after exact selection/live revision checks and before any overlay or patch mutation; stale/missing evidence fails closed by default; the escape hatch is explicit and warns; production/rebase-report gates remain fail-closed; focused tests pass.

## Notes

Supersedes: HI102
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-hip-autotune-hi102

## Change Log

- 2026-09-09T10:48:04.945767+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:04:00.674969+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.785104+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T13:27:22.858646+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:acceptance_criteria
- chg_20260909_132732_pha01-direct-patch-admission-i_5854
- 2026-09-09T13:27:32.588197+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T13:27:52.066363+00:00 (updated-by): Updated: section:steps
- 2026-09-09T13:28:53.082865+00:00 (state-transition): State: pending → in_progress
