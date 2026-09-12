---
id: PRBC02
order: 2
plan: patching-reusable-build-campaign
state: pending
created-at: '2026-09-09T10:59:33.148983+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Extend patch_catalog.py metadata: PLAN_IDS, BACKENDS, SUBSYSTEMS, REQUIRES, CONFLICTS, HARDWARE

## Description

Complete residual catalog metadata/provenance validation while keeping executable REQUIRES/CONFLICTS authoritative in patch modules.

## Steps

Validate merged CatalogSnapshot has exactly one metadata source per registry module (packaged patch.toml or legacy catalog entry, never neither/both); validate plural metadata/plan IDs, backend/options against PatchContext, external-source links and retirement links; expose relationships through snapshot/CLI without duplicating authority.

## Detailed Solution & Technical Design

Catalog metadata is descriptive and merged from packaged descriptors plus legacy catalog. patchset module constants remain executable dependency/conflict authority. Add canonical plan/backend/subsystem/hardware metadata checks and fail closed on dual authority or stale provenance.

## Code Samples & Guidance



## Files

patch_catalog.py; patchset.py; external_sources.py; patches/catalog.toml; catalog/governance tests.

## Validation

1:1 merged metadata source tests, state/agreement and plural-field checks, backend/options applicability, external-source and retirement-link checks, CLI exposure.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Every module has exactly one metadata source, merged metadata is consistent and provenance/retirement links validate; no duplicate REQUIRES/CONFLICTS authority.

## Notes

Supersedes: RE40
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-reusable-build-campaign-re40

GPT roadmap provenance: request req_f7a013040828433c, same session ses_76206cac3e6b4be0, exact pushed bb20f104. Treat this as a residual metadata/provenance audit after PRBC01, not a reason to duplicate existing CatalogSnapshot or registry logic. Current packaged catalog metadata already implements much of this scope; close only the exact residual checks after inventory and tests prove them satisfied, otherwise retain only those residuals.

## Change Log

- 2026-09-09T10:59:33.148983+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:16:38.119374+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events



- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.507395+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.379203+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:21:46.904522+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032208_repaired-the-reusable-build-ca_4665
- 2026-09-10T03:22:08.386926+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T18:53:09.647626+00:00 (updated-by): Updated: order=2, section:notes
- chg_20260912_185547_recorded-the-gpt-guided-non-vu_6004
- 2026-09-12T18:55:48.042620+00:00 (updated-by): Updated: section:ledger-events
- chg_20260912_191132_corrected-the-gpt-guided-imple_8803
- 2026-09-12T19:11:32.072877+00:00 (updated-by): Updated: section:ledger-events
