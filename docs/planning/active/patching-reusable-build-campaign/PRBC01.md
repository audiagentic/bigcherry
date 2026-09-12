---
id: PRBC01
order: 0
plan: patching-reusable-build-campaign
state: pending
created-at: '2026-09-09T10:59:28.152268+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Introduce CatalogSnapshot: one immutable patch-catalog read per command/campaign

## Description

Thread one immutable CatalogSnapshot through campaign resolution, materialization, build planning, and reporting instead of rescanning patches/catalog independently.

## Steps

Fix snapshot digest/immutability prerequisite if needed; construct snapshot at command/campaign boundary; pass through resolve_lane/resolve_patch_set, source planning/materialization, build planning and reports; preserve custom patches_root; reject root/catalog drift; remove lower-level rescans after one-read and mutation tests pass.

## Detailed Solution & Technical Design

CatalogSnapshot must include immutable module bytes/content hashes and canonical metadata digest. Keep exact resolution and patch_set identity unchanged. Optional compatibility construction is allowed only at outer boundary; campaign execution must use one snapshot.

## Code Samples & Guidance



## Files

tools/bigcherry/patch/catalog.py; tools/bigcherry/patch/patchset.py; tools/bigcherry/campaign/resolution.py; tools/bigcherry/campaign/source.py; tools/bigcherry/campaign/build.py; tools/bigcherry/campaign/lane.py; tools/bigcherry/campaign/planner.py; tools/bigcherry/campaign/plan.py; snapshot and campaign tests.

## Validation

Prove one filesystem read per command, custom-root correctness, mutation cannot alter resolved plan, and unchanged IDs/order/hashes/patch_set_id.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Campaign planning and execution resolve from one immutable snapshot with drift rejection, no lower-level rescans, and unchanged exact identity semantics.

## Notes

Supersedes: RE39
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-reusable-build-campaign-re39

GPT roadmap provenance: request req_f7a013040828433c, same session ses_76206cac3e6b4be0, exact pushed bb20f104. This is the first reusable-campaign infrastructure item after BRC01/PA34 and must precede PA33. Thread the existing CatalogSnapshot through PA34 source+experiment/focal-overlay selection, materialization, BuildPlan and reporting; do not redesign it or reimplement resolve_exact()/expand_composition(). The prior file list's root-level patch_catalog.py/campaign_resolution.py names were stale; current owners are under patch/ and campaign/.

## Change Log

- 2026-09-09T10:59:28.152268+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:16:34.062604+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events



- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.502028+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.372600+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:21:39.750213+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032208_repaired-the-reusable-build-ca_4665
- 2026-09-10T03:22:08.366620+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T18:52:56.632221+00:00 (updated-by): Updated: section:files, section:notes
- chg_20260912_185547_recorded-the-gpt-guided-non-vu_6004
- 2026-09-12T18:55:48.038185+00:00 (updated-by): Updated: section:ledger-events
- chg_20260912_191132_corrected-the-gpt-guided-imple_8803
- 2026-09-12T19:11:32.068321+00:00 (updated-by): Updated: section:ledger-events
