---
id: PRBC03
order: 3
plan: patching-reusable-build-campaign
state: pending
created-at: '2026-09-09T10:59:38.602888+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Prove snapshot-aware composition expansion and exact/manual equivalence

## Description

Re-scope the former resolver-split item around the existing resolve_exact() and expand_composition() authorities. Those functions already exist; this item must not reimplement them or introduce a second resolver. After PRBC01 provides one immutable CatalogSnapshot, prove that explicit dependency-closure expansion is snapshot-aware, fail-closed, and equivalent to the current hand-written recipe for an opt-in experiment.

## Steps

1. Use PRBC01's immutable CatalogSnapshot as the sole catalog input and inspect the existing resolve_exact() and expand_composition() contracts.
2. Add only the snapshot-aware reporting/adapter seam needed to expose requested IDs, expanded IDs/order/hashes, materialized source identity, and patch_set_id; preserve existing resolver behavior.
3. Prove closure across dependency chains including a scrambled-order fixture, conflicts/rejections, empty input, missing dependency, and a focal closure whose dependencies are split between base and overlay for PA34.
4. Compare requested/expanded IDs, order, hashes, materialized source, selector identity and patch_set_id with the current manual recipe under one snapshot.
5. Keep composition expansion explicit and opt-in for one non-production experiment; do not simplify production recipes or delete old names until PA20 equivalence gates pass.

## Detailed Solution & Technical Design

Keep resolve_exact() fail-closed and expand_composition() as explicit dependency-closure expansion. Use module REQUIRES/CONFLICTS as authority. Production HIP/replay lanes stay exact until equivalence evidence is accepted; never silently change recipes.

## Code Samples & Guidance



## Files

tools/bigcherry/patch/patchset.py; tools/bigcherry/patch/catalog.py; tools/bigcherry/campaign/resolution.py; tools/bigcherry/campaign/source.py; tools/bigcherry/campaign/lane.py; tools/tests/patch/test_catalog_snapshot.py; tools/tests/patch/** composition/equivalence tests; equivalence artifacts.

## Validation

Snapshot-aware closure and conflict tests; scrambled dependency order; split base/overlay dependency fixture; exact/manual equivalence of IDs, order, hashes, source, selector identity and patch_set_id; explicit opt-in only; source-only compatibility; full offline suite.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Existing resolve_exact() and expand_composition() remain the single authorities; snapshot-aware expansion is explicit, fail-closed, and never silently alters patch identity. Production recipes remain unchanged until all equivalence fields match under one CatalogSnapshot, and PA34 can consume the reported exact composition without a duplicate resolver.

## Notes

Supersedes: RE42
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-reusable-build-campaign-re42

GPT roadmap provenance: request req_f7a013040828433c, same session ses_76206cac3e6b4be0, exact pushed bb20f104. PRBC03 is a residual proof/adapter item, not creation of resolver functions that already exist. It follows PRBC01 and must include PA34 experiment/focal-overlay selection in the one-snapshot invariant. Do not reimplement resolve_exact() or expand_composition(), and do not use it to introduce temporary recipe sources.

## Change Log

- 2026-09-09T10:59:38.602888+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:16:42.735596+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.511906+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.386068+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:21:55.651886+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032208_repaired-the-reusable-build-ca_4665
- 2026-09-10T03:22:08.399557+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T18:53:26.461187+00:00 (updated-by): Updated: section:title, order=3, section:description, section:steps, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260912_185547_recorded-the-gpt-guided-non-vu_6004
- 2026-09-12T18:55:48.046946+00:00 (updated-by): Updated: section:ledger-events
