---
id: PRBC03
order: 0
plan: patching-reusable-build-campaign
state: pending
created-at: '2026-09-09T10:59:38.602888+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Split resolver into resolve_exact() (fail-closed, explicit) and expand_composition() (dependency-closure expansion)

## Description

Adopt explicit composition expansion only after snapshot/metadata prerequisites, with equivalence proof against hand-written recipes.

## Steps

Add snapshot-aware expansion/report API; prove closure across dependency chains including scrambled-order fixture; add opt-in composition_mode=expand for one non-production experiment; compare requested/expanded IDs/order/hashes/materialized source/patch_set_id with current manual recipe; simplify production recipes only after equivalence.

## Detailed Solution & Technical Design

Keep resolve_exact() fail-closed and expand_composition() as explicit dependency-closure expansion. Use module REQUIRES/CONFLICTS as authority. Production HIP/replay lanes stay exact until equivalence evidence is accepted; never silently change recipes.

## Code Samples & Guidance



## Files

patchset.py; campaign_resolution.py; campaign_source.py; config/recipes.toml; patch-resolution/campaign tests; equivalence artifacts.

## Validation

Snapshot-aware closure and conflict tests; scrambled dependency order; exact/manual equivalence of IDs, order, hashes, source and patch_set_id; explicit opt-in only.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Production recipes remain unchanged until all equivalence fields match under one snapshot; expansion is explicit, fail-closed, and never silently alters patch identity.

## Notes

Supersedes: RE42
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-reusable-build-campaign-re42

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
