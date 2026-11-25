---
id: RCS22
order: 0
plan: build-radiance-components
state: pending
created-at: '2026-11-25T00:40:56.481749+00:00'
breadth: ''
skill: advanced
created-by: lead
work: L
priority: P2
---

# Loadable kernel-family plugins composed from a manifest (decided at the RCS15 gate)

## Description

Replace one plugin per architecture by one small plugin per kernel family per architecture, composed at load from a manifest, so a family builds, gates, versions and rolls back on its own.

## Steps

1. Decision gate: the RCS15 baseline says whether a native RDNA3 provider replaces the generated gfx11 tree; build the family plugins on whichever survives. 2. Plugin contract: each family exports its row table, a family version and an ABI hash of the shared headers; shared decoders stay header-only so there are no cross-plugin symbols. 3. Loader and registry: read a family manifest per architecture, merge row tables, error on a row in two families, explicit refusal (never a silent reference-kernel fallback) when a required family is missing; engine loader change as a managed patch. 4. Release and install: bundle the family plugins and manifest; verify checks each family's ABI hash and code objects. 5. Plan-cache identity includes loaded family versions. 6. Build graph: one link and one family test target per family; full all-family build only for release.

## Detailed Solution & Technical Design

Design: docs/design/radiance-kernel-composition.md section 6.

## Code Samples & Guidance



## Files

engines/radiance/components (libr3, registry, loader patch), release packaging, bc-release verify

## Validation

Loader and registry tests; family selftests; full qualification of a release assembled from family plugins against the single-plugin release, bytes equal; refusal tests for a missing family and a duplicate row.

## Effort & Risk

Large. Risks: engine loader change, helper contracts (remote MoE, n-gram) must not change meaning, load time with several plugins.

## Standards



## Acceptance Criteria

A kernel change in one family rebuilds, gates and ships that family only; releases name their family versions; no provider or reference fallback.

## Notes

Gated on RCS15. Not started until the baseline decides which tree survives.

## Change Log

- 2026-11-25T00:40:56.481749+00:00 (created-by): Created by lead
- 2026-11-25T00:41:36.975982+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
