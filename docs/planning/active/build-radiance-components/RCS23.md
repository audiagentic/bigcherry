---
id: RCS23
order: 0
plan: build-radiance-components
state: pending
created-at: '2026-11-25T00:54:41.130343+00:00'
breadth: ''
skill: advanced
created-by: lead
work: L
priority: P1
---

# Owned kernel units: take source ownership of the heavily rewritten IQR family with a manifest and an upstream drift check

## Description

For the libr4d units we already rewrite heavily (the IQR GEMM unit: 440 lines of rewrites over 1,343 upstream lines; the prefill header 134 over 309; the vocabulary head 121 over 536; the embedding gather about 57 over 142; 19 of 48 patch packages touch libr4d) patches and rewrites layered on one text no longer pay off. Take source ownership of those units, split by kernel family and written against the traits seam, while upstream keeps everything else; manage upstream change with a manifest and a drift check. Not a fork of libr4d: about 5 of about 170 files, measured upstream churn on them about three commits a week (14 commits in 12 releases over 30 days; the row-table file, 29 commits, is never owned).

## Steps

1. Inventory: per libr4d unit, the share of lines we rewrite or patch, the patches touching it, and upstream commits per unit per release. 2. Manifest (owned units: file, upstream base commit and blob, family, last upstream commit merged) and the drift check: per owned unit an empty upstream diff advances the base, a non-empty one tries a three-way merge (clean: branch plus gates; conflict: plan item with both diffs); include closure reports a changed upstream header included by an owned unit; run it over the last twelve releases and check it reproduces the measured churn. 3. Take ownership of one family first (the fused tile and the embedding gather: small, recent): split into per-kernel files on the traits seam, retire their patches and rewrites, byte identical and resource equal to the shipped build. 4. The remaining IQR units. 5. Exit rule: when upstream adopts a change the drift check shows equality and the unit leaves the set, local patch archived unchanged. 6. Build takes owned units from our repository and the rest from the pinned source; the engine's per-architecture libraries (libr4d, libr3, libr2) keep their names and loader contract.

## Detailed Solution & Technical Design

Design: docs/design/radiance-kernel-composition.md section 11.

## Code Samples & Guidance



## Files

engines/radiance (owned unit sources, manifest, drift tool), engines/radiance/components/libr3, patches, tools/bigcherry pin triage, docs/design

## Validation

Drift check over the last twelve releases matches the measured churn; each owned family: resource records equal, byte identity at every affected shape, cold and warm replay no regression against the shipped build, gates 1 to 4 (RCS19) pass for both architectures; a pin bump with an upstream change to an owned file produces the merge branch or the plan item automatically.

## Effort & Risk

Large. Risks: merge burden (bounded by the measured churn, automated), silent drift (manifest base commit and per-bump report), upstream ABI change (include closure plus gates 2 and 3), a first decomposition that changes bytes (staged one family at a time, each byte identical).

## Standards



## Acceptance Criteria

The IQR family is owned source with a manifest, retired patches and rewrites, one source for gfx11 and gfx12, and an automated drift check at each pin bump; engine-facing library names and loader contract unchanged.

## Notes

Depends on RCS19 (gates, ownership records) and RCS21 (traits seam); sequenced in parallel with the RCS15 baseline, which decides whether a native gfx11 provider replaces the generated tree for the unowned units. Review by GPT only (code returned as text).

## Change Log

- 2026-11-25T00:54:41.130343+00:00 (created-by): Created by lead
- 2026-11-25T00:54:59.083112+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
