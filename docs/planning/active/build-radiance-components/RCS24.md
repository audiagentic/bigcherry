---
id: RCS24
order: 0
plan: build-radiance-components
state: pending
created-at: '2026-11-25T01:02:59.740271+00:00'
breadth: ''
skill: intermediate
created-by: lead
work: M
priority: P1
---

# Kernel and build cleanup pass: stale docs, tests and dead code found while redesigning composition

## Description

Redesigning kernel composition surfaced stale or contradictory material. Fix it as part of the work, docs first, so nothing contradicts the design and no dead or legacy code stays behind.

## Steps

Done in the docs-first pass (PR 1543): one glossary and programme map in the design doc; status banner on the native libr3d assessment (pin 1.17); libr3 README terms; stale unverified-unit notes removed from exclude.txt; the shipped chained-tile block no longer labelled a scratch experiment; planned drift check noted in PIN_BUMP. Open: 1. Build GC docs and code after the policy change (bigcherry-2): BUILD_CACHE_GC.md and RUNS_LAYOUT.md still describe the removed protections (unadopted flat runs, symlinks pinning, run and serving manifests pinning, quarantine); the legacy-runs adoption and migrate machinery, the run-record and stored-manifest pin scanners and their tests are dead if nothing else reads them (no legacy readers or shims); test_legacy_runs fails on main against the new policy. 2. Patch tests: producer.py lost its write_json_atomic import (16 patch tests failed): fixed (PR 1542) with an undefined-names test. 3. native-shelved/: each file's reason is in component.toml; decide per file keep, re-port or retire (archive unchanged) at the RCS15 gate. 4. RCS15 to RCS18 text: align the wording with the glossary (libr3d is upstream's native library, libr3 ours, libbcr3 after RCS23 stage 0). 5. The unmerged family compile targets (RCS20). 6. Review the engine docs under docs/reference/engines/radiance for statements about libraries and the libr3 provider; fix or point at the glossary. 7. The patch lifecycle docs and the patch-author and bump-radiance skills: add the composition gates (RCS19) and the owned-unit drift check (RCS23) when they exist, not before.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Each item closed by a merged PR; the doc and test suites pass offline; grep for the removed concepts finds no live references.

## Effort & Risk

Medium; mostly mechanical, risk is deleting something with a hidden caller: grep callers first and keep what has one.

## Standards



## Acceptance Criteria

No document describes behaviour the code no longer has, no test asserts removed behaviour, no dead adoption or pin machinery remains, and every kernel-library term has one meaning.

## Notes

Created 2026-11-25 from the design review pass. Related: RCS19 to RCS23.

## Change Log

- 2026-11-25T01:02:59.740271+00:00 (created-by): Created by lead
- 2026-11-25T01:03:17.409050+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
