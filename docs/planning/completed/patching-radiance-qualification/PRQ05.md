---
id: PRQ05
order: 0
plan: patching-radiance-qualification
state: completed
created-at: '2026-11-16T05:16:37.157081+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P1
---

# A diagnostic patch state: the state says where a patch can build

## Description

Diagnostics cannot honestly become validated and so sit at untested forever, mixing intentional leftovers with real qualification gaps. Add one state, `diagnostic`, defined by where the patch may build, so what is in or out of production is stated by the state and enforced by one rule, not by tags, source names or hidden logic.

Rule: validated may be in any source including production; diagnostic may be in diagnostic and evidence sources only, never in a production source; untested may be in no production or diagnostic source (experiments only); rejected and superseded are out everywhere with history kept. Production sources hold only validated patches. Diagnostic sources hold validated plus diagnostic patches. Reaching `diagnostic` needs the lighter proof: patch tests and lint pass and the patch's own marker or counter shows it ran; no performance or identity claim. A diagnostic whose question is answered is retired (superseded), not promoted.

Effects: rad0004 and rad0010 become diagnostic (rad0010's kind is wrongly enhancement and becomes diagnostic); rad0006 becomes superseded (its question, whether the mover causes the MTP acceptance drop, is answered: no, RAD14); rad0023 and rad0031 stay validated (compiled in, inert unless their env is set). One test checks every source's patches against its allowed states and replaces the diagnostic-build special case in patch-lint and the untested-in-production exception list.

## Steps

1. Add `diagnostic` to the patch states in the registry, patch.toml schema, gates (G5 lifecycle: eligibility for diagnostic needs mechanics plus activation marker only) and patch-promote (a `--diagnostic` mode that writes the state and the evidence record).
2. Source classes in recipes: declare each source as production or diagnostic; one test checks allowed states per class; remove the diagnostic-build tag and its lint special case if the state subsumes it; keep the exception list only if still needed.
3. Move rad0004 and rad0010 to diagnostic and rad0006 to superseded through the new path; fix rad0010's kind.
4. Update docs/reference/testing/PATCH_VALIDATION.md (state table and the where-it-builds rule), patch-status/patch-explain output (state shown with its allowed sources), generated docs.
5. Migrate every caller in the same change (no compatibility path).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

A test fails when an untested or diagnostic patch is in a production source, when a diagnostic source holds an untested patch, and when a diagnostic patch has no activation evidence; patch-rebase-check, patch-lint, bigcherry check and the patch/release suites pass; patch-status lists no untested radiance patch outside an experiment.

## Effort & Risk



## Standards



## Acceptance Criteria

Diagnostic state and source admission are enforced without kind/tag naming shims or a second special-case gate; original diagnostic patch state transitions and documentation are complete.

## Notes

2026-11-22 reviewed closure RV4330: Source-class and diagnostic state machine, G4/G5 activation proof and docs landed in #995. Current fixtures affirm exact 2.2 source-class composition and negative cases. Evidence: docs/evidence/2026-11-22-prq01-prq05-reconciliation/README.md. Focused negative/positive source-class regression 11 passed, 5 subtests. No new GPU, throughput or code changes; preserve 2.2 RUM/LTX owners.

## Acceptance

Production builds only validated patches by one stated rule; diagnostics are closed out as diagnostic or retired; no hidden derived rules remain.

## Change Log

- 2026-11-16T05:16:37.157081+00:00 (created-by): Created by agent

## Reviews

- RV4330
- 2026-11-22T04:59:46.336969+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
- 2026-11-22T04:59:56.650690+00:00 (state-transition): State: pending → completed
