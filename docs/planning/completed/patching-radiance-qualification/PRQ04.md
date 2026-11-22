---
id: PRQ04
order: 0
plan: patching-radiance-qualification
state: completed
created-at: '2026-11-15T22:57:12.711851+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Split production source.radiance from unqualified patches and gate it in CI

## Description

source.radiance currently lists untested sets (host-compat, helper-config, vtp14-mixed-tp) while build_evidence only checks validated sets, so the production source has carried untested and non-applying patches unnoticed. Split: source.radiance = validated sets only; unqualified patches live in a named experimental source (e.g. radiance-experimental, production plus the PRQ03 patches, layered like radiance-diag) so they stay buildable and tracked. Add a CI check that patch-rebase-check --source radiance (and the experimental source) is clean, and that every patch in source.radiance is validated.

## Steps

1. After PRQ01/PRQ02, move unpromoted patches to the experimental source in recipes.toml.
2. Add the rebase-clean check and a validated-only test for source.radiance (extend test_radiance_sources_share_production).
3. Document the split in docs/reference/testing/PATCH_VALIDATION.md.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

New tests fail on a deliberately broken anchor and on an untested patch in source.radiance; bigcherry check passes.

## Effort & Risk



## Standards



## Acceptance Criteria

Production sources admit validated patches only, diagnostic sources admit diagnostics with activation proof, and the negative source-class tests pass on the current 2.2 recipe.

## Notes

2026-11-22 reviewed closure RV4329: Revised scope delivered in #990: production source admission is guarded against nonvalidated patches and vendor pin drift; splitting a second experimental source would break the originally selected serving layouts. Evidence: docs/evidence/2026-11-22-prq01-prq05-reconciliation/README.md. Focused negative/positive source-class regression 11 passed, 5 subtests. No new GPU, throughput or code changes; preserve 2.2 RUM/LTX owners.

## Acceptance

Production serves only validated patches; every excluded patch is listed in PRQ03.

## Change Log

- 2026-11-15T22:57:12.711851+00:00 (created-by): Created by agent

## Revision 2026-11-16

Splitting out the remaining untested patches would break layouts that use them (rad0030 by qwen27b-tp3-int8, rad0036 by the -long configs). The split is therefore replaced by qualifying them in place (PRQ03) and a guard: a test that every patch in source.radiance is validated or listed here with a reason, so an untested patch cannot enter production unnoticed, plus the vendor-pin guard from PRQ01.

## Outcome 2026-11-16

The guard is in main (#990): the vendor checkout must match the recipes pin, and source.radiance contains only validated patches or listed exceptions. PRQ01's cause (vendor drift) is guarded by the same pin check. The exception list is now empty and is superseded by the source-class rule of PRQ05.

## Reviews

- RV4329
- 2026-11-22T04:59:26.505376+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
- 2026-11-22T04:59:36.856182+00:00 (state-transition): State: pending → completed
