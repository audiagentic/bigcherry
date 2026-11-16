---
id: PRQ04
order: 0
plan: patching-radiance-qualification
state: pending
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



## Notes



## Acceptance

Production serves only validated patches; every excluded patch is listed in PRQ03.

## Change Log

- 2026-11-15T22:57:12.711851+00:00 (created-by): Created by agent


## Revision 2026-11-16

Splitting out the remaining untested patches would break layouts that use them (rad0030 by qwen27b-tp3-int8, rad0036 by the -long configs). The split is therefore replaced by qualifying them in place (PRQ03) and a guard: a test that every patch in source.radiance is validated or listed here with a reason, so an untested patch cannot enter production unnoticed, plus the vendor-pin guard from PRQ01.
