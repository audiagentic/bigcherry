---
id: PRQ02
order: 0
plan: patching-radiance-qualification
state: pending
created-at: '2026-11-15T22:56:57.003682+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Lite-promote the evidenced radiance patches from existing runs (QFP18)

## Description

Promote with patch-promote --profile-only using evidence taken from runs we already have, in dependency order, one PR per batch. Evidence drafts: docs/planning/active/patching-radiance-qualification/evidence/e1-tp3-base.md (rad0016-0019; VTP14 KLD TP3 vs TP2 exact 0.00307 mean), e2-prefill.md (rad0020, 0025-0027; VTP31 experiments 8, 9, 11), e3-allreduce.md (rad0021, 0022, 0024, 0032, 0033; VTP14/31/36), e4-ownership.md (rad0028, rad0029; VTP31 experiment 13, VTP33). Also candidates: rad0012 (hardware identity run recorded in its SUMMARY), diagnostics rad0004, rad0006, rad0010, rad0023, rad0031 on their own emitted markers. Move them into serving-core as the tool does.

## Steps

1. After PRQ01, run patch-promote per batch (dependency order: 0016-0019, 0020+0021+0022, 0024, 0025-0027, 0028-0029, 0032-0033).
2. Re-read each evidence line against the cited log before running; edit lines that overstate.
3. Merge each PR before the next (recipes.toml conflicts).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

patch-verify-evidence passes for each promoted patch; patch-gates promote intent passes; PR checks green.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes



## Acceptance

Every patch with run evidence is validated or has a recorded reason in PRQ03.

## Change Log

- 2026-11-15T22:56:57.003682+00:00 (created-by): Created by agent


## Progress 2026-11-16

Promoted through the lightweight tier and merged: rad0016-0019 (#983), rad0020 and rad0025-0027 (#984), rad0021, rad0022, rad0024, rad0031, rad0032, rad0033 (#985), rad0028 and rad0029 (#986), rad0012 (#987, evidence in this folder's e5 via the rsc08-s3-gate and flashnext identity runs), rad0023 (#988). All are in serving-core. Diagnostics rad0004, rad0006 and rad0010 stay untested by design (diagnostic-only sources; promoting them would put them in production); recorded in PRQ03.
