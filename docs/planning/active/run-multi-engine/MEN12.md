---
id: MEN12
order: 0
plan: run-multi-engine
state: pending
created-at: '2026-10-14T22:16:19.350063+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: M
---

# First radiance bump to the latest upstream release with patches and components

## Description

Fire-test MEN10: bump radiance from 1.3.0 (89cee7c) to the latest upstream release, rebase rad0001-rad0003, rebuild libr3/libr4d/libr3moe and the helper, and re-run the smoke and a Flash-Next and 27B check on dev.

## Steps

1. Read upstream changes since 1.3.0 for anything that supersedes our patches or components (sources check equivalent).
2. pin-bump --engine radiance.
3. Fix anchors inside the owning patches; fix components against ABI changes.
4. Build + runtime smoke; Flash-Next layout and 27B two-XTX numbers and KLD vs the previous pin.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

pin-status --engine radiance --complete PASS; smoke passes; prefill/decode within drift of the previous pin or explained; KLD within noise.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Depends on MEN10.

## Change Log

- 2026-10-14T22:16:19.350063+00:00 (created-by): Created by agent
