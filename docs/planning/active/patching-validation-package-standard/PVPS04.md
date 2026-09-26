---
id: PVPS04
order: 0
plan: patching-validation-package-standard
state: pending
created-at: '2026-09-25T12:43:53.159999+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: M
---

# Skip tune/replay builds on producer campaigns (they are built but never executed)

## Description

Every standard producer session builds five trees (tune, replay, stock, control, validation-subject). On the producer path tune/replay binaries are never run; they only feed campaign_build_identities into the campaign identity digest and the evidence record. Dropping them saves ~40% of per-session build work.

## Steps

1. Confirm no producer or dispatcher path executes tune_bin/replay_bin.
2. Make campaign_build_identities producer-path = {stock} (plus the PVPS03 base arm), migrate the identity digest and evidence schema (no legacy fallback), update verify-evidence for new records while keeping historical records verifiable as history.
3. Keep tune/replay builds for the tuning campaign path that actually runs them.
4. Tests + one hardware session showing 3 builds (stock, control, subject).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Offline suite; patch-verify-evidence on existing records still reports them as historical; a new session persists evidence with the new identity shape.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Filed 2026-09-25 alongside the ccache fix (9c4dee01): ccache was at the 5 GiB default with 3658 cleanups and a 55% hit rate because content-addressed worktree paths defeated cross-tree hits.

## Change Log

- 2026-09-25T12:43:53.159999+00:00 (created-by): Created by agent
