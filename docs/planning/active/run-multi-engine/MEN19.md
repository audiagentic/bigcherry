---
id: MEN19
order: 0
plan: run-multi-engine
state: pending
created-at: '2026-10-17T09:09:41.576164+00:00'
breadth: ''
skill: basic
created-by: agent
priority: P2
work: S
---

# Bump-time native rewrite drift check, and pin-release confirm-then-run

## Description

Owner-approved (2026-10-17): (a) radiance pin-bump runs r3_rewrite.py offline against the new upstream as an early phase and reports failing native blocks per file (the 1.7.0 port found 38 only by hand); generic via an adapter check. (b) pin-release runs all stages in one go with a single explicit confirmation before tagging.

## Steps

1. Adapter rewrite-drift check before compile. 2. pin-release --confirm prompt / flag before the tag phase. 3. Tests.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

A bump against a tree with a broken native block stops in the drift phase with the block list.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-10-17T09:09:41.576164+00:00 (created-by): Created by agent
