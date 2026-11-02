---
id: RSC04
order: 4
plan: run-serving-configuration
state: pending
created-at: '2026-11-02T00:53:10.689343+00:00'
breadth: ''
skill: basic
created-by: agent
priority: P2
work: S
---

# Retire legacy launch surfaces once their callers migrate

## Description

After RSC02/RSC03 and RCS10/RHP09: remove the bash launcher, runtime-profile argument duplication in engines/radiance/recipes.toml (build recipes and patch sets stay), obsolete or superseded R3_* aliases (e.g. R3_MOE_WORKER_MIB, R3_MOE_WORKER_PLUGIN fallbacks, R3_MOE_WORKER_QUEUE_WARM), and hand-written layout budgets, migrating every caller in the same change.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

No stale callers (grep); docs and schema inventory agree; RHP12 gate on every available layout.

## Effort & Risk



## Standards



## Acceptance Criteria

The 'Derived, not hand-picked' literal inventory is empty and the legacy surfaces are gone.

## Notes

Source: GPT configuration assessment 2026-11-02.

## Change Log

- 2026-11-02T00:53:10.689343+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261102_005338_planning-one-serve-command_5259
- 2026-11-02T00:53:56.311686+00:00 (updated-by): Updated: section:ledger-events
