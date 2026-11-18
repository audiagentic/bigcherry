---
id: BHK03
order: 3
plan: build-host-kernel
state: pending
created-at: '2026-11-18T01:29:42.197170+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P2
work: M
---

# Establish a verified Linux runtime timing-write path before any tuning

## Description

Depends on BHK02. BIOS realtime timing is enabled, but no supported Alder Lake DDR5 write implementation has been established. Document writable fields, controller/DRAM synchronization, live versus retraining settings, exact driver interface and recovery. Keep prototype writes disabled until code review and a maintenance window authorize a named change.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Source-level write-path and field review; readback/dry-run validation. No hardware writes or broad fuzzing in this investigation.

## Effort & Risk



## Standards



## Acceptance Criteria

A justified supported writer and narrowly scoped staged experiment are documented, or the item concludes no verified writer and retains BIOS/reboot tuning. No claim that a kernel update alone enables writes.

## Notes

Plan-only follow-up to the earlier RAM baseline and timing investigation. BHK01 already owns kernel recovery; do not duplicate it or modify live host settings as part of planning.

## Change Log

- 2026-11-18T01:29:42.197170+00:00 (created-by): Created by codex
