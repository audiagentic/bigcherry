---
id: BHK02
order: 2
plan: build-host-kernel
state: pending
created-at: '2026-11-18T01:29:35.902016+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P2
work: M
---

# Read active memory timings through an ABI-matched host diagnostic

## Description

Depends on BHK01's verified matching headers and Module.symvers. Audit a pinned read-only Alder Lake DDR5 reader (RamSleuth/CoreFreq candidate), compile it separately, and export a host snapshot for the container. Preserve BIOS-reported speed versus SPD capabilities versus live register data. No force-load, kernel replacement, container restart or timing writes during planning.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Check ABI/build identity, read-only code paths and decoded field provenance; one live readout only after an explicitly scheduled module-load step.

## Effort & Risk



## Standards



## Acceptance Criteria

Active speed/timings are reported with source and confidence, or a concrete unsupported-register reason; protected kernel state is preserved.

## Notes

Plan-only follow-up to the earlier RAM baseline and timing investigation. BHK01 already owns kernel recovery; do not duplicate it or modify live host settings as part of planning.

## Change Log

- 2026-11-18T01:29:35.902016+00:00 (created-by): Created by codex
