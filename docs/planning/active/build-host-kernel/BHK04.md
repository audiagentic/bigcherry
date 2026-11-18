---
id: BHK04
order: 4
plan: build-host-kernel
state: pending
created-at: '2026-11-18T01:29:48.426851+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P2
work: M
---

# Audit and repeat the RAM baseline after the four-DIMM upgrade

## Description

Use docs/evidence/2026-11-18-memory-baseline as historical evidence. Audit the STREAM-style harness byte accounting (Triad reads two arrays and writes one), size/alignment and run identity before reuse; retain original reported numbers without silently rewriting history. Record 2-versus-4-DIMM slots, capacity, BIOS settings, active timing source, compiler, affinity, governor and background load.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Only a short repeatable before/after bandwidth/latency sample under comparable conditions; do not treat this as a full stability test.

## Effort & Risk



## Standards



## Acceptance Criteria

A corrected method and traceable comparable report quantify the capacity/bandwidth/latency change; any incomparable historical rate or missing timing is explicitly labeled.

## Notes

Plan-only follow-up to the earlier RAM baseline and timing investigation. BHK01 already owns kernel recovery; do not duplicate it or modify live host settings as part of planning.

## Change Log

- 2026-11-18T01:29:48.426851+00:00 (created-by): Created by codex
