---
id: PWR02
order: 0
plan: run-power-efficiency
state: pending
created-at: '2026-10-27T02:15:44.041717+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: M
---

# GPU board power cap sweep: tokens per joule against throughput for decode and prefill

## Description

Decode is memory/latency bound, so a lower board power cap on the XTX helpers and the R9700 may cost little decode speed while saving watts; prefill is the phase that would slow. Sweep power1_cap per card and measure tok/s and tok/J for tg512/tg2048/pp1024/pp4096. A winning cap must be derived by the release build (measured per device x build, cached), not hand-configured.

## Steps

1. Harness: set/restore caps (sudo, sysfs), run the bench per cap level, rotate order.
2. Curve per card and phase.
3. If worthwhile: release-build calibration that picks the cap per phase, with a no-regression guard.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Curves recorded; any adopted cap has an A/B showing the accepted speed cost and the energy saved.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Reversible host setting; needs explicit hardware authorisation and sudo. Owner rule: derive over configure.

## Change Log

- 2026-10-27T02:15:44.041717+00:00 (created-by): Created by agent
