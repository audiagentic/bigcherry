---
id: PWR03
order: 0
plan: run-power-efficiency
state: pending
created-at: '2026-10-27T02:15:48.543469+00:00'
breadth: ''
skill: basic
created-by: agent
priority: P3
work: S
---

# CPU power: amd_pstate mode/EPP and C-state residency on the serving host

## Description

Check the CPU frequency driver mode and energy-performance preference and whether idle cores reach deep C-states, at idle and while serving. The helper lane threads' yield loops block C-states during bursts (RAD25/RAD26 remove them), so measure before and after those. Host checks belong in the tooling's start-up environment check with a warning, per the owner's 'check the host and warn' rule.

## Steps

1. Read cpufreq/amd_pstate status, EPP, cpuidle residency (idle, decode).
2. Try EPP settings with a decode/prefill A/B.
3. Add the chosen expectation to the start-up host check.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

C-state residency and package power recorded; any setting change has an A/B with no regression.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Depends on PWR01 for the baseline; most useful after RAD25.

## Change Log

- 2026-10-27T02:15:48.543469+00:00 (created-by): Created by agent
