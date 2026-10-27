---
id: PWR04
order: 0
plan: run-power-efficiency
state: pending
created-at: '2026-10-27T10:13:52.881547+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Per-device power settings in environment config, a privileged apply/restore helper and a guarded controller

## Description

Power settings found by PWR02/PWR03 (XTX voltage offset, optional clock limit, R9700 cap/offset, CPU EPP) must be configuration, never code: per device in the environment settings' device inventory (keyed by locator, so each card has its own calibrated value), with tooling that applies them. Each setting records how it was validated (offset, stability soak, output identity, measured W and tok/s) and when.

## Steps

1. Schema: optional power block per [[host.*.devices]] entry (vddgfx_offset_mv, sclk_max_mhz, power_cap_w, profile) and a host cpu block (epp); validated by environment.py; absent = driver defaults.
2. A small privileged helper (one-time root install) that applies only those configured values to the listed devices, reads them back, and restores defaults on stop or lease expiry; unprivileged callers talk to it over a local socket.
3. Start-up host check warns when configured settings are not applied or the RAPL read rule is missing.
4. Later: coarse phase-aware selection (idle/decode/prefill) with dwell time and automatic rollback on regression (decode/prefill > ~1-2%, any GPU error).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Unit tests for the schema and helper command building; on the bench host the configured offset is applied and read back, a kill/stop restores defaults, and a PWR01 run with the setting matches the sweep's numbers.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Owner rules: settings configurable, not hard coded; hosts by role; derive over configure where possible (calibrate per device x build, cache). Prior art: LACT daemon model, amdgpu-clocks (PWR02 notes).

## Change Log

- 2026-10-27T10:13:52.881547+00:00 (created-by): Created by agent
