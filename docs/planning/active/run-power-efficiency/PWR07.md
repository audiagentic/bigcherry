---
id: PWR07
order: 0
plan: run-power-efficiency
state: pending
created-at: '2026-11-10T13:12:56.453273+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Portable power capability probe, power receipt and safe-failure policy (slice 1 of power management)

## Description

Power management must never fail a start on hardware it does not support (Intel or AMD CPUs, other cpufreq drivers, VMs/containers with no cpufreq or sysfs, Windows dev PCs, AMD GPUs without OverDrive, other vendors). Slice 1: a read-only capability probe (CPU vendor/driver/governor/EPP/boost/energy counters; per GPU driver, OverDrive and power files, hwmon sensors, ranges; each item supported / unsupported / unreadable / not_permitted with the reason, never an exception), a power receipt written into every run (observed state of the layout's devices and the CPU at start and stop; requested/applied/restored when a profile is used; absent data recorded as unavailable), and a failure policy (no request -> never touches hardware, never fails a start; an explicit request for an unsupported setting refuses the start before any write; apply only to the layout's devices with a durable snapshot, readback and rollback).

## Steps

1. Probe and receipt modules without Linux-only imports at import time.
2. Fake-sysfs fixture trees for Intel, AMD (amd_pstate / acpi-cpufreq), no cpufreq, read-only, missing OverDrive, no GPU, permission denied; unit tests per case; a read-only test on this host.
3. Launcher and benchmark integration: receipt in the run output and a launcher line.
4. Cases we cannot test (AMD Ryzen/EPYC, Windows, NVIDIA) are covered by the same capability logic and listed as assumed, fixtures only.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Fixture tests for every named case, a read-only probe test on this host, and the launcher start unchanged on a host with nothing to apply.

## Effort & Risk



## Standards



## Acceptance Criteria

A start never fails because a power feature is absent; every run records its observed power state; an explicit request on unsupported hardware refuses before any write with the reason.

## Notes

Plan: gateway request req_main_8ad977841284 (power slice 1). Brief: runs/rad49-gpt-review/followups/POWER-PORTABILITY-PLAN-BRIEF.md. Parent PWR04.

## Change Log

- 2026-11-10T13:12:56.453273+00:00 (created-by): Created by agent
