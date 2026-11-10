---
id: PWR07
order: 0
plan: run-power-efficiency
state: in_progress
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


## Update 2026-11-10 (GPT power manifest review, req_main_eb983900e21b)

Scope narrowed to portability and observability: CPU probe (PR #810), GPU capability probe, receipt types. Manifest-owned profiles moved to PWR08 (bigcherry/v2) and the transactional controller to PWR09; no manifest field is added by this item.


## Update 2026-11-10 (PWR06 R9700 GFXOFF finding)

R9700 receipt fields (from PWR06): MES/KIQ/SMC firmware, kernel, amdgpu srcversion, measured idle watts, gpu_busy_percent, VDDGFX (GFXOFF-active proxy) and queue count where observable; observation only, kernel/firmware are host prerequisites.


## Implementation status: 2026-11-10

Merged: portable CPU probe #810, GPU OverDrive/capability probe #812; durable receipt and transaction infrastructure #815; gpu-power command and fail-closed shell launcher transaction adapter #816. Python supervisor receipt integration is the PWR09/RSC11 S12 card-free slice. Still open: receipts for every production serving/benchmark start, host read-only smoke, portability on additional real drivers and hardware validation. Observe-only starts require no supported controls and must never write hardware.

## XTX forensic receipt requirements — 2026-11-11 (research only)

The October `pwr01-baseline-10271708` vs `pwr01-baseline-10272216` comparison captured board `amd-smi SOCKET_POWER` and CPU RAPL watts, **not** real SCLK, VDDGFX, DPM, throttles, firmware, ambient or per-arm cap. November `xtx-undervolt-ab/journals/D1.json` records pre-offset OD SCLK max **2950/2935 MHz**, but power cap `power_cap_uw=0`: **not trustworthy evidence of a zero cap or a restored stock 303 W cap**. November's 1-Hz `power1_average` was XTX-only in sustained decode; short-window watt averages conflate workloads. Do not present October summed medians as integrated energy.

For each XTX BDF and each phase, require a receipt of: exact real cap/default/min/max with sensor identification and status, requested/actual OD SCLK max and effective clock distribution, VDDGFX offset and voltage, DPM/perf/profile mode (observation only), frequency, board watts, busy, temperature/fan/throttle, firmware MES/SMU, kernel/amdgpu srcversion, ROCm, model/build/layout/MTP, sampler source/cadence, ambient or **unknown**, time-window markers, token counts and GPU+CPU energy. Preserve supported/unsupported/unreadable distinctions. Do not infer from a single numeric zero that a control is supported. This is **a requirement for future instrumentation**, not a claim that current receipts expose those fields or that a live probe has been performed. See ranked acceptance in the November power evidence appendix.


## Recheck instrumentation failure modes — 2026-11-11

The `xtx-undervolt-ab/watts.sh` source measures 1-Hz `power1_average` via DRM `card0/card1` without recording their BDF identities; October's `pwr01-baseline.sh` used BDF-mapped AMD-SMI `SOCKET_POWER` and CPU RAPL. These are related *averaged socket/SoC power* interfaces, not proven identical on this host; `power1_input` (when available) is instantaneous, not interchangeable with the averaged output. Demand simultaneous time-aligned `amd-smi SOCKET_POWER`, BDF-mapped hwmon `power1_average` and `power1_input` in the **same stock and -125 mV decode windows**, plus a readback of `pp_od_clk_voltage` at apply, at mid-decode and before stop. Capture real clocks `freq1_input`/AMD-SMI, DPM occupancy, power cap (zero in prior journal ambiguous), MES/SMU, temperature/hotspot, busy, fan and throttle, with thermal baselines. Record phase boundaries for 40-s October-harness phases. No current XTX A/B has demonstrated during-decode offset persistence, sensor equivalence, or matched thermal state. Existing `summarise.py` only prints power sample **counts**, not a watt or energy comparison. This is a requirement for future receipts, not a completed capability.
