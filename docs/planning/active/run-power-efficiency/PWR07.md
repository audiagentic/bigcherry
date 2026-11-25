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

**Transferred PWR04 acceptance (independent gate):** Carry original PWR04 startup-host observability contract under the existing portable power probe/receipt: warn, do not fail, when optional configured power state differs from observed state or when the CPU package RAPL/energy read rule is unavailable; distinguish missing capability, unreadable sensor, permission and intentional observe-only. Every start still records source/BDF/CPU observations; no power writes in observe-only or unsupported-host paths and no false success from ambiguous cap=0. Use PWR07's existing read-only probe and receipt, never a second host check or implicit power apply.

Native probe/receipt follow-up: use selected BDF and verified amdgpu hwmon name, exact units and current/min/max/default; 0/missing/unreadable/inconsistent cap is invalid, never a restore target or silently replaced by factory cap. Refuse mutations where original state cannot be safely restored. Distinguish unsupported, permission, invalid and unavailable observations; warn on optional missing RAPL/host mismatch, record actual readback, no fabricated stock. Record requested/admitted/effective/restore settings, firmware/kernel/driver, phase markers, actual clock/voltage and sensor identity; pair AMD-SMI and BDF hwmon in qualification, integrate energy with PWR01. Fake malformed/zero/ambiguous cap tests, read-only host smoke and actual native receipt validation. Probe and backend support for gfx1201 SCLK offset must be independently reviewed with RSC02 before enabling it; retain voltage/absolute SCLK/MCLK/cap controls.

## Notes

Plan: gateway request req_main_8ad977841284 (power slice 1). Brief: runs/rad49-gpt-review/followups/POWER-PORTABILITY-PLAN-BRIEF.md. Parent PWR04.

CONSOLIDATION 2026-11-22 POWER-LIFECYCLE: PWR04's original residual requirement is now owned here. PWR07 is the already deployed read-only CPU/GPU capability and receipt owner, and PWR04 explicitly delegates probe and warning work to it. Neither this transfer nor previous observational power results satisfy live device, rollback or v2 migration acceptance.

2026-11-25 audit: native C++ transaction/probe and diagnostic hooks exist; normal receipts are stock/observe. Both XTXs have reported 0 W caps in historic receipts although later direct sysfs reads were 303 W. Root cause not established. session.cpp already labels zero_cap_suspect in one path; power_linux.cpp cap.current and power_records.cpp need independent audit. R9700 SCLK offset has no native reversible backend. PWR08 consumes typed fresh capability; RSC02 owns writes/restore.

2026-11-25 owner direction: no launch manifest, all authored serving configuration YAML, no compatibility shims or legacy backports. docs/design/radiance-native-yaml-launch.md supersedes earlier manifest-launch instructions; completed RUM04/05/06/09/12 evidence is historical, not proof of the new launch path. Existing capture manifests may remain output-only. No new programme or second resolver. Probe/receipt implementation remains here. Typed owner PowerPlan uses native LinuxHardware/actual BDF, positive cap/original/readback and existing journal. Capture/receipt may link execution identity but cannot authorize power from manifest. Retain zero-cap correction and reversible gfx1201 offset proof; no second probe.

2026-11-25 cross-programme bump review: docs/design/radiance-bump-integration.md is the shared implementation-order and compatibility specification; MEN24 owns sequence and timing, domain implementation remains here. No duplicate checker, bump driver, manifest launch or legacy fallback. Bump preflight verifies actual hardware probe/restore capability and typed owner receipt on selected BDFs, invalid zero cap refusal, kernel/driver/firmware/board identity. No manifest authorization and no power writes from compiler/triage. Changes to driver/firmware require appropriate fresh capability/qualification; preserve independent host idle plans.

2026-11-25 second integration review: docs/design/radiance-integration-review-20261125.md is the exact-source review and controls conflicting earlier wording. Shipping pin remains b6b8751 (2.2.0), upstream observed main 1c104de4 is a development candidate, 119 commits and 327 changed files; no automatic repin or backport. No new compatibility layer, launch manifest, bump driver or identity machinery. Candidate Fleet telemetry adds BDF lookup, clocks/temp/memory busy/fan only on watched dashboard. Reuse presentation where useful; DPM selected level is not actual frequency. Does not replace exact amdgpu hwmon cap/current/min/max, invalid zero handling, typed restore or phase energy. No intrusive dashboard polling for idle; owner probes selected BDF only.

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

## 2026-11-11 XTX sweep: receipt requirements revealed by raw data

Every `runs/pwr-sclk/<arm>/applied.txt` captures both XTX OverDrive readbacks, and each `wl.*.log` points to an indexed, versioned `result.json`. The `pwr-phases@1` JSON explicitly records `power_sensor: sysfs:power1_average`, while the campaign was described as AMD-SMI board watt sampling; do not silently equate the two sensor identities. Reconcile with paired BDF-mapped AMD-SMI `SOCKET_POWER` in a future receipt. Throughput and watts are phase-specific; 44K doc and six-stream results are *not* paired energy measurements. CPU power was `not_sampled`. The R9700 and 6900 XT settings were unmodified.

For a **validating** `xtx-cap2700-uv125` profile, bind per-BDF selected OD max, offset and actual effective SCLK/voltage, sensor and sampling interval, model/build/kernel/driver, manifest/owner map, phase workload ID, identity hashes and output checks, thermal/health, transaction journal/readback/restore and soak/replication refs. `runs/pwr-sclk/soak27uv125` has no completion status at inspection, and container `dmesg` was denied: require **host-side amdgpu errors/resets** check. Do not accept an ambiguous `power1_cap=0` as a real cap. Future MCLK (reported 97–1250 MHz per XTX, verify fresh) and pp_power_profile_mode experiments also require full capability, reversible snapshot, receipt and no-regression validation before an apply plan can refer to them. Reference table: `docs/reference/power-tuning-evidence-2026-11.md`.

## Reviews

- RV4545
- 2026-11-22T11:42:13.063598+00:00 (updated-by): Updated: section:notes, section:acceptance_criteria
- RV4577
- 2026-11-25T01:02:37.422892+00:00 (updated-by): Updated: section:notes, section:acceptance_criteria
- RV4596
- 2026-11-25T01:39:34.429611+00:00 (updated-by): Updated: section:notes
- RV4614
- 2026-11-25T02:08:53.144048+00:00 (updated-by): Updated: section:notes
- RV4645
- 2026-11-25T02:47:05.699591+00:00 (updated-by): Updated: section:notes
