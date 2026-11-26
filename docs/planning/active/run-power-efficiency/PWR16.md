---
id: PWR16
order: 0
plan: run-power-efficiency
state: pending
created-at: '2026-11-26T08:12:38.552070+00:00'
breadth: ''
skill: ''
created-by: agent
---

# Implement RDNA4 shader-offset transaction support and qualify R9700

## Description

R9700 exposes signed OD_SCLK_OFFSET, not an absolute shader ceiling. Both current apply owners refuse it; voltage-only restore assumes a shader ceiling. No raw sysfs fallback.

## Steps

Verify driver command/coupling; extend existing platform and native typed state, probes, writers, journals, receipts and recovery; fault-inject nonzero originals, coupled drift, lease loss, partial commits, legacy journals; offline suites, reciprocal admission PWR15, supervised physical pressure and native matched repeats.

## Detailed Solution & Technical Design

Derive control kind from actual supported capability; enforce mutual exclusion, signed ranges, exact coupled originals, per-write lease/identity, terminal readback. Extend gpu_power/PowerTransaction and native Tuning/ControlId/Linux parser/render/records/qualification. Old active records without required originals refuse; maintain existing apply refusal until complete. Cap-only sweep independent.

## Code Samples & Guidance



## Files



## Validation

Python hardware/CLI and native transaction acceptance tests; driver source review; supervised live readback, restored originals, numerical and energy evidence.

## Effort & Risk



## Standards



## Acceptance Criteria

Both owners pass fault and offline suites, actual gfx1201 apply/restore and pressure tests, matched inference and extended stability before promotion. No transferred XTX offsets or default changes.

## Notes

2026-11-26: R9700 read-only probe shows OD_SCLK_OFFSET (signed -500..1000 MHz, current 0), voltage -200..0 mV current 0, MCLK1259, cap300 W range210..330. Independent BF16/FP32 768 square hipBLASLt baseline passes in captures 20261126T080808Z-gpu-ai-screen-21022b43708e and 20261126T080906Z-gpu-ai-screen-e326d84ca65f. Mixed settings request refused before writes in 20261126T080920Z-gpu-ai-screen-1e457e9a4260: no platform shader-offset transaction backend. Native resolver also refuses offsets; both restore models assume absolute shader ceiling for voltage. All four original snapshots matched. PWR16 owns backend work. Cap-only 270/240/210 W sweep queued behind release acceptance; no new cap result yet.

## Change Log

- 2026-11-26T08:12:38.552070+00:00 (created-by): Created by agent
- 2026-11-26T08:13:11.345658+00:00 (updated-by): Updated: section:notes
