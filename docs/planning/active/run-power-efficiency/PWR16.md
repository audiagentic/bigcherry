---
id: PWR16
order: 0
plan: run-power-efficiency
state: in_progress
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

2026-11-26: platform and native typed signed shader offset implementation completed. Source grammar cross-checked against upstream SMU14 driver, verified physically on gfx1201 using managed platform writes. Signed path stages/restores actual shader offset, memory ceiling, voltage and verifies all observed controls; native signed journals save identity and refuse missing original or changed identity; legacy active platform journals without complete new state refuse manual recovery. 323 hardware/core/CLI tests pass, 1 skipped; native power acceptance executable compiled/run with exact generated contract fingerprint, signed recovery/fault/zero/receipt cases and existing Python golden profiles pass; resolver/site/manifest binding syntax checks pass. No full release or native physical run. Cap-only 20/20 cases 203328 GEMMs, three restores; offset 32/32 cases 323104 GEMMs, six restores, all-card exact snapshots. Captures and per-arm energy in AI_POWER_SCREEN.md. Finalists 300W/-200MHz/-50mV and 210W/-200MHz/-75mV. Native physical owner, independent extended pressure, combined inference and long soak remain pending. GPU held by release parity job; no production profile change.

2026-11-26: 30+60 s FP32 reference 20261126T083443Z-gpu-ai-screen-c566b175881a thermal nonpass at 80 C after ~33 s; no candidate applied, all original snapshots matched. Fresh 30+60 s BF16 bracket passed 4/4: stock A1 468608 GEMMs/7481.768 J/58 C; 300W/-200MHz/-50mV 473344/7491.458 J/61 C; 210W/-200MHz/-75mV 474208/7436.050 J/62 C; stock A2 468480/7626.687 J/64 C. Both restores and all-card exact snapshot pass, lease free. BF16 energy savings ~1.8%/2.7% against stock aggregate; not inference. CI first pass 39/40 serving tests, bc_manifest_power_test fake populated mutually exclusive shader capabilities. Corrected fixture in 1381502de without changing admission; rerun pending. Driver health incomplete. Native live owner, VRAM/burst/mixed pressure and model ABBA pending; do not promote profiles.

## Change Log

- 2026-11-26T08:12:38.552070+00:00 (created-by): Created by agent
- 2026-11-26T08:13:11.345658+00:00 (updated-by): Updated: section:notes
- 2026-11-26T08:32:54.215371+00:00 (updated-by): Updated: section:notes
- 2026-11-26T08:32:59.239311+00:00 (state-transition): State: pending → in_progress
- 2026-11-26T08:44:35.266324+00:00 (updated-by): Updated: section:notes
