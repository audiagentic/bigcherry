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

Current scope (takes precedence over earlier breadth): 2026-11-27: absorb PWR02/PWR10. After PWR15/PWR12, qualify stock and existing efficient XTX/R9700 finalists (210 vs 300 W); distinct performance setting only if justified. Repeat matched stock/candidate/closing stock, extended/concurrent inference, pinned engine/model/kernel, dense and relevant MoE roles, quality, throughput/TTFT/phase energy, thermals, health and exact restore. PWR01 system measurement. No exhaustive search or promotion without approval; 210 W recovery failure remains nonpass.

Final owner-selected scope supersedes the earlier finalist comparison: stock vs efficient (210 W R9700) only; verify deployed recovery/fence and available lease, match stock/efficient/stock, complete PWR12 coverage and PWR01 phase energy in same campaign.

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

Current acceptance scope: 2026-11-27: absorb PWR02/PWR10. After PWR15/PWR12, qualify stock and existing efficient XTX/R9700 finalists (210 vs 300 W); distinct performance setting only if justified. Repeat matched stock/candidate/closing stock, extended/concurrent inference, pinned engine/model/kernel, dense and relevant MoE roles, quality, throughput/TTFT/phase energy, thermals, health and exact restore. PWR01 system measurement. No exhaustive search or promotion without approval; 210 W recovery failure remains nonpass.

Final owner-selected scope: use only stock and efficient at the specified 210 W settings. Working numeric thresholds in latest note, strict correctness and restoration. Preserve historical 300 W comparison without further runs.

## Notes

2026-11-26: R9700 read-only probe shows OD_SCLK_OFFSET (signed -500..1000 MHz, current 0), voltage -200..0 mV current 0, MCLK1259, cap300 W range210..330. Independent BF16/FP32 768 square hipBLASLt baseline passes in captures 20261126T080808Z-gpu-ai-screen-21022b43708e and 20261126T080906Z-gpu-ai-screen-e326d84ca65f. Mixed settings request refused before writes in 20261126T080920Z-gpu-ai-screen-1e457e9a4260: no platform shader-offset transaction backend. Native resolver also refuses offsets; both restore models assume absolute shader ceiling for voltage. All four original snapshots matched. PWR16 owns backend work. Cap-only 270/240/210 W sweep queued behind release acceptance; no new cap result yet.

2026-11-26: platform and native typed signed shader offset implementation completed. Source grammar cross-checked against upstream SMU14 driver, verified physically on gfx1201 using managed platform writes. Signed path stages/restores actual shader offset, memory ceiling, voltage and verifies all observed controls; native signed journals save identity and refuse missing original or changed identity; legacy active platform journals without complete new state refuse manual recovery. 323 hardware/core/CLI tests pass, 1 skipped; native power acceptance executable compiled/run with exact generated contract fingerprint, signed recovery/fault/zero/receipt cases and existing Python golden profiles pass; resolver/site/manifest binding syntax checks pass. No full release or native physical run. Cap-only 20/20 cases 203328 GEMMs, three restores; offset 32/32 cases 323104 GEMMs, six restores, all-card exact snapshots. Captures and per-arm energy in AI_POWER_SCREEN.md. Finalists 300W/-200MHz/-50mV and 210W/-200MHz/-75mV. Native physical owner, independent extended pressure, combined inference and long soak remain pending. GPU held by release parity job; no production profile change.

2026-11-26: 30+60 s FP32 reference 20261126T083443Z-gpu-ai-screen-c566b175881a thermal nonpass at 80 C after ~33 s; no candidate applied, all original snapshots matched. Fresh 30+60 s BF16 bracket passed 4/4: stock A1 468608 GEMMs/7481.768 J/58 C; 300W/-200MHz/-50mV 473344/7491.458 J/61 C; 210W/-200MHz/-75mV 474208/7436.050 J/62 C; stock A2 468480/7626.687 J/64 C. Both restores and all-card exact snapshot pass, lease free. BF16 energy savings ~1.8%/2.7% against stock aggregate; not inference. CI first pass 39/40 serving tests, bc_manifest_power_test fake populated mutually exclusive shader capabilities. Corrected fixture in 1381502de without changing admission; rerun pending. Driver health incomplete. Native live owner, VRAM/burst/mixed pressure and model ABBA pending; do not promote profiles.

2026-11-26 expanded voltage-only and clock screens complete: 64/64 cases, 659136 verified GEMMs, 12/12 candidate restores and all-card exact snapshots. Original clock 300 W through -150 mV passed, -14.9% estimated energy/GEMM and +1.3% rate. At -125 mV, offsets 0/-100/-200/-300/-400 MHz all pass; 210 W/-200 also passes. 300 W/-300/-125 saves 20.0% energy/GEMM at -0.36% rate, 300 W/-400/-125 saves 22.4% at -1.03% rate. Capture IDs in AI_POWER_SCREEN.md. Fresh 90 s BF16 sustained finalists selected from evidence; no measured failure boundary, driver health incomplete, no default changes. Native owner release and further pressure remain required.

2026-11-26: expanded finalists complete. Both 300 W/-300/-125 and -400/-125 pass 90 s BF16, 4/4 including bracketed references, 1890656 GEMMs, two restores. -300/-125 selected on sustained energy and throughput (8.5% lower J/GEMM, +1.1% rate vs stock). Fresh matched 28546 MiB independent VRAM 2/2 pass (two full patterns each), 60 s mixed resident 2/2 pass (one full sweep each, -7.1% J/GEMM, +0.4% rate), 15 s x 4 burst 8/8 pass (-7.2% J/GEMM, +0.3% aggregate rate). Combined expanded cohort 80/80 cases, 3343488 verified GEMMs, 17 exact candidate restores. All captures harvested, all-card originals match, leases free. Retain -300/-125/300 W as supervised research finalist. Prior sustained FP32 thermal stop, missing driver log and native/extended/concurrent qualification still outstanding; installed 2020.23.0 predates PWR16, use a verified build containing 4eabbd6c7 or later. No production defaults changed.

2026-11-27 closeout: Native 2020.23.1 comparison completed 16 matching inference requests. Exact 210 W R9700 candidate passed independent screen; native second XTX restore reported a write fault. All physical snapshots match originals but journal remains unresolved. Closing stock not run. Verified diagnostic capture 20261126T212407Z-native-power-cap-0cc14c765951 preserves 112 sources. Full metrics in power-tuning-evidence-2026-11.md. Keep plan active for native recovery, repeated comparison, driver health and extended stability. No production promotion.

2026-11-27: absorb PWR02/PWR10. After PWR15/PWR12, qualify stock and existing efficient XTX/R9700 finalists (210 vs 300 W); distinct performance setting only if justified. Repeat matched stock/candidate/closing stock, extended/concurrent inference, pinned engine/model/kernel, dense and relevant MoE roles, quality, throughput/TTFT/phase energy, thermals, health and exact restore. PWR01 system measurement. No exhaustive search or promotion without approval; 210 W recovery failure remains nonpass.

Owner direction 2026-11-27: stock and efficient only. Efficient: both XTX 2700 MHz/-125 mV, R9700 -300 MHz/-125 mV/210 W, memory unchanged. Preserve 300 W evidence but do not repeat it. Working acceptance: <=2% throughput loss each phase, <=10% TTFT increase, >=15% GPU J/token reduction, zero new numerical errors/resets/throttling, exact restore. Stock/efficient/stock matched dense, MoE, concurrent, sustained and burst. No further cap sweeps. PWR15 deploy and lease gates remain; no implicit promotion.

2026-11-27 exact 210W/-300MHz/-125mV R9700 final independent pressure/matrix brackets: 48/48 cases across 12 managed runs, 2931360 verified GEMMs plus six full memory patterns, four exact candidate restores and full fleet original snapshot equality. Refer PWR12 for IDs. Existing XTX same-settings evidence reused. Tagged 2020.23.1 remains older than merged owner fixes; no repeat native model test or release promotion. CPU counters and kernel health log unavailable.

2026-11-27 owner-selected intent locked in canonical power-profiles.yaml: stock empty; efficient gfx1100 2700 MHz/-125 mV and gfx1201 -300 MHz/-125 mV/210 W for engine and helpers, gfx1030 and memory unchanged. Observe-only default; existing per-device evidence and apply checks unchanged. Legacy names retained for historical compatibility, not new tuning targets. Python hardware/CLI tests 304 pass, 1 skipped. No automatic application, release promotion or closure of native stability gates.

Owner requested model selection: all 18 shipped native YAMLs select efficient as named intent. Stock parity and PWR01 baseline explicitly select stock; no change to explicit apply, lease, qualification or observe mode. Native production application still awaits acceptance and updated release.

## Change Log

- 2026-11-26T08:12:38.552070+00:00 (created-by): Created by agent
- 2026-11-26T08:13:11.345658+00:00 (updated-by): Updated: section:notes
- 2026-11-26T08:32:54.215371+00:00 (updated-by): Updated: section:notes
- 2026-11-26T08:32:59.239311+00:00 (state-transition): State: pending → in_progress
- 2026-11-26T08:44:35.266324+00:00 (updated-by): Updated: section:notes
- 2026-11-26T08:58:07.547084+00:00 (updated-by): Updated: section:notes
- 2026-11-26T09:13:45.754693+00:00 (updated-by): Updated: section:notes
- 2026-11-26T21:26:10.842990+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4682
- 2026-11-26T23:12:50.159623+00:00 (updated-by): Updated: section:notes, section:steps, section:acceptance_criteria
- 2026-11-27T02:41:59.212978+00:00 (updated-by): Updated: section:notes
- 2026-11-27T02:42:10.223512+00:00 (updated-by): Updated: section:steps, section:acceptance_criteria
- 2026-11-27T03:00:31.112506+00:00 (updated-by): Updated: section:notes
- 2026-11-27T03:03:01.761424+00:00 (updated-by): Updated: section:notes
- 2026-11-27T03:14:05.940167+00:00 (updated-by): Updated: section:notes
