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

2026-10-27: power caps cannot help the XTX (driver minimum 272 W, above the 155-198 W they draw); R9700 cap minimum 210 W vs ~216 W draw. Max shader clock (OD_SCLK, OverDrive enabled) is the lever. Sweep (pwr02-xtx-sclk-sweep.sh, 2 rotated runs a level, runs pwr01-baseline-10271930..54): default (2960-3010 MHz) decode 91.0 tok/s, prefill 3274 tok/s, XTX 156/198 W each, decode 6.78 J/tok; 2500 MHz: decode -2.2%, prefill -2.5%, XTX 95/138 W (about -120 W across both), decode 5.53 J/tok (-18%); 2000 MHz: -7.3%/-7.5%, XTX 72/113 W, 5.24 J/tok. R9700 and CPU unchanged. Next: 2800/2700 MHz to find the no-penalty point; then a voltage-offset (undervolt) test at default clock only with fixed-text quality checks; a winning setting must be applied by tooling (privileged, restored on exit) and derived per device x build.

2026-10-27 clock sweep 2 (runs pwr01-baseline-10271959..10272023): 2800 MHz decode -2.6%, prefill -1.3%, XTX 118/162 W; 2700 MHz -1.7%/-2.0%, 110/156 W. No penalty-free clock limit (decode runs ~3050 MHz). Undervolt sweep (VDDGFX offset at default clock, 2 rotated runs a level, runs 10272031..10272104): -25 mV no change; -50 mV XTX 156/192 W, speed unchanged; -75 mV XTX 149/186 W (about -12 W decode, -13 W prefill per card), decode 89.6 vs 89.1 tok/s, prefill 3337 vs 3304 tok/s; decode text hash set identical at every level; no amdgpu faults/resets. Next: -100/-125/-150 mV with longer stability runs, then tooling to apply the chosen offset (privileged, restored on exit) per device.

2026-10-27 prior art (GPT req_bigcherry_32d421bc02bc; key source checked): szafranski/rx7900xtx-qwen3.8-27b-tuning docs/power-and-undervolt.md - XTX at 272 W cap: SCLK 2400 MHz -0.4% tok/s, 2200 MHz -3.1%; without a clock cap -125 mV raised tok/s 58.3 -> 60.6; with a 2200 MHz cap the offset barely changes VDDGFX (715.6 vs 704.8 mV at -75 vs -200 mV): clock cap pins voltage; kept -75 mV after a 30-min soak. R9700: 4x R9700 vLLM 300 -> 210 W cap with no tok/s change (grosan.co.uk); -100 mV hung; per-card stable offsets differ (-30 vs -70/-120 mV, daimonionnn toolkit). COMPUTE profile: no matched saving; CachyOS #888 regression. MCLK changes risky (R9700 -6.6% decode). Next: XTX -100/-125 mV at stock clock with soak; R9700 210 W cap and its own undervolt sweep; helper threads on E-cores (12700KF has 4) vs package power.

2026-10-27 deep undervolt (runs 10272121..45): -100 mV XTX 142/180 W, -125 mV 136/175 W, speed equal or +1%, identical decode text set, no errors. Soak at -125 mV (pwr01-soak, 40 min, run 10272216): XTX 133/177 W, decode 90.5 tok/s, prefill 3344 tok/s, 53 responses with the stock text set, no amdgpu faults/resets. Recommendation: -125 mV per XTX as the calibrated setting ([host.*.devices.power], evidence = this run), applied by `bigcherry gpu-power apply`; R9700 sweep next.

## Change Log

- 2026-10-27T02:15:44.041717+00:00 (created-by): Created by agent
- 2026-10-27T08:59:53.073139+00:00 (updated-by): Updated: section:notes
- 2026-10-27T10:09:45.576821+00:00 (updated-by): Updated: section:notes
- 2026-10-27T10:11:28.723089+00:00 (updated-by): Updated: section:notes
- 2026-10-27T11:59:08.273830+00:00 (updated-by): Updated: section:notes
