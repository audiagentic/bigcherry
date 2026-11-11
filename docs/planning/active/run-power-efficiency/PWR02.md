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

RECONCILED 2026-11-10 (gateway audit): DONE: the XTX sweeps and one 40-minute soak at -125 mV (53 decode responses with matching text, no amdgpu faults or resets): ~24 W per XTX saved in decode with no measured throughput penalty; the XTX minimum board cap sits above the workload's draw (the cap is ineffective); a 2,500 MHz clock limit cut two-XTX power by ~120 W for a 2-3% speed loss (a trade, not free). Evidence caveats: J/token came from summed median watts and throughput, not synchronised energy integration; the October 27 figure included the CPU package (~590 W decode) and the October 29 one was GPU-only (~498 W): not comparable. STILL OPEN: per-card qualification (each XTX and the R9700 separately; -125 mV is evidence for the tested cards only, not a default), the R9700 sweep, quality/identity gates per operating point, named profiles (PWR04).

## Change Log

- 2026-10-27T02:15:44.041717+00:00 (created-by): Created by agent
- 2026-10-27T08:59:53.073139+00:00 (updated-by): Updated: section:notes
- 2026-10-27T10:09:45.576821+00:00 (updated-by): Updated: section:notes
- 2026-10-27T10:11:28.723089+00:00 (updated-by): Updated: section:notes
- 2026-10-27T11:59:08.273830+00:00 (updated-by): Updated: section:notes
- 2026-11-10T13:13:15.318951+00:00 (updated-by): Updated: section:notes

## Correction / current-build XTX recovery queue — 2026-11-11 (read-only research)

The 2026-10-27 notes recommending -125 mV as calibrated describe **historical** RX 7900 XTX evidence, not a presently valid setting. In `runs/xtx-undervolt-ab/`, 2026-11-11 current Radiance 1.17.0, 7.0.13-gfxoff, four-GPU/MTP3 **C1/D1/D2/C2** sustained ABBA found ~190–191 W stock vs ~192–196 W at -125 mV per XTX, decode 325–326 tok/s in both; **no power saving**. Short ABBA A1/B1/B2/A2 agreed. Stress suggests boost converted voltage headroom to +243 MHz and +16 W, not a lower board watt reading. The October XTX cap minimum was 272 W and decode/prefill drew 154–198 W; **the October cards were not board-power-capped**. A cap sweep is conditional on a measured near-cap serving phase.

**Rank and dependencies (all unvalidated for the current build):** 0) instrumented, phase-isolated stock baseline (both decode modes, 8K prefill, idle; current build; cap/thermal/DPM/SCLK/voltage readback); 1) reversible 2900/2800/2700 MHz OD_SCLK max candidates, 0 mV, one arm/lease; 2) the *same* speed-safe SCLK with -125 mV vs SCLK alone; 3) cap 290/280/272 W only if current min/max, real cap readback and near-cap prefill justify; 4) DPM/performance-level read-only diagnostic (write path lacks PowerTransaction support); 5) per-phase static comparison before any dynamic switching design. SCLK values and combinations are candidate arms, **not approved production settings**. Each needs current-build ABBA with separate decode and prefill no-regression + integrated J/token gates. Exact low-level knobs, lease pattern and pass/fail are in `docs/reference/power-tuning-evidence-2026-11.md`, “Forensic XTX October-to-November reconciliation”. No power profile should auto-select a historic offset.


## Lead addendum: stock-power shift first, October-harness recheck before cap/clock sweeps (2026-11-11)

**First investigate the +35–37 W *stock* XTX decode rise** (October 154/155.5 W to November 190–191 W): compare single-stream vs six-stream duty, two-XTX vs three-helper tail expert ownership and routing, actual SCLK/DPM/busy, and `amd-smi SOCKET_POWER` vs hwmon `power1_average` sampled *together by BDF*. Both builds already used MTP depth 3 and 22528 MiB per XTX; speculation and raw XTX expert budget are **not** November-only additions. Kernel 7.0.13→7.0.13-gfxoff and Radiance source changes are candidates, not demonstrated causes.

**Revised dependency-ranked owner queue (one short lease per arm):** 0A) `bash tools/lab/radiance/pwr01-baseline.sh 40` stock/observe on current pinned build, retaining October single-stream decode + unique-prefix long prefill + two idles, AMD-SMI `SOCKET_POWER` and RAPL; 0B) same PWR01 script at -125 mV through fenced transaction + during-decode OD readback; repeat stock/UV/UV/stock; 0C) after *measuring* actual stock decode clocks, use separate **clock-only** `Setting(sclk_max_mhz=N, vddgfx_offset_mv=0)` and **clock+UV** `Setting(sclk_max_mhz=N, vddgfx_offset_mv=-125)` with legal, per-BDF readback-valid N, same PWR01 script. Do not assume October's reported ~3050 MHz is an allowed OD max (November journal OD maxima 2950/2935). Then revisit current six-stream shape, fine SCLK ceilings, conditional board cap, DPM read-only and later phase controls. Rank 0A–0C is now **ahead of** the prior 2900/2800/2700 sweep.

**Critical implementation guard:** `pwr01-baseline.sh` calls `rad_build` automatically and today's default launcher is `flashnext-4gpu`. Record/lock the actual compiled build and layout each time; running the October protocol is not equivalent to running the October binary/layout. `40` is the October short ~40-second-per-phase screening run; `600` is the old 40-minute soak, **not** a short lease and only a separate later approval. See full command, thermal, sensor and rollback instructions in `docs/reference/power-tuning-evidence-2026-11.md` lead addendum. Results remain **unvalidated**; no default/profile activation.

## 2026-11-11 owner-run XTX SCLK/offset sweep (measured; qualification still open)

New raw sweep: `${BC_WORK_ROOT}/runs/pwr-sclk/`, on `flashnext-4gpu`, build `daf793f13be8d5762deecb73e3372641/a29e62ada0fe691746707b2b87c3e9f3`, `7.0.13-gfxoff`. All 12 arms have 1 `pwr-phases@1`, 1 `prefill-doc-44k@1`, and 1 `decode-6stream@1` registered run with individual result JSON; both XTX OD settings were read back. See the **complete measured table and V/F CSV medians** in `docs/reference/power-tuning-evidence-2026-11.md`, “XTX SCLK × VDDGFX sweep”. The result JSON attributes phase power to `sysfs:power1_average`, **not confirmed equivalent to AMD-SMI SOCKET_POWER**. The two other cards remained at stock settings.

At 2700 MHz, comparing SCLK-only with the same SCLK plus -125 mV: per-XTX decode 109/109 → 88/88 W; prefill 165/167 → 138.5/138 W; 44K doc-prefill 3645.1 → 3658.1 tok/s; 6-stream engine 351.2 → 349.8 tok/s. `c2700uv125` row IDs: phase `587b81b5fa3b4835`, doc `b69a3764952f49b2`, decode `d9177d830ee64d89`; control `s2700`: `9331ee86e4b34adc`, `8e2d6ec68b3f4196`, `706d606817fe4288`. Three stock repeats vary by up to ~4 W/XTX in decode, 1.7% six-stream and **3.9% full-range doc-prefill**; one arm per setting does not qualify the speed/power curve. This sweep is a candidate discovery, not a completed ABBA.

For an *illustrative* whole-layout **GPU-only** J/token proxy, `pwr-phases` summed per-BDF median decode W / same-run decode tok/s and separately prefill W / same-run prefill tok/s: stock repeats 6.838/0.233, 6.002/0.250, 5.900/0.252 (decode/prefill); SCLK 2700 only 5.568/0.208; 2700/-125 5.048/0.191. These are **not integrated joules**, lack CPU energy and must never be used with 44K-doc or six-stream speed measured in another benchmark. The saturating XTX-03 CSV confirms that without an SCLK cap, -125 mV raised clock by ~247 MHz and power by 16 W; a clock cap allowed a voltage reduction. **Offset alone remains withdrawn.** The new `xtx-cap2700-uv125` name is candidate-only; 40-minute soak, output identity, host amdgpu log check, per-device stable ABBA and safe restore remain **pending**.

Next: evaluate XTX OD_MCLK within its reported 97–1250 MHz supported range (per-BDF re-probe; memory-bound decode), and observe then test pp_power_profile_mode BOOTUP_DEFAULT/COMPUTE/POWER_SAVING only with rollback coverage. R9700 210 W minimum-cap trial is still separate, not executed by this XTX sweep. No new XTX/R9700/6900 XT setting is production-validated.
