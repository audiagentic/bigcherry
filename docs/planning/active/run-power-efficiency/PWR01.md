---
id: PWR01
order: 0
plan: run-power-efficiency
state: pending
created-at: '2026-10-27T02:15:39.545328+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: S
---

# Whole-system power baseline: every GPU and CPU package at idle, decode and prefill (incl. R9700 engine idle)

## Description

Only the XTX helpers' idle power has been gated (RAD20, 6-7 W). Measure where the watts go across the serving host: board power of each GPU (amd-smi) and CPU package energy (RAPL/amd_energy) at idle, single-stream decode and long prefill, in the production Flash-Next layout. Includes the R9700 engine's idle power, never measured; if high, check the same two causes as the XTX (pending AQL wait, more than num_cp_queues compute queues).

## Steps

1. Extend burst-idle.sh (or a sibling) to sample every inventory GPU plus CPU package energy per phase; devices from env settings.
2. One run; record watts per device per phase and joules per token for decode/prefill.
3. If the R9700 idles high, profile its queues/waits like RAD20.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Per-device, per-phase table recorded on the model page and in the research record.

## Effort & Risk



## Standards



## Acceptance Criteria

**Transferred PWR05 acceptance (independent gate):** PWR05 four-card baseline acceptance: retain actual measured flashnext-4gpu loaded idle after one warm-up/60 seconds: gfx1100 XTX helpers 5-6 W each, gfx1030 tail 6 W, gfx1201 engine 20 W (~17 W pre-process), engine 1.1% CPU across 48 threads and helpers 0%. This is historical one-run evidence, not a current-pin or multi-repeat guarantee. Verify current same-build/driver/BDF, idle all devices and per-thread CPU under a lease, explain any >15 W GPU or >2% CPU consumer with queue/ASPM wait evidence; PWR06 alone owns the GFX12 R9700 GFXOFF/driver/kernel causal mitigation, no second driver patch. Measure both GPU board sensors and CPU RAPL for matched single-stream decode, long prefill, 42K and 8-session phases, with explicit sensor identity/cadence/phase boundaries, output identity and W/J per token; do not merge cross-date single-stream and 6-stream numbers or infer J/token from one sample. Any root-only ASPM/power mode change is a separate owner-authorised experiment, not a default.

## Notes

Measurement only, no speed risk. RX 6900 XT runs an unrelated service; record it but out of scope for tuning.

2026-10-27: host change on the build/bench host (owner-approved, test machine): udev rule /etc/udev/rules.d/99-rapl-energy-read.rules makes /sys/class/powercap/intel-rapl*/energy_uj readable by the video group (read-only), so CPU package power is measured without root. The start-up host check should warn when it is missing (owner rule: check the host and warn). Harness tools/lab/radiance/pwr01-baseline.sh (branch feat/pwr01-power-baseline): every inventory GPU via amd-smi + RAPL package power per phase, J/token for decode/prefill.

CONSOLIDATION 2026-11-22 POWER-BASELINE: PWR05's unique acceptance is retained under this established owner; no hardware changes or runtime activation are implied.

## Change Log

- 2026-10-27T02:15:39.545328+00:00 (created-by): Created by agent
- 2026-10-27T06:03:16.386794+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4549
- 2026-11-22T11:48:53.486970+00:00 (updated-by): Updated: section:notes, section:acceptance_criteria
