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

Current scope (takes precedence over earlier breadth): 2026-11-27: bound to matched stock/candidate idle/decode/prefill system measurement within PWR16, synchronized GPU and CPU package energy, wall meter if available, label component estimates and unavailable sensors. Record throughput/latency, clocks, temperature, energy/token. No host-wide writes or duplicate sweep.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Per-device, per-phase table recorded on the model page and in the research record.

## Effort & Risk



## Standards



## Acceptance Criteria

**Transferred PWR05 acceptance (independent gate):** PWR05 four-card baseline acceptance: retain actual measured flashnext-4gpu loaded idle after one warm-up/60 seconds: gfx1100 XTX helpers 5-6 W each, gfx1030 tail 6 W, gfx1201 engine 20 W (~17 W pre-process), engine 1.1% CPU across 48 threads and helpers 0%. This is historical one-run evidence, not a current-pin or multi-repeat guarantee. Verify current same-build/driver/BDF, idle all devices and per-thread CPU under a lease, explain any >15 W GPU or >2% CPU consumer with queue/ASPM wait evidence; PWR06 alone owns the GFX12 R9700 GFXOFF/driver/kernel causal mitigation, no second driver patch. Measure both GPU board sensors and CPU RAPL for matched single-stream decode, long prefill, 42K and 8-session phases, with explicit sensor identity/cadence/phase boundaries, output identity and W/J per token; do not merge cross-date single-stream and 6-stream numbers or infer J/token from one sample. Any root-only ASPM/power mode change is a separate owner-authorised experiment, not a default.

Time-aligned sensor samples and actual tokens, integrate W*dt per phase; RAPL counter wrap and missing/unavailable explicit; report GPU, CPU+GPU and wall energy separately; no double count. Current dense and MoE baselines, replicated controls and dispersion, immutable run refs and receipts. Preserve previous transferred acceptance.

Current acceptance scope: 2026-11-27: bound to matched stock/candidate idle/decode/prefill system measurement within PWR16, synchronized GPU and CPU package energy, wall meter if available, label component estimates and unavailable sensors. Record throughput/latency, clocks, temperature, energy/token. No host-wide writes or duplicate sweep.

## Notes

Measurement only, no speed risk. RX 6900 XT runs an unrelated service; record it but out of scope for tuning.

2026-10-27: host change on the build/bench host (owner-approved, test machine): udev rule /etc/udev/rules.d/99-rapl-energy-read.rules makes /sys/class/powercap/intel-rapl*/energy_uj readable by the video group (read-only), so CPU package power is measured without root. The start-up host check should warn when it is missing (owner rule: check the host and warn). Harness tools/lab/radiance/pwr01-baseline.sh (branch feat/pwr01-power-baseline): every inventory GPU via amd-smi + RAPL package power per phase, J/token for decode/prefill.

CONSOLIDATION 2026-11-22 POWER-BASELINE: PWR05's unique acceptance is retained under this established owner; no hardware changes or runtime activation are implied.

2026-11-25 add dense TP3 2020.11.4 and current-build stock/controlled phases to baseline. Old four-GPU MoE results are separate cohorts. Owner trial power windows mix decode/prefill; do not divide by one phase throughput to infer energy. Measure per BDF, CPU RAPL and wall meter if available; phase markers, cache/warmup/thermal conditions, actual settings and sensor identity. Record unloaded, HIP-resident, loaded idle and post-request idle without intrusive polling, all cards including unused 6900 (observation only). PWR10 references this protocol.

2026-11-26 roadmap reconciliation at 165d6f724: External reference: Sweet-Spot Tuner v3.5.15, commit a2a31ce63cc1d40af72997de621115eaa2dcd54d (https://gitlab.com/spawnitems-group/sweetspot-tuner/-/tree/v3.5.15); see docs/reference/power-tuning-evidence-2026-11.md, hot AI qualification section. For clock/voltage comparisons record stock before and after, same workload, warmed thermal envelope/normal fan policy, hotspot and memory temperature if readable, actual clocks and throttle; integrate synchronized GPU+CPU energy by phase and token count. Include cold-to-boost and warm idle-to-burst transitions, not just sustained GPU saturation. Do not force RX9070-specific 70–90 C range or assume 94 C limit for all boards; use supported device limits, observed safe operating envelope, and explicit bounded soak objective; thermal stop or missing required telemetry is inconclusive, never instability/pass. No extra matrix for already qualified stock and cap-only.

2026-11-27: bound to matched stock/candidate idle/decode/prefill system measurement within PWR16, synchronized GPU and CPU package energy, wall meter if available, label component estimates and unavailable sensors. Record throughput/latency, clocks, temperature, energy/token. No host-wide writes or duplicate sweep.

2026-11-27 read-only CPU capability preflight: Intel package energy domains absent in the execution environment; existing probe reports unsupported/no_energy_counters. RAPL energy and range files not exposed. Require host-side readable package counters or equivalent synchronized instrumentation before claiming system energy acceptance. No host configuration change. GPU held by unrelated serving workload.

## Change Log

- 2026-10-27T02:15:39.545328+00:00 (created-by): Created by agent
- 2026-10-27T06:03:16.386794+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4549
- 2026-11-22T11:48:53.486970+00:00 (updated-by): Updated: section:notes, section:acceptance_criteria
- RV4582
- 2026-11-25T01:05:11.696313+00:00 (updated-by): Updated: section:notes, section:acceptance_criteria
- 2026-11-25T19:01:36.788268+00:00 (updated-by): Updated: section:notes
- RV4683
- 2026-11-26T23:13:06.718785+00:00 (updated-by): Updated: section:notes, section:steps, section:acceptance_criteria
- 2026-11-27T00:36:03.069944+00:00 (updated-by): Updated: section:notes
