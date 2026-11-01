---
id: PWR05
order: 0
plan: run-power-efficiency
state: pending
created-at: '2026-11-01T11:26:28.930896+00:00'
breadth: ''
skill: intermediate
created-by: agent
work: S
priority: P2
---

# Idle and load power of the flashnext-4gpu default: tail, engine, CPU threads; power defaults

## Description

RAD20 brought an idle XTX helper from ~95 W to 6-12 W (R3_MOE_WORKER_POWER_SAVE, default on) and PWR01-PWR04 found a -125 mV XTX offset saves ~24 W a card under load at no speed cost (opt-in, --gpu-power). Not measured: the RX 6900 XT tail and the R9700 engine idle in the four-card default, helper 0's combine pool and the helpers' CPU threads at idle (any spinning), and whether the measured voltage offsets should be the launcher default.

## Steps

1. Idle: start flashnext-4gpu, one warm-up request, 60 s idle; per-card power (rocm-smi) and per-process/thread CPU (top -H); runs/pwr-idle4.
2. Any card above ~15 W or any thread above ~2% CPU at idle: find the wait that keeps it busy (pending AQL wait, polling loop) with the RAD20 method.
3. Load: 42K probe and 8-session run with --gpu-power (XTX -125 mV) against without: wall power per card and tok/s.
4. Decide defaults: --gpu-power on by default if speed and output are unchanged; record in the model page.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Idle table per card and process; every idle consumer above the thresholds explained or fixed; load A/B with identical output and speed within noise.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Measurements need the gpu lease; idle check is queued after RHP10 (runs/pwr-idle4).

## Change Log

- 2026-11-01T11:26:28.930896+00:00 (created-by): Created by agent

## Idle result (2026-11-01, `runs/pwr-idle4`)

flashnext-4gpu loaded, after one warm-up request, 60 s idle: XTX 5-6 W each, 6900 XT 6 W, R9700 20 W (17 W before the
server started); engine 1.1% CPU (48 threads), helpers 0.0%. Only the R9700 is above ~15 W, and mostly at its own floor:
the open checks are PCIe ASPM (policy `default`; `powersupersave` lets idle links sleep) and its power profile
(BOOTUP_DEFAULT; POWER_SAVING exists), both root-only. Review: report joules per token, qualify the XTX -125 mV offset per
card (-75/-100/-125 mV, stress, errors) before any default; audit queues for armed waits at idle.
