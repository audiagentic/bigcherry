---
id: RHP16
order: 0
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-11-03T00:02:23.382624+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Calibrate live-placement observability and move triggers from measured quantities (no hand-picked policy constants)

## Description

worker_heat.h's placement policy (slice B) and the slice D live-move loop are governed by literals that break the no-hand-picked-constants rule: half_life_observations 1000, margin 0.10, max_moves 8, max_moved_bytes 256 MiB, min_residency 30 min, max_heat_age 5 min, min_samples 0, Device.cost 1.0 ('calibrated curves come later'), Device.layer_cap 0, and the epoch cadence (R3_MOE_WORKER_EPOCH_CYCLE_S). Each must be derived from a measured quantity, re-derived when the layout or workload changes, and recorded with its evidence. Observability exists: RHP02 route histograms with R3_MOE_WORKER_HISTOGRAM_PERIOD_S snapshots (windowed heat), R3_MOE_WORKER_TRACE per-request publish/wait/helper-GPU times (the cost signal), the slice C admission ledger high-water and drain timings, and radiance's own local HeatEngine line (64 moves/step, decay 0.95, hysteresis 1.25, min-gain 1.5, credit cap 512), which is a separate lever, coordinated rather than duplicated. Starts after slice D (RHP11 step 3 / RHP06) can move experts live; until then calibration would tune a model, not the system.

## Steps

1. Device.cost: per-helper expert cost curves (us per request against rows and routes) from R3_MOE_WORKER_TRACE helper_gpu_us across the three helper cards; replaces the 1.0 literal.
2. max_moved_bytes and cooldown: measured helper residency copy bandwidth over host memory (no P2P) and the measured epoch drain/commit time from slice C cycles, so a transaction costs a stated fraction of throughput.
3. min_residency: break-even, i.e. measured move cost divided by predicted saving per second.
4. half_life and max_heat_age: heat-distribution drift measured from histogram snapshots across workload mixes (decode-heavy, long-context, multi-stream, MTP on).
5. margin and min_samples: from the measured noise floor of predicted peak layer load across repeated identical windows.
6. A calibration store entry per layout (like serving calibration), consumed by the helper policy; literals deleted.
7. Coordinate with radiance's engine-local HeatEngine: document which lever owns which tier, and prevent ping-pong between engine-local promotion and helper placement.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

End-to-end A/B on Flash-Next (R9700 engine, two 7900 XTX helpers, 6900 XT tail), decode and prefill reported separately, three workload mixes: (a) slice D with calibrated triggers vs static heat placement (R3_MOE_WORKER_HEAT); (b) the moved-bytes and drain-time budget holds (throughput dip during a transaction within the derived bound); (c) no oscillation over a 2 h mixed soak (moves per hour stay bounded, no expert moved back within min_residency). Identical text in deterministic mode. Every derived value is recorded with the run that produced it.

## Effort & Risk



## Standards



## Acceptance Criteria

Slice D with calibrated triggers beats static placement end to end on at least the decode-heavy and multi-stream mixes, by more than slice C's measured admission cost (the justification for slices C and D), with no regression on any mix; no policy literal remains in worker_heat.h; each value has a derivation and evidence file.

## Notes

Depends on RHP11 slice D (live moves) and RHP06. Follows the owner's direction: live placement, no restarts, auto-calibrate by default, no regressions.

## Change Log

- 2026-11-03T00:02:23.382624+00:00 (created-by): Created by agent
