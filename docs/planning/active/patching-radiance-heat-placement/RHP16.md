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

1. (done by slice J, 2026-11-04) Device.cost and Device.layer_cap, and R3_MOE_POLICY_DEVICE_COSTS / LAYER_CAPS, no longer exist: the live plan is the static placer's plan for the live heat, whose head/tail order already encodes card speed. The stage-1 literals half_life 1000, margin 0.10, max_moves 8, 256 MiB, 30 min dwell and 5 min age are gone from code; the policy now takes 14 declared settings with NO defaults (see the table in the notes) and only experiment values were used on hardware.
2. Add the measurement hooks the derivations need, in the driver and the existing epoch cycle, not a new collector: (a) per-transaction wall time and bytes (already in the evidence as transition_ns and bytes_moved: expose p50/p95/max and bytes per ns per helper); (b) abort-to-quiescent time from the slice C cycle; (c) heat source delta interval history (max and p99 gap between deltas) and observation rate; (d) top-K stability across consecutive windows (overlap of the K hottest experts, K = tail capacity in experts), which is a measured drift rate; (e) the lane-service noise estimate the effect gate already builds from its quiet windows, exported as a number; (f) the per-expert share's relative sampling error from the observation counts.
3. Derive, each from its stated measurement, re-derive when the layout, budgets or workload class change, and record value plus evidence in the run log: see the table. Nothing is derived from a literal; where an objective is unavoidable (how much a transaction may disturb service) it is a declared objective expressed in measured units, never a raw number of seconds or bytes.
4. Calibration store: one entry per layout (as serving calibration, RSC03), written by the driver after a settle period, consumed at start; an explicit R3_MOE_POLICY_* setting overrides a derived value and is printed as an override; with no store entry the policy runs in observe-only mode (evaluates, proposes nothing) until it has calibrated, so default-on never moves experts on an uncalibrated guess (owner decision: auto-calibrate by default, no regressions).
5. Delete the settings that become derived or event-driven (MAX_MOVES from bytes over stride; LEARNED_ELIGIBILITY_TTL_NS replaced by invalidation on any ledger or layout change); keep as overrides only those with an operator reason. No compatibility defaults.
6. Coordinate with radiance's engine-local HeatEngine (64 moves/step, decay 0.95, hysteresis 1.25, min-gain 1.5, credit cap 512): document which lever owns which tier and test that the two do not ping-pong (RHP13 for the single allocator).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Card-free tests: each derivation from a recorded series of measurements (golden series from policy-live-5 to policy-live-8 evidence), re-derivation on a layout change, override printing, observe-only before calibration, deleted settings absent. Hardware, decode and prefill reported separately, on Flash-Next with the four-card layout: (a) the workload shift in both orders (filler then docs, docs then filler) with derived settings reaches the same end state as the experiment settings (filler TTFT 13.3 s, documentation prefill within 2% of its start, no aborts); (b) a transaction's step-time disturbance stays inside the measured noise floor; (c) a 2 h mixed soak: moves per hour bounded, no expert moved back within the derived residency, decode step time equal to policy off; (d) KLD and top-1 against BF16 within the gate. Every derived value is recorded with the run that produced it.

## Effort & Risk



## Standards



## Acceptance Criteria

The policy starts with no hand-picked policy number: every trigger, size and age is derived from a measurement recorded in a per-layout calibration store or is an explicit operator override that is printed; before calibration it observes only; the experiment-value runs are reproduced by derived values within the validation bounds.

## Notes

Depends on RHP11 slice D (live moves) and RHP06. Follows the owner's direction: live placement, no restarts, auto-calibrate by default, no regressions.

## Derivation table for the 14 declared policy settings (2026-11-05, after slices H, I, J)

| Setting | Derived from | Measured by |
|---|---|---|
| R3_MOE_POLICY_CADENCE_NS | at least the p95 transaction wall time plus one effect window, so a proposal never overlaps the previous one's settling | step 2a, 2e |
| R3_MOE_POLICY_MAX_MOVED_BYTES | host-to-VRAM copy bytes per ns of the slowest helper (from past transitions) times the declared objective expressed as a number of effect windows a transaction may last | 2a, 2e |
| R3_MOE_POLICY_MAX_MOVES | MAX_MOVED_BYTES divided by the expert stride: no separate setting | stride known to the driver |
| R3_MOE_POLICY_MIN_RESIDENCY_NS | break-even: measured move cost (bytes over bandwidth plus the lane time lost) divided by the predicted saving per second (change of the tail's heat share times the tail lane's service rate) | 2a, 2e |
| R3_MOE_POLICY_HEAT_HALF_LIFE_OBSERVATIONS | the observation count at which the overlap of the K hottest experts of consecutive windows falls to one half (a drift half-life by definition), K = tail capacity | 2d |
| R3_MOE_POLICY_MAX_HEAT_AGE_NS | the largest gap actually seen between source updates over the calibration window (the heat file's own update interval), so a stale source is one that missed its own rhythm | 2c |
| R3_MOE_POLICY_MIN_SAMPLES | observations per expert at which the relative sampling error of its share is below MARGIN (a statistic of MARGIN, not a number) | 2f |
| R3_MOE_POLICY_MARGIN | the largest relative change of the tail heat share between consecutive windows with no move (the score's own noise floor) | 2d, 2e |
| R3_MOE_POLICY_ABORT_COOLDOWN_NS | the measured abort-to-quiescent time of the epoch cycle | 2b |
| R3_MOE_POLICY_LEARNED_ELIGIBILITY_TTL_NS | no time at all: a learned ineligibility is invalidated by any ledger, budget or layout change (an event) | driver |
| R3_MOE_POLICY_EFFECT_WINDOW_SAMPLES | requests per lane class at which the noise estimate of the lane service rate stops changing (variance convergence) | 2e |
| R3_MOE_POLICY_EFFECT_MIN_QUIET_WINDOWS | quiet windows until the noise estimate's confidence interval is inside MARGIN | 2e |
| R3_MOE_POLICY_EFFECT_NOISE_MULTIPLIER | replaced by a declared false-revert rate and the empirical noise quantile it implies, not a factor | 2e |
| R3_MOE_POLICY_EFFECT_REVERT_TTL_NS | the heat drift half-life (a worse destination is blacklisted for as long as the workload stays the same) | 2d |

The values used on hardware in policy-live-5 to policy-live-8 (for comparison with what the derivations produce, not as defaults): cadence 5 s, margin 0.005, 128 moves, 2 GiB, residency 30 s, half-life 20,000,000 observations, heat age 10 s, min samples 3, abort cooldown 30 s, eligibility TTL 600 s, effect window 32, 3 quiet windows, noise multiplier 3.0, revert TTL 600 s. Measured so far from those runs: transactions of 0.42 s average (15 in 6.3 s, 4.87 GB) in the churn gate and one of 3.0 s for 162 MB; 8 and 9 committed with no aborts in the repeat and reverse shifts.

PROTECTED_LAYERS and ACCEPT_UNWATERMARKED_HEAT are not tunables: protected layers are a model fact to be read from the container metadata (RCS12 guard owns hardcoded model facts) and the unwatermarked opt-in goes away when the heat source carries completeness (RHP01 found the existing feed complete in the production layout; RHP02 owns the single feed).

## Change Log

- 2026-11-03T00:02:23.382624+00:00 (created-by): Created by agent
- 2026-11-05T01:35:06.283737+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria
- 2026-11-05T01:35:26.223649+00:00 (updated-by): Updated: section:notes

## Ledger-events

- chg_20261105_013526_planned-how-the-live-expert-pl_9955
- 2026-11-05T01:35:30.040386+00:00 (updated-by): Updated: section:ledger-events
