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

## Offline replay of the policy, 2026-11-05 (slim harness for steps 3-4)

r3_policy_replay (engines/radiance/components/moe-remote/worker_policy_replay.h, worker_policy_replay_main.cpp, test worker_policy_replay_test.cpp) feeds a recorded SERIES of the engine's cumulative route-count snapshots through the driver's own heat accounting and planner (heat::Heat, heat::propose, DriftTracker, marginal_relative_error), applying each proposed transaction at once. It runs in seconds with no GPU and no server, and prints per snapshot the tail's heat share before and after, the experts moved, and at the end transactions, bytes moved, the hot-set drift half-life and the marginal sampling error. Every setting is an explicit argument (no defaults). It does NOT model transaction time and its disturbance of service, aborts, the effect gate or eligibility learned from refused destinations, so it answers what the plan does to the placement, not what it costs; the cost side still needs a serve run. tools/lab/radiance/heat-placement/record_heat_series.py copies the live route-count file (PREFIX.rank0, rewritten atomically by --record-heat) to numbered snapshots whenever it changes, so a run leaves the whole trajectory; it is the input of the replay. placement_sim.py in the same folder is an older simulator with its own cost-aware policy; the replay is the one that runs the real planner. Card-free tests: convergence off the tail, fixed point, determinism, a hot set that shifts twice (followed, drift measured), residency holding just-moved experts, refusal of bad input, a counter that goes back. Next: record the series in the next hardware runs and use the replay to try the step 3 derivations (cadence, move budget, residency, half-life, margin) against real drift, then confirm on hardware.

## Step 2 implemented, 2026-11-05 (measurement hooks, no derivation yet)

worker_policy_measure.h (pure, no clock or thread): Series (bounded, exact nearest-rank quantiles), Throughput (bytes over wall time), topk_overlap, DriftTracker (ring of heat snapshots; overlap of the k hottest experts of the newest snapshot against each older one; the drift half-life is the lag at which the overlap first falls to one half, interpolated; resets when the source counter goes back or the shape changes), relative_share_error and marginal_relative_error (multinomial sampling error of the k-th hottest expert). The driver feeds them where it already observes: a closed transaction (committed: wall time and bytes moved over that time; aborted: abort-to-quiescent time), a consumed heat delta (gap between deltas, the drift tracker, the marginal error), and restarts the drift measurement on a heat baseline or a stale source. k is the number of experts the tail helper holds (its budget over the expert stride). Evidence adds one line: `policy measure transactions= txn_p50_ns= txn_p95_ns= txn_max_ns= copy_bytes_per_ns= aborts= abort_p95_ns= delta_gaps= delta_gap_p99_ns= delta_gap_max_ns= hot_set_k= drift_half_life_observations= marginal_relative_error=`; Driver::measured() returns a copy for tests. The effect gate's noise floor was already in the evidence (effect_noise_floor), so step 2e needed no new code. Card-free tests: worker_policy_measure_test (quantiles, bounds, non-finite samples, overlap with ties, a synthetic sliding hot window whose half-life is exactly 200 observations, discontinuities, sampling error) and assertions in worker_policy_driver_test (two committed transactions timed, an abort timed as an abort with no measured bytes, update gaps recorded, hot-set size 2 for a budget of 20 over a stride of 10). NOT done: the derivations themselves (step 3), the calibration store (step 4), observe-only until calibrated, and a hardware run that reads the new evidence line under the workload shifts to see real values (next).

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
- 2026-11-05T01:58:26.403837+00:00 (updated-by): Updated: section:notes
- chg_20261105_015826_the-live-expert-placement-poli_7498
- 2026-11-05T01:58:30.239165+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-05T02:46:46.794753+00:00 (updated-by): Updated: section:notes
- chg_20261105_024646_added-a-fast-offline-way-to-te_8266
- 2026-11-05T02:46:50.926414+00:00 (updated-by): Updated: section:ledger-events
