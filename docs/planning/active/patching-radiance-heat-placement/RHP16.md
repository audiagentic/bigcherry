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
## First measured values and a measurement flaw they exposed, 2026-11-05 (policy-live-9)

Run policy-live-9: the merged policy with the measurement hooks, effect gate on, Flash-Next four GPUs, eight 27K filler prefills then ten ~50K documentation prefills. The shift result reproduced again (filler TTFT 19.1 to 13.3 s, documentation prefill 3,705 to 3,668 tok/s, no aborts, decode 120 tok/s at one stream and 242 at six). The evidence line `policy measure` at the end of the run:

| Measured | Value | Feeds |
|---|---|---|
| transactions | 10 committed, 0 aborted | |
| transaction wall time | p50 1.77 s, p95 2.72 s, max 2.72 s | cadence, max moved bytes |
| copy bandwidth (bytes moved over transaction wall time) | 0.16 to 0.20 GB/s over the run | max moved bytes, residency |
| gap between heat deltas | 5.04 s (p99 = max), the policy's own cadence | max heat age |
| hot set | k = 5,931 experts (the tail's budget over the stride) | |
| marginal expert's relative sampling error | 5.2% after the first window, 2.3% at the end | min samples, margin |
| abort-to-quiescent | none seen | abort cooldown |

Two things follow. (1) A transaction moves about 330 MB on average (2.6 GB in 8) and takes 1.8 s, so the achieved bandwidth is 0.17 GB/s: far below the link, because it includes the epoch protocol (reserve, copy, quiesce, commit), not the copy alone. The bytes budget must be derived from this achieved rate, not from link speed. (2) The drift half-life read 0 (not measured) at every sample but one, where it read 57,085,762 observations during the filler-to-docs shift. That is a MEASUREMENT FLAW, found by this run: the tracker was fed the accumulated heat, which is decayed by the very half-life setting the measurement is meant to inform (the experiment value was 20,000,000 observations), so it measured the setting and could barely see the workload. Fixed in the same change that records this note: the tracker now takes the counts of ONE source update interval (never accumulated or decayed), compares the hot sets of the newest interval with each older one, and takes the half-life relative to the overlap at the nearest lag (two adjacent intervals of the same traffic never match perfectly), and the evidence line gains drift_horizon_observations so a 0 means 'not seen within this horizon' and not 'zero'. The 57M reading and the zeros of policy-live-9 are therefore NOT valid drift values and must not be used to derive the half-life; the next run measures it properly. The offline replay (PR 620) needs the same change (it also fed decayed heat) and gets it once both are in.

## Step 2 implemented, 2026-11-05 (measurement hooks, no derivation yet)

worker_policy_measure.h (pure, no clock or thread): Series (bounded, exact nearest-rank quantiles), Throughput (bytes over wall time), topk_overlap, DriftTracker (ring of heat snapshots; overlap of the k hottest experts of the newest snapshot against each older one; the drift half-life is the lag at which the overlap first falls to one half, interpolated; resets when the source counter goes back or the shape changes), relative_share_error and marginal_relative_error (multinomial sampling error of the k-th hottest expert). The driver feeds them where it already observes: a closed transaction (committed: wall time and bytes moved over that time; aborted: abort-to-quiescent time), a consumed heat delta (gap between deltas, the drift tracker, the marginal error), and restarts the drift measurement on a heat baseline or a stale source. k is the number of experts the tail helper holds (its budget over the expert stride). Evidence adds one line: `policy measure transactions= txn_p50_ns= txn_p95_ns= txn_max_ns= copy_bytes_per_ns= aborts= abort_p95_ns= delta_gaps= delta_gap_p99_ns= delta_gap_max_ns= hot_set_k= drift_half_life_observations= marginal_relative_error=`; Driver::measured() returns a copy for tests. The effect gate's noise floor was already in the evidence (effect_noise_floor), so step 2e needed no new code. Card-free tests: worker_policy_measure_test (quantiles, bounds, non-finite samples, overlap with ties, a synthetic sliding hot window whose half-life is exactly 200 observations, discontinuities, sampling error) and assertions in worker_policy_driver_test (two committed transactions timed, an abort timed as an abort with no measured bytes, update gaps recorded, hot-set size 2 for a budget of 20 over a stride of 10). NOT done: the derivations themselves (step 3), the calibration store (step 4), observe-only until calibrated, and a hardware run that reads the new evidence line under the workload shifts to see real values (next).

## Derivation table for the 14 declared policy settings (2026-11-05, after slices H, I, J)

| Setting | Derived from | Measured by |
|---|---|---|
| R3_MOE_POLICY_CADENCE_NS | p95 transaction wall time plus k measured effect windows, so quiet comparisons complete before the next poll | step 2a, 2e |
| R3_MOE_POLICY_MAX_MOVED_BYTES | host-to-VRAM copy bytes per ns of the slowest helper (from past transitions) times the declared objective expressed as a number of effect windows a transaction may last | 2a, 2e |
| R3_MOE_POLICY_MAX_MOVES | MAX_MOVED_BYTES divided by the expert stride: no separate setting | stride known to the driver |
| R3_MOE_POLICY_MIN_RESIDENCY_NS | break-even: measured move cost (bytes over bandwidth plus the lane time lost) divided by the predicted saving per second (change of the tail's heat share times the tail lane's service rate) | 2a, 2e |
| R3_MOE_POLICY_HEAT_HALF_LIFE_OBSERVATIONS | the observation count at which the overlap of the K hottest experts of consecutive windows falls to one half (a drift half-life by definition), K = tail capacity | 2d |
| R3_MOE_POLICY_MAX_HEAT_AGE_NS | the largest gap actually seen between source updates over the calibration window (the heat file's own update interval), so a stale source is one that missed its own rhythm | 2c |
| R3_MOE_POLICY_MIN_SAMPLES | observations per expert at which the relative sampling error of its share is below MARGIN (a statistic of MARGIN, not a number) | 2f |
| R3_MOE_POLICY_MARGIN | the largest relative change of the tail heat share between consecutive windows with no move (the score's own noise floor) | 2d, 2e |
| R3_MOE_POLICY_ABORT_COOLDOWN_NS | the measured abort-to-quiescent time of the epoch cycle | 2b |
| R3_MOE_POLICY_LEARNED_ELIGIBILITY_TTL_NS | no time at all: a learned ineligibility is invalidated by any ledger, budget or layout change (an event) | driver |
| R3_MOE_POLICY_EFFECT_WINDOW_SAMPLES | smallest measured tail window L with observed RMS fractional noise(L) × z(α) ≤ predicted E / tail_lane_row_ns | published-tail row-window noise ladder and objective α |
| R3_MOE_POLICY_EFFECT_MIN_QUIET_WINDOWS | smallest positive k with transaction_effect_windows × p(n)^k ≤ α for measured-window noise-only crossing chance p(L), assuming independent comparisons (not independent rows) | 2e, declared transaction windows and α |
| R3_MOE_POLICY_EFFECT_NOISE_MULTIPLIER | retained locked-v1 multiplier: observed max absolute quiet delta / observed median absolute quiet delta; conservative envelope, not an objective-specific quantile | 2e |
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
- 2026-11-05T02:50:13.944855+00:00 (updated-by): Updated: section:notes
- chg_20261105_025013_fixed-how-the-placement-policy_9631
- 2026-11-05T02:50:17.774448+00:00 (updated-by): Updated: section:ledger-events

## Offline sweep of the policy settings over a recorded series, 2026-11-05 (replay of policy-live-10)

162 combinations in 48 s on one workload (the filler-then-docs shift, 275 snapshots): max moves per transaction {32, 128, 512} x residency {0, 30, 120 s} x half-life {2M, 20M, 200M observations} x margin {0.005, 0.02, 0.05} x cadence {5, 10} snapshots, with 2 GiB per transaction, minimum samples 3. Score = mean and final share of the heat on the tail helper (lower is closer to the static plan) and experts moved. The replay applies each transaction at once and models neither its cost nor the effect gate, so it ranks settings by what the plan does, not by what it costs.

| Setting | Values | Mean tail share | Experts moved |
|---|---|---:|---:|
| max moves | 32 / 128 / 512 | 0.115 / 0.104 / 0.094 | 148 / 1,348 / 11,425 |
| margin | 0.005 / 0.02 / 0.05 | 0.1035 / 0.1033 / 0.1065 | 6,728 / 4,269 / 1,925 |
| cadence (snapshots) | 5 / 10 | 0.1035 / 0.1054 | 5,589 / 3,026 |
| half-life (observations) | 2M / 20M / 200M | 0.0990 / 0.0975 / 0.1168 (final 0.116 / 0.058 / 0.064) | 4,923 / 4,481 / 3,519 |
| residency (s) | 0 / 30 / 120 | 0.1087 / 0.1006 / 0.1040 | 3,394 / 3,848 / 5,680 |

Reading. (1) The move budget per transaction is the lever that costs: 4x the moves buys about 10% less tail share, 8x moves for 18%. (2) Margin is nearly free to raise: 0.005 to 0.05 moves 71% fewer experts for 3% more tail share; the measured marginal sampling error of the run was 2.3%, and a margin near it (0.02) sits at the knee, which supports deriving margin from the sampling error (RHP16 table) and says the experiment value 0.005 was needlessly low. (3) Cadence 5 to 10 s halves the moves for 2% more tail share. (4) A half-life of 20M observations gives the lowest final tail share (0.058, against 0.116 at 2M, which chases noise, and 0.064 at 200M); the measured drift scale in the same run was 59M to 65M observations during the shift, so the experiment value was within about 3x of it. (5) Residency is not monotone: 120 s moved more experts than 30 s (5,680 against 3,848) because experts held back were moved in larger groups later; 0 s was worst on tail share.
Caveats: one workload, one recorded run, the gate and transaction cost not modelled. It is a screening tool: it tells which settings are worth deriving first (margin, the move budget, cadence) and which are second order here (residency), and the derived values must be confirmed on hardware.

Replay result on the recorded series of policy-live-10 (the run with the fixed drift tracker; 275 snapshots, the run's own settings: 128 moves, 2 GiB, 30 s residency, margin 0.005, half-life 20M, cadence 5): 33 transactions, 4,219 experts (10.7 GB) moved, the tail's heat share 19.2% to 5.8%, in 0.35 s. The hardware run committed 8 transactions and moved about 2.6 GB with the effect gate on, so the gate is what holds the real policy back; the replay does not model it. The replay now feeds the drift tracker per-interval counts, as the driver does.

## Steps 3-4 now sit on the manifest, not the helper (2026-11-05, after contract amendment 10 / PR #633)

The helper process no longer reads `R3_MOE_POLICY_*`: contract amendment 10 makes the live policy typed manifest fields (`/spec/handoff/placement/live_policy`: enabled, effect_gate, accept_unwatermarked_heat, the eleven integers, the three scaled values as ppm/milli integers, protected_layers[]), every member required, no defaults, and the helper loads them through the shared reader. Consequences for this item:

- **Derivation happens where the manifest is written.** `bc-serve --write-manifest` resolves `live_policy` from the layout, the detected hardware and the per-layout calibration store (RSC03's resource core), and writes the derived numbers into the manifest with provenance (`derived` or `override`). The driver in the helper never derives anything at start; it only measures (step 2 hooks) and writes calibration records.
- **Calibration store** (step 4): one record per plan-cache key (which includes the layout) in the existing calibration TOML. **Correction to the earlier design:** v1 omits `live_policy` entirely when no complete calibration exists. An omitted policy does *not* run the helper policy thread or its measurement hooks. v1 has no legal extra store-path field; the helper therefore emits one machine-readable log record from an explicit lab live policy, and `bc-serve calibration-ingest` writes the store.
- **Overrides** are an explicit `bc-serve --live-policy FILE --experiment FILE` (lab-only full v1 live-policy JSON and lab experiment JSON), with CLI provenance. Production may only use derived complete records.
- **Deleted settings** (step 5) are deleted from the registry in the next contract version, not hidden: MAX_MOVES (bytes over stride), LEARNED_ELIGIBILITY_TTL_NS (invalidation instead). Because amendment 10 closed v1 for field changes, these deletions are a v2 change and wait for the next contract version; until then they are derived values like the others.
- **Order of work:** #633 merges first; then the derivations (step 3) land in `bc-serve` against the recorded policy-live series and the `r3_policy_replay` harness; then the calibration store (step 4); the helper-side deletions ride the v2 contract.


## How a layout gets calibrated (RHP16 step 4)

The production start does **not** auto-enable the live policy for an uncalibrated layout: in locked v1 the policy is absent, so there is no policy thread to measure anything. Calibrate with the launcher's `--calibrate` lab run: `--live-policy POLICY.json` (a non-empty v1 live-policy object; no `enabled` member, presence is activation) and `--calibration-prompts-a FILE` plus `--calibration-prompts-b FILE` (distinct, representative workload sets, one prompt per line), and no `--experiment`. The launcher passes `bc-serve --calibration-run`, which writes a lab-profile manifest with the explicit policy and no `spec.experiment` (v1 refuses an empty experiment), so the measured plan, and therefore the plan-cache key, is the production one. `calibration-ingest` accepts only that shape: lab profile, explicit live policy, no experiment.

```sh
bash tools/lab/radiance/launch-flashnext-best.sh start \
  --layout flashnext-4gpu --live-policy /path/to/explicit-lab-policy.json \
  --calibrate --calibration-prompts-a /path/to/filler-prompts.txt \
  --calibration-prompts-b /path/to/documentation-prompts.txt
```

This is a *hardware lab procedure*, not part of the card-free verification suite. The launcher's mandatory measured plan check still applies. The helper uses the measurement hooks to wait for the retained hot-set drift horizon and accepted heat-delta observations. It then emits exactly one `RHP16_CALIBRATION_JSON {...}` line in helper0's log. Choose prompts whose observed routed-expert heat differs (e.g. filler versus documentation); distinct files alone cannot prove that model routing shifts. The launcher plays both workload orders (A then B, B then A), repeating complete cycles until the helper emits the marker. The launcher never accepts a marker before both orders complete. It runs `bc-serve calibration-ingest --from-log HELPER_LOG --calibration STORE --manifest LAB_MANIFEST`, stops its processes, and reports the store location at `$BC_WORK_ROOT/live-policy-calibration.toml`. Ingestion validates the exact v1 manifest key, layout, objective and lab profile, re-resolves the exact launch and checks its measured cached plan, then maps the verified v1 helper key to the distinct RSC03 plan-store key used by production. It locks the cache and writes with a temporary file and atomic rename. It refuses malformed, mismatched and duplicate records; to remeasure a plan, archive and deliberately remove its previous record rather than silently replacing evidence.

The log route is required because v1 has **no manifest field for the calibration store path**. No v1 contract field or fingerprint is modified. Every known measured input appears in the TOML record. Unmeasurable inputs are recorded under `absent = { field = "reason" }` rather than filled with made-up values. The existing reader accepts and preserves partial records, but **production derives no live policy from one**; a genuinely complete set of empirical inputs is required. Step 5 now measures per-helper READY ACK copy rates, sampled large-lane service at RESERVED, quiesce-to-reopen downtime at CLEANED, quiet-window nanosecond noise, and committed tail-share gains. The follow-up removes `noise_false_revert_rate_ppm`, `effect_window_samples` and `effect_quiet_windows` from measured records: one was an objective, and two merely reflected configured gate values. `effect_row_noise_ns` was added as a diagnostic; follow-up 7 derives the gate window from the measured tail-window ladder instead. The renamed `effect_noise_max_ns` states the observed maximum rather than claiming an objective-specific quantile. A record with all required observations can now derive policy; new hardware evidence is still pending. The existing policy-live-9 golden series tests the complete round-trip with its explicitly synthetic supplemental inputs; that test fixture is not production evidence.

For manual ingestion after a lab run, use:

```sh
bc-serve calibration-ingest --from-log /path/to/helper0.log --calibration /path/to/live-policy-calibration.toml --manifest /path/to/verified-lab-manifest.json --plan-cache /path/to/plan-cache --engine-binary /path/to/radiance --site /path/to/plan-site.toml --plan-extra EXAMPLE_PLAN_EXTRA --set KEY=VALUE  # all original --set arguments required
```

For manual ingestion, pass the **same** site, plan-extra, and all --set arguments used by the lab launch; the launcher supplies these automatically. The helper record bears the typed v1 key, and the stored TOML record bears the validated RSC03 plan-store key.

## First hardware calibration, 2026-11-06 (rhp16-calib attempt 6): ingest works, the record is incomplete

Run `rhp16-calib` (lab run folder; Flash-Next, flashnext-4gpu, build `45af79e5`, `--calibrate --record-period 1`, 40 documentation prompts of about 6K tokens, the policy-live-9 policy). The helper converged and emitted `RHP16_CALIBRATION_JSON`; `calibration-ingest` recorded it in `$BC_WORK_ROOT/live-policy-calibration.toml` (plan key `634585fb...`). Attempt 5 had been refused by ingest, which still required the old empty-experiment manifest; fixed on this PR with a card-free test that ingests against a real `--calibration-run` manifest and refuses the production one.

Measured: `heat_gap_max_ns` 5,001,931,591 (the 5 s cadence, not the 1 s record period), `hot_set_k` 5,931, `marginal_error_ppm` 22,678, `marginal_observations` 121,478,390, `observation_rate_per_s` 1,560,435. Absent, 15 of 20, so production derives no policy from this record (`live_policy_manifest.cpp`: incomplete record, no policy) and the planned on/off A/B was stopped before its first arm, because both arms would have served without a policy.

Two causes, which are the step-5 work:
1. **The calibration workload must shift heat.** One kind of prompt gives a stable hot set: the policy proposed nothing (`proposed=0`), so no transaction closed (`txn_p50_ns`, `txn_p95_ns`, `copy_bytes_per_s` and `abort_quiescent_ns` absent) and the drift track never crossed half overlap (`drift_half_life_observations` absent). policy-live-9 committed moves because its workload shifted (filler then documentation). The calibration prompts must contain the shift in both orders.
2. **Measurement hooks the helper does not have:** effect gate quantities in nanoseconds and counts (`effect_noise_floor_ns`, `effect_noise_quantile_ns`, `effect_window_samples`, `effect_quiet_windows`, `effect_rows_per_s`, `noise_false_revert_rate_ppm`), tail-lane service (`tail_lane_service_ns_per_s`, `tail_share_gain_ppm`, `tail_share_noise_ppm`, `lane_lost_ns`) and the slowest-helper copy rate. Each needs a hook where the helper already observes the event (effect gate windows, the tail lane's served requests, the transaction copy), not a new collector, and a card-free test from a recorded series.

Until both land no layout can be calibrated, and the live policy stays lab-only (`--live-policy`).


## Step 5 follow-up 1: empirical noise dispersion (2026-11-06)

The helper records per-row sigma from signed differences of quiet-window
means normalized by `sqrt(1/n_previous + 1/n_current)` and the maximum
absolute quiet-window deviation (`effect_noise_max_ns`). It does **not**
serialize configured gate counts or the false-revert objective as readings.
The first follow-up derived the window from the median quiet difference,
which depends on calibration window size; that threshold was superseded by
follow-up 2. Earlier hardware evidence uses the old schema and is not a
complete current-format record.

## Step 5 follow-up 2: committed-move effect size (2026-11-06)

The tail helper snapshots large-lane service nanoseconds **and**
completed routed-row counts together. The transaction RESERVED ACK conveys
both deltas and elapsed wall nanoseconds without a shared-memory layout change.
The driver records the median positive `service_ns / completed_rows` as
`tail_lane_row_ns` (ns/row); `tail_lane_service_ns_per_s` remains
service ns / wall ns. The settled record requires a valid per-row sample.

Let `G = tail_share_gain_ppm` be the median **predicted relative reduction**
of committed placement plans, `S = tail_lane_row_ns`, and
`E = ceil(S * G / 1000000)` nanoseconds **per routed row**. E is a predicted
effect, not a measured post-move saving; its factors are independent of the
calibration gate's configured window. The strict record removes
`effect_noise_floor_ns` and refuses the old key.

| Derived effect setting | Measured and declared inputs | Formula |
|---|---|---|
| `effect_window_samples` (n) | per-row noise sigma, E, declared false-revert fraction α | smallest n ≥ 1 with `noise_ppm(n) / 1e6 * z(α) <= E / tail_lane_row_ns`, `erfc(z/sqrt(2)) = α` |
| `effect_min_quiet_windows` (k) | E, sigma, derived n, declared opportunities T and α | smallest k ≥ 1 with `T * p^k <= α`; `p = erfc((E/tail_lane_row_ns) / (sqrt(2) * noise_ppm(n)/1e6))` |
| `effect_noise_multiplier_milli` | declared false-revert fraction alpha | `ceil(1000 * z(alpha))`, multiplying the effect gate's fractional quiet noise |

The Gaussian and independent quiet-crossing assumptions, including the
union bound over T, are model assumptions not empirically certified false
reversion rates. Synthetic quiet series with identical per-row sigma
at configured widths 2 and 8 derive the same n and k; hardware evidence
is still required. All settings retain `derived` provenance.

## Watermark follows the measured feed (2026-11-06)

The rhp16-calib2 hardware run produced a COMPLETE record and a derived
production policy, but helper0 reported `evaluated=0`, `no_change=24`,
`delta_gaps=0`: the policy always set `accept_unwatermarked_heat=false`.
The only available production feed is `HistogramFileSource`, which cannot
supply a complete observation/step watermark. Calibration had accepted its
deltas only because the explicit lab policy allowed authenticated histogram
files. Production therefore rejected every delta as `heat_unwatermarked`.

The helper now records `heat_watermarked` over **accepted** heat deltas:
`1` means every accepted delta had `complete_observations`; `0`
means at least one accepted delta lacked it. This is a measured, zero-valid
record field, required for a complete calibration; no accepted deltas means
an explicitly absent field. Serving derives `accept_unwatermarked_heat =
(heat_watermarked == 0)` with derived provenance. The existing helper-side
capability check still limits the allowance to the authenticated engine
histogram-file source; it does not authorize unknown feeds.

**Records carry across pin bumps (2026-11-06, owner direction).** Calibration records are keyed by
`calibration_key`: the plan-key inputs (model, card arch and VRAM, layout, engine arguments and engine settings)
without `engine_build`. What calibration measures belongs to the cards, the layout, the model and the engine
settings, not to the pinned engine commit, so a daily bump keeps the live policy instead of dropping it until a
recalibration. A change of any remaining input, or of the record schema, needs a new calibration.

**RHP02:** a histogram-file record says `heat_watermarked = 0`, so the policy derived from it keeps accepting
unwatermarked heat after RHP02's watermarked feed arrives: as permissive as today, never less safe. Taking the
benefit of the watermark (refusing incoherent snapshots) is a deliberate recalibration once RHP02 ships, not an
automatic invalidation on every engine change.

## Step 5 follow-up 4: planner-aligned calibration (2026-11-06)

Calib5 produced a complete record but a zero-action derived policy. The margin
was dominated by the median workload-shift change (224,359 ppm) instead of the
decayed heat estimator's multinomial sampling error (31,905 ppm). The strict
record now **rejects** `tail_share_noise_ppm`; it is neither sampled nor a
convergence requirement. `margin_ppm = marginal_error_ppm` alone.

An end-to-end card-free test derives the policy from the calib5 readings, passes
the 17 members through the typed helper adapter and verifies the real CPU heat
planner: unchanged heat yields no move; filler-to-docs synthetic heat moves.
It separately verifies all coverage, heat-age, cooldown, residency, move-count
and byte-budget gates, and a quiet effect gate that can become ready and retain
a demonstrably faster lane.

The derivation uses ns versus ns, ppm versus fractional shares, observation
counts versus accepted heat **windows**, and the `half_life_observations_milli`
1000x ABI conversion. The per-transaction copy budget is based on measured
transaction p95, not T future effect windows; the effect multiplier applies the
Gaussian z objective to an **already fractional** quiet noise envelope, not
absolute nanoseconds divided by predicted per-row effect. Age, learned
eligibility and effect-revert TTLs survive cadence, recovery, rollback and
minimum residency. See `engines/radiance/components/serving/live_policy_calibration.md`
for formulas and the separate Follow-up 4 report for all 17 comparisons.

## Step 5 follow-up 5: routed-row effect units (2026-11-06)

The calib6 on-arm hardware result overturned the request-denominator change
from follow-up 4: request sizes can alternate 27,000 and 64 rows, producing
a 70% fractional quiet envelope and hiding a real ~9% placement improvement.
The **current** contract is nanoseconds per **routed row** across the entire
effect pipeline, not per sampled request. `EffectGate::rate`, quiet deltas,
normalised sigma (`effect_row_noise_ns`), observed quiet noise, and the
after/before verdict divide by routed rows. `effect_window_samples` retains
the v1 name but counts rows (`d.rows >= window_samples`); the window
normalizer uses `sqrt(1/rows_current + 1/rows_previous)`.
The tail-service RESERVED ACK carries same-interval routed rows; the
settled record requires `tail_lane_row_ns` (ns/row) and
`effect_rows_per_s` (sampled lane rows/wall second), refusing the old
request-unit field names. The predicted effect is
`E = ceil(tail_lane_row_ns * tail_share_gain_ppm / 1e6)` ns/row;
`n` is the smallest routed-row count satisfying
`noise_ppm(n) * z(alpha)/1e6 <= E/tail_lane_row_ns` and
`W = ceil(n * 1e9 / effect_rows_per_s)` controls cadence and quiet-gate
recovery / TTLs. The calib5 request-unit measurements are **not**
hardware measurements of these new row fields; CPU acceptance tests use
explicitly synthetic conversions alongside the genuine unchanged calib5
planner measurements. See `live_policy_calibration.md`.

## Hardware: the derived production policy works (2026-11-06, runs rhp16-calib2 to calib7)

Each A/B from calib2 to calib6 found one more derivation defect that kept the derived policy from moving, fixed in turn on
this branch: the policy refused the unwatermarked histogram feed (heat_watermarked now recorded and followed); the margin was
the largest or median window-to-window tail-share change, which is the deliberate workload shift, not noise (margin is now the
multinomial sampling error); min_samples was in routed observations while the planner counts heat windows; eight further
unit/scale mismatches between derived settings and the helper's comparisons (follow-up 4, each with a test, plus a calib5-numbers
planner test that must move on a shift and stay still on steady load); and the effect gate timed lanes per request, which on a
mixed workload swings ~70% with request size, so it reverted good moves (now per routed row end to end). Separately, every
assert in the serving and helper component tests had been compiled out (RelWithDebInfo, -DNDEBUG); fixed on main (#688).
Calibration records key on the plan inputs without engine_build, so they carry across pin bumps.

calib7 (flashnext-4gpu, production profile, policy derived from its own calibration record, two on/off pairs, policy-live-9
workload, same build both arms):

| | policy off (2 arms) | policy on (2 arms) | change |
|---|---|---|---|
| filler TTFT, fills 6-8 (after the shift) | 19.49 s | 13.42 s | -31.2% |
| documentation prefill, mean of 20 | 3,663 tok/s | 3,652 tok/s | -0.3% (A/A spread about 1%) |
| decode step, 1 stream / 6 streams | 25.5 / 50.0-50.3 ms | 25.5 / 49.6-49.8 ms | none |
| transactions | - | 30 committed, 1 abort, 37 GB moved | |

Decode tok/s at one stream on the single seed-7 prompt differs between arms (step time does not): moved experts are computed on
another card, so greedy text and therefore MTP acceptance change. That is a numerics question, gated below.

Not yet default-on, two gates left:
1. Quality: KLD and top-1 against BF16 with the policy moving (moves put hot experts on the 6900 XT tail, RAD46). The campaign
   quality arms cannot start a manifest-configured engine with --kld-* yet (rad0009) and cannot pass a calibration store; until
   RSC11 provides that, run it by hand (helpers from the calibrated manifest, a KLD-mode engine on the same helpers).
2. Churn: the effect gate judged 21 windows, kept 0, reverted 9, undecided 12. The gain survives, but reverts and re-moves cost
   copy bandwidth (37 GB in two arms) and can oscillate over a long soak. The gate must be able to keep a move it can see helping.


## Step 5 follow-up 6: judge the continuously published tail service (2026-11-06)

The calib7 21-window effect-gate result (0 kept, 9 reverted, 12
undecided despite a 31.2% filler TTFT reduction) was judged using helper
0's process-private service. A move that relieves the tail helper need not
speed up coordinator helper 0; its local signal is therefore not a valid
effect or calibration signal for tail relief.

Each policy-enabled helper now snapshots the **existing** per-class
LaneServiceSource totals (sampled requests, routed rows, service ns) on
its CPU epoch tick and publishes both lanes into its own single-writer
shared-memory slot. The slot occupies unused space in the placement ACK
page: no larger mapping, collector thread or GPU sample. A seqlock over
atomic fields prevents torn cross-process reads; uninitialised, future
and stale monotonic timestamps fail closed. The segment protocol is
**v24**, and v23 attachments are refused without compatibility parsing.

worker_heat.h::propose optimises **only the last helper's tail share**:
plan.baseline - plan.predicted is the predicted relative heat-share
saving of that helper, even if other helpers send or receive experts.
Consequently the effect gate for such a plan must read that last helper's
published lane-class counters, not helper 0's. It retains the follow-up
5 ns-per-routed-row maths, noise envelope, quiet windows, Gaussian row
sigma and sampled rows/s. Calibration derives effect_row_noise_ns and
effect_rows_per_s from exactly this continuous **tail** source as well.
The transaction-phase RESERVED ACK still supplies the separate tail
service-rate / ns-per-row calibration inputs.

Card-free synthetic tests publish an unchanged helper 0 and a tail
improved by 9% per row: the published tail gate returns KEPT, while the
retired coordinator-local source returns UNDECIDED. A 10% slower tail
returns WORSE; seqlock and freshness rejection are tested. The calib5
planner acceptance test (steady no move, shifted heat moves) remains.
Hardware churn and quality require follow-up validation; no new GPU
measurement is implied by these CPU tests.

## Step 5 follow-up 7: size from correlated tail-window noise (2026-11-06)

Calib8's published-tail effect gate judged ten windows: **0 kept,
2 reverted, 8 undecided**, with a 236% consecutive-window noise floor
despite an i.i.d.-derived 49,718-row window (~0.29 s). The request
mix and tail service are bursty and correlated: estimating the
variance of two window means as `sigma_row * sqrt(2/n)` is unsound.

The existing CPU epoch heartbeat now samples the **same published tail**
service slot as the production gate, without a GPU query, collector
thread or shared-memory ABI change. A bounded ordered tick history
produces a geometric ladder of routed-row window lengths and the
observed RMS fractional differences of consecutive non-overlapping
window means. The larger noise over the two lane classes is retained
at each length. At least three differences per length and three
lengths spanning one lab cadence are needed to settle; otherwise
`effect_window_noise_ladder` is absent with a short-series reason.
Old calibration records are missing a mandatory measurement and
cannot silently derive.

Let `E = ceil(tail_lane_row_ns * tail_share_gain_ppm / 1e6)`,
`f = E/tail_lane_row_ns`, and z(alpha) be the declared
two-sided Gaussian objective quantile. Production takes the
**smallest measured** L with `z(alpha) * noise_ppm(L)/1e6 <= f`;
k uses the selected measured noise and the declared T and alpha.
For `W = ceil(L * 1e9 / effect_rows_per_s)`, the policy chooses
`cadence_ns = txn_p95_ns + k * W`, enlarging cadence and all
dependent timeouts when a longer observed window is needed.
No unmeasured length is extrapolated; if no rung resolves the
predicted effect, production fails closed pending more calibration
evidence instead of shortening the gate window.

Card-free tests include seeded AR(1) and i.i.d. series (the former
demonstrates i.i.d. undersizing, the latter agreement), a KEPT 10%
tail improvement and UNDECIDED unchanged tail, insufficient and
invalid ladders, unresolvable effects, cadence expansion, and
the existing calib5-numbers heat-shift planner acceptance test.
No hardware remeasurement is claimed.

## calib8 (2026-11-06): effect gate on the published tail lane, churn halved

The effect gate now judges the tail helper's published lane service (follow-up 6). Two on/off pairs, same workload and build:
filler TTFT fills 6-8 on 12.83 s vs off 19.85 s (-35.3%; -34% against off1 alone, off2 ran slow overall), documentation prefill
on 3,660 vs off 3,564 tok/s (+0.7% against off1), decode step 25.3 / 49.8 ms vs 25.5 / 50.1 ms, 23 transactions, 0 aborts,
16.6 GB moved (calib7: 30, 1, 37 GB). The gate still kept no move: on the tail lane consecutive 0.29 s windows differed by 236%
(bursty, correlated service), so the i.i.d. window was far too short; follow-up 7 sizes the window from measured window-to-window
noise over a ladder of lengths and lengthens the cadence instead of shortening the window.

## Step 5 follow-up 8: calibration cannot stop serving; judge relative tail service (2026-11-06)

The calib9 record's absolute tail-service ladder did not resolve its
6.6249% predicted reduction: the smallest empirical RMS noise was
32,781 ppm at 1,281,536 rows, versus a required maximum of
66,249 / 2.576 = 25,718 ppm. This is a failure of the *absolute*
signal under a workload phase change, not a reason to stop serving.

A production start with a **schema-valid** record that is incomplete
or cannot derive a live policy now writes the measured serving manifest
without `live_policy`. Stderr and the existing placement provenance
include the record key and reason. Unparseable TOML, schema errors,
unknown keys and malformed ladders still refuse the file; they are
not treated as a non-derivable valid record.

The gate's measured quantity is now
`r = (tail service ns / tail routed rows) /
(sum(other helper service ns) / sum(other helper routed rows))`,
computed for each lane class from *the same windows* of the continuously
published, validated v24 service slots. The denominator is the
row-weighted peer mean. All peer slots must be fresh and coherent;
missing or stalled peer measurements cannot silently fall back to
absolute service. The CPU tick ladder uses r, still requiring four
observed windows per length and at least three comparisons. Its row-rate
basis remains the sampled tail lane's routed rows per wall second.

For a committed plan with predicted fractional tail-share gain
`g = tail_share_gain_ppm / 1e6`, the predicted relative improvement
is `r_after = r_before * (1-g)`. This assumes per-row tail service
scales with the tail's predicted share and the other helpers'
row-weighted per-row service is unchanged. The quiet envelope and the
pre/post effect gate verdict compare r, not raw tail ns/row. A common
30% workload-wide service increase therefore cancels; a simultaneous
7% improvement in r is KEPT, no relative change is UNDECIDED, and a
7% increase in r is WORSE in card-free synthetic tests.

The old calib9 absolute numbers remain explicitly unresolvable, not
relabelled as a measured relative ladder. A fresh hardware calibration
is required before a production relative policy can be claimed to
resolve this signal; the card-free results establish only the
implementation and invariance of the new statistic.
