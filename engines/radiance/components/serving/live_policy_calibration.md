# RHP16 per-plan live-policy calibration

The optional TOML supplied to `bc-serve --calibration FILE` supports normal
resource `[[entry]]` parameters and `[[live_policy_record]]` with
`schema_version = 1`. Each live record is identified by a lowercase 64-character
SHA-256 plan-cache key plus required layout, objective, seeded date and evidence
description. Unknown fields, duplicate keys and malformed or unsupported old
records fail closed; a missing key means uncalibrated.

All measurements are integers. Only `lane_lost_ns` and `heat_watermarked` can be zero. An unobserved
quantity must have an explicit `absent = { quantity = "reason" }` entry;
partial records are preserved but cannot derive a production live policy.

| Required measured field | Meaning and units |
|---|---|
| txn_p50_ns, txn_p95_ns | Committed transaction wall time, ns |
| copy_bytes_per_s | Slowest receiver helper's observed transfer rate |
| heat_gap_max_ns | Maximum consumed heat-delta gap |
| heat_watermarked | 1 if every accepted delta has a complete observation watermark; 0 if any accepted delta lacks it |
| hot_set_k | Tail capacity / expert stride |
| marginal_error_ppm, marginal_observations | Marginal expert sampling error, observed requests |
| drift_half_life_observations | Measured top-K overlap half-life |
| observation_rate_per_s | Observations per second |
| tail_share_gain_ppm | Positive predicted tail-share reduction from a committed plan |
| tail_lane_service_ns_per_s | Sampled tail service per wall second |
| tail_lane_row_ns | Same-interval tail large-lane service ns / completed routed rows (ns/row), via RESERVED ACK |
| lane_lost_ns | Observed quiesce-to-reopen downtime (zero permitted) |
| effect_rows_per_s | Observed sampled-lane routed rows per wall second |
| effect_noise_max_ns | Maximum absolute quiet-window difference; **not** a statistical quantile |
| effect_row_noise_ns | Per-routed-row service-time dispersion σ from signed quiet-window deltas |
| abort_quiescent_ns | Optional abort-to-quiescent p95, ns |

The helper accumulates consecutive quiet-window per-row means and their actual routed-row
counts. For each signed delta of two adjacent means of sizes n_previous and
n_current, it records `delta / sqrt(1/n_previous + 1/n_current)`. The SD of
these normalized deltas estimates per-row σ; for equal-size n this is
`SD(delta) * sqrt(n/2)`. This is actual observed dispersion, not a configured
gate count. A positive observed quiet deviation and at least two variable
signed comparisons are required for convergence. The helper cannot see the objective
false-revert rate, so the measured maximum (renamed from `effect_noise_quantile_ns`) is
retained as a diagnostic only. It is not a false-revert quantile and does not
rescale the fractional quiet-window noise at the effect gate.

`calibration.live_policy.transaction_effect_windows` (T) and
`calibration.live_policy.false_revert_rate_ppm` (α in parts per million) are
**declared objectives**, never helper measurements. The removed record fields
`effect_window_samples`, `effect_quiet_windows` and
`noise_false_revert_rate_ppm` are refused, not silently accepted.

## Derived policy

With α = false_revert_rate_ppm / 1000000, let z(α) be the two-sided
Gaussian quantile defined by `erfc(z/sqrt(2)) = α`. Let
`E = ceil(tail_lane_row_ns * tail_share_gain_ppm / 1000000)`.
The gain is the median **predicted relative reduction** of committed
plans; the service input is the median over pre-move intervals of large-lane
service nanoseconds divided by completed routed rows from **the same
snapshot interval**. E is thus a predicted service-time change **per routed row**,
not a fraction of a wall-second and not an observed post-move improvement.
Neither factor depends on the configured quiet-window size.

The derived effect window is the smallest positive integer n with
`σ * sqrt(2/n) * z(α) <= E`. At this n the Gaussian noise-only
crossing probability is
`p = erfc(E / (sqrt(2) * σ * sqrt(2/n)))`.
The derived quiet count is the smallest positive integer k with
`T * p^k <= α`, applying a union bound over T transaction-effect
opportunities and modelling consecutive quiet crossings as independent.
Adjacent quiet differences share a window, so this is an **unverified**
Gaussian/independent-crossing model, not measured false-reversion outcomes.

Let `W = ceil(n * 1e9 / effect_rows_per_s)`, where `n` and the v1
`effect_window_samples` control count **routed rows**, not request completions.
Both cadence and gate-recovery TTLs use this same measured row throughput.

- cadence_ns = transaction p95 + W. The ONE-transaction byte allowance is
  min(slowest helper budget, copy_bytes_per_s * transaction p95 / 1e9),
  not T times the window copy allowance; max_moves = floor(bytes / stride).
- min_residency_ns = (committed transaction p95 + observed lane downtime) /
  (predicted tail-share gain * sampled tail service per wall second).
- half_life_observations_milli = 1000 * top-K drift half-life observations:
  the helper applies exp2(-routed_observations / half_life), not a heat-window decay.
- max_heat_age_ns = cadence + largest accepted heat-delta gap.
- margin_ppm = marginal_error_ppm, the multinomial sampling error of the
  k-th hottest expert's share; no workload shift is treated as noise.
  min_samples = ceil(marginal observations / ceil(observation rate * cadence)),
  in accepted heat WINDOWS, not routed requests.
- abort_cooldown_ns = max(max_heat_age_ns, measured abort-to-quiescent or cadence).
- Let recovery = (k + 1) * W: the quiet gate needs its initial window plus
  k consecutive comparisons after an abort or effect judgment.
  learned_eligibility_ttl_ns = max_heat_age_ns + recovery, and
  effect_revert_ttl_ns = max(drift half-life / observation rate, learned TTL,
  transaction p95 + min_residency_ns + cadence). The revert ban survives
  the rollback and the earliest per-expert residency-expired retry.
- effect_window_samples = n **routed rows** (the legacy member name is
  retained in v1); effect_min_quiet_windows = k.
- effect_noise_multiplier_milli = ceil(1000 * z(alpha)). The effect gate
  already compares fractional rates against fractional observed quiet noise;
  dividing absolute service-time noise by E multiplied noise twice.
  Both the quiet fraction and verdict compare ns per routed row, as do
  the measured sigma and predicted E; variable request sizes would inflate
  noise by mixing 27K-row prefills and short decodes if divided by requests.

| Derived effect setting | Inputs | Rule |
|---|---|---|
| effect_window_samples (routed rows) | per-row σ, predicted E, declared α | minimum n with `σ sqrt(2/n) z(α) <= E` |
| effect_min_quiet_windows | E, σ, derived n, declared T and α | minimum k with `T * p^k <= α` |
| effect_noise_multiplier_milli | declared false-revert fraction alpha | `ceil(1000 * z(alpha))` |

Equal-σ synthetic quiet-window series at configured widths 2 and 8
derive the same n, k and multiplier. The observed maximum may differ
with the calibration window, but is not used by the gate.

Each emitted policy setting has `origin = derived` provenance. An uncalibrated
locked-v1 plan omits the policy entirely, since v1 has no observe-only state.
The policy requires the effect gate and derives `accept_unwatermarked_heat`
from the measured `heat_watermarked` (zero means allow the authenticated
histogram-file feed; one means refuse unwatermarked deltas). RHP02's
watermarked engine producer changes the engine build identity included in
the plan-store key, forcing a new calibration rather than reusing the old
histogram-file record (or RHP02 must explicitly version the feed in the key
if its engine build fingerprint does not change). It leaves
protected_layers empty until model metadata supplies globally protected IDs
(the helper separately guards missing weights).

The synthetic quiet-window, drift and lane-service fixture measurements in
`live_policy_test.cpp` are test inputs, not production hardware evidence.

## Store key

Records are stored and looked up by `calibration_key(plan key inputs)`: the plan-key inputs without `engine_build`,
so a record survives pin bumps. Changing the model, the cards, the layout, the engine arguments or settings, or the
record schema needs a new calibration.

