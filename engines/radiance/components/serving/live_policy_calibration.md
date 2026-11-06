# RHP16 per-plan live-policy calibration

The optional TOML supplied to `bc-serve --calibration FILE` supports normal
resource `[[entry]]` parameters and `[[live_policy_record]]` with
`schema_version = 1`. Each live record is identified by a lowercase 64-character
SHA-256 plan-cache key plus required layout, objective, seeded date and evidence
description. Unknown fields, duplicate keys and malformed or unsupported old
records fail closed; a missing key means uncalibrated.

All measurements are integers. Only `lane_lost_ns` can be zero. An unobserved
quantity must have an explicit `absent = { quantity = "reason" }` entry;
partial records are preserved but cannot derive a production live policy.

| Required measured field | Meaning and units |
|---|---|
| txn_p50_ns, txn_p95_ns | Committed transaction wall time, ns |
| copy_bytes_per_s | Slowest receiver helper's observed transfer rate |
| heat_gap_max_ns | Maximum consumed heat-delta gap |
| hot_set_k | Tail capacity / expert stride |
| marginal_error_ppm, marginal_observations | Marginal expert sampling error, observed requests |
| drift_half_life_observations | Measured top-K overlap half-life |
| observation_rate_per_s | Observations per second |
| tail_share_noise_ppm | Largest relative no-move tail-share change |
| tail_share_gain_ppm | Positive predicted tail-share reduction from a committed plan |
| tail_lane_service_ns_per_s | Sampled tail service per wall second |
| lane_lost_ns | Observed quiesce-to-reopen downtime (zero permitted) |
| effect_requests_per_s | Observed sampled lane requests per wall second |
| effect_noise_floor_ns | Median absolute quiet-window service-time difference per request |
| effect_noise_max_ns | Maximum absolute quiet-window difference; **not** a statistical quantile |
| effect_sample_noise_ns | Per-request service-time dispersion σ from signed quiet-window deltas |
| abort_quiescent_ns | Optional abort-to-quiescent p95, ns |

The helper accumulates consecutive quiet-window means and their actual sample
counts. For each signed delta of two adjacent means of sizes n_previous and
n_current, it records `delta / sqrt(1/n_previous + 1/n_current)`. The SD of
these normalized deltas estimates per-request σ; for equal-size n this is
`SD(delta) * sqrt(n/2)`. This is actual observed dispersion, not a configured
gate count. A nonzero median noise floor and at least two variable signed
comparisons are required for convergence. The helper cannot see the objective
false-revert rate, so the maximum (renamed from the misleading
`effect_noise_quantile_ns`) supplies the conservative locked-v1 noise
multiplier, rather than purporting to be an objective-specific empirical quantile.

`calibration.live_policy.transaction_effect_windows` (T) and
`calibration.live_policy.false_revert_rate_ppm` (α in parts per million) are
**declared objectives**, never helper measurements. The removed record fields
`effect_window_samples`, `effect_quiet_windows` and
`noise_false_revert_rate_ppm` are refused, not silently accepted.

## Derived policy

With α = false_revert_rate_ppm / 1000000, let z(α) be the two-sided
Gaussian quantile, computed by inverting `erfc(z/sqrt(2)) = α`, and let F be
the observed `effect_noise_floor_ns`. The derived effect window is the
smallest positive integer n such that

`σ * sqrt(2/n) * z(α) <= F`.

For this n the Gaussian noise-only threshold-crossing probability is
`p = erfc(F / (sqrt(2) * σ * sqrt(2/n)))`. The derived quiet count is the
smallest positive integer k such that `T * p^k <= α`, treating the T
transaction-effect opportunities with a union bound and the k consecutive
noise crossings as independent. This model assumes Gaussian differences and
independent crossings. Adjacent consecutive comparisons share a window, so
these independence assumptions require hardware validation; they are **not**
independently observed false-reversion outcomes.

Let `W = ceil(n * 1e9 / effect_requests_per_s)`.

- cadence_ns = transaction p95 + W; max_moved_bytes = min(slowest helper
  budget, copy_bytes_per_s * W * T / 1e9).
- max_moves = floor(max_moved_bytes / expert stride).
- min_residency_ns = (move bytes / observed copy bandwidth + observed lost
  lane time) / (predicted tail-share gain * sampled tail service rate).
- half_life_observations_milli = 1000 * observed drift half-life;
  max_heat_age_ns = largest heat-delta gap.
- margin_ppm = max(no-move tail-share noise, marginal sampling error);
  min_samples = ceil(marginal samples * (marginal error / margin)^2).
- abort_cooldown_ns = measured abort-to-quiescent, or the labelled cadence
  fallback when no abort was observed.
- effect_window_samples = n; effect_min_quiet_windows = k.
- effect_noise_multiplier_milli = ceil(1000 * effect_noise_max_ns / F),
  a conservative observed maximum-to-floor envelope.
- effect_revert_ttl_ns = drift half-life / observation rate;
  learned_eligibility_ttl_ns = one observed heat gap as a locked-v1
  approximation (event invalidation needs v2).

Each emitted policy setting has `origin = derived` provenance. An uncalibrated
locked-v1 plan omits the policy entirely, since v1 has no observe-only state.
The policy requires the effect gate, refuses unwatermarked heat, and leaves
protected_layers empty until model metadata supplies globally protected IDs
(the helper separately guards missing weights).

The synthetic quiet-window, drift and lane-service fixture measurements in
`live_policy_test.cpp` are test inputs, not production hardware evidence.
