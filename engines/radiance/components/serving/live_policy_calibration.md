# RHP16 per-plan live-policy calibration (step 3: read only)

The optional TOML supplied to bc-serve --calibration FILE supports normal resource
[[entry]] parameters plus [[live_policy_record]]. The file requires
schema_version = 1. Each live record is keyed by its 64-character lowercase
plan_cache_key (sha256), with required layout, objective, seeded (measurement
date) and evidence (source run). A missing key means uncalibrated; a malformed
record or duplicate key fails closed, even if it does not match this plan.
Step 4 will write these records after a helper settle period.

Every measurement is an integer. Except lane_lost_ns (which may be zero), all
are strictly positive. Unknown fields are refused.

| Required record field | Meaning and units |
|---|---|
| txn_p50_ns, txn_p95_ns | Closed transaction wall time, ns |
| copy_bytes_per_s | Slowest helper's achieved transfer bytes per wall second, including the epoch protocol |
| heat_gap_max_ns | Largest gap between consumed heat deltas |
| hot_set_k | Tail capacity divided by the device expert stride, experts |
| marginal_error_ppm, marginal_observations | Measured marginal-expert sampling error and observations |
| drift_half_life_observations | Half-life of top-K overlap from non-decayed interval counts |
| observation_rate_per_s | Heat observations per second |
| tail_share_noise_ppm | Largest no-move relative tail-share change |
| tail_share_gain_ppm | Predicted relative change from the proposed placement |
| tail_lane_service_ns_per_s | Measured tail lane service nanoseconds per wall second |
| lane_lost_ns | Lost service time during a move (can be 0) |
| effect_window_samples, effect_requests_per_s | Variance-convergence window count and observed lane request rate |
| effect_quiet_windows | Quiet windows to confidence convergence |
| effect_noise_floor_ns, effect_noise_quantile_ns | Quiet-window noise and empirical quantile |
| noise_false_revert_rate_ppm | Measured quantile's false-revert rate |
| abort_quiescent_ns | Optional measured abort-to-quiescent p95, ns |

The seeded calibration entries calibration.live_policy.transaction_effect_windows
and calibration.live_policy.false_revert_rate_ppm are explicit objectives with
written reasons, not hidden time/byte/step thresholds. The empirical noise
quantile must match the requested false-revert rate.

## Derivations

Let W = ceil(effect_window_samples * 1e9 / effect_requests_per_s).

- cadence_ns = transaction p95 + W.
- max_moved_bytes = min(smallest helper budget, copy_bytes_per_s * W *
  declared transaction_effect_windows / 1e9).
- max_moves = floor(max_moved_bytes / device-layout expert stride).
- min_residency_ns = (move_bytes / achieved copy bandwidth + lost lane time)
  / (predicted relative tail gain * measured lane service rate).
- half_life_observations_milli = 1000 * observed top-K drift half-life;
  max_heat_age_ns = largest heat update gap.
- margin_ppm = max(no-move tail-share noise, marginal sampling error);
  min_samples = ceil(marginal_observations * (marginal_error / margin)^2).
- abort_cooldown_ns = measured abort-to-quiescent, if available; otherwise
  cadence_ns as a labelled conservative settling fallback.
- effect_window_samples and effect_min_quiet_windows = observed convergence;
  effect_noise_multiplier_milli = 1000 * empirical noise quantile / noise floor.
- effect_revert_ttl_ns = observed drift observations / observations per second.
- learned_eligibility_ttl_ns = one observed heat gap as a mandatory v1
  approximation; event invalidation is not representable in v1.

The gate is required and unwatermarked heat is refused. Protected layer IDs
cannot be discovered through the current container ABI; the policy emits an
empty protected_layers array, while the helper still guards missing weights.
RCS12 model metadata must supply model-specific protected layers when needed.

The locked v1 manifest has no enabled flag and no observe-only mode. Without
calibration, the policy object is absent, disabling the policy thread and its
driver-owned measurement hooks. This cannot satisfy the original observe-only
request without a v2 contract change. v2 should introduce an explicit
observe-only state and event-driven learned eligibility; v1 also requires the
time-based learned TTL and a multiplier even though their replacements are
event-based and an empirical quantile.

The fixture writer in live_policy_test.cpp uses actual policy-live-9 p50
1.77 s, p95 2.72 s, copy 0.16 GB/s, gap 5.04 s, marginal error 2.3% and
hot-set k=5931. The missing effect-gate, noise, drift and lane-service
readings in that test are explicitly synthetic, not production calibration.
