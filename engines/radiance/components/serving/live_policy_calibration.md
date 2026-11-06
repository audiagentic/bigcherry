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
A valid but incomplete or non-derivable record (including an unresolvable
noise window or out-of-range budget) **never blocks production serving**:
`bc-serve` omits `live_policy`, writes the record key and specific reason on
stderr and under the existing `/spec/handoff/placement` manifest provenance
record. A malformed TOML file or schema violation remains a hard error.

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
| effect_row_noise_ns | Diagnostic per-row dispersion σ (not used to size a correlated window) |
| effect_window_noise_ladder | Array of [routed rows, RMS adjacent-window fractional difference in ppm] of r = tail ns/row / row-weighted mean peer ns/row, over the same published service windows; at least three measured lengths spanning a lab cadence |
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
Gaussian quantile defined by `erfc(z/sqrt(2)) = α`. The gate compares
**relative service** `r = (tail lane service ns / tail routed rows) /
(sum(peer lane service ns) / sum(peer routed rows))`, using all other
policy-enabled helpers and the same published sampling interval. The
peer denominator is the **row-weighted mean**, not a mean of helper
ratios. Large and small lanes remain separate.

Let `g = tail_share_gain_ppm / 1000000`, the committed plan's predicted
*fractional* tail-share reduction. Assuming tail per-row service is
proportional to predicted tail share and that the peers' row-weighted
per-row service remains unchanged after moving, `r_after = r_before * (1-g)`,
so the predicted **fractional reduction in r is g**. This is a model
assumption, not an observed per-row saving. The earlier absolute prediction
`E = ceil(tail_lane_row_ns * g)` cannot be used as a verdict against a
workload-wide shift, since all helpers can become slower together.

The helper retains a bounded CPU-epoch-tick history of the same
published tail and peer helper service counters that the effect gate judges. For
geometrically increasing routed-row lengths L, starting at the largest
single observed tick and continuing only while at least four disjoint
windows fit, it computes RMS fractional differences between consecutive
non-overlapping window ratios r (tail ns/row divided by peer row-weighted ns/row). At least three differences per length
are required; the worse measured lane class sets each point. It emits
`effect_window_noise_ladder = [[L, noise_ppm], ...]` with at least three
lengths spanning one lab cadence. Otherwise the ladder is absent
with a named short-series reason and cannot activate the effect gate;
production may still derive the planner policy with `effect_gate=false`.

Let `f = g` be the predicted relative effect on r.
The smallest *observed* L satisfying `noise_ppm(L)/1e6 * z(alpha) <= f`
sets `effect_window_samples`. At that L,
`p = erfc(f / (sqrt(2) * noise_ppm(L)/1e6))` (p=0 if noise=0), and
the minimum positive k satisfies `T * p^k <= alpha`. Gaussian window
tails and independence of successive crossings remain assumptions;
**routed rows are not assumed independent**. If no measured L resolves
f, derive `effect_gate=false` with the unresolvable-ladder reason in
provenance. The planner policy remains active; no shorter fictitious
window or i.i.d. extrapolation is substituted.

Let `W = ceil(L * 1e9 / effect_rows_per_s)`; the legacy
`effect_window_samples` control counts routed rows. Cadence is
`txn_p95_ns + k * W`, growing to fit the chosen measured length and
every required quiet comparison. The same derived cadence feeds
heat-age, sample-count, eligibility and revert TTLs.

- cadence_ns = transaction p95 + k * W, never capped to an earlier
  shorter cadence. The ONE-transaction byte allowance is
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
  Both the quiet fraction and verdict compare the tail/peer service ratio r,
  whose predicted fractional effect is g; the legacy measured absolute
  per-row sigma is diagnostic only; variable request sizes would inflate
  noise by mixing 27K-row prefills and short decodes if divided by requests.

| Derived effect setting | Inputs | Rule |
|---|---|---|
| effect_window_samples (routed rows) | observed relative-service ladder, predicted tail-share gain g, declared α | smallest measured L with `noise_ppm(L)/1e6 * z(α) <= g` |
| effect_min_quiet_windows | selected window noise(L), predicted fractional effect, declared T and α | minimum k with `T * p^k <= α`, using the measured-window Gaussian crossing probability |
| effect_noise_multiplier_milli | declared false-revert fraction alpha | `ceil(1000 * z(alpha))` |

Equivalent observed noise ladders from different configured gate
windows derive the same n, k and multiplier. The legacy row-σ
diagnostic and observed quiet-window maximum may differ, but neither
determines the selected window length.

Each emitted policy setting has `origin = derived` provenance. A record
missing **planner-required** evidence (transaction time, helper copy rate,
heat-gap, marginal error/observations, drift, tail gain/service or lane
downtime) omits the policy; serving continues without live placement.
Effect-only evidence (`effect_rows_per_s`, `effect_noise_max_ns`,
`effect_row_noise_ns`, `effect_window_noise_ladder`) may instead be
absent with explicit reasons. In that case the derived policy stays active
with `effect_gate=false`; the gate-specific window and quiet controls
are inert positive v1 schema values (1, 1), not alleged measurements.
Cadence derives from max(transaction p95, observed heat-feed gap),
with no imaginary quiet wait;
heat age, samples, residency, margin, move caps and TTLs still derive from
planner evidence. A non-judgable lane (tail LARGE with no peer LARGE
rows on flashnext-4gpu) is never compared using absolute tail service.
The gate-off provenance records why the relative effect cannot be
judged, and that margin, residency and move caps bound churn.

The policy derives `accept_unwatermarked_heat`
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


### Calibration bounds and drift retention (RHP16 step 5 follow-ups 13-14)

Helper 0 lists unmet planner measurements in a deduplicated
"calibration waiting:" line with specific reasons. The watchdog cannot
start at the initial heartbeat or a heat baseline: its first basis is
accepted heat deltas with a positive measured inter-delta gap and a
retained drift snapshot. Copy-ACK requires the first committed transaction;
tail service and lane-loss require the first non-revert committed transaction
involving the tail. A condition with no measurable opportunity has no
expiring deadline, even if the process has run for a long time.

Once armed, each missing condition has its own no-progress watch, bounded
by the larger of measured heat gap, policy cadence and maximum heat age,
multiplied by the drift ring capacity. The duration freezes on arming;
later heat gaps cannot extend it. Progress in that *specific* missing
measurement renews the watch, whereas unrelated activity does not.
Effect-only missing evidence follows the same basis and tracks its own
quiet/ladder progress; the relative publisher timeout also cannot arm
before the accepted heat feed. On expiry, a partial record contains actual
observations and explicit absent entries. Valid partial planner records
serve without live policy; malformed records remain hard errors. Effect-only
absences can instead derive an effect-gate-off policy when planner inputs
are present.

The drift series retains source-interval route counts with growing
observation-count spacing, adaptively thinning its bounded ring in
pairs. The newest source interval remains available for comparison
without overwriting the spaced samples. The measured route rate and
ring capacity determine the initial spacing, and the observed span
determines subsequent coarsening, independent of evaluation cadence.
