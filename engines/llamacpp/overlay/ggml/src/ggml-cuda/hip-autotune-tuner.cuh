// bigcherry: the tuning engine (HI12).
//
// Given a signature, measure every eligible candidate and pick a winner. The
// rules that make the result mean anything:
//
// **Complete-path timing** (standards 7.1). Time everything a candidate needs
// to produce its output -- activation quantisation, per-call workspace prep,
// the kernel, reductions, output conversion -- not just the terminal kernel.
// MMQ's activation quantisation is a large fraction of its cost; a measurement
// that excluded it would make MMQ look artificially good against MMVQ, which
// has no such step.
//
// **Native is the reference and the baseline** (standards 7.2, 7.3). Its output
// is what correctness is judged against, and a challenger must beat it by a
// real margin to replace it. If native cannot be measured for a signature, the
// whole tuning run for that signature is rejected -- a winner chosen without a
// baseline is a winner chosen against nothing.
//
// **Hard eligibility before launch** (standards 12.4). `can_execute` is asked
// first; an ineligible candidate is never launched, and the reason is recorded
// so a suspiciously thin candidate set is visible rather than silent.
//
// **Coverage is part of the output.** A run that measured four of forty
// candidates reports its winner exactly as confidently as one that measured all
// forty. The counts below are what tell those two apart, and without them
// "we tuned it" is an assumption.

#pragma once

#include "hip-autotune-types.h"

#if defined(GGML_USE_HIP) && defined(GGML_HIP_AUTOTUNE)

#include <stddef.h>
#include <limits>
#include <string>

// Why a candidate was not measured, or was measured and rejected. Recorded per
// candidate so a thin measurement set can be explained rather than guessed at.
enum ggml_hip_reject_reason {
    GGML_HIP_REJECT_NONE = 0,
    GGML_HIP_REJECT_ARCHITECTURE,   // architecture_mask excludes this GPU
    GGML_HIP_REJECT_INELIGIBLE,     // can_execute said no
    GGML_HIP_REJECT_WORKSPACE,      // over max_workspace_bytes (standards 7.3)
    GGML_HIP_REJECT_LAUNCH_FAILED,
    GGML_HIP_REJECT_NAN_INF,        // introduced NaN or Inf
    GGML_HIP_REJECT_TOLERANCE,      // NMSE or max error beyond policy
    GGML_HIP_REJECT_UNSTABLE,       // non-deterministic beyond policy (winner-only recheck)
    GGML_HIP_REJECT_NOISY,          // timing dispersion too wide to rank (HI12 E4)
    GGML_HIP_REJECT_COUNT
};

// Tuning parameters. Defaults are the values standards 7 and 8 specify; they
// are here rather than hard-coded so a long run can trade precision for time.
// HI65: pre-sample intervention mode for the tuner's measurement loop.
// ONE enum, not independent booleans: an invalid combination (an eviction
// without a defined post-eviction state) is unrepresentable rather than
// runtime-rejected. Resolved once in ggml_hip_tuner_get_config from
// GGML_HIP_TUNE_FLUSH_L2 / GGML_HIP_TUNE_FLUSH_REWARM; setting both is a
// fail-closed configuration error.
enum ggml_hip_pre_sample_mode {
    // No intervention; the back-to-back hot state (production default).
    GGML_HIP_PRE_SAMPLE_NONE = 0,
    // Evict + stream sync before each timed sample (Slice B0).
    GGML_HIP_PRE_SAMPLE_EVICT = 1,
    // Evict + sync, then ONE untimed rewarm launch + sync before the
    // measurement clocks start: isolates cache-state preconditioning from
    // residency itself (HI65 attribution matrix).
    GGML_HIP_PRE_SAMPLE_EVICT_REWARM = 2,
};

struct ggml_hip_tuner_config {
    bool   valid                   = true;
    int    warmup_launches       = 15;   // 10-20 (standards 11.4)
    int    screen_samples        = 20;   // 15-30
    int    final_samples         = 100;  // standards 11.5

    // HI34: adaptive shared launch batching, superseding a static
    // launches_per_sample. Native is piloted once per
    // signature at one launch per sample; the batch size that would put its
    // median above min_sample_us is derived from that pilot and then reused
    // -- unchanged -- for native's real measurement and every candidate's,
    // native and candidate alike, for this signature. Per-candidate
    // calibration would bias the comparison by amortising fixed launch
    // overhead differently across candidates of different true cost.
    int    pilot_samples           = 3;
    double min_sample_us           = 100.0;   // GGML_HIP_TUNE_MIN_SAMPLE_US
    int    max_launches_per_sample = 32;      // GGML_HIP_TUNE_MAX_LPS

    // HI64: bounded retry, WITHIN one timed sample, for a spurious non-
    // positive/non-finite hipEventElapsedTime() reading only -- real-hardware
    // evidence (Windows/WDDM, RX 7900 GRE) found this specific silent branch
    // firing on a sub-millisecond kernel with every checked HIP API call
    // still reporting success, consistent with event-timestamp precision
    // rather than a hard fault. A genuine HIP API failure (anything hip_ok()
    // itself catches) is NEVER retried -- it stays immediately fatal, same as
    // before this item: "a wrong measurement is worse than no measurement"
    // only applies to trusting a bad number, not to masking a real error.
    // 0 disables retry, restoring the original immediately-fatal behavior.
    int    elapsed_time_retry_max  = 2;       // GGML_HIP_TUNE_ELAPSED_RETRY

    // HI64 (2026-08-23, second real-hardware finding): the retry above was
    // originally back-to-back with no delay, and one run in six still
    // exhausted all 3 attempts and poisoned. If the underlying WDDM
    // event-timestamp anomaly is tied to the driver's own submission/
    // scheduling batching rather than a single independent per-launch coin
    // flip, hammering the identical measurement again in the same
    // scheduling window would not be expected to help -- a real wait gives
    // the driver a chance to leave that window before the next attempt.
    // Linear backoff (attempt number * this value), not exponential: the
    // retry budget is small (2 by default) and unbounded exponential growth
    // has no evidence behind it here, only a bare hypothesis that SOME delay
    // beats none. 0 disables the wait, matching the original immediate-
    // retry behavior.
    double elapsed_time_retry_backoff_us = 2000.0;  // GGML_HIP_TUNE_ELAPSED_RETRY_BACKOFF_US

    // HI34 step 3 (Slice B0): cache eviction between timed samples.
    // Diagnostic only, default OFF: the residency experiment needs a cold
    // extreme to compare against the hot back-to-back state. Sizing is
    // explicit megabytes rather than 2 x hipDeviceAttributeL2CacheSize --
    // on Navi 31 that attribute reports the 6 MB L2 and knows nothing about
    // the 64-96 MB Infinity Cache behind it, so a derived size would flush
    // the wrong level and produce a null result by construction. When
    // enabled, get_config forces max_launches_per_sample to 1: a flush cannot
    // reach inside a batch, so a batched sample would measure one cold launch
    // plus lps-1 hot ones and report the mean as if it were one number.
    // HI65: measurement code branches on pre_sample_mode, never on the two
    // request booleans. flush_l2 remains only as the resolved 0/1 wire format
    // for artifact emission and backward compatibility (set by get_config).
    ggml_hip_pre_sample_mode pre_sample_mode = GGML_HIP_PRE_SAMPLE_NONE;
    int flush_l2       = 0;      // wire mirror: 1 iff pre_sample_mode != NONE
    int flush_evict_mb = 256;    // GGML_HIP_TUNE_FLUSH_MB

    // HI34: fresh, disjoint confirmation holdout for a provisional winner
    // (selection samples never participate, removing winner's-curse from
    // promotion evidence). Rounds alternate first-measured by schedule_seed
    // so neither side is systematically measured under worse thermal state.
    int    confirmation_samples    = 100;     // GGML_HIP_TUNE_CONFIRM_SAMPLES

    // Screening retention (standards 11.4). Native is always kept.
    int    screen_keep_top       = 3;
    double screen_keep_within_pct = 10.0;

    // HI24: when the same kernel measured twice differs by more than this, the
    // signature's timings are not trustworthy and are taken again.
    //
    // The threshold is a statement about the harness, not the kernels: native
    // and a forced J equal to J_best run identical code (RV21), so any gap
    // between them is measurement error by construction. 5% is chosen against
    // the observed floor -- a quiet machine reproduced the same kernel to 0.6%,
    // while a 3-sample screen on a busy one diverged by 14%.
    double noise_canary_pct      = 5.0;
    // HI68: probe allowance. 0 sends a failed initial canary straight to
    // UNRESOLVED; any positive value allows exactly ONE pair-only stability
    // probe, followed (only if the probe passes) by one fresh complete
    // finalist block that becomes the sole ranking dataset. The fresh
    // block's own canary is never retried at any setting.
    int    noise_canary_retries  = 1;

    // HI24 step 4: measure native a second time as a same-kernel twin so
    // every family (not only the ~9% where native is MMQ with a J-best pair)
    // gets a repeatability canary. The twin is a synthetic measurement role,
    // never a candidate: it cannot win, is not counted in result.measured,
    // and is emitted as "<stable_name>#twin".
    int    double_native         = 1;   // GGML_HIP_TUNE_DOUBLE_NATIVE

    // HI24 steps 5-6: cumulative call-weighted impact share (from
    // `python -m bigcherry.inventory hot-list`, GGML_HIP_TUNE_HOT_SIGNATURES)
    // at or below which a signature is "hot" and skips screening's
    // noise-driven elimination. Only takes effect when a hot list is
    // actually loaded; with no list, every signature behaves exactly as it
    // did before this item.
    double hot_share_pct         = 80.0;   // GGML_HIP_TUNE_HOT_SHARE

    // Winner selection (standards 7.3).
    double replacement_threshold_pct = 1.0;
    double tie_pct                   = 0.5;
    size_t max_workspace_bytes       = 0;  // 0 = unlimited

    // HI50: which compiled-in ranking policy actually governs promotion
    // (determinism recheck, confirmation holdout, replay cache) versus which
    // policies merely shadow-evaluate and get recorded for offline
    // comparison. An unrecognized production_policy or malformed active list
    // fails closed before measurement. "all" is the standing default for
    // active_policies; explicit lists must contain known, unique names and
    // include the production policy.
    std::string production_policy    = "latency-v1";
    std::string active_policies      = "all";

    // Correctness (standards 7.2), judged against native's own output.
    double max_nmse              = 1e-6;
    double max_abs_error         = 1e-2;

    // HI12 E1: a challenger must be both materially and *consistently*
    // faster. 0.025 is one-sided; at final_samples = 15 that means winning
    // at least 12 of the 15 paired rounds. Below min_paired_rounds usable
    // pairs the test is declared inconclusive rather than run underpowered.
    double confidence_alpha    = 0.025;
    int    min_paired_rounds   = 8;

    // HI12 E4: dispersion beyond this fraction of the median means the
    // candidate cannot be ranked at all, whatever its median says.
    double noisy_mad_ratio     = 0.25;
    int    verify_determinism  = 1;   // winner-only bitwise recheck

    // HI12 E2: emit raw finalist samples so a winner is recomputable offline.
    int    emit_samples        = 1;
};

// HI99: single source of truth for every direct scalar env-backed
// ggml_hip_tuner_config override. Before this macro, the env-override
// parsing block and the measurements.jsonl header-emission fprintf were two
// independently hand-maintained lists that were free to drift -- confirmed
// they already had (11 of the fields below were env-overridable but never
// appeared in the header, so a real tuning run's own provenance did not
// record that they had been set). This is the same failure-mode class as
// the historical incident where the `compiler` header field silently went
// missing for weeks and nobody caught it.
//
// F(TYPE, cpp_field, wire_key, env_name, min_expr, max_expr) -- TYPE in
// {INT, SIZE, DOUBLE} selects which env-parsing helper
// (int_env/size_env/double_env, hip-autotune-tuner.cu) and which fprintf
// specifier apply. wire_key is spelled out explicitly rather than derived
// from cpp_field because the two already disagree for at least one field
// (confidence_alpha is emitted as "alpha") -- inferring one from the other
// would either silently rename the wire format or require yet another
// hidden mapping this macro exists to eliminate.
//
// Deliberately NOT covered here: GGML_HIP_TUNE_FLUSH_L2 and
// GGML_HIP_TUNE_FLUSH_REWARM resolve into ONE enum (pre_sample_mode) plus a
// derived 0/1 wire mirror (flush_l2) -- not a 1:1 scalar override, so that
// resolution logic (and its emission) stays hand-written in
// ggml_hip_tuner_get_config()/ggml_hip_tuner_flush() rather than being
// forced into this table. flush_evict_mb is an ordinary scalar and IS
// covered below. Also deliberately NOT covered: screen_keep_top,
// screen_keep_within_pct, noise_canary_retries, tie_pct, max_nmse,
// max_abs_error, min_paired_rounds -- these are not env-overridable today,
// and promoting them to be so is a separate tuning-policy decision, not
// this anti-drift refactor's scope.
#define GGML_HIP_TUNER_CONFIG_FIELDS(F) \
    F(INT,    final_samples,                 "final_samples",                 "GGML_HIP_TUNE_FINAL_SAMPLES",             2,    100000) \
    F(INT,    screen_samples,                "screen_samples",                "GGML_HIP_TUNE_SCREEN_SAMPLES",            1,    100000) \
    F(SIZE,   max_workspace_bytes,           "max_workspace_bytes",           "GGML_HIP_TUNE_MAX_WORKSPACE",             (size_t) 0, std::numeric_limits<size_t>::max()) \
    F(DOUBLE, noise_canary_pct,              "noise_canary_pct",              "GGML_HIP_TUNE_NOISE_PCT",                 0.0,  std::numeric_limits<double>::max()) \
    F(INT,    double_native,                 "double_native",                 "GGML_HIP_TUNE_DOUBLE_NATIVE",             0,    1) \
    F(DOUBLE, hot_share_pct,                 "hot_share_pct",                 "GGML_HIP_TUNE_HOT_SHARE",                 0.0,  100.0) \
    F(DOUBLE, confidence_alpha,              "alpha",                         "GGML_HIP_TUNE_ALPHA",                     0.0,  1.0) \
    F(DOUBLE, noisy_mad_ratio,               "noisy_mad_ratio",               "GGML_HIP_TUNE_NOISY_MAD",                 0.0,  std::numeric_limits<double>::max()) \
    F(INT,    verify_determinism,            "verify_determinism",            "GGML_HIP_TUNE_VERIFY_DETERMINISM",        0,    1) \
    F(INT,    emit_samples,                  "emit_samples",                  "GGML_HIP_TUNE_EMIT_SAMPLES",              0,    1) \
    F(INT,    pilot_samples,                 "pilot_samples",                 "GGML_HIP_TUNE_PILOT_SAMPLES",             1,    100000) \
    F(DOUBLE, min_sample_us,                 "min_sample_us",                 "GGML_HIP_TUNE_MIN_SAMPLE_US",             0.0,  std::numeric_limits<double>::max()) \
    F(INT,    max_launches_per_sample,       "max_launches_per_sample",       "GGML_HIP_TUNE_MAX_LPS",                   1,    100000) \
    F(INT,    elapsed_time_retry_max,        "elapsed_time_retry_max",        "GGML_HIP_TUNE_ELAPSED_RETRY",             0,    10) \
    F(DOUBLE, elapsed_time_retry_backoff_us, "elapsed_time_retry_backoff_us", "GGML_HIP_TUNE_ELAPSED_RETRY_BACKOFF_US", -1.0,  std::numeric_limits<double>::max()) \
    F(INT,    confirmation_samples,          "confirmation_samples",          "GGML_HIP_TUNE_CONFIRM_SAMPLES",           2,    100000) \
    F(INT,    flush_evict_mb,                "flush_evict_mb",                "GGML_HIP_TUNE_FLUSH_MB",                  1,    65536)

const ggml_hip_tuner_config & ggml_hip_tuner_get_config();

// HI64 (2026-08-22): the header comment above this struct used to claim
// ggml_hip_tuner_resolve() "returns nullptr only when the run had to be
// rejected" -- stale; the real implementation always returns
// native.candidate on every rejection/failure path, never nullptr.
// Corrected below, and this struct now makes the fatal-measurement-failure
// case an explicit, checkable fact rather than something a caller could
// only infer from an unusual reason string.
struct ggml_hip_tuner_resolution {
    // The tuner's selected candidate. Native is returned for an ordinary
    // native-retained result and also as the safe workload choice after a
    // fatal measurement failure -- `measurement_failure` is what tells the
    // two apart, not this pointer's identity.
    const ggml_hip_candidate_descriptor * winner = nullptr;

    // True only when this invocation hit a fatal/mechanical measurement
    // failure (Result::measurement_failure). Such an outcome is local to
    // the current HIP device/context and MUST NOT be used to populate a
    // process-global dispatch binding shared with other devices via the
    // portable (device-ordinal-free) hardware/dispatch key -- confirmed on
    // real dual-XTX hardware that doing so lets one device's transient
    // fault silently block tuning on a second, healthy, identical GPU. It
    // may still be retained as diagnostic/journal evidence.
    //
    // This flag says nothing about HI67 production correctness or
    // promotion; tune-mode workload dispatch remains native regardless of
    // the selected winner either way.
    bool measurement_failure = false;
};

// Resolve a signature by measuring it. Measures once per portable dispatch
// key and caches a NON-FAILED result there; the resolver's process cache
// then keeps later executions (on this or another device sharing that
// portable key) free. A fatal measurement failure is never cached as a
// resolution -- see ggml_hip_tuner_resolution::measurement_failure and
// record_result()'s failure->success replacement in the .cu file.
ggml_hip_tuner_resolution ggml_hip_tuner_resolve(
    ggml_backend_cuda_context & ctx,
    const ggml_hip_dispatch_signature_v1 & sig,
    const ggml_hip_hardware_key_v1 & hw,
    const ggml_hip_digest & dispatch_digest,
    const ggml_hip_native_selection & native,
    const ggml_hip_launch_context & lc);

// Write measurements and winners as JSON Lines, and the coverage report
// alongside. Called from ggml_hip_autotune_flush.
void ggml_hip_tuner_flush();

#endif // GGML_USE_HIP && GGML_HIP_AUTOTUNE
