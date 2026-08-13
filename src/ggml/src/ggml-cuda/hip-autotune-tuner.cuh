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
    int    noise_canary_retries  = 1;

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

const ggml_hip_tuner_config & ggml_hip_tuner_get_config();

// Resolve a signature by measuring it.
//
// Returns the winner, or `native.candidate` when nothing beat it -- which is
// the common and correct outcome. Returns nullptr only when the run had to be
// rejected, meaning native itself could not be measured.
//
// Measures once per dispatch key and caches; the resolver's process cache then
// keeps later executions free.
const ggml_hip_candidate_descriptor * ggml_hip_tuner_resolve(
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
