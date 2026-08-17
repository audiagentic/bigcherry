// bigcherry: HI68 noise-canary transition logic (host-only, CPU-testable).
//
// Deliberately free of every GPU/ggml include: the canary decides *between*
// measurement blocks, which is a pure function of (stage, pair medians,
// threshold). Keeping it out of the GPU translation unit lets a host unit
// test compile it directly and pin the transition table without a device, a
// driver, or a full build.
//
// The state machine (docs/planning/active/hip-autotune/HI68.md):
//
//   initial block --pass--> RANK (rank the original complete block)
//   initial block --fail--> probe: pair-only stability check. Its samples
//        |                     answer "did the environment settle?" and
//        v                     their statistics are DISCARDED (RV49/F2: the
//     RANK                      old code overwrote the ranked medians with
//                               exactly this self-selected fresh draw).
//   probe --pass--> fresh block: ONE complete finalist measurement pass.
//                    It is the SOLE ranking dataset; the original block --
//                    precisely the one whose QC failed -- is discarded for
//                    ranking. Its canary is evaluated exactly once.
//   probe --fail--> UNRESOLVED (native retained on the ORIGINAL block, which
//                    was never swapped).
//   fresh --pass--> RANK (retried_pass);  fresh --fail--> UNRESOLVED.
//
// The fresh stage has no retry branch at any setting: retrying until quiet
// would re-introduce the self-selected baseline this header exists to remove.

#pragma once

#include <cmath>

// Which block the medians being judged came from.
enum ggml_hip_canary_stage {
    GGML_HIP_CANARY_STAGE_INITIAL = 0,  // original complete finalist block
    GGML_HIP_CANARY_STAGE_PROBE   = 1,  // pair-only stability probe after an initial failure
    GGML_HIP_CANARY_STAGE_FRESH   = 2,  // one fresh complete finalist block after a passed probe
};

// What to do next. Only RUN_PROBE and RUN_FRESH request further GPU work;
// both are terminal in the sense that each is taken at most once per run.
enum ggml_hip_canary_next {
    GGML_HIP_CANARY_RANK            = 0,  // stop measuring; rank on the current block
    GGML_HIP_CANARY_RUN_PROBE       = 1,  // run the pair-only stability probe (once)
    GGML_HIP_CANARY_RUN_FRESH       = 2,  // run one fresh complete finalist block (once)
    GGML_HIP_CANARY_STOP_UNRESOLVED = 3,  // stop measuring; native is retained
};

struct ggml_hip_canary_verdict {
    double pct = -1.0;  // same-kernel divergence in percent; -1.0: not evaluable
    bool passed = false;  // true only when pct >= 0 and within threshold
    ggml_hip_canary_next next = GGML_HIP_CANARY_STOP_UNRESOLVED;
};

// Same-kernel pair divergence in percent of the native median. Returns -1.0
// when either side is not a valid non-negative timing (NaN, negative, or a
// zero native): an unmeasurable pair must read as "cannot confirm", never as
// "quiet".
inline double ggml_hip_canary_divergence_pct(double native_us, double twin_us) {
    if (!(native_us > 0.0) || !(twin_us >= 0.0)) {
        return -1.0;
    }
    return 100.0 * std::fabs(native_us - twin_us) / native_us;
}

// Pure transition function. `retries_allowed` (config.noise_canary_retries at
// entry) is consulted only in the initial stage: it decides whether a failed
// initial evaluation gets its one stability probe. The probe and the fresh
// block are each taken at most once regardless of that value, by design.
inline ggml_hip_canary_verdict ggml_hip_canary_transition(
        ggml_hip_canary_stage stage, double native_us, double twin_us,
        double threshold_pct, int retries_allowed) {
    ggml_hip_canary_verdict v;
    v.pct = ggml_hip_canary_divergence_pct(native_us, twin_us);
    v.passed = (v.pct >= 0.0) && (v.pct <= threshold_pct);
    switch (stage) {
        case GGML_HIP_CANARY_STAGE_INITIAL:
            if (v.passed) {
                v.next = GGML_HIP_CANARY_RANK;
            } else if (v.pct < 0.0) {
                // Unmeasurable pair: the probe's question ("did the
                // environment settle?") is unanswerable without a readable
                // pair, so fail closed rather than spend a GPU block on it.
                v.next = GGML_HIP_CANARY_STOP_UNRESOLVED;
            } else if (retries_allowed > 0) {
                v.next = GGML_HIP_CANARY_RUN_PROBE;
            } else {
                v.next = GGML_HIP_CANARY_STOP_UNRESOLVED;
            }
            break;
        case GGML_HIP_CANARY_STAGE_PROBE:
            v.next = v.passed ? GGML_HIP_CANARY_RUN_FRESH
                              : GGML_HIP_CANARY_STOP_UNRESOLVED;
            break;
        case GGML_HIP_CANARY_STAGE_FRESH:
            // Terminal by construction: pass ranks the fresh block, failure
            // retains native. No further branch exists -- and none may be
            // added without re-opening HI68.
            v.next = v.passed ? GGML_HIP_CANARY_RANK
                              : GGML_HIP_CANARY_STOP_UNRESOLVED;
            break;
    }
    return v;
}
