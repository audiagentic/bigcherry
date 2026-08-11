// bigcherry: the tuning engine (HI12).

#include "hip-autotune-tuner.cuh"

#if defined(GGML_USE_HIP) && defined(GGML_HIP_AUTOTUNE)

#include "hip-autotune-build-hash.h"
#include "hip-autotune-dispatch.cuh"
#include "hip-autotune-journal.h"
#include "hip-autotune-signature.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <ctime>
#include <limits>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <vector>

// E6: compiler identity, so a measurements file that outlives a ROCm upgrade
// can say which compiler produced it (RV01-RV05 were invalidated by exactly
// this kind of unrecorded build-side difference).
#if defined(__clang_version__)
#  define GGML_HIP_COMPILER_STR "clang " __clang_version__
#elif defined(__VERSION__)
#  define GGML_HIP_COMPILER_STR __VERSION__
#else
#  define GGML_HIP_COMPILER_STR "unknown"
#endif

#define GGML_HIP_STRINGIFY_(x) #x
#define GGML_HIP_STRINGIFY(x) GGML_HIP_STRINGIFY_(x)

#if defined(HIP_VERSION_MAJOR) && defined(HIP_VERSION_MINOR)
#  define GGML_HIP_VERSION_STR \
       GGML_HIP_STRINGIFY(HIP_VERSION_MAJOR) "." GGML_HIP_STRINGIFY(HIP_VERSION_MINOR)
#else
#  define GGML_HIP_VERSION_STR "unknown"
#endif

namespace {

struct Measurement {
    const ggml_hip_candidate_descriptor * candidate = nullptr;
    ggml_hip_reject_reason reason      = GGML_HIP_REJECT_NONE;
    bool     measured                  = false;
    double   median_us                 = 0.0;
    double   mad_us                    = 0.0;   // standards: GPU times are
    double   p95_us                    = 0.0;   // right-skewed; MAD survives an
    double   host_median_us            = 0.0;   // outlier where stddev does not
    double   nmse                      = 0.0;
    double   max_abs_error             = 0.0;
    size_t   workspace_bytes           = 0;
    int      samples                   = 0;

    // E3: max(gpu_median, host_median - host_sync_overhead) -- whichever
    // resource is actually binding. Used for ranking; median_us stays the
    // recorded and reported figure.
    double   effective_us              = 0.0;

    // E1/E2: round-aligned final-stage samples. NaN marks a round where this
    // candidate failed to launch, so index r always means "round r" even
    // after a failure -- required for the paired sign test to compare like
    // rounds, and persisted verbatim as measurement.samples_json.
    std::vector<double> final_gpu_us;
    std::vector<double> final_host_us;

    // E1: one-sided paired sign-test result against native, over final_gpu_us.
    double   sign_p                    = 1.0;
    int      sign_wins                 = 0;
    int      sign_rounds               = 0;
};

struct Result {
    const ggml_hip_candidate_descriptor * winner = nullptr;
    std::vector<Measurement> measurements;
    int generated  = 0;   // candidates in the registry for this family/shape
    int applicable = 0;   // E5: right arch + right src0_type (varies; generated does not)
    int eligible   = 0;
    int measured   = 0;
    double improvement_pct = 0.0;
    double confidence      = 0.0;   // E1: 1 - sign_p of the chosen winner
    std::string reason;

    // The two halves the dispatch digest was built from.
    //
    // Recorded because the dispatch digest also mixes in this build's manifest
    // hash and source revision (standards 13.1), so it is only meaningful to
    // the build that produced it. Without the components, no tool can compute
    // any other key from a measurements file -- not a re-keyed cache for a
    // slimmed catalog, not a migration after an upstream bump, not even an
    // offline check of which signatures two runs share. See HI23.
    ggml_hip_digest signature_digest = {};
    ggml_hip_digest hardware_digest  = {};
    std::string canonical_json;

    // HI24 noise canary. `canary_pct` is the divergence between two
    // measurements of the *same kernel*; anything above zero is pure
    // measurement error. -1 means the check could not be run for this
    // signature (no same-kernel pair reached final measurement).
    double canary_pct    = -1.0;
    int    canary_retries = 0;
    std::string canary_pair;   // which two entries were compared
};

struct DigestHash {
    size_t operator()(const ggml_hip_digest & d) const {
        size_t v = 0;
        for (size_t i = 0; i < sizeof(size_t) && i < GGML_HIP_DIGEST_BYTES; ++i) {
            v |= (size_t) d.bytes[i] << (8 * i);
        }
        return v;
    }
};

struct DigestEqual {
    bool operator()(const ggml_hip_digest & a, const ggml_hip_digest & b) const {
        return ggml_hip_digest_equal(a, b);
    }
};

std::unordered_map<ggml_hip_digest, Result, DigestHash, DigestEqual> g_results;
std::mutex g_mutex;

// UTC, second resolution, ISO-ish -- only used to name a journal experiment
// uniquely per process, not parsed back by anything, so exact format is not
// load-bearing.
std::string utc_timestamp() {
    std::time_t now = std::time(nullptr);
    std::tm tm_utc;
#if defined(_WIN32)
    gmtime_s(&tm_utc, &now);
#else
    gmtime_r(&now, &tm_utc);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%dT%H%M%SZ", &tm_utc);
    return std::string(buf);
}

// A compact per-candidate summary for the journal's "result" event -- not
// the full ~80-field measurements-file record (that is built separately by
// ggml_hip_tuner_flush()), just enough to identify which candidate won,
// why, and how much evidence backed the answer, so a killed run's partial
// journal is still useful without a human re-deriving it from raw samples.
std::string journal_result_summary(
        const ggml_hip_digest & dispatch_digest, const Result & result) {
    return std::string("{\"kind\":\"result\",\"dispatch\":\"") +
        ggml_hip_digest_hex(dispatch_digest) +
        "\",\"winner\":\"" +
        (result.winner != nullptr ? result.winner->stable_name : "(none)") +
        "\",\"reason\":\"" + result.reason +
        "\",\"generation\":1" +
        ",\"generated\":" + std::to_string(result.generated) +
        ",\"applicable\":" + std::to_string(result.applicable) +
        ",\"eligible\":" + std::to_string(result.eligible) +
        ",\"measured\":" + std::to_string(result.measured) +
        ",\"improvement_pct\":" + std::to_string(result.improvement_pct) +
        ",\"confidence\":" + std::to_string(result.confidence) + "}";
}

// Called from every g_results.emplace() site (there are several -- most are
// early "native not eligible/rejected" exits). Emplacing into g_results and
// appending to the journal are two independent, differently-mutexed
// operations on purpose: g_mutex protects the in-memory map that
// ggml_hip_tuner_flush() reads at shutdown, while the journal's own mutex
// (inside hip-autotune-journal.cpp) guards the file handle -- holding
// g_mutex across a filesystem fsync would serialise every concurrent
// dispatch behind disk I/O for no reason.
void record_result(const ggml_hip_digest & dispatch_digest, const Result & result) {
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_results.emplace(dispatch_digest, result);
    }

    // Lazy, attempt-once open: the common case (no GGML_HIP_DISPATCH_DB, or
    // a tune run with journaling not requested) must cost nothing beyond
    // this one atomic check on every subsequent call.
    static std::atomic<bool> journal_open_attempted{false};
    if (!journal_open_attempted.exchange(true)) {
        const char * db_path = getenv("GGML_HIP_DISPATCH_DB");
        if (db_path != nullptr && db_path[0] != '\0') {
            const std::string journal_path = std::string(db_path) + ".journal.jsonl";
            const std::string experiment_id = "tune-" + utc_timestamp();
            if (!ggml_hip_journal_open(
                    journal_path.c_str(), experiment_id,
                    GGML_HIP_AUTOTUNE_SOURCE_REVISION_STR,
                    GGML_HIP_AUTOTUNE_MANIFEST_HASH_STR, result.hardware_digest)) {
                GGML_LOG_WARN("bigcherry: cannot open tuning journal '%s'; "
                              "a killed run will not be incrementally recoverable\n",
                              journal_path.c_str());
            }
        }
    }
    if (ggml_hip_journal_is_open()) {
        ggml_hip_journal_append_result(journal_result_summary(dispatch_digest, result));
    }
}

const char * reason_name(ggml_hip_reject_reason r) {
    switch (r) {
        case GGML_HIP_REJECT_NONE:          return "ok";
        case GGML_HIP_REJECT_ARCHITECTURE:  return "architecture";
        case GGML_HIP_REJECT_INELIGIBLE:    return "ineligible";
        case GGML_HIP_REJECT_WORKSPACE:     return "workspace";
        case GGML_HIP_REJECT_LAUNCH_FAILED: return "launch_failed";
        case GGML_HIP_REJECT_NAN_INF:       return "nan_inf";
        case GGML_HIP_REJECT_TOLERANCE:     return "tolerance";
        case GGML_HIP_REJECT_UNSTABLE:      return "unstable";
        case GGML_HIP_REJECT_NOISY:         return "noisy";
        default:                            return "?";
    }
}

// E1: NaN marks a round where a candidate failed to launch (position must
// still equal round number, so the pairing survives). std::sort over a range
// containing NaN is undefined behaviour, not merely an odd answer, so every
// statistic below filters first.
std::vector<double> finite_only(const std::vector<double> & v) {
    std::vector<double> out;
    out.reserve(v.size());
    for (double x : v) {
        if (!std::isnan(x)) out.push_back(x);
    }
    return out;
}

double median_of(const std::vector<double> & v_in) {
    std::vector<double> v = finite_only(v_in);
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double percentile_of(const std::vector<double> & v_in, double p) {
    std::vector<double> v = finite_only(v_in);
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t idx = (size_t) (p * (double) (v.size() - 1));
    return v[idx];
}

// Median absolute deviation. Preferred over stddev here because a single
// preemption or clock excursion makes stddev meaningless while MAD barely
// moves -- and GPU timings are full of both.
double mad_of(const std::vector<double> & v_in, double median) {
    std::vector<double> v = finite_only(v_in);
    if (v.empty()) return 0.0;
    std::vector<double> deviations;
    deviations.reserve(v.size());
    for (double x : v) {
        deviations.push_back(std::fabs(x - median));
    }
    return median_of(deviations);
}

// E1: P(X >= k) for X ~ Binomial(n, 0.5), exact. Summed downward from i = n
// with a running ratio, so no binomial coefficient is ever materialised and
// nothing overflows for any n the tuner can produce.
//   C(n, i-1) = C(n, i) * i / (n - i + 1)
double binomial_tail_ge(int k, int n) {
    if (k <= 0) return 1.0;
    if (k >  n) return 0.0;
    double term = std::exp(-(double) n * std::log(2.0));   // C(n,n) / 2^n
    double sum  = term;
    for (int i = n; i > k; --i) {
        term *= (double) i / (double) (n - i + 1);
        sum  += term;
    }
    return std::min(sum, 1.0);
}

// E1: paired sign test over the interleaved final rounds. Only the direction
// of each paired difference is used, so one preempted round costs a single
// vote instead of dominating a mean -- the same reasoning that put MAD in
// this file rather than stddev. Returns the one-sided p-value for "candidate
// is faster than baseline", and 1.0 when too few usable rounds survive to
// say anything.
double paired_sign_test(const std::vector<double> & baseline,
                        const std::vector<double> & candidate,
                        int min_rounds, int & wins_out, int & rounds_out) {
    wins_out = rounds_out = 0;
    const size_t rounds = std::min(baseline.size(), candidate.size());
    int wins = 0;
    int n    = 0;
    for (size_t r = 0; r < rounds; ++r) {
        if (std::isnan(baseline[r]) || std::isnan(candidate[r])) continue;
        if (candidate[r] == baseline[r]) continue;   // tie: excluded, standard
        if (candidate[r] < baseline[r]) ++wins;
        ++n;
    }
    wins_out = wins; rounds_out = n;
    return n < min_rounds ? 1.0 : binomial_tail_ge(wins, n);
}

// E5: aggregate the per-candidate reject reasons so a 1151-signature sweep is
// readable during the run rather than only after post-processing.
void reject_counts(const Result & r, int counts[GGML_HIP_REJECT_COUNT]) {
    std::fill(counts, counts + GGML_HIP_REJECT_COUNT, 0);
    for (const Measurement & m : r.measurements) {
        ++counts[m.reason];
    }
}

// Compare a candidate's output against native's, on the host.
//
// Both NMSE and max absolute error, because they fail differently: NMSE catches
// a systematically wrong result, max-abs catches a single catastrophic element
// that averaging would hide.
bool compare_outputs(const std::vector<float> & reference,
                     const std::vector<float> & actual,
                     double & nmse, double & max_abs) {
    nmse = 0.0;
    max_abs = 0.0;
    if (reference.size() != actual.size() || reference.empty()) {
        return false;
    }

    double sq_error = 0.0;
    double sq_ref   = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double a = actual[i];
        if (std::isnan(a) || std::isinf(a)) {
            return false;   // standards 7.2: introduced NaN/Inf is fatal
        }
        const double d = a - reference[i];
        sq_error += d * d;
        sq_ref   += (double) reference[i] * reference[i];
        max_abs = std::max(max_abs, std::fabs(d));
    }
    nmse = sq_ref > 0.0 ? sq_error / sq_ref : sq_error;
    return true;
}

// Diagnostic only -- attributes a device-side crash (illegal memory access,
// XNACK fault) to the candidate that caused it. Not a replacement for the
// crash-safe journal, a companion to it: a candidate that faults never
// reaches record_result(), so the journal's last "result" entry alone is one
// candidate short of the truth. This writes an "attempt" event into the SAME
// journal, *before* the risky GPU work, so the crashing candidate is the
// last line even when the process never executes another line of C++
// afterward.
//
// Only writes once the journal is already open -- opening it here would need
// a hardware digest that does not exist until a candidate has completed at
// least once (see record_result), and plumbing that into this hot path for
// a diagnostic-only feature is not worth it. Practical effect: the very
// first candidate of a run has no attempt coverage; every candidate after
// the first completed result does. Off by default (one getenv + atomic
// check when unset) -- do not enable for a real tuning run, fsync-per-launch
// is the wrong trade when nothing is crashing. Correlate against
// `HIP_LAUNCH_BLOCKING=1`: without it, several launches can be in-flight on
// the device ahead of the one that actually faults, and the last logged
// attempt would name an innocent candidate that merely happened to be
// dispatched most recently, not the one whose kernel corrupted memory.
void trace_launch_attempt(const char * stable_name) {
    static std::atomic<bool> enabled{false};
    static std::atomic<bool> checked{false};
    if (!checked.exchange(true)) {
        const char * flag = getenv("GGML_HIP_TUNE_TRACE_ATTEMPTS");
        enabled = (flag != nullptr && flag[0] != '\0' && flag[0] != '0');
    }
    if (!enabled || !ggml_hip_journal_is_open()) return;
    const std::string payload =
        "{\"candidate\":\"" + std::string(stable_name ? stable_name : "(null)") +
        "\",\"t_us\":" + std::to_string(ggml_time_us()) + "}";
    ggml_hip_journal_append_attempt(payload);
}

// Launch one candidate into scratch and time the complete path.
//
// `lc` already points at scratch, so the caller's real destination is never
// touched -- a candidate that produces garbage must not corrupt the run it is
// being measured inside.
bool time_candidate(const ggml_hip_candidate_descriptor * candidate,
                    const ggml_hip_launch_context & lc,
                    int warmup, int samples, int launches_per_sample,
                    std::vector<double> & gpu_us,
                    std::vector<double> & host_us) {
    trace_launch_attempt(candidate ? candidate->stable_name : nullptr);

    hipEvent_t start;
    hipEvent_t stop;
    if (hipEventCreate(&start) != hipSuccess) return false;
    if (hipEventCreate(&stop)  != hipSuccess) { hipEventDestroy(start); return false; }

    ggml_hip_candidate_descriptor effective = *candidate;

    for (int i = 0; i < warmup; ++i) {
        effective.launch(&effective, lc);
    }
    if (hipStreamSynchronize(lc.stream) != hipSuccess) {
        hipEventDestroy(start); hipEventDestroy(stop);
        return false;
    }
    if (hipGetLastError() != hipSuccess) {
        hipEventDestroy(start); hipEventDestroy(stop);
        return false;
    }

    gpu_us.reserve(samples);
    host_us.reserve(samples);

    for (int s = 0; s < samples; ++s) {
        const int64_t host_start = ggml_time_us();
        hipEventRecord(start, lc.stream);
        // Several launches per sample when one kernel is below event
        // resolution; the mean of the batch is the sample.
        for (int i = 0; i < launches_per_sample; ++i) {
            effective.launch(&effective, lc);
        }
        hipEventRecord(stop, lc.stream);
        if (hipEventSynchronize(stop) != hipSuccess) {
            hipEventDestroy(start); hipEventDestroy(stop);
            return false;
        }
        const int64_t host_end = ggml_time_us();

        float ms = 0.0f;
        hipEventElapsedTime(&ms, start, stop);
        gpu_us.push_back((double) ms * 1000.0 / (double) launches_per_sample);
        host_us.push_back((double) (host_end - host_start)
                          / (double) launches_per_sample);
    }

    hipEventDestroy(start);
    hipEventDestroy(stop);
    return hipGetLastError() == hipSuccess;
}

// E3: the floor cost of one timed sample with no work in it -- two event
// records, a synchronize, and the two host clock reads. Subtracting it is
// what makes a host-side number comparable with a GPU-side one. Measured
// once per process on the tuning stream; cheap, and constant enough that a
// per-signature recalibration would only add noise. Cached at namespace
// scope (not function-local) so `ggml_hip_tuner_flush` can report it in the
// measurements header for audit.
double g_host_sync_overhead_us = -1.0;

double host_sync_overhead_us(cudaStream_t stream) {
    if (g_host_sync_overhead_us >= 0.0) return g_host_sync_overhead_us;

    hipEvent_t a, b;
    hipEventCreate(&a); hipEventCreate(&b);
    std::vector<double> samples;
    for (int i = 0; i < 20; ++i) {
        const int64_t t0 = ggml_time_us();
        hipEventRecord(a, stream);
        hipEventRecord(b, stream);
        hipEventSynchronize(b);
        samples.push_back((double) (ggml_time_us() - t0));
    }
    hipEventDestroy(a); hipEventDestroy(b);
    g_host_sync_overhead_us = median_of(samples);
    return g_host_sync_overhead_us;
}

// E3: max(), not a mode switch -- whichever resource is binding is the one
// that bounds the call, and the other term is smaller by construction. When
// the host runs ahead of the GPU (large kernels, deep queue) the second term
// is smaller and the GPU number wins; when submission dominates (tiny
// kernels) the host term wins.
double effective_us_of(double median_us, double host_median_us, double sync_overhead_us) {
    return std::max(median_us, host_median_us - sync_overhead_us);
}

// E4: one launch, then fetch the result to host. Used only for the winner's
// determinism recheck -- two launches and two copies, not the full timing
// harness above.
bool launch_and_fetch(const ggml_hip_candidate_descriptor * candidate,
                      const ggml_hip_launch_context & lc,
                      void * dst_device, size_t dst_bytes,
                      std::vector<float> & out_host) {
    ggml_hip_candidate_descriptor effective = *candidate;
    effective.launch(&effective, lc);
    if (hipStreamSynchronize(lc.stream) != hipSuccess) return false;
    if (hipGetLastError() != hipSuccess) return false;
    return hipMemcpy(out_host.data(), dst_device, dst_bytes,
                     hipMemcpyDeviceToHost) == hipSuccess;
}

// E2: raw finalist samples as a JSON array, so a winner is recomputable
// offline without a GPU. NaN (a round where the candidate failed to launch)
// serialises as `null` -- JSON has no NaN literal, and dropping the entry
// entirely would silently reintroduce the round-misalignment bug E1 fixes.
std::string samples_json(const std::vector<double> & v) {
    std::string out = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) out += ",";
        if (std::isnan(v[i])) {
            out += "null";
        } else {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.3f", v[i]);
            out += buf;
        }
    }
    out += "]";
    return out;
}

} // namespace

const ggml_hip_tuner_config & ggml_hip_tuner_get_config() {
    static ggml_hip_tuner_config config = [] {
        ggml_hip_tuner_config c;
        // Environment overrides exist so a long production tune can be traded
        // against precision without a rebuild.
        if (const char * v = getenv("GGML_HIP_TUNE_FINAL_SAMPLES")) {
            c.final_samples = atoi(v);
        }
        if (const char * v = getenv("GGML_HIP_TUNE_SCREEN_SAMPLES")) {
            c.screen_samples = atoi(v);
        }
        if (const char * v = getenv("GGML_HIP_TUNE_MAX_WORKSPACE")) {
            c.max_workspace_bytes = (size_t) atoll(v);
        }
        if (const char * v = getenv("GGML_HIP_TUNE_NOISE_PCT")) {
            c.noise_canary_pct = atof(v);
        }
        if (const char * v = getenv("GGML_HIP_TUNE_ALPHA")) {
            c.confidence_alpha = atof(v);
        }
        if (const char * v = getenv("GGML_HIP_TUNE_NOISY_MAD")) {
            c.noisy_mad_ratio = atof(v);
        }
        if (const char * v = getenv("GGML_HIP_TUNE_VERIFY_DETERMINISM")) {
            c.verify_determinism = atoi(v);
        }
        if (const char * v = getenv("GGML_HIP_TUNE_EMIT_SAMPLES")) {
            c.emit_samples = atoi(v);
        }
        return c;
    }();
    return config;
}

const ggml_hip_candidate_descriptor * ggml_hip_tuner_resolve(
        ggml_backend_cuda_context & ctx,
        const ggml_hip_dispatch_signature_v1 & sig,
        const ggml_hip_hardware_key_v1 & hw,
        const ggml_hip_digest & dispatch_digest,
        const ggml_hip_native_selection & native,
        const ggml_hip_launch_context & lc_in) {
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto found = g_results.find(dispatch_digest);
        if (found != g_results.end()) {
            return found->second.winner;
        }
    }

    const ggml_hip_tuner_config & config = ggml_hip_tuner_get_config();
    Result result;
    result.winner = native.candidate;
    // Recomputed rather than threaded down from the resolver: the two are the
    // same values, and recomputing here keeps this function's signature stable
    // for a field the caller has no other use for. Cold path, once per
    // signature.
    result.signature_digest = ggml_hip_signature_digest(sig);
    result.hardware_digest  = ggml_hip_hardware_digest(hw);
    result.canonical_json   = ggml_hip_signature_json(sig, true);

    // Held for the whole run, not just around each launch.
    //
    // Every measurement launch goes through a family entry point, and HI13
    // made those entry points collection sites. Without this the first
    // measured launch re-enters the hook, which resolves, which calls the
    // tuner, which launches again -- unbounded recursion, and the stack dies
    // inside the HSA runtime where no bigcherry frame is visible. That is
    // exactly how this was found.
    const ggml_hip_dispatch_scope no_reentry;

    // Runs once per dispatch key, inline, on the first encounter. It stalls
    // that one execution; the resolver's process cache makes every later one
    // free. Inline rather than on a background thread so the measured launches
    // sit on the same stream and device as the real work -- which is the only
    // way the numbers describe production rather than a laboratory.

    // --- gather the candidate set ---------------------------------------
    //
    // Whether other families compete depends on whether this operation is
    // fused, and the two rules pull in opposite directions:
    //
    //   plan 11.3   -- for an ordinary signature, include every legal family
    //                  default and BLAS auto. Upstream picks the family from a
    //                  heuristic ladder (`should_use_mmq` and friends), and
    //                  measuring whether that heuristic is right for a given
    //                  shape is the single biggest lever this project has.
    //                  Restricting to native's family throws it away.
    //
    //   standards 11.1 -- for a *fused* pattern, tune only within the family
    //                  already selected. A fused MMVQ is a different semantic
    //                  operation, and comparing it against an unfused MMQ
    //                  decomposition is a graph-level question that the matmul
    //                  tuner is not entitled to answer.
    //
    // So: cross-family for unfused operations, single-family for fused ones.
    const bool fused = sig.fusion != GGML_HIP_FUSION_NONE;

    for (size_t i = 0; i < ggml_hip_registry_size(); ++i) {
        const ggml_hip_candidate_descriptor * candidate = ggml_hip_registry_at(i);

        if (fused && candidate->family != native.candidate->family) {
            continue;
        }
        ++result.generated;

        Measurement m;
        m.candidate = candidate;

        const bool arch_ok = ggml_hip_candidate_supports_arch(*candidate, hw);

        // E5: a candidate is *applicable* when it could plausibly serve this
        // signature at all: right architecture, and built for this src0 type.
        // Without this counter `generated` is the registry size and never
        // varies, which is why every signature in TUNING-DETAIL.md reports
        // 270 and why COVERAGE_AUDIT's "generated high, eligible low"
        // diagnostic can never fire. variant.src0_type is 0 for native
        // wrappers, which are admitted unconditionally (hip-autotune-types.h:197).
        const bool type_ok = candidate->variant.src0_type == 0
                          || candidate->variant.src0_type == sig.src0_type;
        if (arch_ok && type_ok) {
            ++result.applicable;
        }

        if (!arch_ok) {
            m.reason = GGML_HIP_REJECT_ARCHITECTURE;
        } else if (!candidate->can_execute(candidate, sig, hw)) {
            m.reason = GGML_HIP_REJECT_INELIGIBLE;
        } else {
            m.workspace_bytes = candidate->workspace(candidate, sig);
            if (config.max_workspace_bytes != 0
                    && m.workspace_bytes > config.max_workspace_bytes) {
                m.reason = GGML_HIP_REJECT_WORKSPACE;
            } else {
                ++result.eligible;
            }
        }
        result.measurements.push_back(m);
    }

    std::vector<Measurement *> screening;
    for (Measurement & m : result.measurements) {
        if (m.reason == GGML_HIP_REJECT_NONE) {
            screening.push_back(&m);
        }
    }

    Measurement * native_m = nullptr;
    for (Measurement * m : screening) {
        if (m->candidate == native.candidate) { native_m = m; break; }
    }
    if (native_m == nullptr) {
        // Standards 7.3: without a measured native there is no correctness
        // reference and no baseline, so this signature is rejected rather than
        // producing a winner chosen against nothing.
        result.reason = "native not eligible; run rejected";
        record_result(dispatch_digest, result);
        return native.candidate;
    }

    // --- scratch destinations -------------------------------------------
    //
    // Two buffers: one holds native's output as the correctness reference, the
    // other receives each candidate under test. The caller's real destination
    // is never written by a measurement, so a candidate producing garbage
    // cannot corrupt the run it is being measured inside.
    const size_t dst_bytes = ggml_nbytes(lc_in.dst);
    const size_t dst_floats = dst_bytes / sizeof(float);

    ggml_cuda_pool_alloc<char> reference_buf(ctx.pool(), dst_bytes);
    ggml_cuda_pool_alloc<char> candidate_buf(ctx.pool(), dst_bytes);

    std::vector<float> reference_host(dst_floats);
    std::vector<float> candidate_host(dst_floats);

    ggml_tensor scratch_dst = *lc_in.dst;
    ggml_hip_launch_context lc = lc_in;
    lc.dst = &scratch_dst;

    // --- native reference, and native's own timing ----------------------
    scratch_dst.data = reference_buf.get();
    {
        std::vector<double> gpu;
        std::vector<double> host;
        if (!time_candidate(native.candidate, lc, config.warmup_launches,
                            config.screen_samples, config.launches_per_sample,
                            gpu, host)) {
            native_m->reason = GGML_HIP_REJECT_LAUNCH_FAILED;
            result.reason = "native failed to launch; run rejected";
            record_result(dispatch_digest, result);
            return native.candidate;
        }
        native_m->median_us      = median_of(gpu);
        native_m->mad_us         = mad_of(gpu, native_m->median_us);
        native_m->p95_us         = percentile_of(gpu, 0.95);
        native_m->host_median_us = median_of(host);
        native_m->samples        = (int) gpu.size();
        native_m->measured       = true;
        ++result.measured;

        CUDA_CHECK(hipMemcpyAsync(reference_host.data(), reference_buf.get(),
                                  dst_bytes, hipMemcpyDeviceToHost, lc.stream));
        CUDA_CHECK(hipStreamSynchronize(lc.stream));
    }

    // --- screening (standards 11.4) -------------------------------------
    scratch_dst.data = candidate_buf.get();
    for (Measurement * m : screening) {
        if (m->candidate == native.candidate) {
            continue;
        }
        std::vector<double> gpu;
        std::vector<double> host;
        if (!time_candidate(m->candidate, lc, config.warmup_launches,
                            config.screen_samples, config.launches_per_sample,
                            gpu, host)) {
            m->reason = GGML_HIP_REJECT_LAUNCH_FAILED;
            continue;
        }

        CUDA_CHECK(hipMemcpyAsync(candidate_host.data(), candidate_buf.get(),
                                  dst_bytes, hipMemcpyDeviceToHost, lc.stream));
        CUDA_CHECK(hipStreamSynchronize(lc.stream));

        double nmse = 0.0;
        double max_abs = 0.0;
        if (!compare_outputs(reference_host, candidate_host, nmse, max_abs)) {
            m->reason = GGML_HIP_REJECT_NAN_INF;
            continue;
        }
        m->nmse          = nmse;
        m->max_abs_error = max_abs;
        if (nmse > config.max_nmse || max_abs > config.max_abs_error) {
            m->reason = GGML_HIP_REJECT_TOLERANCE;
            continue;
        }

        m->median_us      = median_of(gpu);
        m->mad_us         = mad_of(gpu, m->median_us);
        m->p95_us         = percentile_of(gpu, 0.95);
        m->host_median_us = median_of(host);
        m->samples        = (int) gpu.size();
        m->measured       = true;
        ++result.measured;
    }

    // --- retain finalists -----------------------------------------------
    //
    // native always, plus the top 3 by median, plus everything within 10% of
    // the best. Finalist count dominates total tune time, and without this the
    // final stage would re-measure every candidate at 100 samples.
    std::vector<Measurement *> survivors;
    for (Measurement * m : screening) {
        if (m->measured) survivors.push_back(m);
    }
    std::sort(survivors.begin(), survivors.end(),
              [](const Measurement * a, const Measurement * b) {
                  return a->median_us < b->median_us;
              });

    std::vector<Measurement *> finalists;
    const double best = survivors.empty() ? 0.0 : survivors.front()->median_us;
    for (size_t i = 0; i < survivors.size(); ++i) {
        Measurement * m = survivors[i];
        const bool is_native = m->candidate == native.candidate;
        const bool in_top    = (int) i < config.screen_keep_top;
        const bool near_best = best > 0.0 &&
            m->median_us <= best * (1.0 + config.screen_keep_within_pct / 100.0);
        if (is_native || in_top || near_best) {
            finalists.push_back(m);
        }
    }

    // --- final measurement (standards 11.5, 8.2) ------------------------
    //
    // Interleaved rather than one candidate at a time. Measuring A to
    // completion and then B lets thermal drift or a clock change across the
    // run masquerade as a difference between them.
    if (finalists.size() > 1) {
        for (Measurement * m : finalists) {
            m->final_gpu_us.assign(config.final_samples,
                                   std::numeric_limits<double>::quiet_NaN());
            m->final_host_us.assign(config.final_samples,
                                    std::numeric_limits<double>::quiet_NaN());
        }
        for (int round = 0; round < config.final_samples; ++round) {
            for (Measurement * m : finalists) {
                std::vector<double> g1;
                std::vector<double> h1;
                if (time_candidate(m->candidate, lc, 0, 1,
                                   config.launches_per_sample, g1, h1) && !g1.empty()) {
                    m->final_gpu_us[round]  = g1[0];
                    m->final_host_us[round] = h1[0];
                }
                // else leave NaN: position must equal round, or E1's paired
                // sign test would silently compare round 7 of one candidate
                // against round 8 of another. Medians never noticed; the sign
                // test would be quietly wrong instead of loudly so.
            }
        }
        for (Measurement * m : finalists) {
            if (finite_only(m->final_gpu_us).empty()) continue;
            m->median_us      = median_of(m->final_gpu_us);
            m->mad_us         = mad_of(m->final_gpu_us, m->median_us);
            m->p95_us         = percentile_of(m->final_gpu_us, 0.95);
            m->host_median_us = median_of(m->final_host_us);
            m->samples        = (int) finite_only(m->final_gpu_us).size();
        }
    }

    // E4: dispersion rejection, after the final stage. A candidate whose own
    // spread is wider than the margins being decided cannot be ranked,
    // however good its median looks.
    for (Measurement * m : finalists) {
        if (!m->measured || m->median_us <= 0.0) continue;
        if (m->mad_us / m->median_us > config.noisy_mad_ratio) {
            m->reason = GGML_HIP_REJECT_NOISY;
        }
    }
    if (native_m->reason == GGML_HIP_REJECT_NOISY) {
        // Same treatment as an unmeasurable native (standards 7.3): a
        // baseline that cannot be pinned down is not a baseline, and every
        // improvement_pct in this signature would be computed against a
        // number that moved.
        result.reason = "native timing unstable; run rejected";
        record_result(dispatch_digest, result);
        return native.candidate;
    }

    // --- noise canary (HI24) --------------------------------------------
    //
    // Native and a forced MMQ candidate at J == J_best are the *same kernel*:
    // `mul_mat_q_switch_J` overwrites J_best with forced_J and calls one
    // launcher, so forcing J_best is native rather than merely equivalent to
    // it (RV21). Any divergence between their medians is therefore measurement
    // error, and needs no external reference to calibrate.
    //
    // This is worth more than a repeatability check on native alone, because
    // it also holds the forced path to producing native's timing -- the
    // invariant the whole dispatch design rests on. If an upstream change ever
    // breaks it, this fires constantly, which is the correct alarm.
    //
    // Costs nothing when the pair is already present: both were going to be
    // measured anyway.
    {
        Measurement * twin = nullptr;
        if (native.candidate != nullptr &&
                native.candidate->family == GGML_HIP_FAMILY_MMQ) {
            const bool fb = (sig.ne0[1] % 128) != 0;
            const int j_best = ggml_cuda_mmq_native_j_best(
                (ggml_type) sig.src0_type, fb, sig.ned[1]);
            if (j_best != 0) {
                for (Measurement * m : finalists) {
                    if (m == native_m || !m->measured || m->candidate == nullptr) {
                        continue;
                    }
                    if (m->candidate->family == GGML_HIP_FAMILY_MMQ &&
                            m->candidate->variant.primary == j_best &&
                            (m->candidate->variant.fallback != 0) == fb) {
                        twin = m;
                        break;
                    }
                }
            }
        }

        for (int attempt = 0; twin != nullptr && native_m->median_us > 0.0; ++attempt) {
            const double diff = std::fabs(native_m->median_us - twin->median_us);
            result.canary_pct  = 100.0 * diff / native_m->median_us;
            result.canary_pair = twin->candidate->stable_name;
            if (result.canary_pct <= config.noise_canary_pct ||
                    attempt >= config.noise_canary_retries) {
                if (result.canary_pct > config.noise_canary_pct) {
                    // Report rather than discard. The winner may still be
                    // right; what is established is that this signature's
                    // margins are not resolvable at these sample counts.
                    GGML_LOG_WARN("bigcherry: noise canary %.1f%% on this "
                                  "signature (native vs %s, identical "
                                  "kernels); timings are unreliable at these "
                                  "sample counts\n",
                                  result.canary_pct, result.canary_pair.c_str());
                }
                break;
            }
            // Re-measure the pair interleaved, exactly as the final stage does.
            ++result.canary_retries;
            Measurement * pair[2] = { native_m, twin };
            std::vector<std::vector<double>> g(2), h(2);
            for (int round = 0; round < config.final_samples; ++round) {
                for (int i = 0; i < 2; ++i) {
                    std::vector<double> g1, h1;
                    if (time_candidate(pair[i]->candidate, lc, 0, 1,
                                       config.launches_per_sample, g1, h1)) {
                        g[i].insert(g[i].end(), g1.begin(), g1.end());
                        h[i].insert(h[i].end(), h1.begin(), h1.end());
                    }
                }
            }
            for (int i = 0; i < 2; ++i) {
                if (g[i].empty()) continue;
                pair[i]->median_us      = median_of(g[i]);
                pair[i]->mad_us         = mad_of(g[i], pair[i]->median_us);
                pair[i]->p95_us         = percentile_of(g[i], 0.95);
                pair[i]->host_median_us = median_of(h[i]);
                pair[i]->samples        = (int) g[i].size();
            }
        }
    }

    // E3: rank on whichever resource is actually binding, computed after the
    // canary section above (which can re-measure native and its twin) so
    // this is never stale relative to what median_us/host_median_us hold.
    // For most candidates host_median ~= gpu_median + sync_overhead and the
    // adjustment does nothing; it matters when host-side submission cost is
    // a material fraction of a tiny call (TUNING-DETAIL.md's 12.36us
    // signatures).
    {
        const double sync_overhead = host_sync_overhead_us(lc.stream);
        for (Measurement * m : finalists) {
            if (!m->measured) continue;
            m->effective_us = effective_us_of(m->median_us, m->host_median_us,
                                              sync_overhead);
        }
    }

    // --- winner selection (standards 7.3, HI12 E1) -----------------------
    Measurement * best_m = native_m;
    for (Measurement * m : finalists) {
        if (m->measured && m->reason == GGML_HIP_REJECT_NONE &&
                m->effective_us < best_m->effective_us) {
            best_m = m;
        }
    }

    Measurement * winner_m = nullptr;
    bool any_short_rounds = false;

    if (best_m != native_m) {
        // Near-tie resolution: everything within tie_pct of the best, ordered
        // by p95, then workspace, then native-preferred, then name. Median
        // alone is a coin flip inside the noise band; p95 is what a latency
        // budget actually feels.
        std::vector<Measurement *> tied;
        for (Measurement * m : finalists) {
            if (m->measured && m->reason == GGML_HIP_REJECT_NONE &&
                    m->effective_us <= best_m->effective_us * (1.0 + config.tie_pct / 100.0)) {
                tied.push_back(m);
            }
        }

        auto tie_order = [&](const Measurement * a, const Measurement * b) {
            if (a->p95_us != b->p95_us) return a->p95_us < b->p95_us;
            if (a->workspace_bytes != b->workspace_bytes) {
                return a->workspace_bytes < b->workspace_bytes;
            }
            const bool an = a->candidate == native.candidate;
            const bool bn = b->candidate == native.candidate;
            if (an != bn) return an;   // native wins a genuine tie
            return strcmp(a->candidate->stable_name,
                          b->candidate->stable_name) < 0;
        };

        // E1: effect size and consistency are different questions and both
        // must answer yes. Effect without consistency is RV21: a 14% "win"
        // between two runs of the same kernel. Consistency without effect is
        // a reproducible 0.05% that is not worth a cache entry. The sign
        // test runs over the raw paired GPU rounds (final_gpu_us); the
        // effect-size gate runs over effective_us.
        std::vector<Measurement *> qualified;
        for (Measurement * m : tied) {
            if (m == native_m) { qualified.push_back(m); continue; }
            int wins = 0, rounds = 0;
            m->sign_p = paired_sign_test(native_m->final_gpu_us, m->final_gpu_us,
                                         config.min_paired_rounds, wins, rounds);
            m->sign_wins   = wins;
            m->sign_rounds = rounds;
            if (rounds < config.min_paired_rounds) any_short_rounds = true;
            const double impr = native_m->effective_us > 0.0
                ? 100.0 * (native_m->effective_us - m->effective_us) / native_m->effective_us
                : 0.0;
            if (impr >= config.replacement_threshold_pct &&
                    m->sign_p <= config.confidence_alpha) {
                qualified.push_back(m);
            }
        }
        std::sort(qualified.begin(), qualified.end(), tie_order);

        // E4: winner-only determinism recheck. Bounded at two extra launches
        // regardless of catalog size -- the winner is the only candidate
        // whose determinism this run ends up asserting in a shipped cache.
        for (;;) {
            winner_m = qualified.empty() ? nullptr : qualified.front();
            if (winner_m == nullptr || winner_m == native_m ||
                    !config.verify_determinism || !winner_m->candidate->deterministic) {
                break;
            }
            std::vector<float> first(dst_floats);
            std::vector<float> second(dst_floats);
            const bool launched =
                launch_and_fetch(winner_m->candidate, lc, candidate_buf.get(),
                                 dst_bytes, first) &&
                launch_and_fetch(winner_m->candidate, lc, candidate_buf.get(),
                                 dst_bytes, second);
            if (!launched) {
                winner_m->reason = GGML_HIP_REJECT_LAUNCH_FAILED;
                qualified.erase(qualified.begin());
                continue;
            }
            if (std::memcmp(first.data(), second.data(),
                            dst_floats * sizeof(float)) != 0) {
                winner_m->reason = GGML_HIP_REJECT_UNSTABLE;
                qualified.erase(qualified.begin());
                continue;
            }
            break;
        }
    }

    if (winner_m == nullptr || winner_m == native_m) {
        result.winner          = native.candidate;
        result.improvement_pct = 0.0;
        result.reason          = any_short_rounds
            ? "native retained (too few paired rounds to confirm any challenger)"
            : "native retained";
    } else {
        result.winner          = winner_m->candidate;
        result.improvement_pct = native_m->effective_us > 0.0
            ? 100.0 * (native_m->effective_us - winner_m->effective_us) / native_m->effective_us
            : 0.0;
        result.confidence      = 1.0 - winner_m->sign_p;
        result.reason          = "measured winner, " +
            std::to_string(winner_m->sign_wins) + "/" +
            std::to_string(winner_m->sign_rounds) + " paired rounds" +
            (winner_m->candidate->family == native.candidate->family
                ? "" : " (different family from native)");
    }

    {
        int counts[GGML_HIP_REJECT_COUNT];
        reject_counts(result, counts);
        GGML_LOG_INFO("bigcherry: tuned %s -- gen/appl/elig/meas %d/%d/%d/%d, "
                      "rejects arch=%d inelig=%d ws=%d launch=%d nan=%d tol=%d "
                      "noisy=%d unstable=%d, winner %s (%.2f%% vs native, "
                      "confidence=%.4f)\n",
                      ggml_hip_digest_hex(dispatch_digest).c_str(),
                      result.generated, result.applicable, result.eligible,
                      result.measured,
                      counts[GGML_HIP_REJECT_ARCHITECTURE],
                      counts[GGML_HIP_REJECT_INELIGIBLE],
                      counts[GGML_HIP_REJECT_WORKSPACE],
                      counts[GGML_HIP_REJECT_LAUNCH_FAILED],
                      counts[GGML_HIP_REJECT_NAN_INF],
                      counts[GGML_HIP_REJECT_TOLERANCE],
                      counts[GGML_HIP_REJECT_NOISY],
                      counts[GGML_HIP_REJECT_UNSTABLE],
                      result.winner->stable_name, result.improvement_pct,
                      result.confidence);
    }

    record_result(dispatch_digest, result);
    return result.winner;
}

void ggml_hip_tuner_flush() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_results.empty()) {
        return;
    }

    const char * path = getenv("GGML_HIP_DISPATCH_DB");
    if (path == nullptr || path[0] == '\0') {
        GGML_LOG_WARN("bigcherry: tuned %zu signature(s) but "
                      "GGML_HIP_DISPATCH_DB is unset; nothing written\n",
                      g_results.size());
        return;
    }

    std::string measurements_path = std::string(path) + ".measurements.jsonl";
    FILE * file = fopen(measurements_path.c_str(), "w");
    if (file == nullptr) {
        GGML_LOG_WARN("bigcherry: cannot write '%s'\n",
                      measurements_path.c_str());
        return;
    }

    const ggml_hip_tuner_config & config = ggml_hip_tuner_get_config();

    fprintf(file,
            "{\"kind\":\"header\",\"artifact_version\":%d,"
            "\"source_revision\":\"%s\",\"manifest_hash\":\"%s\","
            "\"compiler\":\"%s\",\"hip_version\":\"%s\","
            "\"host_sync_overhead_us\":%.3f,\"final_samples\":%d,"
            "\"alpha\":%.4f}\n",
            GGML_HIP_AUTOTUNE_ARTIFACT_VERSION,
            GGML_HIP_AUTOTUNE_SOURCE_REVISION_STR,
            GGML_HIP_AUTOTUNE_MANIFEST_HASH_STR,
            GGML_HIP_COMPILER_STR, GGML_HIP_VERSION_STR,
            g_host_sync_overhead_us, config.final_samples,
            config.confidence_alpha);

    for (const auto & entry : g_results) {
        const Result & r = entry.second;
        int counts[GGML_HIP_REJECT_COUNT];
        reject_counts(r, counts);

        // Coverage first, because it is what says whether the winner means
        // anything. generated/applicable/eligible/measured being far apart
        // is the signal that an eligibility predicate is too strict or a
        // candidate set never matched the observed shapes.
        fprintf(file,
                "{\"kind\":\"result\",\"dispatch\":\"%s\","
                "\"signature\":\"%s\",\"hardware\":\"%s\","
                "\"canonical\":%s,\"winner\":\"%s\","
                "\"improvement_pct\":%.3f,\"confidence\":%.4f,"
                "\"generated\":%d,\"applicable\":%d,\"eligible\":%d,"
                "\"measured\":%d,\"reason\":\"%s\","
                "\"rejects\":{\"architecture\":%d,\"ineligible\":%d,"
                "\"workspace\":%d,\"launch_failed\":%d,\"nan_inf\":%d,"
                "\"tolerance\":%d,\"noisy\":%d,\"unstable\":%d},"
                "\"canary_pct\":%.3f,\"canary_retries\":%d,"
                "\"canary_pair\":\"%s\",\"candidates\":[",
                ggml_hip_digest_hex(entry.first).c_str(),
                ggml_hip_digest_hex(r.signature_digest).c_str(),
                ggml_hip_digest_hex(r.hardware_digest).c_str(),
                r.canonical_json.c_str(),
                r.winner ? r.winner->stable_name : "",
                r.improvement_pct, r.confidence,
                r.generated, r.applicable, r.eligible, r.measured,
                r.reason.c_str(),
                counts[GGML_HIP_REJECT_ARCHITECTURE],
                counts[GGML_HIP_REJECT_INELIGIBLE],
                counts[GGML_HIP_REJECT_WORKSPACE],
                counts[GGML_HIP_REJECT_LAUNCH_FAILED],
                counts[GGML_HIP_REJECT_NAN_INF],
                counts[GGML_HIP_REJECT_TOLERANCE],
                counts[GGML_HIP_REJECT_NOISY],
                counts[GGML_HIP_REJECT_UNSTABLE],
                r.canary_pct, r.canary_retries, r.canary_pair.c_str());

        bool first = true;
        for (const Measurement & m : r.measurements) {
            fprintf(file,
                    "%s{\"name\":\"%s\",\"status\":\"%s\",\"median_us\":%.3f,"
                    "\"effective_us\":%.3f,"
                    "\"mad_us\":%.3f,\"p95_us\":%.3f,\"host_median_us\":%.3f,"
                    "\"nmse\":%.6g,\"max_abs\":%.6g,\"workspace\":%zu,"
                    "\"samples\":%d,\"sign_p\":%.4f,\"sign_wins\":%d,"
                    "\"sign_rounds\":%d",
                    first ? "" : ",",
                    m.candidate ? m.candidate->stable_name : "",
                    reason_name(m.reason), m.median_us, m.effective_us,
                    m.mad_us, m.p95_us,
                    m.host_median_us, m.nmse, m.max_abs_error,
                    m.workspace_bytes, m.samples, m.sign_p, m.sign_wins,
                    m.sign_rounds);
            if (config.emit_samples && !m.final_gpu_us.empty()) {
                fprintf(file, ",\"samples_us\":%s",
                        samples_json(m.final_gpu_us).c_str());
            }
            fprintf(file, "}");
            first = false;
        }
        fprintf(file, "]}\n");
    }
    fclose(file);

    GGML_LOG_INFO("bigcherry: wrote %zu tuning result(s) to '%s'\n",
                  g_results.size(), measurements_path.c_str());
}

#endif // GGML_USE_HIP && GGML_HIP_AUTOTUNE
