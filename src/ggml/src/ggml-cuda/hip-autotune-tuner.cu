// bigcherry: the tuning engine (HI12).

#include "hip-autotune-tuner.cuh"

#if defined(GGML_USE_HIP) && defined(GGML_HIP_AUTOTUNE)

#include "hip-autotune-build-hash.h"
#include "hip-autotune-dispatch.cuh"
#include "hip-autotune-signature.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <vector>

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
};

struct Result {
    const ggml_hip_candidate_descriptor * winner = nullptr;
    std::vector<Measurement> measurements;
    int generated  = 0;   // candidates in the registry for this family/shape
    int eligible   = 0;
    int measured   = 0;
    double improvement_pct = 0.0;
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
        default:                            return "?";
    }
}

double median_of(std::vector<double> & v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double percentile_of(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t idx = (size_t) (p * (double) (v.size() - 1));
    return v[idx];
}

// Median absolute deviation. Preferred over stddev here because a single
// preemption or clock excursion makes stddev meaningless while MAD barely
// moves -- and GPU timings are full of both.
double mad_of(const std::vector<double> & v, double median) {
    if (v.empty()) return 0.0;
    std::vector<double> deviations;
    deviations.reserve(v.size());
    for (double x : v) {
        deviations.push_back(std::fabs(x - median));
    }
    return median_of(deviations);
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

        if (!ggml_hip_candidate_supports_arch(*candidate, hw)) {
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
        std::lock_guard<std::mutex> lock(g_mutex);
        g_results.emplace(dispatch_digest, result);
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
            std::lock_guard<std::mutex> lock(g_mutex);
            g_results.emplace(dispatch_digest, result);
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
        std::vector<std::vector<double>> gpu(finalists.size());
        std::vector<std::vector<double>> host(finalists.size());
        for (int round = 0; round < config.final_samples; ++round) {
            for (size_t i = 0; i < finalists.size(); ++i) {
                std::vector<double> g1;
                std::vector<double> h1;
                if (time_candidate(finalists[i]->candidate, lc, 0, 1,
                                   config.launches_per_sample, g1, h1)) {
                    gpu[i].insert(gpu[i].end(), g1.begin(), g1.end());
                    host[i].insert(host[i].end(), h1.begin(), h1.end());
                }
            }
        }
        for (size_t i = 0; i < finalists.size(); ++i) {
            if (gpu[i].empty()) continue;
            finalists[i]->median_us      = median_of(gpu[i]);
            finalists[i]->mad_us         = mad_of(gpu[i], finalists[i]->median_us);
            finalists[i]->p95_us         = percentile_of(gpu[i], 0.95);
            finalists[i]->host_median_us = median_of(host[i]);
            finalists[i]->samples        = (int) gpu[i].size();
        }
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

    // --- winner selection (standards 7.3) -------------------------------
    const double native_median = native_m->median_us;
    Measurement * best_m = native_m;
    for (Measurement * m : finalists) {
        if (m->measured && m->median_us < best_m->median_us) {
            best_m = m;
        }
    }

    const double improvement = native_median > 0.0
        ? 100.0 * (native_median - best_m->median_us) / native_median : 0.0;

    if (best_m == native_m || improvement < config.replacement_threshold_pct) {
        result.winner         = native.candidate;
        result.improvement_pct = 0.0;
        result.reason          = "native retained";
    } else {
        // Near-tie resolution: everything within tie_pct of the best, ordered
        // by p95, then workspace, then native-preferred, then name. Median
        // alone is a coin flip inside the noise band; p95 is what a latency
        // budget actually feels.
        std::vector<Measurement *> tied;
        for (Measurement * m : finalists) {
            if (m->measured && m->median_us
                    <= best_m->median_us * (1.0 + config.tie_pct / 100.0)) {
                tied.push_back(m);
            }
        }
        std::sort(tied.begin(), tied.end(),
                  [&](const Measurement * a, const Measurement * b) {
                      if (a->p95_us != b->p95_us) return a->p95_us < b->p95_us;
                      if (a->workspace_bytes != b->workspace_bytes) {
                          return a->workspace_bytes < b->workspace_bytes;
                      }
                      const bool an = a->candidate == native.candidate;
                      const bool bn = b->candidate == native.candidate;
                      if (an != bn) return an;   // native wins a genuine tie
                      return strcmp(a->candidate->stable_name,
                                    b->candidate->stable_name) < 0;
                  });
        result.winner          = tied.empty() ? native.candidate
                                              : tied.front()->candidate;
        result.improvement_pct = improvement;
        result.reason          = result.winner->family == native.candidate->family
            ? "measured winner"
            : "measured winner (different family from native)";
    }

    GGML_LOG_INFO("bigcherry: tuned %s -- gen/elig/meas %d/%d/%d, winner %s "
                  "(%.2f%% vs native)\n",
                  ggml_hip_digest_hex(dispatch_digest).c_str(),
                  result.generated, result.eligible, result.measured,
                  result.winner->stable_name, result.improvement_pct);

    std::lock_guard<std::mutex> lock(g_mutex);
    g_results.emplace(dispatch_digest, result);
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

    fprintf(file,
            "{\"kind\":\"header\",\"artifact_version\":%d,"
            "\"source_revision\":\"%s\",\"manifest_hash\":\"%s\"}\n",
            GGML_HIP_AUTOTUNE_ARTIFACT_VERSION,
            GGML_HIP_AUTOTUNE_SOURCE_REVISION_STR,
            GGML_HIP_AUTOTUNE_MANIFEST_HASH_STR);

    for (const auto & entry : g_results) {
        const Result & r = entry.second;
        // Coverage first, because it is what says whether the winner means
        // anything. generated/eligible/measured being far apart is the signal
        // that an eligibility predicate is too strict or a candidate set never
        // matched the observed shapes.
        fprintf(file,
                "{\"kind\":\"result\",\"dispatch\":\"%s\","
                "\"signature\":\"%s\",\"hardware\":\"%s\",\"winner\":\"%s\","
                "\"improvement_pct\":%.3f,\"generated\":%d,\"eligible\":%d,"
                "\"measured\":%d,\"reason\":\"%s\","
                "\"canary_pct\":%.3f,\"canary_retries\":%d,"
                "\"canary_pair\":\"%s\",\"candidates\":[",
                ggml_hip_digest_hex(entry.first).c_str(),
                ggml_hip_digest_hex(r.signature_digest).c_str(),
                ggml_hip_digest_hex(r.hardware_digest).c_str(),
                r.winner ? r.winner->stable_name : "",
                r.improvement_pct, r.generated, r.eligible, r.measured,
                r.reason.c_str(),
                r.canary_pct, r.canary_retries, r.canary_pair.c_str());

        bool first = true;
        for (const Measurement & m : r.measurements) {
            fprintf(file,
                    "%s{\"name\":\"%s\",\"status\":\"%s\",\"median_us\":%.3f,"
                    "\"mad_us\":%.3f,\"p95_us\":%.3f,\"host_median_us\":%.3f,"
                    "\"nmse\":%.6g,\"max_abs\":%.6g,\"workspace\":%zu,"
                    "\"samples\":%d}",
                    first ? "" : ",",
                    m.candidate ? m.candidate->stable_name : "",
                    reason_name(m.reason), m.median_us, m.mad_us, m.p95_us,
                    m.host_median_us, m.nmse, m.max_abs_error,
                    m.workspace_bytes, m.samples);
            first = false;
        }
        fprintf(file, "]}\n");
    }
    fclose(file);

    GGML_LOG_INFO("bigcherry: wrote %zu tuning result(s) to '%s'\n",
                  g_results.size(), measurements_path.c_str());
}

#endif // GGML_USE_HIP && GGML_HIP_AUTOTUNE
