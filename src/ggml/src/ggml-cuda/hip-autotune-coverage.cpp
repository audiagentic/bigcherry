// bigcherry: dispatch coverage counters.

#include "hip-autotune-coverage.h"

#if defined(GGML_USE_HIP) && defined(GGML_HIP_DISPATCH)

#ifdef GGML_HIP_DISPATCH_REPLAY
#include "hip-autotune-replay.h"
#endif

#include <atomic>
#include <stdio.h>
#include <stdlib.h>

namespace {

// Relaxed ordering throughout: these are diagnostics, and the only consumer
// reads them after every contributing thread has finished. Paying for
// sequential consistency on a per-launch counter would be buying an ordering
// guarantee nothing uses.
std::atomic<uint64_t> g_executed[GGML_HIP_FAMILY_COUNT];
std::atomic<uint64_t> g_dispatched[GGML_HIP_FAMILY_COUNT];

const char * family_name(int family) {
    switch (family) {
        case GGML_HIP_FAMILY_MMVQ: return "mmvq";
        case GGML_HIP_FAMILY_MMQ:  return "mmq";
        case GGML_HIP_FAMILY_MMVF: return "mmvf";
        case GGML_HIP_FAMILY_MMF:  return "mmf";
        case GGML_HIP_FAMILY_BLAS: return "blas";
        default:                   return "?";
    }
}

} // namespace

void ggml_hip_coverage_count_executed(ggml_hip_kernel_family family) {
    if (family < GGML_HIP_FAMILY_COUNT) {
        g_executed[family].fetch_add(1, std::memory_order_relaxed);
    }
}

void ggml_hip_coverage_count_dispatched(ggml_hip_kernel_family family) {
    if (family < GGML_HIP_FAMILY_COUNT) {
        g_dispatched[family].fetch_add(1, std::memory_order_relaxed);
    }
}

void ggml_hip_coverage_report() {
    uint64_t executed[GGML_HIP_FAMILY_COUNT];
    uint64_t dispatched[GGML_HIP_FAMILY_COUNT];
    uint64_t total_executed = 0;
    uint64_t total_dispatched = 0;

    for (int i = 0; i < GGML_HIP_FAMILY_COUNT; ++i) {
        executed[i]   = g_executed[i].load(std::memory_order_relaxed);
        dispatched[i] = g_dispatched[i].load(std::memory_order_relaxed);
        total_executed   += executed[i];
        total_dispatched += dispatched[i];
    }

    if (total_executed == 0) {
        return;
    }

    const char * path = getenv("GGML_HIP_DISPATCH_COVERAGE");
    FILE * out = nullptr;
    if (path != nullptr && path[0] != '\0') {
        out = fopen(path, "w");
    }

    if (out != nullptr) {
        fprintf(out, "{\n  \"families\": {\n");
        for (int i = 0; i < GGML_HIP_FAMILY_COUNT; ++i) {
            fprintf(out,
                    "    \"%s\": {\"executed\": %llu, \"dispatched\": %llu}%s\n",
                    family_name(i), (unsigned long long) executed[i],
                    (unsigned long long) dispatched[i],
                    i + 1 < GGML_HIP_FAMILY_COUNT ? "," : "");
        }
        fprintf(out,
                "  },\n  \"total_executed\": %llu,\n  \"total_dispatched\": %llu",
                (unsigned long long) total_executed,
                (unsigned long long) total_dispatched);
#ifdef GGML_HIP_DISPATCH_REPLAY
        // Replay provenance belongs in the artifact, not only in a log line.
        //
        // "dispatched == executed" says every launch went through the layer.
        // It does *not* say the cache supplied a winner -- a miss is still a
        // dispatch, to native. So a fully-covered run can be entirely untuned,
        // and reading only the coverage totals cannot tell the difference.
        // That confusion has already cost one wrong conclusion (RV12).
        //
        // `stale` says the winners were measured against a different candidate
        // set: still valid, possibly no longer best (HI23).
        //
        // schema_version 2 restores the pre-reset (2c2fe7c) per-lookup
        // resolution classification -- see the enum in hip-autotune-replay.h
        // for what each bucket means. "misses" here is the unconditional
        // per-lookup MISS count from ggml_hip_replay_lookup(), distinct from
        // ggml_hip_replay_miss_count() (miss_log_calls below), which is only
        // populated when GGML_HIP_DISPATCH_MISS=native-record is set.
        fprintf(out,
                ",\n  \"replay\": {\"schema_version\": 2, \"entries\": %zu, "
                "\"miss_log_calls\": %zu, \"exact\": %zu, "
                "\"candidate_unavailable\": %zu, \"rerun_required\": %zu, "
                "\"incompatible\": %zu, \"misses\": %zu, \"stale\": %s}",
                ggml_hip_replay_entry_count(),
                ggml_hip_replay_miss_count(),
                ggml_hip_replay_resolution_count(GGML_HIP_RESOLVE_EXACT),
                ggml_hip_replay_resolution_count(GGML_HIP_RESOLVE_CANDIDATE_UNAVAILABLE),
                ggml_hip_replay_resolution_count(GGML_HIP_RESOLVE_RERUN_REQUIRED),
                ggml_hip_replay_resolution_count(GGML_HIP_RESOLVE_INCOMPATIBLE),
                ggml_hip_replay_resolution_count(GGML_HIP_RESOLVE_MISS),
                ggml_hip_replay_is_stale() ? "true" : "false");
#endif
        // The hot-path counters go in the artifact for the same reason the
        // replay provenance above does: the log channel does not survive
        // llama-server, so a number that only ever appears in a log line is a
        // number no benchmark can use.
        ggml_hip_dispatch_counters_write_json(out);
        fprintf(out, "\n}\n");
        fclose(out);
    }

    GGML_LOG_INFO("bigcherry: dispatch coverage -- %llu of %llu matmul launches "
                  "(%.1f%%) reached measured dispatch\n",
                  (unsigned long long) total_dispatched,
                  (unsigned long long) total_executed,
                  100.0 * (double) total_dispatched / (double) total_executed);
#ifdef GGML_HIP_DISPATCH_REPLAY
    // Stated alongside coverage because the two are routinely conflated: full
    // coverage with every lookup missing is a correct, entirely untuned run.
    GGML_LOG_INFO(
        "bigcherry:   replay v2 %zu winner(s); exact=%zu unavailable=%zu "
        "rerun-required=%zu incompatible=%zu miss=%zu%s\n",
        ggml_hip_replay_entry_count(),
        ggml_hip_replay_resolution_count(GGML_HIP_RESOLVE_EXACT),
        ggml_hip_replay_resolution_count(GGML_HIP_RESOLVE_CANDIDATE_UNAVAILABLE),
        ggml_hip_replay_resolution_count(GGML_HIP_RESOLVE_RERUN_REQUIRED),
        ggml_hip_replay_resolution_count(GGML_HIP_RESOLVE_INCOMPATIBLE),
        ggml_hip_replay_resolution_count(GGML_HIP_RESOLVE_MISS),
        ggml_hip_replay_is_stale()
            ? " -- tuned against a different candidate set" : "");
#endif
    for (int i = 0; i < GGML_HIP_FAMILY_COUNT; ++i) {
        if (executed[i] == 0) {
            continue;
        }
        GGML_LOG_INFO("bigcherry:   %-5s executed %10llu  dispatched %10llu  "
                      "(%.1f%%)\n",
                      family_name(i), (unsigned long long) executed[i],
                      (unsigned long long) dispatched[i],
                      100.0 * (double) dispatched[i] / (double) executed[i]);
    }
}

#endif // GGML_USE_HIP && GGML_HIP_DISPATCH
