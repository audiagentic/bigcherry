// bigcherry: public interface to the HIP measured-dispatch autotuner.
//
// This header is deliberately small and C-linkage. Everything a host program
// needs -- checking whether a build carries the autotuner, reading the current
// dispatch mode, flushing the record database at shutdown -- lives here; the
// dispatch machinery itself is internal to the HIP backend.
//
// The header is safe to include unconditionally. In a build without autotune
// support the functions are still defined and report an inert state, so callers
// do not need their own #ifdef ladder.

#pragma once

#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

    // Dispatch mode, selected by GGML_HIP_DISPATCH_MODE.
    enum ggml_hip_dispatch_mode {
        // Upstream selection only. The autotuner observes nothing and changes
        // nothing. This is what a build without GGML_HIP_AUTOTUNE or
        // GGML_HIP_DISPATCH_REPLAY always reports.
        GGML_HIP_DISPATCH_MODE_NATIVE = 0,
        // Record signatures and native candidates to the SQLite database. No
        // benchmarking, no behaviour change.
        GGML_HIP_DISPATCH_MODE_RECORD = 1,
        // Benchmark candidates and persist measurements and winners.
        GGML_HIP_DISPATCH_MODE_TUNE   = 2,
        // Resolve stored winners from the compact cache. Never benchmarks.
        GGML_HIP_DISPATCH_MODE_REPLAY = 3,
    };

    // What a replay build does when a dispatch key is not in the cache.
    enum ggml_hip_miss_policy {
        GGML_HIP_MISS_NATIVE        = 0, // fall back silently
        GGML_HIP_MISS_NATIVE_RECORD = 1, // fall back and log the miss
    };

    // True when this build contains the dispatch layer at all -- that is, when
    // it was configured with GGML_HIP_AUTOTUNE or GGML_HIP_DISPATCH_REPLAY.
    GGML_BACKEND_API bool ggml_hip_autotune_available(void);

    // True when this build contains the tuning engine. Production replay builds
    // return false: standards 9.1 forbids them from benchmarking.
    GGML_BACKEND_API bool ggml_hip_autotune_can_tune(void);

    // The effective mode after applying build capability to the environment. A
    // replay-only build asked for `tune` reports NATIVE, not TUNE.
    GGML_BACKEND_API enum ggml_hip_dispatch_mode ggml_hip_dispatch_get_mode(void);

    // Hash of the candidate manifest this binary was generated from, as a
    // NUL-terminated lowercase hex string. Together with the llama.cpp source
    // revision it forms the build namespace (standards 13.1). Returns "" when
    // the build has no manifest.
    GGML_BACKEND_API const char * ggml_hip_autotune_manifest_hash(void);

    // The llama.cpp source revision embedded at build time, or "" if unknown.
    GGML_BACKEND_API const char * ggml_hip_autotune_source_revision(void);

    // Number of candidates registered in this binary.
    GGML_BACKEND_API size_t ggml_hip_autotune_candidate_count(void);

    // Stable name of candidate `index`, or NULL when out of range. The returned
    // pointer is owned by the registry and outlives the caller.
    GGML_BACKEND_API const char * ggml_hip_autotune_candidate_name(size_t index);

    // Flush pending record-mode observations and tuning measurements to the
    // database. Called automatically at backend teardown; exposed so a host can
    // checkpoint a long run. No-op outside record/tune modes.
    GGML_BACKEND_API void ggml_hip_autotune_flush(void);

    // Write a human-readable coverage report (observed signatures, native
    // candidates, frequency ranking) to `path`. No-op outside record/tune.
    GGML_BACKEND_API void ggml_hip_autotune_write_report(const char * path);

#ifdef __cplusplus
}
#endif
