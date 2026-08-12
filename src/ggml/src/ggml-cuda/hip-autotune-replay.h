// bigcherry: compact immutable replay cache (HI11).
//
// Production resolves a dispatch key to a stored winner and launches it. That
// is the whole job. There is no benchmarking here, no SQLite, and no way to
// reach either (standards 9.1) -- a replay build does not link them.
//
// The cache is a flat binary file rather than a database because the only
// operation is "look up this 128-bit digest", performed once per distinct
// signature and then cached in process. A file that can be mapped, validated by
// checksum, and turned into a hash map in one pass is the right shape for that.
//
// Everything about the loader is written to fail *safe*: a cache from another
// build, a truncated file, a corrupted entry, or a winner whose candidate no
// longer exists all resolve to "use native selection", never to a wrong launch.
// A production binary that silently ran the wrong kernel would be far worse
// than one that quietly ran the upstream default.

#pragma once

#include "hip-autotune-types.h"

#if defined(GGML_USE_HIP) && defined(GGML_HIP_DISPATCH)

#include <stddef.h>
#include <stdint.h>

// "BCHY". Bumped only if the container layout changes incompatibly; the
// payload's compatibility is checked separately against the manifest hash and
// the ABI schema versions.
#define GGML_HIP_REPLAY_MAGIC   0x59484342u
#define GGML_HIP_REPLAY_VERSION 2

// Header of the on-disk cache. Fixed size, little-endian, no padding assumed --
// fields are read individually rather than by struct overlay, so the file is
// portable across compilers.
struct ggml_hip_replay_header {
    uint32_t magic;
    uint32_t format_version;
    uint32_t artifact_version;
    uint16_t signature_schema;
    uint16_t hardware_schema;
    uint32_t entry_count;
    uint32_t string_bytes;
    uint8_t  manifest_hash[GGML_HIP_DIGEST_BYTES];
    uint8_t  content_digest[GGML_HIP_DIGEST_BYTES]; // over entries + strings
};

// One stored winner.
struct ggml_hip_replay_entry {
    uint8_t  dispatch_digest[GGML_HIP_DIGEST_BYTES];
    uint8_t  signature_digest[GGML_HIP_DIGEST_BYTES];
    uint32_t stable_name_offset;   // into the string table
    int32_t  variant_primary;
    int32_t  variant_secondary;
    int32_t  variant_width;
    uint8_t  variant_acc_f16;
    uint8_t  variant_fallback;
    uint8_t  variant_small_k;
    uint8_t  variant_src0_type;
};

// Load the cache named by GGML_HIP_DISPATCH_CACHE, if any. Safe to call more
// than once; the first call wins. Returns false when no usable cache was
// loaded, which is not an error -- it means native selection.
bool ggml_hip_replay_init();

// Resolve a dispatch key to a stored winner.
//
// Returns false on a miss, on a cache that failed validation, and when the
// stored winner names a candidate this binary does not contain. The caller then
// uses native selection.
bool ggml_hip_replay_lookup(const ggml_hip_digest & dispatch_digest,
                            const ggml_hip_digest & signature_digest,
                            const ggml_hip_candidate_descriptor ** out_candidate,
                            ggml_hip_variant_params * out_variant);

// Record a miss. Bounded: after a fixed number of distinct keys the log stops
// growing, because an unbounded miss log in a long-running server is a leak
// wearing a diagnostic's clothing.
void ggml_hip_replay_record_miss(const ggml_hip_digest & dispatch_digest,
                                 const ggml_hip_digest & signature_digest,
                                 const ggml_hip_dispatch_signature_v1 & sig,
                                 const ggml_hip_hardware_key_v1 & hw,
                                 const ggml_hip_candidate_descriptor * fallback);

// Write the miss log where GGML_HIP_DISPATCH_MISS asked for it. Called at
// teardown; no-op unless the policy is `native-record`.
void ggml_hip_replay_flush_misses();

#ifdef GGML_HIP_REPLAY_DIAGNOSTICS
// Present only in diagnostic replay builds: production replay has no hit-log
// call or branch on the dispatch hot path.
void ggml_hip_replay_record_hit(const ggml_hip_digest & dispatch_digest,
                                const ggml_hip_digest & signature_digest,
                                const ggml_hip_candidate_descriptor * candidate);
void ggml_hip_replay_flush_hits();
#endif

// Counts for diagnostics and tests.
size_t ggml_hip_replay_entry_count();
size_t ggml_hip_replay_miss_count();

// True when a cache was rejected because its producer manifest differs from
// this build. Rejected caches never expose winners to the resolver.
bool ggml_hip_replay_is_stale();

#endif // GGML_USE_HIP && GGML_HIP_DISPATCH
