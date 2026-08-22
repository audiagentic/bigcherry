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
// build, a truncated file, a corrupted entry, or a winner whose provenance or
// candidate no longer matches all resolve to "use native selection", never to
// a wrong launch.
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
// HI31/HI74: v4 -> v5 adds a per-entry transform_id (uint16, 0 = no
// transform) and a match_kind discriminator (uint8, see
// ggml_hip_replay_match_kind below). A v4 cache is rejected outright as
// RERUN_REQUIRED by the existing format-version check below -- no dual-
// layout reader, matching this project's fail-closed exact-version-match
// policy for every other schema field here (signature_schema,
// hardware_schema). Transformed entries additionally require EXACT
// manifest/source provenance even in GGML_HIP_DISPATCH_REPLAY_REVISION_
// MATCH=0 mode: transforms carry no implementation_version of their own yet
// to validate safely against a relaxed match (see hip-autotune-replay.cpp's
// loader).
#define GGML_HIP_REPLAY_VERSION 5

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

// HI74: which key form this entry was matched by. EXACT is the only kind
// any producer emits today; the rest of the range is reserved so a future
// generalised-entry feature (HI36b, e.g. matching on a coarser MMQ K-multiple
// or MMVQ ne0 class rather than the exact dispatch digest) has an extension
// point that does not force a v6 wire-format bump by itself. An unrecognised
// value is a REJECT (INCOMPATIBLE), never a silent reinterpretation as EXACT
// -- see hip-autotune-replay.cpp's loader.
enum ggml_hip_replay_match_kind : uint8_t {
    GGML_HIP_REPLAY_MATCH_EXACT = 0,
    // 1-255 reserved for future match kinds (HI36b).
};

// One stored winner.
struct ggml_hip_replay_entry {
    uint8_t  dispatch_digest[GGML_HIP_DIGEST_BYTES];
    uint8_t  signature_digest[GGML_HIP_DIGEST_BYTES];
    uint32_t stable_name_offset;   // into the string table
    uint16_t implementation_version;
    int32_t  variant_primary;
    int32_t  variant_secondary;
    int32_t  variant_width;
    uint8_t  variant_acc_f16;
    uint8_t  variant_fallback;
    uint8_t  variant_small_k;
    uint8_t  variant_src0_type;
    uint8_t  manifest_hash[GGML_HIP_DIGEST_BYTES];
    uint8_t  source_revision_digest[GGML_HIP_DIGEST_BYTES];
    uint32_t generation;
    // HI31 (v5): 0 = GGML_HIP_TRANSFORM_NONE, a plain candidate winner.
    // Raw uint16 rather than ggml_hip_transform_id -- that enum only exists
    // under GGML_HIP_ROUTING_TRANSFORM, but the wire format and the loader
    // in hip-autotune-replay.cpp are unconditionally compiled either way (a
    // cache built with transforms may still be read by a build without the
    // feature, which must then reject every non-zero entry, never crash or
    // silently drop the field). Deliberately NOT folded into the dispatch
    // digest: the digest identifies the operation being decided, transform
    // is the stored decision -- a separate fact about the same key.
    uint16_t transform_id;
    // HI74 (v5): see ggml_hip_replay_match_kind above. Every producer today
    // writes GGML_HIP_REPLAY_MATCH_EXACT; the loader rejects anything else.
    uint8_t  match_kind;
};

// Recovery note (RE27): this five-way classification and the always-on
// counters below restore what existed pre-reset (commit 2c2fe7c) and that
// the Python coverage validator (ab_benchmark.validate_replay_coverage())
// was already written and tested against. It was never a speculative v2
// design -- the runtime producer was simply lost in the reset. Semantics,
// per the recovery reconciliation against v4's multi-generation cache:
//   exact                  -- a winner resolved AND is actually usable now:
//                              candidate registered, supports this arch,
//                              can_execute() accepts this signature, and the
//                              stored variant matches the compiled candidate.
//   miss                   -- the cache loaded fine (or no cache was
//                              configured); this dispatch key simply has no
//                              stored entry at all.
//   rerun_required          -- the cache (or the matching entries for this
//                              key) is from an obsolete-but-recognisable
//                              producer: format/ABI/schema mismatch at
//                              load time, or (v4 multi-generation specific)
//                              entries exist for this key+signature but all
//                              are non-fresh while revision matching is
//                              required. A tune rerun against the current
//                              build would fix it.
//   incompatible            -- malformed/corrupt/structurally wrong cache
//                              (bad magic, truncation, checksum, entry
//                              provenance), or a fresh matching entry whose
//                              signature digest disagrees with this lookup's.
//   candidate_unavailable   -- the key (and, where applicable, signature)
//                              matched a stored winner, but it cannot be
//                              used: unregistered candidate, architecture
//                              rejection, can_execute() rejection, or a
//                              variant-identity mismatch against the
//                              compiled candidate.
// incompatible/rerun_required are frequently whole-cache-level failures
// (see g_load_failure in the .cpp) surfaced as a per-lookup outcome purely
// for uniform coverage accounting -- see ggml_hip_coverage_report().
enum ggml_hip_resolution_v2 : uint8_t {
    GGML_HIP_RESOLVE_EXACT = 0,
    GGML_HIP_RESOLVE_CANDIDATE_UNAVAILABLE,
    GGML_HIP_RESOLVE_RERUN_REQUIRED,
    GGML_HIP_RESOLVE_INCOMPATIBLE,
    GGML_HIP_RESOLVE_MISS,
    GGML_HIP_RESOLVE_COUNT,
};

// Load the cache named by GGML_HIP_DISPATCH_CACHE, if any. Safe to call more
// than once; the first call wins. Returns false when no usable cache was
// loaded, which is not an error -- it means native selection.
bool ggml_hip_replay_init();

// Resolve a dispatch key to a stored winner. sig/hw are needed to run the
// same arch/can_execute/variant checks that decide EXACT vs
// CANDIDATE_UNAVAILABLE -- classification happens here, not after the
// caller has already committed to a binding, so a candidate that resolves
// but then fails those checks is never silently counted as a hit.
//
// Only GGML_HIP_RESOLVE_EXACT sets *out_candidate/*out_variant/*out_transform_id;
// every other outcome leaves them untouched and the caller uses native
// selection.
//
// HI31: *out_transform_id != 0 (GGML_HIP_TRANSFORM_NONE) means the stored
// winner was reached via a routing transform (HI27/HI28), NOT the plain
// candidate the caller's `sig` describes. This function only validates
// candidate registration and architecture support for such an entry --
// it deliberately does NOT run `winner.candidate->can_execute(candidate,
// sig, hw)` against the ORIGINAL signature for a transformed entry, since
// that is exactly the wrong check (the candidate was never expected to
// accept the untransformed shape; that mismatch is the whole reason the
// transform exists). EXACT here is therefore NECESSARY but not SUFFICIENT
// for a transformed entry: the caller (ggml_hip_dispatch_resolve(), which
// has the real ggml_hip_launch_context this loader does not) MUST still
// resolve the transform, confirm transform->equivalence_verified, call
// transform->apply() to obtain the actual transformed signature, and
// check *out_candidate->can_execute(*out_candidate, transformed_sig, hw)
// before treating this as a valid binding -- exactly as it already must
// for a freshly-tuned transformed winner. A build without
// GGML_HIP_ROUTING_TRANSFORM compiled in must instead treat any nonzero
// *out_transform_id as unusable and fail to native selection; it has no
// way to run that second-layer check at all.
ggml_hip_resolution_v2 ggml_hip_replay_lookup(
                            const ggml_hip_digest & dispatch_digest,
                            const ggml_hip_digest & signature_digest,
                            const ggml_hip_dispatch_signature_v1 & sig,
                            const ggml_hip_hardware_key_v1 & hw,
                            const ggml_hip_candidate_descriptor ** out_candidate,
                            ggml_hip_variant_params * out_variant,
                            uint16_t * out_transform_id);

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

// Per-outcome resolution counters. Always tracked in every replay build --
// five relaxed atomics incremented once per ggml_hip_replay_lookup() call,
// the same cost class as the existing dispatched/executed coverage
// counters (standards: the resolver binds a dispatch key once per process
// and reuses the binding, so this is not a per-launch increment). Not
// gated by GGML_HIP_REPLAY_DIAGNOSTICS -- that flag is reserved for the
// heavier per-key hit log, never for this aggregate.
size_t ggml_hip_replay_resolution_count(ggml_hip_resolution_v2 outcome);
const char * ggml_hip_replay_resolution_name(ggml_hip_resolution_v2 outcome);

// True when the loaded cache contains one or more retained generations whose
// provenance does not match this build. Such entries remain available to a
// matching older binary but never become a stale production decision here.
bool ggml_hip_replay_is_stale();

// The loader's whole-cache rejection classification, if any. Meaningful when
// ggml_hip_replay_init() returned false: GGML_HIP_RESOLVE_RERUN_REQUIRED for
// an obsolete-but-recognisable producer, GGML_HIP_RESOLVE_INCOMPATIBLE for a
// structurally wrong or corrupt file, GGML_HIP_RESOLVE_MISS when no cache was
// configured. A loaded cache has no failure. Exposed so the offline inspect
// tool can report the same reason the production lookups would surface, not
// a reimplementation's guess at it.
ggml_hip_resolution_v2 ggml_hip_replay_load_failure();

// One winner slot as the loader's own tables hold it -- the exact state a
// production lookup consumes, exposed for inspection only. The lookup hot
// path never touches this API.
struct ggml_hip_replay_winner_info {
    const char * candidate_name;      // registry name, or the file's string for an unregistered entry
    bool         registered;          // this build's registry knows the name
    bool         stale_impl_version;  // implementation_version drifted from the registry
    bool         unrecognized_match;  // match_kind this build does not recognise
    bool         fresh;               // entry provenance matches this build exactly
    uint32_t     generation;
    uint16_t     transform_id;        // 0 = GGML_HIP_TRANSFORM_NONE
    uint8_t      match_kind;          // see ggml_hip_replay_match_kind
};

// Visit every winner slot of the loaded cache, all keys and all generations.
// Returns the number of slots visited (0 when no usable cache was loaded);
// `visit` returning false stops early. Used by the offline inspect tool; no
// production code calls it.
size_t ggml_hip_replay_foreach_winner(
    bool (*visit)(const ggml_hip_digest & dispatch_digest,
                  const ggml_hip_replay_winner_info * info, void * user),
    void * user);

#endif // GGML_USE_HIP && GGML_HIP_DISPATCH
