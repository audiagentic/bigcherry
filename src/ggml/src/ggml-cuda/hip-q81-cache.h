#pragma once
// bigcherry (RD09): per-graph Q8_1 activation-quantization cache.
//
// Ported and hardened from stew675/llama.cpp commit
// 299f6eaf73b5eeb888bd94eaa66122d003136e6a (rebased v2 snapshot of the
// fork's active branch; original commit
// ff6fde5046ffb86672e05da640d2bfb20d4bfdfc, "CUDA: cache quantized Q8_1
// matmul inputs per graph"). Two deliberate departures from the fork, both
// required by this item's own risk framing ("stale cache entries could
// corrupt outputs"):
//
//   1. The cache key here includes the exact view data address (`data`
//      below). The fork's key omits it, keying only on the view-root
//      pointer plus shape/stride/stream -- so two views of the SAME root
//      tensor with identical shape/stride but different byte offsets can
//      collide there. That is a real false-hit bug in the source, not a
//      hypothetical: this item's own acceptance criteria require the
//      opposite (same-root-different-offset must miss).
//   2. Backing storage here is a bounded, growable-but-never-relocated set
//      of stable slabs (see ggml_hip_q81_cache_reserve), not the fork's
//      single relocatable arena that grows by allocate-copy-free. HIP/CUDA
//      graph capture bakes pointer values into the captured graph; an
//      address that later moves would silently corrupt every subsequent
//      replay. A stable slab set has no such failure mode.
//
// Ownership: one cache instance per ggml_backend_cuda_context, not per
// physical device (dev-gpt-agent review, session ses_76b0fef0c94c434a,
// req_9bd7db08f19d4e53, 2026-08-23: a device-global singleton is unsound
// because upstream does not guarantee exactly one context per device --
// two contexts sharing a device would otherwise race each other's
// begin_generation() against live, still-in-flight reservations). The
// context owns the cache's lifetime via
// ggml_hip_q81_cache_destroy_for_context, called from
// ggml_backend_cuda_context's own destructor.
//
// Stage 1 (this file + hip-q81-cache.cpp): cache foundation only. Nothing
// in mmvq.cu references this yet -- that wiring is RD09 stage 2. Until a
// caller exists, this code is entirely inert: its mere presence in the
// build cannot change model output, by construction.
//
// docs/planning/active/rdna-boost-experiments/RD09.md has the full staged
// plan and the correctness matrix this cache must satisfy before any
// performance claim is made.

#include <cstddef>
#include <cstdint>

struct ggml_tensor;
struct ggml_backend_cuda_context;

enum ggml_hip_q81_cache_mode {
    GGML_HIP_Q81_CACHE_OFF    = 0,
    GGML_HIP_Q81_CACHE_ON     = 1,
    GGML_HIP_Q81_CACHE_VERIFY = 2,
};

// GGML_HIP_Q8_1_CACHE_MODE=off|on|verify, default off. Checked once and
// cached (same checked-once-atomic shape as ggml_hip_tuner's own env
// lookups), not re-read on every call. Deliberately independent of
// GGML_HIP_DISPATCH_MODE: this cache sits underneath MMVQ activation
// materialization and must behave identically regardless of how the MMVQ
// geometry itself was chosen (native or BigCherry-dispatched), so coupling
// the two would poison the native control that other experiments rely on.
ggml_hip_q81_cache_mode ggml_hip_q81_cache_mode_get();

// Emits a one-line summary of lifetime stats to stderr when
// GGML_HIP_Q8_1_CACHE_STATS=1 is set. Cheap to call unconditionally; the
// env check is itself checked-once.
bool ggml_hip_q81_cache_stats_enabled();

// Cache identity is scoped to one ggml_backend_cuda_context, so the key
// never needs a device field: every entry in a given instance is
// implicitly scoped to that context's device.
struct ggml_hip_q81_cache_key {
    uint64_t generation = 0;
    const ggml_tensor * view_root = nullptr; // tensor identity, not a view
    const void * data = nullptr;             // exact view byte start
    int stream_no = -1;
    int64_t ne0 = 0;
    int64_t ne0_padded = 0;
    int64_t ne1 = 0;
    int64_t ne2 = 0;
    int64_t ne3 = 0;
    int64_t s1 = 0;
    int64_t s2 = 0;
    int64_t s3 = 0;

    bool operator==(const ggml_hip_q81_cache_key & other) const {
        return generation  == other.generation
            && view_root   == other.view_root
            && data        == other.data
            && stream_no   == other.stream_no
            && ne0         == other.ne0
            && ne0_padded  == other.ne0_padded
            && ne1         == other.ne1
            && ne2         == other.ne2
            && ne3         == other.ne3
            && s1          == other.s1
            && s2          == other.s2
            && s3          == other.s3;
    }
};

struct ggml_hip_q81_cache_key_hash {
    size_t operator()(const ggml_hip_q81_cache_key & k) const;
};

struct ggml_hip_q81_cache_stats {
    uint64_t lookups                    = 0;
    uint64_t hits                       = 0;
    uint64_t misses                     = 0;
    uint64_t quantize_launches          = 0;
    uint64_t quantize_launches_saved    = 0;
    uint64_t capacity_bypasses          = 0;
    uint64_t capture_capacity_bypasses  = 0;
    uint64_t entry_cap_bypasses         = 0;
    uint64_t verify_mismatches          = 0;
    size_t   current_bytes              = 0;
    size_t   high_water_bytes           = 0;
    uint64_t generations                = 0;
};

// Reserved-but-not-yet-published allocation returned by reserve(). The
// caller (mmvq.cu, stage 2) must quantize into `ptr` and only then call
// publish() -- a miss must never become a visible cache entry before the
// producer kernel writing it has at least been enqueued. Making a miss
// visible earlier would let a second, concurrent lookup on the same key
// observe a not-yet-written buffer as if it were a hit.
struct ggml_hip_q81_cache_reservation {
    void * ptr = nullptr;
    bool   ok  = false;
};

struct ggml_hip_q81_cache;

// Lazily constructs the cache on first use for this context and returns
// it. The context owns the returned reference's lifetime; callers must
// not retain it past the context's own lifetime.
ggml_hip_q81_cache & ggml_hip_q81_cache_for_context(ggml_backend_cuda_context & ctx);

// Must be called exactly once, from ggml_backend_cuda_context's own
// destructor, before the context itself finishes tearing down. A no-op if
// no cache was ever constructed for this context (the common case for a
// context that never ran an MMVQ op, or ran with the cache off).
void ggml_hip_q81_cache_destroy_for_context(ggml_backend_cuda_context & ctx);

// Call once per graph evaluation, before any lookup against this
// context's cache for that evaluation. Bumps and returns the generation
// counter (so every key from the previous evaluation misses) and resets
// the LOGICAL bump position of the stable slabs back to their start.
// Backing memory is never freed or moved here -- see the stable-slab
// rationale above. This only produces stable per-generation addresses if
// the sequence of reserve() calls within a generation is itself
// deterministic for a given graph shape, which is already a precondition
// HIP/CUDA graph capture imposes on the whole compute path.
uint64_t ggml_hip_q81_cache_begin_generation(ggml_hip_q81_cache & cache);

// The generation set by the most recent begin_generation() call (0 before
// the first call). Exposed so callers build cache keys from a single
// authoritative source rather than tracking a second counter of their
// own -- see ggml_hip_q81_cache_make_key, which does this for them.
uint64_t ggml_hip_q81_cache_current_generation(const ggml_hip_q81_cache & cache);

// Builds a key against the cache's current generation, so a caller never
// hand-assembles cache identity (and can never omit a field a future
// reviewer would otherwise have to catch by hand -- see the fork's own
// missing-offset bug this cache exists to not repeat).
ggml_hip_q81_cache_key ggml_hip_q81_cache_make_key(
    const ggml_hip_q81_cache & cache, const ggml_tensor * view_root, const void * data,
    int stream_no, int64_t ne0, int64_t ne0_padded, int64_t ne1, int64_t ne2, int64_t ne3,
    int64_t s1, int64_t s2, int64_t s3);

// nullptr on miss.
void * ggml_hip_q81_cache_find(ggml_hip_q81_cache & cache, const ggml_hip_q81_cache_key & key);

// Bump-allocates `bytes` from the current generation's live region of the
// stable slab set. Walks forward through any already-existing slabs
// first (a request that doesn't fit the remainder of the current slab
// advances to the next retained slab, never splitting one reservation
// across two slabs); only once every existing slab is exhausted does this
// append a brand-new slab, and only when growth is currently allowed (see
// ggml_hip_q81_cache_set_capture_active). A new slab is an ADDITIONAL
// allocation, never a reallocation of an existing one, so no
// previously-returned pointer is ever invalidated.
//
// Returns ok=false (never partial) on any failure. The caller's contract
// on failure is to fall back to the exact existing per-call pool
// allocation and quantizer for that one call, never to retry or degrade
// the request.
ggml_hip_q81_cache_reservation ggml_hip_q81_cache_reserve(ggml_hip_q81_cache & cache, size_t bytes);

// Makes a successful reserve() visible to later find() calls with the same
// key for the remainder of this generation. Must only be called after the
// producer kernel writing `reservation.ptr` has been enqueued on the same
// stream the key names. A no-op past the per-generation entry cap (the
// reservation's memory is still valid for the caller's own immediate use;
// it simply will not be found again by a later lookup this generation).
void ggml_hip_q81_cache_publish(ggml_hip_q81_cache & cache, const ggml_hip_q81_cache_key & key,
                                 const ggml_hip_q81_cache_reservation & reservation);

// Graph capture is in flight on this context: reserve() must not grow the
// backing slab set (a growth would allocate a new address after a graph
// may already have captured pointers computed from the current layout).
// While active, a reserve() that would otherwise need to grow instead
// fails closed -- counted as a capture_capacity_bypass, not a plain
// capacity_bypass -- so the caller falls back to the native path for that
// one call rather than risking a captured graph referencing memory that
// was never actually grown into. Reusing an already-existing later slab
// is still permitted while capture is active; only growth (a brand-new
// allocation) is blocked.
void ggml_hip_q81_cache_set_capture_active(ggml_hip_q81_cache & cache, bool active);

ggml_hip_q81_cache_stats ggml_hip_q81_cache_get_stats(const ggml_hip_q81_cache & cache);

// Testing only: drops all cached entries, frees all backing slabs, and
// zeroes stats/generation. Not for use on any real inference path -- a
// live graph capture holding addresses into a freed slab is exactly the
// corruption this cache exists to prevent.
void ggml_hip_q81_cache_reset_for_test(ggml_hip_q81_cache & cache);
