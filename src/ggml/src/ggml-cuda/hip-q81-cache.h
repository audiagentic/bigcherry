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

// One cache instance per physical device. Device id is intentionally NOT
// part of the key: the accessor below already routes to the right
// instance, so the key never needs to disambiguate devices, and every
// entry in a given instance is implicitly scoped to that device.
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

ggml_hip_q81_cache & ggml_hip_q81_cache_for_device(int device);

// Call once per graph evaluation, before any lookup against this device's
// cache for that evaluation. Bumps the generation counter (so every key
// from the previous evaluation misses) and resets the LOGICAL bump
// position of the stable slabs back to their start. Backing memory is
// never freed or moved here -- see the stable-slab rationale above. This
// only produces stable per-generation addresses if the sequence of
// reserve() calls within a generation is itself deterministic for a given
// graph shape, which is already a precondition HIP/CUDA graph capture
// imposes on the whole compute path.
void ggml_hip_q81_cache_begin_generation(ggml_hip_q81_cache & cache);

// nullptr on miss.
void * ggml_hip_q81_cache_find(ggml_hip_q81_cache & cache, const ggml_hip_q81_cache_key & key);

// Bump-allocates `bytes` from the current generation's live region of the
// stable slab set, appending a new slab if the existing ones are
// exhausted and growth is currently allowed (see
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
// stream the key names.
void ggml_hip_q81_cache_publish(ggml_hip_q81_cache & cache, const ggml_hip_q81_cache_key & key,
                                 const ggml_hip_q81_cache_reservation & reservation);

// Graph capture is in flight on this device: reserve() must not grow the
// backing slab set (a growth would allocate a new address after a graph
// may already have captured pointers computed from the current layout).
// While active, a reserve() that would otherwise need to grow instead
// fails closed -- counted as a capture_capacity_bypass, not a plain
// capacity_bypass -- so the caller falls back to the native path for that
// one call rather than risking a captured graph referencing memory that
// was never actually grown into.
void ggml_hip_q81_cache_set_capture_active(ggml_hip_q81_cache & cache, bool active);

ggml_hip_q81_cache_stats ggml_hip_q81_cache_get_stats(const ggml_hip_q81_cache & cache);

// Testing only: drops all cached entries, frees all backing slabs, and
// zeroes stats/generation. Not for use on any real inference path -- a
// live graph capture holding addresses into a freed slab is exactly the
// corruption this cache exists to prevent.
void ggml_hip_q81_cache_reset_for_test(ggml_hip_q81_cache & cache);
