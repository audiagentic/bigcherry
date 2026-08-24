// bigcherry (RD09): per-graph Q8_1 activation-quantization cache
// implementation. See hip-q81-cache.h for the design rationale and the two
// deliberate departures from the stew675 fork source.

#include "hip-q81-cache.h"

#include "common.cuh"
#include "ggml.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

ggml_hip_q81_cache_mode parse_mode(const char * s) {
    if (s == nullptr) {
        return GGML_HIP_Q81_CACHE_OFF;
    }
    if (std::strcmp(s, "on") == 0) {
        return GGML_HIP_Q81_CACHE_ON;
    }
    if (std::strcmp(s, "verify") == 0) {
        return GGML_HIP_Q81_CACHE_VERIFY;
    }
    // "off" and anything unrecognized both mean off -- fail closed rather
    // than silently enabling an experimental path on a typo.
    return GGML_HIP_Q81_CACHE_OFF;
}

// Default byte cap: bounded, generous enough for real per-graph activation
// traffic without letting a runaway workload grow this indefinitely.
// Overridable for isolated-bench headroom testing, never for production
// tuning -- there is deliberately no per-workload auto-sizing here.
constexpr size_t k_default_max_bytes = 64ull << 20; // 64 MiB per context

// Default entry cap: publish() enforces this so one pathological
// generation issuing an unbounded number of distinct cache keys cannot
// grow the entries map without limit even though total bytes are capped
// (many small reservations could otherwise exhaust the entry map's own
// memory well before the byte cap engages).
constexpr size_t k_default_max_entries = 4096;

size_t max_bytes_from_env() {
    const char * s = std::getenv("GGML_HIP_Q8_1_CACHE_MAX_MB");
    if (s == nullptr) {
        return k_default_max_bytes;
    }
    char * end = nullptr;
    const long mb = std::strtol(s, &end, 10);
    if (end == s || mb <= 0) {
        return k_default_max_bytes;
    }
    return (size_t) mb << 20;
}

} // namespace

ggml_hip_q81_cache_mode ggml_hip_q81_cache_mode_get() {
    static const ggml_hip_q81_cache_mode mode = parse_mode(std::getenv("GGML_HIP_Q8_1_CACHE_MODE"));
    return mode;
}

bool ggml_hip_q81_cache_stats_enabled() {
    static const bool enabled = [] {
        const char * s = std::getenv("GGML_HIP_Q8_1_CACHE_STATS");
        return s != nullptr && std::strcmp(s, "1") == 0;
    }();
    return enabled;
}

size_t ggml_hip_q81_cache_key_hash::operator()(const ggml_hip_q81_cache_key & k) const {
    // A plain FNV-1a mix over the key fields. This cache's correctness
    // never depends on hash quality -- operator== is the authority, this
    // only buckets candidates -- so a simple, fast, allocation-free mix is
    // preferable to pulling in a heavier hash for a fixed-size POD key.
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix(k.generation);
    mix((uint64_t) (uintptr_t) k.view_root);
    mix((uint64_t) (uintptr_t) k.data);
    mix((uint64_t) (int64_t) k.stream_no);
    mix((uint64_t) k.ne0);
    mix((uint64_t) k.ne0_padded);
    mix((uint64_t) k.ne1);
    mix((uint64_t) k.ne2);
    mix((uint64_t) k.ne3);
    mix((uint64_t) k.s1);
    mix((uint64_t) k.s2);
    mix((uint64_t) k.s3);
    return (size_t) h;
}

namespace {

// One append-only, never-relocated allocation. A cache instance owns a
// vector of these; growth appends a new one and never touches an existing
// entry's pointer or size.
struct slab {
    char * data  = nullptr;
    size_t bytes = 0;
};

} // namespace

struct ggml_hip_q81_cache {
    std::mutex mutex;

    std::vector<slab> slabs;
    // Logical write cursor within the CURRENT generation, expressed as a
    // byte offset from the start of the concatenated slab sequence. Reset
    // to 0 by begin_generation(); never causes any slab to be freed or
    // moved. May point past the end of an existing slab mid-reservation
    // walk (see reserve()); it always lands exactly on a slab boundary or
    // inside a slab's own bytes, never inside a gap.
    size_t cursor = 0;
    size_t total_slab_bytes = 0;

    uint64_t generation = 0;
    bool capture_active = false;

    std::unordered_map<ggml_hip_q81_cache_key, void *, ggml_hip_q81_cache_key_hash> entries;

    ggml_hip_q81_cache_stats stats;

    ~ggml_hip_q81_cache() {
        for (slab & s : slabs) {
            if (s.data != nullptr) {
                // Deliberately ignoring the [[nodiscard]] result: this
                // runs from a destructor / test-only reset, where there is
                // no way to propagate a failure and no correctness action
                // to take differently based on it.
                (void) hipFree(s.data);
            }
        }
    }
};

namespace {

// Translates a logical cursor position into a physical (slab, offset)
// pair, without allocating. Returns false if the position falls beyond
// every existing slab.
bool locate(const std::vector<slab> & slabs, size_t pos, size_t & slab_index, size_t & offset) {
    size_t base = 0;
    for (size_t i = 0; i < slabs.size(); i++) {
        if (pos < base + slabs[i].bytes) {
            slab_index = i;
            offset = pos - base;
            return true;
        }
        base += slabs[i].bytes;
    }
    return false;
}

// Byte offset of the start of slabs[index] within the concatenated slab
// sequence.
size_t slab_base(const std::vector<slab> & slabs, size_t index) {
    size_t base = 0;
    for (size_t i = 0; i < index; i++) {
        base += slabs[i].bytes;
    }
    return base;
}

} // namespace

ggml_hip_q81_cache & ggml_hip_q81_cache_for_context(ggml_backend_cuda_context & ctx) {
    if (ctx.q81_cache == nullptr) {
        ctx.q81_cache = new ggml_hip_q81_cache();
    }
    return *static_cast<ggml_hip_q81_cache *>(ctx.q81_cache);
}

void ggml_hip_q81_cache_destroy_for_context(ggml_backend_cuda_context & ctx) {
    if (ctx.q81_cache != nullptr) {
        delete static_cast<ggml_hip_q81_cache *>(ctx.q81_cache);
        ctx.q81_cache = nullptr;
    }
}

uint64_t ggml_hip_q81_cache_begin_generation(ggml_hip_q81_cache & cache) {
    std::lock_guard<std::mutex> lock(cache.mutex);
    cache.generation++;
    cache.cursor = 0;
    cache.entries.clear();
    cache.stats.generations++;
    cache.stats.current_bytes = 0;
    return cache.generation;
}

uint64_t ggml_hip_q81_cache_current_generation(const ggml_hip_q81_cache & cache) {
    return cache.generation;
}

ggml_hip_q81_cache_key ggml_hip_q81_cache_make_key(
        const ggml_hip_q81_cache & cache, const ggml_tensor * view_root, const void * data,
        int stream_no, int64_t ne0, int64_t ne0_padded, int64_t ne1, int64_t ne2, int64_t ne3,
        int64_t s1, int64_t s2, int64_t s3) {
    ggml_hip_q81_cache_key key;
    key.generation  = cache.generation;
    key.view_root   = view_root;
    key.data        = data;
    key.stream_no   = stream_no;
    key.ne0         = ne0;
    key.ne0_padded  = ne0_padded;
    key.ne1         = ne1;
    key.ne2         = ne2;
    key.ne3         = ne3;
    key.s1          = s1;
    key.s2          = s2;
    key.s3          = s3;
    return key;
}

void * ggml_hip_q81_cache_find(ggml_hip_q81_cache & cache, const ggml_hip_q81_cache_key & key) {
    std::lock_guard<std::mutex> lock(cache.mutex);
    cache.stats.lookups++;
    auto it = cache.entries.find(key);
    if (it == cache.entries.end()) {
        cache.stats.misses++;
        return nullptr;
    }
    cache.stats.hits++;
    cache.stats.quantize_launches_saved++;
    return it->second;
}

ggml_hip_q81_cache_reservation ggml_hip_q81_cache_reserve(ggml_hip_q81_cache & cache, size_t bytes) {
    std::lock_guard<std::mutex> lock(cache.mutex);

    // Round up so consecutive reservations stay naturally aligned for any
    // reasonable Q8_1 block size without this cache needing to know the
    // block layout itself.
    constexpr size_t k_align = 256;
    const size_t aligned_bytes = (bytes + k_align - 1) / k_align * k_align;

    // Walk forward through any already-existing slabs before ever
    // growing. A request that doesn't fit the remainder of the current
    // slab abandons that remainder and advances the cursor to the start
    // of the next retained slab -- reused across generations exactly like
    // the current slab is, so a warmed multi-slab cache stays fully
    // reusable (including during graph capture, where growth is
    // otherwise blocked but reuse of existing slabs is not).
    for (;;) {
        size_t slab_index = 0, offset = 0;
        if (!locate(cache.slabs, cache.cursor, slab_index, offset)) {
            break; // cursor is at/past every existing slab -- must grow
        }
        if (offset + aligned_bytes <= cache.slabs[slab_index].bytes) {
            void * ptr = cache.slabs[slab_index].data + offset;
            cache.cursor = slab_base(cache.slabs, slab_index) + offset + aligned_bytes;
            cache.stats.current_bytes = cache.cursor;
            if (cache.cursor > cache.stats.high_water_bytes) {
                cache.stats.high_water_bytes = cache.cursor;
            }
            return {ptr, true};
        }
        // Doesn't fit the remainder of this slab: abandon that remainder
        // (never split one reservation across two slabs) and try the
        // next existing slab, without allocating anything yet.
        cache.cursor = slab_base(cache.slabs, slab_index) + cache.slabs[slab_index].bytes;
    }

    // Every existing slab is exhausted: growth means appending a
    // brand-new slab -- never touching an existing one -- so no
    // previously-published pointer can be invalidated by this call.
    if (cache.capture_active) {
        cache.stats.capture_capacity_bypasses++;
        return {nullptr, false};
    }

    const size_t max_bytes = max_bytes_from_env();
    if (cache.total_slab_bytes >= max_bytes) {
        cache.stats.capacity_bypasses++;
        return {nullptr, false};
    }

    const size_t new_slab_bytes = std::max(aligned_bytes, (size_t) (4ull << 20));
    if (cache.total_slab_bytes + new_slab_bytes > max_bytes) {
        cache.stats.capacity_bypasses++;
        return {nullptr, false};
    }

    char * p = nullptr;
    if (hipMalloc((void **) &p, new_slab_bytes) != hipSuccess || p == nullptr) {
        cache.stats.capacity_bypasses++;
        return {nullptr, false};
    }

    // The new slab starts exactly where the logical cursor already sits
    // (the end of every existing slab, since the walk above never leaves
    // a gap), so this reservation is contiguous from the caller's point
    // of view even though it now lives in a different physical
    // allocation.
    cache.slabs.push_back({p, new_slab_bytes});
    cache.total_slab_bytes += new_slab_bytes;

    void * ptr = p;
    cache.cursor += aligned_bytes;
    cache.stats.current_bytes = cache.cursor;
    if (cache.cursor > cache.stats.high_water_bytes) {
        cache.stats.high_water_bytes = cache.cursor;
    }
    return {ptr, true};
}

void ggml_hip_q81_cache_publish(ggml_hip_q81_cache & cache, const ggml_hip_q81_cache_key & key,
                                 const ggml_hip_q81_cache_reservation & reservation) {
    if (!reservation.ok) {
        return;
    }
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (cache.entries.find(key) == cache.entries.end() && cache.entries.size() >= k_default_max_entries) {
        // The reservation's memory is still valid for the caller's own
        // immediate use; it just will not be found again by a later
        // lookup this generation.
        cache.stats.entry_cap_bypasses++;
        return;
    }
    cache.entries[key] = reservation.ptr;
    cache.stats.quantize_launches++;
}

void ggml_hip_q81_cache_set_capture_active(ggml_hip_q81_cache & cache, bool active) {
    std::lock_guard<std::mutex> lock(cache.mutex);
    cache.capture_active = active;
}

ggml_hip_q81_cache_stats ggml_hip_q81_cache_get_stats(const ggml_hip_q81_cache & cache) {
    // Cast away const for the lock only -- stats reporting must not race a
    // concurrent mutator, but it observes state rather than changing it.
    std::lock_guard<std::mutex> lock(const_cast<ggml_hip_q81_cache &>(cache).mutex);
    return cache.stats;
}

void ggml_hip_q81_cache_reset_for_test(ggml_hip_q81_cache & cache) {
    std::lock_guard<std::mutex> lock(cache.mutex);
    for (slab & s : cache.slabs) {
        if (s.data != nullptr) {
            hipFree(s.data);
        }
    }
    cache.slabs.clear();
    cache.entries.clear();
    cache.cursor = 0;
    cache.total_slab_bytes = 0;
    cache.generation = 0;
    cache.capture_active = false;
    cache.stats = ggml_hip_q81_cache_stats{};
}
