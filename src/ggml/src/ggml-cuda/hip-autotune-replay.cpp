// bigcherry: compact immutable replay cache (HI11).

#include "hip-autotune-replay.h"

#if defined(GGML_USE_HIP) && defined(GGML_HIP_DISPATCH)

#include "hip-autotune-blake2b.h"
#include "hip-autotune-build-hash.h"
#include "hip-autotune-dispatch.cuh"
#include "hip-autotune-signature.h"

#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// An unbounded miss log in a server that runs for weeks is a memory leak with a
// justification attached. Past this many distinct keys we keep counting but
// stop remembering which ones.
constexpr size_t MAX_TRACKED_MISSES = 4096;

struct Winner {
    const ggml_hip_candidate_descriptor * candidate;
    ggml_hip_variant_params               variant;
    ggml_hip_digest                       signature_digest;
};

// A recorded miss.
//
// `calls` is very nearly always 1, and that is by design rather than a bug: the
// process cache in the resolver binds the native fallback on first encounter,
// so every later execution of the same dispatch key is a dictionary hit that
// never reaches this function (standards 15.2). Counting them would mean
// incrementing a mutex-protected counter on the hottest path in the system, to
// produce a number record mode already produces properly.
//
// So the miss log answers "which signatures is the cache not covering", not
// "how hot are they". Frequency ranking for tuning priority (standards 7.4)
// comes from record-mode observations, which are built for exactly that.
struct Miss {
    ggml_hip_digest signature_digest;
    std::string     canonical_json;
    std::string     fallback_name;
    uint64_t        calls;   // encounters that reached the resolver's cold path
};

#ifdef GGML_HIP_REPLAY_DIAGNOSTICS
struct Hit {
    ggml_hip_digest signature_digest;
    std::string     candidate_name;
    uint64_t        calls;
};
#endif

struct DigestHash {
    size_t operator()(const ggml_hip_digest & digest) const {
        size_t value = 0;
        for (size_t i = 0; i < sizeof(size_t) && i < GGML_HIP_DIGEST_BYTES; ++i) {
            value |= (size_t) digest.bytes[i] << (8 * i);
        }
        return value;
    }
};

struct DigestEqual {
    bool operator()(const ggml_hip_digest & a, const ggml_hip_digest & b) const {
        return ggml_hip_digest_equal(a, b);
    }
};

std::unordered_map<ggml_hip_digest, Winner, DigestHash, DigestEqual> g_winners;
std::unordered_map<ggml_hip_digest, Miss, DigestHash, DigestEqual>   g_misses;
#ifdef GGML_HIP_REPLAY_DIAGNOSTICS
std::unordered_map<ggml_hip_digest, Hit, DigestHash, DigestEqual>     g_hits;
std::mutex g_hits_mutex;
#endif
std::mutex g_misses_mutex;
uint64_t   g_misses_total = 0;
bool       g_loaded = false;
// The loaded cache was tuned against a different candidate set. Kept as state
// rather than only logged so the coverage report can say so too -- a log line
// in a server that has been up for a week is not a usable answer to "is this
// build actually tuned".
bool       g_stale  = false;

uint32_t read_u32(const uint8_t * p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8)
         | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

uint16_t read_u16(const uint8_t * p) {
    return (uint16_t) ((uint16_t) p[0] | ((uint16_t) p[1] << 8));
}

int32_t read_i32(const uint8_t * p) {
    return (int32_t) read_u32(p);
}

// Byte offsets within the header. Written out rather than using a struct
// overlay so the file layout does not depend on the compiler's padding.
constexpr size_t HDR_MAGIC        = 0;
constexpr size_t HDR_FORMAT       = 4;
constexpr size_t HDR_ARTIFACT     = 8;
constexpr size_t HDR_SIG_SCHEMA   = 12;
constexpr size_t HDR_HW_SCHEMA    = 14;
constexpr size_t HDR_ENTRY_COUNT  = 16;
constexpr size_t HDR_STRING_BYTES = 20;
constexpr size_t HDR_MANIFEST     = 24;
constexpr size_t HDR_CONTENT      = 24 + GGML_HIP_DIGEST_BYTES;
constexpr size_t HDR_SIZE         = HDR_CONTENT + GGML_HIP_DIGEST_BYTES;

constexpr size_t ENT_DISPATCH  = 0;
constexpr size_t ENT_SIGNATURE = GGML_HIP_DIGEST_BYTES;
constexpr size_t ENT_NAME_OFF  = 2 * GGML_HIP_DIGEST_BYTES;
constexpr size_t ENT_PRIMARY   = ENT_NAME_OFF + 4;
constexpr size_t ENT_SECONDARY = ENT_PRIMARY + 4;
constexpr size_t ENT_WIDTH     = ENT_SECONDARY + 4;
constexpr size_t ENT_ACC_F16   = ENT_WIDTH + 4;
constexpr size_t ENT_FALLBACK  = ENT_ACC_F16 + 1;
// These two were reserved bytes that nothing read, which meant the two most
// recently added members of ggml_hip_variant_params could not round-trip
// through the cache at all:
//
//   small_k   -- every top winner on the real MTP workload is an `sk1`
//                geometry (RV10, 14-20% each). Replayed as sk0 they name a
//                different compiled instance, and the (w6,nw4,rpb4,sk0)
//                combination is not generated, so the launch reaches the HI09
//                missing-instance abort rather than merely losing the gain.
//   src0_type -- added by RV01, so a replayed candidate would carry type 0
//                (F32) and be rejected by its own family's can_execute on
//                every quantised signature, silently falling back to native.
//
// Taken from the reserved space, so ENT_SIZE is unchanged and no version bump
// is needed: nothing has ever written this format (RV09), so there is no cache
// in existence to be misread.
constexpr size_t ENT_SMALL_K   = ENT_FALLBACK + 1;
constexpr size_t ENT_SRC0_TYPE = ENT_SMALL_K + 1;
constexpr size_t ENT_SIZE      = ENT_SRC0_TYPE + 1;

std::vector<uint8_t> read_file(const char * path) {
    std::vector<uint8_t> bytes;
    FILE * file = fopen(path, "rb");
    if (file == nullptr) {
        return bytes;
    }
    fseek(file, 0, SEEK_END);
    const long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (size > 0) {
        bytes.resize((size_t) size);
        if (fread(bytes.data(), 1, bytes.size(), file) != bytes.size()) {
            bytes.clear();
        }
    }
    fclose(file);
    return bytes;
}

// Parse the manifest hash from its hex form in the generated header, so the
// on-disk 16 raw bytes can be compared against what this binary was built from.
bool manifest_hash_matches(const uint8_t * stored) {
    const char * hex = GGML_HIP_AUTOTUNE_MANIFEST_HASH_STR;
    if (strlen(hex) != GGML_HIP_DIGEST_BYTES * 2) {
        return false;
    }
    for (size_t i = 0; i < GGML_HIP_DIGEST_BYTES; ++i) {
        unsigned value = 0;
        if (sscanf(hex + 2 * i, "%2x", &value) != 1) {
            return false;
        }
        if ((uint8_t) value != stored[i]) {
            return false;
        }
    }
    return true;
}

void reject(const char * path, const char * why) {
    // Deliberately a warning, not a fatal error: an unusable cache means this
    // process runs upstream's own selection, which is correct, merely untuned.
    GGML_LOG_WARN("bigcherry: ignoring replay cache '%s': %s. "
                  "Falling back to native selection.\n", path, why);
}

} // namespace

bool ggml_hip_replay_init() {
    static std::once_flag once;
    std::call_once(once, []() {
        const char * path = getenv("GGML_HIP_DISPATCH_CACHE");
        if (path == nullptr || path[0] == '\0') {
            return;
        }

        const std::vector<uint8_t> bytes = read_file(path);
        if (bytes.size() < HDR_SIZE) {
            reject(path, "file is shorter than its header");
            return;
        }
        const uint8_t * data = bytes.data();
        bool stale = false;

        if (read_u32(data + HDR_MAGIC) != GGML_HIP_REPLAY_MAGIC) {
            reject(path, "bad magic -- not a bigcherry replay cache");
            return;
        }
        if (read_u32(data + HDR_FORMAT) != GGML_HIP_REPLAY_VERSION) {
            reject(path, "container format version mismatch");
            return;
        }
        if (read_u32(data + HDR_ARTIFACT) != GGML_HIP_AUTOTUNE_ARTIFACT_VERSION) {
            reject(path, "artifact version mismatch");
            return;
        }
        // Standards 5.3: a signature schema change reinterprets hashed fields,
        // so every stored key from an older build is meaningless, not merely
        // stale.
        if (read_u16(data + HDR_SIG_SCHEMA) != GGML_HIP_SIGNATURE_SCHEMA_VERSION) {
            reject(path, "signature schema version mismatch");
            return;
        }
        if (read_u16(data + HDR_HW_SCHEMA) != GGML_HIP_HARDWARE_SCHEMA_VERSION) {
            reject(path, "hardware key schema version mismatch");
            return;
        }
        // A different candidate set means these winners were chosen from
        // different options. That makes them possibly sub-optimal, not
        // invalid, so it is reported rather than fatal.
        //
        // This used to reject the cache outright (standards 13.1). Combined
        // with the manifest hash being part of the lookup key, it meant any
        // rebuild that touched the catalog threw away every tuning result --
        // hours of exclusive GPU time -- with no way to keep using it. The
        // guards that matter are per entry and still enforced below: an entry
        // naming a candidate this binary lacks is dropped, and the resolver
        // re-runs `can_execute` before launching a stored winner.
        //
        // Interim measure; HI23 replaces it with per-entry provenance and
        // several retained generations.
        stale = !manifest_hash_matches(data + HDR_MANIFEST);

        const uint32_t entry_count  = read_u32(data + HDR_ENTRY_COUNT);
        const uint32_t string_bytes = read_u32(data + HDR_STRING_BYTES);
        const size_t   entries_at   = HDR_SIZE;
        const size_t   strings_at   = entries_at + (size_t) entry_count * ENT_SIZE;
        const size_t   expected     = strings_at + string_bytes;
        if (bytes.size() != expected) {
            reject(path, "file is truncated");
            return;
        }

        // Checksum before trusting any offset in the payload.
        ggml_hip_digest content;
        ggml_hip_blake2b(content.bytes, GGML_HIP_DIGEST_BYTES,
                         data + entries_at, expected - entries_at,
                         GGML_HIP_PERSON_DISPATCH);
        if (memcmp(content.bytes, data + HDR_CONTENT, GGML_HIP_DIGEST_BYTES) != 0) {
            reject(path, "content checksum mismatch -- the file is corrupt");
            return;
        }

        const char * strings = (const char *) (data + strings_at);
        size_t unknown = 0;

        for (uint32_t i = 0; i < entry_count; ++i) {
            const uint8_t * entry = data + entries_at + (size_t) i * ENT_SIZE;

            const uint32_t name_offset = read_u32(entry + ENT_NAME_OFF);
            if (name_offset >= string_bytes) {
                reject(path, "entry names a string outside the string table");
                g_winners.clear();
                return;
            }
            // The table must be NUL-terminated within its own bounds, or
            // strcmp below would walk off the end.
            if (memchr(strings + name_offset, '\0',
                       string_bytes - name_offset) == nullptr) {
                reject(path, "string table is not NUL-terminated");
                g_winners.clear();
                return;
            }

            const ggml_hip_candidate_descriptor * candidate =
                ggml_hip_registry_find(strings + name_offset);
            if (candidate == nullptr) {
                // A winner naming a candidate this binary does not carry. Skip
                // it rather than rejecting the whole cache: the remaining
                // entries are still valid, and this one falls back to native.
                ++unknown;
                continue;
            }

            ggml_hip_digest key;
            memcpy(key.bytes, entry + ENT_DISPATCH, GGML_HIP_DIGEST_BYTES);

            if (g_winners.find(key) != g_winners.end()) {
                reject(path, "duplicate dispatch digest");
                g_winners.clear();
                return;
            }

            Winner winner = {};
            winner.candidate = candidate;
            memcpy(winner.signature_digest.bytes, entry + ENT_SIGNATURE,
                   GGML_HIP_DIGEST_BYTES);
            winner.variant.primary   = read_i32(entry + ENT_PRIMARY);
            winner.variant.secondary = read_i32(entry + ENT_SECONDARY);
            winner.variant.width     = read_i32(entry + ENT_WIDTH);
            winner.variant.acc_f16   = entry[ENT_ACC_F16];
            winner.variant.fallback  = entry[ENT_FALLBACK];
            winner.variant.small_k   = entry[ENT_SMALL_K];
            winner.variant.src0_type = entry[ENT_SRC0_TYPE];

            g_winners.emplace(key, winner);
        }

        g_loaded = true;
        g_stale  = stale;
        GGML_LOG_INFO("bigcherry: replay cache '%s' loaded, %zu winner(s)"
                      "%s\n", path, g_winners.size(),
                      unknown ? " (some entries name candidates absent from "
                                "this build and were skipped)" : "");
        if (unknown) {
            GGML_LOG_WARN("bigcherry: %zu cache entry/entries name unknown "
                          "candidates; those signatures use native selection\n",
                          unknown);
        }
        if (stale) {
            // Worth a warning rather than a note: the winners are still valid
            // and will be used, but they were chosen from a different set of
            // candidates than this binary carries, so "tuned" no longer means
            // "tuned for this build". Re-tune when the difference matters.
            GGML_LOG_WARN("bigcherry: replay cache was tuned against a "
                          "different candidate set (manifest hash differs from "
                          "this build's %s). Its winners are still valid and "
                          "are being used, but may no longer be the best "
                          "available; re-tune to refresh them.\n",
                          GGML_HIP_AUTOTUNE_MANIFEST_HASH_STR);
        }
    });
    return g_loaded;
}

bool ggml_hip_replay_lookup(const ggml_hip_digest & dispatch_digest,
                            const ggml_hip_digest & signature_digest,
                            const ggml_hip_candidate_descriptor ** out_candidate,
                            ggml_hip_variant_params * out_variant) {
    ggml_hip_replay_init();

    const auto found = g_winners.find(dispatch_digest);
    if (found == g_winners.end()) {
        return false;
    }
    if (!ggml_hip_digest_equal(found->second.signature_digest, signature_digest)) {
        return false;
    }
    *out_candidate = found->second.candidate;
    *out_variant   = found->second.variant;
    return true;
}

#ifdef GGML_HIP_REPLAY_DIAGNOSTICS
void ggml_hip_replay_record_hit(const ggml_hip_digest & dispatch_digest,
                                const ggml_hip_digest & signature_digest,
                                const ggml_hip_candidate_descriptor * candidate) {
    if (candidate == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_hits_mutex);
    auto found = g_hits.find(dispatch_digest);
    if (found != g_hits.end()) {
        ++found->second.calls;
        return;
    }
    Hit hit;
    hit.signature_digest = signature_digest;
    hit.candidate_name = candidate->stable_name;
    hit.calls = 1;
    g_hits.emplace(dispatch_digest, std::move(hit));
}

void ggml_hip_replay_flush_hits() {
    std::lock_guard<std::mutex> lock(g_hits_mutex);
    if (g_hits.empty()) {
        return;
    }
    const char * path = getenv("GGML_HIP_DISPATCH_HIT_LOG");
    if (path == nullptr || path[0] == '\0') {
        return;
    }
    FILE * file = fopen(path, "w");
    if (file == nullptr) {
        GGML_LOG_WARN("bigcherry: cannot write replay hit log '%s'\\n", path);
        return;
    }
    for (const auto & [digest, hit] : g_hits) {
        fprintf(file,
                "{\"dispatch\":\"%s\",\"signature\":\"%s\","
                "\"candidate\":\"%s\",\"calls\":%llu}\n",
                ggml_hip_digest_hex(digest).c_str(),
                ggml_hip_digest_hex(hit.signature_digest).c_str(),
                hit.candidate_name.c_str(),
                (unsigned long long) hit.calls);
    }
    fclose(file);
    GGML_LOG_INFO("bigcherry: wrote %zu replay hit(s) to '%s'\\n",
                  g_hits.size(), path);
}
#endif

void ggml_hip_replay_record_miss(const ggml_hip_digest & dispatch_digest,
                                 const ggml_hip_digest & signature_digest,
                                 const ggml_hip_dispatch_signature_v1 & sig,
                                 const ggml_hip_hardware_key_v1 & hw,
                                 const ggml_hip_candidate_descriptor * fallback) {
    GGML_UNUSED(hw);

    const char * policy = getenv("GGML_HIP_DISPATCH_MISS");
    if (policy == nullptr || strcmp(policy, "native-record") != 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_misses_mutex);
    ++g_misses_total;

    const auto existing = g_misses.find(dispatch_digest);
    if (existing != g_misses.end()) {
        ++existing->second.calls;
        return;
    }
    if (g_misses.size() >= MAX_TRACKED_MISSES) {
        return; // still counted in g_misses_total, no longer remembered
    }

    Miss miss;
    miss.signature_digest = signature_digest;
    miss.canonical_json   = ggml_hip_signature_json(sig, true);
    miss.fallback_name    = fallback ? fallback->stable_name : "";
    miss.calls            = 1;
    g_misses.emplace(dispatch_digest, miss);
}

void ggml_hip_replay_flush_misses() {
    std::lock_guard<std::mutex> lock(g_misses_mutex);
    if (g_misses.empty()) {
        return;
    }

    const char * path = getenv("GGML_HIP_DISPATCH_MISS_LOG");
    if (path == nullptr || path[0] == '\0') {
        GGML_LOG_INFO("bigcherry: %zu distinct replay miss(es), %llu call(s) "
                      "total. Set GGML_HIP_DISPATCH_MISS_LOG to record them.\n",
                      g_misses.size(), (unsigned long long) g_misses_total);
        return;
    }

    FILE * file = fopen(path, "w");
    if (file == nullptr) {
        GGML_LOG_WARN("bigcherry: cannot write miss log '%s'\n", path);
        return;
    }
    // JSON lines: appendable, and readable by the tools that turn misses into
    // the next round's inventory without needing a parser for a bespoke format.
    for (const auto & [digest, miss] : g_misses) {
        fprintf(file,
                "{\"dispatch\":\"%s\",\"signature\":\"%s\",\"calls\":%llu,"
                "\"fallback\":\"%s\",\"canonical\":%s}\n",
                ggml_hip_digest_hex(digest).c_str(),
                ggml_hip_digest_hex(miss.signature_digest).c_str(),
                (unsigned long long) miss.calls,
                miss.fallback_name.c_str(),
                miss.canonical_json.c_str());
    }
    fclose(file);
    GGML_LOG_INFO("bigcherry: wrote %zu replay miss(es) to '%s'\n",
                  g_misses.size(), path);
}

size_t ggml_hip_replay_entry_count() {
    return g_winners.size();
}

size_t ggml_hip_replay_miss_count() {
    return (size_t) g_misses_total;
}

bool ggml_hip_replay_is_stale() {
    ggml_hip_replay_init();
    return g_stale;
}

#endif // GGML_USE_HIP && GGML_HIP_DISPATCH
