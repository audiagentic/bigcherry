// bigcherry: compact immutable replay cache (HI11).

#include "hip-autotune-replay.h"

#if defined(GGML_USE_HIP) && defined(GGML_HIP_DISPATCH)

#include "hip-autotune-blake2b.h"
#include "hip-autotune-build-hash.h"
#include "hip-autotune-dispatch.cuh"
#include "hip-autotune-signature.h"

#include <atomic>
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

// An entry whose stable name this build's registry does not recognise, or
// whose implementation_version has drifted, still occupies a slot here
// (with candidate == nullptr / stale_impl == true) rather than being
// dropped at load time -- a lookup that reaches such a slot must classify
// as CANDIDATE_UNAVAILABLE, not fall through indistinguishably to MISS.
struct Winner {
    const ggml_hip_candidate_descriptor * candidate; // nullptr: unregistered
    bool                                   stale_impl_version;
    ggml_hip_variant_params               variant;
    ggml_hip_digest                       signature_digest;
    ggml_hip_digest                       manifest_hash;
    ggml_hip_digest                       source_revision_digest;
    uint32_t                              generation;
    bool                                  fresh;
    uint16_t                               transform_id; // 0 = GGML_HIP_TRANSFORM_NONE
    uint8_t                                match_kind;    // see ggml_hip_replay_match_kind
    // The stable name as written in the file's string table. For a registered
    // candidate the registry's own name is the canonical one (and any
    // divergence from it is itself a finding worth reporting); for an
    // unregistered one this is the only surviving copy of the name, because
    // the file's string table does not outlive ggml_hip_replay_init. Load-
    // time cost only; the lookup hot path never reads it.
    std::string                            stored_name;
};

std::unordered_map<ggml_hip_digest, std::vector<Winner>, DigestHash, DigestEqual> g_winners;
std::unordered_map<ggml_hip_digest, Miss, DigestHash, DigestEqual>   g_misses;
#ifdef GGML_HIP_REPLAY_DIAGNOSTICS
std::unordered_map<ggml_hip_digest, Hit, DigestHash, DigestEqual>     g_hits;
std::mutex g_hits_mutex;
#endif
std::mutex g_misses_mutex;
uint64_t   g_misses_total = 0;
std::atomic<size_t> g_outcomes[GGML_HIP_RESOLVE_COUNT] {};
// Whole-cache/entry-table level rejection reason, applied to every lookup
// once set: RERUN_REQUIRED for an obsolete-but-recognisable producer
// (format/ABI/schema drift), INCOMPATIBLE for a structurally wrong or
// corrupt file. Defaults to MISS (no cache configured is a plain miss, not
// a failure).
ggml_hip_resolution_v2 g_load_failure = GGML_HIP_RESOLVE_MISS;
bool       g_loaded = false;
// The loaded cache was tuned against a different candidate set. Kept as state
// rather than only logged so the coverage report can say so too -- a log line
// in a server that has been up for a week is not a usable answer to "is this
// build actually tuned".
bool       g_stale  = false;
// GGML_HIP_DISPATCH_REPLAY_REVISION_MATCH=OFF lets lookup fall back to the
// newest available entry for a dispatch key even when its source_revision or
// manifest_hash does not match this binary. implementation_version is the
// actual safety check in that mode (each entry is validated per-candidate at
// load time, see the loader loop); ON (default) keeps today's exact-match,
// fail-closed behavior.
bool       g_require_revision_match = true;

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
constexpr size_t ENT_IMPL_VER  = ENT_NAME_OFF + 4;
constexpr size_t ENT_PRIMARY   = ENT_IMPL_VER + 2;
constexpr size_t ENT_SECONDARY = ENT_PRIMARY + 4;
constexpr size_t ENT_WIDTH     = ENT_SECONDARY + 4;
constexpr size_t ENT_ACC_F16   = ENT_WIDTH + 4;
constexpr size_t ENT_FALLBACK  = ENT_ACC_F16 + 1;
// These two bytes were reserved in the v2 layout; v3 added implementation
// version before the variant fields so the loader can verify the manifest's
// candidate ABI rather than trusting the stable-name suffix alone. The two
// most recently added members of ggml_hip_variant_params could not round-trip
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
constexpr size_t ENT_SMALL_K   = ENT_FALLBACK + 1;
constexpr size_t ENT_SRC0_TYPE = ENT_SMALL_K + 1;
constexpr size_t ENT_MANIFEST  = ENT_SRC0_TYPE + 1;
constexpr size_t ENT_REVISION  = ENT_MANIFEST + GGML_HIP_DIGEST_BYTES;
constexpr size_t ENT_GENERATION = ENT_REVISION + GGML_HIP_DIGEST_BYTES;
// HI31 (v5): 0 = GGML_HIP_TRANSFORM_NONE. See hip-autotune-replay.h's
// ggml_hip_replay_entry::transform_id comment for why this is a raw
// uint16, not the (feature-gated) enum type.
constexpr size_t ENT_TRANSFORM = ENT_GENERATION + 4;
// HI74 (v5): see ggml_hip_replay_match_kind in hip-autotune-replay.h.
constexpr size_t ENT_MATCH_KIND = ENT_TRANSFORM + 2;
constexpr size_t ENT_SIZE      = ENT_MATCH_KIND + 1;

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

bool source_revision_matches(const uint8_t * stored) {
    const char * revision = GGML_HIP_AUTOTUNE_SOURCE_REVISION_STR;
    if (strlen(revision) != 40) {
        return false;
    }
    char normalized[41];
    for (size_t i = 0; i < 40; ++i) {
        const char ch = revision[i];
        normalized[i] = ch >= 'A' && ch <= 'F' ? (char) (ch - 'A' + 'a') : ch;
    }
    normalized[40] = '\0';
    ggml_hip_digest digest;
    ggml_hip_blake2b(digest.bytes, GGML_HIP_DIGEST_BYTES,
                     normalized, 40, GGML_HIP_PERSON_DISPATCH);
    return memcmp(digest.bytes, stored, GGML_HIP_DIGEST_BYTES) == 0;
}

ggml_hip_resolution_v2 counted(ggml_hip_resolution_v2 outcome) {
    g_outcomes[outcome].fetch_add(1, std::memory_order_relaxed);
    return outcome;
}

void reject(const char * path, const char * why, ggml_hip_resolution_v2 outcome) {
    // Deliberately a warning, not a fatal error: an unusable cache means this
    // process runs upstream's own selection, which is correct, merely untuned.
    g_load_failure = outcome;
    GGML_LOG_WARN("bigcherry: ignoring replay cache '%s': %s. "
                  "Falling back to native selection.\n", path, why);
}

} // namespace

bool ggml_hip_replay_init() {
    static std::once_flag once;
    std::call_once(once, []() {
        const char * revision_match = getenv("GGML_HIP_DISPATCH_REPLAY_REVISION_MATCH");
        if (revision_match != nullptr && strcmp(revision_match, "0") == 0) {
            g_require_revision_match = false;
        }

        const char * path = getenv("GGML_HIP_DISPATCH_CACHE");
        if (path == nullptr || path[0] == '\0') {
            return;
        }

        const std::vector<uint8_t> bytes = read_file(path);
        if (bytes.size() < HDR_SIZE) {
            reject(path, "file is shorter than its header", GGML_HIP_RESOLVE_INCOMPATIBLE);
            return;
        }
        const uint8_t * data = bytes.data();
        bool stale = false;

        if (read_u32(data + HDR_MAGIC) != GGML_HIP_REPLAY_MAGIC) {
            reject(path, "bad magic -- not a bigcherry replay cache", GGML_HIP_RESOLVE_INCOMPATIBLE);
            return;
        }
        if (read_u32(data + HDR_FORMAT) != GGML_HIP_REPLAY_VERSION) {
            reject(path, "container format version mismatch", GGML_HIP_RESOLVE_RERUN_REQUIRED);
            return;
        }
        if (read_u32(data + HDR_ARTIFACT) != GGML_HIP_AUTOTUNE_ARTIFACT_VERSION) {
            reject(path, "artifact version mismatch", GGML_HIP_RESOLVE_RERUN_REQUIRED);
            return;
        }
        // Standards 5.3: a signature schema change reinterprets hashed fields,
        // so every stored key from an older build is meaningless, not merely
        // stale.
        if (read_u16(data + HDR_SIG_SCHEMA) != GGML_HIP_SIGNATURE_SCHEMA_VERSION) {
            reject(path, "signature schema version mismatch", GGML_HIP_RESOLVE_RERUN_REQUIRED);
            return;
        }
        if (read_u16(data + HDR_HW_SCHEMA) != GGML_HIP_HARDWARE_SCHEMA_VERSION) {
            reject(path, "hardware key schema version mismatch", GGML_HIP_RESOLVE_RERUN_REQUIRED);
            return;
        }
        // v4 moves producer identity to each entry. A global mismatch is no
        // longer a reason to discard the file: a single cache may retain
        // several generations. Only an entry whose own manifest and source
        // revision match this binary is eligible for production replay.
        stale = !manifest_hash_matches(data + HDR_MANIFEST);

        const uint32_t entry_count  = read_u32(data + HDR_ENTRY_COUNT);
        const uint32_t string_bytes = read_u32(data + HDR_STRING_BYTES);
        const size_t   entries_at   = HDR_SIZE;
        if ((size_t) entry_count > (SIZE_MAX - entries_at) / ENT_SIZE) {
            reject(path, "entry table size overflows", GGML_HIP_RESOLVE_INCOMPATIBLE);
            return;
        }
        const size_t entries_bytes = (size_t) entry_count * ENT_SIZE;
        if (string_bytes > SIZE_MAX - entries_at - entries_bytes) {
            reject(path, "string table size overflows", GGML_HIP_RESOLVE_INCOMPATIBLE);
            return;
        }
        const size_t strings_at   = entries_at + entries_bytes;
        const size_t expected     = strings_at + string_bytes;
        if (bytes.size() != expected) {
            reject(path, "file is truncated", GGML_HIP_RESOLVE_INCOMPATIBLE);
            return;
        }

        // Checksum before trusting any offset in the payload.
        ggml_hip_digest content;
        ggml_hip_blake2b(content.bytes, GGML_HIP_DIGEST_BYTES,
                         data + entries_at, expected - entries_at,
                         GGML_HIP_PERSON_DISPATCH);
        if (memcmp(content.bytes, data + HDR_CONTENT, GGML_HIP_DIGEST_BYTES) != 0) {
            reject(path, "content checksum mismatch -- the file is corrupt", GGML_HIP_RESOLVE_INCOMPATIBLE);
            return;
        }

        const char * strings = (const char *) (data + strings_at);
        size_t unknown = 0;
        size_t stale_impl_version = 0;
        size_t unrecognized_match_kind = 0;

        for (uint32_t i = 0; i < entry_count; ++i) {
            const uint8_t * entry = data + entries_at + (size_t) i * ENT_SIZE;

            const uint32_t name_offset = read_u32(entry + ENT_NAME_OFF);
            if (name_offset >= string_bytes) {
                reject(path, "entry names a string outside the string table", GGML_HIP_RESOLVE_INCOMPATIBLE);
                g_winners.clear();
                return;
            }
            // The table must be NUL-terminated within its own bounds, or
            // strcmp below would walk off the end.
            if (memchr(strings + name_offset, '\0',
                       string_bytes - name_offset) == nullptr) {
                reject(path, "string table is not NUL-terminated", GGML_HIP_RESOLVE_INCOMPATIBLE);
                g_winners.clear();
                return;
            }

            const ggml_hip_candidate_descriptor * candidate =
                ggml_hip_registry_find(strings + name_offset);

            ggml_hip_digest key;
            memcpy(key.bytes, entry + ENT_DISPATCH, GGML_HIP_DIGEST_BYTES);

            // An unregistered candidate or an implementation_version
            // mismatch means this one entry cannot be used, but a lookup
            // that reaches its key/signature must still say so explicitly
            // (CANDIDATE_UNAVAILABLE) rather than falling through to MISS,
            // which would misreport "never tuned" as the reason. So the
            // slot is still recorded, just with candidate == nullptr (or
            // stale_impl_version set) rather than being dropped. It is not
            // evidence about any other entry, including other generations
            // of the same build or entries retained from other builds --
            // only this one slot is marked unusable (previously an
            // implementation_version mismatch cleared g_winners entirely).
            bool stale_impl = candidate != nullptr &&
                read_u16(entry + ENT_IMPL_VER) != candidate->implementation_version;
            // HI74: an entry whose match_kind this loader does not recognise
            // must be rejected, never silently reinterpreted as EXACT (that
            // would validate a future generalised-entry key form as if it
            // were a plain exact digest match). Marked unusable the same
            // way as an unregistered candidate -- this one slot only, not
            // the whole cache.
            const bool unrecognized_match =
                entry[ENT_MATCH_KIND] != GGML_HIP_REPLAY_MATCH_EXACT;
            if (candidate == nullptr) {
                ++unknown;
            } else if (stale_impl) {
                ++stale_impl_version;
            }
            if (unrecognized_match) {
                ++unrecognized_match_kind;
            }

            Winner winner = {};
            winner.candidate          = (stale_impl || unrecognized_match) ? nullptr : candidate;
            winner.stale_impl_version = stale_impl;
            winner.stored_name        = strings + name_offset;
            memcpy(winner.signature_digest.bytes, entry + ENT_SIGNATURE,
                   GGML_HIP_DIGEST_BYTES);
            winner.variant.primary   = read_i32(entry + ENT_PRIMARY);
            winner.variant.secondary = read_i32(entry + ENT_SECONDARY);
            winner.variant.width     = read_i32(entry + ENT_WIDTH);
            winner.variant.acc_f16   = entry[ENT_ACC_F16];
            winner.variant.fallback  = entry[ENT_FALLBACK];
            winner.variant.small_k   = entry[ENT_SMALL_K];
            winner.variant.src0_type = entry[ENT_SRC0_TYPE];
            memcpy(winner.manifest_hash.bytes, entry + ENT_MANIFEST,
                   GGML_HIP_DIGEST_BYTES);
            memcpy(winner.source_revision_digest.bytes, entry + ENT_REVISION,
                   GGML_HIP_DIGEST_BYTES);
            winner.generation = read_u32(entry + ENT_GENERATION);
            winner.transform_id = read_u16(entry + ENT_TRANSFORM);
            winner.match_kind   = entry[ENT_MATCH_KIND];

            auto & generations = g_winners[key];
            for (const Winner & prior : generations) {
                if (prior.generation == winner.generation &&
                    ggml_hip_digest_equal(prior.manifest_hash, winner.manifest_hash) &&
                    ggml_hip_digest_equal(prior.source_revision_digest,
                                          winner.source_revision_digest)) {
                    reject(path, "duplicate generation identity", GGML_HIP_RESOLVE_INCOMPATIBLE);
                    g_winners.clear();
                    return;
                }
            }
            winner.fresh = manifest_hash_matches(winner.manifest_hash.bytes) &&
                           source_revision_matches(winner.source_revision_digest.bytes);
            if (!winner.fresh) {
                stale = true;
            }
            generations.push_back(winner);
        }

        g_loaded = true;
        g_stale  = stale;
        size_t winner_count = 0;
        for (const auto & [digest, generations] : g_winners) {
            GGML_UNUSED(digest);
            winner_count += generations.size();
        }
        GGML_LOG_INFO("bigcherry: replay cache '%s' loaded, %zu winner(s)"
                      " across %zu key(s)%s\n", path, winner_count,
                      g_winners.size(),
                      unknown ? " (some entries name candidates absent from "
                                "this build and were skipped)" : "");
        if (unknown) {
            GGML_LOG_WARN("bigcherry: %zu cache entry/entries name unknown "
                          "candidates; those signatures use native selection\n",
                          unknown);
        }
        if (stale_impl_version) {
            GGML_LOG_WARN("bigcherry: %zu cache entry/entries have an "
                          "implementation_version that no longer matches "
                          "this build's candidate registry and were skipped; "
                          "other entries in the cache were retained\n",
                          stale_impl_version);
        }
        if (unrecognized_match_kind) {
            // HI74: expected once generalised entries (HI36b) exist and this
            // binary predates that feature -- not a corruption signal, so a
            // warning, not a reject().
            GGML_LOG_WARN("bigcherry: %zu cache entry/entries use a "
                          "match_kind this build does not recognise and "
                          "were skipped; other entries in the cache were "
                          "retained\n", unrecognized_match_kind);
        }
    });
    return g_loaded;
}

ggml_hip_resolution_v2 ggml_hip_replay_lookup(
                            const ggml_hip_digest & dispatch_digest,
                            const ggml_hip_digest & signature_digest,
                            const ggml_hip_dispatch_signature_v1 & sig,
                            const ggml_hip_hardware_key_v1 & hw,
                            const ggml_hip_candidate_descriptor ** out_candidate,
                            ggml_hip_variant_params * out_variant,
                            uint16_t * out_transform_id) {
    if (!ggml_hip_replay_init()) {
        // No cache configured is a plain miss; a cache that failed to load
        // carries the specific reason recorded by reject() during init.
        return counted(g_load_failure);
    }

    const auto found = g_winners.find(dispatch_digest);
    if (found == g_winners.end()) {
        return counted(GGML_HIP_RESOLVE_MISS);
    }

    bool saw_signature_match     = false;
    bool saw_stale_generation    = false;
    bool saw_candidate_rejected  = false;

    // Entries are stored newest-generation-first per key (the exporter sorts
    // descending by generation), so the first usable match here is always
    // the newest available winner for this dispatch key.
    for (const Winner & winner : found->second) {
        if (!ggml_hip_digest_equal(winner.signature_digest, signature_digest)) {
            continue;
        }
        saw_signature_match = true;
        // HI31: a transformed entry (transform_id != 0) always requires
        // exact revision match, even when GGML_HIP_DISPATCH_REPLAY_REVISION_
        // MATCH=0 relaxes that for plain candidates -- transforms carry no
        // implementation_version of their own yet to validate safely
        // against a relaxed match (see hip-autotune-replay.h).
        const bool must_be_fresh = winner.transform_id != 0 || g_require_revision_match;
        if (must_be_fresh && !winner.fresh) {
            saw_stale_generation = true;
            continue;
        }
        // EXACT requires more than "the digest matched": the candidate must
        // still be registered and must support this architecture -- checked
        // here, not after the caller has already committed to the binding,
        // so a candidate that resolves but then fails these checks is never
        // silently counted as a hit (the blind spot the pre-reset design
        // closed).
        //
        // HI31: can_execute() against `sig` is run ONLY for a plain entry
        // (transform_id == 0). For a transformed entry, `sig` is the
        // ORIGINAL untransformed signature -- checking the candidate
        // against it is exactly the wrong question (the candidate was
        // never expected to accept that shape; that mismatch is why the
        // transform exists). This function has no way to compute the
        // actual transformed signature (it only has `sig`/`hw`, not the
        // real ggml_hip_launch_context transform->apply() needs), so EXACT
        // here is NECESSARY but not SUFFICIENT for a transformed entry --
        // see this function's declaration in hip-autotune-replay.h for the
        // caller's required second-layer validation.
        if (winner.candidate == nullptr ||
            !ggml_hip_candidate_supports_arch(*winner.candidate, hw) ||
            (winner.transform_id == 0 &&
             !winner.candidate->can_execute(winner.candidate, sig, hw))) {
            saw_candidate_rejected = true;
            continue;
        }
        *out_candidate     = winner.candidate;
        *out_variant       = winner.variant;
        *out_transform_id  = winner.transform_id;
        return counted(GGML_HIP_RESOLVE_EXACT);
    }

    if (saw_candidate_rejected) {
        return counted(GGML_HIP_RESOLVE_CANDIDATE_UNAVAILABLE);
    }
    if (saw_stale_generation) {
        // Only non-fresh generations exist for this key+signature while
        // revision matching is required: a tune rerun against the current
        // build is what would produce a usable entry.
        return counted(GGML_HIP_RESOLVE_RERUN_REQUIRED);
    }
    if (saw_signature_match) {
        // Unreachable given the loop above (every signature match falls
        // into one of the two branches), kept as a defensive fallback.
        return counted(GGML_HIP_RESOLVE_CANDIDATE_UNAVAILABLE);
    }
    // The dispatch key has entries, but none share this lookup's signature
    // digest: the cache and this binary disagree about what the key means.
    return counted(GGML_HIP_RESOLVE_INCOMPATIBLE);
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
    size_t count = 0;
    for (const auto & [digest, generations] : g_winners) {
        GGML_UNUSED(digest);
        count += generations.size();
    }
    return count;
}

size_t ggml_hip_replay_miss_count() {
    return (size_t) g_misses_total;
}

size_t ggml_hip_replay_resolution_count(ggml_hip_resolution_v2 outcome) {
    return outcome < GGML_HIP_RESOLVE_COUNT
        ? g_outcomes[outcome].load(std::memory_order_relaxed) : 0;
}

const char * ggml_hip_replay_resolution_name(ggml_hip_resolution_v2 outcome) {
    switch (outcome) {
        case GGML_HIP_RESOLVE_EXACT:                 return "exact";
        case GGML_HIP_RESOLVE_CANDIDATE_UNAVAILABLE: return "candidate_unavailable";
        case GGML_HIP_RESOLVE_RERUN_REQUIRED:        return "rerun_required";
        case GGML_HIP_RESOLVE_INCOMPATIBLE:          return "incompatible";
        case GGML_HIP_RESOLVE_MISS:                  return "misses";
        case GGML_HIP_RESOLVE_COUNT:                 break;
    }
    return "unknown";
}

bool ggml_hip_replay_is_stale() {
    ggml_hip_replay_init();
    return g_stale;
}

ggml_hip_resolution_v2 ggml_hip_replay_load_failure() {
    // The loader's own classification, unchanged: MISS when nothing was
    // configured or no cache was found, RERUN_REQUIRED / INCOMPATIBLE when a
    // cache was rejected. Callers check ggml_hip_replay_init()'s return first
    // and treat a loaded cache as having no failure.
    return g_load_failure;
}

size_t ggml_hip_replay_foreach_winner(
        bool (*visit)(const ggml_hip_digest & dispatch_digest,
                      const ggml_hip_replay_winner_info * info, void * user),
        void * user) {
    if (visit == nullptr || !ggml_hip_replay_init()) {
        return 0;
    }
    size_t visited = 0;
    for (const auto & [digest, generations] : g_winners) {
        for (const Winner & winner : generations) {
            ggml_hip_replay_winner_info info = {};
            info.candidate_name = winner.candidate != nullptr
                ? winner.candidate->stable_name : winner.stored_name.c_str();
            info.registered = winner.candidate != nullptr;
            info.stale_impl_version = winner.stale_impl_version;
            info.unrecognized_match = winner.match_kind != GGML_HIP_REPLAY_MATCH_EXACT;
            info.fresh = winner.fresh;
            info.generation = winner.generation;
            info.transform_id = winner.transform_id;
            info.match_kind = winner.match_kind;
            ++visited;
            if (!visit(digest, &info, user)) {
                return visited;
            }
        }
    }
    return visited;
}

#endif // GGML_USE_HIP && GGML_HIP_DISPATCH
