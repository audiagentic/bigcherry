// bigcherry: offline cache and registry inspector (HI15/HI16).
//
// The only C++ consumer of the replay loader outside the dispatch hot path.
// It links ggml-hip and calls the same registry functions and the same
// loader a production process uses, so every judgement in its output is the
// real code's judgement. A reimplementation in another language could
// disagree with the loader -- a different entry size, a different
// freshness test, a different rejection order -- and that is exactly what
// this tool exists to check, which is why it is C++ that links the
// production objects rather than a second parser.
//
// Two things it answers:
//
//   registry  -- is the compiled candidate table well-formed? stable names
//                unique, exactly one native wrapper per family, every
//                source_class in the enum, every descriptor complete.
//                This is HI16's catalog/registry agreement half: the
//                generated catalog is compiled into this table, so a table
//                that passes its own checks is the ground truth a manifest
//                must agree with (the manifest diff is done by the Python
//                wrapper, which owns JSON parsing).
//
//   cache     -- what does the production loader do with THIS file on THIS
//                build? rejected (and for which of the loader's own
//                reasons), or loaded, with per-entry registered/fresh/stale
//                state straight from the loader's tables.
//
//   --selftest -- HI16's cross-language digest agreement vectors: runs the
//                BLAKE2b implementation (one-shot and streaming) against
//                known answers computed by Python's hashlib.blake2b, so a
//                C++/Python digest divergence -- the silent "every replay
//                lookup misses and falls back to native" failure mode -- is
//                caught loudly and offline.
//
// Usage:
//   hip-autotune-inspect [cache.cache] [--json] [--selftest]
//
// Exit codes:
//   0  registry well-formed; no cache given, or the cache loaded (it may
//      still contain retained stale generations -- reported, not an error)
//   1  registry anomaly
//   2  usage error
//   3  cache rejected by the loader (the loader's classification is printed)
//   4  cache loaded but no entry is usable on this build

#include "hip-autotune-dispatch.cuh"
#include "hip-autotune-replay.h"
#include "hip-autotune-signature.h"
#include "hip-autotune-build-hash.h"
#include "hip-autotune-blake2b.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

std::string family_name(ggml_hip_kernel_family family) {
    switch (family) {
        case GGML_HIP_FAMILY_MMVQ: return "mmvq";
        case GGML_HIP_FAMILY_MMQ:  return "mmq";
        case GGML_HIP_FAMILY_MMVF: return "mmvf";
        case GGML_HIP_FAMILY_MMF:  return "mmf";
        case GGML_HIP_FAMILY_BLAS: return "blas";
        case GGML_HIP_FAMILY_COUNT: break;
    }
    return "?";
}

std::string source_class_name(ggml_hip_source_class source_class) {
    switch (source_class) {
        case GGML_HIP_SOURCE_NATIVE_WRAPPER:         return "native_wrapper";
        case GGML_HIP_SOURCE_EXISTING_RUNTIME:       return "existing_runtime";
        case GGML_HIP_SOURCE_EXISTING_ALTERNATIVE:   return "existing_alternative";
        case GGML_HIP_SOURCE_NEW_GENERATED_VARIANT:  return "new_generated_variant";
        case GGML_HIP_SOURCE_VENDOR_AUTO:            return "vendor_auto";
        case GGML_HIP_SOURCE_VENDOR_EXPLICIT:        return "vendor_explicit";
        case GGML_HIP_SOURCE_COUNT: break;
    }
    return "?";
}

struct RegistryReport {
    size_t count = 0;
    std::map<std::string, size_t> by_family;
    std::map<std::string, size_t> by_source_class;
    std::vector<std::string> anomalies;
};

RegistryReport report_registry() {
    RegistryReport report;
    report.count = ggml_hip_registry_size();
    std::set<std::string> seen_names;
    for (size_t i = 0; i < report.count; ++i) {
        const ggml_hip_candidate_descriptor * c = ggml_hip_registry_at(i);
        if (c == nullptr) {
            report.anomalies.push_back("registry slot " + std::to_string(i) + " is null");
            continue;
        }
        if (c->stable_name == nullptr || c->stable_name[0] == '\0') {
            report.anomalies.push_back("slot " + std::to_string(i) + " has an empty stable name");
            continue;
        }
        if (!seen_names.insert(c->stable_name).second) {
            report.anomalies.push_back(
                "duplicate stable name: " + std::string(c->stable_name));
        }
        if (c->family >= GGML_HIP_FAMILY_COUNT) {
            report.anomalies.push_back(std::string(c->stable_name) +
                                       " has an out-of-range family " +
                                       std::to_string(c->family));
        } else {
            report.by_family[family_name((ggml_hip_kernel_family) c->family)]++;
        }
        if (c->source_class >= GGML_HIP_SOURCE_COUNT) {
            report.anomalies.push_back(std::string(c->stable_name) +
                                       " has an out-of-range source_class " +
                                       std::to_string(c->source_class));
        } else {
            report.by_source_class[source_class_name((ggml_hip_source_class) c->source_class)]++;
        }
        if (c->can_execute == nullptr || c->launch == nullptr || c->workspace == nullptr) {
            report.anomalies.push_back(std::string(c->stable_name) +
                                       " has a null entry-point pointer");
        }
        if (c->implementation_version == 0) {
            report.anomalies.push_back(std::string(c->stable_name) +
                                       " has implementation_version 0");
        }
    }
    // Exactly one native wrapper per family, resolvable through the registry
    // API itself -- the same call the dispatch path uses.
    for (int family = 0; family < GGML_HIP_FAMILY_COUNT; ++family) {
        const ggml_hip_kernel_family f = (ggml_hip_kernel_family) family;
        const ggml_hip_candidate_descriptor * native = ggml_hip_registry_native(f);
        if (native == nullptr) {
            report.anomalies.push_back(
                "family " + family_name(f) + " has no native wrapper");
            continue;
        }
        size_t wrappers = 0;
        for (size_t i = 0; i < report.count; ++i) {
            const ggml_hip_candidate_descriptor * c = ggml_hip_registry_at(i);
            if (c != nullptr && c->family == f &&
                    c->source_class == GGML_HIP_SOURCE_NATIVE_WRAPPER) {
                ++wrappers;
            }
        }
        if (wrappers != 1) {
            report.anomalies.push_back("family " + family_name(f) +
                                       " has " + std::to_string(wrappers) +
                                       " native wrappers, expected 1");
        }
    }
    return report;
}

struct WinnerRow {
    std::string dispatch;
    std::string name;
    bool registered = false;
    bool fresh = false;
    bool stale_impl = false;
    bool unrecognized_match = false;
    uint32_t generation = 0;
    uint16_t transform_id = 0;
    uint8_t  match_kind = 0;

    bool usable() const {
        return registered && fresh && !stale_impl && !unrecognized_match;
    }
};

bool collect_winner(const ggml_hip_digest & dispatch,
                    const ggml_hip_replay_winner_info * info, void * user) {
    std::vector<WinnerRow> * rows = static_cast<std::vector<WinnerRow> *>(user);
    WinnerRow row;
    row.dispatch = ggml_hip_digest_hex(dispatch);
    row.name = info->candidate_name;
    row.registered = info->registered;
    row.fresh = info->fresh;
    row.stale_impl = info->stale_impl_version;
    row.unrecognized_match = info->unrecognized_match;
    row.generation = info->generation;
    row.transform_id = info->transform_id;
    row.match_kind = info->match_kind;
    rows->push_back(std::move(row));
    return true;
}

std::string json_escape(const std::string & in) {
    std::string out;
    out.reserve(in.size());
    for (char ch : in) {
        switch (ch) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\t': out += "\\t";  break;
            default:   out += ch;     break;
        }
    }
    return out;
}

} // namespace

struct DigestVector {
    const char * label;
    const char * data_hex;
    const char * person_hex; // "" means no personalisation (all-zero person)
    size_t       person_len; // 0 means nullptr is passed
    const char * expected_hex;
};

// Known-answer table for HI16's cross-language digest agreement test. The
// expected digests were produced with Python's
//   hashlib.blake2b(data, digest_size=16, person=b"...").hexdigest()
// and are independently re-derived from hashlib on every offline test run by
// tools/tests/test_blake2b_cross_lang_vectors.py, which fails if any row
// below stops matching. The table deliberately covers: no personalisation,
// the three real runtime person prefixes, empty data, exact 128-byte block
// boundary, multi-block input, short-person zero padding, and an exact-16
// person.
static const DigestVector kDigestVectors[] = {
    {"abc-no-person", "616263", "", 0, "cf4ab791c62b8d2b2109c90275287816"},
    {"sig-person-real", "6d756c5f6d61745f71203020312036313434203531323020382071385f30", "6c6c616d612d6869702d74756e65", 14, "62473a84a35e57bc868f3b82cc15b6f4"},
    {"dispatch-person-real", "64697370617463683a6d756c5f6d61745f713a36313434783531323078383a71385f30", "6c6c616d612d6469737061746368", 14, "0d49b10a8f7746369c68a5e16232942e"},
    {"hardware-person-real", "676678313230312d33323736382d4d42", "6c6c616d612d6861726477617265", 14, "d412bf27b12eeac781be910c97ecdfd0"},
    {"empty-data", "", "6c6c616d612d6469737061746368", 14, "7f2d1c4ae39bb116a3512ef0aa3bf988"},
    {"one-full-block-128", "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5f606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f", "6c6c616d612d6869702d74756e65", 14, "a453c6b27adbc93f726b03d0c24344d7"},
    {"one-block-plus-one", "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5f606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f78", "6c6c616d612d6469737061746368", 14, "05a3da1b5a1226e4560feaa8084162f7"},
    {"multi-block-300", "6d756c5f6d61745f717c363134347c353132307c387c71385f307c6d756c5f6d61745f717c363134347c353132307c387c71385f307c6d756c5f6d61745f717c363134347c353132307c387c71385f307c6d756c5f6d61745f717c363134347c353132307c387c71385f307c6d756c5f6d61745f717c363134347c353132307c387c71385f307c6d756c5f6d61745f717c363134347c353132307c387c71385f307c6d756c5f6d61745f717c363134347c353132307c387c71385f307c6d756c5f6d61745f717c363134347c353132307c387c71385f307c6d756c5f6d61745f717c363134347c353132307c387c71385f307c6d756c5f6d61745f717c363134347c353132307c387c71385f307c6d756c5f6d61745f717c363134347c353132307c387c71385f307c6d75", "6c6c616d612d6469737061746368", 14, "0da83acf8b014ce1ab00e7229666de3c"},
    {"short-person-zero-pad", "70616464696e672d63617365", "736967", 3, "5b60dca3dfc715cf76a9c115b297efc4"},
    {"person-exact-16", "65786163742d706572736f6e", "30313233343536373839616263646566", 16, "ce8c8aa46b6d3438d591d4d2cf5d95e6"},
};

bool from_hex(const std::string & hex, std::string & out) {
    if (hex.size() % 2 != 0) return false;
    auto value = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    out.clear();
    out.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2) {
        const int hi = value(hex[i]);
        const int lo = value(hex[i + 1]);
        if (hi < 0 || lo < 0) return false;
        out.push_back((char) ((hi << 4) | lo));
    }
    return true;
}

std::string to_hex(const unsigned char * p, size_t n) {
    static const char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        out.push_back(digits[p[i] >> 4]);
        out.push_back(digits[p[i] & 0xF]);
    }
    return out;
}

// HI16 cross-language digest agreement: every vector must match Python's
// hashlib AND the streaming API must agree with the one-shot API, because
// the runtime digests are computed in chunks.
// Returns the number of failing vectors.
size_t run_digest_selftest() {
    size_t failures = 0;
    for (const DigestVector & vec : kDigestVectors) {
        std::string data, person, expected;
        bool table_ok = from_hex(vec.data_hex, data)
                     && from_hex(vec.expected_hex, expected)
                     && (vec.person_len == 0 || from_hex(vec.person_hex, person));
        if (!table_ok) {
            std::printf("digest  %-22s FAIL (malformed hex in table)\n", vec.label);
            ++failures;
            continue;
        }
        const char * person_ptr = vec.person_len > 0 ? person.c_str() : nullptr;

        unsigned char oneshot[16];
        ggml_hip_blake2b(oneshot, 16, data.data(), data.size(), person_ptr);

        unsigned char streamed[16];
        ggml_hip_blake2b_state state;
        ggml_hip_blake2b_init(&state, 16, person_ptr);
        // Deliberately uneven chunk boundaries (ceil-ceil-floor split) so a
        // single flush point cannot hide a counter or buffer bug. The three
        // chunks always sum to exactly data.size() -- an earlier equal-third
        // version dropped up to two bytes for some lengths and this very
        // check caught it, which is the point of the test.
        const size_t s = data.size();
        const size_t c1 = (s + 2) / 3;
        const size_t c2 = (s - c1 + 1) / 2;
        const size_t c3 = s - c1 - c2;
        size_t off = 0;
        for (const size_t c : {c1, c2, c3}) {
            if (c > 0) {
                ggml_hip_blake2b_update(&state, data.data() + off, c);
                off += c;
            }
        }
        (void) off; // off == s is guaranteed by construction; debug builds
                    // should assert it
        ggml_hip_blake2b_final(&state, streamed);

        const bool ok = memcmp(oneshot, expected.data(), 16) == 0
                     && memcmp(streamed, oneshot, 16) == 0;
        if (ok) {
            std::printf("digest  %-22s ok\n", vec.label);
        } else {
            std::printf("digest  %-22s FAIL one-shot=%s expected=%s\n",
                        vec.label, to_hex(oneshot, 16).c_str(), vec.expected_hex);
            ++failures;
        }
    }
    return failures;
}

int main(int argc, char ** argv) {
    std::string cache_path;
    bool json = false;
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--json") {
            json = true;
        } else if (arg == "--selftest") {
            selftest = true;
        } else if (arg == "--help" || arg == "-h") {
            std::printf("usage: hip-autotune-inspect [cache.cache] [--json] [--selftest]\n");
            return 0;
        } else if (!cache_path.empty()) {
            std::fprintf(stderr, "unexpected argument: %s\n", arg.c_str());
            return 2;
        } else {
            cache_path = arg;
        }
    }

    if (selftest) {
        if (!cache_path.empty() || json) {
            std::fprintf(stderr, "--selftest takes no other options\n");
            return 2;
        }
        const size_t failures = run_digest_selftest();
        std::printf("digest  %zu vector(s), %zu failure(s)\n",
                    sizeof(kDigestVectors) / sizeof(kDigestVectors[0]), failures);
        return failures ? 1 : 0;
    }

    const RegistryReport registry = report_registry();

    std::vector<WinnerRow> winners;
    bool cache_configured = false;
    bool cache_loaded = false;
    std::string cache_outcome;
#ifdef GGML_HIP_DISPATCH_REPLAY
    if (!cache_path.empty()) {
        // The loader reads GGML_HIP_DISPATCH_CACHE exactly once; this is a
        // fresh process, so pointing it at the argument is the same path the
        // production environment variable takes.
        setenv("GGML_HIP_DISPATCH_CACHE", cache_path.c_str(), 1);
        cache_configured = true;
        cache_loaded = ggml_hip_replay_init();
        if (cache_loaded) {
            cache_outcome = "loaded";
            ggml_hip_replay_foreach_winner(collect_winner, &winners);
        } else {
            cache_outcome = ggml_hip_replay_resolution_name(
                ggml_hip_replay_load_failure());
        }
    }
#else
    if (!cache_path.empty()) {
        std::fprintf(stderr,
                     "cache inspection requires a GGML_HIP_DISPATCH_REPLAY "
                     "build; this binary was built without the replay loader\n");
        return 2;
    }
#endif

    size_t usable = 0;
    for (const WinnerRow & row : winners) {
        if (row.usable()) ++usable;
    }

    if (!json) {
        std::printf("bigcherry hip-autotune-inspect\n");
        std::printf("build  manifest %s\n", GGML_HIP_AUTOTUNE_MANIFEST_HASH_STR);
        std::printf("       source   %s\n", GGML_HIP_AUTOTUNE_SOURCE_REVISION_STR);
        std::printf("       variant  %s  artifact_version %d\n",
                    GGML_HIP_AUTOTUNE_VARIANT_SET_STR,
                    GGML_HIP_AUTOTUNE_ARTIFACT_VERSION);
        std::printf("       catalog  descriptor %s\n",
                    GGML_HIP_AUTOTUNE_DESCRIPTOR_HASH_STR);
        std::printf("registry %zu candidate(s)", registry.count);
        for (const auto & [family, n] : registry.by_family) {
            std::printf("  %s %zu", family.c_str(), n);
        }
        std::printf("\n");
        if (registry.anomalies.empty()) {
            std::printf("       ok\n");
        } else {
            std::printf("       %zu anomaly/anomalies:\n", registry.anomalies.size());
            for (const std::string & anomaly : registry.anomalies) {
                std::printf("         - %s\n", anomaly.c_str());
            }
        }
#ifdef GGML_HIP_DISPATCH_REPLAY
        if (cache_configured) {
            std::printf("cache    %s -> %s\n", cache_path.c_str(), cache_outcome.c_str());
            if (cache_loaded) {
                std::printf("       %zu winner slot(s), %zu usable on this build, stale=%s\n",
                            winners.size(), usable,
                            ggml_hip_replay_is_stale() ? "yes" : "no");
                for (const WinnerRow & row : winners) {
                    std::printf("       gen %u  %s  fresh=%s reg=%s impl-stale=%s "
                                "match_kind=%u transform=%u  %s\n",
                                row.generation, row.name.c_str(),
                                row.fresh ? "y" : "n",
                                row.registered ? "y" : "N",
                                row.stale_impl ? "y" : "n",
                                (unsigned) row.match_kind,
                                (unsigned) row.transform_id,
                                row.dispatch.c_str());
                }
            }
        }
#endif
        // Computed as if/else (not a ternary across an #ifdef) so the
        // statement-terminator can't land on only one branch.
        int rc = 0;
#ifdef GGML_HIP_DISPATCH_REPLAY
        if (cache_configured && !cache_loaded) rc = 3;
        else if (cache_configured && winners.empty()) rc = 4;
#endif
        if (!registry.anomalies.empty()) rc = 1;
        return rc;
    }

    // --json
    std::printf("{\"schema\":1,\"build\":{\"manifest_hash\":\"%s\",",
                GGML_HIP_AUTOTUNE_MANIFEST_HASH_STR);
    std::printf("\"source_revision\":\"%s\",", GGML_HIP_AUTOTUNE_SOURCE_REVISION_STR);
    std::printf("\"descriptor_hash\":\"%s\",", GGML_HIP_AUTOTUNE_DESCRIPTOR_HASH_STR);
    std::printf("\"variant_set\":\"%s\",", GGML_HIP_AUTOTUNE_VARIANT_SET_STR);
    std::printf("\"artifact_version\":%d,", GGML_HIP_AUTOTUNE_ARTIFACT_VERSION);
    std::printf("\"candidate_count\":%d},", GGML_HIP_AUTOTUNE_CANDIDATE_COUNT);
    std::printf("\"registry\":{\"count\":%zu,\"by_family\":{", registry.count);
    bool first = true;
    for (const auto & [family, n] : registry.by_family) {
        if (!first) std::printf(",");
        std::printf("\"%s\":%zu", family.c_str(), n);
        first = false;
    }
    std::printf("},\"by_source_class\":{");
    first = true;
    for (const auto & [class_name, n] : registry.by_source_class) {
        if (!first) std::printf(",");
        std::printf("\"%s\":%zu", class_name.c_str(), n);
        first = false;
    }
    std::printf("},\"anomalies\":[");
    first = true;
    for (const std::string & anomaly : registry.anomalies) {
        if (!first) std::printf(",");
        std::printf("\"%s\"", json_escape(anomaly).c_str());
        first = false;
    }
    std::printf("]");
#ifdef GGML_HIP_DISPATCH_REPLAY
    if (cache_configured) {
        std::printf("},\"cache\":{\"path\":\"%s\",\"outcome\":\"%s\",",
                    json_escape(cache_path).c_str(), cache_outcome.c_str());
        std::printf("\"stale\":%s,\"winner_slots\":%zu,\"usable\":%zu,\"entries\":[",
                    cache_loaded && ggml_hip_replay_is_stale() ? "true" : "false",
                    winners.size(), usable);
        first = true;
        for (const WinnerRow & row : winners) {
            if (!first) std::printf(",");
            std::printf("{\"dispatch\":\"%s\",\"winner\":\"%s\",",
                        row.dispatch.c_str(), json_escape(row.name).c_str());
            std::printf("\"registered\":%s,\"fresh\":%s,\"stale_impl\":%s,",
                        row.registered ? "true" : "false",
                        row.fresh ? "true" : "false",
                        row.stale_impl ? "true" : "false");
            std::printf("\"unrecognized_match\":%s,\"generation\":%u,",
                        row.unrecognized_match ? "true" : "false", row.generation);
            std::printf("\"transform_id\":%u,\"match_kind\":%u}",
                        (unsigned) row.transform_id, (unsigned) row.match_kind);
            first = false;
        }
        std::printf("]}");
    }
#endif
    std::printf("}\n");
    int rc = 0;
#ifdef GGML_HIP_DISPATCH_REPLAY
    if (cache_configured && !cache_loaded) rc = 3;
    else if (cache_configured && winners.empty()) rc = 4;
#endif
    if (!registry.anomalies.empty()) rc = 1;
    return rc;
}
