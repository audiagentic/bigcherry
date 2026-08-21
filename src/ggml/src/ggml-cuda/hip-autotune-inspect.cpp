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
// Usage:
//   hip-autotune-inspect [cache.cache] [--json]
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

int main(int argc, char ** argv) {
    std::string cache_path;
    bool json = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--json") {
            json = true;
        } else if (arg == "--help" || arg == "-h") {
            std::printf("usage: hip-autotune-inspect [cache.cache] [--json]\n");
            return 0;
        } else if (!cache_path.empty()) {
            std::fprintf(stderr, "unexpected argument: %s\n", arg.c_str());
            return 2;
        } else {
            cache_path = arg;
        }
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
        return (!registry.anomalies.empty()) ? 1
#ifdef GGML_HIP_DISPATCH_REPLAY
               : (cache_configured && !cache_loaded) ? 3
               : (cache_configured && winners.empty() ? 4 : 0)
#else
               : 0;
#endif
    }

    // --json
    std::printf("{\"schema\":1,\"build\":{\"manifest_hash\":\"%s\",",
                GGML_HIP_AUTOTUNE_MANIFEST_HASH_STR);
    std::printf("\"source_revision\":\"%s\",", GGML_HIP_AUTOTUNE_SOURCE_REVISION_STR);
    std::printf("\"descriptor_hash\":\"%s\",", GGML_HIP_AUTOTUNE_DESCRIPTOR_HASH_STR);
    std::printf("\"variant_set\":\"%s\",", GGML_HIP_AUTOTUNE_VARIANT_SET_STR);
    std::printf("\"artifact_version\":%d,", GGML_HIP_AUTOTUNE_ARTIFACT_VERSION);
    std::printf("\"candidate_count\":%zu},", GGML_HIP_AUTOTUNE_CANDIDATE_COUNT);
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
    return (!registry.anomalies.empty()) ? 1
#ifdef GGML_HIP_DISPATCH_REPLAY
           : (cache_configured && !cache_loaded) ? 3
           : (cache_configured && winners.empty() ? 4 : 0)
#else
           : 0;
#endif
}
