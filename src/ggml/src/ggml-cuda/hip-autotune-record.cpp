// bigcherry: signature collection, record mode (HI10).

#include "hip-autotune-record.h"

#if defined(GGML_USE_HIP) && defined(GGML_HIP_AUTOTUNE_RECORD)

#include "hip-autotune-build-hash.h"
#include "hip-autotune-io.h"
#include "hip-autotune-signature.h"

#include <algorithm>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// One observed signature. Standards 15.1: a second sighting increments `calls`
// and merges its site rather than creating a second record, so the file has one
// row per distinct operation regardless of how often it ran.
struct Observation {
    ggml_hip_digest signature_digest;
    ggml_hip_digest hardware_digest;
    std::string     canonical_json;
    std::string     hardware_json;
    std::string     native_stable_name;
    // Telemetry only: these describe the established BLAS wrapper hook and
    // never participate in signature or dispatch identity.
    std::string     effective_api;
    std::string     effective_call_api;
    std::string     blas_operand_a_type;
    std::string     blas_operand_b_type;
    std::string     blas_output_type;
    std::string     blas_accumulation_type;
    std::string     blas_source_a_conversion;
    std::string     blas_source_b_conversion;
    std::string     blas_output_conversion;
    std::string     blas_requested_precision;
    std::string     blas_effective_provider;
    std::string     blas_effective_backend;
    uint64_t        blas_source_a_temp_bytes = 0;
    uint64_t        blas_source_b_temp_bytes = 0;
    uint64_t        blas_output_temp_bytes = 0;
    uint64_t        workspace_bytes;
    uint64_t        calls;
    uint64_t        est_bytes;
    // Device *ordinals* seen, which is diagnostic only -- never part of the
    // key. Two GPUs of the same class share a hardware digest (standards 10.2)
    // and this records that it actually happened rather than assuming it.
    std::vector<int> devices;
};

struct DigestHash {
    size_t operator()(const ggml_hip_digest & d) const {
        size_t v = 0;
        for (size_t i = 0; i < sizeof(size_t) && i < GGML_HIP_DIGEST_BYTES; ++i) {
            v |= (size_t) d.bytes[i] << (8 * i);
        }
        return v;
    }
};

struct DigestEqual {
    bool operator()(const ggml_hip_digest & a, const ggml_hip_digest & b) const {
        return ggml_hip_digest_equal(a, b);
    }
};

// Keyed on the *dispatch-relevant pair*: the same operation on two different
// GPU classes is two observations, because it may well have two different
// winners. On identical hardware it is one.
struct PairKey {
    ggml_hip_digest signature;
    ggml_hip_digest hardware;
};

struct PairHash {
    size_t operator()(const PairKey & k) const {
        return DigestHash()(k.signature) ^ (DigestHash()(k.hardware) << 1);
    }
};

struct PairEqual {
    bool operator()(const PairKey & a, const PairKey & b) const {
        return ggml_hip_digest_equal(a.signature, b.signature)
            && ggml_hip_digest_equal(a.hardware, b.hardware);
    }
};

std::unordered_map<PairKey, Observation, PairHash, PairEqual> g_observations;
std::mutex g_mutex;
thread_local PairKey g_active_key = {};
thread_local bool g_has_active_key = false;

int64_t estimate_bytes(const ggml_hip_dispatch_signature_v1 & sig) {
    // Rough traffic estimate, used only to rank hot signatures for tuning
    // priority (standards 7.4). Exactness would buy nothing -- the ranking is
    // by order of magnitude, not by percent.
    const int64_t k = sig.ne0[0];
    return sig.ne0[1] * k + sig.ne1[1] * k + sig.ned[0] * sig.ned[1];
}

} // namespace

void ggml_hip_record_observation(
        ggml_backend_cuda_context & ctx,
        const ggml_hip_dispatch_signature_v1 & sig,
        const ggml_hip_hardware_key_v1 & hw,
        const ggml_hip_digest & signature_digest,
        const ggml_hip_digest & hardware_digest,
        const ggml_hip_native_selection & native,
        const char * effective_api,
        size_t workspace_bytes) {
    PairKey key;
    key.signature = signature_digest;
    key.hardware  = hardware_digest;

    std::lock_guard<std::mutex> lock(g_mutex);

    auto found = g_observations.find(key);
    if (found != g_observations.end()) {
        ++found->second.calls;
        // The device list is a set, kept small and unsorted because it has at
        // most a handful of entries.
        if (std::find(found->second.devices.begin(), found->second.devices.end(),
                      ctx.device) == found->second.devices.end()) {
            found->second.devices.push_back(ctx.device);
        }
        g_active_key = key;
        g_has_active_key = true;
        return;
    }

    Observation observation;
    observation.signature_digest   = signature_digest;
    observation.hardware_digest    = hardware_digest;
    observation.canonical_json     = ggml_hip_signature_json(sig, true);
    observation.hardware_json      = ggml_hip_hardware_json(hw);
    observation.native_stable_name = native.candidate != nullptr
        ? native.candidate->stable_name : "";
    observation.effective_api     = effective_api != nullptr ? effective_api : "";
    observation.workspace_bytes   = (uint64_t) workspace_bytes;
    observation.calls              = 1;
    observation.est_bytes          = (uint64_t) estimate_bytes(sig);
    observation.devices.push_back(ctx.device);

    g_observations.emplace(key, observation);
    g_active_key = key;
    g_has_active_key = true;
}

void ggml_hip_record_touch(const ggml_hip_digest & signature_digest,
                           const ggml_hip_digest & hardware_digest,
                           int device) {
    PairKey key;
    key.signature = signature_digest;
    key.hardware  = hardware_digest;

    std::lock_guard<std::mutex> lock(g_mutex);
    auto found = g_observations.find(key);
    if (found == g_observations.end()) {
        // A warm-path hit for something never recorded cannot normally happen,
        // but silently inventing an observation with no canonical JSON would
        // corrupt the inventory, so do nothing rather than guess.
        return;
    }
    ++found->second.calls;
    if (std::find(found->second.devices.begin(), found->second.devices.end(),
                  device) == found->second.devices.end()) {
        found->second.devices.push_back(device);
    }
    g_active_key = key;
    g_has_active_key = true;
}

void ggml_hip_record_effective_call_api(const char * api) {
    if (api == nullptr || !g_has_active_key) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto found = g_observations.find(g_active_key);
    if (found != g_observations.end()) {
        found->second.effective_call_api = api;
    }
}

void ggml_hip_record_blas_metadata(const ggml_hip_blas_observation_v1 & metadata) {
    if (!g_has_active_key) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto found = g_observations.find(g_active_key);
    if (found == g_observations.end()) {
        return;
    }
    Observation & observation = found->second;
    observation.blas_operand_a_type = metadata.operand_a_type != nullptr
        ? metadata.operand_a_type : "unknown";
    observation.blas_operand_b_type = metadata.operand_b_type != nullptr
        ? metadata.operand_b_type : "unknown";
    observation.blas_output_type = metadata.output_type != nullptr
        ? metadata.output_type : "unknown";
    observation.blas_accumulation_type = metadata.accumulation_type != nullptr
        ? metadata.accumulation_type : "unknown";
    observation.blas_source_a_conversion = metadata.source_a_conversion != nullptr
        ? metadata.source_a_conversion : "unknown";
    observation.blas_source_b_conversion = metadata.source_b_conversion != nullptr
        ? metadata.source_b_conversion : "unknown";
    observation.blas_output_conversion = metadata.output_conversion != nullptr
        ? metadata.output_conversion : "unknown";
    observation.blas_requested_precision = metadata.requested_precision != nullptr
        ? metadata.requested_precision : "unknown";
    observation.blas_effective_provider = metadata.effective_provider != nullptr
        ? metadata.effective_provider : "unknown";
    observation.blas_effective_backend = metadata.effective_backend != nullptr
        ? metadata.effective_backend : "unknown";
    observation.blas_source_a_temp_bytes = metadata.source_a_temp_bytes;
    observation.blas_source_b_temp_bytes = metadata.source_b_temp_bytes;
    observation.blas_output_temp_bytes = metadata.output_temp_bytes;
}

void ggml_hip_record_flush() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_observations.empty()) {
        return;
    }

    const char * path = getenv("GGML_HIP_DISPATCH_DB");
    if (path == nullptr || path[0] == '\0') {
        GGML_LOG_WARN("bigcherry: record mode observed %zu signature(s) but "
                      "GGML_HIP_DISPATCH_DB is unset, so nothing was written\n",
                      g_observations.size());
        return;
    }

    // Rewritten, not appended: a checkpoint mid-run must leave a complete
    // document, and the in-memory map is already the merged truth. HI48:
    // atomic same-directory replacement, so a crash mid-rewrite leaves the
    // previous good file in place rather than a truncated one overwriting it.
    ggml_hip_atomic_file file_atomic;
    if (!ggml_hip_atomic_begin(path, file_atomic)) {
        GGML_LOG_WARN("bigcherry: cannot write record file '%s'\n", path);
        return;
    }
    FILE * file = file_atomic.file;

    // One header line, then one line per observation. JSON Lines rather than a
    // single document so a truncated file is still readable up to its last
    // complete line -- which is the whole point of surviving a killed run.
    fprintf(file,
            "{\"kind\":\"header\",\"artifact_version\":%d,"
            "\"source_revision\":\"%s\",\"manifest_hash\":\"%s\","
            "\"variant_set\":\"%s\",\"signature_schema\":%d,"
            "\"hardware_schema\":%d,\"signatures\":%zu}\n",
            GGML_HIP_AUTOTUNE_ARTIFACT_VERSION,
            GGML_HIP_AUTOTUNE_SOURCE_REVISION_STR,
            GGML_HIP_AUTOTUNE_MANIFEST_HASH_STR,
            GGML_HIP_AUTOTUNE_VARIANT_SET_STR,
            GGML_HIP_SIGNATURE_SCHEMA_VERSION,
            GGML_HIP_HARDWARE_SCHEMA_VERSION,
            g_observations.size());

    for (const auto & entry : g_observations) {
        const Observation & o = entry.second;
        fprintf(file,
                "{\"kind\":\"observation\",\"signature\":\"%s\","
                "\"hardware\":\"%s\",\"native\":\"%s\",\"calls\":%llu,"
                "\"est_bytes\":%llu,\"effective_api\":\"%s\","
                "\"effective_call_api\":\"%s\","
                "\"workspace_bytes\":%llu,\"blas_metadata\":{"
                "\"operand_a_type\":\"%s\",\"operand_b_type\":\"%s\","
                "\"output_type\":\"%s\",\"accumulation_type\":\"%s\","
                "\"source_a_conversion\":\"%s\",\"source_b_conversion\":\"%s\","
                "\"output_conversion\":\"%s\",\"requested_precision\":\"%s\","
                "\"effective_provider\":\"%s\",\"effective_backend\":\"%s\","
                "\"source_a_temp_bytes\":%llu,\"source_b_temp_bytes\":%llu,"
                "\"output_temp_bytes\":%llu},\"devices\":[",
                ggml_hip_digest_hex(o.signature_digest).c_str(),
                ggml_hip_digest_hex(o.hardware_digest).c_str(),
                o.native_stable_name.c_str(),
                (unsigned long long) o.calls,
                (unsigned long long) o.est_bytes,
                o.effective_api.c_str(),
                o.effective_call_api.c_str(),
                (unsigned long long) o.workspace_bytes,
                o.blas_operand_a_type.c_str(), o.blas_operand_b_type.c_str(),
                o.blas_output_type.c_str(), o.blas_accumulation_type.c_str(),
                o.blas_source_a_conversion.c_str(), o.blas_source_b_conversion.c_str(),
                o.blas_output_conversion.c_str(), o.blas_requested_precision.c_str(),
                o.blas_effective_provider.c_str(), o.blas_effective_backend.c_str(),
                (unsigned long long) o.blas_source_a_temp_bytes,
                (unsigned long long) o.blas_source_b_temp_bytes,
                (unsigned long long) o.blas_output_temp_bytes);
        for (size_t i = 0; i < o.devices.size(); ++i) {
            fprintf(file, "%s%d", i ? "," : "", o.devices[i]);
        }
        fprintf(file, "],\"canonical\":%s,\"hardware_key\":%s}\n",
                o.canonical_json.c_str(), o.hardware_json.c_str());
    }
    if (!ggml_hip_atomic_commit(file_atomic)) {
        GGML_LOG_WARN("bigcherry: atomic record replacement failed for '%s'\n", path);
        return;
    }

    GGML_LOG_INFO("bigcherry: recorded %zu signature(s) to '%s'\n",
                  g_observations.size(), path);
}

void ggml_hip_record_write_report(const char * path) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_observations.empty() || path == nullptr) {
        return;
    }

    // Ranked by call count, because that is the order the tuner will work in
    // (standards 7.4) and the first thing worth eyeballing is whether the top
    // of that list is what you expected the workload to be doing.
    std::vector<const Observation *> ranked;
    ranked.reserve(g_observations.size());
    for (const auto & entry : g_observations) {
        ranked.push_back(&entry.second);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const Observation * a, const Observation * b) {
                  if (a->calls != b->calls) return a->calls > b->calls;
                  return a->est_bytes > b->est_bytes;
              });

    ggml_hip_atomic_file report_atomic;
    if (!ggml_hip_atomic_begin(path, report_atomic)) {
        return;
    }
    FILE * file = report_atomic.file;

    uint64_t total_calls = 0;
    for (const Observation * o : ranked) {
        total_calls += o->calls;
    }

    fprintf(file, "bigcherry record coverage\n");
    fprintf(file, "  source revision : %s\n", GGML_HIP_AUTOTUNE_SOURCE_REVISION_STR);
    fprintf(file, "  manifest hash   : %s\n", GGML_HIP_AUTOTUNE_MANIFEST_HASH_STR);
    fprintf(file, "  signatures      : %zu\n", ranked.size());
    fprintf(file, "  total calls     : %llu\n\n",
            (unsigned long long) total_calls);
    fprintf(file, "  %-34s %10s %7s  %s\n",
            "signature", "calls", "share", "native candidate");

    uint64_t cumulative = 0;
    for (size_t i = 0; i < ranked.size(); ++i) {
        const Observation * o = ranked[i];
        cumulative += o->calls;
        fprintf(file, "  %-34s %10llu %6.2f%%  %s\n",
                ggml_hip_digest_hex(o->signature_digest).c_str(),
                (unsigned long long) o->calls,
                100.0 * (double) o->calls / (double) total_calls,
                o->native_stable_name.c_str());
        // Where the workload's mass actually sits. If ten signatures cover 90%
        // of calls, tuning the other several hundred is not where the time
        // should go.
        if (i + 1 == 10 || i + 1 == 50) {
            fprintf(file, "  -- top %zu cover %.1f%% of calls --\n",
                    i + 1, 100.0 * (double) cumulative / (double) total_calls);
        }
    }
    ggml_hip_atomic_commit(report_atomic);
}

size_t ggml_hip_record_signature_count() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_observations.size();
}

#endif // GGML_USE_HIP && GGML_HIP_AUTOTUNE_RECORD
