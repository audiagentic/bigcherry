// bigcherry: telemetry-only SPLIT_REDUCE observation (HI58).

#include "hip-autotune-reduce-telemetry.h"

#if defined(GGML_USE_HIP) && defined(GGML_HIP_DISPATCH)

#include "../../include/ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

namespace {

std::mutex g_mutex;

const char * telemetry_path() {
    const char * path = std::getenv("GGML_HIP_REDUCE_TELEMETRY");
    return path != nullptr && path[0] != '\0' ? path : nullptr;
}

const char * json_type(const ggml_tensor * tensor) {
    return tensor == nullptr ? "unknown" : ggml_type_name(tensor->type);
}

const char * provider_label(const char * value) {
    if (value == nullptr) {
        return "unknown";
    }
    if (std::string(value) == "internal" || std::string(value) == "rccl" ||
        std::string(value) == "meta" || std::string(value) == "unknown" ||
        std::string(value) == "provider_declined") {
        return value;
    }
    return "unknown";
}

const char * handoff_label(const char * value) {
    if (value == nullptr || std::string(value) == "none") {
        return "none";
    }
    if (std::string(value) == "provider_declined_handoff_meta") {
        return value;
    }
    return "unknown";
}

void write_event(const int * devices,
                 size_t device_count,
                 ggml_tensor ** tensors,
                 const char * requested,
                 const char * effective,
                 const char * handoff,
                 size_t fallback_depth) {
    const char * path = telemetry_path();
    if (path == nullptr || tensors == nullptr || tensors[0] == nullptr) {
        return;
    }

    // A null device array carries no topology evidence. Never manufacture a
    // list of -1 ordinals, and never claim a fallback depth without a handoff.
    const size_t normalized_device_count = devices == nullptr ? 0 : device_count;
    const size_t normalized_fallback_depth =
        handoff == nullptr || std::string(handoff) == "none" ? 0 : fallback_depth;

    // The context fields are intentionally observed, not changed. Keep this
    // JSONL channel independent from the tuning/replay artifacts.
    std::lock_guard<std::mutex> lock(g_mutex);
    FILE * file = std::fopen(path, "ab");
    if (file == nullptr) {
        return;
    }
    const long long timestamp = static_cast<long long>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    std::fprintf(file,
        "{\"kind\":\"split_reduce_observation\",\"timestamp_us\":%lld,"
        "\"requested_provider\":\"%s\",\"effective_provider\":\"%s\","
        "\"handoff\":\"%s\",\"fallback_depth\":%zu,"
        "\"element_count\":%lld,\"element_type\":\"%s\","
        "\"device_count\":%zu,\"devices\":[",
        timestamp, provider_label(requested), provider_label(effective),
        handoff_label(handoff), normalized_fallback_depth,
        static_cast<long long>(ggml_nelements(tensors[0])), json_type(tensors[0]),
        normalized_device_count);
    for (size_t i = 0; i < normalized_device_count; ++i) {
        std::fprintf(file, "%s%d", i == 0 ? "" : ",", devices != nullptr ? devices[i] : -1);
    }
    std::fprintf(file, "]}\n");
    std::fflush(file);
    std::fclose(file);
}

} // namespace

void ggml_hip_reduce_telemetry_provider(
        const int * devices,
        size_t device_count,
        ggml_tensor ** tensors,
        const char * requested_provider,
        bool provider_succeeded) {
    write_event(devices, device_count, tensors, requested_provider,
                provider_succeeded ? requested_provider : "provider_declined",
                provider_succeeded ? "none" : "provider_declined_handoff_meta",
                provider_succeeded ? 0 : 1);
}

void ggml_hip_reduce_telemetry_fallback(
        const int * devices,
        size_t device_count,
        ggml_tensor ** tensors,
        const char * requested_provider,
        const char * handoff,
        size_t fallback_depth) {
    // The meta backend proves the fallback execution.  Keep its effective
    // provider in the stable telemetry vocabulary rather than promoting an
    // implementation detail (butterfly) into provider identity.
    write_event(devices, device_count, tensors, requested_provider, "meta", handoff,
                fallback_depth);
}

#endif
