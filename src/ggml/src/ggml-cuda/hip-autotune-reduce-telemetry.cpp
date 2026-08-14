// bigcherry: telemetry-only SPLIT_REDUCE observation (HI58).

#include "hip-autotune-reduce-telemetry.h"

#if defined(GGML_USE_HIP) && defined(GGML_HIP_DISPATCH)

#include "../../include/ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <string>

#include <hip/hip_runtime_api.h>

extern bool ggml_hip_reduce_telemetry_context_snapshot(
        void * comm_ctx,
        const int ** devices,
        size_t * device_count,
        const char ** requested_provider);

namespace {

std::mutex g_mutex;
thread_local void * g_last_comm_ctx = nullptr;
thread_local std::string g_last_requested_provider;

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
    if (std::string(value) == "auto" || std::string(value) == "internal" ||
        std::string(value) == "rccl" ||
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

struct topology_observation {
    std::string key;
    const char * peer_access = "unknown";
};

topology_observation observe_topology(const int * devices, size_t device_count) {
    topology_observation result;
    std::ostringstream matrix;
    matrix << "n" << device_count << ":peer";
    bool complete = true;
    bool known = true;
    for (size_t i = 0; i < device_count; ++i) {
        for (size_t j = 0; j < device_count; ++j) {
            int can_access = i == j ? 1 : 0;
            const hipError_t status = i == j
                ? hipSuccess
                : hipDeviceCanAccessPeer(&can_access, devices[i], devices[j]);
            if (status != hipSuccess) {
                known = false;
                matrix << "?";
                continue;
            }
            matrix << (can_access != 0 ? "1" : "0");
            complete = complete && can_access != 0;
        }
    }
    if (!known) {
        result.peer_access = "unknown";
    } else if (complete) {
        result.peer_access = "complete";
    } else {
        result.peer_access = "partial";
    }
    result.key = matrix.str();
    return result;
}

ggml_hip_reduce_signature_v1 make_signature(
        const int * devices, size_t device_count, ggml_tensor ** tensors) {
    static thread_local std::string topology_key;
    static thread_local topology_observation topology;
    topology = observe_topology(devices, device_count);
    topology_key = topology.key;

    ggml_hip_reduce_signature_v1 signature{};
    signature.element_count = tensors != nullptr && tensors[0] != nullptr
        ? ggml_nelements(tensors[0]) : 0;
    if (tensors != nullptr && tensors[0] != nullptr) {
        for (size_t i = 0; i < 4; ++i) {
            signature.slice_shape[i] = tensors[0]->ne[i];
        }
        signature.element_type = json_type(tensors[0]);
    } else {
        signature.element_type = "unknown";
    }
    signature.device_count = devices == nullptr ? 0 : device_count;
    signature.devices = devices;
    signature.topology_key = topology_key.c_str();
    signature.peer_access = topology.peer_access;
    return signature;
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

    const ggml_hip_reduce_signature_v1 signature =
        make_signature(devices, normalized_device_count, tensors);

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
        "\"reduction_signature\":{"
        "\"version\":1,\"element_count\":%lld,\"element_type\":\"%s\","
        "\"slice_shape\":[%lld,%lld,%lld,%lld],\"topology_key\":\"%s\","
        "\"peer_access\":\"%s\"},"
        "\"device_count\":%zu,\"devices\":[",
        timestamp, provider_label(requested), provider_label(effective),
        handoff_label(handoff), normalized_fallback_depth,
        static_cast<long long>(signature.element_count), signature.element_type,
        static_cast<long long>(signature.element_count), signature.element_type,
        static_cast<long long>(signature.slice_shape[0]),
        static_cast<long long>(signature.slice_shape[1]),
        static_cast<long long>(signature.slice_shape[2]),
        static_cast<long long>(signature.slice_shape[3]),
        signature.topology_key, signature.peer_access, normalized_device_count);
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
        const char * effective_provider,
        bool provider_succeeded) {
    write_event(devices, device_count, tensors, requested_provider,
                provider_succeeded ? effective_provider : "provider_declined",
                provider_succeeded ? "none" : "provider_declined_handoff_meta",
                provider_succeeded ? 0 : 1);
}

void ggml_hip_reduce_telemetry_set_requested_provider(
        void * comm_ctx, const char * requested_provider) {
    g_last_comm_ctx = comm_ctx;
    g_last_requested_provider = requested_provider != nullptr
        ? requested_provider : "auto";
}

const char * ggml_hip_reduce_telemetry_requested_provider(
        void * comm_ctx, const char * fallback_provider) {
    if (g_last_comm_ctx == comm_ctx && !g_last_requested_provider.empty()) {
        return g_last_requested_provider.c_str();
    }
    return fallback_provider;
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

void ggml_hip_reduce_telemetry_fallback_context(
        void * comm_ctx,
        ggml_tensor ** tensors,
        const char * handoff,
        size_t fallback_depth) {
    const int * devices = nullptr;
    size_t device_count = 0;
    const char * requested_provider = nullptr;
    if (!ggml_hip_reduce_telemetry_context_snapshot(
            comm_ctx, &devices, &device_count, &requested_provider)) {
        return;
    }
    const bool explicitly_requested_meta =
        g_last_comm_ctx == comm_ctx && g_last_requested_provider == "meta";
    ggml_hip_reduce_telemetry_fallback(
            devices, device_count, tensors, requested_provider,
            explicitly_requested_meta ? "none" : handoff,
            explicitly_requested_meta ? 0 : fallback_depth);
}

#endif
