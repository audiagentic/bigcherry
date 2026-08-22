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
thread_local std::chrono::steady_clock::time_point g_reduce_start;
thread_local bool g_reduce_timer_active = false;

// HI18 test-capture state. thread_local because the standalone probe is
// single-threaded and this must never become a second, accidental
// cross-thread state channel alongside g_last_comm_ctx above.
thread_local ggml_hip_reduce_test_snapshot_v1 g_reduce_test_snapshot;
thread_local uint64_t g_reduce_test_sequence = 0;
thread_local bool g_reduce_test_captured = false;

void copy_label(char * dst, size_t dst_size, const char * src) {
    if (src == nullptr) {
        dst[0] = '\0';
        return;
    }
    std::snprintf(dst, dst_size, "%s", src);
}

// Forward-declared: defined further below, alongside write_event()'s own
// use of it. Recomputing it here (rather than threading a shared value
// through both call sites) keeps this test-only seam decoupled from the
// production telemetry hot path -- it only runs when a probe is actually
// capturing, never in a normal inference process.
ggml_hip_reduce_signature_v1 make_signature(
        const int * devices, size_t device_count, ggml_tensor ** tensors);

void capture_reduce_test_snapshot(
        const int * devices, size_t device_count, ggml_tensor ** tensors,
        const char * requested_provider, const char * effective_provider,
        const char * handoff, size_t fallback_depth, bool provider_succeeded) {
    ggml_hip_reduce_test_snapshot_v1 & snap = g_reduce_test_snapshot;
    snap.version = 1;
    snap.sequence = ++g_reduce_test_sequence;
    snap.device_count = devices == nullptr
        ? 0
        : std::min<size_t>(device_count, GGML_HIP_REDUCE_TEST_CAPTURE_MAX_DEVICES);
    for (size_t i = 0; i < snap.device_count; ++i) {
        snap.devices[i] = devices[i];
        snap.tensors[i] = tensors != nullptr ? tensors[i] : nullptr;
    }
    copy_label(snap.requested_provider, sizeof(snap.requested_provider), requested_provider);
    copy_label(snap.effective_provider, sizeof(snap.effective_provider), effective_provider);
    copy_label(snap.handoff, sizeof(snap.handoff), handoff);
    snap.fallback_depth = fallback_depth;
    snap.provider_succeeded = provider_succeeded;

    if (tensors != nullptr && tensors[0] != nullptr) {
        const ggml_hip_reduce_signature_v1 sig = make_signature(devices, snap.device_count, tensors);
        snap.element_count = sig.element_count;
        for (size_t i = 0; i < 4; ++i) {
            snap.slice_shape[i] = sig.slice_shape[i];
        }
        copy_label(snap.element_type, sizeof(snap.element_type), sig.element_type);
        copy_label(snap.topology_key, sizeof(snap.topology_key), sig.topology_key);
        copy_label(snap.peer_access, sizeof(snap.peer_access), sig.peer_access);
    }
    g_reduce_test_captured = true;
}

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

bool reduce_sync_requested() {
    const char * value = std::getenv("GGML_HIP_REDUCE_TIMING");
    if (value == nullptr) {
        return false;
    }
    const std::string mode(value);
    return mode == "sync" || mode == "synchronized" || mode == "1" || mode == "true";
}

bool synchronize_reduce_devices(const int * devices, size_t device_count) {
    if (devices == nullptr || device_count == 0) {
        return false;
    }
    int previous_device = -1;
    if (hipGetDevice(&previous_device) != hipSuccess) {
        return false;
    }
    bool ok = true;
    for (size_t i = 0; i < device_count; ++i) {
        if (hipSetDevice(devices[i]) != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
            ok = false;
            break;
        }
    }
    if (previous_device >= 0 && hipSetDevice(previous_device) != hipSuccess) {
        ok = false;
    }
    return ok;
}

struct reduce_timing {
    int64_t elapsed_us = 0;
    const char * mode = "host_control";
};

reduce_timing reduce_elapsed_us(const int * devices, size_t device_count,
                                bool synchronize) {
    reduce_timing result;
    if (synchronize && reduce_sync_requested()) {
        result.mode = synchronize_reduce_devices(devices, device_count)
            ? "device_synchronized" : "sync_failed";
    }
    if (!g_reduce_timer_active) {
        return result;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - g_reduce_start).count();
    result.elapsed_us = std::max<int64_t>(0, elapsed);
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
                 size_t fallback_depth,
                 const reduce_timing & timing) {
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
        "\"elapsed_us\":%lld,"
        "\"timing_mode\":\"%s\","
        "\"requested_provider\":\"%s\",\"effective_provider\":\"%s\","
        "\"handoff\":\"%s\",\"fallback_depth\":%zu,"
        "\"element_count\":%lld,\"element_type\":\"%s\","
        "\"reduction_signature\":{"
        "\"version\":1,\"element_count\":%lld,\"element_type\":\"%s\","
        "\"slice_shape\":[%lld,%lld,%lld,%lld],\"topology_key\":\"%s\","
        "\"peer_access\":\"%s\"},"
        "\"device_count\":%zu,\"devices\":[",
        timestamp, static_cast<long long>(timing.elapsed_us), timing.mode,
        provider_label(requested), provider_label(effective),
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
    const reduce_timing timing = reduce_elapsed_us(
        devices, device_count, provider_succeeded);
    write_event(devices, device_count, tensors, requested_provider,
                provider_succeeded ? effective_provider : "provider_declined",
                provider_succeeded ? "none" : "provider_declined_handoff_meta",
                provider_succeeded ? 0 : 1, timing);
    if (provider_succeeded) {
        capture_reduce_test_snapshot(
            devices, device_count, tensors, requested_provider, effective_provider,
            "none", 0, true);
        g_reduce_timer_active = false;
    }
}

void ggml_hip_reduce_telemetry_set_requested_provider(
        void * comm_ctx, const char * requested_provider) {
    g_last_comm_ctx = comm_ctx;
    g_last_requested_provider = requested_provider != nullptr
        ? requested_provider : "auto";
    g_reduce_start = std::chrono::steady_clock::now();
    g_reduce_timer_active = true;
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
    const reduce_timing timing = reduce_elapsed_us(devices, device_count, true);
    write_event(devices, device_count, tensors, requested_provider, "meta", handoff,
                fallback_depth, timing);
    capture_reduce_test_snapshot(
        devices, device_count, tensors, requested_provider, "meta",
        handoff, fallback_depth, false);
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
    g_reduce_timer_active = false;
}

void ggml_hip_reduce_test_capture_reset() {
    g_reduce_test_snapshot = ggml_hip_reduce_test_snapshot_v1{};
    g_reduce_test_captured = false;
}

bool ggml_hip_reduce_test_capture_snapshot(ggml_hip_reduce_test_snapshot_v1 * out) {
    if (out == nullptr || !g_reduce_test_captured) {
        return false;
    }
    *out = g_reduce_test_snapshot;
    return true;
}

#endif
