// The engine's wait for a helper to consume a published layer map (RUM06). Split out of the hook so it is tested on
// the host. It is bounded by the helper protocol's own readiness deadline (r3::epoch::Deadlines::ready_ns, the phase
// bound the coordinator gives helpers to become ready), not by a literal of its own: a helper that is alive but never
// consumes ends the wait with the sticky helper fault, which unblocks every other wait and lets teardown proceed.
#pragma once
#include "moe_remote_shm.h"
#include "worker_epoch.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <string>
#include <vector>

enum class MapWait { Consumed, Faulted, Timeout, Error };

/* The absolute CLOCK_MONOTONIC deadline `bound` from now, in the form wait_for takes. */
inline timespec deadline_after(std::chrono::nanoseconds bound) {
    timespec deadline{};
    clock_gettime(CLOCK_MONOTONIC, &deadline);
    const int64_t ns = static_cast<int64_t>(bound.count()) + deadline.tv_nsec;
    deadline.tv_sec += static_cast<time_t>(ns / 1000000000);
    deadline.tv_nsec = static_cast<long>(ns % 1000000000);
    return deadline;
}

inline MapWait await_layer_consumed(r3::moe_shm::Header* h, int layer, std::chrono::nanoseconds bound) {
    using namespace r3::moe_shm;
    timespec deadline = deadline_after(bound);
    const uint32_t* consumed = &h->layer_map_consumed[layer];
    switch (wait_for(h, &deadline, [&] { return load(consumed) == 1; })) {
        case WaitResult::Ready: return MapWait::Consumed;
        case WaitResult::Fault: return MapWait::Faulted;
        case WaitResult::Timeout: raise_fault(h); return MapWait::Timeout;   /* sticky: unblocks every other wait */
        default: raise_fault(h); return MapWait::Error;
    }
}

/* The bound the engine gives a helper to register, be watched, or consume a layer map: the protocol's readiness
 * deadline (the phase bound helpers have to become ready). */
inline std::chrono::nanoseconds layer_map_bound() {
    return std::chrono::nanoseconds(r3::epoch::Deadlines{}.ready_ns);
}

/* Startup-only record of how long the engine waited, per layer, for helpers to consume published maps. The waits happen
 * while the engine declares its layers, so their sum is the part of engine startup the helpers' bootstrap holds up.
 * Written under the hook's lock by publish_layer_map; read once, when the first rank attaches after the bootstrap seal. */
struct MapWaitStats {
    struct Layer { uint64_t calls = 0, total_ns = 0, max_ns = 0; };
    std::vector<Layer> layer;
    void record(uint32_t l, uint64_t ns) {
        if (layer.size() <= l) layer.resize(size_t(l) + 1);
        Layer& x = layer[l];
        ++x.calls; x.total_ns += ns; x.max_ns = std::max(x.max_ns, ns);
    }
    Layer total() const {
        Layer t;
        for (const Layer& x : layer) { t.calls += x.calls; t.total_ns += x.total_ns; t.max_ns = std::max(t.max_ns, x.max_ns); }
        return t;
    }
    /* One line: `libr3: layer_map_wait {"calls":N,"total_ns":T,"max_ns":M,"per_layer":[[layer,calls,total_ns,max_ns],...]}` */
    std::string line() const {
        const Layer t = total();
        std::string o = "libr3: layer_map_wait {\"calls\":" + std::to_string(t.calls) + ",\"total_ns\":" +
                        std::to_string(t.total_ns) + ",\"max_ns\":" + std::to_string(t.max_ns) + ",\"per_layer\":[";
        bool first = true;
        for (size_t l = 0; l < layer.size(); ++l) {
            if (!layer[l].calls) continue;
            if (!first) o += ',';
            first = false;
            o += "[" + std::to_string(l) + "," + std::to_string(layer[l].calls) + "," +
                 std::to_string(layer[l].total_ns) + "," + std::to_string(layer[l].max_ns) + "]";
        }
        return o + "]}";
    }
};

/* await_layer_consumed, with the monotonic wall of the wait added to `stats` whatever its outcome. */
inline MapWait await_layer_consumed_timed(r3::moe_shm::Header* h, int layer, std::chrono::nanoseconds bound,
                                          MapWaitStats* stats) {
    const auto t0 = std::chrono::steady_clock::now();
    const MapWait r = await_layer_consumed(h, layer, bound);
    if (stats && layer >= 0)
        stats->record(uint32_t(layer), uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - t0).count()));
    return r;
}
