// The engine's wait for a helper to consume a published layer map (RUM06). Split out of the hook so it is tested on
// the host. It is bounded by the helper protocol's own readiness deadline (r3::epoch::Deadlines::ready_ns, the phase
// bound the coordinator gives helpers to become ready), not by a literal of its own: a helper that is alive but never
// consumes ends the wait with the sticky helper fault, which unblocks every other wait and lets teardown proceed.
#pragma once
#include "moe_remote_shm.h"
#include "worker_epoch.h"
#include <chrono>
#include <ctime>

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
