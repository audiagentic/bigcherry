// The engine's open barrier for the helpers (RUM06): every helper has published its identity and the watcher holds a
// validated pidfd for it. Split out of open_mailbox so the real wait is tested on the host with fake helpers.
// The bound is the helper STARTUP protocol's deadline (kStartupWaitSeconds, the one the helpers themselves use for
// segment attach and the bootstrap seal), not the placement readiness deadline (r3::epoch::Deadlines), which bounds
// placement phases. A miss, a helper fault or a wait error raises the sticky helper fault.
#pragma once
#include "layer_map_wait.h"
#include "moe_helper_watch.h"
#include "moe_remote_shm.h"
#include <chrono>

inline std::chrono::nanoseconds registration_bound() {
    return std::chrono::seconds(r3::moe_shm::kStartupWaitSeconds);
}

inline r3::moe_shm::WaitResult await_helpers_registered(r3::moe_shm::Header* h, HelperWatch& watch,
                                                        std::chrono::nanoseconds bound) {
    using namespace r3::moe_shm;
    timespec deadline = deadline_after(bound);
    const WaitResult r = wait_for(h, &deadline, [&] {
        watch.notice();
        const uint32_t n = load(&h->helper_count);
        for (uint32_t i = 0; i < n; ++i)
            if (!watch.watching(i)) return false;
        return n > 0;
    });
    if (r != WaitResult::Ready) raise_fault(h);   /* sticky: the generation opens, its first remote step fails */
    return r;
}
