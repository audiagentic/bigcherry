#pragma once
#include <cstdint>

// ENG05: engine clock stamps are a single device's wall_clock64 domain.
// Helper clocks are a *different* device and must not be subtracted here.
namespace r3::moe_trace {
enum Index : unsigned {
    PublishBegin = 0, PublishReady = 1, WaitExit = 2,
    FinishBegin = 3, FinishEnd = 4, CombineWaitTicks = 5
};
struct Metrics {
    double publish_us = 0, wait_bookkeeping_us = 0;
    double finish_us = 0, combine_engine_wait_us = 0;
    bool valid = false;
};
inline Metrics compute(const uint64_t* stamps, uint64_t wait_seen,
                       double us_per_tick) {
    Metrics m{};
    if (!stamps || !(us_per_tick > 0) ||
        !stamps[PublishBegin] || !stamps[PublishReady] ||
        !stamps[WaitExit] || !stamps[FinishBegin] || !stamps[FinishEnd] ||
        stamps[PublishReady] < stamps[PublishBegin] ||
        stamps[WaitExit] < wait_seen ||
        stamps[FinishEnd] < stamps[FinishBegin] ||
        stamps[FinishBegin] < stamps[WaitExit]) return m;
    m.publish_us = (stamps[PublishReady] - stamps[PublishBegin]) * us_per_tick;
    m.wait_bookkeeping_us = (stamps[WaitExit] - wait_seen) * us_per_tick;
    m.finish_us = (stamps[FinishEnd] - stamps[FinishBegin]) * us_per_tick;
    m.combine_engine_wait_us = stamps[CombineWaitTicks] * us_per_tick;
    m.valid = true;
    return m;
}
} // namespace r3::moe_trace
