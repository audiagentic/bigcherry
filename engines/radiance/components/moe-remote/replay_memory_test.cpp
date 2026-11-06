#include "worker_replay_memory.h"
#include <cstdio>
#include <cstdlib>
#include <limits>

static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "replay memory: FAIL %s\n", what); std::exit(1); }
}
int main() {
    using namespace r3::moe_shm;
    using namespace r3::replay;
    const uint64_t mib = 1024ull * 1024ull;
    uint64_t budget = 0;
    // The manifest's 22,528 MiB is a serving ceiling, not a replay arena.
    check(expert_budget(2400 * 1024ull, 476, 22528 * mib, &budget) &&
          budget == 476ull * 2400 * 1024 && budget < 22528 * mib,
          "476 selected experts require 476 strides, not the full manifest budget");
    check(!expert_budget(2400 * 1024ull, 10000, 22528 * mib, &budget),
          "selected experts must still respect the manifest ceiling");
    check(!expert_budget(std::numeric_limits<uint64_t>::max(), 2,
                         std::numeric_limits<uint64_t>::max(), &budget),
          "overflow cannot wrap the arena");
    check(!expert_budget(10, 0, 100, &budget), "zero selected experts refused");

    std::array<uint32_t, kSmallClasses> classes{8, 0, 0, 0};
    LanePlan plan{};
    check(lane_plan(classes, 1, 2048, 143, 10, 1, &plan), "capture plan");
    check(plan.large_rows == 2048 && plan.dense_routes == 2048 * 10,
          "capture 2048 rows use dense route capacity, not 4096 rows x top-k");
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        check(plan.open[lane] == (lane == lane_of(0, kSmallClasses)),
              "only the live large lane of one engine rank is allocated");
    check(lane_plan(classes, 1, 8, 4, 10, 2, &plan), "small class plan");
    check(plan.large_rows == 8 && plan.dense_routes == 8 * 10, "small request dense capacity");
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        check(plan.open[lane] == (lane == lane_of(0, 0) || lane == lane_of(1, 0)),
              "two-rank small request prepares only two active lanes");
    check(lane_plan(classes, 1, 1024, 90, 10, 1, &plan) &&
          plan.large_rows == 1024 && plan.dense_routes == 1024 * 10,
          "a later smaller capture gets a smaller plan, not the prior maximum");
    classes = {4, 8, 16, 24};
    check(lane_plan(classes, 4, 12, 9, 10, 1, &plan) &&
          plan.large_rows == 24 && plan.dense_routes == 16 * 10 && plan.open[lane_of(0, 2)],
          "middle class retains its fixed 16 rows while satisfying 24-row validation");
    check(!lane_plan(classes, 4, 12, 121, 10, 1, &plan), "route count bounded by rows x top-k");
    check(!lane_plan(classes, 4, 4097, 1, 10, 1, &plan), "row protocol bound");
    std::puts("replay memory sizing OK");
}
