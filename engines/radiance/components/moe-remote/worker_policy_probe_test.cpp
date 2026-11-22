// The calibration probe's pure rules, card-free.
#include "worker_policy_probe.h"
#include <cstdio>
#include <cstdlib>
using namespace r3::policy;
static void check(bool b, const char* m) {
    if (!b) { std::fprintf(stderr, "policy probe FAIL: %s\n", m); std::exit(1); }
}
static std::vector<MissingPlannerCondition> missing(std::initializer_list<const char*> names) {
    std::vector<MissingPlannerCondition> out;
    for (const char* n : names) out.push_back({n, "x", 0});
    return out;
}
int main() {
    check(probe_ready({}) && probe_ready(missing({"txn_p50_ns", "drift_half_life_observations"})),
          "only transaction and drift-crossing evidence may be missing when probing starts");
    for (const char* n : {"heat_watermarked", "drift_window", "heat_gap_max_ns", "hot_set_k", "marginal_error_ppm",
                          "marginal_observations", "observation_rate_per_s"})
        check(!probe_ready(missing({n})), "passive evidence gates the probe");
    check(!probe_needed({}) && !probe_needed(missing({"drift_window", "drift_half_life_observations"})),
          "no transaction evidence missing: no probe");
    for (const char* n : {"txn_p50_ns", "tail_share_gain_ppm", "copy_bytes_per_s", "tail_lane_service_ns_per_s",
                          "tail_lane_row_ns", "lane_lost_ns"})
        check(probe_needed(missing({n})), "each transaction quantity needs a probe");

    r3::heat::Plan plan;
    plan.valid = true; plan.no_change = false; plan.baseline = 0.6; plan.predicted = 0.4;
    plan.delta.push_back({7, 0, 2, 0});   // head 0 -> tail 2 (three helpers)
    check(probe_acceptable(plan, 3), "a positive-gain move onto the tail is a probe");
    plan.delta[0] = {7, 2, 1, 0};
    check(probe_acceptable(plan, 3), "a move off the tail is a probe");
    plan.delta[0] = {7, 0, 1, 0};
    check(!probe_acceptable(plan, 3), "a move between heads never involves the tail");
    plan.delta[0] = {7, 0, 2, 0};
    plan.predicted = plan.baseline;
    check(!probe_acceptable(plan, 3), "no predicted gain: not a probe");
    plan.predicted = 0.4; plan.no_change = true;
    check(!probe_acceptable(plan, 3), "no change: not a probe");
    plan.no_change = false; plan.valid = false;
    check(!probe_acceptable(plan, 3), "invalid plan: not a probe");

    // The ramp starts at one stride, doubles, and stops at what the smallest budget holds.
    ProbeRamp ramp(10, {100, 45, 80});
    check(ramp.usable() && ramp.moves() == 1 && ramp.cap() == 4, "one stride, bounded by the smallest budget");
    PolicyControls base;
    base.max_moves = 99; base.max_moved_bytes = 9999; base.margin = 0.5;
    auto c = ramp.controls(base);
    check(c.max_moves == 1 && c.max_moved_bytes == 10 && c.margin == 0.5, "only the size comes from the ramp");
    ramp.grow(); check(ramp.moves() == 2 && ramp.controls(base).max_moved_bytes == 20, "factor two");
    ramp.grow(); check(ramp.moves() == 4, "factor two again, now at the cap");
    ramp.grow(); check(ramp.moves() == 4, "never past the budget");
    ProbeRamp odd(10, {70});
    odd.grow(); odd.grow(); odd.grow();
    check(odd.moves() == 7 && odd.cap() == 7, "a cap that is not a power of two is reached exactly");
    check(!ProbeRamp(10, {5}).usable() && !ProbeRamp(0, {50}).usable() && !ProbeRamp(10, {}).usable() &&
          !ProbeRamp().usable(), "a budget that cannot hold one stride gives no probe");
    std::puts("worker_policy_probe_test: ok");
    return 0;
}
