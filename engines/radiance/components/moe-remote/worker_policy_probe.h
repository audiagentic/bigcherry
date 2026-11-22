#pragma once
// RHP16 calibration probe: the pure rules of what a calibrating driver may propose before any policy setting is derived.
// No state of the driver lives here. The probe is the static placer's own plan on the live heat, restricted to moves that
// involve the tail helper and predict a positive tail-share gain, sized by a ramp of the physical expert stride.
#include "worker_policy_calibration.h"
#include "worker_policy_input.h"
#include "worker_heat.h"
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace r3::policy {

namespace probe_detail {
inline bool named(const std::vector<MissingPlannerCondition>& missing, const char* name) {
    return std::any_of(missing.begin(), missing.end(),
                       [&](const MissingPlannerCondition& c) { return c.name == name; });
}
} // namespace probe_detail

// Passive observation the probe must not run ahead of: accepted heat, the retained drift window, the heat gaps and the
// hot-set size. The drift half-life crossing is NOT required (it needs the workload to move, not a transaction).
inline bool probe_ready(const std::vector<MissingPlannerCondition>& missing) {
    for (const char* name : {"heat_watermarked", "drift_window", "heat_gap_max_ns", "hot_set_k",
                             "marginal_error_ppm", "marginal_observations", "observation_rate_per_s"})
        if (probe_detail::named(missing, name)) return false;
    return true;
}
// The quantities only a committed (tail-involved) transaction can supply.
inline bool probe_needed(const std::vector<MissingPlannerCondition>& missing) {
    for (const char* name : {"txn_p50_ns", "tail_share_gain_ppm", "copy_bytes_per_s", "tail_lane_service_ns_per_s",
                             "tail_lane_row_ns", "lane_lost_ns"})
        if (probe_detail::named(missing, name)) return true;
    return false;
}
// A probe must move something on or off the tail helper (the last) and predict that the tail's heat share falls.
inline bool probe_acceptable(const r3::heat::Plan& plan, uint32_t helpers) {
    if (!plan.valid || plan.no_change || plan.delta.empty() || !helpers) return false;
    if (!(plan.predicted < plan.baseline)) return false;
    const int32_t tail = int32_t(helpers) - 1;
    return std::any_of(plan.delta.begin(), plan.delta.end(),
                       [&](const auto& move) { return move.from == tail || move.to == tail; });
}

// The size ramp: one expert stride, doubled each time a probe round leaves the evidence still missing, never past what the
// smallest helper budget can hold (the byte budgets the launch already derived).
class ProbeRamp {
public:
    ProbeRamp() = default;
    ProbeRamp(uint64_t stride, const std::vector<uint64_t>& budgets) {
        if (!stride || budgets.empty()) return;
        const uint64_t slowest = *std::min_element(budgets.begin(), budgets.end());
        cap_ = uint32_t(std::min<uint64_t>(slowest / stride, UINT32_MAX));
        stride_ = stride;
        moves_ = cap_ ? 1 : 0;
    }
    bool usable() const { return moves_ != 0; }
    uint32_t moves() const { return moves_; }
    uint32_t cap() const { return cap_; }
    // Another round is needed and this size did not yield a plan or enough evidence.
    void grow() { if (moves_ && moves_ < cap_) moves_ = moves_ > cap_ / 2 ? cap_ : moves_ * 2; }
    // The planner controls of one probe round: only the size comes from the ramp.
    PolicyControls controls(PolicyControls base) const {
        base.max_moves = moves_;
        base.max_moved_bytes = uint64_t(moves_) * stride_;
        return base;
    }
private:
    uint32_t cap_ = 0, moves_ = 0;
    uint64_t stride_ = 0;
};

} // namespace r3::policy
