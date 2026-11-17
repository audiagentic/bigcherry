#pragma once
// RSC02 lab calibration: bounded ABBA workload and measured-drift envelope.
// No launcher, Python watchdog or shell operations in this module.
#include "generated/v2/serving_types.h"
#include "v2/error.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace bc::serving::session {

struct CalibrationPrompts {
    std::vector<std::string> a, b;
};
struct DriftProgress {
    uint64_t capacity = 0;
    uint64_t intervals = 0;
    uint64_t horizon_observations = 0;
    uint64_t observation_rate_per_s = 0;
    uint64_t accepted_gap_ns = 0;
    uint64_t accepted_heat_deltas = 0;
};
struct CalibrationEnvelope {
    std::chrono::seconds limit{};
    std::chrono::seconds prefeed{};
    std::chrono::seconds report_period{};
    bool accepted_feed = false;
};
v2::Result<CalibrationPrompts> read_calibration_prompts(const std::filesystem::path& a,
                                                        const std::filesystem::path& b);
v2::Result<std::optional<DriftProgress>> read_calibration_drift(const std::filesystem::path& helper_log);
v2::Result<CalibrationEnvelope> calibration_envelope(const v2::LivePolicy& policy, size_t a_count,
                                                     size_t b_count, std::chrono::seconds request_timeout,
                                                     const std::optional<DriftProgress>& drift);
// An evolving drift ring may shorten its own latest estimate, but cannot
// retract a deadline the owner already admitted. Keep the original prefeed
// horizon anchored to the same uninterrupted stopwatch.
CalibrationEnvelope monotonic_calibration_envelope(const CalibrationEnvelope& admitted,
                                                   const CalibrationEnvelope& observed);
v2::Result<bool> complete_calibration_marker(const std::filesystem::path& helper_log);

} // namespace bc::serving::session
