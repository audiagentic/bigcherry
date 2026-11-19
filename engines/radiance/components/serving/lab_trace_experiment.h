#pragma once
#include <cstdint>
#include <nlohmann/json.hpp>
namespace bc::serving {
// Lab-only trace: strict admission of the existing Nth-request diagnostic.
// No shape, force, kernel override or other experiment changes.
inline bool lab_lane_trace_only(const nlohmann::json& experiment) {
    if (!experiment.is_object() || experiment.size() != 1 ||
        !experiment.contains("diagnostics")) return false;
    const auto& diagnostics = experiment.at("diagnostics");
    if (!diagnostics.is_object() || diagnostics.size() != 1 ||
        !diagnostics.contains("lanes")) return false;
    const auto& lanes = diagnostics.at("lanes");
    if (!lanes.is_object() || lanes.size() != 1 ||
        !lanes.contains("trace_level")) return false;
    const auto& n = lanes.at("trace_level");
    return n.is_number_integer() && n.get<int64_t>() > 0 &&
           n.get<int64_t>() <= UINT32_MAX;
}
} // namespace bc::serving
