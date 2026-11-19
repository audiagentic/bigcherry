#pragma once
// MRC03: platform v1 feature-state snapshot derived from an admitted serving manifest.
// This adapter is native C++ and has no Python or engine-runtime dependency.

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace bc::serving::feature_state {

inline constexpr char kFileName[] = "feature-states.v1.json";

struct Snapshot {
    nlohmann::json entries = nlohmann::json::array();
    std::vector<std::string> warnings;  // owner.log only; never required to parse status
};

// Pure function over the exact manifest chosen for a serving start. Throws on
// missing or contradictory required facts instead of pretending a feature is on.
Snapshot from_manifest(const nlohmann::json& manifest, const std::string& observed_at);

} // namespace bc::serving::feature_state
