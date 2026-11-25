#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace bc::serving {

// The closed, typed v1 `x-bigcherry` section of a native serving YAML (docs/design/radiance-native-config-x-bigcherry.md).
// It holds only what BigCherry owns: rank architectures, helper groups, service policy, memory-plan overrides, handoff
// and helper process settings, kernel switches and variant provenance. Radiance's parser skips it; the engine's
// inspector returns it as JSON from the same sealed bytes and this decodes it.
struct XRank {
    std::string arch;
    std::int64_t count;
};
struct XHelperGroup {
    std::string arch;
    std::int64_t count;  // -1: every unused card of the arch ("all")
    std::string placement;  // "head" or "tail"
    std::optional<std::int64_t> budget_mib, share;
};
struct XBigcherry {
    std::string description;
    std::optional<std::string> variant_of;
    std::vector<std::string> differs;
    std::vector<XRank> ranks;
    std::vector<XHelperGroup> helpers;
    // Validated sections: closed key sets, typed values, dashed key names as authored.
    nlohmann::json power = nlohmann::json::object(), service = nlohmann::json::object(), memory = nlohmann::json::object(),
                   handoff = nlohmann::json::object(), helper = nlohmann::json::object(),
                   kernels = nlohmann::json::object();

    std::int64_t rank_count() const;
    // The named power profile of engines/radiance/power-profiles.yaml this config selects; stock when absent.
    std::string power_profile() const { return power.contains("profile") ? power.at("profile").get<std::string>() : "stock"; }
};

// Decodes the inspector's `x_bigcherry` value. Unknown keys, wrong types, a missing required key, an unsupported
// schema and malformed helper or rank groups are refused with std::invalid_argument naming the YAML path.
XBigcherry decode_x_bigcherry(const nlohmann::json& section);

}  // namespace bc::serving
