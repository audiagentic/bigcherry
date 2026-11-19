#include "native_fleet_config.h"

#include <stdexcept>
#include <utility>

namespace bc::serving {

NativeFleetConfig NativeFleetConfig::read(const std::filesystem::path& authored) {
    return NativeFleetConfig(NativeConfigSnapshot::read(authored), authored.string());
}

NativeFleetConfig NativeFleetConfig::replay(NativeConfigSnapshot snapshot) {
    return NativeFleetConfig(std::move(snapshot), "spec.native_config.yaml_utf8");
}

NativeFleetConfig::NativeFleetConfig(NativeConfigSnapshot snapshot, const std::string& origin)
    : snapshot_(std::move(snapshot)) {
    snapshot_.require_execution_root();
    if (rad::fleet::parse(snapshot_.yaml_utf8(), origin, &fleet_) != RAD_OK)
        throw std::invalid_argument("Radiance refused the native fleet YAML: " + origin);
    if (fleet_.models.size() != 1)
        throw std::invalid_argument("managed native fleet YAML requires exactly one model");
    // Native provenance includes inherited and aliased server options. Checking
    // the option, not its resolved value, also permits environment references.
    if (fleet_.models.front().cfg.given.contains("--api-key"))
        throw std::invalid_argument("managed native fleet YAML requires server.api-key-env for credentials");
}

}  // namespace bc::serving
