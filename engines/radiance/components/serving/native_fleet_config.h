#pragma once

#include "fleet/config.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace bc::serving {

// One managed, single-model fleet document. The site supplies non-secret YAML,
// including comments and unused anchors, since a snapshot retains every byte.
// The caller runs the native parser under the deployment's resolution root.
// The enclosing manifest owns build identity and live admission.
class NativeFleetConfig final {
  public:
    static NativeFleetConfig read(const std::filesystem::path& authored);
    // The spec.native_config field of the successor manifest. Replay takes the
    // field from a strictly decoded, identity-verified document.
    nlohmann::json manifest_field() const;
    static NativeFleetConfig replay(const nlohmann::json& native_config);

    const std::string& yaml_utf8() const { return yaml_; }
    const std::string& sha256() const { return sha256_; }
    const rad::fleet::FleetConfig& fleet() const { return fleet_; }

  private:
    NativeFleetConfig(std::string yaml_utf8, std::string resolution_root, const std::string& origin,
                      std::optional<std::string_view> expected_sha256);

    std::string yaml_;
    std::string sha256_;
    std::string resolution_root_;
    rad::fleet::FleetConfig fleet_;
};

}  // namespace bc::serving
