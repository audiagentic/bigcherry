#pragma once

#include "fleet/config.h"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace bc::serving {

// One managed, single-model fleet document. The site supplies non-secret YAML,
// including comments and unused anchors, since a snapshot retains every byte.
// The measured manifest owns build identity, resolution root and live admission.
class NativeFleetConfig final {
  public:
    static NativeFleetConfig read(const std::filesystem::path& authored);
    // Both inputs come from the verified manifest; replay never opens the author's file.
    static NativeFleetConfig replay(std::string yaml_utf8, std::string_view expected_sha256);

    const std::string& yaml_utf8() const { return yaml_; }
    const std::string& sha256() const { return sha256_; }
    const rad::fleet::FleetConfig& fleet() const { return fleet_; }

  private:
    NativeFleetConfig(std::string yaml_utf8, const std::string& origin,
                      std::optional<std::string_view> expected_sha256);

    std::string yaml_;
    std::string sha256_;
    rad::fleet::FleetConfig fleet_;
};

}  // namespace bc::serving
