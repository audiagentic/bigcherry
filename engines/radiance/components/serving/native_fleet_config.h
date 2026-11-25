#pragma once

#include "fleet/config.h"
#include "v2/manifest.h"
#include "v2/native_config_snapshot.h"

#include <optional>
#include <string>

namespace bc::serving {

// The top-level x-bigcherry subtree of YAML text as JSON (null when absent); see native_fleet_config.cpp.
nlohmann::json x_bigcherry_subtree(const std::string& yaml_utf8);
// The whole document of a YAML file as JSON, by the same parser and converter (anchors and merge keys resolved, a
// duplicate key refused naming its YAML path): the power profile file, which bc-serve cannot parse itself.
nlohmann::json yaml_document_json(const std::string& yaml_utf8, const char* name);

// One managed, single-model fleet document. The site supplies non-secret YAML,
// including comments and unused anchors, since a snapshot retains every byte.
// The caller runs the native parser under the deployment's resolution root.
// The enclosing manifest owns build identity and live admission.
class NativeFleetConfig final {
  public:
    static NativeFleetConfig read(const std::filesystem::path& authored,
                                  const std::filesystem::path& resolution_root);
    // Replay takes a snapshot from the strictly decoded, identity-verified
    // spec.native_config field.
    static NativeFleetConfig replay(NativeConfigSnapshot snapshot);
    static NativeFleetConfig replay(const v2::NativeConfigView& admitted);

    const NativeConfigSnapshot& snapshot() const { return snapshot_; }
    const rad::fleet::FleetConfig& fleet() const { return fleet_; }

  private:
    NativeFleetConfig(NativeConfigSnapshot snapshot, const std::string& origin);

    NativeConfigSnapshot snapshot_;
    rad::fleet::FleetConfig fleet_;
};

// Handle the engine's read-only inspection command. A disengaged result means
// this is an ordinary Radiance invocation; otherwise the value is its exit code.
std::optional<int> inspect_native_fleet_config_command(int argc, char** argv);

}  // namespace bc::serving
