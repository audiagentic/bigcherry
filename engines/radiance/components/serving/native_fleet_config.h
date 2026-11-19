#pragma once

#include "fleet/config.h"
#include "v2/native_config_snapshot.h"

namespace bc::serving {

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

    const NativeConfigSnapshot& snapshot() const { return snapshot_; }
    const rad::fleet::FleetConfig& fleet() const { return fleet_; }

  private:
    NativeFleetConfig(NativeConfigSnapshot snapshot, const std::string& origin);

    NativeConfigSnapshot snapshot_;
    rad::fleet::FleetConfig fleet_;
};

}  // namespace bc::serving
