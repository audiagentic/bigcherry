#pragma once
// Manifest-authenticated power ownership for RSC02. No raw callback or Linux
// sysfs operation is accessible through this interface.
#include "power_transaction.h"
#include "v2/manifest.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace bc::serving::session {

// The power run is an execution identity, not the immutable manifest origin.
struct PowerExecution {
    std::string run_id;
    std::string manifest_origin_instance_id;
};

class ManifestPower final {
  public:
    ManifestPower(power::Hardware& hardware, power::Records& records, power::Sensor& sensor,
                  power::SessionPower::Predicate lease, std::string workload = {});
    // `plan` is the S10 output (a strictly read, typed manifest); its digest is a record in the receipt, not an input.
    v2::Status begin(const v2::Manifest& plan, v2::PlanUse use);
    v2::Status verify_serving(const std::filesystem::path& manifest);
    v2::Status capture(power::WindowPhase phase);
    void children_verified() noexcept;
    v2::Status finish();
    const std::optional<PowerExecution>& execution() const noexcept {
        return execution_;
    }

  private:
    power::Hardware& hardware_;
    power::Records& records_;
    power::Sensor& sensor_;
    power::SessionPower::Predicate lease_;
    bool verified_exit_ = false;
    std::string workload_;   // the config that ran; receipt evidence only
    std::unique_ptr<power::SessionPower> transaction_;
    std::optional<PowerExecution> execution_;
    std::optional<power::PowerPlan> bound_;
    std::vector<power::PciAddress> engine_;
    std::vector<std::optional<power::Window>> floor_;
};

// Verify an actual agent-lease JSON record by expiry, instance and a live
// ancestor PID. No parsing of 'agent-lease status' or inherited claim of a
// held lease. The owner calls this only for an explicitly opted-in apply.
bool owner_has_gpu_lease(const std::filesystem::path& leases = "/control/leases",
                         const std::filesystem::path& proc = "/proc");

} // namespace bc::serving::session
