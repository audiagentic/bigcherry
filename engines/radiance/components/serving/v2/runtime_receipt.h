#pragma once
// Inactive S13 runtime receipt: separates the immutable S10 v2 manifest
// spec.handoff.placement.live_policy_state startup decision from actual runtime
// policy degradation / placement-stopped evidence. No v1 activation and no
// production linkage. Bounded snapshots are derived only from authoritative
// worker_epoch.h PlacementControl/PlacementJournal telemetry; missing evidence
// is fail-closed and never reported as successful runtime policy.
#include "generated/v2/serving_types.h"
#include "v2/error.h"
#include "../../moe-remote/moe_remote_shm.h"

#include <cstdint>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace bc::serving::v2 {

struct RuntimeSnapshot {
  std::string manifest_digest;
  uint64_t epoch = 0;
  uint64_t owner_generation_digest = 0;
  std::string owner_generation_digest_hex;
  uint32_t state = 0;
  uint32_t fault_code = 0;
  uint32_t fault_actor = 0;
  uint64_t transaction_id = 0;
  uint32_t cleanup_fault = 0;
  uint32_t cleanup_actor = 0;
  bool placement_stopped = false;
  bool degraded = false;
};

// One typed producer. The caller must supply the genuine worker_epoch
// PlacementControl and the matching current journal; no fallback fields or
// aliases are accepted. Missing, zero, mismatched or stale telemetry fails
// closed.
Result<nlohmann::json> publish_runtime_receipt(
    const LivePolicyStartup& startup_decision,
    const std::string& manifest_content_digest,
    uint64_t epoch,
    uint64_t owner_generation_digest,
    const r3::moe_shm::PlacementControl& control,
    const r3::moe_shm::PlacementJournal& journal);

} // namespace bc::serving::v2