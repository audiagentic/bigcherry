#pragma once
// Inactive P1 native publication boundary. The caller must supply the actual
// engine plan, owner map, selected worker geometry and live host observations.
// No v1 manifest or calibrated KV target is accepted.
#include "generated/v2/serving_types.h"
#include "v2/error.h"
#include "worker_footprint.h"

#include <cstdint>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace bc::serving::v2 {

struct P1Helper {
  std::string role;
  std::string bdf;
  r3::worker_footprint::Shape shape;
  r3::worker_footprint::Lanes lanes;
  uint64_t largest_expert_device_bytes = 0;
};

struct P1Gpu {
  std::string bdf;
  int64_t required_bytes = 0;
  int64_t fixed_bytes = 0;
  int64_t claimable_bytes = 0; // the rank's live claimable VRAM after the rank split
};

struct P1Native {
  EnginePlan engine_plan;      // radiance --plan-memory result, not an estimate
  OwnerMap owner_map;          // engine's selected, verified owner map
  MarginCalibration host_margin; // P2 qualified record, never synthesized
  std::vector<P1Helper> helpers;
  std::vector<P1Gpu> gpu;
  std::string planning_key;    // keyed model/build/layout/owner/targets/inventory
  std::string provenance;      // retained measurement run identity
  std::string model_digest, engine_build_digest, helper_build_digest;
  std::string layout_digest, inventory_digest;
  int64_t unowned_expert_bytes = -1; // actual model ownership accounting
  int64_t resident_other_bytes = -1; // measuring-start resident host bytes
  int64_t context_target_tokens = 0;
  int64_t retention_target_tokens = 0;
  int64_t concurrent_sessions = 0;
  int64_t host_reserve_bytes = -1; // measured/qualified host reserve
  int64_t mem_available_bytes = -1; // live MemAvailable
  int64_t cgroup_headroom_bytes = -1; // effective cgroup limit - usage
  int64_t pin_available_bytes = -1; // actual pinnable headroom
};

// Serializes only observations with a complete source. In particular,
// KV bytes/token is recovered from the engine's full-context KV footprint.
// Missing or non-integral geometry is refused, never repaired with a target.
Result<nlohmann::json> publish_p1_measurement(const P1Native &m);

} // namespace bc::serving::v2
