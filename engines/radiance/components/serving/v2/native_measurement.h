#pragma once
// RSC11 S13 P1 (inactive): native engine/owner/worker measurements to S8 and S11.
// The rad0011 EnginePlan is an out-of-band *measurement*, not a startup manifest.
#include "generated/v2/serving_types.h"
#include "v2/s8_rules.h"
#include "worker_footprint.h"

#include <cstdint>
#include <string>
#include <vector>

namespace bc::serving::v2 {

struct NativeEngineGpu {
  std::string bdf;
  int64_t required_bytes = 0; // direct per-rank pre-weight engine plan
  int64_t fixed_bytes = 0;    // direct per-rank non-elastic engine plan
  int64_t claimable_bytes = 0; // live rank admission
  std::string evidence;
};

struct NativeHelper {
  S8HelperRoom room;
  r3::worker_footprint::Shape shape;
  r3::worker_footprint::Lanes lanes;
  uint64_t incoming_stride_bytes = 0;
  std::string footprint_evidence; // worker_footprint.h input/model identity
};

struct NativeMeasurement {
  std::string planning_key;
  // Produced by the real radiance --plan-memory, not a calibration target.
  EnginePlan engine_plan;
  S8Scope scope;
  std::string actual_owner_map_digest; // exact bc::experts owner publisher digest
  int64_t unowned_expert_bytes = -1; // owner planner, NOT guessed by launcher
  std::string unowned_expert_evidence;
  int64_t resident_other_bytes = -1; // measured engine resident, outside other entries
  std::string resident_other_evidence;
  int64_t concurrent_sessions = 0;
  int64_t mem_available_bytes = 0;
  int64_t cgroup_headroom_bytes = 0;
  int64_t pin_available_bytes = 0;
  std::string host_live_evidence;
  MarginCalibration host_margin; // qualified calibration record only
  std::vector<NativeHelper> helpers;
  std::vector<NativeEngineGpu> gpu;
  int64_t host_reserve_bytes = -1;
  std::string host_reserve_evidence;
  int64_t dispatch_row_capacity = 0;
  std::string transport_shm;
  std::optional<std::string> histogram_file;
  bool live_policy_requested = false;
};

struct NativeMeasurementOutput {
  S8Measured measured;
  std::string s11_wire_json; // directly consumed by Python S11 MeasurementSource
};

// Source must supply actual engine/owner/BDF/cgroup/pin evidence. No inferred
// KV from host-KV calibration; derives a conservative full-session charge
// from the engine's own measured kv_one_session / target context.
// Checked integer arithmetic and scope/owner/provenance/footprint binding.
Result<NativeMeasurementOutput> produce_native_measurement(const NativeMeasurement &in);

} // namespace bc::serving::v2
