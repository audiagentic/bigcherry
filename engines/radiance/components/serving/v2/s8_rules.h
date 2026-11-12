#pragma once
// RSC11 S8: card-free, inactive v2 derivations. S10 is the first consumer.
// No v1 reader, writer, engine, or launcher is linked to this module.
#include "generated/v2/serving_types.h"
#include "v2/error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace bc::serving::v2 {

// A calibration is qualified only for the actual measured model, owner map,
// implementation and service objective. A cache entry is not a portable
// default.
struct S8Scope {
  std::string model_digest;
  std::string engine_build_digest;
  std::string helper_build_digest;
  std::string layout_digest;
  std::string inventory_digest; // selected PCI role/BDF/arch/VRAM snapshot
  std::string owner_map_digest;
  int64_t context_target_tokens = 0;
  int64_t retention_target_tokens = 0;
  int64_t kv_bytes_per_token = 0;
  bool operator==(const S8Scope &) const = default;
};

// A persisted record requires the raw A/B or admission samples, not just a
// selected value. All timestamps are caller-supplied UTC Unix seconds.
struct S8Calibration {
  std::string key;
  int64_t value = 0;
  std::string evidence_id;
  std::string raw_run;
  S8Scope scope;
  int64_t measured_at_unix_s = 0;
  int64_t valid_until_unix_s = 0;
  std::vector<int64_t>
      measured_candidates; // tested settings, including selected
  std::string method; // e.g. ABBA with output/health/effect gates, or host-room
                      // sampling
  CalibrationSource source = CalibrationSource::kCache;
  std::string
      recorded_date; // ISO date of the measured evidence, no implicit seed
};

struct S8HelperRoom {
  std::string role;
  std::string bdf;
  int64_t physical_vram_bytes = 0;  // current BDF inventory
  int64_t live_claimable_bytes = 0; // current engine/helper admission probe
  int64_t arch_reserve_bytes = 0;   // qualified architecture reserve
  std::string reserve_evidence;
  int64_t owned_experts = 0; // actual owner-map set, never inferred by Python
};

// A setting's value when no calibration record applies, and where it came from (site.toml, layouts.toml, a seed).
struct S8Setting {
  int64_t value = 0;
  std::string source;
};

struct S8Measured {
  MeasuredResourceInputs resource_inputs; // S9 typed v2 wire members
  S8Scope scope;
  std::string
      actual_owner_map_digest; // from the engine's real owner-map publisher
  std::vector<S8HelperRoom> helpers; // exact role/BDF match with footprints
  int64_t host_reserve_bytes = 0;    // S7 host ledger, with explicit provenance
  std::string host_reserve_evidence;
  int64_t engine_vram_required_bytes = 0;
  int64_t engine_vram_claimable_bytes = 0;
  int64_t checkpoint_slots = 0; // the slots the host ledger fitted between the plan's floor and want
  int64_t checkpoint_slot_bytes = 0; // one slot, from the engine's plan (--plan-memory)
  int64_t dispatch_row_capacity = 0; // engine/worker measured dispatch limit
  std::string transport_shm;
  std::optional<std::string> histogram_file; // absent means disabled
  bool live_policy_requested =
      false; // no live cadence requirement when policy off
  std::map<std::string, S8Setting> settings; // by calibration key: the value used when no record applies
};

struct S8HelperBudget {
  std::string role;
  std::string bdf;
  int64_t expert_budget_mib = 0;
  int64_t slack_bytes = 0;
};

// S10 writes the returned typed values directly into v2 spec.*;
// S13 verifies source records and strict covering provenance.
struct S8Resolution {
  MeasuredResourceInputs resource_inputs;
  HostResources host;
  std::vector<S8HelperBudget> helpers;
  std::optional<HandoffTransport> transport; // absent without remote helpers
  std::optional<HandoffKernels> kernels;
  std::optional<HandoffTelemetry> telemetry;
  std::optional<int64_t> live_cadence_ns; // qualified RHP18 startup decision
  std::map<std::string, std::string> evidence_by_path;
  std::vector<CalibrationEntry>
      calibration_entries; // merge once into spec.calibration
  std::map<std::string, ProvenanceRecord> provenance_by_path;
  std::vector<std::string> skipped_records; // records given but not used, and why
};

// A record replaces a setting's given value when it matches the measurement scope and is current; others are skipped
// and reported, never refused. now_unix_s is injected for replayable tests.
Result<S8Resolution> resolve_s8(const S8Measured &measured,
                                const std::vector<S8Calibration> &records,
                                int64_t now_unix_s);

// The pinned host pool S8 derives from the measured inputs: host embeddings, unowned experts, staging and the margin,
// rounded up to whole MiB. A plan measured with another pool is re-measured with this one (bc-serve exit 11).
std::optional<int64_t> derived_host_pool_mib(const MeasuredResourceInputs &inputs);

// Reusable engine VRAM admission: no hidden tolerance (including the observed
// 2026-11 R9700 1,036,288-byte shortfall).
Status require_measured_vram(int64_t required_bytes, int64_t claimable_bytes,
                             const std::string &bdf);

} // namespace bc::serving::v2
