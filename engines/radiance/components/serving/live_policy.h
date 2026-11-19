#pragma once
// RHP16: the card-free, read-only per-plan live-policy calibration and derivation.
#include "calibration.h"
#include <filesystem>
#include <optional>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace bc::experts { struct Container; }
namespace bc::serving {
struct ManifestInputs;
struct LivePolicyRecord {
    std::string key, layout, objective, seeded, evidence, source;
    nlohmann::json measurements;
    // Records bind source evidence to the narrow, typed hardware/service scope; a record without it is refused.
    std::optional<nlohmann::json> calibration_inputs;
    std::string source_plan_key, source_engine_build;
    // In-memory activation state: 1 includes verified post-#1169 scoped
    // records (later revision/run/host provenance remains unavailable).
    uint32_t record_schema_version = 0;
    std::string calibration_kind, source_revision, run_id, host_facts_digest;
    nlohmann::json calibration_key_fields = nlohmann::json::array();
    // Partial evidence has explicit missing reasons; never derive from it.
    nlohmann::json absent = nlohmann::json::object();
};
bool live_policy_record_complete(const LivePolicyRecord& record);


// Reads [[live_policy_record]] from the existing --calibration TOML cache.
// Every record is validated, including nonmatching records; missing key means no calibration.
std::optional<LivePolicyRecord> read_live_policy_record(const std::filesystem::path& cache,
                                                       const std::string& plan_key);

// Recompute all policy values when the expert stride, helper budgets or calibrated readings change.
// Provenance is keyed by the full manifest JSON Pointer, one record per leaf.
struct DerivedLivePolicy {
    nlohmann::json policy;
    nlohmann::json provenance;
};
DerivedLivePolicy derive_live_policy(const LivePolicyRecord& record, const Calibration& calibration,
                                     const CalibrationMatch& match, uint64_t expert_stride_bytes,
                                     const std::vector<uint64_t>& helper_budget_bytes);
std::optional<DerivedLivePolicy> derive_manifest_live_policy(const ManifestInputs& in,
    const bc::experts::Container& container, const nlohmann::json& spec,
    std::string* skipped_reason = nullptr);
} // namespace bc::serving
