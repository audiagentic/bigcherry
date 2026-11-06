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
};

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
    const bc::experts::Container& container);
} // namespace bc::serving
