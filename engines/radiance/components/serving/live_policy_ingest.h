#pragma once
// RHP16: ingest the helper's one-line, measured calibration evidence into the TOML cache.
// No GPU or manifest-contract changes; caller supplies a strict v1 lab manifest.
#include <filesystem>
#include <optional>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace bc::serving {
// Card-free verified-identity ingestion primitive; the CLI wrapper also checks
// the strict v1 lab manifest before supplying these identity values.
struct LivePolicyProvenance {
    nlohmann::json inputs;
    std::string source_plan_key, source_engine_build;
    // Audit-only provenance: never fed back into policy_calibration_key().
    std::string source_revision, run_id, host_facts_digest;
};
void ingest_live_policy_record(const std::filesystem::path& log,
                               const std::filesystem::path& cache,
                               const std::string& key, const std::string& layout,
                               const std::string& objective,
                               const std::string& target_store_key,
                               const LivePolicyProvenance& provenance);
void ingest_live_policy_log(const std::filesystem::path& log,
                            const std::filesystem::path& cache,
                            const std::filesystem::path& manifest,
                            const std::string& target_store_key);
// Remove the rows the strict reader refuses (no calibration_inputs: hash-only evidence from before scoped keys) from a
// calibration store. Returns the plan_cache_key of each row that is, or with apply was, removed. Takes the store's lock,
// keeps every other byte, validates the result with the strict reader, and replaces the file atomically.
std::vector<std::string> prune_live_policy_store(const std::filesystem::path& cache, bool apply);
} // namespace bc::serving
