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
// The strict single-line marker payload (JSON after RHP16_CALIBRATION_JSON ): the one writer. A complete row replaces
// only a partial one for the same key; a complete row is never replaced and a partial row never displaces another.
void ingest_live_policy_marker(const std::string& payload, const std::filesystem::path& cache,
                               const std::string& key, const std::string& layout,
                               const std::string& objective,
                               const std::string& target_store_key,
                               const LivePolicyProvenance& provenance);
// Everything the primitive needs, taken from one strict serving manifest (no lab-run checks).
struct LivePolicyTarget {
    std::string plan_key, layout, objective, store_key;
    LivePolicyProvenance provenance;
};
LivePolicyTarget live_policy_target_from_manifest(const std::filesystem::path& manifest);
// The lab calibration-run entry: checks that the manifest is a lab calibration run, then ingests.
void ingest_live_policy_log(const std::filesystem::path& log,
                            const std::filesystem::path& cache,
                            const std::filesystem::path& manifest,
                            const std::string& target_store_key);
// Remove the rows the strict reader refuses (no calibration_inputs or record_schema_version: evidence from before native v1 records) from a
// calibration store. Returns the plan_cache_key of each row that is, or with apply was, removed. Takes the store's lock,
// keeps every other byte, validates the result with the strict reader, and replaces the file atomically.
std::vector<std::string> prune_live_policy_store(const std::filesystem::path& cache, bool apply);
} // namespace bc::serving
