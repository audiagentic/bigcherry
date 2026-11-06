#pragma once
// RHP16: ingest the helper's one-line, measured calibration evidence into the TOML cache.
// No GPU or manifest-contract changes; caller supplies a strict v1 lab manifest.
#include <filesystem>
#include <string>

namespace bc::serving {
// Card-free verified-identity ingestion primitive; the CLI wrapper also checks
// the strict v1 lab manifest before supplying these identity values.
void ingest_live_policy_record(const std::filesystem::path& log,
                               const std::filesystem::path& cache,
                               const std::string& key, const std::string& layout,
                               const std::string& objective,
                               const std::string& target_store_key = "");
void ingest_live_policy_log(const std::filesystem::path& log,
                            const std::filesystem::path& cache,
                            const std::filesystem::path& manifest,
                            const std::string& target_store_key);
} // namespace bc::serving
