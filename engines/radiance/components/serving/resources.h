#pragma once
// RSC03 resource admission: measured model bytes, verified card footprints and host limits.
// The calculation is independent of manifest JSON and of any process or hardware access.
#include "v2/error.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace bc::serving::v2 {
struct Manifest;
}

namespace bc::serving::resources {

struct ModelFootprint {
    std::string owner_map_digest;
    uint64_t embedding_bytes = 0;
    uint64_t unowned_expert_bytes = 0;
    uint64_t staging_bytes = 0;
    uint64_t ngram_bytes = 0;
    uint64_t checkpoint_bytes = 0;
    uint64_t other_engine_bytes = 0;
    uint64_t kv_bytes_per_token = 0;
};

struct RetentionTarget {
    uint64_t context_tokens = 0;
    uint64_t concurrent_sessions = 0;
    uint64_t retained_tokens = 0;
    uint64_t pool_margin_bytes = 0;
    uint64_t host_reserve_bytes = 0;
};

struct HostCapacity {
    uint64_t mem_available_bytes = 0;
    uint64_t cgroup_headroom_bytes = 0;
    uint64_t pin_available_bytes = 0;
};

struct HelperCapacity {
    std::string bdf;
    std::string arch;
    uint64_t vram_mib = 0;
    uint64_t architecture_reserve_mib = 0;
    uint64_t worker_footprint_bytes = 0;
    std::optional<uint64_t> planned_budget_mib;
};

struct ResourceRequest {
    ModelFootprint model;
    RetentionTarget target;
    HostCapacity host;
    std::vector<HelperCapacity> helpers;
    uint64_t engine_ranks = 1; // host_pool_mib is allocated once per selected engine rank
};

enum class Allocation : uint8_t {
    embedding,
    unowned_experts,
    staging,
    pool_margin,
    ngram_ram,
    checkpoints,
    other_engine,
    retained_kv,
    host_reserve,
    count
};
constexpr size_t AllocationCount = static_cast<size_t>(Allocation::count);

struct HelperBudget {
    std::string bdf;
    std::string arch;
    uint64_t vram_mib = 0;
    uint64_t reserve_mib = 0;
    uint64_t worker_footprint_mib = 0;
    uint64_t expert_budget_mib = 0;
    uint64_t slack_bytes = 0;
};

struct ResourceLedger {
    std::string owner_map_digest;
    uint64_t host_pool_mib = 0;
    uint64_t host_kv_mib = 0;
    std::vector<HelperBudget> helpers;
    std::array<uint64_t, AllocationCount> allocations{};
    uint64_t available_bytes = 0;
    uint64_t committed_bytes = 0;
    uint64_t remaining_bytes = 0;
    uint64_t pin_available_bytes = 0;
    uint64_t pin_committed_bytes = 0;
};

// Pure, deterministic admission; invalid measurements and budget mismatches are typed errors.
v2::Result<ResourceLedger> calculate(const ResourceRequest& request);

// The resolver and v2 manifest are JSON on disk. This adapter validates them, reads only
// RAD directory metadata, and builds a ResourceRequest before invoking calculate().
// The published owner map, not a recalculated one, determines GPU-owned expert weights.
struct LaunchDocuments {
    const nlohmann::json& resolved_values;
    const nlohmann::json& resolved_helpers;
    const nlohmann::json& detected_facts;
    const nlohmann::json& unmeasured_manifest;
};
v2::Result<ResourceLedger> derive_launch(const LaunchDocuments& documents,
                                         const std::filesystem::path& calibration_cache = {});

// The measuring start has no measured engine footprint yet. Still admit the
// effective allocation from the exact measuring manifest against fresh RAM, cgroup and pin room.
v2::Status preflight_host_pool(const v2::Manifest& measuring_manifest,
                               const nlohmann::json& detected_facts);

// Wire format is needed only for status and the frozen Python-oracle parity fixture.
nlohmann::json to_json(const ResourceLedger& ledger);

} // namespace bc::serving::resources
