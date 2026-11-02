#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace bc::serving {
using Json = nlohmann::json;

struct Service {
    std::string objective, retention, quality;
    uint32_t max_sessions = 0, max_context = 0;
};
struct Layout {
    std::string name, engine_card;
    std::vector<std::string> helper_cards;
};
struct Memory {
    uint64_t host_pool_mib = 0, host_kv_mib = 0, tail_budget_mib = 0;
    std::vector<uint64_t> helper_budgets_mib;
};
// Shared launcher COMMON fields live outside the exact WorkerConfig struct.
struct Handoff {
    std::string shm, heat, histogram, placement;
    uint32_t max_rows = 0, small_rows = 0, gemv_large_rows = 0;
    uint32_t histogram_period_s = 0;
    bool moe_down_tile = true, skip_owned = true;
};
// Exact field set of moe-remote/worker_config.h::WorkerConfig.
struct Helper {
    uint32_t count = 0, index = 0, engine_ranks = 0;
    bool cpu = false, power_save = true, queue_warm = false, combine = false;
    uint32_t combine_min_rows = 256, combine_threads = 8;
    bool cpu_set_auto = true;
};
struct Engine {
    std::string model, radiance_home, kernels, placement, ngram_placement;
    std::string kv_cache_dtype, rope_scaling, prefix_resume;
    uint32_t port = 0, tp = 0, max_num_seqs = 0, max_num_batched_tokens = 0;
    uint32_t checkpoint_interval = 0, num_speculative_tokens = 0, max_model_len = 0;
    uint64_t host_pool_mib = 0, prefix_cache_mib = 0;
};
struct Config {
    uint32_t schema_version = 1;
    Service service;
    Layout layout;
    Memory memory;
    Handoff handoff;
    Helper helper;
    Engine engine;
};
struct Options {
    std::filesystem::path sys_root = "/";
    std::filesystem::path site, profile;
    std::vector<std::string> sets;
};
struct Resolved {
    Config typed;
    Json values, provenance, facts, manifest;
    std::string hash;
};
Json detect(const std::filesystem::path& root = "/");
Json json_schema();
Resolved resolve(const Options& options, const Json& facts);
std::string sha256_hex(const std::string& text);
} // namespace bc::serving
