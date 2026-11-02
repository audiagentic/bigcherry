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
    std::string profile = "production";
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
    uint32_t moe_down_tile = 1;
    bool skip_owned = true;
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
// Immutable per-process view, selected by --role from a verified manifest.
struct HelperSettings {
    Helper helper;
    Handoff handoff;
    std::string role, content_sha256, model, radiance_home;
    std::string plugin, plugin_file, formats, heat;
    std::vector<uint64_t> budgets_mib, shares;
    std::vector<double> balance_speeds;
    std::vector<uint32_t> small_lanes;
    uint64_t budget_mib = 0;
    uint32_t gemv = 1, gemv_h2 = 2, moe_form = 0, moe_long_rows = 0;
    bool gemv_wide = true, gemv_w4h2 = true, gemv_mr = false;
    bool rows = false, rows_small = false, rows_kernel = false;
    bool fuseroute = true, vecio = true, sparse_act = true, lane_quant = true;
    bool native = true, fp16 = true, ahalf = true;
    bool balance = false;
    std::string capture, poison = "0", test_stop_at;
    uint32_t capture_every = 37, capture_max = 16, trace = 0;
    uint32_t moe_probe = 0;
    bool refusal_trace = false;
    bool sweep_fast = false;
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
// Strict manifest reader validates schema, cross-field invariants, and content hash.
Json read_manifest(const std::filesystem::path& path);
// Select helper<N> and reject any unknown/missing settings before GPU initialization.
HelperSettings load_helper_settings(const std::filesystem::path& path, const std::string& role);
std::string sha256_hex(const std::string& text);
} // namespace bc::serving
