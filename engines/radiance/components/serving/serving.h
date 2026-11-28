#pragma once
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "xbigcherry.h"

namespace bc::serving {

/* The engine options the registry binds to a field of their own beyond the hand-written resolver keys (contract amendment 13):
 * one `section.name` key each (`sampling.temp`, `limits.max_body_mib`, ...), the manifest leaf it becomes, and its shape. Read
 * from the generated registry, so a field added there is a resolver key and a manifest leaf without another table. */
struct OptionField {
    std::string section, name, key, path, kind, doc;
    std::string env;          // the environment variable the field stands in for (amendment 14), or empty for a flag
    bool is_switch = false;   // the flag takes no value: the key's one value is "on"
};
const std::vector<OptionField>& engine_option_fields();
using Json = nlohmann::json;

struct Service {
    std::string objective, retention, quality;
    uint32_t max_sessions = 0, max_context = 0;
};
struct Layout {
    std::string name;
    std::vector<std::string> engine_cards;
    std::vector<std::string> helper_cards;
};
struct Memory {
    uint64_t host_pool_mib = 0, host_kv_mib = 0, host_reserve_mib = 0, tail_budget_mib = 0;
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
    uint64_t host_pool_mib = 0, prefix_cache_mib = 0, host_headroom_mib = 0;
    // Set only when an operator gave them (RSC11 A1); absent leaves radiance's own default.
    std::optional<std::string> tp_wire;
    bool no_prefix_cache = false;
    std::optional<uint32_t> gpu_headroom_mib;
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
// The representable engine settings reported by the matched native parser.
struct NativeEngineSettings {
    uint32_t port, max_context, max_sequences, max_batched_tokens, checkpoint_interval;
    std::optional<uint32_t> speculative_tokens;   // unset: native "auto" depth, the engine decides
    std::vector<std::string> kernels;
    std::string placement, ngram_placement, kv_dtype, rope_scaling, prefix_resume;   // rope_scaling empty: model-defined
    bool prefix_cache;
    bool profile_ops = false;   // per-op timing (models.<id>.profile-ops): the runtime MTP replay guard exempts it
    std::optional<uint32_t> prefill_interleave;
    // The native tuning projection (kNativeTuningFields). The engine validates YAML;
    // the resolver validates the manifest field types and ranges.
    std::map<std::string, nlohmann::json> tuning;
};
// One serving config as its native YAML states it, read by the engine's own parser through the device-free inspector
// (session.h inspect_config_file): the config name is the file stem. This replaces the RSC12 TOML catalogue as the
// resolver's source of a config's model, ranks, helpers, wire and BigCherry-owned settings.
struct InspectedConfig {
    std::string name;
    std::filesystem::path yaml_path;   // the file read; its directory's sibling heat/ holds @bundle/heat seeds
    std::string sha256;                // digest of the exact bytes the inspector parsed
    std::filesystem::path radiance_home;   // the canonical runtime home that parsed them (a start must keep it)
    std::string model;                 // as authored: absolute, or relative to the model root
    std::optional<std::string> tp_wire;
    XBigcherry x;
};
struct Options {
    std::filesystem::path sys_root = "/";
    std::filesystem::path site, profile;
    // Directories holding NAME.yaml configs (--configs); empty: BIGCHERRY_CONFIGS, else the installed bundle's configs/.
    std::vector<std::filesystem::path> configs;
    // The root a config's relative model path is under (--model-root); bc-serve reads no environment for it.
    std::filesystem::path model_root;
    // Internal session-owner context: no CLI flag. Only serve/replay populates
    // it, to derive mutable handoff.histogram under the run directory.
    std::filesystem::path run_dir;
    // A calibration cache (RHP12 refits) layered over the compiled-in seeds; empty: seeds only.
    std::filesystem::path calibration_cache;
    std::vector<std::string> sets;
    // The config named by layout.name, read from its YAML: when set, it is the one config the resolver knows (the
    // catalogue readers are gone); the caller runs the inspector first. Empty only for tests of layout files.
    std::optional<InspectedConfig> inspected;
};
// The string a layered setting `section.key` takes before any config is resolved: the last --set, else the profile, else the
// site file (the resolver's own precedence among them); nothing when none sets it. A --set value is JSON or bare text.
std::optional<std::string> layered_string(const Options& options, const std::string& section, const std::string& key);
struct Resolved {
    Config typed;
    Json values, provenance, facts, manifest;
    std::string hash;
};
Json detect(const std::filesystem::path& root = "/");
Json json_schema();
Resolved resolve(const Options& options, const Json& facts,
                 const NativeEngineSettings* native = nullptr);
/* The one named config of Options::inspected as a layout, with its file: {"layout": {name: table}, "source": {name: path},
 * "kind": {name: "config"}}. Refused when no config was read. */
Json load_layouts(const Options& options);
/* archs.toml reserve-mib of a card type: the VRAM a card of it keeps free. */
uint64_t arch_reserve_mib(const std::string& arch);
} // namespace bc::serving
