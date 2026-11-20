#define _GNU_SOURCE 1
#include "native_fleet_config.h"

#include "fleet/fleet.h"

#include <charconv>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace bc::serving {

NativeFleetConfig NativeFleetConfig::read(const std::filesystem::path& authored,
                                          const std::filesystem::path& resolution_root) {
    return NativeFleetConfig(NativeConfigSnapshot::read(authored, resolution_root), authored.string());
}

NativeFleetConfig NativeFleetConfig::replay(NativeConfigSnapshot snapshot) {
    return NativeFleetConfig(std::move(snapshot), "spec.native_config.yaml_utf8");
}

NativeFleetConfig NativeFleetConfig::replay(const v2::NativeConfigView& admitted) {
    return replay(NativeConfigSnapshot::admitted(std::string(admitted.yaml_utf8),
                                                 std::string(admitted.identity.sha256),
                                                 std::string(admitted.identity.resolution_root)));
}

NativeFleetConfig::NativeFleetConfig(NativeConfigSnapshot snapshot, const std::string& origin)
    : snapshot_(std::move(snapshot)) {
    snapshot_.require_execution_root();
    if (rad::fleet::parse(snapshot_.yaml_utf8(), origin, &fleet_) != RAD_OK)
        throw std::invalid_argument("Radiance refused the native fleet YAML: " + origin);
    if (fleet_.models.size() != 1)
        throw std::invalid_argument("managed native fleet YAML requires exactly one model");
    if (const std::string unsupported = rad::fleet::unsupported(fleet_); !unsupported.empty())
        throw std::invalid_argument("Radiance build does not support managed fleet settings: " + unsupported);
    // Managed manifests construct one Engine from the model Config. These
    // settings have no effect without the upstream Fleet lifecycle.
    const auto& model = fleet_.models.front();
    const rad::fleet::FleetConfig defaults;
    const std::pair<bool, const char*> fleet_only[] = {
        {fleet_.host_mib != defaults.host_mib, "memory.host-mib"},
        {fleet_.disk_tier != defaults.disk_tier, "memory.disk-tier"},
        {fleet_.load_timeout_ms != defaults.load_timeout_ms, "scheduler.load-timeout"},
        {fleet_.drain_timeout_ms != defaults.drain_timeout_ms, "scheduler.drain-timeout"},
        {fleet_.keepalive_ms != defaults.keepalive_ms, "scheduler.keepalive"},
        {fleet_.stuck_exit_ms != defaults.stuck_exit_ms, "scheduler.stuck-exit"},
        {fleet_.warm_up != defaults.warm_up, "scheduler.warm-up"},
        {!model.aliases.empty(), "aliases"},
        {model.priority != 0, "priority"},
        {model.pin, "pin"},
        {model.ttl_ms != 0, "ttl"},
        {model.flex != rad::fleet::FlexSpec{}, "flex"},
        {model.park != rad::fleet::ParkSpec{}, "park"},
    };
    for (const auto& [configured, option] : fleet_only)
        if (configured)
            throw std::invalid_argument(std::string("managed single-engine YAML cannot apply fleet-only setting: ") +
                                        option);
    // Native provenance includes inherited and aliased server options. Checking
    // the option, not its resolved value, also permits environment references.
    if (model.cfg.given.contains("--api-key"))
        throw std::invalid_argument("managed native fleet YAML requires server.api-key-env for credentials");
}

std::optional<int> inspect_native_fleet_config_command(int argc, char** argv) {
    constexpr const char* command = "--bc-inspect-native-config";
    if (argc < 2 || argv[1] == nullptr || std::strcmp(argv[1], command) != 0) return std::nullopt;
    if (argc != 3 || argv[2] == nullptr) {
        std::cerr << "radiance: " << command << " requires one inherited sealed descriptor\n";
        return 2;
    }

#if defined(__linux__)
    try {
        int fd = -1;
        const char* end = argv[2] + std::strlen(argv[2]);
        const auto [ptr, error] = std::from_chars(argv[2], end, fd);
        if (error != std::errc{} || ptr != end || fd < 3)
            throw std::invalid_argument("descriptor must be an integer greater than 2");

        NativeFleetConfig managed = NativeFleetConfig::replay(NativeConfigSnapshot::from_sealed_fd(fd));
        const auto& fleet = managed.fleet();
        const auto& cfg = fleet.models.front().cfg;
        auto source = [&](const char* key) -> nlohmann::json {
            const auto it = cfg.given.find(key);
            return it == cfg.given.end() ? nlohmann::json(nullptr) : nlohmann::json(it->second);
        };

        const nlohmann::json report = {
            {"protocol", "bigcherry.native-fleet-inspect/v2"},
            {"snapshot", {{"sha256", managed.snapshot().sha256()},
                           {"resolution_root", managed.snapshot().resolution_root()}}},
            {"engine", {{"abi_version", RAD_ABI_VERSION},
                         {"model", cfg.model},
                         {"radiance_home", cfg.radiance_home},
                         {"radiance_home_source", source("--radiance-home")},
                         {"kernels", cfg.kernel_hierarchy},
                         {"kernels_source", source("--kernels")},
                         {"listener", {{"host", cfg.host}, {"port", cfg.port}}},
                         {"tp", cfg.tp},
                         {"cards", fleet.cards},
                         {"rank_devices", cfg.rank_devices},
                         {"max_context", cfg.max_ctx},
                         {"max_sequences", cfg.max_seqs},
                         // The tuning options the authored YAML states, as the strict reader accepted them (null: absent).
                         {"tuning", {{"p2p", source("--p2p")},
                                      {"pipelined_prefill", source("--pipelined-prefill")},
                                      {"pipelined_prefill_min_rows",
                                       cfg.given.contains("--pipelined-prefill-min-rows")
                                           ? nlohmann::json(cfg.pipelined_prefill_min_rows) : nlohmann::json(nullptr)},
                                      {"prefill_governor", source("--prefill-governor")},
                                      {"tp_ownership", source("--tp-ownership")},
                                      {"tune_pin", source("--tune-pin")}}},
                         {"settings", {{"max_batched_tokens", cfg.max_tok},
                                       {"checkpoint_interval", cfg.checkpoint_interval},
                                       {"placement", cfg.placement},
                                       {"ngram_placement", rad::table_placement_name(cfg.ngram_placement)},
                                       {"kv_dtype", cfg.kv_cache_dtype},
                                       {"speculative_tokens", cfg.n_spec},
                                       {"rope_scaling", cfg.rope_scaling},
                                       {"prefix_resume", cfg.prefix_exact ? "exact" : "nearest"},
                                       {"prefix_cache", cfg.prefix_cache},
                                       {"prefill_interleave", cfg.prefill_interleave_set && cfg.prefill_interleave >= 0
                                           ? nlohmann::json(cfg.prefill_interleave) : nlohmann::json(nullptr)}}}}}
        };
        std::cout << report.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "radiance: " << command << ": " << error.what() << '\n';
        return 2;
    }
#else
    std::cerr << "radiance: " << command << " is supported only on Linux\n";
    return 2;
#endif
}

}  // namespace bc::serving
