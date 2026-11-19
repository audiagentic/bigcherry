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
    // Native provenance includes inherited and aliased server options. Checking
    // the option, not its resolved value, also permits environment references.
    if (fleet_.models.front().cfg.given.contains("--api-key"))
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
            {"protocol", "bigcherry.native-fleet-inspect/v1"},
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
                         {"max_sequences", cfg.max_seqs}}}
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
