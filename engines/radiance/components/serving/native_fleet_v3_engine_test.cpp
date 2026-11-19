#include "native_fleet_config.h"
#include "bc_serving_manifest.h"
#include "bc_plan_memory.h"
#include "v2/manifest.h"

#include <fstream>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "rad_abi.h"

namespace {
namespace wire = bc::serving::v2;
using Json = nlohmann::json;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

wire::Manifest read(const std::string& path, wire::PlanUse use) {
    const auto result = wire::load_manifest(path, use);
    if (!result) throw std::runtime_error(wire::render_human("v3 engine consumer", result.error()));
    return result.value();
}

bc::serving::NativeFleetConfig replay(const wire::Manifest& manifest) {
    const auto native = wire::native_config_view(manifest);
    check(native.has_value(), "admitted v3 manifest omitted its native config");
    return bc::serving::NativeFleetConfig::replay(*native);
}

void reject_wrong_use(const std::string& path, wire::PlanUse use) {
    const auto result = wire::load_manifest(path, use);
    check(!result && result.error().code == wire::ConfigErrorCode::plan_mismatch,
          "v3 reader accepted a manifest for the wrong plan role");
}

void check_native_measuring_main(const std::string& path, const wire::Manifest& manifest) {
    const auto native = wire::native_config_view(manifest);
    check(native.has_value(), "measuring manifest has no native YAML view");
    std::filesystem::current_path(native->identity.resolution_root);
    std::vector<std::string> storage = {"radiance", "--manifest", path, "--role", "engine",
                                        "--plan-memory", "unused-plan.json", "--plan-memory-ranks",
                                        "unused-ranks.json"};
    std::vector<char*> argv;
    for (std::string& arg : storage) argv.push_back(arg.data());
    int argc = static_cast<int>(argv.size());
    argv.push_back(nullptr);
    char** raw_argv = argv.data();
    check(bc_serving_engine_argv(&argc, &raw_argv), "rad0009 rejected the native measuring manifest");
    check(bc_plan_memory_take_flag(&argc, &raw_argv) == 0,
          "rad0011 rejected the native measuring output flags");
    check(argc == 1, "native measuring argv retained flags that config_parse would consume");
    rad::Config cfg;
    check(bc_serving_take_native_measuring_config(&cfg) == 1,
          "rad0009 did not replay the native Config after rad0011 captured its baseline");
    const auto& spec = manifest.spec;
    check(cfg.mixed_step_target_ms == 17 &&
              cfg.given.at("--mixed-step-target-ms") == "models.tiny.mixed-step-target-ms",
          "native Config value or YAML provenance was lost before Engine::start");
    check(cfg.devices.size() == static_cast<size_t>(cfg.tp) && cfg.tp == spec.processes.engine.scheduler.tp &&
              cfg.rank_devices == spec.processes.engine.device.rank_devices.value_or(std::vector<std::string>{}),
          "native device ordinals or the admitted rank-device list changed");
    check(cfg.model == spec.intent.model && cfg.radiance_home == spec.processes.engine.runtime.radiance_home &&
              cfg.kernel_hierarchy == spec.processes.engine.runtime.kernels &&
              cfg.port == spec.processes.engine.server.port &&
              cfg.max_ctx == spec.intent.service.max_context_tokens &&
              cfg.max_seqs == spec.intent.service.max_sessions &&
              cfg.host_pool_mib == spec.resources.host.pool_mib,
          "native measuring Config differs from the independent admitted engine settings");
}
} // namespace

int main(int argc, char** argv) {
    static_assert(wire::kContractVersion == 3, "consumer must link only the v3 generated reader");
    static_assert(RAD_ABI_VERSION == 44u, "consumer must match the ABI44 candidate engine");
    try {
        check(argc == 3, "expected measuring and serving v3 manifests");
        const auto measuring = read(argv[1], wire::PlanUse::measuring);
        const auto serving = read(argv[2], wire::PlanUse::serving);
        const auto measuring_native = wire::native_config_view(measuring);
        const auto serving_native = wire::native_config_view(serving);
        check(measuring_native && serving_native &&
                  measuring_native->identity.sha256 == serving_native->identity.sha256 &&
                  measuring_native->identity.resolution_root == serving_native->identity.resolution_root,
              "measuring and serving manifests do not share the admitted native snapshot");

        for (const auto* manifest : {&measuring, &serving}) {
            const auto native = replay(*manifest);
            check(native.fleet().models.size() == 1, "replayed fleet did not contain one model");
            const auto& model = native.fleet().models.front();
            check(model.cfg.mixed_step_target_ms == 17 &&
                      model.cfg.given.at("--mixed-step-target-ms") == "models.tiny.mixed-step-target-ms",
                  "upstream-only YAML setting or its provenance was not replayed");
            check(native.snapshot().yaml_utf8().find("# café\r\n") == 0,
                  "native YAML bytes were not preserved exactly");
        }
        check_native_measuring_main(argv[1], measuring);
        reject_wrong_use(argv[1], wire::PlanUse::serving);
        reject_wrong_use(argv[2], wire::PlanUse::measuring);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
