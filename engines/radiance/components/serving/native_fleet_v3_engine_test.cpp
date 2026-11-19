#include "native_fleet_config.h"
#include "v2/manifest.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

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
        reject_wrong_use(argv[1], wire::PlanUse::serving);
        reject_wrong_use(argv[2], wire::PlanUse::measuring);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
