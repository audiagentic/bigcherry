#include "native_fleet_config.h"
#include "bc_serving_manifest.h"
#include "bc_plan_check.h"
#include "bc_plan_memory.h"
#include "sha256.h"
#include "v2/manifest.h"
#include "v2/s10_resolver.h"
#include "serving_codec.h"

#include <fstream>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

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

bool serving_entry(const std::string& path, const std::vector<std::string>& run_args = {}) {
    std::vector<std::string> storage = {"radiance", "--manifest", path, "--role", "engine"};
    storage.insert(storage.end(), run_args.begin(), run_args.end());
    std::vector<char*> argv;
    for (std::string& arg : storage) argv.push_back(arg.data());
    int argc = static_cast<int>(argv.size());
    argv.push_back(nullptr);
    char** raw_argv = argv.data();
    if (!bc_serving_engine_argv(&argc, &raw_argv)) return false;
    check(bc_plan_memory_take_flag(&argc, &raw_argv) == 0 && argc == 1 + static_cast<int>(run_args.size()),
          "native serving left unexpected flags after rad0011 extraction");
    return true;
}

void reject_direct_v3_serving_start(const std::string& path) {
    check(!serving_entry(path), "direct engine startup accepted an unsealed v3 serving manifest");
}

struct Descriptor {
    int fd = ::memfd_create("v3-fixture", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    ~Descriptor() { if (fd >= 0) ::close(fd); }
    void write_bytes(const std::string& bytes) const {
        check(fd >= 3 && !bytes.empty() &&
                  ::write(fd, bytes.data(), bytes.size()) == static_cast<ssize_t>(bytes.size()),
              "cannot prepare exact manifest bytes in the fixture descriptor");
    }
    void seal() const {
        constexpr int seals = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
        check(::fcntl(fd, F_ADD_SEALS, seals) == 0, "cannot seal the fixture descriptor");
    }
    std::string path() const { return "/proc/self/fd/" + std::to_string(fd); }
};

std::string with_yaml_host_pool(wire::Manifest manifest, std::optional<int64_t> total_mib) {
    auto& native = manifest.spec.native_config;
    auto& yaml = native.yaml_utf8;
    const auto begin = yaml.find("\r\n    host-pool-mib:");
    check(begin != std::string::npos, "fixture YAML lost its explicit host-pool");
    const auto end = yaml.find("\r\n", begin + 2);
    check(end != std::string::npos, "fixture YAML lost its line ending");
    yaml.replace(begin, end - begin,
                 total_mib ? "\r\n    host-pool-mib: " + std::to_string(*total_mib) : "");
    native.sha256 = bc::serving::sha256_hex(yaml);
    if (manifest.spec.plan.engine_memory)
        manifest.spec.plan.engine_memory->cache_key = wire::plan_cache_key(manifest);
    const auto encoded = wire::write_s10(manifest);
    if (!encoded) throw std::runtime_error(wire::render_human("host pool fixture", encoded.error()));
    return encoded.value();
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
    check(bc_serving_take_native_config(&cfg) == 1,
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
    const auto yaml_config = replay(manifest).fleet().models.front().cfg;
    const auto given = yaml_config.given.find("--host-pool-mib");
    const std::string pool_source = given == yaml_config.given.end() ?
        "manifest spec.resources.host.pool_mib" : given->second;
    check(cfg.given.at("--host-pool-mib") == pool_source, "measuring Config lost host-pool provenance");
}

void check_native_serving_main(const std::string& path, const wire::Manifest& manifest) {
    const auto native = wire::native_config_view(manifest);
    check(native.has_value(), "serving manifest has no native YAML view");
    std::filesystem::current_path(native->identity.resolution_root);
    std::ifstream authored(path, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(authored)), std::istreambuf_iterator<char>());
    const Descriptor descriptor;
    descriptor.write_bytes(bytes);
    const std::string sealed_path = descriptor.path();
    check(!serving_entry(sealed_path), "v3 serving accepted an unsealed inherited descriptor");
    descriptor.seal();
    check(!serving_entry(sealed_path, {"--plan-memory", "plan.json", "--plan-memory-ranks", "ranks.json",
                                       "--kld-ref", "ref"}),
          "sealed v3 manifest accepted mixed measuring and quality flags");
    // Replace the mutable artifact after sealing the exact admitted bytes.
    {
        std::ofstream changed(path, std::ios::binary | std::ios::trunc);
        check(static_cast<bool>(changed), "cannot mutate serving artifact path for replay fixture");
        changed << "{}\n";
    }
    const auto root = std::filesystem::path(native->identity.resolution_root);
    std::filesystem::current_path(root.parent_path());
    rad::Config wrong_root;
    wire::EnginePlan expected;
    int64_t floor_ppm = 0;
    check(serving_entry(sealed_path) && bc_serving_take_native_config(&wrong_root) < 0 &&
              !bc_serving_expected_plan(&expected, &floor_ppm) && bc_serving_engine_bdfs().empty(),
          "wrong-root native start retained measured plan or device admission");
    std::filesystem::current_path(root);
    rad::Config cfg;
    check(serving_entry(sealed_path) && bc_serving_take_native_config(&cfg) == 1,
          "real-engine entry did not prepare the sealed measured serving Config");
    const auto& spec = manifest.spec;
    check(cfg.mixed_step_target_ms == 17 &&
              cfg.given.at("--mixed-step-target-ms") == "models.tiny.mixed-step-target-ms",
          "serving Config lost its upstream-only YAML value or provenance");
    check(cfg.given.at("--host-pool-mib") == "models.tiny.host-pool-mib",
          "serving Config lost an explicit YAML host-pool provenance");
    check(cfg.devices.size() == static_cast<size_t>(cfg.tp) && cfg.tp == spec.processes.engine.scheduler.tp &&
              cfg.rank_devices == spec.processes.engine.device.rank_devices.value_or(std::vector<std::string>{}),
          "serving Config device ordinals or admitted rank-device order changed");
    check(cfg.model == spec.intent.model && cfg.radiance_home == spec.processes.engine.runtime.radiance_home &&
              cfg.kernel_hierarchy == spec.processes.engine.runtime.kernels &&
              cfg.port == spec.processes.engine.server.port &&
              cfg.max_ctx == spec.intent.service.max_context_tokens &&
              cfg.max_seqs == spec.intent.service.max_sessions &&
              cfg.host_pool_mib == spec.resources.host.pool_mib,
          "serving Config differs from its independent measured manifest fields");
    check(bc_serving_expected_plan(&expected, &floor_ppm),
          "native serving did not retain the strict measured-plan admission");
    check(bc_plan::mode() == bc_plan::Mode::check, "rad0011 did not arm live plan admission for native serving");
    const auto admitted_view = wire::engine_view(manifest);
    check(admitted_view.ok(), "serving fixture has no strict measured engine view");
    check(wire::encode(expected) == wire::encode(spec.plan.engine_memory->expected) &&
              floor_ppm == admitted_view.value().kv_backing_floor_ppm &&
              bc_serving_engine_bdfs() == spec.processes.engine.device.gpus,
          "native serving changed the expected plan, KV floor, or ordered BDFs");

    const auto quality_config = [&](const std::vector<std::string>& flags) {
        rad::Config quality;
        check(serving_entry(sealed_path, flags) && bc_serving_take_native_config(&quality) == 1 &&
                  quality.kld() && bc_plan::mode() == bc_plan::Mode::none &&
                  !bc_serving_expected_plan(&expected, &floor_ppm),
              "native quality did not retain its offline plan role");
        check(quality.given.at("--mixed-step-target-ms") == cfg.given.at("--mixed-step-target-ms") &&
                  quality.mixed_step_target_ms == cfg.mixed_step_target_ms &&
                  quality.devices == cfg.devices && quality.rank_devices == cfg.rank_devices &&
                  quality.host_pool_mib == cfg.host_pool_mib,
              "native quality overlay changed YAML provenance, ordered devices or per-rank pool");
        return quality;
    };
    const auto recording = quality_config({"--kld-record", "/ref", "--kld-corpus", "/corpus.jsonl"});
    check(recording.kld_record == "/ref" && recording.kld_corpus == "/corpus.jsonl",
          "native quality lost reference recording paths");
    const auto score = quality_config({"--kld-ref", "/ref", "--kld-out", "/result.json"});
    check(score.kld_ref == "/ref" && score.kld_out == "/result.json" && score.kld_record.empty(),
          "native quality lost score paths");
    rad::Config invalid;
    check(serving_entry(sealed_path, {"--kld-out", "/result.json"}) &&
              bc_serving_take_native_config(&invalid) < 0 && !bc_serving_expected_plan(&expected, &floor_ppm),
          "an invalid KL request could fall through to serving");

    const Descriptor without_pool;
    without_pool.write_bytes(with_yaml_host_pool(manifest, std::nullopt));
    without_pool.seal();
    check(spec.resources.host.pool_mib > 0, "fixture needs a nonzero admitted per-rank pool");
    for (const auto& flags : {std::vector<std::string>{},
                              std::vector<std::string>{"--kld-ref", "/ref", "--kld-out", "/result.json"}}) {
        rad::Config derived;
        check(serving_entry(without_pool.path(), flags) && bc_serving_take_native_config(&derived) == 1 &&
                  derived.host_pool_mib == spec.resources.host.pool_mib &&
                  derived.given.at("--host-pool-mib") == "manifest spec.resources.host.pool_mib" &&
                  derived.kld() == !flags.empty(),
              "omitted YAML host pool did not retain the admitted per-rank value and provenance");
    }

    const Descriptor explicit_zero;
    explicit_zero.write_bytes(with_yaml_host_pool(manifest, 0));
    explicit_zero.seal();
    check(serving_entry(explicit_zero.path()) && bc_serving_take_native_config(&invalid) < 0 &&
              !bc_serving_expected_plan(&expected, &floor_ppm) && bc_serving_engine_bdfs().empty(),
          "explicit mismatched YAML host pool must refuse and clear measured admission");
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
        const Descriptor measuring_without_pool;
        measuring_without_pool.write_bytes(with_yaml_host_pool(measuring, std::nullopt));
        measuring_without_pool.seal();
        check_native_measuring_main(measuring_without_pool.path(),
                                    read(measuring_without_pool.path(), wire::PlanUse::measuring));
        reject_wrong_use(argv[1], wire::PlanUse::serving);
        reject_wrong_use(argv[2], wire::PlanUse::measuring);
        reject_direct_v3_serving_start(argv[2]);
        check_native_serving_main(argv[2], serving);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
