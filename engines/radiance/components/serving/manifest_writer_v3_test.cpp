#include "manifest_writer.h"
#include "test_manifest_fixture.h"
#include "test_rad_fixture.h"
#include "v2/manifest.h"
#include "v2/p1_producer.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
namespace fs = std::filesystem;
namespace sv = bc::serving;
namespace wire = bc::serving::v2;
using sv::Json;

struct TempDirectory {
    fs::path path;

    TempDirectory() {
        std::string name = (fs::temp_directory_path() / "bc-v3-writer-XXXXXX").string();
        if (!::mkdtemp(name.data()))
            throw std::runtime_error("cannot create writer fixture directory");
        path = fs::canonical(name);
    }

    ~TempDirectory() {
        std::error_code error;
        fs::remove_all(path, error);
    }
};

void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void check_snapshot_required(const sv::ManifestInputs& inputs) {
    try {
        (void)sv::build_manifest(inputs);
    } catch (const std::runtime_error& error) {
        check(std::string_view(error.what()).find("requires an admitted native fleet YAML snapshot") !=
                  std::string_view::npos,
              error.what());
        return;
    }
    throw std::runtime_error("the v3 writer accepted an absent native snapshot");
}

void check_snapshot(const sv::BuiltManifest& manifest, wire::PlanUse use,
                    const sv::NativeConfigSnapshot& snapshot, const std::string& yaml, const fs::path& root) {
    const auto read = wire::read_manifest(manifest.text, "v3 writer fixture", use);
    if (!read)
        throw std::runtime_error(wire::render_human("v3 writer test", read.error()));
    const auto& native = read.value().spec.native_config;
    check(read.value().api_version == "bigcherry/v3" && read.value().metadata.contract_version == 3,
          "the real writer emitted another contract edition");
    check(native.yaml_utf8 == yaml && native.sha256 == snapshot.sha256() &&
              native.resolution_root == root.string(),
          "the real writer changed the admitted native YAML bytes, digest or root");
    const auto& record = manifest.document.at("provenance").at("/spec/native_config");
    check(record.at("origin") == "detected" && record.at("class") == "fact" &&
              record.at("confidence") == "exact" && record.at("source") == "admitted native fleet YAML",
          "the real writer lost native snapshot provenance");
}
} // namespace

int main(int argc, char** argv) {
    try {
        check(argc == 2, "expected the engine-only named config fixture source");
        TempDirectory temporary;
        const fs::path model = temporary.path / "tiny.rad";
        bc::test::write_rad(model, 0, 0, 128, 128, {});
        const Json facts = Json::parse(R"json({
            "source": "v3 writer fixture",
            "host": {
                "mem_total_mib": 131072,
                "mem_available_mib": 92160,
                "swap_total_mib": 8192,
                "shm_available_mib": 4096,
                "memory_cgroups": [],
                "amdgpu_pin": {"loaded": true, "lifted": false, "cap_mib": 127488},
                "memlock": {"limit_mib": 8, "cap_ipc_lock": false},
                "cpu_count": 32
            },
            "gpus": [{
                "bdf": "0000:09:00.0",
                "arch": "gfx1201",
                "hip_ordinal": 0,
                "vram_total_mib": 32624,
                "vram_available_mib": 32496
            }]
        })json");
        sv::Options options;
        options.configs = {argv[1]};
        options.model_root = temporary.path;
        options.sets = {"layout.name=\"flashnext-r9700\"", "memory.host_pool_mib=9216",
                        "engine.model=" + Json(model.string()).dump(),
                        "engine.radiance_home=" + Json(temporary.path.string()).dump()};
        sv::ManifestInputs inputs;
        inputs.resolved = sv::resolve(options, facts);
        inputs.profile = "production";
        inputs.instance_id = "11111111-1111-4111-8111-111111111111";
        inputs.producer_sha256 = std::string(64, 'a');
        inputs.s10.power_profile = "stock";
        inputs.s10.now_unix_s = 1800000000;
        inputs.s10.calibration_records = Json::array();
        check_snapshot_required(inputs);

        const std::string yaml = "# café\r\ncards: [0]\r\nmodels:\r\n  tiny: {model: tiny.rad}\r\n";
        const fs::path authored = temporary.path / "fleet.yaml";
        {
            std::ofstream out(authored, std::ios::binary);
            out.exceptions(std::ios::failbit | std::ios::badbit);
            out << yaml;
            out.close();
        }
        inputs.native_config = sv::NativeConfigSnapshot::read(authored, temporary.path);
        fs::remove(authored);

        const auto measuring = sv::build_manifest(inputs);
        check_snapshot(measuring, wire::PlanUse::measuring, *inputs.native_config, yaml, temporary.path);
        bc::test::set_measured_manifest_fixture(inputs, measuring, facts);
        // Run the real P1 producer with the same admitted snapshot that the
        // v3 writer will persist. The authored path is already gone here.
        const fs::path binary = temporary.path / "radiance-test-binary";
        {
            std::ofstream out(binary, std::ios::binary);
            out.exceptions(std::ios::failbit | std::ios::badbit);
            out << "matched engine fixture\n";
        }
        const fs::path gtt = temporary.path / "sys/bus/pci/devices/0000:09:00.0/mem_info_gtt_used";
        fs::create_directories(gtt.parent_path());
        {
            std::ofstream out(gtt);
            out.exceptions(std::ios::failbit | std::ios::badbit);
            out << "0\n";
        }
        Json native_plan = inputs.measured->plan.plan_json;
        native_plan["kv_one_session"]["bytes"] = inputs.resolved.typed.service.max_context;
        native_plan["host_pool_mib"] = inputs.resolved.typed.memory.host_pool_mib;
        // The derived pool is the measured host embedding plus the 1 MiB
        // margin, matching this fixture's configured 9216 MiB pool.
        native_plan["host"] = Json::array({{{"name", "embedding"},
                                            {"bytes", 9215LL * 1048576},
                                            {"residence", "pinned"},
                                            {"from", "rad0011 v3 writer fixture"},
                                            {"remedy", ""}}});
        const Json rank_demand = {
            {"schema_version", 1}, {"kind", "radiance_p1_rank_demand"},
            {"gpu", Json::array({{{"bdf", "0000:09:00.0"}, {"required_bytes", 10},
                                  {"fixed_bytes", 1}, {"claimable_bytes", 10},
                                  {"evidence", "rad0011 v3 writer fixture rank split"}}})},
            {"resident_other_bytes", 0}};
        const Json native_sources = {
            {"engine_plan", native_plan}, {"rank_demand", rank_demand},
            {"engine_binary", binary.string()}, {"helper_binary", ""},
            {"retention_target_tokens", 0},
            {"margin", {{"key", "host_ledger_margin.mib"}, {"source", "cache"},
                        {"evidence", "v3 writer fixture measured room"}, {"mib", 1}}},
            {"run_id", "bc_serving_v3_writer_test"}};
        inputs.s10.p1_record = wire::produce_p1(inputs, facts, temporary.path, native_sources);
        check(inputs.s10.p1_record.at("kind") == "bc_serve_v2_measured_inputs",
              "real P1 producer did not emit the expected measured input record");
        inputs.measured->key_inputs["native_config"] = {
            {"sha256", inputs.native_config->sha256()},
            {"resolution_root", inputs.native_config->resolution_root()}};
        const Json measured_key = inputs.measured->key_inputs;
        inputs.measured->key_inputs["native_config"]["sha256"] = std::string(64, 'f');
        try {
            (void)sv::build_manifest(inputs);
        } catch (const std::runtime_error& error) {
            check(std::string_view(error.what()).find("not keyed to this native fleet YAML snapshot") !=
                      std::string_view::npos,
                  error.what());
            inputs.measured->key_inputs = measured_key;
        }
        check(inputs.measured->key_inputs == measured_key,
              "the real writer accepted a measured plan from a different native YAML snapshot");
        inputs.measured->key_inputs["native_config"]["resolution_root"] = temporary.path.parent_path().string();
        try {
            (void)sv::build_manifest(inputs);
        } catch (const std::runtime_error& error) {
            check(std::string_view(error.what()).find("not keyed to this native fleet YAML snapshot") !=
                      std::string_view::npos,
                  error.what());
            inputs.measured->key_inputs = measured_key;
        }
        check(inputs.measured->key_inputs == measured_key,
              "the real writer accepted a measured plan from a different execution root");
        const auto serving = sv::build_manifest(inputs);
        check_snapshot(serving, wire::PlanUse::serving, *inputs.native_config, yaml, temporary.path);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
