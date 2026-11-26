#include "manifest_writer.h"
#include "test_config_fixture.h"
#include "test_manifest_fixture.h"
#include "test_rad_fixture.h"
#include "plan_cache.h"
#include "rank_demand.h"
#include "sha256.h"
#include "v2/manifest.h"
#include "v2/p1_producer.h"

#include <cstdlib>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>

#include <fcntl.h>
#include <linux/memfd.h>
#include <sys/mman.h>

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
    const auto native_view = wire::native_config_view(read.value());
    check(read.value().api_version == "bigcherry/v3" && read.value().metadata.contract_version == 3,
          "the real writer emitted another contract edition");
    check(native_view && native_view->yaml_utf8 == yaml &&
              native_view->identity.sha256 == snapshot.sha256() &&
              native_view->identity.resolution_root == root.string(),
          "role-neutral native snapshot view is unavailable for this manifest state");
    check(native.yaml_utf8 == yaml && native.sha256 == snapshot.sha256() &&
              native.resolution_root == root.string(),
          "the real writer changed the admitted native YAML bytes, digest or root");
    const auto& record = manifest.document.at("provenance").at("/spec/native_config");
    check(record.at("origin") == "detected" && record.at("class") == "fact" &&
              record.at("confidence") == "exact" && record.at("source") == "admitted native fleet YAML",
          "the real writer lost native snapshot provenance");
}

void run_engine_consumer(const fs::path& consumer, const fs::path& root,
                         const sv::BuiltManifest& measuring, const sv::BuiltManifest& serving) {
    const fs::path measuring_path = root / "measuring.v3.json";
    const fs::path serving_path = root / "serving.v3.json";
    for (const auto& [path, bytes] : {std::pair{measuring_path, measuring.text},
                                      std::pair{serving_path, serving.text}}) {
        std::ofstream out(path, std::ios::binary);
        out.exceptions(std::ios::failbit | std::ios::badbit);
        out << bytes;
        out.close();
    }
    const pid_t child = ::fork();
    if (child < 0) throw std::runtime_error("cannot start v3 engine consumer");
    if (child == 0) {
        if (::chdir(root.c_str()) != 0) _exit(126);
        ::execl(consumer.c_str(), consumer.c_str(), measuring_path.c_str(), serving_path.c_str(), nullptr);
        _exit(127);
    }
    int status = 0;
    pid_t waited;
    do {
        waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    check(waited == child, "could not reap v3 engine consumer");
    check(WIFEXITED(status) && WEXITSTATUS(status) == 0, "v3 engine consumer rejected writer output");
}
} // namespace

int main(int argc, char** argv) {
    try {
        check(argc == 1 || argc == 2, "expected an optional engine consumer executable");
        TempDirectory temporary;
        const fs::path model = temporary.path / "tiny.rad";
        const fs::path runtime_home = temporary.path / "runtime-home";
        fs::create_directories(runtime_home);
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
        options.model_root = temporary.path;
        options.sets = {"memory.host_pool_mib=9216",
                        "engine.model=" + Json(model.string()).dump(),
                        "engine.radiance_home=" + Json(runtime_home.string()).dump()};
        sv::fixture::use(options, "flashnext-r9700");   // test_config_fixture.h: the preset named after the shipped YAML
        sv::ManifestInputs inputs;
        inputs.resolved = sv::resolve(options, facts);
        inputs.profile = "production";
        inputs.instance_id = "11111111-1111-4111-8111-111111111111";
        inputs.producer_sha256 = std::string(64, 'a');
        inputs.s10.power_profile = "stock";
        inputs.s10.now_unix_s = 1800000000;
        inputs.s10.calibration_records = Json::array();
        check_snapshot_required(inputs);

        const Json& values = inputs.resolved.values;
        const Json& engine_values = values.at("engine");
        const int64_t tp = engine_values.at("tp").get<int64_t>();
        const int64_t pool_total = inputs.resolved.typed.memory.host_pool_mib * tp;
        std::string cards = "cards: [";
        for (int64_t rank = 0; rank < tp; ++rank) {
            if (rank) cards += ", ";
            cards += std::to_string(rank);
        }
        cards += "]";
        const std::string yaml =
            "# café\r\nserver: {host: 127.0.0.1, port: " +
            std::to_string(engine_values.at("port").get<int64_t>()) +
            ", radiance-home: " + Json(runtime_home.string()).dump() +
            ", kernels: " + Json(engine_values.at("kernels").get<std::string>()).dump() +
            "}\r\n" + cards + "\r\nmodels:\r\n  tiny:\r\n    model: " + Json(model.string()).dump() +
            "\r\n    tp: " + std::to_string(tp) +
            "\r\n    max-num-seqs: " + std::to_string(inputs.resolved.typed.service.max_sessions) +
            "\r\n    max-model-len: " + std::to_string(inputs.resolved.typed.service.max_context) +
            "\r\n    host-pool-mib: " + std::to_string(pool_total) +
            "\r\n    mixed-step-target-ms: 17\r\n";
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
        native_plan["vram"]["cards"][0]["free"] = 10;
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
        // RUM05 B: the native YAML's omitted ngram-placement ("auto")
        // becomes an explicit measured tier. The exact rank receipt, not
        // another resolver default, must own RAM/DISK/VRAM selection.
        {
            const std::vector<std::string> bdfs = {"0000:09:00.0"};
            Json measured_ram = rank_demand;
            measured_ram["ngram_placement"] = "ram";
            auto ram = sv::decode_engine_rank_demand(measured_ram, bdfs, "ram fixture");
            check(sv::admitted_ngram_placement(ram, true, "auto") == "ram",
                  "native auto did not seal the measured pinned-RAM placement");
            Json measured_disk = rank_demand;
            measured_disk["ngram_placement"] = "disk";
            auto disk = sv::decode_engine_rank_demand(measured_disk, bdfs, "disk fixture");
            check(sv::admitted_ngram_placement(disk, false, "auto") == "disk",
                  "native auto did not seal the measured disk fallback");
            check(sv::admitted_ngram_placement(disk, false, "disk") == "disk",
                  "explicit disk override was rewritten");
            bool refused_missing = false, refused_mismatch = false, refused_invalid = false;
            try {
                const auto old = sv::decode_engine_rank_demand(rank_demand, bdfs, "old receipt");
                (void)sv::admitted_ngram_placement(old, false, "auto");
            } catch (const std::runtime_error& e) {
                refused_missing = std::string(e.what()).find("remeasure") != std::string::npos;
            }
            try {
                (void)sv::admitted_ngram_placement(ram, false, "auto");
            } catch (const std::runtime_error& e) {
                refused_mismatch = std::string(e.what()).find("pinned-table") != std::string::npos;
            }
            measured_disk["ngram_placement"] = "auto";
            try {
                (void)sv::decode_engine_rank_demand(measured_disk, bdfs, "invalid receipt");
            } catch (const std::runtime_error& e) {
                refused_invalid = std::string(e.what()).find("ngram_placement") != std::string::npos;
            }
            check(refused_missing && refused_mismatch && refused_invalid,
                  "an absent, contradictory or unresolved n-gram receipt was admitted");
        }
        const std::string model_identity = sv::container_identity(model);
        const std::string engine_identity = sv::engine_identity(binary, runtime_home);
        const Json native_key = sv::plan_key_inputs(inputs.resolved.values, facts, model_identity,
                                                     engine_identity, "", &*inputs.native_config);
        { // A native key binds the YAML digest, which covers every tuning option the YAML states: tuning values add
          // nothing to it, and a changed YAML (hence a changed tuning value) changes the digest and so the key.
            Json tuned = inputs.resolved.values;
            tuned["tuning"] = {{"tp_ownership", "kv=1,1,2;ffn=5632,5632,6144;lk=5,5,6;lv=15,15,18"}};
            const Json tuned_key = sv::plan_key_inputs(tuned, facts, model_identity, engine_identity, "",
                                                       &*inputs.native_config);
            // Native YAML digest alone owns the authored choice. The P1
            // engine result may change from auto to ram/disk without making
            // the sealed owner attempt look like another static configuration.
            Json chosen = inputs.resolved.values;
            chosen["engine"]["ngram_placement"] = "auto";
            const auto auto_key = sv::plan_key_inputs(chosen, facts, model_identity, engine_identity, "",
                                                      &*inputs.native_config);
            chosen["engine"]["ngram_placement"] = "ram";
            const auto ram_key = sv::plan_key_inputs(chosen, facts, model_identity, engine_identity, "",
                                                     &*inputs.native_config);
            chosen["engine"]["ngram_placement"] = "disk";
            const auto disk_key = sv::plan_key_inputs(chosen, facts, model_identity, engine_identity, "",
                                                      &*inputs.native_config);
            check(auto_key == native_key && ram_key == native_key && disk_key == native_key,
                  "derived native n-gram tier must not change the admitted YAML plan key");
            check(!native_key.contains("tuning") && tuned_key == native_key,
                  "a native plan key must bind tuning through the YAML digest, not a second copy");
            check(native_key.at("native_config").at("sha256") == inputs.native_config->sha256(),
                  "the native plan key does not carry the YAML digest");
            // Two YAMLs that differ only in one tuning value: different digest, so different keys.
            const auto key_for = [&](const std::string& text, const char* name) {
                const fs::path path = temporary.path / name;
                { std::ofstream out(path, std::ios::binary); out << text; }
                const auto snapshot = sv::NativeConfigSnapshot::read(path, temporary.path);
                fs::remove(path);
                return sv::plan_key_inputs(inputs.resolved.values, facts, model_identity, engine_identity, "", &snapshot);
            };
            const std::string with_pin = yaml + "    tune-pin: \"all_reduce:nblocks=4\"\r\n";
            const std::string other_pin = yaml + "    tune-pin: \"all_reduce:nblocks=8\"\r\n";
            check(key_for(with_pin, "pin-a.yaml") != key_for(other_pin, "pin-b.yaml"),
                  "two YAMLs differing only in tune-pin gave the same native plan key");
            check(key_for(with_pin, "pin-c.yaml") == key_for(with_pin, "pin-d.yaml"),
                  "the same YAML bytes gave different native plan keys");
        }
        const fs::path cache_dir = temporary.path / "plan-cache";
        const sv::OwnerMeasurementEvidence owner{"v3-writer-attempt", sv::sha256_hex(measuring.text)};
        const auto stored = sv::plan_cache_store_owned(cache_dir, native_key, native_plan, rank_demand,
                                                       inputs.resolved.values,
                                                       sv::HostResident{1536, 1024}, owner);
        const auto cached = sv::plan_cache_lookup(cache_dir, native_key);
        check(cached && cached->key == stored.key && cached->owner &&
                  cached->owner->attempt_id == owner.attempt_id &&
                  cached->owner->measuring_manifest_sha256 == owner.measuring_manifest_sha256,
              "the measured native plan did not round-trip through its owner cache receipt");
        inputs.measured->plan = *cached;
        inputs.measured->key_inputs = native_key;
        inputs.engine_binary = binary;
        inputs.plan_engine_args = "";
        const Json native_sources = {
            {"engine_plan", cached->plan_json}, {"rank_demand", cached->rank_demand},
            {"engine_binary", binary.string()}, {"helper_binary", ""},
            {"retention_target_tokens", 0},
            {"margin", {{"key", "host_ledger_margin.mib"}, {"source", "cache"},
                        {"evidence", "v3 writer fixture measured room"}, {"mib", 1}}},
            {"run_id", "bc_serving_v3_writer_test"}};
        inputs.s10.p1_record = wire::produce_p1(inputs, facts, temporary.path, native_sources);
        check(inputs.s10.p1_record.at("kind") == "bc_serve_v2_measured_inputs",
              "real P1 producer did not emit the expected measured input record");
        const Json measured_key = inputs.measured->key_inputs;
        inputs.measured->key_inputs["native_config"]["sha256"] = std::string(64, 'f');
        try {
            (void)sv::build_manifest(inputs);
        } catch (const std::runtime_error& error) {
            check(std::string_view(error.what()).find("key differs from the resolved native YAML") !=
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
            check(std::string_view(error.what()).find("key differs from the resolved native YAML") !=
                      std::string_view::npos,
                  error.what());
            inputs.measured->key_inputs = measured_key;
        }
        check(inputs.measured->key_inputs == measured_key,
              "the real writer accepted a measured plan from a different execution root");
        const auto refuses_rekeyed_input = [&](Json changed, const char* why) {
            inputs.measured->key_inputs = std::move(changed);
            inputs.measured->plan.key = sv::plan_key(inputs.measured->key_inputs);
            try {
                (void)sv::build_manifest(inputs);
            } catch (const std::runtime_error& error) {
                check(std::string_view(error.what()).find("key differs from the resolved native YAML") !=
                          std::string_view::npos,
                      error.what());
                inputs.measured->key_inputs = measured_key;
                inputs.measured->plan.key = cached->key;
                return;
            }
            throw std::runtime_error(why);
        };
        Json wrong_model = measured_key;
        wrong_model["model"] = std::string(64, 'c');
        refuses_rekeyed_input(wrong_model, "writer accepted a plan with a re-keyed model identity");
        Json wrong_build = measured_key;
        wrong_build["engine_build"] = std::string(64, 'd');
        refuses_rekeyed_input(wrong_build, "writer accepted a plan with a re-keyed engine build identity");
        Json wrong_args = measured_key;
        wrong_args["engine_args"] = "--kv-resident 1234";
        refuses_rekeyed_input(wrong_args, "writer accepted a plan with re-keyed engine arguments");
        Json wrong_devices = measured_key;
        wrong_devices["native_config"]["rank_devices"][0]["bdf"] = "0000:ff:00.0";
        refuses_rekeyed_input(wrong_devices, "writer accepted a plan keyed to another ordered GPU identity");
        const auto serving = sv::build_manifest(inputs);
        check_snapshot(serving, wire::PlanUse::serving, *inputs.native_config, yaml, temporary.path);
        // The measured engine tier is a derived leaf in the final strict
        // v3 wire contract (not a second native/YAML policy value).
        inputs.resolved.provenance["engine.ngram_placement"] =
            Json{{"origin", "derived"}, {"source", "rad0011 measured rank n-gram tier"},
                 {"lifecycle", "startup-only"}};
        inputs.resolved.manifest["provenance"] = inputs.resolved.provenance;
        const auto measured_tier = sv::build_manifest(inputs);
        check_snapshot(measured_tier, wire::PlanUse::serving, *inputs.native_config, yaml, temporary.path);
        check(measured_tier.document.at("provenance")
                  .at("/spec/processes/engine/execution/ngram_placement")
                  .at("origin") == "derived" &&
                  measured_tier.document.at("spec").at("processes").at("engine")
                      .at("execution").at("ngram_placement") == "disk",
              "strict v3 writer lost the explicit measured tier/provenance");
        check(serving.document.at("spec").at("plan").at("engine_memory").at("expected") == cached->plan_json &&
                  serving.document.at("spec").at("plan").at("engine_memory").at("model_identity") == model_identity &&
                  serving.document.at("spec").at("plan").at("engine_memory").at("engine_identity") == engine_identity,
              "the final serving manifest did not carry the owner-cached plan and real build identities");
        if (argc == 2) run_engine_consumer(argv[1], temporary.path, measuring, serving);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
