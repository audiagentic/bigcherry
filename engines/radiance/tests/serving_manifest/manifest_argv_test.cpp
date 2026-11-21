// Card-free test of the patched engine's manifest admission (core/bc_serving_manifest.cpp from rad0009) on edition 3:
// a serving start takes the owner's sealed manifest descriptor (/proc/self/fd/N) with --role engine, a measuring start takes
// a manifest file with --plan-memory and --plan-memory-ranks, and the engine's own settings come from the admitted native YAML,
// so no registry field is projected to a flag: what is left for radiance's parser is the run flags.
// Built by tools/tests/patch/test_radiance_patch_packages.py against radiance's real option table and a mirror of config_parse's shape rules.
// usage: manifest_argv_test SCRATCH_DIR V3_EXAMPLE_MANIFEST OPTIONS_FILE
#include "bc_plan_check.h"
#include "manifest.h"
#include "bc_serving_manifest.h"
#include "bc_engine_settings.h"
#include "jcs.h"
#include "sha256.h"
#include "serving_codec.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

// radiance's real option table (g_flags in core/config.cpp at 4a61d72), extracted by the Python harness into a file
// of "name<TAB>ARG" lines, ARG "-" for a boolean. A hand-written option list once let a wrong manifest shape pass.
static std::vector<std::string> g_options;
static std::map<std::string, bool> g_takes_arg;

static void load_options(const char* path) {
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        const std::string name = line.substr(0, tab);
        g_options.push_back(name);
        g_takes_arg[name] = line.substr(tab + 1) != "-";
    }
}

namespace rad {
namespace diag {
std::vector<std::string> config_option_names() { return g_options; }
}  // namespace diag
}  // namespace rad

// A mirror of the shape rules of radiance's config_parse (core/config.cpp, 4a61d72): an option that takes an
// argument calls need(), which fails when argv has nothing left and otherwise takes the next token whatever it is;
// a boolean takes none; anything else, such as a stray value, is "unknown option". This is what rejects a value
// given to a boolean option and a bare flag given to an option that needs an argument.
static bool parse_like_radiance(int argc, char** argv, std::map<std::string, std::string>* given) {
    for (int i = 1; i < argc; ++i) {
        const auto it = g_takes_arg.find(argv[i]);
        if (it == g_takes_arg.end()) return false;   // "unknown option"
        if (it->second) {
            if (i + 1 >= argc) return false;          // "needs an argument"
            ++i;
        }
        (*given)[it->first] = "command line";
    }
    return true;
}

static int g_failed = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_failed; } \
    } while (0)

static std::string g_dir;
static std::map<std::string, std::string> g_given;   // what the stub parse recorded as given
static bool g_parsed = false;   // whether the stub of radiance's parser accepts the last run() argv

static std::string write(const std::string& name, const std::string& body) {
    const std::string p = g_dir + "/" + name;
    std::ofstream(p) << body;
    return p;
}

static std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// Runs bc_serving_engine_argv on `args` (argv[0] is added) and returns the result argv joined by space,
// or "ERROR" when it refused.
static std::string run(std::vector<std::string> args) {
    std::vector<std::string> store{"engine"};
    for (auto& a : args) store.push_back(a);
    std::vector<char*> ptrs;
    for (auto& s : store) ptrs.push_back(&s[0]);
    ptrs.push_back(nullptr);
    int argc = (int)store.size();
    char** argv = ptrs.data();
    g_parsed = false;
    g_given.clear();
    if (!bc_serving_engine_argv(&argc, &argv)) return "ERROR";
    g_parsed = parse_like_radiance(argc, argv, &g_given);
    std::string out;
    for (int i = 0; i < argc; ++i) out += (i ? " " : "") + std::string(argv[i]);
    CHECK(argv[argc] == nullptr);
    return out;
}

// main()'s real call order (core/main.cpp, checked by the Python harness): rad0009's manifest reader first, then rad0011's
// plan flags. Returns the argv left for config_parse joined by space, or "ERROR" when the manifest reader refused it, or
// "EXIT <rc>" when the plan flags did.
static std::string run_main_order(std::vector<std::string> args) {
    std::vector<std::string> store{"engine"};
    for (auto& a : args) store.push_back(a);
    std::vector<char*> ptrs;
    for (auto& s : store) ptrs.push_back(&s[0]);
    ptrs.push_back(nullptr);
    int argc = (int)store.size();
    char** argv = ptrs.data();
    if (!bc_serving_engine_argv(&argc, &argv)) return "ERROR";
    if (int rc = bc_plan::take_flags(&argc, &argv)) return "EXIT " + std::to_string(rc);
    std::string out;
    for (int i = 0; i < argc; ++i) out += (i ? " " : "") + std::string(argv[i]);
    return out;
}


// The owner hands the engine its manifest as a fully sealed memfd, named /proc/self/fd/N. Returns that path; the descriptor
// stays open for the process, and can be read as often as a test needs.
static std::string sealed(const std::string& body) {
    const int fd = memfd_create("manifest", MFD_ALLOW_SEALING);
    if (fd < 0 || write(fd, body.data(), body.size()) != (ssize_t)body.size() ||
        fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) != 0) {
        std::fprintf(stderr, "FAIL: cannot make a sealed manifest descriptor\n");
        std::exit(1);
    }
    return "/proc/self/fd/" + std::to_string(fd);
}

// A manifest document with its content digest recomputed after an edit.
static std::string seal_digest(nlohmann::json doc) {
    std::string digest, why;
    doc["metadata"].erase("content_digest");
    CHECK(bc::serving::manifest_content_digest(doc, &digest, &why));
    doc["metadata"]["content_digest"] = digest;
    return doc.dump(2) + "\n";
}

// The measuring start's document: plan.state unmeasured, nothing measured yet (no engine_memory, resource_inputs or
// checkpoint slots), as the owner writes it for the measuring engine.
static nlohmann::json unmeasured_of(nlohmann::json doc) {
    doc["spec"]["plan"]["state"] = "unmeasured";
    doc["spec"]["plan"].erase("engine_memory");
    doc["spec"]["plan"].erase("resource_inputs");
    for (auto it = doc["provenance"].begin(); it != doc["provenance"].end();)
        it = it.key().rfind("/spec/plan/resource_inputs", 0) == 0 ? doc["provenance"].erase(it) : std::next(it);
    doc["spec"]["resources"]["host"].erase("checkpoint_slots");
    doc["provenance"]["/spec/resources"]["inputs"] = nlohmann::json::array({"/spec/facts/host", "/spec/calibration"});
    return doc;
}

int main(int, char** argv) {
    g_dir = argv[1];
    load_options(argv[3]);
    // rad0011 adds --plan-memory FILE and --plan-memory-ranks FILE to radiance's option table (the table the harness
    // extracted is the unpatched pin's)
    for (const char* flag : {"--plan-memory", "--plan-memory-ranks"}) {
        g_options.push_back(flag);
        g_takes_arg[flag] = true;
    }
    CHECK(g_options.size() > 100);
    const std::string example = argv[2];
    const nlohmann::json shape = nlohmann::json::parse(slurp(example));
    CHECK(shape.at("apiVersion") == "bigcherry/v3" && shape.at("spec").contains("native_config"));
    const std::string measured = sealed(slurp(example));

    // A serving start: the strict reader admits the owner's sealed descriptor, and the engine takes its settings from the
    // admitted native YAML, so nothing of the manifest becomes a flag for radiance's parser.
    const std::string got = run({"--manifest", measured, "--role", "engine"});
    CHECK(got == "engine" && g_parsed);
    CHECK(bc_serving_mark_given(&g_given) && g_given.empty());
    CHECK(run({"--manifest", measured, "--role", "engine", "-v"}) == "engine -v");
    CHECK(run({"--manifest", measured, "--role", "engine"}) == got);   // deterministic
    {
        bc::serving::v2::EnginePlan plan;
        int64_t floor_ppm = -1;
        CHECK(bc_serving_expected_plan(&plan, &floor_ppm));   // a serving start arms the plan check with the manifest's own plan
        CHECK(bc_serving_engine_bdfs() ==
              shape.at("spec").at("processes").at("engine").at("device").at("gpus").get<std::vector<std::string>>());
    }

    // The arithmetic of the host-pool budget (per rank in the manifest, the aggregate in Radiance 2.1's own parser).
    int64_t aggregate_pool_mib = 0;
    CHECK(bc_serving_manifest_host_pool_total(2048, 3, &aggregate_pool_mib));
    CHECK(aggregate_pool_mib == 6144);
    CHECK(!bc_serving_manifest_host_pool_total(std::numeric_limits<int64_t>::max(), 2, &aggregate_pool_mib));
    CHECK(!bc_serving_manifest_host_pool_total(2048, 0, &aggregate_pool_mib));

    // A serving start takes only the owner's descriptor and --role engine: a path, a missing role, a wrong role, an unsealed
    // or partly sealed descriptor and any other descriptor path are each refused before the manifest is read.
    CHECK(run({"--manifest", example, "--role", "engine"}) == "ERROR");              // a file on disk is not the owner's
    CHECK(run({"--manifest", measured}) == "ERROR");                                  // no role
    CHECK(run({"--manifest", measured, "--role", "helper"}) == "ERROR");
    CHECK(run({"--manifest", "/proc/self/fd/not-a-number", "--role", "engine"}) == "ERROR");
    CHECK(run({"--manifest", "/proc/self/fd/2", "--role", "engine"}) == "ERROR");     // below the first free descriptor
    {
        const int open_fd = memfd_create("unsealed", MFD_ALLOW_SEALING);
        CHECK(open_fd >= 3 && write(open_fd, "{}", 2) == 2);
        CHECK(run({"--manifest", "/proc/self/fd/" + std::to_string(open_fd), "--role", "engine"}) == "ERROR");
        CHECK(fcntl(open_fd, F_ADD_SEALS, F_SEAL_WRITE) == 0);                         // write-sealed only
        CHECK(run({"--manifest", "/proc/self/fd/" + std::to_string(open_fd), "--role", "engine"}) == "ERROR");
    }

    // RSC02: the production launch owner passes one authenticated heat publication identity as an ephemeral flag, *outside*
    // the signed manifest. The strict reader consumes it before radiance config_parse and makes it available only as a
    // typed engine snapshot; it must never leak into argv.
    const std::string heat_nonce = "0123456789abcdef0123456789abcdef";
    const std::string with_heat =
        run({"--manifest", measured, "--role", "engine", "--bc-heat-launch-nonce", heat_nonce});
    CHECK(with_heat == got && g_parsed);
    CHECK(with_heat.find(heat_nonce) == std::string::npos);
    BcEngineSettings heat_settings{};
    CHECK(bc_serving_engine_settings(&heat_settings));
    CHECK(heat_settings.heat_launch_nonce != nullptr);
    if (heat_settings.heat_launch_nonce)
        CHECK(std::string(heat_settings.heat_launch_nonce) == heat_nonce);
    const auto serve = [&](std::vector<std::string> more) {
        more.insert(more.begin(), {"--manifest", measured, "--role", "engine"});
        return run(more);
    };
    CHECK(serve({"--bc-heat-launch-nonce", "bad"}) == "ERROR");
    CHECK(serve({"--bc-heat-launch-nonce", "0123456789ABCDEF0123456789abcdef"}) == "ERROR");
    CHECK(serve({"--bc-heat-launch-nonce", heat_nonce, "--bc-heat-launch-nonce", heat_nonce}) == "ERROR");
    CHECK(serve({"--bc-heat-launch-nonce"}) == "ERROR");
    CHECK(serve({"--plan-memory", "memory.json", "--bc-heat-launch-nonce", heat_nonce}) == "ERROR");
    CHECK(serve({}) == got);
    BcEngineSettings no_heat{};
    CHECK(bc_serving_engine_settings(&no_heat));
    CHECK(no_heat.heat_launch_nonce == nullptr);

    // Hostile inherited R3_* variables must be inert. The manifest's typed handoff settings are authoritative; the startup
    // code must not rewrite them.
    {
        setenv("R3_MOE_WORKER", "0", 1);
        setenv("R3_MOE_WORKER_SHM", "/stale", 1);
        setenv("R3_MOE_WORKER_MAX_ROWS", "1", 1);
        setenv("R3_MOE_WORKER_COMBINE", "0", 1);
        setenv("R3_MOE_WORKER_HISTOGRAM", "/stale-heat", 1);
        setenv("R3_MOE_WORKER_MASK_ONLY", "2", 1);
        CHECK(serve({}) != "ERROR");
        BcEngineSettings cfg{};
        CHECK(bc_serving_engine_settings(&cfg));
        CHECK(cfg.manifest == 1);
        const auto& h = shape.at("spec").at("handoff");
        CHECK(cfg.handoff == 1);
        CHECK(cfg.shm != nullptr && std::string(cfg.shm) == h.at("transport").at("shm").get<std::string>());
        CHECK(cfg.max_rows == h.at("transport").at("max_rows").get<long long>());
        CHECK(cfg.skip_owned == (h.at("transport").at("skip_owned").get<bool>() ? 1u : 0u));
        CHECK(cfg.combine == (shape.at("spec").at("processes").at("helper_common").at("combine").at("enabled").get<bool>() ? 1u : 0u));
        if (h.at("telemetry").at("histogram_file").is_string()) {
            CHECK(cfg.histogram_file != nullptr);
            if (cfg.histogram_file)
                CHECK(std::string(cfg.histogram_file) == h.at("telemetry").at("histogram_file").get<std::string>());
        }
        CHECK(cfg.histogram_period_s == h.at("telemetry").at("period_s").get<long long>());
        CHECK(std::string(std::getenv("R3_MOE_WORKER")) == "0");
        CHECK(std::string(std::getenv("R3_MOE_WORKER_SHM")) == "/stale");
        CHECK(std::string(std::getenv("R3_MOE_WORKER_MAX_ROWS")) == "1");
        CHECK(std::string(std::getenv("R3_MOE_WORKER_COMBINE")) == "0");
        CHECK(std::string(std::getenv("R3_MOE_WORKER_HISTOGRAM")) == "/stale-heat");
        CHECK(std::string(std::getenv("R3_MOE_WORKER_MASK_ONLY")) == "2");
        for (const char* v : {"R3_MOE_WORKER", "R3_MOE_WORKER_SHM", "R3_MOE_WORKER_MAX_ROWS",
                               "R3_MOE_WORKER_COMBINE", "R3_MOE_WORKER_HISTOGRAM", "R3_MOE_WORKER_MASK_ONLY"}) unsetenv(v);
    }

    // The measuring start. An unmeasured manifest (no engine_memory, no checkpoint slots) is a file taken only with
    // --plan-memory FILE and --plan-memory-ranks FILE, both forwarded to radiance; a serving start refuses it, and
    // --plan-memory with a measured manifest is refused too: neither state is read as the other.
    const nlohmann::json unmeasured_doc = unmeasured_of(shape);
    const std::string unmeasured = write("unmeasured.json", seal_digest(unmeasured_doc));
    const std::string plan_out = g_dir + "/plan.json", ranks_out = g_dir + "/ranks.json";
    {
        const std::string measuring = run({"--manifest", unmeasured, "--plan-memory", plan_out, "--plan-memory-ranks", ranks_out});
        CHECK(measuring == "engine --plan-memory " + plan_out + " --plan-memory-ranks " + ranks_out && g_parsed);
        CHECK(bc_serving_mark_given(&g_given));
        CHECK(g_given.at("--plan-memory") == "command line");   // the flag stays the operator's, not the manifest's
        CHECK(bc_serving_engine_bdfs() ==
              shape.at("spec").at("processes").at("engine").at("device").at("gpus").get<std::vector<std::string>>());
        CHECK(run({"--manifest", unmeasured, "--plan-memory", plan_out}) == "ERROR");   // v3 measurement needs the rank file
        CHECK(run({"--manifest", unmeasured, "--plan-memory-ranks", ranks_out}) == "ERROR");
        CHECK(run({"--manifest", unmeasured, "--plan-memory", "a", "--plan-memory", "b", "--plan-memory-ranks", ranks_out}) == "ERROR");
        CHECK(run({"--manifest", unmeasured, "--plan-memory", plan_out, "--plan-memory-ranks"}) == "ERROR");
        CHECK(run({"--manifest", unmeasured, "--role", "engine"}) == "ERROR");        // a serving start with no plan
        CHECK(run({"--manifest", sealed(seal_digest(unmeasured_doc)), "--role", "engine"}) == "ERROR");
        CHECK(run({"--manifest", example, "--plan-memory", plan_out, "--plan-memory-ranks", ranks_out}) == "ERROR");   // a measuring start with a plan
    }

    // RSC08 S3: the plan check comes from the manifest, through main()'s real order (the manifest reader, then the plan flags).
    // A measuring start is measuring and both flags are off argv for config_parse; a start from the measured manifest arms the
    // check with the manifest's own plan and floor; no start without a manifest arms one.
    {
        const std::string measuring = run_main_order({"--manifest", unmeasured, "--plan-memory", plan_out, "--plan-memory-ranks", ranks_out});
        CHECK(measuring == "engine" && bc_plan::mode() == bc_plan::Mode::measure && bc_plan::measure_out() == plan_out);

        const std::string serving = run_main_order({"--manifest", measured, "--role", "engine"});
        CHECK(serving == got);   // nothing of the plan machinery reaches radiance's own parser
        CHECK(bc_plan::mode() == bc_plan::Mode::check);

        // the armed check is the manifest's: its own expected plan is admitted by itself, and a drifted live plan is a plan mismatch
        bc::serving::v2::EnginePlan expected;
        CHECK(!bc::serving::v2::decode(shape.at("spec").at("plan").at("engine_memory").at("expected"), "", &expected));
        std::fflush(stderr);
        CHECK(bc_plan::check(expected) == 0);
        bc::serving::v2::EnginePlan drifted = expected;
        drifted.vram.statics += 1;
        CHECK(bc_plan::check(drifted) == 2);

        // without a manifest there is nothing to check; --plan-memory still measures
        const std::string dense = run_main_order({"--model", "m.rad", "--plan-memory", plan_out});
        CHECK(dense == "engine --model m.rad" && bc_plan::mode() == bc_plan::Mode::measure);
        CHECK(run_main_order({"--model", "m.rad"}) == "engine --model m.rad" && bc_plan::mode() == bc_plan::Mode::none);
        // the retired flags are not the plan machinery's: radiance's own parser refuses them as unknown options
        CHECK(run_main_order({"--model", "m.rad", "--plan-check", "x", "--plan-check-floor-ppm", "0"}) ==
              "engine --model m.rad --plan-check x --plan-check-floor-ppm 0");
    }

    // Amendment 14/15: typed kernel settings, with independent defaults.
    {
        nlohmann::json doc = shape;
        doc["spec"]["processes"]["engine"]["kernels"] = {{"moe_native", false}, {"moe_fp16", false}, {"moe_ahalf", true},
            {"dec11", false}, {"gemv", true}, {"fuse_stage", false}, {"all_reduce_wire4", true}};
        doc["spec"]["intent"]["service"]["profile"] = "lab";
        doc["spec"]["experiment"] = {{"engine", {{"routing_evidence", true}, {"mask_only", 2}}}};
        doc["provenance"]["/spec/experiment"] = {{"origin", "profile"}, {"source", "lab"}, {"class", "diagnostic"},
            {"lifecycle", "startup_only"}, {"confidence", "exact"}, {"inputs", nlohmann::json::array()}};
        doc["provenance"]["/spec/processes/engine/kernels"] = {{"origin", "profile"}, {"source", "kernel switches"}, {"class", "policy"},
            {"lifecycle", "startup_only"}, {"confidence", "exact"}, {"inputs", nlohmann::json::array()}};
        const std::string kernels = sealed(seal_digest(doc));
        setenv("R3_MOE_NATIVE", "1", 1);
        setenv("R3_MOE_FP16", "1", 1);
        setenv("R3_MOE_AHALF", "0", 1);
        setenv("R3_DEC11", "1", 1);
        setenv("R3_GEMV", "0", 1);
        setenv("R3_MOE_WORKER_FUSESTAGE", "1", 1);
        setenv("R3_MOE_WORKER_MASK_ONLY", "0", 1);
        CHECK(run({"--manifest", kernels, "--role", "engine"}) != "ERROR");
        BcEngineSettings cfg{};
        CHECK(bc_serving_engine_settings(&cfg));
        CHECK(cfg.moe_native == 0 && cfg.moe_fp16 == 0 && cfg.moe_ahalf == 1);
        CHECK(cfg.dec11 == 0 && cfg.gemv == 1 && cfg.fuse_stage == 0);
        CHECK(cfg.all_reduce_wire4 == 1 && cfg.mask_only == 2);
        CHECK(std::string(std::getenv("R3_MOE_NATIVE")) == "1");
        CHECK(std::string(std::getenv("R3_MOE_FP16")) == "1");
        CHECK(std::string(std::getenv("R3_MOE_AHALF")) == "0");
        CHECK(std::string(std::getenv("R3_DEC11")) == "1");
        CHECK(std::string(std::getenv("R3_GEMV")) == "0");
        CHECK(std::string(std::getenv("R3_MOE_WORKER_FUSESTAGE")) == "1");
        CHECK(std::string(std::getenv("R3_MOE_WORKER_MASK_ONLY")) == "0");
        CHECK(std::string(std::getenv("RAD_ROUTING_EVIDENCE")) == "1");
        for (const char* v : {"R3_MOE_NATIVE", "R3_MOE_FP16", "R3_MOE_AHALF", "R3_DEC11", "R3_GEMV",
                               "R3_MOE_WORKER_FUSESTAGE", "R3_MOE_WORKER_MASK_ONLY",
                               "RAD_ROUTING_EVIDENCE", "R3_AR_WIRE4"}) unsetenv(v);
    }

    // The KL mode is a kind of run, not configuration: its flags accompany the manifest, pass through unchanged and keep their
    // command-line label; no other flag is accepted, and a flag without its path is refused. A manifest that has helpers is
    // scored only by its matched owner (the process that holds the descriptor and runs the build's bc-serve), which a test
    // process is not; a measuring start cannot be a quality run.
    for (const char* flag : {"--kld-record", "--kld-corpus", "--kld-ref", "--kld-out"}) {
        if (!g_takes_arg.count(flag)) g_options.push_back(flag);
        g_takes_arg[flag] = true;
    }
    {
        CHECK(serve({"--kld-ref", "/ref", "--kld-out", "/out.json"}) == "ERROR");   // helpers: needs the matched owner
        CHECK(serve({"--kld-ref"}) == "ERROR");
        CHECK(run({"--manifest", unmeasured, "--plan-memory", plan_out, "--plan-memory-ranks", ranks_out,
                   "--kld-ref", "/ref", "--kld-out", "/out.json"}) == "ERROR");
        // a manifest with no helper processes is a trunk-only quality run
        nlohmann::json trunk = shape;
        trunk["spec"]["processes"]["helpers"] = nlohmann::json::array();
        trunk["spec"]["plan"]["owner_map"]["helpers"] = nlohmann::json::array();
        trunk["spec"]["plan"]["owner_map"]["owner_map_sha256"] = bc::serving::sha256_hex("");
        trunk["spec"]["processes"].erase("helper_common");
        trunk["spec"].erase("handoff");
        auto& devices = trunk["spec"]["plan"]["power"]["devices"];
        devices.erase(std::remove_if(devices.begin(), devices.end(),
                                     [](const nlohmann::json& d) { return d.at("role") != "engine"; }),
                      devices.end());
        for (auto it = trunk["provenance"].begin(); it != trunk["provenance"].end();)
            it = (it.key().rfind("/spec/handoff", 0) == 0 || it.key().rfind("/spec/processes/helpers/", 0) == 0 ||
                  it.key() == "/spec/processes/helper_common") ? trunk["provenance"].erase(it) : std::next(it);
        // the measured plan is keyed by the configuration it measured, which just changed: rekey it
        bc::serving::v2::Manifest typed;
        CHECK(!bc::serving::v2::decode(trunk, "", &typed));
        trunk["spec"]["plan"]["engine_memory"]["cache_key"] = bc::serving::v2::plan_cache_key(typed);
        const std::string quality = sealed(seal_digest(trunk));
        const std::string kld = run({"--manifest", quality, "--role", "engine", "--kld-ref", "/ref", "--kld-out", "/out.json"});
        CHECK(kld != "ERROR" && g_parsed);
        CHECK(kld.find("--kld-ref /ref --kld-out /out.json") != std::string::npos);
        CHECK(bc_serving_mark_given(&g_given));
        CHECK(g_given.at("--kld-ref") == "command line" && g_given.at("--kld-out") == "command line");
        CHECK(run({"--manifest", quality, "--role", "engine", "--kld-record", "/ref", "--kld-corpus", "/corpus.jsonl"}) != "ERROR");
        CHECK(run({"--manifest", quality, "--role", "engine", "--kld-ref", "/r", "--tp", "2"}) == "ERROR");
    }

    // without --manifest argv is untouched
    CHECK(run({"--model", "x", "--tp-wire", "exact"}) == "engine --model x --tp-wire exact");

    // other flags, a duplicate, a missing path
    CHECK(serve({"--tp", "2"}) == "ERROR");
    CHECK(serve({"--manifest", measured}) == "ERROR");
    CHECK(run({"--manifest"}) == "ERROR");
    CHECK(run({"--help", "--manifest", measured, "--role", "engine"}) == "ERROR");
    CHECK(run({"--manifest", g_dir + "/missing.json", "--plan-memory", plan_out, "--plan-memory-ranks", ranks_out}) == "ERROR");

    // the shared strict reader refuses what is not an edition 3 manifest, with its own errors
    CHECK(run({"--manifest", sealed("{not json"), "--role", "engine"}) == "ERROR");
    CHECK(run({"--manifest", sealed("{}"), "--role", "engine"}) == "ERROR");
    // the interim bc-serve format (flat schema_version / values / content_sha256) is gone: refused, not silently read
    CHECK(run({"--manifest", sealed("{\"schema_version\":1,\"values\":{\"engine\":{\"radiance_home\":\"/h\","
                                    "\"kernels\":\"libr4d\"}},\"process_helpers\":[],"
                                    "\"hash_algorithm\":\"sha256\",\"content_sha256\":\"" + std::string(64, '0') + "\"}"),
               "--role", "engine"}) == "ERROR");
    // a change to any value without a new digest is refused (the digest is checked, not trusted)
    std::string tampered = slurp(example);
    const size_t at = tampered.find("\"tp\":1");
    CHECK(at != std::string::npos);
    if (at != std::string::npos) tampered.replace(at, 6, "\"tp\":2");
    CHECK(run({"--manifest", sealed(tampered), "--role", "engine"}) == "ERROR");
    // a duplicate key is refused by the strict JSON layer
    std::string dup = slurp(example);
    const size_t kind = dup.find("\"kind\"");
    CHECK(kind != std::string::npos);
    if (kind != std::string::npos) dup.insert(kind, "\"kind\":\"x\",");
    CHECK(run({"--manifest", sealed(dup), "--role", "engine"}) == "ERROR");

    if (g_failed) { std::fprintf(stderr, "%d failed\n", g_failed); return 1; }
    std::puts("ok");
    return 0;
}
