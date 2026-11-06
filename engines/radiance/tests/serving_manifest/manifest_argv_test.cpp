// Card-free test of the patched engine's manifest-to-argv logic (core/bc_serving_manifest.cpp from rad0009): a serving
// manifest v1 read by the shared strict reader, its registry-bound fields as flags.
// Built by tools/tests/patch/test_radiance_patch_packages.py against radiance's real option table and a mirror of config_parse's shape rules.
// usage: manifest_argv_test SCRATCH_DIR V1_EXAMPLE_MANIFEST OPTIONS_FILE LAB_EXAMPLE_MANIFEST
#include "bc_plan_check.h"
#include "bc_serving_manifest.h"
#include "jcs.h"
#include "serving_codec.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

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

int main(int, char** argv) {
    g_dir = argv[1];
    const char* argc4 = argv[4];
    load_options(argv[3]);
    // rad0011 adds --plan-memory FILE to radiance's option table (the table the harness extracted is the unpatched pin's)
    g_options.push_back("--plan-memory");
    g_takes_arg["--plan-memory"] = true;
    CHECK(g_options.size() > 100);
    // A real v1 manifest (the design example for the production layout): the strict reader accepts it, and every field
    // the registry binds to a radiance option becomes that flag, in registry order.
    const std::string example = argv[2];
    const std::string got = run({"--manifest", example});
    CHECK(got != "ERROR" && g_parsed);
    CHECK(got == "engine --model /models/flash-next.rad --max-num-seqs 8 --max-model-len 262144 --host-pool-mib 9216 "
                 "--prefix-cache-mib 12288 --host-headroom-mib 4800 --checkpoint-slots 43 "
                 "--radiance-home /opt/radiance/radiance_home --kernels libr4d,libr3moe,libref --port 36080 --tp 1 "
                 "--max-num-batched-tokens 2048 --checkpoint-interval 4096 --placement expert_tiered "
                 "--ngram-placement ram --kv-cache-dtype fp8 --num-speculative-tokens 3 --rope-scaling none "
                 "--prefix-resume nearest");
    CHECK(bc_serving_mark_given(&g_given));
    for (const auto& kv : g_given) CHECK(kv.second == "manifest 44bdbe1a071a");   // "manifest <first 12 hex of the digest>"
    CHECK(run({"--manifest", example, "--role", "engine", "-v"}).rfind("engine -v --model ", 0) == 0);
    CHECK(run({"--manifest", example}) == got);   // deterministic

    // The measuring start. An unmeasured manifest (plan.state unmeasured, no engine_memory, no checkpoint slots) is taken only
    // with --plan-memory FILE, which is forwarded to radiance; a serving start refuses it, and --plan-memory with a measured
    // manifest is refused too: neither state is read as the other.
    {
        nlohmann::json doc = nlohmann::json::parse(slurp(example));
        doc["spec"]["plan"]["state"] = "unmeasured";
        doc["spec"]["plan"].erase("engine_memory");
        doc["spec"]["resources"]["host"].erase("checkpoint_slots");
        doc["provenance"]["/spec/resources"]["inputs"] = nlohmann::json::array({"/spec/facts/host", "/spec/calibration"});
        std::string digest, why;
        doc["metadata"].erase("content_digest");
        CHECK(bc::serving::manifest_content_digest(doc, &digest, &why));
        doc["metadata"]["content_digest"] = digest;
        const std::string unmeasured = write("unmeasured.json", doc.dump(2));
        const std::string measuring = run({"--manifest", unmeasured, "--plan-memory", g_dir + "/plan.json"});
        CHECK(measuring != "ERROR" && g_parsed);
        CHECK(measuring.find("--plan-memory " + g_dir + "/plan.json") != std::string::npos);
        CHECK(measuring.find("--checkpoint-slots") == std::string::npos);   // derived from a plan there is not yet
        CHECK(bc_serving_mark_given(&g_given));
        CHECK(g_given.at("--plan-memory") == "command line");               // the flag stays the operator's, not the manifest's
        CHECK(g_given.at("--model").rfind("manifest ", 0) == 0);
        CHECK(run({"--manifest", unmeasured}) == "ERROR");
        CHECK(run({"--manifest", example, "--plan-memory", g_dir + "/plan.json"}) == "ERROR");
        CHECK(run({"--manifest", unmeasured, "--plan-memory"}) == "ERROR");
        CHECK(run({"--manifest", unmeasured, "--plan-memory", "a", "--plan-memory", "b"}) == "ERROR");
    }

    // Amendment 11: the engine's other flags are manifest fields. Value-less flags (profile_ops, debug_selection, deterministic,
    // no_prefix_cache) are emitted without a value, the others with theirs, and radiance's own parser takes all of it.
    {
        const std::string lab = argc4;
        const std::string lab_argv = run({"--manifest", lab});
        CHECK(lab_argv != "ERROR" && g_parsed);
        for (const char* flag : {" --profile-ops", " --debug-selection", " --deterministic", " --no-prefix-cache"}) {
            const size_t at = lab_argv.find(flag);
            CHECK(at != std::string::npos);
            // no value follows: the next token is another flag or the end
            if (at != std::string::npos) {
                const size_t next = at + std::strlen(flag);
                CHECK(next == lab_argv.size() || lab_argv.compare(next, 3, " --") == 0);
            }
        }
        CHECK(lab_argv.find(" --kv-resident 65536") != std::string::npos);
        CHECK(lab_argv.find(" --tp-wire exact") != std::string::npos);
        CHECK(lab_argv.find(" --gpu-headroom-mib 96") != std::string::npos);
        CHECK(bc_serving_mark_given(&g_given));
        CHECK(g_given.at("--profile-ops").rfind("manifest ", 0) == 0 && g_given.at("--no-prefix-cache").rfind("manifest ", 0) == 0);
    }

    // without --manifest argv is untouched
    CHECK(run({"--model", "x", "--tp-wire", "exact"}) == "engine --model x --tp-wire exact");

    // RSC08 S3: the plan check comes from the manifest, through main()'s real order (the manifest reader, then the plan flags).
    // A measuring start (unmeasured manifest + --plan-memory) is measuring and the flag is off argv for config_parse; a start
    // from the measured manifest arms the check with the manifest's own plan and floor; no start without a manifest arms one.
    {
        nlohmann::json doc = nlohmann::json::parse(slurp(example));
        doc["spec"]["plan"]["state"] = "unmeasured";
        doc["spec"]["plan"].erase("engine_memory");
        doc["spec"]["resources"]["host"].erase("checkpoint_slots");
        doc["provenance"]["/spec/resources"]["inputs"] = nlohmann::json::array({"/spec/facts/host", "/spec/calibration"});
        std::string digest, why;
        doc["metadata"].erase("content_digest");
        CHECK(bc::serving::manifest_content_digest(doc, &digest, &why));
        doc["metadata"]["content_digest"] = digest;
        const std::string unmeasured = write("unmeasured-main.json", doc.dump(2));
        const std::string plan_out = g_dir + "/measured-plan.json";

        const std::string measuring = run_main_order({"--manifest", unmeasured, "--plan-memory", plan_out});
        CHECK(measuring != "ERROR" && measuring.rfind("EXIT", 0) != 0);
        CHECK(measuring.find("--plan-memory") == std::string::npos && measuring.find("--model ") != std::string::npos);
        CHECK(bc_plan::mode() == bc_plan::Mode::measure && bc_plan::measure_out() == plan_out);

        const std::string serving = run_main_order({"--manifest", example});
        CHECK(serving == got);   // nothing of the plan machinery reaches radiance's own parser
        CHECK(bc_plan::mode() == bc_plan::Mode::check);

        // the armed check is the manifest's: its own expected plan is admitted by itself, and a drifted live plan is a plan mismatch
        const nlohmann::json measured = nlohmann::json::parse(slurp(example));
        bc::serving::v1::EnginePlan expected;
        CHECK(!bc::serving::v1::decode(measured.at("spec").at("plan").at("engine_memory").at("expected"), "", &expected));
        std::fflush(stderr);
        CHECK(bc_plan::check(expected) == 0);
        bc::serving::v1::EnginePlan drifted = expected;
        drifted.vram.statics += 1;
        CHECK(bc_plan::check(drifted) == 2);

        // refused states: neither manifest state is read as the other, whatever the plan flags say
        CHECK(run_main_order({"--manifest", unmeasured}) == "ERROR");                           // a serving start with no plan
        CHECK(run_main_order({"--manifest", example, "--plan-memory", plan_out}) == "ERROR");   // a measuring start with a plan

        // without a manifest there is nothing to check; --plan-memory still measures (a layout with no manifest writer yet)
        const std::string dense = run_main_order({"--model", "m.rad", "--plan-memory", plan_out});
        CHECK(dense == "engine --model m.rad" && bc_plan::mode() == bc_plan::Mode::measure);
        CHECK(run_main_order({"--model", "m.rad"}) == "engine --model m.rad" && bc_plan::mode() == bc_plan::Mode::none);
        // the retired flags are not the plan machinery's: radiance's own parser refuses them as unknown options
        CHECK(run_main_order({"--model", "m.rad", "--plan-check", "x", "--plan-check-floor-ppm", "0"}) ==
              "engine --model m.rad --plan-check x --plan-check-floor-ppm 0");
    }

    // other flags, a wrong role, duplicates, a missing path
    CHECK(run({"--manifest", example, "--tp", "2"}) == "ERROR");
    CHECK(run({"--manifest", example, "--role", "helper"}) == "ERROR");
    CHECK(run({"--manifest", example, "--manifest", example}) == "ERROR");
    CHECK(run({"--manifest"}) == "ERROR");
    CHECK(run({"--help", "--manifest", example}) == "ERROR");
    CHECK(run({"--manifest", g_dir + "/missing.json"}) == "ERROR");

    // the shared strict reader refuses what is not a v1 manifest, with its own errors
    CHECK(run({"--manifest", write("junk.json", "{not json")}) == "ERROR");
    CHECK(run({"--manifest", write("empty.json", "{}")}) == "ERROR");
    // the interim bc-serve format (flat schema_version / values / content_sha256) is gone: refused, not silently read
    CHECK(run({"--manifest", write("interim.json", "{\"schema_version\":1,\"values\":{\"engine\":{\"radiance_home\":\"/h\","
                                                   "\"kernels\":\"libr4d\"}},\"process_helpers\":[],"
                                                   "\"hash_algorithm\":\"sha256\",\"content_sha256\":\"" + std::string(64, '0') +
                                                   "\"}")}) == "ERROR");
    // a change to any value without a new digest is refused (the digest is checked, not trusted)
    std::string tampered = slurp(example);
    const size_t at = tampered.find("\"tp\": 1");
    CHECK(at != std::string::npos);
    if (at != std::string::npos) tampered.replace(at, 7, "\"tp\": 2");
    CHECK(run({"--manifest", write("tampered.json", tampered)}) == "ERROR");
    // a duplicate key is refused by the strict JSON layer
    std::string dup = slurp(example);
    const size_t kind = dup.find("\"kind\"");
    CHECK(kind != std::string::npos);
    if (kind != std::string::npos) dup.insert(kind, "\"kind\": \"x\", ");
    CHECK(run({"--manifest", write("dup.json", dup)}) == "ERROR");

    // an emitted option that the parser did not take as given fails the mark (a swallowed flag)
    std::map<std::string, std::string> swallowed;
    CHECK(run({"--manifest", example}) == got);
    swallowed = g_given;
    swallowed.erase("--tp");
    CHECK(!bc_serving_mark_given(&swallowed));

    if (g_failed) { std::fprintf(stderr, "%d failed\n", g_failed); return 1; }
    std::puts("ok");
    return 0;
}
