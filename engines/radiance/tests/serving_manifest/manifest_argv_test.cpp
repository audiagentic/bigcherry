// Card-free test of the patched engine's manifest-to-argv logic (core/bc_serving_manifest.cpp from rad0009).
// Built by tools/tests/patch/test_radiance_patch_packages.py against radiance's real option table and a mirror of config_parse's shape rules.
// usage: manifest_argv_test SCRATCH_DIR BC_SERVE_FIXTURE OPTIONS_FILE
#include "bc_serving_manifest.h"

#include <cstdio>
#include <cstring>
#include <fstream>
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
static const char* kSha = "0123456789abcdef0123456789abcdef";

static std::string write(const std::string& name, const std::string& body) {
    const std::string p = g_dir + "/" + name;
    std::ofstream(p) << body;
    return p;
}

static std::string manifest(const std::string& engine_values, const std::string& schema = "1") {
    // the shape bc-serve writes: each section nested under values (see the shared bc-serve fixture test below)
    return "{\"schema_version\":" + schema + ",\"values\":{\"service\":{\"max_sessions\":4},\"engine\":{" +
           engine_values + "}},\"process_helpers\":[],\"hash_algorithm\":\"sha256\",\"content_sha256\":\"" + kSha + "\"}";
}

static const std::string kBase = "\"radiance_home\":\"/h\",\"kernels\":\"libr4d,libref\"";

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

int main(int, char** argv) {
    g_dir = argv[1];
    load_options(argv[3]);
    CHECK(g_options.size() > 100);
    // The shape bc-serve really writes (bc_serve_dry_run.json: `bc-serve --dry-run` against a synthetic sysfs tree, so it
    // names no real host): values nested by section. The reader must accept it and emit the engine section as flags.
    // A hand-written manifest cannot stand in for this: the flat-key fixtures once hid that rad0009 read the wrong shape.
    {
        const std::string real = argv[2];
        const std::string got = run({"--manifest", real});
        CHECK(got != "ERROR" && g_parsed);
        for (const char* want : {"--kernels libr4d,libr3moe,libref", "--max-num-seqs 8", "--model /models/flash-next.rad",
                                 "--radiance-home /opt/radiance/radiance_home", "--prefix-cache-mib 12288", "--tp 1"})
            CHECK(got.find(want) != std::string::npos);
        CHECK(bc_serving_mark_given(&g_given));
    }
    // The flat-key shape is not the contract and is refused, not silently read as "no engine settings".
    CHECK(run({"--manifest", write("flat.json", "{\"schema_version\":1,\"values\":{\"engine.radiance_home\":\"/h\","
                                                "\"engine.kernels\":\"libr4d\"},\"process_helpers\":[],"
                                                "\"hash_algorithm\":\"sha256\",\"content_sha256\":\"" + std::string(kSha) +
                                                "\"}")}) == "ERROR");

    // without --manifest argv is untouched
    CHECK(run({"--model", "x", "--tp-wire", "exact"}) == "engine --model x --tp-wire exact");

    // sorted keys, integers and strings, booleans
    const std::string ok = write("ok.json", manifest(kBase + ",\"max_num_seqs\":6,\"tp_wire\":\"wht6\","
                                                              "\"live\":true,\"deterministic\":false,"
                                                              "\"model\":\"/m.rad\""));
    CHECK(run({"--manifest", ok}) == "engine --kernels libr4d,libref --max-num-seqs 6 --model /m.rad "
                                     "--radiance-home /h --tp-wire wht6 --live");
    CHECK(run({"--manifest", ok, "--role", "engine", "-v"}) == "engine -v --kernels libr4d,libref --max-num-seqs 6 "
                                                               "--model /m.rad --radiance-home /h --tp-wire wht6 --live");
    // deterministic
    CHECK(run({"--manifest", ok}) == run({"--manifest", ok}));

    // other flags, a wrong role, duplicates, a missing path
    CHECK(run({"--manifest", ok, "--tp", "2"}) == "ERROR");
    CHECK(run({"--manifest", ok, "--role", "helper"}) == "ERROR");
    CHECK(run({"--manifest", ok, "--manifest", ok}) == "ERROR");
    CHECK(run({"--manifest"}) == "ERROR");
    CHECK(run({"--help", "--manifest", ok}) == "ERROR");
    CHECK(run({"--manifest", g_dir + "/missing.json"}) == "ERROR");

    // schema, values, hash fields
    CHECK(run({"--manifest", write("v2.json", manifest(kBase, "2"))}) == "ERROR");
    CHECK(run({"--manifest", write("nov.json", "{\"schema_version\":1,\"hash_algorithm\":\"sha256\","
                                                "\"content_sha256\":\"0123456789abcdef\"}")}) == "ERROR");
    CHECK(run({"--manifest", write("junk.json", "{not json")}) == "ERROR");
    CHECK(run({"--manifest", write("nosha.json", "{\"schema_version\":1,\"values\":{\"engine\":{" + kBase + "}},"
                                                  "\"hash_algorithm\":\"sha256\"}")}) == "ERROR");

    // an unknown engine key, and one that is not an engine option at all
    CHECK(run({"--manifest", write("unk.json", manifest(kBase + ",\"no_such_thing\":1"))}) == "ERROR");
    CHECK(run({"--manifest", write("man.json", manifest(kBase + ",\"manifest\":\"x\""))}) == "ERROR");
    CHECK(run({"--manifest", write("help.json", manifest(kBase + ",\"help\":true"))}) == "ERROR");

    // the argv shape follows the JSON type alone; shape mismatches are radiance's config_parse to reject
    CHECK(run({"--manifest", ok}) != "ERROR" && g_parsed);
    CHECK(run({"--manifest", write("fl.json", manifest(kBase + ",\"max_num_seqs\":2.5"))}) ==
          "engine --kernels libr4d,libref --max-num-seqs 2.5 --radiance-home /h");
    CHECK(run({"--manifest", write("big.json", manifest(kBase + ",\"max_num_seqs\":4294967296"))}) ==
          "engine --kernels libr4d,libref --max-num-seqs 4294967296 --radiance-home /h");
    // a value for a boolean option: emitted as "--live yes", which radiance rejects as an unknown option
    CHECK(run({"--manifest", write("b2.json", manifest(kBase + ",\"live\":\"yes\""))}) ==
          "engine --kernels libr4d,libref --live yes --radiance-home /h");
    CHECK(!g_parsed);
    CHECK(run({"--manifest", write("b3.json", manifest(kBase + ",\"live\":1"))}) != "ERROR" && !g_parsed);
    // true for an option that needs an argument: a bare "--max-num-seqs", which radiance rejects
    CHECK(run({"--manifest", write("b1.json", manifest(kBase + ",\"max_num_seqs\":true"))}) ==
          "engine --kernels libr4d,libref --radiance-home /h --max-num-seqs");
    CHECK(!g_parsed);
    // false adds no flag
    CHECK(run({"--manifest", write("f.json", manifest(kBase + ",\"live\":false"))}) ==
          "engine --kernels libr4d,libref --radiance-home /h");
    // null, arrays, objects are errors here
    CHECK(run({"--manifest", write("n.json", manifest(kBase + ",\"max_num_seqs\":null"))}) == "ERROR");
    CHECK(run({"--manifest", write("a.json", manifest(kBase + ",\"model\":[1]"))}) == "ERROR");
    CHECK(run({"--manifest", write("o.json", manifest(kBase + ",\"model\":{}"))}) == "ERROR");

    // RADIANCE_HOME / RADIANCE_KERNELS must not be able to fill in for the manifest
    CHECK(run({"--manifest", write("nohome.json", manifest("\"kernels\":\"libref\""))}) == "ERROR");
    CHECK(run({"--manifest", write("nokern.json", manifest("\"radiance_home\":\"/h\""))}) == "ERROR");
    CHECK(run({"--manifest", write("emptyk.json", manifest("\"radiance_home\":\"/h\",\"kernels\":\"\""))}) == "ERROR");
    CHECK(run({"--manifest", write("emptyh.json", manifest("\"radiance_home\":\"\",\"kernels\":\"libref\""))}) == "ERROR");

    // the report source: the manifest's options are relabelled, -v and the environment's are not
    run({"--manifest", ok, "-v"});
    CHECK(g_parsed);
    g_given["--kernels"] = "environment (RADIANCE_KERNELS)";
    CHECK(!bc_serving_mark_given(&g_given));   // --kernels is a manifest option: its source must be the command line
    run({"--manifest", ok, "-v"});
    CHECK(bc_serving_mark_given(&g_given));
    CHECK(g_given["--model"] == "manifest 0123456789ab");
    CHECK(g_given["--live"] == "manifest 0123456789ab");
    CHECK(g_given["-v"] == "command line");

    // a bare flag for an option that needs an argument swallows the next flag: "--max-num-seqs --no-cors"
    // parses, but --no-cors was not parsed as given, so start-up fails
    CHECK(run({"--manifest", write("sw.json", manifest(kBase + ",\"max_num_seqs\":true,"
                                                                "\"no_cors\":true"))}) ==
          "engine --kernels libr4d,libref --radiance-home /h --max-num-seqs --no-cors");
    CHECK(g_parsed);
    CHECK(!bc_serving_mark_given(&g_given));
    CHECK(g_given["--max-num-seqs"] == "command line");   // not relabelled on failure

    std::printf(g_failed ? "FAILED\n" : "ok\n");
    return g_failed ? 1 : 0;
}
