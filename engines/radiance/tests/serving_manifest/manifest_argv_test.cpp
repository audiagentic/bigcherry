// Card-free test of the patched engine's manifest-to-argv logic (core/bc_serving_manifest.cpp from rad0009).
// Built by tools/tests/patch/test_radiance_patch_packages.py against a stub of the engine's config_option_names, plus a mirror of config_parse's shape rules.
// usage: manifest_argv_test SCRATCH_DIR
#include "bc_serving_manifest.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace rad {
namespace diag {
std::vector<std::string> config_option_names() {
    return {"--model", "--radiance-home", "--kernels", "--max-num-seqs", "--tp-wire", "--live", "--prefix-cache",
            "-v", "-vv"};
}
}  // namespace diag
}  // namespace rad

// A mirror of the shape rules of radiance's config_parse (core/config.cpp, 4a61d72): an option that takes an
// argument calls need(), which fails when argv has nothing left and otherwise takes the next token whatever it is;
// a boolean takes none; anything else, such as a stray value, is "unknown option". This is what rejects a value
// given to a boolean option and a bare flag given to an option that needs an argument.
static bool parse_like_radiance(int argc, char** argv, std::map<std::string, std::string>* given) {
    static const char* takes_arg[] = {"--model", "--radiance-home", "--kernels", "--max-num-seqs", "--tp-wire"};
    static const char* bare[] = {"--live", "--prefix-cache", "-v", "-vv"};
    for (int i = 1; i < argc; ++i) {
        bool matched = false;
        for (const char* n : takes_arg)
            if (!std::strcmp(argv[i], n)) {
                if (i + 1 >= argc) return false;   // "needs an argument"
                (*given)[n] = "command line";
                ++i;
                matched = true;
            }
        for (const char* n : bare)
            if (!matched && !std::strcmp(argv[i], n)) {
                (*given)[n] = "command line";
                matched = true;
            }
        if (!matched) return false;                // "unknown option"
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
    return "{\"schema_version\":" + schema + ",\"values\":{\"service.max_sessions\":4," + engine_values +
           "},\"process_helpers\":[],\"hash_algorithm\":\"sha256\",\"content_sha256\":\"" + kSha + "\"}";
}

static const std::string kBase = "\"engine.radiance_home\":\"/h\",\"engine.kernels\":\"libr4d,libref\"";

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
    // without --manifest argv is untouched
    CHECK(run({"--model", "x", "--tp-wire", "exact"}) == "engine --model x --tp-wire exact");

    // sorted keys, integers and strings, booleans
    const std::string ok = write("ok.json", manifest(kBase + ",\"engine.max_num_seqs\":6,\"engine.tp_wire\":\"wht6\","
                                                              "\"engine.live\":true,\"engine.prefix_cache\":false,"
                                                              "\"engine.model\":\"/m.rad\""));
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
    CHECK(run({"--manifest", write("nosha.json", "{\"schema_version\":1,\"values\":{" + kBase + "},"
                                                  "\"hash_algorithm\":\"sha256\"}")}) == "ERROR");

    // an unknown engine key, and one that is not an engine option at all
    CHECK(run({"--manifest", write("unk.json", manifest(kBase + ",\"engine.no_such_thing\":1"))}) == "ERROR");
    CHECK(run({"--manifest", write("man.json", manifest(kBase + ",\"engine.manifest\":\"x\""))}) == "ERROR");
    CHECK(run({"--manifest", write("help.json", manifest(kBase + ",\"engine.help\":true"))}) == "ERROR");

    // the argv shape follows the JSON type alone; shape mismatches are radiance's config_parse to reject
    CHECK(run({"--manifest", ok}) != "ERROR" && g_parsed);
    CHECK(run({"--manifest", write("fl.json", manifest(kBase + ",\"engine.max_num_seqs\":2.5"))}) ==
          "engine --kernels libr4d,libref --max-num-seqs 2.5 --radiance-home /h");
    CHECK(run({"--manifest", write("big.json", manifest(kBase + ",\"engine.max_num_seqs\":4294967296"))}) ==
          "engine --kernels libr4d,libref --max-num-seqs 4294967296 --radiance-home /h");
    // a value for a boolean option: emitted as "--live yes", which radiance rejects as an unknown option
    CHECK(run({"--manifest", write("b2.json", manifest(kBase + ",\"engine.live\":\"yes\""))}) ==
          "engine --kernels libr4d,libref --live yes --radiance-home /h");
    CHECK(!g_parsed);
    CHECK(run({"--manifest", write("b3.json", manifest(kBase + ",\"engine.live\":1"))}) != "ERROR" && !g_parsed);
    // true for an option that needs an argument: a bare "--max-num-seqs", which radiance rejects
    CHECK(run({"--manifest", write("b1.json", manifest(kBase + ",\"engine.max_num_seqs\":true"))}) ==
          "engine --kernels libr4d,libref --radiance-home /h --max-num-seqs");
    CHECK(!g_parsed);
    // false adds no flag
    CHECK(run({"--manifest", write("f.json", manifest(kBase + ",\"engine.live\":false"))}) ==
          "engine --kernels libr4d,libref --radiance-home /h");
    // null, arrays, objects are errors here
    CHECK(run({"--manifest", write("n.json", manifest(kBase + ",\"engine.max_num_seqs\":null"))}) == "ERROR");
    CHECK(run({"--manifest", write("a.json", manifest(kBase + ",\"engine.model\":[1]"))}) == "ERROR");
    CHECK(run({"--manifest", write("o.json", manifest(kBase + ",\"engine.model\":{}"))}) == "ERROR");

    // RADIANCE_HOME / RADIANCE_KERNELS must not be able to fill in for the manifest
    CHECK(run({"--manifest", write("nohome.json", manifest("\"engine.kernels\":\"libref\""))}) == "ERROR");
    CHECK(run({"--manifest", write("nokern.json", manifest("\"engine.radiance_home\":\"/h\""))}) == "ERROR");
    CHECK(run({"--manifest", write("emptyk.json", manifest("\"engine.radiance_home\":\"/h\",\"engine.kernels\":\"\""))}) == "ERROR");
    CHECK(run({"--manifest", write("emptyh.json", manifest("\"engine.radiance_home\":\"\",\"engine.kernels\":\"libref\""))}) == "ERROR");

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

    // a bare flag for an option that needs an argument swallows the next flag: "--max-num-seqs --prefix-cache"
    // parses, but --prefix-cache was not parsed as given, so start-up fails
    CHECK(run({"--manifest", write("sw.json", manifest(kBase + ",\"engine.max_num_seqs\":true,"
                                                                "\"engine.prefix_cache\":true"))}) ==
          "engine --kernels libr4d,libref --radiance-home /h --max-num-seqs --prefix-cache");
    CHECK(g_parsed);
    CHECK(!bc_serving_mark_given(&g_given));
    CHECK(g_given["--max-num-seqs"] == "command line");   // not relabelled on failure

    std::printf(g_failed ? "FAILED\n" : "ok\n");
    return g_failed ? 1 : 0;
}
