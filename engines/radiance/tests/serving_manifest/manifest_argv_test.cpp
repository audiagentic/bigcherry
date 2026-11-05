// Card-free test of the patched engine's manifest-to-argv logic (core/bc_serving_manifest.cpp from rad0009): a serving
// manifest v1 read by the shared strict reader, its registry-bound fields as flags.
// Built by tools/tests/patch/test_radiance_patch_packages.py against radiance's real option table and a mirror of config_parse's shape rules.
// usage: manifest_argv_test SCRATCH_DIR V1_EXAMPLE_MANIFEST OPTIONS_FILE
#include "bc_serving_manifest.h"

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

int main(int, char** argv) {
    g_dir = argv[1];
    load_options(argv[3]);
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
    for (const auto& kv : g_given) CHECK(kv.second == "manifest e2e8066eafd8");   // "manifest <first 12 hex of the digest>"
    CHECK(run({"--manifest", example, "--role", "engine", "-v"}).rfind("engine -v --model ", 0) == 0);
    CHECK(run({"--manifest", example}) == got);   // deterministic

    // without --manifest argv is untouched
    CHECK(run({"--model", "x", "--tp-wire", "exact"}) == "engine --model x --tp-wire exact");

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
