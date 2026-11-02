#ifdef NDEBUG
#undef NDEBUG // Assertions are test requirements even in RelWithDebInfo.
#endif
#include "serving.h"

#undef NDEBUG   /* the checks below are assert()s: keep them live in Release component builds */
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
using bc::serving::Json;
using bc::serving::Options;
using bc::serving::Resolved;

void write(const fs::path& path, const std::string& content) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot create fixture file");
    out << content;
    if (!out) throw std::runtime_error("cannot write fixture file");
}

struct Fixture {
    fs::path root;
    Json facts;

    Fixture() : root(fs::current_path() / "bc-serving-fixture") {
        // This test only writes in its own CTest build directory.
        fs::remove_all(root);
        const auto nodes = root / "sys/class/kfd/kfd/topology/nodes";
        write(nodes / "0/properties", "gfx_target_version 0\n");
        // KFD node order, not BDF sort or device names, defines HIP ordinals.
        const std::vector<int> versions = {110000, 110000, 120001, 100300};
        const std::vector<int> buses = {3, 6, 9, 23};
        const std::vector<int> simds = {192, 192, 128, 160};
        const std::vector<int> vram = {24560, 24560, 32624, 16368};
        for (int i = 0; i != 4; ++i) {
            const auto node = nodes / std::to_string(i + 1) / "properties";
            write(node, "gfx_target_version " + std::to_string(versions[i]) +
                        "\nsimd_count " + std::to_string(simds[i]) +
                        "\nlocation_id " + std::to_string(buses[i] * 256) +
                        "\ndomain 0\ndrm_render_minor " + std::to_string(128 + i) +
                        "\n");
            char bdf[24];
            std::snprintf(bdf, sizeof(bdf), "0000:%02x:00.0", buses[i]);
            const auto pci = root / "sys/bus/pci/devices" / bdf;
            write(pci / "mem_info_vram_total", std::to_string(uint64_t(vram[i]) * 1048576));
            write(pci / "mem_info_vram_used", std::to_string(uint64_t(128) * 1048576));
            write(pci / "current_link_speed", (i == 2 ? "32.0" : "16.0") +
                  std::string(" GT/s PCIe\n"));
            write(pci / "max_link_speed", (i == 2 ? "32.0" : "16.0") +
                  std::string(" GT/s PCIe\n"));
            write(pci / "current_link_width", "16\n");
            write(pci / "max_link_width", "16\n");
            write(pci / "numa_node", "0\n");
        }
        write(root / "proc/meminfo",
              "MemTotal:       134217728 kB\n"
              "MemAvailable:    94371840 kB\n"
              "SwapTotal:       8388608 kB\n");
        fs::create_directories(root / "dev/shm");
        facts = bc::serving::detect(root);
    }
    ~Fixture() { fs::remove_all(root); }
    fs::path file(const char* name, const std::string& content) {
        const auto path = root / name;
        write(path, content);
        return path;
    }
    Options standard() {
        Options opts;
        opts.site = file("site.toml", "[engine]\nmodel = \"/model.rad\"\n"
                                      "radiance_home = \"/runtime\"\n"
                                      "[handoff]\nheat = \"/heat.hot-first\"\n"
                                      "histogram = \"/heat.live\"\n");
        return opts;
    }
};

void fail(const Options& opts, const Json& facts, const std::string& needle) {
    try {
        (void)bc::serving::resolve(opts, facts);
    } catch (const std::exception& e) {
        if (std::string(e.what()).find(needle) == std::string::npos) {
            throw std::runtime_error("wrong diagnostic for " + needle + ": " + e.what());
        }
        return;
    }
    throw std::runtime_error("expected rejection containing " + needle);
}

void flatten_keys(const Json& j, const std::string& prefix,
                  std::vector<std::string>& keys) {
    if (!j.is_object()) { keys.push_back(prefix); return; }
    for (auto it = j.begin(); it != j.end(); ++it)
        flatten_keys(it.value(), prefix.empty() ? it.key() : prefix + "." + it.key(),
                     keys);
}

void detection(const Fixture& t) {
    auto g = t.facts.at("gpus");
    assert(g.size() == 4);
    assert(g[0].at("arch") == "gfx1100");
    assert(g[1].at("arch") == "gfx1100");
    assert(g[2].at("arch") == "gfx1201");
    assert(g[3].at("arch") == "gfx1030");
    assert(g[0].at("hip_ordinal") == 0 && g[3].at("hip_ordinal") == 3);
    assert(g[0].at("vram_total_mib") == 24560);
    assert(g[2].at("vram_total_mib") == 32624);
    assert(g[3].at("vram_total_mib") == 16368);
    assert(g[2].at("bdf") == "0000:09:00.0");
    assert(g[2].at("current_link_speed") == "32.0 GT/s PCIe");
    assert(g[2].at("current_link_width") == 16);
    assert(g[0].at("simd_count") == 192);
    assert(t.facts.at("host").at("mem_total_mib") == 131072);
    assert(t.facts.at("host").at("mem_available_mib") == 92160);
}

void unknown_keys(Fixture& t) {
    auto opts = t.standard();
    opts.site = t.file("unknown-site.toml",
                       "[engine]\nmodel = \"/model.rad\"\n"
                       "radiance_home = \"/runtime\"\n"
                       "max_numm_seqs = 7\n");
    fail(opts, t.facts, "unknown-site.toml: unknown key engine.max_numm_seqs");
    fail(opts, t.facts, "engine.max_num_seqs");
    opts = t.standard();
    opts.profile = t.file("unknown-profile.toml", "[service]\nmax_sessionz = 7\n");
    fail(opts, t.facts, "unknown-profile.toml: unknown key service.max_sessionz");
    opts = t.standard();
    opts.sets = {"engine.potr=1234"};
    fail(opts, t.facts, "nearest valid key: engine.port");
}

void type_range_required(Fixture& t) {
    Options opts;
    fail(opts, t.facts, "missing required key: engine.model");
    opts = t.standard();
    opts.profile = t.file("type.toml", "[service]\nmax_sessions = \"many\"\n");
    fail(opts, t.facts, "service.max_sessions: expected non-negative integer");
    opts.profile = t.file("range.toml", "[helper]\ncombine_threads = 0\n");
    fail(opts, t.facts, "helper.combine_threads: range 1..1024");
    opts.profile = t.file("enum.toml", "[service]\nobjective = \"fast\"\n");
    fail(opts, t.facts, "service.objective: invalid enum value");
    opts.profile = t.file("badarr.toml", "[memory]\nhelper_budgets_mib = [3, \"x\"]\n");
    fail(opts, t.facts, "memory.helper_budgets_mib: expected non-negative integer");
    opts.profile = t.file("badcount.toml", "[helper]\ncount = 1\nindex = 2\n");
    fail(opts, t.facts, "helper.index must be below helper.count");
    opts.profile = t.file("invalidlayout.toml", "[layout]\nname = \"not-a-layout\"\n");
    fail(opts, t.facts, "unknown named layout");
    opts.profile = t.file("derived.toml", "[engine]\nmax_num_seqs = 99\n");
    fail(opts, t.facts, "derived and cannot be overridden");
}

void precedence_and_caps(Fixture& t) {
    auto opts = t.standard();
    opts.site = t.file("precedence-site.toml",
        "[engine]\nmodel = \"/model.rad\"\nradiance_home = \"/runtime\"\nport = 4700\n"
        "[handoff]\nheat = \"/heat.hot-first\"\nhistogram = \"/heat.live\"\n");
    opts.profile = t.file("precedence-profile.toml", "[engine]\nport = 4800\n");
    opts.sets = {"engine.port=4900"};
    const auto r = bc::serving::resolve(opts, t.facts);
    assert(r.typed.engine.port == 4900);
    assert(r.provenance.at("engine.port").at("origin") == "cli");
    opts.sets.clear();
    const auto p = bc::serving::resolve(opts, t.facts);
    assert(p.typed.engine.port == 4800);
    opts.profile.clear();
    const auto s = bc::serving::resolve(opts, t.facts);
    assert(s.typed.engine.port == 4700);
    opts.site = t.standard().site;
    const auto d = bc::serving::resolve(opts, t.facts);
    assert(d.typed.engine.port == 36080);

    opts.site = t.file("cap-site.toml",
        "[engine]\nmodel = \"/model.rad\"\nradiance_home = \"/runtime\"\n"
        "[memory]\nhost_pool_mib = 7000\n"
        "[hard_caps.memory]\nhost_pool_mib = 8000\n"
        "[handoff]\nheat = \"/heat.hot-first\"\nhistogram = \"/heat.live\"\n");
    auto good = bc::serving::resolve(opts, t.facts);
    assert(good.typed.memory.host_pool_mib == 7000);
    opts.profile = t.file("cap-profile.toml", "[memory]\nhost_pool_mib = 9000\n");
    fail(opts, t.facts, "hard cap refusal: memory.host_pool_mib");
    opts.profile = t.file("cap-illegal.toml", "[hard_caps.memory]\nhost_pool_mib = 9000\n");
    fail(opts, t.facts, "profile cannot change hard caps");
    opts.profile.clear();
    opts.sets = {"hard_caps.memory.host_pool_mib=9000"};
    fail(opts, t.facts, "hard caps are site-only");
    opts.sets.clear();
    opts.site = t.file("overcap-site.toml",
        "[engine]\nmodel = \"/model.rad\"\nradiance_home = \"/runtime\"\n"
        "[hard_caps.memory]\nhost_pool_mib = 999999\n"
        "[handoff]\nheat = \"/heat.hot-first\"\nhistogram = \"/heat.live\"\n");
    fail(opts, t.facts, "exceeds detected RAM");
}

void provenance_hash_and_parity(Fixture& t) {
    auto opts = t.standard();
    opts.sets = {"layout.name=flashnext-2xtx-rdna2-tail"};
    const auto r = bc::serving::resolve(opts, t.facts);
    const auto again = bc::serving::resolve(opts, t.facts);
    assert(r.hash == again.hash);
    assert(r.manifest.dump() == again.manifest.dump());
    Json unhashed = r.manifest;
    unhashed.erase("content_sha256");
    assert(bc::serving::sha256_hex(unhashed.dump()) == r.hash);
    std::vector<std::string> keys;
    flatten_keys(r.values, "", keys);
    assert(keys.size() == r.provenance.size());
    for (const auto& key : keys) {
        assert(r.provenance.contains(key));
        const auto& pv = r.provenance.at(key);
        for (const char* field : {"origin", "source", "inputs", "lifecycle",
                                  "class", "confidence"})
            assert(pv.contains(field));
        assert(pv.at("inputs").is_array());
    }

    // RSC01 launcher-equivalence golden. Values transcribed from the
    // 2026-11-02 launch-flashnext-best.sh engine argv and COMMON/WorkerConfig.
    // RSC02 deletes the launcher and replaces this with a manifest golden.
    const auto& e = r.values.at("engine");
    const std::vector<std::pair<std::string, Json>> expected_engine = {
        {"model", "/model.rad"}, {"radiance_home", "/runtime"}, {"port", 36080},
        {"kernels", "libr4d,libr3moe,libref"}, {"placement", "expert_tiered"},
        {"ngram_placement", "ram"}, {"host_pool_mib", 9216},
        {"prefix_cache_mib", 12288}, {"tp", 1}, {"max_num_seqs", 8},
        {"max_num_batched_tokens", 2048}, {"checkpoint_interval", 4096},
        {"kv_cache_dtype", "fp8"}, {"num_speculative_tokens", 3},
        {"max_model_len", 262144}, {"rope_scaling", "none"},
        {"prefix_resume", "nearest"}
    };
    for (const auto& [name, value] : expected_engine)
        if (e.at(name) != value)
            throw std::runtime_error("launcher engine parity failed: " + name);
    // The launcher COMMON values; paths are required rather than guessed.
    const auto& a = r.values.at("handoff");
    const std::vector<std::pair<std::string, Json>> expected_common = {
        {"shm", "/r3-moe-helper"}, {"heat", "/heat.hot-first"},
        {"histogram", "/heat.live"}, {"histogram_period_s", 60},
        {"max_rows", 4096}, {"small_rows", 8}, {"gemv_large_rows", 32},
        {"moe_down_tile", true}, {"skip_owned", true},
        {"placement", "tail"}
    };
    for (const auto& [key, val] : expected_common)
        if (a.at(key) != val)
            throw std::runtime_error("launcher COMMON parity failed: " + key);
    const auto& h = r.values.at("helper");
    const std::vector<std::pair<std::string, Json>> expected_helper = {
        {"count", 3}, {"index", 0}, {"engine_ranks", 1},
        {"cpu", false}, {"power_save", true}, {"queue_warm", false},
        {"combine", false}, {"combine_min_rows", 256},
        {"combine_threads", 8}, {"cpu_set_auto", true}
    };
    for (const auto& [name, value] : expected_helper)
        if (h.at(name) != value)
            throw std::runtime_error("launcher WorkerConfig parity failed: " + name);
    assert(r.typed.memory.helper_budgets_mib ==
           (std::vector<uint64_t>{22528, 22528, 2048}));
    assert(r.typed.memory.tail_budget_mib == 2048);
    assert(r.manifest.at("process_helpers").size() == 3);
    const auto& helpers = r.manifest.at("process_helpers");
    assert(helpers[0]["plugin_file"] == "libr3.so");
    assert(helpers[1]["formats"].size() == 2);
    assert(helpers[2]["plugin_file"] == "libr2.so");
    assert(helpers[2]["rows_kernel"] == true);
    assert(helpers[2]["placement"] == "tail");
    assert(helpers[2]["share"] == 0);

    for (size_t i = 0; i < 3; ++i)
        assert(r.manifest.at("process_helpers")[i]["config"]["index"] == i);
    assert(r.typed.layout.engine_card == "0000:09:00.0");
    assert(r.typed.layout.helper_cards[2] == "0000:17:00.0");

    const auto schema = bc::serving::json_schema();
    assert(schema["properties"]["helper"]["properties"].size() == 10);
    assert(schema["properties"]["engine"]["properties"]["port"]["maximum"] == 65535);
    assert(schema["properties"]["service"]["properties"]["objective"]["enum"].size() == 3);
    assert(schema["properties"]["memory"]["properties"]["host_pool_mib"].contains("x-origin"));
    assert(schema["properties"]["engine"]["required"].size() == 2);
    assert(schema["properties"]["handoff"]["required"].size() == 2);
    assert(schema["properties"]["helper"]["properties"]["index"]["maximum"] == 3);
}
} // namespace

static void sha256_vectors() {   /* FIPS 180-4 examples, incl. the two-block 56-byte boundary */
    using bc::serving::sha256_hex;
    assert(sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
           "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

int main() {
    try {
        sha256_vectors();
        Fixture t;
        detection(t);
        unknown_keys(t);
        type_range_required(t);
        precedence_and_caps(t);
        provenance_hash_and_parity(t);
        std::cout << "PASS: fake KFD/PCI/proc detection; strict keys; types/ranges; required; "
                     "precedence/caps; provenance/hash; launcher engine/WorkerConfig parity\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
