#ifdef NDEBUG
#undef NDEBUG // Assertions are test requirements even in RelWithDebInfo.
#endif
#include "../moe-remote/heat_publication.h"
#include "calibration.h"
#include "config_loader.h"
#include "experts.h"
#include "ledger.h"
#include "live_policy.h"
#include "live_policy_ingest.h"
#include "manifest_writer.h"
#include "plan_cache.h"
#include "resources.h"
#include "runtime_guards.h"
#include "s8_replay.h"
#include "sdk_check.h"
#include "serving.h"
#include "session.h"
#include "sha256.h"
#include "stored_manifest_cache.h"
#include "test_manifest_fixture.h"
#include "test_rad_fixture.h"
#include "v2/helper_settings.h"
#include "v2/manifest.h"
#include "v2/p1_producer.h"
#include "worker_footprint.h"

#undef NDEBUG /* the checks below are assert()s: keep them live in Release component builds */
#include <arpa/inet.h>
#include <csignal>
#include <cstdio>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <cerrno>
#include <optional>
#ifdef NDEBUG
#error "assert-based test built with NDEBUG: its assertions would not run"
#endif
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
namespace fs = std::filesystem;
using bc::serving::Json;
using bc::serving::Options;
using bc::serving::Resolved;
Json g_session_manifest_template = Json::object();

void write(const fs::path& path, const std::string& content) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("cannot create fixture file");
    out << content;
    if (!out)
        throw std::runtime_error("cannot write fixture file");
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
            write(node, "gfx_target_version " + std::to_string(versions[i]) + "\nsimd_count " +
                            std::to_string(simds[i]) + "\nlocation_id " + std::to_string(buses[i] * 256) +
                            "\ndomain 0\ndrm_render_minor " + std::to_string(128 + i) + "\n");
            char bdf[24];
            std::snprintf(bdf, sizeof(bdf), "0000:%02x:00.0", buses[i]);
            const auto pci = root / "sys/bus/pci/devices" / bdf;
            write(pci / "mem_info_vram_total", std::to_string(uint64_t(vram[i]) * 1048576));
            write(pci / "mem_info_vram_used", std::to_string(uint64_t(128) * 1048576));
            write(pci / "current_link_speed", (i == 2 ? "32.0" : "16.0") + std::string(" GT/s PCIe\n"));
            write(pci / "max_link_speed", (i == 2 ? "32.0" : "16.0") + std::string(" GT/s PCIe\n"));
            write(pci / "current_link_width", "16\n");
            write(pci / "max_link_width", "16\n");
            write(pci / "numa_node", "0\n");
        }
        write(root / "proc/meminfo", "MemTotal:       134217728 kB\n"
                                     "MemAvailable:    94371840 kB\n"
                                     "SwapTotal:       8388608 kB\n");
        fs::create_directories(root / "dev/shm");
        // A v2 cgroup with a limit at the leaf and none above; memlock limited, no CAP_IPC_LOCK;
        // amdgpu loaded with its system memory limit in force.
        write(root / "proc/self/cgroup", "0::/agents/serve\n");
        write(root / "sys/fs/cgroup/agents/serve/memory.max", std::to_string(uint64_t(64) << 30U) + "\n");
        write(root / "sys/fs/cgroup/agents/serve/memory.current", std::to_string(uint64_t(10) << 30) + "\n");
        write(root / "sys/fs/cgroup/agents/serve/memory.stat",
              "anon 1\nfile " + std::to_string(uint64_t(4) << 30) + "\n");
        write(root / "sys/fs/cgroup/agents/memory.max", "max\n");
        write(root / "sys/fs/cgroup/memory.max", "max\n");
        write(root / "proc/self/limits",
              "Limit                     Soft Limit           Hard Limit           Units\n"
              "Max locked memory         8388608              8388608              bytes\n");
        write(root / "proc/self/status", "Name:\tbc-serve\nCapEff:\t00000000a80425fb\n");
        write(root / "sys/module/amdgpu/parameters/no_system_mem_limit", "N\n");
        bc::test::write_rad(root / "model.rad", 4, 16, 128, 128, {});
        facts = bc::serving::detect(root);
        // detect() counts this machine's CPUs; the fixture fixes the count so the result does not depend on the
        // runner.
        facts["host"]["cpu_count"] = 32;
    }
    ~Fixture() {
        fs::remove_all(root);
    }
    fs::path file(const char* name, const std::string& content) {
        const auto path = root / name;
        write(path, content);
        return path;
    }
    Options standard() {
        Options opts;
        opts.site = file("site.toml", "[engine]\nmodel = \"" + (root / "model.rad").string() +
                                          "\"\n"
                                          "radiance_home = \"/runtime\"\n"
                                          "[handoff]\nheat = \"/heat.hot-first\"\n"
                                          "histogram = \"/heat.live\"\n"
                                          "[layout]\nname = \"flashnext-4gpu\"\n");
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

void flatten_keys(const Json& j, const std::string& prefix, std::vector<std::string>& keys) {
    if (!j.is_object()) {
        keys.push_back(prefix);
        return;
    }
    for (auto it = j.begin(); it != j.end(); ++it)
        flatten_keys(it.value(), prefix.empty() ? it.key() : prefix + "." + it.key(), keys);
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
    // cgroup: 64 GiB limit less (10 GiB current - 4 GiB file pages) = 58 GiB; "max" levels skipped.
    const Json cg = t.facts.at("host").at("memory_cgroups");
    assert(cg.size() == 1);
    assert(cg[0].at("dir") == "/sys/fs/cgroup/agents/serve" && cg[0].at("version") == 2);
    assert(cg[0].at("room_mib") == 58 * 1024);
    // memlock 8 MiB soft limit; CapEff bit 14 (CAP_IPC_LOCK) clear in a80425fb.
    assert(t.facts.at("host").at("memlock").at("limit_mib") == 8);
    assert(t.facts.at("host").at("memlock").at("cap_ipc_lock") == false);
    // amdgpu pin cap on 128 GiB: 128 GiB - 2 GiB (1/64) - 1.5 GiB.
    const Json pin = t.facts.at("host").at("amdgpu_pin");
    assert(pin.at("loaded") == true && pin.at("lifted") == false);
    assert(pin.at("cap_mib") == 131072 - 2048 - 1536);
}

void detection_limits(const Fixture& t) {
    // Unlimited memlock, CAP_IPC_LOCK held, the pin cap lifted, a v1 memory controller.
    const fs::path root = fs::current_path() / "bc-serving-limits";
    fs::remove_all(root);
    fs::copy(t.root, root, fs::copy_options::recursive);
    write(root / "proc/self/cgroup", "4:cpu,memory:/job\n1:name=systemd:/\n");
    write(root / "sys/fs/cgroup/memory/job/memory.limit_in_bytes", std::to_string(uint64_t(32) << 30));
    write(root / "sys/fs/cgroup/memory/job/memory.usage_in_bytes", std::to_string(uint64_t(3) << 30));
    write(root / "sys/fs/cgroup/memory/job/memory.stat", "cache 1073741824\ntotal_cache 2147483648\n");
    write(root / "sys/fs/cgroup/memory/memory.limit_in_bytes", "9223372036854771712");
    write(root / "proc/self/limits",
          "Max locked memory         unlimited            unlimited            bytes\n");
    write(root / "proc/self/status", "CapEff:\t0000000000004000\n");
    write(root / "sys/module/amdgpu/parameters/no_system_mem_limit", "Y\n");
    const Json h = bc::serving::detect(root).at("host");
    assert(h.at("memory_cgroups").size() == 1);
    assert(h.at("memory_cgroups")[0].at("version") == 1);
    assert(h.at("memory_cgroups")[0].at("room_mib") == 31 * 1024); // 32 GiB - (3 - 2 total_cache)
    assert(h.at("memlock").at("limit_mib").is_null() && h.at("memlock").at("cap_ipc_lock") == true);
    assert(h.at("amdgpu_pin").at("lifted") == true && h.at("amdgpu_pin").at("cap_mib").is_null());
    fs::remove(root / "proc/self/limits");
    bool refused = false;
    try {
        (void)bc::serving::detect(root);
    } catch (const std::exception& e) {
        refused = std::string(e.what()).find("proc/self/limits") != std::string::npos;
    }
    assert(refused);
    fs::remove_all(root);
}

void unknown_keys(Fixture& t) {
    auto opts = t.standard();
    opts.site = t.file("unknown-site.toml", "[engine]\nmodel = \"" + (t.root / "model.rad").string() +
                                                "\"\n"
                                                "radiance_home = \"/runtime\"\n"
                                                "max_numm_seqs = 7\n"
                                                "[layout]\nname = \"flashnext-4gpu\"\n");
    fail(opts, t.facts, "unknown-site.toml: unknown key engine.max_numm_seqs");
    fail(opts, t.facts, "engine.max_num_seqs");
    opts = t.standard();
    opts.profile = t.file("unknown-profile.toml", "[service]\nmax_sessionz = 7\n");
    fail(opts, t.facts, "unknown-profile.toml: unknown key service.max_sessionz");
    opts = t.standard();
    opts.sets = {"engine.potr=1234"};
    fail(opts, t.facts, "nearest valid key: engine.port");
}

/* The layouts are run-time data: no default layout, no built-in table, and a name defined twice is refused. */
void layout_sources(Fixture& t) {
    auto opts = t.standard();
    opts.site = t.file("nolayout-site.toml", "[engine]\nmodel = \"" + (t.root / "model.rad").string() +
                                                 "\"\n"
                                                 "radiance_home = \"/runtime\"\n");
    fail(opts, t.facts, "layout.name is required");
    opts = t.standard();
    opts.layouts = {t.root / "no-such-layouts"};
    fail(opts, t.facts, "neither a layout file nor a directory");
    const std::string one = "schema = 1\n[layout.solo]\nmodel = \"m.rad\"\nengine = { arch = \"gfx1201\" }\n";
    opts.layouts = {t.file("a.toml", one), t.file("b.toml", one)};
    opts.sets = {"layout.name=\"solo\""};
    fail(opts, t.facts, "layout solo is defined in both");
    opts.layouts = {t.file("noschema.toml", "[layout.solo]\nengine = { arch = \"gfx1201\" }\n")};
    fail(opts, t.facts, "schema = 1 is required");
    const Json all =
        bc::serving::load_layouts(Options{}); // BIGCHERRY_LAYOUTS: the repository's example layouts
    assert(all.at("layout").contains("flashnext-4gpu") && all.at("layout").contains("qwen27b-2xtx"));
}

void type_range_required(Fixture& t) {
    Options opts;
    opts.sets = {"layout.name=\"flashnext-4gpu\""};
    fail(opts, t.facts, "missing required key: engine.model");
    opts.sets.clear();
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
    { // the standard profile's 8 combine threads on a 4-CPU host
        Json few = t.facts;
        few["host"]["cpu_count"] = 4;
        fail(t.standard(), few, "helper.combine_threads exceeds discovered CPU count");
    }
}

void precedence_and_caps(Fixture& t) {
    auto opts = t.standard();
    opts.site = t.file("precedence-site.toml",
                       "[engine]\nmodel = \"" + (t.root / "model.rad").string() +
                           "\"\nradiance_home = \"/runtime\"\nport = 4700\n"
                           "[handoff]\nheat = \"/heat.hot-first\"\nhistogram = \"/heat.live\"\n" +
                           std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
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

    opts.site =
        t.file("cap-site.toml", "[engine]\nmodel = \"" + (t.root / "model.rad").string() +
                                    "\"\nradiance_home = \"/runtime\"\n"
                                    "[memory]\nhost_pool_mib = 7000\n"
                                    "[hard_caps.memory]\nhost_pool_mib = 8000\n"
                                    "[handoff]\nheat = \"/heat.hot-first\"\nhistogram = \"/heat.live\"\n" +
                                    std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
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
    opts.site =
        t.file("overcap-site.toml", "[engine]\nmodel = \"" + (t.root / "model.rad").string() +
                                        "\"\nradiance_home = \"/runtime\"\n"
                                        "[hard_caps.memory]\nhost_pool_mib = 999999\n"
                                        "[handoff]\nheat = \"/heat.hot-first\"\nhistogram = \"/heat.live\"\n" +
                                        std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
    fail(opts, t.facts, "exceeds detected RAM");
}

void provenance_hash_and_parity(Fixture& t) {
    auto opts = t.standard();
    opts.sets = {"layout.name=flashnext-2xtx-rdna2-tail"};
    const auto r = bc::serving::resolve(opts, t.facts);
    const auto again = bc::serving::resolve(opts, t.facts);
    assert(r.hash == again.hash);
    assert(r.manifest.dump() == again.manifest.dump());
    assert(bc::serving::sha256_hex(r.manifest.dump()) == r.hash);
    assert(!r.manifest.contains("content_sha256") &&
           !r.manifest.contains("hash_algorithm")); // no second manifest format
    std::vector<std::string> keys;
    flatten_keys(r.values, "", keys);
    assert(keys.size() == r.provenance.size());
    for (const auto& key : keys) {
        assert(r.provenance.contains(key));
        const auto& pv = r.provenance.at(key);
        for (const char* field : {"origin", "source", "inputs", "lifecycle", "class", "confidence"})
            assert(pv.contains(field));
        assert(pv.at("inputs").is_array());
    }

    // RSC01 launcher-equivalence golden. Values transcribed from the
    // 2026-11-02 launch-flashnext-best.sh engine argv and COMMON/WorkerConfig.
    // RSC02 deletes the launcher and replaces this with a manifest golden.
    const auto& e = r.values.at("engine");
    const std::vector<std::pair<std::string, Json>> expected_engine = {
        {"model", (t.root / "model.rad").string()},
        {"radiance_home", "/runtime"},
        {"port", 36080},
        {"kernels", "libr4d,libr3moe,libref"},
        {"placement", "expert_tiered"},
        {"ngram_placement", "ram"},
        {"host_pool_mib", 9216},
        {"prefix_cache_mib", 12288},
        {"tp", 1},
        {"max_num_seqs", 8},
        {"max_num_batched_tokens", 2048},
        {"checkpoint_interval", 4096},
        {"kv_cache_dtype", "fp8"},
        {"num_speculative_tokens", 3},
        {"max_model_len", 262144},
        {"rope_scaling", "none"},
        {"prefix_resume", "nearest"}};
    for (const auto& [name, value] : expected_engine)
        if (e.at(name) != value)
            throw std::runtime_error("launcher engine parity failed: " + name);
    // The launcher COMMON values; paths are required rather than guessed.
    const auto& a = r.values.at("handoff");
    const std::vector<std::pair<std::string, Json>> expected_common = {
        {"shm", "/r3-moe-helper"},  {"heat", "/heat.hot-first"}, {"histogram", "/heat.live"},
        {"histogram_period_s", 60}, {"max_rows", 4096},          {"small_rows", 8},
        {"gemv_large_rows", 32},    {"moe_down_tile", true},     {"skip_owned", true},
        {"placement", "tail"}};
    for (const auto& [key, val] : expected_common)
        if (a.at(key) != val)
            throw std::runtime_error("launcher COMMON parity failed: " + key);
    const auto& h = r.values.at("helper");
    const std::vector<std::pair<std::string, Json>> expected_helper = {
        {"count", 3},           {"index", 0},          {"engine_ranks", 1}, {"cpu", false},
        {"power_save", true},   {"queue_warm", false}, {"combine", false},  {"combine_min_rows", 256},
        {"combine_threads", 8}, {"cpu_set_auto", true}};
    for (const auto& [name, value] : expected_helper)
        if (h.at(name) != value)
            throw std::runtime_error("launcher WorkerConfig parity failed: " + name);
    const auto& derived_budgets = r.typed.memory.helper_budgets_mib;
    assert(derived_budgets.size() == 3);
    for (size_t i = 0; i < derived_budgets.size(); ++i)
        assert(derived_budgets[i] > 0 &&
               derived_budgets[i] < t.facts.at("gpus")[i < 2 ? i : 3].at("vram_total_mib").get<uint64_t>());
    assert(r.typed.memory.tail_budget_mib == derived_budgets.back());
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
    assert(r.typed.layout.engine_cards == std::vector<std::string>{"0000:09:00.0"});
    assert(r.typed.layout.helper_cards[2] == "0000:17:00.0");

    const auto schema = bc::serving::json_schema();
    assert(schema["properties"]["helper"]["properties"].size() == 10);
    assert(schema["properties"]["engine"]["properties"]["port"]["maximum"] == 65535);
    assert(schema["properties"]["service"]["properties"]["objective"]["enum"].size() == 3);
    assert(schema["properties"]["memory"]["properties"]["host_pool_mib"].contains("x-origin"));
    assert(schema["properties"]["engine"]["required"].size() == 2);
    assert(!schema["properties"]["handoff"].contains(
        "required")); // heat and histogram are required by a layout with helpers, not by the schema
    assert(schema["properties"]["helper"]["properties"]["index"]["maximum"] == 3);
}
} // namespace

static void sha256_vectors() { /* FIPS 180-4 examples, incl. the two-block 56-byte boundary */
    using bc::serving::sha256_hex;
    assert(sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
           "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

void calibration_store(Fixture& t) {
    using bc::serving::Calibration;
    using bc::serving::CalibrationMatch;
    const std::string seed =
        "schema_version = 1\n"
        "[[entry]]\nkey = \"calibration.x.v\"\nvalue = 1\nseeded = \"2026-01-01\"\nevidence = \"any\"\n"
        "[[entry]]\nkey = \"calibration.x.v\"\nmatch = { layout = \"L\" }\nvalue = 2\nseeded = \"2026-01-02\"\n"
        "evidence = \"layout L\"\n"
        "[[entry]]\nkey = \"calibration.x.v\"\nmatch = { layout = \"L\", objective = \"latency\" }\nvalue = 3\n"
        "seeded = \"2026-01-03\"\nevidence = \"L latency\"\n";
    const Calibration c = Calibration::load(seed);
    assert(c.integer("calibration.x.v", {"m", "other", "latency"}) == 1);
    assert(c.integer("calibration.x.v", {"m", "L", "throughput"}) == 2);
    assert(c.integer("calibration.x.v", {"m", "L", "latency"}) == 3);
    const auto got = c.get("calibration.x.v", {"m", "L", "latency"});
    assert(got.source == "seed" && got.seeded == "2026-01-03");
    assert(got.describe("calibration.x.v") == "calibrated (seed: 2026-01-03) calibration.x.v: L latency");
    // A cache refit wins at equal specificity, and a less specific cache entry does not beat a narrower seed.
    const auto cache = t.file("refit.toml", "schema_version = 1\n"
                                            "[[entry]]\nkey = \"calibration.x.v\"\nmatch = { layout = \"L\" "
                                            "}\nvalue = 20\nseeded = \"2026-02-01\"\n"
                                            "evidence = \"RHP12 refit\"\n");
    const Calibration r = Calibration::load(seed, cache);
    assert(r.integer("calibration.x.v", {"m", "L", "throughput"}) == 20);
    assert(r.get("calibration.x.v", {"m", "L", "throughput"}).source == "cache");
    assert(r.integer("calibration.x.v", {"m", "L", "latency"}) == 3);
    auto refused = [](const std::function<void()>& f, const std::string& needle) {
        try {
            f();
        } catch (const std::exception& e) {
            if (std::string(e.what()).find(needle) != std::string::npos)
                return;
            throw std::runtime_error("wrong calibration diagnostic for " + needle + ": " + e.what());
        }
        throw std::runtime_error("expected calibration refusal containing " + needle);
    };
    refused([&] { (void)c.get("calibration.y.v", {"m", "L", "latency"}); }, "no entry for calibration.y.v");
    refused(
        [&] {
            (void)Calibration::load("schema_version = 1\n[[entry]]\nkey = \"calibration.a\"\nvalue = 1\n"
                                    "seeded = \"d\"\nevidence = \"e\"\nnote = 1\n");
        },
        "unknown key note");
    refused(
        [&] {
            (void)Calibration::load("schema_version = 1\n[[entry]]\nkey = \"calibration.a\"\nvalue = 1\n"
                                    "seeded = \"d\"\nevidence = \"e\"\nmatch = { host = \"h\" }\n");
        },
        "allowed: model, layout, objective");
    refused(
        [&] {
            (void)Calibration::load("schema_version = 1\n[[entry]]\nkey = \"calibration.a\"\nvalue = 1\n"
                                    "seeded = \"d\"\nevidence = \"\"\n");
        },
        "evidence must be a non-empty");
    refused(
        [&] {
            (void)Calibration::load(seed + "[[entry]]\nkey = \"calibration.x.v\"\nvalue = 9\n"
                                           "seeded = \"d\"\nevidence = \"e\"\n");
        },
        "duplicate entry");
    // The compiled-in seeds parse and carry evidence for every rule parameter the plan reads.
    const Calibration standard = Calibration::standard();
    for (const char* objective : {"latency", "balanced"})
        assert(!standard.get("calibration.kv_backing_floor.fraction_ppm", {"m", "flashnext-4gpu", objective})
                    .evidence.empty());
    refused(
        [&] {
            (void)standard.get("calibration.kv_backing_floor.fraction_ppm",
                               {"m", "flashnext-4gpu", "throughput"});
        },
        "no entry");
    // No floor is selected for throughput: find() says so (the plan check then takes 0), where get() is an
    // error.
    assert(!standard.find("calibration.kv_backing_floor.fraction_ppm", {"m", "flashnext-4gpu", "throughput"}));
    assert(standard.integer("calibration.kv_backing_floor.fraction_ppm", {"m", "flashnext-4gpu", "balanced"}) ==
           600000);
    assert(standard.integer("calibration.kv_backing_floor.fraction_ppm", {"m", "flashnext-4gpu", "latency"}) ==
           500000);
    // v1: a calibration value is an integer, a string or a boolean; a fraction is parts per million.
    refused(
        [&] {
            (void)Calibration::load(seed + "[[entry]]\nkey = \"calibration.y.v\"\nvalue = 0.5\n"
                                           "seeded = \"d\"\nevidence = \"e\"\n");
        },
        "non-integer number");
}

void host_reserve(Fixture& t) {
    // 128 GiB fixture: 5% of 131072 MiB is 6554 MiB, above the 4096 MiB floor.
    auto r = bc::serving::resolve(t.standard(), t.facts);
    assert(r.values.at("memory").at("host_reserve_mib") == 6554);
    assert(r.values.at("engine").at("host_headroom_mib") == 6554);
    assert(r.provenance.at("engine.host_headroom_mib").at("inputs") ==
           Json::array({"memory.host_reserve_mib"}));
    const Json& p = r.provenance.at("memory.host_reserve_mib");
    assert(p.at("origin") == "calibrated");
    assert(p.at("source").get<std::string>().find(
               "calibrated (seed: 2026-11-02) calibration.host_reserve.floor_mib") != std::string::npos);
    assert(p.at("inputs").size() == 3);
    // The site may only raise it.
    auto opts = t.standard();
    opts.site =
        t.file("floor-site.toml", "[engine]\nmodel = \"" + (t.root / "model.rad").string() +
                                      "\"\nradiance_home = \"/runtime\"\n"
                                      "[handoff]\nheat = \"/heat.hot-first\"\nhistogram = \"/heat.live\"\n"
                                      "[hard_floors.memory]\nhost_reserve_mib = 8192\n" +
                                      std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
    r = bc::serving::resolve(opts, t.facts);
    assert(r.values.at("memory").at("host_reserve_mib") == 8192);
    assert(r.provenance.at("memory.host_reserve_mib").at("origin") == "site");
    opts.site =
        t.file("low-floor-site.toml", "[engine]\nmodel = \"" + (t.root / "model.rad").string() +
                                          "\"\nradiance_home = \"/runtime\"\n"
                                          "[handoff]\nheat = \"/heat.hot-first\"\nhistogram = \"/heat.live\"\n"
                                          "[hard_floors.memory]\nhost_reserve_mib = 1024\n" +
                                          std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
    r = bc::serving::resolve(opts, t.facts);
    assert(r.values.at("memory").at("host_reserve_mib") == 6554);
    opts.site =
        t.file("bad-floor-site.toml", "[engine]\nmodel = \"" + (t.root / "model.rad").string() +
                                          "\"\nradiance_home = \"/runtime\"\n"
                                          "[handoff]\nheat = \"/heat.hot-first\"\nhistogram = \"/heat.live\"\n"
                                          "[hard_floors.memory]\nhost_pool_mib = 1024\n" +
                                          std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
    fail(opts, t.facts, "unknown floor hard_floors.memory.host_pool_mib");
    opts = t.standard();
    opts.profile = t.file("floor-profile.toml", "[hard_floors.memory]\nhost_reserve_mib = 8192\n");
    fail(opts, t.facts, "hard_floors: site-only");
    opts = t.standard();
    opts.sets = {"memory.host_reserve_mib=1"};
    fail(opts, t.facts, "derived, cannot override");
    // A refit in the calibration cache moves the reserve, and explain names the cache.
    opts = t.standard();
    opts.calibration_cache =
        t.file("reserve-refit.toml",
               "schema_version = 1\n[[entry]]\nkey = \"calibration.host_reserve.total_fraction_ppm\"\n"
               "match = { layout = \"flashnext-4gpu\" }\nvalue = 100000\nseeded = \"2026-12-01\"\nevidence = "
               "\"refit\"\n");
    r = bc::serving::resolve(opts, t.facts);
    assert(r.values.at("memory").at("host_reserve_mib") == 13108);
    assert(r.provenance.at("memory.host_reserve_mib")
               .at("source")
               .get<std::string>()
               .find("calibrated (cache: 2026-12-01)") != std::string::npos);
}

void host_ledger_plan() {
    using bc::serving::CheckpointStore;
    using bc::serving::HostConsumer;
    using bc::serving::LedgerInputs;
    using bc::serving::Residence;
    constexpr uint64_t MiB = 1048576;
    // The measured host (flashnext-4gpu, 2026-11-02): 94082 MiB RAM, 83020 MiB available, pin cap 91076 MiB.
    Json host = {{"mem_total_mib", 94082},
                 {"mem_available_mib", 83020},
                 {"memory_cgroups", Json::array()},
                 {"memlock", {{"limit_mib", nullptr}, {"cap_ipc_lock", false}}},
                 {"amdgpu_pin", {{"loaded", true}, {"lifted", false}, {"cap_mib", 91076}}}};
    const std::vector<HostConsumer> measured = {
        {"ngram_table", 48824 * MiB, Residence::pinned, "engine plan (weight_footprint)",
         "set engine.ngram_placement = \"disk\""},
        {"host_pool", 7300 * MiB, Residence::pinned, "owner map + weight_footprint", ""},
        {"helpers", 3072 * MiB, Residence::pageable, "helper plan", ""}};
    // 72 snapshots wanted (8064 MiB), a floor of 16 (1792 MiB).
    const CheckpointStore store{112 * MiB, 16, 72, "engine plan (size_checkpoint_store)", "want", "seed"};
    LedgerInputs in{host, 4705, measured, store, true, 4 * 1024 * MiB};
    auto l = bc::serving::host_ledger(in);
    // 83020 - (48824 + 7300 + 3072) - 8064 - 4705 = 11055 MiB, with the store at its want.
    assert(l.checkpoint_slots == 72);
    assert(l.host_kv_mib == 11055);
    assert(l.explain.at("host_kv_why").get<std::string>().find("MemAvailable") != std::string::npos);
    assert(l.explain.at("checkpoints").at("why").get<std::string>().find("calibrated share (want)") !=
           std::string::npos);
    assert(l.explain.at("limits").size() == 2);
    // The floor share plans only the floor and the tier takes the difference.
    in.checkpoints.share = "floor";
    l = bc::serving::host_ledger(in);
    assert(l.checkpoint_slots == 16 && l.host_kv_mib == 11055 + 8064 - 1792);
    in.checkpoints.share = "want";
    // Want plus the minimum tier does not fit: the store gives the tier its minimum, down toward the floor.
    in.min_tier_bytes = 12000 * MiB; // 945 MiB short: 9 slots of 112 MiB
    l = bc::serving::host_ledger(in);
    assert(l.checkpoint_slots == 63 && l.host_kv_mib == 11055 + 9 * 112);
    assert(l.explain.at("checkpoints").at("why").get<std::string>().find("shrunk from 72 (want) to 63") !=
           std::string::npos);
    // Even the floor leaves too little: the store stops at its floor and the tier is 0, saying so.
    in.min_tier_bytes = 30000 * MiB;
    l = bc::serving::host_ledger(in);
    assert(l.checkpoint_slots == 16 && l.host_kv_mib == 0);
    assert(l.explain.at("checkpoints").at("why").get<std::string>().find("reaching the floor") !=
           std::string::npos);
    assert(l.explain.at("host_kv_why").get<std::string>().find("no whole session") != std::string::npos);
    in.min_tier_bytes = 4 * 1024 * MiB;
    // The pin cap binds when the pinned consumers and the store leave less under it than MemAvailable does.
    in.host_facts["amdgpu_pin"]["cap_mib"] = 75000;
    l = bc::serving::host_ledger(in);
    assert(l.host_kv_mib == 75000 - (48824 + 7300 + 8064)); // 10812, under the 11055 MemAvailable leaves
    assert(l.explain.at("host_kv_why").get<std::string>().find("amdgpu pin cap") != std::string::npos);
    in.host_facts = host;
    // A memory cgroup is a limit of its own.
    in.host_facts["memory_cgroups"] =
        Json::array({{{"dir", "/sys/fs/cgroup/job"}, {"version", 2}, {"room_mib", 80000}}});
    l = bc::serving::host_ledger(in);
    assert(l.host_kv_mib == 80000 - (48824 + 7300 + 3072 + 8064) - 4705);
    // Too little room even for the floor: refused, naming the floor, the largest consumer and what to change.
    in.host_facts = host;
    in.host_facts["mem_available_mib"] = 60000;
    bool refused = false;
    try {
        (void)bc::serving::host_ledger(in);
    } catch (const std::exception& e) {
        const std::string w = e.what();
        refused = w.find("under MemAvailable") != std::string::npos &&
                  w.find("4705 MiB reserve") != std::string::npos &&
                  w.find("checkpoint floor of 16 slot(s), 1792 MiB") != std::string::npos &&
                  w.find("largest consumer is ngram_table (48824 MiB") != std::string::npos &&
                  w.find("engine.ngram_placement = \"disk\"") != std::string::npos;
    }
    assert(refused);
    // RLIMIT_MEMLOCK counts only mlocked copies, and CAP_IPC_LOCK lifts it.
    in.host_facts = host;
    in.host_facts["memlock"] = {{"limit_mib", 64}, {"cap_ipc_lock", false}};
    in.consumers.push_back({"locked_copy", 128 * MiB, Residence::mlocked, "engine plan (copy_table)", ""});
    refused = false;
    try {
        (void)bc::serving::host_ledger(in);
    } catch (const std::exception& e) {
        refused = std::string(e.what()).find("RLIMIT_MEMLOCK") != std::string::npos;
    }
    assert(refused);
    in.host_facts["memlock"]["cap_ipc_lock"] = true;
    assert(bc::serving::host_ledger(in).host_kv_mib == 11055 - 128);
    in.consumers.pop_back();
    in.host_facts = host;
    // The tier is 0 when sessions are evicted; the store keeps its want.
    in.retention_keep = false;
    l = bc::serving::host_ledger(in);
    assert(l.host_kv_mib == 0 && l.checkpoint_slots == 72);
    in.retention_keep = true;
    in.checkpoints.share = "most";
    refused = false;
    try {
        (void)bc::serving::host_ledger(in);
    } catch (const std::exception& e) {
        refused = std::string(e.what()).find("want, floor or tier_first") != std::string::npos;
    }
    assert(refused);
    // tier_first: the tier takes its target out of what the floor leaves, the store the remainder, up to its
    // want.
    in.checkpoints.share = "tier_first";
    in.tier_target_bytes = 12288 * MiB;
    l = bc::serving::host_ledger(in);
    // floor 16 x 112 MiB leaves 83020 - (48824 + 7300 + 3072) - 1792 - 4705 = 17327 MiB; the tier takes 12288
    // and the store 16 + (17327 - 12288) / 112 = 60 slots of a want of 72.
    assert(l.host_kv_mib == 12288 && l.checkpoint_slots == 60);
    assert(l.explain.at("checkpoints").at("want_slots") == 72 &&
           l.explain.at("checkpoints").at("floor_slots") == 16);
    assert(l.explain.at("checkpoints").at("why").get<std::string>().find("takes 12288 MiB first") !=
           std::string::npos);
    assert(l.explain.at("host_kv_why").get<std::string>().find("calibrated target") != std::string::npos);
    // A target larger than the room: the tier takes the room the floor leaves and the store stays at its floor.
    in.tier_target_bytes = 40000 * MiB;
    l = bc::serving::host_ledger(in);
    assert(l.host_kv_mib == 17327 && l.checkpoint_slots == 16);
    assert(l.explain.at("host_kv_why").get<std::string>().find("below its calibrated target") !=
           std::string::npos);
    in.checkpoints.share = "want";
    in.tier_target_bytes = 0;
    // The compiled-in share reproduces today's launch: tier first, 12 GiB.
    const auto standard = bc::serving::Calibration::standard();
    assert(standard.get("calibration.checkpoint_share", {"m", "flashnext-4gpu", "latency"}).value ==
           "tier_first");
    assert(standard.integer("calibration.host_kv_tier.target_mib", {"m", "flashnext-4gpu", "latency"}) ==
           12288);

    // The engine plan (rad0011's JSON) is strict.
    const Json plan = Json::parse(R"json({
      "plan_version": 1, "producer": "radiance --plan-memory (rad0011)",
      "host": [{"name": "ngram_table", "bytes": 51196575744, "residence": "pinned",
                "from": "engine plan (rad0011: weight_footprint, resolve_table_placement)",
                "remedy": "set engine.ngram_placement = \"disk\""}],
      "checkpoints": {"slot_bytes": 117941452, "floor_slots": 8, "want_slots": 72,
                      "from": "engine plan (rad0011: KVManager::checkpoint_bytes, checkpoint_slots_want)"},
      "kv_one_session": {"bytes": 4000000000, "from": "engine plan (rad0011: VramFacts::kv_one)"},
      "vram": {"claimable": 31224000000, "statics": 5800000000, "elastic": 25400000000, "experts": 0,
               "kv": 22800000000, "stage": 2600000000, "kv_resolved": 22799187968, "expert_floor": 0, "kv_ceiling": 40000000000,
               "expert_ceiling": 9000000000, "from": "engine plan (rad0011: vram_budget_resolve)",
               "cards": [{"capacity": 34208743424, "free": 33900000000, "engine_expert_plane": 9000000000,
                          "expert_unit_max": 2537472}]},
      "host_pool_mib": 9216})json");
    const auto parsed = bc::serving::decode_engine_plan(plan, "test");
    assert(parsed.host.size() == 1 && parsed.host[0].residence == bc::serving::v2::Residence::kPinned);
    assert(parsed.checkpoints.want_slots == 72 && parsed.checkpoints.floor_slots == 8);
    assert(parsed.kv_one_session.bytes == 4000000000ll && parsed.host_pool_mib == 9216);
    assert(parsed.vram.cards.size() == 1 && parsed.vram.kv_resolved == 22799187968ll);
    auto rejects = [](Json j, const std::string& needle) {
        try {
            (void)bc::serving::decode_engine_plan(j, "test");
        } catch (const std::exception& e) {
            if (std::string(e.what()).find(needle) != std::string::npos)
                return;
            throw std::runtime_error("wrong engine plan diagnostic for " + needle + ": " + e.what());
        }
        throw std::runtime_error("expected engine plan refusal containing " + needle);
    };
    Json bad = plan;
    bad["extra"] = 1;
    rejects(bad, "extra: unknown member");
    bad = plan;
    bad["plan_version"] = 2;
    rejects(bad, "must be 1");
    bad = plan;
    bad["host"][0]["residence"] = "swap";
    rejects(bad, "is not one of pinned, mlocked, pageable");
    bad = plan;
    bad["host"][0]["bytes"] = -1;
    rejects(bad, "below the minimum 0");
    bad = plan;
    bad["host"].push_back(plan["host"][0]);
    rejects(bad, "duplicate consumer ngram_table");
    bad = plan;
    bad["host"][0].erase("from");
    rejects(bad, "from: missing required member");
    bad = plan;
    bad["checkpoints"]["floor_slots"] = 99;
    rejects(bad, "floor_slots exceeds want_slots");
    bad = plan;
    bad["vram"].erase("kv");
    rejects(bad, "kv: missing required member");
    bad = plan;
    bad["vram"]["cards"] = Json::array();
    rejects(bad, "needs at least 1 item");
    bad = plan;
    bad["vram"]["cards"][0]["arch"] = "gfx1201";
    rejects(bad, "arch: unknown member");
}

void measured_plan_cache(Fixture& t) {
    namespace sv = bc::serving;
    // Container identity: the metadata tables, not the payload.
    const auto model = t.root / "plan-model.rad";
    bc::test::write_rad(model, 2, 8, 128, 128, {});
    const std::string id = sv::container_identity(model);
    assert(id.size() == 64 && sv::container_identity(model) == id);
    bc::test::write_rad(model, 2, 9, 128, 128, {});
    assert(sv::container_identity(model) != id);
    // Engine identity: the binary and every runtime-home file.
    const auto bin = t.file("engine/bin/radiance", "engine-binary-v1");
    (void)t.file("engine/runtime-home/kernels/libr4d.so", "kernels-v1");
    const std::string eid = sv::engine_identity(bin, t.root / "engine/runtime-home");
    (void)t.file("engine/runtime-home/kernels/libr4d.so", "kernels-v2");
    assert(sv::engine_identity(bin, t.root / "engine/runtime-home") != eid);
    // The key covers the planning inputs, the card and the host pool the plan echoes; not the memory outputs
    // the plan feeds (host KV tier, headroom) nor where the engine listens or lives (port, runtime home, model
    // path).
    const auto r = sv::resolve(t.standard(), t.facts);
    const Json k = sv::plan_key_inputs(r.values, t.facts, id, eid, "");
    assert(k.at("card").at("arch") == "gfx1201" && k.at("card").at("vram_total_mib") == 32624);
    assert(k.at("layout") == "flashnext-4gpu");
    for (const char* absent : {"prefix_cache_mib", "host_headroom_mib", "port", "radiance_home", "model"})
        assert(!k.at("engine").contains(absent));
    assert(k.at("engine").contains("host_pool_mib") && k.at("engine").contains("max_num_seqs"));
    const auto native_file = t.file("plan-key-fleet.yaml", "cards: [0]\nmodels: {tiny: {model: tiny.rad}}\n");
    const auto native = sv::NativeConfigSnapshot::read(native_file, t.root);
    const Json native_key = sv::plan_key_inputs(r.values, t.facts, id, eid, "", &native);
    assert(native_key.at("cache_version") == 3 && native_key.at("native_config").at("sha256") == native.sha256() &&
           native_key.at("native_config").at("resolution_root") == t.root.string());
    assert(native_key.at("native_config").at("rank_devices").size() == r.values.at("layout").at("engine_cards").size() +
                                                                          r.values.at("layout").at("helper_cards").size());
    Json reordered_values = r.values;
    assert(reordered_values.at("layout").at("helper_cards").size() > 1);
    std::swap(reordered_values["layout"]["helper_cards"][0],
              reordered_values["layout"]["helper_cards"][1]);
    assert(sv::plan_key(sv::plan_key_inputs(reordered_values, t.facts, id, eid, "", &native)) !=
           sv::plan_key(native_key));
    assert(sv::plan_key(native_key) != sv::plan_key(k));
    const auto other_root = t.root / "other-root";
    std::filesystem::create_directories(other_root);
    const auto same_bytes_other_root = sv::NativeConfigSnapshot::read(native_file, other_root);
    assert(sv::plan_key(sv::plan_key_inputs(r.values, t.facts, id, eid, "", &same_bytes_other_root)) !=
           sv::plan_key(native_key));
    Json v = r.values;
    for (const char* moved : {"prefix_cache_mib", "host_headroom_mib", "port"}) {
        v = r.values;
        v["engine"][moved] = 1;
        assert(sv::plan_key(sv::plan_key_inputs(v, t.facts, id, eid, "")) == sv::plan_key(k));
    }
    v = r.values;
    v["engine"]["host_pool_mib"] =
        1; // the plan echoes the pool: a plan measured under another pool is another plan
    assert(sv::plan_key(sv::plan_key_inputs(v, t.facts, id, eid, "")) != sv::plan_key(k));
    v = r.values;
    v["engine"]["checkpoint_interval"] = 2048;
    assert(sv::plan_key(sv::plan_key_inputs(v, t.facts, id, eid, "")) != sv::plan_key(k));
    // Engine arguments passed through verbatim can change the plan, so they are part of the key.
    assert(sv::plan_key(sv::plan_key_inputs(r.values, t.facts, id, eid, "--kv-resident 65536")) !=
           sv::plan_key(k));
    assert(sv::plan_key(sv::plan_key_inputs(r.values, t.facts, id, eid, "--kv-resident 65536")) ==
           sv::plan_key(sv::plan_key_inputs(r.values, t.facts, id, eid, "--kv-resident 65536")));
    // Every launcher setting that reaches the engine changes the key (a plan is never found for another
    // configuration); only the port, the runtime-home and model paths and the memory outputs the plan feeds do
    // not.
    {
        const std::string base_key = sv::plan_key(k);
        const std::pair<const char*, const char*> changes[] = {
            {"service.objective", "latency"},
            {"service.max_sessions", "4"},
            {"service.max_context", "131072"},
            {"engine.max_num_batched_tokens", "1024"},
            {"engine.checkpoint_interval", "2048"},
            {"engine.kv_cache_dtype", "fp8_v4"},
            {"engine.num_speculative_tokens", "2"},
            {"engine.rope_scaling", "yarn:4"},
            {"engine.ngram_placement", "disk"},
            {"engine.prefix_resume", "exact"},
            {"engine.tp", "2"},
            {"memory.helper_budgets_mib", "[22528,22528,7168]"},
            {"tuning.tp_ownership", "\"kv=1,1,2;ffn=8,8,16\""},
            {"tuning.state_cache_dtype", "\"bf16\""}};
        for (const auto& [key, value] : changes) {
            auto opts = t.standard();
            opts.sets = {std::string(key) + "=" + value};
            try {
                const auto changed = sv::resolve(opts, t.facts);
                if (sv::plan_key(sv::plan_key_inputs(changed.values, t.facts, id, eid, "")) == base_key)
                    throw std::runtime_error(std::string("the plan key ignores ") + key);
            } catch (const std::exception& e) {
                // a value the schema refuses is no configuration at all; every other change must move the key
                if (std::string(e.what()).find(key) == std::string::npos)
                    throw;
            }
        }
        const std::pair<const char*, const char*> same[] = {{"engine.port", "36081"},
                                                            {"memory.host_kv_mib", "6144"}};
        for (const auto& [key, value] : same) {
            auto opts = t.standard();
            opts.sets = {std::string(key) + "=" + value};
            assert(sv::plan_key(sv::plan_key_inputs(sv::resolve(opts, t.facts).values, t.facts, id, eid, "")) ==
                   base_key);
        }
    }
    // Store, then lookup; a miss is nothing; a tampered entry is an error.
    const auto dir = t.root / "plan-cache";
    assert(!sv::plan_cache_lookup(dir, k).has_value());
    const Json plan = Json::parse(R"json({
      "plan_version": 1, "producer": "radiance --plan-memory (rad0011)", "host": [],
      "checkpoints": {"slot_bytes": 1048576, "floor_slots": 8, "want_slots": 72, "from": "engine plan (test)"},
      "kv_one_session": {"bytes": 4096, "from": "engine plan (test)"},
      "vram": {"claimable": 10, "statics": 1, "elastic": 9, "experts": 0, "kv": 9, "stage": 0, "kv_resolved": 9, "expert_floor": 0,
               "kv_ceiling": 9, "expert_ceiling": 0, "from": "engine plan (test)",
               "cards": [{"capacity": 10, "free": 10, "engine_expert_plane": 0, "expert_unit_max": 1}]},
      "host_pool_mib": 9216})json");
    const Json ranks = {{"schema_version", 1},
                        {"kind", "radiance_p1_rank_demand"},
                        {"gpu", Json::array()},
                        {"resident_other_bytes", 0}};
    const auto stored = sv::plan_cache_store(dir, k, plan, ranks, sv::HostResident{2304, 1536});
    const auto found = sv::plan_cache_lookup(dir, k);
    assert(found && found->key == stored.key && found->plan.checkpoints.want_slots == 72 &&
           !found->measured_at.empty());
    const auto native_dir = t.root / "native-plan-cache";
    bool raw_native_refused = false;
    try {
        (void)sv::plan_cache_store(native_dir, native_key, plan, ranks, sv::HostResident{2304, 1536});
    } catch (const std::runtime_error& error) {
        raw_native_refused = std::string(error.what()).find("require owner measurement provenance") !=
                             std::string::npos;
    }
    assert(raw_native_refused);
    Json native_plan = plan;
    if (r.values.at("engine").at("ngram_placement") == "ram")
        native_plan["host"].push_back({{"name", "ngram_table"}, {"bytes", 1048576},
                                       {"residence", "pinned"}, {"from", "rad0011 native test"},
                                       {"remedy", ""}});
    Json native_ranks = ranks;
    native_ranks["gpu"].push_back({{"bdf", r.values.at("layout").at("engine_cards").at(0)},
                                   {"required_bytes", 10}, {"fixed_bytes", 1}, {"claimable_bytes", 10},
                                   {"evidence", "rad0011 test rank"}});
    auto refuse_owned_ranks = [&](Json bad_ranks, const std::string& expected) {
        try {
            (void)sv::plan_cache_store_owned(
                native_dir, native_key, native_plan, bad_ranks, r.values, sv::HostResident{2304, 1536},
                sv::OwnerMeasurementEvidence{"invalid-native-attempt", sv::sha256_hex("measuring-manifest")});
        } catch (const std::runtime_error& error) {
            assert(std::string(error.what()).find(expected) != std::string::npos);
            return;
        }
        assert(false && "invalid owned rank demand must refuse");
    };
    Json bad_bdf = native_ranks;
    bad_bdf["gpu"][0]["bdf"] = "0000:ff:00.0";
    refuse_owned_ranks(bad_bdf, "differs from selected ordered engine devices");
    Json bad_bytes = native_ranks;
    bad_bytes["gpu"][0]["required_bytes"] = 0;
    refuse_owned_ranks(bad_bytes, "out of measured wire range");
    const auto native_stored = sv::plan_cache_store_owned(
        native_dir, native_key, native_plan, native_ranks, r.values, sv::HostResident{2304, 1536},
        sv::OwnerMeasurementEvidence{"native-test-attempt", sv::sha256_hex("measuring-manifest")});
    const auto native_found = sv::plan_cache_lookup(native_dir, native_key);
    assert(native_found && native_found->key == native_stored.key);
    const fs::path native_entry_path = native_dir / (native_stored.key + ".json");
    Json native_entry;
    {
        std::ifstream in(native_entry_path, std::ios::binary);
        in >> native_entry;
    }
    assert(native_entry.at("cache_version") == 3);
    Json changed_resident = native_entry;
    changed_resident["resident"]["engine_mib"] = 1537;
    write(native_entry_path, changed_resident.dump(2) + "\n");
    try {
        (void)sv::plan_cache_lookup(native_dir, native_key);
        assert(false && "owned measurement receipt must bind resident memory");
    } catch (const std::runtime_error& error) {
        assert(std::string(error.what()).find("owner measurement digest mismatch") != std::string::npos);
    }
    write(native_entry_path, native_entry.dump(2) + "\n");
    native_entry["cache_version"] = 2;
    write(native_entry_path, native_entry.dump(2) + "\n");
    try {
        (void)sv::plan_cache_lookup(native_dir, native_key);
        assert(false && "native cache entry with v2 envelope must be refused");
    } catch (const std::runtime_error& error) {
        assert(std::string(error.what()).find("cache_version differs from key inputs") != std::string::npos);
    }
    // A plan is stored only for the configuration it belongs to: the host pool it echoes and the expected host
    // consumer list (the n-gram table exactly when it lives in RAM; nothing unknown).
    {
        auto decoded = [&](Json j) { return sv::decode_engine_plan(j, "test"); };
        auto refused_for = [&](const Json& values, const Json& plan_json, const std::string& needle) {
            try {
                sv::check_plan_for_configuration(values, decoded(plan_json));
            } catch (const std::exception& e) {
                if (std::string(e.what()).find(needle) != std::string::npos)
                    return;
                throw std::runtime_error("wrong store diagnostic for " + needle + ": " + e.what());
            }
            throw std::runtime_error("expected store refusal containing " + needle);
        };
        Json values = r.values;
        values["engine"]["ngram_placement"] = "disk";
        sv::check_plan_for_configuration(values, decoded(plan)); // host [] with the table on disk: fine
        Json with_table = plan;
        with_table["host"] = Json::array({{{"name", "ngram_table"},
                                           {"bytes", 1},
                                           {"residence", "pinned"},
                                           {"from", "engine plan (test)"},
                                           {"remedy", ""}}});
        refused_for(values, with_table, "places the n-gram table on disk");
        Json in_ram = r.values;
        in_ram["engine"]["ngram_placement"] = "ram";
        refused_for(in_ram, plan, "no ngram_table consumer");
        sv::check_plan_for_configuration(in_ram, decoded(with_table));
        Json other_pool = values;
        other_pool["engine"]["host_pool_mib"] = 1024;
        refused_for(other_pool, plan, "measured with a 9216 MiB host pool");
        Json unknown = with_table;
        unknown["host"].push_back(unknown["host"][0]);
        unknown["host"][1]["name"] = "mystery";
        refused_for(in_ram, unknown, "unknown host consumer mystery");
    }
    // The cache entry's own plan is what a manifest carries as the expected plan (the engine reads it from the
    // manifest, never from a file next to the cache): an entry edited by one byte validates, and the engine's
    // CHECK is what refuses it.
    {
        const auto file = dir / (stored.key + ".json");
        const std::string original = [&] {
            std::ifstream in(file);
            std::ostringstream t;
            t << in.rdbuf();
            return t.str();
        }();
        Json e = Json::parse(original);
        e["plan"]["host"] = Json::array({{{"name", "host_arenas"},
                                          {"bytes", 100},
                                          {"residence", "pinned"},
                                          {"from", "engine plan (test)"},
                                          {"remedy", ""}}});
        std::ofstream(file, std::ios::trunc) << e.dump();
        const auto before = sv::plan_cache_lookup(dir, k);
        assert(before && before->plan.host.at(0).bytes == 100);
        e["plan"]["host"][0]["bytes"] = 101;
        std::ofstream(file, std::ios::trunc) << e.dump();
        const auto drifted = sv::plan_cache_lookup(dir, k);
        assert(drifted && drifted->plan.host.at(0).bytes == 101);
        std::ofstream(file, std::ios::trunc) << original; // restore for the tests below
    }
    { // an entry from before the resident host memory was measured is a miss: the configuration is measured
      // again
        const auto file = dir / (stored.key + ".json");
        const std::string original = [&] {
            std::ifstream in(file);
            std::ostringstream t;
            t << in.rdbuf();
            return t.str();
        }();
        Json e = Json::parse(original);
        assert(e.at("resident").at("helpers_mib") == 2304 && e.at("resident").at("engine_mib") == 1536);
        e.erase("resident");
        std::ofstream(file, std::ios::trunc) << e.dump();
        assert(!sv::plan_cache_lookup(dir, k));
        e["resident"] = {{"helpers_mib", 5}, {"engine_mib", 0}};
        std::ofstream(file, std::ios::trunc) << e.dump();
        bool zero_refused = false;
        try {
            (void)sv::plan_cache_lookup(dir, k);
        } catch (const std::exception&) {
            zero_refused = true;
        }
        assert(zero_refused);
        std::ofstream(file, std::ios::trunc) << original;
    }
    {
        const auto file = dir / (stored.key + ".json");
        Json e = Json::parse(std::ifstream(file));
        e["key_inputs"]["layout"] = "other";
        std::ofstream(file, std::ios::trunc) << e.dump();
    }
    bool refused = false;
    try {
        (void)sv::plan_cache_lookup(dir, k);
    } catch (const std::exception& e) {
        refused = std::string(e.what()).find("key inputs differ") != std::string::npos;
    }
    assert(refused);
    // Plans and cache entries are read as the engine reads them: a repeated member is refused before any
    // decoding.
    {
        const std::string text = plan.dump();
        auto dup_refused = [&](const std::string& doc, const char* what) {
            try {
                (void)sv::parse_plan_json(doc, "dup.json");
            } catch (const std::exception& e) {
                return std::string(e.what()).find("dup.json") != std::string::npos;
            }
            std::fprintf(stderr, "duplicate member accepted: %s\n", what);
            return false;
        };
        assert(sv::parse_plan_json(text, "ok.json") == plan);
        assert(dup_refused("{\"host_pool_mib\":1,\"host_pool_mib\":2}", "top level"));
        assert(dup_refused("{\"vram\":{\"statics\":1,\"statics\":2}}", "nested vram"));
        assert(dup_refused("{\"host\":[{\"name\":\"a\",\"name\":\"b\"}]}", "inside a host consumer"));
        assert(dup_refused("{\"plan_version\":1", "truncated"));
        const auto dup_file = dir / "dup-plan.json";
        std::ofstream(dup_file) << "{\"a\":1,\"a\":1}";
        std::ostringstream read_back;
        read_back << std::ifstream(dup_file).rdbuf();
        assert(dup_refused(read_back.str(), "from a file"));
    }
    Json bad = plan;
    bad.erase("vram");
    refused = false;
    try {
        (void)sv::plan_cache_store(dir, k, bad, ranks, sv::HostResident{2304, 1536});
    } catch (const std::exception& e) {
        refused = std::string(e.what()).find("vram: missing required member") != std::string::npos;
    }
    assert(refused);
}

/* BRVP06 item 5: bc-serve refuses to run against a ROCm SDK other than the one its build recorded. */
void sdk_environment_step(Fixture& t) {
    namespace sv = bc::serving;
    namespace fs = std::filesystem;
    const fs::path build = t.root / "sdk-build";
    const fs::path exe = build / "radiance_home" / "bin" / "bc-serve";
    fs::create_directories(exe.parent_path());
    std::ofstream(exe) << "";
    const fs::path sdk = t.root / "rocm-7.2.4", other = t.root / "rocm-7.14";
    for (const fs::path& d : {sdk / "lib", other / "lib"}) {
        fs::create_directories(d);
        std::ofstream(d / "libamdhip64.so") << "";
    }
    auto metadata = [&](const std::string& toolchain) {
        std::ofstream(build / "bigcherry-build-metadata-radiance.json", std::ios::trunc)
            << "{\"toolchain\": " << toolchain << "}";
    };
    metadata("[[\"rocm_root\", \"" + sdk.string() + "\"], [\"rocm_version\", \"7.2.4\"]]");
    assert(!sv::sdk_refusal(exe, {})); // the recorded SDK, nothing set
    assert(
        !sv::sdk_refusal(exe, {sdk.string(), sdk.string(), (sdk / "lib").string()})); // the environment agrees
    assert(
        !sv::sdk_refusal(exe, {sdk.string() + "/", "", "/usr/lib/x86_64-linux-gnu:" + (sdk / "lib").string()}));
    auto refused = [&](const sv::SdkEnvironment& env, const std::string& needle) {
        const auto r = sv::sdk_refusal(exe, env);
        return r && r->find(needle) != std::string::npos;
    };
    assert(refused({other.string(), "", ""}, "ROCM_PATH=")); // another SDK by ROCM_PATH
    assert(refused({"", other.string(), ""}, "HIP_PATH="));  // ... or HIP_PATH
    assert(refused({"", "", (other / "lib").string()},
                   "outside the recorded SDK")); // ... or its libraries on the path
    assert(refused({"", "", "/usr/lib:" + (other / "lib").string()}, "outside the recorded SDK"));
    assert(!sv::sdk_refusal(exe,
                            {"", "", (t.root / "no-rocm-libs").string()})); // a directory with no ROCm runtime
    metadata("[[\"rocm_version\", \"7.2.4\"]]");
    assert(refused({}, "records no rocm_root"));
    metadata("[[\"rocm_root\", \"" + (t.root / "gone").string() + "\"]]");
    assert(refused({}, "is not installed"));
    std::ofstream(build / "bigcherry-build-metadata-radiance.json", std::ios::trunc) << "{not json";
    assert(refused({}, "cannot be read"));
    // A recognizable build/release path without provenance is not a reason to bypass SDK admission.
    fs::remove(build / "bigcherry-build-metadata-radiance.json");
    assert(refused({}, "metadata missing"));
    // an executable that is not inside a build has no recorded SDK and is not checked
    assert(!sv::sdk_refusal(t.root / "elsewhere" / "bc-serve",
                            {other.string(), other.string(), (other / "lib").string()}));
    assert(!sv::sdk_refusal(build / "bin" / "bc-serve", {other.string(), "", ""}));
}

struct SessionManifestPair {
    Json measuring;
    Json serving;
};

namespace sv = bc::serving;

SessionManifestPair session_manifest_pair(const sv::Resolved& resolved, const Json& facts) {
    sv::ManifestInputs inputs;
    inputs.resolved = resolved;
    inputs.instance_id = "11111111-1111-4111-8111-111111111111";
    inputs.producer_sha256 = std::string(64, 'a');
    inputs.profile = "production";
    inputs.s10.power_profile = "stock";
    inputs.s10.now_unix_s = 1800000000;
    inputs.s10.calibration_records = Json::array();
    const sv::BuiltManifest unmeasured = sv::build_manifest(inputs);

    bc::test::set_measured_manifest_fixture(inputs, unmeasured, facts);
    const sv::BuiltManifest serving = sv::build_manifest(inputs);
    return {unmeasured.document, serving.document};
}

void manifest_writer_step(Fixture& t) {
    namespace sv = bc::serving;
    // The container, its heat ranking and the measured plan this configuration's manifest is built from.
    const auto model = t.root / "writer-model.rad";
    bc::test::write_rad(model, 4, 16, 128, 128, {});
    // The original minimal planner fixture has no string table. RSC03's real RAD v2
    // resource reader (Python and now C++) requires valid name offsets even for experts.
    // Attach an expert-only name table without touching tensor payloads or model geometry.
    {
        std::fstream file(model, std::ios::binary | std::ios::in | std::ios::out);
        RadFileHeader header{};
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        assert(file && header.str_bytes == 0);
        constexpr char label[] = "expert";
        header.str_off = header.file_bytes;
        header.str_bytes = sizeof(label);
        header.file_bytes += sizeof(label);
        file.seekp(std::streamoff(header.str_off));
        file.write(label, sizeof(label));
        file.seekp(0);
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));
        assert(file);
    }
    const auto heat = t.root / "writer.hot-first";
    {
        std::vector<uint32_t> counts(128 * 16, 0);
        for (uint32_t l = 0; l < 4; ++l)
            for (uint32_t e = 0; e < 16; ++e)
                counts[(l * 16) + e] = (((e * 7) + (l * 3)) % 16) + 1;
        std::ofstream(heat, std::ios::binary)
            .write(reinterpret_cast<const char*>(counts.data()),
                   static_cast<std::streamsize>(counts.size() * sizeof(counts[0])));
    }
    Options opts;
    opts.site = t.file("writer-site.toml", "[engine]\nmodel = \"" + model.string() +
                                               "\"\nradiance_home = \"/runtime\"\n"
                                               "[handoff]\nheat = \"" +
                                               heat.string() + "\"\nhistogram = \"/heat.live\"\n" +
                                               std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
    const auto r = sv::resolve(opts, t.facts);
    const Json plan = Json::parse(R"json({
      "plan_version": 1, "producer": "radiance --plan-memory (rad0011)",
      "host": [{"name": "ngram_table", "bytes": 1048576, "residence": "pinned", "from": "engine plan (test)", "remedy": ""}],
      "checkpoints": {"slot_bytes": 1048576, "floor_slots": 8, "want_slots": 72, "from": "engine plan (test)"},
      "kv_one_session": {"bytes": 4096, "from": "engine plan (test)"},
      "vram": {"claimable": 10, "statics": 1, "elastic": 9, "experts": 0, "kv": 9, "stage": 0, "kv_resolved": 9, "expert_floor": 0,
               "kv_ceiling": 9, "expert_ceiling": 0, "from": "engine plan (test)",
               "cards": [{"capacity": 10, "free": 10, "engine_expert_plane": 0, "expert_unit_max": 1}]},
      "host_pool_mib": 9216})json");
    sv::ManifestInputs in;
    in.resolved = r;
    in.instance_id = "11111111-1111-4111-8111-111111111111";
    in.producer_sha256 = std::string(64, 'a');
    in.profile = "production";
    in.s10.power_profile = "stock";
    in.s10.now_unix_s = 1800000000;
    in.s10.calibration_records = Json::array();
    {
        // The unmeasured manifest of the same configuration has the owner map the P1 record is measured
        // against.
        const auto unmeasured = sv::build_manifest(in);
        // RSC02 step 2: exercise real RAD directory, the exact published owner map and this
        // synthetic KFD/cgroup/pin detector fixture. The standalone parity test checks the pure
        // arithmetic against Python, this test proves the wiring into native manifest inputs.
        const auto ledger = sv::resources::derive_launch(
            {r.values, r.manifest.at("process_helpers"), t.facts, unmeasured.document});
        assert(ledger.ok());
        assert(ledger.value().helpers.size() == r.manifest.at("process_helpers").size());
        assert(ledger.value().owner_map_digest == unmeasured.document.at("spec")
                                                      .at("plan")
                                                      .at("owner_map")
                                                      .at("owner_map_sha256")
                                                      .get<std::string>());
        assert(ledger.value().host_pool_mib > 0 && ledger.value().host_kv_mib > 0);
        {
            auto tp3_values = r.values;
            tp3_values["engine"]["tp"] = 3;
            auto one_rank_room = t.facts;
            const uint64_t room_mib = (ledger.value().committed_bytes + (1ULL << 20) - 1) >> 20;
            one_rank_room["host"]["mem_available_mib"] = room_mib;
            for (auto& group : one_rank_room["host"]["memory_cgroups"])
                group["room_mib"] = room_mib;
            const auto denied = sv::resources::derive_launch(
                {tp3_values, r.manifest.at("process_helpers"), one_rank_room, unmeasured.document});
            assert(!denied.ok() && denied.error().code == bc::serving::v2::ConfigErrorCode::admission_refused);
        }
        {
            auto altered = r.manifest.at("process_helpers");
            altered[0]["bdf"] = "0000:fe:00.0";
            assert(!sv::resources::derive_launch({r.values, altered, t.facts, unmeasured.document}));
        }
        {
            auto altered = unmeasured.document;
            altered["spec"]["plan"]["owner_map"]["owner_map_sha256"] = "forged";
            assert(
                !sv::resources::derive_launch({r.values, r.manifest.at("process_helpers"), t.facts, altered}));
        }
        {
            auto constrained = t.facts;
            constrained["host"]["mem_available_mib"] = 1;
            assert(!sv::resources::derive_launch(
                {r.values, r.manifest.at("process_helpers"), constrained, unmeasured.document}));
        }
        in.s10.p1_record = bc::test::p1_record_for(unmeasured.document, 9216, 1024,
                                         r.values.at("memory").at("host_reserve_mib").get<int64_t>());
    }
    {
        sv::MeasuredPlan measured;
        measured.plan.plan = sv::decode_engine_plan(plan, "writer plan");
        measured.plan.plan_json = plan;
        measured.plan.resident = {1536, 1024};
        measured.key_inputs =
            sv::plan_key_inputs(r.values, t.facts, sv::container_identity(model), std::string(64, 'b'), "");
        in.measured = std::move(measured);
    }

    const auto built = sv::build_manifest(in);
    g_session_manifest_template = built.document;
    // It is a v1 manifest a consumer accepts, and says what the resolver resolved.
    const auto back =
        bc::serving::v2::read_manifest(built.text, "writer test", bc::serving::v2::PlanUse::serving);
    assert(back.ok());
    { // RED: strict native writer + real runtime guard must not kill recording-only serving.
        namespace ss = bc::serving::session;
        const fs::path file = t.root / "recording-only-guard-manifest.json";
        sv::write_manifest_file(file, built.text);
        const fs::path log = t.file("recording-only-guard-engine.log", "recorded pass: draft pass 8 token\n");
        ss::EngineStart engine;
        engine.manifest = file;
        engine.handoff_shm =
            built.document.at("spec").at("handoff").at("transport").at("shm").get<std::string>();
        engine.engine_launched_at = std::chrono::system_clock::now() - std::chrono::seconds(3);
        engine.heat_session_nonce = "0123456789abcdef0123456789abcdef";
        ss::RuntimeGuards guard;
        const auto result = guard.startup(engine, log, {});
        if (result)
            throw std::runtime_error("recording-only native runtime guard refused: " + result->message);
        if (guard.policy_enabled())
            throw std::runtime_error("uncalibrated recording-only manifest incorrectly activated policy");
        if (const auto ongoing = guard.poll())
            throw std::runtime_error("recording-only heat file affected serving: " + ongoing->message);
    }
    // RED P1-5: measured producer P1 + raw P2 with a bounded valid_until.
    // Build the strict v2 consumer document via the REAL S10 writer, and
    // re-qualify that same native P1 with today's scope using the same S8.
    // The P2 value deliberately equals the operator fallback seed: expiry
    // must still invalidate its evidence, rather than comparing only values.
    {
        namespace ss = bc::serving::session;
        namespace v2 = bc::serving::v2;
        sv::ManifestInputs qualified = in;
        sv::Resolved current = qualified.resolved;
        const fs::path home = t.root / "s8-native-home";
        const fs::path binary = t.root / "s8-native-engine";
        fs::create_directories(home);
        std::ofstream(binary, std::ios::binary) << "S8 fake process binary identity";
        current.values["engine"]["radiance_home"] = home.string();
        current.typed.engine.radiance_home = home;
        qualified.resolved = current;
        Json& native = qualified.s10.p1_record;
        Json& scope = native["scope"];
        scope["model_digest"] = sv::container_identity(model);
        scope["engine_build_digest"] = sv::engine_identity(binary, home);
        scope["helper_build_digest"] = sv::file_sha256(binary);
        scope["layout_digest"] = sv::sha256_hex(current.values.at("layout").dump());
        scope["inventory_digest"] = sv::gpu_inventory_digest(t.facts.at("gpus"));
        scope["context_target_tokens"] = current.values.at("service").at("max_context");
        native["resource_inputs"]["context_target_tokens"] = scope["context_target_tokens"];
        const int64_t same_seed = current.values.at("handoff").at("small_rows").get<int64_t>();
        constexpr int64_t expiry = 1800000110;
        qualified.s10.calibration_records =
            Json::array({{{"key", "handoff.small_rows"},
                          {"value", same_seed},
                          {"evidence_id", "s8-expiry-same-seed"},
                          {"raw_run", "runs/s8-typed-acceptance"},
                          {"method", "bounded native P2 check"},
                          {"recorded_date", "2027-01-15"},
                          {"source", "cache"},
                          {"measured_at_unix_s", 1800000000},
                          {"valid_until_unix_s", expiry},
                          {"measured_candidates", Json::array({same_seed, same_seed + 1})},
                          {"scope", scope}}});
        // The actual P1 producer must generate the same typed wire domain
        // the S10/S8 consumer accepts, rather than a hand-made alternate P1.
        // All OS facts here originate in the test detector's real sysroot.
        const Json rank = {{"schema_version", 1},
                           {"kind", "radiance_p1_rank_demand"},
                           {"gpu", Json::array({{{"bdf", current.values.at("layout").at("engine_cards").at(0)},
                                                 {"required_bytes", 10},
                                                 {"fixed_bytes", 1},
                                                 {"claimable_bytes", 10},
                                                 {"evidence", "rad0011 rank split fixture"}}})},
                           {"resident_other_bytes", 0}};
        // A tiny test RAD has all experts fitting into two of three helpers.
        // The REAL tail placement contract requires at least two helpers, so
        // select exactly those two actually owning experts. Never invent a
        // third helper's footprint, which the producer would refuse.
        sv::Resolved producer_config = current;
        producer_config.typed.layout.helper_cards.resize(2);
        producer_config.values["layout"]["helper_cards"] =
            Json::array({current.values.at("layout").at("helper_cards").at(0),
                         current.values.at("layout").at("helper_cards").at(1)});
        producer_config.manifest["process_helpers"] = Json::array(
            {current.manifest.at("process_helpers").at(0), current.manifest.at("process_helpers").at(1)});
        producer_config.typed.helper.count = 2;
        producer_config.values["helper"]["count"] = 2;
        producer_config.manifest["process_helpers"][1]["placement"] = "tail";
        producer_config.manifest["process_helpers"][1]["share"] = 0;
        for (size_t i = 0; i < 2; ++i) {
            producer_config.manifest["process_helpers"][i]["config"]["count"] = 2;
            producer_config.manifest["process_helpers"][i]["config"]["index"] = i;
        }
        // 64 tiny routed experts together take about 1.6 MiB. A 1 MiB
        // expert budget makes both selected helpers actually own experts
        // under the real worker footprint and tail placement contract.
        producer_config.typed.memory.helper_budgets_mib = {1, 1};
        producer_config.values["memory"]["helper_budgets_mib"] = Json::array({1, 1});
        producer_config.typed.memory.tail_budget_mib = 1;
        producer_config.values["memory"]["tail_budget_mib"] = 1;
        for (auto& row : producer_config.manifest["process_helpers"])
            row["budget_mib"] = 1;
        Json native_plan = plan;
        native_plan["kv_one_session"]["bytes"] = current.typed.service.max_context;
        // The actual native owner map covers all routed experts; the host
        // pool is the 1024 MiB P2 margin and no unowned/embedding/staging.
        native_plan["host_pool_mib"] = 1024;
        const Json native_sources = {{"engine_plan", native_plan},
                                     {"rank_demand", rank},
                                     {"engine_binary", binary.string()},
                                     {"helper_binary", binary.string()},
                                     {"retention_target_tokens", 4096},
                                     {"margin",
                                      {{"key", "host_ledger_margin.mib"},
                                       {"source", "cache"},
                                       {"evidence", "typed S8 fixture"},
                                       {"mib", 1024}}},
                                     {"run_id", "rsc02-test-real-p1-producer"}};
        for (const auto& gpu : t.facts.at("gpus"))
            write(t.root / "sys/bus/pci/devices" / gpu.at("bdf").get<std::string>() / "mem_info_gtt_used",
                  "0\n");
        sv::ManifestInputs producer_inputs;
        producer_inputs.resolved = producer_config;
        const Json published = v2::produce_p1(producer_inputs, t.facts, t.root, native_sources);
        assert(published.at("kind") == "bc_serve_v2_measured_inputs");
        assert(published.at("resource_inputs").at("helper_footprints").size() == 2);
        assert(published.at("resource_inputs").at("owner_map_digest") ==
               published.at("scope").at("owner_map_digest"));
        // The strict manifest consumes the exact native producer bytes.
        qualified.resolved = producer_config;
        qualified.measured->plan.plan = sv::decode_engine_plan(native_plan, "native S8 plan");
        qualified.measured->plan.plan_json = native_plan;
        native = published;
        for (auto& rec : qualified.s10.calibration_records)
            rec["scope"] = published.at("scope");
        const auto issued = sv::build_manifest(qualified);
        const auto typed = v2::read_manifest(issued.text, "native S8 expiry document", v2::PlanUse::serving);
        assert(typed.ok());
        const Json raw = {{"version", 1},
                          {"manifest_digest", issued.digest},
                          {"produced_at_unix_s", qualified.s10.now_unix_s},
                          {"p1_record", native},
                          {"calibration_records", qualified.s10.calibration_records}};
        const fs::path sibling = t.root / "native-s8-evidence.json";
        sv::write_manifest_file(sibling, raw.dump(2) + "\n");
        const ss::StoredS8Evidence measured = ss::read_stored_s8_evidence(sibling, issued.digest);
        v2::P1LiveRoom room{};
        constexpr int64_t MiB = 1048576;
        room.mem_available_bytes = 92160LL * MiB;
        room.cgroup_headroom_bytes = 92160LL * MiB;
        room.pin_available_bytes = 92160LL * MiB;
        const auto before = ss::verify_stored_s8(typed.value(), producer_config, t.facts, room, measured,
                                                 binary, binary, expiry - 1);
        if (before)
            throw std::runtime_error("valid native S8 replay failed: " +
                                     v2::render_human("valid before expiry", *before));
        { // The stored inventory scope is an identity snapshot (selected card, BDF, architecture, VRAM total). Live VRAM
          // use and the PCIe link state differ from one start to the next (the previous engine's pages are still being
          // released), and must never refuse the replay of an unchanged configuration.
            Json drifted = t.facts;
            for (Json& g : drifted["gpus"]) {
                g["vram_used_mib"] = g.value("vram_used_mib", 0) + 7;
                g["vram_available_mib"] = std::max<int64_t>(0, g.value("vram_available_mib", 0) - 7);
                g["current_link_speed"] = "2.5 GT/s PCIe";
                g["current_link_width"] = 1;
            }
            const auto drift = ss::verify_stored_s8(typed.value(), producer_config, drifted, room, measured, binary,
                                                    binary, expiry - 1);
            if (drift)
                throw std::runtime_error("replay refused for live VRAM drift: " + v2::render_human("drift", *drift));
            // ... while a changed card identity still refuses.
            Json moved = t.facts;
            moved["gpus"][0]["arch"] = "gfx-foreign";
            const auto changed = ss::verify_stored_s8(typed.value(), producer_config, moved, room, measured, binary,
                                                      binary, expiry - 1);
            assert(changed.has_value());
            Json resized = t.facts;
            resized["gpus"][0]["vram_total_mib"] = resized["gpus"][0].at("vram_total_mib").get<int64_t>() + 1024;
            assert(ss::verify_stored_s8(typed.value(), producer_config, resized, room, measured, binary, binary,
                                        expiry - 1).has_value());
        }
        const auto after = ss::verify_stored_s8(typed.value(), producer_config, t.facts, room, measured, binary,
                                                binary, expiry);
        assert(after.has_value());
        assert(after->message.find("valid_until_unix_s=1800000110") != std::string::npos &&
               "expired calibrated P2 must give its exact live deadline, not silently use identical seed");
    }
    const Json& d = built.document;
    assert(d.at("metadata").at("content_digest") == built.digest && built.digest.rfind("sha256:", 0) == 0);
    const Json& spec = d.at("spec");
    assert(spec.at("intent").at("layout") == "flashnext-4gpu" &&
           spec.at("intent").at("model") == model.string());
    assert(spec.at("processes").at("helpers").size() == 3 &&
           spec.at("processes").at("helpers")[2].at("tier") == "tail");
    assert(spec.at("processes").at("helpers")[2].at("rows_kernel") == true &&
           spec.at("processes").at("helpers")[2].at("share") == 0);
    assert(spec.at("processes").at("engine").at("scheduler").at("tp") == r.values.at("engine").at("tp"));
    assert(spec.at("handoff").at("transport").at("max_rows") == r.values.at("handoff").at("max_rows"));
    assert(spec.at("handoff").at("kernels").at("moe_down_tile") == 1);
    assert(spec.at("resources").at("host").at("pool_mib") == r.values.at("memory").at("host_pool_mib"));
    size_t leaves = 0;
    for (const auto& [group, body] : spec.at("processes").at("helper_common").items())
        leaves += body.size();
    assert(leaves == 25); // every helper_common leaf of helper_profile.toml; balance_speeds_milli is absent
                          // while balance is off
    size_t owned = 0;
    for (const Json& h : spec.at("plan").at("owner_map").at("helpers"))
        owned += h.at("count").get<size_t>();
    assert(owned == 64); // 4 layers x 16 experts: with budgets this large the helpers own them all
    assert(spec.at("plan").at("engine_memory").at("expected") == plan);
    // The checkpoint slots follow what is really free (RSC08 S3): the host pool and the memory the helpers and
    // the engine hold are counted against MemAvailable as read when the manifest is written. With the tier at
    // its target (12288 MiB), a store floor of 8 and the plan's 1 MiB consumer, the room above the tier is
    // `left` MiB and the store takes 8 + left of its want 72.
    {
        const uint64_t reserve = r.values.at("memory").at("host_reserve_mib").get<uint64_t>();
        // the calibrated margin the ledger leaves unplanned (amendment 11) is part of the manifest's
        // calibration and of the room
        uint64_t margin = 0;
        for (const Json& e : spec.at("calibration"))
            if (e.at("key") == "host_ledger_margin.mib")
                margin = e.at("value").get<uint64_t>();
        assert(margin == 1024);
        auto slots_for = [&](uint64_t helpers_mib, int64_t extra_left_mib,
                             int64_t available_shift_mib) -> uint64_t {
            sv::ManifestInputs in3 = in;
            in3.measured->plan.resident = {helpers_mib, 1024};
            const int64_t available = int64_t(reserve + margin + 9216 + 1 + 8 + 12288 + helpers_mib + 1024) +
                                      extra_left_mib + available_shift_mib;
            in3.resolved.facts["host"]["mem_available_mib"] = available;
            const auto built3 = sv::build_manifest(in3);
            return built3.document.at("spec").at("resources").at("host").at("checkpoint_slots").get<uint64_t>();
        };
        assert(slots_for(1536, 22, 0) == 8 + 22); // 22 MiB above the tier: 22 more slots than the floor
        assert(slots_for(1536, 22, -10) ==
               8 + 12); // a manifest written when MemAvailable is lower: fewer slots
        assert(slots_for(1536, 22, +10) == 8 + 32); // and when it is higher: more
        // the margin is a seed with its evidence in spec.calibration, and a refit (a cache entry) replaces it:
        // the manifest says so (source cache), its digest moves, and the slots follow; the engine plan and its
        // key do not (the margin is not part of what the engine measured)
        {
            const Json* seed_entry = nullptr;
            for (const Json& e : spec.at("calibration"))
                if (e.at("key") == "host_ledger_margin.mib")
                    seed_entry = &e;
            assert(seed_entry && seed_entry->at("source") == "seed" &&
                   !seed_entry->at("evidence").get<std::string>().empty());
            sv::ManifestInputs seeded = in;
            seeded.measured->plan.resident = {1536, 1024};
            seeded.resolved.facts["host"]["mem_available_mib"] =
                reserve + margin + 9216 + 1 + 8 + 12288 + 1536 + 1024 + 30;
            const auto before = sv::build_manifest(seeded);
            sv::ManifestInputs refit = seeded;
            refit.calibration_cache = t.file(
                "margin-refit.toml",
                "schema_version = 1\n[[entry]]\nkey = \"calibration.host_ledger_margin.mib\"\nvalue = 1052\n"
                "seeded = \"2026-11-06\"\nevidence = \"a start stopped 28 MiB short at the engine's host-room "
                "check\"\n");
            const auto after = sv::build_manifest(refit);
            const Json* refit_entry = nullptr;
            for (const Json& e : after.document.at("spec").at("calibration"))
                if (e.at("key") == "host_ledger_margin.mib")
                    refit_entry = &e;
            assert(refit_entry && refit_entry->at("source") == "cache" &&
                   refit_entry->at("value") == margin + 28);
            assert(after.digest != before.digest);
            auto slots_of = [](const sv::BuiltManifest& b) {
                return b.document.at("spec").at("resources").at("host").at("checkpoint_slots").get<uint64_t>();
            };
            assert(slots_of(before) == 8 + 30 &&
                   slots_of(after) == 8 + 2); // 28 MiB more margin, 28 fewer slots of 1 MiB
            assert(before.document.at("spec").at("plan").at("engine_memory").at("cache_key") ==
                   after.document.at("spec").at("plan").at("engine_memory").at("cache_key"));
        }
        // helpers holding more resident memory take the same room: fewer slots at the same MemAvailable
        {
            sv::ManifestInputs in3 = in;
            in3.measured->plan.resident = {1536, 1024};
            const uint64_t available = in3.resolved.facts.at("host").at("mem_available_mib").get<uint64_t>();
            in3.resolved.facts["host"]["mem_available_mib"] =
                reserve + margin + 9216 + 1 + 8 + 12288 + 1536 + 1024 + 22;
            const auto base = sv::build_manifest(in3);
            in3.measured->plan.resident = {1536 + 10, 1024};
            const auto more = sv::build_manifest(in3);
            assert(more.document.at("spec").at("resources").at("host").at("checkpoint_slots").get<uint64_t>() +
                       10 ==
                   base.document.at("spec").at("resources").at("host").at("checkpoint_slots").get<uint64_t>());
            (void)available;
        }
        // the figures are measured, never defaulted: a plan stored without them is refused, and a cache entry
        // written before they were measured is a miss (the configuration is measured again)
        sv::ManifestInputs in4 = in;
        in4.measured->plan.resident = {0, 0};
        bool refused = false;
        try {
            (void)sv::build_manifest(in4);
        } catch (const std::exception&) {
            refused = true;
        }
        assert(refused);
    }
    // The kv_backing_floor the engine admits with is the selected calibration, carried in the manifest (never a
    // launcher literal): balanced 600000, latency 500000, throughput none (0). engine_view hands the engine
    // exactly this.
    {
        auto floor_for = [&](const char* objective) {
            Options o = opts;
            o.sets = {std::string("service.objective=") + objective};
            const auto r2 = sv::resolve(o, t.facts);
            sv::ManifestInputs in2 = in;
            in2.resolved = r2;
            in2.measured->key_inputs = sv::plan_key_inputs(r2.values, t.facts, sv::container_identity(model),
                                                           std::string(64, 'b'), "");
            const auto built2 = sv::build_manifest(in2);
            const auto m2 =
                bc::serving::v2::read_manifest(built2.text, "floor test", bc::serving::v2::PlanUse::serving);
            assert(m2.ok());
            const auto view = bc::serving::v2::engine_view(m2.value());
            assert(view.ok());
            return view.value().kv_backing_floor_ppm;
        };
        assert(floor_for("balanced") == 600000 && floor_for("latency") == 500000 &&
               floor_for("throughput") == 0);
    }
    assert(spec.at("plan").at("state") == "measured" &&
           spec.at("resources").at("host").contains("checkpoint_slots"));
    assert(!spec.at("handoff").at("placement").contains("live_policy") && !spec.contains("experiment"));

    // RHP16: a matching measured cache record derives production policy leaves; v1
    // read-back validates the digest, each leaf's provenance and the pinned contract.
    // The transaction, copy, heat-gap, margin and k figures are from policy-live-9.
    // This synthetic tiny model has synthetic additional effect/lanes/drift readings.
    {
        sv::ManifestInputs calibrated = in;
        const Json policy_inputs = sv::policy_calibration_inputs(built.document.at("spec"));
        const std::string key = sv::policy_calibration_key(policy_inputs);
        { // Rebuilds, helper VRAM budgets and host memory do not change measured policy scope.
            auto changed = built.document.at("spec");
            changed["plan"]["engine_memory"]["engine_identity"] = std::string(64, 'f');
            changed["processes"]["helpers"][0]["vram_budget_mib"] = 1234;
            changed["resources"]["host"]["pool_mib"] = 1234;
            assert(sv::policy_calibration_key(sv::policy_calibration_inputs(changed)) == key);
            changed = built.document.at("spec");
            changed["intent"]["layout"] = "another-layout";
            assert(sv::policy_calibration_key(sv::policy_calibration_inputs(changed)) != key);
            changed = built.document.at("spec");
            changed["intent"]["service"]["objective"] = "latency";
            assert(sv::policy_calibration_key(sv::policy_calibration_inputs(changed)) != key);
            changed = built.document.at("spec");
            changed["handoff"]["telemetry"]["period_s"] = 1;
            assert(sv::policy_calibration_key(sv::policy_calibration_inputs(changed)) != key);
            changed = built.document.at("spec");
            changed["processes"]["helpers"][0]["gpu"] = "0000:06:00.0";
            assert(sv::policy_calibration_key(sv::policy_calibration_inputs(changed)) != key);
            changed = built.document.at("spec");
            changed["plan"]["engine_memory"]["model_identity"] = std::string(64, 'f');
            assert(sv::policy_calibration_key(sv::policy_calibration_inputs(changed)) != key);
            changed = built.document.at("spec");
            changed["facts"]["gpus"][0]["arch"] = "gfx1030";
            assert(sv::policy_calibration_key(sv::policy_calibration_inputs(changed)) != key);
        }
        calibrated.calibration_cache =
            t.file("live-policy-record.toml",
                   "schema_version = 1\n[[live_policy_record]]\n"
                   "plan_cache_key = \"" +
                       key +
                       "\"\n" +

                       "calibration_inputs = " + Json(policy_inputs.dump()).dump() + "\n" +
                       "source_plan_key = \"" + sv::plan_key(calibrated.measured->key_inputs) + "\"\n" +
                       "source_engine_build = \"" +
                       calibrated.measured->key_inputs.at("engine_build").get<std::string>() + "\"\n" +
                       "layout = \"flashnext-4gpu\"\nobjective = \"balanced\"\n"
                       "seeded = \"2026-11-05\"\nevidence = \"policy-live-9 timing and sampling; synthetic "
                       "noise/drift/lane readings\"\n"
                       "txn_p50_ns = 1770000000\ntxn_p95_ns = 2720000000\n"
                       "copy_bytes_per_s = 160000000\nheat_gap_max_ns = 5040000000\nheat_watermarked = 0\n"
                       "hot_set_k = 5931\nmarginal_error_ppm = 23000\n"
                       "marginal_observations = 3\ndrift_half_life_observations = 59000000\n"
                       "observation_rate_per_s = 10000000\n"
                       "tail_share_gain_ppm = 45000\ntail_lane_service_ns_per_s = 750000000\ntail_lane_row_ns "
                       "= 22222222\n"
                       "lane_lost_ns = 100000000\neffect_rows_per_s = 64\n"
                       "effect_noise_max_ns = 2500000\n"
                       "effect_row_noise_ns = 1550000\n"
                       "effect_window_noise_ladder = [[8, 80000], [16, 40000], [32, 16000], [64, 10000]]\n"
                       "abort_quiescent_ns = 1500000000\n");
        const auto with_calibration = sv::build_manifest(calibrated);
        { // A different engine build consumes the SAME qualified hardware evidence,
          // with a distinct strict memory-plan identity and a new manifest digest.
            sv::ManifestInputs rebuilt = calibrated;
            rebuilt.measured->key_inputs["engine_build"] = std::string(64, 'f');
            const auto carried = sv::build_manifest(rebuilt);
            assert(carried.document.at("spec").at("handoff").at("placement").contains("live_policy"));
            assert(carried.document.at("spec").at("plan").at("engine_memory").at("engine_identity") !=
                   with_calibration.document.at("spec").at("plan").at("engine_memory").at("engine_identity"));
            assert(carried.digest != with_calibration.digest);
            assert(bc::serving::v2::read_manifest(carried.text, "carried calibration",
                                                  bc::serving::v2::PlanUse::serving).ok());
        }
        { // A policy-enabled manifest must STILL refuse missing live heat.
            namespace ss = bc::serving::session;
            const fs::path document = t.root / "policy-enabled-guard-manifest.json";
            sv::write_manifest_file(document, with_calibration.text);
            ss::EngineStart active;
            active.manifest = document;
            active.handoff_shm =
                with_calibration.document.at("spec").at("handoff").at("transport").at("shm").get<std::string>();
            active.engine_launched_at = std::chrono::system_clock::now() - std::chrono::seconds(3);
            active.heat_session_nonce = "0123456789abcdef0123456789abcdef";
            ss::RuntimeGuards guard;
            const auto failure = guard.startup(active, t.root / "recording-only-guard-engine.log", {});
            if (!guard.policy_enabled() || !failure || failure->message.find("histogram") == std::string::npos)
                throw std::runtime_error("qualified live policy lost strict histogram admission");
        }
        const auto& lp = with_calibration.document.at("spec").at("handoff").at("placement").at("live_policy");
        const uint64_t quiet_windows = lp.at("effect_min_quiet_windows").get<uint64_t>();
        const uint64_t window_samples = lp.at("effect_window_samples").get<uint64_t>();
        const uint64_t window_ns = ((window_samples * 1000000000ULL) + 63ULL) / 64ULL;
        assert(lp.at("cadence_ns") == (2720000000ULL + (quiet_windows * window_ns)));
        assert(lp.at("accept_unwatermarked_heat") == true);
        assert(lp.at("max_moved_bytes") == 435200000ull && lp.at("margin_ppm") == 23000);
        assert(lp.at("half_life_observations_milli") == 59000000000ull);
        const auto& pv = with_calibration.document.at("provenance");
        assert(pv.at("/spec/handoff/placement/live_policy/cadence_ns").at("origin") == "derived");
        assert(pv.at("/spec/handoff/placement/live_policy/margin_ppm").at("class") == "policy");
        assert(bc::serving::v2::read_manifest(with_calibration.text, "calibrated policy",
                                              bc::serving::v2::PlanUse::serving)
                   .ok());
        assert(with_calibration.digest != built.digest);
        std::ifstream cal_input(calibrated.calibration_cache);
        const std::string cal_text((std::istreambuf_iterator<char>(cal_input)),
                                   std::istreambuf_iterator<char>());
        // Capture the production diagnostic, not only the helper's return value.
        auto no_policy = [&](const std::string& file_text, const std::string& reason) {
            sv::ManifestInputs failed = calibrated;
            failed.calibration_cache = t.file("failed-live-policy.toml", file_text);
            FILE* log = std::tmpfile();
            assert(log);
            std::fflush(stderr);
            const int saved = ::dup(STDERR_FILENO);
            assert(saved >= 0 && ::dup2(::fileno(log), STDERR_FILENO) >= 0);
            sv::BuiltManifest no_live;
            try {
                no_live = sv::build_manifest(failed);
            } catch (...) {
                std::fflush(stderr);
                assert(::dup2(saved, STDERR_FILENO) >= 0);
                ::close(saved);
                std::fclose(log);
                throw;
            }
            std::fflush(stderr);
            assert(::dup2(saved, STDERR_FILENO) >= 0);
            ::close(saved);
            std::rewind(log);
            std::string diagnostic;
            char chunk[512];
            while (const size_t count = std::fread(chunk, 1, sizeof(chunk), log))
                diagnostic.append(chunk, count);
            std::fclose(log);
            const auto& placement = no_live.document.at("spec").at("handoff").at("placement");
            assert(!placement.contains("live_policy"));
            const auto& origin = no_live.document.at("provenance")
                                     .at("/spec/handoff/placement")
                                     .at("source")
                                     .get_ref<const std::string&>();
            assert(origin.find(key) != std::string::npos && origin.find(reason) != std::string::npos);
            assert(diagnostic.find("serving without live_policy") != std::string::npos &&
                   diagnostic.find(key) != std::string::npos && diagnostic.find(reason) != std::string::npos);
            assert(bc::serving::v2::read_manifest(no_live.text, "fallback", bc::serving::v2::PlanUse::serving)
                       .ok());
            sv::write_manifest_file(t.root / ("fallback-" + reason + ".json"), no_live.text);
        };
        auto replace_line = [&](std::string src, const std::string& name, const std::string& value) {
            const auto from = src.find(name + " = ");
            assert(from != std::string::npos);
            const auto end = src.find('\n', from);
            assert(end != std::string::npos);
            src.replace(from, end - from, name + " = " + value);
            return src;
        };
        const std::string calib9_ladder = "[[10012, 256051], [20024, 199777], [40048, 110659], "
                                          "[80096, 79012], [160192, 64111], [320384, 63886], "
                                          "[640768, 42779], [1281536, 32781], [2563072, 53553]]";
        auto noisy = replace_line(cal_text, "effect_window_noise_ladder", calib9_ladder);
        noisy = replace_line(noisy, "tail_share_gain_ppm", "66249");
        noisy = replace_line(noisy, "tail_lane_row_ns", "4287");
        noisy = replace_line(noisy, "effect_rows_per_s", "175034");
        sv::ManifestInputs noisy_input = calibrated;
        noisy_input.calibration_cache = t.file("unresolvable-noise.toml", noisy);
        const auto gate_off = sv::build_manifest(noisy_input);
        const auto& off = gate_off.document.at("spec").at("handoff").at("placement").at("live_policy");
        assert(off.at("effect_gate") == false);
        assert(off.at("margin_ppm") == 23000 && off.at("max_moves") > 0);
        assert(gate_off.document.at("provenance")
                   .at("/spec/handoff/placement/live_policy/effect_gate")
                   .at("source")
                   .get<std::string>()
                   .find("cannot resolve predicted tail effect") != std::string::npos);
        assert(bc::serving::v2::read_manifest(gate_off.text, "unresolvable", bc::serving::v2::PlanUse::serving)
                   .ok());
        // Calib12-shaped: all planner measurements present, but no peer
        // reference on the tail LARGE lane and all effect fields absent.
        auto structural = cal_text;
        for (const char* name : {"effect_rows_per_s", "effect_noise_max_ns", "effect_row_noise_ns",
                                 "effect_window_noise_ladder"}) {
            const auto pos = structural.find(std::string(name) + " = ");
            assert(pos != std::string::npos);
            const auto end = structural.find('\n', pos);
            assert(end != std::string::npos);
            structural.erase(pos, end - pos + 1);
        }
        structural +=
            "absent = { effect_rows_per_s = \"tail large lane has no peer reference on this layout\", "
            "effect_noise_max_ns = \"tail large lane has no peer reference on this layout\", "
            "effect_row_noise_ns = \"tail large lane has no peer reference on this layout\", "
            "effect_window_noise_ladder = \"tail large lane has no peer reference on this layout\" }\n";
        sv::ManifestInputs structural_input = calibrated;
        structural_input.calibration_cache = t.file("calib12-structural.toml", structural);
        const auto structural_manifest = sv::build_manifest(structural_input);
        const auto& structural_policy =
            structural_manifest.document.at("spec").at("handoff").at("placement").at("live_policy");
        assert(structural_policy.at("effect_gate") == false);
        assert(structural_policy.at("max_moved_bytes") == 435200000ULL);
        assert(structural_policy.at("margin_ppm") == 23000);
        assert(structural_manifest.document.at("provenance")
                   .at("/spec/handoff/placement/live_policy/effect_gate")
                   .at("source")
                   .get<std::string>()
                   .find("tail large lane has no peer reference") != std::string::npos);
        assert(bc::serving::v2::read_manifest(structural_manifest.text, "calib12",
                                              bc::serving::v2::PlanUse::serving)
                   .ok());
        auto incomplete = replace_line(cal_text, "copy_bytes_per_s", "1");
        // A schema-valid but unusable positive reading is not a malformed file.
        no_policy(incomplete, "measured budget buys fewer");
        auto missing = cal_text;
        const std::string copy_line = "copy_bytes_per_s = 160000000\n";
        assert(missing.find(copy_line) != std::string::npos);
        missing.erase(missing.find(copy_line), copy_line.size());
        missing += "absent = { copy_bytes_per_s = \"no DMA observation\" }\n";
        no_policy(missing, "copy_bytes_per_s");
        auto malformed = calibrated;
        malformed.calibration_cache =
            t.file("malformed-live-policy.toml", cal_text + "unexpected_guess = 12\n");
        try {
            (void)sv::build_manifest(malformed);
            assert(false);
        } catch (const std::runtime_error& e) {
            assert(std::string(e.what()).find("unknown key") != std::string::npos);
        }
        malformed.calibration_cache =
            t.file("unparseable-live-policy.toml", cal_text + "unexpected_guess = \"unterminated\n");
        try {
            (void)sv::build_manifest(malformed);
            assert(false);
        } catch (const std::runtime_error& e) {
            assert(std::string(e.what()).find("TOML parse error") != std::string::npos);
        }

        sv::ManifestInputs forbidden = calibrated;
        forbidden.live_policy = lp;
        forbidden.live_policy_from = "--live-policy lab-policy.json";
        try {
            (void)sv::build_manifest(forbidden);
            assert(false);
        } catch (const std::runtime_error& e) {
            assert(std::string(e.what()).find("lab-only") != std::string::npos);
        }
        // The calibration run (bc-serve --calibration-run, the launcher's --calibrate): lab profile, the
        // explicit policy and no experiment. calibration-ingest accepts exactly that manifest, and refuses the
        // production one.
        sv::ManifestInputs calibration_run = in;
        calibration_run.profile = "lab";
        calibration_run.live_policy = lp;
        calibration_run.live_policy_from = "--live-policy policy.json";
        const auto cr = sv::build_manifest(calibration_run);
        assert(!cr.document.at("spec").contains("experiment"));
        // The record a calibration run files must be the one a production start of the same configuration looks up:
        // the lab and production manifests carry the same calibrated scope.
        assert(sv::policy_calibration_key(sv::policy_calibration_inputs(cr.document.at("spec"))) ==
                   sv::policy_calibration_key(sv::policy_calibration_inputs(built.document.at("spec"))) &&
               sv::policy_calibration_inputs(cr.document.at("spec")) ==
                   sv::policy_calibration_inputs(built.document.at("spec")));
        const auto cr_path = t.root / "calibration-run-manifest.json";
        const auto prod_path = t.root / "production-manifest.json";
        sv::write_manifest_file(cr_path, cr.text);
        sv::write_manifest_file(prod_path, built.text);
        const auto cr_loaded = bc::serving::v2::load_manifest(cr_path, bc::serving::v2::PlanUse::serving);
        assert(cr_loaded.ok());
        const auto& crm = cr_loaded.value();
        Json measured = {
            {"txn_p50_ns", 1770000000ULL},
            {"txn_p95_ns", 2720000000ULL},
            {"copy_bytes_per_s", 160000000ULL},
            {"heat_gap_max_ns", 5040000000ULL},
            {"heat_watermarked", 0},
            {"hot_set_k", 5931},
            {"marginal_error_ppm", 23000},
            {"marginal_observations", 3},
            {"drift_half_life_observations", 59000000},
            {"observation_rate_per_s", 10000000},
            {"tail_share_gain_ppm", 45000},
            {"tail_lane_service_ns_per_s", 750000000},
            {"lane_lost_ns", 100000000},
            {"effect_rows_per_s", 64},
            {"tail_lane_row_ns", 22222222},
            {"effect_noise_max_ns", 2500000},
            {"effect_row_noise_ns", 1550000},
            {"effect_window_noise_ladder",
             Json::array(
                 {Json::array({uint64_t(8), uint64_t(80000)}), Json::array({uint64_t(16), uint64_t(40000)}),
                  Json::array({uint64_t(32), uint64_t(16000)}), Json::array({uint64_t(64), uint64_t(10000)})})},
            {"abort_quiescent_ns", 1500000000ULL}};
        { // the v2 plan key covers the engine tuning settings (a changed TP split is another plan) and, in a
          // world of
            // more than one card type, every rank's card; an absent or empty tuning section keeps the key
            namespace v2 = bc::serving::v2;
            auto a = crm;
            a.spec.processes.engine.tuning.reset();
            const std::string k0 = v2::plan_cache_key(a);
            assert(!k0.empty());
            auto b = a;
            b.spec.processes.engine.tuning = v2::EngineTuning{};
            assert(v2::plan_cache_key(b) == k0);
            b.spec.processes.engine.tuning->tp_ownership = "kv=1,1,2;ffn=8,8,16";
            const std::string k1 = v2::plan_cache_key(b);
            assert(k1 != k0);
            b.spec.processes.engine.tuning->tp_ownership = "kv=1,1,2;ffn=8,16,8";
            assert(v2::plan_cache_key(b) != k1);
            if (a.spec.processes.engine.device.gpus.size() == 1 && a.spec.facts.gpus.size() > 1) {
                auto c = a; // a second rank on another card type moves the key
                for (const auto& g : c.spec.facts.gpus)
                    if (g.arch != c.spec.facts.gpus.front().arch &&
                        g.bdf != c.spec.processes.engine.device.gpus[0]) {
                        c.spec.processes.engine.device.gpus.push_back(g.bdf);
                        break;
                    }
                if (c.spec.processes.engine.device.gpus.size() == 2)
                    assert(v2::plan_cache_key(c) != k0);
            }
        }
        const Json row = {
            {"plan_cache_key", bc::serving::v2::plan_cache_key(crm)},
            {"layout", crm.spec.intent.layout},
            {"objective",
             std::string(
                 bc::serving::v2::kObjectiveNames[static_cast<size_t>(crm.spec.intent.service.objective)])},
            {"seeded", "2026-11-06"},
            {"evidence", "calibration-run ingest test"},
            {"measurements", measured},
            {"absent", Json::object()}};
        const auto helper_log = t.root / "calibration-helper0.log";
        {
            std::ofstream out(helper_log);
            out << "RHP16_CALIBRATION_JSON " << row.dump() << "\n";
        }
        const auto store = t.root / "calibration-run-store.toml";
        const std::string store_key = sv::policy_calibration_key(
            sv::policy_calibration_inputs(cr.document.at("spec")));
        try {
            sv::ingest_live_policy_log(helper_log, store, prod_path, store_key);
            assert(false);
        } catch (const std::runtime_error& e) {
            assert(std::string(e.what()).find("calibration-run manifest") != std::string::npos);
        }
        assert(!fs::exists(store));
        bool bad_scope_rejected = false;
        try {
            sv::ingest_live_policy_log(helper_log, store, cr_path, std::string(64, 'e'));
        } catch (const std::runtime_error& e) {
            bad_scope_rejected = std::string(e.what()).find("target policy key") != std::string::npos;
        }
        assert(bad_scope_rejected && !fs::exists(store));
        sv::ingest_live_policy_log(helper_log, store, cr_path, store_key);
        const auto ingested = sv::read_live_policy_record(store, store_key);
        assert(ingested && sv::live_policy_record_complete(*ingested) &&
               ingested->layout == crm.spec.intent.layout &&
               ingested->calibration_inputs &&
               *ingested->calibration_inputs == sv::policy_calibration_inputs(cr.document.at("spec")));
    }
    // The unmeasured manifest of the measuring start: owner map and settings, no plan and nothing derived from
    // one. A serving start refuses it, a measuring start reads it, and the measured manifest is the reverse.
    {
        sv::ManifestInputs u = in;
        u.measured.reset();
        const auto um = sv::build_manifest(u);
        const Json& us = um.document.at("spec");
        assert(us.at("plan").at("state") == "unmeasured" && !us.at("plan").contains("engine_memory"));
        assert(!us.at("resources").at("host").contains("checkpoint_slots") &&
               us.at("plan").at("owner_map").at("helpers").size() == 3);
        assert(us.at("resources").at("host").at("pool_mib") == spec.at("resources").at("host").at("pool_mib"));
        assert(um.document.at("provenance").at("/spec/resources").at("inputs").size() == 2);
        assert(!bc::serving::v2::read_manifest(um.text, "unmeasured", bc::serving::v2::PlanUse::serving).ok());
        assert(bc::serving::v2::read_manifest(um.text, "unmeasured", bc::serving::v2::PlanUse::measuring).ok());
        assert(
            !bc::serving::v2::read_manifest(built.text, "measured", bc::serving::v2::PlanUse::measuring).ok());
        assert(sv::build_manifest(u).text == um.text); // deterministic, like the measured one
    }
    // A live policy and a lab experiment come from operator-supplied objects, are written verbatim with their
    // own provenance record, and are read back by the shared reader; a partial policy or a production
    // experiment is an error, never a file.
    {
        sv::ManifestInputs c = in;
        c.profile = "lab";
        c.live_policy = Json::parse(
            R"json({"effect_gate": true, "accept_unwatermarked_heat": false, "cadence_ns": 60000000000,
            "max_moves": 8, "max_moved_bytes": 4294967296, "min_residency_ns": 300000000000, "max_heat_age_ns": 120000000000,
            "min_samples": 1000, "abort_cooldown_ns": 30000000000, "learned_eligibility_ttl_ns": 600000000000,
            "effect_window_samples": 5000, "effect_min_quiet_windows": 3, "effect_revert_ttl_ns": 120000000000,
            "margin_ppm": 100000, "half_life_observations_milli": 4000, "effect_noise_multiplier_milli": 2000,
            "protected_layers": [0, 1]})json");
        c.live_policy_from = "--live-policy policy.json";
        const auto with_policy = sv::build_manifest(c);
        assert(
            with_policy.document.at("spec").at("handoff").at("placement").at("live_policy").at("max_moves") ==
            8);
        const Json& rec = with_policy.document.at("provenance").at("/spec/handoff/placement/live_policy");
        assert(rec.at("origin") == "cli" && rec.at("source") == "--live-policy policy.json" &&
               rec.at("class") == "policy");
        assert(with_policy.digest != built.digest);
        sv::ManifestInputs partial = c;
        partial.live_policy.erase("margin_ppm");
        try {
            (void)sv::build_manifest(partial);
            assert(false);
        } catch (const std::runtime_error& e) {
            assert(std::string(e.what()).find("margin_ppm") != std::string::npos);
        }
        sv::ManifestInputs lab = in;
        lab.profile = "lab";
        lab.experiment = Json::parse(R"json({"placement": {"mode": "observe", "epoch_cycle_s": 30}})json");
        lab.experiment_from = "--experiment lab.json";
        const auto with_experiment = sv::build_manifest(lab);
        assert(with_experiment.document.at("spec").at("intent").at("service").at("profile") == "lab");
        assert(with_experiment.document.at("spec").at("experiment").at("placement").at("epoch_cycle_s") == 30);
        assert(with_experiment.document.at("provenance").at("/spec/experiment").at("origin") == "experiment");
        sv::ManifestInputs forbidden = lab;
        forbidden.profile = "production";
        try {
            (void)sv::build_manifest(forbidden);
            assert(false);
        } catch (const std::runtime_error& e) {
            assert(std::string(e.what()).find("needs the lab profile") != std::string::npos);
        }
    }
    // RSC11 A1: the three production engine options with no default are written only when set (--set), each at
    // its registry path, and an unset one leaves no trace in the document or the resolved typed view.
    {
        const Json& base_engine = built.document.at("spec").at("processes").at("engine");
        assert(!base_engine.at("execution").contains("tp_wire") &&
               !base_engine.at("execution").contains("no_prefix_cache"));
        assert(!base_engine.at("device").contains("headroom_mib"));
        assert(!r.typed.engine.tp_wire && !r.typed.engine.no_prefix_cache && !r.typed.engine.gpu_headroom_mib);
        Options with = opts;
        with.sets = {"engine.tp_wire=\"wht6\"", "engine.no_prefix_cache=\"on\"", "engine.gpu_headroom_mib=320"};
        const auto r2 = sv::resolve(with, t.facts);
        assert(r2.typed.engine.tp_wire == "wht6" && r2.typed.engine.no_prefix_cache &&
               r2.typed.engine.gpu_headroom_mib == 320U);
        assert(r2.provenance.at("engine.tp_wire").at("origin") == "cli");
        sv::ManifestInputs c = in;
        c.resolved = r2;
        c.measured.reset();
        const auto set_built = sv::build_manifest(c);
        const Json& eng = set_built.document.at("spec").at("processes").at("engine");
        assert(eng.at("execution").at("tp_wire") == "wht6" &&
               eng.at("execution").at("no_prefix_cache") == "on");
        assert(eng.at("device").at("headroom_mib") == 320);
        assert(
            bc::serving::v2::read_manifest(set_built.text, "options test", bc::serving::v2::PlanUse::measuring)
                .ok());
        Options bad = opts;
        bad.sets = {"engine.tp_wire=\"lossy\""};
        try {
            (void)sv::resolve(bad, t.facts);
            assert(false);
        } catch (const std::runtime_error& e) {
            assert(std::string(e.what()).find("tp_wire") != std::string::npos);
        }
    }
    // The helper section follows the resolver (a layout, the site file or --set), not the table's defaults.
    {
        sv::ManifestInputs c = in;
        c.resolved.values["helper"]["combine"] = true;
        c.resolved.values["helper"]["combine_min_rows"] = 512;
        c.resolved.values["helper"]["combine_threads"] = 4;
        const Json hc =
            sv::build_manifest(c).document.at("spec").at("processes").at("helper_common").at("combine");
        assert(hc.at("enabled") == true && hc.at("min_rows") == 512 && hc.at("threads") == 4);
        c.resolved.values["helper"]["combine"] = false;
        assert(sv::build_manifest(c)
                   .document.at("spec")
                   .at("processes")
                   .at("helper_common")
                   .at("combine")
                   .at("enabled") == false);
    }
    // One value for the host KV tier: the ledger plans for the manifest's kv_mib, and a calibration target that
    // disagrees is refused.
    {
        sv::ManifestInputs c = in;
        c.resolved.values["memory"]["host_kv_mib"] = 8192;
        try {
            (void)sv::build_manifest(c);
            assert(false);
        } catch (const std::runtime_error& e) {
            assert(std::string(e.what()).find("differs from calibration host_kv_tier.target_mib") !=
                   std::string::npos);
        }
    }
    // An operator override keeps its own origin: --set and the site file are never filed under the section's
    // default.
    {
        sv::ManifestInputs c = in;
        c.resolved.provenance["service.max_sessions"] = {
            {"origin", "cli"}, {"source", "--set"}, {"inputs", Json::array({"service.max_sessions=4"})}};
        c.resolved.provenance["engine.port"] = {
            {"origin", "site"}, {"source", "site.toml"}, {"inputs", Json::array()}};
        c.resolved.provenance["engine.prefix_resume"] = {
            {"origin", "derived"}, {"source", "rule"}, {"inputs", Json::array()}};
        const auto overridden = sv::build_manifest(c);
        const Json& pv = overridden.document.at("provenance");
        assert(pv.at("/spec/intent/service/max_sessions").at("origin") == "cli" &&
               pv.at("/spec/intent/service/max_sessions").at("source") == "--set");
        assert(pv.at("/spec/intent/service/max_sessions").at("class") ==
               pv.at("/spec/intent/service").at("class"));
        assert(pv.at("/spec/processes/engine/server/port").at("origin") == "site");
        assert(pv.at("/spec/processes/engine/server/port").at("source") == "site.toml");
        assert(!pv.contains(
            "/spec/processes/engine/execution/prefix_resume")); // derived keeps its section's record
        assert(!built.document.at("provenance").contains("/spec/intent/service/max_sessions"));
    }
    // Same inputs, same bytes; another instance, another digest; nothing else moves it.
    assert(sv::build_manifest(in).text == built.text);
    sv::ManifestInputs other = in;
    other.instance_id = "22222222-2222-4222-8222-222222222222";
    assert(sv::build_manifest(other).digest != built.digest);
    // A model it cannot open, or a plan that is not this configuration's, is an error, never a file.
    sv::ManifestInputs bad = in;
    bad.resolved.values["engine"]["model"] = (t.root / "no-such.rad").string();
    try {
        (void)sv::build_manifest(bad);
        assert(false);
    } catch (const std::runtime_error& e) {
        assert(std::string(e.what()).find("cannot open the model container") != std::string::npos);
    }
    // Written once, read-only, atomically; an existing path is never replaced.
    const auto path = t.root / "written-manifest.json";
    sv::write_manifest_file(path, built.text);
    struct stat st{};
    assert(::stat(path.c_str(), &st) == 0 && (st.st_mode & 0777U) == 0444 &&
           !fs::exists(path.string() + ".tmp"));
    assert(bc::serving::v2::load_manifest(path, bc::serving::v2::PlanUse::serving).ok());
    try {
        sv::write_manifest_file(path, built.text);
        assert(false);
    } catch (const std::runtime_error& e) {
        assert(std::string(e.what()).find("immutable") != std::string::npos);
    }
    // Racing writers to one path: exactly one wins, the others are refused, the winner's bytes are intact, and
    // no temporary file is left behind.
    const auto race = t.root / "raced-manifest.json";
    std::atomic<int> won{0}, refused{0};
    std::vector<std::thread> writers;
    for (int i = 0; i < 8; ++i)
        writers.emplace_back([&, i] {
            try {
                sv::write_manifest_file(race, "writer-" + std::to_string(i) + "\n");
                ++won;
            } catch (const std::runtime_error& e) {
                if (std::string(e.what()).find("immutable") != std::string::npos)
                    ++refused;
            }
        });
    for (auto& w : writers)
        w.join();
    assert(won == 1 && refused == 7);
    for (const auto& e : fs::directory_iterator(t.root))
        assert(e.path().filename().string().find("raced-manifest.json.tmp") == std::string::npos);
    std::ifstream in_race(race);
    std::string got((std::istreambuf_iterator<char>(in_race)), {});
    assert(got.rfind("writer-", 0) == 0 && got.back() == '\n');
}

// The named layouts' unmeasured manifests, digested: the layout data in layouts.toml must produce exactly these
// documents (RSC11 A2). `bc_serving_test --write-golden` rewrites golden/layout_manifests.txt (a deliberate
// change, reviewed against the old file); `--dump-manifests DIR` writes the documents.
void layout_manifests_golden(Fixture& t, bool write_golden, const char* dump_dir) {
    namespace sv = bc::serving;
    const auto model = t.root / "golden-model.rad";
    bc::test::write_rad(model, 4, 16, 128, 128, {});
    const auto heat = t.root / "golden.hot-first";
    {
        std::vector<uint32_t> counts(128 * 16, 0);
        for (uint32_t l = 0; l < 4; ++l)
            for (uint32_t e = 0; e < 16; ++e)
                counts[(l * 16) + e] = (((e * 7) + (l * 3)) % 16) + 1;
        std::ofstream(heat, std::ios::binary)
            .write(reinterpret_cast<const char*>(counts.data()),
                   static_cast<std::streamsize>(counts.size() * sizeof(counts[0])));
    }
    std::string got;
    for (const char* name :
         {"flashnext-4gpu", "flashnext-2xtx", "flashnext-2xtx-combine", "flashnext-2xtx-rdna2-head",
          "flashnext-2xtx-rdna2-tail", "flashnext-2xtx-rdna2-tail-combine", "flashnext-tp2xtx-r9700",
          "flashnext-tp2xtx"}) {
        Options opts;
        opts.site = t.file("golden-site.toml", "[engine]\nmodel = \"" + model.string() +
                                                   "\"\nradiance_home = \"/runtime\"\n"
                                                   "[handoff]\nheat = \"" +
                                                   heat.string() + "\"\nhistogram = \"/heat.live\"\n" +
                                                   std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
        opts.sets = {std::string("layout.name=\"") + name + "\""};
        sv::ManifestInputs in;
        in.resolved = sv::resolve(opts, t.facts);
        in.instance_id = "11111111-1111-4111-8111-111111111111";
        in.producer_sha256 = std::string(64, 'a');
        in.profile = "production";
        in.s10.power_profile = "stock";
        const auto built = sv::build_manifest(in);
        if (dump_dir)
            write(fs::path(dump_dir) / (std::string(name) + ".json"), built.text);
        // Normalised to what the layout decides: not this run's directory, the host's free /dev/shm, or the
        // digest they feed.
        std::string text = built.text;
        for (size_t at; (at = text.find(t.root.string())) != std::string::npos;)
            text.replace(at, t.root.string().size(), "<root>");
        text =
            std::regex_replace(text, std::regex("\"shm_available_mib\": [0-9]+"), "\"shm_available_mib\": 0");
        text = std::regex_replace(text, std::regex("\"content_digest\": \"sha256:[0-9a-f]+\""),
                                  "\"content_digest\": \"-\"");
        got += std::string(name) + " sha256:" + sv::sha256_hex(text) + "\n";
    }
    const fs::path golden = fs::path(__FILE__).parent_path() / "golden/layout_manifests.txt";
    if (write_golden) {
        write(golden, got);
        return;
    }
    std::ifstream in(golden);
    std::string want((std::istreambuf_iterator<char>(in)), {});
    if (want != got)
        throw std::runtime_error("layout manifests differ from golden/layout_manifests.txt:\n" + got);
}

// An engine-only layout (contract amendment 12): two RDNA3 ranks, no helpers, no handoff, read back by the
// strict reader.
void engine_only_layout(Fixture& t) {
    namespace sv = bc::serving;
    // A dense model: a container with no routed experts, which the expert reader refuses. Engine-only needs no
    // experts.
    const auto model = t.root / "dense-model.rad";
    bc::test::write_rad(model, 0, 0, 128, 128, {});
    {
        bc::experts::Container refused;
        assert(!refused.open(model.c_str()));
    }
    Options opts;
    opts.site = t.file("dense-site.toml", "[engine]\nmodel = \"" + model.string() +
                                              "\"\nradiance_home = \"/runtime\"\n" +
                                              std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
    opts.sets = {"layout.name=\"qwen27b-2xtx\"", "service.max_sessions=6", "service.max_context=8192"};
    const auto r = sv::resolve(opts, t.facts); // no [handoff] heat/histogram: nothing here uses them
    assert(r.typed.layout.engine_cards.size() == 2 && r.typed.layout.helper_cards.empty());
    { // no helper, so no combine threads: a host with fewer CPUs than the combine default resolves it
        Json few = t.facts;
        few["host"]["cpu_count"] = 4;
        (void)sv::resolve(opts, few);
    }
    assert(r.typed.engine.tp == 2 && r.typed.engine.tp_wire == "wht6" &&
           r.typed.engine.kernels == "libr3,libref");
    assert(r.typed.helper.count == 0 && r.manifest.at("process_helpers").empty());
    assert(r.provenance.at("engine.tp").at("origin") == "derived");
    sv::ManifestInputs in;
    in.resolved = r;
    in.instance_id = "11111111-1111-4111-8111-111111111111";
    in.producer_sha256 = std::string(64, 'a');
    in.profile = "production";
    in.s10.power_profile = "stock";
    const auto built = sv::build_manifest(in);
    const Json& spec = built.document.at("spec");
    assert(!spec.contains("handoff") && !spec.at("processes").contains("helper_common"));
    assert(spec.at("processes").at("helpers").empty() && spec.at("plan").at("owner_map").at("helpers").empty());
    assert(spec.at("processes").at("engine").at("device").at("gpus").size() == 2);
    assert(spec.at("processes").at("engine").at("execution").at("tp_wire") == "wht6");
    assert(spec.at("processes").at("engine").at("runtime").at("kernels") == Json::array({"libr3", "libref"}));
    assert(!built.document.at("provenance").contains("/spec/handoff/transport") &&
           !built.document.at("provenance").contains("/spec/processes/helper_common"));
    assert(bc::serving::v2::read_manifest(built.text, "engine-only", bc::serving::v2::PlanUse::measuring).ok());
    // the sections tie to the helpers both ways, and an engine-only manifest refuses a helper-side experiment
    Json with_handoff = built.document;
    with_handoff["spec"]["handoff"] = Json::object();
    assert(!bc::serving::v2::read_manifest(with_handoff.dump(), "x", bc::serving::v2::PlanUse::measuring).ok());
    // Amendment 13: engine options with a field of their own are `section.name` keys and manifest leaves; a
    // switch's one value is "on"
    {
        Options more = opts;
        more.sets = {"layout.name=\"qwen27b-2xtx\"",
                     "sampling.temp=\"0.7\"",
                     "sampling.top_k=40",
                     "limits.max_body_mib=64",
                     "limits.no_cors=\"on\"",
                     "tuning.p2p=\"off\"",
                     "vision.video_fps=\"2\"",
                     "diagnostics.debug_placement=\"on\"",
                     "kernels.moe_native=true",
                     "kernels.dec11=false"};
        sv::ManifestInputs mi = in;
        mi.resolved = sv::resolve(more, t.facts);
        const auto with_options = sv::build_manifest(mi);
        const Json& e = with_options.document.at("spec").at("processes").at("engine");
        assert(e.at("sampling").at("temp") == "0.7" && e.at("sampling").at("top_k") == 40);
        assert(e.at("limits").at("max_body_mib") == 64 && e.at("limits").at("no_cors") == "on");
        assert(e.at("tuning").at("p2p") == "off" && e.at("vision").at("video_fps") == "2" &&
               e.at("diagnostics").at("debug_placement") == "on");
        assert(e.at("kernels") == Json({{"moe_native", true},
                                        {"dec11", false}})); // amendment 14: environment settings are keys too
        assert(!e.contains("lora") && !e.contains("chat"));  // a section appears only with its members
        assert(bc::serving::v2::read_manifest(with_options.text, "options", bc::serving::v2::PlanUse::measuring)
                   .ok());
        assert(
            bc::serving::engine_option_fields().size() ==
            96 /* VTP44: --expert-route-record and --expert-route-replay join; VTP42: four prefill options become one governor spec (87 flags + 9 environment settings) */);
        Options bad = opts;
        bad.sets = {"layout.name=\"qwen27b-2xtx\"", "sampling.no_such_option=1"};
        fail(bad, t.facts, "no_such_option");
        bad.sets = {"layout.name=\"qwen27b-2xtx\"", "limits.max_body_mib=\"big\""};
        fail(bad, t.facts, "max_body_mib");
        bad.sets = {"layout.name=\"qwen27b-2xtx\"", "limits.no_cors=\"off\""};
        fail(bad, t.facts, "no_cors");
    }
    // a layout with helpers still needs its heat and histogram files
    Options helpers = opts;
    // A helper layout needs routed experts before it can derive a VRAM budget;
    // keep this test focused on the missing handoff inputs.
    helpers.site =
        t.file("helper-required-site.toml", "[engine]\nmodel = \"" + (t.root / "model.rad").string() +
                                                "\"\nradiance_home = \"/runtime\"\n" +
                                                std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
    helpers.sets = {"layout.name=\"flashnext-2xtx\""};
    fail(helpers, t.facts, "missing required key: handoff.heat");
    // a two-rank engine without its transport is refused
    Options no_wire = opts;
    no_wire.sets = {"layout.name=\"qwen27b-2xtx\"", "engine.tp_wire=\"exact\""};
    (void)sv::resolve(no_wire, t.facts);
}

// RAD49: a tensor-parallel engine (the two RDNA3 cards) with the one remote-MoE helper the mailbox protocol
// allows behind it. The expert budget bc-serve derives for an omitted budget-mib (RAD49): the card less its
// archs.toml reserve less the helper's own lanes, pointer banks, slot map and incoming slot
// (worker_footprint.h, which the worker checks its lanes against), for this model and this helper's lanes (one
// small class of handoff.small_rows, the large lane of handoff.max_rows, power save on).
uint64_t derived_budget_mib(const fs::path& model, uint64_t card_mib, uint64_t reserve_mib, uint32_t helpers,
                            uint32_t index, uint32_t world, const r3::worker_footprint::SmallClasses& classes,
                            uint32_t max_rows, bool power_save) {
    namespace fp = r3::worker_footprint;
    bc::experts::Container c;
    MoeShape cs{};
    assert(c.open(model.c_str()) && bc::experts::container_shape(c, &cs));
    uint64_t largest = 0;
    for (uint32_t l = 0; l < c.layers; ++l)
        for (uint32_t e = 0; e < c.experts; ++e)
            if (c.weight(l, e, 0) && c.weight(l, e, 1))
                largest = std::max(largest, uint64_t(bc::experts::laid_out_bytes(c, true, l, e)));
    const fp::Lanes lanes = fp::lanes_for(classes, max_rows, helpers, index, world, power_save);
    const fp::Shape shape{c.layers,
                          cs.experts,
                          cs.hidden,
                          cs.intermediate,
                          cs.top_k ? cs.top_k : r3::moe_shm::kTopK,
                          r3::moe_shm::kActGroup};
    const uint64_t fixed = fp::fixed_bytes(shape, lanes, fp::stride_bound(largest));
    const uint64_t budget = fp::expert_budget_bytes(card_mib << 20, reserve_mib << 20, fixed) >> 20;
    assert(budget > 0 &&
           budget < card_mib - reserve_mib); /* strictly less than the old whole-card derivation */
    return budget;
}

/* the schema defaults: helper_profile.toml's empty small-lane list (one class of handoff.small_rows 8),
 * max_rows 4096, power save */
uint64_t derived_budget_mib(const fs::path& model, uint64_t card_mib, uint64_t reserve_mib, uint32_t helpers,
                            uint32_t index, uint32_t world) {
    return derived_budget_mib(model, card_mib, reserve_mib, helpers, index, world,
                              r3::worker_footprint::small_classes({}, 8, 4096), 4096, true);
}

/* PR #771 review: the budget bc-serve derives is the one the worker's own settings imply. For both power-save
 * modes the manifest's helper_common carries the resolved value, and each helper's settings as the worker reads
 * them (v1 helper settings: its small lanes, hand-off rows, index, engine width, power save) give, through the
 * worker's functions (small_classes, lanes_for), exactly the budget the manifest declares. An experiment that
 * patches a helper's lanes is refused while the budgets are derived. */
void derived_budget_is_the_workers(Fixture& t) {
    namespace sv = bc::serving;
    const auto model = t.root / "derived-budget-model.rad";
    bc::test::write_rad(model, 4, 16, 128, 128, {});
    const auto heat = t.root / "derived-budget.hot-first";
    {
        std::vector<uint32_t> counts(128 * 16, 1);
        std::ofstream(heat, std::ios::binary)
            .write(reinterpret_cast<const char*>(counts.data()),
                   static_cast<std::streamsize>(counts.size() * sizeof(counts[0])));
    }
    uint64_t budgets_on[2] = {}, budgets_off[2] = {};
    for (const bool power_save : {true, false}) {
        Options opts;
        opts.site =
            t.file("derived-budget-site.toml", "[engine]\nmodel = \"" + model.string() +
                                                   "\"\nradiance_home = \"/runtime\"\n"
                                                   "[handoff]\nheat = \"" +
                                                   heat.string() + "\"\nhistogram = \"/heat.live\"\n" +
                                                   std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
        opts.sets = {"layout.name=\"flashnext-tp2xtx\"",
                     std::string("helper.power_save=") + (power_save ? "true" : "false")};
        sv::ManifestInputs in;
        in.resolved = sv::resolve(opts, t.facts);
        in.instance_id = "11111111-1111-4111-8111-111111111111";
        in.producer_sha256 = std::string(64, 'a');
        in.profile = "lab";
        in.s10.power_profile = "stock";
        const auto built = sv::build_manifest(in);
        assert(built.document.at("spec").at("processes").at("helper_common").at("runtime").at("power_save") ==
               power_save);
        auto read =
            bc::serving::v2::read_manifest(built.text, "derived-budget", bc::serving::v2::PlanUse::measuring);
        assert(read.ok());
        const uint64_t card[2] = {32624, 16368}, reserve[2] = {2048, 1024};
        for (uint32_t i = 0; i < 2; ++i) {
            auto s = bc::serving::helper_settings(read.value(), "helper" + std::to_string(i));
            assert(s.ok());
            const auto& hs = s.value();
            assert(hs.helper.power_save == power_save); /* the worker runs the mode the budget was sized for */
            const auto classes =
                r3::worker_footprint::small_classes(hs.small_lanes, hs.handoff.small_rows, hs.handoff.max_rows);
            const uint64_t want =
                derived_budget_mib(model, card[i], reserve[i], hs.helper.count, hs.helper.index,
                                   hs.helper.engine_ranks, classes, hs.handoff.max_rows, hs.helper.power_save);
            assert(hs.budget_mib == want);
            (power_save ? budgets_on : budgets_off)[i] = hs.budget_mib;
        }
        if (!power_save) { /* an experiment patching a helper's lanes while the budgets are derived is refused
                            */
            sv::ManifestInputs patched = in;
            patched.experiment = Json{
                {"shape",
                 {{"helpers",
                   Json::array({Json{{"role", "helper1"},
                                     {"settings", {{"lanes", {{"small_lanes", Json::array({4, 8})}}}}}}})}}}};
            patched.experiment_from = "test";
            bool refused = false;
            try {
                (void)sv::build_manifest(patched);
            } catch (const std::exception& e) {
                refused = std::string(e.what()).find("budget-mib") != std::string::npos;
                if (!refused)
                    std::fprintf(stderr, "derived_budget_is_the_workers: refused for another reason: %s\n",
                                 e.what());
            }
            assert(refused);
        }
    }
    /* power save off opens every protocol lane: more lanes, smaller budgets */
    assert(budgets_off[0] < budgets_on[0] && budgets_off[1] < budgets_on[1]);
}

/* A world of two card types: engine groups in rank order, one plugin each, and every rank's card in the plan
 * key. */
void mixed_arch_engine_layout(Fixture& t) {
    namespace sv = bc::serving;
    const auto model = t.root / "mixed-model.rad";
    bc::test::write_rad(model, 0, 0, 128, 128, {});
    Options opts;
    opts.site = t.file("mixed-site.toml",
                       "[engine]\nmodel = \"" + model.string() + "\"\nradiance_home = \"/runtime\"\n");
    opts.layouts = {t.file(
        "mixed-layouts.toml",
        "schema = 1\n[layout.mixed-tp3]\nmodel = \"m.rad\"\n"
        "engine = [{ arch = \"gfx1100\", count = 2 }, { arch = \"gfx1201\", count = 1 }]\ntp-wire = \"exact\"\n"
        "[layout.mixed-tp3.settings]\ntuning.tp_ownership = \"kv=1,1,2;ffn=8,8,16\"\ntuning.pipelined_prefill "
        "= \"on\"\n"
        "tuning.tune_pin = \"all_reduce:nblocks=4,nthreads=256\"\n"
        "[layout.badtuning]\nmodel = \"m.rad\"\nengine = { arch = \"gfx1201\" "
        "}\n[layout.badtuning.settings]\ntuning.no_such = 1\n"
        "[layout.twice]\nmodel = \"m.rad\"\nengine = [{ arch = \"gfx1100\" }, { arch = \"gfx1100\" }]\ntp-wire "
        "= \"exact\"\n")};
    opts.sets = {"layout.name=\"mixed-tp3\"", "service.max_sessions=4", "service.max_context=8192"};
    const auto r = sv::resolve(opts, t.facts);
    const auto& g = t.facts.at("gpus");
    assert(r.typed.layout.engine_cards.size() == 3);
    for (size_t i = 0; i < 3; ++i)
        assert(r.typed.layout.engine_cards[i] == g[i].at("bdf").get<std::string>());
    assert(r.typed.engine.tp == 3 && r.typed.engine.kernels == "libr3,libr4d,libref");
    // a mixed world hands the engine its ranks' cards (--rank-devices); the layout's tuning is a policy layer
    assert(r.values.at("layout").at("rank_devices") == r.values.at("layout").at("engine_cards"));
    assert(r.values.at("tuning").at("tp_ownership") == "kv=1,1,2;ffn=8,8,16");
    assert(r.values.at("tuning").at("pipelined_prefill") == "on");
    assert(r.values.at("tuning").at("tune_pin") == "all_reduce:nblocks=4,nthreads=256");
    assert(r.provenance.at("tuning.tp_ownership").at("source").get<std::string>().find("layout mixed-tp3") ==
           0);
    {
        Options over = opts; // a profile (or --set) overrides the layout's tuning
        over.sets.push_back("tuning.pipelined_prefill=\"off\"");
        assert(sv::resolve(over, t.facts).values.at("tuning").at("pipelined_prefill") == "off");
        over.sets = {"layout.name=\"badtuning\""};
        fail(over, t.facts, "unknown key tuning.no_such");
        over.layouts = {t.file(
            "oldenv.toml", "schema = 1\n[layout.oldenv]\nmodel = \"m.rad\"\nengine = { arch = \"gfx1201\" }\n"
                           "[layout.oldenv.env]\nR3_MOE_WORKER_COMBINE = \"1\"\n")};
        over.sets = {"layout.name=\"oldenv\""};
        fail(over, t.facts, "env is not a layout key");
    }
    const Json key = sv::plan_key_inputs(r.values, t.facts, "m", "e", "");
    assert(key.contains("rank_cards") && key.at("rank_cards").size() == 3 &&
           key.at("rank_cards")[2].at("arch") == "gfx1201");
    opts.sets = {"layout.name=\"twice\""};
    fail(opts, t.facts, "in two groups");
    // a single-architecture world keys as before: no rank_cards member
    Options dense = opts;
    dense.layouts.clear();
    dense.sets = {"layout.name=\"qwen27b-2xtx\"", "service.max_sessions=6", "service.max_context=8192"};
    const auto uniform = sv::resolve(dense, t.facts);
    assert(!sv::plan_key_inputs(uniform.values, t.facts, "m", "e", "").contains("rank_cards"));
    assert(!uniform.values.at("layout").contains("rank_devices")); // a uniform world needs no --rank-devices
}

void tp_engine_with_helper_layout(Fixture& t) {
    namespace sv = bc::serving;
    const auto model = t.root / "tp-helper-model.rad";
    bc::test::write_rad(model, 4, 16, 128, 128, {});
    const auto heat = t.root / "tp-helper.hot-first";
    {
        std::vector<uint32_t> counts(128 * 16, 1);
        std::ofstream(heat, std::ios::binary)
            .write(reinterpret_cast<const char*>(counts.data()),
                   static_cast<std::streamsize>(counts.size() * sizeof(counts[0])));
    }
    Options opts;
    opts.site = t.file("tp-helper-site.toml", "[engine]\nmodel = \"" + model.string() +
                                                  "\"\nradiance_home = \"/runtime\"\n"
                                                  "[handoff]\nheat = \"" +
                                                  heat.string() + "\"\nhistogram = \"/heat.live\"\n" +
                                                  std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
    opts.sets = {"layout.name=\"flashnext-tp2xtx-r9700\"", "service.max_sessions=6",
                 "service.max_context=65536"};
    const auto r = sv::resolve(opts, t.facts);
    assert(r.typed.layout.engine_cards.size() == 2 && r.typed.layout.helper_cards.size() == 1);
    assert(r.typed.engine.tp == 2 && r.typed.engine.tp_wire == "exact");
    assert(r.typed.engine.kernels == "libr3,libr3moe,libref");
    assert(r.typed.helper.count == 1 && r.typed.helper.engine_ranks == 2);
    // no budget-mib in the layout: the helper's card (the fixture's R9700, 32624 MiB) less its archs.toml
    // reserve, less its own lanes and tables (one slot a TP rank: both lanes of both ranks), the planner's and
    // the worker's one derivation
    assert(r.values.at("memory").at("helper_budgets_mib") ==
           Json::array({derived_budget_mib(model, 32624, 2048, 1, 0, 2)}));
    assert(r.provenance.at("memory.helper_budgets_mib").at("origin") == "derived");
    sv::ManifestInputs in;
    in.resolved = r;
    in.instance_id = "11111111-1111-4111-8111-111111111111";
    in.producer_sha256 = std::string(64, 'a');
    in.profile = "production";
    in.s10.power_profile = "stock";
    const auto built = sv::build_manifest(in);
    const Json& spec = built.document.at("spec");
    assert(spec.at("processes").at("engine").at("device").at("gpus").size() == 2);
    assert(spec.at("processes").at("engine").at("scheduler").at("tp") == 2);
    assert(spec.at("processes").at("engine").at("execution").at("tp_wire") == "exact");
    assert(spec.at("processes").at("helpers").size() == 1 && spec.contains("handoff"));
    assert(bc::serving::v2::read_manifest(built.text, "tp-helper", bc::serving::v2::PlanUse::measuring).ok());
    // the transport is required with helpers too
    Options no_wire = opts;
    no_wire.sets.push_back("engine.tp_wire=\"wht6\"");
    assert(sv::resolve(no_wire, t.facts).typed.engine.tp_wire == "wht6");
}

// RAD49 steps 3-4: two helpers behind a TP2 engine. The budgets are each card less its archs.toml reserve (no
// budget-mib), the RX 6900 XT is the tail, the combine is on, and the manifest carries both helpers behind the
// two ranks.
void tp_engine_with_two_helpers_layout(Fixture& t) {
    namespace sv = bc::serving;
    const auto model = t.root / "tp-two-helper-model.rad";
    bc::test::write_rad(model, 4, 16, 128, 128, {});
    const auto heat = t.root / "tp-two-helper.hot-first";
    {
        std::vector<uint32_t> counts(128 * 16, 1);
        std::ofstream(heat, std::ios::binary)
            .write(reinterpret_cast<const char*>(counts.data()),
                   static_cast<std::streamsize>(counts.size() * sizeof(counts[0])));
    }
    Options opts;
    opts.site = t.file("tp-two-helper-site.toml", "[engine]\nmodel = \"" + model.string() +
                                                      "\"\nradiance_home = \"/runtime\"\n"
                                                      "[handoff]\nheat = \"" +
                                                      heat.string() + "\"\nhistogram = \"/heat.live\"\n" +
                                                      std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
    opts.sets = {"layout.name=\"flashnext-tp2xtx\""};
    const auto r = sv::resolve(opts, t.facts);
    assert(r.typed.layout.engine_cards.size() == 2 && r.typed.layout.helper_cards.size() == 2);
    assert(r.typed.engine.tp == 2 && r.typed.engine.tp_wire == "exact");
    assert(r.typed.helper.count == 2 && r.typed.helper.engine_ranks == 2 && r.typed.helper.combine);
    // no budget-mib: the R9700 (32624 MiB) less 2048, the RX 6900 XT (16368 MiB) less 1024; the tail is the
    // last one
    const uint64_t head = derived_budget_mib(model, 32624, 2048, 2, 0, 2),
                   tail = derived_budget_mib(model, 16368, 1024, 2, 1, 2);
    assert(r.values.at("memory").at("helper_budgets_mib") == Json::array({head, tail}));
    assert(r.typed.memory.tail_budget_mib == tail);
    sv::ManifestInputs in;
    in.resolved = r;
    in.instance_id = "11111111-1111-4111-8111-111111111111";
    in.producer_sha256 = std::string(64, 'a');
    in.profile = "production";
    in.s10.power_profile = "stock";
    const auto built = sv::build_manifest(in);
    const Json& spec = built.document.at("spec");
    assert(spec.at("processes").at("engine").at("scheduler").at("tp") == 2);
    assert(spec.at("processes").at("helpers").size() == 2 && spec.contains("handoff"));
    // the manifest's helper budgets are the derived expert budgets the owner map was planned with
    // (build_manifest checks the map against them with verify_owner_map), and the ones the worker sizes its
    // arena from
    assert(spec.at("processes").at("helpers")[0].at("vram_budget_mib") == head &&
           spec.at("processes").at("helpers")[1].at("vram_budget_mib") == tail);
    assert(
        bc::serving::v2::read_manifest(built.text, "tp-two-helper", bc::serving::v2::PlanUse::measuring).ok());
    // a helper count other than the cards the layout selects is refused (the typed v26 representation of
    // helpers x ranks is checked on the same path; moe_remote_shm_test covers slots_fit)
    Options three = opts;
    three.sets = {"layout.name=\"flashnext-tp2xtx\"", "helper.count=3"};
    bool refused = false;
    try {
        (void)sv::resolve(three, t.facts);
    } catch (const std::exception&) {
        refused = true;
    }
    assert(refused);
}

/* RSC02 S16: preserve the production static session identity invariant.
 * This exercises real RSC12 named configs and the native resolver, but does
 * not inspect or emulate retired shell launcher text. The old layout-vs-shell
 * parity test was retired with the launcher and is no longer an acceptance gate. */
void native_static_resolution_identity(Fixture& t) {
    namespace sv = bc::serving;
    const char* env = std::getenv("BIGCHERRY_LAYOUTS");
    if (!env || !fs::is_directory(env))
        throw std::runtime_error("native_static_resolution_identity: BIGCHERRY_LAYOUTS required");
    const fs::path configs = fs::path(env).parent_path() / "configs";
    const auto model = t.root / "parity-model.rad";
    bc::test::write_rad(model, 4, 16, 128, 128, {});
    const auto heat = t.root / "parity.hot-first";
    {
        std::vector<uint32_t> counts(128 * 16, 1);
        std::ofstream(heat, std::ios::binary)
            .write(reinterpret_cast<const char*>(counts.data()),
                   static_cast<std::streamsize>(counts.size() * sizeof(counts[0])));
    }
    const auto site = t.file("parity-site.toml", "[engine]\nmodel = \"" + model.string() +
                                                     "\"\nradiance_home = \"/runtime\"\n"
                                                     "[handoff]\nheat = \"" +
                                                     heat.string() + "\"\nhistogram = \"/heat.live\"\n");
    // RED P0-1: project the real RSC12 config resolver's producer values,
    // not a hand-built "manifest-shaped" JSON. The actual writer chooses
    // ephemeral shm and S8-derived pool/KV, which a replay must not turn into
    // new operator settings when authenticating the same static resolution.
    {
        namespace ss = bc::serving::session;
        Options selected;
        selected.site = site;
        selected.configs = {configs};
        selected.model_root = t.root;
        selected.sets = {"layout.name=\"flashnext-tp2xtx-r9700\""};
        const Resolved actual = sv::resolve(selected, t.facts);
        ss::StaticResolutionInputs input;
        input.effective_values = actual.values;
        input.ordered_devices = Json::array();
        for (const char* group : {"engine_cards", "helper_cards"}) {
            for (const Json& selected_bdf : actual.values.at("layout").at(group)) {
                bool found = false;
                for (const Json& gpu : t.facts.at("gpus")) {
                    if (gpu.at("bdf") != selected_bdf)
                        continue;
                    input.ordered_devices.push_back({{"role_group", group},
                                                     {"bdf", selected_bdf},
                                                     {"arch", gpu.at("arch")},
                                                     {"hip_ordinal", gpu.at("hip_ordinal")},
                                                     {"vram_total_mib", gpu.at("vram_total_mib")}});
                    found = true;
                    break;
                }
                assert(found);
            }
        }
        input.producer_digest = ss::complete_file_identity("/proc/self/exe");
        input.engine_digest = ss::complete_file_identity(BC_FAKE_ENGINE);
        input.helper_digest = ss::complete_file_identity(BC_FAKE_ENGINE);
        input.model_digest = ss::complete_file_identity(model);
        input.power_profile = "stock";
        input.file_digests[site.string()] = ss::complete_file_identity(site);
        for (const fs::path& file : sv::config_sources({configs}))
            input.file_digests[file.string()] = ss::complete_file_identity(file);
        const auto original_key = ss::static_resolution_key(input);
        input.effective_values["handoff"]["shm"] = "/r3-serve-immutable-real-fixture";
        input.effective_values["memory"]["host_pool_mib"] =
            actual.values.at("memory").at("host_pool_mib").get<int64_t>() + 256;
        input.effective_values["memory"]["host_kv_mib"] =
            actual.values.at("memory").at("host_kv_mib").get<int64_t>() + 128;
        assert(ss::static_resolution_key(input) == original_key &&
               "one static projection must ignore S8-derived budgets and per-start shm");
        input.effective_values["service"]["max_context"] = 4096;
        assert(ss::static_resolution_key(input) != original_key &&
               "operator-controlled context remains a static identity input");
    }
}

/* RSC12: a config resolves exactly as the layout it replaces (every effective value; only the provenance names
 * the config file), a variant changes only what it states, and the loader refuses what it does not know. */

// Exercise the shipped bc-serve executable against the actual native
// config-loader/resolver/detector with a fake sys-root. No GPU or planner
// stubs and no Python parsing or permission-dependent assertions.
struct CommandResult {
    int exit_code;
    std::string output;
};
std::string serve_test_executable = BC_SERVE_EXE;
CommandResult run_bc_serve(const std::vector<std::string>& arguments) {
    int fds[2];
    if (::pipe(fds) != 0)
        throw std::runtime_error("cli test: pipe failed");
    const pid_t pid = ::fork();
    if (pid == -1)
        throw std::runtime_error("cli test: fork failed");
    if (pid == 0) {
        ::close(fds[0]);
        if (::dup2(fds[1], STDOUT_FILENO) == -1 || ::dup2(fds[1], STDERR_FILENO) == -1)
            ::_exit(127);
        ::close(fds[1]);
        std::vector<char*> argv;
        const char* binary = serve_test_executable.c_str();
        argv.push_back(const_cast<char*>(binary));
        for (const auto& arg : arguments)
            argv.push_back(const_cast<char*>(arg.c_str()));
        argv.push_back(nullptr);
        ::execv(binary, argv.data());
        ::_exit(127);
    }
    ::close(fds[1]);
    std::string output;
    char buf[2048];
    for (;;) {
        const auto n = ::read(fds[0], buf, sizeof(buf));
        if (n == 0)
            break;
        if (n < 0) {
            if (errno == EINTR)
                continue;
            ::close(fds[0]);
            (void)::waitpid(pid, nullptr, 0);
            throw std::runtime_error("cli test: read failed");
        }
        output.append(buf, static_cast<size_t>(n));
    }
    ::close(fds[0]);
    int status = 0;
    if (::waitpid(pid, &status, 0) != pid)
        throw std::runtime_error("cli test: child wait failed");
    return {WIFEXITED(status) ? WEXITSTATUS(status) : -1, output};
}
void v3_writer_cli_rejects_unowned_imports(Fixture& t) {
    const auto no_snapshot = run_bc_serve(
        {"--write-manifest", "unused.json", "--measured-plan", "missing-plan.json", "--rank-demand",
         "missing-ranks.json"});
    if (no_snapshot.exit_code != 2 ||
        no_snapshot.output.find("v3 --write-manifest requires --bc-native-config-fd") == std::string::npos ||
        no_snapshot.output.find("cannot read missing-plan.json") != std::string::npos)
        throw std::runtime_error("v3 writer did not reject a missing sealed snapshot before plan-file access: " +
                                 no_snapshot.output);

    const auto raw_import = run_bc_serve(
        {"--write-manifest", "unused.json", "--bc-native-config-fd", "3", "--measured-plan",
         "missing-plan.json", "--rank-demand", "missing-ranks.json"});
    if (raw_import.exit_code != 2 ||
        raw_import.output.find("v3 --write-manifest does not import --measured-plan or --rank-demand") ==
            std::string::npos ||
        raw_import.output.find("cannot read missing-plan.json") != std::string::npos)
        throw std::runtime_error("v3 writer did not refuse raw measured-plan imports before file access: " +
                                 raw_import.output);

    // A native v3 measured writer must accept the frozen-key guard too. Use an
    // invalid descriptor so the CLI reaches sealed-snapshot loading without
    // opening cache files or starting any process.
    const char* layouts_dir = std::getenv("BIGCHERRY_LAYOUTS");
    if (!layouts_dir)
        throw std::runtime_error("v3 measured-key CLI test requires BIGCHERRY_LAYOUTS");
    const fs::path configs_dir = fs::path(layouts_dir).parent_path() / "configs";
    const fs::path model = t.root / "v3-measured-key-model.rad";
    bc::test::write_rad(model, 0, 0, 128, 128, {});
    const fs::path site = t.file("v3-measured-key-site.toml", "[engine]\nmodel = \"" + model.string() +
                                                                   "\"\nradiance_home = \"/runtime\"\n");
    const auto measured_key = run_bc_serve(
        {"--write-manifest", "unused.json", "--bc-native-config-fd", "99", "--expected-plan-key",
         std::string(64, '0'), "--plan-cache", "missing-cache", "--engine-binary", "/missing/radiance",
         "--power-profile", "stock", "--configs", configs_dir.string(), "--site", site.string(), "--model-root",
         (t.root / "model-root").string(), "--sys-root", t.root.string(), "--set",
         "layout.name=\"qwen27b-tp3-int8\""});
    if (measured_key.output.find("native snapshot descriptor must be a fully sealed memfd") == std::string::npos)
        throw std::runtime_error("v3 measured writer rejected the frozen-key guard before snapshot loading: " +
                                 measured_key.output);
}
void native_explain_cli(Fixture& t) {
    const char* env = std::getenv("BIGCHERRY_LAYOUTS");
    if (!env || !fs::is_directory(env))
        throw std::runtime_error("native_explain_cli: BIGCHERRY_LAYOUTS required");
    const fs::path configs = fs::path(env).parent_path() / "configs";
    const auto model = t.root / "explain-model.rad";
    bc::test::write_rad(model, 0, 0, 128, 128, {});
    const auto site = t.file("explain-site.toml",
                             "[engine]\nmodel = \"" + model.string() + "\"\nradiance_home = \"/runtime\"\n");
    auto command = [&](const std::string& name) {
        return std::vector<std::string>{"explain",
                                        name,
                                        "--json",
                                        "--configs",
                                        configs.string(),
                                        "--site",
                                        site.string(),
                                        "--model-root",
                                        (t.root / "model-root").string(),
                                        "--sys-root",
                                        t.root.string()};
    };
    const auto base = run_bc_serve(command("qwen27b-tp3-int8"));
    assert(base.exit_code == 0);
    const Json base_doc = Json::parse(base.output);
    assert(base_doc.at("config") == "qwen27b-tp3-int8");
    assert(base_doc.at("values").at("layout").at("name") == "qwen27b-tp3-int8");
    assert(base_doc.at("values").at("service").at("max_context") == 8192);
    assert(base_doc.at("provenance")
               .at("tuning.tp_ownership")
               .at("source")
               .get<std::string>()
               .find("config qwen27b.toml") != std::string::npos);
    const auto variant = run_bc_serve(command("qwen27b-tp3-int8-556"));
    assert(variant.exit_code == 0);
    const Json variant_doc = Json::parse(variant.output);
    assert(variant_doc.at("values").at("tuning").at("tp_ownership") !=
           base_doc.at("values").at("tuning").at("tp_ownership"));
    assert(variant_doc.at("provenance")
               .at("tuning.tp_ownership")
               .at("source")
               .get<std::string>()
               .find("config qwen27b.toml") != std::string::npos);
    const auto text_args = std::vector<std::string>{
        "explain",    "qwen27b-tp3-int8", "--configs",    configs.string(),
        "--site",     site.string(),      "--model-root", (t.root / "model-root").string(),
        "--sys-root", t.root.string()};
    const auto text_result = run_bc_serve(text_args);
    assert(text_result.exit_code == 0);
    assert(text_result.output.find("layout: qwen27b-tp3-int8") != std::string::npos);

    auto bad = command("absent-config");
    assert(run_bc_serve(bad).exit_code != 0);
    bad = command("qwen27b-tp3-int8");
    bad.push_back("--set");
    bad.push_back("layout.name=\"qwen27b-tp3-int8-556\"");
    const auto override = run_bc_serve(bad);
    assert(override.exit_code != 0);
    assert(override.output.find("cannot be replaced") != std::string::npos);
    bad = command("qwen27b-tp3-int8");
    bad.push_back("--unexpected-flag");
    assert(run_bc_serve(bad).exit_code != 0);
}

void config_files(Fixture& t) {
    namespace sv = bc::serving;
    const char* env = std::getenv("BIGCHERRY_LAYOUTS");
    if (!env || !fs::is_directory(env))
        throw std::runtime_error("config_files: BIGCHERRY_LAYOUTS must name the layouts directory");
    const fs::path configs = fs::path(env).parent_path() / "configs";
    const auto model = t.root / "config-model.rad";
    bc::test::write_rad(model, 4, 16, 128, 128, {});
    const auto site = t.file("config-site.toml",
                             "[engine]\nmodel = \"" + model.string() + "\"\nradiance_home = \"/runtime\"\n");
    // Native RSC12 dense config: host pool/KV and ngram placement are explicit
    // effective values, and a TP engine defaults to no peer-to-peer transfers.
    Options layout;
    layout.site = site;
    layout.sets = {"layout.name=\"qwen27b-tp3-int8\"",
                   "service.max_context=8192",
                   "service.max_sessions=8",
                   "engine.kv_cache_dtype=\"fp8\"",
                   "engine.num_speculative_tokens=3",
                   "memory.host_pool_mib=0",
                   "memory.host_kv_mib=0",
                   "engine.ngram_placement=\"disk\"",
                   "tuning.p2p=\"off\""};
    const auto from_layout = sv::resolve(layout, t.facts);
    Options config;
    config.site = site;
    config.configs = {configs};
    config.model_root = t.root / "model-root"; // the site's model wins over the config's (config < site)
    config.sets = {"layout.name=\"qwen27b-tp3-int8\""};
    const auto from_config = sv::resolve(config, t.facts);
    if (from_layout.values != from_config.values) {
        std::fprintf(stderr, "layout: %s\nconfig: %s\n", from_layout.values.dump().c_str(),
                     from_config.values.dump().c_str());
        assert(false);
    }
    assert(from_config.provenance.at("tuning.tp_ownership")
               .at("source")
               .get<std::string>()
               .find("config qwen27b.toml") != std::string::npos);
    assert(from_config.provenance.at("service.max_context")
               .at("source")
               .get<std::string>()
               .find("config qwen27b.toml") != std::string::npos);
    { // VTP42: the prefill governor is one optional string field that reaches the engine options
        Options gov = config;
        gov.sets = {"layout.name=\"qwen27b-tp3-int8\"", "tuning.prefill_governor=\"kind=budget;factor=1.5\""};
        assert(sv::resolve(gov, t.facts).values.at("tuning").at("prefill_governor") ==
               "kind=budget;factor=1.5");
        const auto& options = bc::serving::engine_option_fields();
        auto it = std::find_if(options.begin(), options.end(),
                               [](const auto& f) { return f.key == "tuning.prefill_governor"; });
        assert(it != options.end() && it->path == "/spec/processes/engine/tuning/prefill_governor");
        gov.sets.back() = "tuning.prefill_governor=\"\"";
        fail(gov, t.facts, "prefill_governor");
    }
    // the variant: only the split differs
    config.sets = {"layout.name=\"qwen27b-tp3-int8-556\""};
    const auto variant = sv::resolve(config, t.facts);
    Json a = from_config.values, b = variant.values;
    assert(b.at("tuning").at("tp_ownership") == "kv=1,1,2;ffn=5632,5632,6144;lk=5,5,6;lv=15,15,18");
    a["tuning"].erase("tp_ownership");
    b["tuning"].erase("tp_ownership");
    a["layout"].erase("name");
    b["layout"].erase("name");
    assert(a == b);
    { // where each value came from: the variant's own key, and the base's for the rest
        const auto set = sv::load_configs(sv::config_sources({configs}));
        const Json& o = set.origin.at("qwen27b-tp3-int8-556");
        assert(o.at("tuning.tp_ownership") == "qwen27b-tp3-int8-556");
        assert(o.at("tuning.pipelined_prefill") == "qwen27b-tp3-int8" && o.at("ranks") == "qwen27b-tp3-int8");
        assert(set.chain.at("qwen27b-tp3-int8-556") ==
               Json::array({"qwen27b-tp3-int8-556", "qwen27b-tp3-int8"}));
    }
    { // the long-context variants: the 5,5,6 split at 262,144 tokens, fp8 and bf16 cache, VTP39/VTP42 prefill
      // tuning; nothing else moves
        Options lv = config;
        lv.sets = {"layout.name=\"qwen27b-tp3-int8-556-long\""};
        const auto fp8 = sv::resolve(lv, t.facts);
        lv.sets = {"layout.name=\"qwen27b-tp3-int8-556-long-bf16\""};
        const auto bf16 = sv::resolve(lv, t.facts);
        assert(fp8.values.at("service").at("max_context") == 262144 &&
               fp8.values.at("engine").at("kv_cache_dtype") == "fp8");
        assert(bf16.values.at("service").at("max_context") == 262144 &&
               bf16.values.at("engine").at("kv_cache_dtype") == "bf16");
        // VTP39/VTP42 scheduling defaults for long prompts, inherited by -long-bf16
        for (const auto* r : {&fp8, &bf16}) {
            assert(r->values.at("tuning").at("prefill_interleave") == 2);
            assert(r->values.at("tuning").at("prefill_governor") ==
                   "kind=budget;factor=1.5;long_tokens=16384;order=remaining;age_ms=120000");
        }
        Json x = variant.values, y = fp8.values, z = bf16.values;
        for (Json* j : {&x, &y, &z}) {
            (*j)["tuning"].erase("prefill_interleave");
            (*j)["tuning"].erase("prefill_governor");
        }
        for (Json* j : {&x, &y, &z}) {
            (*j)["layout"].erase("name");
            (*j)["service"].erase("max_context");
            (*j)["engine"].erase("max_model_len");
            (*j)["engine"].erase("kv_cache_dtype");
        }
        if (!(x == y && y == z)) {
            std::fprintf(stderr, "long variants differ beyond context and cache: %s / %s\n",
                         Json::diff(x, y).dump().c_str(), Json::diff(y, z).dump().c_str());
            assert(false);
        }
    }
    { // ABI43 defaults absent Flash-Next interleave to auto:1; the measured v1.17 lane used zero.
        const auto flashnext_site = t.file(
            "flashnext-profile-site.toml",
            "[engine]\nmodel = \"" + model.string() + "\"\nradiance_home = \"/runtime\"\n[handoff]\nheat = \"" +
                (t.root / "writer.hot-first").string() + "\"\nhistogram = \"" +
                (t.root / "writer.histogram").string() + "\"\n");
        for (const char* profile : {"flashnext-4gpu", "flashnext-2xtx-rdna2-tail-combine"}) {
            Options flashnext = config;
            flashnext.site = flashnext_site;
            flashnext.sets = {std::string("layout.name=\"") + profile + "\""};
            const auto pinned = sv::resolve(flashnext, t.facts);
            assert(pinned.values.at("tuning").at("prefill_interleave") == 0);
        }
        const auto& options = bc::serving::engine_option_fields();
        const auto field = std::find_if(options.begin(), options.end(),
                                        [](const auto& option) {
                                            return option.key == "tuning.prefill_interleave";
                                        });
        assert(field != options.end() && field->path == "/spec/processes/engine/tuning/prefill_interleave" &&
               field->kind == "ref");
    }
    // a profile and --set still override a config (config < site < profile < --set)
    config.sets.push_back("tuning.pipelined_prefill=\"off\"");
    assert(sv::resolve(config, t.facts).values.at("tuning").at("pipelined_prefill") == "off");
    { // every cache format the engine offers is a setting, bf16 (16-bit K and V) included, and reaches the
      // manifest
        Options kv = config;
        kv.sets = {"layout.name=\"qwen27b-tp3-int8\"", "engine.kv_cache_dtype=\"bf16\""};
        const auto r = sv::resolve(kv, t.facts);
        assert(r.values.at("engine").at("kv_cache_dtype") == "bf16");
        sv::ManifestInputs in;
        in.resolved = r;
        in.instance_id = "11111111-1111-4111-8111-111111111111";
        in.producer_sha256 = std::string(64, 'a');
        in.profile = "production";
        in.s10.power_profile = "stock";
        const auto built = sv::build_manifest(in);
        assert(built.document.at("spec").at("processes").at("engine").at("execution").at("kv_cache_dtype") ==
               "bf16");
        assert(
            bc::serving::v2::read_manifest(built.text, "qwen27b-tp3-int8", bc::serving::v2::PlanUse::measuring)
                .ok());
        kv.sets.back() = "engine.kv_cache_dtype=\"int8\"";
        bool refused = false;
        try {
            (void)sv::resolve(kv, t.facts);
        } catch (const std::exception&) {
            refused = true;
        }
        assert(refused); // not an engine format
    }

    const std::string head =
        "schema = 1\n[config.base]\ndescription = \"b\"\nmodel = \"m.rad\"\n"
        "ranks = [{ arch = \"gfx1201\" }]\n[config.base.service]\nmax_context = 8192\nmax_sessions = 4\n";
    auto refuses = [&](const std::string& name, const std::string& text, const std::string& needle) {
        const auto f = t.file(name.c_str(), text);
        try {
            (void)sv::load_configs({f});
        } catch (const std::exception& e) {
            if (std::string(e.what()).find(needle) == std::string::npos) {
                std::fprintf(stderr, "config refusal %s: wanted '%s', got: %s\n", name.c_str(), needle.c_str(),
                             e.what());
                assert(false);
            }
            return;
        }
        std::fprintf(stderr, "config %s: not refused\n", name.c_str());
        assert(false);
    };
    refuses("c-unknown.toml", head + "[config.v]\nbase = \"base\"\ndescription = \"v\"\ncontext = 1\n",
            "unknown key context");
    refuses("c-flat-setting.toml", head + "[config.v]\nbase = \"base\"\ndescription = \"v\"\nmax_context = 1\n",
            "unknown key max_context");
    refuses("c-server-args.toml",
            head + "[config.v]\nbase = \"base\"\ndescription = \"v\"\nserver-args = [\"--max-model-len\"]\n",
            "unknown key server-args");
    refuses("c-nobase.toml", head + "[config.v]\nbase = \"gone\"\ndescription = \"v\"\n",
            "base gone is not a defined config");
    refuses(
        "c-cycle.toml",
        head + "[config.x]\nbase = \"y\"\ndescription = \"x\"\n[config.y]\nbase = \"x\"\ndescription = \"y\"\n",
        "a cycle");
    refuses("c-nodesc.toml", head + "[config.v]\nbase = \"base\"\n", "description is required");
    refuses("c-nomodel.toml", "schema = 1\n[config.v]\ndescription = \"v\"\nranks = [{ arch = \"gfx1201\" }]\n",
            "model is required");
    refuses("c-rankkey.toml",
            "schema = 1\n[config.v]\ndescription = \"v\"\nmodel = \"m\"\nranks = [{ arch = \"gfx1201\", "
            "devices = [0] }]\n",
            "ranks: unknown key devices");
    refuses("c-schema.toml", "[config.v]\ndescription = \"v\"\n", "schema = 1 is required");
    { // a name defined in two files
        const auto f1 = t.file("c-dup1.toml", head);
        const auto f2 = t.file("c-dup2.toml", head);
        bool refused = false;
        try {
            (void)sv::load_configs({f1, f2});
        } catch (const std::exception& e) {
            refused = std::string(e.what()).find("defined in both") != std::string::npos;
        }
        assert(refused);
    }
    { // tables merge key by key; a list replaces
        const auto f =
            t.file("c-merge.toml", head + "[config.v]\nbase = \"base\"\ndescription = \"v\"\nranks = [{ arch = "
                                          "\"gfx1100\", count = 2 }]\ntp_wire = \"exact\"\n"
                                          "[config.v.service]\nmax_sessions = 2\n");
        const auto set = sv::load_configs({f});
        const Json& v = set.config.at("v");
        assert(v.at("service").at("max_context") == 8192 && v.at("service").at("max_sessions") == 2);
        assert(v.at("ranks").size() == 1 && v.at("ranks")[0].at("arch") == "gfx1100");
        assert(set.origin.at("v").at("service.max_context") == "base" &&
               set.origin.at("v").at("service.max_sessions") == "v");
        assert(set.origin.at("v").at("ranks") == "v" &&
               set.origin.at("base").at("ranks") == "base"); // a replaced list
        const Json l = sv::config_as_layout(v);
        assert(l.at("engine") == v.at("ranks") && l.at("tp-wire") == "exact" && !l.contains("server-args"));
        assert(l.at("settings").at("service").at("max_sessions") == 2);
    }
    { // the config's model: under the model root, the lowest engine.model; a site's model wins; relative with
      // no root
        // is refused when that config is selected
        const auto root = t.root / "models";
        fs::create_directories(root / "m");
        bc::test::write_rad(root / "m" / "own.rad", 0, 0, 128, 128, {});
        const auto own = t.file("c-model.toml",
                                "schema = 1\n[config.own]\ndescription = \"o\"\nmodel = \"m/own.rad\"\n"
                                "ranks = [{ arch = \"gfx1201\" }]\n[config.own.service]\nmax_context = 8192\n"
                                "max_sessions = 4\n");
        Options o;
        o.site = t.file("c-nomodel-site.toml", "[engine]\nradiance_home = \"/runtime\"\n");
        o.configs = {own};
        o.model_root = root;
        o.sets = {"layout.name=\"own\""};
        const auto r = sv::resolve(o, t.facts);
        assert(r.values.at("engine").at("model") == (root / "m" / "own.rad").string());
        o.site = site;
        assert(sv::resolve(o, t.facts).values.at("engine").at("model") == model.string());
        o.model_root.clear();
        fail(o, t.facts, "is relative and no model root is given");
    }
    refuses("c-placement.toml",
            head + "[config.v]\nbase = \"base\"\ndescription = \"v\"\n"
                   "[[config.v.helpers]]\narch = \"gfx1100\"\ncount = 1\nplacement = \"sideways\"\n",
            "placement is required");
    refuses("c-helper-count.toml",
            head + "[config.v]\nbase = \"base\"\ndescription = \"v\"\n"
                   "[[config.v.helpers]]\narch = \"gfx1100\"\nplacement = \"head\"\n",
            "count is required");
    refuses("c-budget.toml",
            head +
                "[config.v]\nbase = \"base\"\ndescription = \"v\"\n"
                "[[config.v.helpers]]\narch = \"gfx1100\"\ncount = 1\nplacement = \"head\"\nbudget_mib = 1.5\n",
            "budget_mib");
    refuses("c-frac.toml",
            "schema = 1\n[config.v]\ndescription = \"v\"\nmodel = \"m\"\nranks = [{ arch = \"gfx1100\", count "
            "= 2.5 }]\n",
            "count is a whole number");
    refuses("c-bool.toml",
            "schema = 1\n[config.v]\ndescription = \"v\"\nmodel = \"m\"\nranks = [{ arch = \"gfx1100\", count "
            "= true }]\n",
            "count is a whole number");
    refuses("c-schema-float.toml", "schema = 1.0\n", "schema = 1 is required");
    refuses("c-date.toml", head + "[config.base.tuning]\nwhen = 2026-11-14\n", "unsupported TOML value");
    refuses("c-config-scalar.toml", "schema = 1\nconfig = 3\n", "config is a table");
    refuses("c-model-escape.toml",
            head + "[config.v]\nbase = \"base\"\ndescription = \"v\"\nmodel = \"../other.rad\"\n",
            "leaves the model root");
    refuses("c-engine-model.toml", head + "[config.base.engine]\nmodel = \"x\"\n", "give the model as `model`");
    { // dotted keys are tables: `tuning.x = ...` in the config table is the tuning table; three levels of
      // bases;
        // a scalar replaced by a table keeps no stale origin
        const auto f = t.file(
            "c-dotted.toml",
            head + "[config.v1]\nbase = \"base\"\ndescription = \"1\"\ntuning.pipelined_prefill = \"on\"\n"
                   "[config.v2]\nbase = \"v1\"\ndescription = \"2\"\n[config.v2.service]\nmax_sessions = 2\n"
                   "[config.v3]\nbase = \"v2\"\ndescription = \"3\"\ntuning.tp_ownership = \"kv=1\"\n");
        const auto set = sv::load_configs({f});
        const Json& v3 = set.config.at("v3");
        assert(v3.at("tuning").at("pipelined_prefill") == "on" && v3.at("tuning").at("tp_ownership") == "kv=1");
        assert(v3.at("service").at("max_sessions") == 2 && v3.at("service").at("max_context") == 8192);
        assert(set.origin.at("v3").at("tuning.pipelined_prefill") == "v1" &&
               set.origin.at("v3").at("service.max_sessions") == "v2");
        assert(set.chain.at("v3") == Json::array({"v3", "v2", "v1", "base"}));
        const auto g =
            t.file("c-replace.toml",
                   head + "[config.w]\nbase = \"base\"\ndescription = \"w\"\nmodel = { x = \"y\" }\n");
        try {
            (void)sv::load_configs({g});
            assert(false);
        } // a model that is a table is not a string
        catch (const std::exception& e) {
            assert(std::string(e.what()).find("model is a string") != std::string::npos);
        }
    }
    { // --configs wins over BIGCHERRY_CONFIGS
        const auto f = t.file("c-given.toml", head);
        const char* before = std::getenv("BIGCHERRY_CONFIGS");
        const std::optional<std::string> saved = before ? std::optional<std::string>(before) : std::nullopt;
        ::setenv("BIGCHERRY_CONFIGS", (t.root / "no-such-dir").c_str(), 1);
        assert(sv::config_sources({f}) == std::vector<fs::path>{f});
        bool refused = false;
        try {
            (void)sv::config_sources({});
        } catch (const std::exception&) {
            refused = true;
        }
        assert(refused);
        ::unsetenv("BIGCHERRY_CONFIGS");
        assert(sv::config_sources({}).empty());
        if (saved)
            ::setenv("BIGCHERRY_CONFIGS", saved->c_str(), 1);
    }
    { // the resolver checks the settings: an unknown tuning key is refused, naming the config file
        Options bad;
        bad.site = site;
        bad.configs = {t.file("c-badtuning.toml", head + "[config.base.tuning]\nno_such = 1\n")};
        bad.model_root = t.root;
        bad.sets = {"layout.name=\"base\""};
        fail(bad, t.facts, "unknown key tuning.no_such");
        // with --layouts given as well, a config may not take a layout's name
        bad.configs = {configs};
        bad.layouts = {fs::path(env)};
        bad.sets = {"layout.name=\"qwen27b-tp3-int8\""};
        fail(bad, t.facts, "has the name of a layout");
    }
}

#ifndef BC_FAKE_ENGINE
#error "BC_FAKE_ENGINE must name the fake engine binary (session_fake_engine.cpp)"
#endif
// RSC02 step 1: the session owner's engine-only plan step against the fixture's cards: the qwen27b config
// resolved, its engine's device visibility from the resolver's rank order, and the manifest rounds with
// bc-serve faked: a cache miss measures with the engine alone (--plan-memory against the unmeasured manifest),
// the measured plan is stored with the engine's reported start-up memory, a derived host pool goes round once
// more, and the plan-extra is the bash launcher's.
void session_engine_plan(Fixture& t) {
    namespace ss = bc::serving::session;
    const fs::path configs = fs::path(std::getenv("BIGCHERRY_LAYOUTS")).parent_path() / "configs";
    const auto model = t.root / "session-model.rad";
    bc::test::write_rad(model, 0, 0, 128, 128, {});
    const fs::path home = t.root / "session-build" / "radiance_home";
    fs::create_directories(home);
    const auto site = t.file("session-site.toml", "[engine]\nmodel = \"" + model.string() +
                                                      "\"\nradiance_home = \"" + home.string() + "\"\n");
    const fs::path run = t.root / "session-run";
    fs::create_directories(run);
    ss::EnginePlanInputs in;
    in.options.sys_root = t.root;
    in.options.site = site;
    in.options.configs = {configs};
    in.options.model_root = t.root / "model-root"; // the site's model wins; the root is still required
    in.options.sets = {"layout.name=\"qwen27b-tp3-int8\""};
    in.self = t.root / "bc-serve-test";
    fs::create_symlink(BC_SERVE_EXE, in.self);
    in.engine_binary = BC_FAKE_ENGINE;
    const Json session_facts = bc::serving::detect(t.root);
    const auto session_resolved = bc::serving::resolve(in.options, session_facts);
    SessionManifestPair session_manifests;
#ifndef BC_NATIVE_V3_SESSION_ONLY
    session_manifests = session_manifest_pair(session_resolved, session_facts);
#endif
    { // A private native admission request inspects and reconciles exact bytes, then stops before all v2-only side effects.
        ss::EnginePlanInputs gated = in;
        const fs::path native_root = t.root / "native-admission-root";
        fs::create_directories(native_root);
        const fs::path native_yaml = native_root / "fleet.yaml";
        const std::string native_yaml_bytes = "# caf\xc3\xa9\r\ncards: [0]\r\nmodels:\r\n  tiny: {model: " +
                                               Json(model.string()).dump() + "}\r\n";
        std::ofstream(native_yaml, std::ios::binary) << native_yaml_bytes;
        Json expected_kernels = Json::array();
        const std::string kernel_text = session_resolved.values.at("engine").at("kernels").get<std::string>();
        for (size_t begin = 0; begin < kernel_text.size();) {
            const size_t end = kernel_text.find(',', begin);
            expected_kernels.push_back(kernel_text.substr(begin, end == std::string::npos
                                                                    ? std::string::npos
                                                                    : end - begin));
            if (end == std::string::npos) break;
            begin = end + 1;
        }
        ss::Env native_overrides = {
            {"TEST_NATIVE_INSPECT_TP", "3"},
            {"TEST_NATIVE_INSPECT_RANK_DEVICES", session_resolved.values.at("layout").at("engine_cards").dump()},
            {"TEST_NATIVE_INSPECT_CARDS", "[0,1,2]"},
            {"TEST_NATIVE_INSPECT_MODEL", model.string()},
            {"TEST_NATIVE_INSPECT_EXPECT_YAML", native_yaml_bytes},
            {"TEST_NATIVE_INSPECT_PORT", std::to_string(session_resolved.values.at("engine").at("port").get<int>())},
            {"TEST_NATIVE_INSPECT_MAX_CONTEXT",
             std::to_string(session_resolved.values.at("service").at("max_context").get<int>())},
            {"TEST_NATIVE_INSPECT_MAX_SEQUENCES",
             std::to_string(session_resolved.values.at("service").at("max_sessions").get<int>())},
            {"TEST_NATIVE_INSPECT_KERNELS", expected_kernels.dump()}};
        gated.native_admission = ss::NativeAdmissionRequest{native_yaml, native_root, std::move(native_overrides)};
        gated.named_config = "qwen27b-tp3-int8";
        const fs::path gated_run = run / "native-admission";
        fs::create_directories(gated_run);
        const fs::path cache_tripwire = gated_run /
            ("stored-" + bc::serving::sha256_hex(gated.named_config) + ".index.json");
        std::ofstream(cache_tripwire) << "deliberately malformed cache index";
        int writer_calls = 0;
#ifndef BC_NATIVE_V3_SESSION_ONLY
        gated.runner = [&](const std::vector<std::string>&) {
            ++writer_calls;
            return ss::ManifestRun{};
        };
#endif
        bool power_admitted = false;
        std::vector<std::string> inspector_roles;
        std::vector<pid_t> reaped;
        ss::Hooks admission_hooks;
        admission_hooks.wait = [](std::chrono::milliseconds) { return true; };
        admission_hooks.started = [&](ss::ChildKind kind, const std::string& role, pid_t,
                                      const fs::path&) {
            assert(kind == ss::ChildKind::native_config_inspector || kind == ss::ChildKind::manifest_writer);
            inspector_roles.push_back(role);
        };
        admission_hooks.ended = [&](pid_t pid) {
            if (reaped.empty()) {
                std::error_code remove_error;
                fs::remove(native_yaml, remove_error);
                if (remove_error) throw std::runtime_error("cannot remove authored YAML after inspection");
            }
            reaped.push_back(pid);
        };
        admission_hooks.admit_power = [&](const fs::path&, bc::serving::v2::PlanUse) {
            power_admitted = true;
            return bc::serving::v2::Status{};
        };
        bool stopped_before_v2 = false;
        std::string native_gate_error;
        try {
            (void)ss::plan_engine_only(gated, gated_run, admission_hooks);
        } catch (const ss::PlanFailed& error) {
            native_gate_error = error.what();
#ifdef BC_NATIVE_V3_SESSION_ONLY
            stopped_before_v2 = std::string_view(error.what()).find("native v3 measuring draft verified") !=
                                std::string_view::npos;
#else
            stopped_before_v2 = std::string_view(error.what()).find("native writer/P1/measurement/replay") !=
                                std::string_view::npos;
#endif
        }
#ifdef BC_NATIVE_V3_SESSION_ONLY
        if (!(stopped_before_v2 && writer_calls == 0 && !power_admitted && inspector_roles.size() == 2 &&
              inspector_roles[0] == "native config inspector" &&
              inspector_roles[1] == "native v3 measuring draft writer" && reaped.size() == 2 &&
              !fs::exists(gated_run / "plan-cache") && !fs::exists(native_yaml) &&
              fs::exists(cache_tripwire)))
            throw std::runtime_error("native v3 session did not verify a real writer draft before its closed gate: " +
                                     native_gate_error);
#else
        if (!(stopped_before_v2 && writer_calls == 0 && !power_admitted && inspector_roles.size() == 1 &&
              inspector_roles[0] == "native config inspector" && reaped.size() == 1))
            throw std::runtime_error("native admission did not reach its expected fail-closed gate: " +
                                     native_gate_error);
#endif

        ss::EnginePlanInputs reserved_override = gated;
        reserved_override.native_admission->runtime_overrides["CUDA_VISIBLE_DEVICES"] = "0";
        int forbidden_inspector_calls = 0;
        reserved_override.runner = [&](const std::vector<std::string>&) {
            ++forbidden_inspector_calls;
            return ss::ManifestRun{};
        };
        bool selector_refused = false;
        try {
            (void)ss::plan_engine_only(reserved_override, gated_run, admission_hooks);
        } catch (const ss::PlanFailed& error) {
            selector_refused = std::string_view(error.what()).find("runtime override cannot select CUDA_VISIBLE_DEVICES") !=
                               std::string_view::npos;
        }
        assert(selector_refused && forbidden_inspector_calls == 0 && !power_admitted);
    }
#ifdef BC_NATIVE_V3_SESSION_ONLY
    return;
#endif
    std::vector<std::vector<std::string>> calls;
    auto has = [](const std::vector<std::string>& argv, const std::string& flag, const std::string& value) {
        for (size_t i = 0; i + 1 < argv.size(); ++i)
            if (argv[i] == flag && argv[i + 1] == value)
                return true;
        return false;
    };
    auto flag = [](const std::vector<std::string>& argv, const std::string& f) {
        return std::find(argv.begin(), argv.end(), f) != argv.end();
    };
    in.runner = [&](const std::vector<std::string>& argv) {
        calls.push_back(argv);
        ss::ManifestRun r;
        switch (calls.size()) {
        case 1:
            r.exit = 10;
            break; // no measured plan
        case 2:
            assert(flag(argv, "--unmeasured"));
            r = ss::run_to_completion(argv, ss::child_environment(environ, {}));
            break;
        case 3:
            r.exit = 11;
            r.out = "derived_host_pool_mib=2048\n";
            break; // stored; another pool wanted
        case 4:
            r.exit = 10;
            break; // the changed pool has a different measured-plan key
        case 5: {
            assert(flag(argv, "--unmeasured"));
            r = ss::run_to_completion(argv, ss::child_environment(environ, {}));
            break;
        }
        case 6:
            r.exit = 0;
            r.out = "manifest: x\ndigest: 0123abcd\n";
            std::ofstream(argv[2]) << session_manifests.serving.dump(2) << '\n';
            break;
        default:
            assert(!"unexpected bc-serve call");
        }
        return r;
    };
    std::vector<std::string> started;
    std::vector<pid_t> ended;
    ss::Hooks wait;
    wait.wait = [](std::chrono::milliseconds) { return true; };
    wait.started = [&](ss::ChildKind, const std::string& role, pid_t pid, const fs::path&) {
        started.push_back(role + " " + std::to_string(pid));
    };
    wait.ended = [&](pid_t pid) { ended.push_back(pid); };
    const ss::EngineStart s = ss::plan_engine_only(in, run, wait);
    assert(calls.size() == 6);
    // the measuring engine was the owner's to know about, and its exit was verified
    assert(started.size() == 2 && ended.size() == 2);
    for (size_t i = 0; i < started.size(); ++i)
        assert(started[i].rfind("measuring engine ", 0) == 0 &&
               started[i] == "measuring engine " + std::to_string(ended[i]));
    for (const auto& c : calls) {
        assert(c[0] == in.self.string() && c[1] == "--write-manifest" && has(c, "--power-profile", "stock"));
        assert(has(c, "--site", site.string()) && has(c, "--configs", (configs / "qwen27b.toml").string()) &&
               !flag(c, "--layouts") && has(c, "--set", "layout.name=\"qwen27b-tp3-int8\"") &&
               has(c, "--sys-root", t.root.string()));
    }
    for (size_t i : {0, 2, 3, 5})
        assert(has(calls[i], "--plan-extra", "budgets: default") &&
               has(calls[i], "--engine-binary", BC_FAKE_ENGINE) &&
               has(calls[i], "--plan-cache", (run / "plan-cache").string()));
    for (size_t i : {size_t(1), size_t(4)})
        assert(flag(calls[i], "--plan-extra") && has(calls[i], "--engine-binary", BC_FAKE_ENGINE) &&
               flag(calls[i], "--expected-plan-key") && !flag(calls[i], "--plan-cache"));
    // The owner publishes exact artifacts from the tracked attempt, then the writer reads cache-only and checks its attempt receipt.
    assert(flag(calls[2], "--expected-owner-attempt") && !flag(calls[2], "--measured-plan") &&
           !flag(calls[2], "--rank-demand") && flag(calls[5], "--expected-owner-attempt"));
    assert(!flag(calls[2], "memory.host_pool_mib=2048") && has(calls[3], "--set", "memory.host_pool_mib=2048"));
    {
        std::set<std::string> expected_attempts;
        for (size_t i : {size_t(2), size_t(5)})
            for (size_t j = 0; j + 1 < calls[i].size(); ++j)
                if (calls[i][j] == "--expected-owner-attempt")
                    expected_attempts.insert(calls[i][j + 1]);
        std::set<std::string> published_attempts;
        for (const auto& entry : fs::directory_iterator(run / "plan-cache")) {
            std::ifstream file(entry.path(), std::ios::binary);
            Json record;
            file >> record;
            assert(record.at("cache_version") == 2 && record.contains("owner"));
            published_attempts.insert(record.at("owner").at("attempt_id").get<std::string>());
        }
        assert(expected_attempts.size() == 2 && published_attempts == expected_attempts);
    }
    // the engine: from the measured manifest only, its three ranks on the resolver's cards in rank order
    assert(s.argv.size() == 5 && s.argv[0] == BC_FAKE_ENGINE && s.argv[1] == "--manifest" &&
           s.argv[2].rfind("/proc/self/fd/", 0) == 0 && s.argv[3] == "--role" && s.argv[4] == "engine" &&
           s.manifest == calls[3][2] && s.execution_manifest() == s.argv[2] && s.pinned_manifest &&
           s.stored_sha256 == bc::serving::sha256_hex(session_manifests.serving.dump(2) + "\n"));
    std::string rocr;
    const Json resolved = bc::serving::resolve(in.options, bc::serving::detect(t.root)).values;
    for (const Json& bdf : resolved.at("layout").at("engine_cards"))
        for (const Json& g : t.facts.at("gpus"))
            if (g.at("bdf") == bdf)
                rocr += (rocr.empty() ? "" : ",") + std::to_string(g.at("hip_ordinal").get<int>());
    assert(s.runtime.at("ROCR_VISIBLE_DEVICES") == rocr && s.runtime.at("HIP_VISIBLE_DEVICES") == "0,1,2");
    assert(s.runtime.at("RADIANCE_HOME") == home.string() && s.runtime.size() == 3);
    assert(s.port == 36080 && s.summary.at("config") == "qwen27b-tp3-int8" &&
           s.summary.at("manifest_digest") == "0123abcd");
    { // A profile changed only while the unmeasured child resolves must not slip behind the frozen cache key.
        ss::EnginePlanInputs drifting = in;
        const fs::path profile = t.file("transient-profile.toml", "[engine]\nnum_speculative_tokens = 2\n");
        drifting.options.profile = profile;
        const fs::path drift_run = run / "transient-drift";
        fs::create_directories(drift_run);
        int writer_calls = 0;
        drifting.runner = [&](const std::vector<std::string>& argv) {
            if (++writer_calls == 1)
                return ss::ManifestRun{10, "", "cache miss"};
            assert(writer_calls == 2 && flag(argv, "--expected-plan-key"));
            std::ofstream(profile, std::ios::trunc) << "[engine]\nnum_speculative_tokens = 1\n";
            return ss::run_to_completion(argv, ss::child_environment(environ, {}));
        };
        bool drift_refused = false;
        try {
            (void)ss::plan_engine_only(drifting, drift_run, {});
        } catch (const ss::PlanFailed& error) {
            drift_refused = std::string(error.what()).find(
                                "effective measuring-manifest inputs differ from the session's frozen plan key") !=
                            std::string::npos;
        }
        assert(drift_refused && writer_calls == 2 &&
               !fs::exists(drift_run / "measurement-attempts"));
    }
    { // Engine-only TP3 cache miss refuses a pool that fits one rank but not all ranks, before power or child hooks.
        ss::EnginePlanInputs limited = in;
        const fs::path limited_root = t.root / "tp3-low-room-root";
        fs::create_directories(limited_root);
        for (const char* name : {"proc", "sys", "dev"})
            fs::copy(t.root / name, limited_root / name, fs::copy_options::recursive);
        write(limited_root / "proc/meminfo", "MemTotal:       134217728 kB\n"
                                              "MemAvailable:     9216000 kB\n"
                                              "SwapTotal:        8388608 kB\n");
        limited.options.sys_root = limited_root;
        limited.options.sets.insert(limited.options.sets.end(), {
            "memory.host_pool_mib=2048", "memory.host_kv_mib=256"});
        const Json low_room_facts = bc::serving::detect(limited_root);
        const SessionManifestPair low_room_manifests = session_manifest_pair(
            bc::serving::resolve(limited.options, session_facts), session_facts);
        int writer_calls = 0;
        limited.runner = [&](const std::vector<std::string>& argv) {
            if (++writer_calls == 1)
                return ss::ManifestRun{10, "", "uncached"};
            assert(writer_calls == 2 && flag(argv, "--unmeasured"));
            std::ofstream(argv[2]) << low_room_manifests.measuring.dump(2) << '\n';
            return ss::ManifestRun{};
        };
        bool power_admitted = false;
        bool child_started = false;
        ss::Hooks preflight_hooks;
        preflight_hooks.wait = [](std::chrono::milliseconds) { return true; };
        preflight_hooks.admit_power = [&](const fs::path&, bc::serving::v2::PlanUse) {
            power_admitted = true;
            return bc::serving::v2::Status{};
        };
        preflight_hooks.started = [&](ss::ChildKind, const std::string&, pid_t, const fs::path&) {
            child_started = true;
        };
        bool refused_all_ranks = false;
        try {
            (void)ss::plan_engine_only(limited, run / "tp3-low-room", preflight_hooks);
        } catch (const ss::PlanFailed& error) {
            refused_all_ranks = std::string(error.what()).find(
                "host RAM admission refused for host pool across 3 engine ranks") != std::string::npos;
        }
        assert(refused_all_ranks && writer_calls == 2 && !power_admitted && !child_started);
    }
    { // A later plan miss must recheck the effective pool from its sealed manifest, not the original resolver values.
        ss::EnginePlanInputs revised = in;
        const fs::path revised_root = t.root / "tp3-revised-room-root";
        fs::create_directories(revised_root);
        for (const char* name : {"proc", "sys", "dev"})
            fs::copy(t.root / name, revised_root / name, fs::copy_options::recursive);
        revised.options.sys_root = revised_root;
        revised.options.sets = {"layout.name=\"qwen27b-tp3-int8\"", "memory.host_pool_mib=0",
                                "memory.host_kv_mib=0"};
        const auto original_resolved = bc::serving::resolve(revised.options, session_facts);
        assert(original_resolved.values.at("memory").at("host_pool_mib").get<int64_t>() == 0);
        auto effective_options = revised.options;
        effective_options.sets.push_back("memory.host_pool_mib=2048");
        const auto effective_resolved = bc::serving::resolve(effective_options, session_facts);
        const auto original_manifests = session_manifest_pair(original_resolved, session_facts);
        const auto effective_manifests = session_manifest_pair(effective_resolved, session_facts);
        const uint64_t kv_mib = effective_resolved.values.at("memory").at("host_kv_mib").get<uint64_t>();
        const uint64_t reserve_mib = effective_resolved.values.at("memory").at("host_reserve_mib").get<uint64_t>();
        const uint64_t room_mib = reserve_mib + std::max<uint64_t>(kv_mib, 4096) + 1;
        assert(room_mib < reserve_mib + kv_mib + 3 * 2048);
        const uint64_t total_kib = session_facts.at("host").at("mem_total_mib").get<uint64_t>() * 1024;
        write(revised_root / "proc/meminfo", std::string("MemTotal:       ") + std::to_string(total_kib) + " kB\n" +
                                                "MemAvailable:     " + std::to_string(room_mib * 1024) + " kB\n" +
                                                "SwapTotal:        8388608 kB\n");
        int writer_calls = 0;
        revised.runner = [&](const std::vector<std::string>& argv) {
            ss::ManifestRun result;
            switch (++writer_calls) {
            case 1:
                result.exit = 10;
                break;
            case 2:
                assert(flag(argv, "--unmeasured"));
                assert(original_manifests.measuring.at("spec").at("resources").at("host").at("pool_mib") == 0);
                std::ofstream(argv[2]) << original_manifests.measuring.dump(2) << '\n';
                break;
            case 3:
                result.exit = 11;
                result.out = "derived_host_pool_mib=2048\n";
                break;
            case 4:
                result.exit = 10;
                break; // the increased pool has no measured plan yet
            case 5:
                assert(flag(argv, "--unmeasured") && has(argv, "--set", "memory.host_pool_mib=2048"));
                assert(effective_manifests.measuring.at("spec").at("resources").at("host").at("pool_mib") ==
                       2048);
                std::ofstream(argv[2]) << effective_manifests.measuring.dump(2) << '\n';
                break;
            default:
                assert(!"unexpected writer call after the effective-pool refusal");
            }
            return result;
        };
        int power_admissions = 0;
        std::vector<std::string> revised_started;
        std::vector<pid_t> revised_ended;
        ss::Hooks revised_hooks;
        revised_hooks.wait = [](std::chrono::milliseconds) { return true; };
        revised_hooks.admit_power = [&](const fs::path&, bc::serving::v2::PlanUse) {
            ++power_admissions;
            return bc::serving::v2::Status{};
        };
        revised_hooks.started = [&](ss::ChildKind, const std::string& role, pid_t pid, const fs::path&) {
            revised_started.push_back(role + " " + std::to_string(pid));
        };
        revised_hooks.ended = [&](pid_t pid) { revised_ended.push_back(pid); };
        bool refused_effective_pool = false;
        try {
            (void)ss::plan_engine_only(revised, run / "tp3-revised-room", revised_hooks);
        } catch (const ss::PlanFailed& error) {
            refused_effective_pool = std::string(error.what()).find(
                "host RAM admission refused for host pool across 3 engine ranks") != std::string::npos;
        }
        assert(refused_effective_pool && writer_calls == 5 && power_admissions == 1 &&
               revised_started.size() == 1 && revised_started.front().rfind("measuring engine ", 0) == 0 &&
               revised_ended.size() == 1);
    }
    { // configs from BIGCHERRY_CONFIGS alone reach the manifest writer, whose environment has no BIGCHERRY_*:
      // as files
        ss::EnginePlanInputs env_only = in;
        env_only.options.configs.clear();
        const char* saved = std::getenv("BIGCHERRY_CONFIGS");
        const std::optional<std::string> keep = saved ? std::optional<std::string>(saved) : std::nullopt;
        ::setenv("BIGCHERRY_CONFIGS", configs.c_str(), 1);
        calls.clear();
        ended.clear();
        auto old = in.runner;
        env_only.runner = [&](const std::vector<std::string>& argv) {
            calls.push_back(argv);
            ss::ManifestRun r;
            std::ofstream(argv[2]) << session_manifests.serving.dump(2) << '\n';
            return r;
        };
        ss::plan_engine_only(env_only, run, wait);
        if (keep)
            ::setenv("BIGCHERRY_CONFIGS", keep->c_str(), 1);
        else
            ::unsetenv("BIGCHERRY_CONFIGS");
        assert(calls.size() == 1 && has(calls[0], "--configs", (configs / "qwen27b.toml").string()) &&
               !flag(calls[0], "--layouts"));
    }
    { // A calibration run files its record under the key of the manifest it served with: the one production looks
      // up. The host pool is re-measured between the two writes (1033 -> 2247 MiB on flashnext-4gpu), so a key that
      // depended on it, or on the engine args, could never match; the written manifest's own scope is the key.
        ss::EnginePlanInputs cal = in;
        cal.calibration_run = true;
        cal.live_policy_file = t.file("session-cal-policy.json", "{\"accept_unwatermarked_heat\": true}\n");
        cal.calibration_prompts_a = t.file("session-cal-a.txt", "prompt a\n");
        cal.calibration_prompts_b = t.file("session-cal-b.txt", "prompt b\n");
        cal.calibration_store = t.file("session-cal-store.toml", "");
        std::string keys[2];
        int i = 0;
        for (const int pool : {1033, 2247}) {
            const Json written = g_session_manifest_template;
            Json calibration_scope = written.at("spec");
            calibration_scope["resources"]["host"]["pool_mib"] = pool;
            cal.runner = [&](const std::vector<std::string>& argv) {
                ss::ManifestRun r;
                std::ofstream(argv[2]) << written.dump();
                return r;
            };
            const ss::EngineStart cs = ss::plan_engine_only(cal, run, wait);
            keys[i++] = cs.summary.at("calibration_store_key").get<std::string>();
            assert(keys[i - 1] ==
                   sv::policy_calibration_key(sv::policy_calibration_inputs(calibration_scope)));
        }
        assert(keys[0] == keys[1]); // a re-measured host pool does not move the record
    }
    // a failed manifest write and a layout (server-args are the bash launcher's) are refused before any engine
    // starts
    calls.clear();
    in.runner = [&](const std::vector<std::string>& argv) {
        calls.push_back(argv);
        return ss::ManifestRun{2, "", "bc-serve: no such thing"};
    };
    bool refused = false;
    try {
        ss::plan_engine_only(in, run, wait);
    } catch (const ss::PlanFailed& e) {
        refused = std::string(e.what()).find("exit 2") != std::string::npos;
    }
    assert(refused && calls.size() == 1);
    // An injected manifest runner must never leak any thrown type through the
    // v2 boundary. Failures at this stage cannot launch measuring children.
    for (int kind = 0; kind < 3; ++kind) {
        const size_t original_starts = started.size();
        in.runner = [kind](const std::vector<std::string>&) -> ss::ManifestRun {
            if (kind == 0)
                throw std::runtime_error("injected runner");
            if (kind == 1)
                throw 29; // NOLINT(bugprone-std-exception-baseclass)
            struct UnusualRunnerException {};
            throw UnusualRunnerException{}; // NOLINT(bugprone-std-exception-baseclass)
        };
        bool typed_refusal = false;
        try {
            (void)ss::plan_engine_only(in, run, wait);
        } catch (const ss::PlanFailed& e) {
            typed_refusal = std::string(e.what()).find("callback threw") != std::string::npos;
        }
        assert(typed_refusal && started.size() == original_starts);
    }
    ss::EnginePlanInputs layout = in;
    layout.options.configs.clear();
    layout.options.layouts = {fs::path(std::getenv("BIGCHERRY_LAYOUTS"))};
    layout.options.sets = {"layout.name=\"qwen27b-tp3-int8\""};
    refused = false;
    calls.clear();
    try {
        ss::plan_engine_only(layout, run, wait);
    } catch (const ss::PlanFailed& e) {
        refused = std::string(e.what()).find("is a layout") != std::string::npos;
    }
    assert(refused && calls.empty());
    { // bc-serve inside a build serves only that build's engine and runtime home
        const fs::path build = t.root / "session-build";
        fs::create_directories(build / "bin");
        std::ofstream(build / "bin" / "radiance") << "#!/bin/sh\n";
        fs::permissions(build / "bin" / "radiance", fs::perms::owner_all);
        ss::EnginePlanInputs own = in;
        own.self = home / "bin" / "bc-serve";
        calls.clear();
        refused = false;
        try {
            ss::plan_engine_only(own, run, wait);
        } catch (const ss::PlanFailed& e) {
            refused = std::string(e.what()).find("is not this bc-serve's build") != std::string::npos;
        }
        assert(refused && calls.empty());
    }
    { // Calibration parity: invalid prompt batches must be refused before the writer or any GPU child.
        ss::EnginePlanInputs bad_calibration = in;
        bad_calibration.named_config = "qwen27b-tp3-int8";
        bad_calibration.self = BC_SERVE_EXE;
        bad_calibration.calibration_run = true;
        bad_calibration.live_policy_file = t.file("session-calibration-policy.json", "{}");
        bad_calibration.calibration_prompts_a = t.file("session-calibration-a.txt", "same prompt\n");
        bad_calibration.calibration_prompts_b = t.file("session-calibration-b.txt", "same prompt\n");
        bad_calibration.calibration_store = t.file("session-calibration-store.toml", "");
        bool wrote = false;
        bad_calibration.runner = [&](const std::vector<std::string>&) {
            wrote = true;
            return ss::ManifestRun{2, "", "invalid calibration must never reach writer"};
        };
        bool refused_prompts = false;
        try {
            (void)ss::plan_engine_only(bad_calibration, run, wait);
        } catch (const ss::PlanFailed& e) {
            refused_prompts = std::string(e.what()).find("distinct") != std::string::npos;
        }
        assert(refused_prompts && !wrote);
        // RHP18 parity: invalid live-policy source is rejected before any writer
        // subprocess or measuring/helper child, not as a late manifest error.
        ss::EnginePlanInputs malformed = bad_calibration;
        malformed.calibration_prompts_b = t.file("session-calibration-distinct-b.txt", "distinct prompt\n");
        malformed.live_policy_file = t.file("session-calibration-malformed-policy.json", "{not valid JSON");
        bool called_writer = false;
        malformed.runner = [&](const std::vector<std::string>&) {
            called_writer = true;
            return ss::ManifestRun{2, "", "unexpected writer for malformed live policy"};
        };
        bool refused_policy = false;
        try {
            (void)ss::plan_engine_only(malformed, run, wait);
        } catch (const ss::PlanFailed& e) {
            refused_policy = std::string(e.what()).find("live policy") != std::string::npos;
        }
        if (!refused_policy || called_writer)
            throw std::runtime_error("calibration live-policy preflight failed before writer admission");

        // The positive calibration path crosses the real resolver, writer and
        // foreground owner. The OS process boundary blocks GPU measuring.
        ss::EnginePlanInputs valid = malformed;
        valid.named_config = "flashnext-2xtx";
        valid.helper_binary = BC_FAKE_ENGINE;
        valid.options.sets = {"layout.name=\"flashnext-2xtx\""};
        // Keep the card-free fixture valid even on a one-vCPU CI runner.
        valid.options.sets.push_back("helper.combine_threads=1");
        valid.options.sets.push_back("handoff.heat=\"" + (t.root / "parity.hot-first").string() + "\"");
        valid.options.sets.push_back("handoff.histogram=\"" +
                                     (t.root / "session-calibration-histogram").string() + "\"");
        valid.options.site = t.file("session-calibration-helper-site.toml",
                                    "[engine]\nmodel = \"" + (t.root / "model.rad").string() +
                                        "\"\nradiance_home = \"" + home.string() + "\"\n");
        valid.live_policy_file =
            t.file("session-calibration-valid-policy.json",
                   R"json({"effect_gate":true,"accept_unwatermarked_heat":false,"cadence_ns":60000000000,
            "max_moves":8,"max_moved_bytes":4294967296,"min_residency_ns":300000000000,
            "max_heat_age_ns":120000000000,"min_samples":1000,"abort_cooldown_ns":30000000000,
            "learned_eligibility_ttl_ns":600000000000,"effect_window_samples":5000,
            "effect_min_quiet_windows":3,"effect_revert_ttl_ns":120000000000,
            "margin_ppm":100000,"half_life_observations_milli":4000,"effect_noise_multiplier_milli":2000,
            "protected_layers":[0,1]})json");
        // RED: a strict observation policy cannot consume raw file heat.
        bool strict_writer_called = false;
        ss::EnginePlanInputs strict_heat = valid;
        strict_heat.runner = [&](const std::vector<std::string>&) {
            strict_writer_called = true;
            return ss::ManifestRun{2, "", "unexpected writer for strict file heat"};
        };
        bool refused_strict_heat = false;
        try {
            (void)ss::plan_engine_only(strict_heat, run, wait);
        } catch (const ss::PlanFailed& e) {
            const std::string why = e.what();
            refused_strict_heat = why.find("accept_unwatermarked_heat") != std::string::npos &&
                                  why.find("histogram") != std::string::npos;
        }
        if (!refused_strict_heat || strict_writer_called)
            throw std::runtime_error("calibration accepted non-watermarked histogram with a strict policy");

        std::ifstream policy_input(valid.live_policy_file);
        std::string opted_in_policy{std::istreambuf_iterator<char>(policy_input),
                                    std::istreambuf_iterator<char>()};
        const std::string strict_setting = "\"accept_unwatermarked_heat\":false";
        const auto setting_pos = opted_in_policy.find(strict_setting);
        if (setting_pos == std::string::npos)
            throw std::runtime_error("calibration policy fixture lost strict file heat setting");
        opted_in_policy.replace(setting_pos, strict_setting.size(), "\"accept_unwatermarked_heat\":true");
        valid.live_policy_file = t.file("session-calibration-opted-in-policy.json", opted_in_policy);

        bool missing_store_writer = false;
        ss::EnginePlanInputs missing_store = valid;
        missing_store.calibration_store = t.root / "new-calibration-store.toml";
        missing_store.runner = [&](const std::vector<std::string>&) {
            missing_store_writer = true;
            return ss::ManifestRun{2, "", "unexpected writer for missing store"};
        };
        bool refused_store = false;
        try {
            (void)ss::plan_engine_only(missing_store, run, wait);
        } catch (const ss::PlanFailed& e) {
            const std::string why = e.what();
            refused_store = why.find("calibration store") != std::string::npos &&
                            why.find("empty") != std::string::npos &&
                            why.find(missing_store.calibration_store.string()) != std::string::npos;
        }
        if (!refused_store || missing_store_writer)
            throw std::runtime_error("calibration store preflight did not explain missing store");
        const fs::path calibration_run = t.root / "session-calibration-real-writer-owner";
        bool submitted_to_writer = false;
        fs::path written_calibration;
        valid.runner = [&](const std::vector<std::string>& argv) {
            submitted_to_writer =
                submitted_to_writer ||
                (flag(argv, "--calibration-run") &&
                 has(argv, "--live-policy", valid.live_policy_file.string()) && !flag(argv, "--experiment"));
            if (!flag(argv, "--unmeasured"))
                return ss::ManifestRun{10, "", "uncached measured plan"};
            const auto actual = run_bc_serve({argv.begin() + 1, argv.end()});
            if (actual.exit_code != 0)
                throw std::runtime_error("calibration strict writer: " + actual.output);
            written_calibration = argv.at(2);
            return ss::ManifestRun{2, "", "card-free: refuse before the measuring GPU child"};
        };
        ss::Session calibration_owner;
        calibration_owner.run_dir = calibration_run;
        calibration_owner.plan = [valid](const fs::path& rd, const ss::Hooks& hooks) {
            return ss::plan_engine_only(valid, rd, hooks);
        };
        if (ss::serve(calibration_owner) == 0 || !submitted_to_writer || written_calibration.empty() ||
            !fs::is_regular_file(written_calibration))
            throw std::runtime_error("calibration owner did not execute the real native writer");
        const auto produced =
            bc::serving::v2::load_manifest(written_calibration, bc::serving::v2::PlanUse::measuring);
        if (!produced || produced.value().spec.intent.service.profile != bc::serving::v2::Profile::kLab ||
            !produced.value().spec.handoff || produced.value().spec.handoff->placement.live_policy ||
            produced.value().spec.handoff->placement.live_policy_state.reason_code != "measuring_start" ||
            produced.value().provenance.contains("/spec/handoff/placement/live_policy") ||
            produced.value().spec.experiment)
            throw std::runtime_error("calibration owner wrote an invalid unmeasured lab manifest");
    }
    // the measuring start that fails is refused, naming its log
    in.runner = [&](const std::vector<std::string>& argv) {
        ss::ManifestRun r;
        if (std::find(argv.begin(), argv.end(), "--unmeasured") != argv.end()) {
            std::ofstream(argv[2]) << session_manifests.measuring.dump(2) << '\n';
            return r;
        }
        r.exit = 10;
        return r;
    };
    in.engine_binary = "/bin/false";
    refused = false;
    try {
        ss::plan_engine_only(in, run, wait);
    } catch (const ss::PlanFailed& e) {
        refused = std::string(e.what()).find("measuring start failed") != std::string::npos;
    }
    assert(refused);
    { // Real named resolver and process owner, fake measuring engine OS boundary.
        ss::EnginePlanInputs ram = in;
        ram.engine_binary = t.root / "ram-measurement-error.sh";
        write(ram.engine_binary, "#!/bin/sh\nprintf '%s\\n' 'E --ngram-placement ram: the n-gram table (47.68 GiB) cannot be held as a pinned copy: 82.00 GiB of memory is available, 64.60 GiB more is still to be allocated, and this wants 47.69 GiB with 4.59 GiB to spare' >&2\nexit 1\n");
        fs::permissions(ram.engine_binary, fs::perms::owner_exec | fs::perms::owner_read | fs::perms::owner_write);
        ram.runner = [&](const std::vector<std::string>& argv) {
            if (flag(argv, "--unmeasured")) {
                std::ofstream(argv[2]) << session_manifests.measuring.dump(2) << '\n';
                return ss::ManifestRun{};
            }
            return ss::ManifestRun{10, "", "cache miss"};
        };
        bool host_ram_refused = false;
        try {
            (void)ss::plan_engine_only(ram, run, wait);
        } catch (const ss::PlanFailed& e) {
            const std::string detail = e.what();
            host_ram_refused = detail.find("BC_CFG_ADMISSION_REFUSED") != std::string::npos &&
                               detail.find("host RAM") != std::string::npos &&
                               detail.find("47.69 GiB") != std::string::npos &&
                               detail.find("82.00 GiB") != std::string::npos;
        }
        assert(host_ram_refused);
    }
    { // RSC04 lab experiments use the real named resolver and writer call graph.
        ss::EnginePlanInputs lab = in;
        lab.named_config = "qwen27b-tp3-int8";
        lab.experiment_file =
            t.file("lab-geometry.json", R"({"placement":{"mode":"observe","epoch_cycle_s":30}})");
        lab.engine_binary = BC_FAKE_ENGINE;
        const fs::path lab_run = t.root / "lab-session-run";
        fs::create_directories(lab_run);
        bool writer_checked = false;
        fs::path produced;
        lab.runner = [&](const std::vector<std::string>& argv) {
            writer_checked =
                has(argv, "--experiment", lab.experiment_file.string()) && !flag(argv, "--calibration-run");
            if (!flag(argv, "--unmeasured"))
                return ss::ManifestRun{10, "", "no cached measured plan"};
            const auto written = run_bc_serve({argv.begin() + 1, argv.end()});
            if (written.exit_code != 0)
                throw std::runtime_error("lab strict writer: " + written.output);
            produced = argv.at(2);
            return ss::ManifestRun{2, "", "deliberately stop before the measuring GPU child"};
        };
        try {
            (void)ss::plan_engine_only(lab, lab_run, wait);
        } catch (const ss::PlanFailed& e) {
            std::cerr << "lab graph: " << e.what() << "\n";
        }
        if (!writer_checked || produced.empty() || !fs::is_regular_file(produced))
            throw std::runtime_error("lab graph: writer_argv=" + std::to_string(writer_checked) +
                                     " produced=" + produced.string());
        // Real named resolver, real C++ writer, strict-v2 producer/reader bytes.
        const auto native = bc::serving::v2::load_manifest(produced, bc::serving::v2::PlanUse::measuring);
        assert(native && native.value().spec.experiment &&
               native.value().spec.intent.service.profile == bc::serving::v2::Profile::kLab);
        const auto detached = run_bc_serve({"start", "qwen27b-tp3-int8", "--run-dir", run.string(),
                                            "--lab-experiment", lab.experiment_file.string()});
        assert(detached.exit_code != 0 && detached.output.find("requires foreground") != std::string::npos);
        // An unsupported lab shape changes measurements: refuse before the writer/child.
        ss::EnginePlanInputs shaped = lab;
        shaped.experiment_file = t.file("lab-shape-rejected.json", R"({"shape":{}})");
        bool child_writer_called = false;
        shaped.runner = [&](const std::vector<std::string>&) {
            child_writer_called = true;
            return ss::ManifestRun{2, "", "unsupported lab experiment"};
        };
        bool refused_shape = false;
        try {
            (void)ss::plan_engine_only(shaped, lab_run / "shape-rejected", wait);
        } catch (const ss::PlanFailed& e) {
            refused_shape = std::string(e.what()).find("observe-only") != std::string::npos;
        }
        assert(refused_shape && !child_writer_called);
    }
}

// Real plan_engine_only + real session owner, with the fake KFD/proc tree,
// a child-gated CPU manifest writer and a fake engine. Both manifest cache
// branches must preserve writer -> durable power admission -> first GPU child.
// The injected writer response and power transaction never touch hardware.
void session_owner_real_planner_admission(Fixture& t) {
    namespace ss = bc::serving::session;
    namespace v2 = bc::serving::v2;
    const fs::path config_dir = fs::path(std::getenv("BIGCHERRY_LAYOUTS")).parent_path() / "configs";
    const fs::path model = t.root / "owner-plan-model.rad";
    bc::test::write_rad(model, 0, 0, 128, 128, {});
    const fs::path home = t.root / "owner-plan-home" / "radiance_home";
    fs::create_directories(home);
    auto select_free_port = [] {
        const int sock = ::socket(AF_INET, SOCK_STREAM, 0);
        assert(sock >= 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        assert(::bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0);
        socklen_t len = sizeof addr;
        assert(::getsockname(sock, reinterpret_cast<sockaddr*>(&addr), &len) == 0);
        ::close(sock);
        return int(ntohs(addr.sin_port));
    };

    for (const std::string mode : {"miss", "hit", "partial"}) {
        const fs::path run = t.root / ("owner-plan-" + mode);
        fs::create_directories(run);
        const int port = select_free_port();
        const fs::path site = t.file(("owner-plan-" + mode + ".toml").c_str(),
                                     "[engine]\nmodel = \"" + model.string() + "\"\nradiance_home = \"" +
                                         home.string() + "\"\nport = " + std::to_string(port) + "\n");
        ss::EnginePlanInputs in;
        in.options.sys_root = t.root;
        in.options.site = site;
        in.options.configs = {config_dir};
        in.options.model_root = t.root / "model-root";
        in.options.sets = {"layout.name=\"qwen27b-tp3-int8\""};
        in.named_config = "qwen27b-tp3-int8";
        in.self = BC_SERVE_EXE;
        in.engine_binary = BC_FAKE_ENGINE;
        const Json facts = bc::serving::detect(t.root);
        const SessionManifestPair manifests = session_manifest_pair(bc::serving::resolve(in.options, facts), facts);

        const fs::path trace = run / "power.trace";
        ss::Session session;
        session.run_dir = run;
        session.timing.health_timeout = std::chrono::seconds(6);
        session.timing.health_interval = std::chrono::milliseconds(40);
        session.timing.canary_timeout = std::chrono::seconds(5);
        session.timing.stop_grace = std::chrono::seconds(1);
        session.timing.poll = std::chrono::milliseconds(20);
        session.plan = [in, mode, manifests](const fs::path& dir, const ss::Hooks& hooks) -> ss::EngineStart {
            ss::EnginePlanInputs actual = in;
            unsigned call = 0;
            actual.runner = [mode, manifests, &hooks,
                             &call](const std::vector<std::string>& argv) mutable -> ss::ManifestRun {
                // One real fork + start-gate + registration + verified reap for
                // *each* writer attempt. Only its output is synthetic.
                const auto child = ss::run_to_completion({"/bin/true"}, {}, hooks);
                assert(child.exit == 0);
                ss::ManifestRun result;
                const unsigned turn = ++call;
                if (mode != "hit" && turn == 1) {
                    result.exit = 10; // planner cache miss
                } else {
                    const auto output = fs::path(argv.at(2));
                    const bool unmeasured = std::find(argv.begin(), argv.end(), "--unmeasured") != argv.end();
                    std::ofstream(output) << (unmeasured ? manifests.measuring : manifests.serving).dump(2) << '\n';
                    if (!unmeasured) {
                        for (size_t i = 0; i + 1 < argv.size(); ++i) {
                            if (argv[i] == "--s8-replay-evidence") {
                                std::ofstream(argv[i + 1]) << "synthetic named-cache S8 evidence\n";
                                break;
                            }
                        }
                    }
                }
                return result;
            };
            return ss::plan_engine_only(actual, dir, hooks);
        };
        session.power_hook.begin_manifest = [mode, trace](const fs::path&, v2::PlanUse use) -> v2::Status {
            // Partial begin models a hardware write after durable journalling,
            // followed by a returned error. The owner must still finish/reap.
            std::ofstream(trace, std::ios::app) << "begin\n";
            if (mode == "partial")
                return v2::make_error(v2::ConfigErrorCode::admission_refused, "test.power",
                                      "partial admission refused");
            assert(use == (mode == "hit" ? v2::PlanUse::serving : v2::PlanUse::measuring));
            return std::nullopt;
        };
        session.power_hook.verify_serving = [](const fs::path&) -> v2::Status { return std::nullopt; };
        session.power_hook.verified_exit = [trace] {
            std::ofstream(trace, std::ios::app) << "verified_exit\n";
        };
        session.power_hook.restore = [trace]() -> v2::Status {
            std::ofstream(trace, std::ios::app) << "restore\n";
            return std::nullopt;
        };

        std::cout.flush();
        const pid_t owner = ::fork();
        assert(owner >= 0);
        if (owner == 0) {
            try {
                std::_Exit(ss::serve(session));
            } catch (const std::exception& e) {
                std::cerr << "real planner owner failed: " << e.what() << '\n';
                std::_Exit(98);
            }
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(18);
        Json terminal;
        bool observed_running = false;
        for (;;) {
            terminal = ss::read_record(run).value_or(Json::object());
            const auto state = terminal.value("state", std::string());
            if (mode != "partial" && state == "running") {
                observed_running = true;
                break;
            }
            if (state == "failed" || state == "stopped" || std::chrono::steady_clock::now() > deadline)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
        if (observed_running) {
            const auto reply = ss::control(run, "stop", std::chrono::seconds(5));
            assert(reply && reply->value("ok", false));
        }
        int status = 0;
        bool reaped = false;
        for (int n = 0; n < 200; ++n) {
            if (::waitpid(owner, &status, WNOHANG) == owner) {
                reaped = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
        if (!reaped) {
            ::kill(owner, SIGTERM);
            ::waitpid(owner, &status, 0);
        }
        assert(reaped && WIFEXITED(status));
        assert((WEXITSTATUS(status) == 0) == (mode != "partial"));
        assert(observed_running == (mode != "partial"));
        const auto completed = ss::read_record(run);
        if (!completed)
            throw std::runtime_error("real planner owner failed to persist its terminal record");
        terminal = *completed;
        assert(terminal.value("state", std::string()) == (mode == "partial" ? "failed" : "stopped"));

        const auto cache_index = ss::stored_index_path(run, "qwen27b-tp3-int8");
        if (mode == "partial") {
            assert(!fs::exists(cache_index));
        } else {
            const auto cache_entry = ss::read_stored_index(cache_index);
            const auto named_hit = ss::stored_manifest_for(run, "qwen27b-tp3-int8", cache_entry.key);
            assert(named_hit && named_hit->filename() == cache_entry.manifest_filename &&
                   ss::complete_file_identity(*named_hit) == cache_entry.digest);
        }

        size_t first_writer = SIZE_MAX, admit = SIZE_MAX, first_gpu = SIZE_MAX;
        size_t writer_count = 0, gpu_count = 0;
        const auto& events = terminal.at("events");
        for (size_t i = 0; i < events.size(); ++i) {
            const auto& item = events.at(i);
            const std::string kind = item.value("kind", std::string());
            const std::string role = item.value("role", std::string());
            if (kind == "started" && role == "manifest writer") {
                first_writer = std::min(first_writer, i);
                ++writer_count;
            }
            if (kind == "power_applied")
                admit = i;
            if (kind == "started" && (role == "measuring engine" || role == "engine")) {
                first_gpu = std::min(first_gpu, i);
                ++gpu_count;
            }
        }
        assert(writer_count == (mode == "hit" ? 1U : mode == "miss" ? 3U : 2U));
        assert(first_writer != SIZE_MAX);
        if (mode == "partial") {
            assert(admit == SIZE_MAX && gpu_count == 0);
        } else {
            assert(first_writer < admit && admit < first_gpu);
            assert(gpu_count == (mode == "miss" ? 2U : 1U));
        }
        for (const auto& child : terminal.at("processes")) {
            assert(child.value("exited", false));
            assert(child.value("pid", 0) > 0);
        }
        const auto evidence = [&] {
            std::stringstream out;
            out << std::ifstream(trace).rdbuf();
            return out.str();
        }();
        assert(evidence.find("begin\n") == 0 && evidence.find("verified_exit\n") != std::string::npos &&
               evidence.find("restore\n") > evidence.find("verified_exit\n"));
    }
}

// End-to-end CPU-only planner boundary: real resolver and bc-serve writer,
// fake engine/helper process boundary and BDF-specific sysfs fixtures.
void helper_vram_owner_graph(Fixture& t) {
    namespace ss = bc::serving::session;
    const fs::path configs = fs::path(std::getenv("BIGCHERRY_LAYOUTS")).parent_path() / "configs";
    const fs::path home = t.root / "session-build" / "radiance_home";
    // The normal lightweight RAD fixture omits name strings. The real
    // resource ledger requires an actual bounded name table in its directory.
    const fs::path model = t.root / "helper-vram-model.rad";
    bc::test::write_rad(model, 4, 16, 128, 128, {});
    {
        std::fstream rad(model, std::ios::in | std::ios::out | std::ios::binary);
        RadFileHeader header{};
        rad.read(reinterpret_cast<char*>(&header), sizeof header);
        header.str_off = header.file_bytes;
        header.str_bytes = 7;
        header.file_bytes += header.str_bytes;
        rad.seekp(static_cast<std::streamoff>(header.str_off));
        rad.write("weight", 7);
        rad.seekp(0);
        rad.write(reinterpret_cast<const char*>(&header), sizeof header);
        assert(rad);
    }
    const fs::path site =
        t.file("helper-vram-graph.toml",
               "[engine]\nmodel = \"" + model.string() + "\"\nradiance_home = \"" + home.string() + "\"\n");
    ss::EnginePlanInputs in;
    in.options.site = site;
    in.options.sys_root = t.root;
    in.options.configs = {configs};
    in.options.model_root = t.root;
    in.options.sets = {"layout.name=\"flashnext-2xtx\"", "helper.combine_threads=1",
                       "handoff.heat=\"" + (t.root / "writer.hot-first").string() + "\"",
                       "handoff.histogram=\"" + (t.root / "vram-histogram").string() + "\""};
    in.self = BC_SERVE_EXE;
    in.engine_binary = BC_FAKE_ENGINE;
    in.helper_binary = BC_FAKE_ENGINE;
    int writers = 0;
    std::vector<fs::path> measuring_histograms;
    std::vector<std::vector<std::string>> serving_writer_args;
    in.runner = [&](const std::vector<std::string>& args) -> ss::ManifestRun {
        ++writers;
        auto result = run_bc_serve({args.begin() + 1, args.end()});
        const std::string file = fs::path(args.at(2)).filename().string();
        if (file.rfind("manifest-", 0) == 0 && file.rfind("manifest-measuring-", 0) != 0 &&
            file.rfind("manifest-resources-", 0) != 0)
            serving_writer_args.push_back(args);
        if (result.exit_code == 0 && fs::is_regular_file(args.at(2))) {
            if (file.rfind("manifest-measuring-", 0) == 0 ||
                (file.rfind("manifest-", 0) == 0 && file.rfind("manifest-resources-", 0) != 0)) {
                const Json doc = Json::parse(std::ifstream(args.at(2)));
                const fs::path prefix =
                    doc.at("spec").at("handoff").at("telemetry").at("histogram_file").get<std::string>();
                if (file.rfind("manifest-measuring-", 0) == 0)
                    measuring_histograms.push_back(prefix);
            }
        }
        return {result.exit_code, result.output, result.output};
    };
    in.helper_vram.read_used = [](const fs::path& path) -> uint64_t {
        throw std::runtime_error("injected unreadable BDF-specific counter: " + path.string());
    };
    ss::Hooks hooks;
    hooks.wait = [](std::chrono::milliseconds) { return true; };
    bool blocked_baseline = false;
    try {
        (void)ss::plan_engine_only(in, t.root / "vram-graph-run", hooks);
    } catch (const ss::PlanFailed& e) {
        const std::string why = e.what();
        blocked_baseline = why.find("BC_CFG_ADMISSION_REFUSED") != std::string::npos &&
                           why.find("cannot read VRAM") != std::string::npos;
        if (!blocked_baseline)
            std::cerr << "helper graph: " << why << "\n";
    }
    if (!blocked_baseline || writers < 1)
        throw std::runtime_error("helper graph did not reach the pre-start BDF baseline");

    for (const std::string mode : {"immediate", "late", "never"}) {
        ss::EnginePlanInputs active = in;
        std::map<std::string, unsigned> samples;
        active.helper_vram.read_used = [&](const fs::path& path) -> uint64_t {
            std::ifstream used(path);
            uint64_t bytes = 0;
            if (!(used >> bytes))
                throw std::runtime_error("missing fake BDF usage file");
            const unsigned n = ++samples[path.string()];
            // Three pre-start baseline observations; GPU usage lingers AFTER
            // reaped fake helpers, then settles. No actual GPU is contacted.
            if (n > 3 && (mode == "never" || (mode == "late" && n < 7)))
                return bytes + uint64_t{256} * 1024 * 1024;
            return bytes;
        };
        ss::Hooks live_hooks;
        live_hooks.wait = [](std::chrono::milliseconds) { return true; };
        live_hooks.helper_stop_timeout = std::chrono::milliseconds(400);
        const int before = writers;
        std::string refusal;
        const fs::path run_dir = t.root / ("vram-graph-" + mode);
        fs::create_directories(run_dir);
        try {
            (void)ss::plan_engine_only(active, run_dir, live_hooks);
        } catch (const ss::PlanFailed& e) {
            refusal = e.what();
        }
        if (mode == "never") {
            if (refusal.find("measuring helper VRAM not released") == std::string::npos ||
                refusal.find("baseline=") == std::string::npos || writers - before != 3)
                throw std::runtime_error("helper graph timeout: " + refusal +
                                         " writers=" + std::to_string(writers - before));
        } else if (writers - before < 4) {
            std::ostringstream message;
            message << "helper graph " << mode << " stopped before final writer: " << refusal
                    << " writers=" << (writers - before);
            throw std::runtime_error(message.str());
        }
    }
    // Each measuring writer gets a private telemetry generation. A future
    // engine that publishes during --plan-memory must never replace the
    // serving engine's rank0 histogram at the operator's destination.
    if (measuring_histograms.empty())
        throw std::runtime_error("real measuring writer histogram fixture did not run");
    for (const fs::path& path : measuring_histograms) {
        if (path == t.root / "vram-histogram" || path.parent_path().parent_path() != t.root ||
            path.parent_path().filename().string().rfind("vram-graph-", 0) != 0 ||
            path.filename().string().rfind("histogram-measuring-", 0) != 0)
            throw std::runtime_error("measuring histogram not isolated: " + path.string());
    }
    if (serving_writer_args.empty())
        throw std::runtime_error("real serving writer was never invoked");
    const std::string site_recording = "handoff.histogram=" + Json((t.root / "vram-histogram").string()).dump();
    for (const auto& args : serving_writer_args) {
        size_t histogram_overrides = 0;
        for (size_t i = 0; i + 1 < args.size(); ++i)
            if (args[i] == "--set" && args[i + 1].rfind("handoff.histogram=", 0) == 0) {
                if (args[i + 1] != site_recording)
                    throw std::runtime_error("private measuring prefix leaked to serving writer");
                ++histogram_overrides;
            }
        if (histogram_overrides != 1)
            throw std::runtime_error("serving writer lost the operator's recording destination");
    }

    // Exercise the real CPU publisher twice as future --plan-memory and serving
    // producers would: neither generation may overwrite the other's files.
    std::vector<uint32_t> counts(static_cast<size_t>(r3::moe_shm::kMaxLayers) * 16, 0);
    counts[0] = 42;
    const std::string serving_prefix = (t.root / "vram-histogram").string();
    const std::string measurement_prefix = measuring_histograms.back().string();
    const std::string nonce = "0123456789abcdef0123456789abcdef";
    if (!r3::heat_publication::publish(measurement_prefix, 0, 16, counts) ||
        !r3::heat_publication::publish(serving_prefix, 0, 16, counts, nonce))
        throw std::runtime_error("engine CPU histogram publication failed");
    const ss::HistogramExpectation expected{
        4, 16, std::chrono::system_clock::now() - std::chrono::seconds(3), {}, nonce};
    const fs::path rank = serving_prefix + ".rank0";
    if (ss::histogram_verdict(rank, std::chrono::system_clock::now() + std::chrono::seconds(1),
                              std::chrono::seconds(30), expected) != ss::HistogramVerdict::fresh)
        throw std::runtime_error("serving producer proof must qualify independently of measurement");
    counts[0] += 1;
    if (!r3::heat_publication::publish(measurement_prefix, 0, 16, counts) ||
        ss::histogram_verdict(rank, std::chrono::system_clock::now() + std::chrono::seconds(1),
                              std::chrono::seconds(30), expected) != ss::HistogramVerdict::fresh)
        throw std::runtime_error("later measuring publication affected the serving generation");
}

int main(int argc, char** argv) {
#ifdef BC_NATIVE_V3_SESSION_ONLY
    (void)argc;
    (void)argv;
    try {
        Fixture t;
        v3_writer_cli_rejects_unowned_imports(t);
        session_engine_plan(t);
        std::cout << "PASS: native v3 sealed snapshot session draft and pre-cache gate\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
#else
    bool write_golden = false;
    const char* dump_dir = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--write-golden") {
            write_golden = true;
        } else if (std::string(argv[i]) == "--dump-manifests" && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (std::string(argv[i]) == "--bc-serve-exe" && i + 1 < argc) {
            serve_test_executable = argv[++i];
        } else {
            std::cerr << "usage: bc_serving_test [--write-golden] [--dump-manifests DIR] [--bc-serve-exe PATH]\n";
            return 2;
        }
    }
    try {
        sha256_vectors();
        Fixture t;
        detection(t);
        detection_limits(t);
        unknown_keys(t);
        layout_sources(t);
        type_range_required(t);
        precedence_and_caps(t);
        provenance_hash_and_parity(t);
        calibration_store(t);
        host_reserve(t);
        host_ledger_plan();
        measured_plan_cache(t);
        manifest_writer_step(t);
        layout_manifests_golden(t, write_golden, dump_dir);
        engine_only_layout(t);
        tp_engine_with_helper_layout(t);
        tp_engine_with_two_helpers_layout(t);
        mixed_arch_engine_layout(t);
        config_files(t);
        native_explain_cli(t);
        native_static_resolution_identity(t);
        session_engine_plan(t);
        helper_vram_owner_graph(t);
        session_owner_real_planner_admission(t);
        derived_budget_is_the_workers(t);
        sdk_environment_step(t);
        std::cout << "PASS: fake KFD/PCI/proc detection; strict keys; types/ranges; required; "
                     "precedence/caps; provenance/hash; launcher engine/WorkerConfig parity; "
                     "calibration store; calibrated host reserve; host ledger; measured plan cache; measuring "
                     "step; manifest writer\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
#endif
}
