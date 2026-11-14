#ifdef NDEBUG
#undef NDEBUG // Assertions are test requirements even in RelWithDebInfo.
#endif
#include "serving.h"
#include "calibration.h"
#include "config_loader.h"
#include "ledger.h"
#include "live_policy_ingest.h"
#include "experts.h"
#include "live_policy.h"
#include "manifest_writer.h"
#include "v2/manifest.h"
#include "plan_cache.h"
#include "sdk_check.h"
#include "test_rad_fixture.h"
#include "worker_footprint.h"
#include "v2/helper_settings.h"
#include "sha256.h"

#undef NDEBUG   /* the checks below are assert()s: keep them live in Release component builds */
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>

#include <atomic>
#include <cassert>
#ifdef NDEBUG
#error "assert-based test built with NDEBUG: its assertions would not run"
#endif
#include <thread>
#include <vector>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <fstream>
#include <regex>
#include <sstream>
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
        // A v2 cgroup with a limit at the leaf and none above; memlock limited, no CAP_IPC_LOCK;
        // amdgpu loaded with its system memory limit in force.
        write(root / "proc/self/cgroup", "0::/agents/serve\n");
        write(root / "sys/fs/cgroup/agents/serve/memory.max", std::to_string(uint64_t(64) << 30) + "\n");
        write(root / "sys/fs/cgroup/agents/serve/memory.current", std::to_string(uint64_t(10) << 30) + "\n");
        write(root / "sys/fs/cgroup/agents/serve/memory.stat", "anon 1\nfile " + std::to_string(uint64_t(4) << 30) + "\n");
        write(root / "sys/fs/cgroup/agents/memory.max", "max\n");
        write(root / "sys/fs/cgroup/memory.max", "max\n");
        write(root / "proc/self/limits",
              "Limit                     Soft Limit           Hard Limit           Units\n"
              "Max locked memory         8388608              8388608              bytes\n");
        write(root / "proc/self/status", "Name:\tbc-serve\nCapEff:\t00000000a80425fb\n");
        write(root / "sys/module/amdgpu/parameters/no_system_mem_limit", "N\n");
        bc::test::write_rad(root / "model.rad", 4, 16, 128, 128, {});
        facts = bc::serving::detect(root);
        // detect() counts this machine's CPUs; the fixture fixes the count so the result does not depend on the runner.
        facts["host"]["cpu_count"] = 32;
    }
    ~Fixture() { fs::remove_all(root); }
    fs::path file(const char* name, const std::string& content) {
        const auto path = root / name;
        write(path, content);
        return path;
    }
    Options standard() {
        Options opts;
        opts.site = file("site.toml", "[engine]\nmodel = \"" + (root / "model.rad").string() + "\"\n"
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
    write(root / "proc/self/limits", "Max locked memory         unlimited            unlimited            bytes\n");
    write(root / "proc/self/status", "CapEff:\t0000000000004000\n");
    write(root / "sys/module/amdgpu/parameters/no_system_mem_limit", "Y\n");
    const Json h = bc::serving::detect(root).at("host");
    assert(h.at("memory_cgroups").size() == 1);
    assert(h.at("memory_cgroups")[0].at("version") == 1);
    assert(h.at("memory_cgroups")[0].at("room_mib") == 31 * 1024);   // 32 GiB - (3 - 2 total_cache)
    assert(h.at("memlock").at("limit_mib").is_null() && h.at("memlock").at("cap_ipc_lock") == true);
    assert(h.at("amdgpu_pin").at("lifted") == true && h.at("amdgpu_pin").at("cap_mib").is_null());
    fs::remove(root / "proc/self/limits");
    bool refused = false;
    try { (void)bc::serving::detect(root); } catch (const std::exception& e) {
        refused = std::string(e.what()).find("proc/self/limits") != std::string::npos;
    }
    assert(refused);
    fs::remove_all(root);
}

void unknown_keys(Fixture& t) {
    auto opts = t.standard();
    opts.site = t.file("unknown-site.toml",
                       "[engine]\nmodel = \"" + (t.root / "model.rad").string() + "\"\n"
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
    opts.site = t.file("nolayout-site.toml", "[engine]\nmodel = \"" + (t.root / "model.rad").string() + "\"\n"
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
    const Json all = bc::serving::load_layouts(Options{});   // BIGCHERRY_LAYOUTS: the repository's example layouts
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
    {   // the standard profile's 8 combine threads on a 4-CPU host
        Json few = t.facts;
        few["host"]["cpu_count"] = 4;
        fail(t.standard(), few, "helper.combine_threads exceeds discovered CPU count");
    }
}

void precedence_and_caps(Fixture& t) {
    auto opts = t.standard();
    opts.site = t.file("precedence-site.toml",
        "[engine]\nmodel = \"" + (t.root / "model.rad").string() + "\"\nradiance_home = \"/runtime\"\nport = 4700\n"
        "[handoff]\nheat = \"/heat.hot-first\"\nhistogram = \"/heat.live\"\n" + std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
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
        "[engine]\nmodel = \"" + (t.root / "model.rad").string() + "\"\nradiance_home = \"/runtime\"\n"
        "[memory]\nhost_pool_mib = 7000\n"
        "[hard_caps.memory]\nhost_pool_mib = 8000\n"
        "[handoff]\nheat = \"/heat.hot-first\"\nhistogram = \"/heat.live\"\n" + std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
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
        "[engine]\nmodel = \"" + (t.root / "model.rad").string() + "\"\nradiance_home = \"/runtime\"\n"
        "[hard_caps.memory]\nhost_pool_mib = 999999\n"
        "[handoff]\nheat = \"/heat.hot-first\"\nhistogram = \"/heat.live\"\n" + std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
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
    assert(!r.manifest.contains("content_sha256") && !r.manifest.contains("hash_algorithm"));   // no second manifest format
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
        {"model", (t.root / "model.rad").string()}, {"radiance_home", "/runtime"}, {"port", 36080},
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
    assert(!schema["properties"]["handoff"].contains("required"));   // heat and histogram are required by a layout with helpers, not by the schema
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
    const auto cache = t.file("refit.toml",
        "schema_version = 1\n"
        "[[entry]]\nkey = \"calibration.x.v\"\nmatch = { layout = \"L\" }\nvalue = 20\nseeded = \"2026-02-01\"\n"
        "evidence = \"RHP12 refit\"\n");
    const Calibration r = Calibration::load(seed, cache);
    assert(r.integer("calibration.x.v", {"m", "L", "throughput"}) == 20);
    assert(r.get("calibration.x.v", {"m", "L", "throughput"}).source == "cache");
    assert(r.integer("calibration.x.v", {"m", "L", "latency"}) == 3);
    auto refused = [](const std::function<void()>& f, const std::string& needle) {
        try { f(); } catch (const std::exception& e) {
            if (std::string(e.what()).find(needle) != std::string::npos) return;
            throw std::runtime_error("wrong calibration diagnostic for " + needle + ": " + e.what());
        }
        throw std::runtime_error("expected calibration refusal containing " + needle);
    };
    refused([&] { (void)c.get("calibration.y.v", {"m", "L", "latency"}); }, "no entry for calibration.y.v");
    refused([&] { (void)Calibration::load("schema_version = 1\n[[entry]]\nkey = \"calibration.a\"\nvalue = 1\n"
                                          "seeded = \"d\"\nevidence = \"e\"\nnote = 1\n"); }, "unknown key note");
    refused([&] { (void)Calibration::load("schema_version = 1\n[[entry]]\nkey = \"calibration.a\"\nvalue = 1\n"
                                          "seeded = \"d\"\nevidence = \"e\"\nmatch = { host = \"h\" }\n"); },
            "allowed: model, layout, objective");
    refused([&] { (void)Calibration::load("schema_version = 1\n[[entry]]\nkey = \"calibration.a\"\nvalue = 1\n"
                                          "seeded = \"d\"\nevidence = \"\"\n"); }, "evidence must be a non-empty");
    refused([&] { (void)Calibration::load(seed + "[[entry]]\nkey = \"calibration.x.v\"\nvalue = 9\n"
                                          "seeded = \"d\"\nevidence = \"e\"\n"); }, "duplicate entry");
    // The compiled-in seeds parse and carry evidence for every rule parameter the plan reads.
    const Calibration standard = Calibration::standard();
    for (const char* objective : {"latency", "balanced"})
        assert(!standard.get("calibration.kv_backing_floor.fraction_ppm",
                             {"m", "flashnext-4gpu", objective}).evidence.empty());
    refused([&] { (void)standard.get("calibration.kv_backing_floor.fraction_ppm", {"m", "flashnext-4gpu", "throughput"}); },
            "no entry");
    // No floor is selected for throughput: find() says so (the plan check then takes 0), where get() is an error.
    assert(!standard.find("calibration.kv_backing_floor.fraction_ppm", {"m", "flashnext-4gpu", "throughput"}));
    assert(standard.integer("calibration.kv_backing_floor.fraction_ppm", {"m", "flashnext-4gpu", "balanced"}) == 600000);
    assert(standard.integer("calibration.kv_backing_floor.fraction_ppm", {"m", "flashnext-4gpu", "latency"}) == 500000);
    // v1: a calibration value is an integer, a string or a boolean; a fraction is parts per million.
    refused([&] { (void)Calibration::load(seed + "[[entry]]\nkey = \"calibration.y.v\"\nvalue = 0.5\n"
                                          "seeded = \"d\"\nevidence = \"e\"\n"); }, "non-integer number");
}

void host_reserve(Fixture& t) {
    // 128 GiB fixture: 5% of 131072 MiB is 6554 MiB, above the 4096 MiB floor.
    auto r = bc::serving::resolve(t.standard(), t.facts);
    assert(r.values.at("memory").at("host_reserve_mib") == 6554);
    assert(r.values.at("engine").at("host_headroom_mib") == 6554);
    assert(r.provenance.at("engine.host_headroom_mib").at("inputs") == Json::array({"memory.host_reserve_mib"}));
    const Json& p = r.provenance.at("memory.host_reserve_mib");
    assert(p.at("origin") == "calibrated");
    assert(p.at("source").get<std::string>().find("calibrated (seed: 2026-11-02) calibration.host_reserve.floor_mib") !=
           std::string::npos);
    assert(p.at("inputs").size() == 3);
    // The site may only raise it.
    auto opts = t.standard();
    opts.site = t.file("floor-site.toml", "[engine]\nmodel = \"" + (t.root / "model.rad").string() + "\"\nradiance_home = \"/runtime\"\n"
                                          "[handoff]\nheat = \"/heat.hot-first\"\nhistogram = \"/heat.live\"\n"
                                          "[hard_floors.memory]\nhost_reserve_mib = 8192\n" + std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
    r = bc::serving::resolve(opts, t.facts);
    assert(r.values.at("memory").at("host_reserve_mib") == 8192);
    assert(r.provenance.at("memory.host_reserve_mib").at("origin") == "site");
    opts.site = t.file("low-floor-site.toml", "[engine]\nmodel = \"" + (t.root / "model.rad").string() + "\"\nradiance_home = \"/runtime\"\n"
                                              "[handoff]\nheat = \"/heat.hot-first\"\nhistogram = \"/heat.live\"\n"
                                              "[hard_floors.memory]\nhost_reserve_mib = 1024\n" + std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
    r = bc::serving::resolve(opts, t.facts);
    assert(r.values.at("memory").at("host_reserve_mib") == 6554);
    opts.site = t.file("bad-floor-site.toml", "[engine]\nmodel = \"" + (t.root / "model.rad").string() + "\"\nradiance_home = \"/runtime\"\n"
                                              "[handoff]\nheat = \"/heat.hot-first\"\nhistogram = \"/heat.live\"\n"
                                              "[hard_floors.memory]\nhost_pool_mib = 1024\n" + std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
    fail(opts, t.facts, "unknown floor hard_floors.memory.host_pool_mib");
    opts = t.standard();
    opts.profile = t.file("floor-profile.toml", "[hard_floors.memory]\nhost_reserve_mib = 8192\n");
    fail(opts, t.facts, "hard_floors: site-only");
    opts = t.standard();
    opts.sets = {"memory.host_reserve_mib=1"};
    fail(opts, t.facts, "derived, cannot override");
    // A refit in the calibration cache moves the reserve, and explain names the cache.
    opts = t.standard();
    opts.calibration_cache = t.file("reserve-refit.toml",
        "schema_version = 1\n[[entry]]\nkey = \"calibration.host_reserve.total_fraction_ppm\"\n"
        "match = { layout = \"flashnext-4gpu\" }\nvalue = 100000\nseeded = \"2026-12-01\"\nevidence = \"refit\"\n");
    r = bc::serving::resolve(opts, t.facts);
    assert(r.values.at("memory").at("host_reserve_mib") == 13108);
    assert(r.provenance.at("memory.host_reserve_mib").at("source").get<std::string>().find(
               "calibrated (cache: 2026-12-01)") != std::string::npos);
}

void host_ledger_plan() {
    using bc::serving::CheckpointStore;
    using bc::serving::HostConsumer;
    using bc::serving::LedgerInputs;
    using bc::serving::Residence;
    constexpr uint64_t MiB = 1048576;
    // The measured host (flashnext-4gpu, 2026-11-02): 94082 MiB RAM, 83020 MiB available, pin cap 91076 MiB.
    Json host = {{"mem_total_mib", 94082}, {"mem_available_mib", 83020}, {"memory_cgroups", Json::array()},
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
    assert(l.explain.at("checkpoints").at("why").get<std::string>().find("calibrated share (want)") != std::string::npos);
    assert(l.explain.at("limits").size() == 2);
    // The floor share plans only the floor and the tier takes the difference.
    in.checkpoints.share = "floor";
    l = bc::serving::host_ledger(in);
    assert(l.checkpoint_slots == 16 && l.host_kv_mib == 11055 + 8064 - 1792);
    in.checkpoints.share = "want";
    // Want plus the minimum tier does not fit: the store gives the tier its minimum, down toward the floor.
    in.min_tier_bytes = 12000 * MiB;   // 945 MiB short: 9 slots of 112 MiB
    l = bc::serving::host_ledger(in);
    assert(l.checkpoint_slots == 63 && l.host_kv_mib == 11055 + 9 * 112);
    assert(l.explain.at("checkpoints").at("why").get<std::string>().find("shrunk from 72 (want) to 63") != std::string::npos);
    // Even the floor leaves too little: the store stops at its floor and the tier is 0, saying so.
    in.min_tier_bytes = 30000 * MiB;
    l = bc::serving::host_ledger(in);
    assert(l.checkpoint_slots == 16 && l.host_kv_mib == 0);
    assert(l.explain.at("checkpoints").at("why").get<std::string>().find("reaching the floor") != std::string::npos);
    assert(l.explain.at("host_kv_why").get<std::string>().find("no whole session") != std::string::npos);
    in.min_tier_bytes = 4 * 1024 * MiB;
    // The pin cap binds when the pinned consumers and the store leave less under it than MemAvailable does.
    in.host_facts["amdgpu_pin"]["cap_mib"] = 75000;
    l = bc::serving::host_ledger(in);
    assert(l.host_kv_mib == 75000 - (48824 + 7300 + 8064));   // 10812, under the 11055 MemAvailable leaves
    assert(l.explain.at("host_kv_why").get<std::string>().find("amdgpu pin cap") != std::string::npos);
    in.host_facts = host;
    // A memory cgroup is a limit of its own.
    in.host_facts["memory_cgroups"] = Json::array({{{"dir", "/sys/fs/cgroup/job"}, {"version", 2}, {"room_mib", 80000}}});
    l = bc::serving::host_ledger(in);
    assert(l.host_kv_mib == 80000 - (48824 + 7300 + 3072 + 8064) - 4705);
    // Too little room even for the floor: refused, naming the floor, the largest consumer and what to change.
    in.host_facts = host;
    in.host_facts["mem_available_mib"] = 60000;
    bool refused = false;
    try { (void)bc::serving::host_ledger(in); } catch (const std::exception& e) {
        const std::string w = e.what();
        refused = w.find("under MemAvailable") != std::string::npos && w.find("4705 MiB reserve") != std::string::npos &&
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
    try { (void)bc::serving::host_ledger(in); } catch (const std::exception& e) {
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
    try { (void)bc::serving::host_ledger(in); } catch (const std::exception& e) {
        refused = std::string(e.what()).find("want, floor or tier_first") != std::string::npos;
    }
    assert(refused);
    // tier_first: the tier takes its target out of what the floor leaves, the store the remainder, up to its want.
    in.checkpoints.share = "tier_first";
    in.tier_target_bytes = 12288 * MiB;
    l = bc::serving::host_ledger(in);
    // floor 16 x 112 MiB leaves 83020 - (48824 + 7300 + 3072) - 1792 - 4705 = 17327 MiB; the tier takes 12288 and the
    // store 16 + (17327 - 12288) / 112 = 60 slots of a want of 72.
    assert(l.host_kv_mib == 12288 && l.checkpoint_slots == 60);
    assert(l.explain.at("checkpoints").at("want_slots") == 72 && l.explain.at("checkpoints").at("floor_slots") == 16);
    assert(l.explain.at("checkpoints").at("why").get<std::string>().find("takes 12288 MiB first") != std::string::npos);
    assert(l.explain.at("host_kv_why").get<std::string>().find("calibrated target") != std::string::npos);
    // A target larger than the room: the tier takes the room the floor leaves and the store stays at its floor.
    in.tier_target_bytes = 40000 * MiB;
    l = bc::serving::host_ledger(in);
    assert(l.host_kv_mib == 17327 && l.checkpoint_slots == 16);
    assert(l.explain.at("host_kv_why").get<std::string>().find("below its calibrated target") != std::string::npos);
    in.checkpoints.share = "want";
    in.tier_target_bytes = 0;
    // The compiled-in share reproduces today's launch: tier first, 12 GiB.
    const auto standard = bc::serving::Calibration::standard();
    assert(standard.get("calibration.checkpoint_share", {"m", "flashnext-4gpu", "latency"}).value == "tier_first");
    assert(standard.integer("calibration.host_kv_tier.target_mib", {"m", "flashnext-4gpu", "latency"}) == 12288);

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
        try { (void)bc::serving::decode_engine_plan(j, "test"); } catch (const std::exception& e) {
            if (std::string(e.what()).find(needle) != std::string::npos) return;
            throw std::runtime_error("wrong engine plan diagnostic for " + needle + ": " + e.what());
        }
        throw std::runtime_error("expected engine plan refusal containing " + needle);
    };
    Json bad = plan; bad["extra"] = 1; rejects(bad, "extra: unknown member");
    bad = plan; bad["plan_version"] = 2; rejects(bad, "must be 1");
    bad = plan; bad["host"][0]["residence"] = "swap"; rejects(bad, "is not one of pinned, mlocked, pageable");
    bad = plan; bad["host"][0]["bytes"] = -1; rejects(bad, "below the minimum 0");
    bad = plan; bad["host"].push_back(plan["host"][0]); rejects(bad, "duplicate consumer ngram_table");
    bad = plan; bad["host"][0].erase("from"); rejects(bad, "from: missing required member");
    bad = plan; bad["checkpoints"]["floor_slots"] = 99; rejects(bad, "floor_slots exceeds want_slots");
    bad = plan; bad["vram"].erase("kv"); rejects(bad, "kv: missing required member");
    bad = plan; bad["vram"]["cards"] = Json::array(); rejects(bad, "needs at least 1 item");
    bad = plan; bad["vram"]["cards"][0]["arch"] = "gfx1201"; rejects(bad, "arch: unknown member");
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
    // The key covers the planning inputs, the card and the host pool the plan echoes; not the memory outputs the plan
    // feeds (host KV tier, headroom) nor where the engine listens or lives (port, runtime home, model path).
    const auto r = sv::resolve(t.standard(), t.facts);
    const Json k = sv::plan_key_inputs(r.values, t.facts, id, eid, "");
    assert(k.at("card").at("arch") == "gfx1201" && k.at("card").at("vram_total_mib") == 32624);
    assert(k.at("layout") == "flashnext-4gpu");
    for (const char* absent : {"prefix_cache_mib", "host_headroom_mib", "port", "radiance_home", "model"})
        assert(!k.at("engine").contains(absent));
    assert(k.at("engine").contains("host_pool_mib") && k.at("engine").contains("max_num_seqs"));
    Json v = r.values;
    for (const char* moved : {"prefix_cache_mib", "host_headroom_mib", "port"}) {
        v = r.values;
        v["engine"][moved] = 1;
        assert(sv::plan_key(sv::plan_key_inputs(v, t.facts, id, eid, "")) == sv::plan_key(k));
    }
    v = r.values;
    v["engine"]["host_pool_mib"] = 1;   // the plan echoes the pool: a plan measured under another pool is another plan
    assert(sv::plan_key(sv::plan_key_inputs(v, t.facts, id, eid, "")) != sv::plan_key(k));
    v = r.values;
    v["engine"]["checkpoint_interval"] = 2048;
    assert(sv::plan_key(sv::plan_key_inputs(v, t.facts, id, eid, "")) != sv::plan_key(k));
    // Engine arguments passed through verbatim can change the plan, so they are part of the key.
    assert(sv::plan_key(sv::plan_key_inputs(r.values, t.facts, id, eid, "--kv-resident 65536")) != sv::plan_key(k));
    assert(sv::plan_key(sv::plan_key_inputs(r.values, t.facts, id, eid, "--kv-resident 65536")) ==
           sv::plan_key(sv::plan_key_inputs(r.values, t.facts, id, eid, "--kv-resident 65536")));
    // Every launcher setting that reaches the engine changes the key (a plan is never found for another configuration);
    // only the port, the runtime-home and model paths and the memory outputs the plan feeds do not.
    {
        const std::string base_key = sv::plan_key(k);
        const std::pair<const char*, const char*> changes[] = {
            {"service.objective", "latency"}, {"service.max_sessions", "4"}, {"service.max_context", "131072"},
            {"engine.max_num_batched_tokens", "1024"}, {"engine.checkpoint_interval", "2048"},
            {"engine.kv_cache_dtype", "fp8_v4"}, {"engine.num_speculative_tokens", "2"}, {"engine.rope_scaling", "yarn:4"},
            {"engine.ngram_placement", "disk"}, {"engine.prefix_resume", "exact"}, {"engine.tp", "2"},
            {"memory.helper_budgets_mib", "[22528,22528,7168]"}, {"tuning.tp_ownership", "\"kv=1,1,2;ffn=8,8,16\""},
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
                if (std::string(e.what()).find(key) == std::string::npos) throw;
            }
        }
        const std::pair<const char*, const char*> same[] = {{"engine.port", "36081"}, {"memory.host_kv_mib", "6144"}};
        for (const auto& [key, value] : same) {
            auto opts = t.standard();
            opts.sets = {std::string(key) + "=" + value};
            assert(sv::plan_key(sv::plan_key_inputs(sv::resolve(opts, t.facts).values, t.facts, id, eid, "")) == base_key);
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
    const Json ranks = {{"schema_version", 1}, {"kind", "radiance_p1_rank_demand"}, {"gpu", Json::array()}, {"resident_other_bytes", 0}};
    const auto stored = sv::plan_cache_store(dir, k, plan, ranks, sv::HostResident{2304, 1536});
    const auto found = sv::plan_cache_lookup(dir, k);
    assert(found && found->key == stored.key && found->plan.checkpoints.want_slots == 72 && !found->measured_at.empty());
    // A plan is stored only for the configuration it belongs to: the host pool it echoes and the expected host consumer
    // list (the n-gram table exactly when it lives in RAM; nothing unknown).
    {
        auto decoded = [&](Json j) { return sv::decode_engine_plan(j, "test"); };
        auto refused_for = [&](const Json& values, const Json& plan_json, const std::string& needle) {
            try { sv::check_plan_for_configuration(values, decoded(plan_json)); } catch (const std::exception& e) {
                if (std::string(e.what()).find(needle) != std::string::npos) return;
                throw std::runtime_error("wrong store diagnostic for " + needle + ": " + e.what());
            }
            throw std::runtime_error("expected store refusal containing " + needle);
        };
        Json values = r.values;
        values["engine"]["ngram_placement"] = "disk";
        sv::check_plan_for_configuration(values, decoded(plan));                 // host [] with the table on disk: fine
        Json with_table = plan;
        with_table["host"] = Json::array({{{"name", "ngram_table"}, {"bytes", 1}, {"residence", "pinned"},
                                           {"from", "engine plan (test)"}, {"remedy", ""}}});
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
    // The cache entry's own plan is what a manifest carries as the expected plan (the engine reads it from the manifest, never
    // from a file next to the cache): an entry edited by one byte validates, and the engine's CHECK is what refuses it.
    {
        const auto file = dir / (stored.key + ".json");
        const std::string original = [&] { std::ifstream in(file); std::ostringstream t; t << in.rdbuf(); return t.str(); }();
        Json e = Json::parse(original);
        e["plan"]["host"] = Json::array({{{"name", "host_arenas"}, {"bytes", 100}, {"residence", "pinned"},
                                          {"from", "engine plan (test)"}, {"remedy", ""}}});
        std::ofstream(file, std::ios::trunc) << e.dump();
        const auto before = sv::plan_cache_lookup(dir, k);
        assert(before && before->plan.host.at(0).bytes == 100);
        e["plan"]["host"][0]["bytes"] = 101;
        std::ofstream(file, std::ios::trunc) << e.dump();
        const auto drifted = sv::plan_cache_lookup(dir, k);
        assert(drifted && drifted->plan.host.at(0).bytes == 101);
        std::ofstream(file, std::ios::trunc) << original;                      // restore for the tests below
    }
    {   // an entry from before the resident host memory was measured is a miss: the configuration is measured again
        const auto file = dir / (stored.key + ".json");
        const std::string original = [&] { std::ifstream in(file); std::ostringstream t; t << in.rdbuf(); return t.str(); }();
        Json e = Json::parse(original);
        assert(e.at("resident").at("helpers_mib") == 2304 && e.at("resident").at("engine_mib") == 1536);
        e.erase("resident");
        std::ofstream(file, std::ios::trunc) << e.dump();
        assert(!sv::plan_cache_lookup(dir, k));
        e["resident"] = {{"helpers_mib", 5}, {"engine_mib", 0}};
        std::ofstream(file, std::ios::trunc) << e.dump();
        bool zero_refused = false;
        try { (void)sv::plan_cache_lookup(dir, k); } catch (const std::exception&) { zero_refused = true; }
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
    try { (void)sv::plan_cache_lookup(dir, k); } catch (const std::exception& e) {
        refused = std::string(e.what()).find("key inputs differ") != std::string::npos;
    }
    assert(refused);
    // Plans and cache entries are read as the engine reads them: a repeated member is refused before any decoding.
    {
        const std::string text = plan.dump();
        auto dup_refused = [&](const std::string& doc, const char* what) {
            try { (void)sv::parse_plan_json(doc, "dup.json"); } catch (const std::exception& e) {
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
    Json bad = plan; bad.erase("vram");
    refused = false;
    try { (void)sv::plan_cache_store(dir, k, bad, ranks, sv::HostResident{2304, 1536}); } catch (const std::exception& e) {
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
        std::ofstream(build / "bigcherry-build-metadata-radiance.json", std::ios::trunc) << "{\"toolchain\": " << toolchain << "}";
    };
    metadata("[[\"rocm_root\", \"" + sdk.string() + "\"], [\"rocm_version\", \"7.2.4\"]]");
    assert(!sv::sdk_refusal(exe, {}));                                                                   // the recorded SDK, nothing set
    assert(!sv::sdk_refusal(exe, {sdk.string(), sdk.string(), (sdk / "lib").string()}));                 // the environment agrees
    assert(!sv::sdk_refusal(exe, {sdk.string() + "/", "", "/usr/lib/x86_64-linux-gnu:" + (sdk / "lib").string()}));
    auto refused = [&](const sv::SdkEnvironment& env, const std::string& needle) {
        const auto r = sv::sdk_refusal(exe, env);
        return r && r->find(needle) != std::string::npos;
    };
    assert(refused({other.string(), "", ""}, "ROCM_PATH="));                                             // another SDK by ROCM_PATH
    assert(refused({"", other.string(), ""}, "HIP_PATH="));                                              // ... or HIP_PATH
    assert(refused({"", "", (other / "lib").string()}, "outside the recorded SDK"));                     // ... or its libraries on the path
    assert(refused({"", "", "/usr/lib:" + (other / "lib").string()}, "outside the recorded SDK"));
    assert(!sv::sdk_refusal(exe, {"", "", (t.root / "no-rocm-libs").string()}));                         // a directory with no ROCm runtime
    metadata("[[\"rocm_version\", \"7.2.4\"]]");
    assert(refused({}, "records no rocm_root"));
    metadata("[[\"rocm_root\", \"" + (t.root / "gone").string() + "\"]]");
    assert(refused({}, "is not installed"));
    std::ofstream(build / "bigcherry-build-metadata-radiance.json", std::ios::trunc) << "{not json";
    assert(refused({}, "cannot be read"));
    // an executable that is not inside a build has no recorded SDK and is not checked
    assert(!sv::sdk_refusal(t.root / "elsewhere" / "bc-serve", {other.string(), other.string(), (other / "lib").string()}));
    assert(!sv::sdk_refusal(build / "bin" / "bc-serve", {other.string(), "", ""}));
}

/* A native P1 measurement record (bc_serve_v2_measured_inputs) consistent with the manifest `unmeasured` describes: its
 * owner map and helpers, a host pool that sums to `pool_mib` with the given margin, and live room far above the need.
 * Test-only: real records come from bc-serve --publish-v2-p1 on the cards. */
Json p1_record_for(const Json& unmeasured, int64_t pool_mib, int64_t margin_mib, int64_t reserve_mib) {
    constexpr int64_t MiB = 1048576;
    const Json& spec = unmeasured.at("spec");
    const std::string owner = spec.at("plan").at("owner_map").at("owner_map_sha256");
    Json footprints = Json::array();
    for (const Json& h : spec.at("processes").at("helpers"))
        footprints.push_back({{"role", h.at("role")}, {"bdf", h.at("gpu")}, {"lanes_bytes", MiB}, {"pointer_banks_bytes", MiB},
                              {"slot_map_bytes", MiB}, {"incoming_slot_bytes", MiB}, {"total_bytes", 4 * MiB}});
    const Json scope = {{"model_digest", std::string(64, '1')}, {"engine_build_digest", std::string(64, '2')},
                        {"helper_build_digest", std::string(64, '3')}, {"layout_digest", std::string(64, '4')},
                        {"inventory_digest", std::string(64, '5')}, {"owner_map_digest", owner},
                        {"context_target_tokens", 4096}, {"retention_target_tokens", 4096}, {"kv_bytes_per_token", 1}};
    const Json inputs = {{"host_embedding_bytes", (pool_mib - margin_mib) * MiB}, {"unowned_expert_bytes", 0},
                         {"staging_bytes", 0}, {"ngram_bytes", MiB}, {"checkpoint_bytes", 0}, {"resident_other_bytes", 0},
                         {"kv_bytes_per_token", 1}, {"context_target_tokens", 4096}, {"retention_target_tokens", 4096},
                         {"mem_available_bytes", 92160 * MiB}, {"cgroup_headroom_bytes", 92160 * MiB},
                         {"pin_available_bytes", 92160 * MiB}, {"owner_map_digest", owner},
                         {"host_pool_margin_calibration",
                          {{"key", "host_ledger_margin.mib"}, {"source", "cache"}, {"evidence", "test"}, {"mib", margin_mib}}},
                         {"helper_footprints", footprints}};
    return {{"schema_version", 1}, {"kind", "bc_serve_v2_measured_inputs"}, {"planning_key", std::string(64, '6')},
            {"provenance", "serving_test synthetic record"}, {"resource_inputs", inputs},
            {"gpu", Json::array({{{"bdf", spec.at("processes").at("engine").at("device").at("gpus")[0]},
                                  {"required_bytes", 10}, {"fixed_bytes", 1}, {"claimable_bytes", 10}}})},
            {"concurrent_sessions", 1}, {"host_reserve_bytes", reserve_mib * MiB}, {"scope", scope}};
}

void manifest_writer_step(Fixture& t) {
    namespace sv = bc::serving;
    // The container, its heat ranking and the measured plan this configuration's manifest is built from.
    const auto model = t.root / "writer-model.rad";
    bc::test::write_rad(model, 4, 16, 128, 128, {});
    const auto heat = t.root / "writer.hot-first";
    {
        std::vector<uint32_t> counts(128 * 16, 0);
        for (uint32_t l = 0; l < 4; ++l)
            for (uint32_t e = 0; e < 16; ++e) counts[l * 16 + e] = (e * 7 + l * 3) % 16 + 1;
        std::ofstream(heat, std::ios::binary).write(reinterpret_cast<const char*>(counts.data()), counts.size() * 4);
    }
    Options opts;
    opts.site = t.file("writer-site.toml", "[engine]\nmodel = \"" + model.string() + "\"\nradiance_home = \"/runtime\"\n"
                                          "[handoff]\nheat = \"" + heat.string() + "\"\nhistogram = \"/heat.live\"\n" + std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
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
        // The unmeasured manifest of the same configuration has the owner map the P1 record is measured against.
        const auto unmeasured = sv::build_manifest(in);
        in.s10.p1_record = p1_record_for(unmeasured.document, 9216, 1024,
                                         r.values.at("memory").at("host_reserve_mib").get<int64_t>());
    }
    {
        sv::MeasuredPlan measured;
        measured.plan.plan = sv::decode_engine_plan(plan, "writer plan");
        measured.plan.plan_json = plan;
        measured.plan.resident = {1536, 1024};
        measured.key_inputs = sv::plan_key_inputs(r.values, t.facts, sv::container_identity(model), std::string(64, 'b'), "");
        in.measured = std::move(measured);
    }

    const auto built = sv::build_manifest(in);
    // It is a v1 manifest a consumer accepts, and says what the resolver resolved.
    const auto back = bc::serving::v2::read_manifest(built.text, "writer test", bc::serving::v2::PlanUse::serving);
    assert(back.ok());
    const Json& d = built.document;
    assert(d.at("metadata").at("content_digest") == built.digest && built.digest.rfind("sha256:", 0) == 0);
    const Json& spec = d.at("spec");
    assert(spec.at("intent").at("layout") == "flashnext-4gpu" && spec.at("intent").at("model") == model.string());
    assert(spec.at("processes").at("helpers").size() == 3 && spec.at("processes").at("helpers")[2].at("tier") == "tail");
    assert(spec.at("processes").at("helpers")[2].at("rows_kernel") == true && spec.at("processes").at("helpers")[2].at("share") == 0);
    assert(spec.at("processes").at("engine").at("scheduler").at("tp") == r.values.at("engine").at("tp"));
    assert(spec.at("handoff").at("transport").at("max_rows") == r.values.at("handoff").at("max_rows"));
    assert(spec.at("handoff").at("kernels").at("moe_down_tile") == 1);
    assert(spec.at("resources").at("host").at("pool_mib") == r.values.at("memory").at("host_pool_mib"));
    size_t leaves = 0;
    for (const auto& [group, body] : spec.at("processes").at("helper_common").items()) leaves += body.size();
    assert(leaves == 25);   // every helper_common leaf of helper_profile.toml; balance_speeds_milli is absent while balance is off
    size_t owned = 0;
    for (const Json& h : spec.at("plan").at("owner_map").at("helpers")) owned += h.at("count").get<size_t>();
    assert(owned == 64);   // 4 layers x 16 experts: with budgets this large the helpers own them all
    assert(spec.at("plan").at("engine_memory").at("expected") == plan);
    // The checkpoint slots follow what is really free (RSC08 S3): the host pool and the memory the helpers and the engine hold
    // are counted against MemAvailable as read when the manifest is written. With the tier at its target (12288 MiB), a store
    // floor of 8 and the plan's 1 MiB consumer, the room above the tier is `left` MiB and the store takes 8 + left of its want 72.
    {
        const uint64_t reserve = r.values.at("memory").at("host_reserve_mib").get<uint64_t>();
        // the calibrated margin the ledger leaves unplanned (amendment 11) is part of the manifest's calibration and of the room
        uint64_t margin = 0;
        for (const Json& e : spec.at("calibration"))
            if (e.at("key") == "host_ledger_margin.mib") margin = e.at("value").get<uint64_t>();
        assert(margin == 1024);
        auto slots_for = [&](uint64_t helpers_mib, int64_t extra_left_mib, int64_t available_shift_mib) -> uint64_t {
            sv::ManifestInputs in3 = in;
            in3.measured->plan.resident = {helpers_mib, 1024};
            const int64_t available = int64_t(reserve + margin + 9216 + 1 + 8 + 12288 + helpers_mib + 1024) + extra_left_mib + available_shift_mib;
            in3.resolved.facts["host"]["mem_available_mib"] = available;
            const auto built3 = sv::build_manifest(in3);
            return built3.document.at("spec").at("resources").at("host").at("checkpoint_slots").get<uint64_t>();
        };
        assert(slots_for(1536, 22, 0) == 8 + 22);                 // 22 MiB above the tier: 22 more slots than the floor
        assert(slots_for(1536, 22, -10) == 8 + 12);               // a manifest written when MemAvailable is lower: fewer slots
        assert(slots_for(1536, 22, +10) == 8 + 32);               // and when it is higher: more
        // the margin is a seed with its evidence in spec.calibration, and a refit (a cache entry) replaces it: the manifest says
        // so (source cache), its digest moves, and the slots follow; the engine plan and its key do not (the margin is not part
        // of what the engine measured)
        {
            const Json* seed_entry = nullptr;
            for (const Json& e : spec.at("calibration"))
                if (e.at("key") == "host_ledger_margin.mib") seed_entry = &e;
            assert(seed_entry && seed_entry->at("source") == "seed" && !seed_entry->at("evidence").get<std::string>().empty());
            sv::ManifestInputs seeded = in;
            seeded.measured->plan.resident = {1536, 1024};
            seeded.resolved.facts["host"]["mem_available_mib"] = reserve + margin + 9216 + 1 + 8 + 12288 + 1536 + 1024 + 30;
            const auto before = sv::build_manifest(seeded);
            sv::ManifestInputs refit = seeded;
            refit.calibration_cache = t.file("margin-refit.toml",
                "schema_version = 1\n[[entry]]\nkey = \"calibration.host_ledger_margin.mib\"\nvalue = 1052\n"
                "seeded = \"2026-11-06\"\nevidence = \"a start stopped 28 MiB short at the engine's host-room check\"\n");
            const auto after = sv::build_manifest(refit);
            const Json* refit_entry = nullptr;
            for (const Json& e : after.document.at("spec").at("calibration"))
                if (e.at("key") == "host_ledger_margin.mib") refit_entry = &e;
            assert(refit_entry && refit_entry->at("source") == "cache" && refit_entry->at("value") == margin + 28);
            assert(after.digest != before.digest);
            auto slots_of = [](const sv::BuiltManifest& b) {
                return b.document.at("spec").at("resources").at("host").at("checkpoint_slots").get<uint64_t>();
            };
            assert(slots_of(before) == 8 + 30 && slots_of(after) == 8 + 2);   // 28 MiB more margin, 28 fewer slots of 1 MiB
            assert(before.document.at("spec").at("plan").at("engine_memory").at("cache_key") ==
                   after.document.at("spec").at("plan").at("engine_memory").at("cache_key"));
        }
        // helpers holding more resident memory take the same room: fewer slots at the same MemAvailable
        {
            sv::ManifestInputs in3 = in;
            in3.measured->plan.resident = {1536, 1024};
            const uint64_t available = in3.resolved.facts.at("host").at("mem_available_mib").get<uint64_t>();
            in3.resolved.facts["host"]["mem_available_mib"] = reserve + margin + 9216 + 1 + 8 + 12288 + 1536 + 1024 + 22;
            const auto base = sv::build_manifest(in3);
            in3.measured->plan.resident = {1536 + 10, 1024};
            const auto more = sv::build_manifest(in3);
            assert(more.document.at("spec").at("resources").at("host").at("checkpoint_slots").get<uint64_t>() + 10 ==
                   base.document.at("spec").at("resources").at("host").at("checkpoint_slots").get<uint64_t>());
            (void)available;
        }
        // the figures are measured, never defaulted: a plan stored without them is refused, and a cache entry written before
        // they were measured is a miss (the configuration is measured again)
        sv::ManifestInputs in4 = in;
        in4.measured->plan.resident = {0, 0};
        bool refused = false;
        try { (void)sv::build_manifest(in4); } catch (const std::exception&) { refused = true; }
        assert(refused);
    }
    // The kv_backing_floor the engine admits with is the selected calibration, carried in the manifest (never a launcher literal):
    // balanced 600000, latency 500000, throughput none (0). engine_view hands the engine exactly this.
    {
        auto floor_for = [&](const char* objective) {
            Options o = opts;
            o.sets = {std::string("service.objective=") + objective};
            const auto r2 = sv::resolve(o, t.facts);
            sv::ManifestInputs in2 = in;
            in2.resolved = r2;
            in2.measured->key_inputs = sv::plan_key_inputs(r2.values, t.facts, sv::container_identity(model), std::string(64, 'b'), "");
            const auto built2 = sv::build_manifest(in2);
            const auto m2 = bc::serving::v2::read_manifest(built2.text, "floor test", bc::serving::v2::PlanUse::serving);
            assert(m2.ok());
            const auto view = bc::serving::v2::engine_view(m2.value());
            assert(view.ok());
            return view.value().kv_backing_floor_ppm;
        };
        assert(floor_for("balanced") == 600000 && floor_for("latency") == 500000 && floor_for("throughput") == 0);
    }
    assert(spec.at("plan").at("state") == "measured" && spec.at("resources").at("host").contains("checkpoint_slots"));
    assert(!spec.at("handoff").at("placement").contains("live_policy") && !spec.contains("experiment"));

    // RHP16: a matching measured cache record derives production policy leaves; v1
    // read-back validates the digest, each leaf's provenance and the pinned contract.
    // The transaction, copy, heat-gap, margin and k figures are from policy-live-9.
    // This synthetic tiny model has synthetic additional effect/lanes/drift readings.
    {
        sv::ManifestInputs calibrated = in;
        const std::string key = sv::calibration_key(calibrated.measured->key_inputs);
        {   // The calibration key ignores the pinned engine commit (records carry across bumps) and nothing else.
            auto bumped = calibrated.measured->key_inputs;
            bumped["engine_build"] = std::string(64, 'f');
            assert(sv::calibration_key(bumped) == key && sv::plan_key(bumped) != sv::plan_key(calibrated.measured->key_inputs));
            auto other = calibrated.measured->key_inputs;
            other["layout"] = "another-layout";
            assert(sv::calibration_key(other) != key);
        }
        calibrated.calibration_cache = t.file("live-policy-record.toml",
            "schema_version = 1\n[[live_policy_record]]\n"
            "plan_cache_key = \"" + key + "\"\n"
            "layout = \"flashnext-4gpu\"\nobjective = \"balanced\"\n"
            "seeded = \"2026-11-05\"\nevidence = \"policy-live-9 timing and sampling; synthetic noise/drift/lane readings\"\n"
            "txn_p50_ns = 1770000000\ntxn_p95_ns = 2720000000\n"
            "copy_bytes_per_s = 160000000\nheat_gap_max_ns = 5040000000\nheat_watermarked = 0\n"
            "hot_set_k = 5931\nmarginal_error_ppm = 23000\n"
            "marginal_observations = 3\ndrift_half_life_observations = 59000000\n"
            "observation_rate_per_s = 10000000\n"
            "tail_share_gain_ppm = 45000\ntail_lane_service_ns_per_s = 750000000\ntail_lane_row_ns = 22222222\n"
            "lane_lost_ns = 100000000\neffect_rows_per_s = 64\n"
            "effect_noise_max_ns = 2500000\n"
            "effect_row_noise_ns = 1550000\n"
            "effect_window_noise_ladder = [[8, 80000], [16, 40000], [32, 16000], [64, 10000]]\n"
            "abort_quiescent_ns = 1500000000\n");
        const auto with_calibration = sv::build_manifest(calibrated);
        const auto& lp = with_calibration.document.at("spec").at("handoff").at("placement").at("live_policy");
        assert(lp.at("cadence_ns") == 2720000000ull+
            lp.at("effect_min_quiet_windows").get<uint64_t>()*
            ((lp.at("effect_window_samples").get<uint64_t>()*1000000000ull+63)/64));
        assert(lp.at("accept_unwatermarked_heat") == true);
        assert(lp.at("max_moved_bytes") == 435200000ull && lp.at("margin_ppm") == 23000);
        assert(lp.at("half_life_observations_milli") == 59000000000ull);
        const auto& pv = with_calibration.document.at("provenance");
        assert(pv.at("/spec/handoff/placement/live_policy/cadence_ns").at("origin") == "derived");
        assert(pv.at("/spec/handoff/placement/live_policy/margin_ppm").at("class") == "policy");
        assert(bc::serving::v2::read_manifest(with_calibration.text, "calibrated policy",
                                              bc::serving::v2::PlanUse::serving).ok());
        assert(with_calibration.digest != built.digest);
        std::ifstream cal_input(calibrated.calibration_cache);
        const std::string cal_text((std::istreambuf_iterator<char>(cal_input)),
                                   std::istreambuf_iterator<char>());
        // Capture the production diagnostic, not only the helper's return value.
        auto no_policy=[&](const std::string& file_text,const std::string& reason) {
            sv::ManifestInputs failed=calibrated;
            failed.calibration_cache=t.file("failed-live-policy.toml",file_text);
            FILE* log=std::tmpfile();
            assert(log);
            std::fflush(stderr);
            const int saved=::dup(STDERR_FILENO);
            assert(saved>=0 && ::dup2(::fileno(log),STDERR_FILENO)>=0);
            sv::BuiltManifest no_live;
            try { no_live=sv::build_manifest(failed); }
            catch(...) {
                std::fflush(stderr);
                assert(::dup2(saved,STDERR_FILENO)>=0);
                ::close(saved);
                std::fclose(log);
                throw;
            }
            std::fflush(stderr);
            assert(::dup2(saved,STDERR_FILENO)>=0);
            ::close(saved);
            std::rewind(log);
            std::string diagnostic;
            char chunk[512];
            while(const size_t count=std::fread(chunk,1,sizeof(chunk),log))
                diagnostic.append(chunk,count);
            std::fclose(log);
            const auto& placement=no_live.document.at("spec").at("handoff").at("placement");
            assert(!placement.contains("live_policy"));
            const auto& origin=no_live.document.at("provenance")
                .at("/spec/handoff/placement").at("source").get_ref<const std::string&>();
            assert(origin.find(key)!=std::string::npos && origin.find(reason)!=std::string::npos);
            assert(diagnostic.find("serving without live_policy")!=std::string::npos &&
                   diagnostic.find(key)!=std::string::npos &&
                   diagnostic.find(reason)!=std::string::npos);
            assert(bc::serving::v2::read_manifest(no_live.text,"fallback",
                bc::serving::v2::PlanUse::serving).ok());
            sv::write_manifest_file(t.root / ("fallback-" + reason + ".json"),no_live.text);
        };
        auto replace_line=[&](std::string src,const std::string& name,
                              const std::string& value) {
            const auto from=src.find(name + " = ");
            assert(from!=std::string::npos);
            const auto end=src.find('\n',from);
            assert(end!=std::string::npos);
            src.replace(from,end-from,name+" = "+value);
            return src;
        };
        const std::string calib9_ladder=
            "[[10012, 256051], [20024, 199777], [40048, 110659], "
            "[80096, 79012], [160192, 64111], [320384, 63886], "
            "[640768, 42779], [1281536, 32781], [2563072, 53553]]";
        auto noisy=replace_line(cal_text,"effect_window_noise_ladder",calib9_ladder);
        noisy=replace_line(noisy,"tail_share_gain_ppm","66249");
        noisy=replace_line(noisy,"tail_lane_row_ns","4287");
        noisy=replace_line(noisy,"effect_rows_per_s","175034");
        sv::ManifestInputs noisy_input=calibrated;
        noisy_input.calibration_cache=t.file("unresolvable-noise.toml",noisy);
        const auto gate_off=sv::build_manifest(noisy_input);
        const auto& off=gate_off.document.at("spec").at("handoff").at("placement")
            .at("live_policy");
        assert(off.at("effect_gate")==false);
        assert(off.at("margin_ppm")==23000 && off.at("max_moves")>0);
        assert(gate_off.document.at("provenance")
            .at("/spec/handoff/placement/live_policy/effect_gate").at("source")
            .get<std::string>().find("cannot resolve predicted tail effect")!=std::string::npos);
        assert(bc::serving::v2::read_manifest(gate_off.text,"unresolvable",
            bc::serving::v2::PlanUse::serving).ok());
        // Calib12-shaped: all planner measurements present, but no peer
        // reference on the tail LARGE lane and all effect fields absent.
        auto structural=cal_text;
        for (const char* name : {"effect_rows_per_s","effect_noise_max_ns",
                                 "effect_row_noise_ns","effect_window_noise_ladder"}) {
            const auto pos=structural.find(std::string(name)+" = ");
            assert(pos!=std::string::npos);
            const auto end=structural.find('\n',pos);
            assert(end!=std::string::npos);
            structural.erase(pos,end-pos+1);
        }
        structural+="absent = { effect_rows_per_s = \"tail large lane has no peer reference on this layout\", "
                    "effect_noise_max_ns = \"tail large lane has no peer reference on this layout\", "
                    "effect_row_noise_ns = \"tail large lane has no peer reference on this layout\", "
                    "effect_window_noise_ladder = \"tail large lane has no peer reference on this layout\" }\n";
        sv::ManifestInputs structural_input=calibrated;
        structural_input.calibration_cache=t.file("calib12-structural.toml",structural);
        const auto structural_manifest=sv::build_manifest(structural_input);
        const auto& structural_policy=structural_manifest.document.at("spec").at("handoff")
            .at("placement").at("live_policy");
        assert(structural_policy.at("effect_gate")==false);
        assert(structural_policy.at("max_moved_bytes")==435200000ull);
        assert(structural_policy.at("margin_ppm")==23000);
        assert(structural_manifest.document.at("provenance")
            .at("/spec/handoff/placement/live_policy/effect_gate").at("source")
            .get<std::string>().find("tail large lane has no peer reference")!=std::string::npos);
        assert(bc::serving::v2::read_manifest(structural_manifest.text,"calib12",
            bc::serving::v2::PlanUse::serving).ok());
        auto incomplete=replace_line(cal_text,"copy_bytes_per_s","1");
        // A schema-valid but unusable positive reading is not a malformed file.
        no_policy(incomplete,"measured budget buys fewer");
        auto missing=cal_text;
        const std::string copy_line="copy_bytes_per_s = 160000000\n";
        assert(missing.find(copy_line)!=std::string::npos);
        missing.erase(missing.find(copy_line),copy_line.size());
        missing+="absent = { copy_bytes_per_s = \"no DMA observation\" }\n";
        no_policy(missing,"copy_bytes_per_s");
        auto malformed=calibrated;
        malformed.calibration_cache=t.file("malformed-live-policy.toml",
                                            cal_text+"unexpected_guess = 12\n");
        try { (void)sv::build_manifest(malformed); assert(false); }
        catch(const std::runtime_error& e) {
            assert(std::string(e.what()).find("unknown key")!=std::string::npos);
        }
        malformed.calibration_cache=t.file("unparseable-live-policy.toml", cal_text+"unexpected_guess = \"unterminated\n");
        try { (void)sv::build_manifest(malformed); assert(false); }
        catch(const std::runtime_error& e) {
            assert(std::string(e.what()).find("TOML parse error")!=std::string::npos);
        }

        sv::ManifestInputs forbidden = calibrated;
        forbidden.live_policy = lp;
        forbidden.live_policy_from = "--live-policy lab-policy.json";
        try { (void)sv::build_manifest(forbidden); assert(false); }
        catch (const std::runtime_error& e) {
            assert(std::string(e.what()).find("lab-only") != std::string::npos);
        }
        // The calibration run (bc-serve --calibration-run, the launcher's --calibrate): lab profile, the explicit policy and
        // no experiment. calibration-ingest accepts exactly that manifest, and refuses the production one.
        sv::ManifestInputs calibration_run = in;
        calibration_run.profile = "lab";
        calibration_run.live_policy = lp;
        calibration_run.live_policy_from = "--live-policy policy.json";
        const auto cr = sv::build_manifest(calibration_run);
        assert(!cr.document.at("spec").contains("experiment"));
        const auto cr_path = t.root / "calibration-run-manifest.json";
        const auto prod_path = t.root / "production-manifest.json";
        sv::write_manifest_file(cr_path, cr.text);
        sv::write_manifest_file(prod_path, built.text);
        const auto cr_loaded = bc::serving::v2::load_manifest(cr_path, bc::serving::v2::PlanUse::serving);
        assert(cr_loaded.ok());
        const auto& crm = cr_loaded.value();
        Json measured = {
            {"txn_p50_ns",1770000000ull}, {"txn_p95_ns",2720000000ull}, {"copy_bytes_per_s",160000000ull},
            {"heat_gap_max_ns",5040000000ull}, {"heat_watermarked",0}, {"hot_set_k",5931}, {"marginal_error_ppm",23000},
            {"marginal_observations",3}, {"drift_half_life_observations",59000000}, {"observation_rate_per_s",10000000},
            {"tail_share_gain_ppm",45000}, {"tail_lane_service_ns_per_s",750000000},
            {"lane_lost_ns",100000000}, {"effect_rows_per_s",64},
            {"tail_lane_row_ns",22222222}, {"effect_noise_max_ns",2500000},
            {"effect_row_noise_ns",1550000},
            {"effect_window_noise_ladder",Json::array({
                Json::array({uint64_t(8),uint64_t(80000)}),
                Json::array({uint64_t(16),uint64_t(40000)}),
                Json::array({uint64_t(32),uint64_t(16000)}),
                Json::array({uint64_t(64),uint64_t(10000)})})},
            {"abort_quiescent_ns",1500000000ull}};
        {   // the v2 plan key covers the engine tuning settings (a changed TP split is another plan) and, in a world of
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
                auto c = a;   // a second rank on another card type moves the key
                for (const auto& g : c.spec.facts.gpus)
                    if (g.arch != c.spec.facts.gpus.front().arch && g.bdf != c.spec.processes.engine.device.gpus[0]) {
                        c.spec.processes.engine.device.gpus.push_back(g.bdf);
                        break;
                    }
                if (c.spec.processes.engine.device.gpus.size() == 2) assert(v2::plan_cache_key(c) != k0);
            }
        }
        const Json row = {{"plan_cache_key", bc::serving::v2::plan_cache_key(crm)}, {"layout", crm.spec.intent.layout},
            {"objective", std::string(bc::serving::v2::kObjectiveNames[static_cast<size_t>(crm.spec.intent.service.objective)])},
            {"seeded","2026-11-06"}, {"evidence","calibration-run ingest test"},
            {"measurements", measured}, {"absent", Json::object()}};
        const auto helper_log = t.root / "calibration-helper0.log";
        { std::ofstream out(helper_log); out << "RHP16_CALIBRATION_JSON " << row.dump() << "\n"; }
        const auto store = t.root / "calibration-run-store.toml";
        const std::string store_key(64, 'e');
        try { sv::ingest_live_policy_log(helper_log, store, prod_path, store_key); assert(false); }
        catch (const std::runtime_error& e) {
            assert(std::string(e.what()).find("calibration-run manifest") != std::string::npos);
        }
        assert(!fs::exists(store));
        sv::ingest_live_policy_log(helper_log, store, cr_path, store_key);
        const auto ingested = sv::read_live_policy_record(store, store_key);
        assert(ingested && sv::live_policy_record_complete(*ingested) && ingested->layout == crm.spec.intent.layout);
    }
    // The unmeasured manifest of the measuring start: owner map and settings, no plan and nothing derived from one. A serving
    // start refuses it, a measuring start reads it, and the measured manifest is the reverse.
    {
        sv::ManifestInputs u = in;
        u.measured.reset();
        const auto um = sv::build_manifest(u);
        const Json& us = um.document.at("spec");
        assert(us.at("plan").at("state") == "unmeasured" && !us.at("plan").contains("engine_memory"));
        assert(!us.at("resources").at("host").contains("checkpoint_slots") && us.at("plan").at("owner_map").at("helpers").size() == 3);
        assert(us.at("resources").at("host").at("pool_mib") == spec.at("resources").at("host").at("pool_mib"));
        assert(um.document.at("provenance").at("/spec/resources").at("inputs").size() == 2);
        assert(!bc::serving::v2::read_manifest(um.text, "unmeasured", bc::serving::v2::PlanUse::serving).ok());
        assert(bc::serving::v2::read_manifest(um.text, "unmeasured", bc::serving::v2::PlanUse::measuring).ok());
        assert(!bc::serving::v2::read_manifest(built.text, "measured", bc::serving::v2::PlanUse::measuring).ok());
        assert(sv::build_manifest(u).text == um.text);   // deterministic, like the measured one
    }
    // A live policy and a lab experiment come from operator-supplied objects, are written verbatim with their own provenance
    // record, and are read back by the shared reader; a partial policy or a production experiment is an error, never a file.
    {
        sv::ManifestInputs c = in;
        c.profile = "lab";
        c.live_policy = Json::parse(R"json({"effect_gate": true, "accept_unwatermarked_heat": false, "cadence_ns": 60000000000,
            "max_moves": 8, "max_moved_bytes": 4294967296, "min_residency_ns": 300000000000, "max_heat_age_ns": 120000000000,
            "min_samples": 1000, "abort_cooldown_ns": 30000000000, "learned_eligibility_ttl_ns": 600000000000,
            "effect_window_samples": 5000, "effect_min_quiet_windows": 3, "effect_revert_ttl_ns": 120000000000,
            "margin_ppm": 100000, "half_life_observations_milli": 4000, "effect_noise_multiplier_milli": 2000,
            "protected_layers": [0, 1]})json");
        c.live_policy_from = "--live-policy policy.json";
        const auto with_policy = sv::build_manifest(c);
        assert(with_policy.document.at("spec").at("handoff").at("placement").at("live_policy").at("max_moves") == 8);
        const Json& rec = with_policy.document.at("provenance").at("/spec/handoff/placement/live_policy");
        assert(rec.at("origin") == "cli" && rec.at("source") == "--live-policy policy.json" && rec.at("class") == "policy");
        assert(with_policy.digest != built.digest);
        sv::ManifestInputs partial = c;
        partial.live_policy.erase("margin_ppm");
        try { (void)sv::build_manifest(partial); assert(false); } catch (const std::runtime_error& e) {
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
        try { (void)sv::build_manifest(forbidden); assert(false); } catch (const std::runtime_error& e) {
            assert(std::string(e.what()).find("needs the lab profile") != std::string::npos);
        }
    }
    // RSC11 A1: the three production engine options with no default are written only when set (--set), each at its
    // registry path, and an unset one leaves no trace in the document or the resolved typed view.
    {
        const Json& base_engine = built.document.at("spec").at("processes").at("engine");
        assert(!base_engine.at("execution").contains("tp_wire") && !base_engine.at("execution").contains("no_prefix_cache"));
        assert(!base_engine.at("device").contains("headroom_mib"));
        assert(!r.typed.engine.tp_wire && !r.typed.engine.no_prefix_cache && !r.typed.engine.gpu_headroom_mib);
        Options with = opts;
        with.sets = {"engine.tp_wire=\"wht6\"", "engine.no_prefix_cache=\"on\"", "engine.gpu_headroom_mib=320"};
        const auto r2 = sv::resolve(with, t.facts);
        assert(r2.typed.engine.tp_wire == "wht6" && r2.typed.engine.no_prefix_cache && r2.typed.engine.gpu_headroom_mib == 320u);
        assert(r2.provenance.at("engine.tp_wire").at("origin") == "cli");
        sv::ManifestInputs c = in;
        c.resolved = r2;
        c.measured.reset();
        const auto set_built = sv::build_manifest(c);
        const Json& eng = set_built.document.at("spec").at("processes").at("engine");
        assert(eng.at("execution").at("tp_wire") == "wht6" && eng.at("execution").at("no_prefix_cache") == "on");
        assert(eng.at("device").at("headroom_mib") == 320);
        assert(bc::serving::v2::read_manifest(set_built.text, "options test", bc::serving::v2::PlanUse::measuring).ok());
        Options bad = opts;
        bad.sets = {"engine.tp_wire=\"lossy\""};
        try { (void)sv::resolve(bad, t.facts); assert(false); } catch (const std::runtime_error& e) {
            assert(std::string(e.what()).find("tp_wire") != std::string::npos);
        }
    }
    // The helper section follows the resolver (a layout, the site file or --set), not the table's defaults.
    {
        sv::ManifestInputs c = in;
        c.resolved.values["helper"]["combine"] = true;
        c.resolved.values["helper"]["combine_min_rows"] = 512;
        c.resolved.values["helper"]["combine_threads"] = 4;
        const Json hc = sv::build_manifest(c).document.at("spec").at("processes").at("helper_common").at("combine");
        assert(hc.at("enabled") == true && hc.at("min_rows") == 512 && hc.at("threads") == 4);
        c.resolved.values["helper"]["combine"] = false;
        assert(sv::build_manifest(c).document.at("spec").at("processes").at("helper_common").at("combine").at("enabled") == false);
    }
    // One value for the host KV tier: the ledger plans for the manifest's kv_mib, and a calibration target that disagrees
    // is refused.
    {
        sv::ManifestInputs c = in;
        c.resolved.values["memory"]["host_kv_mib"] = 8192;
        try { (void)sv::build_manifest(c); assert(false); } catch (const std::runtime_error& e) {
            assert(std::string(e.what()).find("differs from calibration host_kv_tier.target_mib") != std::string::npos);
        }
    }
    // An operator override keeps its own origin: --set and the site file are never filed under the section's default.
    {
        sv::ManifestInputs c = in;
        c.resolved.provenance["service.max_sessions"] = {{"origin", "cli"}, {"source", "--set"}, {"inputs", Json::array({"service.max_sessions=4"})}};
        c.resolved.provenance["engine.port"] = {{"origin", "site"}, {"source", "site.toml"}, {"inputs", Json::array()}};
        c.resolved.provenance["engine.prefix_resume"] = {{"origin", "derived"}, {"source", "rule"}, {"inputs", Json::array()}};
        const auto overridden = sv::build_manifest(c);
        const Json& pv = overridden.document.at("provenance");
        assert(pv.at("/spec/intent/service/max_sessions").at("origin") == "cli" && pv.at("/spec/intent/service/max_sessions").at("source") == "--set");
        assert(pv.at("/spec/intent/service/max_sessions").at("class") == pv.at("/spec/intent/service").at("class"));
        assert(pv.at("/spec/processes/engine/server/port").at("origin") == "site");
        assert(pv.at("/spec/processes/engine/server/port").at("source") == "site.toml");
        assert(!pv.contains("/spec/processes/engine/execution/prefix_resume"));   // derived keeps its section's record
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
    try { (void)sv::build_manifest(bad); assert(false); } catch (const std::runtime_error& e) {
        assert(std::string(e.what()).find("cannot open the model container") != std::string::npos);
    }
    // Written once, read-only, atomically; an existing path is never replaced.
    const auto path = t.root / "written-manifest.json";
    sv::write_manifest_file(path, built.text);
    struct stat st{};
    assert(::stat(path.c_str(), &st) == 0 && (st.st_mode & 0777) == 0444 && !fs::exists(path.string() + ".tmp"));
    assert(bc::serving::v2::load_manifest(path, bc::serving::v2::PlanUse::serving).ok());
    try { sv::write_manifest_file(path, built.text); assert(false); } catch (const std::runtime_error& e) {
        assert(std::string(e.what()).find("immutable") != std::string::npos);
    }
    // Racing writers to one path: exactly one wins, the others are refused, the winner's bytes are intact, and no
    // temporary file is left behind.
    const auto race = t.root / "raced-manifest.json";
    std::atomic<int> won{0}, refused{0};
    std::vector<std::thread> writers;
    for (int i = 0; i < 8; ++i)
        writers.emplace_back([&, i] {
            try { sv::write_manifest_file(race, "writer-" + std::to_string(i) + "\n"); ++won; }
            catch (const std::runtime_error& e) {
                if (std::string(e.what()).find("immutable") != std::string::npos) ++refused;
            }
        });
    for (auto& w : writers) w.join();
    assert(won == 1 && refused == 7);
    for (const auto& e : fs::directory_iterator(t.root))
        assert(e.path().filename().string().find("raced-manifest.json.tmp") == std::string::npos);
    std::ifstream in_race(race);
    std::string got((std::istreambuf_iterator<char>(in_race)), {});
    assert(got.rfind("writer-", 0) == 0 && got.back() == '\n');
}

// The named layouts' unmeasured manifests, digested: the layout data in layouts.toml must produce exactly these documents
// (RSC11 A2). `bc_serving_test --write-golden` rewrites golden/layout_manifests.txt (a deliberate change, reviewed against the old file); `--dump-manifests DIR` writes the documents.
void layout_manifests_golden(Fixture& t, bool write_golden, const char* dump_dir) {
    namespace sv = bc::serving;
    const auto model = t.root / "golden-model.rad";
    bc::test::write_rad(model, 4, 16, 128, 128, {});
    const auto heat = t.root / "golden.hot-first";
    {
        std::vector<uint32_t> counts(128 * 16, 0);
        for (uint32_t l = 0; l < 4; ++l)
            for (uint32_t e = 0; e < 16; ++e) counts[l * 16 + e] = (e * 7 + l * 3) % 16 + 1;
        std::ofstream(heat, std::ios::binary).write(reinterpret_cast<const char*>(counts.data()), counts.size() * 4);
    }
    std::string got;
    for (const char* name : {"flashnext-4gpu", "flashnext-2xtx", "flashnext-2xtx-combine", "flashnext-2xtx-rdna2-head",
                             "flashnext-2xtx-rdna2-tail", "flashnext-2xtx-rdna2-tail-combine", "flashnext-tp2xtx-r9700",
                             "flashnext-tp2xtx"}) {
        Options opts;
        opts.site = t.file("golden-site.toml", "[engine]\nmodel = \"" + model.string() + "\"\nradiance_home = \"/runtime\"\n"
                                              "[handoff]\nheat = \"" + heat.string() + "\"\nhistogram = \"/heat.live\"\n" + std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
        opts.sets = {std::string("layout.name=\"") + name + "\""};
        sv::ManifestInputs in;
        in.resolved = sv::resolve(opts, t.facts);
        in.instance_id = "11111111-1111-4111-8111-111111111111";
        in.producer_sha256 = std::string(64, 'a');
        in.profile = "production";
        in.s10.power_profile = "stock";
        const auto built = sv::build_manifest(in);
        if (dump_dir) write(fs::path(dump_dir) / (std::string(name) + ".json"), built.text);
        // Normalised to what the layout decides: not this run's directory, the host's free /dev/shm, or the digest they feed.
        std::string text = built.text;
        for (size_t at; (at = text.find(t.root.string())) != std::string::npos;) text.replace(at, t.root.string().size(), "<root>");
        text = std::regex_replace(text, std::regex("\"shm_available_mib\": [0-9]+"), "\"shm_available_mib\": 0");
        text = std::regex_replace(text, std::regex("\"content_digest\": \"sha256:[0-9a-f]+\""), "\"content_digest\": \"-\"");
        got += std::string(name) + " sha256:" + sv::sha256_hex(text) + "\n";
    }
    const fs::path golden = fs::path(__FILE__).parent_path() / "golden/layout_manifests.txt";
    if (write_golden) { write(golden, got); return; }
    std::ifstream in(golden);
    std::string want((std::istreambuf_iterator<char>(in)), {});
    if (want != got) throw std::runtime_error("layout manifests differ from golden/layout_manifests.txt:\n" + got);
}

// An engine-only layout (contract amendment 12): two RDNA3 ranks, no helpers, no handoff, read back by the strict reader.
void engine_only_layout(Fixture& t) {
    namespace sv = bc::serving;
    // A dense model: a container with no routed experts, which the expert reader refuses. Engine-only needs no experts.
    const auto model = t.root / "dense-model.rad";
    bc::test::write_rad(model, 0, 0, 128, 128, {});
    {
        bc::experts::Container refused;
        assert(!refused.open(model.c_str()));
    }
    Options opts;
    opts.site = t.file("dense-site.toml", "[engine]\nmodel = \"" + model.string() + "\"\nradiance_home = \"/runtime\"\n" + std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
    opts.sets = {"layout.name=\"qwen27b-2xtx\"", "service.max_sessions=6", "service.max_context=8192"};
    const auto r = sv::resolve(opts, t.facts);   // no [handoff] heat/histogram: nothing here uses them
    assert(r.typed.layout.engine_cards.size() == 2 && r.typed.layout.helper_cards.empty());
    assert(r.typed.engine.tp == 2 && r.typed.engine.tp_wire == "wht6" && r.typed.engine.kernels == "libr3,libref");
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
    assert(!built.document.at("provenance").contains("/spec/handoff/transport") && !built.document.at("provenance").contains("/spec/processes/helper_common"));
    assert(bc::serving::v2::read_manifest(built.text, "engine-only", bc::serving::v2::PlanUse::measuring).ok());
    // the sections tie to the helpers both ways, and an engine-only manifest refuses a helper-side experiment
    Json with_handoff = built.document;
    with_handoff["spec"]["handoff"] = Json::object();
    assert(!bc::serving::v2::read_manifest(with_handoff.dump(), "x", bc::serving::v2::PlanUse::measuring).ok());
    // Amendment 13: engine options with a field of their own are `section.name` keys and manifest leaves; a switch's one value is "on"
    {
        Options more = opts;
        more.sets = {"layout.name=\"qwen27b-2xtx\"", "sampling.temp=\"0.7\"", "sampling.top_k=40", "limits.max_body_mib=64",
                     "limits.no_cors=\"on\"", "tuning.p2p=\"off\"", "vision.video_fps=\"2\"", "diagnostics.debug_placement=\"on\"",
                     "kernels.moe_native=true", "kernels.dec11=false"};
        sv::ManifestInputs mi = in;
        mi.resolved = sv::resolve(more, t.facts);
        const auto with_options = sv::build_manifest(mi);
        const Json& e = with_options.document.at("spec").at("processes").at("engine");
        assert(e.at("sampling").at("temp") == "0.7" && e.at("sampling").at("top_k") == 40);
        assert(e.at("limits").at("max_body_mib") == 64 && e.at("limits").at("no_cors") == "on");
        assert(e.at("tuning").at("p2p") == "off" && e.at("vision").at("video_fps") == "2" && e.at("diagnostics").at("debug_placement") == "on");
        assert(e.at("kernels") == Json({{"moe_native", true}, {"dec11", false}}));   // amendment 14: environment settings are keys too
        assert(!e.contains("lora") && !e.contains("chat"));   // a section appears only with its members
        assert(bc::serving::v2::read_manifest(with_options.text, "options", bc::serving::v2::PlanUse::measuring).ok());
        assert(bc::serving::engine_option_fields().size() == 93 /* 84 flags (amendment 18: +3, amendment 20: +1) + 9 environment settings */);
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
    helpers.site = t.file("helper-required-site.toml",
                          "[engine]\nmodel = \"" + (t.root / "model.rad").string() +
                          "\"\nradiance_home = \"/runtime\"\n" + std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
    helpers.sets = {"layout.name=\"flashnext-2xtx\""};
    fail(helpers, t.facts, "missing required key: handoff.heat");
    // a two-rank engine without its transport is refused
    Options no_wire = opts;
    no_wire.sets = {"layout.name=\"qwen27b-2xtx\"", "engine.tp_wire=\"exact\""};
    (void)sv::resolve(no_wire, t.facts);
}

// RAD49: a tensor-parallel engine (the two RDNA3 cards) with the one remote-MoE helper the mailbox protocol allows behind it.
// The expert budget bc-serve derives for an omitted budget-mib (RAD49): the card less its archs.toml reserve less the helper's own
// lanes, pointer banks, slot map and incoming slot (worker_footprint.h, which the worker checks its lanes against), for this
// model and this helper's lanes (one small class of handoff.small_rows, the large lane of handoff.max_rows, power save on).
uint64_t derived_budget_mib(const fs::path& model, uint64_t card_mib, uint64_t reserve_mib, uint32_t helpers, uint32_t index,
                            uint32_t world, const r3::worker_footprint::SmallClasses& classes, uint32_t max_rows, bool power_save) {
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
    const fp::Shape shape{c.layers, cs.experts, cs.hidden, cs.intermediate, cs.top_k ? cs.top_k : r3::moe_shm::kTopK,
                          r3::moe_shm::kActGroup};
    const uint64_t fixed = fp::fixed_bytes(shape, lanes, fp::stride_bound(largest));
    const uint64_t budget = fp::expert_budget_bytes(card_mib << 20, reserve_mib << 20, fixed) >> 20;
    assert(budget > 0 && budget < card_mib - reserve_mib);   /* strictly less than the old whole-card derivation */
    return budget;
}

/* the schema defaults: helper_profile.toml's empty small-lane list (one class of handoff.small_rows 8), max_rows 4096, power save */
uint64_t derived_budget_mib(const fs::path& model, uint64_t card_mib, uint64_t reserve_mib, uint32_t helpers, uint32_t index,
                            uint32_t world) {
    return derived_budget_mib(model, card_mib, reserve_mib, helpers, index, world,
                              r3::worker_footprint::small_classes({}, 8, 4096), 4096, true);
}

/* PR #771 review: the budget bc-serve derives is the one the worker's own settings imply. For both power-save modes the
 * manifest's helper_common carries the resolved value, and each helper's settings as the worker reads them (v1 helper settings:
 * its small lanes, hand-off rows, index, engine width, power save) give, through the worker's functions (small_classes,
 * lanes_for), exactly the budget the manifest declares. An experiment that patches a helper's lanes is refused while the
 * budgets are derived. */
void derived_budget_is_the_workers(Fixture& t) {
    namespace sv = bc::serving;
    const auto model = t.root / "derived-budget-model.rad";
    bc::test::write_rad(model, 4, 16, 128, 128, {});
    const auto heat = t.root / "derived-budget.hot-first";
    {
        std::vector<uint32_t> counts(128 * 16, 1);
        std::ofstream(heat, std::ios::binary).write(reinterpret_cast<const char*>(counts.data()), counts.size() * 4);
    }
    uint64_t budgets_on[2] = {}, budgets_off[2] = {};
    for (const bool power_save : {true, false}) {
        Options opts;
        opts.site = t.file("derived-budget-site.toml", "[engine]\nmodel = \"" + model.string() + "\"\nradiance_home = \"/runtime\"\n"
                                                        "[handoff]\nheat = \"" + heat.string() + "\"\nhistogram = \"/heat.live\"\n" + std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
        opts.sets = {"layout.name=\"flashnext-tp2xtx\"", std::string("helper.power_save=") + (power_save ? "true" : "false")};
        sv::ManifestInputs in;
        in.resolved = sv::resolve(opts, t.facts);
        in.instance_id = "11111111-1111-4111-8111-111111111111";
        in.producer_sha256 = std::string(64, 'a');
        in.profile = "lab";
        in.s10.power_profile = "stock";
        const auto built = sv::build_manifest(in);
        assert(built.document.at("spec").at("processes").at("helper_common").at("runtime").at("power_save") == power_save);
        auto read = bc::serving::v2::read_manifest(built.text, "derived-budget", bc::serving::v2::PlanUse::measuring);
        assert(read.ok());
        const uint64_t card[2] = {32624, 16368}, reserve[2] = {2048, 1024};
        for (uint32_t i = 0; i < 2; ++i) {
            auto s = bc::serving::helper_settings(read.value(), "helper" + std::to_string(i));
            assert(s.ok());
            const auto& hs = s.value();
            assert(hs.helper.power_save == power_save);   /* the worker runs the mode the budget was sized for */
            const auto classes = r3::worker_footprint::small_classes(hs.small_lanes, hs.handoff.small_rows, hs.handoff.max_rows);
            const uint64_t want = derived_budget_mib(model, card[i], reserve[i], hs.helper.count, hs.helper.index,
                                                     hs.helper.engine_ranks, classes, hs.handoff.max_rows, hs.helper.power_save);
            assert(hs.budget_mib == want);
            (power_save ? budgets_on : budgets_off)[i] = hs.budget_mib;
        }
        if (!power_save) {   /* an experiment patching a helper's lanes while the budgets are derived is refused */
            sv::ManifestInputs patched = in;
            patched.experiment = Json{{"shape", {{"helpers", Json::array({Json{{"role", "helper1"},
                                        {"settings", {{"lanes", {{"small_lanes", Json::array({4, 8})}}}}}}})}}}};
            patched.experiment_from = "test";
            bool refused = false;
            try { (void)sv::build_manifest(patched); } catch (const std::exception& e) {
                refused = std::string(e.what()).find("budget-mib") != std::string::npos;
                if (!refused) std::fprintf(stderr, "derived_budget_is_the_workers: refused for another reason: %s\n", e.what());
            }
            assert(refused);
        }
    }
    /* power save off opens every protocol lane: more lanes, smaller budgets */
    assert(budgets_off[0] < budgets_on[0] && budgets_off[1] < budgets_on[1]);
}

/* A world of two card types: engine groups in rank order, one plugin each, and every rank's card in the plan key. */
void mixed_arch_engine_layout(Fixture& t) {
    namespace sv = bc::serving;
    const auto model = t.root / "mixed-model.rad";
    bc::test::write_rad(model, 0, 0, 128, 128, {});
    Options opts;
    opts.site = t.file("mixed-site.toml", "[engine]\nmodel = \"" + model.string() + "\"\nradiance_home = \"/runtime\"\n");
    opts.layouts = {t.file("mixed-layouts.toml",
        "schema = 1\n[layout.mixed-tp3]\nmodel = \"m.rad\"\n"
        "engine = [{ arch = \"gfx1100\", count = 2 }, { arch = \"gfx1201\", count = 1 }]\ntp-wire = \"exact\"\n"
        "[layout.mixed-tp3.settings]\ntuning.tp_ownership = \"kv=1,1,2;ffn=8,8,16\"\ntuning.pipelined_prefill = \"on\"\n"
        "tuning.tune_pin = \"all_reduce:nblocks=4,nthreads=256\"\n"
        "[layout.badtuning]\nmodel = \"m.rad\"\nengine = { arch = \"gfx1201\" }\n[layout.badtuning.settings]\ntuning.no_such = 1\n"
        "[layout.twice]\nmodel = \"m.rad\"\nengine = [{ arch = \"gfx1100\" }, { arch = \"gfx1100\" }]\ntp-wire = \"exact\"\n")};
    opts.sets = {"layout.name=\"mixed-tp3\"", "service.max_sessions=4", "service.max_context=8192"};
    const auto r = sv::resolve(opts, t.facts);
    const auto& g = t.facts.at("gpus");
    assert(r.typed.layout.engine_cards.size() == 3);
    for (size_t i = 0; i < 3; ++i) assert(r.typed.layout.engine_cards[i] == g[i].at("bdf").get<std::string>());
    assert(r.typed.engine.tp == 3 && r.typed.engine.kernels == "libr3,libr4d,libref");
    // a mixed world hands the engine its ranks' cards (--rank-devices); the layout's tuning is a policy layer
    assert(r.values.at("layout").at("rank_devices") == r.values.at("layout").at("engine_cards"));
    assert(r.values.at("tuning").at("tp_ownership") == "kv=1,1,2;ffn=8,8,16");
    assert(r.values.at("tuning").at("pipelined_prefill") == "on");
    assert(r.values.at("tuning").at("tune_pin") == "all_reduce:nblocks=4,nthreads=256");
    assert(r.provenance.at("tuning.tp_ownership").at("source").get<std::string>().find("layout mixed-tp3") == 0);
    {
        Options over = opts;   // a profile (or --set) overrides the layout's tuning
        over.sets.push_back("tuning.pipelined_prefill=\"off\"");
        assert(sv::resolve(over, t.facts).values.at("tuning").at("pipelined_prefill") == "off");
        over.sets = {"layout.name=\"badtuning\""};
        fail(over, t.facts, "unknown key tuning.no_such");
        over.layouts = {t.file("oldenv.toml", "schema = 1\n[layout.oldenv]\nmodel = \"m.rad\"\nengine = { arch = \"gfx1201\" }\n"
                                              "[layout.oldenv.env]\nR3_MOE_WORKER_COMBINE = \"1\"\n")};
        over.sets = {"layout.name=\"oldenv\""};
        fail(over, t.facts, "env is not a layout key");
    }
    const Json key = sv::plan_key_inputs(r.values, t.facts, "m", "e", "");
    assert(key.contains("rank_cards") && key.at("rank_cards").size() == 3 && key.at("rank_cards")[2].at("arch") == "gfx1201");
    opts.sets = {"layout.name=\"twice\""};
    fail(opts, t.facts, "in two groups");
    // a single-architecture world keys as before: no rank_cards member
    Options dense = opts;
    dense.layouts.clear();
    dense.sets = {"layout.name=\"qwen27b-2xtx\"", "service.max_sessions=6", "service.max_context=8192"};
    const auto uniform = sv::resolve(dense, t.facts);
    assert(!sv::plan_key_inputs(uniform.values, t.facts, "m", "e", "").contains("rank_cards"));
    assert(!uniform.values.at("layout").contains("rank_devices"));   // a uniform world needs no --rank-devices
}

void tp_engine_with_helper_layout(Fixture& t) {
    namespace sv = bc::serving;
    const auto model = t.root / "tp-helper-model.rad";
    bc::test::write_rad(model, 4, 16, 128, 128, {});
    const auto heat = t.root / "tp-helper.hot-first";
    {
        std::vector<uint32_t> counts(128 * 16, 1);
        std::ofstream(heat, std::ios::binary).write(reinterpret_cast<const char*>(counts.data()), counts.size() * 4);
    }
    Options opts;
    opts.site = t.file("tp-helper-site.toml", "[engine]\nmodel = \"" + model.string() + "\"\nradiance_home = \"/runtime\"\n"
                                              "[handoff]\nheat = \"" + heat.string() + "\"\nhistogram = \"/heat.live\"\n" + std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
    opts.sets = {"layout.name=\"flashnext-tp2xtx-r9700\"", "service.max_sessions=6", "service.max_context=65536"};
    const auto r = sv::resolve(opts, t.facts);
    assert(r.typed.layout.engine_cards.size() == 2 && r.typed.layout.helper_cards.size() == 1);
    assert(r.typed.engine.tp == 2 && r.typed.engine.tp_wire == "exact");
    assert(r.typed.engine.kernels == "libr3,libr3moe,libref");
    assert(r.typed.helper.count == 1 && r.typed.helper.engine_ranks == 2);
    // no budget-mib in the layout: the helper's card (the fixture's R9700, 32624 MiB) less its archs.toml reserve, less its own
    // lanes and tables (one slot a TP rank: both lanes of both ranks), the planner's and the worker's one derivation
    assert(r.values.at("memory").at("helper_budgets_mib") == Json::array({derived_budget_mib(model, 32624, 2048, 1, 0, 2)}));
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

// RAD49 steps 3-4: two helpers behind a TP2 engine. The budgets are each card less its archs.toml reserve (no budget-mib), the
// RX 6900 XT is the tail, the combine is on, and the manifest carries both helpers behind the two ranks.
void tp_engine_with_two_helpers_layout(Fixture& t) {
    namespace sv = bc::serving;
    const auto model = t.root / "tp-two-helper-model.rad";
    bc::test::write_rad(model, 4, 16, 128, 128, {});
    const auto heat = t.root / "tp-two-helper.hot-first";
    {
        std::vector<uint32_t> counts(128 * 16, 1);
        std::ofstream(heat, std::ios::binary).write(reinterpret_cast<const char*>(counts.data()), counts.size() * 4);
    }
    Options opts;
    opts.site = t.file("tp-two-helper-site.toml", "[engine]\nmodel = \"" + model.string() + "\"\nradiance_home = \"/runtime\"\n"
                                                  "[handoff]\nheat = \"" + heat.string() + "\"\nhistogram = \"/heat.live\"\n" + std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
    opts.sets = {"layout.name=\"flashnext-tp2xtx\""};
    const auto r = sv::resolve(opts, t.facts);
    assert(r.typed.layout.engine_cards.size() == 2 && r.typed.layout.helper_cards.size() == 2);
    assert(r.typed.engine.tp == 2 && r.typed.engine.tp_wire == "exact");
    assert(r.typed.helper.count == 2 && r.typed.helper.engine_ranks == 2 && r.typed.helper.combine);
    // no budget-mib: the R9700 (32624 MiB) less 2048, the RX 6900 XT (16368 MiB) less 1024; the tail is the last one
    const uint64_t head = derived_budget_mib(model, 32624, 2048, 2, 0, 2), tail = derived_budget_mib(model, 16368, 1024, 2, 1, 2);
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
    // the manifest's helper budgets are the derived expert budgets the owner map was planned with (build_manifest checks the map
    // against them with verify_owner_map), and the ones the worker sizes its arena from
    assert(spec.at("processes").at("helpers")[0].at("vram_budget_mib") == head &&
           spec.at("processes").at("helpers")[1].at("vram_budget_mib") == tail);
    assert(bc::serving::v2::read_manifest(built.text, "tp-two-helper", bc::serving::v2::PlanUse::measuring).ok());
    // a helper count other than the cards the layout selects is refused (the slot bound, helpers x ranks <= kMaxSlots, is
    // checked on the same path; moe_remote_shm_test covers slots_fit)
    Options three = opts;
    three.sets = {"layout.name=\"flashnext-tp2xtx\"", "helper.count=3"};
    bool refused = false;
    try { (void)sv::resolve(three, t.facts); } catch (const std::exception&) { refused = true; }
    assert(refused);
}

/* RSC12: a config resolves exactly as the layout it replaces (every effective value; only the provenance names the config
 * file), a variant changes only what it states, and the loader refuses what it does not know. */
void config_files(Fixture& t) {
    namespace sv = bc::serving;
    const char* env = std::getenv("BIGCHERRY_LAYOUTS");
    if (!env || !fs::is_directory(env)) throw std::runtime_error("config_files: BIGCHERRY_LAYOUTS must name the layouts directory");
    const fs::path configs = fs::path(env).parent_path() / "configs";
    const auto model = t.root / "config-model.rad";
    bc::test::write_rad(model, 0, 0, 128, 128, {});
    const auto site = t.file("config-site.toml", "[engine]\nmodel = \"" + model.string() + "\"\nradiance_home = \"/runtime\"\n");
    // the layout as the launcher resolves it: its server-args become --set (tools/lab/radiance/lib/plan-check.sh)
    Options layout;
    layout.site = site;
    layout.sets = {"layout.name=\"qwen27b-tp3-int8\"", "service.max_context=8192", "service.max_sessions=8",
                   "engine.kv_cache_dtype=\"fp8\"", "engine.num_speculative_tokens=3"};
    const auto from_layout = sv::resolve(layout, t.facts);
    Options config;
    config.site = site;
    config.configs = {configs};
    config.model_root = t.root / "model-root";   // the site's model wins over the config's (config < site)
    config.sets = {"layout.name=\"qwen27b-tp3-int8\""};
    const auto from_config = sv::resolve(config, t.facts);
    if (from_layout.values != from_config.values) {
        std::fprintf(stderr, "layout: %s\nconfig: %s\n", from_layout.values.dump().c_str(), from_config.values.dump().c_str());
        assert(false);
    }
    assert(from_config.provenance.at("tuning.tp_ownership").at("source").get<std::string>().find("config qwen27b.toml") !=
           std::string::npos);
    assert(from_config.provenance.at("service.max_context").at("source").get<std::string>().find("config qwen27b.toml") !=
           std::string::npos);
    // the variant: only the split differs
    config.sets = {"layout.name=\"qwen27b-tp3-int8-556\""};
    const auto variant = sv::resolve(config, t.facts);
    Json a = from_config.values, b = variant.values;
    assert(b.at("tuning").at("tp_ownership") == "kv=1,1,2;ffn=5632,5632,6144;lk=5,5,6;lv=15,15,18");
    a["tuning"].erase("tp_ownership"); b["tuning"].erase("tp_ownership");
    a["layout"].erase("name"); b["layout"].erase("name");
    assert(a == b);
    {   // where each value came from: the variant's own key, and the base's for the rest
        const auto set = sv::load_configs(sv::config_sources({configs}));
        const Json& o = set.origin.at("qwen27b-tp3-int8-556");
        assert(o.at("tuning.tp_ownership") == "qwen27b-tp3-int8-556");
        assert(o.at("tuning.pipelined_prefill") == "qwen27b-tp3-int8" && o.at("ranks") == "qwen27b-tp3-int8");
        assert(set.chain.at("qwen27b-tp3-int8-556") == Json::array({"qwen27b-tp3-int8-556", "qwen27b-tp3-int8"}));
    }
    // a profile and --set still override a config (config < site < profile < --set)
    config.sets.push_back("tuning.pipelined_prefill=\"off\"");
    assert(sv::resolve(config, t.facts).values.at("tuning").at("pipelined_prefill") == "off");

    const std::string head = "schema = 1\n[config.base]\ndescription = \"b\"\nmodel = \"m.rad\"\n"
                             "ranks = [{ arch = \"gfx1201\" }]\n[config.base.service]\nmax_context = 8192\nmax_sessions = 4\n";
    auto refuses = [&](const std::string& name, const std::string& text, const std::string& needle) {
        const auto f = t.file(name.c_str(), text);
        try { (void)sv::load_configs({f}); }
        catch (const std::exception& e) {
            if (std::string(e.what()).find(needle) == std::string::npos) {
                std::fprintf(stderr, "config refusal %s: wanted '%s', got: %s\n", name.c_str(), needle.c_str(), e.what());
                assert(false);
            }
            return;
        }
        std::fprintf(stderr, "config %s: not refused\n", name.c_str());
        assert(false);
    };
    refuses("c-unknown.toml", head + "[config.v]\nbase = \"base\"\ndescription = \"v\"\ncontext = 1\n", "unknown key context");
    refuses("c-flat-setting.toml", head + "[config.v]\nbase = \"base\"\ndescription = \"v\"\nmax_context = 1\n",
            "unknown key max_context");
    refuses("c-server-args.toml", head + "[config.v]\nbase = \"base\"\ndescription = \"v\"\nserver-args = [\"--max-model-len\"]\n",
            "unknown key server-args");
    refuses("c-nobase.toml", head + "[config.v]\nbase = \"gone\"\ndescription = \"v\"\n", "base gone is not a defined config");
    refuses("c-cycle.toml", head + "[config.x]\nbase = \"y\"\ndescription = \"x\"\n[config.y]\nbase = \"x\"\ndescription = \"y\"\n",
            "a cycle");
    refuses("c-nodesc.toml", head + "[config.v]\nbase = \"base\"\n", "description is required");
    refuses("c-nomodel.toml", "schema = 1\n[config.v]\ndescription = \"v\"\nranks = [{ arch = \"gfx1201\" }]\n",
            "model is required");
    refuses("c-rankkey.toml", "schema = 1\n[config.v]\ndescription = \"v\"\nmodel = \"m\"\nranks = [{ arch = \"gfx1201\", devices = [0] }]\n",
            "ranks: unknown key devices");
    refuses("c-schema.toml", "[config.v]\ndescription = \"v\"\n", "schema = 1 is required");
    {   // a name defined in two files
        const auto f1 = t.file("c-dup1.toml", head);
        const auto f2 = t.file("c-dup2.toml", head);
        bool refused = false;
        try { (void)sv::load_configs({f1, f2}); } catch (const std::exception& e) {
            refused = std::string(e.what()).find("defined in both") != std::string::npos;
        }
        assert(refused);
    }
    {   // tables merge key by key; a list replaces
        const auto f = t.file("c-merge.toml", head +
            "[config.v]\nbase = \"base\"\ndescription = \"v\"\nranks = [{ arch = \"gfx1100\", count = 2 }]\ntp_wire = \"exact\"\n"
            "[config.v.service]\nmax_sessions = 2\n");
        const auto set = sv::load_configs({f});
        const Json& v = set.config.at("v");
        assert(v.at("service").at("max_context") == 8192 && v.at("service").at("max_sessions") == 2);
        assert(v.at("ranks").size() == 1 && v.at("ranks")[0].at("arch") == "gfx1100");
        assert(set.origin.at("v").at("service.max_context") == "base" && set.origin.at("v").at("service.max_sessions") == "v");
        const Json l = sv::config_as_layout(v);
        assert(l.at("engine") == v.at("ranks") && l.at("tp-wire") == "exact" && !l.contains("server-args"));
        assert(l.at("settings").at("service").at("max_sessions") == 2);
    }
    {   // the config's model: under the model root, the lowest engine.model; a site's model wins; relative with no root
        // is refused when that config is selected
        const auto root = t.root / "models";
        fs::create_directories(root / "m");
        bc::test::write_rad(root / "m" / "own.rad", 0, 0, 128, 128, {});
        const auto own = t.file("c-model.toml", "schema = 1\n[config.own]\ndescription = \"o\"\nmodel = \"m/own.rad\"\n"
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
        if (!std::getenv("BC_MODEL_ROOT")) fail(o, t.facts, "is relative and no model root is given");
    }
    refuses("c-placement.toml", head + "[config.v]\nbase = \"base\"\ndescription = \"v\"\n"
            "[[config.v.helpers]]\narch = \"gfx1100\"\ncount = 1\nplacement = \"sideways\"\n", "placement is required");
    refuses("c-helper-count.toml", head + "[config.v]\nbase = \"base\"\ndescription = \"v\"\n"
            "[[config.v.helpers]]\narch = \"gfx1100\"\nplacement = \"head\"\n", "count is required");
    refuses("c-budget.toml", head + "[config.v]\nbase = \"base\"\ndescription = \"v\"\n"
            "[[config.v.helpers]]\narch = \"gfx1100\"\ncount = 1\nplacement = \"head\"\nbudget_mib = 1.5\n", "budget_mib");
    refuses("c-frac.toml", "schema = 1\n[config.v]\ndescription = \"v\"\nmodel = \"m\"\nranks = [{ arch = \"gfx1100\", count = 2.5 }]\n",
            "count is a whole number");
    refuses("c-bool.toml", "schema = 1\n[config.v]\ndescription = \"v\"\nmodel = \"m\"\nranks = [{ arch = \"gfx1100\", count = true }]\n",
            "count is a whole number");
    refuses("c-schema-float.toml", "schema = 1.0\n", "schema = 1 is required");
    refuses("c-date.toml", head + "[config.base.tuning]\nwhen = 2026-11-14\n", "unsupported TOML value");
    refuses("c-config-scalar.toml", "schema = 1\nconfig = 3\n", "config is a table");
    refuses("c-engine-model.toml", head + "[config.base.engine]\nmodel = \"x\"\n", "give the model as `model`");
    {   // dotted keys are tables: `tuning.x = ...` in the config table is the tuning table; three levels of bases;
        // a scalar replaced by a table keeps no stale origin
        const auto f = t.file("c-dotted.toml", head +
            "[config.v1]\nbase = \"base\"\ndescription = \"1\"\ntuning.pipelined_prefill = \"on\"\n"
            "[config.v2]\nbase = \"v1\"\ndescription = \"2\"\n[config.v2.service]\nmax_sessions = 2\n"
            "[config.v3]\nbase = \"v2\"\ndescription = \"3\"\ntuning.tp_ownership = \"kv=1\"\n");
        const auto set = sv::load_configs({f});
        const Json& v3 = set.config.at("v3");
        assert(v3.at("tuning").at("pipelined_prefill") == "on" && v3.at("tuning").at("tp_ownership") == "kv=1");
        assert(v3.at("service").at("max_sessions") == 2 && v3.at("service").at("max_context") == 8192);
        assert(set.origin.at("v3").at("tuning.pipelined_prefill") == "v1" && set.origin.at("v3").at("service.max_sessions") == "v2");
        assert(set.chain.at("v3") == Json::array({"v3", "v2", "v1", "base"}));
        const auto g = t.file("c-replace.toml", head + "[config.w]\nbase = \"base\"\ndescription = \"w\"\nmodel = { x = \"y\" }\n");
        try { (void)sv::load_configs({g}); assert(false); }   // a model that is a table is not a string
        catch (const std::exception& e) { assert(std::string(e.what()).find("model is a string") != std::string::npos); }
    }
    {   // --configs wins over BIGCHERRY_CONFIGS
        const auto f = t.file("c-given.toml", head);
        ::setenv("BIGCHERRY_CONFIGS", (t.root / "no-such-dir").c_str(), 1);
        assert(sv::config_sources({f}) == std::vector<fs::path>{f});
        bool refused = false;
        try { (void)sv::config_sources({}); } catch (const std::exception&) { refused = true; }
        assert(refused);
        ::unsetenv("BIGCHERRY_CONFIGS");
        assert(sv::config_sources({}).empty());
    }
    {   // the resolver checks the settings: an unknown tuning key is refused, naming the config file
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

int main(int argc, char** argv) {
    bool write_golden = false;
    const char* dump_dir = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--write-golden") write_golden = true;
        else if (std::string(argv[i]) == "--dump-manifests" && i + 1 < argc) dump_dir = argv[++i];
        else { std::cerr << "usage: bc_serving_test [--write-golden] [--dump-manifests DIR]\n"; return 2; }
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
        derived_budget_is_the_workers(t);
        sdk_environment_step(t);
        std::cout << "PASS: fake KFD/PCI/proc detection; strict keys; types/ranges; required; "
                     "precedence/caps; provenance/hash; launcher engine/WorkerConfig parity; "
                     "calibration store; calibrated host reserve; host ledger; measured plan cache; measuring step; manifest writer\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
