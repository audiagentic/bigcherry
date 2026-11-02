#ifdef NDEBUG
#undef NDEBUG // Assertions are test requirements even in RelWithDebInfo.
#endif
#include "serving.h"
#include "calibration.h"
#include "ledger.h"
#include "plan_cache.h"
#include "test_rad_fixture.h"
#include "sha256.h"

#undef NDEBUG   /* the checks below are assert()s: keep them live in Release component builds */
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <fstream>
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
    {   // the standard profile's 8 combine threads on a 4-CPU host
        Json few = t.facts;
        few["host"]["cpu_count"] = 4;
        fail(t.standard(), few, "helper.combine_threads exceeds discovered CPU count");
    }
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
    opts.site = t.file("floor-site.toml", "[engine]\nmodel = \"/model.rad\"\nradiance_home = \"/runtime\"\n"
                                          "[handoff]\nheat = \"/heat.hot-first\"\nhistogram = \"/heat.live\"\n"
                                          "[hard_floors.memory]\nhost_reserve_mib = 8192\n");
    r = bc::serving::resolve(opts, t.facts);
    assert(r.values.at("memory").at("host_reserve_mib") == 8192);
    assert(r.provenance.at("memory.host_reserve_mib").at("origin") == "site");
    opts.site = t.file("low-floor-site.toml", "[engine]\nmodel = \"/model.rad\"\nradiance_home = \"/runtime\"\n"
                                              "[handoff]\nheat = \"/heat.hot-first\"\nhistogram = \"/heat.live\"\n"
                                              "[hard_floors.memory]\nhost_reserve_mib = 1024\n");
    r = bc::serving::resolve(opts, t.facts);
    assert(r.values.at("memory").at("host_reserve_mib") == 6554);
    opts.site = t.file("bad-floor-site.toml", "[engine]\nmodel = \"/model.rad\"\nradiance_home = \"/runtime\"\n"
                                              "[handoff]\nheat = \"/heat.hot-first\"\nhistogram = \"/heat.live\"\n"
                                              "[hard_floors.memory]\nhost_pool_mib = 1024\n");
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
    assert(parsed.host.size() == 1 && parsed.host[0].residence == bc::serving::v1::Residence::kPinned);
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
    bc::test::write_rad(model, 2, 8, 128, 128);
    const std::string id = sv::container_identity(model);
    assert(id.size() == 64 && sv::container_identity(model) == id);
    bc::test::write_rad(model, 2, 9, 128, 128);
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
            {"engine.ngram_placement", "disk"}, {"engine.prefix_resume", "off"}, {"engine.tp", "2"}};
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
    const auto stored = sv::plan_cache_store(dir, k, plan);
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
    // The hardware gate's one-byte drift edits the cache entry's own plan: the next lookup hands that plan to the engine as the
    // expected one (bc-serve rewrites <key>.expected-plan.json from the entry at every launch, so editing that file is moot).
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
        assert(drifted && drifted->plan.host.at(0).bytes == 101);              // the entry validates; the CHECK is what refuses it
        const auto args2 = sv::plan_check_args(dir, *drifted, 600000);
        std::ifstream handed(args2[1]);
        assert(sv::decode_engine_plan(Json::parse(handed), "test").host.at(0).bytes == 101);   // the engine is handed the drifted plan
        std::ofstream(file, std::ios::trunc) << original;                      // restore for the tests below
    }
    // The engine's check flags: the cached plan as a v1 file, and the floor the calibration selects (never a literal).
    {
        const auto args = sv::plan_check_args(dir, *found, 600000);
        assert(args.size() == 4 && args[0] == "--plan-check" && args[2] == "--plan-check-floor-ppm" && args[3] == "600000");
        assert(args[1] == (dir / (stored.key + ".expected-plan.json")).string());
        std::ifstream written(args[1]);
        const auto round = sv::decode_engine_plan(Json::parse(written), args[1]);
        assert(round.vram.kv_resolved == 9 && round.host_pool_mib == 9216);
        auto bad_ppm = [&](int64_t n) {
            try { (void)sv::plan_check_args(dir, *found, n); } catch (const std::exception& e) {
                return std::string(e.what()).find("must be in [0, 1000000)") != std::string::npos;
            }
            return false;
        };
        assert(bad_ppm(-1) && bad_ppm(1000000));
        // the floor is the selected calibration: balanced 600000, latency 500000, throughput none (0)
        auto floor_for = [&](const char* objective) {
            auto opts = t.standard();
            opts.sets = {std::string("service.objective=") + objective};
            return sv::kv_backing_floor_ppm(sv::resolve(opts, t.facts).values, {});
        };
        assert(floor_for("balanced") == 600000 && floor_for("latency") == 500000 && floor_for("throughput") == 0);
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
    try { (void)sv::plan_cache_store(dir, k, bad); } catch (const std::exception& e) {
        refused = std::string(e.what()).find("vram: missing required member") != std::string::npos;
    }
    assert(refused);
}

void measuring_step(Fixture& t) {
    namespace sv = bc::serving;
    const Json plan = Json::parse(R"json({
      "plan_version": 1, "producer": "radiance --plan-memory (rad0011)", "host": [],
      "checkpoints": {"slot_bytes": 1048576, "floor_slots": 8, "want_slots": 72, "from": "engine plan (test)"},
      "kv_one_session": {"bytes": 4096, "from": "engine plan (test)"},
      "vram": {"claimable": 10, "statics": 1, "elastic": 9, "experts": 0, "kv": 9, "stage": 0, "kv_resolved": 9, "expert_floor": 0,
               "kv_ceiling": 9, "expert_ceiling": 0, "from": "engine plan (test)",
               "cards": [{"capacity": 10, "free": 10, "engine_expert_plane": 0, "expert_unit_max": 1}]},
      "host_pool_mib": 9216})json");
    // A stand-in engine: checks it sees only the engine card and its arguments, then writes the plan where asked.
    const auto engine = t.file("fake-engine/radiance",
        "#!/bin/sh\n"
        "[ \"$ROCR_VISIBLE_DEVICES\" = 2 ] && [ \"$HIP_VISIBLE_DEVICES\" = 0 ] || exit 7\n"
        "[ \"$1\" = --manifest ] && [ \"$3\" = --role ] && [ \"$4\" = engine ] && [ \"$5\" = --plan-memory ] || exit 8\n"
        "cat > \"$6\" <<'PLAN'\n" + plan.dump() + "\nPLAN\n");
    fs::permissions(engine, fs::perms::owner_all);
    const auto manifest = t.file("fake-engine/manifest.json", "{}");
    double wall = -1;
    const Json got = sv::measure_engine_plan(engine, manifest, 2, t.root / "fake-engine/plan.json", &wall);
    assert(got == plan && wall >= 0);
    // The device selection is the engine card's, whatever the caller's environment had.
    setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3", 1);
    assert(sv::measure_engine_plan(engine, manifest, 2, t.root / "fake-engine/plan.json", &wall) == plan);
    unsetenv("ROCR_VISIBLE_DEVICES");
    auto refused = [&](uint32_t ordinal, const std::string& needle) {
        try { (void)sv::measure_engine_plan(engine, manifest, ordinal, t.root / "fake-engine/plan.json", &wall); }
        catch (const std::exception& e) {
            if (std::string(e.what()).find(needle) != std::string::npos) return;
            throw std::runtime_error("wrong measuring diagnostic for " + needle + ": " + e.what());
        }
        throw std::runtime_error("expected measuring refusal containing " + needle);
    };
    refused(1, "exited 7");
    const auto bad = t.file("fake-engine/bad", "#!/bin/sh\necho '{\"plan_version\": 1}' > \"$6\"\n");
    fs::permissions(bad, fs::perms::owner_all);
    bool strict = false;
    try { (void)sv::measure_engine_plan(bad, manifest, 0, t.root / "fake-engine/plan.json", &wall); }
    catch (const std::exception& e) { strict = std::string(e.what()).find("missing") != std::string::npos; }
    assert(strict);
    // What it measured goes into the cache under its key.
    const Json key = {{"cache_version", 1}, {"model", "m"}, {"engine_build", "e"}, {"card", {{"arch", "gfx1201"}}},
                      {"layout", "flashnext-4gpu"}, {"engine", Json::object()}};
    sv::plan_cache_store(t.root / "fake-engine/cache", key, got);
    assert(sv::plan_cache_lookup(t.root / "fake-engine/cache", key)->plan.checkpoints.want_slots == 72);
}

int main() {
    try {
        sha256_vectors();
        Fixture t;
        detection(t);
        detection_limits(t);
        unknown_keys(t);
        type_range_required(t);
        precedence_and_caps(t);
        provenance_hash_and_parity(t);
        calibration_store(t);
        host_reserve(t);
        host_ledger_plan();
        measured_plan_cache(t);
        measuring_step(t);
        std::cout << "PASS: fake KFD/PCI/proc detection; strict keys; types/ranges; required; "
                     "precedence/caps; provenance/hash; launcher engine/WorkerConfig parity; "
                     "calibration store; calibrated host reserve; host ledger; measured plan cache; measuring step\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
