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
#include "test_config_fixture.h"
#include "runtime_guards.h"
#include "s8_replay.h"
#include "sdk_check.h"
#include "serving.h"
#include "lab_experiment.h"
#include "session.h"
#include "sha256.h"
#include "serving_codec.h"
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
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/mman.h>
#include <sys/resource.h>
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
#include <future>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sched.h>
#include <thread>
#include <vector>

namespace {
// An engine-free resolver test names a config; production gets it from its YAML through the inspector, so the test hands
// the resolver the matching preset (test_config_fixture.h: one per shipped engines/radiance/configs/NAME.yaml).
bc::serving::Resolved tresolve(bc::serving::Options options, const nlohmann::json& facts,
                               const bc::serving::NativeEngineSettings* native = nullptr) {
    if (!options.inspected)
        if (const auto name = bc::serving::layered_string(options, "layout", "name");
            name && bc::serving::fixture::has_preset(*name))
            bc::serving::fixture::use(options, *name);
    return bc::serving::resolve(options, facts, native);
}
}  // namespace

namespace {
namespace fs = std::filesystem;
using bc::serving::Json;
using bc::serving::Options;
using bc::serving::Resolved;
Json g_session_manifest_template = Json::object();

int select_free_port() {
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
}

std::string read_text(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}
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
    fs::path work_root() const { return root / "work"; }   // the platform work root a start gets as --work-root

    Fixture() : root(fs::current_path() / "bc-serving-fixture") {
        // This test only writes in its own CTest build directory.
        fs::remove_all(root);
        const auto cache_root = root / "work";
        fs::create_directories(cache_root);
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
        (void)tresolve(opts, facts);
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
    {   // cpu_count is the CPUs this process may run on (its affinity set), which is what the helper inherits
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        assert(sched_getaffinity(0, sizeof allowed, &allowed) == 0);
        assert(h.at("cpu_count") == uint64_t(CPU_COUNT(&allowed)));
        cpu_set_t one;
        CPU_ZERO(&one);
        int first = 0;
        while (first < CPU_SETSIZE && !CPU_ISSET(first, &allowed)) ++first;
        CPU_SET(first, &one);
        assert(sched_setaffinity(0, sizeof one, &one) == 0);   // pinned to one CPU: detect must report one
        const Json pinned = bc::serving::detect(root).at("host");
        assert(sched_setaffinity(0, sizeof allowed, &allowed) == 0);
        assert(pinned.at("cpu_count") == 1);
    }
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


void type_range_required(Fixture& t) {
    // (engine.model is no longer a missing-key case: a config's YAML always names its model.)
    Options opts = t.standard();
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
    fail(opts, t.facts, "no config was read");
    opts.profile = t.file("derived.toml", "[engine]\nmax_num_seqs = 99\n");
    fail(opts, t.facts, "derived and cannot be overridden");
    { // the combine thread count is derived from the discovered CPUs, never a literal the host may not have
        namespace sv = bc::serving;
        const auto threads_on = [&](uint64_t cpus, Options o) {
            Json host = t.facts;
            host["host"]["cpu_count"] = cpus;
            return tresolve(o, host);
        };
        const Options standard = t.standard();
        const auto four = threads_on(4, standard);
        assert(four.values.at("helper").at("combine_threads") == 4);   // a 4-CPU host resolves instead of refusing
        assert(four.provenance.at("helper.combine_threads").at("origin") == "derived");
        const auto twenty = threads_on(20, standard);
        assert(twenty.values.at("helper").at("combine_threads") == 8);  // the calibrated ceiling on a larger host
        assert(twenty.provenance.at("helper.combine_threads").at("origin") == "derived");
        assert(threads_on(1, standard).values.at("helper").at("combine_threads") == 1);
        // an explicit value is the operator's: it keeps its origin, and one above the CPU count is still refused
        Options explicit_set = standard;
        explicit_set.sets.push_back("helper.combine_threads=6");
        const auto six = threads_on(20, explicit_set);
        assert(six.values.at("helper").at("combine_threads") == 6);
        assert(six.provenance.at("helper.combine_threads").at("origin") != "derived");
        Json few = t.facts;
        few["host"]["cpu_count"] = 4;
        fail(explicit_set, few, "helper.combine_threads exceeds discovered CPU count");
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
    const auto r = tresolve(opts, t.facts);
    assert(r.typed.engine.port == 4900);
    assert(r.provenance.at("engine.port").at("origin") == "cli");
    opts.sets.clear();
    const auto p = tresolve(opts, t.facts);
    assert(p.typed.engine.port == 4800);
    opts.profile.clear();
    const auto s = tresolve(opts, t.facts);
    assert(s.typed.engine.port == 4700);
    opts.site = t.standard().site;
    const auto d = tresolve(opts, t.facts);
    assert(d.typed.engine.port == 36080);

    opts.site =
        t.file("cap-site.toml", "[engine]\nmodel = \"" + (t.root / "model.rad").string() +
                                    "\"\nradiance_home = \"/runtime\"\n"
                                    "[memory]\nhost_pool_mib = 7000\n"
                                    "[hard_caps.memory]\nhost_pool_mib = 8000\n"
                                    "[handoff]\nheat = \"/heat.hot-first\"\nhistogram = \"/heat.live\"\n" +
                                    std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
    auto good = tresolve(opts, t.facts);
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
    const auto r = tresolve(opts, t.facts);
    const auto again = tresolve(opts, t.facts);
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
    auto r = tresolve(t.standard(), t.facts);
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
    r = tresolve(opts, t.facts);
    assert(r.values.at("memory").at("host_reserve_mib") == 8192);
    assert(r.provenance.at("memory.host_reserve_mib").at("origin") == "site");
    opts.site =
        t.file("low-floor-site.toml", "[engine]\nmodel = \"" + (t.root / "model.rad").string() +
                                          "\"\nradiance_home = \"/runtime\"\n"
                                          "[handoff]\nheat = \"/heat.hot-first\"\nhistogram = \"/heat.live\"\n"
                                          "[hard_floors.memory]\nhost_reserve_mib = 1024\n" +
                                          std::string("[layout]\nname = \"flashnext-4gpu\"\n"));
    r = tresolve(opts, t.facts);
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
    r = tresolve(opts, t.facts);
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
    // Identity validation must not allocate entire runtime libraries on the
    // heap: that changes live host room while a native start measures it.
    // The child derives a memory limit from its own address space and the
    // fixture size; there is no machine-specific memory budget in this test.
    {
        const fs::path large = t.file("engine/runtime-home/kernels/large-test.so",
                                      std::string(32u * 1024u * 1024u, 'x'));
        const std::string expected = sv::engine_identity(bin, t.root / "engine/runtime-home");
        const pid_t child = ::fork();
        assert(child >= 0);
        if (child == 0) {
            size_t pages = 0;
            std::ifstream("/proc/self/statm") >> pages;
            struct rlimit limit{};
            if (pages == 0 || ::getrlimit(RLIMIT_AS, &limit) != 0) _exit(2);
            const rlim_t extra = static_cast<rlim_t>(fs::file_size(large) / 8);
            const rlim_t requested = static_cast<rlim_t>(pages) * static_cast<rlim_t>(::sysconf(_SC_PAGESIZE)) + extra;
            limit.rlim_cur = limit.rlim_max == RLIM_INFINITY ? requested : std::min(requested, limit.rlim_max);
            if (::setrlimit(RLIMIT_AS, &limit) != 0) _exit(3);
            try {
                _exit(sv::engine_identity(bin, t.root / "engine/runtime-home") == expected ? 0 : 4);
            } catch (...) {
                _exit(5);
            }
        }
        int status = 0;
        assert(::waitpid(child, &status, 0) == child);
        assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
    // The key covers the planning inputs, the card and the host pool the plan echoes; not the memory outputs
    // the plan feeds (host KV tier, headroom) nor where the engine listens or lives (port, runtime home, model
    // path).
    const auto r = tresolve(t.standard(), t.facts);
    const Json k = sv::plan_key_inputs(r.values, t.facts, id, eid, "");
    assert(k.at("card").at("arch") == "gfx1201" && k.at("card").at("vram_total_mib") == 32624);
    assert(k.at("layout") == "flashnext-4gpu");
    for (const char* absent : {"prefix_cache_mib", "host_headroom_mib", "port", "radiance_home", "model"})
        assert(!k.at("engine").contains(absent));
    assert(k.at("engine").contains("host_pool_mib") && k.at("engine").contains("max_num_seqs"));
    const auto native_file = t.file("plan-key-fleet.yaml", "cards: [0]\nmodels: {tiny: {model: tiny.rad}}\n");
    const auto native = sv::NativeConfigSnapshot::read(native_file, t.root);
    const auto worker = t.file("engine/runtime-home/helpers/r3_moe_worker", "helper-binary-v1");
    Json native_values = r.values;
    native_values["engine"]["radiance_home"] = (t.root / "engine/runtime-home").string();
    const Json native_key = sv::plan_key_inputs(native_values, t.facts, id, eid, "", &native);
    assert(native_key.at("cache_version") == 3 && native_key.at("native_config").at("sha256") == native.sha256() &&
           native_key.at("native_config").at("resolution_root") == t.root.string());
    assert(native_key.at("native_config").at("rank_devices").size() == r.values.at("layout").at("engine_cards").size() +
                                                                          r.values.at("layout").at("helper_cards").size());
    // Volatile free host/GPU memory is checked by fresh admission, not
    // included in the reusable plan identity or its derived helper budgets.
    // Re-resolve between captures to catch a hidden dependency as well as
    // an accidental facts field added directly to the plan key.
    {
        Json live = t.facts;
        live["host"]["mem_available_mib"] = live["host"].at("mem_available_mib").get<int64_t>() - 1;
        for (auto& card : live["gpus"])
            card["vram_available_mib"] = card.at("vram_available_mib").get<int64_t>() - 1;
        const Resolved changed = tresolve(t.standard(), live);
        Json changed_native_values = changed.values;
        changed_native_values["engine"]["radiance_home"] = native_values.at("engine").at("radiance_home");
        assert(sv::plan_key_inputs(changed_native_values, live, id, eid, "", &native) == native_key);
        assert(sv::plan_key_inputs(changed.values, live, id, eid, "") == k);
        // Physical VRAM capacity, unlike availability, genuinely changes the
        // card's identity and must invalidate this exact key.
        live["gpus"][2]["vram_total_mib"] = live["gpus"][2].at("vram_total_mib").get<int64_t>() - 1;
        assert(sv::plan_key_inputs(native_values, live, id, eid, "", &native) != native_key);
    }
    // Native helper build identity must also hash in bounded memory. All
    // helper-based native layouts bind this digest in their frozen key.
    {
        const fs::path large_worker = t.file("engine/runtime-home/helpers/r3_moe_worker",
                                             std::string(32u * 1024u * 1024u, 'h'));
        const Json expected_key = sv::plan_key_inputs(native_values, t.facts, id, eid, "", &native);
        const pid_t child = ::fork();
        assert(child >= 0);
        if (child == 0) {
            size_t pages = 0;
            std::ifstream("/proc/self/statm") >> pages;
            struct rlimit limit{};
            if (pages == 0 || ::getrlimit(RLIMIT_AS, &limit) != 0) _exit(2);
            const rlim_t extra = static_cast<rlim_t>(fs::file_size(large_worker) / 8);
            const rlim_t requested = static_cast<rlim_t>(pages) * static_cast<rlim_t>(::sysconf(_SC_PAGESIZE)) + extra;
            limit.rlim_cur = limit.rlim_max == RLIM_INFINITY ? requested : std::min(requested, limit.rlim_max);
            if (::setrlimit(RLIMIT_AS, &limit) != 0) _exit(3);
            try {
                _exit(sv::plan_key_inputs(native_values, t.facts, id, eid, "", &native) == expected_key ? 0 : 4);
            } catch (...) {
                _exit(5);
            }
        }
        int status = 0;
        assert(::waitpid(child, &status, 0) == child);
        assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
        write(worker, "helper-binary-v1");
    }
    Json reordered_values = native_values;
    assert(reordered_values.at("layout").at("helper_cards").size() > 1);
    std::swap(reordered_values["layout"]["helper_cards"][0],
              reordered_values["layout"]["helper_cards"][1]);
    assert(sv::plan_key(sv::plan_key_inputs(reordered_values, t.facts, id, eid, "", &native)) !=
           sv::plan_key(native_key));
    assert(sv::plan_key(native_key) != sv::plan_key(k));
    const auto other_root = t.root / "other-root";
    std::filesystem::create_directories(other_root);
    const auto same_bytes_other_root = sv::NativeConfigSnapshot::read(native_file, other_root);
    assert(sv::plan_key(sv::plan_key_inputs(native_values, t.facts, id, eid, "", &same_bytes_other_root)) !=
           sv::plan_key(native_key));
    write(worker, "helper-binary-v2");
    assert(sv::plan_key(sv::plan_key_inputs(native_values, t.facts, id, eid, "", &native)) !=
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
                const auto changed = tresolve(opts, t.facts);
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
            assert(sv::plan_key(sv::plan_key_inputs(tresolve(opts, t.facts).values, t.facts, id, eid, "")) ==
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
    Json inconsistent_claimable = native_ranks;
    inconsistent_claimable["gpu"][0]["claimable_bytes"] = 11;
    refuse_owned_ranks(inconsistent_claimable,
                       "owner rank claimable VRAM differs from the engine plan's same-rank free VRAM facts");
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



// RAD49 steps 3-4: two helpers behind a TP2 engine. The budgets are each card less its archs.toml reserve (no
// budget-mib), the RX 6900 XT is the tail, the combine is on, and the manifest carries both helpers behind the
// two ranks.


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
namespace sv = bc::serving;

// A build-tree start takes no --set for its runtime home: it is <build>/radiance_home of the --engine-binary, and this bc-serve
// must be the one in it.
void build_tree_runtime_home(Fixture& t) {
    namespace ss = bc::serving::session;
    const fs::path build = t.root / "tree-build";
    fs::create_directories(build / "bin");
    fs::create_directories(build / "radiance_home/bin");
    std::ofstream(build / "bin/radiance") << "#!/bin/sh\n";
    const fs::path owner = build / "radiance_home/bin/bc-serve";
    std::ofstream(owner) << "#!/bin/sh\n";
    const fs::path home = fs::weakly_canonical(build / "radiance_home");
    assert(ss::derive_build_runtime_home(owner, build / "bin/radiance") == home);
    const auto refused = [&](const fs::path& self, const fs::path& engine, const std::string& why) {
        try {
            (void)ss::derive_build_runtime_home(self, engine);
        } catch (const ss::PlanFailed& error) {
            assert(std::string(error.what()).find(why) != std::string::npos);
            return;
        }
        throw std::runtime_error("derive_build_runtime_home accepted: " + why);
    };
    // another build's bc-serve with this build's engine
    const fs::path other = t.root / "other-build";
    fs::create_directories(other / "radiance_home/bin");
    std::ofstream(other / "radiance_home/bin/bc-serve") << "#!/bin/sh\n";
    refused(other / "radiance_home/bin/bc-serve", build / "bin/radiance", "run the bc-serve of the build");
    // an engine that is not <build>/bin/radiance, and a build without a runtime home
    refused(owner, build / "radiance", "is not <build>/bin/radiance");
    fs::create_directories(t.root / "bare/bin");
    std::ofstream(t.root / "bare/bin/radiance") << "#!/bin/sh\n";
    refused(owner, t.root / "bare/bin/radiance", "has no radiance_home");
}

// A serving start is native and production-only: the lab calibration and experiment options of the foreground
// owner no longer exist, so they are unknown options, not a refusal that a later start could lift.
void serve_has_no_lab_options() {
    for (const std::vector<std::string>& option :
         std::vector<std::vector<std::string>>{{"--calibration-run"}, {"--lab-experiment", "x.json"},
                                               {"--live-policy", "x.json"}, {"--calibration-prompts-a", "a"},
                                               {"--calibration-prompts-b", "b"}, {"--calibration-store", "s"}}) {
        std::vector<std::string> args = {"serve", "qwen27b-tp3", "--run-dir", "lab-option-run"};
        args.insert(args.end(), option.begin(), option.end());
        const auto result = run_bc_serve(args);
        if (result.exit_code == 0 || result.output.find("serve: unknown option " + option.front()) == std::string::npos)
            throw std::runtime_error("serve accepted the removed lab option " + option.front() + ": " + result.output);
    }
}

// `quality --lab-experiment FILE` takes a forced placement schedule and nothing else; every malformed file is refused
// before any plan, and serve (above) still has no such option.
void quality_lab_experiment_file(Fixture& t) {
    const auto read = [&](const std::string& name, const std::string& text) {
        return bc::serving::read_lab_placement_experiment(t.file(name.c_str(), text));
    };
    const auto refused = [&](const std::string& name, const std::string& text, const std::string& why) {
        try {
            read(name, text);
        } catch (const std::runtime_error& e) {
            if (std::string(e.what()).find(why) != std::string::npos) return;
            throw std::runtime_error("lab experiment " + name + " refused for another reason: " + e.what());
        }
        throw std::runtime_error("lab experiment " + name + " was accepted");
    };
    const auto swaps = read("lab-swaps.json", R"({"placement":{"mode":"forced_churn","forced":{"repeat":2,"steps":[
        {"kind":"swap","a_layer":0,"a_expert":3,"b_layer":0,"b_expert":9},
        {"kind":"move","a_layer":1,"a_expert":4,"destination":1}]}}})");
    if (swaps.transactions != 4 || swaps.helper_limit != 2)
        throw std::runtime_error("lab experiment: the explicit swap list was not counted");
    const auto automatic = read("lab-auto.json", R"({"placement":{"mode":"forced_churn","forced":{"repeat":3,"steps":[
        {"kind":"auto_swap","a_layer":0,"a_expert":2}]}}})");
    if (automatic.transactions != 3 || automatic.helper_limit != 3)
        throw std::runtime_error("lab experiment: the auto_swap schedule was not counted");
    bc::serving::require_experiment_helpers(automatic, 3);
    for (const size_t too_few : {size_t{0}, size_t{1}, size_t{2}}) {
        try {
            bc::serving::require_experiment_helpers(automatic, too_few);
        } catch (const std::runtime_error&) {
            continue;
        }
        throw std::runtime_error("lab experiment: a schedule naming helper 2 was accepted for " +
                                 std::to_string(too_few) + " helpers");
    }
    refused("lab-not-json.json", "{", "is not a JSON object");
    refused("lab-array.json", "[]", "is not a JSON object");
    refused("lab-shape.json", R"({"shape":{"helper_common":{"gemv":{"h2":4}}}})", "must hold exactly `placement`");
    refused("lab-two.json", R"({"placement":{"mode":"observe"},"diagnostics":{}})", "must hold exactly `placement`");
    refused("lab-observe.json", R"({"placement":{"mode":"observe"}})", "forced_churn");
    refused("lab-cycle.json", R"({"placement":{"mode":"forced_churn","epoch_cycle_s":5,"forced":{"repeat":1,"steps":[
        {"kind":"auto_swap","a_layer":0,"a_expert":1}]}}})", "forced_churn");
    refused("lab-no-steps.json", R"({"placement":{"mode":"forced_churn","forced":{"repeat":1,"steps":[]}}})", "steps");
    refused("lab-swap-b.json", R"({"placement":{"mode":"forced_churn","forced":{"repeat":1,"steps":[
        {"kind":"swap","a_layer":0,"a_expert":3}]}}})", "swap names both experts");
    refused("lab-move.json", R"({"placement":{"mode":"forced_churn","forced":{"repeat":1,"steps":[
        {"kind":"move","a_layer":0,"a_expert":3}]}}})", "a move needs a destination");
    refused("lab-same.json", R"({"placement":{"mode":"forced_churn","forced":{"repeat":1,"steps":[
        {"kind":"auto_swap","a_layer":1,"a_expert":1}]}}})", "two different helpers");
    refused("lab-unknown.json", R"({"placement":{"mode":"forced_churn","forced":{"repeat":1,"steps":[
        {"kind":"auto_swap","a_layer":0,"a_expert":1,"colour":1}]}}})", "colour");
    try {
        bc::serving::read_lab_placement_experiment(t.root / "no-such-experiment.json");
        throw std::runtime_error("a missing lab experiment file was accepted");
    } catch (const std::runtime_error& e) {
        if (std::string(e.what()).find("cannot read") == std::string::npos) throw;
    }

    // Through the command line: quality names the file, a bad one is refused before the plan; serve and start never
    // read it.
    const std::string bad = t.file("lab-cli-bad.json", R"({"placement":{"mode":"observe"}})").string();
    const std::string good = t.file("lab-cli-good.json", R"({"placement":{"mode":"forced_churn","forced":{
        "repeat":1,"steps":[{"kind":"auto_swap","a_layer":0,"a_expert":1}]}}})").string();
    const auto quality = [&](const std::string& file) {
        return run_bc_serve({"quality", "qwen27b-tp3", "--run-dir", (t.root / "lab-cli-run").string(),
                             "--lab-experiment", file});
    };
    const auto refusal = quality(bad);
    if (refusal.exit_code == 0 || refusal.output.find("--lab-experiment") == std::string::npos ||
        refusal.output.find("forced_churn") == std::string::npos)
        throw std::runtime_error("quality did not refuse a non-forced experiment: " + refusal.output);
    const auto accepted = quality(good);
    if (accepted.output.find("--lab-experiment") != std::string::npos)
        throw std::runtime_error("quality refused a valid forced placement file: " + accepted.output);
    for (const char* cmd : {"serve", "start"}) {
        const auto served = run_bc_serve({cmd, "qwen27b-tp3", "--run-dir", (t.root / "lab-cli-run").string(),
                                          "--lab-experiment", good});
        if (served.exit_code == 0 || served.output.find(std::string(cmd) + ": unknown option --lab-experiment") ==
                                         std::string::npos)
            throw std::runtime_error(std::string(cmd) + " accepted --lab-experiment: " + served.output);
    }
}

void v3_writer_cli_rejects_unowned_imports(Fixture& t) {
    const auto no_snapshot = run_bc_serve({"--write-manifest", "unused.json"});
    if (no_snapshot.exit_code != 2 ||
        no_snapshot.output.find("v3 --write-manifest requires --bc-native-config-fd") == std::string::npos)
        throw std::runtime_error("v3 writer accepted a missing sealed snapshot: " + no_snapshot.output);

    for (const auto* flag : {"--measured-plan", "--rank-demand", "--helpers-resident-mib",
                             "--engine-resident-mib", "--publish-v2-p1", "--publish-v2-live-room"}) {
        const auto result = run_bc_serve({"--write-manifest", "unused.json", flag});
        if (result.exit_code != 2 ||
            result.output.find(std::string("unknown option ") + flag) == std::string::npos)
            throw std::runtime_error(std::string("writer accepted removed option ") + flag + ": " +
                                     result.output);
    }

    // A native v3 measured writer must accept the frozen-key guard too. Use an
    // invalid descriptor so the CLI reaches sealed-snapshot loading without
    // opening cache files or starting any process.
    const fs::path configs_dir = fs::path(BC_CONFIGS_DIR);
    const fs::path model = t.root / "v3-measured-key-model.rad";
    bc::test::write_rad(model, 0, 0, 128, 128, {});
    const fs::path site = t.file("v3-measured-key-site.toml", "[engine]\nmodel = \"" + model.string() +
                                                                   "\"\nradiance_home = \"/runtime\"\n");
    const std::vector<std::string> measured_args = {
        "--write-manifest", "unused.json", "--bc-native-config-fd", "99", "--expected-plan-key",
        std::string(64, '0'), "--expected-owner-attempt", "native-test-attempt", "--plan-cache", "missing-cache",
        "--engine-binary", "/missing/radiance", "--configs", configs_dir.string(),
        "--site", site.string(), "--model-root", (t.root / "model-root").string(), "--sys-root", t.root.string(),
        "--set", "layout.name=\"qwen27b-tp3\""};
    const auto measured_key = run_bc_serve(measured_args);
    if (measured_key.output.find("native writer descriptor must be a fully sealed memfd") == std::string::npos)
        throw std::runtime_error("v3 measured writer rejected the frozen-key guard before snapshot loading: " +
                                 measured_key.output);
    std::vector<std::string> missing_owner_args = measured_args;
    const auto owner_arg = std::find(missing_owner_args.begin(), missing_owner_args.end(), "--expected-owner-attempt");
    missing_owner_args.erase(owner_arg, owner_arg + 2);
    const auto missing_owner = run_bc_serve(missing_owner_args);
    if (missing_owner.output.find("v3 measured cache reads require --expected-owner-attempt") == std::string::npos)
        throw std::runtime_error("v3 measured writer did not require its owner attempt before snapshot access: " +
                                 missing_owner.output);
}




#ifndef BC_FAKE_ENGINE
#error "BC_FAKE_ENGINE must name the fake engine binary (session_fake_engine.cpp)"
#endif
// RSC02 step 1: the session owner's native plan step against the fixture's cards: the qwen27b config's YAML
// admitted, its engine's device visibility from the resolver's rank order, a cache miss measured by the engine
// (--plan-memory against the unmeasured manifest) and stored with the engine's reported start-up memory, and a
// derived host pool measured once more.
void session_engine_plan(Fixture& t) {
    namespace ss = bc::serving::session;
    const fs::path configs = fs::path(BC_CONFIGS_DIR);
    const auto model = t.root / "session-model.rad";
    bc::test::write_rad(model, 0, 0, 128, 128, {});
    const fs::path home = t.root / "session-build" / "radiance_home";
    fs::create_directories(home);
    std::string site_text = "[engine]\nmodel = \"" + model.string() +
                            "\"\nradiance_home = \"" + home.string() + "\"\n";
    site_text += "port = " + std::to_string(select_free_port()) + "\n";
    const auto site = t.file("session-site.toml", site_text);
    const fs::path run = t.root / "session-run";
    fs::create_directories(run);
    ss::EnginePlanInputs in;
    in.work_root = t.work_root();
    in.options.sys_root = t.root;
    in.options.site = site;
    in.options.configs = {configs};
    in.options.model_root = t.root / "model-root"; // the site's model wins; the root is still required
    in.options.sets = {"layout.name=\"qwen27b-tp3\"", "memory.host_pool_mib=1024"};
    in.self = t.root / "bc-serve-test";
    fs::create_symlink(BC_SERVE_EXE, in.self);
    in.engine_binary = BC_FAKE_ENGINE;
    const Json session_facts = bc::serving::detect(t.root);
    const auto session_resolved = tresolve(in.options, session_facts);
    for (const Json& bdf : session_resolved.values.at("layout").at("engine_cards")) {
        const fs::path gtt = t.root / "sys/bus/pci/devices" / bdf.get<std::string>() / "mem_info_gtt_used";
        fs::create_directories(gtt.parent_path());
        std::ofstream(gtt) << "0\n";
    }
    { // Private native admission inspects exact bytes before measurement; the v2 build refuses the v3 path.
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
        std::reverse(expected_kernels.begin(), expected_kernels.end());
        const int native_port = select_free_port();
        const Json execution_settings = {
            {"max_batched_tokens", 384}, {"checkpoint_interval", 1536}, {"kv_dtype", "bf16"},
            {"ngram_placement", session_resolved.values.at("engine").at("ngram_placement")},
            {"speculative_tokens", 2}, {"rope_scaling", "yarn:4"}, {"prefix_resume", "exact"},
            {"prefill_interleave", 1}};
        ss::Env native_overrides = {
            {"TEST_NATIVE_INSPECT_TP", "3"},
            {"TEST_NATIVE_INSPECT_TP_WIRE", "exact"},
            {"TEST_NATIVE_INSPECT_XBIGCHERRY",
             Json{{"schema", 1}, {"description", "qwen27b-tp3 fixture"},
                  {"ranks", Json::array({{{"arch", "gfx1100"}, {"count", 2}}, {{"arch", "gfx1201"}, {"count", 1}}})},
                  {"memory", {{"host-pool-mib", 0}, {"host-kv-mib", 0}}}}   // a dense model: the engine sizes its own
                 .dump()},
            {"TEST_NATIVE_INSPECT_RANK_DEVICES", session_resolved.values.at("layout").at("engine_cards").dump()},
            {"TEST_NATIVE_INSPECT_CARDS", "[0,1,2]"},
            {"TEST_NATIVE_INSPECT_MODEL", model.string()},
            {"TEST_NATIVE_INSPECT_EXPECT_YAML", native_yaml_bytes},
            {"TEST_NATIVE_INSPECT_PORT", std::to_string(native_port)},
            {"TEST_NATIVE_INSPECT_SETTINGS", execution_settings.dump()},
            {"TEST_NATIVE_INSPECT_MAX_CONTEXT",
             std::to_string(session_resolved.values.at("service").at("max_context").get<int>())},
            {"TEST_NATIVE_INSPECT_MAX_SEQUENCES",
             std::to_string(session_resolved.values.at("service").at("max_sessions").get<int>())},
            {"TEST_NATIVE_INSPECT_KERNELS", expected_kernels.dump()}};
        gated.native_admission = ss::NativeAdmissionRequest{native_yaml, native_root, std::move(native_overrides)};
        gated.named_config = "qwen27b-tp3";
        // An installed start names its runtime home with `--set engine.radiance_home=...` (main.cpp adds it), so it is
        // an operator set. Which build runs a start is not part of what the stored plan is keyed by.
        const std::string runtime_home_set =
            "engine.radiance_home=" + session_resolved.values.at("engine").at("radiance_home").dump();
        gated.options.sets.push_back(runtime_home_set);
        // Offline preflight (`bc-serve explain CONFIG --native-config YAML`): the planner's inspection and reconcile,
        // stopped before any writer, power admission, engine or helper child. One pass and one refusal per rule.
        if (bc::serving::v2::kContractVersion == 3) {
            int preflight_runs = 0;
            const auto preflight = [&](auto&& tweak) -> std::string {
                ss::EnginePlanInputs pre = gated;
                pre.native_preflight = true;
                tweak(pre.native_admission->runtime_overrides);
                const fs::path pre_run = run / ("native-preflight-" + std::to_string(++preflight_runs));
                fs::create_directories(pre_run);
                std::vector<ss::ChildKind> kinds;
                ss::Hooks hooks;
                hooks.wait = [](std::chrono::milliseconds) { return true; };
                hooks.started = [&](ss::ChildKind kind, const std::string&, pid_t, const fs::path&) {
                    kinds.push_back(kind);
                };
                std::string refusal;
                try {
                    const ss::EngineStart plan = ss::plan_engine_only(pre, pre_run, hooks);
                    assert(plan.native_preflight_receipt.at("protocol") == "bigcherry.native-fleet-inspect/v2");
                    assert(plan.argv.empty() && plan.helpers.empty());   // nothing was prepared to start
                } catch (const ss::PlanFailed& error) {
                    refusal = error.what();
                }
                for (const ss::ChildKind kind : kinds)
                    assert(kind == ss::ChildKind::native_config_inspector);   // never a writer, helper or engine
                return refusal;
            };
            const auto refused_for = [&](const std::string& why, auto&& tweak) {
                const std::string refusal = preflight(tweak);
                if (refusal.find(why) == std::string::npos)
                    throw std::runtime_error("preflight: expected a refusal naming '" + why + "', got '" + refusal + "'");
            };
            if (const std::string pass = preflight([](ss::Env&) {}); !pass.empty())
                throw std::runtime_error("preflight: a matching YAML was refused: " + pass);
            // The tuning options the YAML states are the resolver's tuning values; ill-typed ones are refused.
            {
                ss::EnginePlanInputs pre = gated;
                pre.native_preflight = true;
                pre.native_admission->runtime_overrides["TEST_NATIVE_INSPECT_TUNING"] =
                    Json{{"tp_ownership", "kv=1,1,2;ffn=5632,5632,6144;lk=5,5,6;lv=15,15,18"},
                         {"tune_pin", "all_reduce:nblocks=4"}, {"p2p", "off"}, {"pipelined_prefill_min_rows", 256}}
                        .dump();
                const fs::path tuned_run = run / "native-preflight-tuning";
                fs::create_directories(tuned_run);
                const ss::EngineStart tuned = ss::plan_engine_only(pre, tuned_run, ss::Hooks{});
                const Json& tuning = tuned.native_preflight_values.at("tuning");
                assert(tuning.at("tp_ownership") == "kv=1,1,2;ffn=5632,5632,6144;lk=5,5,6;lv=15,15,18");
                assert(tuning.at("tune_pin") == "all_reduce:nblocks=4" && tuning.at("p2p") == "off");
                assert(tuning.at("pipelined_prefill_min_rows") == 256 && !tuning.contains("prefill_governor"));
            }
            {   // The BigCherry-owned section: one pass with every table, then one refusal per rule.
                const auto section = [](Json section) {
                    return [section](ss::Env& e) { e["TEST_NATIVE_INSPECT_XBIGCHERRY"] = section.dump(); };
                };
                const Json ranks = Json::array({{{"arch", "gfx1100"}, {"count", 2}}, {{"arch", "gfx1201"}, {"count", 1}}});
                const Json full = {{"schema", 1}, {"description", "d"}, {"ranks", ranks},
                                   {"service", {{"objective", "balanced"}}},
                                   {"memory", {{"host-pool-mib", 0}}},
                                   {"handoff", {{"heat", "@bundle/heat/x"}, {"moe-down-tile", true}}},
                                   {"helper", {{"combine", true}}}, {"kernels", {{"dec11", true}}}};
                if (const std::string pass = preflight(section(full)); !pass.empty())
                    throw std::runtime_error("preflight: a full x-bigcherry section was refused: " + pass);
                const auto with = [&](const char* key, Json value) { Json j = full; j[key] = value; return j; };
                refused_for("x-bigcherry.ranks name 2 ranks, but the YAML's tp is 3",
                            section({{"schema", 1}, {"description", "d"},
                                     {"ranks", Json::array({{{"arch", "gfx1100"}, {"count", 2}}})}}));
                refused_for("x-bigcherry.bogus is not a key", section(with("bogus", 1)));
                refused_for("x-bigcherry.helper.combine must be true or false", section(with("helper", {{"combine", "yes"}})));
                refused_for("x-bigcherry.helper.nope is not a key", section(with("helper", {{"nope", true}})));
                refused_for("x-bigcherry.schema is 2", section(with("schema", 2)));
                refused_for("x-bigcherry.description is required", section({{"schema", 1}, {"ranks", ranks}}));
                refused_for("x-bigcherry.helpers[0].placement must be head or tail",
                            section(with("helpers", Json::array({{{"arch", "a"}, {"count", 1}, {"placement", "mid"}}}))));
                refused_for("x-bigcherry.memory.host-pool-mib must be an integer",
                            section(with("memory", {{"host-pool-mib", "0"}})));
                refused_for("x-bigcherry.variant-of needs differs",
                            section(with("variant-of", "other")));
                refused_for("is required: a serving config file carries an x-bigcherry section", section(nullptr));
            }
            refused_for("ill-typed tuning tp_ownership", [](ss::Env& e) {
                e["TEST_NATIVE_INSPECT_TUNING"] = Json{{"tp_ownership", 5}}.dump(); });
            refused_for("ill-typed tuning tune_pin", [](ss::Env& e) {
                e["TEST_NATIVE_INSPECT_TUNING"] = Json{{"tune_pin", ""}}.dump(); });
            refused_for("ill-typed tuning pipelined_prefill_min_rows", [](ss::Env& e) {
                e["TEST_NATIVE_INSPECT_TUNING"] = Json{{"pipelined_prefill_min_rows", "256"}}.dump(); });
            refused_for("model-defined context", [](ss::Env& e) { e["TEST_NATIVE_INSPECT_MAX_CONTEXT"] = "0"; });
            // Radiance's measured auto n-gram placement is admitted as an intent (the engine decides, the owner seals it).
            if (const std::string pass = preflight([](ss::Env& e) {
                    e["TEST_NATIVE_INSPECT_SETTINGS"] = Json{{"ngram_placement", "auto"}}.dump(); });
                !pass.empty())
                throw std::runtime_error("preflight: an auto ngram placement was refused: " + pass);
            refused_for("set num-speculative-tokens", [](ss::Env& e) {
                e["TEST_NATIVE_INSPECT_SETTINGS"] = Json{{"speculative_tokens", -1}}.dump(); });
            refused_for("set rope-scaling", [](ss::Env& e) {
                e["TEST_NATIVE_INSPECT_SETTINGS"] = Json{{"rope_scaling", ""}}.dump(); });
            refused_for("YAML model differs", [](ss::Env& e) { e["TEST_NATIVE_INSPECT_MODEL"] = "/elsewhere/other.rad"; });
            // (tp against the selected ranks cannot differ: the resolver's ranks are the YAML's x-bigcherry.ranks, and
            // their count against the YAML's tp is the refusal tested above.)
            refused_for("card ordinals differ", [](ss::Env& e) { e["TEST_NATIVE_INSPECT_CARDS"] = "[2,1,0]"; });
            refused_for("rank-device order differs", [](ss::Env& e) {
                e["TEST_NATIVE_INSPECT_RANK_DEVICES"] = Json::array({"0000:09:00.0", "0000:09:01.0", "0000:09:02.0"}).dump(); });
            refused_for("loopback listener", [](ss::Env& e) { e["TEST_NATIVE_INSPECT_HOST"] = "0.0.0.0"; });
            refused_for("runtime home differs", [](ss::Env& e) { e["TEST_NATIVE_INSPECT_REPORTED_HOME"] = "/another/home"; });
        }
        const fs::path gated_run = run / "native-admission";
        fs::create_directories(gated_run);
        // The measured manifest is stored for the next start of this config, from any run dir, in the work root.
        const fs::path cache_tripwire = ss::named_store_index(t.work_root(), gated.named_config);
        int writer_calls = 0;
        bool power_admitted = false;
        int power_admit_calls = 0;
        std::string admitted_measurement_manifest_sha256;
        std::vector<std::string> inspector_roles;
        std::vector<pid_t> reaped;
        ss::Hooks admission_hooks;
        admission_hooks.wait = [](std::chrono::milliseconds) { return true; };
        admission_hooks.started = [&](ss::ChildKind kind, const std::string& role, pid_t,
                                      const fs::path&) {
            assert(kind == ss::ChildKind::native_config_inspector || kind == ss::ChildKind::manifest_writer ||
                   kind == ss::ChildKind::measuring_engine);
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
        admission_hooks.admit_power = [&](const fs::path& manifest_path, const bc::serving::v2::Manifest&, bc::serving::v2::PlanUse) {
            power_admitted = true;
            ++power_admit_calls;
            std::ifstream manifest_file(manifest_path, std::ios::binary);
            if (!manifest_file) throw std::runtime_error("cannot read pinned native measuring manifest in test hook");
            const std::string manifest_bytes((std::istreambuf_iterator<char>(manifest_file)),
                                             std::istreambuf_iterator<char>());
            admitted_measurement_manifest_sha256 = bc::serving::sha256_hex(manifest_bytes);
            return bc::serving::v2::Status{};
        };
        bool admitted_expected_path = false;
        std::string native_gate_error;
        try {
            const auto plan = ss::plan_engine_only(gated, gated_run, admission_hooks);
            admitted_expected_path = plan.native_launch && plan.pinned_manifest && plan.helpers.empty() &&
                plan.execution_manifest() != plan.manifest;
        } catch (const ss::PlanFailed& error) {
            native_gate_error = error.what();
        }
        const fs::path attempts = gated_run / "measurement-attempts";
        std::vector<fs::path> attempt_dirs;
        if (fs::is_directory(attempts))
            for (const auto& attempt : fs::directory_iterator(attempts))
                if (attempt.is_directory()) attempt_dirs.push_back(attempt.path());
        bool owner_receipt_verified = false;
        bool measured_artifact_verified = false;
        fs::path original_manifest;
        if (attempt_dirs.size() == 1 && !admitted_measurement_manifest_sha256.empty()) {
            const fs::path attempt = attempt_dirs.front();
            std::vector<fs::path> cache_entries;
            const fs::path owner_cache = attempt / "plan-cache";
            if (fs::is_directory(owner_cache))
                for (const auto& entry : fs::directory_iterator(owner_cache))
                    if (entry.is_regular_file() && entry.path().extension() == ".json")
                        cache_entries.push_back(entry.path());
            if (cache_entries.size() == 1) {
                std::ifstream cache_file(cache_entries.front(), std::ios::binary);
                Json cache_record;
                cache_file >> cache_record;
                const Json& owner = cache_record.at("owner");
                const Json& rank_demand = cache_record.at("rank_demand");
                owner_receipt_verified = owner.at("attempt_id") == attempt.filename().string() &&
                    owner.at("measuring_manifest_sha256") == admitted_measurement_manifest_sha256 &&
                    cache_record.at("resident").at("engine_mib") ==
                        std::max<int64_t>(1, (rank_demand.at("resident_other_bytes").get<int64_t>() +
                                               ((int64_t{1} << 20) - 1)) / (int64_t{1} << 20));
                const fs::path measured_path = gated_run / ("manifest-native-" + attempt.filename().string() + ".json");
                std::ifstream measured_file(measured_path, std::ios::binary);
                if (owner_receipt_verified && measured_file) {
                    const std::string measured_bytes((std::istreambuf_iterator<char>(measured_file)),
                                                     std::istreambuf_iterator<char>());
                    const auto parsed = bc::serving::v2::read_manifest(
                        measured_bytes, "native v3 session measurement test", bc::serving::v2::PlanUse::serving);
                    if (parsed.ok()) {
                        const auto& spec = parsed.value().spec;
                        const auto native = bc::serving::v2::native_config_view(parsed.value());
                        const auto& engine = spec.processes.engine;
                        measured_artifact_verified = native && native->yaml_utf8 == native_yaml_bytes &&
                            engine.server.port == native_port &&
                            engine.runtime.kernels == expected_kernels.get<std::vector<std::string>>() &&
                            engine.scheduler.max_num_batched_tokens == 384 &&
                            engine.scheduler.checkpoint_interval == 1536 &&
                            engine.execution.kv_cache_dtype == bc::serving::v2::KvCacheDtype::kBf16 &&
                            engine.execution.num_speculative_tokens == 2 &&
                            engine.execution.rope_scaling == "yarn:4" &&
                            engine.execution.prefix_resume == bc::serving::v2::PrefixResume::kExact &&
                            engine.tuning && engine.tuning->prefill_interleave == 1 &&
                            spec.intent.model == model.string() && spec.processes.engine.device.gpus ==
                                session_resolved.values.at("layout").at("engine_cards").get<std::vector<std::string>>() &&
                            spec.plan.engine_memory &&
                            bc::serving::v2::encode(spec.plan.engine_memory->expected) == cache_record.at("plan") &&
                            spec.plan.engine_memory->model_identity == bc::serving::container_identity(model) &&
                            spec.plan.resource_inputs &&
                            spec.plan.resource_inputs->resident_other_bytes ==
                                rank_demand.at("resident_other_bytes").get<int64_t>();
                        if (measured_artifact_verified) original_manifest = measured_path;
                    }
                }
            }
        }
        if (!(admitted_expected_path && writer_calls == 0 && power_admitted && inspector_roles.size() == 3 &&
              inspector_roles[0] == "native config inspector" &&   // the device-free parse that resolves the config
              inspector_roles[1] == "native config inspector" &&   // the admission inspect with the selected cards
              inspector_roles[2] == "native v3 measuring engine" &&   // the manifests are written in the owner, not by children
              reaped.size() == 3 && owner_receipt_verified &&
              measured_artifact_verified && !fs::exists(native_yaml) && fs::exists(cache_tripwire)))
            throw std::runtime_error("native v3 session did not complete one owner-tracked measurement and measured "
                                     "writer with a sealed serving handoff: " +
                                     native_gate_error);

        write(t.root / "proc/self/limits",
              "Limit                     Soft Limit           Hard Limit          Units\n"
              "Max locked memory         unlimited            unlimited           bytes\n");
        const auto attestation_path = original_manifest.string() + ".replay-index.json";
        const auto original_index = ss::read_stored_index(attestation_path);
        assert(original_index.native_measurement &&
               original_index.manifest_filename == original_manifest.filename().string());
        ss::EnginePlanInputs replay = gated;
        replay.stored_manifest = original_manifest;
        replay.options.model_root.clear();
        ss::Hooks stored_hooks;
        stored_hooks.wait = [](std::chrono::milliseconds) { return true; };
        unsigned replay_inspections = 0;
        unsigned replay_admissions = 0;
        stored_hooks.started = [&](ss::ChildKind kind, const std::string&, pid_t, const fs::path&) {
            assert(kind == ss::ChildKind::native_config_inspector);
            ++replay_inspections;
        };
        stored_hooks.admit_power = [&](const fs::path&, const bc::serving::v2::Manifest&, bc::serving::v2::PlanUse use) {
            assert(use == bc::serving::v2::PlanUse::serving && replay_inspections > replay_admissions);
            ++replay_admissions;
            return bc::serving::v2::Status{};
        };
        const auto refuse_stored = [&](const ss::EnginePlanInputs& rejected, std::string_view why) {
            try {
                (void)ss::replay_stored_manifest(rejected, gated_run, stored_hooks);
            } catch (const std::exception& error) {
                if (std::string_view(error.what()).find(why) != std::string_view::npos) return;
                throw;
            }
            throw std::runtime_error("stored v3 replay unexpectedly admitted: " + std::string(why));
        };
        auto tampered = original_index;
        tampered.native_measurement->sha256 = std::string(64, '0');
        ss::write_stored_index(attestation_path, tampered);
        refuse_stored(replay, "owner measurement no longer matches");
        ss::write_stored_index(attestation_path, original_index);
        const auto replace_fixture_bytes = [&](const fs::path& path, const std::string& bytes) {
            const fs::path replacement = path.string() + ".fixture-replacement";
            write(replacement, bytes);
            fs::rename(replacement, path);
        };
        const auto change_bytes_and_refuse = [&](const fs::path& path, std::string_view why) {
            std::ifstream original(path, std::ios::binary);
            const std::string bytes((std::istreambuf_iterator<char>(original)), std::istreambuf_iterator<char>());
            assert(!bytes.empty());
            replace_fixture_bytes(path, bytes + '\n');
            assert(ss::complete_file_identity(path) != ss::complete_file_identity_bytes(bytes));
            refuse_stored(replay, why);
            replace_fixture_bytes(path, bytes);
        };
        change_bytes_and_refuse(original_manifest, "no longer binds the original manifest bytes");
        change_bytes_and_refuse(original_manifest.string() + ".s8-evidence.json", "producer evidence was tampered");
        auto changed_build = replay;
        changed_build.engine_binary = t.root / "changed-native-engine";
        fs::copy_file(BC_FAKE_ENGINE, changed_build.engine_binary);
        std::ofstream(changed_build.engine_binary, std::ios::binary | std::ios::app) << '\n';
        refuse_stored(changed_build, "engine, kernels or helper build changed");
        auto wrong_root = replay;
        wrong_root.native_admission->resolution_root = t.root;
        refuse_stored(wrong_root, "requested resolution root differs");
        auto low_replay = replay;
        low_replay.replay_live_room = [](const auto&, const auto&) {
            return bc::serving::v2::P1LiveRoom{1, 1, 1};
        };
        refuse_stored(low_replay, "live MemAvailable/cgroup room");
        assert(replay_admissions == 0 && !fs::exists(native_yaml));
        replay_inspections = 0;

        std::ofstream(native_yaml, std::ios::binary) << native_yaml_bytes;
        std::ostringstream hit_err;
        std::streambuf* const err_before = std::cerr.rdbuf(hit_err.rdbuf());
        const auto hit = [&] {
            struct Restore { std::streambuf* b; ~Restore() { std::cerr.rdbuf(b); } } restore{err_before};
            return ss::plan_engine_only(gated, gated_run, stored_hooks);
        }();
        // The reuse is named once on stderr and recorded with the planning phase durations.
        const std::string hit_line = hit_err.str();
        assert(hit_line.find("bc-serve: stored plan reused") == 0 &&
               std::count(hit_line.begin(), hit_line.end(), '\n') == 1);
        assert(hit.plan_reuse.at("outcome") == "hit" && hit.phase_ms.contains("yaml_inspect") &&
               hit.phase_ms.contains("runtime_identity") && hit.phase_ms.contains("store_lookup") &&
               hit.phase_ms.contains("replay"));
        assert(hit.manifest == original_manifest && hit.summary.at("replay") == true &&
               hit.stored_sha256 == bc::serving::file_sha256(original_manifest));
        assert(replay_inspections == 2 && replay_admissions == 1);   // the device-free config parse and the admission inspect
        fs::remove(native_yaml);

        // A later start of the same config in ANOTHER run directory reuses the stored measurement: no writer,
        // no measuring engine, the admission inspect only.
        replay_inspections = 0;
        replay_admissions = 0;
        const fs::path later_run = run / "native-later-start";
        fs::create_directories(later_run);
        std::ofstream(native_yaml, std::ios::binary) << native_yaml_bytes;
        const auto later = ss::plan_engine_only(gated, later_run, stored_hooks);
        assert(later.manifest == original_manifest && later.summary.at("replay") == true &&
               later.summary.at("plan_measurement_attempt") == original_index.native_measurement->attempt_id);
        assert(replay_inspections == 2 && replay_admissions == 1 && !fs::exists(later_run / "measurement-attempts"));
        fs::remove(native_yaml);

        // A start under a DIFFERENT engine build still reuses the stored measurement: no measuring engine runs,
        // the stored plan is sealed again under the current build (draft and measured writers), the old build's
        // identity is kept as provenance, and the new manifest replaces the stored one.
        const fs::path rebuilt_run = run / "native-rebuilt-start";
        fs::create_directories(rebuilt_run);
        std::ofstream(native_yaml, std::ios::binary) << native_yaml_bytes;
        auto rebuilt = gated;
        // The installed layout of a newer release: <root>/releases/<tag>/radiance_home.
        const fs::path rebuilt_home = t.root / "installed-root/releases/bc-radiance-2020.0.0/radiance_home";
        fs::create_directories(rebuilt_home);
        for (auto& set : rebuilt.options.sets)
            if (set == runtime_home_set) set = "engine.radiance_home=" + Json(rebuilt_home.string()).dump();
        rebuilt.engine_binary = t.root / "rebuilt-native-engine";
        fs::copy_file(BC_FAKE_ENGINE, rebuilt.engine_binary);
        std::ofstream(rebuilt.engine_binary, std::ios::binary | std::ios::app) << '\n';
        const size_t roles_before = inspector_roles.size();
        const auto carried = ss::plan_engine_only(rebuilt, rebuilt_run, admission_hooks);
        const std::vector<std::string> carried_roles(inspector_roles.begin() + static_cast<long>(roles_before),
                                                     inspector_roles.end());
        assert(!carried.summary.contains("replay") && carried.manifest.parent_path() == rebuilt_run);
        assert(carried.plan_reuse.at("outcome") == "live_admission_refused" && carried.phase_ms.contains("measure"));
        assert(std::find(carried_roles.begin(), carried_roles.end(), "native v3 measuring engine") ==
                   carried_roles.end() &&
               carried_roles.size() == 2);   // the two inspectors only: the plan is sealed again in the owner
        assert(carried.summary.at("plan_carried_from").at("attempt") == original_index.native_measurement->attempt_id &&
               carried.summary.at("plan_carried_from").at("engine_identity") !=
                   bc::serving::engine_identity(rebuilt.engine_binary, home));
        assert(ss::lookup_stored_manifest(cache_tripwire, ss::read_stored_index(cache_tripwire).key).manifest == carried.manifest);
        fs::remove(native_yaml);

        // A stored measurement the new inputs cannot use measures afresh: here the stored plan's rank record is
        // removed, so nothing can be carried.
        {
            const auto entry = ss::read_stored_index(cache_tripwire);
            const fs::path attempt_cache = fs::path(entry.directory) / "measurement-attempts" /
                                           entry.native_measurement->attempt_id / "plan-cache";
            for (const auto& file : fs::directory_iterator(attempt_cache)) fs::remove(file.path());
            const fs::path unusable_run = run / "native-unusable-store";
            fs::create_directories(unusable_run);
            std::ofstream(native_yaml, std::ios::binary) << native_yaml_bytes;
            const size_t before = inspector_roles.size();
            std::ostringstream miss_err;
            std::streambuf* const miss_before = std::cerr.rdbuf(miss_err.rdbuf());
            const auto measured_again = [&] {
                struct Restore { std::streambuf* b; ~Restore() { std::cerr.rdbuf(b); } } restore{miss_before};
                return ss::plan_engine_only(rebuilt, unusable_run, admission_hooks);
            }();
            // A miss that measures again says why, in one line.
            const std::string miss_line = miss_err.str();
            assert(measured_again.plan_reuse.at("outcome") == "carried_plan_unusable" &&
                   miss_line.find("bc-serve: no stored plan reused (stored plan unusable: ") == 0 &&
                   std::count(miss_line.begin(), miss_line.end(), '\n') == 1);
            assert(!measured_again.summary.contains("plan_carried_from") &&
                   std::find(inspector_roles.begin() + static_cast<long>(before), inspector_roles.end(),
                             "native v3 measuring engine") != inspector_roles.end());
            fs::remove(native_yaml);
        }

        // The same release installed in a SECOND prefix, the first prefix since removed: the engine and runtime home
        // hold the same bytes, only where they live differs. That is decided up front from the engine identity: no
        // replay is attempted (it would refuse for the paths the stored manifest embeds), the one reuse line names
        // the carry, no measuring engine runs, and the stored measurement is sealed again under this prefix.
        {
            const auto entry_before = ss::read_stored_index(cache_tripwire);
            const fs::path first_prefix = rebuilt_home.parent_path().parent_path().parent_path();
            fs::remove_all(first_prefix);
            const fs::path second_home = t.root / "installed-root-second/releases/bc-radiance-2020.0.0/radiance_home";
            fs::create_directories(second_home);
            auto second = rebuilt;
            for (auto& set : second.options.sets)
                if (set == "engine.radiance_home=" + Json(rebuilt_home.string()).dump())
                    set = "engine.radiance_home=" + Json(second_home.string()).dump();
            const fs::path second_run = run / "native-second-prefix";
            fs::create_directories(second_run);
            std::ofstream(native_yaml, std::ios::binary) << native_yaml_bytes;
            const unsigned replays_before = replay_inspections;
            const size_t before = inspector_roles.size();
            std::ostringstream carry_err;
            std::streambuf* const carry_before = std::cerr.rdbuf(carry_err.rdbuf());
            const auto second_start = [&] {
                struct Restore { std::streambuf* b; ~Restore() { std::cerr.rdbuf(b); } } restore{carry_before};
                return ss::plan_engine_only(second, second_run, admission_hooks);
            }();
            const std::string carry_line = carry_err.str();
            assert(carry_line.find("bc-serve: stored plan carried") == 0 && carry_line.find("not replayable") == std::string::npos &&
                   carry_line.find("measuring") == std::string::npos &&
                   std::count(carry_line.begin(), carry_line.end(), '\n') == 1);
            assert(second_start.plan_reuse.at("outcome") == "carried" && replay_inspections == replays_before &&
                   !second_start.summary.contains("replay") && second_start.manifest.parent_path() == second_run);
            assert(std::find(inspector_roles.begin() + static_cast<long>(before), inspector_roles.end(),
                             "native v3 measuring engine") == inspector_roles.end());
            assert(second_start.summary.at("plan_carried_from").at("attempt") ==
                   entry_before.native_measurement->attempt_id);
            assert(ss::lookup_stored_manifest(cache_tripwire, ss::read_stored_index(cache_tripwire).key).manifest ==
                   second_start.manifest);
            fs::remove(native_yaml);
        }

        std::ofstream(native_yaml, std::ios::binary) << native_yaml_bytes;
        const fs::path owner_run = run / "native-serving-owner";
        fs::create_directories(owner_run);
        ss::EnginePlanInputs fresh = gated;
        fresh.work_root = t.root / "work-owner-measures";   // no stored measurement here: this owner measures
        fresh.native_admission->runtime_overrides["TEST_NATIVE_EXPECT_FROZEN"] = "before";
        assert(::setenv("TEST_NATIVE_FROZEN", "before", 1) == 0);
        ss::Session owner;
        owner.run_dir = owner_run;
        owner.timing.health_timeout = std::chrono::seconds(6);
        owner.timing.health_interval = std::chrono::milliseconds(30);
        owner.timing.stop_grace = std::chrono::seconds(1);
        owner.timing.poll = std::chrono::milliseconds(20);
        owner.plan = [&](const fs::path& dir, const ss::Hooks& hooks) {
            return fresh.stored_manifest.empty() ? ss::plan_engine_only(fresh, dir, hooks)
                                                : ss::replay_stored_manifest(fresh, dir, hooks);
        };
        bool serving_power_verified = false;
        bool cleared_children = false;
        owner.power_hook.begin_manifest = [&](const fs::path& path, const bc::serving::v2::Manifest&, bc::serving::v2::PlanUse use) {
            assert(use == (fresh.stored_manifest.empty() ? bc::serving::v2::PlanUse::measuring
                                                        : bc::serving::v2::PlanUse::serving));
            assert(bc::serving::v2::load_manifest(path, use));
            fs::remove(native_yaml);
            return bc::serving::v2::Status{};
        };
        owner.power_hook.verify_serving = [&](const fs::path& path) {
            const auto parsed = bc::serving::v2::load_manifest(path, bc::serving::v2::PlanUse::serving);
            assert(parsed && parsed.value().spec.plan.engine_memory);
            serving_power_verified = true;
            return bc::serving::v2::Status{};
        };
        owner.power_hook.verified_exit = [&] { cleared_children = true; };
        owner.power_hook.restore = [&] {
            assert(cleared_children);
            return bc::serving::v2::Status{};
        };
        owner.before_gpu_start = [&](const ss::EngineStart& plan) {
            assert(serving_power_verified && plan.execution_manifest() != plan.manifest);
            replace_fixture_bytes(plan.manifest, "{}\n");
            assert(::setenv("TEST_NATIVE_FROZEN", "changed-after-admission", 1) == 0);
            return bc::serving::v2::Status{};
        };
        bool serving_handoff_seen = false;
        owner.runtime_guard = [&](const ss::EngineStart&, const fs::path& log) {
            std::ifstream file(log);
            const std::string contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            serving_handoff_seen = contents.find("native v3 serving snapshot, root and environment verified") !=
                                   std::string::npos;
            return bc::serving::v2::Status{};
        };
        const auto run_owner = [&] {
            auto stop_owner = std::async(std::launch::async, [&] {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(12);
                while (std::chrono::steady_clock::now() < deadline) {
                    const auto record = ss::read_record(owner.run_dir).value_or(Json::object());
                    const std::string state = record.value("state", std::string());
                    if (state == "failed" || state == "stopped") return false;
                    if (state == "running") {
                        const auto reply = ss::control(owner.run_dir, "stop", std::chrono::seconds(3));
                        return reply && reply->value("ok", false);
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
                (void)ss::control(owner.run_dir, "stop", std::chrono::seconds(3));
                return false;
            });
            const int owner_result = ss::serve(owner);
            const bool owner_stopped = stop_owner.get();
            assert(::unsetenv("TEST_NATIVE_FROZEN") == 0);
            const auto completed = ss::read_record(owner.run_dir).value_or(Json::object());
            if (!(owner_result == 0 && owner_stopped && serving_handoff_seen && cleared_children &&
                  completed.value("state", std::string()) == "stopped" && ss::alive_recorded(completed).empty()))
                throw std::runtime_error("native v3 session did not complete sealed serving and verified stop: " +
                                         completed.dump());
            return completed;
        };
        (void)run_owner();
        assert(!fs::exists(native_yaml));
        owner.run_dir = run / "native-stored-owner";
        fs::create_directories(owner.run_dir);
        fresh.stored_manifest = original_manifest;
        assert(::setenv("TEST_NATIVE_FROZEN", "before", 1) == 0);
        serving_power_verified = cleared_children = serving_handoff_seen = false;
        const auto replay_record = run_owner();
        if (replay_record.at("selection").value("replay", false) != true ||
            replay_record.at("selection").at("plan_measurement_attempt") != original_index.native_measurement->attempt_id ||
            replay_record.at("processes").size() != 3)   // the two inspectors (config parse, admission) and the engine
            throw std::runtime_error("native v3 stored replay record: " + replay_record.at("processes").dump() + replay_record.at("selection").dump());
        assert(replay_record.at("selection").at("replay") == true &&
               replay_record.at("selection").at("plan_measurement_attempt") == original_index.native_measurement->attempt_id &&
               replay_record.at("processes").size() == 3);

        std::ofstream(native_yaml, std::ios::binary) << native_yaml_bytes;
        const fs::path low_room_run = run / "native-low-room";
        fs::create_directories(low_room_run);
        const fs::path meminfo = t.root / "proc/meminfo";
        std::ifstream prior_room_file(meminfo);
        const std::string prior_room((std::istreambuf_iterator<char>(prior_room_file)), std::istreambuf_iterator<char>());
        ss::Hooks low_room = admission_hooks;
        pid_t measuring_child = -1;
        int measured_children = 0;
        low_room.started = [&](ss::ChildKind kind, const std::string&, pid_t pid, const fs::path&) {
            if (kind == ss::ChildKind::measuring_engine) {
                measuring_child = pid;
                ++measured_children;
            }
        };
        const auto& memory = session_resolved.values.at("memory");
        const int64_t ranks = session_resolved.values.at("engine").at("tp").get<int64_t>();
        const int64_t room_mib = memory.at("host_pool_mib").get<int64_t>() * ranks +
            memory.at("host_kv_mib").get<int64_t>() + memory.at("host_reserve_mib").get<int64_t>();
        std::string reduced_room = prior_room;
        const size_t available = reduced_room.find("MemAvailable:");
        assert(available != std::string::npos);
        reduced_room.replace(available, reduced_room.find('\n', available) - available,
                             "MemAvailable: " + std::to_string(room_mib * 1024) + " kB");
        low_room.ended = [&](pid_t pid) {
            if (pid == measuring_child)
                write(meminfo, reduced_room);
        };
        std::string low_room_error;
        try {
            auto measuring = gated;
            measuring.work_root = t.root / "work-low-room";   // nothing stored: measure, so the room check can fire
            (void)ss::plan_engine_only(measuring, low_room_run, low_room);
        } catch (const ss::PlanFailed& error) {
            low_room_error = error.what();
        }
        write(meminfo, prior_room);
        const auto measured_attempts = low_room_run / "measurement-attempts";
        const auto attempt = fs::is_directory(measured_attempts) ? fs::directory_iterator(measured_attempts)
                                                                : fs::directory_iterator{};
        if (measured_children != 1 ||
            low_room_error.find("native v3 measured writer refused") == std::string::npos ||
            low_room_error.find("host memory plan refused") == std::string::npos ||
            low_room_error.find("MemAvailable (" + std::to_string(room_mib) + " MiB)") == std::string::npos ||
            low_room_error.find("checkpoint floor") == std::string::npos ||
            attempt == fs::directory_iterator{} ||
            fs::exists(low_room_run / ("manifest-native-" + attempt->path().filename().string() + ".json")))
            throw std::runtime_error("native v3 post-measure admission did not refuse live host room: " + low_room_error);

        const fs::path home_alias = t.root / "native-runtime-home-link";
        const fs::path replacement_home = t.root / "native-runtime-home-identical";
        fs::create_directories(replacement_home);
        fs::create_directory_symlink(home, home_alias);
        const fs::path drift_site = t.file(
            "session-site-runtime-link.toml", "[engine]\nmodel = \"" + model.string() + "\"\nradiance_home = \"" +
                home_alias.string() + "\"\nport = " +
                std::to_string(session_resolved.values.at("engine").at("port").get<int>()) + "\n");
        const fs::path drift_yaml = native_root / "fleet-runtime-drift.yaml";
        std::ofstream(drift_yaml, std::ios::binary) << native_yaml_bytes;
        ss::EnginePlanInputs home_drift = gated;
        home_drift.options.site = drift_site;
        // The runtime home comes from the site file here, not the --set; a work root of its own makes this start measure.
        std::erase(home_drift.options.sets, runtime_home_set);
        home_drift.work_root = t.root / "native-runtime-drift-work";
        fs::create_directories(home_drift.work_root);
        home_drift.native_admission = ss::NativeAdmissionRequest{
            drift_yaml, native_root, gated.native_admission->runtime_overrides};
        const fs::path drift_run = t.root / "native-runtime-drift-run";
        fs::create_directories(drift_run);
        ss::Hooks drift_hooks;
        std::map<pid_t, std::string> drift_roles;
        std::vector<std::string> drift_role_order;
        drift_hooks.wait = [](std::chrono::milliseconds) { return true; };
        drift_hooks.started = [&](ss::ChildKind, const std::string& role, pid_t pid, const fs::path&) {
            drift_roles.emplace(pid, role);
            drift_role_order.push_back(role);
        };
        int drift_inspections = 0;
        drift_hooks.ended = [&](pid_t pid) {
            // The first inspector is the device-free config parse; the drift lands after the admission inspect.
            if (drift_roles.at(pid) == "native config inspector" && ++drift_inspections == 2) {
                fs::remove(drift_yaml);
                fs::remove(home_alias);
                fs::create_directory_symlink(replacement_home, home_alias);
            }
        };
        bool drift_power_admitted = false;
        drift_hooks.admit_power = [&](const fs::path&, const bc::serving::v2::Manifest&, bc::serving::v2::PlanUse) {
            drift_power_admitted = true;
            return bc::serving::v2::Status{};
        };
        bool runtime_home_drift_refused = false;
        std::string runtime_home_drift_error;
        try {
            (void)ss::plan_engine_only(home_drift, drift_run, drift_hooks);
        } catch (const ss::PlanFailed& error) {
            runtime_home_drift_error = error.what();
            runtime_home_drift_refused =
                std::string_view(error.what()).find("runtime home changed during measurement") != std::string_view::npos;
        }
        std::vector<fs::path> drift_attempt_dirs;
        const fs::path drift_attempts = drift_run / "measurement-attempts";
        if (fs::is_directory(drift_attempts))
            for (const auto& attempt : fs::directory_iterator(drift_attempts))
                if (attempt.is_directory()) drift_attempt_dirs.push_back(attempt.path());
        if (!(runtime_home_drift_refused && drift_power_admitted && drift_role_order.size() == 3 &&
              drift_role_order[0] == "native config inspector" &&
              drift_role_order[1] == "native config inspector" &&
              drift_role_order[2] == "native v3 measuring engine" && drift_attempt_dirs.size() == 1 &&
              !fs::exists(drift_attempt_dirs.front() / "plan-cache") &&
              !fs::exists(drift_run / ("manifest-native-" + drift_attempt_dirs.front().filename().string() + ".json"))))
            throw std::runtime_error("native v3 session did not refuse runtime-home symlink retarget before owner-cache "
                                     "publication: " + runtime_home_drift_error);

        ss::EnginePlanInputs reserved_override = gated;
        reserved_override.native_admission->runtime_overrides["CUDA_VISIBLE_DEVICES"] = "0";
        const int power_calls_before_invalid_override = power_admit_calls;
        bool selector_refused = false;
        try {
            (void)ss::plan_engine_only(reserved_override, gated_run, admission_hooks);
        } catch (const ss::PlanFailed& error) {
            selector_refused = std::string_view(error.what()).find("runtime override cannot select CUDA_VISIBLE_DEVICES") !=
                               std::string_view::npos;
        }
        assert(selector_refused && power_admit_calls == power_calls_before_invalid_override);
    }
}

// Real plan_engine_only + real session owner, with the fake KFD/proc tree,
// a child-gated CPU manifest writer and a fake engine. Both manifest cache
// branches must preserve writer -> durable power admission -> first GPU child.
// The injected writer response and power transaction never touch hardware.

// End-to-end CPU-only planner boundary: real resolver and bc-serve writer,
// fake engine/helper process boundary and BDF-specific sysfs fixtures.
void helper_vram_owner_graph(Fixture& t) {
    namespace ss = bc::serving::session;
    const fs::path configs = fs::path(BC_CONFIGS_DIR);
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
    std::string site_text = "[engine]\nmodel = \"" + model.string() +
                            "\"\nradiance_home = \"" + home.string() + "\"\n";
    const int port = select_free_port();
    site_text += "port = " + std::to_string(port) + "\n";
    const fs::path site = t.file("helper-vram-graph.toml", site_text);
    ss::EnginePlanInputs in;
    in.work_root = t.work_root();
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
    const fs::path heat = t.root / "writer.hot-first";
    std::vector<uint32_t> counts(128 * 16, 0);
    for (uint32_t l = 0; l < 4; ++l)
        for (uint32_t e = 0; e < 16; ++e)
            counts[l * 16 + e] = (e * 7 + l * 3) % 16 + 1;
    std::ofstream(heat, std::ios::binary).write(reinterpret_cast<const char*>(counts.data()),
        static_cast<std::streamsize>(counts.size() * sizeof(counts[0])));
    fs::create_directories(home / "helpers");
    fs::create_symlink(BC_FAKE_ENGINE, home / "helpers/r3_moe_worker");
    in.named_config = "flashnext-2xtx";
    in.options.sets.insert(in.options.sets.end(),
                          {"memory.helper_budgets_mib=[1,1]", "memory.host_pool_mib=2048", "memory.host_kv_mib=0"});
    const auto selected = tresolve(in.options, bc::serving::detect(t.root));
    const auto& value = selected.values;
    const auto& engine = value.at("engine");
    const auto& service = value.at("service");
    Json kernels = Json::array();
    std::istringstream list(engine.at("kernels").get<std::string>());
    for (std::string kernel; std::getline(list, kernel, ',');)
        kernels.push_back(kernel);
    for (const auto& gpu : t.facts.at("gpus"))
        write(t.root / "sys/bus/pci/devices" / gpu.at("bdf").get<std::string>() / "mem_info_gtt_used", "0\n");
    const std::string yaml = "cards: [0]\nmodels:\n  tiny: {model: " + Json(model.string()).dump() + "}\n";
    const auto authored = t.file("native-helpers.yaml", yaml);
    in.native_admission = ss::NativeAdmissionRequest{authored, t.root, {
        {"TEST_NATIVE_INSPECT_EXPECT_YAML", yaml},
        {"TEST_NATIVE_INSPECT_MODEL", model.string()},
        {"TEST_NATIVE_INSPECT_RANK_DEVICES", value.at("layout").at("engine_cards").dump()},
        {"TEST_NATIVE_INSPECT_PORT", std::to_string(port)},
        {"TEST_NATIVE_INSPECT_MAX_CONTEXT", service.at("max_context").dump()},
        {"TEST_NATIVE_INSPECT_MAX_SEQUENCES", service.at("max_sessions").dump()},
        {"TEST_NATIVE_INSPECT_KERNELS", kernels.dump()},
        {"TEST_NATIVE_INSPECT_SETTINGS", Json({{"ngram_placement", engine.at("ngram_placement")}}).dump()},
        // flashnext-2xtx: the R9700 engine, both 7900 XTX as head helpers (as engines/radiance/configs/flashnext-2xtx.yaml)
        {"TEST_NATIVE_INSPECT_XBIGCHERRY",
         Json{{"schema", 1}, {"description", "flashnext-2xtx fixture"},
              {"ranks", Json::array({{{"arch", "gfx1201"}, {"count", 1}}})},
              {"helpers", Json::array({{{"arch", "gfx1100"}, {"count", "all"}, {"placement", "head"}}})}}.dump()},
    }};
    std::map<std::string, unsigned> samples;
    in.helper_vram.read_used = [&](const fs::path& path) {
        ++samples[path.parent_path().filename().string()];
        return uint64_t{128} << 20;
    };
    ss::Session owner;
    owner.run_dir = t.root / "native-helper-owner";
    fs::create_directories(owner.run_dir);
    owner.timing.health_timeout = owner.timing.helper_timeout = std::chrono::seconds(6);
    owner.timing.health_interval = owner.timing.helper_interval = std::chrono::milliseconds(30);
    owner.timing.stop_grace = std::chrono::seconds(1);
    owner.timing.poll = std::chrono::milliseconds(20);
    owner.plan = [&](const fs::path& dir, const ss::Hooks& hooks) {
        return in.stored_manifest.empty() ? ss::plan_engine_only(in, dir, hooks)
                                          : ss::replay_stored_manifest(in, dir, hooks);
    };
    std::vector<bc::serving::v2::PlanUse> admissions;
    owner.power_hook.begin_manifest = [&](const fs::path& manifest, const bc::serving::v2::Manifest&, bc::serving::v2::PlanUse use) {
        assert(bc::serving::v2::load_manifest(manifest, use));
        admissions.push_back(use);
        if (use == bc::serving::v2::PlanUse::measuring)
            fs::remove(authored);
        return bc::serving::v2::Status{};
    };
    owner.power_hook.verify_serving = [](const fs::path& manifest) {
        const auto measured = bc::serving::v2::load_manifest(manifest, bc::serving::v2::PlanUse::serving);
        assert(measured && measured.value().spec.plan.resource_inputs &&
               measured.value().spec.plan.resource_inputs->helper_footprints.size() == 2);
        return bc::serving::v2::Status{};
    };
    owner.power_hook.verified_exit = [] {};
    owner.power_hook.restore = [] { return bc::serving::v2::Status{}; };
    const auto serve_and_stop = [&](const std::vector<std::string>& expected_roles) {
        auto stop = std::async(std::launch::async, [&] {
            // A slow but healthy measured writer must not be stopped by a test deadline.
            // Wait for the owner's durable running event; CTest owns the overall test timeout.
            for (;;) {
                const auto record = ss::read_record(owner.run_dir).value_or(Json::object());
                const auto state = record.value("state", std::string{});
                if (state == "failed" || state == "exit_unverified") return false;
                if (state == "running") {
                    const auto reply = ss::control(owner.run_dir, "stop", std::chrono::seconds(4));
                    return reply && reply->value("ok", false);
                }
                // This schedules a recheck; elapsed time never authorizes a stop.
                std::this_thread::sleep_for(owner.timing.poll);
            }
        });
        const int code = ss::serve(owner);
        const bool stopped = stop.get();
        const auto record = ss::read_record(owner.run_dir).value_or(Json::object());
        if (code != 0 || !stopped || record.value("state", std::string{}) != "stopped" ||
            !ss::alive_recorded(record).empty())
            throw std::runtime_error("native helper owner did not serve and stop: " + record.dump());
        std::vector<std::string> roles;
        for (const auto& child : record.at("processes")) {
            assert(child.at("exited") == true);
            roles.push_back(child.at("role").get<std::string>());
        }
        if (roles != expected_roles)
            throw std::runtime_error("native helper owner started unexpected roles: " + record.dump());
        return record;
    };
    // The first inspector is the device-free config parse; the second is the admission inspect.
    const auto fresh = serve_and_stop({"native config inspector", "native config inspector",
        "helper0 measuring", "helper1 measuring", "native v3 measuring engine",
        "helper0 measuring", "helper1 measuring", "native v3 measuring engine", "helper0", "helper1", "engine"});
    assert(samples.size() == 2 && !fs::exists(authored));
    for (const auto& [bdf, count] : samples) {
        assert(count >= 4);
        assert(std::find(value.at("layout").at("engine_cards").begin(),
                         value.at("layout").at("engine_cards").end(), bdf) ==
               value.at("layout").at("engine_cards").end());
    }
    const auto measured_samples = samples;
    const auto stored = fs::path(fresh.at("manifest").get<std::string>());
    const auto attempt = fresh.at("selection").at("plan_measurement_attempt");
    size_t attempts = 0;
    for (const auto& dir : fs::directory_iterator(owner.run_dir / "measurement-attempts")) {
        ++attempts;
        assert(fs::is_regular_file(dir.path() / "plan.json"));
        if (dir.path().filename() != attempt.get<std::string>())
            assert(!fs::exists(owner.run_dir / ("manifest-native-" + dir.path().filename().string() + ".json")));
    }
    assert(attempts == 2);
    const auto final = bc::serving::v2::load_manifest(stored, bc::serving::v2::PlanUse::serving);
    assert(final && final.value().spec.resources.host.pool_mib == 1024 &&
           final.value().spec.native_config.yaml_utf8 == yaml);
    const std::vector<std::string> replay_roles{"native config inspector", "native config inspector", "helper0",
                                                "helper1", "engine"};
    const auto check_replay = [&](const Json& replayed) {
        assert(replayed.at("selection").at("replay") == true);
        assert(replayed.at("manifest") == stored.string());
        assert(replayed.at("handoff_shm") == fresh.at("handoff_shm"));
        assert(replayed.at("selection").at("plan_measurement_attempt") == attempt);
        assert(replayed.at("selection").at("manifest_digest") == fresh.at("selection").at("manifest_digest"));
        assert(samples == measured_samples);
    };
    write(authored, yaml);
    check_replay(serve_and_stop(replay_roles));
    fs::remove(authored);
    in.stored_manifest = stored;
    check_replay(serve_and_stop(replay_roles));
    using PlanUse = bc::serving::v2::PlanUse;
    assert(admissions == std::vector<PlanUse>({PlanUse::measuring, PlanUse::serving, PlanUse::serving}));

    const std::string segment = fresh.at("handoff_shm");
    const int occupied = ::shm_open(segment.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    assert(occupied >= 0);
    ::close(occupied);
    const int conflict = ss::serve(owner);
    const auto refused = ss::read_record(owner.run_dir).value_or(Json::object());
    const int still_occupied = ::shm_open(segment.c_str(), O_RDONLY | O_CLOEXEC, 0);
    if (still_occupied >= 0)
        ::close(still_occupied);
    const int cleaned = ::shm_unlink(segment.c_str());
    assert(conflict != 0 && still_occupied >= 0 && cleaned == 0);
    assert(admissions.size() == 3 && samples == measured_samples);
    assert(refused.at("errors").dump().find("still exists after an unverified stop") != std::string::npos);
    assert(refused.at("processes").size() == 2 &&   // the config parse and the admission inspect, then the refusal
           refused.at("processes").at(0).at("role") == "native config inspector" &&
           refused.at("processes").at(1).at("role") == "native config inspector" &&
           ss::alive_recorded(refused).empty());

    const auto meminfo = t.root / "proc/meminfo";
    std::ifstream prior(meminfo);
    const std::string prior_room((std::istreambuf_iterator<char>(prior)), std::istreambuf_iterator<char>());
    const auto reduced = std::regex_replace(prior_room, std::regex("MemAvailable:[^\n]+"),
                                           "MemAvailable: 1024 kB");
    assert(reduced != prior_room);
    owner.before_gpu_start = [&](const ss::EngineStart& plan) {
        assert(plan.summary.at("replay") == true && admissions.size() == 4);
        write(meminfo, reduced);
        return bc::serving::v2::Status{};
    };
    const int low_room = ss::serve(owner);
    write(meminfo, prior_room);
    const auto room_refused = ss::read_record(owner.run_dir).value_or(Json::object());
    assert(low_room != 0 && admissions.size() == 4 && samples == measured_samples);
    assert(room_refused.at("errors").dump().find("host bytes required=") != std::string::npos);
    assert(room_refused.at("processes").size() == 2 &&
           room_refused.at("processes").at(0).at("role") == "native config inspector" &&
           room_refused.at("processes").at(1).at("role") == "native config inspector" &&
           ss::alive_recorded(room_refused).empty());

    owner.before_gpu_start = {};
    const fs::path reference = t.root / "recorded-quality";
    fs::create_directory(reference);
    write(reference / "kld.json", "{}");
    owner.quality = ss::QualityScore{reference, t.root / "helper-quality.json"};
    const int scored = ss::serve(owner);
    const auto quality = ss::read_record(owner.run_dir).value_or(Json::object());
    check_replay(quality);
    const auto& engine_args = quality.at("processes").back().at("argv");
    const auto events = quality.at("events").dump();
    if (scored != 0 || quality.value("state", "") != "stopped" || !ss::alive_recorded(quality).empty() ||
        engine_args.size() != 5 || engine_args[1] != "--config" || engine_args[3] != "--bc-admission-fd" ||   // the run's files are in the admission
        !fs::is_regular_file(owner.quality->output) ||
        events.find("quality_completed") == std::string::npos || events.find("power_restored") == std::string::npos ||
        events.find("canary_passed") != std::string::npos || events.find("healthy") != std::string::npos)
        throw std::runtime_error("native helper finite score did not complete and clean up: " + quality.dump());
    assert(admissions.size() == 5 && fs::file_size(owner.quality->output.string() + ".rows") == 16);

    // Recording a reference is the same finite run with --kld-record DIR --kld-corpus FILE: no --kld-ref or --kld-out,
    // and it only succeeds when the engine wrote the reference's kld.json.
    {
        const fs::path recorded = t.root / "recorded-reference";
        fs::create_directory(recorded);
        const fs::path corpus = t.root / "record-corpus.jsonl";
        write(corpus, "{\"prompt\":\"a\"}\n");
        owner.quality = ss::QualityScore{recorded, {}, corpus};
        const int recorded_exit = ss::serve(owner);
        const auto record_run = ss::read_record(owner.run_dir).value_or(Json::object());
        const auto& record_args = record_run.at("processes").back().at("argv");
        const auto record_events = record_run.at("events").dump();
        if (recorded_exit != 0 || record_run.value("state", "") != "stopped" || !ss::alive_recorded(record_run).empty() ||
            record_args.size() != 5 || record_args[1] != "--config" || record_args[3] != "--bc-admission-fd" ||
            !fs::is_regular_file(recorded / "kld.json") ||
            record_events.find("quality_recorded") == std::string::npos ||
            record_events.find("power_restored") == std::string::npos)
            throw std::runtime_error("native finite reference recording did not complete and clean up: " + record_run.dump());
        // A reference directory that already holds one, or a missing corpus, is refused before any child starts.
        const int again = ss::serve(owner);
        assert(again != 0);
        owner.quality = ss::QualityScore{t.root / "no-such-reference", {}, corpus};
        assert(ss::serve(owner) != 0);
        fs::create_directory(t.root / "empty-reference");
        owner.quality = ss::QualityScore{t.root / "empty-reference", {}, t.root / "no-corpus.jsonl"};
        assert(ss::serve(owner) != 0);
        owner.quality = ss::QualityScore{reference, t.root / "helper-quality.json"};
    }

    // A finite run that times out still stops every helper and restores power.
    owner.quality->output = t.root / "helper-quality-timeout.json";
    owner.timing.quality_timeout = std::chrono::milliseconds(100);
    write(reference / "kld.json", R"({"fake_delay_ms":5000})");
    const int timed_out = ss::serve(owner);
    const auto timeout_record = ss::read_record(owner.run_dir).value_or(Json::object());
    if (timed_out == 0 || timeout_record.value("state", "") != "failed" ||
        !ss::alive_recorded(timeout_record).empty() ||
        timeout_record.at("errors").dump().find("KL scoring timed out") == std::string::npos ||
        timeout_record.at("events").dump().find("power_restored") == std::string::npos)
        throw std::runtime_error("native helper finite score timeout did not clean up: " + timeout_record.dump());

    // A forced placement schedule: measured afresh into a lab-profile manifest that carries it, the named store left
    // as it was, helper0's report read when the score ends, and the owner map written for the next schedule.
    const fs::path stored_before = stored;
    const auto stored_bytes_before = read_text(stored_before);
    const fs::path swaps = t.file("lab-swaps-run.json", R"({"placement":{"mode":"forced_churn","forced":{"repeat":2,
        "steps":[{"kind":"swap","a_layer":0,"a_expert":3,"b_layer":0,"b_expert":9},
                 {"kind":"auto_swap","a_layer":0,"a_expert":1}]}}})");
    in.stored_manifest.clear();   // an explicit replay input is production-only; the schedule needs a fresh plan
    in.lab_experiment = bc::serving::read_lab_placement_experiment(swaps);
    owner.timing.quality_timeout = std::chrono::minutes(1);
    // The owner waits this long for helper0's forced-placement report (and for a process to stop). A
    // loaded CI container can leave a freshly started helper unscheduled for a second.
    owner.timing.stop_grace = std::chrono::seconds(10);
    owner.quality->output = t.root / "helper-quality-forced.json";
    write(reference / "kld.json", "{}");
    write(authored, yaml);   // the measuring admission consumes the authored file
    const int forced = ss::serve(owner);
    const auto forced_record = ss::read_record(owner.run_dir).value_or(Json::object());
    const auto forced_events = forced_record.at("events").dump();
    if (forced != 0 || forced_record.value("state", "") != "stopped" || !ss::alive_recorded(forced_record).empty() ||
        forced_events.find("\"forced_placement\"") == std::string::npos ||
        forced_events.find("\"committed\":4") == std::string::npos ||
        forced_events.find("\"swaps\"") == std::string::npos ||
        forced_events.find("owner_map_written") == std::string::npos)
        throw std::runtime_error("forced placement score did not complete and report: " + forced_record.dump());
    const auto lab_manifest = bc::serving::v2::load_manifest(fs::path(forced_record.at("manifest").get<std::string>()),
                                                             bc::serving::v2::PlanUse::serving);
    if (!lab_manifest || lab_manifest.value().spec.intent.service.profile != bc::serving::v2::Profile::kLab ||
        !lab_manifest.value().spec.experiment || !lab_manifest.value().spec.experiment->placement ||
        !lab_manifest.value().spec.experiment->placement->forced ||
        lab_manifest.value().spec.experiment->placement->forced->steps.size() != 2 ||
        lab_manifest.value().spec.experiment->placement->forced->steps[0].kind != bc::serving::v2::ForcedKind::kSwap ||
        lab_manifest.value().spec.experiment->placement->forced->steps[1].kind != bc::serving::v2::ForcedKind::kAutoSwap ||
        lab_manifest.value().spec.experiment->placement->forced->repeat != 2)
        throw std::runtime_error("the forced schedule did not reach the manifest the engine ran");
    if (read_text(stored_before) != stored_bytes_before ||
        read_text(owner.run_dir / "lab-experiment.json").find("auto_swap") == std::string::npos)
        throw std::runtime_error("a lab experiment changed the stored plan, or its file was not kept");
    const Json owner_map = Json::parse(read_text(owner.run_dir / "owner-map.json"));
    if (owner_map.at("helpers").size() != 2 || owner_map.at("helpers").at(0).at("role") != "helper0" ||
        owner_map.at("helpers").at(0).at("experts").size() != owner_map.at("helpers").at(0).at("count") ||
        !owner_map.at("helpers").at(0).contains("arch") || !owner_map.at("helpers").at(1).contains("tier"))
        throw std::runtime_error("the owner map dump is incomplete: " + owner_map.dump());

    // Swaps that did not all commit fail the score, and the report names them.
    owner.quality->output = t.root / "helper-quality-forced-short.json";
    write(authored, yaml);
    ::setenv("TEST_FAKE_FORCED_COMMITTED", "3", 1);
    // One actual report followed by silence exercises the deadline without
    // relying on the final SIGUSR1 response racing the last poll.
    ::setenv("TEST_FAKE_REPORT_ONCE", "1", 1);
    const int short_forced = ss::serve(owner);
    ::unsetenv("TEST_FAKE_REPORT_ONCE");
    ::unsetenv("TEST_FAKE_FORCED_COMMITTED");
    const auto short_record = ss::read_record(owner.run_dir).value_or(Json::object());
    if (short_forced == 0 || short_record.value("state", "") != "failed" || !ss::alive_recorded(short_record).empty() ||
        short_record.at("errors").dump().find("forced placement incomplete: committed 3 of 4") == std::string::npos)
        throw std::runtime_error("an incomplete forced schedule did not fail the score: " + short_record.dump());

    // Force a fresh measured plan and request an operator stop when the measuring engine starts. This is an event, not
    // a timing guess: the owner must stop/reap the measuring engine and helpers before acknowledging stop.
    in.stored_manifest.clear();
    in.lab_experiment.reset();
    in.work_root = t.root / "writer-stop-cache";
    write(authored, yaml);
    ss::Session stop_owner = owner;
    stop_owner.run_dir = t.root / "writer-stop";
    stop_owner.quality.reset();
    bool saw_measuring_engine = false;
    stop_owner.plan = [&](const fs::path& dir, const ss::Hooks& hooks) {
        ss::Hooks observing = hooks;
        observing.started = [&](ss::ChildKind kind, const std::string& role, pid_t pid, const fs::path& log) {
            hooks.started.require("session.hooks.started", kind, role, pid, log);
            if (role == "native v3 measuring engine") {
                saw_measuring_engine = true;
                ::raise(SIGTERM); // installed owner handler: legal operator stop during the measurement
            }
        };
        return ss::plan_engine_only(in, dir, observing);
    };
    const int interrupted = ss::serve(stop_owner);
    const auto interrupted_record = ss::read_record(stop_owner.run_dir).value_or(Json::object());
    const auto interrupted_events = interrupted_record.value("events", Json::array()).dump();
    if (!saw_measuring_engine || interrupted != 0 || interrupted_record.value("state", "") != "stopped" ||
        !interrupted_record.value("errors", Json::array()).empty() ||
        !ss::alive_recorded(interrupted_record).empty() ||
        interrupted_events.find("stop_requested") == std::string::npos ||
        interrupted_events.find("power_restored") == std::string::npos ||
        interrupted_events.find("segment_retired") == std::string::npos)
        throw std::runtime_error("native measuring-engine stop failed to clean up: " + interrupted_record.dump());
}


// RSC05: the resolver's card selection is by architecture in group order, engine and helpers never share a card, a
// missing card is refused naming its architecture, and the child visibility follows the selected order.
void card_selection(Fixture& t) {
    namespace ss = bc::serving::session;
    const auto bdfs_of = [](const Json& cards) { return cards.get<std::vector<std::string>>(); };
    const auto arch_of = [&](const std::string& bdf) {
        for (const auto& g : t.facts.at("gpus"))
            if (g.at("bdf") == bdf) return g.at("arch").get<std::string>();
        return std::string();
    };
    auto opts = t.standard();
    bc::serving::fixture::use(opts, "qwen27b-tp3");
    const auto forward = tresolve(opts, t.facts);
    const auto engine = bdfs_of(forward.values.at("layout").at("engine_cards"));
    assert(engine.size() == 3 && arch_of(engine[0]) == "gfx1100" && arch_of(engine[1]) == "gfx1100" &&
           arch_of(engine[2]) == "gfx1201" && engine[0] != engine[1]);
    // The rank groups reversed: the card of the other architecture is rank 0, the two boards of one architecture keep
    // their inventory order.
    auto reversed = opts;
    std::reverse(reversed.inspected->x.ranks.begin(), reversed.inspected->x.ranks.end());
    const auto back = bdfs_of(tresolve(reversed, t.facts).values.at("layout").at("engine_cards"));
    assert(back.size() == 3 && back[0] == engine[2] && back[1] == engine[0] && back[2] == engine[1]);
    const auto visible = ss::engine_visibility("/runtime", back, t.facts);
    std::string rocr;
    for (size_t i = 0; i < back.size(); ++i)
        for (const auto& g : t.facts.at("gpus"))
            if (g.at("bdf") == back[i]) rocr += (i ? "," : "") + std::to_string(g.at("hip_ordinal").get<int64_t>());
    assert(visible.at("ROCR_VISIBLE_DEVICES") == rocr && visible.at("HIP_VISIBLE_DEVICES") == "0,1,2");
    assert(visible.at("ROCR_VISIBLE_DEVICES") != ss::engine_visibility("/runtime", engine, t.facts).at("ROCR_VISIBLE_DEVICES"));
    // A card that is not in the inventory (it vanished after selection) is refused, naming it.
    Json without = t.facts;
    Json kept = Json::array();
    for (const auto& g : t.facts.at("gpus"))
        if (g.at("bdf") != engine[2]) kept.push_back(g);
    without["gpus"] = kept;
    try {
        (void)ss::engine_visibility("/runtime", engine, without);
        assert(false && "a vanished card was accepted");
    } catch (const bc::serving::session::PlanFailed& e) {
        assert(std::string(e.what()).find(engine[2]) != std::string::npos);
    }
    // An inventory without the gfx1201 card refuses the layout, naming the architecture and the counts.
    try {
        (void)tresolve(opts, without);
        assert(false && "a layout with a missing card resolved");
    } catch (const std::runtime_error& e) {
        assert(std::string(e.what()).find("layout: missing gfx1201 card(s): need 1, found 0") != std::string::npos);
    }
    // Engine and helpers are disjoint.
    auto helpers = t.standard();
    const auto with_helpers = tresolve(helpers, t.facts);
    const auto hel = with_helpers.values.at("layout");
    for (const auto& h : hel.value("helper_cards", Json::array()))
        for (const auto& e : hel.at("engine_cards")) assert(h != e);
}


// PWR08 S2: x-bigcherry.power.profile selects a profile of the one profile file; the file's JSON is decoded strictly.
void power_profile_selection() {
    namespace v2 = bc::serving::v2;
    Json section = {{"schema", 1}, {"description", "d"}, {"ranks", Json::array({{{"arch", "gfx1100"}, {"count", 1}}})}};
    assert(bc::serving::decode_x_bigcherry(section).power_profile() == "stock");
    section["power"] = {{"profile", "efficient"}};
    assert(bc::serving::decode_x_bigcherry(section).power_profile() == "efficient");
    section["power"] = {{"profile", "efficient"}, {"mode", "apply"}};
    try { (void)bc::serving::decode_x_bigcherry(section); assert(false && "an unknown power key was accepted"); }
    catch (const std::invalid_argument& e) { assert(std::string(e.what()).find("x-bigcherry.power.mode") != std::string::npos); }

    const Json good = {{"schema", 1},
                       {"profiles", {{"stock", {{"default_mode", "observe"}}},
                                     {"cap", {{"default_mode", "observe"},
                                              {"rules", Json::array({{{"role", "engine"}, {"arch", "gfx1201"},
                                                                      {"evidence", "docs/x.md"},
                                                                      {"settings", {{"power_cap_w", 200}}}}})}}}}}};
    const auto loaded = v2::decode_power_profiles(good);
    assert(loaded && loaded.value().size() == 2 && loaded.value().at("cap").rules.size() == 1 &&
           loaded.value().at("cap").rules[0].settings.power_cap_w == 200);
    const auto refused = [&](Json doc, const char* field) {
        const auto r = v2::decode_power_profiles(doc);
        assert(!r && r.error().path.find(field) != std::string::npos);
    };
    Json unknown = good; unknown["profiles"]["cap"]["surprise"] = 1;
    refused(unknown, "surprise");
    Json applies = good; applies["profiles"]["cap"]["default_mode"] = "apply";
    refused(applies, "default_mode");
    Json stock_rules = good;
    stock_rules["profiles"]["stock"]["rules"] = good.at("profiles").at("cap").at("rules");
    refused(stock_rules, "stock");
    Json bad_role = good; bad_role["profiles"]["cap"]["rules"][0]["role"] = "gpu0";
    refused(bad_role, "role");
    Json host_evidence = good; host_evidence["profiles"]["cap"]["rules"][0]["evidence"] = "/abs/run.txt";
    refused(host_evidence, "evidence");
    Json float_setting = good; float_setting["profiles"]["cap"]["rules"][0]["settings"]["power_cap_w"] = 200.5;
    refused(float_setting, "power_cap_w");
    Json no_stock = good; no_stock["profiles"].erase("stock");
    refused(no_stock, "stock");
    // the selection names a profile of the file, or stock; anything else is refused naming the file
    namespace ss = bc::serving::session;
    assert(ss::selected_power_rules(loaded.value(), "stock").rules.empty());
    assert(ss::selected_power_rules(loaded.value(), "cap").rules.size() == 1);
    try { (void)ss::selected_power_rules(loaded.value(), "nope"); assert(false && "an unknown profile was accepted"); }
    catch (const ss::PlanFailed& e) { assert(std::string(e.what()).find("power-profiles.yaml") != std::string::npos); }
    // the compiled-in file is the shipped one: its stock profile exists (the engine decodes the rest in CI's inspector)
    assert(v2::embedded_power_profiles_yaml().find("stock:") != std::string::npos);

    // this host's site file: per-BDF settings only, closed keys, a lower-case BDF, evidence required
    const Json site = {{"schema", 1},
                       {"devices", {{"0000:03:00.0", {{"settings", {{"sclk_max_mhz", 2900}, {"vddgfx_offset_mv", -75}}},
                                                        {"evidence", "run-1"}}}}}};
    const auto site_loaded = v2::decode_power_site(site);
    assert(site_loaded && site_loaded.value().at("0000:03:00.0").sclk_max_mhz == 2900 &&
           site_loaded.value().at("0000:03:00.0").vddgfx_offset_mv == -75);
    assert(v2::decode_power_site(Json(nullptr)).value().empty());
    const auto site_refused = [&](Json doc, const char* field) {
        const auto r = v2::decode_power_site(doc);
        assert(!r && r.error().path.find(field) != std::string::npos);
    };
    Json upper_bdf = site; upper_bdf["devices"]["0000:0A:00.0"] = site["devices"]["0000:03:00.0"];
    site_refused(upper_bdf, "0000:0A:00.0");
    Json site_extra = site; site_extra["devices"]["0000:03:00.0"]["model"] = "x";
    site_refused(site_extra, "model");
    Json site_offset = site; site_offset["devices"]["0000:03:00.0"]["settings"]["sclk_offset_mhz"] = 50;
    site_refused(site_offset, "sclk_offset_mhz");
    Json site_no_evidence = site; site_no_evidence["devices"]["0000:03:00.0"].erase("evidence");
    site_refused(site_no_evidence, "0000:03:00.0");
    // the calibration key sees what the cards run under: a name for observe, the effective per-BDF settings for apply
    Json spec = {{"facts", {{"gpus", Json::array({{{"bdf", "0000:03:00.0"}, {"arch", "gfx1100"}, {"vram", {{"total_mib", 24560}}}}})}}},
                 {"processes", {{"engine", {{"device", {{"gpus", Json::array({"0000:03:00.0"})}}}}},
                                {"helpers", Json::array()}, {"helper_common", Json::object()}}},
                 {"plan", {{"engine_memory", {{"model_identity", "m"}}},
                           {"power", {{"mode", "observe"}, {"devices", Json::array()}}}}},
                 {"intent", {{"layout", "l"}, {"service", {{"objective", "balanced"}}}, {"power", {{"profile", "cap"}}}}},
                 {"handoff", {{"telemetry", {{"period_s", 1}}},
                              {"transport", {{"max_rows", 1}, {"small_rows", 1}, {"gemv_large_rows", 1}, {"skip_owned", true}}},
                              {"kernels", {{"moe_down_tile", 1}}}}}};
    assert(bc::serving::policy_calibration_inputs(spec).at("power_profile") == "cap");
    spec["plan"]["power"] = {{"mode", "apply"}, {"devices", Json::array({{{"role", "engine"}, {"bdf", "0000:03:00.0"},
                                                                         {"settings", {{"power_cap_w", 200}}}}})}};
    const std::string cap200 = bc::serving::policy_calibration_inputs(spec).at("power_profile");
    spec["plan"]["power"]["devices"][0]["settings"]["power_cap_w"] = 190;
    const std::string cap190 = bc::serving::policy_calibration_inputs(spec).at("power_profile");
    assert(cap200.rfind("cap@", 0) == 0 && cap200 != cap190 && cap200 != "cap");
    // --qualify-power is a lab apply: refused on its own, before any config is read
    const auto qualify_alone = run_bc_serve({"serve", "any", "--run-dir", "/tmp/qualify-alone", "--qualify-power"});
    assert(qualify_alone.exit_code != 0 && qualify_alone.output.find("--allow-power-apply") != std::string::npos);
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--bc-serve-exe" && i + 1 < argc) {
            serve_test_executable = argv[++i];
        } else {
            std::cerr << "usage: bc_serving_test [--bc-serve-exe PATH]\n";
            return 2;
        }
    }
    try {
        sha256_vectors();
        Fixture t;
        detection(t);
        detection_limits(t);
        unknown_keys(t);
        card_selection(t);
        power_profile_selection();
        type_range_required(t);
        precedence_and_caps(t);
        provenance_hash_and_parity(t);
        calibration_store(t);
        host_reserve(t);
        host_ledger_plan();
        measured_plan_cache(t);
        serve_has_no_lab_options();
        quality_lab_experiment_file(t);
        v3_writer_cli_rejects_unowned_imports(t);
        build_tree_runtime_home(t);
        session_engine_plan(t);
        helper_vram_owner_graph(t);
        std::cout << "PASS: fake KFD/PCI/proc detection; strict keys; types/ranges; required; precedence/caps; "
                     "provenance/hash; calibration store; calibrated host reserve; host ledger; measured plan cache; "
                     "native v3 sealed snapshot session, config read and pre-cache gate\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
