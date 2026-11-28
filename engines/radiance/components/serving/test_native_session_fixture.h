#pragma once
// Test only: a card-free native (v3) serving host and the plan inputs for it, shared by the tests that run a real
// owner (serving_test, session_test, session_stored_test, release_install_test).
//
//   write_fake_host(root)       a sysfs/procfs tree for four cards (two gfx1100, a gfx1201, a gfx1030), a 128 GiB
//                               host with a cgroup limit, memlock, and a small model.rad
//   native_session_inputs(...)  ss::EnginePlanInputs that admit a native YAML through the fake engine
//                               (session_fake_engine.cpp), whose inspector answers from TEST_NATIVE_INSPECT_*
//
// The fake engine's behaviour (health delay, exit codes, ...) is passed as JSON in TEST_FAKE_BEHAVIOUR; it applies to
// the serving engine only, never to the measuring run or the inspector.
#include "serving.h"
#include "session.h"
#include "test_config_fixture.h"
#include "test_rad_fixture.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef BC_FAKE_ENGINE
#error "BC_FAKE_ENGINE must name the fake engine binary"
#endif
#ifndef BC_SERVE_EXE
#error "BC_SERVE_EXE must name bc-serve"
#endif
#ifndef BC_CONFIGS_DIR
#error "BC_CONFIGS_DIR must name engines/radiance/configs"
#endif

namespace bc::test {

namespace fs = std::filesystem;
using Json = nlohmann::json;

inline int free_port() {
    const int sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) throw std::runtime_error("socket");
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) throw std::runtime_error("bind");
    socklen_t len = sizeof addr;
    if (::getsockname(sock, reinterpret_cast<sockaddr*>(&addr), &len) != 0) throw std::runtime_error("getsockname");
    ::close(sock);
    return int(ntohs(addr.sin_port));
}

inline void write_file(const fs::path& path, const std::string& content) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot create fixture file " + path.string());
    out << content;
    if (!out) throw std::runtime_error("cannot write fixture file " + path.string());
}

/* The fake host under `root`: KFD node order, not BDF sort or device names, defines HIP ordinals. A session that
 * replays a stored plan needs the memlock limit to admit its pinned demand: `unlimited_memlock`. */
inline void write_fake_host(const fs::path& root, bool unlimited_memlock = false) {
    fs::create_directories(root / "work");
    const auto nodes = root / "sys/class/kfd/kfd/topology/nodes";
    write_file(nodes / "0/properties", "gfx_target_version 0\n");
    const std::vector<int> versions = {110000, 110000, 120001, 100300};
    const std::vector<int> buses = {3, 6, 9, 23};
    const std::vector<int> simds = {192, 192, 128, 160};
    const std::vector<int> vram = {24560, 24560, 32624, 16368};
    for (int i = 0; i != 4; ++i) {
        const auto node = nodes / std::to_string(i + 1) / "properties";
        write_file(node, "gfx_target_version " + std::to_string(versions[i]) + "\nsimd_count " +
                             std::to_string(simds[i]) + "\nlocation_id " + std::to_string(buses[i] * 256) +
                             "\ndomain 0\ndrm_render_minor " + std::to_string(128 + i) + "\n");
        char bdf[24];
        std::snprintf(bdf, sizeof(bdf), "0000:%02x:00.0", buses[i]);
        const auto pci = root / "sys/bus/pci/devices" / bdf;
        write_file(pci / "mem_info_vram_total", std::to_string(uint64_t(vram[i]) * 1048576));
        write_file(pci / "mem_info_vram_used", std::to_string(uint64_t(128) * 1048576));
        write_file(pci / "current_link_speed", (i == 2 ? "32.0" : "16.0") + std::string(" GT/s PCIe\n"));
        write_file(pci / "max_link_speed", (i == 2 ? "32.0" : "16.0") + std::string(" GT/s PCIe\n"));
        write_file(pci / "current_link_width", "16\n");
        write_file(pci / "max_link_width", "16\n");
        write_file(pci / "numa_node", "0\n");
    }
    write_file(root / "proc/meminfo", "MemTotal:       134217728 kB\n"
                                      "MemAvailable:    94371840 kB\n"
                                      "SwapTotal:       8388608 kB\n");
    fs::create_directories(root / "dev/shm");
    // A v2 cgroup with a limit at the leaf and none above; memlock limited, no CAP_IPC_LOCK;
    // amdgpu loaded with its system memory limit in force.
    write_file(root / "proc/self/cgroup", "0::/agents/serve\n");
    write_file(root / "sys/fs/cgroup/agents/serve/memory.max", std::to_string(uint64_t(64) << 30U) + "\n");
    write_file(root / "sys/fs/cgroup/agents/serve/memory.current", std::to_string(uint64_t(10) << 30) + "\n");
    write_file(root / "sys/fs/cgroup/agents/serve/memory.stat",
               "anon 1\nfile " + std::to_string(uint64_t(4) << 30) + "\n");
    write_file(root / "sys/fs/cgroup/agents/memory.max", "max\n");
    write_file(root / "sys/fs/cgroup/memory.max", "max\n");
    write_file(root / "proc/self/limits",
               "Limit                     Soft Limit           Hard Limit           Units\n"
               + std::string(unlimited_memlock ? "Max locked memory         unlimited            unlimited            bytes\n"
                                               : "Max locked memory         8388608              8388608              bytes\n"));
    write_file(root / "proc/self/status", "Name:\tbc-serve\nCapEff:\t00000000a80425fb\n");
    write_file(root / "sys/module/amdgpu/parameters/no_system_mem_limit", "N\n");
    write_rad(root / "model.rad", 4, 16, 128, 128, {});
}

/* A fixture config's resolver values for the fake host (what the owner plans against). */
inline serving::Resolved resolve_fixture(serving::Options options, const Json& facts,
                                         const serving::NativeEngineSettings* native = nullptr) {
    if (!options.inspected)
        if (const auto name = serving::layered_string(options, "layout", "name");
            name && serving::fixture::has_preset(*name))
            serving::fixture::use(options, *name);
    return serving::resolve(options, facts, native);
}

/* The plan inputs that admit `fleet.yaml` for the fixture config "qwen27b-tp3-q8" on the fake host, serving the
 * fake engine on `port`. `scratch` holds this start's own files (site, YAML, work root); `behaviour` is the fake
 * engine's behaviour for its serving run. `host` is the write_fake_host tree.
 *
 * The YAML must still exist whenever the plan runs: it is read once, sealed, and every later child gets that
 * snapshot, never the path. */
struct NativeSessionPlan {
    serving::session::EnginePlanInputs inputs;
    fs::path yaml;
    std::string yaml_bytes;
    Json resolved_values;
};

/* engine_only: the fixture config qwen27b-tp3-q8 (three ranks, no helper). helpers: flashnext-2xtx, the gfx1201
 * engine with both gfx1100 cards as head helpers (fake workers, a shared-memory handoff, measured helper footprints). */
enum class SessionLayout { engine_only, helpers };

inline NativeSessionPlan native_helper_session_inputs(const fs::path& host, const fs::path& scratch, int port,
                                                      const Json& behaviour, const fs::path& runtime_home);

inline NativeSessionPlan native_session_inputs(const fs::path& host, const fs::path& scratch, int port,
                                               const Json& behaviour = Json::object(),
                                               SessionLayout layout = SessionLayout::engine_only,
                                               const fs::path& runtime_home = {}) {
    namespace ss = serving::session;
    if (layout == SessionLayout::helpers)
        return native_helper_session_inputs(host, scratch, port, behaviour, runtime_home);
    fs::create_directories(scratch);
    const fs::path model = host / "session-model.rad";
    if (!fs::exists(model)) write_rad(model, 0, 0, 128, 128, {});
    const fs::path home = runtime_home.empty() ? host / "session-build" / "radiance_home" : runtime_home;
    fs::create_directories(home);
    const fs::path site = scratch / "site.toml";
    write_file(site, "[engine]\nmodel = \"" + model.string() + "\"\nradiance_home = \"" + home.string() +
                         "\"\nport = " + std::to_string(port) + "\n");
    NativeSessionPlan out;
    auto& in = out.inputs;
    in.work_root = scratch / "work";
    fs::create_directories(in.work_root);
    in.options.sys_root = host;
    in.options.site = site;
    in.options.configs = {fs::path(BC_CONFIGS_DIR)};
    in.options.model_root = host / "model-root"; // the site's model wins; the root is still required
    in.options.sets = {"layout.name=\"qwen27b-tp3-q8\"", "memory.host_pool_mib=1024"};
    in.self = host / "bc-serve-test";
    if (!fs::exists(in.self)) fs::create_symlink(BC_SERVE_EXE, in.self);
    in.engine_binary = BC_FAKE_ENGINE;
    const Json facts = serving::detect(host);
    const auto resolved = resolve_fixture(in.options, facts);
    out.resolved_values = resolved.values;
    for (const Json& bdf : resolved.values.at("layout").at("engine_cards")) {
        const fs::path gtt = host / "sys/bus/pci/devices" / bdf.get<std::string>() / "mem_info_gtt_used";
        fs::create_directories(gtt.parent_path());
        std::ofstream(gtt) << "0\n";
    }
    const fs::path root = scratch / "native-root";
    fs::create_directories(root);
    out.yaml = root / "fleet.yaml";
    out.yaml_bytes = "# caf\xc3\xa9\r\ncards: [0]\r\nmodels:\r\n  tiny: {model: " + Json(model.string()).dump() +
                     "}\r\n";
    {
        std::ofstream yaml(out.yaml, std::ios::binary);
        yaml << out.yaml_bytes;
    }
    Json kernels = Json::array();
    const std::string kernel_text = resolved.values.at("engine").at("kernels").get<std::string>();
    for (size_t begin = 0; begin < kernel_text.size();) {
        const size_t end = kernel_text.find(',', begin);
        kernels.push_back(kernel_text.substr(begin, end == std::string::npos ? std::string::npos : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    std::reverse(kernels.begin(), kernels.end());
    const Json settings = {{"max_batched_tokens", 384}, {"checkpoint_interval", 1536}, {"kv_dtype", "bf16"},
                           {"ngram_placement", resolved.values.at("engine").at("ngram_placement")},
                           {"speculative_tokens", 2}, {"rope_scaling", "yarn:4"}, {"prefix_resume", "exact"},
                           {"prefill_interleave", 1}};
    ss::Env overrides = {
        {"TEST_NATIVE_INSPECT_TP", "3"},
        {"TEST_NATIVE_INSPECT_TP_WIRE", "exact"},
        {"TEST_NATIVE_INSPECT_XBIGCHERRY",
         Json{{"schema", 1}, {"description", "qwen27b-tp3-q8 fixture"},
              {"ranks", Json::array({{{"arch", "gfx1100"}, {"count", 2}}, {{"arch", "gfx1201"}, {"count", 1}}})},
              {"memory", {{"host-pool-mib", 0}, {"host-kv-mib", 0}}}}
             .dump()},
        {"TEST_NATIVE_INSPECT_RANK_DEVICES", resolved.values.at("layout").at("engine_cards").dump()},
        {"TEST_NATIVE_INSPECT_CARDS", "[0,1,2]"},
        {"TEST_NATIVE_INSPECT_MODEL", model.string()},
        {"TEST_NATIVE_INSPECT_EXPECT_YAML", out.yaml_bytes},
        {"TEST_NATIVE_EXPECT_ROOT", root.string()},
        {"TEST_NATIVE_INSPECT_PORT", std::to_string(port)},
        {"TEST_NATIVE_INSPECT_SETTINGS", settings.dump()},
        {"TEST_NATIVE_INSPECT_MAX_CONTEXT", std::to_string(resolved.values.at("service").at("max_context").get<int>())},
        {"TEST_NATIVE_INSPECT_MAX_SEQUENCES",
         std::to_string(resolved.values.at("service").at("max_sessions").get<int>())},
        {"TEST_NATIVE_INSPECT_KERNELS", kernels.dump()}};
    if (!behaviour.empty()) overrides["TEST_FAKE_BEHAVIOUR"] = behaviour.dump();
    in.native_admission = ss::NativeAdmissionRequest{out.yaml, root, std::move(overrides)};
    in.named_config = "qwen27b-tp3-q8";
    return out;
}

inline NativeSessionPlan native_helper_session_inputs(const fs::path& host, const fs::path& scratch, int port,
                                                      const Json& behaviour, const fs::path& runtime_home) {
    namespace ss = serving::session;
    fs::create_directories(scratch);
    // The resource ledger needs an actual bounded name table in the model's directory.
    const fs::path model = host / "helper-session-model.rad";
    if (!fs::exists(model)) {
        write_rad(model, 4, 16, 128, 128, {});
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
        if (!rad) throw std::runtime_error("cannot write the helper fixture model");
    }
    const fs::path home = runtime_home.empty() ? host / "session-build" / "radiance_home" : runtime_home;
    fs::create_directories(home / "helpers");
    if (!fs::exists(home / "helpers/r3_moe_worker")) fs::create_symlink(BC_FAKE_ENGINE, home / "helpers/r3_moe_worker");
    const fs::path site = scratch / "site.toml";
    write_file(site, "[engine]\nmodel = \"" + model.string() + "\"\nradiance_home = \"" + home.string() +
                         "\"\nport = " + std::to_string(port) + "\n");
    const fs::path heat = host / "writer.hot-first";
    if (!fs::exists(heat)) {
        std::vector<uint32_t> counts(128 * 16, 0);
        for (uint32_t l = 0; l < 4; ++l)
            for (uint32_t e = 0; e < 16; ++e) counts[l * 16 + e] = (e * 7 + l * 3) % 16 + 1;
        std::ofstream(heat, std::ios::binary)
            .write(reinterpret_cast<const char*>(counts.data()), static_cast<std::streamsize>(counts.size() * sizeof counts[0]));
    }
    NativeSessionPlan out;
    auto& in = out.inputs;
    in.work_root = scratch / "work";
    fs::create_directories(in.work_root);
    in.options.site = site;
    in.options.sys_root = host;
    in.options.configs = {fs::path(BC_CONFIGS_DIR)};
    in.options.model_root = host;
    in.options.sets = {"layout.name=\"flashnext-2xtx\"", "helper.combine_threads=1",
                       "handoff.heat=\"" + heat.string() + "\"",
                       "handoff.histogram=\"" + (host / "vram-histogram").string() + "\"",
                       "memory.helper_budgets_mib=[1,1]", "memory.host_pool_mib=2048", "memory.host_kv_mib=0"};
    in.self = BC_SERVE_EXE;
    in.engine_binary = BC_FAKE_ENGINE;
    in.helper_binary = BC_FAKE_ENGINE;
    in.named_config = "flashnext-2xtx";
    const Json facts = serving::detect(host);
    const auto resolved = resolve_fixture(in.options, facts);
    out.resolved_values = resolved.values;
    const auto& engine = resolved.values.at("engine");
    const auto& service = resolved.values.at("service");
    Json kernels = Json::array();
    {
        std::istringstream list(engine.at("kernels").get<std::string>());
        for (std::string kernel; std::getline(list, kernel, ',');) kernels.push_back(kernel);
    }
    for (const auto& gpu : facts.at("gpus"))
        write_file(host / "sys/bus/pci/devices" / gpu.at("bdf").get<std::string>() / "mem_info_gtt_used", "0\n");
    const fs::path root = scratch / "native-root";
    fs::create_directories(root);
    out.yaml = root / "fleet.yaml";
    out.yaml_bytes = "cards: [0]\nmodels:\n  tiny: {model: " + Json(model.string()).dump() + "}\n";
    {
        std::ofstream yaml(out.yaml, std::ios::binary);
        yaml << out.yaml_bytes;
    }
    ss::Env overrides = {
        {"TEST_NATIVE_INSPECT_EXPECT_YAML", out.yaml_bytes},
        {"TEST_NATIVE_EXPECT_ROOT", root.string()},
        {"TEST_NATIVE_INSPECT_MODEL", model.string()},
        {"TEST_NATIVE_INSPECT_RANK_DEVICES", resolved.values.at("layout").at("engine_cards").dump()},
        {"TEST_NATIVE_INSPECT_PORT", std::to_string(port)},
        {"TEST_NATIVE_INSPECT_MAX_CONTEXT", service.at("max_context").dump()},
        {"TEST_NATIVE_INSPECT_MAX_SEQUENCES", service.at("max_sessions").dump()},
        {"TEST_NATIVE_INSPECT_KERNELS", kernels.dump()},
        {"TEST_NATIVE_INSPECT_SETTINGS", Json({{"ngram_placement", engine.at("ngram_placement")}}).dump()},
        // flashnext-2xtx: the R9700 engine, both 7900 XTX as head helpers (as engines/radiance/configs/flashnext-2xtx.yaml)
        {"TEST_NATIVE_INSPECT_XBIGCHERRY",
         Json{{"schema", 1}, {"description", "flashnext-2xtx fixture"},
              {"ranks", Json::array({{{"arch", "gfx1201"}, {"count", 1}}})},
              {"helpers", Json::array({{{"arch", "gfx1100"}, {"count", "all"}, {"placement", "head"}}})}}
             .dump()}};
    if (!behaviour.empty()) overrides["TEST_FAKE_BEHAVIOUR"] = behaviour.dump();
    in.native_admission = ss::NativeAdmissionRequest{out.yaml, root, std::move(overrides)};
    in.helper_vram.read_used = [](const fs::path&) { return uint64_t{128} << 20; };
    return out;
}

/* The hooks of a plan step that is not run by an owner: it may wait forever and its power admission is a no-op. */
inline serving::session::Hooks standalone_plan_hooks() {
    serving::session::Hooks hooks;
    hooks.wait = [](std::chrono::milliseconds) { return true; };
    hooks.admit_power = [](const fs::path&, const serving::v2::Manifest&, serving::v2::PlanUse) { return serving::v2::Status{}; };
    hooks.selected_segment = [](const std::string&) {};
    return hooks;
}

} // namespace bc::test
