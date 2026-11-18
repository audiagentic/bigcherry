// RSC02 step 6 / RSC04 typed-mode parity. Card-free real native bc-serve
// detection, named resolver and strict-v2 manifest writer, no fake planner.
// Fake only the OS KFD/PCI/proc tree; heat uses the actual padded producer shape.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../moe-remote/heat_publication.h"
#include "serving.h"
#include "test_rad_fixture.h"

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef BC_SERVE_EXE
#error "BC_SERVE_EXE must be the compiled native bc-serve"
#endif
#ifndef BC_CONFIGS_DIR
#error "BC_CONFIGS_DIR must be the shipped RSC12 catalogue"
#endif
namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {
void write(const fs::path& file, const std::string& text) {
    fs::create_directories(file.parent_path());
    std::ofstream stream(file, std::ios::binary);
    if (!stream || !(stream << text))
        throw std::runtime_error("cannot write fixture " + file.string());
}
struct Fixture {
    fs::path root = fs::current_path() / "bc-typed-settings-fixture";
    fs::path model = root / "model.rad";
    fs::path heat = root / "route.hot-first";
    fs::path site = root / "site.toml";
    Fixture() {
        fs::remove_all(root);
        write(root / "sys/class/kfd/kfd/topology/nodes/0/properties", "gfx_target_version 0\n");
        const std::vector<int> versions{110000, 110000, 120001};
        const std::vector<int> buses{3, 6, 9};
        const std::vector<int> simds{192, 192, 128};
        const std::vector<int> vrams{24560, 24560, 32624};
        for (size_t i = 0; i < versions.size(); ++i) {
            write(root / "sys/class/kfd/kfd/topology/nodes" / std::to_string(i + 1) / "properties",
                  "gfx_target_version " + std::to_string(versions[i]) + "\nsimd_count " +
                      std::to_string(simds[i]) + "\nlocation_id " + std::to_string(buses[i] * 256) +
                      "\ndomain 0\ndrm_render_minor " + std::to_string(128 + i) + "\n");
            char bdf[24];
            std::snprintf(bdf, sizeof(bdf), "0000:%02x:00.0", buses[i]);
            const auto pci = root / "sys/bus/pci/devices" / bdf;
            write(pci / "mem_info_vram_total", std::to_string(uint64_t(vrams[i]) * 1048576));
            write(pci / "mem_info_vram_used", std::to_string(uint64_t(128) * 1048576));
            write(pci / "current_link_speed", "16.0 GT/s PCIe\n");
            write(pci / "max_link_speed", "16.0 GT/s PCIe\n");
            write(pci / "current_link_width", "16\n");
            write(pci / "max_link_width", "16\n");
            write(pci / "numa_node", "0\n");
        }
        write(root / "proc/meminfo", "MemTotal:       134217728 kB\nMemAvailable:    94371840 kB\n"
                                     "SwapTotal:       8388608 kB\n");
        fs::create_directories(root / "dev/shm");
        write(root / "proc/self/cgroup", "0::/agents/serve\n");
        write(root / "sys/fs/cgroup/agents/serve/memory.max", std::to_string(uint64_t(64) << 30) + "\n");
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
        bc::test::write_rad(model, 4, 16, 128, 128, {});
        // The same uint32[kMaxLayers][model.experts] layout produced by the
        // engine and consumed by both the worker and the runtime heat guard.
        std::vector<uint32_t> counts(size_t(r3::moe_shm::kMaxLayers) * 16);
        for (size_t i = 0; i < counts.size(); ++i)
            counts[i] = uint32_t(i % 16) + 1;
        std::ofstream stream(heat, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(counts.data()),
                     std::streamsize(counts.size() * sizeof(uint32_t)));
        if (!stream)
            throw std::runtime_error("route heat fixture failed");
        write(site, "[engine]\nmodel = \"" + model.string() + "\"\nradiance_home = \"/runtime\"\n");
    }
    ~Fixture() {
        std::error_code ec;
        fs::remove_all(root, ec);
    }
};
struct CommandResult {
    int status;
    std::string output;
};
CommandResult run(const std::vector<std::string>& args) {
    int fd[2];
    if (::pipe(fd) != 0)
        throw std::runtime_error("pipe failed");
    const pid_t child = ::fork();
    if (child < 0)
        throw std::runtime_error("fork failed");
    if (child == 0) {
        ::close(fd[0]);
        if (::dup2(fd[1], STDOUT_FILENO) == -1 || ::dup2(fd[1], STDERR_FILENO) == -1)
            ::_exit(127);
        ::close(fd[1]);
        std::vector<char*> argv{const_cast<char*>(BC_SERVE_EXE)};
        for (const std::string& arg : args)
            argv.push_back(const_cast<char*>(arg.c_str()));
        argv.push_back(nullptr);
        ::execv(BC_SERVE_EXE, argv.data());
        ::_exit(127);
    }
    ::close(fd[1]);
    std::string output;
    char buf[2048];
    for (;;) {
        const ssize_t n = ::read(fd[0], buf, sizeof buf);
        if (n == 0)
            break;
        if (n < 0) {
            if (errno == EINTR)
                continue;
            ::close(fd[0]);
            (void)::waitpid(child, nullptr, 0);
            throw std::runtime_error("read failed");
        }
        output.append(buf, size_t(n));
    }
    ::close(fd[0]);
    int status = 0;
    if (::waitpid(child, &status, 0) != child)
        throw std::runtime_error("waitpid failed");
    return {WIFEXITED(status) ? WEXITSTATUS(status) : -1, output};
}
void verify(const Fixture& f) {
    const std::vector<std::string> common{"--configs",    BC_CONFIGS_DIR,  "--site",     f.site.string(),
                                          "--model-root", f.root.string(), "--sys-root", f.root.string()};
    const std::vector<std::string> values{
        "engine.port=47001", "handoff.heat=\"" + f.heat.string() + "\"", "handoff.histogram=\"/heat/lab.live\"",
        "handoff.histogram_period_s=3", "engine.kv_cache_dtype=\"bf16\"", "engine.ngram_placement=\"disk\"",
        "engine.num_speculative_tokens=0", "service.max_context=8192", "memory.host_pool_mib=4096",
        "memory.host_kv_mib=0", "memory.helper_budgets_mib=[8192,8192]",
        // CPU parallelism follows the CI host, not the fake GPUs.
        "helper.combine_threads=" + std::to_string(std::max(1u, std::thread::hardware_concurrency()))};
    auto append = [&](std::vector<std::string>& args) {
        args.insert(args.end(), common.begin(), common.end());
        for (const auto& value : values) {
            args.push_back("--set");
            args.push_back(value);
        }
    };
    std::vector<std::string> explain{"explain", "flashnext-2xtx", "--json"};
    append(explain);
    const auto inspected = run(explain);
    if (inspected.status != 0)
        throw std::runtime_error("explain CONFIG: " + inspected.output);
    const Json result = Json::parse(inspected.output);
    const Json& v = result.at("values");
    assert(v.at("engine").at("port") == 47001);
    assert(v.at("handoff").at("heat") == f.heat.string());
    assert(v.at("handoff").at("histogram") == "/heat/lab.live");
    assert(v.at("handoff").at("histogram_period_s") == 3);
    assert(v.at("engine").at("kv_cache_dtype") == "bf16");
    assert(v.at("engine").at("ngram_placement") == "disk");
    assert(v.at("engine").at("num_speculative_tokens") == 0);
    assert(v.at("service").at("max_context") == 8192);
    assert(v.at("memory").at("host_pool_mib") == 4096);
    assert(v.at("memory").at("helper_budgets_mib") == Json::array({8192, 8192}));
    for (const char* key :
         {"engine.port", "handoff.heat", "handoff.histogram", "handoff.histogram_period_s",
          "engine.kv_cache_dtype", "engine.ngram_placement", "engine.num_speculative_tokens",
          "service.max_context", "memory.host_pool_mib", "memory.host_kv_mib", "memory.helper_budgets_mib"})
        assert(result.at("provenance").at(key).at("origin") == "cli");

    const fs::path output = f.root / "unmeasured.json";
    std::vector<std::string> writer{"--write-manifest",
                                    output.string(),
                                    "--unmeasured",
                                    "--power-profile",
                                    "stock",
                                    "--set",
                                    "layout.name=\"flashnext-2xtx\""};
    append(writer);
    const auto emitted = run(writer);
    if (emitted.status != 0)
        throw std::runtime_error("strict native manifest writer: " + emitted.output);
    std::ifstream stream(output);
    if (!stream)
        throw std::runtime_error("writer did not publish a manifest");
    const Json m = Json::parse(stream);
    const Json& spec = m.at("spec");
    assert(spec.at("processes").at("engine").at("server").at("port") == 47001);
    assert(spec.at("intent").at("service").at("max_context_tokens") == 8192);
    const Json& execution = spec.at("processes").at("engine").at("execution");
    assert(execution.at("kv_cache_dtype") == "bf16");
    assert(execution.at("ngram_placement") == "disk");
    assert(execution.at("num_speculative_tokens") == 0);
    assert(spec.at("handoff").at("placement").at("heat_file") == f.heat.string());
    assert(spec.at("handoff").at("telemetry").at("histogram_file") == "/heat/lab.live");

    // Unknown typed fields and deprecated shell syntax are not silently coerced.
    for (const char* invalid : {"engine.ngram_device=0", "engine.port=0", "handoff.histogram_period_s=-1",
                                "engine.kv_cache_dtype=\"invalid\"", "engine.ngram_placement=\"invalid\"",
                                "engine.num_speculative_tokens=-1", "service.max_context=0",
                                "memory.host_pool_mib=-1", "memory.helper_budgets_mib=[-1]"}) {
        std::vector<std::string> args{"explain", "flashnext-2xtx", "--json"};
        args.insert(args.end(), common.begin(), common.end());
        args.push_back("--set");
        args.push_back(invalid);
        assert(run(args).status != 0);
    }
    for (const char* alias : {"--port", "--heat", "--record-heat", "--record-period", "--kv-dtype",
                              "--num-spec", "--max-model-len", "--budget", "--ngram-device"}) {
        std::vector<std::string> args{"explain", "flashnext-2xtx"};
        args.insert(args.end(), common.begin(), common.end());
        args.push_back(alias);
        args.push_back("1");
        assert(run(args).status != 0);
    }
}
} // namespace

int main() {
    try {
        Fixture fixture;
        verify(fixture);
        std::cout << "native typed settings: actual resolver and strict writer PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "typed settings: " << e.what() << '\n';
        return 1;
    }
}
