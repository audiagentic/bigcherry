// RAD42 BAR push experiment. HSA pool/access discovery ONLY; no allocation or DMA.
// Run only under the project's gpu lease. This does not prove CPU writes work.
// Build: g++ -std=c++17 -O2 pool_inventory.cpp -lhsa-runtime64 -o pool_inventory
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
struct Agents {
    std::vector<hsa_agent_t> cpu;
    std::vector<hsa_agent_t> gpu;
};
hsa_status_t collect(hsa_agent_t agent, void* opaque) {
    hsa_device_type_t type{};
    const hsa_status_t rc = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (rc != HSA_STATUS_SUCCESS) return rc;
    auto& agents = *static_cast<Agents*>(opaque);
    if (type == HSA_DEVICE_TYPE_CPU) agents.cpu.push_back(agent);
    if (type == HSA_DEVICE_TYPE_GPU) agents.gpu.push_back(agent);
    return HSA_STATUS_SUCCESS;
}
struct PoolContext {
    const Agents* agents;
    unsigned ordinal;
    unsigned count = 0;
    unsigned candidates = 0;
};
hsa_status_t show_pool(hsa_amd_memory_pool_t pool, void* opaque) {
    auto& c = *static_cast<PoolContext*>(opaque);
    hsa_amd_segment_t segment{};
    hsa_amd_memory_pool_location_t location{};
    uint32_t flags = 0;
    bool alloc = false;
    size_t size = 0;
    if (hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment) != HSA_STATUS_SUCCESS ||
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_LOCATION, &location) != HSA_STATUS_SUCCESS ||
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc) != HSA_STATUS_SUCCESS ||
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SIZE, &size) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_ERROR;
    if (segment == HSA_AMD_SEGMENT_GLOBAL &&
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_ERROR;
    bool cpu_accessible = false;
    for (hsa_agent_t cpu : c.agents->cpu) {
        hsa_amd_memory_pool_access_t access = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
        if (hsa_amd_agent_memory_pool_get_info(cpu, pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access) != HSA_STATUS_SUCCESS)
            return HSA_STATUS_ERROR;
        if (access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) cpu_accessible = true;
    }
    const bool candidate = segment == HSA_AMD_SEGMENT_GLOBAL &&
        location == HSA_AMD_MEMORY_POOL_LOCATION_GPU && alloc && cpu_accessible;
    ++c.count;
    if (candidate) ++c.candidates;
    std::printf("gpu=%u pool=%u location=%s size_MiB=%zu alloc=%d flags=0x%x cpu_accessible=%d candidate=%d\n",
        c.ordinal, c.count, location == HSA_AMD_MEMORY_POOL_LOCATION_GPU ? "gpu" : "cpu",
        size / (1024 * 1024), (int)alloc, flags, (int)cpu_accessible, (int)candidate);
    return HSA_STATUS_SUCCESS;
}
}  // namespace

int main() {
    if (hsa_init() != HSA_STATUS_SUCCESS) {
        std::fprintf(stderr, "HSA init failed\n");
        return 2;
    }
    Agents a;
    hsa_status_t rc = hsa_iterate_agents(collect, &a);
    unsigned candidates = 0;
    for (unsigned i = 0; rc == HSA_STATUS_SUCCESS && i < a.gpu.size(); ++i) {
        char name[64] = {};
        uint32_t bdf = 0, domain = 0;
        rc = hsa_agent_get_info(a.gpu[i], HSA_AGENT_INFO_NAME, name);
        if (rc != HSA_STATUS_SUCCESS) break;
        // Report physical identity; don't assume HIP ordinal equals HSA ordinal.
        (void)hsa_agent_get_info(a.gpu[i], (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &bdf);
        (void)hsa_agent_get_info(a.gpu[i], (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DOMAIN, &domain);
        std::printf("gpu=%u name=%s domain=%04x bdf=%04x cpus=%zu\n",
                    i, name, domain, bdf, a.cpu.size());
        PoolContext c{&a, i};
        rc = hsa_amd_agent_iterate_memory_pools(a.gpu[i], show_pool, &c);
        candidates += c.candidates;
    }
    const hsa_status_t shutdown = hsa_shut_down();
    if (rc != HSA_STATUS_SUCCESS || shutdown != HSA_STATUS_SUCCESS || a.gpu.empty()) {
        std::fprintf(stderr, "HSA pool discovery failed\n");
        return 2;
    }
    if (candidates == 0) {
        std::fprintf(stderr, "No CPU-accessible GPU-local pool reported: STOP (no BAR-push assumption).\n");
        return 3;
    }
    std::printf("candidate_pools=%u (discovery only; no CPU write proven)\n", candidates);
    return 0;
}
