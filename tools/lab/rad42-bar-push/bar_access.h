#pragma once
// RAD50 lab-only common GPU/HSA allocation and PCI identity helpers.
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

__host__ __device__ inline uint32_t value_at(uint32_t idx, uint32_t gen) {
    uint32_t x = idx ^ (gen * 0x9e3779b9u);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__global__ void verify_words(const uint32_t* gpu_local,
                             uint32_t count, uint32_t gen, unsigned* errors) {
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    unsigned mismatches = 0;
    for (uint32_t i = id; i < count; i += stride)
        mismatches += gpu_local[i] != value_at(i, gen);
    if (mismatches) atomicAdd(errors, mismatches);
}

struct Inventory {
    std::vector<hsa_agent_t> cpus;
    std::vector<hsa_agent_t> gpus;
};
hsa_status_t collect_agent(hsa_agent_t a, void* ctx) {
    hsa_device_type_t kind{};
    const auto s = hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &kind);
    if (s != HSA_STATUS_SUCCESS) return s;
    auto& v = *static_cast<Inventory*>(ctx);
    if (kind == HSA_DEVICE_TYPE_CPU) v.cpus.push_back(a);
    if (kind == HSA_DEVICE_TYPE_GPU) v.gpus.push_back(a);
    return HSA_STATUS_SUCCESS;
}

struct PoolChoice {
    const Inventory* inv = nullptr;
    bool found = false;
    hsa_amd_memory_pool_t pool{};
    hsa_agent_t cpu{};
    size_t granule = 0;
};
hsa_status_t select_pool(hsa_amd_memory_pool_t p, void* raw) {
    auto& c = *static_cast<PoolChoice*>(raw);
    hsa_amd_segment_t segment{};
    hsa_amd_memory_pool_location_t loc{};
    bool can_alloc = false;
    size_t bytes = 0;
    if (hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment) != HSA_STATUS_SUCCESS ||
        hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_LOCATION, &loc) != HSA_STATUS_SUCCESS ||
        hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &can_alloc) != HSA_STATUS_SUCCESS ||
        hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SIZE, &bytes) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_ERROR;
    if (c.found || segment != HSA_AMD_SEGMENT_GLOBAL ||
        loc != HSA_AMD_MEMORY_POOL_LOCATION_GPU ||
        !can_alloc || bytes < 4u * 1024u * 1024u) return HSA_STATUS_SUCCESS;
    for (const hsa_agent_t cpu : c.inv->cpus) {
        hsa_amd_memory_pool_access_t access = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
        if (hsa_amd_agent_memory_pool_get_info(cpu, p,
                HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access) != HSA_STATUS_SUCCESS)
            return HSA_STATUS_ERROR;
        if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) continue;
        size_t granule = 0;
        if (hsa_amd_memory_pool_get_info(p,
                HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &granule) != HSA_STATUS_SUCCESS)
            return HSA_STATUS_ERROR;
        if (!granule) return HSA_STATUS_ERROR;
        c.pool = p;
        c.cpu = cpu;
        c.granule = granule;
        c.found = true;
        break;
    }
    return HSA_STATUS_SUCCESS;
}

bool parse_pci(const char* bdf, uint32_t* domain, uint32_t* id) {
    unsigned dom = 0, bus = 0, device = 0, function = 0;
    if (std::sscanf(bdf, "%x:%x:%x.%x", &dom, &bus, &device, &function) != 4 ||
        dom > 0xffffu || bus > 0xffu || device > 0x1fu || function > 7u)
        return false;
    *domain = dom;
    *id = (bus << 8) | (device << 3) | function;
    return true;
}

bool check_hip(hipError_t s, const char* op) {
    if (s == hipSuccess) return true;
    std::fprintf(stderr, "FAIL %s: %s\n", op, hipGetErrorString(s));
    return false;
}

} // namespace
