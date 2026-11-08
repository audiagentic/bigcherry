// RAD50 gate 2: single-process CPU -> *GPU-local HSA memory* -> GPU verify.
// EXPERIMENT ONLY. No server/worker, IPC, peer access, or protocol integration.
// Must be run with the project GPU lease; no GPU execution during build.
// Build: hipcc -O2 -std=c++17 --offload-arch=gfx1100,gfx1201
//          bar_write_canary.hip -lhsa-runtime64 -o bar_write_canary
// Run:   ./bar_write_canary --device <HIP-visible-ordinal>
// Safety: refusal if HSA cannot match the HIP GPU PCI identity or if no
// CPU-accessible, HSA GPU-local pool exists; no host fallback masquerading as VRAM.
// This proves only serial launch-after-CPU-store correctness, NOT a real
// concurrent GPU/CPU published-generation protocol. Do not treat as Gate 4.
#include "bar_access.h"

namespace {

bool exercise(hsa_agent_t selected, const Inventory& inventory, int device) {
    PoolChoice choice{&inventory};
    if (hsa_amd_agent_iterate_memory_pools(selected, select_pool, &choice) != HSA_STATUS_SUCCESS ||
        !choice.found) {
        std::fprintf(stderr, "BLOCKED: no CPU-accessible GPU-local allocation pool\n");
        return false;
    }
    constexpr size_t max_bytes = 4u * 1024u * 1024u;
    if (choice.granule > SIZE_MAX - max_bytes) return false;
    const size_t alloc_bytes = ((max_bytes + choice.granule - 1) / choice.granule) * choice.granule;
    void* ptr = nullptr;
    if (hsa_amd_memory_pool_allocate(choice.pool, alloc_bytes, 0, &ptr) != HSA_STATUS_SUCCESS || !ptr) {
        std::fprintf(stderr, "FAIL: GPU-local HSA allocation rejected\n");
        return false;
    }
    unsigned* device_errors = nullptr;
    bool ok = false;
    do {
        // Explicitly grant the CPU access. A CPU-accessible pool is not enough
        // to conclude coherent CPU writes; GPU verification below decides.
        if (hsa_amd_agents_allow_access(1, &choice.cpu, nullptr, ptr) != HSA_STATUS_SUCCESS) {
            std::fprintf(stderr, "FAIL: HSA CPU access grant rejected\n");
            break;
        }
        if (!check_hip(hipMalloc(&device_errors, sizeof(unsigned)), "error-count allocation")) break;
        std::printf("device=%d pool=GPU-local CPU access granted allocation=%zu bytes\n", device, alloc_bytes);
        bool succeeded = true;
        for (const size_t bytes : {size_t{4096}, max_bytes}) {
            for (uint32_t generation = 1; generation <= 12; ++generation) {
                // This is an actual host CPU write through HSA's CPU-accessible
                // mapping. The GPU consumes via a kernel after launch ordering.
                uint32_t* cpu = static_cast<uint32_t*>(ptr);
                const uint32_t words = static_cast<uint32_t>(bytes / sizeof(uint32_t));
                for (uint32_t i = 0; i < words; ++i) cpu[i] = value_at(i, generation);
                std::atomic_thread_fence(std::memory_order_seq_cst);
                if (!check_hip(hipMemset(device_errors, 0, sizeof(unsigned)), "error-count reset")) {
                    succeeded = false;
                    break;
                }
                hipLaunchKernelGGL(verify_words, dim3(256), dim3(256), 0, 0,
                                   static_cast<const uint32_t*>(ptr), words,
                                   generation, device_errors);
                if (!check_hip(hipGetLastError(), "GPU verify launch") ||
                    !check_hip(hipDeviceSynchronize(), "GPU verify synchronize")) {
                    succeeded = false;
                    break;
                }
                unsigned bad = 0;
                if (!check_hip(hipMemcpy(&bad, device_errors, sizeof(bad), hipMemcpyDeviceToHost),
                               "error-count copy") || bad) {
                    std::fprintf(stderr, "FAIL bytes=%zu generation=%u mismatches=%u\n",
                                 bytes, generation, bad);
                    succeeded = false;
                    break;
                }
            }
            if (!succeeded) break;
            std::printf("PASS bytes=%zu generations=12 full GPU-verified coverage\n", bytes);
        }
        ok = succeeded;
    } while (false);
    if (device_errors) (void)hipFree(device_errors);
    if (hsa_amd_memory_pool_free(ptr) != HSA_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: GPU-local allocation cleanup\n");
        ok = false;
    }
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3 || std::strcmp(argv[1], "--device")) {
        std::fprintf(stderr, "usage: bar_write_canary --device HIP_VISIBLE_ORDINAL\n");
        return 2;
    }
    char* end = nullptr;
    long requested = std::strtol(argv[2], &end, 10);
    if (!argv[2][0] || *end || requested < 0 || requested > 127) {
        std::fprintf(stderr, "invalid device ordinal\n");
        return 2;
    }
    int count = 0;
    if (!check_hip(hipGetDeviceCount(&count), "device count") || requested >= count ||
        !check_hip(hipSetDevice(static_cast<int>(requested)), "select HIP device")) return 2;
    char pci[64]{};
    if (!check_hip(hipDeviceGetPCIBusId(pci, sizeof(pci), static_cast<int>(requested)), "HIP PCI ID"))
        return 2;
    uint32_t target_domain = 0, target_id = 0;
    if (!parse_pci(pci, &target_domain, &target_id)) {
        std::fprintf(stderr, "FAIL: unparseable PCI ID, refusing ordinal-only selection\n");
        return 2;
    }
    if (hsa_init() != HSA_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: HSA init\n");
        return 2;
    }
    Inventory all;
    hsa_status_t rc = hsa_iterate_agents(collect_agent, &all);
    hsa_agent_t selected{};
    unsigned matches = 0;
    for (hsa_agent_t a : all.gpus) {
        uint32_t dom = 0, bdf = 0;
        if (hsa_agent_get_info(a, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DOMAIN, &dom) != HSA_STATUS_SUCCESS ||
            hsa_agent_get_info(a, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &bdf) != HSA_STATUS_SUCCESS) {
            rc = HSA_STATUS_ERROR;
            break;
        }
        if (dom == target_domain && bdf == target_id) {
            ++matches;
            selected = a;
        }
    }
    bool success = false;
    if (rc == HSA_STATUS_SUCCESS && matches == 1 && !all.cpus.empty()) {
        std::printf("HIP ordinal=%ld PCI=%s matched HSA GPU\n", requested, pci);
        success = exercise(selected, all, static_cast<int>(requested));
    } else {
        std::fprintf(stderr, "FAIL: HSA GPU/CPU agent discovery or unique HIP PCI match (matches=%u)\n", matches);
    }
    if (hsa_shut_down() != HSA_STATUS_SUCCESS) success = false;
    return success ? 0 : 3;
}
