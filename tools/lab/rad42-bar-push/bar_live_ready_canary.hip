// RAD50 Gate 4 feasibility: LIVE GPU waiter before CPU BAR write + flag.
// Build only until Gate 1/2/2b succeed and the exclusive GPU lease is held.
// This experiment tests whether a CPU release to a HOST flag follows writes
// to separately mapped GPU VRAM in the GPU's observed order. It DOES NOT
// assume std::atomic_thread_fence suffices; a failure disproves that shortcut.
// A pass is a hardware observation, not a general HSA memory-model guarantee.
// No peer GPU, no work queue/serving modifications, no CPU-to-GPU kernel
// dispatch after the CPU's ready publication (waiter is armed in advance).
#include "bar_access.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>

namespace {
struct alignas(64) Control {
    uint32_t armed=0;
    uint32_t ready=0;
    uint32_t done=0;
    uint32_t errors=0;
};
__global__ void live_reader(const uint32_t* gpu_local, uint32_t words, uint32_t gen,
                            Control* ctrl) {
    if (blockIdx.x || threadIdx.x) return;
    // CPU must observe armed before it starts writing VRAM. No kernel
    // launch occurs after CPU publication; the GPU waits autonomously.
    __hip_atomic_store(&ctrl->armed,gen,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_SYSTEM);
    const uint64_t begin=clock64();
    uint32_t loops=0;
    while (__hip_atomic_load(&ctrl->ready,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_SYSTEM)!=gen) {
        __builtin_amdgcn_s_sleep(8);
        // Terminate on missing/stale publication rather than GPU hang.
        if (++loops>100000000u || clock64()-begin>3000000000ull) {
            __hip_atomic_store(&ctrl->errors,0xffffffffu,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_SYSTEM);
            __threadfence_system();
            __hip_atomic_store(&ctrl->done,gen,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_SYSTEM);
            return;
        }
    }
    uint32_t bad=0;
    for (uint32_t i=0;i<words;++i)
        bad += gpu_local[i] != value_at(i,gen);
    __hip_atomic_store(&ctrl->errors,bad,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_SYSTEM);
    __threadfence_system();
    __hip_atomic_store(&ctrl->done,gen,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_SYSTEM);
}
bool wait_host(const uint32_t* ptr,uint32_t gen,uint32_t ms) {
    const auto until=std::chrono::steady_clock::now()+std::chrono::milliseconds(ms);
    while (__atomic_load_n(ptr,__ATOMIC_ACQUIRE)!=gen) {
        if (std::chrono::steady_clock::now()>until) return false;
        // A genuinely asynchronous host waiter; no HIP sync or CPU futex.
        std::this_thread::yield();
    }
    return true;
}
bool select_device(int dev, Inventory* inventory, hsa_agent_t* gpu) {
    int count=0;
    if (!check_hip(hipGetDeviceCount(&count),"HIP count") ||
        dev<0 || dev>=count ||
        !check_hip(hipSetDevice(dev),"HIP select")) return false;
    char pci[64]{};
    uint32_t domain=0,bdf=0;
    if (!check_hip(hipDeviceGetPCIBusId(pci,sizeof(pci),dev),"HIP PCI") ||
        !parse_pci(pci,&domain,&bdf) ||
        hsa_iterate_agents(collect_agent,inventory)!=HSA_STATUS_SUCCESS)
        return false;
    unsigned matched=0;
    for (hsa_agent_t candidate:inventory->gpus) {
        uint32_t d=0,b=0;
        if (hsa_agent_get_info(candidate,(hsa_agent_info_t)HSA_AMD_AGENT_INFO_DOMAIN,&d)!=HSA_STATUS_SUCCESS ||
            hsa_agent_get_info(candidate,(hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID,&b)!=HSA_STATUS_SUCCESS)
            return false;
        if (domain==d && bdf==b) { *gpu=candidate; ++matched; }
    }
    std::printf("gpu=%d PCI=%s HSA-matches=%u\n",dev,pci,matched);
    return matched==1 && !inventory->cpus.empty();
}
bool exercise(hsa_agent_t gpu, const Inventory& inventory) {
    PoolChoice pool{&inventory};
    if (hsa_amd_agent_iterate_memory_pools(gpu,select_pool,&pool)!=HSA_STATUS_SUCCESS
        || !pool.found) return false;
    constexpr size_t kBytes=4u*1024u*1024u;
    if (pool.granule>std::numeric_limits<size_t>::max()-kBytes) return false;
    const size_t alloc=((kBytes+pool.granule-1)/pool.granule)*pool.granule;
    void* vram=nullptr;
    if (hsa_amd_memory_pool_allocate(pool.pool,alloc,0,&vram)!=HSA_STATUS_SUCCESS ||
        !vram) return false;
    Control* ctrl=nullptr;
    Control* dev_ctrl=nullptr;
    bool ok=false;
    do {
        if (hsa_amd_agents_allow_access(1,&pool.cpu,nullptr,vram)!=HSA_STATUS_SUCCESS)
            break;
        if (!check_hip(hipHostMalloc((void**)&ctrl,sizeof(Control),
                       hipHostMallocMapped|hipHostMallocCoherent),"control host allocation") ||
            !check_hip(hipHostGetDevicePointer((void**)&dev_ctrl,ctrl,0),"control device ptr"))
            break;
        bool success=true;
        for (uint32_t bytes : {uint32_t{4096},uint32_t{kBytes}}) {
            for (uint32_t gen=1;gen<=12;++gen) {
                // No reset during a running kernel. Reuse is *only* after
                // last done and synchronization on the previous iteration.
                __atomic_store_n(&ctrl->armed,0u,__ATOMIC_RELAXED);
                __atomic_store_n(&ctrl->ready,0u,__ATOMIC_RELAXED);
                __atomic_store_n(&ctrl->done,0u,__ATOMIC_RELAXED);
                __atomic_store_n(&ctrl->errors,0u,__ATOMIC_RELAXED);
                const uint32_t ticket=gen+((bytes==4096)?0u:100u);
                hipLaunchKernelGGL(live_reader,dim3(1),dim3(1),0,0,
                                   static_cast<const uint32_t*>(vram),
                                   bytes/4,ticket,dev_ctrl);
                if (!check_hip(hipGetLastError(),"start GPU waiter") ||
                    !wait_host(&ctrl->armed,ticket,3000)) {
                    std::fprintf(stderr,"FAIL: GPU never armed ready waiter\n");
                    success=false;break;
                }
                // The GPU is *already polling* and no kernel-launch or sync
                // can establish visibility after this point.
                auto* out=static_cast<uint32_t*>(vram);
                for (uint32_t i=0;i<bytes/4;++i) out[i]=value_at(i,ticket);
                // This fence is only a hypothesis! If posted BAR writes are
                // observed after the host-mapped flag, mismatches will show it.
                std::atomic_thread_fence(std::memory_order_seq_cst);
                __atomic_store_n(&ctrl->ready,ticket,__ATOMIC_RELEASE);
                if (!wait_host(&ctrl->done,ticket,3000)) {
                    std::fprintf(stderr,"FAIL: GPU did not acknowledge request\n");
                    success=false;break;
                }
                const uint32_t errors=__atomic_load_n(&ctrl->errors,__ATOMIC_ACQUIRE);
                if (!check_hip(hipDeviceSynchronize(),"drain live GPU waiter") || errors) {
                    std::fprintf(stderr,"FAIL bytes=%u gen=%u mismatches=%u\n",
                                 bytes,ticket,errors);
                    success=false;break;
                }
            }
            if (!success) break;
            std::printf("PASS live GPU waiting before CPU ready: bytes=%u generations=12\n",bytes);
        }
        ok=success;
    } while (false);
    // Even on an arm/done timeout the bounded kernel may still be exiting.
    // Never free host flags or HSA VRAM while it could be polling them.
    if (!check_hip(hipDeviceSynchronize(),"final kernel drain")) ok=false;
    if (ctrl) (void)hipHostFree(ctrl);
    if (hsa_amd_memory_pool_free(vram)!=HSA_STATUS_SUCCESS) ok=false;
    return ok;
}
} // namespace

int main(int argc,char** argv) {
    if (argc!=3 || std::strcmp(argv[1],"--device")) {
        std::fputs("usage: bar_live_ready_canary --device HIP_ORDINAL\n",stderr);
        return 2;
    }
    char* end=nullptr;
    const long dev=std::strtol(argv[2],&end,10);
    if (!argv[2][0] || *end || dev<0 || dev>127) return 2;
    if (hsa_init()!=HSA_STATUS_SUCCESS) return 2;
    Inventory inventory;
    hsa_agent_t gpu{};
    bool ok=false;
    if (select_device((int)dev,&inventory,&gpu)) ok=exercise(gpu,inventory);
    if (hsa_shut_down()!=HSA_STATUS_SUCCESS) ok=false;
    if (!ok) std::fputs("STOP: concurrent GPU-live BAR publication NOT proven\n",stderr);
    return ok?0:3;
}
