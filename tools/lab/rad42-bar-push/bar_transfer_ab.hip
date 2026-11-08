// RAD50 Gate 3 readiness: strict standalone *single-process* transfer A/B.
// Compile only until Gate 2 passes + an exclusive GPU lease is held.
// NOT a full engine model: the baseline engine adds rows directly from
// mapped host memory rather than copying them into VRAM; this microbench
// measures transfer physics and correctness only, not critical path.
// CPU BAR data is generated in pinned host memory; GPU pull uses a
// mapped-host kernel load with the SAME source and GPU-local destination.
// Per-iteration p50/p95 include launch/thread-start overhead and the full
// GPU verification; copy-only timings exclude the verification.
// One CPU-thread group is spawned for each push; this is deliberately
// pessimistic versus a persistent combine worker pool, and must be labelled.
// Run only after checking pool inventory and canary on the same PCI device.
#include "bar_access.h"
#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {
__global__ void gpu_pull_to_vram(const uint4* mapped_host, uint4* vram, size_t n) {
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i=t; i<n; i+=stride) vram[i] = mapped_host[i];
}
double now_us() {
    using Clock=std::chrono::steady_clock;
    return std::chrono::duration<double,std::micro>(Clock::now().time_since_epoch()).count();
}
double quantile(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(),v.end());
    return v[static_cast<size_t>((v.size()-1)*q)];
}
bool verify(unsigned* errors, const uint32_t* vram, size_t bytes, uint32_t generation) {
    if (!check_hip(hipMemset(errors,0,sizeof(unsigned)), "reset error count"))
        return false;
    hipLaunchKernelGGL(verify_words,dim3(256),dim3(256),0,0,
                       vram,(uint32_t)(bytes / 4),generation,errors);
    if (!check_hip(hipGetLastError(),"verification launch") ||
        !check_hip(hipDeviceSynchronize(),"verification completion"))
        return false;
    unsigned bad=0;
    if (!check_hip(hipMemcpy(&bad,errors,sizeof(bad),hipMemcpyDeviceToHost),
                   "transfer verification count copy") || bad) {
        std::fprintf(stderr,"FAIL bytes=%zu generation=%u mismatches=%u\n",
                     bytes,generation,bad);
        return false;
    }
    return true;
}
bool try_device(int device, hsa_agent_t* selected, Inventory* agents) {
    int count=0;
    if (!check_hip(hipGetDeviceCount(&count),"device count") ||
        device<0 || device>=count ||
        !check_hip(hipSetDevice(device),"select device")) return false;
    char pci[64]{};
    uint32_t domain=0, bdf=0;
    if (!check_hip(hipDeviceGetPCIBusId(pci,sizeof(pci),device),"PCI ID") ||
        !parse_pci(pci,&domain,&bdf)) return false;
    if (hsa_iterate_agents(collect_agent,agents)!=HSA_STATUS_SUCCESS ||
        agents->cpus.empty()) return false;
    unsigned matches=0;
    for (hsa_agent_t agent : agents->gpus) {
        uint32_t d=0, b=0;
        if (hsa_agent_get_info(agent,(hsa_agent_info_t)HSA_AMD_AGENT_INFO_DOMAIN,&d)!=HSA_STATUS_SUCCESS ||
            hsa_agent_get_info(agent,(hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID,&b)!=HSA_STATUS_SUCCESS)
            return false;
        if (d==domain && b==bdf) { ++matches; *selected=agent; }
    }
    if (matches!=1) return false;
    std::printf("gpu=%d pci=%s HSA-PCI-matched=1\n",device,pci);
    return true;
}
bool experiment(const PoolChoice& choice, unsigned iterations) {
    constexpr size_t kMax=21u*1024u*1024u; // RAD40 full two-helper plane
    if (choice.granule > std::numeric_limits<size_t>::max()-kMax) return false;
    const size_t bytes=((kMax+choice.granule-1)/choice.granule)*choice.granule;
    void* vram=nullptr;
    if (hsa_amd_memory_pool_allocate(choice.pool,bytes,0,&vram)!=HSA_STATUS_SUCCESS || !vram)
        return false;
    void* host=nullptr;
    void* host_dev=nullptr;
    unsigned* errors=nullptr;
    bool ok=false;
    do {
        if (hsa_amd_agents_allow_access(1,&choice.cpu,nullptr,vram)!=HSA_STATUS_SUCCESS)
            break;
        if (!check_hip(hipHostMalloc(&host,kMax,hipHostMallocMapped|hipHostMallocCoherent),"host source") ||
            !check_hip(hipHostGetDevicePointer(&host_dev,host,0),"host device pointer") ||
            !check_hip(hipMalloc(&errors,sizeof(unsigned)),"device errors")) break;
        const std::vector<size_t> cases={4096,128u*1024u,256u*1024u,
                                         1024u*1024u,5u*1024u*1024u,
                                         10u*1024u*1024u,kMax};
        for (size_t n : cases) {
            for (int mode=0; mode<2; ++mode) {
                for (int threads : {1,2,4}) {
                    if (mode==1 && threads>1) continue; // GPU pull: one GPU stream
                    std::vector<double> transfer, complete;
                    for (uint32_t iter=1; iter<iterations+4; ++iter) {
                        const uint32_t gen=(uint32_t)(n >> 12) ^ (mode<<20) ^
                                           (threads<<24) ^ (iter*47u);
                        auto* src=static_cast<uint32_t*>(host);
                        for (size_t i=0;i<n/4;++i) src[i]=value_at((uint32_t)i,gen);
                        const double begin=now_us();
                        if (mode==0) {
                            // CPU writes directly to a CPU-mapped GPU-local pool;
                            // no hipMemcpyAsync (which itself performs GPU pull).
                            std::vector<std::thread> pool;
                            for (int k=0;k<threads;++k) {
                                const size_t from=n*k/threads & ~(size_t)15;
                                const size_t to=(n*(k+1)/threads) & ~(size_t)15;
                                pool.emplace_back([=] {
                                    std::memcpy(static_cast<uint8_t*>(vram)+from,
                                                static_cast<const uint8_t*>(host)+from,to-from);
                                });
                            }
                            for (auto& t : pool) t.join();
                            std::atomic_thread_fence(std::memory_order_seq_cst);
                        } else {
                            hipLaunchKernelGGL(gpu_pull_to_vram,dim3(256),dim3(256),0,0,
                                               static_cast<const uint4*>(host_dev),
                                               static_cast<uint4*>(vram),n/16);
                            if (!check_hip(hipGetLastError(),"GPU pull launch") ||
                                !check_hip(hipDeviceSynchronize(),"GPU pull completion"))
                                goto cleanup;
                        }
                        const double transfer_done=now_us();
                        if (!verify(errors,static_cast<const uint32_t*>(vram),n,gen))
                            goto cleanup;
                        const double verified_done=now_us();
                        if (iter>3) {
                            transfer.push_back(transfer_done-begin);
                            complete.push_back(verified_done-begin);
                        }
                    }
                    const double copy_p50=quantile(transfer,.5);
                    std::printf("mode=%s bytes=%zu cpu_threads=%d n=%u "
                                "transfer_p50_us=%.2f transfer_p95_us=%.2f "
                                "verified_p50_us=%.2f verified_p95_us=%.2f "
                                "transfer_GBps=%.3f\n",
                                mode==0?"cpu-bar":"gpu-pull",n,threads,iterations,
                                copy_p50,quantile(transfer,.95),
                                quantile(complete,.5),quantile(complete,.95),
                                copy_p50>0 ? n/(copy_p50*1000.0) : 0);
                }
            }
        }
        ok=true;
    } while (false);
cleanup:
    if (errors) (void)hipFree(errors);
    if (host) (void)hipHostFree(host);
    if (hsa_amd_memory_pool_free(vram)!=HSA_STATUS_SUCCESS) ok=false;
    return ok;
}
} // namespace

int main(int argc,char** argv) {
    if (argc!=5 || std::strcmp(argv[1],"--device") ||
        std::strcmp(argv[3],"--iterations")) {
        std::fputs("usage: bar_transfer_ab --device HIP_ORDINAL --iterations 10..200\n",stderr);
        return 2;
    }
    char* end=nullptr;
    const long dev=std::strtol(argv[2],&end,10);
    if (!argv[2][0] || *end || dev<0 || dev>127) return 2;
    const long iters=std::strtol(argv[4],&end,10);
    if (!argv[4][0] || *end || iters<10 || iters>200) return 2;
    if (hsa_init()!=HSA_STATUS_SUCCESS) return 2;
    Inventory inv;
    hsa_agent_t gpu{};
    bool ok=false;
    if (try_device((int)dev,&gpu,&inv)) {
        PoolChoice choice{&inv};
        if (hsa_amd_agent_iterate_memory_pools(gpu,select_pool,&choice)==HSA_STATUS_SUCCESS
            && choice.found)
            ok=experiment(choice,(unsigned)iters);
    }
    if (hsa_shut_down()!=HSA_STATUS_SUCCESS) ok=false;
    if (!ok) std::fputs("STOP: GPU-local CPU BAR pull A/B is unsupported or failed\n",stderr);
    return ok?0:3;
}
