// RAD50 VMM capability gate, not a replacement for IPC data correctness.
// Tests GPU-local ROCr VMM handle/export, local CPU+GPU mapping and host VMA.
// No CPU data stores, GPU kernels, subprocess or peer-GPU access.
// Requires the exclusive GPU lease. A PASS permits a later cross-process
// DMA-BUF import canary; it does NOT satisfy Gate 2b.
#include "bar_access.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <unistd.h>

namespace {
bool hsa_ok(const char* step,hsa_status_t status) {
    const char* detail=nullptr;
    (void)hsa_status_string(status,&detail);
    std::fprintf(stderr,"RAD50 VMM %s: %d %s\n",step,(int)status,
                 detail?detail:"unknown");
    return status==HSA_STATUS_SUCCESS;
}
bool writable_vma(const void* addr,size_t bytes) {
    const uintptr_t a=reinterpret_cast<uintptr_t>(addr);
    if (!addr || bytes>UINTPTR_MAX-a) return false;
    FILE* f=std::fopen("/proc/self/maps","r");
    if (!f) return false;
    bool yes=false;
    char line[512]{};
    while (std::fgets(line,sizeof(line),f)) {
        unsigned long lo=0,hi=0;
        char perm[5]{};
        if (std::sscanf(line,"%lx-%lx %4s",&lo,&hi,perm)==3 &&
            a>=lo && a+bytes<=hi && perm[0]=='r' && perm[1]=='w') {
            yes=true;break;
        }
    }
    std::fclose(f);
    return yes;
}
bool run(int dev) {
    int count=0;
    if (!check_hip(hipGetDeviceCount(&count),"HIP device count") ||
        dev<0 || dev>=count ||
        !check_hip(hipSetDevice(dev),"HIP select")) return false;
    char pci[64]{};
    uint32_t domain=0,bdf=0;
    if (!check_hip(hipDeviceGetPCIBusId(pci,sizeof(pci),dev),"HIP PCI") ||
        !parse_pci(pci,&domain,&bdf)) return false;
    if (!hsa_ok("init",hsa_init())) return false;
    Inventory inv;
    hsa_agent_t gpu{};
    bool ok=false;
    hsa_amd_vmem_alloc_handle_t handle{};
    bool handle_live=false,va_reserved=false,va_mapped=false;
    void* va=nullptr;
    int dmabuf=-1;
    size_t size=0;
    do {
        if (!hsa_ok("enumerate",hsa_iterate_agents(collect_agent,&inv)) ||
            inv.cpus.empty()) break;
        unsigned matches=0;
        for (auto agent:inv.gpus) {
            uint32_t d=0,b=0;
            if (hsa_agent_get_info(agent,(hsa_agent_info_t)HSA_AMD_AGENT_INFO_DOMAIN,&d)!=HSA_STATUS_SUCCESS ||
                hsa_agent_get_info(agent,(hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID,&b)!=HSA_STATUS_SUCCESS)
                break;
            if (d==domain && b==bdf) { gpu=agent; ++matches; }
        }
        if (matches!=1) {std::fprintf(stderr,"RAD50 VMM PCI match=%u\n",matches);break;}
        std::printf("RAD50 VMM device=%d PCI=%s\n",dev,pci);
        PoolChoice choice{&inv};
        if (!hsa_ok("pool iterate",hsa_amd_agent_iterate_memory_pools(gpu,select_pool,&choice)) ||
            !choice.found) break;
        constexpr size_t kBytes=4u*1024u*1024u;
        if (choice.granule>SIZE_MAX-kBytes) break;
        size=((kBytes+choice.granule-1)/choice.granule)*choice.granule;
        if (!hsa_ok("GPU pool vmem handle create",
              hsa_amd_vmem_handle_create(choice.pool,size,MEMORY_TYPE_PINNED,0,&handle))) break;
        handle_live=true;
        if (!hsa_ok("dma-buf export",
              hsa_amd_vmem_export_shareable_handle(&dmabuf,handle,0)) || dmabuf<0) break;
        std::fprintf(stderr,"RAD50 VMM exported DMA-BUF fd; no peer access\n");
        if (!hsa_ok("address reserve",hsa_amd_vmem_address_reserve(&va,size,0,0))) break;
        va_reserved=true;
        if (!hsa_ok("VMM map",hsa_amd_vmem_map(va,size,0,handle,0))) break;
        va_mapped=true;
        const hsa_amd_memory_access_desc_t grants[]={
            {HSA_ACCESS_PERMISSION_RW,gpu},
            {HSA_ACCESS_PERMISSION_RW,choice.cpu}
        };
        if (!hsa_ok("GPU+CPU set_access",
              hsa_amd_vmem_set_access(va,size,grants,2))) break;
        const bool writable=writable_vma(va,4096);
        hsa_access_permission_t cpu_perms{},gpu_perms{};
        const hsa_status_t cpu_access=hsa_amd_vmem_get_access(va,&cpu_perms,choice.cpu);
        const hsa_status_t gpu_access=hsa_amd_vmem_get_access(va,&gpu_perms,gpu);
        (void)hsa_ok("CPU get_access",cpu_access);
        (void)hsa_ok("GPU get_access",gpu_access);
        std::printf("RAD50 VMM CPU access_status=%d permissions=%d GPU access_status=%d permissions=%d writable_CPU_VMA=%d\n",
                    (int)cpu_access,(int)cpu_perms,(int)gpu_access,(int)gpu_perms,(int)writable);
        if (cpu_access!=HSA_STATUS_SUCCESS ||
            cpu_perms!=HSA_ACCESS_PERMISSION_RW || !writable) {
            std::fputs("STOP: local VMM mapping is not proven CPU writable\n",stderr);
            break;
        }
        std::puts("PASS: GPU-local VMM export and same-process CPU VMA (no stores/imports)");
        ok=true;
    } while(false);
    if (va_mapped && !hsa_ok("unmap",hsa_amd_vmem_unmap(va,size))) ok=false;
    if (va_reserved && !hsa_ok("address free",hsa_amd_vmem_address_free(va,size))) ok=false;
    if (dmabuf>=0 && ::close(dmabuf)!=0) ok=false;
    if (handle_live && !hsa_ok("handle release",hsa_amd_vmem_handle_release(handle))) ok=false;
    if (!hsa_ok("shutdown",hsa_shut_down())) ok=false;
    return ok;
}
} // namespace

int main(int argc,char** argv) {
    if (argc!=3 || std::strcmp(argv[1],"--device")) {
        std::fputs("usage: bar_vmem_cap --device HIP_ORDINAL\n",stderr);
        return 2;
    }
    char* end=nullptr;
    const long n=std::strtol(argv[2],&end,10);
    if (!argv[2][0] || *end || n<0 || n>127) return 2;
    return run((int)n)?0:3;
}
