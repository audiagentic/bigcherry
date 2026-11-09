// RAD50 Gate 2b: two-process CPU-writable GPU VRAM IPC feasibility probe.
// BUILD ONLY until Gate 2 passes and an exclusive GPU lease is held.
// Parent (engine stand-in) exports an HSA GPU-pool allocation, then runs a
// fresh CPU-only helper process via posix_spawn; no helper HIP calls or P2P.
// Child attaches the export with a *CPU* HSA agent, writes generations,
// acknowledges after its CPU fence; parent verifies from the GPU.
// The check is serial (GPU verify launches after child ACK), NOT a proof of
// concurrent VRAM->host-flag ordering. Gate 4 is a separate test.
// Must reject unsupported CPU attachment; never fall back to host memory.
#include "bar_access.h"
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <poll.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;
namespace {
constexpr size_t kBytes=4u*1024u*1024u;
constexpr int kFd=3;
struct Init {
    hsa_amd_ipc_memory_t ipc{};
    uint64_t alloc_bytes=0;
    uint32_t tag=0xBAD50A11;
    uint32_t probe_gpu_only=0;
    uint32_t domain=0;
    uint32_t bdf=0;
};
struct Work {
    uint32_t bytes=0;
    uint32_t generation=0;
};
struct Reply {
    int32_t status=0;
    uint32_t generation=0;
};

bool write_exact(int fd, const void* p, size_t n) {
    const char* x=static_cast<const char*>(p);
    while (n) {
        const ssize_t got=::write(fd,x,n);
        if (got<0 && errno==EINTR) continue;
        if (got<=0) return false;
        x+=got;n-=(size_t)got;
    }
    return true;
}
bool read_exact(int fd, void* p, size_t n) {
    char* x=static_cast<char*>(p);
    while (n) {
        pollfd watch{fd,POLLIN,0};
        const int ready=::poll(&watch,1,10000);
        if (ready<0 && errno==EINTR) continue;
        if (ready<=0 || !(watch.revents&POLLIN)) return false;
        const ssize_t got=::read(fd,x,n);
        if (got<0 && errno==EINTR) continue;
        if (got<=0) return false;
        x+=got;n-=(size_t)got;
    }
    return true;
}
bool writable_cpu_vma(const void* ptr, size_t bytes) {
    const uintptr_t address=reinterpret_cast<uintptr_t>(ptr);
    if (!ptr || bytes > UINTPTR_MAX-address) return false;
    FILE* f=std::fopen("/proc/self/maps","r");
    if (!f) return false;
    bool valid=false;
    char line[512]{};
    while (std::fgets(line,sizeof(line),f)) {
        unsigned long start=0,end=0;
        char perms[5]{};
        if (std::sscanf(line,"%lx-%lx %4s",&start,&end,perms)==3 &&
            address>=start && address+bytes<=end &&
            perms[0]=='r' && perms[1]=='w') {
            valid=true;break;
        }
    }
    std::fclose(f);
    return valid;
}
bool process_chunks(int fd, uint32_t* words) {
    const Reply ready{0,0};
    if (!write_exact(fd,&ready,sizeof(ready))) return false;
    for (;;) {
        Work w{};
        if (!read_exact(fd,&w,sizeof(w))) return false;
        if (!w.bytes) {
            const Reply done{0,0};
            return write_exact(fd,&done,sizeof(done));
        }
        if (w.bytes>kBytes || w.bytes%4 || !w.generation) return false;
        for (uint32_t i=0;i<w.bytes/4;++i) words[i]=value_at(i,w.generation);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const Reply ack{0,w.generation};
        if (!write_exact(fd,&ack,sizeof(ack))) return false;
    }
}
bool hsa_agents(Inventory* inv) {
    return hsa_iterate_agents(collect_agent,inv)==HSA_STATUS_SUCCESS;
}
bool find_selected_gpu(int dev, const Inventory& inv, hsa_agent_t* selected) {
    int count=0;
    if (!check_hip(hipGetDeviceCount(&count),"device count") ||
        dev<0 || dev>=count ||
        !check_hip(hipSetDevice(dev),"set GPU")) return false;
    char pci[64]{};
    uint32_t dom=0,bdf=0;
    if (!check_hip(hipDeviceGetPCIBusId(pci,sizeof(pci),dev),"PCI ID") ||
        !parse_pci(pci,&dom,&bdf)) return false;
    unsigned count_match=0;
    for (const hsa_agent_t gpu:inv.gpus) {
        uint32_t d=0,b=0;
        if (hsa_agent_get_info(gpu,(hsa_agent_info_t)HSA_AMD_AGENT_INFO_DOMAIN,&d)!=HSA_STATUS_SUCCESS ||
            hsa_agent_get_info(gpu,(hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID,&b)!=HSA_STATUS_SUCCESS)
            return false;
        if (d==dom && b==bdf) { *selected=gpu; ++count_match; }
    }
    std::printf("parent HIP GPU=%d PCI=%s HSA match=%u\n",dev,pci,count_match);
    return count_match==1;
}
int child(int fd) {
    if (hsa_init()!=HSA_STATUS_SUCCESS) return 3;
    Inventory inv;
    int exit_code=3;
    void* mapped=nullptr;
    do {
        if (!hsa_agents(&inv) || inv.cpus.empty()) break;
        Init init{};
        if (!read_exact(fd,&init,sizeof(init)) || init.tag!=0xBAD50A11 ||
            init.alloc_bytes<kBytes || init.alloc_bytes>1u<<28) break;
        if (init.probe_gpu_only) {
            hsa_agent_t selected{};
            unsigned count=0;
            for (auto gpu : inv.gpus) {
                uint32_t d=0,b=0;
                if (hsa_agent_get_info(gpu,(hsa_agent_info_t)HSA_AMD_AGENT_INFO_DOMAIN,&d)!=HSA_STATUS_SUCCESS ||
                    hsa_agent_get_info(gpu,(hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID,&b)!=HSA_STATUS_SUCCESS)
                    break;
                if (d==init.domain && b==init.bdf) { selected=gpu; ++count; }
            }
            if (count!=1) { std::fprintf(stderr,"BLOCKED: IPC GPU agent PCI match=%u\n",count); break; }
            const hsa_status_t status=hsa_amd_ipc_memory_attach(&init.ipc,init.alloc_bytes,1,&selected,&mapped);
            const char* description=nullptr;
            (void)hsa_status_string(status,&description);
            std::fprintf(stderr,"RAD50 IPC GPU-agent attach: status=%d %s (no writes yet)\n",
                         (int)status,description?description:"unknown");
            if (status!=HSA_STATUS_SUCCESS || !mapped) break;
            // Diagnostics only; metadata does not itself prove CPU dereference safety.
            hsa_amd_pointer_info_t info{};
            info.size=sizeof(info);
            const hsa_status_t pstatus=hsa_amd_pointer_info(mapped,&info,nullptr,nullptr,nullptr);
            std::fprintf(stderr,"RAD50 IPC GPU import pointer: status=%d type=%d host_addr=%d\n",
                         (int)pstatus,(int)info.type,
                         (int)(pstatus==HSA_STATUS_SUCCESS && info.hostBaseAddress!=nullptr));
            const hsa_status_t grant=hsa_amd_agents_allow_access(
                1,&inv.cpus[0],nullptr,mapped);
            const char* grant_str=nullptr;
            (void)hsa_status_string(grant,&grant_str);
            std::fprintf(stderr,"RAD50 IPC CPU grant after GPU import: %d %s (no writes yet)\n",
                         (int)grant,grant_str?grant_str:"unknown");
            hsa_amd_pointer_info_t after{};
            after.size=sizeof(after);
            const hsa_status_t ast=hsa_amd_pointer_info(mapped,&after,nullptr,nullptr,nullptr);
            std::fprintf(stderr,"RAD50 IPC after grant: info_status=%d host_addr=%d\n",
                         (int)ast,(int)(ast==HSA_STATUS_SUCCESS && after.hostBaseAddress!=nullptr));
            if (init.probe_gpu_only==2) {
                // Only the explicitly requested diagnostic write uses this
                // CPU virtual address. Keep mapped for the actual IPC detach.
                if (grant!=HSA_STATUS_SUCCESS || ast!=HSA_STATUS_SUCCESS ||
                    after.hostBaseAddress==nullptr) {
                    std::fputs("BLOCKED: imported GPU memory lacks CPU mapping\n",stderr);
                    break;
                }
                if (!writable_cpu_vma(after.hostBaseAddress,4096)) {
                    std::fputs("BLOCKED: imported hostBaseAddress is not a writable CPU VMA\n",stderr);
                    break;
                }
                std::fputs("RAD50 IPC: host VMA writable; attempting 4 KiB CPU write\n",stderr);
                exit_code=process_chunks(fd,static_cast<uint32_t*>(after.hostBaseAddress))?0:3;
                break;
            }
            const Reply reply{0,0};
            if (!write_exact(fd,&reply,sizeof(reply))) break;
            exit_code=0;
            break;
        }
        // Request direct *CPU* access, never a peer GPU agent.
        hsa_status_t attached=HSA_STATUS_ERROR;
        // The exporting process selected an accessible CPU NUMA agent.
        // Try each local CPU agent so enumeration order cannot create
        // a false negative on a multi-socket host.
        for (const auto cpu : inv.cpus) {
            attached=hsa_amd_ipc_memory_attach(&init.ipc,init.alloc_bytes,
                                                1,&cpu,&mapped);
            if (attached==HSA_STATUS_SUCCESS && mapped) break;
        }
        if (attached!=HSA_STATUS_SUCCESS || !mapped) {
            const char* description=nullptr;
            (void)hsa_status_string(attached,&description);
            std::fprintf(stderr,"BLOCKED: CPU-agent HSA IPC attach failed: status=%d (%s), agents_tried=%zu\n",
                         (int)attached,description?description:"unknown",inv.cpus.size());
            break;
        }
        exit_code=process_chunks(fd,static_cast<uint32_t*>(mapped))?0:3;
    } while (false);
    if (mapped && hsa_amd_ipc_memory_detach(mapped)!=HSA_STATUS_SUCCESS)
        exit_code=3;
    if (hsa_shut_down()!=HSA_STATUS_SUCCESS) exit_code=3;
    ::close(fd);
    return exit_code;
}
bool verify_gpu(const void* ptr, size_t bytes, uint32_t generation,
                unsigned* errors) {
    if (!check_hip(hipMemset(errors,0,sizeof(unsigned)),"zero verification")) return false;
    hipLaunchKernelGGL(verify_words,dim3(256),dim3(256),0,0,
                       static_cast<const uint32_t*>(ptr),
                       (uint32_t)(bytes/4),generation,errors);
    if (!check_hip(hipGetLastError(),"kernel") ||
        !check_hip(hipDeviceSynchronize(),"verification sync"))
        return false;
    unsigned bad=0;
    if (!check_hip(hipMemcpy(&bad,errors,sizeof(bad),hipMemcpyDeviceToHost),
                   "verification count")) return false;
    if (bad) std::fprintf(stderr,"FAIL generation=%u bytes=%zu mismatches=%u\n",
                           generation,bytes,bad);
    return bad==0;
}
bool run_parent(const char* self,int dev,int mode) {
    if (hsa_init()!=HSA_STATUS_SUCCESS) return false;
    bool ok=false;
    void* ptr=nullptr;
    unsigned* errors=nullptr;
    int pair[2]{-1,-1};
    pid_t pid=-1;
    Inventory inv;
    do {
        hsa_agent_t gpu{};
        if (!hsa_agents(&inv) || !find_selected_gpu(dev,inv,&gpu)) break;
        PoolChoice choice{&inv};
        if (hsa_amd_agent_iterate_memory_pools(gpu,select_pool,&choice)!=HSA_STATUS_SUCCESS ||
            !choice.found) {
            std::fputs("BLOCKED: no CPU-accessible GPU-local pool\n",stderr);
            break;
        }
        if (choice.granule>SIZE_MAX-kBytes) break;
        const size_t size=((kBytes+choice.granule-1)/choice.granule)*choice.granule;
        if (hsa_amd_memory_pool_allocate(choice.pool,size,0,&ptr)!=HSA_STATUS_SUCCESS || !ptr)
            break;
        // Match the successful serial path: grant the CPU agent access
        // *before* sharing. This is a diagnostic hypothesis, not proof that
        // grant survives IPC import into another process.
        const hsa_status_t cpu_grant=hsa_amd_agents_allow_access(
            1,&choice.cpu,nullptr,ptr);
        if (cpu_grant!=HSA_STATUS_SUCCESS) {
            std::fprintf(stderr,"BLOCKED: exporter CPU access grant HSA=%d\n",
                         (int)cpu_grant);
            break;
        }
        std::fprintf(stderr,"RAD50 IPC: parent CPU access granted before export\n");
        hsa_amd_ipc_memory_t handle{};
        const auto ipc_status=hsa_amd_ipc_memory_create(ptr,size,&handle);
        if (ipc_status!=HSA_STATUS_SUCCESS) {
            std::fprintf(stderr,"BLOCKED: HSA export failed status=%d\n",(int)ipc_status);
            break;
        }
        if (!check_hip(hipMalloc(&errors,sizeof(unsigned)),"error slot")) break;
        if (::socketpair(AF_UNIX,SOCK_STREAM,0,pair)) break;
        posix_spawn_file_actions_t actions{};
        if (posix_spawn_file_actions_init(&actions)) break;
        (void)posix_spawn_file_actions_addclose(&actions,pair[0]);
        (void)posix_spawn_file_actions_adddup2(&actions,pair[1],kFd);
        if (pair[1]!=kFd) (void)posix_spawn_file_actions_addclose(&actions,pair[1]);
        char fd_name[16]{};
        std::snprintf(fd_name,sizeof(fd_name),"%d",kFd);
        char* const args[]={const_cast<char*>(self),const_cast<char*>("--helper"),
                            fd_name,nullptr};
        const int spawned=posix_spawn(&pid,self,&actions,nullptr,args,environ);
        (void)posix_spawn_file_actions_destroy(&actions);
        if (spawned) {pid=-1;break;}
        ::close(pair[1]);pair[1]=-1;
        uint32_t domain=0,bdf=0;
        if (hsa_agent_get_info(gpu,(hsa_agent_info_t)HSA_AMD_AGENT_INFO_DOMAIN,&domain)!=HSA_STATUS_SUCCESS ||
            hsa_agent_get_info(gpu,(hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID,&bdf)!=HSA_STATUS_SUCCESS) break;
        const Init init{handle,size,0xBAD50A11,(uint32_t)mode,domain,bdf};
        Reply reply{};
        if (!write_exact(pair[0],&init,sizeof(init)) ||
            !read_exact(pair[0],&reply,sizeof(reply)) || reply.status) break;
        if (mode==1) {
            std::puts("RAD50 IPC GPU-agent-only attach PASS; no CPU access, no payload tested");
            ok=true;break;
        }
        bool all=true;
        for (uint32_t bytes : {uint32_t{4096},uint32_t{kBytes}}) {
            if (mode==2 && bytes>4096) break;
            for (uint32_t gen=1;gen<=(mode==2?1u:12u);++gen) {
                const Work w{bytes,gen+(bytes==4096?0u:100u)};
                if (!write_exact(pair[0],&w,sizeof(w)) ||
                    !read_exact(pair[0],&reply,sizeof(reply)) ||
                    reply.status || reply.generation!=w.generation ||
                    !verify_gpu(ptr,w.bytes,w.generation,errors)) {
                    all=false;break;
                }
            }
            if (!all) break;
            std::printf("PASS IPC parent GPU verify bytes=%u generations=%u mode=%d\n",
                        bytes,mode==2?1u:12u,mode);
        }
        if (!all) break;
        const Work stop{};
        if (!write_exact(pair[0],&stop,sizeof(stop)) ||
            !read_exact(pair[0],&reply,sizeof(reply)) ||
            reply.status) break;
        ok=true;
    } while (false);
    if (pair[0]>=0) ::close(pair[0]);
    if (pair[1]>=0) ::close(pair[1]);
    if (pid>0) {
        // Child is ours; on a failed protocol terminate only this exact PID
        // to avoid a stuck reader. Never touch foreign processes.
        if (!ok) (void)::kill(pid,SIGTERM);
        int status=0;
        const pid_t waited=::waitpid(pid,&status,0);
        if (waited==pid && WIFSIGNALED(status))
            std::fprintf(stderr,"RAD50 IPC helper terminated by signal %d\n",WTERMSIG(status));
        else if (waited==pid && WIFEXITED(status) && WEXITSTATUS(status)!=0)
            std::fprintf(stderr,"RAD50 IPC helper exited %d\n",WEXITSTATUS(status));
        if (waited!=pid || !WIFEXITED(status) || WEXITSTATUS(status)!=0) ok=false;
    }
    if (errors) (void)hipFree(errors);
    if (ptr && hsa_amd_memory_pool_free(ptr)!=HSA_STATUS_SUCCESS) ok=false;
    if (hsa_shut_down()!=HSA_STATUS_SUCCESS) ok=false;
    return ok;
}
} // namespace

int main(int argc,char** argv) {
    if (argc==3 && !std::strcmp(argv[1],"--helper")) {
        const int fd=std::atoi(argv[2]);
        return fd==kFd ? child(fd) : 2;
    }
    const bool gpu_only=argc==4 && !std::strcmp(argv[3],"--probe-attach-gpu-only");
    const bool gpu_cpu=argc==4 && !std::strcmp(argv[3],"--probe-gpu-then-cpu");
    if ((argc!=3 && !gpu_only && !gpu_cpu) || std::strcmp(argv[1],"--device")) {
        std::fputs("usage: bar_ipc_canary --device HIP_ORDINAL\n",stderr);
        return 2;
    }
    char* end=nullptr;
    const long dev=std::strtol(argv[2],&end,10);
    if (!argv[2][0] || *end || dev<0 || dev>127) return 2;
    const bool ok=run_parent(argv[0],(int)dev,gpu_only?1:(gpu_cpu?2:0));
    if (!ok) std::fputs("STOP: two-process HSA CPU->GPU VRAM IPC not proven\n",stderr);
    return ok?0:3;
}
