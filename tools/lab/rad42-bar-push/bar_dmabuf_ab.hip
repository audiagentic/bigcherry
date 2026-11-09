// RAD50 isolated Gate 3 microbenchmark: the SAME CPU-only child process
// writes mapped host memory (production GPU-pull baseline) or GPU-local
// DMA-BUF mmap (proposed CPU BAR push). Persistent mappings and process.
// Parent launches the SAME full-word GPU consumer for each path.
// Serial handoff only: not concurrent GPU visibility or live inference.
// No P2P, no serving changes, no generic command/subprocess runner.
// Requires a separately acquired, exclusive GPU lease.
#include "bar_access.h"
#include "combine_f16.h"
#include "moe_remote_shm.h"
#include <array>
#include <linux/dma-buf.h>
#include <linux/memfd.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <poll.h>
#include <spawn.h>
#include <unistd.h>
#include <thread>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <string>
#include <vector>
extern char** environ;
namespace {
constexpr size_t MAX_BYTES=4u*1024u*1024u;
constexpr uint32_t COMBINE_WIDTH=2560;
constexpr uint32_t COMBINE_ROW_BYTES=2u*COMBINE_WIDTH;
constexpr size_t COMBINE_ROWS[]={1,16,64,256,768};
const char* style_name(int method){
    switch(method) {case 0:return "direct-scalar";case 1:return "staged-memcpy";
    case 2:return "combine-direct";case 3:return "combine-staged";}
    return "invalid";
}
constexpr uint32_t MAGIC=0x50335052u;
constexpr uint32_t VERSION=1;
constexpr int CHILD_SOCKET=3;
constexpr unsigned WARMUPS=3;
using Clock=std::chrono::steady_clock;
double tick_us(){return std::chrono::duration<double,std::micro>(Clock::now().time_since_epoch()).count();}
#include "combine_patterns.hip.h"
#include "live_control.hip.h"
struct Hello {uint32_t magic=MAGIC,version=VERSION,bytes=MAX_BYTES;};
struct Command {uint32_t magic=MAGIC,path=0,bytes=0,generation=0,method=0;};
struct Ack {
    uint32_t magic=MAGIC,generation=0,status=0;
    double prepare_us=0,write_us=0,sync_start_us=0,sync_end_us=0,total_us=0;
};
bool read_exact(int fd,void* dst,size_t bytes) {
    auto* p=static_cast<uint8_t*>(dst);
    while(bytes) {
        pollfd ev{fd,POLLIN,0}; int rc;
        do {rc=::poll(&ev,1,10000);}while(rc<0&&errno==EINTR);
        if(rc<=0 || !(ev.revents&POLLIN))return false;
        ssize_t n=::read(fd,p,bytes);
        if(n<0&&errno==EINTR)continue;
        if(n<=0)return false;
        p+=n;bytes-=static_cast<size_t>(n);
    }
    return true;
}
bool write_exact(int fd,const void* src,size_t bytes) {
    auto* p=static_cast<const uint8_t*>(src);
    while(bytes) {
        ssize_t n=::send(fd,p,bytes,MSG_NOSIGNAL);
        if(n<0&&errno==EINTR)continue;
        if(n<=0)return false;
        p+=n;bytes-=static_cast<size_t>(n);
    }
    return true;
}
bool send_handles(int fd,int dmabuf,int hostfd) {
    Hello hello;
    iovec io{&hello,sizeof(hello)};
    char control[CMSG_SPACE(2*sizeof(int))]{};
    msghdr msg{};
    msg.msg_iov=&io;msg.msg_iovlen=1;
    msg.msg_control=control;msg.msg_controllen=sizeof(control);
    cmsghdr* c=CMSG_FIRSTHDR(&msg);
    c->cmsg_level=SOL_SOCKET;c->cmsg_type=SCM_RIGHTS;
    c->cmsg_len=CMSG_LEN(2*sizeof(int));
    int handles[2]={dmabuf,hostfd};
    std::memcpy(CMSG_DATA(c),handles,sizeof(handles));
    return ::sendmsg(fd,&msg,MSG_NOSIGNAL)==(ssize_t)sizeof(hello);
}
bool receive_handles(int sock,int* gpu,int* host) {
    *gpu=*host=-1;
    pollfd p{sock,POLLIN,0};int rc;
    do {rc=::poll(&p,1,10000);}while(rc<0&&errno==EINTR);
    if(rc<=0||!(p.revents&POLLIN))return false;
    Hello hello{};
    iovec io{&hello,sizeof(hello)};
    char control[CMSG_SPACE(2*sizeof(int))]{};
    msghdr msg{};
    msg.msg_iov=&io;msg.msg_iovlen=1;
    msg.msg_control=control;msg.msg_controllen=sizeof(control);
    if(::recvmsg(sock,&msg,MSG_CMSG_CLOEXEC)!=(ssize_t)sizeof(hello) ||
       (msg.msg_flags&(MSG_TRUNC|MSG_CTRUNC)))return false;
    cmsghdr* c=CMSG_FIRSTHDR(&msg);
    if(!c || c->cmsg_level!=SOL_SOCKET || c->cmsg_type!=SCM_RIGHTS ||
       c->cmsg_len!=CMSG_LEN(2*sizeof(int)))return false;
    int handles[2]{-1,-1};
    std::memcpy(handles,CMSG_DATA(c),sizeof(handles));
    if(hello.magic!=MAGIC || hello.version!=VERSION ||
       hello.bytes!=MAX_BYTES || handles[0]<0 || handles[1]<0) {
        for(int f:handles)if(f>=0)::close(f);
        return false;
    }
    *gpu=handles[0];*host=handles[1];return true;
}
int worker(int sock) {
    int gpu_fd=-1,host_fd=-1;
    if(!receive_handles(sock,&gpu_fd,&host_fd)){
        std::fputs("RAD50 A/B worker: invalid descriptors\n",stderr);
        return 3;
    }
    void* gpu=::mmap(nullptr,MAX_BYTES,PROT_READ|PROT_WRITE,MAP_SHARED,gpu_fd,0);
    void* host=::mmap(nullptr,MAX_BYTES,PROT_READ|PROT_WRITE,MAP_SHARED,host_fd,0);
    const bool mapped=(gpu!=MAP_FAILED&&host!=MAP_FAILED);
    if(!mapped)std::fprintf(stderr,"RAD50 A/B worker mmap failed errno=%d\n",errno);
    // One staging allocation, outside timed loops, for comparing direct
    // computational stores with CPU staging plus memcpy.
    std::vector<uint32_t> staging(MAX_BYTES/sizeof(uint32_t));
    CombineInput combine_inputs; // actual production AVX2/F16C combiner inputs
    Ack start{};start.status=mapped?0:3;
    bool ok=write_exact(sock,&start,sizeof(start))&&mapped;
    while(ok) {
        Command cmd{};
        if(!read_exact(sock,&cmd,sizeof(cmd))) {ok=false;break;}
        if(cmd.magic!=MAGIC) {ok=false;break;}
        if(cmd.bytes==0)break;
        Ack reply{};reply.generation=cmd.generation;
        if(!cmd.generation || cmd.bytes>MAX_BYTES || cmd.bytes%4 ||
           cmd.path>2 || cmd.method>3 ||
           (cmd.path==2 && cmd.method!=2) ||
           (cmd.method>=2 && (cmd.bytes%COMBINE_ROW_BYTES || !combine_cpu_supported()))) {
            reply.status=3; (void)write_exact(sock,&reply,sizeof(reply));
            ok=false;break;
        }
        void* target=cmd.path==0?host:gpu;
        double begin=tick_us();
        dma_buf_sync sync{};
        int rc=0;
        if(cmd.path!=0) {
            sync.flags=DMA_BUF_SYNC_START|DMA_BUF_SYNC_WRITE;
            errno=0;
            do {rc=::ioctl(gpu_fd,DMA_BUF_IOCTL_SYNC,&sync);}
            while(rc<0&&(errno==EINTR||errno==EAGAIN));
            reply.sync_start_us=tick_us()-begin;
            if(rc<0)reply.status=errno?errno:3;
        }
        if(reply.status==0) {
            if(cmd.method==1 || cmd.method==3) {
                const double prep=tick_us();
                if(cmd.method==1) {
                    for(uint32_t i=0;i<cmd.bytes/4;++i)
                        staging[i]=value_at(i,cmd.generation);
                } else if(!combine_inputs.into(reinterpret_cast<uint16_t*>(staging.data()),
                                                cmd.bytes/COMBINE_ROW_BYTES,
                                                cmd.generation)) reply.status=4;
                reply.prepare_us=tick_us()-prep;
            }
            const double write_begin=tick_us();
            if(reply.status==0 && (cmd.method==1 || cmd.method==3))
                std::memcpy(target,staging.data(),cmd.bytes);
            else if(reply.status==0 && cmd.method==2) {
                if(!combine_inputs.into(static_cast<uint16_t*>(target),
                                        cmd.bytes/COMBINE_ROW_BYTES,
                                        cmd.generation)) reply.status=4;
            } else if(reply.status==0) {
                volatile uint32_t* out=static_cast<volatile uint32_t*>(target);
                for(uint32_t i=0;i<cmd.bytes/4;++i)
                    out[i]=value_at(i,cmd.generation);
            }
            std::atomic_thread_fence(std::memory_order_seq_cst);
            reply.write_us=tick_us()-write_begin;
        }
        if(cmd.path!=0 && reply.status==0) {
            sync.flags=DMA_BUF_SYNC_END|DMA_BUF_SYNC_WRITE;
            const double t=tick_us();
            errno=0;
            do {rc=::ioctl(gpu_fd,DMA_BUF_IOCTL_SYNC,&sync);}
            while(rc<0&&(errno==EINTR||errno==EAGAIN));
            reply.sync_end_us=tick_us()-t;
            if(rc<0)reply.status=errno?errno:3;
        }
        // Cross-process GPU-live publication: after DMA-BUF SYNC_END,
        // never before the full FP16 combine has drained.
        if(cmd.path==2 && reply.status==0){
            std::atomic_thread_fence(std::memory_order_seq_cst);
            __atomic_store_n(&static_cast<LiveControl*>(host)->ready,
                             cmd.generation,__ATOMIC_RELEASE);
        }
        reply.total_us=tick_us()-begin;
        if(!write_exact(sock,&reply,sizeof(reply))){ok=false;break;}
        if(reply.status){ok=false;break;}
    }
    if(gpu!=MAP_FAILED&&::munmap(gpu,MAX_BYTES)!=0)ok=false;
    if(host!=MAP_FAILED&&::munmap(host,MAX_BYTES)!=0)ok=false;
    ::close(gpu_fd);::close(host_fd);::close(sock);
    return ok?0:3;
}

#include "live_rounds.hip.h"
#include "live_relay.hip.h"
struct Sample {
    double cpu_prepare=0,cpu_write=0,sync_start=0,sync_end=0;
    double child_total=0,roundtrip=0,gpu_kernel=0,gpu_host=0,serial_total=0;
};
struct Stats {
    std::vector<Sample> samples;
    void push(const Sample& s){samples.push_back(s);}
};
double pct(std::vector<double> v,double q){
    if(v.empty())return 0;
    std::sort(v.begin(),v.end());
    return v[static_cast<size_t>(q*(v.size()-1))];
}
template<typename F> double quant(const Stats& stats,F f,double q){
    std::vector<double> v;v.reserve(stats.samples.size());
    for(const Sample& s:stats.samples)v.push_back(f(s));
    return pct(std::move(v),q);
}
void report(int device,size_t bytes,int iterations,const char* method,int path,const Stats& stats) {
    auto q=[&](auto f,double p){return quant(stats,f,p);};
    const double p50=q([](const Sample& s){return s.serial_total;},.5);
    const double p95=q([](const Sample& s){return s.serial_total;},.95);
    const double p99=q([](const Sample& s){return s.serial_total;},.99);
    std::printf("RAD50_AB device=%d bytes=%zu iterations=%d style=%s path=%s "
                "prepare_p50_us=%.2f cpu_write_p50_us=%.2f "
                "sync_start_p50_us=%.2f sync_end_p50_us=%.2f "
                "child_total_p50_us=%.2f roundtrip_p50_us=%.2f "
                "gpu_kernel_p50_us=%.2f gpu_host_p50_us=%.2f "
                "seq_p50_us=%.2f seq_p95_us=%.2f seq_p99_us=%.2f "
                "seq_GBps=%.3f\n",
                device,bytes,iterations,method,path==0?"mapped-host-pull":"dmabuf-cpu-push",
                q([](const Sample& s){return s.cpu_prepare;},.5),
                q([](const Sample& s){return s.cpu_write;},.5),
                q([](const Sample& s){return s.sync_start;},.5),
                q([](const Sample& s){return s.sync_end;},.5),
                q([](const Sample& s){return s.child_total;},.5),
                q([](const Sample& s){return s.roundtrip;},.5),
                q([](const Sample& s){return s.gpu_kernel;},.5),
                q([](const Sample& s){return s.gpu_host;},.5),
                p50,p95,p99,p50>0?bytes/(p50*1000):0);
}
bool bench(const char* self,int device,unsigned iterations,int method,size_t selected_bytes,bool live,int control_kind,unsigned verifier_blocks) {
    const bool finegrained=control_kind==1;
    const bool relay=control_kind==2;
    const bool parallel=control_kind==2 || control_kind==3;
    int count=0;char pci[64]{};
    uint32_t dom=0,bdf=0;
    if(!check_hip(hipGetDeviceCount(&count),"A/B HIP device count") ||
       device<0||device>=count||
       !check_hip(hipSetDevice(device),"A/B HIP set device") ||
       !check_hip(hipDeviceGetPCIBusId(pci,sizeof(pci),device),"A/B HIP PCI")||
       !parse_pci(pci,&dom,&bdf))return false;
    if(method>=2 && !combine_cpu_supported()){
        std::fputs("RAD50 combine-ab unsupported CPU: AVX2/F16C required\n",stderr);
        return false;
    }
    int gpu_wallclock_khz=0;
    if(live && (!check_hip(hipDeviceGetAttribute(&gpu_wallclock_khz,
                       hipDeviceAttributeWallClockRate,device),
                       "live GPU wall-clock rate") || gpu_wallclock_khz<=0))return false;
    if(hsa_init()!=HSA_STATUS_SUCCESS)return false;
    Inventory agents;
    hsa_amd_vmem_alloc_handle_t handle{};
    bool have_handle=false,have_va=false,have_map=false,have_reg=false;
    void* device_va=nullptr,*host_va=nullptr,*host_device=nullptr;
    void* relay_host=nullptr,*relay_device=nullptr;
    size_t alloc=0;
    int dmafd=-1,hostfd=-1,pair[2]{-1,-1};
    pid_t pid=-1;bool ok=false;
    unsigned* mismatch=nullptr,*relay_wait_timeout=nullptr;
    hipEvent_t event_start=nullptr,event_stop=nullptr;
    do {
        if(hsa_iterate_agents(collect_agent,&agents)!=HSA_STATUS_SUCCESS ||
           agents.cpus.empty())break;
        hsa_agent_t gpu{};unsigned matches=0;
        for(hsa_agent_t a:agents.gpus){
            uint32_t d=0,b=0;
            if(hsa_agent_get_info(a,(hsa_agent_info_t)HSA_AMD_AGENT_INFO_DOMAIN,&d)!=HSA_STATUS_SUCCESS||
               hsa_agent_get_info(a,(hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID,&b)!=HSA_STATUS_SUCCESS){
                matches=0;break;
            }
            if(d==dom&&b==bdf){gpu=a;++matches;}
        }
        if(matches!=1){std::fprintf(stderr,"RAD50 A/B GPU PCI match=%u\n",matches);break;}
        PoolChoice choice{&agents};
        if(hsa_amd_agent_iterate_memory_pools(gpu,select_pool,&choice)!=HSA_STATUS_SUCCESS||
           !choice.found||!choice.granule||choice.granule>SIZE_MAX-MAX_BYTES)break;
        alloc=(MAX_BYTES+choice.granule-1)/choice.granule*choice.granule;
        hsa_status_t st=hsa_amd_vmem_handle_create(choice.pool,alloc,MEMORY_TYPE_PINNED,0,&handle);
        if(st!=HSA_STATUS_SUCCESS){std::fprintf(stderr,"RAD50 A/B VMM create HSA=%d\n",(int)st);break;}
        have_handle=true;
        st=hsa_amd_vmem_export_shareable_handle(&dmafd,handle,0);
        if(st!=HSA_STATUS_SUCCESS||dmafd<0){std::fprintf(stderr,"RAD50 A/B export HSA=%d\n",(int)st);break;}
        if(hsa_amd_vmem_address_reserve(&device_va,alloc,0,0)!=HSA_STATUS_SUCCESS)break;
        have_va=true;
        if(hsa_amd_vmem_map(device_va,alloc,0,handle,0)!=HSA_STATUS_SUCCESS)break;
        have_map=true;
        hsa_amd_memory_access_desc_t grant{HSA_ACCESS_PERMISSION_RW,gpu};
        if(hsa_amd_vmem_set_access(device_va,alloc,&grant,1)!=HSA_STATUS_SUCCESS)break;
        // Host backing is a shared memfd so the separate CPU-only helper
        // can write into exactly the GPU-registered mapped host pages.
        hostfd=static_cast<int>(::syscall(SYS_memfd_create,"rad50-ab-host",MFD_CLOEXEC));
        if(hostfd<0 || ::ftruncate(hostfd,MAX_BYTES)!=0)break;
        host_va=::mmap(nullptr,MAX_BYTES,PROT_READ|PROT_WRITE,MAP_SHARED,hostfd,0);
        if(host_va==MAP_FAILED){host_va=nullptr;break;}
        if(live) std::memset(host_va,0,sizeof(LiveControl));
        const unsigned register_flags=hipHostRegisterMapped |
            (finegrained?hipExtHostRegisterUncached:0u);
        if(!check_hip(hipHostRegister(host_va,MAX_BYTES,register_flags),
                      finegrained?"live fine-grained host register":"A/B HIP host register"))break;
        have_reg=true;
        if(!check_hip(hipHostGetDevicePointer(&host_device,host_va,0),"A/B host GPU pointer") ||
           !check_hip(hipMalloc(&mismatch,sizeof(unsigned)),"A/B mismatch count")||
           !check_hip(hipEventCreate(&event_start),"A/B event start")||
           !check_hip(hipEventCreate(&event_stop),"A/B event stop"))break;
        if(parallel) {
            if(relay) {
                if(!check_hip(hipHostMalloc(&relay_host,sizeof(LiveControl),
                           hipHostMallocCoherent|hipHostMallocMapped),
                           "relay coherent host control allocate") ||
                   !check_hip(hipHostGetDevicePointer(&relay_device,relay_host,0),
                           "relay coherent control GPU pointer"))break;
                std::memset(relay_host,0,sizeof(LiveControl));
            }
            if(!check_hip(hipMalloc(&relay_wait_timeout,5*sizeof(unsigned)),
                          "parallel wait diagnostic device buffer"))break;
        }
        if(::socketpair(AF_UNIX,SOCK_STREAM,0,pair)!=0)break;
        posix_spawn_file_actions_t actions{};
        if(posix_spawn_file_actions_init(&actions)!=0)break;
        (void)posix_spawn_file_actions_addclose(&actions,pair[0]);
        (void)posix_spawn_file_actions_adddup2(&actions,pair[1],CHILD_SOCKET);
        if(pair[1]!=CHILD_SOCKET)
            (void)posix_spawn_file_actions_addclose(&actions,pair[1]);
        char* const av[]={const_cast<char*>(self),const_cast<char*>("--bench-child"),
                          const_cast<char*>("3"),nullptr};
        const int started=::posix_spawn(&pid,self,&actions,nullptr,av,environ);
        (void)posix_spawn_file_actions_destroy(&actions);
        if(started){pid=-1;break;}
        ::close(pair[1]);pair[1]=-1;
        Ack ready{};
        if(!send_handles(pair[0],dmafd,hostfd) ||
           !read_exact(pair[0],&ready,sizeof(ready)) ||
           ready.magic!=MAGIC || ready.status!=0)break;
        if(live){
            std::printf("RAD50_LIVE_CONTEXT device=%d pci=%s rows=%zu "
                        "bytes=%zu iterations=%u cpu_child=persistent "
                        "consumer_armed_before_write=1 dma_buf_sync=1 "
                        "ready_scope=system reuse_after_done=1 control=%s gpu_parallel_blocks=%u\n",
                        device,pci,selected_bytes/COMBINE_ROW_BYTES,
                        selected_bytes,iterations,
                        relay?"hip-host-coherent-relay":
                            (parallel?"hip-register-mapped-direct-parallel":
                            (finegrained?"hip-ext-uncached":"hip-register-mapped")),
                        verifier_blocks);
            std::fflush(stdout);
            const uint32_t count=static_cast<uint32_t>(selected_bytes/COMBINE_ROW_BYTES);
            const bool passed=parallel?
                relay_rounds(pair[0],device,iterations,count,
                             static_cast<LiveControl*>(relay?relay_host:host_va),
                             static_cast<LiveControl*>(relay?relay_device:host_device),
                             device_va,mismatch,relay_wait_timeout,
                             static_cast<uint64_t>(gpu_wallclock_khz)*3000ull,
                             relay,verifier_blocks):
                live_rounds(pair[0],device,iterations,count,
                            static_cast<LiveControl*>(host_va),
                            static_cast<LiveControl*>(host_device),device_va,
                            static_cast<uint64_t>(gpu_wallclock_khz)*3000ull);
            if(!passed)break;
            Command stop{};
            if(!write_exact(pair[0],&stop,sizeof(stop)))break;
            ok=true;
            break; // Parent cleanup waits for worker before success marker.
        }
        std::printf("RAD50_AB_CONTEXT device=%d pci=%s cpu_child=persistent "
                    "dma_buf_mmap=persistent host_memfd=hipHostRegisterMapped "
                    "serial_handoff=1 sync_start_end_per_push=1 "
                    "engine_semantics=direct_gpu_load source_is_synthetic=1 "
                    "bytes_filter=%zu iterations=%u style=%s real_combine=%d row_width=%u\n",
                    device,pci,selected_bytes,iterations,style_name(method),
                    method>=2?1:0,COMBINE_WIDTH);
        constexpr size_t cases_scalar[]={4096,65536,262144,1048576,4194304};
        const size_t* cases=method>=2?COMBINE_ROWS:cases_scalar;
        const size_t ncases=method>=2?(sizeof(COMBINE_ROWS)/sizeof(COMBINE_ROWS[0])):
                                   (sizeof(cases_scalar)/sizeof(cases_scalar[0]));
        bool all=true;
        for(size_t case_idx=0;case_idx<ncases;++case_idx){
            const size_t bytes=cases[case_idx]*(method>=2?COMBINE_ROW_BYTES:1u);
            if(selected_bytes && selected_bytes!=bytes)continue;
            Stats by_path[2];
            for(unsigned iteration=0;iteration<iterations+WARMUPS && all;++iteration){
                // Alternate order per iteration; no concealed extra copy.
                for(int j=0;j<2 && all;++j){
                    const int path=static_cast<int>((iteration+j)&1u);
                    const uint32_t generation=static_cast<uint32_t>(
                        0x182u+(bytes/4096)*313u+(iteration+1)*43u);
                    const Command cmd{MAGIC,static_cast<uint32_t>(path),
                                      static_cast<uint32_t>(bytes),generation,
                                      static_cast<uint32_t>(method)};
                    const double begin=tick_us();
                    Ack ack{};
                    if(!write_exact(pair[0],&cmd,sizeof(cmd)) ||
                       !read_exact(pair[0],&ack,sizeof(ack)) ||
                       ack.magic!=MAGIC || ack.status!=0 ||
                       ack.generation!=generation) {
                        std::fprintf(stderr,"RAD50 A/B child protocol failed path=%d bytes=%zu iter=%u status=%u\n",
                                     path,bytes,iteration,ack.status);
                        all=false;break;
                    }
                    const double cpu_ack=tick_us();
                    if(!check_hip(hipMemset(mismatch,0,sizeof(unsigned)),"A/B reset mismatch")){all=false;break;}
                    const double gpu_begin=tick_us();
                    if(!check_hip(hipEventRecord(event_start),"A/B event start")){all=false;break;}
                    const uint32_t* source=static_cast<const uint32_t*>(
                        path==0?host_device:device_va);
                    if(method>=2)
                        hipLaunchKernelGGL(verify_combine_rows,dim3(256),dim3(256),0,0,
                            reinterpret_cast<const uint16_t*>(source),
                            static_cast<uint32_t>(bytes/2),generation,mismatch);
                    else
                        hipLaunchKernelGGL(verify_words,dim3(256),dim3(256),0,0,
                            source,static_cast<uint32_t>(bytes/4),generation,mismatch);
                    if(!check_hip(hipGetLastError(),"A/B verify launch") ||
                       !check_hip(hipEventRecord(event_stop),"A/B event end") ||
                       !check_hip(hipEventSynchronize(event_stop),"A/B GPU wait")){all=false;break;}
                    float elapsed_ms=0;
                    if(!check_hip(hipEventElapsedTime(&elapsed_ms,event_start,event_stop),
                                  "A/B GPU event time")){all=false;break;}
                    unsigned bad=0;
                    if(!check_hip(hipMemcpy(&bad,mismatch,sizeof(bad),hipMemcpyDeviceToHost),
                                  "A/B GPU mismatch count")||bad) {
                        std::fprintf(stderr,"RAD50 A/B WRONG DATA path=%d bytes=%zu iter=%u errors=%u\n",
                                     path,bytes,iteration,bad);
                        all=false;break;
                    }
                    const double end=tick_us();
                    if(iteration>=WARMUPS)by_path[path].push(Sample{
                        ack.prepare_us,ack.write_us,ack.sync_start_us,ack.sync_end_us,
                        ack.total_us,cpu_ack-begin,elapsed_ms*1000.0,
                        end-gpu_begin,end-begin});
                }
            }
            if(!all)break;
            for(int path=0;path<2;++path)
                report(device,bytes,static_cast<int>(iterations),
                       style_name(method),path,by_path[path]);
            const double base=quant(by_path[0],[](const Sample& s){return s.serial_total;},.5);
            const double push=quant(by_path[1],[](const Sample& s){return s.serial_total;},.5);
            std::printf("RAD50_AB_COMPARE device=%d bytes=%zu method=%s "
                        "baseline_seq_p50_us=%.2f push_seq_p50_us=%.2f "
                        "push_speedup=%.3f delta_us=%.2f "
                        "NOTE=serial_synthetic_not_inference\n",
                        device,bytes,style_name(method),
                        base,push,push>0?base/push:0,base-push);
        }
        if(!all)break;
        Command stop{};stop.bytes=0;
        if(!write_exact(pair[0],&stop,sizeof(stop)))break;
        ok=true;
    }while(false);
    if(pair[0]>=0)::close(pair[0]);
    if(pair[1]>=0)::close(pair[1]);
    if(pid>0) {
        if(!ok)(void)::kill(pid,SIGTERM); // only the child spawned here
        int status=0;
        if(::waitpid(pid,&status,0)!=pid || !WIFEXITED(status) ||
           WEXITSTATUS(status)!=0)ok=false;
    }
    // A failed live arm/publish must not unmap host control or VRAM while
    // the bounded GPU waiter might still be polling it.
    if(live && !check_hip(hipDeviceSynchronize(),"RAD50 final live GPU drain"))ok=false;
    if(event_start&&!check_hip(hipEventDestroy(event_start),"A/B event cleanup"))ok=false;
    if(event_stop&&!check_hip(hipEventDestroy(event_stop),"A/B event cleanup"))ok=false;
    if(relay_wait_timeout &&
       !check_hip(hipFree(relay_wait_timeout),"relay timeout cleanup"))ok=false;
    if(mismatch&&!check_hip(hipFree(mismatch),"A/B GPU error cleanup"))ok=false;
    if(relay_host &&
       !check_hip(hipHostFree(relay_host),"relay coherent host cleanup"))ok=false;
    if(have_reg&&!check_hip(hipHostUnregister(host_va),"A/B host unregister"))ok=false;
    if(host_va&&::munmap(host_va,MAX_BYTES)!=0)ok=false;
    if(hostfd>=0)::close(hostfd);
    if(have_map&&hsa_amd_vmem_unmap(device_va,alloc)!=HSA_STATUS_SUCCESS)ok=false;
    if(have_va&&hsa_amd_vmem_address_free(device_va,alloc)!=HSA_STATUS_SUCCESS)ok=false;
    if(dmafd>=0)::close(dmafd);
    if(have_handle&&hsa_amd_vmem_handle_release(handle)!=HSA_STATUS_SUCCESS)ok=false;
    if(hsa_shut_down()!=HSA_STATUS_SUCCESS)ok=false;
    if(!ok)std::fputs("RAD50 A/B FAILED: no speed claim; check mapped-host registration, correctness, or GPU availability\n",stderr);
    else {
        if(live && parallel)
            std::printf("RAD50_%s_COMPLETE device=%d rows=%zu iterations=%u verified=%u status=0\n",
                        relay?"RELAY":"PARALLEL",device,
                        selected_bytes/COMBINE_ROW_BYTES,iterations,iterations);
        else if(live)
            std::printf("RAD50_LIVE_COMPLETE device=%d rows=%zu iterations=%u verified=%u status=0\n",
                        device,selected_bytes/COMBINE_ROW_BYTES,iterations,iterations);
        else
            std::printf("RAD50_AB_COMPLETE device=%d iterations=%u style=%s bytes_filter=%zu status=0\n",
                        device,iterations,style_name(method),selected_bytes);
        std::fflush(stdout); // Marker only after child exit and cleanup.
    }
    return ok;
}
} // namespace
int main(int argc,char** argv) {
    if(argc==3 && !std::strcmp(argv[1],"--bench-child") &&
       !std::strcmp(argv[2],"3")) return worker(CHILD_SOCKET);
    if((argc==8 || argc==10) && (!std::strcmp(argv[1],"--live") ||
                   !std::strcmp(argv[1],"--live-fg") ||
                   !std::strcmp(argv[1],"--live-relay") ||
                   !std::strcmp(argv[1],"--live-parallel")) &&
       !std::strcmp(argv[2],"--device") &&
       !std::strcmp(argv[4],"--iterations") &&
       !std::strcmp(argv[6],"--rows")) {
        char* end=nullptr;
        const long dev=std::strtol(argv[3],&end,10);
        if(!argv[3][0]||*end||dev<0||dev>127)return 2;
        const long n=std::strtol(argv[5],&end,10);
        if(!argv[5][0]||*end||n<10||n>200)return 2;
        const long rows=std::strtol(argv[7],&end,10);
        if(!argv[7][0]||*end)return 2;
        bool allowed=false;
        for(size_t r:COMBINE_ROWS)if(rows==static_cast<long>(r))allowed=true;
        if(!allowed)return 2;
        const bool fg=!std::strcmp(argv[1],"--live-fg");
        const int control_kind=!std::strcmp(argv[1],"--live-relay")?2:
                               (!std::strcmp(argv[1],"--live-parallel")?3:(fg?1:0));
        unsigned blocks=256;
        if(argc==10) {
            if(control_kind!=2 && control_kind!=3) return 2;
            if(std::strcmp(argv[8],"--verifier-blocks")) return 2;
            const long chosen=std::strtol(argv[9],&end,10);
            if(!argv[9][0] || *end || (chosen!=1 && chosen!=32 &&
               chosen!=64 && chosen!=128 && chosen!=256)) return 2;
            blocks=static_cast<unsigned>(chosen);
        }
        return bench(argv[0],static_cast<int>(dev),static_cast<unsigned>(n),
                     2,static_cast<size_t>(rows)*COMBINE_ROW_BYTES,true,control_kind,blocks)?0:3;
    }
    if((argc!=7 && argc!=9) || std::strcmp(argv[1],"--device") ||
       std::strcmp(argv[3],"--iterations") || std::strcmp(argv[5],"--mode"))
        return 2;
    char* end=nullptr;
    const long d=std::strtol(argv[2],&end,10);
    if(!argv[2][0]||*end||d<0||d>127)return 2;
    const long it=std::strtol(argv[4],&end,10);
    if(!argv[4][0]||*end||it<10||it>200)return 2;
    int method=0;
    if(!std::strcmp(argv[6],"direct"))method=0;
    else if(!std::strcmp(argv[6],"staged"))method=1;
    else if(!std::strcmp(argv[6],"combine-direct"))method=2;
    else if(!std::strcmp(argv[6],"combine-staged"))method=3;
    else return 2;
    size_t selected_bytes=0;
    if(argc==9) {
        if(std::strcmp(argv[7],"--bytes"))return 2;
        const long chosen=std::strtol(argv[8],&end,10);
        if(!argv[8][0]||*end)return 2;
        const bool valid_synthetic=chosen==4096 || chosen==65536 ||
                                   chosen==262144 || chosen==1048576 || chosen==4194304;
        bool valid_combine=false;
        for(size_t rows:COMBINE_ROWS)
            if(chosen==static_cast<long>(rows*COMBINE_ROW_BYTES))valid_combine=true;
        if(!(method>=2?valid_combine:valid_synthetic))return 2;
        selected_bytes=static_cast<size_t>(chosen);
    }
    return bench(argv[0],static_cast<int>(d),static_cast<unsigned>(it),
                 method,selected_bytes,false,0,256)?0:3;
}
