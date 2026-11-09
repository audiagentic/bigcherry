// RAD50: GPU-local VMM DMA-BUF export -> separate CPU-only process
// mmap(PROT_READ|PROT_WRITE) + DMA_BUF_IOCTL_SYNC capability probe.
// No pointer dereference, CPU stores, GPU reads or performance claims.
// Requires exclusive GPU lease. IPC child re-executes the unified runner only.
#include "bar_access.h"
#include <linux/dma-buf.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <poll.h>
#include <spawn.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
extern char** environ;
namespace {
constexpr size_t SIZE=4u*1024u*1024u;
constexpr uint32_t MAGIC=0x52443530u;
struct Msg {uint32_t magic=MAGIC;uint32_t size=SIZE;uint32_t op=0;};
struct Reply {uint32_t magic=MAGIC;int map_error=-1,start_error=-1,end_error=-1;uint32_t wrote=0;};
bool read_reply(int fd,Reply* r) {
    pollfd p{fd,POLLIN,0}; int rc;
    do {rc=::poll(&p,1,8000);}while(rc<0&&errno==EINTR);
    return rc>0&&(p.revents&POLLIN)&&::read(fd,r,sizeof(*r))==(ssize_t)sizeof(*r)
           &&r->magic==MAGIC;
}
bool send_fd(int sock,int fd,int mode) {
    Msg payload; payload.op=static_cast<uint32_t>(mode);
    iovec iov{&payload,sizeof(payload)};
    char control[CMSG_SPACE(sizeof(int))]{};
    msghdr msg{};
    msg.msg_iov=&iov;msg.msg_iovlen=1;
    msg.msg_control=control;msg.msg_controllen=sizeof(control);
    cmsghdr* c=CMSG_FIRSTHDR(&msg);
    c->cmsg_level=SOL_SOCKET;c->cmsg_type=SCM_RIGHTS;
    c->cmsg_len=CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(c),&fd,sizeof(fd));
    return ::sendmsg(sock,&msg,MSG_NOSIGNAL)==(ssize_t)sizeof(payload);
}
int child(int sock) {
    Msg payload{};
    iovec iov{&payload,sizeof(payload)};
    char control[CMSG_SPACE(sizeof(int))]{};
    msghdr msg{};
    msg.msg_iov=&iov;msg.msg_iovlen=1;
    msg.msg_control=control;msg.msg_controllen=sizeof(control);
    pollfd p{sock,POLLIN,0};int rc;
    do {rc=::poll(&p,1,8000);}while(rc<0&&errno==EINTR);
    Reply r{};
    int fd=-1;
    if(rc>0&&(p.revents&POLLIN)&&
       ::recvmsg(sock,&msg,MSG_CMSG_CLOEXEC)==(ssize_t)sizeof(payload) &&
       payload.magic==MAGIC&&payload.size==SIZE&&payload.op<=2&&
       !(msg.msg_flags&(MSG_CTRUNC|MSG_TRUNC))) {
        cmsghdr* c=CMSG_FIRSTHDR(&msg);
        if(c&&c->cmsg_level==SOL_SOCKET&&c->cmsg_type==SCM_RIGHTS&&
           c->cmsg_len==CMSG_LEN(sizeof(int)))
            std::memcpy(&fd,CMSG_DATA(c),sizeof(fd));
    }
    if(fd>=0) {
        errno=0;
        void* va=::mmap(nullptr,SIZE,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
        r.map_error=va==MAP_FAILED?errno:0;
        if(va!=MAP_FAILED) {
            dma_buf_sync sync{};
            sync.flags=DMA_BUF_SYNC_START|DMA_BUF_SYNC_RW;
            errno=0;
            do {rc=::ioctl(fd,DMA_BUF_IOCTL_SYNC,&sync);}
            while(rc<0&&(errno==EAGAIN||errno==EINTR));
            r.start_error=rc<0?errno:0;
            if(!r.start_error) {
                if(payload.op!=0) {
                    const uint32_t count=payload.op==2?uint32_t{SIZE/4}:uint32_t{1024};
                    std::fprintf(stderr,"RAD50 DMA-BUF child: %u-byte CPU write begins (isolated process)\n",count*4u);
                    volatile uint32_t* words=static_cast<volatile uint32_t*>(va);
                    for(uint32_t i=0;i<count;++i) words[i]=value_at(i,7u);
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    r.wrote=1;
                }
                sync.flags=DMA_BUF_SYNC_END|DMA_BUF_SYNC_RW;
                errno=0;
                do {rc=::ioctl(fd,DMA_BUF_IOCTL_SYNC,&sync);}
                while(rc<0&&(errno==EAGAIN||errno==EINTR));
                r.end_error=rc<0?errno:0;
            }
            if(::munmap(va,SIZE)!=0) r.end_error=errno;
        }
        ::close(fd);
    }
    std::fprintf(stderr,"RAD50 DMA-BUF child map_errno=%d sync_start_errno=%d "
                "sync_end_errno=%d wrote=%u GPU_READBACK=0\n",
                r.map_error,r.start_error,r.end_error,r.wrote);
    (void)::write(sock,&r,sizeof(r));
    ::close(sock);
    return !r.map_error&&!r.start_error&&!r.end_error&&(!payload.op||r.wrote)?0:3;
}
bool parent(const char* self,int device,int mode) {
    const bool write=mode!=0;
    int count=0;char pci[64]{};uint32_t dom=0,bdf=0;
    if(!check_hip(hipGetDeviceCount(&count),"device count")||
       device<0||device>=count||!check_hip(hipSetDevice(device),"set device")||
       !check_hip(hipDeviceGetPCIBusId(pci,sizeof(pci),device),"PCI")||
       !parse_pci(pci,&dom,&bdf)) return false;
    if(hsa_init()!=HSA_STATUS_SUCCESS) return false;
    Inventory agents;
    hsa_amd_vmem_alloc_handle_t handle{};
    bool have_handle=false,ok=false,have_va=false,have_map=false;
    void* gpu_va=nullptr;
    size_t gpu_alloc=0;
    unsigned* errors=nullptr;
    int fd=-1,pair[2]{-1,-1};pid_t pid=-1;
    do {
        if(hsa_iterate_agents(collect_agent,&agents)!=HSA_STATUS_SUCCESS||
           agents.cpus.empty()) break;
        hsa_agent_t gpu{};unsigned matches=0;
        for(auto a:agents.gpus) {
            uint32_t d=0,b=0;
            if(hsa_agent_get_info(a,(hsa_agent_info_t)HSA_AMD_AGENT_INFO_DOMAIN,&d)!=HSA_STATUS_SUCCESS||
               hsa_agent_get_info(a,(hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID,&b)!=HSA_STATUS_SUCCESS) break;
            if(d==dom&&b==bdf){gpu=a;++matches;}
        }
        if(matches!=1) break;
        PoolChoice choice{&agents};
        if(hsa_amd_agent_iterate_memory_pools(gpu,select_pool,&choice)!=HSA_STATUS_SUCCESS||
           !choice.found||!choice.granule||choice.granule>SIZE_MAX-SIZE) break;
        size_t alloc=(SIZE+choice.granule-1)/choice.granule*choice.granule;
        gpu_alloc=alloc;
        hsa_status_t st=hsa_amd_vmem_handle_create(choice.pool,alloc,MEMORY_TYPE_PINNED,0,&handle);
        if(st!=HSA_STATUS_SUCCESS){std::fprintf(stderr,"RAD50 VMM create HSA=%d\n",(int)st);break;}
        have_handle=true;
        st=hsa_amd_vmem_export_shareable_handle(&fd,handle,0);
        if(st!=HSA_STATUS_SUCCESS||fd<0){std::fprintf(stderr,"RAD50 export HSA=%d\n",(int)st);break;}
        if(write) {
            if(hsa_amd_vmem_address_reserve(&gpu_va,alloc,0,0)!=HSA_STATUS_SUCCESS) break;
            have_va=true;
            if(hsa_amd_vmem_map(gpu_va,alloc,0,handle,0)!=HSA_STATUS_SUCCESS) break;
            have_map=true;
            const hsa_amd_memory_access_desc_t gpu_grant{HSA_ACCESS_PERMISSION_RW,gpu};
            if(hsa_amd_vmem_set_access(gpu_va,alloc,&gpu_grant,1)!=HSA_STATUS_SUCCESS) break;
            if(!check_hip(hipMalloc(&errors,sizeof(unsigned)),"GPU verify errors allocation")) break;
        }
        if(::socketpair(AF_UNIX,SOCK_STREAM,0,pair)!=0) break;
        posix_spawn_file_actions_t actions{};
        if(posix_spawn_file_actions_init(&actions)!=0) break;
        (void)posix_spawn_file_actions_addclose(&actions,pair[0]);
        (void)posix_spawn_file_actions_adddup2(&actions,pair[1],3);
        if(pair[1]!=3) (void)posix_spawn_file_actions_addclose(&actions,pair[1]);
        char* const argv[]={const_cast<char*>(self),const_cast<char*>("--dmabuf-child"),
                            const_cast<char*>("3"),nullptr};
        int started=::posix_spawn(&pid,self,&actions,nullptr,argv,environ);
        (void)posix_spawn_file_actions_destroy(&actions);
        if(started!=0){pid=-1;break;}
        ::close(pair[1]);pair[1]=-1;
        Reply reply{};
        if(!send_fd(pair[0],fd,mode)||!read_reply(pair[0],&reply)) break;
        std::printf("RAD50 DMA-BUF child device=%d PCI=%s mmap_errno=%d "
                    "start_errno=%d end_errno=%d\n",
                    device,pci,reply.map_error,reply.start_error,reply.end_error);
        ok=!reply.map_error&&!reply.start_error&&!reply.end_error&&(!write||reply.wrote==1);
        if(ok && write) {
            if(!check_hip(hipMemset(errors,0,sizeof(unsigned)),"GPU verify reset")) ok=false;
            if(ok) {
                hipLaunchKernelGGL(verify_words,dim3(32),dim3(128),0,0,
                    static_cast<const uint32_t*>(gpu_va),mode==2?uint32_t{SIZE/4}:uint32_t{1024},uint32_t{7},errors);
                ok=check_hip(hipGetLastError(),"GPU VMM verify launch") &&
                   check_hip(hipDeviceSynchronize(),"GPU VMM verify sync");
            }
            unsigned mismatch=0;
            if(ok) ok=check_hip(hipMemcpy(&mismatch,errors,sizeof(mismatch),
                                hipMemcpyDeviceToHost),"GPU VMM verify readback");
            if(ok && mismatch) {
                std::fprintf(stderr,"FAIL DMA-BUF GPU mismatches=%u\n",mismatch);
                ok=false;
            }
            if(ok) std::printf("PASS separate-process %u-byte CPU stores and parent GPU full-word verification\n",mode==2?unsigned(SIZE):4096u);
        } else if(ok) std::puts("PASS capability only; NO CPU STORES or GPU VERIFICATION");
    }while(false);
    if(pair[0]>=0)::close(pair[0]);
    if(pair[1]>=0)::close(pair[1]);
    if(pid>0) {
        if(!ok)(void)::kill(pid,SIGTERM); // exact child PID only
        int status=0;
        if(::waitpid(pid,&status,0)!=pid||!WIFEXITED(status)||WEXITSTATUS(status)!=0)ok=false;
    }
    if(errors && !check_hip(hipFree(errors),"GPU error buffer cleanup"))ok=false;
    if(have_map && hsa_amd_vmem_unmap(gpu_va,gpu_alloc)!=HSA_STATUS_SUCCESS)ok=false;
    if(have_va && hsa_amd_vmem_address_free(gpu_va,gpu_alloc)!=HSA_STATUS_SUCCESS)ok=false;
    if(fd>=0)::close(fd);
    if(have_handle&&hsa_amd_vmem_handle_release(handle)!=HSA_STATUS_SUCCESS)ok=false;
    if(hsa_shut_down()!=HSA_STATUS_SUCCESS)ok=false;
    return ok;
}
} // namespace
int main(int argc,char** argv) {
    if(argc==3&&!std::strcmp(argv[1],"--dmabuf-child")&&!std::strcmp(argv[2],"3"))
        return child(3);
    int mode=argc==4&&!std::strcmp(argv[3],"--write-verify")?1:(argc==4&&!std::strcmp(argv[3],"--full-verify")?2:0);
    if((argc!=3 && !mode)||std::strcmp(argv[1],"--device"))return 2;
    char* end=nullptr;long device=std::strtol(argv[2],&end,10);
    if(!argv[2][0]||*end||device<0||device>127)return 2;
    bool ok=parent(argv[0],(int)device,mode);
    if(!ok)std::fputs("STOP: DMA-BUF CPU mmap capability not proven\n",stderr);
    return ok?0:3;
}
