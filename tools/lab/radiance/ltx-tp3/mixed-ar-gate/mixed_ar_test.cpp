/* Native Radiance ABI44 mixed-card collective probe, not a TP3 serving path.
 * Exercise the existing exact three-rank kernel rows and BigCherry's actual
 * process-wide shared-slot rendezvous with two gfx1100 and one gfx1201 cards.
 * Inputs use C06's video/audio whole-row Q/K pack and a full output reduce.
 */
#include "rad_abi.h"
#include "bc_shared_slot.h"
#if defined(LTX_C08_AQL_ENGINE)
#include "rad_device.h"
#include "device/device.h"
#endif
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
void check(bool yes, const std::string& what) {
    if (!yes) throw std::runtime_error(what);
}
void hipok(hipError_t x, const char* where) {
    if (x != hipSuccess) throw std::runtime_error(std::string(where) + ": " + hipGetErrorString(x));
}
struct Plugin {
    void* h = nullptr;
    int (*count)() = nullptr;
    const RadKernelInfo* (*at)(int) = nullptr;
    explicit Plugin(const char* path) {
        h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        check(h != nullptr, std::string("dlopen ") + path + ": " + (h ? "" : dlerror()));
        count = (int(*)())dlsym(h,"rad_kernel_count");
        at = (const RadKernelInfo*(*)(int))dlsym(h,"rad_kernel_at");
        check(count && at, "missing native kernel ABI44 table");
    }
    const RadKernelInfo* find(const std::string& name) const {
        for(int i=0; i<count(); ++i) {
            const auto* k=at(i);
            if(k && k->name && name==k->name) return k;
        }
        return nullptr;
    }
    Plugin(const Plugin&)=delete; Plugin& operator=(const Plugin&)=delete;
    ~Plugin() { if(h) dlclose(h); }
};
RadParam pi(const char* k, long long v) {
    RadParam p{}; p.key=k; p.kind=RAD_P_INT; p.ival=v; return p;
}
RadParam ps(const char* k, const char* v) {
    RadParam p{}; p.key=k; p.kind=RAD_P_STR; p.sval=v; return p;
}
RadTensor tbf(void* ptr, int64_t n) {
    RadTensor t{}; t.data=ptr; t.dtype=RAD_BF16;
    t.rank=1; t.shape[0]=n; t.stride[0]=1; return t;
}
uint16_t bf(float f) {
    uint32_t u=0; std::memcpy(&u,&f,4);
    u+=0x7fffU+((u>>16)&1U); return (uint16_t)(u>>16);
}
float f32(uint16_t b) {
    uint32_t u=(uint32_t)b<<16; float f;
    std::memcpy(&f,&u,4);return f;
}
struct Trial {
    int64_t width;
    int rows, hd;
    std::array<int,3> heads;
    bool disjoint;
    bool oop;
    const char* label;
};
constexpr std::array<Trial,11> trials {{
    {4096, 4, 128, {8,8,16}, true, false, "video q/k 8,8,16"},
    {2048, 4, 64,  {8,8,16}, true, false, "audio q/k 8,8,16"},
    {4096, 4, 128, {12,12,8}, true, false, "video q/k 12,12,8"},
    {2048, 4, 64,  {12,12,8}, true, false, "audio q/k 12,12,8"},
    {4096, 4, 128, {8,8,16}, false, false, "video output sum in-place"},
    {2048, 4, 64,  {12,12,8}, false, true, "audio output sum out-of-place"},
    {4096, 1, 128, {12,12,8}, true, true, "video q/k 1-row out-of-place"},
    {4096,16, 128, {8,8,16}, true, false, "video q/k 16 rows"},
    {2048,16, 64, {12,12,8}, true, false, "audio q/k 16 rows"},
    {4096,64, 128, {12,12,8}, true, false, "video q/k 64 rows"},
    {4096,64, 128, {8,8,16}, false, false, "video output sum 64 rows"},
}};
int64_t owner_start(const Trial& t, int rank) {
    int64_t x=0;
    for(int i=0;i<rank;++i)x+=t.heads[(size_t)i]*t.hd;
    return x;
}
float sample(const Trial& t,int rank,int64_t ix) {
    const int64_t col=ix%t.width;
    const int64_t start=owner_start(t,rank), n=t.heads[(size_t)rank]*t.hd;
    if(t.disjoint && (col<start || col>=start+n))return 0;
    // Representable BF16 operands with both signs and unequal contributions.
    return (float)((int)((ix*37+rank*31)%61)-30)*(float)(rank+1)/16.0f;
}
struct Result {
    int rank=-1, device=-1, rc_init=RAD_E_STATE, rc_launch=RAD_OK;
    std::vector<int64_t> wrong;
    std::string error;
};
void run_rank(int rank,int device,const RadKernelInfo* k,Result* result) {
    result->rank=rank;result->device=device;
    void* instance=nullptr;
    void* din=nullptr;void* dout=nullptr;
    hipStream_t stream=nullptr;
    try {
#if defined(LTX_C08_AQL_ENGINE)
        check(rad_dev_set(device)==RAD_OK,"Radiance device bind failed");
        RadStream rs=nullptr;
        check(rad_stream_create(&rs,0)==RAD_OK && rs,
              "Radiance AQL stream create failed");
        stream=(hipStream_t)rs;
#else
        hipok(hipSetDevice(device),"hipSetDevice");
        hipok(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking),"hipStreamCreate");
#endif
        int64_t max=0;
        for(const Trial& trial:trials)
            max=std::max(max,trial.width*trial.rows);
        std::vector<RadParam> param {
            pi("world_size",3),pi("numel",max),ps("dtype","bf16"),pi("exact",1)
        };
        result->rc_init=k->init(param.data(),(int)param.size(),rank,3,&instance);
        std::fprintf(stderr,"rank%d: native init rc=%d instance=%p row=%s\n",rank,
                     result->rc_init,instance,k->name);
        if(result->rc_init!=RAD_OK) throw std::runtime_error("native init rc="+std::to_string(result->rc_init));
        hipok(hipMalloc(&din,max*2),"hipMalloc input");
        hipok(hipMalloc(&dout,max*2),"hipMalloc output");
        for(const Trial& trial:trials) {
            const int64_t numel=trial.width*trial.rows;
            std::vector<uint16_t> send((size_t)numel),receive((size_t)numel);
            for(int64_t i=0;i<numel;++i)send[(size_t)i]=bf(sample(trial,rank,i));
            hipok(hipMemcpy(din,send.data(),send.size()*2,hipMemcpyHostToDevice),"hipMemcpy input");
            if(trial.oop)hipok(hipMemset(dout,0,send.size()*2),"hipMemset output");
            // Reuse the kernel's live instance and exact mode across every
            // sequential collective (including alternating in-/out-of-place).
            RadTensor ts[2]={tbf(din,numel),tbf(dout,numel)};
            RadArgs a{};a.t=ts;a.n_t=trial.oop?2:1;
            std::vector<RadParam> p{
                pi("world_size",3),pi("numel",numel),ps("dtype","bf16"),pi("exact",1)
            };
            a.p=p.data();a.n_p=(int)p.size();
            a.instance=instance;a.rank=rank;a.world_size=3;
            std::fprintf(stderr,"rank%d: native launch %s numel=%lld x=%p y=%p\n",
                         rank,trial.label,(long long)numel,din,trial.oop?dout:din);
#if defined(LTX_C08_AQL_ENGINE)
            rad_dev_peer_launch(1);
#endif
            const int rc=k->launch(&a,stream);
#if defined(LTX_C08_AQL_ENGINE)
            rad_dev_peer_launch(0);
#endif
            if(rc!=RAD_OK && result->rc_launch==RAD_OK) result->rc_launch=rc;
#if defined(LTX_C08_AQL_ENGINE)
            check(rad_stream_sync((RadStream)stream)==RAD_OK,
                  std::string("Radiance AQL collective sync: ")+rad_dev_last_error());
#else
            hipok(hipStreamSynchronize(stream),"collective sync");
#endif
            hipok(hipMemcpy(receive.data(),trial.oop?dout:din,send.size()*2,hipMemcpyDeviceToHost),
                  "hipMemcpy output");
            int64_t bad=0;
            for(int64_t i=0;i<numel;++i) {
                float sum=0;
                for(int r=0;r<3;++r)sum+=f32(bf(sample(trial,r,i)));
                if(receive[(size_t)i]!=bf(sum))++bad;
            }
            result->wrong.push_back(bad);
        }
    } catch(const std::exception& e) { result->error=e.what(); }
    if(din)(void)hipFree(din);
    if(dout)(void)hipFree(dout);
#if defined(LTX_C08_AQL_ENGINE)
    if(stream)rad_stream_destroy((RadStream)stream);
#else
    if(stream)(void)hipStreamDestroy(stream);
#endif
    if(instance && k->fini)k->fini(instance);
}
} // namespace

int main(int argc,char** argv) {
    if(argc<4 || argc>5) {
        std::fprintf(stderr,"usage: mixed_ar_test <libr3.so> <libr4d.so> <row-name> [0,1,2]\n");
        return 2;
    }
    try {
#if defined(LTX_C08_AQL_ENGINE)
        const char* backend=rad::device_backend_name();
        std::printf("Radiance device backend: %s\n",backend);
        check(std::strcmp(backend,"aql")==0,
              "native GPU engine dispatch must use the real AQL backend");
#endif
        const std::string row=argv[3];
        check(row=="ar_oneshot_3rank_exact" || row=="ar_twoshot_3rank_exact" ||
              row=="ar_oneshot_3rank_exact_host" || row=="ar_twoshot_3rank_exact_host",
              "native exact 3-rank row required");
        Plugin r4(argv[2]),r3(argv[1]);
        auto* k3=r3.find(row),*k4=r4.find(row);
        check(k3 && k4 && k3->init && k4->init && k3->launch && k4->launch,
              "same native exact row must exist on both gfx11/gfx12");
        std::array<int,3> dev={0,1,2};
        if(argc==5) {
            check(std::sscanf(argv[4],"%d,%d,%d",&dev[0],&dev[1],&dev[2])==3,
                  "devices must be n,n,n");
        }
        int count=0;hipok(hipGetDeviceCount(&count),"hipGetDeviceCount");
        check(count>std::max({dev[0],dev[1],dev[2]}),"not enough HIP GPUs");
        std::array<bool,3> uses_r4{};
        for(int rank=0;rank<3;++rank) {
            hipDeviceProp_t prop{};
            hipok(hipGetDeviceProperties(&prop,dev[(size_t)rank]),"device properties");
            const std::string arch=prop.gcnArchName;
            check(arch.rfind("gfx1100",0)==0 || arch.rfind("gfx1201",0)==0,
                  "rank "+std::to_string(rank)+" unsupported GPU architecture");
            uses_r4[(size_t)rank]=arch.rfind("gfx1201",0)==0;
            std::printf("rank=%d device=%d arch=%s native=%s\n",
                        rank,dev[(size_t)rank],prop.gcnArchName,
                        uses_r4[(size_t)rank]?"libr4d":"libr3");
        }
        for(int a=0;a<3;++a)for(int b=0;b<3;++b)if(a!=b) {
            int peer=0;hipok(hipDeviceCanAccessPeer(&peer,dev[(size_t)a],dev[(size_t)b]),"P2P check");
            std::printf("peer rank%d->rank%d: %d\n",a,b,peer);
        }
        std::fflush(stdout);
        std::array<Result,3> result{};
        std::thread t0(run_rank,0,dev[0],uses_r4[0]?k4:k3,&result[0]);
        std::thread t1(run_rank,1,dev[1],uses_r4[1]?k4:k3,&result[1]);
        std::thread t2(run_rank,2,dev[2],uses_r4[2]?k4:k3,&result[2]);
        t0.join();t1.join();t2.join();
        int failures=0;
        for(int r=0;r<3;++r) {
            const auto& got=result[(size_t)r];
            std::printf("rank%d init=%d launch=%d error=%s\n",r,got.rc_init,got.rc_launch,
                        got.error.empty()?"none":got.error.c_str());
            if(got.rc_init!=RAD_OK || got.rc_launch!=RAD_OK || !got.error.empty() ||
               got.wrong.size()!=trials.size())++failures;
            for(size_t j=0;j<got.wrong.size();++j) {
                std::printf("  %s: %lld mismatched BF16 elements\n",trials[j].label,
                            (long long)got.wrong[j]);
                if(got.wrong[j])++failures;
            }
        }
        const auto keys=rad::bc_shared_slot_keys("libr4d.ar.rendezvous/v2/group=");
        check(!keys.empty(),"actual native shared-slot rendezvous never used");
        for(const auto& key:keys) {
            const auto users=rad::bc_shared_slot_users(key);
            std::printf("shared rendezvous group=%s users=",key.c_str());
            for(const auto& u:users)std::printf("%s,",u.c_str());
            std::printf("\n");
            const bool mixed=uses_r4[0]!=uses_r4[1] || uses_r4[1]!=uses_r4[2];
            check(users.size()==(mixed?2U:1U),
                  "native shared-slot users do not match participating libraries");
        }
        std::printf("native exact-3rank %s: %s (%d failures)\n",
                    row.c_str(),failures?"FAIL":"PASS",failures);
        return failures?1:0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"native mixed-card probe: %s\n",e.what());
        return 1;
    }
}
