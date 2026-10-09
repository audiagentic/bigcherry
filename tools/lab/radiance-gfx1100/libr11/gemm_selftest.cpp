// Hardware-only: test both RadArgs GEMM candidates with asymmetric values and odd N.
#include "rad_abi.h"
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
using Launch=int (*)(const RadArgs*,RadStream);
static float widen(uint16_t h) {
    uint32_t v=uint32_t(h)<<16; float f; std::memcpy(&f,&v,4); return f;
}
static uint16_t narrow(float f) {
    uint32_t u; std::memcpy(&u,&f,4);
    if((u&0x7fffffffu)>0x7f800000u) return uint16_t((u>>16)|0x40u);
    return uint16_t((u+0x7fffu+((u>>16)&1u))>>16);
}
static int run(Launch scalar,Launch wmma,int m,int n,int k) {
    static const uint16_t bits[]={0x0000,0x3e80,0xbe80,0x3f00,0xbf00,0x3f80,0x4000,0x4040};
    std::vector<uint16_t> a(size_t(m)*k),b(size_t(n)*k),expect(size_t(m)*n),got(size_t(m)*n);
    for(size_t i=0;i<a.size();++i) a[i]=bits[(i*3+1)%8];
    for(size_t i=0;i<b.size();++i) b[i]=bits[(i*5+3)%8];
    for(int i=0;i<m;++i) for(int j=0;j<n;++j) {
        float v=0;
        for(int q=0;q<k;++q) v+=widen(a[size_t(i)*k+q])*widen(b[size_t(j)*k+q]);
        expect[size_t(i)*n+j]=narrow(v);
    }
    uint16_t *da=nullptr,*db=nullptr,*dy=nullptr;
    hipStream_t stream=nullptr;
    if(hipMalloc(reinterpret_cast<void**>(&da),a.size()*2)!=hipSuccess ||
       hipMalloc(reinterpret_cast<void**>(&db),b.size()*2)!=hipSuccess ||
       hipMalloc(reinterpret_cast<void**>(&dy),got.size()*2)!=hipSuccess ||
       hipStreamCreate(&stream)!=hipSuccess) {
        std::fprintf(stderr,"HIP allocation/stream failed\n"); return 1;
    }
    if(hipMemcpy(da,a.data(),a.size()*2,hipMemcpyHostToDevice)!=hipSuccess ||
       hipMemcpy(db,b.data(),b.size()*2,hipMemcpyHostToDevice)!=hipSuccess) return 1;
    RadTensor t[]={
        {da,RAD_BF16,2,{m,k},{k,1}},
        {db,RAD_BF16,2,{n,k},{k,1}},
        {dy,RAD_BF16,2,{m,n},{n,1}},
    };
    RadParam p[]={
        {"M",RAD_P_INT,m,0,nullptr,0.0},
        {"N",RAD_P_INT,n,0,nullptr,0.0},
        {"K",RAD_P_INT,k,0,nullptr,0.0},
        {"dtype",RAD_P_STR,0,0,"bf16",0.0},
    };
    RadArgs args{}; args.t=t;args.n_t=3;args.p=p;args.n_p=4;
    int failed=0;
    for(int which=0;which<2;++which) {
        const int rc=(which?wmma:scalar)(&args,reinterpret_cast<RadStream>(stream));
        if(rc!=RAD_OK || hipStreamSynchronize(stream)!=hipSuccess ||
           hipMemcpy(got.data(),dy,got.size()*2,hipMemcpyDeviceToHost)!=hipSuccess ||
           got!=expect) {
            std::fprintf(stderr,"FAIL %s M=%d N=%d K=%d rc=%d\n",
                         which?"wmma":"scalar",m,n,k,rc);
            if(rc==RAD_OK) for(size_t i=0;i<got.size();++i) if(got[i]!=expect[i]){
                std::fprintf(stderr,"first mismatch [%zu] got=%04x expected=%04x\n",
                             i,unsigned(got[i]),unsigned(expect[i]));break;}
            failed=1;
        }
    }
    // K not a multiple of 16 must be rejected before touching memory.
    p[2]={"K",RAD_P_INT,k-1,0,nullptr,0.0};
    if(scalar(&args,reinterpret_cast<RadStream>(stream))!=RAD_E_SHAPE ||
       wmma(&args,reinterpret_cast<RadStream>(stream))!=RAD_E_SHAPE) failed=1;
    p[2]={"K",RAD_P_INT,k,0,nullptr,0.0};
    t[2].stride[1]=2;
    if(scalar(&args,reinterpret_cast<RadStream>(stream))!=RAD_E_STRIDE ||
       wmma(&args,reinterpret_cast<RadStream>(stream))!=RAD_E_STRIDE) failed=1;
    hipStreamDestroy(stream);hipFree(dy);hipFree(db);hipFree(da);
    if(!failed) std::printf("PASS gfx1100 gemm M=%d N=%d K=%d (both candidates)\n",m,n,k);
    return failed;
}
int main(int argc,char** argv) {
    if(argc!=2){std::fprintf(stderr,"usage: %s libr11.so\n",argv[0]);return 2;}
    void* so=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if(!so){std::fprintf(stderr,"dlopen: %s\n",dlerror());return 1;}
    auto version=reinterpret_cast<uint32_t(*)()>(dlsym(so,"rad_plugin_abi_version"));
    auto scalar=reinterpret_cast<Launch>(dlsym(so,"r11_gemm_bf16_scalar"));
    auto wmma=reinterpret_cast<Launch>(dlsym(so,"r11_gemm_bf16_wmma"));
    if(!version||version()!=RAD_ABI_VERSION||!scalar||!wmma)return 1;
    if(hipSetDevice(0)!=hipSuccess)return 1;
    hipDeviceProp_t d{};
    if(hipGetDeviceProperties(&d,0)!=hipSuccess ||
       std::strncmp(d.gcnArchName,"gfx1100",7)!=0){
        std::fprintf(stderr,"requires a visible gfx1100 device, got %s\n",d.gcnArchName);
        return 2;
    }
    int fail=0;
    fail|=run(scalar,wmma,1,1,16);
    fail|=run(scalar,wmma,9,17,32);
    fail|=run(scalar,wmma,16,33,64);
    fail|=run(scalar,wmma,7,63,256);
    dlclose(so);
    return fail?1:0;
}
