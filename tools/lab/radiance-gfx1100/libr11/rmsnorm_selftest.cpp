// Standalone hardware smoke for independent RDNA3 wave32 and block256 RMSNorm.
#include "rad_abi.h"
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
using Launch=int (*)(const RadArgs*,RadStream);
static uint16_t bf(float f) {
    uint32_t u;std::memcpy(&u,&f,4);
    if((u&0x7fffffffu)>0x7f800000u)return uint16_t((u>>16)|0x40u);
    return uint16_t((u+0x7fffu+((u>>16)&1u))>>16);
}
static float f32(uint16_t h) {
    uint32_t u=uint32_t(h)<<16;float f;std::memcpy(&f,&u,4);return f;
}
static int single(Launch a,Launch b,int m,int n,bool wf32,float wadd) {
    const int xld=n+32, yld=n+32;
    std::vector<uint16_t> x(size_t(m)*xld),y(size_t(m)*yld,0xbaad),wbf(n),result(size_t(m)*yld);
    std::vector<float> wf(n),oracle(size_t(m)*yld);
    for(int r=0;r<m;++r)for(int c=0;c<n;++c) {
        float v=(float((r*17+c*13)%37)-18.f)/16.f;
        x[size_t(r)*xld+c]=bf(v);
    }
    for(int c=0;c<n;++c) {
        wf[c]=float((c*3)%11-5)/32.f;
        wbf[c]=bf(wf[c]);
    }
    float eps=1e-6f;
    for(int row=0;row<m;++row) {
        double ss=0;
        for(int c=0;c<n;++c) {
            double v=f32(x[size_t(row)*xld+c]);ss+=v*v;
        }
        float scale=1.f/std::sqrt(float(ss/n)+eps);
        for(int c=0;c<n;++c) {
            float g=wf32?wf[c]:f32(wbf[c]);
            oracle[size_t(row)*yld+c]=f32(bf((f32(x[size_t(row)*xld+c])*scale)*(g+wadd)));
        }
    }
    uint16_t *dx=nullptr,*dy=nullptr,*dw16=nullptr;
    float *dw32=nullptr;
    hipStream_t stream=nullptr;
    if(hipMalloc(reinterpret_cast<void**>(&dx),x.size()*2)!=hipSuccess ||
       hipMalloc(reinterpret_cast<void**>(&dy),y.size()*2)!=hipSuccess ||
       hipMalloc(reinterpret_cast<void**>(&dw16),wbf.size()*2)!=hipSuccess ||
       hipMalloc(reinterpret_cast<void**>(&dw32),wf.size()*4)!=hipSuccess ||
       hipStreamCreate(&stream)!=hipSuccess) {
       std::fprintf(stderr,"HIP allocation/stream failed\n");return 1;
    }
    if(hipMemcpy(dx,x.data(),x.size()*2,hipMemcpyHostToDevice)!=hipSuccess ||
       hipMemcpy(dw16,wbf.data(),wbf.size()*2,hipMemcpyHostToDevice)!=hipSuccess ||
       hipMemcpy(dw32,wf.data(),wf.size()*4,hipMemcpyHostToDevice)!=hipSuccess)return 1;
    RadTensor t[]={
        {dx,RAD_BF16,2,{m,n},{xld,1}},
        {wf32?static_cast<void*>(dw32):static_cast<void*>(dw16),wf32?RAD_F32:RAD_BF16,1,{n},{1}},
        {dy,RAD_BF16,2,{m,n},{yld,1}}
    };
    RadParam p[]={
        {"M",RAD_P_INT,m,0,nullptr,0.0},
        {"n",RAD_P_INT,n,0,nullptr,0.0},
        {"dtype",RAD_P_STR,0,0,"bf16",0.0},
        {"eps",RAD_P_F64,0,0,nullptr,double(eps)},
        {"wadd",RAD_P_F64,0,0,nullptr,double(wadd)}
    };
    RadArgs args{};args.t=t;args.n_t=3;args.p=p;args.n_p=5;
    int failures=0;
    for(int variant=0;variant<2;++variant) {
        std::fill(y.begin(),y.end(),0xbaad);
        if(hipMemcpy(dy,y.data(),y.size()*2,hipMemcpyHostToDevice)!=hipSuccess)return 1;
        int rc=(variant?b:a)(&args,reinterpret_cast<RadStream>(stream));
        if(rc!=RAD_OK || hipStreamSynchronize(stream)!=hipSuccess ||
           hipMemcpy(result.data(),dy,result.size()*2,hipMemcpyDeviceToHost)!=hipSuccess){
            std::fprintf(stderr,"FAIL launch variant %d M=%d n=%d rc=%d\n",variant,m,n,rc);
            ++failures;continue;
        }
        for(int row=0;row<m;++row)for(int c=0;c<yld;++c){
            size_t i=size_t(row)*yld+c;
            if(c>=n) {
                if(result[i]!=0xbaad) {
                    std::fprintf(stderr,"FAIL redzone variant %d row=%d col=%d\n",variant,row,c);
                    ++failures;break;
                }
                continue;
            }
            const float got=f32(result[i]),ref=oracle[i];
            if(!std::isfinite(got) ||
               std::fabs(got-ref)>0.015f+0.01f*std::fabs(ref)){
                std::fprintf(stderr,"FAIL numerics variant %d row=%d col=%d got=%g ref=%g\n",
                             variant,row,c,double(got),double(ref));
                ++failures;break;
            }
        }
    }
    // Mis-specified row stride must fail before a GPU launch.
    t[0].stride[1]=2;
    if(a(&args,reinterpret_cast<RadStream>(stream))!=RAD_E_STRIDE ||
       b(&args,reinterpret_cast<RadStream>(stream))!=RAD_E_STRIDE)++failures;
    hipStreamDestroy(stream);hipFree(dw32);hipFree(dw16);hipFree(dy);hipFree(dx);
    if(!failures)std::printf("PASS norm M=%d n=%d wf32=%d wadd=%g\n",m,n,wf32,double(wadd));
    return failures?1:0;
}
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    void* lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if(!lib){std::fprintf(stderr,"dlopen %s\n",dlerror());return 1;}
    auto version=reinterpret_cast<uint32_t(*)()>(dlsym(lib,"rad_plugin_abi_version"));
    auto wave=reinterpret_cast<Launch>(dlsym(lib,"r11_rmsnorm_wave32"));
    auto block=reinterpret_cast<Launch>(dlsym(lib,"r11_rmsnorm_block256"));
    if(!version||version()!=RAD_ABI_VERSION||!wave||!block)return 1;
    if(hipSetDevice(0)!=hipSuccess)return 1;
    hipDeviceProp_t p{};
    if(hipGetDeviceProperties(&p,0)!=hipSuccess ||
       std::strncmp(p.gcnArchName,"gfx1100",7)!=0) {
        std::fprintf(stderr,"requires gfx1100, found %s\n",p.gcnArchName);return 2;
    }
    int fail=0;
    fail|=single(wave,block,1,32,true,0);
    fail|=single(wave,block,3,256,true,1);
    fail|=single(wave,block,2,5120,true,1);
    fail|=single(wave,block,4,1024,false,0);
    dlclose(lib);
    return fail?1:0;
}
