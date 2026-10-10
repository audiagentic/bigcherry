// Separate gfx1100 bandwidth candidate: one thread reads/writes two BF16s as a dword.
// Compare against elementwise.hip's one-element/thread path in rad-kbench.
// Only even-width, 4-byte aligned, dense BF16 operands. No hidden fallback.
#include "rad_abi.h"
#include "rad_plugin.h"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>

namespace {
__device__ __forceinline__ float widen(uint16_t v) {
    return __uint_as_float(uint32_t(v)<<16);
}
__device__ __forceinline__ uint16_t narrow(float f) {
    uint32_t v=__float_as_uint(f);
    if((v&0x7fffffffu)>0x7f800000u)return uint16_t((v>>16)|0x40u);
    return uint16_t((v+0x7fffu+((v>>16)&1u))>>16);
}
template<bool MUL,bool ALIGNED>
__global__ void pair_kernel(const uint16_t* a,const uint16_t* b,uint16_t* y,
                            uint32_t pairs,uint32_t n_pairs,bool broadcast) {
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=pairs)return;
    uint32_t bi=broadcast?(i%n_pairs):i;
    uint32_t av,bv;
    if constexpr(ALIGNED) {
        av=reinterpret_cast<const uint32_t*>(a)[i];
        bv=reinterpret_cast<const uint32_t*>(b)[bi];
    } else {
        av=uint32_t(a[2*i])|(uint32_t(a[2*i+1])<<16);
        bv=uint32_t(b[2*bi])|(uint32_t(b[2*bi+1])<<16);
    }
    float a0=widen(uint16_t(av)),a1=widen(uint16_t(av>>16));
    float b0=widen(uint16_t(bv)),b1=widen(uint16_t(bv>>16));
    uint32_t c0=uint32_t(narrow(MUL?a0*b0:a0+b0));
    uint32_t c1=uint32_t(narrow(MUL?a1*b1:a1+b1));
    uint32_t packed=c0|(c1<<16);
    if constexpr(ALIGNED) reinterpret_cast<uint32_t*>(y)[i]=packed;
    else {y[2*i]=uint16_t(packed);y[2*i+1]=uint16_t(packed>>16);}
}
template<bool MUL>
int run(const RadArgs* args,RadStream stream){
    if(!args||!args->t||args->n_t<3)return RAD_E_INVAL;
    const RadTensor& a=args->t[0],&b=args->t[1],&y=args->t[2];
    if(!a.data||!b.data||!y.data)return RAD_E_INVAL;
    const char* d=rad_args_gets(args,"dtype");
    if(!d||std::strcmp(d,"bf16")||a.dtype!=RAD_BF16||b.dtype!=RAD_BF16||
       y.dtype!=RAD_BF16)return RAD_E_DTYPE;
    long long m=0,n=0;
    if(!rad_args_geti(args,"M",&m)||!rad_args_geti(args,"n",&n)||
       m<1||m>512||n<2||n>8192||n%2)return RAD_E_SHAPE;
    const long long total=m*n;
    const long long bn=rad_tensor_numel(&b);
    const bool broadcast=m>1&&bn==n;
    if(rad_tensor_numel(&a)!=total||rad_tensor_numel(&y)!=total||
       (bn!=total&&!broadcast))return RAD_E_SHAPE;
    if(!rad_tensor_is_contiguous(&a)||!rad_tensor_is_contiguous(&b)||
       !rad_tensor_is_contiguous(&y))return RAD_E_STRIDE;
    if(broadcast&&b.data==y.data)return RAD_E_INVAL;
    uint32_t pairs=uint32_t(total/2),n_pairs=uint32_t(n/2);
    const bool aligned=((reinterpret_cast<uintptr_t>(a.data)|
        reinterpret_cast<uintptr_t>(b.data)|
        reinterpret_cast<uintptr_t>(y.data))&3u)==0;
    auto srca=static_cast<const uint16_t*>(a.data);
    auto srcb=static_cast<const uint16_t*>(b.data);
    auto dst=static_cast<uint16_t*>(y.data);
    auto hipstream=reinterpret_cast<hipStream_t>(stream);
    if(aligned) {
        hipLaunchKernelGGL((pair_kernel<MUL,true>),dim3((pairs+255u)/256u),
                           dim3(256),0,hipstream,srca,srcb,dst,pairs,n_pairs,broadcast);
    } else {
        hipLaunchKernelGGL((pair_kernel<MUL,false>),dim3((pairs+255u)/256u),
                           dim3(256),0,hipstream,srca,srcb,dst,pairs,n_pairs,broadcast);
    }
    return hipGetLastError()==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
}
extern "C" int r11_add_bf16_packed(const RadArgs* a,RadStream s){return run<false>(a,s);}
extern "C" int r11_mul_bf16_packed(const RadArgs* a,RadStream s){return run<true>(a,s);}
