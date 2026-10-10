// Independent gfx1100 RMSNorm candidates: one wave/row versus eight waves/row.
// Radiance: y = bf16(x * rsqrt(sum(x*x)/n + eps) * (w + wadd)).
// Reference: radiance/libr4d/r4d_model_bf16.hip and libref; weight may be F32 or BF16.
// The two variants are for side-by-side rad-kbench, never a silent production replacement.
#include "rad_abi.h"
#include "rad_plugin.h"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>
#include <cstring>

namespace {
__device__ __forceinline__ float widen(uint16_t bits) {
    return __uint_as_float(uint32_t(bits) << 16);
}
__device__ __forceinline__ uint16_t narrow(float x) {
    uint32_t bits=__float_as_uint(x);
    if ((bits & 0x7fffffffu) > 0x7f800000u)
        return uint16_t((bits >> 16) | 0x40u);
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
struct Norm {
    const uint16_t* x;
    const void* w;
    uint16_t* y;
    uint32_t m,n,xld,yld;
    float eps,wadd;
    bool wf32;
};
int check(const RadArgs* args, Norm* g) {
    if (!args || !args->t || args->n_t<3) return RAD_E_INVAL;
    const auto& x=args->t[0];
    const auto& w=args->t[1];
    const auto& y=args->t[2];
    if (!x.data || !w.data || !y.data || w.data==y.data) return RAD_E_INVAL;
    const char* dtype=rad_args_gets(args,"dtype");
    if (!dtype || std::strcmp(dtype,"bf16") ||
        x.dtype!=RAD_BF16 || y.dtype!=RAD_BF16 ||
        (w.dtype!=RAD_F32 && w.dtype!=RAD_BF16)) return RAD_E_DTYPE;
    long long m=0,n=0;
    if (!rad_args_geti(args,"M",&m) || !rad_args_geti(args,"n",&n) ||
        m<1 || m>1024 || n<32 || n>8192 || (n%32)) return RAD_E_SHAPE;
    if (x.rank!=2 || y.rank!=2 || w.rank!=1 ||
        x.shape[0]!=m || y.shape[0]!=m || x.shape[1]!=n || y.shape[1]!=n ||
        w.shape[0]<n || w.stride[0]!=1) return RAD_E_SHAPE;
    if (x.stride[1]!=1 || y.stride[1]!=1 ||
        x.stride[0]<n || y.stride[0]<n) return RAD_E_STRIDE;
    const double eps=rad_args_getf_or(args,"eps",1e-6);
    const double wadd=rad_args_getf_or(args,"wadd",0.0);
    if (!std::isfinite(eps) || eps<0.0 || !std::isfinite(wadd))
        return RAD_E_INVAL;
    *g={static_cast<const uint16_t*>(x.data),w.data,
        static_cast<uint16_t*>(y.data),uint32_t(m),uint32_t(n),
        uint32_t(x.stride[0]),uint32_t(y.stride[0]),
        float(eps),float(wadd),w.dtype==RAD_F32};
    return RAD_OK;
}
__device__ __forceinline__ float wave_sum(float v) {
    for (int delta=16;delta>0;delta>>=1)
        v += __shfl_down(v,delta,32);
    return __shfl(v,0,32);
}
__device__ __forceinline__ float gain(const Norm& g,uint32_t col) {
    return (g.wf32 ? static_cast<const float*>(g.w)[col] :
            widen(static_cast<const uint16_t*>(g.w)[col])) + g.wadd;
}
// One full gfx1100 wave32 per row; no LDS, no inter-wave barriers.
__global__ __launch_bounds__(32) void rms_wave32(Norm g) {
    uint32_t lane=threadIdx.x;
    uint32_t row=blockIdx.x;
    float sum=0.f;
    for(uint32_t c=lane;c<g.n;c+=32) {
        float v=widen(g.x[size_t(row)*g.xld+c]);
        sum += v*v;
    }
    sum=wave_sum(sum);
    float inv=rsqrtf(sum/float(g.n)+g.eps);
    for(uint32_t c=lane;c<g.n;c+=32) {
        float v=widen(g.x[size_t(row)*g.xld+c]);
        g.y[size_t(row)*g.yld+c]=narrow((v*inv)*gain(g,c));
    }
}
// AMD ROCm wavefront guide: gfx11 has DPP quad/row operations but NOT
// gfx9 row_bcast15/31. After row-wise permutations, ds_swizzle(0x1e0)
// supplies the other 16-lane row; lane 31 holds the total.
// Reference: rocm-handbook AMD optimization guide / wavefront builtins.
__device__ __forceinline__ float dpp_wave_sum(float sum) {
    constexpr int quad1032=0xb1,quad2301=0x4e,row4=0x124,row8=0x128;
    int bits=__float_as_uint(sum);
    sum += __uint_as_float(__builtin_amdgcn_mov_dpp(bits,quad1032,0xf,0xf,false));
    bits=__float_as_uint(sum);
    sum += __uint_as_float(__builtin_amdgcn_mov_dpp(bits,quad2301,0xf,0xf,false));
    bits=__float_as_uint(sum);
    sum += __uint_as_float(__builtin_amdgcn_mov_dpp(bits,row4,0xf,0xf,false));
    bits=__float_as_uint(sum);
    sum += __uint_as_float(__builtin_amdgcn_mov_dpp(bits,row8,0xf,0xf,false));
    bits=__float_as_uint(sum);
    sum += __uint_as_float(__builtin_amdgcn_ds_swizzle(bits,0x1e0));
    // Broadcast the lane-31 value; don't use gfx9-only row-broadcast DPP.
    return __uint_as_float(__builtin_amdgcn_readlane(__float_as_uint(sum),31));
}
// One wave per row, identical data layout to rms_wave32 but native gfx11 DPP.
__global__ __launch_bounds__(32) void rms_dpp32(Norm g) {
    uint32_t lane=threadIdx.x,row=blockIdx.x;
    float sum=0.f;
    for(uint32_t c=lane;c<g.n;c+=32) {
        float v=widen(g.x[size_t(row)*g.xld+c]);
        sum += v*v;
    }
    float inv=rsqrtf(dpp_wave_sum(sum)/float(g.n)+g.eps);
    for(uint32_t c=lane;c<g.n;c+=32) {
        float v=widen(g.x[size_t(row)*g.xld+c]);
        g.y[size_t(row)*g.yld+c]=narrow((v*inv)*gain(g,c));
    }
}
// Eight waves cooperate on a row; a single LDS barrier combines wave sums.
__global__ __launch_bounds__(256) void rms_block256(Norm g) {
    __shared__ float partials[8];
    uint32_t t=threadIdx.x;
    uint32_t row=blockIdx.x;
    float sum=0.f;
    for(uint32_t c=t;c<g.n;c+=256) {
        float v=widen(g.x[size_t(row)*g.xld+c]);
        sum += v*v;
    }
    sum=wave_sum(sum);
    if((t&31u)==0) partials[t>>5]=sum;
    __syncthreads();
    float total=0.f;
#pragma unroll
    for(int w=0;w<8;++w) total+=partials[w];
    float inv=rsqrtf(total/float(g.n)+g.eps);
    for(uint32_t c=t;c<g.n;c+=256) {
        float v=widen(g.x[size_t(row)*g.xld+c]);
        g.y[size_t(row)*g.yld+c]=narrow((v*inv)*gain(g,c));
    }
}
int run(const RadArgs* args, RadStream stream, int variant) {
    Norm g{};
    int status=check(args,&g);
    if(status!=RAD_OK) return status;
    hipStream_t s=reinterpret_cast<hipStream_t>(stream);
    if(variant==0)
        hipLaunchKernelGGL(rms_wave32,dim3(g.m),dim3(32),0,s,g);
    else if(variant==1)
        hipLaunchKernelGGL(rms_block256,dim3(g.m),dim3(256),0,s,g);
    else
        hipLaunchKernelGGL(rms_dpp32,dim3(g.m),dim3(32),0,s,g);
    return hipGetLastError()==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
}
extern "C" int r11_rmsnorm_wave32(const RadArgs* a,RadStream s){return run(a,s,0);}
extern "C" int r11_rmsnorm_block256(const RadArgs* a,RadStream s){return run(a,s,1);}
extern "C" int r11_rmsnorm_dpp32(const RadArgs* a,RadStream s){return run(a,s,2);}
