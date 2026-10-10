// gfx1100 BF16 GEMM candidates: scalar reference and native wave32 WMMA.
// Y[M,N] = A[M,K] * B[N,K]^T, all row-major. Experimental: M <= 16.
#include "rad_abi.h"
#include "rad_plugin.h"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>

namespace {
__device__ __forceinline__ float widen(uint16_t v) {
    return __uint_as_float(uint32_t(v) << 16);
}
__device__ __forceinline__ uint16_t narrow(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40u);
    return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
struct Gemm { const uint16_t* a; const uint16_t* b; uint16_t* y; uint32_t m,n,k; };
int check(const RadArgs* args, Gemm* g) {
    if (!args || !args->t || args->n_t < 3) return RAD_E_INVAL;
    const RadTensor& a=args->t[0], &b=args->t[1], &y=args->t[2];
    if (!a.data || !b.data || !y.data || a.data==y.data || b.data==y.data)
        return RAD_E_INVAL;
    const char* d=rad_args_gets(args,"dtype");
    if (!d || std::strcmp(d,"bf16") || a.dtype!=RAD_BF16 ||
        b.dtype!=RAD_BF16 || y.dtype!=RAD_BF16) return RAD_E_DTYPE;
    long long m=0,n=0,k=0;
    if (!rad_args_geti(args,"M",&m) || !rad_args_geti(args,"N",&n) ||
        !rad_args_geti(args,"K",&k) || m<1 || m>16 || n<1 || n>32768 ||
        k<16 || k>8192 || k%16) return RAD_E_SHAPE;
    if (a.rank!=2 || b.rank!=2 || y.rank!=2 ||
        a.shape[0]!=m || a.shape[1]!=k || b.shape[0]!=n ||
        b.shape[1]!=k || y.shape[0]!=m || y.shape[1]!=n)
        return RAD_E_SHAPE;
    if (!rad_tensor_is_contiguous(&a) || !rad_tensor_is_contiguous(&b) ||
        !rad_tensor_is_contiguous(&y)) return RAD_E_STRIDE;
    *g={static_cast<const uint16_t*>(a.data),static_cast<const uint16_t*>(b.data),
        static_cast<uint16_t*>(y.data),uint32_t(m),uint32_t(n),uint32_t(k)};
    return RAD_OK;
}
__global__ void scalar_gemm(Gemm g) {
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=g.m*g.n) return;
    uint32_t row=i/g.n,col=i%g.n;
    float sum=0.f;
    for (uint32_t k=0;k<g.k;++k)
        sum+=widen(g.a[row*g.k+k])*widen(g.b[col*g.k+k]);
    g.y[i]=narrow(sum);
}
// RDNA3: both lane halves load identical input fragments indexed by lane%16.
// Output row = 2*acc_index + lane/16 (RDNA4's mapping is DIFFERENT).
using u16x16=uint16_t __attribute__((ext_vector_type(16)));
using bf16x16=__bf16 __attribute__((ext_vector_type(16)));
using f32x8=float __attribute__((ext_vector_type(8)));
__global__ __launch_bounds__(32) void wmma_gfx1100(Gemm g) {
    uint32_t lane=threadIdx.x;
    uint32_t arow=blockIdx.y*16u+(lane&15u);
    uint32_t bcol=blockIdx.x*16u+(lane&15u);
    f32x8 acc={};
    for(uint32_t k=0;k<g.k;k+=16u){
        u16x16 ai={},bi={};
#pragma unroll
        for(int e=0;e<16;++e){
            ai[e]=arow<g.m?g.a[arow*g.k+k+uint32_t(e)]:0;
            bi[e]=bcol<g.n?g.b[bcol*g.k+k+uint32_t(e)]:0;
        }
        acc=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
            __builtin_bit_cast(bf16x16,ai),
            __builtin_bit_cast(bf16x16,bi),acc);
    }
#pragma unroll
    for(int j=0;j<8;++j){
        uint32_t row=blockIdx.y*16u+uint32_t(2*j)+(lane>>4);
        if(row<g.m && bcol<g.n) g.y[row*g.n+bcol]=narrow(acc[j]);
    }
}
int run(const RadArgs* a,RadStream stream,bool wmma) {
    Gemm g{};
    int status=check(a,&g);
    if(status!=RAD_OK) return status;
    auto s=reinterpret_cast<hipStream_t>(stream);
    if(wmma) {
        hipLaunchKernelGGL(wmma_gfx1100,dim3((g.n+15u)/16u,(g.m+15u)/16u),
            dim3(32),0,s,g);
    } else {
        uint32_t count=g.m*g.n;
        hipLaunchKernelGGL(scalar_gemm,dim3((count+255u)/256u),dim3(256),0,s,g);
    }
    return hipGetLastError()==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
}
extern "C" int r11_gemm_bf16_scalar(const RadArgs* a,RadStream s){return run(a,s,false);}
extern "C" int r11_gemm_bf16_wmma(const RadArgs* a,RadStream s){return run(a,s,true);}
