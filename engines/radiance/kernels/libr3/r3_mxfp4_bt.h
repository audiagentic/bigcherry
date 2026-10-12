// r3_mxfp4_bt.h - gfx1100 MXFP4 x fp8 PREFILL GEMM: one wave, one weight row per lane.
//
// DESIGN (RR08). This is a libr3-owned header included by the rewritten r4d_gemm_mxfp4a8_tiled unit.
// The upstream registry row, layout hooks and scratch callback stay intact; R3_BT=1 changes only its launch.
//
// - A gfx11 lane owns weight row n0 + (lane & 15). Per 16-of-K step it reads that row's two
//   fragment-order dwords, folds the per-32 E8M0 exponent against Wref, expands the resulting E4M3
//   bytes to f16 in registers, and reuses that fragment across BT independent 16-row activation tiles.
// - Incoming E4M3 activations are widened once into row-major f16 in the op's existing scratch.
//   A lane then loads one token's 16 K values as one aligned 32-byte gfx11 fragment. The native tiled
//   rewrite already asks the row scratch hook for two bytes per activation element; Radiance is unchanged.
// - Both operands use f16, not bf16. The measured block-fp8 GEMM on gfx1100 was 2.4x faster after the
//   same E4M3-bit to f16 widening. A widened E4M3 value is exactly 2^-8 of its value, including
//   subnormals, so the f32 accumulator is multiplied by 2^16 once in the epilogue.
// - BT native-layout v8f accumulators stay in gfx11 layout until store. BT=8 is 64 f32 accumulator
//   values per lane plus live fragments/scalars. launch_bounds asks for two resident waves; VGPR
//   pressure, not LDS, is expected to set occupancy. R3_BT_B={4,6,8} exposes that tradeoff.
// - Per-(row,128) activation scales rescale each accumulator at a boundary into the next non-zero
//   scale's units. A zero-scale block has zero codes and keeps the previous carried scale.
// - Grid = [N/16, ceil(M/(16*BT))], one wave per block. M=64..512 is a real matrix product, so the
//   decode result "one token row is not a matrix product" does not apply. The gfx1100 result that
//   streaming wants roughly 1,000 waves does apply; wide prefill shapes naturally provide that work.
//
// LIMITS. Every N tile reloads its BT activation rows. If that is the limiter, the second form is an
// eight-wave block sharing an LDS activation tile. If fragment-order weight loads/unpack dominate,
// a row-major layout hook is the next lever.
//
// Structure follows the lane-owned gfx1100 prefill form described from hipfire
// gemm_mq4g256v2_residual_wmma_gfx11_bt.hip (Apache-2.0, Kaden Schutt), adapted to Radiance MXFP4.
#pragma once
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <cstdlib>

static inline bool r3_bt_on() {
    static const bool on=[] { const char* e=std::getenv("R3_BT"); return e && std::atoi(e)!=0; }();
    return on;
}
static inline int r3_bt_batch() {
    static const int v=[] { const char* e=std::getenv("R3_BT_B"); const int x=e?std::atoi(e):8;
                            return x==4||x==6||x==8?x:8; }();
    return v;
}
static inline bool r3_bt_nt() {
    static const bool on=[] { const char* e=std::getenv("R3_BT_NT"); return !e||std::atoi(e)!=0; }();
    return on;
}

__global__ __launch_bounds__(256) void r3_mxfp4_bt_prepare_kernel(
        const unsigned char* __restrict__ a, unsigned short* __restrict__ ah, long long n4) {
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n4) return;
    unsigned h0,h1;
    r3_e4m3x4_to_f16(reinterpret_cast<const unsigned*>(a)[i],h0,h1);
    unsigned* dst=reinterpret_cast<unsigned*>(ah)+i*2;
    dst[0]=h0; dst[1]=h1;
}
static inline void r3_mxfp4_bt_prepare(long a,long ah,int M,int K,long stream) {
    const long long n4=(long long)M*K/4;
    hipLaunchKernelGGL(r3_mxfp4_bt_prepare_kernel,dim3((unsigned)((n4+255)/256)),dim3(256),0,
                       (hipStream_t)stream,(const unsigned char*)a,(unsigned short*)ah,n4);
}

__device__ __forceinline__ r3_v16h r3_mxfp4_bt_weight16(unsigned c0,unsigned c1,unsigned t0,unsigned t1) {
    r3_u8 out; unsigned lo,hi,a,b;
    r4d_mxfp4_unpack8_t(c0,t0,t1,&lo,&hi);
    r3_e4m3x4_to_f16(lo,a,b); out[0]=a; out[1]=b;
    r3_e4m3x4_to_f16(hi,a,b); out[2]=a; out[3]=b;
    r4d_mxfp4_unpack8_t(c1,t0,t1,&lo,&hi);
    r3_e4m3x4_to_f16(lo,a,b); out[4]=a; out[5]=b;
    r3_e4m3x4_to_f16(hi,a,b); out[6]=a; out[7]=b;
    return __builtin_bit_cast(r3_v16h,out);
}

template<int BT,bool NT,bool ABLK>
__global__ __launch_bounds__(32,2) void r3_gemm_mxfp4a8_bt_gfx11_kernel(
        const unsigned short* __restrict__ A,const float* __restrict__ As,
        const unsigned int* __restrict__ W,const unsigned char* __restrict__ Ws,
        const unsigned char* __restrict__ Wref,__hip_bfloat16* __restrict__ C,
        int M,int K,int N,int as_rs) {
    const int lane=threadIdx.x&31,col=lane&15,half=lane>>4;
    const int n0=(int)blockIdx.x*16,nrow=n0+col,m0=(int)blockIdx.y*(16*BT),ksteps=K/16;
    const unsigned int* __restrict__ wp=W+(size_t)(n0>>4)*ksteps*32;
    const int ref=(int)Wref[nrow];
    const unsigned short* ar[BT]; const float* sr[BT]; float carried[BT]; r3_v8f acc[BT];
#pragma unroll
    for(int b=0;b<BT;++b){
        const int m=m0+b*16+col,mc=m<M?m:M-1;
        ar[b]=A+(size_t)mc*K; sr[b]=As+(size_t)mc*as_rs;
        carried[b]=ABLK?(sr[b][0]>0.f?sr[b][0]:1.f):1.f;
        acc[b]=r3_v8f{0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    }
    for(int k0=0;k0<K;k0+=32){
        const int d=r4d_mxfp4_fold_d(ref,(int)Ws[(size_t)(k0>>5)*N+nrow]);
        const unsigned t0=kMxfp4Mag[d][0],t1=kMxfp4Mag[d][1];
#pragma unroll
        for(int s=0;s<2;++s){
            const unsigned int* __restrict__ slot=wp+(size_t)((k0>>4)+s)*32;
            const unsigned c0=NT?__builtin_nontemporal_load(slot+col):slot[col];
            const unsigned c1=NT?__builtin_nontemporal_load(slot+col+16):slot[col+16];
            const r3_v16h wf=r3_mxfp4_bt_weight16(c0,c1,t0,t1);
#pragma unroll
            for(int b=0;b<BT;++b){
                const r3_u8 av=*reinterpret_cast<const r3_u8*>(ar[b]+k0+s*16);
                acc[b]=r3_wmma_f16_native(wf,__builtin_bit_cast(r3_v16h,av),acc[b]);
            }
        }
        if constexpr(ABLK){
            const int next_k=k0+32;
            if(next_k<K&&(next_k&127)==0){
                const int kb=next_k>>7;
#pragma unroll
                for(int b=0;b<BT;++b){
                    const float next=sr[b][kb];
                    if(next>0.f){ acc[b]*=carried[b]/next; carried[b]=next; }
                }
            }
        }
    }
#pragma unroll
    for(int b=0;b<BT;++b){
        const int m=m0+b*16+col;
        if(m>=M) continue;
        const float sc=ABLK?carried[b]:sr[b][0];
#pragma unroll
        for(int e=0;e<8;++e){
            const int n=n0+2*e+half;
            const float rf=__int_as_float((int)Wref[n]<<23);
            C[(size_t)m*N+n]=(__hip_bfloat16)(acc[b][e]*r3_fp8_f16_unscale*rf*sc);
        }
    }
}

static inline void r3_mxfp4_bt_launch(long ah,long ascale,long wq,long ws,long wref,long c,
                                      int M,int K,int N,int as_group,int as_rs,long stream) {
    const int bt=r3_bt_batch(); const bool nt=r3_bt_nt(),ablk=as_group==128;
    const dim3 grid((unsigned)(N/16),(unsigned)((M+16*bt-1)/(16*bt)),1),block(32);
    hipStream_t st=(hipStream_t)stream;
    auto Ap=(const unsigned short*)ah; auto Sp=(const float*)ascale; auto Wp=(const unsigned int*)wq;
    auto Zp=(const unsigned char*)ws; auto Rp=(const unsigned char*)wref; auto Cp=(__hip_bfloat16*)c;
#define R3_BT_L(B_,NT_,AB_) hipLaunchKernelGGL((r3_gemm_mxfp4a8_bt_gfx11_kernel<B_,NT_,AB_>),grid,block,0,st,Ap,Sp,Wp,Zp,Rp,Cp,M,K,N,as_rs)
#define R3_BT_AB(B_,NT_) do{if(ablk)R3_BT_L(B_,NT_,true);else R3_BT_L(B_,NT_,false);}while(0)
#define R3_BT_NT(B_) do{if(nt)R3_BT_AB(B_,true);else R3_BT_AB(B_,false);}while(0)
    if(bt==4)R3_BT_NT(4);else if(bt==6)R3_BT_NT(6);else R3_BT_NT(8);
#undef R3_BT_NT
#undef R3_BT_AB
#undef R3_BT_L
}
