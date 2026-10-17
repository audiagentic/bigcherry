// r3_mxfp4_bt.h - gfx1100 MXFP4 x fp8 PREFILL GEMMs used by RR08.
//
// R3_BT=1 is the measured one-wave lane-owned form. It remains unchanged as the comparison arm.
// R3_BT=2 is the second form: one 256-thread block has eight wave32 N tiles. The block covers
// (16*BT) activation rows x 128 weight rows. For each 128-wide K slab the block cooperatively
// stages the widened f16 activation tile in LDS, in [kstep][row][16] order, and all eight waves
// reuse it. Each wave keeps lane-owned f16 weights and BT native-layout f32 accumulators.
//
// At BT=8 a slab is 128 x 128 f16 = 32 KiB of LDS. Relative to the established tiled kernel's
// M=512 TN=2/LBK=128 block (256 M x 64 N), both blocks issue 512 16x16x16 WMMAs per K slab.
// R3_BT=2 cuts the activation loads for the same output area from 128 KiB to 32 KiB and removes
// the tiled form's 64 KiB of weight-LDS reads plus the final gfx11->gfx12 accumulator shuffle.
// It doubles packed-weight traffic/expansion per output area (128 N rows rather than 64). Thus
// it is an occupancy/load-instruction experiment, not a claim that the arithmetic can become 2x.
//
// R3_BT_ABL is a timing-only bit mask; any non-zero value makes results intentionally garbage:
//   1 weight expansion -> constants; 2 activation widening -> constants;
//   4 WMMA products -> skipped;      8 epilogue/store -> skipped.
// The mask is runtime-only and defaults to zero.
//
// Both native forms use f16 operands. An E4M3 code widened by placing its bits in a half is exactly
// 2^-8 of its E4M3 value, including subnormals; the epilogue therefore applies 2^16 once.
// Per-(row,128) activation scales carry each accumulator in the current scale's units.
//
// Structure of R3_BT=2 follows the LDS-shared multi-wave form in hipfire
// gemm_hfq4g256_residual_wmma_gfx1100_muse_lds_g256.hip (Apache-2.0, Kaden Schutt), adapted to
// Radiance's MXFP4 per-32 exponent and fragment-order weight layout.
#pragma once
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <cstdlib>

enum {
    R3_BT_ABL_WEIGHT   = 1,
    R3_BT_ABL_ACT      = 2,
    R3_BT_ABL_WMMA     = 4,
    R3_BT_ABL_EPILOGUE = 8,
};

static inline int r3_bt_mode() {
    static const int v=[] {
        const char* e=std::getenv("R3_BT");
        const int x=e?std::atoi(e):0;
        return x==1||x==2?x:0;
    }();
    return v;
}
static inline bool r3_bt_on() { return r3_bt_mode()!=0; }
static inline int r3_bt_batch() {
    static const int v=[] { const char* e=std::getenv("R3_BT_B"); const int x=e?std::atoi(e):8;
                            return x==4||x==6||x==8?x:8; }();
    return v;
}
static inline bool r3_bt_nt() {
    static const bool on=[] { const char* e=std::getenv("R3_BT_NT"); return !e||std::atoi(e)!=0; }();
    return on;
}
static inline int r3_bt_abl() {
    static const int v=[] { const char* e=std::getenv("R3_BT_ABL"); return e?std::atoi(e):0; }();
    return v & 15;
}

__device__ __forceinline__ r3_v16h r3_bt_const_f16() {
    const r3_u8 v={0x3c003c00u,0x3c003c00u,0x3c003c00u,0x3c003c00u,
                   0x3c003c00u,0x3c003c00u,0x3c003c00u,0x3c003c00u};
    return __builtin_bit_cast(r3_v16h,v);
}
__device__ __forceinline__ void r3_bt_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup","local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup","local");
}

__global__ __launch_bounds__(256) void r3_mxfp4_bt_prepare_kernel(
        const unsigned char* __restrict__ a,unsigned short* __restrict__ ah,long long n4,int K,int lda,int abl) {
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n4) return;
    const long long row=i/(K/4),c4=i%(K/4);  // a row of the activation is `lda` bytes apart, the packed copy K
    unsigned* dst=reinterpret_cast<unsigned*>(ah)+i*2;
    if(abl&R3_BT_ABL_ACT) {
        dst[0]=0x3c003c00u; dst[1]=0x3c003c00u;
    } else {
        unsigned h0,h1;
        r3_e4m3x4_to_f16(reinterpret_cast<const unsigned*>(a+(size_t)row*lda)[c4],h0,h1);
        dst[0]=h0; dst[1]=h1;
    }
}
static inline void r3_mxfp4_bt_prepare(long a,long ah,int M,int K,int lda,long stream) {
    const long long n4=(long long)M*K/4;
    hipLaunchKernelGGL(r3_mxfp4_bt_prepare_kernel,dim3((unsigned)((n4+255)/256)),dim3(256),0,
                       (hipStream_t)stream,(const unsigned char*)a,(unsigned short*)ah,n4,K,lda,r3_bt_abl());
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

// R3_BT=1: original RR08 direct one-wave form.
template<int BT,bool NT,bool ABLK>
__global__ __launch_bounds__(32,2) void r3_gemm_mxfp4a8_bt_gfx11_kernel(
        const unsigned short* __restrict__ A,const float* __restrict__ As,
        const unsigned int* __restrict__ W,const unsigned char* __restrict__ Ws,
        const unsigned char* __restrict__ Wref,__hip_bfloat16* __restrict__ C,
        int M,int K,int N,int as_rs,int ldws,int ldc,int abl) {
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
        const int d=r4d_mxfp4_fold_d(ref,(int)Ws[(size_t)(k0>>5)*ldws+nrow]);
#pragma unroll
        for(int s=0;s<2;++s){
            const unsigned int* __restrict__ slot=wp+(size_t)((k0>>4)+s)*32;
            const unsigned c0=NT?__builtin_nontemporal_load(slot+col):slot[col];
            const unsigned c1=NT?__builtin_nontemporal_load(slot+col+16):slot[col+16];
            r3_v16h wf;
            if(abl&R3_BT_ABL_WEIGHT) wf=r3_bt_const_f16();
            else {
                const unsigned t0=kMxfp4Mag[d][0],t1=kMxfp4Mag[d][1];
                wf=r3_mxfp4_bt_weight16(c0,c1,t0,t1);
            }
#pragma unroll
            for(int b=0;b<BT;++b){
                const r3_u8 av=*reinterpret_cast<const r3_u8*>(ar[b]+k0+s*16);
                if(!(abl&R3_BT_ABL_WMMA))
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
    if(abl&R3_BT_ABL_EPILOGUE) return;
#pragma unroll
    for(int b=0;b<BT;++b){
        const int m=m0+b*16+col;
        if(m>=M) continue;
        const float sc=ABLK?carried[b]:sr[b][0];
#pragma unroll
        for(int e=0;e<8;++e){
            const int n=n0+2*e+half;
            const float rf=__int_as_float((int)Wref[n]<<23);
            C[(size_t)m*ldc+n]=(__hip_bfloat16)(acc[b][e]*r3_fp8_f16_unscale*rf*sc);
        }
    }
}

// R3_BT=2: 8 N-waves share a tile-major LDS activation slab.
// BT=8 => output tile 128 M x 128 N; a K=128 slab is 32 KiB LDS.
template<int BT,bool NT,bool ABLK>
__global__ __launch_bounds__(256,1) void r3_gemm_mxfp4a8_bt2_gfx11_kernel(
        const unsigned short* __restrict__ A,const float* __restrict__ As,
        const unsigned int* __restrict__ W,const unsigned char* __restrict__ Ws,
        const unsigned char* __restrict__ Wref,__hip_bfloat16* __restrict__ C,
        int M,int K,int N,int as_rs,int ldws,int ldc,int abl) {
    constexpr int MR=BT*16,KS=128,MAX_STEPS=KS/16;
    __shared__ unsigned short sA[MR*KS];

    const int tid=threadIdx.x,lane=tid&31,wave=tid>>5,col=lane&15,half=lane>>4;
    const int m0=(int)blockIdx.y*MR;
    const int nb=(int)blockIdx.x*128+wave*16;
    const bool nalive=nb<N;
    const int nc=nalive?nb:N-16,nrow=nc+col,ksteps=K/16;
    const unsigned int* __restrict__ wp=W+(size_t)(nc>>4)*ksteps*32;
    const int ref=(int)Wref[nrow];

    const float* sr[BT]; float carried[BT]; r3_v8f acc[BT];
#pragma unroll
    for(int b=0;b<BT;++b){
        const int m=m0+b*16+col,mc=m<M?m:M-1;
        sr[b]=As+(size_t)mc*as_rs;
        carried[b]=ABLK?(sr[b][0]>0.f?sr[b][0]:1.f):1.f;
        acc[b]=r3_v8f{0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    }

    for(int kb0=0;kb0<K;kb0+=KS){
        const int steps=((K-kb0)<KS?(K-kb0):KS)/16;
        const int vecs=MR*steps;
        for(int v=tid;v<vecs;v+=256){
            const int st=v/MR,r=v-st*MR;
            const int gm=(m0+r)<M?m0+r:M-1;
            const r3_v16h av=*reinterpret_cast<const r3_v16h*>(A+(size_t)gm*K+kb0+st*16);
            *reinterpret_cast<r3_v16h*>(&sA[(st*MR+r)*16])=av;
        }
        r3_bt_lds_barrier();

        for(int st=0;st<MAX_STEPS;++st){
            if(st>=steps) break;
            const int k=kb0+st*16;
            const int d=r4d_mxfp4_fold_d(ref,(int)Ws[(size_t)(k>>5)*ldws+nrow]);
            const unsigned int* __restrict__ slot=wp+(size_t)(k>>4)*32;
            const unsigned c0=NT?__builtin_nontemporal_load(slot+col):slot[col];
            const unsigned c1=NT?__builtin_nontemporal_load(slot+col+16):slot[col+16];
            r3_v16h wf;
            if(abl&R3_BT_ABL_WEIGHT) wf=r3_bt_const_f16();
            else {
                const unsigned t0=kMxfp4Mag[d][0],t1=kMxfp4Mag[d][1];
                wf=r3_mxfp4_bt_weight16(c0,c1,t0,t1);
            }
#pragma unroll
            for(int b=0;b<BT;++b){
                const r3_u8 av=*reinterpret_cast<const r3_u8*>(&sA[(st*MR+b*16+col)*16]);
                if(!(abl&R3_BT_ABL_WMMA))
                    acc[b]=r3_wmma_f16_native(wf,__builtin_bit_cast(r3_v16h,av),acc[b]);
            }
            if constexpr(ABLK){
                const int next_k=k+16;
                if(next_k<K&&(next_k&127)==0){
                    const int skb=next_k>>7;
#pragma unroll
                    for(int b=0;b<BT;++b){
                        const float next=sr[b][skb];
                        if(next>0.f){ acc[b]*=carried[b]/next; carried[b]=next; }
                    }
                }
            }
        }
        r3_bt_lds_barrier();
    }

    if(abl&R3_BT_ABL_EPILOGUE) return;
    if(!nalive) return;
#pragma unroll
    for(int b=0;b<BT;++b){
        const int m=m0+b*16+col;
        if(m>=M) continue;
        const float sc=ABLK?carried[b]:sr[b][0];
#pragma unroll
        for(int e=0;e<8;++e){
            const int n=nb+2*e+half;
            if(n>=N) continue;
            const float rf=__int_as_float((int)Wref[n]<<23);
            C[(size_t)m*ldc+n]=(__hip_bfloat16)(acc[b][e]*r3_fp8_f16_unscale*rf*sc);
        }
    }
}

static inline void r3_mxfp4_bt_launch(long ah,long ascale,long wq,long ws,long wref,long c,
                                      int M,int K,int N,int as_group,int as_rs,long stream,int ldws,int ldc) {
    const int bt=r3_bt_batch(),abl=r3_bt_abl(); const bool nt=r3_bt_nt(),ablk=as_group==128;
    const dim3 grid((unsigned)(N/16),(unsigned)((M+16*bt-1)/(16*bt)),1),block(32);
    hipStream_t st=(hipStream_t)stream;
    auto Ap=(const unsigned short*)ah; auto Sp=(const float*)ascale; auto Wp=(const unsigned int*)wq;
    auto Zp=(const unsigned char*)ws; auto Rp=(const unsigned char*)wref; auto Cp=(__hip_bfloat16*)c;
#define R3_BT_L(B_,NT_,AB_) hipLaunchKernelGGL((r3_gemm_mxfp4a8_bt_gfx11_kernel<B_,NT_,AB_>),grid,block,0,st,Ap,Sp,Wp,Zp,Rp,Cp,M,K,N,as_rs,ldws,ldc,abl)
#define R3_BT_AB(B_,NT_) do{if(ablk)R3_BT_L(B_,NT_,true);else R3_BT_L(B_,NT_,false);}while(0)
#define R3_BT_NT(B_) do{if(nt)R3_BT_AB(B_,true);else R3_BT_AB(B_,false);}while(0)
    if(bt==4)R3_BT_NT(4);else if(bt==6)R3_BT_NT(6);else R3_BT_NT(8);
#undef R3_BT_NT
#undef R3_BT_AB
#undef R3_BT_L
}

static inline void r3_mxfp4_bt2_launch(long ah,long ascale,long wq,long ws,long wref,long c,
                                       int M,int K,int N,int as_group,int as_rs,long stream,int ldws,int ldc) {
    const int bt=r3_bt_batch(),abl=r3_bt_abl(); const bool nt=r3_bt_nt(),ablk=as_group==128;
    const dim3 grid((unsigned)((N+127)/128),(unsigned)((M+16*bt-1)/(16*bt)),1),block(256);
    hipStream_t st=(hipStream_t)stream;
    auto Ap=(const unsigned short*)ah; auto Sp=(const float*)ascale; auto Wp=(const unsigned int*)wq;
    auto Zp=(const unsigned char*)ws; auto Rp=(const unsigned char*)wref; auto Cp=(__hip_bfloat16*)c;
#define R3_BT2_L(B_,NT_,AB_) hipLaunchKernelGGL((r3_gemm_mxfp4a8_bt2_gfx11_kernel<B_,NT_,AB_>),grid,block,0,st,Ap,Sp,Wp,Zp,Rp,Cp,M,K,N,as_rs,ldws,ldc,abl)
#define R3_BT2_AB(B_,NT_) do{if(ablk)R3_BT2_L(B_,NT_,true);else R3_BT2_L(B_,NT_,false);}while(0)
#define R3_BT2_NT(B_) do{if(nt)R3_BT2_AB(B_,true);else R3_BT2_AB(B_,false);}while(0)
    if(bt==4)R3_BT2_NT(4);else if(bt==6)R3_BT2_NT(6);else R3_BT2_NT(8);
#undef R3_BT2_NT
#undef R3_BT2_AB
#undef R3_BT2_L
}
