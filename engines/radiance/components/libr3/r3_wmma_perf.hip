// Native gfx11 WMMA arithmetic roof for libr3's rewritten r4d_selftest.
//
// Register rows have no memory traffic in the timed loop. 1/2/4/8 independent
// accumulator chains expose WMMA issue latency; grid and workgroup geometry are
// swept independently. The best register geometry is rerun with one LDS read
// and one global read per operand fragment. Every accumulator is folded into a
// checksum after the loop so the compiler cannot remove the matrix operations.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

typedef unsigned r3w_u2 __attribute__((ext_vector_type(2)));
typedef unsigned r3w_u4 __attribute__((ext_vector_type(4)));
typedef unsigned r3w_u8 __attribute__((ext_vector_type(8)));
typedef int r3w_i2 __attribute__((ext_vector_type(2)));
typedef int r3w_i4 __attribute__((ext_vector_type(4)));
typedef int r3w_i8 __attribute__((ext_vector_type(8)));
typedef float r3w_f8 __attribute__((ext_vector_type(8)));
typedef _Float16 r3w_h16 __attribute__((ext_vector_type(16)));
typedef __bf16 r3w_b16 __attribute__((ext_vector_type(16)));

enum { R3W_REG = 0, R3W_LDS = 1, R3W_GLOBAL = 2 };
enum { R3W_F16 = 0, R3W_BF16 = 1, R3W_IU8 = 2, R3W_IU4 = 3 };
enum { R3W_WAVE = 32, R3W_DWORDS_PER_LANE = 16 };

__device__ __forceinline__ r3w_u8 r3w_ld8(const volatile unsigned* p) {
    r3w_u8 v;
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = p[i];
    return v;
}
__device__ __forceinline__ r3w_u4 r3w_ld4(const volatile unsigned* p) {
    r3w_u4 v;
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = p[i];
    return v;
}
__device__ __forceinline__ r3w_u2 r3w_ld2(const volatile unsigned* p) {
    r3w_u2 v;
    v[0] = p[0]; v[1] = p[1];
    return v;
}

__device__ __forceinline__ const volatile unsigned* r3w_gp(
        const unsigned* in, unsigned slot_mask, int rep, int chain) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned wave = threadIdx.x >> 5;
    const unsigned gw = blockIdx.x * (blockDim.x >> 5) + wave;
    const unsigned slot = (gw * 4099u + (unsigned)rep * 131u + (unsigned)chain * 17u) & slot_mask;
    return (const volatile unsigned*)in
         + (size_t)slot * R3W_WAVE * R3W_DWORDS_PER_LANE
         + (size_t)lane * R3W_DWORDS_PER_LANE;
}

template <int N>
__device__ __forceinline__ unsigned long long r3w_hash_f(const r3w_f8 (&a)[N]) {
    unsigned long long h = 1469598103934665603ull ^ (unsigned)(blockIdx.x * blockDim.x + threadIdx.x);
#pragma unroll
    for (int n = 0; n < N; ++n)
#pragma unroll
        for (int i = 0; i < 8; ++i)
            h = (h ^ __builtin_bit_cast(unsigned, a[n][i])) * 1099511628211ull;
    return h;
}
template <int N>
__device__ __forceinline__ unsigned long long r3w_hash_i(const r3w_i8 (&a)[N]) {
    unsigned long long h = 1469598103934665603ull ^ (unsigned)(blockIdx.x * blockDim.x + threadIdx.x);
#pragma unroll
    for (int n = 0; n < N; ++n)
#pragma unroll
        for (int i = 0; i < 8; ++i)
            h = (h ^ (unsigned)a[n][i]) * 1099511628211ull;
    return h;
}

template <int MODE, int ACCS>
__global__ __launch_bounds__(256) void r3w_f16(
        const unsigned* in, unsigned mask, int iters, unsigned long long* out) {
    extern __shared__ unsigned sm[];
    const unsigned seed = (unsigned)(blockIdx.x * blockDim.x + threadIdx.x);
    r3w_u8 ar, br;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        ar[i] = 0x3c003c00u ^ (((seed + (unsigned)i) & 3u) * 0x00010001u);
        br[i] = 0x38003800u ^ (((seed + (unsigned)i + 1u) & 3u) * 0x00010001u);
    }
    if constexpr (MODE == R3W_LDS) {
        volatile unsigned* p = sm + (size_t)threadIdx.x * R3W_DWORDS_PER_LANE;
#pragma unroll
        for (int i = 0; i < 8; ++i) { p[i] = ar[i]; p[8 + i] = br[i]; }
        __syncthreads();
    }
    r3w_f8 acc[ACCS];
#pragma unroll
    for (int n = 0; n < ACCS; ++n)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[n][i] = (float)(n + 1) * 0.0001f;
    for (int rep = 0; rep < iters; ++rep) {
#pragma unroll
        for (int n = 0; n < ACCS; ++n) {
            r3w_u8 a, b;
            if constexpr (MODE == R3W_REG) { a = ar; b = br; }
            else if constexpr (MODE == R3W_LDS) {
                const volatile unsigned* p = sm + (size_t)threadIdx.x * R3W_DWORDS_PER_LANE;
                a = r3w_ld8(p); b = r3w_ld8(p + 8);
            } else {
                const volatile unsigned* p = r3w_gp(in, mask, rep, n);
                a = r3w_ld8(p); b = r3w_ld8(p + 8);
            }
            acc[n] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                __builtin_bit_cast(r3w_h16, a), __builtin_bit_cast(r3w_h16, b), acc[n]);
        }
    }
    out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = r3w_hash_f(acc);
}

template <int MODE, int ACCS>
__global__ __launch_bounds__(256) void r3w_bf16(
        const unsigned* in, unsigned mask, int iters, unsigned long long* out) {
    extern __shared__ unsigned sm[];
    const unsigned seed = (unsigned)(blockIdx.x * blockDim.x + threadIdx.x);
    r3w_u8 ar, br;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        ar[i] = 0x3f803f80u ^ (((seed + (unsigned)i) & 3u) * 0x00010001u);
        br[i] = 0x3f003f00u ^ (((seed + (unsigned)i + 1u) & 3u) * 0x00010001u);
    }
    if constexpr (MODE == R3W_LDS) {
        volatile unsigned* p = sm + (size_t)threadIdx.x * R3W_DWORDS_PER_LANE;
#pragma unroll
        for (int i = 0; i < 8; ++i) { p[i] = ar[i]; p[8 + i] = br[i]; }
        __syncthreads();
    }
    r3w_f8 acc[ACCS];
#pragma unroll
    for (int n = 0; n < ACCS; ++n)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[n][i] = (float)(n + 1) * 0.0001f;
    for (int rep = 0; rep < iters; ++rep) {
#pragma unroll
        for (int n = 0; n < ACCS; ++n) {
            r3w_u8 a, b;
            if constexpr (MODE == R3W_REG) { a = ar; b = br; }
            else if constexpr (MODE == R3W_LDS) {
                const volatile unsigned* p = sm + (size_t)threadIdx.x * R3W_DWORDS_PER_LANE;
                a = r3w_ld8(p); b = r3w_ld8(p + 8);
            } else {
                const volatile unsigned* p = r3w_gp(in, mask, rep, n);
                a = r3w_ld8(p); b = r3w_ld8(p + 8);
            }
            acc[n] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                __builtin_bit_cast(r3w_b16, a), __builtin_bit_cast(r3w_b16, b), acc[n]);
        }
    }
    out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = r3w_hash_f(acc);
}

template <int MODE, int ACCS>
__global__ __launch_bounds__(256) void r3w_iu8(
        const unsigned* in, unsigned mask, int iters, unsigned long long* out) {
    extern __shared__ unsigned sm[];
    const unsigned seed = (unsigned)(blockIdx.x * blockDim.x + threadIdx.x);
    r3w_u4 ar, br;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        ar[i] = 0x01010101u * (1u + ((seed + (unsigned)i) & 3u));
        br[i] = 0x01010101u * (1u + ((seed + (unsigned)i + 1u) & 3u));
    }
    if constexpr (MODE == R3W_LDS) {
        volatile unsigned* p = sm + (size_t)threadIdx.x * R3W_DWORDS_PER_LANE;
#pragma unroll
        for (int i = 0; i < 4; ++i) { p[i] = ar[i]; p[4 + i] = br[i]; }
        __syncthreads();
    }
    r3w_i8 acc[ACCS];
#pragma unroll
    for (int n = 0; n < ACCS; ++n)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[n][i] = n + i;
    for (int rep = 0; rep < iters; ++rep) {
#pragma unroll
        for (int n = 0; n < ACCS; ++n) {
            r3w_u4 a, b;
            if constexpr (MODE == R3W_REG) { a = ar; b = br; }
            else if constexpr (MODE == R3W_LDS) {
                const volatile unsigned* p = sm + (size_t)threadIdx.x * R3W_DWORDS_PER_LANE;
                a = r3w_ld4(p); b = r3w_ld4(p + 4);
            } else {
                const volatile unsigned* p = r3w_gp(in, mask, rep, n);
                a = r3w_ld4(p); b = r3w_ld4(p + 4);
            }
            acc[n] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                false, __builtin_bit_cast(r3w_i4, a), false, __builtin_bit_cast(r3w_i4, b), acc[n], false);
        }
    }
    out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = r3w_hash_i(acc);
}

template <int MODE, int ACCS>
__global__ __launch_bounds__(256) void r3w_iu4(
        const unsigned* in, unsigned mask, int iters, unsigned long long* out) {
    extern __shared__ unsigned sm[];
    const unsigned seed = (unsigned)(blockIdx.x * blockDim.x + threadIdx.x);
    r3w_u2 a0r, b0r, a1r, b1r;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const unsigned x = 0x11111111u * (1u + ((seed + (unsigned)i) & 1u));
        a0r[i] = x; b0r[i] = x ^ 0x11111111u;
        a1r[i] = x ^ 0x22222222u; b1r[i] = x ^ 0x33333333u;
    }
    if constexpr (MODE == R3W_LDS) {
        volatile unsigned* p = sm + (size_t)threadIdx.x * R3W_DWORDS_PER_LANE;
        p[0]=a0r[0]; p[1]=a0r[1]; p[2]=b0r[0]; p[3]=b0r[1];
        p[4]=a1r[0]; p[5]=a1r[1]; p[6]=b1r[0]; p[7]=b1r[1];
        __syncthreads();
    }
    r3w_i8 acc[ACCS];
#pragma unroll
    for (int n = 0; n < ACCS; ++n)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[n][i] = n + i;
    for (int rep = 0; rep < iters; ++rep) {
#pragma unroll
        for (int n = 0; n < ACCS; ++n) {
            r3w_u2 a0, b0, a1, b1;
            if constexpr (MODE == R3W_REG) { a0=a0r; b0=b0r; a1=a1r; b1=b1r; }
            else if constexpr (MODE == R3W_LDS) {
                const volatile unsigned* p = sm + (size_t)threadIdx.x * R3W_DWORDS_PER_LANE;
                a0=r3w_ld2(p); b0=r3w_ld2(p+2); a1=r3w_ld2(p+4); b1=r3w_ld2(p+6);
            } else {
                const volatile unsigned* p = r3w_gp(in, mask, rep, n);
                a0=r3w_ld2(p); b0=r3w_ld2(p+2); a1=r3w_ld2(p+4); b1=r3w_ld2(p+6);
            }
            r3w_i8 v = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                false, __builtin_bit_cast(r3w_i2, a0), false, __builtin_bit_cast(r3w_i2, b0), acc[n], false);
            acc[n] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                false, __builtin_bit_cast(r3w_i2, a1), false, __builtin_bit_cast(r3w_i2, b1), v, false);
        }
    }
    out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = r3w_hash_i(acc);
}

struct R3WResult {
    double us = 0.0, rate = 0.0;
    unsigned long long checksum = 0;
    hipError_t error = hipSuccess;
};

template <int KIND, int MODE, int ACCS>
static R3WResult r3w_run_t(unsigned* in, unsigned mask, unsigned long long* out,
                           int cu, int wpb, int bpc, int iters) {
    R3WResult r;
    const int threads = wpb * R3W_WAVE;
    const int blocks = cu * bpc;
    const size_t shmem = MODE == R3W_LDS ? (size_t)threads * R3W_DWORDS_PER_LANE * 4 : 0;
    auto launch = [&](int n) {
        if constexpr (KIND == R3W_F16)
            hipLaunchKernelGGL((r3w_f16<MODE, ACCS>), dim3(blocks), dim3(threads), shmem, 0, in, mask, n, out);
        else if constexpr (KIND == R3W_BF16)
            hipLaunchKernelGGL((r3w_bf16<MODE, ACCS>), dim3(blocks), dim3(threads), shmem, 0, in, mask, n, out);
        else if constexpr (KIND == R3W_IU8)
            hipLaunchKernelGGL((r3w_iu8<MODE, ACCS>), dim3(blocks), dim3(threads), shmem, 0, in, mask, n, out);
        else
            hipLaunchKernelGGL((r3w_iu4<MODE, ACCS>), dim3(blocks), dim3(threads), shmem, 0, in, mask, n, out);
    };
    launch(std::max(32, iters / 32));
    r.error = hipGetLastError();
    if (r.error == hipSuccess) r.error = hipDeviceSynchronize();
    if (r.error != hipSuccess) return r;

    hipEvent_t e0 = nullptr, e1 = nullptr;
    if ((r.error = hipEventCreate(&e0)) != hipSuccess) return r;
    if ((r.error = hipEventCreate(&e1)) != hipSuccess) { hipEventDestroy(e0); return r; }
    hipEventRecord(e0, 0);
    launch(iters);
    r.error = hipGetLastError();
    hipEventRecord(e1, 0);
    if (r.error == hipSuccess) r.error = hipEventSynchronize(e1);
    float ms = 0.0f;
    if (r.error == hipSuccess) r.error = hipEventElapsedTime(&ms, e0, e1);
    hipEventDestroy(e0); hipEventDestroy(e1);
    if (r.error != hipSuccess || ms <= 0.0f) return r;

    const size_t nth = (size_t)blocks * threads;
    std::vector<unsigned long long> h(nth);
    r.error = hipMemcpy(h.data(), out, nth * sizeof(*out), hipMemcpyDeviceToHost);
    if (r.error != hipSuccess) return r;
    for (unsigned long long v : h) r.checksum ^= v;
    r.us = (double)ms * 1000.0;
    const double ops_per = KIND == R3W_IU4 ? 16384.0 : 8192.0;
    const double ops = (double)blocks * wpb * iters * ACCS * ops_per;
    r.rate = ops / ((double)ms * 1.0e9);
    return r;
}

template <int KIND, int MODE>
static R3WResult r3w_run_acc(int accs, unsigned* in, unsigned mask, unsigned long long* out,
                             int cu, int wpb, int bpc, int iters) {
    switch (accs) {
        case 1: return r3w_run_t<KIND, MODE, 1>(in, mask, out, cu, wpb, bpc, iters);
        case 2: return r3w_run_t<KIND, MODE, 2>(in, mask, out, cu, wpb, bpc, iters);
        case 4: return r3w_run_t<KIND, MODE, 4>(in, mask, out, cu, wpb, bpc, iters);
        default:return r3w_run_t<KIND, MODE, 8>(in, mask, out, cu, wpb, bpc, iters);
    }
}

static R3WResult r3w_run(int kind, int mode, int accs, unsigned* in, unsigned mask,
                         unsigned long long* out, int cu, int wpb, int bpc, int iters) {
#define R3W_CASE(K) do { \
    if (mode == R3W_REG) return r3w_run_acc<K, R3W_REG>(accs,in,mask,out,cu,wpb,bpc,iters); \
    if (mode == R3W_LDS) return r3w_run_acc<K, R3W_LDS>(accs,in,mask,out,cu,wpb,bpc,iters); \
    return r3w_run_acc<K, R3W_GLOBAL>(accs,in,mask,out,cu,wpb,bpc,iters); \
} while (0)
    switch (kind) {
        case R3W_F16:  R3W_CASE(R3W_F16);
        case R3W_BF16: R3W_CASE(R3W_BF16);
        case R3W_IU8:  R3W_CASE(R3W_IU8);
        default:       R3W_CASE(R3W_IU4);
    }
#undef R3W_CASE
}

struct R3WBest { double rate = -1.0; int wpb = 0, bpc = 0, accs = 0; R3WResult r; };
static const char* r3w_kn(int k) { return k==0 ? "f16" : k==1 ? "bf16" : k==2 ? "iu8" : "iu4"; }
static const char* r3w_un(int k) { return k < 2 ? "TFLOP/s" : "TOPS"; }
static const char* r3w_sn(int m) { return m==0 ? "reg" : m==1 ? "lds" : "global"; }

extern "C" int r3_perfwmma(int iters) {
    if (iters < 64) iters = 64;
    int dev = 0;
    hipDeviceProp_t p{};
    hipError_t e = hipGetDevice(&dev);
    if (e == hipSuccess) e = hipGetDeviceProperties(&p, dev);
    if (e != hipSuccess) {
        std::printf("--perfwmma: device query failed: %s\n", hipGetErrorString(e));
        return 1;
    }
    const int cu = p.multiProcessorCount;
    const size_t global_bytes = (size_t)128 << 20;
    const size_t slot_bytes = (size_t)R3W_WAVE * R3W_DWORDS_PER_LANE * sizeof(unsigned);
    const unsigned slots = (unsigned)(global_bytes / slot_bytes);
    if (!slots || (slots & (slots - 1))) {
        std::printf("--perfwmma: bad global ring geometry\n");
        return 1;
    }
    unsigned* in = nullptr;
    unsigned long long* out = nullptr;
    const size_t max_threads = (size_t)cu * 8 * 8 * R3W_WAVE;
    e = hipMalloc((void**)&in, global_bytes);
    if (e == hipSuccess) e = hipMalloc((void**)&out, max_threads * sizeof(*out));
    if (e != hipSuccess) {
        std::printf("--perfwmma: allocation failed: %s\n", hipGetErrorString(e));
        if (in) hipFree(in); if (out) hipFree(out);
        return 1;
    }
    hipMemset(in, 0x3c, global_bytes);

    std::printf("\nWMMA arithmetic roof device=%s CUs=%d iters=%d global_ring=128MiB\n", p.name, cu, iters);
    std::printf("dtype src wpb blocks/CU grid-waves/CU acc us rate unit checksum\n");
    const int wpbs[] = {1,2,4,8}, bpcs[] = {1,2,4,8}, accv[] = {1,2,4,8};
    R3WBest best[4];
    for (int kind = 0; kind < 4; ++kind) {
        for (int wpb : wpbs) for (int bpc : bpcs) for (int accs : accv) {
            R3WResult r = r3w_run(kind, R3W_REG, accs, in, slots - 1, out, cu, wpb, bpc, iters);
            if (r.error != hipSuccess) {
                std::printf("%-4s reg %d %d %d %d SKIP %s\n", r3w_kn(kind), wpb, bpc, wpb*bpc, accs, hipGetErrorString(r.error));
                continue;
            }
            std::printf("%-4s reg %d %d %d %d %.1f %.2f %s %016llx\n",
                        r3w_kn(kind), wpb, bpc, wpb*bpc, accs, r.us, r.rate, r3w_un(kind), r.checksum);
            if (r.rate > best[kind].rate) {
                best[kind].rate=r.rate; best[kind].wpb=wpb; best[kind].bpc=bpc; best[kind].accs=accs; best[kind].r=r;
            }
        }
    }

    std::printf("\nBEST (LDS/global reuse the winning register geometry and ILP)\n");
    for (int kind = 0; kind < 4; ++kind) {
        if (best[kind].rate < 0.0) continue;
        for (int mode : {R3W_REG, R3W_LDS, R3W_GLOBAL}) {
            R3WResult r = mode == R3W_REG ? best[kind].r :
                r3w_run(kind, mode, best[kind].accs, in, slots - 1, out, cu,
                        best[kind].wpb, best[kind].bpc, iters);
            if (r.error != hipSuccess) {
                std::printf("BEST %-4s %-6s SKIP %s\n", r3w_kn(kind), r3w_sn(mode), hipGetErrorString(r.error));
                continue;
            }
            std::printf("BEST %-4s %-6s wpb=%d blocks/CU=%d grid-waves/CU=%d acc=%d %.1f us %.2f %s checksum=%016llx\n",
                        r3w_kn(kind), r3w_sn(mode), best[kind].wpb, best[kind].bpc,
                        best[kind].wpb*best[kind].bpc, best[kind].accs, r.us, r.rate, r3w_un(kind), r.checksum);
        }
    }
    hipFree(out); hipFree(in);
    return 0;
}
