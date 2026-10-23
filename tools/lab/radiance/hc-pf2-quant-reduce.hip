// RAD15: isolated test of the cross-thread amax reduction hc_down_pf2_kernel's real FP8 down-product would need.
// Does NOT touch r4d_hc_bf16.hip -- replicates its exact thread/slice layout (kThreads=256, kKV=8, RT=64 rows,
// kTKT=64-wide slices) against the real ground-truth N already dumped (ndump.bin, RAD15), to prove the
// per-128-of-K amax-then-quantise bookkeeping is correct before ever writing it into the production kernel.
//
// Layout (from r4d_hc_bf16.hip's hc_down_pf2_kernel): kKV = kTKT/8 = 8 threads share one row's 64-wide slice,
// each owning a fixed 8-element window (k8a = (tid%kKV)*8); ar = tid/kKV selects which of the RT=64 rows. Two
// consecutive 64-wide slices (k0, k0+64) cover one 128-wide scale group; the 8 threads at a given ar each see
// only 16 of those 128 elements (8 from each slice) -- getting the group's amax needs a reduction across those
// 8 threads, done here via LDS (simplest to get right; the real kernel can tighten to shuffles once correct).
//
// Build: /opt/rocm-7.2.4/bin/hipcc --offload-arch=gfx1201 -O2 -I/opt/rocm-7.2.4/include hc-pf2-quant-reduce.hip \
//          -o hc-pf2-quant-reduce
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>

using rocwmma::float8_t;

constexpr int kThreads = 256, kKV = 8, kTKT = 64, KB = 128;
constexpr int kRS = kThreads / kKV;   // 32: rows between a thread's A rows (matches r4d_hc_bf16.hip)
constexpr int RT = 64, kAV = RT / kRS;  // kAV=2 rows a thread owns (RT=64, kRS=32)

// One workgroup = one row-tile (RT=64 rows), looping K in kTKT=64 slices, same shape as hc_down_pf2_kernel.
// Input: ground-truth N for ONE row (ndump dumped row 0 only), broadcast to all RT rows here since we only
// have one real row -- this test is about the BOOKKEEPING, not about having 64 distinct ground-truth rows.
__global__ void quant_reduce(const float* __restrict__ n_row, uint8_t* __restrict__ codes_out,
                             float* __restrict__ scale_out, int K) {
    __shared__ float red[kThreads];
    const int tid = threadIdx.x;
    const int ar = tid / kKV, k8a = (tid % kKV) * 8;

    for (int k0 = 0; k0 < K; k0 += 2 * kTKT) {  // one iteration per 128-wide group (2 slices)
        float local_max = 0.0f;
        float vals[16];  // this thread's 8 elements from slice 0, then 8 from slice 1
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            vals[e] = n_row[k0 + k8a + e];
            local_max = std::max(local_max, std::fabs(vals[e]));
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            vals[8 + e] = n_row[k0 + kTKT + k8a + e];
            local_max = std::max(local_max, std::fabs(vals[8 + e]));
        }
        red[tid] = local_max;
        __syncthreads();
        // Reduce over the kKV=8 threads sharing this ar (tree reduction within the 8-thread group in LDS).
        float group_max = 0.0f;
        const int base = ar * kKV;
        for (int i = 0; i < kKV; ++i) group_max = std::max(group_max, red[base + i]);
        __syncthreads();  // before red[] is reused next iteration

        const float scale = group_max > 0.0f ? group_max / 448.0f : 1.0f;
        if (k8a == 0 && ar == 0) scale_out[k0 / (2 * kTKT)] = scale;  // one scale per group, written once
#pragma unroll
        for (int e = 0; e < 16; ++e) {
            const int k = (e < 8) ? k0 + k8a + e : k0 + kTKT + k8a + (e - 8);
            const float8_t c = float8_t(vals[e] / scale);
            codes_out[k] = *reinterpret_cast<const uint8_t*>(&c);
        }
    }
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "ndump.bin";
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return 1; }
    const int K = 10240, NKB = K / KB;
    std::vector<float> host_n(K);
    size_t got = fread(host_n.data(), sizeof(float), K, f);
    fclose(f);
    if (got != (size_t)K) { printf("short read\n"); return 1; }

    float* d_n; uint8_t* d_codes; float* d_scale;
    hipMalloc(&d_n, K * sizeof(float));
    hipMalloc(&d_codes, K * sizeof(uint8_t));
    hipMalloc(&d_scale, NKB * sizeof(float));
    hipMemcpy(d_n, host_n.data(), K * sizeof(float), hipMemcpyHostToDevice);

    quant_reduce<<<1, kThreads>>>(d_n, d_codes, d_scale, K);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("FAIL launch: %s\n", hipGetErrorString(e)); return 1; }

    std::vector<uint8_t> codes(K);
    std::vector<float> scale(NKB);
    hipMemcpy(codes.data(), d_codes, K, hipMemcpyDeviceToHost);
    hipMemcpy(scale.data(), d_scale, NKB * sizeof(float), hipMemcpyDeviceToHost);

    // Offline ground truth (same method as ndump-quant-check.hip): recompute amax/scale/codes directly from
    // host_n, per 128-group, and compare bit-for-bit against the kernel's cross-thread-reduced result.
    int mismatches = 0;
    double sumsq_err = 0.0, sumsq_ref = 0.0;
    for (int g = 0; g < NKB; ++g) {
        float amax = 0.0f;
        for (int k = 0; k < KB; ++k) amax = std::max(amax, std::fabs(host_n[g * KB + k]));
        const float ref_scale = amax > 0.0f ? amax / 448.0f : 1.0f;
        if (std::fabs(ref_scale - scale[g]) > 1e-9f * std::max(1.0f, ref_scale)) {
            printf("group %d: scale mismatch kernel=%.9g ref=%.9g\n", g, scale[g], ref_scale);
            mismatches++;
        }
        for (int k = 0; k < KB; ++k) {
            const float8_t ref_c = float8_t(host_n[g * KB + k] / ref_scale);
            const uint8_t ref_bits = *reinterpret_cast<const uint8_t*>(&ref_c);
            if (ref_bits != codes[g * KB + k]) {
                if (mismatches < 5)
                    printf("group %d k %d: code mismatch kernel=0x%02x ref=0x%02x\n", g, k, codes[g * KB + k], ref_bits);
                mismatches++;
            }
            const float back_k = float(*reinterpret_cast<const float8_t*>(&codes[g * KB + k])) * scale[g];
            const double d = back_k - host_n[g * KB + k];
            sumsq_err += d * d;
            sumsq_ref += (double)host_n[g * KB + k] * host_n[g * KB + k];
        }
    }
    printf("cross-thread amax reduction vs offline ground truth: %d mismatches of %d scales + %d codes\n",
           mismatches, NKB, K);
    printf("RMS relative error (kernel's own reconstruction): %.4f%%\n",
           100.0 * std::sqrt(sumsq_err / K) / std::sqrt(sumsq_ref / K));
    printf(mismatches == 0 ? "PASS: the cross-thread reduction is bit-for-bit correct.\n"
                           : "FAIL: the bookkeeping has a real bug -- do not port this into production.\n");
    return mismatches == 0 ? 0 : 1;
}
