// RAD17: standalone validation of hc_pf2_quant_a8_kernel (GPT dev-agent design, req_bigcherry_757a112f3c07) --
// the FP8 activation quantisation pre-pass. One wave (32 lanes) owns one (row, 128-of-K group): each lane reads
// 4 elements, a full-wave max reduction (matching r4d_common.h's r4d_wave_max -- a plain shuffle butterfly is
// numerically identical for max, which is order-independent, unlike the sum reductions r4d_common.h takes care
// to keep bit-identical) gives the group's amax, then every lane quantises its own 4 elements to E4M3 at
// scale=amax/448. Writes A8[rows][HN] uint8 and Ascale[rows][HN/128] f32 -- the layout the proven WMMA kernel
// already expects (ldm=HN for matrix_a, no change needed there).
//
// Validates against the real ground-truth N (ndump.bin) and the already-proven offline recomputation
// (ndump-quant-check.hip's method), to prove THIS exact wave-per-128-group structure (not the earlier
// 8-lane/two-slice scheme, which GPT's design replaces for the first implementation).
//
// Build: /opt/rocm-7.2.4/bin/hipcc --offload-arch=gfx1201 -O2 -I/opt/rocm-7.2.4/include hc-pf2-quant-a8.hip \
//          -o hc-pf2-quant-a8
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cmath>
#include <vector>

using rocwmma::float8_t;

constexpr int KB = 128;

// grid.x = rows, grid.y = HN/128; block = 32 (one wave per (row, group)).
__global__ void hc_pf2_quant_a8_kernel(const float* __restrict__ n_in, uint8_t* __restrict__ a8_out,
                                       float* __restrict__ ascale_out, int HN) {
    const int row = blockIdx.x, g = blockIdx.y, lane = threadIdx.x;
    const float* n_row = n_in + (size_t)row * HN + (size_t)g * KB;
    float v[4];
    float local_max = 0.0f;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        v[e] = n_row[lane * 4 + e];
        local_max = fmaxf(local_max, fabsf(v[e]));
    }
    // Full-wave butterfly max (order-independent, so a plain shuffle loop matches r4d_wave_max's result exactly).
    for (int off = 16; off; off >>= 1) local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    const float scale = local_max > 0.0f ? local_max / 448.0f : 1.0f;
    if (lane == 0) ascale_out[(size_t)row * (HN / KB) + g] = scale;
    uint8_t* out = a8_out + (size_t)row * HN + (size_t)g * KB;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float8_t c = float8_t(v[e] / scale);
        out[lane * 4 + e] = *reinterpret_cast<const uint8_t*>(&c);
    }
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/mnt/data/bigcherry-work/ndump.bin";
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return 1; }
    const int HN = 10240, NKB = HN / KB;
    std::vector<float> host_n(HN);
    size_t got = fread(host_n.data(), sizeof(float), HN, f);
    fclose(f);
    if (got != (size_t)HN) { printf("short read\n"); return 1; }
    // Replicate the real row count isn't needed to validate the KERNEL's per-group bookkeeping; one real row
    // suffices since each wave only ever touches its own 128-of-K group of this one row.
    const int ROWS = 1;

    float* dN; uint8_t* dA8; float* dScale;
    hipMalloc(&dN, HN * sizeof(float));
    hipMalloc(&dA8, HN);
    hipMalloc(&dScale, NKB * sizeof(float));
    hipMemcpy(dN, host_n.data(), HN * sizeof(float), hipMemcpyHostToDevice);

    hc_pf2_quant_a8_kernel<<<dim3(ROWS, NKB), 32>>>(dN, dA8, dScale, HN);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("FAIL launch: %s\n", hipGetErrorString(e)); return 1; }

    std::vector<uint8_t> a8(HN);
    std::vector<float> scale(NKB);
    hipMemcpy(a8.data(), dA8, HN, hipMemcpyDeviceToHost);
    hipMemcpy(scale.data(), dScale, NKB * sizeof(float), hipMemcpyDeviceToHost);

    int mismatches = 0;
    double sumsq_err = 0.0, sumsq_ref = 0.0;
    for (int g = 0; g < NKB; ++g) {
        float amax = 0.0f;
        for (int k = 0; k < KB; ++k) amax = fmaxf(amax, fabsf(host_n[g * KB + k]));
        const float ref_scale = amax > 0.0f ? amax / 448.0f : 1.0f;
        if (fabsf(ref_scale - scale[g]) > 1e-9f * fmaxf(1.0f, ref_scale)) {
            printf("group %d: scale mismatch kernel=%.9g ref=%.9g\n", g, scale[g], ref_scale);
            mismatches++;
        }
        for (int k = 0; k < KB; ++k) {
            const float8_t ref_c = float8_t(host_n[g * KB + k] / ref_scale);
            const uint8_t ref_bits = *reinterpret_cast<const uint8_t*>(&ref_c);
            if (ref_bits != a8[g * KB + k]) {
                if (mismatches < 5)
                    printf("group %d k %d: code mismatch kernel=0x%02x ref=0x%02x\n", g, k, a8[g * KB + k], ref_bits);
                mismatches++;
            }
            const float back = float(*reinterpret_cast<const float8_t*>(&a8[g * KB + k])) * scale[g];
            const double d = back - host_n[g * KB + k];
            sumsq_err += d * d;
            sumsq_ref += (double)host_n[g * KB + k] * host_n[g * KB + k];
        }
    }
    printf("hc_pf2_quant_a8_kernel vs offline ground truth: %d mismatches of %d scales + %d codes\n",
           mismatches, NKB, HN);
    printf("RMS relative error (kernel's own reconstruction): %.4f%%\n",
           100.0 * sqrt(sumsq_err / HN) / sqrt(sumsq_ref / HN));
    printf(mismatches == 0 ? "PASS: the wave-per-128-group quantiser is bit-for-bit correct.\n"
                           : "FAIL: do not port this into the real patch.\n");
    return mismatches == 0 ? 0 : 1;
}
