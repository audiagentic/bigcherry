// RAD17: the completed hc_down_pf2_fp8_kernel epilogue (the TODO left in rad0007's scaffold), validated at the
// REAL scale: rows=kPf2RT=64, lr=320 (20 column tiles), kWaves=8 waves. Uses store_matrix_sync's row-major
// output directly (part_tile[i][j] = accumulator row i, col j, regardless of whatever internal per-lane
// layout rocwmma's fragment uses) so the epilogue only has to match the OUTPUT ARRAY's addressing contract,
// not replicate hc_down_pf2_kernel's hand-rolled e0/nn lane mapping -- store_matrix_sync already abstracts
// that away. Each of the 32 threads in a wave cooperatively writes 8 of the 256 tile elements (idx =
// threadIdx.x, +32 each step), giving a clean per-thread total[8] register accumulator across the K-group
// loop, matching the real kernel's per-lane element count without needing its exact assignment.
//
// Build: /opt/rocm-7.2.4/bin/hipcc --offload-arch=gfx1201 -O2 -I/opt/rocm-7.2.4/include \
//          hc-down-fp8-fullscale.hip -o hc-down-fp8-fullscale
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace rocwmma;

constexpr int RT = 64, MT = RT / 16, kThreads = 256, kWave = 32, kWaves = kThreads / kWave;
constexpr long HN = 10240, KB = 128, NKB = HN / KB;
constexpr long LR = 320, NTT = LR / 16;   // 20 column tiles
constexpr long DLD = 10560;                // row_bytes(HN, 128), confirmed against the container

using Afrag = fragment<matrix_a, 16, 16, 16, float8_t, row_major>;
using Bfrag = fragment<matrix_b, 16, 16, 16, float8_t, col_major>;
using Cfrag = fragment<accumulator, 16, 16, 16, float>;

// One workgroup = RT=64 rows, all LR=320 output columns (the real hc_down_pf2_fp8_kernel's shape). a8/ascale
// from the quant pre-pass; down is the real packed [codes][scales] weight; dout gets silu(v)/hc) directly
// (ks==1 path -- the ks>1 split-K path writes `part` unactivated instead, not exercised by this harness).
__global__ __launch_bounds__(kThreads) void hc_down_pf2_fp8_kernel(
    const uint8_t* __restrict__ a8, const float* __restrict__ ascale,
    const uint8_t* __restrict__ down, float* __restrict__ dout) {
    const int wave = threadIdx.x / kWave, lane = threadIdx.x % kWave;
    // Indexed by wave: 8 waves run concurrently (different nt each), so a single shared buffer here would
    // have every wave overwrite the others' partial sums -- caught by this harness (the exact bug class
    // already found once this session in hc-down-fp8-validate.hip).
    __shared__ float part_tile[kWaves][16][16];

    for (long nt = wave; nt < NTT; nt += kWaves) {
        const long col0 = nt * 16;
        for (int m = 0; m < MT; ++m) {
            float total[8] = {};
            for (long g = 0; g < NKB; ++g) {
                Cfrag fc;
                fill_fragment(fc, 0.0f);
                for (int s = 0; s < KB / 16; ++s) {
                    const long k0 = g * KB + s * 16;
                    Afrag fa; Bfrag fb;
                    load_matrix_sync(fa, reinterpret_cast<const float8_t*>(a8 + (size_t)(m * 16) * HN + k0), HN);
                    load_matrix_sync(fb, reinterpret_cast<const float8_t*>(down + (size_t)col0 * DLD + k0), DLD);
                    mma_sync(fc, fa, fb, fc);
                }
                store_matrix_sync(&part_tile[wave][0][0], fc, 16, mem_row_major);
                __syncwarp();
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const int idx = lane + t * kWave;
                    const int i = idx / 16, j = idx % 16;
                    const float asc = ascale[(size_t)(m * 16 + i) * NKB + g];
                    const float* wsc = reinterpret_cast<const float*>(down + (size_t)(col0 + j) * DLD + HN);
                    total[t] += part_tile[wave][i][j] * asc * wsc[g];
                }
            }
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const int idx = lane + t * kWave;
                const int i = idx / 16, j = idx % 16;
                const long row = m * 16 + i, col = col0 + j;
                const float v = total[t] * 0.25f;  // inv_hc for hc=4, matching the real epilogue's v*inv_hc
                dout[row * LR + col] = v / (1.0f + expf(-v));  // silu
            }
        }
    }
}

int main(int argc, char** argv) {
    const char* npath = argc > 1 ? argv[1] : "/mnt/data/bigcherry-work/ndump.bin";
    const char* wpath = argc > 2 ? argv[2] : "/mnt/data/bigcherry-work/wdump.bin";
    FILE* fn = fopen(npath, "rb"), *fw = fopen(wpath, "rb");
    if (!fn || !fw) { printf("cannot open inputs\n"); return 1; }
    std::vector<float> n_row(HN);
    fread(n_row.data(), sizeof(float), HN, fn);
    fclose(fn);
    std::vector<uint8_t> w_raw(LR * DLD);
    const size_t got = fread(w_raw.data(), 1, LR * DLD, fw);
    fclose(fw);
    if (got != (size_t)(LR * DLD)) { printf("short weight read: got %zu want %ld\n", got, LR * DLD); return 1; }

    std::vector<float> Af(RT * HN);
    for (int m = 0; m < RT; ++m) std::copy(n_row.begin(), n_row.end(), Af.begin() + (size_t)m * HN);

    std::vector<uint8_t> Aq(RT * HN);
    std::vector<float> Asc(RT * NKB);
    for (int m = 0; m < RT; ++m)
        for (long g = 0; g < NKB; ++g) {
            float amax = 0.0f;
            for (long k = 0; k < KB; ++k) amax = fmaxf(amax, fabsf(Af[(size_t)m * HN + g * KB + k]));
            const float scale = amax > 0.0f ? amax / 448.0f : 1.0f;
            Asc[m * NKB + g] = scale;
            for (long k = 0; k < KB; ++k) {
                const float8_t c = float8_t(Af[(size_t)m * HN + g * KB + k] / scale);
                Aq[(size_t)m * HN + g * KB + k] = *reinterpret_cast<const uint8_t*>(&c);
            }
        }

    std::vector<float> Wdq(LR * HN);
    for (long n = 0; n < LR; ++n)
        for (long g = 0; g < NKB; ++g) {
            const float wsc = *reinterpret_cast<const float*>(w_raw.data() + n * DLD + HN + g * 4);
            for (long k = 0; k < KB; ++k) {
                const float8_t c = *reinterpret_cast<const float8_t*>(&w_raw[n * DLD + g * KB + k]);
                Wdq[(size_t)n * HN + g * KB + k] = float(c) * wsc;
            }
        }
    std::vector<float> Adq(RT * HN);
    for (int m = 0; m < RT; ++m)
        for (long g = 0; g < NKB; ++g)
            for (long k = 0; k < KB; ++k) {
                const float8_t c = *reinterpret_cast<const float8_t*>(&Aq[(size_t)m * HN + g * KB + k]);
                Adq[(size_t)m * HN + g * KB + k] = float(c) * Asc[m * NKB + g];
            }
    std::vector<float> Dref(RT * LR, 0.0f);
    for (int m = 0; m < RT; ++m)
        for (long n = 0; n < LR; ++n) {
            double s = 0;
            for (long k = 0; k < HN; ++k) s += (double)Adq[(size_t)m * HN + k] * Wdq[(size_t)n * HN + k];
            const double v = s * 0.25;
            Dref[m * LR + n] = (float)(v / (1.0 + exp(-v)));
        }

    uint8_t *dAq, *dW; float *dAsc, *dOut;
    hipMalloc(&dAq, Aq.size()); hipMalloc(&dW, w_raw.size());
    hipMalloc(&dAsc, Asc.size() * sizeof(float)); hipMalloc(&dOut, RT * LR * sizeof(float));
    hipMemcpy(dAq, Aq.data(), Aq.size(), hipMemcpyHostToDevice);
    hipMemcpy(dW, w_raw.data(), w_raw.size(), hipMemcpyHostToDevice);
    hipMemcpy(dAsc, Asc.data(), Asc.size() * sizeof(float), hipMemcpyHostToDevice);

    hc_down_pf2_fp8_kernel<<<1, kThreads>>>(dAq, dAsc, dW, dOut);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("FAIL launch: %s\n", hipGetErrorString(e)); return 1; }
    std::vector<float> D(RT * LR);
    hipMemcpy(D.data(), dOut, RT * LR * sizeof(float), hipMemcpyDeviceToHost);

    double maxabs = 0, maxrel = 0;
    for (size_t i = 0; i < D.size(); ++i) {
        const double d = fabs(D[i] - Dref[i]);
        maxabs = std::max(maxabs, d);
        if (fabs(Dref[i]) > 1e-4) maxrel = std::max(maxrel, d / fabs(Dref[i]));
    }
    printf("full-scale (RT=64, lr=320, 8 waves) FP8 down + silu epilogue vs reference:\n");
    printf("  max abs diff=%.6g max rel diff=%.6g\n", maxabs, maxrel);
    printf("D[0]=%f Dref[0]=%f  D[last]=%f Dref[last]=%f\n", D[0], Dref[0], D.back(), Dref.back());
    return maxrel < 1e-2 ? 0 : 1;
}
