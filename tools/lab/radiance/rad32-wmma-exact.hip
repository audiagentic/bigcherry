/* RAD32: does a gfx11 WMMA sum small exact products exactly? radiance 1.13.0's MXFP4 MoE self-test pairs a 2^125 E8M0 scale
 * with a 32-wide group whose products cancel exactly; a residue in the matrix sum becomes ~1e30 after the scale.
 * Inputs: activations A/4 with A in [-16,16] and weights W/2 with W in {0,+-1,+-2,+-3,+-4,+-6,+-8,+-12} (E4M3 and E2M1
 * values, exact in bf16 and f16); every product is a multiple of 1/8 and every 32-term sum fits f32 exactly, so the oracle is
 * an int64 sum of A*W divided by 8. Modes (a 16x16x16 product per 16 of K, row m of A against row n of B, diagonal checked):
 *   0 bf16 chained over K=32   1 bf16 one product (K=16)   2 bf16 two products from zero, added in f32
 *   3 f16  chained over K=32   4 f16  one product (K=16)   5 f16  two products from zero, added in f32
 * Prints, per mode: sums not equal to the exact value, exact-zero sums that came back non-zero, the mean signed error and
 * the worst absolute error. Usage: rad32-wmma-exact [trials] */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

template <class V, class T, int MODE>
__device__ static v8f run(const T* A, const T* B, int lane, int row) {
    v8f acc = {0, 0, 0, 0, 0, 0, 0, 0}, part = acc;
    const int steps = (MODE == 1 || MODE == 4) ? 1 : 2;
    for (int s = 0; s < steps; ++s) {
        V a, b;
        for (int i = 0; i < 16; ++i) { a[i] = A[row * 32 + s * 16 + i]; b[i] = B[row * 32 + s * 16 + i]; }
        v8f zero = {0, 0, 0, 0, 0, 0, 0, 0};
        if constexpr (MODE == 2 || MODE == 5) {
            v8f p;
            if constexpr (sizeof(T) == 2 && MODE == 2) p = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32((v16bf)a, (v16bf)b, zero);
            else p = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((v16h)a, (v16h)b, zero);
            for (int j = 0; j < 8; ++j) part[j] += p[j];
        } else if constexpr (MODE <= 2) {
            acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32((v16bf)a, (v16bf)b, acc);
        } else {
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((v16h)a, (v16h)b, acc);
        }
    }
    return (MODE == 2 || MODE == 5) ? part : acc;
}

template <int MODE>
__global__ void k(const __bf16* Ab, const __bf16* Bb, const _Float16* Ah, const _Float16* Bh, float* D) {
    const int lane = threadIdx.x & 31, row = lane & 15;
    v8f r;
    if constexpr (MODE <= 2) r = run<v16bf, __bf16, MODE>(Ab, Bb, lane, row);
    else r = run<v16h, _Float16, MODE>(Ah, Bh, lane, row);
    for (int j = 0; j < 8; ++j) D[(2 * j + (lane >> 4)) * 16 + (lane & 15)] = r[j];   // gfx11 C layout: row 2j + (lane >> 4)
}

int main(int argc, char** argv) {
    const int trials = argc > 1 ? atoi(argv[1]) : 2000;
    static const int wmag[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    srand(7);
    __bf16 *dAb, *dBb; _Float16 *dAh, *dBh; float* dD;
    hipMalloc(&dAb, 1024); hipMalloc(&dBb, 1024); hipMalloc(&dAh, 1024); hipMalloc(&dBh, 1024); hipMalloc(&dD, 1024);
    const char* names[6] = {"bf16 chained K32", "bf16 one K16", "bf16 2x K16 + f32 add", "f16 chained K32", "f16 one K16", "f16 2x K16 + f32 add"};
    long bad[6] = {0}, zbad[6] = {0}, zeros[6] = {0}, total = 0; double sgn[6] = {0}, worst[6] = {0};
    for (int t = 0; t < trials; ++t) {
        std::vector<__bf16> Ab(512), Bb(512); std::vector<_Float16> Ah(512), Bh(512); std::vector<int> ai(512), wi(512);
        for (int i = 0; i < 512; ++i) {
            ai[i] = (rand() % 33) - 16; wi[i] = ((rand() & 8) ? -1 : 1) * wmag[rand() & 7];
            Ab[i] = (__bf16)(ai[i] / 4.0f); Bb[i] = (__bf16)(wi[i] / 2.0f); Ah[i] = (_Float16)(ai[i] / 4.0f); Bh[i] = (_Float16)(wi[i] / 2.0f);
        }
        hipMemcpy(dAb, Ab.data(), 1024, hipMemcpyHostToDevice); hipMemcpy(dBb, Bb.data(), 1024, hipMemcpyHostToDevice);
        hipMemcpy(dAh, Ah.data(), 1024, hipMemcpyHostToDevice); hipMemcpy(dBh, Bh.data(), 1024, hipMemcpyHostToDevice);
        for (int mode = 0; mode < 6; ++mode) {
            switch (mode) {
                case 0: k<0><<<1, 32>>>(dAb, dBb, dAh, dBh, dD); break; case 1: k<1><<<1, 32>>>(dAb, dBb, dAh, dBh, dD); break;
                case 2: k<2><<<1, 32>>>(dAb, dBb, dAh, dBh, dD); break; case 3: k<3><<<1, 32>>>(dAb, dBb, dAh, dBh, dD); break;
                case 4: k<4><<<1, 32>>>(dAb, dBb, dAh, dBh, dD); break; default: k<5><<<1, 32>>>(dAb, dBb, dAh, dBh, dD); break;
            }
            float D[256]; hipMemcpy(D, dD, 1024, hipMemcpyDeviceToHost);
            const int kmax = (mode == 1 || mode == 4) ? 16 : 32;
            for (int m = 0; m < 16; ++m) {
                int64_t sum = 0;
                for (int kk = 0; kk < kmax; ++kk) sum += (int64_t)ai[m * 32 + kk] * wi[m * 32 + kk];
                const double want = sum / 8.0, got = D[m * 16 + m];
                zeros[mode] += sum == 0;
                if (got != want) { ++bad[mode]; zbad[mode] += sum == 0; sgn[mode] += got - want; worst[mode] = fmax(worst[mode], fabs(got - want)); }
            }
        }
        total += 16;
    }
    for (int mode = 0; mode < 6; ++mode)
        printf("%-24s %6ld of %ld sums not exact, %ld of %ld exact-zero sums non-zero, mean signed err %+.3g, worst %.3g\n",
               names[mode], bad[mode], total, zbad[mode], zeros[mode], sgn[mode] / total, worst[mode]);
    return 0;
}
