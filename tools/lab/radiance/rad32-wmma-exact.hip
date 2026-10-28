/* RAD32: is a gfx11 bf16 WMMA sum of small exact E4M3-by-E2M1 products exactly equal to the host's? radiance 1.13.0's
 * MXFP4 MoE test pairs a 2^125 E8M0 scale with a 32-wide group whose products cancel exactly; any residue in the matrix
 * sum becomes ~1e29 after the scale. This runs 16x16x16 products (two chained, as the kernels do) of random E4M3
 * activations and E2M1 weights with the gfx11 bf16 WMMA and compares each element with a double-precision sum, counting
 * results that are not bit-equal to the exact value and the largest absolute difference. Usage: rad32-wmma-exact [trials] */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

__global__ void k(const __bf16* A, const __bf16* B, float* D) {   // A: 16x32, B: 16x32 (rows of K), D: 16x16
    const int lane = threadIdx.x & 31, row = lane & 15;
    v8f acc = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int s = 0; s < 2; ++s) {
        v16bf a, b;
        for (int i = 0; i < 16; ++i) { a[i] = A[row * 32 + s * 16 + i]; b[i] = B[row * 32 + s * 16 + i]; }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
    }
    for (int j = 0; j < 8; ++j) D[(2 * j + (lane >> 4)) * 16 + (lane & 15)] = acc[j];   // gfx11: row 2j + (lane >> 4)
}

static float bf(float x) { uint32_t u; memcpy(&u, &x, 4); u &= 0xffff0000u; memcpy(&x, &u, 4); return x; }

int main(int argc, char** argv) {
    const int trials = argc > 1 ? atoi(argv[1]) : 2000;
    static const double mag[8] = {0, .5, 1, 1.5, 2, 3, 4, 6};
    srand(7);
    __bf16 *dA, *dB; float* dD;
    hipMalloc(&dA, 16 * 32 * 2); hipMalloc(&dB, 16 * 32 * 2); hipMalloc(&dD, 256 * 4);
    long bad = 0, cancel_bad = 0, cancel = 0; double worst = 0;
    for (int t = 0; t < trials; ++t) {
        std::vector<__bf16> A(512), B(512); std::vector<double> a(512), b(512);
        for (int i = 0; i < 512; ++i) {
            // activations: multiples of 1/4 in [-4, 4] (E4M3-exact), weights: signed E2M1 magnitudes
            const double av = ((rand() % 33) - 16) / 4.0, bv = ((rand() & 8) ? -1 : 1) * mag[rand() & 7];
            a[i] = av; b[i] = bv; A[i] = (__bf16)(float)av; B[i] = (__bf16)(float)bv;
        }
        hipMemcpy(dA, A.data(), 1024, hipMemcpyHostToDevice); hipMemcpy(dB, B.data(), 1024, hipMemcpyHostToDevice);
        k<<<1, 32>>>(dA, dB, dD);
        float D[256]; hipMemcpy(D, dD, 1024, hipMemcpyDeviceToHost);
        // The kernel multiplies row m of A with row m of B (lane row for both), so the check is the diagonal
        for (int m = 0; m < 16; ++m) {
            double want = 0;
            for (int kk = 0; kk < 32; ++kk) want += a[m * 32 + kk] * b[m * 32 + kk];
            const float got = D[m * 16 + m];
            cancel += want == 0;
            if ((double)got != want) { ++bad; cancel_bad += want == 0; worst = fmax(worst, fabs(got - want)); }
        }
    }
    printf("trials %d x 16 sums: %ld not bit-equal to the exact sum (%ld of the %ld exactly-zero sums), worst abs diff %g\n",
           trials, bad, cancel_bad, cancel, worst);
    return 0;
}
