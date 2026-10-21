// RAD15 debug: isolate whether rocwmma::fragment's load_matrix_sync handles a large leading dimension (a 16x16
// sub-tile inside a much wider stored matrix) correctly, independent of the multi-tile/multi-wave harness.
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace rocwmma;
using Afrag = fragment<matrix_a, 16, 16, 16, float8_t, row_major>;
using Bfrag = fragment<matrix_b, 16, 16, 16, float8_t, col_major>;
using Cfrag = fragment<accumulator, 16, 16, 16, float>;

__global__ void mm16_strided(const float8_t* a, const float8_t* b, float* c, int ld) {
    Afrag fa; Bfrag fb; Cfrag fc;
    fill_fragment(fc, 0.0f);
    load_matrix_sync(fa, a, ld);
    load_matrix_sync(fb, b, ld);
    mma_sync(fc, fa, fb, fc);
    store_matrix_sync(c, fc, 16, mem_row_major);
}

int main() {
    const int LD = 10240;  // the real K
    std::vector<float8_t> Af(16 * LD, float8_t(0.0f)), Bf(16 * LD, float8_t(0.0f));
    std::vector<float> Aref(16 * 16), Bref(16 * 16);
    for (int i = 0; i < 16; ++i)
        for (int k = 0; k < 16; ++k) {
            const float v = (float)((i + k) % 3) - 1.0f;
            Af[i * LD + k] = float8_t(v);   // row i, col k of the 16x16 tile at the front of a LD-wide row
            Aref[i * 16 + k] = v;
        }
    for (int k = 0; k < 16; ++k)
        for (int j = 0; j < 16; ++j) {
            const float v = (float)((k * 2 + j) % 3) - 1.0f;
            // B col-major with ld=LD: element (row=k,col=j) at offset j*LD+k
            Bf[j * LD + k] = float8_t(v);
            Bref[k * 16 + j] = v;
        }
    std::vector<float> Cref(256, 0.0f);
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j) {
            float s = 0;
            for (int k = 0; k < 16; ++k) s += Aref[i * 16 + k] * Bref[k * 16 + j];
            Cref[i * 16 + j] = s;
        }
    float8_t *da, *db; float *dc;
    hipMalloc(&da, Af.size()); hipMalloc(&db, Bf.size()); hipMalloc(&dc, 256 * sizeof(float));
    hipMemcpy(da, Af.data(), Af.size(), hipMemcpyHostToDevice);
    hipMemcpy(db, Bf.data(), Bf.size(), hipMemcpyHostToDevice);
    mm16_strided<<<1, 32>>>(da, db, dc, LD);
    hipDeviceSynchronize();
    std::vector<float> C(256);
    hipMemcpy(C.data(), dc, 256 * sizeof(float), hipMemcpyDeviceToHost);
    double maxabs = 0;
    for (int i = 0; i < 256; ++i) maxabs = std::max(maxabs, (double)std::fabs(C[i] - Cref[i]));
    printf("large-ld (%d) single-tile test: C[0]=%f ref[0]=%f C[255]=%f ref[255]=%f max abs diff=%.6g\n",
           LD, C[0], Cref[0], C[255], Cref[255], maxabs);
    return 0;
}
