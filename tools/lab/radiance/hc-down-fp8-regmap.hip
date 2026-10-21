// RAD17: empirically derives rocwmma::fragment<accumulator,...>'s per-thread operator[] (register-direct
// access) index-to-(row,col) mapping, by comparing it in the SAME kernel against the already-proven
// store_matrix_sync+mem_row_major reference (hc-down-fp8-fullscale.hip's method). rocWMMA's own docs say
// "vector elements have no guaranteed order or locality" for operator[] -- this does NOT assume a mapping,
// it DISCOVERS one empirically and cross-checks it is a true bijection (every (row,col) hit exactly once)
// before trusting it, exactly this project's standing methodology (probe with known data, derive, verify).
//
// Motivation: hc_down_pf2_fp8_kernel's measured 0.688ms (more than the entire 530.7us real-hardware PF2
// baseline) is dominated by store_matrix_sync -> LDS -> __syncwarp() -> read-back, once per 128-K-group (80
// times a call), needed only to extract fc's opaque per-thread values for the block-scaled rescale. If
// operator[] gives the same 8 values per thread directly, in SOME fixed (if undocumented) per-thread order,
// the rescale can read them straight from registers with zero LDS/barrier traffic -- the real fix, not a
// workaround.
//
// Build: /opt/rocm-7.2.4/bin/hipcc --offload-arch=gfx1201 -O2 -I/opt/rocm-7.2.4/include \
//          hc-down-fp8-regmap.hip -o hc-down-fp8-regmap
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace rocwmma;

using Afrag = fragment<matrix_a, 16, 16, 16, float8_t, row_major>;
using Bfrag = fragment<matrix_b, 16, 16, 16, float8_t, col_major>;
using Cfrag = fragment<accumulator, 16, 16, 16, float>;

// One 16x16 tile, one wave. Writes BOTH extraction methods' results, plus which (row,col) each store_matrix_sync
// slot corresponds to for each thread (so the host can build the operator[] index -> (row,col) map).
__global__ void probe(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                      float* __restrict__ via_store, float* __restrict__ via_reg,
                      int* __restrict__ row_of, int* __restrict__ col_of) {
    const int lane = threadIdx.x;
    Afrag fa; Bfrag fb; Cfrag fc;
    fill_fragment(fc, 0.0f);
    load_matrix_sync(fa, reinterpret_cast<const float8_t*>(A), 16);
    load_matrix_sync(fb, reinterpret_cast<const float8_t*>(B), 16);
    mma_sync(fc, fa, fb, fc);

    __shared__ float tile[16][16];
    store_matrix_sync(&tile[0][0], fc, 16, mem_row_major);
    __syncwarp();
    // Known-correct reference layout, matching hc-down-fp8-fullscale.hip exactly: thread `lane` owns
    // output elements idx = lane + t*32 for t in 0..7, i=idx/16, j=idx%16.
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const int idx = lane + t * 32;
        const int i = idx / 16, j = idx % 16;
        via_store[lane * 8 + t] = tile[i][j];
        row_of[lane * 8 + t] = i;
        col_of[lane * 8 + t] = j;
    }
    // operator[] direct register access: Size should also be 8 for this config (32 lanes, 256 elements).
#pragma unroll
    for (uint32_t e = 0; e < fc.num_elements; ++e) via_reg[lane * fc.num_elements + e] = fc[e];
}

int main() {
    // A = identity, so out[i][j] = B[i][j] exactly (one nonzero term, no summation/rounding beyond B's own
    // E4M3 representation). B[i][j] is set to 2^t where idx=i*16+j, t=idx/32 -- the known store-mapping's
    // OWN slot index for (i,j) (idx=lane+t*32, i=idx/16, j=idx%16, so t=idx/32, lane=idx%32). Powers of two
    // from 2^0 to 2^7 are exactly representable in E4M3 with zero rounding error, and -- critically -- EVERY
    // lane's own 8 slots (t=0..7) get 8 PAIRWISE DISTINCT values, which is all the matching below needs
    // (values may repeat ACROSS different lanes; matching is done independently per lane).
    std::vector<uint8_t> A(16 * 16, 0), B(16 * 16, 0);
    for (int i = 0; i < 16; ++i) {
        const float8_t one(1.0f);
        A[i * 16 + i] = *reinterpret_cast<const uint8_t*>(&one);
    }
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j) {
            const int idx = i * 16 + j, t = idx / 32;
            const float8_t v((float)(1 << t));
            B[i * 16 + j] = *reinterpret_cast<const uint8_t*>(&v);
        }
    uint8_t *dA, *dB; float *dStore, *dReg; int *dRow, *dCol;
    hipMalloc(&dA, A.size()); hipMalloc(&dB, B.size());
    hipMalloc(&dStore, 256 * sizeof(float)); hipMalloc(&dReg, 256 * sizeof(float));
    hipMalloc(&dRow, 256 * sizeof(int)); hipMalloc(&dCol, 256 * sizeof(int));
    hipMemcpy(dA, A.data(), A.size(), hipMemcpyHostToDevice);
    hipMemcpy(dB, B.data(), B.size(), hipMemcpyHostToDevice);

    probe<<<1, 32>>>(dA, dB, dStore, dReg, dRow, dCol);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("FAIL launch: %s\n", hipGetErrorString(e)); return 1; }

    std::vector<float> store(256), reg(256);
    std::vector<int> rowo(256), colo(256);
    hipMemcpy(store.data(), dStore, 256 * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(reg.data(), dReg, 256 * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(rowo.data(), dRow, 256 * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(colo.data(), dCol, 256 * sizeof(int), hipMemcpyDeviceToHost);

    // Sanity: store-path must reproduce B's own values exactly (out[i][j] = B[i][j] since A is identity).
    int store_bad = 0;
    for (int k = 0; k < 256; ++k) {
        const int idx = rowo[k] * 16 + colo[k], t = idx / 32;
        const float want = (float)(1 << t);
        if (store[k] != want) ++store_bad;
    }
    printf("store-path sanity: %d/256 wrong (want 0)\n", store_bad);

    // For each thread's 8 register slots, find which (row,col) it matches by value: since the true matrix
    // is the identity, a register slot holding 1.0 must be the (row,col) where row==col for SOME row this
    // thread's registers cover; a slot holding 0.0 is consistent with many (row,col) pairs. So instead:
    // directly compare reg[] against store[] value-for-value PER THREAD (same 8 global values, unknown
    // permutation within the thread's own 8) -- find the permutation per lane, then check it is the SAME
    // permutation for every lane (expected, since all threads run the same hardware instruction pattern).
    std::vector<int> perm(8, -1);
    bool consistent = true;
    for (int lane = 0; lane < 32; ++lane) {
        int this_perm[8];
        bool used[8] = {};
        for (int a = 0; a < 8; ++a) {
            int match = -1;
            for (int b = 0; b < 8; ++b) {
                if (used[b]) continue;
                if (store[lane * 8 + a] == reg[lane * 8 + b]) { match = b; break; }
            }
            if (match < 0) { printf("lane %d slot %d: no matching register value (store=%.1f)\n",
                                     lane, a, store[lane * 8 + a]); consistent = false; break; }
            used[match] = true;
            this_perm[a] = match;
        }
        if (!consistent) break;
        if (lane == 0) { std::memcpy(perm.data(), this_perm, sizeof(this_perm)); }
        else {
            for (int a = 0; a < 8; ++a)
                if (this_perm[a] != perm[a]) {
                    printf("lane %d: permutation differs from lane 0 at slot %d (%d vs %d)\n",
                           lane, a, this_perm[a], perm[a]);
                    consistent = false;
                }
        }
    }
    if (consistent) {
        printf("CONSISTENT bijection found, same for all 32 lanes:\n  store-slot t -> register-index: [");
        for (int a = 0; a < 8; ++a) printf("%d%s", perm[a], a < 7 ? ", " : "");
        printf("]\n");
        printf("i.e. fc[%%d] == store-slot t where t = inverse_perm[reg_index]. Safe to use fc[] directly "
               "if every lane applies this SAME fixed mapping.\n");
    } else {
        printf("NOT SAFE: no consistent per-lane bijection -- fc[] cannot be used as a drop-in replacement "
               "for store_matrix_sync without per-lane-variable handling (or at all).\n");
    }
    return consistent ? 0 : 1;
}
