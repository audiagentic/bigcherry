// RAD16 step 1: the decisive, no-engine-change experiment the plan calls for -- project the MTP drafter's GEMM
// cost on the 6900 XT (gfx1030) at decode shapes (M=1 per draft pass), so RAD16's "go only if >= +10%
// end-to-end at 1 stream" criterion can be checked against a real number instead of staying undecided.
//
// gfx1030 (RDNA2) has no WMMA and no FP8; its matrix throughput comes from packed f16 dot products
// (v_dot2c_f32_f16, exposed here as __builtin_amdgcn_fdot2 -- the same idiom already proven working in
// bc_moe_gemv.hip this session, just ported to a plain GEMV shape instead of MoE routing).
//
// Shapes are Qwen3.8-Flash-Next's MTP layer (hidden_size=2560, one full-attention transformer block) and its
// lm_head (vocab_size=248320), read via rad-info --meta on the real container. The per-projection split
// (q/o roughly hidden x hidden, kv at head_dim=256 x 2 kv heads, gate_up/down at a placeholder n_ff) is an
// order-of-magnitude approximation, not an exact replica of the real op graph -- good enough for RAD16's
// go/no-go threshold, not for a production estimate.
//
// Build: /opt/rocm-7.2.4/bin/hipcc --offload-arch=gfx1030 -O3 rdna2-draft-bench.hip -o rdna2-draft-bench
// Run:   ROCR_VISIBLE_DEVICES=<6900 XT's index> ./rdna2-draft-bench
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include <vector>

typedef _Float16 h2v __attribute__((ext_vector_type(2)));

// out[N] = W[N,K] @ in[K], f16 in/out, f32 accumulate, packed f16 dot. One wave per output row.
__global__ void gemv_f16(const _Float16* __restrict__ W, const _Float16* __restrict__ x, _Float16* __restrict__ y,
                         uint32_t N, uint32_t K) {
    const uint32_t row = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
    if (row >= N) return;
    const uint32_t lane = threadIdx.x % warpSize;
    const h2v* wr = reinterpret_cast<const h2v*>(W + (uint64_t)row * K);
    const h2v* xr = reinterpret_cast<const h2v*>(x);
    float acc = 0.0f;
    for (uint32_t k = lane; k < K / 2; k += warpSize)
        acc = __builtin_amdgcn_fdot2(wr[k], xr[k], acc, false);
    for (uint32_t o = warpSize / 2; o; o >>= 1) acc += __shfl_down(acc, o);
    if (lane == 0) y[row] = (_Float16)acc;
}

double time_gemv(uint32_t N, uint32_t K, int reps) {
    _Float16 *W, *x, *y;
    hipError_t e;
    e = hipMalloc(&W, (size_t)N * K * sizeof(_Float16));
    e = hipMalloc(&x, (size_t)K * sizeof(_Float16));
    e = hipMalloc(&y, (size_t)N * sizeof(_Float16));
    e = hipMemset(W, 0x3C, (size_t)N * K * sizeof(_Float16));  // ~1.0 in fp16 bit pattern
    e = hipMemset(x, 0x3C, (size_t)K * sizeof(_Float16));
    (void)e;
    const uint32_t waves_per_block = 4;  // 128 threads
    dim3 grid((N + waves_per_block - 1) / waves_per_block);
    dim3 block(waves_per_block * 32);
    for (int i = 0; i < 5; ++i) gemv_f16<<<grid, block>>>(W, x, y, N, K);
    hipDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) gemv_f16<<<grid, block>>>(W, x, y, N, K);
    hipDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    hipFree(W); hipFree(x); hipFree(y);
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
}

// A small PCIe round trip (hidden-state transfer for drafter placement), timed the same way a real handoff
// would move it: host-pinned-free staging via a plain device buffer and hipMemcpy, repeated.
double time_roundtrip(size_t bytes, int reps) {
    void* d; hipMalloc(&d, bytes);
    std::vector<uint8_t> h(bytes, 0);
    for (int i = 0; i < 5; ++i) { hipMemcpy(d, h.data(), bytes, hipMemcpyHostToDevice); hipMemcpy(h.data(), d, bytes, hipMemcpyDeviceToHost); }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) { hipMemcpy(d, h.data(), bytes, hipMemcpyHostToDevice); hipMemcpy(h.data(), d, bytes, hipMemcpyDeviceToHost); }
    auto t1 = std::chrono::high_resolution_clock::now();
    hipFree(d);
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
}

int main() {
    const uint32_t H = 2560, VOCAB = 248320, NFF = 640;  // n_ff from rad-info --meta; MTP shape TBD beyond this
    const double q = time_gemv(H, H, 200);
    const double o = time_gemv(H, H, 200);
    const double kv = time_gemv(256 * 2, H, 200);
    const double gate_up = time_gemv(2 * NFF, H, 200);
    const double down = time_gemv(H, NFF, 200);
    const double head = time_gemv(VOCAB, H, 20);
    const double layer_total = q + o + kv + gate_up + down;
    const double rt_small = time_roundtrip(5 * 1024, 100);
    const double rt_large = time_roundtrip(30 * 1024, 100);

    printf("gfx1030 GEMV microbench (decode-shaped, M=1), us per call:\n");
    printf("  q_proj  [%u x %u]      %.1f us\n", H, H, q);
    printf("  o_proj  [%u x %u]      %.1f us\n", H, H, o);
    printf("  kv_proj [%u x %u]       %.1f us\n", 512, H, kv);
    printf("  gate_up [%u x %u]      %.1f us\n", 2 * NFF, H, gate_up);
    printf("  down    [%u x %u]       %.1f us\n", H, NFF, down);
    printf("  one MTP layer total:          %.1f us\n", layer_total);
    printf("  lm_head [%u x %u]  %.1f us\n", VOCAB, H, head);
    printf("  3 serial draft passes (layer only, no head): %.1f us\n", layer_total * 3);
    printf("  3 serial draft passes + head each pass:      %.1f us\n", (layer_total + head) * 3);
    printf("PCIe round trip: 5 KB %.1f us, 30 KB %.1f us\n", rt_small, rt_large);
    printf("\nRAD16 reference point: the R9700 runs the real drafter chain (3 serial passes + head) in\n");
    printf("~0.8 ms today. Compare that against the totals above (plus the round trips) to see whether\n");
    printf("offloading to gfx1030 could plausibly clear RAD16's go/no-go bar (>= +10%% end-to-end at 1\n");
    printf("stream) before any engine-patch work -- this is an order-of-magnitude projection, not a\n");
    printf("production estimate (see the shape caveats in the file header).\n");
    return 0;
}
