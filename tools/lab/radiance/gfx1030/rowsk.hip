// RAD23 microbench: the few-rows w4nl kernel (moe_lane_gemm.h lane_moe_rows_w4) on one card, synthetic w4nl weights and the 14 GiB
// gfx1030 tail's live rows-an-expert mix (58 experts, ~1,080 routes a 2048-row request), timed per tile shape and checked against the
// shipped shape's output. Reproduces the live tail's gate/up and down stage times within ~5%. No model, a few seconds a run.
// Build: hipcc -O3 --offload-arch=gfx1030 -I engines/radiance/components/moe-remote -I <radiance prefix>/include/radiance \
//          tools/lab/radiance/gfx1030/rowsk.hip -o rowsk
// Run:   HIP_VISIBLE_DEVICES=<card> ./rowsk [live|r1|r12|r100]   (rN: 58 experts of N rows)
#include "moe_remote_shm.h"
#include "moe_lane_gemm.h"
#include <random>
#include <cmath>
#include <vector>
#include <string>
using namespace r3_moe_gpu;
int main(int c, char** v) {
    std::string mix = c > 1 ? v[1] : "live";
    std::vector<int> rows;
    auto add = [&](int n, int r) { for (int i = 0; i < n; ++i) rows.push_back(r); };
    if (mix == "live") { add(12,1); add(7,2); add(8,3); add(8,6); add(8,12); add(6,24); add(5,48); add(5,100); }
    else add(58, std::stoi(mix.substr(1)));
    const uint32_t E = rows.size(), top_k = 10;
    uint32_t R = 0; for (int r : rows) R += r;
    const uint32_t T = 2048;
    std::mt19937 rng(1);
    for (int pass = 0; pass < 2; ++pass) {
        const bool gu = pass == 0;
        const uint32_t K = gu ? 2560 : 640, N = gu ? 1280 : 2560, M = gu ? T : R;   // activation rows
        std::vector<int32_t> off(2 * E + 1, 0), sorted(R);
        uint32_t at = 0;
        for (uint32_t e = 0; e < E; ++e) { off[2 * e] = at; for (int i = 0; i < rows[e]; ++i) { sorted[at] = (int32_t)(((at * 7919u) % T) * top_k); ++at; } off[2 * e + 1] = at; }
        off[2 * E] = at;
        std::vector<void*> cp(E), sp(E);
        for (uint32_t e = 0; e < E; ++e) {
            hipMalloc(&cp[e], (size_t)N * K / 2); { std::vector<uint8_t> h((size_t)N*K/2); for (auto& x : h) x = (uint8_t)rng(); hipMemcpy(cp[e], h.data(), h.size(), hipMemcpyHostToDevice); }
            hipMalloc(&sp[e], (size_t)N * K / 64); { std::vector<uint8_t> h((size_t)N*K/64); for (auto& x : h) x = (uint8_t)(0x28 + rng() % 16); hipMemcpy(sp[e], h.data(), h.size(), hipMemcpyHostToDevice); }
        }
        void **dcp, **dsp; hipMalloc(&dcp, E * 8); hipMalloc(&dsp, E * 8);
        hipMemcpy(dcp, cp.data(), E * 8, hipMemcpyHostToDevice); hipMemcpy(dsp, sp.data(), E * 8, hipMemcpyHostToDevice);
        uint8_t* aq; float* as; uint16_t* out; int32_t *doff, *dsorted;
        hipMalloc(&aq, (size_t)M * K); { std::vector<uint8_t> h((size_t)M*K); for (auto& x : h) { x = (uint8_t)(rng() % 0x78); if (rng() & 1) x |= 0x80; } hipMemcpy(aq, h.data(), h.size(), hipMemcpyHostToDevice); }
        hipMalloc(&as, (size_t)M * (K / 128) * 4); { std::vector<float> h((size_t)M*(K/128)); for (auto& x : h) x = 0.002f + (rng() % 1000) * 1e-5f; hipMemcpy(as, h.data(), h.size()*4, hipMemcpyHostToDevice); }
        hipMalloc(&out, (size_t)R * N * 2); uint16_t* out2; hipMalloc(&out2, (size_t)R * N * 2);
        hipMalloc(&doff, off.size() * 4); hipMemcpy(doff, off.data(), off.size() * 4, hipMemcpyHostToDevice);
        hipMalloc(&dsorted, R * 4); hipMemcpy(dsorted, sorted.data(), R * 4, hipMemcpyHostToDevice);
        const dim3 grid(2 * E, (N + kRowsTileN - 1) / kRowsTileN);
        auto launch = [&](auto kern, uint32_t tile, uint16_t* o) {
            hipLaunchKernelGGL(kern, dim3(2 * E, (N + tile - 1) / tile), dim3(256), 0, 0, aq, as, dsorted, doff, top_k, dcp, dsp, dcp, dsp, N, N, K, K, K, N, o, 0u);
        };
        const int ref = gu ? 2 : 4;   /* kRowsGuRB x kRowsGuOut = 16x4, kRowsDnRB x kRowsDnOut = 12x2 */
        auto run = [&](int v) {
            uint16_t* o = v == ref ? out : out2;   /* the shipped shape writes the reference */
            if (gu) switch (v) {
                case 0: launch(lane_moe_rows_w4<true, 8, 4>, 32, o); break;
                case 1: launch(lane_moe_rows_w4<true, 12, 4>, 32, o); break;
                case 2: launch(lane_moe_rows_w4<true, 16, 4>, 32, o); break;
                case 3: launch(lane_moe_rows_w4<true, 24, 4>, 32, o); break;
                case 4: launch(lane_moe_rows_w4<true, 12, 2>, 16, o); break;
                case 5: launch(lane_moe_rows_w4<true, 24, 2>, 16, o); break;
                case 6: launch(lane_moe_rows_w4<true, 32, 2>, 16, o); break;
                case 7: launch(lane_moe_rows_w4<true, 8, 8>, 64, o); break;
            } else switch (v) {
                case 0: launch(lane_moe_rows_w4<false, 8, 4>, 32, o); break;
                case 1: launch(lane_moe_rows_w4<false, 12, 4>, 32, o); break;
                case 2: launch(lane_moe_rows_w4<false, 16, 4>, 32, o); break;
                case 3: launch(lane_moe_rows_w4<false, 24, 4>, 32, o); break;
                case 4: launch(lane_moe_rows_w4<false, 12, 2>, 16, o); break;
                case 5: launch(lane_moe_rows_w4<false, 24, 2>, 16, o); break;
                case 6: launch(lane_moe_rows_w4<false, 32, 2>, 16, o); break;
                case 7: launch(lane_moe_rows_w4<false, 8, 8>, 64, o); break;
            }
        };
        const char* names[8] = {"w4 8x4", "w4 12x4", "w4 16x4", "w4 24x4", "w4 12x2", "w4 24x2", "w4 32x2", "w4 8x8"};
        run(ref); hipDeviceSynchronize();
        for (int v = 0; v < 8; ++v) {
            for (int w = 0; w < 3; ++w) run(v);
            hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
            hipEventRecord(a); for (int i = 0; i < 20; ++i) run(v); hipEventRecord(b); hipEventSynchronize(b);
            float ms; hipEventElapsedTime(&ms, a, b); const double t = ms * 1000 / 20;
            std::vector<uint16_t> h1((size_t)R * N), h2((size_t)R * N);
            hipMemcpy(h1.data(), out, h1.size() * 2, hipMemcpyDeviceToHost); hipMemcpy(h2.data(), out2, h2.size() * 2, hipMemcpyDeviceToHost);
            double num = 0, den = 0; for (size_t i = 0; i < h1.size(); ++i) { uint32_t u1 = (uint32_t)h1[i] << 16, u2 = (uint32_t)h2[i] << 16; float f1, f2; memcpy(&f1, &u1, 4); memcpy(&f2, &u2, 4); num += (double)(f1 - f2) * (f1 - f2); den += (double)f1 * f1; }
            printf("%s %s %-9s %8.1f us  %.2f TFMA/s  rel_l2 %.1e %s\n", mix.c_str(), gu ? "gate_up" : "down   ", names[v], t, (double)R * N * K / t / 1e6, v == ref ? 0.0 : std::sqrt(num / std::max(den, 1e-30)), hipGetErrorString(hipGetLastError()));
        }
    }
}
