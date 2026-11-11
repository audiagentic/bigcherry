/* XDP04: isolated gfx1100 native W4A4 WMMA oracle and timing probe.
 *
 * DO NOT REGISTER IN RADIANCE. No vendor edits. No hidden model dependencies.
 * A is signed 4-bit (two's complement packed); W is unsigned 4-bit packed.
 * Per-128-group C = (sum(A4*W4) - zero_w*sum(A4)) * scale_a * scale_w.
 * Exact signed/unsigned 4-bit integer dot product; f32 output for testing.
 *
 * 1 wave32 per 16x16 output tile. Direct global loads are deliberately
 * unoptimized: measure feasibility/correctness, NOT competitive throughput.
 * GPU timing excludes H2D transfers, activation quantization, and engine costs.
 *
 * M,N multiples of 16, K multiple of 128. Usage: probe M N K iters seed
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using v2i = int __attribute__((ext_vector_type(2)));
using v8i = int __attribute__((ext_vector_type(8)));

static bool check(hipError_t status, const char* what) {
    if (status == hipSuccess) return true;
    std::fprintf(stderr, "XDP04 HIP error at %s: %s\n", what, hipGetErrorString(status));
    return false;
}

__global__ __launch_bounds__(32) static void native_w4a4_oracle(
    const unsigned char* __restrict__ apacked,
    const unsigned char* __restrict__ wpacked,
    const float* __restrict__ a_scale,
    const float* __restrict__ w_scale,
    const int* __restrict__ a_sum,
    const int* __restrict__ w_zero,
    float* __restrict__ output, int M, int N, int K) {
    const int lane = int(threadIdx.x) & 31;
    const int wrow = int(blockIdx.x) * 16 + (lane & 15);
    const int arow = int(blockIdx.y) * 16 + (lane & 15);
    const int G = K / 128;
    float sums[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    // Physical operands are native gfx11 2 x 32-bit = 16 nibbles per lane.
    // Each 16-lane operand row is repeated across halves of a wave32.
    for (int g = 0; g < G; ++g) {
        v8i acc = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int step = 0; step < 8; ++step) {
            const size_t byte = size_t(g * 128 + step * 16) / 2;
            const auto* w = reinterpret_cast<const unsigned*>(
                wpacked + size_t(wrow) * size_t(K / 2) + byte);
            const auto* a = reinterpret_cast<const unsigned*>(
                apacked + size_t(arow) * size_t(K / 2) + byte);
            v2i wp = {int(w[0]), int(w[1])};
            v2i ap = {int(a[0]), int(a[1])};
            // W unsigned 0..15; A signed -8..7; exact int32 accumulate.
            acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                false, wp, true, ap, acc, false);
        }
        // RDNA3 C: lane (half, col), register e -> C row 2*e+half, col.
        // WMMA C row is W (srcA), C column is A (srcB). The desired
        // serving layout C[m,n] is the transpose of native C[n,m].
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int n = int(blockIdx.x) * 16 + 2 * e + (lane >> 4);
            const int m = int(blockIdx.y) * 16 + (lane & 15);
            const int corrected = acc[e] - w_zero[n * G + g] * a_sum[m * G + g];
            sums[e] = fmaf(float(corrected) * a_scale[m * G + g],
                           w_scale[n * G + g], sums[e]);
        }
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int n = int(blockIdx.x) * 16 + 2 * e + (lane >> 4);
        const int m = int(blockIdx.y) * 16 + (lane & 15);
        output[size_t(m) * N + n] = sums[e];
    }
}

static void pack_value(std::vector<unsigned char>& target, int row, int K,
                       int offset, int v) {
    const size_t at = size_t(row) * (K / 2) + offset / 2;
    const unsigned char nib = static_cast<unsigned char>(v & 15);
    if (offset & 1) target[at] = static_cast<unsigned char>((target[at] & 15) | (nib << 4));
    else target[at] = static_cast<unsigned char>((target[at] & 240) | nib);
}
static int unpack_value(const std::vector<unsigned char>& source, int row,
                        int K, int offset, bool sign) {
    const unsigned char byte = source[size_t(row) * (K / 2) + offset / 2];
    int code = (byte >> ((offset & 1) * 4)) & 15;
    if (sign && code >= 8) code -= 16;
    return code;
}

static int run(int M, int N, int K, int iters, unsigned seed) {
    if (M < 16 || M > 512 || M % 16 || N < 16 || N > 2048 || N % 16
            || K < 128 || K > 8192 || K % 128 || iters < 1 || iters > 10000
            || uint64_t(M) * N * K > 25'000'000ull) {
        std::fprintf(stderr, "XDP04 invalid shape/iters: M,N multiple of 16; K multiple of 128; bounded 25M products\n");
        return 2;
    }
    int count = 0;
    if (!check(hipGetDeviceCount(&count), "device count") || count < 1) return 3;
    hipDeviceProp_t props{};
    if (!check(hipGetDeviceProperties(&props, 0), "device props")) return 3;
    if (std::strncmp(props.gcnArchName, "gfx1100", 7) != 0) {
        std::fprintf(stderr, "XDP04 requires gfx1100, device 0 is %s\n", props.gcnArchName);
        return 3;
    }
    const int G = K / 128;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> aq(-7, 7), wq(0, 15), wz(2, 13);
    std::uniform_real_distribution<float> as(0.008f, 0.045f), ws(0.01f, 0.09f);
    std::vector<unsigned char> A(size_t(M) * (K / 2), 0), W(size_t(N) * (K / 2), 0);
    std::vector<float> Sa(size_t(M) * G), Sw(size_t(N) * G);
    std::vector<int> Az(size_t(M) * G, 0), Zw(size_t(N) * G);
    for (int m = 0; m < M; ++m) {
        for (int g = 0; g < G; ++g) {
            Sa[size_t(m) * G + g] = as(rng);
            for (int k = g * 128; k < (g + 1) * 128; ++k) {
                int code = aq(rng);
                pack_value(A, m, K, k, code);
                Az[size_t(m) * G + g] += code;
            }
        }
    }
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < G; ++g) {
            Sw[size_t(n) * G + g] = ws(rng);
            Zw[size_t(n) * G + g] = wz(rng);
            for (int k = g * 128; k < (g + 1) * 128; ++k)
                pack_value(W, n, K, k, wq(rng));
        }
    }

    std::vector<float> expected(size_t(M) * N, 0);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float result = 0;
            for (int g = 0; g < G; ++g) {
                int64_t direct = 0, raw = 0;
                for (int k = g * 128; k < (g + 1) * 128; ++k) {
                    const int ai = unpack_value(A, m, K, k, true);
                    const int wi = unpack_value(W, n, K, k, false);
                    raw += int64_t(ai) * wi;
                    direct += int64_t(ai) * (wi - Zw[size_t(n) * G + g]);
                }
                if (raw - int64_t(Zw[size_t(n) * G + g]) * Az[size_t(m) * G + g] != direct) {
                    std::fprintf(stderr, "XDP04 CPU correction invariant failed\n");
                    return 4;
                }
                result = std::fma(float(direct) * Sa[size_t(m) * G + g],
                                  Sw[size_t(n) * G + g], result);
            }
            expected[size_t(m) * N + n] = result;
        }
    }

    unsigned char *dA = nullptr, *dW = nullptr;
    float *dSa = nullptr, *dSw = nullptr, *dC = nullptr;
    int *dAz = nullptr, *dZw = nullptr;
    auto allocate = [](auto** p, size_t bytes) -> bool {
        return check(hipMalloc(reinterpret_cast<void**>(p), bytes), "hipMalloc");
    };
    if (!allocate(&dA, A.size()) || !allocate(&dW, W.size())
            || !allocate(&dSa, Sa.size() * sizeof(float))
            || !allocate(&dSw, Sw.size() * sizeof(float))
            || !allocate(&dAz, Az.size() * sizeof(int))
            || !allocate(&dZw, Zw.size() * sizeof(int))
            || !allocate(&dC, expected.size() * sizeof(float))) return 5;
    if (!check(hipMemcpy(dA, A.data(), A.size(), hipMemcpyHostToDevice), "A copy")
            || !check(hipMemcpy(dW, W.data(), W.size(), hipMemcpyHostToDevice), "W copy")
            || !check(hipMemcpy(dSa, Sa.data(), Sa.size() * sizeof(float), hipMemcpyHostToDevice), "Sa copy")
            || !check(hipMemcpy(dSw, Sw.data(), Sw.size() * sizeof(float), hipMemcpyHostToDevice), "Sw copy")
            || !check(hipMemcpy(dAz, Az.data(), Az.size() * sizeof(int), hipMemcpyHostToDevice), "Az copy")
            || !check(hipMemcpy(dZw, Zw.data(), Zw.size() * sizeof(int), hipMemcpyHostToDevice), "Zw copy")) return 5;
    auto launch = [&]() {
        hipLaunchKernelGGL(native_w4a4_oracle, dim3(N / 16, M / 16), dim3(32), 0, 0,
                           dA, dW, dSa, dSw, dAz, dZw, dC, M, N, K);
    };
    launch();
    if (!check(hipGetLastError(), "kernel launch")
            || !check(hipDeviceSynchronize(), "kernel sync")) return 6;
    std::vector<float> actual(expected.size());
    if (!check(hipMemcpy(actual.data(), dC, actual.size() * sizeof(float),
                         hipMemcpyDeviceToHost), "C copy")) return 6;
    size_t bad = 0;
    double worst = 0, sumerr = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        double delta = std::fabs(double(actual[i]) - expected[i]);
        worst = std::max(worst, delta);
        sumerr += delta;
        if (!std::isfinite(actual[i]) || delta > (0.001 + 0.0002 * std::fabs(expected[i]))) {
            if (bad < 5) std::fprintf(stderr, "XDP04 mismatch row=%zu col=%zu ref=%.7f gpu=%.7f delta=%.7f\n",
                                      i / N, i % N, expected[i], actual[i], delta);
            ++bad;
        }
    }
    if (bad) {
        std::fprintf(stderr, "XDP04_GPU_FAIL bad=%zu/%zu worst=%.7f\n",
                     bad, actual.size(), worst);
        return 7;
    }
    hipEvent_t start{}, stop{};
    if (!check(hipEventCreate(&start), "event start")
            || !check(hipEventCreate(&stop), "event stop")) return 8;
    // Warmup untimed; timings here are an UNOPTIMIZED kernel-only feasibility
    // measurement, with prequantized operands and scales in device memory.
    for (int i = 0; i < 4; ++i) launch();
    if (!check(hipDeviceSynchronize(), "warmup sync")) return 8;
    if (!check(hipEventRecord(start), "start record")) return 8;
    for (int i = 0; i < iters; ++i) launch();
    if (!check(hipEventRecord(stop), "stop record")
            || !check(hipEventSynchronize(stop), "stop sync")) return 8;
    float ms = 0;
    if (!check(hipEventElapsedTime(&ms, start, stop), "elapsed")) return 8;
    std::printf("XDP04_GPU_PASS arch=%s M=%d N=%d K=%d groups=%d iters=%d avg_kernel_us=%.3f worst_abs=%.7g mean_abs=%.7g\n",
                props.gcnArchName, M, N, K, G, iters,
                (double(ms) * 1000.0) / iters, worst, sumerr / actual.size());
    bool cleanup = true;
    cleanup = check(hipEventDestroy(start), "destroy start") && cleanup;
    cleanup = check(hipEventDestroy(stop), "destroy stop") && cleanup;
    cleanup = check(hipFree(dA), "free A") && cleanup;
    cleanup = check(hipFree(dW), "free W") && cleanup;
    cleanup = check(hipFree(dSa), "free Sa") && cleanup;
    cleanup = check(hipFree(dSw), "free Sw") && cleanup;
    cleanup = check(hipFree(dAz), "free Az") && cleanup;
    cleanup = check(hipFree(dZw), "free Zw") && cleanup;
    cleanup = check(hipFree(dC), "free C") && cleanup;
    return cleanup ? 0 : 9;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--suite") == 0) {
        struct Case { int m, n, k; unsigned seed; };
        const Case cases[] = {
            {16, 16, 128, 7}, {16, 32, 256, 8},
            {32, 64, 512, 19}, {64, 128, 1024, 23},
            {128, 256, 256, 29}, {16, 16, 1024, 103},
        };
        for (const auto& c : cases) {
            int rc = run(c.m, c.n, c.k, 25, c.seed);
            if (rc) return rc;
        }
        std::puts("XDP04_GPU_SUITE_PASS cases=6");
        return 0;
    }
    if (argc != 6) {
        std::fprintf(stderr, "usage: gfx1100-w4a4-probe M N K iters seed\n");
        return 2;
    }
    const int m = std::atoi(argv[1]), n = std::atoi(argv[2]),
              k = std::atoi(argv[3]), iters = std::atoi(argv[4]);
    const unsigned seed = static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10));
    return run(m, n, k, iters, seed);
}
