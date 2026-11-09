/* XDP06: independent wide-tiled native gfx1100 W4A4 + CPU oracle.
 *
 * DO NOT REGISTER IN RADIANCE. Never use timings as an end-to-end serving score.
 * 8 waves/block: one M32xN64 rectangle per wave, 128x128 output.
 * Cooperatively stage 16KiB of tile-major A4/W4 packed operands in LDS per
 * group of K=128. Exactly 64 native IU4 WMMA instructions per wave/scale group.
 * Store C[M,N] as float32 with unsigned-weight zero correction.
 *
 * Tile-major A/W input layout, 16 rows per tile:
 * tile[(row/16 * G + g)*1024 + step*128 + (row%16)*8 + byte].
 * A signed 4-bit; W unsigned 4-bit. Two copies of each operand row across
 * the two halves of a wave32 (gfx11 native fragment mapping).
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

using vi2 = int __attribute__((ext_vector_type(2)));
using vi8 = int __attribute__((ext_vector_type(8)));

static bool ok(hipError_t e, const char *site) {
    if (e == hipSuccess) return true;
    std::fprintf(stderr, "XDP06_HIP_FAIL %s: %s\n", site, hipGetErrorString(e));
    return false;
}

// XDP06: 128x128 output, eight wave32s; each wave owns M32xN64.
// Group128 signed-A4/unsigned-W4 K16 WMMA, exactly XDP05's arithmetic.
template <bool COALESCED>
__global__ __launch_bounds__(256) void w4a4_wide8(
    const unsigned* __restrict__ aq, const unsigned* __restrict__ wq,
    const float* __restrict__ sa, const float* __restrict__ sw,
    const int* __restrict__ asum, const int* __restrict__ wz,
    float* __restrict__ out, int M, int N, int G) {
    constexpr int STAGE_WORDS = 4096, OUT_STRIDE = 129;
    __shared__ unsigned stage[COALESCED ? 32 * OUT_STRIDE : STAGE_WORDS];
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm_pair = wave >> 1, wn0 = (wave & 1) * 4;
    const int m0 = int(blockIdx.y) * 128 + 32 * wm_pair;
    const int n0 = int(blockIdx.x) * 128 + 16 * wn0;
    float sums[2][4][8] = {};
    for (int g = 0; g < G; ++g) {
#pragma unroll 1
        for (int i = 0; i < STAGE_WORDS / 256; ++i) {
            const int word = tid + i * 256;
            unsigned val = 0;
            if (word < 2048) {
                const int mt = int(blockIdx.y) * 8 + word / 256;
                if (mt * 16 < M)
                    val = aq[(size_t(mt) * G + g) * 256 + word % 256];
            } else {
                const int x = word - 2048;
                const int nt = int(blockIdx.x) * 8 + x / 256;
                if (nt * 16 < N)
                    val = wq[(size_t(nt) * G + g) * 256 + x % 256];
            }
            stage[word] = val;
        }
        __syncthreads();
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            const int m = m0 + 16 * mi + (lane & 15);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                vi8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll 1
                for (int step = 0; step < 8; ++step) {
                    const unsigned* ap = stage + (2 * wm_pair + mi) * 256
                        + step * 32 + (lane & 15) * 2;
                    const unsigned* wp = stage + 2048 + (wn0 + c) * 256
                        + step * 32 + (lane & 15) * 2;
                    const vi2 a = {int(ap[0]), int(ap[1])};
                    const vi2 w = {int(wp[0]), int(wp[1])};
                    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                        false, w, true, a, acc, false);
                }
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int n = n0 + 16 * c + 2 * e + (lane >> 4);
                    if (m < M && n < N) {
                        const int fixed = acc[e] - wz[size_t(n) * G + g]
                                           * asum[size_t(m) * G + g];
                        sums[mi][c][e] = fmaf(
                            float(fixed) * sa[size_t(m) * G + g],
                            sw[size_t(n) * G + g], sums[mi][c][e]);
                    }
                }
            }
        }
        __syncthreads();
    }
    if constexpr (COALESCED) {
        float* tile = reinterpret_cast<float*>(stage);
        for (int slice = 0; slice < 4; ++slice) {
            if (wm_pair == slice) {
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    const int lm = 16 * mi + (lane & 15);
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
#pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            const int ln = 16 * (wn0 + c) + 2 * e + (lane >> 4);
                            tile[lm * OUT_STRIDE + ln] = sums[mi][c][e];
                        }
                    }
                }
            }
            __syncthreads();
#pragma unroll
            for (int t = tid; t < 4096; t += 256) {
                const int lm = t / 128, ln = t % 128;
                const int m = int(blockIdx.y) * 128 + slice * 32 + lm;
                const int n = int(blockIdx.x) * 128 + ln;
                if (m < M && n < N)
                    out[size_t(m) * N + n] = tile[lm * OUT_STRIDE + ln];
            }
            __syncthreads();
        }
    } else {
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            const int m = m0 + 16 * mi + (lane & 15);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int n = n0 + 16 * c + 2 * e + (lane >> 4);
                    if (m < M && n < N)
                        out[size_t(m) * N + n] = sums[mi][c][e];
                }
            }
        }
    }
}

struct Layout {
    static size_t bytes(int rows, int G) {
        return size_t((rows + 15) / 16) * G * 1024;
    }
    static size_t offset(int row, int k, int G) {
        const int g = k / 128, step = (k % 128) / 16;
        return (size_t(row / 16) * G + g) * 1024
            + step * 128 + (row & 15) * 8 + ((k % 16) >> 1);
    }
    static void put(std::vector<unsigned char>& dst, int row, int k,
                    int G, int code) {
        const size_t at = offset(row, k, G);
        const unsigned char nibble = static_cast<unsigned char>(code & 15);
        if (k & 1) dst[at] = static_cast<unsigned char>((dst[at] & 0x0f) | (nibble << 4));
        else       dst[at] = static_cast<unsigned char>((dst[at] & 0xf0) | nibble);
    }
    static int get(const std::vector<unsigned char>& src, int row, int k,
                   int G, bool signed4) {
        const unsigned char packed = src[offset(row, k, G)];
        int code = (packed >> (4 * (k & 1))) & 15;
        return signed4 && code >= 8 ? code - 16 : code;
    }
};

struct Host {
    int M, N, K, G;
    std::vector<unsigned char> A, W;
    std::vector<float> Sa, Sw;
    std::vector<int> Asum, Wzero;
    explicit Host(int m, int n, int k, unsigned seed) :
        M(m), N(n), K(k), G(k / 128),
        A(Layout::bytes(m, G), 0), W(Layout::bytes(n, G), 0),
        Sa(size_t(m) * G), Sw(size_t(n) * G),
        Asum(size_t(m) * G, 0), Wzero(size_t(n) * G) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> aq(-7, 7), wq(0, 15), wz(2, 13);
        std::uniform_real_distribution<float> sa(0.008f, 0.045f), sw(0.01f, 0.09f);
        for (int mrow = 0; mrow < M; ++mrow)
            for (int g = 0; g < G; ++g) {
                Sa[size_t(mrow) * G + g] = sa(rng);
                for (int j = 0; j < 128; ++j) {
                    int v = aq(rng);
                    Layout::put(A, mrow, g * 128 + j, G, v);
                    Asum[size_t(mrow) * G + g] += v;
                }
            }
        for (int nrow = 0; nrow < N; ++nrow)
            for (int g = 0; g < G; ++g) {
                Sw[size_t(nrow) * G + g] = sw(rng);
                Wzero[size_t(nrow) * G + g] = wz(rng);
                for (int j = 0; j < 128; ++j)
                    Layout::put(W, nrow, g * 128 + j, G, wq(rng));
            }
    }
    float oracle(int m, int n) const {
        float result = 0;
        for (int g = 0; g < G; ++g) {
            int64_t sum = 0, raw = 0;
            for (int j = 0; j < 128; ++j) {
                const int k = g * 128 + j;
                const int a = Layout::get(A, m, k, G, true);
                const int w = Layout::get(W, n, k, G, false);
                raw += int64_t(a) * w;
                sum += int64_t(a) * (w - Wzero[size_t(n) * G + g]);
            }
            if (raw - int64_t(Wzero[size_t(n) * G + g]) * Asum[size_t(m) * G + g] != sum) {
                std::fprintf(stderr, "XDP06_CPU_FAIL correction invariant\n");
                std::abort();
            }
            result = std::fma(float(sum) * Sa[size_t(m) * G + g],
                              Sw[size_t(n) * G + g], result);
        }
        return result;
    }
};

struct Device {
    unsigned *A = nullptr, *W = nullptr;
    float *Sa = nullptr, *Sw = nullptr, *C = nullptr;
    int *Asum = nullptr, *Wzero = nullptr;
    bool alloc_and_copy(const Host &h) {
        auto xfer = [](auto** d, const auto& src, const char* site) -> bool {
            if (!ok(hipMalloc(reinterpret_cast<void**>(d), src.size() *
                    sizeof(typename std::decay_t<decltype(src)>::value_type)), site))
                return false;
            return ok(hipMemcpy(*d, src.data(), src.size() *
                      sizeof(typename std::decay_t<decltype(src)>::value_type),
                      hipMemcpyHostToDevice), site);
        };
        // HIP code reads uint32 for coalesced packed operands. Host
        // byte vectors have exact 4B multiples and are cast on GPU.
        if (!xfer(&A, h.A, "A") || !xfer(&W, h.W, "W")
            || !xfer(&Sa, h.Sa, "Sa") || !xfer(&Sw, h.Sw, "Sw")
            || !xfer(&Asum, h.Asum, "Asum") || !xfer(&Wzero, h.Wzero, "Wzero"))
            return false;
        return ok(hipMalloc(&C, size_t(h.M) * h.N * sizeof(float)), "C");
    }
    ~Device() {
        if (A) (void)hipFree(A);
        if (W) (void)hipFree(W);
        if (Sa) (void)hipFree(Sa);
        if (Sw) (void)hipFree(Sw);
        if (Asum) (void)hipFree(Asum);
        if (Wzero) (void)hipFree(Wzero);
        if (C) (void)hipFree(C);
    }
};

static int one(int M, int N, int K, int iters, unsigned seed, bool full,
               bool coalesced) {
    if (M < 1 || M > 512 || N < 1 || N > 17408 || K < 128 ||
        K > 5120 || K % 128 || iters < 1 || iters > 1000) {
        std::fprintf(stderr, "XDP06_INVALID_SHAPE M=%d N=%d K=%d\n", M, N, K);
        return 2;
    }
    hipDeviceProp_t props{};
    if (!ok(hipGetDeviceProperties(&props, 0), "device props")) return 3;
    if (std::strncmp(props.gcnArchName, "gfx1100", 7)) {
        std::fprintf(stderr, "XDP06_WRONG_GPU %s\n", props.gcnArchName);
        return 3;
    }
    Host h(M, N, K, seed);
    Device d;
    if (!d.alloc_and_copy(h)) return 4;
    const dim3 grid((N + 127) / 128, (M + 127) / 128), block(256);
    auto launch = [&] {
        if (coalesced)
            hipLaunchKernelGGL((w4a4_wide8<true>), grid, block, 0, 0,
                              d.A, d.W, d.Sa, d.Sw, d.Asum, d.Wzero, d.C,
                              M, N, h.G);
        else
            hipLaunchKernelGGL((w4a4_wide8<false>), grid, block, 0, 0,
                              d.A, d.W, d.Sa, d.Sw, d.Asum, d.Wzero, d.C,
                              M, N, h.G);
    };
    launch();
    if (!ok(hipGetLastError(), "launch") || !ok(hipDeviceSynchronize(), "sync"))
        return 5;
    std::vector<float> output(size_t(M) * N);
    if (!ok(hipMemcpy(output.data(), d.C, output.size() * sizeof(float),
                      hipMemcpyDeviceToHost), "C D2H")) return 5;
    for (size_t i = 0; i < output.size(); ++i) {
        if (!std::isfinite(output[i])) {
            std::fprintf(stderr, "XDP06_NONFINITE index=%zu\n", i);
            return 6;
        }
    }
    std::vector<size_t> checked;
    checked.reserve(1024);
    // Full oracle on smaller cases. Exact-shaped 27B uses 512+ samples
    // spanning tile interiors/borders and explicitly both extreme rows.
    if (full && size_t(M) * N <= 65536) {
        for (size_t i = 0; i < output.size(); ++i) checked.push_back(i);
    } else {
        auto sample = [&](int m, int n) {
            if (m >= 0 && m < M && n >= 0 && n < N)
                checked.push_back(size_t(m) * N + n);
        };
        const int special_m[] = {0, 1, 15, 16, 17, 31, 32, 33, 63, 64,
                                 M / 2, M - 33, M - 32, M - 17, M - 16, M - 1};
        const int special_n[] = {0, 1, 15, 16, 17, 31, 32, 33, 63, 64,
                                 N / 2, N - 65, N - 64, N - 33, N - 32, N - 1};
        for (int m : special_m)
            for (int n : special_n) sample(m, n);
        std::mt19937 rng(seed ^ 0x34A45u);
        for (int i = 0; i < 512; ++i)
            sample(int(rng() % unsigned(M)), int(rng() % unsigned(N)));
        std::sort(checked.begin(), checked.end());
        checked.erase(std::unique(checked.begin(), checked.end()), checked.end());
    }
    size_t mismatch = 0;
    double worst = 0, avg = 0;
    for (size_t pos : checked) {
        const int m = int(pos / N), n = int(pos % N);
        const float expected = h.oracle(m, n), actual = output[pos];
        const double delta = std::fabs(double(expected) - actual);
        worst = std::max(worst, delta);
        avg += delta;
        if (delta > 0.00025 + 0.00005 * std::fabs(expected)) {
            if (mismatch < 8)
                std::fprintf(stderr, "XDP06_MISMATCH m=%d n=%d ref=%g gpu=%g err=%.7g\n",
                             m, n, expected, actual, delta);
            ++mismatch;
        }
    }
    if (mismatch) {
        std::fprintf(stderr, "XDP06_GPU_FAIL mismatches=%zu checked=%zu worst=%g\n",
                     mismatch, checked.size(), worst);
        return 7;
    }

    // Timing is kernel-only. A4 quantizer, layout packing and memcpy
    // occur outside GPU events. Repeated event rounds avoid one-run winner.
    hipEvent_t start{}, stop{};
    if (!ok(hipEventCreate(&start), "event create")
        || !ok(hipEventCreate(&stop), "event create")) return 8;
    std::vector<double> timing;
    for (int round = 0; round < 3; ++round) {
        for (int i = 0; i < 4; ++i) launch();
        if (!ok(hipDeviceSynchronize(), "warmup")) return 8;
        if (!ok(hipEventRecord(start), "record start")) return 8;
        for (int i = 0; i < iters; ++i) launch();
        if (!ok(hipEventRecord(stop), "record stop")
            || !ok(hipEventSynchronize(stop), "event sync")) return 8;
        float milliseconds = 0;
        if (!ok(hipEventElapsedTime(&milliseconds, start, stop), "elapsed"))
            return 8;
        timing.push_back(1000.0 * milliseconds / iters);
    }
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    std::sort(timing.begin(), timing.end());
    const double ops = 2.0 * double(M) * double(N) * double(K);
    std::printf("XDP06_WIDE8_PASS arch=%s tile=128x128 waves=8 store=%s M=%d N=%d K=%d groups=%d checked=%zu/%zu worst_abs=%g mean_abs=%g iters=%d min_us=%.3f median_us=%.3f max_us=%.3f tops_median=%.3f prepacked=1 gpu_a4_quant=0\n",
        props.gcnArchName, coalesced ? "coalesced" : "direct",
        M, N, K, h.G, checked.size(), output.size(),
        worst, avg / checked.size(), iters, timing[0], timing[1],
        timing[2], ops / (timing[1] * 1.0e6));
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--suite") == 0) {
        struct Shape { int m, n, k; unsigned seed; };
        const Shape shapes[] = {
            {16, 16, 128, 7}, {16, 32, 256, 8}, {32, 64, 512, 19},
            {64, 128, 1024, 23}, {128, 256, 256, 29},
            {16, 16, 1024, 103}, {17, 65, 384, 61},
            {33, 63, 512, 67}, {1, 1, 128, 71},
            {129, 129, 256, 79}, {128, 257, 512, 83},
            {257, 128, 128, 89}
        };
        for (const auto& shape : shapes) {
            for (bool coalesced : {false, true}) {
                int rc = one(shape.m, shape.n, shape.k, 20, shape.seed,
                             true, coalesced);
                if (rc) return rc;
            }
        }
        std::puts("XDP06_WIDE8_SUITE_PASS cases=24");
        return 0;
    }
    if (argc >= 2 && std::strcmp(argv[1], "--shape-27b") == 0) {
        if (argc != 3) return 2;
        // One lease, same deterministic Host seed and shape in both modes.
        // This is sequential A/B, NOT a fully interleaved temperature A/B.
        if (std::strcmp(argv[2], "paired") == 0) {
            int rc = one(512, 17408, 5120, 15, 73, false, false);
            if (rc) return rc;
            rc = one(512, 17408, 5120, 15, 73, false, true);
            if (rc) return rc;
            std::puts("XDP06_WIDE8_PAIRED_PASS modes=2 M=512 N=17408 K=5120");
            return 0;
        }
        if (std::strcmp(argv[2], "direct") != 0 &&
            std::strcmp(argv[2], "coalesced") != 0) return 2;
        return one(512, 17408, 5120, 15, 73, false,
                   std::strcmp(argv[2], "coalesced") == 0);
    }
    if (argc != 7 ||
        (std::strcmp(argv[6], "direct") != 0 &&
         std::strcmp(argv[6], "coalesced") != 0)) {
        std::fprintf(stderr, "usage: wide8-probe --suite | --shape-27b direct|coalesced | M N K iters seed direct|coalesced\\n");
        return 2;
    }
    return one(std::atoi(argv[1]), std::atoi(argv[2]),
               std::atoi(argv[3]), std::atoi(argv[4]),
               unsigned(std::strtoul(argv[5], nullptr, 10)), true,
               std::strcmp(argv[6], "coalesced") == 0);
}
