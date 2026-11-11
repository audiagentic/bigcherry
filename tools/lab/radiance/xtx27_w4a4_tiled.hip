/* XDP05: experimental tiled gfx1100 W4A4 WMMA + independent host oracle.
 *
 * DO NOT REGISTER IN RADIANCE. Never use timings as an end-to-end serving score.
 * 8 waves/block = two 16-row activation tiles x four 16-row weight tiles.
 * Cooperatively stage 6KiB of tile-major A4/W4 packed operands in LDS per
 * group of K=128. Exactly 8 native iu4 WMMA instructions per scale group.
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
    std::fprintf(stderr, "XDP05_HIP_FAIL %s: %s\n", site, hipGetErrorString(e));
    return false;
}

template <int WM, int WN>
__global__ __launch_bounds__(256) void w4a4_tiled(
    const unsigned* __restrict__ aq, const unsigned* __restrict__ wq,
    const float* __restrict__ sa, const float* __restrict__ sw,
    const int* __restrict__ asum, const int* __restrict__ wz,
    float* __restrict__ out, int M, int N, int G) {
    // Cooperatively stage (WM+WN) x 16 rows of 64 packed bytes.
    static_assert(WM * WN == 8 && (WM == 2 || WM == 4 || WM == 8));
    constexpr int WORDS = (WM + WN) * 256;
    __shared__ unsigned stage[WORDS];
    const int tid = threadIdx.x;
    const int wave = tid >> 5, lane = tid & 31;
    const int wm = wave / WN, wn = wave % WN;
    const int m = int(blockIdx.y) * (16 * WM) + wm * 16 + (lane & 15);
    const int ntile = int(blockIdx.x) * (16 * WN) + wn * 16;
    float sums[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int g = 0; g < G; ++g) {
        // 2x4 and 4x2 stage six dwords per thread, 8x1 stages nine.
#pragma unroll
        for (int iter = 0; iter < WORDS / 256; ++iter) {
            int word = tid + iter * 256;
            unsigned value = 0;
            if (word < WM * 256) {
                int mtile = int(blockIdx.y) * WM + word / 256;
                if (mtile * 16 < M)
                    value = aq[(size_t(mtile) * G + g) * 256 + (word % 256)];
            } else {
                int wword = word - WM * 256;
                int wtile = int(blockIdx.x) * WN + wword / 256;
                if (wtile * 16 < N)
                    value = wq[(size_t(wtile) * G + g) * 256 + (wword % 256)];
            }
            stage[word] = value;
        }
        __syncthreads();
        vi8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int step = 0; step < 8; ++step) {
            // Index: subtile 256 dwords + step 32 dwords +
            // duplicated row 2 dwords.
            const unsigned* ap = stage + wm * 256 + step * 32 + (lane & 15) * 2;
            const unsigned* wp = stage + WM * 256 + wn * 256 + step * 32 + (lane & 15) * 2;
            vi2 a = {int(ap[0]), int(ap[1])};
            vi2 w = {int(wp[0]), int(wp[1])};
            acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                false, w, true, a, acc, false);
        }
        // Native gfx11 C is W rows x A rows. C lanes:
        // n = n_tile+2*e+(lane>>4), m = m_tile+(lane&15).
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int n = ntile + 2 * e + (lane >> 4);
            if (m < M && n < N) {
                const int adjusted = acc[e] - wz[size_t(n) * G + g] * asum[size_t(m) * G + g];
                sums[e] = fmaf(float(adjusted) * sa[size_t(m) * G + g],
                               sw[size_t(n) * G + g], sums[e]);
            }
        }
        // No wave may overwrite the packed group before others consume it.
        __syncthreads();
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int n = ntile + 2 * e + (lane >> 4);
        if (m < M && n < N) out[size_t(m) * N + n] = sums[e];
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
                std::fprintf(stderr, "XDP05_CPU_FAIL correction invariant\n");
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
               int tile_m) {
    if (M < 1 || M > 512 || N < 1 || N > 17408 || K < 128 ||
        K > 5120 || K % 128 || iters < 1 || iters > 1000) {
        std::fprintf(stderr, "XDP05_INVALID_SHAPE M=%d N=%d K=%d\n", M, N, K);
        return 2;
    }
    hipDeviceProp_t props{};
    if (!ok(hipGetDeviceProperties(&props, 0), "device props")) return 3;
    if (std::strncmp(props.gcnArchName, "gfx1100", 7)) {
        std::fprintf(stderr, "XDP05_WRONG_GPU %s\n", props.gcnArchName);
        return 3;
    }
    Host h(M, N, K, seed);
    Device d;
    if (!d.alloc_and_copy(h)) return 4;
    const int tile_n = 8 / tile_m;
    const dim3 grid((N + 16 * tile_n - 1) / (16 * tile_n),
                    (M + 16 * tile_m - 1) / (16 * tile_m)), block(256);
    auto launch = [&] {
        if (tile_m == 2)
            hipLaunchKernelGGL((w4a4_tiled<2, 4>), grid, block, 0, 0,
                              d.A, d.W, d.Sa, d.Sw, d.Asum, d.Wzero, d.C, M, N, h.G);
        else if (tile_m == 4)
            hipLaunchKernelGGL((w4a4_tiled<4, 2>), grid, block, 0, 0,
                              d.A, d.W, d.Sa, d.Sw, d.Asum, d.Wzero, d.C, M, N, h.G);
        else
            hipLaunchKernelGGL((w4a4_tiled<8, 1>), grid, block, 0, 0,
                              d.A, d.W, d.Sa, d.Sw, d.Asum, d.Wzero, d.C, M, N, h.G);
    };
    launch();
    if (!ok(hipGetLastError(), "launch") || !ok(hipDeviceSynchronize(), "sync"))
        return 5;
    std::vector<float> output(size_t(M) * N);
    if (!ok(hipMemcpy(output.data(), d.C, output.size() * sizeof(float),
                      hipMemcpyDeviceToHost), "C D2H")) return 5;
    for (size_t i = 0; i < output.size(); ++i) {
        if (!std::isfinite(output[i])) {
            std::fprintf(stderr, "XDP05_NONFINITE index=%zu\n", i);
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
                std::fprintf(stderr, "XDP05_MISMATCH m=%d n=%d ref=%g gpu=%g err=%.7g\n",
                             m, n, expected, actual, delta);
            ++mismatch;
        }
    }
    if (mismatch) {
        std::fprintf(stderr, "XDP05_GPU_FAIL mismatches=%zu checked=%zu worst=%g\n",
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
    std::printf("XDP05_TILED_PASS arch=%s tile=%dx%d M=%d N=%d K=%d groups=%d checked=%zu/%zu worst_abs=%g mean_abs=%g iters=%d min_us=%.3f median_us=%.3f max_us=%.3f tops_median=%.3f prepacked=1 gpu_a4_quant=0\n",
        props.gcnArchName, tile_m, tile_n, M, N, K, h.G, checked.size(), output.size(),
        worst, avg / checked.size(), iters, timing[0], timing[1],
        timing[2], ops / (timing[1] * 1.0e6));
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--suite") == 0) {
        struct Shape { int m, n, k; unsigned seed; };
        const Shape shapes[] = {
            {16, 16, 128, 7}, {16, 32, 256, 8}, {32, 64, 512, 19},
            {64, 128, 1024, 23}, {128, 256, 256, 29}, {16, 16, 1024, 103},
            {17, 65, 384, 61}, {33, 63, 512, 67}, {1, 1, 128, 71}
        };
        for (const auto& s : shapes) {
            for (int tm : {2, 4, 8}) {
                int rc = one(s.m, s.n, s.k, 20, s.seed, true, tm);
                if (rc) return rc;
            }
        }
        std::puts("XDP05_TILED_SUITE_PASS cases=27");
        return 0;
    }
    if (argc >= 2 && std::strcmp(argv[1], "--shape-27b") == 0) {
        if (argc > 3) return 2;
        const int tm = argc == 3 ? std::atoi(argv[2]) : 2;
        if (tm != 2 && tm != 4 && tm != 8) return 2;
        return one(512, 17408, 5120, 15, 73, false, tm);
    }
    if (argc != 6 && argc != 7) {
        std::fprintf(stderr, "usage: tiled-probe --suite | --shape-27b [tile_m] | M N K iters seed [tile_m]\\n");
        return 2;
    }
    const int tm = argc == 7 ? std::atoi(argv[6]) : 2;
    if (tm != 2 && tm != 4 && tm != 8) return 2;
    return one(std::atoi(argv[1]), std::atoi(argv[2]),
               std::atoi(argv[3]), std::atoi(argv[4]),
               unsigned(std::strtoul(argv[5], nullptr, 10)), true, tm);
}
