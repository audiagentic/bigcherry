#include "r3_ngram_api.h"
#include "rad_plugin.h"

#include <dlfcn.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct Api {
    void* h = nullptr;
    decltype(&r3_ngram_helper_abi) abi = nullptr;
    decltype(&r3_ngram_helper_prepare) prepare = nullptr;
    decltype(&r3_ngram_helper_load) load = nullptr;
    decltype(&r3_ngram_helper_gather) gather = nullptr;
    decltype(&r3_ngram_helper_release) release = nullptr;
};

static bool open_api(const char* so, Api* a) {
    a->h = dlopen(so, RTLD_NOW | RTLD_LOCAL);
    if (!a->h) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return false; }
#define SYM(n) a->n = reinterpret_cast<decltype(a->n)>(dlsym(a->h, "r3_ngram_helper_" #n))
    SYM(abi); SYM(prepare); SYM(load); SYM(gather); SYM(release);
#undef SYM
    return a->abi && a->abi() == R3_NGRAM_HELPER_ABI && a->prepare && a->load && a->gather && a->release;
}

static uint16_t ref(uint8_t q, float scale) {
    return rad_f32_to_bf16(rad_fp8e4m3_to_f32(q) * scale);
}

static int selftest(Api& a, int dev) {
    constexpr int64_t V = 4096, FIRST = 1024, ROWS = 1024;
    constexpr int N = 160;
    const float scale = rad_bf16_to_f32(rad_f32_to_bf16(1.9932e-4f));
    std::vector<uint8_t> table((size_t)V * N);
    for (int64_t r = 0; r < V; ++r)
        for (int c = 0; c < N; ++c)
            table[(size_t)r * N + c] = (uint8_t)((r * 17 + c * 13) & 0x7e);

    const int32_t ids[] = {1023, 1024, 1025, 1536, 2047, 2048};
    constexpr int M = (int)(sizeof(ids) / sizeof(ids[0]));
    std::vector<uint16_t> got((size_t)M * N), want((size_t)M * N);

    int rc = a.prepare(dev, FIRST, ROWS, N, M);
    if (rc != RAD_OK) return rc;
    rc = a.load(dev, table.data() + (size_t)FIRST * N, ROWS * N);
    if (rc != RAD_OK) return rc;
    rc = a.gather(dev, ids, M, 0, scale, got.data());
    if (rc != RAD_OK) return rc;

    for (int m = 0; m < M; ++m) {
        const bool helper = ids[m] >= FIRST && ids[m] < FIRST + ROWS;
        for (int c = 0; c < N; ++c) {
            const uint16_t v = ref(table[(size_t)ids[m] * N + c], scale);
            want[(size_t)m * N + c] = v;
            if (!helper) got[(size_t)m * N + c] = v;  // the host wrapper's libavx miss merge
        }
    }
    size_t bad = 0;
    for (size_t i = 0; i < got.size(); ++i) bad += got[i] != want[i];
    a.release(dev);
    std::printf("R3_NGRAM_SELFTEST boundary_rows=%d bad=%zu/%zu\n", M, bad, got.size());
    return bad ? 1 : 0;
}

static int perf(Api& a, int dev, int M, int iters) {
    constexpr int64_t FIRST = 0, ROWS = 1 << 20;
    constexpr int N = 160;
    const float scale = rad_bf16_to_f32(rad_f32_to_bf16(1.9932e-4f));
    std::vector<uint8_t> table((size_t)ROWS * N);
    for (size_t i = 0; i < table.size(); ++i) table[i] = (uint8_t)(i & 0x7e);
    std::vector<int32_t> ids((size_t)M);
    std::vector<uint16_t> out((size_t)M * N);
    for (int i = 0; i < M; ++i) ids[i] = (int32_t)((uint64_t)i * 104729u % ROWS);

    int rc = a.prepare(dev, FIRST, ROWS, N, M);
    if (rc != RAD_OK) return rc;
    rc = a.load(dev, table.data(), (int64_t)table.size());
    if (rc != RAD_OK) return rc;
    for (int i = 0; i < 3; ++i) if ((rc = a.gather(dev, ids.data(), M, 0, scale, out.data())) != RAD_OK) return rc;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) if ((rc = a.gather(dev, ids.data(), M, 0, scale, out.data())) != RAD_OK) return rc;
    const auto t1 = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    std::printf("R3_NGRAM_PERF M=%d iters=%d us=%.2f rows_per_s=%.0f\n",
                M, iters, us, (double)M * 1e6 / us);
    a.release(dev);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <helper.so> --selftest|--perf <device> [M [iters]]\n", argv[0]);
        return 2;
    }
    Api a;
    if (!open_api(argv[1], &a)) return 2;
    const int dev = std::atoi(argv[3]);
    if (!std::strcmp(argv[2], "--selftest")) return selftest(a, dev);
    if (!std::strcmp(argv[2], "--perf") && argc >= 5)
        return perf(a, dev, std::atoi(argv[4]), argc >= 6 ? std::atoi(argv[5]) : 20);
    return 2;
}
