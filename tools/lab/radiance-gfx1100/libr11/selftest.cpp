// HIP runtime smoke test for the two experimental plugin kernels.
// dlopen the built plugin: prove ABI exports, actual launches and BF16 numerical results.
#include "rad_abi.h"
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
#define HIP_CHECK(call) do { auto status = (call); if (status != hipSuccess) { \
    std::fprintf(stderr, "%s failed: %s\n", #call, hipGetErrorString(status)); return 1; \
} } while (0)

using Launch = int (*)(const RadArgs*, RadStream);

int run_case(Launch launch, const uint16_t expected[4], const char* label,
             uint16_t* da, uint16_t* db, uint16_t* dy, hipStream_t stream) {
    const RadTensor tensors[] = {
        {da, RAD_BF16, 2, {2, 2}, {2, 1}},
        {db, RAD_BF16, 2, {1, 2}, {2, 1}},
        {dy, RAD_BF16, 2, {2, 2}, {2, 1}},
    };
    const RadParam params[] = {
        {"M", RAD_P_INT, 2, 0, nullptr, 0.0},
        {"n", RAD_P_INT, 2, 0, nullptr, 0.0},
        {"dtype", RAD_P_STR, 0, 0, "bf16", 0.0},
    };
    RadArgs a{};
    a.t = tensors;
    a.n_t = 3;
    a.p = params;
    a.n_p = 3;
    int rc = launch(&a, reinterpret_cast<RadStream>(stream));
    if (rc != RAD_OK) {
        std::fprintf(stderr, "%s launch error %d\n", label, rc);
        return 1;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) return 1;
    uint16_t got[4]{};
    if (hipMemcpy(got, dy, sizeof(got), hipMemcpyDeviceToHost) != hipSuccess) return 1;
    if (std::memcmp(got, expected, sizeof(got)) != 0) {
        std::fprintf(stderr, "%s mismatch", label);
        for (int i=0; i<4; ++i) std::fprintf(stderr, " %04x/%04x", got[i], expected[i]);
        std::fprintf(stderr, "\n");
        return 1;
    }
    std::printf("PASS %s bf16 broadcast 2x2\n", label);
    return 0;
}
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s /path/to/libr11.so\n", argv[0]);
        return 2;
    }
    void* module = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!module) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    auto version = reinterpret_cast<uint32_t (*)()>(dlsym(module, "rad_plugin_abi_version"));
    auto add = reinterpret_cast<Launch>(dlsym(module, "r11_add_bf16"));
    auto mul = reinterpret_cast<Launch>(dlsym(module, "r11_mul_bf16"));
    if (!version || version() != RAD_ABI_VERSION || !add || !mul) {
        std::fprintf(stderr, "plugin ABI mismatch or missing kernels\n");
        return 1;
    }
    // A=[[1,2],[3,4]], B=[[0.5,-1]]: one-row broadcast.
    const uint16_t ah[] = {0x3f80, 0x4000, 0x4040, 0x4080};
    const uint16_t bh[] = {0x3f00, 0xbf80};
    const uint16_t sum[] = {0x3fc0, 0x3f80, 0x4060, 0x4040};
    const uint16_t prod[] = {0x3f00, 0xc000, 0x3fc0, 0xc080};
    uint16_t *da=nullptr, *db=nullptr, *dy=nullptr;
    hipStream_t stream=nullptr;
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&da), sizeof(ah)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&db), sizeof(bh)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dy), sizeof(ah)));
    HIP_CHECK(hipMemcpy(da, ah, sizeof(ah), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(db, bh, sizeof(bh), hipMemcpyHostToDevice));
    HIP_CHECK(hipStreamCreate(&stream));
    int failed = run_case(add,sum,"add",da,db,dy,stream) |
                 run_case(mul,prod,"mul",da,db,dy,stream);
    hipStreamDestroy(stream);
    hipFree(dy);
    hipFree(db);
    hipFree(da);
    dlclose(module);
    return failed ? 1 : 0;
}
