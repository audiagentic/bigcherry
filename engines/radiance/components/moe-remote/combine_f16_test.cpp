// The combine's f32 -> fp16 conversion (combine_f16.h) against the compiler's _Float16: every finite fp16 value round-trips,
// random floats round identically, values past 65504 report overflow, and a NaN is never the fallback marker.
#include "combine_f16.h"
#include "moe_remote_shm.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

static void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

int main() {
    for (uint32_t h = 0; h < 0x10000; ++h) {
        if (((h >> 10) & 31u) == 31u) continue;
        _Float16 x;
        std::memcpy(&x, &h, 2);
        bool overflow = false;
        check(combine_f32_f16_rn((float)x, &overflow) == h && !overflow, "finite fp16 round-trips");
    }
    uint32_t s = 1;
    for (int i = 0; i < 4000000; ++i) {
        s = s * 1664525u + 1013904223u;
        float f;
        std::memcpy(&f, &s, 4);
        if (!std::isfinite(f) || std::fabs(f) > 65504.0f) continue;
        bool overflow = false;
        const _Float16 x = (_Float16)f;
        uint16_t want;
        std::memcpy(&want, &x, 2);
        check(combine_f32_f16_rn(f, &overflow) == want, "random float rounds as the compiler does");
    }
    bool overflow = false;
    check(combine_f32_f16_rn(65519.0f, &overflow) == 0x7bffu && !overflow, "65519 rounds to 65504");
    check(combine_f32_f16_rn(65520.0f, &overflow) == 0x7c00u && overflow, "65520 overflows");
    check(combine_f32_f16_rn(NAN, &overflow) != r3::moe_shm::kCombineFallback, "NaN is not the fallback marker");
    std::printf("combine fp16 conversion: ok\n");
    return 0;
}
