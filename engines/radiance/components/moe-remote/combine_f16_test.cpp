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
    // the F16C row path (combine_row_f16) writes exactly what the scalar conversion of each fp32 sum would, for every
    // helper mask, a row length that is not a multiple of 8, and rows with and without a value past fp16 range
    if (combine_cpu_supported()) {
        const uint32_t n = 2048 + 5;
        uint16_t a[n], b[n], out[n];
        for (int trial = 0; trial < 200; ++trial) {
            for (uint32_t c = 0; c < n; ++c) {
                s = s * 1664525u + 1013904223u;
                a[c] = (uint16_t)(0x3800u + (s >> 20) % 0x0c00u) | (uint16_t)((s & 1u) << 15);  // |v| in ~[2^-15, 2^9)
                s = s * 1664525u + 1013904223u;
                b[c] = (uint16_t)(0x3800u + (s >> 20) % 0x0c00u) | (uint16_t)((s & 1u) << 15);
            }
            const bool big = trial % 4 == 3;
            if (big) a[trial % n] = 0x4780u;  // 65536: past fp16 range
            const uint8_t mask = (uint8_t)(1 + trial % 3);
            const bool over = combine_row_f16(a, b, mask, out, n);
            bool want_over = false;
            for (uint32_t c = 0; c < n; ++c) {
                const float v = ((mask & 1u) ? combine_bf16_to_f32(a[c]) : 0.0f) + ((mask & 2u) ? combine_bf16_to_f32(b[c]) : 0.0f);
                bool o = false;
                check(out[c] == combine_f32_f16_rn(v, &o), "row path matches the scalar conversion");
                want_over |= std::fabs(v) >= 65520.0f;
            }
            check(over == want_over, "row path reports a value past fp16 range");
        }
    }
    std::printf("combine fp16 conversion: ok\n");
    return 0;
}
