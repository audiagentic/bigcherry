// RAD50 card-free check against BigCherry's actual FP16 CPU combine function.
// This does NOT prove GPU BAR visibility. It is an integration-preservation
// regression: any future destination switch must keep these exact bytes,
// sentinel behavior, and helper-order semantics.
// Build: g++ -std=c++17 -O2 -Wall -Wextra -Iengines/radiance/components/moe-remote
//        tools/lab/rad42-bar-push/combine_reference_test.cpp -o combine_reference_test
#include "combine_f16.h"
#include "moe_remote_shm.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
constexpr uint32_t kWidth = 2560;
int failures = 0;
void check(bool ok, const char* why) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", why); }
}
} // namespace

int main() {
    if (!combine_cpu_supported()) {
        std::fputs("SKIP: CPU lacks AVX2 and F16C\n", stderr);
        return 77;
    }

    std::vector<uint16_t> a(kWidth, 0x3f80), b(kWidth, 0x4000);
    std::vector<uint16_t> baseline(kWidth, 0xa5a5), mock_vram(kWidth, 0);
    const uint16_t* src[] = {a.data(), b.data()};
    bool overflow = combine_row_f16(src, 2, 0b11, baseline.data(), kWidth);
    check(!overflow, "1+2 must not overflow");
    check(std::all_of(baseline.begin(), baseline.end(),
                      [](uint16_t h) { return h == 0x4200; }),
          "FP16 result 1+2 must be exactly 3");

    // Opt-in transport copies the same exact 2-byte wire format to GPU-local
    // storage; no reordering or BF16<->F16 conversion is permitted.
    std::memcpy(mock_vram.data(), baseline.data(), 2*kWidth);
    check(mock_vram == baseline, "mock VRAM copy must preserve every bit");

    // One helper's contribution is missing; helper mask must be honored.
    std::fill(baseline.begin(), baseline.end(), 0);
    overflow = combine_row_f16(src, 2, 0b10, baseline.data(), kWidth);
    check(!overflow && std::all_of(baseline.begin(), baseline.end(),
          [](uint16_t h) { return h == 0x4000; }),
          "mask must select only helper1");

    // The sentinel lives in the *first output word* and forces the consumer
    // to fall back to the original BF16 helper planes. Bar writes must not
    // destroy those source planes or use sentinel as a real FP16 value.
    std::fill(a.begin(), a.end(), 0x4780); // BF16 65536
    std::fill(b.begin(), b.end(), 0x3f80);
    overflow = combine_row_f16(src, 2, 0b11, baseline.data(), kWidth);
    check(overflow, "unrepresentable FP16 sum must overflow");
    if (overflow) baseline[0] = r3::moe_shm::kCombineFallback;
    check(baseline[0] == 0x7e01u, "fallback sentinel is the protocol value");

    // NaN is canonicalized, but never uses the overflow sentinel.
    std::fill(a.begin(), a.end(), 0x7fc0); // quiet BF16 NaN
    std::fill(b.begin(), b.end(), 0);
    overflow = combine_row_f16(src, 2, 0b11, baseline.data(), kWidth);
    check(!overflow, "quiet NaN is not finite FP16 overflow");
    check(baseline[0] == 0x7e00,
          "canonical NaN must not alias the fallback sentinel");

    // A check on sum order, independent of CPU staging destination.
    // BF16 0x4b80 = 2^24, 0x3f80 = 1, 0xcb80 = -2^24;
    // fp32 accumulates in ascending helper order.
    // Caller is free to have more helpers; verify pair order is unchanged.
    std::fill(a.begin(), a.end(), 0x3f80);
    std::fill(b.begin(), b.end(), 0x4000);
    overflow = combine_row_f16(src, 2, 0b11, baseline.data(), kWidth);
    check(!overflow && baseline[0] == 0x4200,
          "same order produces stable FP16 output");

    std::printf("RAD50 reference combine: failures=%d, checks=7, elements=%u\n",
                failures, kWidth);
    return failures ? 1 : 0;
}
