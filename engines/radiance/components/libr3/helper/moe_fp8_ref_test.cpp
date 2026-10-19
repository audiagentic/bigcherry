// Offline test of the FP8 block-scaled reference (moe_fp8_ref.h): exact E4M3 decode, exact tile indexing at the edges, and the expert
// forward against an independent per-element computation. No GPU, no model.
#include "moe_fp8_ref.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "r3_moe_fp8_ref_test: FAILED %s\n", what); std::exit(1); }
}

using namespace r3::moe_fp8_ref;

int main() {
    // E4M3FN decode: exact values, every one representable in float
    check(decode_e4m3(0x00) == 0.0f, "+0");
    check(std::signbit(decode_e4m3(0x80)) && decode_e4m3(0x80) == 0.0f, "-0");
    check(decode_e4m3(0x01) == std::ldexp(1.0f, -9), "smallest subnormal is 2^-9");
    check(decode_e4m3(0x07) == std::ldexp(7.0f, -9), "largest subnormal is 7 * 2^-9");
    check(decode_e4m3(0x08) == std::ldexp(1.0f, -6), "smallest normal is 2^-6");
    check(decode_e4m3(0x38) == 1.0f, "exponent 7 mantissa 0 is 1.0");
    check(decode_e4m3(0x3C) == 1.5f, "1.5");
    check(decode_e4m3(0x7E) == 448.0f, "largest finite is 448");
    check(decode_e4m3(0xFE) == -448.0f, "smallest finite is -448");
    check(std::isnan(decode_e4m3(0x7F)) && std::isnan(decode_e4m3(0xFF)), "exponent 15 mantissa 7 is NaN");
    for (int b = 0; b < 256; ++b) {                                  // no infinities; only 0x7F and 0xFF are NaN
        const float v = decode_e4m3((uint8_t)b);
        check(!std::isinf(v), "E4M3FN has no infinities");
        check(std::isnan(v) == ((b & 0x7F) == 0x7F), "NaN only at 0x7F and 0xFF");
    }

    // tile indexing at the edges: 130 x 130 with 128 x 128 tiles has 2 x 2 scales
    Matrix m;
    m.rows = 130; m.cols = 130; m.block_n = 128; m.block_k = 128;
    m.code.assign(130 * 130, 0x38);                                  // every weight 1.0 before scaling
    m.scale = {1.0f, 2.0f, 4.0f, 8.0f};
    check(m.tiles_k() == 2, "two tile columns");
    check(m.weight(0, 0) == 1.0f && m.weight(127, 127) == 1.0f, "last element of tile (0,0)");
    check(m.weight(0, 128) == 2.0f && m.weight(127, 129) == 2.0f, "first columns of tile (0,1)");
    check(m.weight(128, 0) == 4.0f && m.weight(129, 127) == 4.0f, "first rows of tile (1,0)");
    check(m.weight(128, 128) == 8.0f && m.weight(129, 129) == 8.0f, "corner tile (1,1)");

    // expert forward against an independent per-element computation (no shared helper but decode_e4m3 and the formula)
    const uint64_t H = 160, I = 96;                                  // not multiples of 128: partial edge tiles
    Expert ex{synthetic_matrix(2 * I, H, 128, 128, 1), synthetic_matrix(H, I, 128, 128, 2)};
    std::vector<float> x(H);
    for (uint64_t k = 0; k < H; ++k) x[k] = (float)((int)(k * 37 % 19) - 9) / 16.0f;
    const std::vector<float> y = expert_forward(ex, x);
    check(y.size() == H, "output width is hidden");
    std::vector<double> h(I);
    for (uint64_t i = 0; i < I; ++i) {
        double g = 0, u = 0;
        for (uint64_t k = 0; k < H; ++k) {
            g += (double)decode_e4m3(ex.gate_up.code[i * H + k]) * ex.gate_up.scale[(i / 128) * 2 + k / 128] * x[k];
            u += (double)decode_e4m3(ex.gate_up.code[(I + i) * H + k]) * ex.gate_up.scale[((I + i) / 128) * 2 + k / 128] * x[k];
        }
        h[i] = g / (1.0 + std::exp(-g)) * u;
    }
    double worst = 0, mag = 0;
    for (uint64_t n = 0; n < H; ++n) {
        double acc = 0;
        for (uint64_t k = 0; k < I; ++k)
            acc += (double)decode_e4m3(ex.down.code[n * I + k]) * ex.down.scale[(n / 128) * 1 + k / 128] * h[k];
        worst = std::fmax(worst, std::fabs(acc - y[n]));
        mag = std::fmax(mag, std::fabs(acc));
    }
    check(mag > 0, "the synthetic expert produces a nonzero output");
    check(worst <= 1e-4 * mag, "float reference agrees with the double computation");

    // synthetic generation is deterministic and never produces NaN codes
    const Matrix a = synthetic_matrix(40, 40, 16, 16, 7), b = synthetic_matrix(40, 40, 16, 16, 7);
    check(a.code == b.code && a.scale == b.scale, "same seed, same matrix");
    for (uint8_t c : a.code) check(!std::isnan(decode_e4m3(c)), "no NaN codes");

    std::puts("r3_moe_fp8_ref_test: ok");
    return 0;
}
