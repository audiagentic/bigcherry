// Host reference for FP8 block-scaled MoE experts (RAD11): the numerical definition a gfx1100 FP8 kernel is checked against, written
// straight from the container's logical layout and independent of any GPU relayout.
//
// A weight matrix W[N, K] is stored as E4M3 codes (one byte per element) with one scale per block_n x block_k tile:
//     w[n, k] = decode_e4m3(code[n, k]) * scale[n / block_n][k / block_k]
// An expert is gate+up fused (rows 0..I-1 gate, I..2I-1 up) over hidden K = H, then down [H, I]:
//     h = silu(Wg x) * (Wu x);   y = Wd h
// Everything is float; this is the reference, not a fast path. Header-only, no HIP, no radiance headers.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace r3::moe_fp8_ref {

// OCP E4M3FN: 1 sign, 4 exponent (bias 7), 3 mantissa bits; no infinities; exponent 15 with mantissa 7 is NaN; exponent 0 is subnormal.
inline float decode_e4m3(uint8_t b) {
    const bool neg = b & 0x80;
    const int e = (b >> 3) & 0xF, m = b & 7;
    if (e == 0xF && m == 7) return std::nanf("");
    const float v = e == 0 ? std::ldexp((float)m / 8.0f, -6) : std::ldexp(1.0f + (float)m / 8.0f, e - 7);
    return neg ? -v : v;
}

struct Matrix {
    uint64_t rows = 0, cols = 0, block_n = 128, block_k = 128;
    std::vector<uint8_t> code;    // rows * cols, row-major
    std::vector<float> scale;     // ceil(rows / block_n) * ceil(cols / block_k), row-major over tiles
    uint64_t tiles_k() const { return (cols + block_k - 1) / block_k; }
    float weight(uint64_t n, uint64_t k) const {
        return decode_e4m3(code[n * cols + k]) * scale[(n / block_n) * tiles_k() + k / block_k];
    }
};

struct Expert {
    Matrix gate_up;   // [2 * I, H]
    Matrix down;      // [H, I]
};

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

// y[H] = down( silu(gate x) * (up x) ) for one input row x[H].
inline std::vector<float> expert_forward(const Expert& e, const std::vector<float>& x) {
    const uint64_t inter = e.gate_up.rows / 2, hidden = e.gate_up.cols;
    std::vector<float> h(inter);
    for (uint64_t i = 0; i < inter; ++i) {
        float g = 0.f, u = 0.f;
        for (uint64_t k = 0; k < hidden; ++k) {
            g += e.gate_up.weight(i, k) * x[k];
            u += e.gate_up.weight(inter + i, k) * x[k];
        }
        h[i] = silu(g) * u;
    }
    std::vector<float> y(e.down.rows, 0.f);
    for (uint64_t n = 0; n < e.down.rows; ++n)
        for (uint64_t k = 0; k < e.down.cols; ++k) y[n] += e.down.weight(n, k) * h[k];
    return y;
}

// A deterministic synthetic matrix (xorshift codes, never the NaN code; scales a few distinct powers of two) so tests need no model.
inline Matrix synthetic_matrix(uint64_t rows, uint64_t cols, uint64_t block_n, uint64_t block_k, uint64_t seed) {
    Matrix m;
    m.rows = rows; m.cols = cols; m.block_n = block_n; m.block_k = block_k;
    uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1;
    auto next = [&] { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    m.code.resize(rows * cols);
    for (auto& c : m.code) {
        uint8_t b = (uint8_t)(next() & 0xFF);
        if ((b & 0x7F) == 0x7F) b ^= 0x01;                  // 0x7F / 0xFF are NaN
        // keep magnitudes small so a few hundred terms stay well inside f32 and f16 range
        if (((b >> 3) & 0xF) > 9) b = (uint8_t)((b & 0x87) | (9 << 3));
        c = b;
    }
    const uint64_t tn = (rows + block_n - 1) / block_n, tk = (cols + block_k - 1) / block_k;
    m.scale.resize(tn * tk);
    for (auto& v : m.scale) v = std::ldexp(1.0f, (int)(next() % 5) - 6);
    return m;
}

}  // namespace r3::moe_fp8_ref
