// r3_e4m3_f16.h against an independent host reference, exhaustively. Card-free: plain C++17, no HIP.
//
// The reference decodes each of the 256 E4M3 codes to a double from the format definition and encodes that double
// to binary16 with an encoder that insists the value is exactly representable. Nothing from the header under test
// is used to build it.
//
// The existing emulation (r3_e4m3_to_f32 in r3_compat.h) cannot be included here, because r3_compat.h pulls in the
// HIP runtime header; its logic is restated below, line for line, and checked against the same reference.
#include "r3_e4m3_f16.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

int g_fail = 0;
int g_shown = 0;

void report(const char* what, uint64_t input, uint64_t got, uint64_t want) {
    ++g_fail;
    if (g_shown < 8) {
        ++g_shown;
        std::fprintf(stderr, "r3_e4m3_f16_test: MISMATCH %s input=0x%llx got=0x%llx want=0x%llx\n", what,
                     static_cast<unsigned long long>(input), static_cast<unsigned long long>(got),
                     static_cast<unsigned long long>(want));
    }
}

[[noreturn]] void die(const char* why, uint32_t code) {
    std::fprintf(stderr, "r3_e4m3_f16_test: reference error: %s (code 0x%02x)\n", why, code);
    std::exit(2);
}

// E4M3 definition: value = (-1)^s * 2^(e-7) * (1 + m/8) for e != 0, (-1)^s * 2^-6 * (m/8) for e == 0.
// Returns false for the NaN code S.1111.111.
bool decode_e4m3(uint32_t code, double* out) {
    const int s = (code >> 7) & 1;
    const int e = (code >> 3) & 15;
    const int m = code & 7;
    if (e == 15 && m == 7) return false;
    double v = (e == 0) ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + m / 8.0, e - 7);
    *out = s ? -v : v;
    return true;
}

// IEEE binary16 encoder for a finite double that must be exactly representable (no rounding is ever applied).
uint32_t encode_f16_exact(double v, uint32_t code) {
    const uint32_t sign = std::signbit(v) ? 0x8000u : 0u;
    const double a = std::fabs(v);
    if (a == 0.0) return sign;
    int ex;
    const double f = std::frexp(a, &ex);  // a = f * 2^ex, f in [0.5, 1)
    const int E = ex - 1;                 // a = 1.x * 2^E
    if (E > 15) die("value above the binary16 range", code);
    if (E < -14) {                        // binary16 subnormal: a = q * 2^-24
        const double q = std::ldexp(a, 24);
        if (q != std::floor(q) || q >= 1024.0) die("not exactly representable (subnormal)", code);
        return sign | static_cast<uint32_t>(q);
    }
    const double q = (f * 2.0 - 1.0) * 1024.0;
    if (q != std::floor(q) || q < 0.0 || q >= 1024.0) die("not exactly representable", code);
    return sign | (static_cast<uint32_t>(E + 15) << 10) | static_cast<uint32_t>(q);
}

uint32_t g_ref[256];
bool g_is_nan[256];

void build_reference() {
    for (uint32_t c = 0; c < 256; ++c) {
        double v;
        if (!decode_e4m3(c, &v)) {
            g_is_nan[c] = true;
            g_ref[c] = 0x7e00u | ((c & 0x80u) << 8);
        } else {
            g_is_nan[c] = false;
            g_ref[c] = encode_f16_exact(v, c);
        }
    }
}

// Restatement of r3_e4m3_to_f32 from r3_compat.h (same statements, host types).
float old_e4m3_to_f32(uint32_t b) {
    const uint32_t s = (b & 0x80u) << 24;
    const uint32_t e = (b >> 3) & 0xfu;
    uint32_t m = b & 0x7u;
    uint32_t u;
    if (e == 0u) {
        if (m == 0u) {
            u = s;
        } else {
            const uint32_t sh = m >= 4u ? 1u : (m >= 2u ? 2u : 3u);
            m = (m << sh) & 0x7u;
            u = s | ((127u - 6u - sh) << 23) | (m << 20);
        }
    } else if (e == 0xfu && m == 0x7u) {
        u = s | 0x7fc00000u;
    } else {
        u = s | ((e - 7u + 127u) << 23) | (m << 20);
    }
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

uint32_t f32_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

struct Rng {
    uint64_t s;
    uint32_t next() {  // xorshift64*
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return static_cast<uint32_t>((s * 0x2545f4914f6cdd1dull) >> 32);
    }
};

uint32_t pair_ref(uint32_t two) { return g_ref[two & 0xffu] | (g_ref[(two >> 8) & 0xffu] << 16); }

}  // namespace

int main() {
    build_reference();

    // 1. one code, all 256
    for (uint32_t c = 0; c < 256; ++c) {
        const uint32_t got = r3_e4m3_to_f16_bits(c);
        if (got != g_ref[c]) report("e4m3_to_f16_bits", c, got, g_ref[c]);
        // the other 24 bits of the argument are not part of the code
        const uint32_t got2 = r3_e4m3_to_f16_bits(c | 0xabcdef00u);
        if (got2 != g_ref[c]) report("e4m3_to_f16_bits(high bits set)", c | 0xabcdef00u, got2, g_ref[c]);
    }

    // 2. both halves, all 65,536 pairs, with several patterns in the unused half
    Rng rng{0x9e3779b97f4a7c15ull};
    uint64_t pair_checks = 0;
    for (uint32_t two = 0; two < 65536u; ++two) {
        const uint32_t want = pair_ref(two);
        const uint32_t junk[5] = {0x0000u, 0xffffu, 0x5555u, rng.next() & 0xffffu, rng.next() & 0xffffu};
        for (uint32_t j : junk) {
            const uint32_t lo_in = (j << 16) | two;
            const uint32_t hi_in = (two << 16) | j;
            const uint32_t got_lo = r3_e4m3x2_to_f16x2(lo_in);
            const uint32_t got_hi = r3_e4m3x2hi_to_f16x2(hi_in);
            if (got_lo != want) report("e4m3x2_to_f16x2", lo_in, got_lo, want);
            if (got_hi != want) report("e4m3x2hi_to_f16x2", hi_in, got_hi, want);
            pair_checks += 2;
        }
    }

    // 3. eight codes: edge patterns, then pseudo-random
    uint64_t x8_checks = 0;
    auto check8 = [&](uint32_t lo, uint32_t hi) {
        uint32_t w4[4] = {0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu};
        r3_e4m3x8_to_f16x4w(lo, hi, w4);
        const uint32_t want[4] = {pair_ref(lo), pair_ref(lo >> 16), pair_ref(hi), pair_ref(hi >> 16)};
        for (int i = 0; i < 4; ++i)
            if (w4[i] != want[i]) report("e4m3x8_to_f16x4w", (static_cast<uint64_t>(hi) << 32) | lo, w4[i], want[i]);
        ++x8_checks;
    };
    const uint32_t edge_bytes[] = {0x00, 0x80, 0x7f, 0xff, 0x7e, 0xfe, 0x01, 0x81, 0x02, 0x03, 0x04,
                                   0x05, 0x06, 0x07, 0x87, 0x08, 0x88, 0x38, 0xb8, 0x77, 0xf7};
    for (uint32_t b : edge_bytes) {
        const uint32_t rep = b * 0x01010101u;
        check8(rep, rep);
        check8(rep, 0u);
        check8(0u, rep);
    }
    check8(0x07060504u, 0x03020100u);  // the subnormal codes in order
    check8(0x87868584u, 0x83828180u);
    check8(0x7f7e0100u, 0xff80fe7fu);
    for (int i = 0; i < 200000; ++i) check8(rng.next(), rng.next());

    // 4. against the existing emulation: its f32 result converted to binary16 must give the same bits
    uint32_t old_nan_diff = 0;
    uint32_t old_checked = 0;
    uint32_t old_nan_f32[2] = {0, 0};
    for (uint32_t c = 0; c < 256; ++c) {
        const float f = old_e4m3_to_f32(c);
        if (g_is_nan[c]) {
            old_nan_f32[c >> 7] = f32_bits(f);
            // an f32 qNaN converts to the f16 qNaN with the sign kept and the payload truncated: sign | 0x7e00 | payload >> 13
            const uint32_t u = f32_bits(f);
            const uint32_t as_f16 = ((u >> 16) & 0x8000u) | 0x7c00u | ((u >> 13) & 0x3ffu);
            if (as_f16 != g_ref[c]) ++old_nan_diff;
            continue;
        }
        ++old_checked;
        const uint32_t via = encode_f16_exact(static_cast<double>(f), c);
        if (via != g_ref[c]) report("old emulation f32->f16", c, via, g_ref[c]);
        if (r3_e4m3_to_f16_bits(c) != via) report("direct vs old emulation", c, r3_e4m3_to_f16_bits(c), via);
    }

    if (g_fail != 0) {
        std::fprintf(stderr, "r3_e4m3_f16_test: FAILED %d mismatches\n", g_fail);
        return 1;
    }
    std::printf("r3_e4m3_f16_test: ok codes=256 x2/x2hi=%llu checks (65536 pairs, 5 unused-half patterns) x8=%llu "
                "old_emulation_nonnan=%u nan: old f32 0x%08x/0x%08x -> f16 %s the direct 0x7e00/0xfe00\n",
                static_cast<unsigned long long>(pair_checks), static_cast<unsigned long long>(x8_checks),
                old_checked, old_nan_f32[0], old_nan_f32[1], old_nan_diff == 0 ? "equals" : "DIFFERS from");
    return 0;
}
