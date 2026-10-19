// Expert-execution backend contract for the remote-MoE helper (RCS05, RAD11).
//
// The helper (transport, lanes, scheduling, expert ownership) is model- and format-independent. What differs per weight format and
// per GPU architecture - how an expert is sized and laid out, and the kernels that run it - sits behind this contract, so a new format
// (FP8 block-scaled) or a new architecture (RDNA2) is a new backend and not an edit to the helper.
//
// This header is plain C-compatible data and pure functions: no HIP, no radiance headers, no allocation. It is what the helper's
// ownership and sizing logic and the offline tests share. Every size is computed with checked 64-bit arithmetic and a failure is a
// negative value, never a wrapped number.
#pragma once

#include <cstdint>

#define MOE_BACKEND_ABI 1

enum MoeWeightFormat : uint32_t {
    MOE_FMT_NONE = 0,
    MOE_FMT_W4NL64 = 1,         // 4-bit nonlinear codes, one E4M3 scale per 64 columns (Flash-Next, "w4nl64a8h")
    MOE_FMT_FP8_E4M3_BLOCK = 2, // E4M3 codes, one scale per block_n x block_k tile (Qwen3.6-35B-A3B-FP8: 128 x 128)
};

enum MoeActivationFormat : uint32_t {
    MOE_ACT_NONE = 0,
    MOE_ACT_FP8_E4M3_GROUP = 1, // E4M3 activations with a group scale (the W4 path's "a8h")
    MOE_ACT_FP16 = 2,
    MOE_ACT_BF16 = 3,
};

enum MoeScaleDtype : uint32_t {
    MOE_SCALE_NONE = 0,
    MOE_SCALE_E4M3 = 1, // 1 byte
    MOE_SCALE_E8M0 = 2, // 1 byte
    MOE_SCALE_F16 = 3,  // 2 bytes
    MOE_SCALE_BF16 = 4, // 2 bytes
    MOE_SCALE_F32 = 5,  // 4 bytes
};

// A model's expert geometry, read from the container (RCS09); never a compile-time model constant.
struct MoeShape {
    uint32_t layers, experts, hidden, top_k;
    uint32_t intermediate;     // per-expert intermediate width (gate and up are each this wide)
    uint32_t shared_experts;
    uint32_t max_rows;         // largest step handed off (a runtime setting)
};

struct MoeWeightDesc {
    uint32_t format;           // MoeWeightFormat
    uint32_t activation;       // MoeActivationFormat the backend will be fed
    uint32_t scale_dtype;      // MoeScaleDtype
    uint32_t block_n, block_k; // scale tile (FP8 block-scaled); block_k is the group width for W4NL64
    uint32_t layout_id;        // backend-private physical layout selector
};

// One expert as the container describes it: the two matrices every MoE expert has.
struct MoeExpertDesc {
    MoeWeightDesc weight;
    uint64_t gu_rows, gu_cols; // gate+up fused: rows = 2 * intermediate, cols = hidden
    uint64_t dn_rows, dn_cols; // down: rows = hidden, cols = intermediate
};

struct MoeDeviceCaps {
    char arch[32];             // gcnArchName prefix, e.g. "gfx1100"
    uint64_t lds_bytes;
    uint32_t wave_size;
    uint32_t has_wmma_f16;     // 1 when the device has f16 WMMA
    uint32_t has_fp8_matrix;   // 1 when it has native FP8 matrix instructions (RDNA4, not RDNA3)
};

enum MoeStatus : int {
    MOE_OK = 0,
    MOE_UNSUPPORTED_FORMAT = -1,
    MOE_UNSUPPORTED_ARCH = -2,
    MOE_BAD_GEOMETRY = -3,
    MOE_OVERFLOW = -4,
};

namespace r3::moe_backend {

constexpr uint32_t kW4Group = 64;

inline bool mul_ok(uint64_t a, uint64_t b, uint64_t* out) {
    if (a && b > UINT64_MAX / a) return false;
    *out = a * b;
    return true;
}
inline bool add_ok(uint64_t a, uint64_t b, uint64_t* out) {
    if (b > UINT64_MAX - a) return false;
    *out = a + b;
    return true;
}
inline uint64_t ceil_div(uint64_t a, uint64_t b) { return b ? a / b + (a % b != 0) : 0; }

inline uint32_t scale_bytes(uint32_t dtype) {
    switch (dtype) {
        case MOE_SCALE_E4M3: case MOE_SCALE_E8M0: return 1;
        case MOE_SCALE_F16: case MOE_SCALE_BF16: return 2;
        case MOE_SCALE_F32: return 4;
        default: return 0;
    }
}

// Bytes of one matrix [rows, cols] in this weight format, or a MoeStatus (negative) when the geometry does not fit the format.
inline int64_t matrix_bytes(const MoeWeightDesc& w, uint64_t rows, uint64_t cols) {
    if (!rows || !cols) return MOE_BAD_GEOMETRY;
    uint64_t codes = 0, scales = 0, n = 0;
    switch (w.format) {
        case MOE_FMT_W4NL64:
            if (cols % kW4Group || !mul_ok(rows, cols, &n) || (n & 1)) return MOE_BAD_GEOMETRY;
            codes = n / 2;                                              // two 4-bit codes per byte
            if (!mul_ok(rows, cols / kW4Group, &scales)) return MOE_OVERFLOW;
            scales *= scale_bytes(w.scale_dtype ? w.scale_dtype : MOE_SCALE_E4M3);
            break;
        case MOE_FMT_FP8_E4M3_BLOCK: {
            const uint32_t sb = scale_bytes(w.scale_dtype);
            if (!w.block_n || !w.block_k || !sb) return MOE_BAD_GEOMETRY;
            if (!mul_ok(rows, cols, &codes)) return MOE_OVERFLOW;       // one byte per weight
            if (!mul_ok(ceil_div(rows, w.block_n), ceil_div(cols, w.block_k), &scales)) return MOE_OVERFLOW;
            scales *= sb;
            break;
        }
        default:
            return MOE_UNSUPPORTED_FORMAT;
    }
    uint64_t total = 0;
    if (!add_ok(codes, scales, &total) || total > (uint64_t)INT64_MAX) return MOE_OVERFLOW;
    return (int64_t)total;
}

// Bytes the container holds for one expert, with no GPU allocation. Negative is a MoeStatus.
inline int64_t expert_bytes(const MoeExpertDesc& e) {
    const int64_t gu = matrix_bytes(e.weight, e.gu_rows, e.gu_cols);
    if (gu < 0) return gu;
    const int64_t dn = matrix_bytes(e.weight, e.dn_rows, e.dn_cols);
    if (dn < 0) return dn;
    uint64_t t = 0;
    if (!add_ok((uint64_t)gu, (uint64_t)dn, &t) || t > (uint64_t)INT64_MAX) return MOE_OVERFLOW;
    return (int64_t)t;
}

// The expert geometry a shape implies: gate+up fused over [2 * intermediate, hidden], down over [hidden, intermediate].
inline MoeExpertDesc expert_for(const MoeShape& s, const MoeWeightDesc& w) {
    MoeExpertDesc e{};
    e.weight = w;
    e.gu_rows = 2ull * s.intermediate; e.gu_cols = s.hidden;
    e.dn_rows = s.hidden;              e.dn_cols = s.intermediate;
    return e;
}

// Whether a backend with the given capabilities can run this combination. Called before ownership is chosen, so an unsupported model
// is refused with a reason instead of failing at its first expert.
struct Backend {
    const char* name;
    uint32_t arch_prefix_len;  // compares the first n characters of caps.arch, 0 = any
    const char* arch_prefix;
    uint32_t formats;          // bit per MoeWeightFormat
    uint32_t requires_wmma_f16;
};

inline int probe(const Backend& b, const MoeShape& s, const MoeWeightDesc& w, const MoeDeviceCaps& caps) {
    if (!s.layers || !s.experts || !s.hidden || !s.top_k || !s.intermediate || s.top_k > s.experts) return MOE_BAD_GEOMETRY;
    if (b.arch_prefix_len) {
        for (uint32_t i = 0; i < b.arch_prefix_len; ++i)
            if (caps.arch[i] != b.arch_prefix[i]) return MOE_UNSUPPORTED_ARCH;
    }
    if (b.requires_wmma_f16 && !caps.has_wmma_f16) return MOE_UNSUPPORTED_ARCH;
    if (w.format == MOE_FMT_NONE || w.format >= 32 || !((b.formats >> w.format) & 1u)) return MOE_UNSUPPORTED_FORMAT;
    const int64_t bytes = expert_bytes(expert_for(s, w));
    return bytes < 0 ? (int)bytes : MOE_OK;
}

// The gfx1100 backend: W4NL64 today (the existing kernels), FP8 block-scaled to follow (RAD11).
inline const Backend& gfx1100_backend() {
    static const Backend b{"gfx1100", 7, "gfx1100", (1u << MOE_FMT_W4NL64), 1};
    return b;
}

}  // namespace r3::moe_backend
