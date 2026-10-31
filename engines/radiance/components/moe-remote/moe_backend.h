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
#include <cstring>

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

enum MoeStatus : int {
    MOE_OK = 0,
    MOE_UNSUPPORTED_FORMAT = -1,
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

// A card type's backend, as data (RAD23): the weight formats its kernels run. Which card types exist, and which formats each runs, is
// engines/radiance/archs.toml; the launcher and the harness pass the list to the helper (R3_MOE_WORKER_FORMATS). Nothing here names an
// architecture - the plugin the helper loads for its card is checked against that card when it is opened.
struct Backend {
    char name[32];             // the card's architecture, for messages
    uint32_t formats;          // bit per MoeWeightFormat
};

inline const char* format_name(uint32_t fmt) {
    return fmt == MOE_FMT_W4NL64 ? "w4nl64" : fmt == MOE_FMT_FP8_E4M3_BLOCK ? "fp8_e4m3_block" : "unrecognised";
}

// "w4nl64,fp8_e4m3_block" -> a format mask. False (mask untouched) on an empty list, an unknown name or a repeat.
inline bool parse_formats(const char* text, uint32_t* mask) {
    if (!text || !*text || !mask) return false;
    uint32_t m = 0;
    for (const char* p = text;;) {
        const char* e = p;
        while (*e && *e != ',') ++e;
        uint32_t bit = 0;
        static const uint32_t kKnown[2] = {MOE_FMT_W4NL64, MOE_FMT_FP8_E4M3_BLOCK};
        for (uint32_t f : kKnown) {
            const char* n = format_name(f);
            if ((size_t)(e - p) == std::strlen(n) && std::strncmp(p, n, (size_t)(e - p)) == 0) bit = 1u << f;
        }
        if (!bit || (m & bit)) return false;
        m |= bit;
        if (!*e) break;
        p = e + 1;
    }
    *mask = m;
    return true;
}

// Whether the backend can run this combination. Called before ownership is chosen, so an unsupported model is refused with a reason
// instead of failing at its first expert.
inline int probe(const Backend& b, const MoeShape& s, const MoeWeightDesc& w) {
    if (!s.layers || !s.experts || !s.hidden || !s.top_k || !s.intermediate || s.top_k > s.experts) return MOE_BAD_GEOMETRY;
    if (w.format == MOE_FMT_NONE || w.format >= 32 || !((b.formats >> w.format) & 1u)) return MOE_UNSUPPORTED_FORMAT;
    const int64_t bytes = expert_bytes(expert_for(s, w));
    return bytes < 0 ? (int)bytes : MOE_OK;
}

}  // namespace r3::moe_backend
