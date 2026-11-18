#pragma once

// Host-only GEMV LDS geometry and pre-launch eligibility, independent of HIP.
// This mirrors the compiled staging layouts in moe_lane_gemm.h and
// bc-gfx12/bc_moe_gemv.hip. It does not change kernel dispatch by itself.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <variant>

namespace r3_moe_gpu::gemv_geometry {

enum class Variant {
    GenericFp8,       // lane_moe_gemv<TOKEN, true>
    GenericW4,        // lane_moe_gemv<TOKEN, false>
    WideFp8,          // lane_moe_gemv_wide<TOKEN>
    H2Fp8R1,          // lane_moe_gemv_h2<TOKEN, 1>
    H2Fp8R2,          // lane_moe_gemv_h2<TOKEN, 2>
    H2Fp8R4,          // lane_moe_gemv_h2<TOKEN, 4>
    W4H2,             // lane_moe_gemv_w4h2<TOKEN>
    MultiRowFp8R4,    // lane_moe_gemv_mr<TOKEN>, kMrRows=4
    Gfx12Fp8R2,       // bc-gfx12 moe_gemv_fp8_h2<TOKEN>, kRows=2
};

enum class Reason {
    ZeroK,
    UnsupportedScaleGroup,
    MisalignedK,
    ArithmeticOverflow,
    KernelUnavailable,
    KernelDynamicLdsExceeded,
    BlockLdsExceeded,
    OccupancyUnavailable,
    UnsupportedVariant,
};

struct Refuse {
    Reason reason;
};

struct LdsLayout {
    uint32_t k;
    uint32_t scale_groups;
    uint32_t rows;
    uint32_t planes;
    uint32_t activation_row_stride_bytes;
    uint32_t scale_row_stride_bytes;
    uint32_t activation_bytes;
    uint32_t scale_offset_bytes;  // offset from the 16-byte-aligned dynamic LDS base
    uint32_t scale_bytes;
    uint32_t dynamic_bytes;       // passed to HIP as the dynamic shared-memory argument
    uint32_t static_bytes;        // scalar __shared__ words declared in the current kernel
    uint32_t total_bytes;         // dynamic + source-declared static (before compiler padding)
};

using Geometry = std::variant<LdsLayout, Refuse>;

namespace detail {
constexpr uint32_t kActivationScaleGroup = 128; // E4M3 activation scale indexing in every variant
constexpr uint32_t kVectorBytes = 16;          // float4 / uint4 vector LDS access
constexpr uint32_t kW4WeightGroup = 64;        // W4 expert's weight scale grouping

inline bool multiply(uint64_t a, uint64_t b, uint64_t& out) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b) return false;
    out = a * b;
    return true;
}

inline bool add(uint64_t a, uint64_t b, uint64_t& out) {
    if (a > std::numeric_limits<uint64_t>::max() - b) return false;
    out = a + b;
    return true;
}

inline bool fits_lds_arg(uint64_t bytes) {
    return bytes <= std::numeric_limits<uint32_t>::max();
}
}  // namespace detail

// 'group' is the activation E4M3 scale group, NOT the W4 weight group.
// All compiled variants index activations in groups of 128. The W4 kernels
// also require weight groups of 64; whole activation groups imply that here.
inline Geometry required_lds_bytes(Variant variant, uint32_t k, uint32_t group) {
    if (!k) return Refuse{Reason::ZeroK};
    if (group != detail::kActivationScaleGroup) return Refuse{Reason::UnsupportedScaleGroup};
    if (k % group) return Refuse{Reason::MisalignedK};

    uint32_t rows = 1, planes = 1, activation_bytes_per_element = sizeof(float);
    uint32_t static_bytes = sizeof(uint32_t);  // sh_e in generic, wide and W4H2
    bool has_scales = false;
    switch (variant) {
        case Variant::GenericFp8:
        case Variant::GenericW4:
            break;
        case Variant::WideFp8:
            planes = 4;
            break;
        case Variant::H2Fp8R1:
            rows = 1; planes = 2; activation_bytes_per_element = sizeof(uint16_t);
            static_bytes = 2 * sizeof(uint32_t); has_scales = true; break;
        case Variant::H2Fp8R2:
        case Variant::Gfx12Fp8R2:
            rows = 2; planes = 2; activation_bytes_per_element = sizeof(uint16_t);
            static_bytes = 2 * sizeof(uint32_t); has_scales = true; break;
        case Variant::H2Fp8R4:
            rows = 4; planes = 2; activation_bytes_per_element = sizeof(uint16_t);
            static_bytes = 2 * sizeof(uint32_t); has_scales = true; break;
        case Variant::W4H2:
            activation_bytes_per_element = sizeof(uint16_t);
            has_scales = true;
            break;
        case Variant::MultiRowFp8R4:
            rows = 4; planes = 4; static_bytes = 0; break;
        default:
            return Refuse{Reason::UnsupportedVariant};
    }
    // All vector-LDS kernels have aligned K-dependent row strides. The
    // generic scalar variant still uses the same activation group contract.
    const uint32_t vector_alignment = (planes > 1 || variant == Variant::W4H2)
        ? detail::kVectorBytes : alignof(float);
    if ((static_cast<uint64_t>(k) * activation_bytes_per_element) % vector_alignment)
        return Refuse{Reason::MisalignedK};
    if ((variant == Variant::GenericW4 || variant == Variant::W4H2) &&
        (k % detail::kW4WeightGroup))
        return Refuse{Reason::MisalignedK};

    uint64_t row_activation = 0, row_scales = 0, activation = 0, scales = 0;
    uint64_t scale_offset = 0, dynamic = 0, total = 0;
    const uint64_t groups = k / group;
    if (!detail::multiply(k, activation_bytes_per_element, row_activation) ||
        !detail::multiply(groups, has_scales ? sizeof(float) : 0, row_scales) ||
        !detail::multiply(row_activation, rows, activation) ||
        !detail::multiply(row_scales, rows, scales) ||
        !detail::add(activation, 0, scale_offset) ||
        !detail::add(scale_offset, scales, dynamic) ||
        !detail::add(dynamic, static_bytes, total) ||
        !detail::fits_lds_arg(row_activation) ||
        !detail::fits_lds_arg(row_scales) ||
        !detail::fits_lds_arg(activation) ||
        !detail::fits_lds_arg(scale_offset) ||
        !detail::fits_lds_arg(scales) ||
        !detail::fits_lds_arg(dynamic) ||
        !detail::fits_lds_arg(total))
        return Refuse{Reason::ArithmeticOverflow};
    // K is a whole activation group; a, a[rows][K], and asc start on
    // 16-byte boundaries. No padding is present in today's fixed arrays.
    if (scale_offset % detail::kVectorBytes)
        return Refuse{Reason::MisalignedK};
    return LdsLayout{k, static_cast<uint32_t>(groups), rows, planes,
                     static_cast<uint32_t>(row_activation), static_cast<uint32_t>(row_scales),
                     static_cast<uint32_t>(activation), static_cast<uint32_t>(scale_offset),
                     static_cast<uint32_t>(scales), static_cast<uint32_t>(dynamic),
                     static_bytes, static_cast<uint32_t>(total)};
}

// The preference order is the production launch order; use one testable
// contract for it and for the HIP dispatcher.
struct Preferences {
    bool fp8;
    bool w4h2;
    bool multi_row;
    uint32_t h2_rows;
    bool wide;
};
inline Variant preferred_variant(const Preferences& p, uint32_t k0, uint32_t k1) {
    if (!p.fp8)
        return p.w4h2 && !(k0 % detail::kW4WeightGroup) &&
               !(k1 % detail::kW4WeightGroup) ? Variant::W4H2 : Variant::GenericW4;
    if (p.multi_row && !(k0 % detail::kVectorBytes) && !(k1 % detail::kVectorBytes))
        return Variant::MultiRowFp8R4;
    if (!(k0 % detail::kVectorBytes) && !(k1 % detail::kVectorBytes)) {
        switch (p.h2_rows) {
            case 4: return Variant::H2Fp8R4;
            case 2: return Variant::H2Fp8R2;
            case 1: return Variant::H2Fp8R1;
            default: break;
        }
        if (p.wide) return Variant::WideFp8;
    }
    return Variant::GenericFp8;
}

struct Capabilities {
    uint32_t per_block_lds_limit;     // runtime actual/opted-in per-block limit
    uint32_t kernel_dynamic_lds_limit; // runtime limit for this compiled kernel
    uint32_t kernel_static_bytes;     // hipFuncAttributes.sharedSizeBytes
    uint32_t active_blocks_per_cu;    // occupancy query with 256 threads + dynamic LDS
    bool kernel_available;            // kernel compiled and launchable on this device
    bool generic_gemm_available;      // verified generic GEMM for this format/stage/device
};

struct Gemv {
    Variant variant;
    uint32_t dynamic_lds;
};
struct GenericGemm {
    Reason gemv_ineligible;
};
using Choice = std::variant<Gemv, GenericGemm, Refuse>;

// Both parity K values share one launch; reserve the largest, even when
// the smaller class executes first. The caller passes the appropriate
// projection's pair (gate/up: hidden; down: intermediate), never model.hidden
// in place of the down-projection K.
inline Choice choose_gemv(Variant variant, uint32_t k0, uint32_t k1,
                          uint32_t group, const Capabilities& caps) {
    auto fallback = [&](Reason reason) -> Choice {
        return caps.generic_gemm_available ? Choice{GenericGemm{reason}}
                                           : Choice{Refuse{reason}};
    };
    if (!k0 || !k1) return Refuse{Reason::ZeroK};
    const Geometry layout = required_lds_bytes(variant, std::max(k0, k1), group);
    if (const auto* invalid = std::get_if<Refuse>(&layout))
        return fallback(invalid->reason);
    // Check the smaller parity too: an invalid K cannot be made valid
    // merely by sizing the scratch region for the other parity.
    const Geometry other = required_lds_bytes(variant, std::min(k0, k1), group);
    if (const auto* invalid = std::get_if<Refuse>(&other))
        return fallback(invalid->reason);
    if (!caps.kernel_available) return fallback(Reason::KernelUnavailable);
    const auto& selected = std::get<LdsLayout>(layout);
    if (selected.dynamic_bytes > caps.kernel_dynamic_lds_limit)
        return fallback(Reason::KernelDynamicLdsExceeded);
    // The compiler's recorded static size can include padding/extra words.
    // Never assume less than the source-declared scalar __shared__ words.
    const uint64_t total = static_cast<uint64_t>(selected.dynamic_bytes) +
                           std::max(caps.kernel_static_bytes, selected.static_bytes);
    if (total > caps.per_block_lds_limit)
        return fallback(Reason::BlockLdsExceeded);
    if (!caps.active_blocks_per_cu)
        return fallback(Reason::OccupancyUnavailable);
    return Gemv{variant, selected.dynamic_bytes};
}

// Disabled and rejected GEMV both require a checked grouped kernel.
inline Choice choose_stage(bool requested, Variant variant, uint32_t k0,
                           uint32_t k1, uint32_t group, const Capabilities& caps) {
    if (!k0 || !k1) return Refuse{Reason::ZeroK};
    if (!requested)
        return caps.generic_gemm_available ? Choice{GenericGemm{Reason::KernelUnavailable}}
                                           : Choice{Refuse{Reason::KernelUnavailable}};
    return choose_gemv(variant, k0, k1, group, caps);
}

} // namespace r3_moe_gpu::gemv_geometry
