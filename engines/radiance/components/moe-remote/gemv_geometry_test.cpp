#include "gemv_geometry.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <variant>
#include <vector>

using namespace r3_moe_gpu::gemv_geometry;

namespace {
int failures = 0;

void check(bool ok, const char* message, int line) {
    if (!ok) {
        std::cerr << "gemv_geometry_test:" << line << ": " << message << '\n';
        ++failures;
    }
}
#define CHECK(c) check(static_cast<bool>(c), #c, __LINE__)

constexpr uint32_t kFixtureK = 2560; // current kernel's fixed-array regression fixture, not a K limit
constexpr uint32_t kFixtureGroups = kFixtureK / 128;

// The declarations deliberately match the __shared__ arrays and scalar words
// in moe_lane_gemm.h and bc-gfx12/bc_moe_gemv.hip at the old fixed K.
struct FixedGeneric { float a[kFixtureK]; uint32_t sh_e; };
struct FixedW4H2 { uint16_t a[kFixtureK]; float asc[kFixtureGroups]; uint32_t sh_e; };
template <uint32_t R> struct FixedH2 {
    uint32_t a[R][kFixtureK / 2];
    float asc[R][kFixtureGroups];
    uint32_t sh_e, sh_first;
};
struct FixedMultiRow { float a[4][kFixtureK]; };
static_assert(sizeof(FixedGeneric) == sizeof(float[kFixtureK]) + sizeof(uint32_t));
static_assert(sizeof(FixedW4H2) == sizeof(uint16_t[kFixtureK]) + sizeof(float[kFixtureGroups]) + sizeof(uint32_t));
static_assert(sizeof(FixedH2<1>) == sizeof(uint32_t[1][kFixtureK / 2]) + sizeof(float[1][kFixtureGroups]) + 2 * sizeof(uint32_t));
static_assert(sizeof(FixedH2<2>) == sizeof(uint32_t[2][kFixtureK / 2]) + sizeof(float[2][kFixtureGroups]) + 2 * sizeof(uint32_t));
static_assert(sizeof(FixedH2<4>) == sizeof(uint32_t[4][kFixtureK / 2]) + sizeof(float[4][kFixtureGroups]) + 2 * sizeof(uint32_t));
static_assert(sizeof(FixedMultiRow) == sizeof(float[4][kFixtureK]));

constexpr std::array<Variant, 9> variants = {
    Variant::GenericFp8, Variant::GenericW4, Variant::WideFp8,
    Variant::H2Fp8R1, Variant::H2Fp8R2, Variant::H2Fp8R4,
    Variant::W4H2, Variant::MultiRowFp8R4, Variant::Gfx12Fp8R2,
};
constexpr std::array<uint32_t, 9> old_sizes = {
    sizeof(FixedGeneric), sizeof(FixedGeneric), sizeof(FixedGeneric),
    sizeof(FixedH2<1>), sizeof(FixedH2<2>), sizeof(FixedH2<4>),
    sizeof(FixedW4H2), sizeof(FixedMultiRow), sizeof(FixedH2<2>),
};

const LdsLayout& geometry(Variant variant, uint32_t k) {
    // Only call for expected-valid cases; copy into stable storage.
    static LdsLayout result{};
    auto answer = required_lds_bytes(variant, k, 128);
    CHECK(std::holds_alternative<LdsLayout>(answer));
    if (const auto* valid = std::get_if<LdsLayout>(&answer)) result = *valid;
    return result;
}

void check_reason(const Geometry& result, Reason expected) {
    CHECK(std::holds_alternative<Refuse>(result));
    if (const auto* failure = std::get_if<Refuse>(&result))
        CHECK(failure->reason == expected);
}
void check_choice_reason(const Choice& choice, Reason expected, bool generic) {
    CHECK(generic ? std::holds_alternative<GenericGemm>(choice)
                  : std::holds_alternative<Refuse>(choice));
    if (const auto* alternative = std::get_if<GenericGemm>(&choice))
        CHECK(alternative->gemv_ineligible == expected);
    if (const auto* refusal = std::get_if<Refuse>(&choice))
        CHECK(refusal->reason == expected);
}
Capabilities caps_for(const LdsLayout& layout) {
    return Capabilities{layout.total_bytes, layout.dynamic_bytes, layout.static_bytes,
                        1, true, true};
}

void test_sizes_and_shapes() {
    check_reason(required_lds_bytes(static_cast<Variant>(999), 2560, 128),
                 Reason::UnsupportedVariant);
    for (size_t i = 0; i < variants.size(); ++i) {
        const auto v = variants[i];
        const auto baseline = geometry(v, kFixtureK);
        CHECK(baseline.total_bytes == old_sizes[i]);
        CHECK(baseline.dynamic_bytes + baseline.static_bytes == old_sizes[i]);
        CHECK(baseline.scale_groups == kFixtureGroups);
        CHECK(baseline.scale_offset_bytes == baseline.activation_bytes);
        CHECK(baseline.scale_offset_bytes % 16 == 0);

        // Below, equal and above 2560 all remain valid; 2560 is not a cap.
        for (uint32_t k : {uint32_t{2432}, kFixtureK, uint32_t{2688}}) {
            const auto a = geometry(v, k);
            CHECK(a.k == k);
            CHECK(a.scale_groups == k / 128);
            CHECK(a.activation_bytes == a.rows * a.activation_row_stride_bytes);
            CHECK(a.scale_bytes == a.rows * a.scale_row_stride_bytes);
            CHECK(a.dynamic_bytes == a.activation_bytes + a.scale_bytes);
            CHECK(a.total_bytes == a.dynamic_bytes + a.static_bytes);
            CHECK(a.scale_offset_bytes % 16 == 0);
            if (v == Variant::H2Fp8R1 || v == Variant::H2Fp8R2 ||
                v == Variant::H2Fp8R4 || v == Variant::Gfx12Fp8R2) {
                CHECK(a.planes == 2);
                CHECK(a.activation_row_stride_bytes == 2 * k);
                CHECK(a.scale_row_stride_bytes == 4 * (k / 128));
                CHECK(a.static_bytes == 2 * sizeof(uint32_t));
            } else if (v == Variant::W4H2) {
                CHECK(a.activation_row_stride_bytes == 2 * k);
                CHECK(a.scale_row_stride_bytes == 4 * (k / 128));
                CHECK(a.static_bytes == sizeof(uint32_t));
            } else if (v == Variant::MultiRowFp8R4) {
                CHECK(a.rows == 4);
                CHECK(a.planes == 4);
                CHECK(a.activation_row_stride_bytes == 4 * k);
                CHECK(a.scale_bytes == 0);
                CHECK(a.static_bytes == 0);
            } else {
                CHECK(a.activation_row_stride_bytes == 4 * k);
                CHECK(a.scale_bytes == 0);
                CHECK(a.static_bytes == sizeof(uint32_t));
            }
        }

        check_reason(required_lds_bytes(v, 0, 128), Reason::ZeroK);
        check_reason(required_lds_bytes(v, 127, 128), Reason::MisalignedK);
        check_reason(required_lds_bytes(v, 2560, 64), Reason::UnsupportedScaleGroup);
        check_reason(required_lds_bytes(v, 2560, 256), Reason::UnsupportedScaleGroup);
        // Aligned to 128, but every variant's full LDS size exceeds the
        // 32-bit launch-byte representation; reject, never wrap.
        constexpr uint32_t huge = std::numeric_limits<uint32_t>::max() - 127;
        static_assert(huge % 128 == 0);
        check_reason(required_lds_bytes(v, huge, 128), Reason::ArithmeticOverflow);
    }

    CHECK(geometry(Variant::H2Fp8R1, kFixtureK).rows == 1);
    CHECK(geometry(Variant::H2Fp8R2, kFixtureK).rows == 2);
    CHECK(geometry(Variant::H2Fp8R4, kFixtureK).rows == 4);
    CHECK(geometry(Variant::Gfx12Fp8R2, kFixtureK).rows == 2);
}

void test_layout_indices() {
    // A CPU bijection oracle for the exact four float4 planes of wide/MR,
    // and the two half2 planes of H2. Each logical K value is stored once.
    for (uint32_t k : {uint32_t{128}, kFixtureK, uint32_t{3072}}) {
        const uint32_t k16 = k / 16;
        std::vector<unsigned> wide(k, 0), half(k, 0);
        for (uint32_t i = 0; i < k; ++i) {
            const uint32_t d = i >> 4, q = (i >> 2) & 3u, c = i & 3u;
            const uint32_t wide_index = (q * k16 + d) * 4u + c;
            const uint32_t j = i & 3u;
            const uint32_t half_index = ((((j & 1u) * k16 + d) * 4u + q) << 1) + (j >> 1);
            CHECK(wide_index < k);
            CHECK(half_index < k);
            if (wide_index < k) ++wide[wide_index];
            if (half_index < k) ++half[half_index];
        }
        for (uint32_t i = 0; i < k; ++i) {
            CHECK(wide[i] == 1);
            CHECK(half[i] == 1);
        }
        const auto h2 = geometry(Variant::H2Fp8R4, k);
        CHECK(h2.scale_offset_bytes == 4u * h2.activation_row_stride_bytes);
        CHECK(h2.scale_offset_bytes + 4u * h2.scale_row_stride_bytes == h2.dynamic_bytes);
        const auto mr = geometry(Variant::MultiRowFp8R4, k);
        CHECK(mr.rows * mr.activation_row_stride_bytes == mr.dynamic_bytes);
    }
}

void test_prelaunch_dispatch() {
    for (const auto v : variants) {
        const auto layout = geometry(v, kFixtureK);
        auto caps = caps_for(layout);
        Choice c = choose_gemv(v, kFixtureK, kFixtureK, 128, caps);
        CHECK(std::holds_alternative<Gemv>(c));
        if (const auto* selected = std::get_if<Gemv>(&c)) {
            CHECK(selected->variant == v);
            CHECK(selected->dynamic_lds == layout.dynamic_bytes);
        }

        // Exact boundary (including static shared words) fits.
        --caps.per_block_lds_limit;
        check_choice_reason(choose_gemv(v, kFixtureK, kFixtureK, 128, caps),
                            Reason::BlockLdsExceeded, true);
        caps.per_block_lds_limit += 2;
        CHECK(std::holds_alternative<Gemv>(choose_gemv(v, kFixtureK, kFixtureK, 128, caps)));

        caps = caps_for(layout);
        --caps.kernel_dynamic_lds_limit;
        check_choice_reason(choose_gemv(v, kFixtureK, kFixtureK, 128, caps),
                            Reason::KernelDynamicLdsExceeded, true);

        caps = caps_for(layout);
        caps.kernel_available = false;
        check_choice_reason(choose_gemv(v, kFixtureK, kFixtureK, 128, caps),
                            Reason::KernelUnavailable, true);

        caps = caps_for(layout);
        caps.active_blocks_per_cu = 0;
        check_choice_reason(choose_gemv(v, kFixtureK, kFixtureK, 128, caps),
                            Reason::OccupancyUnavailable, true);

        caps = caps_for(layout);
        caps.kernel_static_bytes += 16; // compiler may report extra static LDS
        check_choice_reason(choose_gemv(v, kFixtureK, kFixtureK, 128, caps),
                            Reason::BlockLdsExceeded, true);

        caps = caps_for(layout);
        caps.generic_gemm_available = false;
        caps.per_block_lds_limit -= 1;
        check_choice_reason(choose_gemv(v, kFixtureK, kFixtureK, 128, caps),
                            Reason::BlockLdsExceeded, false);
        check_choice_reason(choose_gemv(v, 0, kFixtureK, 128, caps), Reason::ZeroK, false);

        caps = caps_for(layout);
        check_choice_reason(choose_gemv(v, 0, kFixtureK, 128, caps), Reason::ZeroK, false);
        check_choice_reason(choose_gemv(v, kFixtureK, 128, 64, caps),
                            Reason::UnsupportedScaleGroup, true);
        check_choice_reason(choose_gemv(v, kFixtureK, 129, 128, caps),
                            Reason::MisalignedK, true);
    }

    const auto wide = Variant::H2Fp8R2;
    auto caps = caps_for(geometry(wide, 3072));
    const Choice both = choose_gemv(wide, 2560, 3072, 128, caps);
    CHECK(std::holds_alternative<Gemv>(both));
    if (const auto* gemv = std::get_if<Gemv>(&both))
        CHECK(gemv->dynamic_lds == geometry(wide, 3072).dynamic_bytes);
    CHECK(std::holds_alternative<Gemv>(choose_gemv(wide, 3072, 2560, 128, caps)));

    // Gate/up's hidden K fits but down's intermediate K does not. The
    // caller must invoke the selector with the down pair, not hidden.
    caps = caps_for(geometry(wide, 2560));
    CHECK(std::holds_alternative<Gemv>(choose_gemv(wide, 2560, 2560, 128, caps)));
    check_choice_reason(choose_gemv(wide, 3072, 3072, 128, caps),
                        Reason::KernelDynamicLdsExceeded, true);
    caps.generic_gemm_available = false;
    check_choice_reason(choose_gemv(wide, 3072, 3072, 128, caps),
                        Reason::KernelDynamicLdsExceeded, false);
}

void test_stage_dispatch() {
    const Preferences fp8{true, false, false, 2, true};
    const Preferences w4{false, true, false, 0, false};
    CHECK(preferred_variant(fp8, 2560, 3072) == Variant::H2Fp8R2);
    CHECK(preferred_variant(w4, 2560, 3072) == Variant::W4H2);
    CHECK(preferred_variant({true, false, true, 4, true}, 2560, 3072) ==
          Variant::MultiRowFp8R4);
    CHECK(preferred_variant({true, false, false, 0, true}, 2560, 3072) ==
          Variant::WideFp8);
    CHECK(preferred_variant({false, false, false, 0, false}, 2560, 3072) ==
          Variant::GenericW4);
    const auto variant = preferred_variant(fp8, 2560, 3072);
    auto caps = caps_for(geometry(variant, 2560));
    CHECK(std::holds_alternative<Gemv>(
        choose_stage(true, variant, 2560, 2560, 128, caps)));
    check_choice_reason(choose_stage(true, variant, 3072, 3072, 128, caps),
                        Reason::KernelDynamicLdsExceeded, true);
    // One parity's larger K must not be erased by the smaller one.
    check_choice_reason(choose_stage(true, variant, 2560, 3072, 128, caps),
                        Reason::KernelDynamicLdsExceeded, true);
    check_choice_reason(choose_stage(true, variant, 3072, 2560, 128, caps),
                        Reason::KernelDynamicLdsExceeded, true);
    // A larger hidden never implies refusal if grouped GEMM works.
    check_choice_reason(choose_stage(true, variant, 4096, 4096, 128, caps),
                        Reason::KernelDynamicLdsExceeded, true);
    check_choice_reason(choose_stage(false, variant, 2560, 2560, 128, caps),
                        Reason::KernelUnavailable, true);
    caps.generic_gemm_available = false;
    check_choice_reason(choose_stage(true, variant, 3072, 3072, 128, caps),
                        Reason::KernelDynamicLdsExceeded, false);
    check_choice_reason(choose_stage(false, variant, 2560, 2560, 128, caps),
                        Reason::KernelUnavailable, false);
    // Debug W4 and gfx12 decode paths obey the same selector.
    check_choice_reason(choose_stage(true, Variant::GenericW4, 3072, 3072, 128, caps),
                        Reason::KernelDynamicLdsExceeded, false);
    check_choice_reason(choose_stage(true, Variant::Gfx12Fp8R2, 3072, 3072, 128, caps),
                        Reason::KernelDynamicLdsExceeded, false);
}

}  // namespace

int main() {
    test_sizes_and_shapes();
    test_layout_indices();
    test_prelaunch_dispatch();
    test_stage_dispatch();
    if (failures) {
        std::cerr << failures << " GEMV geometry checks failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "gemv geometry: all checks passed\n";
    return EXIT_SUCCESS;
}
