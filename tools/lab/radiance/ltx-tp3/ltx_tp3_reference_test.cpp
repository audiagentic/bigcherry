#include "ltx_tp3_reference.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace bc_ltx_tp3_reference;

template <class F> void must_refuse(F&& f) {
    bool refused = false;
    try { f(); }
    catch (const std::invalid_argument&) { refused = true; }
    assert(refused && "invalid model/device plan accepted");
}

void test_head_ownership() {
    for (const Shares counts : {Shares{8, 8, 16}, Shares{12, 12, 8}, Shares{10, 10, 12}}) {
        for (const auto& [heads, dim] : {std::pair{32, 128}, std::pair{32, 64}}) {
            const auto owned = attention_slices(heads, dim, counts);
            assert(owned[0].offset == 0);
            assert(owned[1].offset == counts[0] * dim);
            assert(owned[2].offset == (counts[0] + counts[1]) * dim);
            assert(owned[2].offset + owned[2].width == heads * dim);
            for (const Slice p : owned) {
                assert(p.width > 0 && p.width % dim == 0);
            }
        }
    }
    must_refuse([] { attention_slices(32, 128, {11, 11, 11}); });
    must_refuse([] { attention_slices(32, 128, {0, 16, 16}); });
    must_refuse([] { attention_slices(32, 128, {8, 8, 15}); });
    must_refuse([] { attention_slices(32, 0, {8, 8, 16}); });
}

void test_fused_qkv_bank_mapping() {
    for (const int dim : {128, 64}) {
        const int inner = 32 * dim;
        const auto owned = attention_slices(32, dim, {8, 8, 16});
        std::vector<int> reconstructed(3 * inner, -1);
        for (const auto o : owned) {
            const auto pieces = fused_qkv_pieces(inner, o);
            std::vector<int> compact(3 * o.width, -1);
            for (int bank = 0; bank < 3; ++bank) {
                const auto p = pieces[bank];
                assert(p.source_row == bank * inner + o.offset);
                assert(p.local_row == bank * o.width);
                assert(p.count == o.width);
                for (int i = 0; i < p.count; ++i)
                    compact[p.local_row + i] = p.source_row + i;
            }
            for (const auto p : pieces)
                for (int i = 0; i < p.count; ++i) {
                    const int idx = p.source_row + i;
                    assert(reconstructed[idx] == -1 && "head overlap");
                    reconstructed[idx] = compact[p.local_row + i];
                }
        }
        for (int i = 0; i < 3 * inner; ++i)
            assert(reconstructed[i] == i && "missing Q/K/V row");
    }
    must_refuse([] { fused_qkv_pieces(4096, {4000, 128}); });
    must_refuse([] { fused_qkv_pieces(std::numeric_limits<int>::max(), {0, 1}); });
}

void test_iq8r_alignment() {
    const auto good = iq8r_input_slices(4096, {1024, 1024, 2048});
    assert(good[1].offset == 1024 && good[2].offset == 2048);
    // The same 32 heads are not the same K-scale partition in video/audio.
    // Video 10/10/12 is 1280/1280/1536 and valid, audio is
    // 640/640/768 and splits IQ8R scales even though heads are whole.
    const auto video_uneven = iq8r_input_slices(4096, {1280, 1280, 1536});
    assert(video_uneven[2].offset == 2560);
    const auto audio_good = iq8r_input_slices(2048, {512, 512, 1024});
    assert(audio_good[2].offset == 1024);
    const auto audio_other = iq8r_input_slices(2048, {768, 768, 512});
    assert(audio_other[2].offset == 1536);
    must_refuse([] { iq8r_input_slices(2048, {640, 640, 768}); });
    // Audio FFN-down uses 8192 input channels, four times the audio
    // attention width. Its 10/10/12 K cuts are whole 256-wide groups.
    const auto audio_ff = iq8r_input_slices(8192, {2560, 2560, 3072});
    assert(audio_ff[2].offset == 5120);
    must_refuse([] { iq8r_input_slices(4096, {128, 1920, 2048}); });
    must_refuse([] { iq8r_input_slices(4096, {1024, 1024, 1792}); });
    must_refuse([] { iq8r_input_slices(4032, {1024, 1024, 1984}); });
    // Row-parallel K shard alignment is not evidence of correct IQ8R kernel support:
    // the encoded scale/rotation planes and the global activation quantizer remain gates.
}

void test_all_head_rms() {
    for (const int width : {4096, 2048}) {
        const int head_dim = width / 32;
        const auto owned = attention_slices(32, head_dim, {8, 8, 16});
        std::array<std::vector<float>, 3> x, gains;
        std::vector<float> full_x(width), full_gain(width);
        for (int i = 0; i < width; ++i) {
            // Unequal norms by rank: deliberately exposes the incorrect local-RMS shortcut.
            const float factor = i < width / 4 ? 0.12f : i < width / 2 ? 4.0f : 1.7f;
            full_x[i] = factor * (float((i * 19) % 43) - 15.0f) / 13.0f;
            full_gain[i] = 0.75f + float(i % 7) / 16.0f;
        }
        for (int r = 0; r < 3; ++r) {
            const auto p = owned[r];
            x[r].assign(full_x.begin() + p.offset, full_x.begin() + p.offset + p.width);
            gains[r].assign(full_gain.begin() + p.offset, full_gain.begin() + p.offset + p.width);
        }
        const double eps = 1e-6;
        double sum = 0;
        for (float v : full_x) sum += double(v) * double(v);
        const double denominator = std::sqrt(sum / width + eps);
        const auto output = global_rms(x, gains, eps);
        bool local_would_be_wrong = false;
        for (int r = 0; r < 3; ++r) {
            double local_sq = 0;
            for (float v : x[r]) local_sq += double(v) * double(v);
            const double local_den = std::sqrt(local_sq / x[r].size() + eps);
            for (std::size_t i = 0; i < x[r].size(); ++i) {
                const double reference = double(x[r][i]) * gains[r][i] / denominator;
                assert(std::fabs(output[r][i] - reference) < 1e-5);
                if (std::fabs(reference - double(x[r][i]) * gains[r][i] / local_den) > .03)
                    local_would_be_wrong = true;
            }
        }
        assert(local_would_be_wrong && "test lacks coverage of global-head RMS");
    }
    std::array<std::vector<float>, 3> x{{{1}, {2}, {3}}}, g{{{1}, {1}, {1}}};
    must_refuse([&] { global_rms(x, g, 0); });
    must_refuse([&] { global_rms(x, g, -1); });
    g[1].clear();
    must_refuse([&] { global_rms(x, g, 1e-6); });
}

FourCardPlan mock_plan() {
    return {{{{0, Arch::gfx1100, 24576, 1024},
              {1, Arch::gfx1100, 24576, 1024},
              {2, Arch::gfx1201, 32768, 2048}}},
            {3, Arch::gfx1030, 16384, 1024},
            13 * 1024, 1024, 256};
}

void test_disjoint_text_card_and_memory() {
    const FourCardPlan good = mock_plan();
    validate_four_card_plan(good);
    auto p = good;
    p.text.id = p.dit[1].id;
    must_refuse([&] { validate_four_card_plan(p); });
    p = good;
    p.dit[2].id = p.dit[0].id;
    must_refuse([&] { validate_four_card_plan(p); });
    p = good;
    p.text.arch = Arch::gfx1100;
    must_refuse([&] { validate_four_card_plan(p); });
    p = good;
    p.text_workspace_mib = 3072;
    must_refuse([&] { validate_four_card_plan(p); });
    p = good;
    p.text_workspace_mib = 0; // unmeasured working set may not be silently zero
    must_refuse([&] { validate_four_card_plan(p); });
    p = good;
    p.dit[2].reserved_mib = p.dit[2].vram_mib;
    must_refuse([&] { validate_four_card_plan(p); });
}

void test_conditioning_transfer() {
    // One VCOND: positive + negative contexts, 1024 rows each, bf16.
    // Gemma 4 48 layers + embedding, 3840 channels: full stack is ~735 MiB.
    assert(gemma_stack_bytes(1024, 2, 3840, 48, 2) == 770703360LL);
    must_refuse([] { gemma_stack_bytes(1024, 2, 3840, 0, 2); });
    assert(conditioning_bytes(1024, 2, 4096, 2048, 2) == 24LL * 1024 * 1024);
    must_refuse([] { conditioning_bytes(1024, 0, 4096, 2048, 2); });
    const auto hi=std::numeric_limits<int>::max();
    must_refuse([&] { gemma_stack_bytes(hi,hi,hi,hi,hi); });
    must_refuse([&] { conditioning_bytes(hi,hi,hi,hi,hi); });
    must_refuse([&] { attention_slices(hi,hi,{hi,hi,hi}); });
}

int main() {
    test_head_ownership();
    test_fused_qkv_bank_mapping();
    test_iq8r_alignment();
    test_all_head_rms();
    test_disjoint_text_card_and_memory();
    test_conditioning_transfer();
    std::cout << "LTX TP3 card-free reference: 6 test groups passed\n";
}
