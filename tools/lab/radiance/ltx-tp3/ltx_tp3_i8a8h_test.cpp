// C00/C01: CPU reference of libr4d had_quant_act_i8g's IQ8R activation rule.
//
// Source contract: r4d_quant_act_fp8.hip, ROT=true/I8=true/SUB=2.
// This is arithmetic and partition testing, NOT the device kernel, layout
// permutation, native BF16 GEMM, or a measured GPU result.
#include "ltx_tp3_reference.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace bc_ltx_tp3_reference;
namespace {
constexpr int kH = 128;
constexpr int kGroup = 256;
struct Codes {
    std::vector<std::int8_t> q;
    std::vector<float> scale;
};

void had128(float* v) {
    for (int h = 1; h < kH; h *= 2)
        for (int i = 0; i < kH; i += h * 2)
            for (int j = i; j < i + h; ++j) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
}
float bf16(float input) {
    if (!std::isfinite(input)) throw std::invalid_argument("nonfinite input");
    const auto bits = std::bit_cast<std::uint32_t>(input);
    const std::uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return std::bit_cast<float>(rounded & 0xffff0000u);
}
int round_i8(float value) {
    // Source's rintf(fminf(fmaxf(value, -127), 127)); FE_TONEAREST.
    const float clamped = std::max(-127.0f, std::min(127.0f, value));
    return static_cast<int>(std::nearbyint(clamped));
}
Codes encode(const std::vector<float>& input) {
    if (input.size() < kGroup || input.size() % kGroup)
        throw std::invalid_argument("IQ8R input must align to 256");
    Codes result;
    result.q.resize(input.size());
    for (std::size_t start = 0; start < input.size(); start += kGroup) {
        float transformed[kGroup];
        for (int i = 0; i < kGroup; ++i) {
            if (!std::isfinite(input[start + i]))
                throw std::invalid_argument("nonfinite source activation");
            transformed[i] = input[start + i];
        }
        had128(transformed);
        had128(transformed + kH);
        float maximum = 0.0f;
        for (float value : transformed) maximum = std::max(maximum, std::fabs(value));
        const float scale = maximum > 0 ? maximum * (1.0f / 127.0f) : 1.0f;
        result.scale.push_back(scale);
        const float inv = 1.0f / scale;
        for (int i = 0; i < kGroup; ++i) {
            const int q = round_i8(transformed[i] * inv);
            assert(q >= -127 && q <= 127);
            result.q[start + i] = static_cast<std::int8_t>(q);
        }
    }
    return result;
}
std::vector<float> signal(int width, int seed) {
    std::vector<float> output(width);
    for (int i = 0; i < width; ++i) {
        const float dynamic = 1.0f + float((i / kH) % 7) * .71f;
        const float x = (float((i * 131 + seed * 17) % 211) - 105.0f) *
            dynamic / 101.0f;
        output[i] = bf16(x);
    }
    return output;
}
template<class F> void refuse(F f) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}
void check_partition(int width, Shares widths) {
    const auto full_input = signal(width, 7);
    const auto full = encode(full_input);
    const auto slices = iq8r_input_slices(width, widths);
    std::vector<std::int8_t> codes;
    std::vector<float> scales;
    for (const auto slice : slices) {
        const std::vector<float> part(
            full_input.begin() + slice.offset,
            full_input.begin() + slice.offset + slice.width);
        const auto local = encode(part);
        codes.insert(codes.end(), local.q.begin(), local.q.end());
        scales.insert(scales.end(), local.scale.begin(), local.scale.end());
    }
    assert(codes == full.q);
    assert(scales.size() == full.scale.size());
    for (std::size_t i = 0; i < scales.size(); ++i)
        assert(std::bit_cast<std::uint32_t>(scales[i]) ==
               std::bit_cast<std::uint32_t>(full.scale[i]));
}
void check_dot() {
    // Synthetic IQ8R W8A8 dot with a BF16 second-level row scale.
    const auto input = signal(4096, 17);
    const auto quant = encode(input);
    const auto cuts = iq8r_input_slices(4096, {1024, 1024, 2048});
    constexpr float row_scale = .125f;
    std::vector<std::int8_t> w(input.size());
    std::vector<std::uint8_t> ws(input.size() / kGroup);
    for (std::size_t i = 0; i < w.size(); ++i)
        w[i] = std::int8_t((i * 73u) % 255u - 127);
    for (std::size_t g = 0; g < ws.size(); ++g)
        ws[g] = std::uint8_t((g * 31u) % 255u + 1u);
    auto shard_dot = [&](int begin, int end) {
        double total = 0;
        for (int i = begin; i < end; i += kGroup) {
            std::int64_t accum = 0;
            for (int j = 0; j < kGroup; ++j)
                accum += int(w[i + j]) * int(quant.q[i + j]);
            total += double(accum) * double(ws[i / kGroup]) *
                     double(row_scale) * double(quant.scale[i / kGroup]);
        }
        return total;
    };
    const double dense = shard_dot(0, 4096);
    double sum = 0;
    for (auto s : cuts) sum += shard_dot(s.offset, s.offset + s.width);
    assert(std::abs(dense - sum) <= 1e-10 * (1.0 + std::abs(dense)));
}
void negative_controls() {
    assert(std::fesetround(FE_TONEAREST) == 0);
    assert(round_i8(2.5f) == 2);
    assert(round_i8(3.5f) == 4);
    assert(round_i8(-2.5f) == -2);
    assert(round_i8(-129) == -127);
    assert(round_i8(129) == 127);
    assert(bf16(1.0f) == 1.0f);
    refuse([]{ encode(std::vector<float>(128, 1.0f)); });
    refuse([]{ encode(std::vector<float>(384, 1.0f)); });
    auto nan = signal(512, 3);
    nan[10] = std::numeric_limits<float>::quiet_NaN();
    refuse([&]{ encode(nan); });
    // An incorrect 128-wide quantizer computes two scales where the
    // kernel's SUB=2 variant requires a single scale over both H128s.
    std::vector<float> mixed(512, .125f);
    for (int i = 128; i < 256; ++i) mixed[i] *= 80;
    const auto correct = encode(mixed);
    assert(correct.scale[0] > correct.scale[1]);
    float block0[kH], block1[kH];
    std::copy(mixed.begin(), mixed.begin() + kH, block0);
    std::copy(mixed.begin() + kH, mixed.begin() + 2 * kH, block1);
    had128(block0);
    had128(block1);
    float a = 0, b = 0;
    for (float x : block0) a = std::max(a, std::fabs(x));
    for (float x : block1) b = std::max(b, std::fabs(x));
    assert(a < b); // independent scales per H128 would be wrong.
    assert(std::abs(correct.scale[0] - a / 127.0f) > 1e-4f);
    refuse([]{ iq8r_input_slices(2048, {640, 640, 768}); });
}
} // namespace

int main() {
    check_partition(2048, {512, 512, 1024});
    check_partition(2048, {768, 768, 512});
    check_partition(4096, {1024, 1024, 2048});
    check_partition(4096, {1280, 1280, 1536});
    check_partition(8192, {2048, 2048, 4096});
    check_dot();
    negative_controls();
    std::cout << "LT3 had_quant_act_i8g CPU reference and TP3 shard invariance: passed\n";
}
