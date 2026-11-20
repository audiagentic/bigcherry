// LT3 component harness: fused QKV, KV, gate, output and FFN sharding.
// Synthetic f64, no GPU or production hooks.
#include "ltx_tp3_reference.h"

#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace bc_ltx_tp3_reference;
namespace {
using Vec = std::vector<double>;
using Mat = std::vector<Vec>;

Mat weights(int n, int k, int seed) {
    Mat w(n, Vec(k));
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < k; ++c)
            w[r][c] = double((r * 17 + c * 31 + seed) % 79 - 39) / 110.0;
    return w;
}
Vec input(int k, int seed) {
    Vec v(k);
    for (int c = 0; c < k; ++c)
        v[c] = double((c * 23 + seed * 7) % 67 - 33) / 47.0;
    return v;
}
double project_row(const Vec& w, const Vec& x) {
    assert(w.size() == x.size());
    double sum = 0;
    for (std::size_t i = 0; i < x.size(); ++i) sum += w[i] * x[i];
    return sum;
}
void near(double a, double b) { assert(std::abs(a - b) < 1e-9 * (1.0 + std::abs(a))); }
template<class F> void refuse(F f) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}
// A fused row projection must cut each Q/K/V (or K/V) bank independently.
void fused(int banks, int hd, Shares shares) {
    const int inner = 32 * hd;
    const int k = 11;
    const Mat w = weights(banks * inner, k, 7 + banks);
    const Vec x = input(k, 4);
    const auto owned = attention_slices(32, hd, shares);
    std::vector<int> seen(banks * inner, 0);
    for (const auto sl : owned) {
        std::vector<double> compact;
        for (int bank = 0; bank < banks; ++bank)
            for (int j = 0; j < sl.width; ++j) {
                const int source = bank * inner + sl.offset + j;
                compact.push_back(project_row(w[source], x));
                seen[source] += 1;
            }
        for (int bank = 0; bank < banks; ++bank)
            for (int j = 0; j < sl.width; ++j)
                near(compact[bank * sl.width + j],
                     project_row(w[bank * inner + sl.offset + j], x));
        if (banks == 3) {
            const auto p = fused_qkv_pieces(inner, sl);
            for (int b = 0; b < 3; ++b) {
                assert(p[b].source_row == b * inner + sl.offset);
                assert(p[b].local_row == b * sl.width);
                assert(p[b].count == sl.width);
            }
        }
    }
    for (int visits : seen) assert(visits == 1);
}
void head_gates(int hd, Shares shares) {
    const auto cut = attention_slices(32, hd, shares);
    const Mat w = weights(32, hd * 32, 42);
    const Vec x = input(hd * 32, 13);
    std::vector<int> seen(32, 0);
    Vec dense(32);
    for (int head = 0; head < 32; ++head)
        dense[head] = project_row(w[head], x);
    Vec reconstructed(32, 0);
    for (const auto slice : cut) {
        Vec compact;
        for (int head = slice.offset / hd; head < (slice.offset + slice.width) / hd; ++head)
            compact.push_back(project_row(w[head], x));
        for (std::size_t i = 0; i < compact.size(); ++i) {
            const int head = slice.offset / hd + static_cast<int>(i);
            reconstructed[head] = compact[i];
            seen[head]++;
        }
    }
    for (int head = 0; head < 32; ++head) {
        assert(seen[head] == 1);
        near(reconstructed[head], dense[head]);
    }
}
void row_parallel(int k, int n, Shares shares, bool has_bias = true) {
    assert(k % 32 == 0);
    const auto slices = iq8r_input_slices(k, {
        shares[0] * k / 32, shares[1] * k / 32, shares[2] * k / 32});
    const Mat w = weights(n, k, 23);
    const Vec x = input(k, 71);
    for (int row = 0; row < n; ++row) {
        const double bias = has_bias ? (row + 1) * .029 : 0.0;
        double sum = bias; // exactly once
        for (const auto sl : slices)
            for (int c = sl.offset; c < sl.offset + sl.width; ++c)
                sum += w[row][c] * x[c];
        near(sum, project_row(w[row], x) + bias);
        const double wrong = sum + 2 * bias;
        if (has_bias) assert(std::abs(sum - wrong) > 0.01);
        else assert(sum == wrong);
    }
}
void negative_alignment() {
    refuse([]{ iq8r_input_slices(2048, {640, 640, 768}); });
    refuse([]{ iq8r_input_slices(4096, {1024, 1024, 1792}); });
    refuse([]{ fused_qkv_pieces(128, {120, 16}); });
    refuse([]{ attention_slices(32, 64, {8, 8, 15}); });
}
} // namespace

int main() {
    for (auto shares : {Shares{8, 8, 16}, Shares{12, 12, 8}, Shares{10, 10, 12}}) {
        for (int hd : {64, 128}) {
            fused(3, hd, shares); // self QKV
            fused(2, hd, shares); // text cross/cross-modal KV
            head_gates(hd, shares);
        }
    }
    for (auto shares : {Shares{8, 8, 16}, Shares{12, 12, 8}}) {
        row_parallel(4096, 9, shares); // video output
        row_parallel(2048, 9, shares); // audio and cross-modal output
    }
    row_parallel(16384, 9, {8, 8, 16}, false); // video FFN-down has no bias
    row_parallel(8192, 9, {10, 10, 12}, true); // audio FFN-down has bias
    negative_alignment();
    std::cout << "LT3 linear banks, gates, output and FFN: passed\n";
}
