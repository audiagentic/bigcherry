// LT3 component harness: Q and K have independent token rows, shared all-head norm.
// Synthetic f64 reference / float shard result; not device reduction-order parity.
#include "ltx_tp3_reference.h"

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace bc_ltx_tp3_reference;
namespace {
using Vec = std::vector<float>;
using Pack = std::array<Vec, 3>;
template<class F> void refuse(F f) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}
float sample(int i, int row, int phase) {
    const float strength = i < 128 ? .15f : i < 768 ? 3.2f : 1.7f;
    return strength * float((i * 17 + row * 37 + phase * 19) % 91 - 43) / 37.0f;
}
void rows(int width, Shares shares, int nrows, int phase, double eps) {
    const auto cut = attention_slices(32, width / 32, shares);
    for (int row = 0; row < nrows; ++row) {
        Pack x, gain;
        double sq = 0;
        for (int rank = 0; rank < 3; ++rank)
            for (int j = 0; j < cut[rank].width; ++j) {
                const int i = cut[rank].offset + j;
                const float v = sample(i, row, phase);
                x[rank].push_back(v);
                gain[rank].push_back(.6f + float((i * 11 + phase) % 13) / 29.0f);
                sq += double(v) * v;
            }
        const double global = 1.0 / std::sqrt(sq / width + eps);
        const auto distributed = global_rms(x, gain, eps);
        // C05's existing-op assembly: put disjoint shard columns into a
        // zeroed whole-width row, sum rank replicas, full-width RMS,
        // then extract each rank's local columns. Independent of the
        // direct sum-of-squares TP3 oracle above.
        std::array<Vec, 3> padded;
        for (int rank = 0; rank < 3; ++rank) {
            padded[rank].assign(width, 0.0f);
            for (std::size_t j = 0; j < x[rank].size(); ++j)
                padded[rank][cut[rank].offset + j] = x[rank][j];
        }
        Vec assembled(width, 0.0f);
        for (int i = 0; i < width; ++i)
            for (int rank = 0; rank < 3; ++rank)
                assembled[i] += padded[rank][i];
        double ss = 0;
        for (float v : assembled) ss += double(v) * v;
        const double packed_factor = 1.0 / std::sqrt(ss / width + eps);
        for (int rank = 0; rank < 3; ++rank)
            for (std::size_t j = 0; j < x[rank].size(); ++j) {
                const double dense = x[rank][j] * global * gain[rank][j];
                const double packed = assembled[cut[rank].offset + j] *
                                      packed_factor * gain[rank][j];
                assert(std::abs(distributed[rank][j] - dense) < 2e-6 * (1 + std::abs(dense)));
                assert(std::abs(packed - dense) < 2e-6 * (1 + std::abs(dense)));
            }
    }
}
void controls() {
    Pack empty{{{0}, {0}, {0}}}, g{{{1}, {1}, {1}}};
    const auto zero = global_rms(empty, g, 1e-6);
    for (const auto& part : zero) assert(part[0] == 0);
    empty[1][0] = std::numeric_limits<float>::quiet_NaN();
    refuse([&] { global_rms(empty, g, 1e-6); });
    empty[1][0] = std::numeric_limits<float>::infinity();
    refuse([&] { global_rms(empty, g, 1e-6); });
    empty[1][0] = 0;
    g[0][0] = std::numeric_limits<float>::infinity();
    refuse([&] { global_rms(empty, g, 1e-6); });
    g[0][0] = 1;
    refuse([&] { global_rms(empty, g, 0); });
    g[2].clear();
    refuse([&] { global_rms(empty, g, 1e-6); });
}
} // namespace

int main() {
    for (auto shares : {Shares{8, 8, 16}, Shares{12, 12, 8}, Shares{10, 10, 12}}) {
        for (int width : {2048, 4096}) {
            rows(width, shares, 3, 1, 1e-6); // three Q rows
            rows(width, shares, 5, 2, 1e-5); // five K rows (cross attention)
        }
    }
    controls();
    std::cout << "LT3 Q/K global all-head RMS: passed\n";
}
