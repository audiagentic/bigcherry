/* C01: compile the same native C01Slice header used by the LTX patch.
 * This exercises actual opt-in admission metadata, not a shadow geometry
 * implementation, and does not enable multi-rank execution.
 */
#include "bc_ltx_tp3_c01.h"
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>

using ltx2::C01Slice;
constexpr int64_t group = 256; // IQ8R test-fixture scale group
namespace {
void rows(int64_t bank, int64_t hd, std::array<int, 3> shares, int n) {
    int64_t cursor = 0;
    for (int heads : shares) {
        const int64_t width = heads * hd;
        const auto spec = C01Slice::rows(bank, cursor, width, hd, n);
        assert(spec.valid(n * width, bank, group));
        assert(!spec.valid(n * width + 1, bank, group));
        assert(!spec.valid(n * width, bank - 1, group)); // no IQ8R 256-K group
        cursor += width;
    }
    assert(cursor == bank);
}
void columns(int64_t K, std::array<int, 3> shares) {
    int64_t cursor = 0;
    bool any_bad = false;
    for (int heads : shares) {
        const int64_t width = heads * (K / 32);
        const auto spec = C01Slice::columns(K, cursor, width);
        const bool good = (cursor % 256 == 0) && (width % 256 == 0);
        assert(spec.valid(9, width, group) == good);
        assert(!spec.valid(9, width + 1, group));
        any_bad |= !good;
        cursor += width;
    }
    assert(cursor == K);
    if (K == 2048 && shares == std::array<int, 3>{10, 10, 12}) assert(any_bad);
}
void negative() {
    const auto r = C01Slice::rows(4096, 1024, 1024, 128, 3);
    assert(r.valid(3072, 4096, group));
    assert(!C01Slice::rows(4096, 1025, 1024, 128, 3).valid(3072, 4096, group));
    assert(!C01Slice::rows(4096, 1024, 513, 128, 3).valid(1539, 4096, group));
    assert(!C01Slice::rows(4096, 1024, 1024, 0, 3).valid(3072, 4096, group));
    assert(!C01Slice::rows(4096, 1024, 1024, 128, 1).valid(1024, 4096, group));
    assert(!C01Slice::rows(4096, 4090, 1024, 128, 3).valid(3072, 4096, group));
    assert(!C01Slice::columns(2048, 640, 640).valid(7, 640, group));
    assert(!C01Slice::columns(2048, 512, 256).valid(7, 512, group));
    assert(!C01Slice::columns(2048, 2048, 256).valid(7, 256, group));
    assert(!C01Slice::columns(2050, 512, 256).valid(7, 256, group));
    // A biased output is admitted only with an explicit deferred-bias contract.
    const auto defer = C01Slice::columns_deferred(4096, 1024, 1024);
    assert(defer.valid(4096, 1024, group));
    assert(defer.defer_bias);
    assert(!C01Slice::columns_deferred(2048, 640, 640).valid(2048, 640, group));
    const auto huge = std::numeric_limits<int64_t>::max();
    assert(!C01Slice::rows(huge, 0, huge, 128, 3).valid(42, 4096, group));
}
} // namespace

int main() {
    for (auto shares : {std::array<int,3>{8,8,16},
                        std::array<int,3>{12,12,8},
                        std::array<int,3>{10,10,12}}) {
        rows(4096, 128, shares, 3);
        rows(4096, 128, shares, 2);
        rows(2048, 64, shares, 3);
        rows(2048, 64, shares, 2);
        columns(4096, shares);
        columns(2048, shares);
        columns(8192, shares);
        columns(16384, shares);
    }
    negative();
    std::cout << "LT3 C01 native LTX builder-shard geometry: passed\n";
}
