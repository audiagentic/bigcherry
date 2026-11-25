/* C01: compile the same native C01Slice header used by the LTX patch.
 * This exercises actual opt-in admission metadata, not a shadow geometry
 * implementation, and does not enable multi-rank execution.
 */
#include "bc_ltx_tp3_c01.h"
#include <algorithm>
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
    // One, two or three fused banks are valid (the FFN's single bank since #1407); none or four are not.
    assert(C01Slice::rows(4096, 1024, 1024, 128, 1).valid(1024, 4096, group));
    assert(!C01Slice::rows(4096, 1024, 1024, 128, 1).valid(2048, 4096, group));   // n_out is banks x width
    assert(!C01Slice::rows(4096, 1024, 1024, 128, 0).valid(0, 4096, group));
    assert(!C01Slice::rows(4096, 1024, 1024, 128, 4).valid(4096, 4096, group));
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
// A group view must cover each head exactly once, independent of the
// physical device order. C01Slice remains the single production
// geometry authority; RUM06 owns admission and group lifecycle.
void ownership() {
    constexpr int64_t hd=64, full=2048;
    const std::array<C01Slice,3> owners{
        C01Slice::rows(full,    0,  8*hd, hd, 3),
        C01Slice::rows(full, 8*hd,  8*hd, hd, 3),
        C01Slice::rows(full,16*hd, 16*hd, hd, 3),
    };
    const auto unique = [](const std::array<C01Slice,3>& ranks) {
        std::array<int,32> heads{};
        for (const auto& slice : ranks) {
            if (!slice.valid(3*slice.width, 2048, group)) return false;
            for (int64_t i=slice.offset/slice.head_dim;
                 i<(slice.offset+slice.width)/slice.head_dim; ++i)
                ++heads[static_cast<size_t>(i)];
        }
        return std::all_of(heads.begin(),heads.end(),
                           [](int n) { return n==1; });
    };
    assert(unique(owners));
    assert(unique({owners[2],owners[0],owners[1]})); // alternate rank order
    assert(!unique({owners[0],owners[1],owners[1]})); // duplicated ownership
    assert(!unique({owners[0],owners[1],
                    C01Slice::rows(full,16*hd,0,hd,3)})); // zero width/gap
    assert(!C01Slice::columns(2048,0,0).valid(1024,0,group));
    assert(!C01Slice::rows(full,0,0,hd,3).valid(0,full,group));
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
    ownership();
    std::cout << "LT3 C01 native LTX builder-shard geometry: passed\n";
}
