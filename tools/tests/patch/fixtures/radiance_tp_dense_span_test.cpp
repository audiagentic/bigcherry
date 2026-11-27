// Card-free test of the actual composed DenseQ::span overload.
#include <arch/rad_linear.h>
#include <cassert>
#include <tuple>
#include <vector>

using Span = std::tuple<rad_weight, int64_t, int64_t>;
static std::vector<Span> spans;
static rad_weight fail_weight = 0;
extern "C" int rad_weight_shard_span(RadBuilder*, rad_weight w, int64_t lo, int64_t hi) {
    spans.emplace_back(w, lo, hi);
    return w == fail_weight ? RAD_E_STATE : RAD_OK;
}
using rad::arch::DenseQ;

static void expect(const DenseQ& d, int shard, int64_t lo, int64_t hi,
                   std::initializer_list<Span> wanted) {
    spans.clear();
    assert(d.span(nullptr, shard, lo, hi) == RAD_OK);
    assert(spans == std::vector<Span>(wanted));
}

int main() {
    DenseQ d;
    d.w = 1;
    d.s = 2;
    d.w8h = true;
    d.g8 = 256;
    expect(d, RAD_SHARD_ROW, 16, 48, {{1, 16, 48}, {2, 16, 48}});
    expect(d, RAD_SHARD_COL, 256, 768, {{1, 256, 768}, {2, 1, 3}});
    d.w8h_s_full = true;
    expect(d, RAD_SHARD_COL, 256, 768, {{1, 256, 768}, {2, 256, 768}});
    spans.clear();
    assert(d.span(nullptr, RAD_SHARD_COL, 128, 768) == RAD_E_INVAL && spans.empty());
    assert(d.span(nullptr, RAD_SHARD_ROW, 8, 48) == RAD_E_INVAL && spans.empty());
    assert(d.span(nullptr, RAD_SHARD_NONE, 16, 48) == RAD_E_INVAL && spans.empty());
    d.g8 = 0;
    assert(d.span(nullptr, RAD_SHARD_COL, 256, 768) == RAD_E_INVAL && spans.empty());
    d.w8h = false;
    d.mx = true;
    d.ref = 3;
    expect(d, RAD_SHARD_ROW, 16, 48, {{1, 16, 48}, {2, 16, 48}, {3, 16, 48}});
    expect(d, RAD_SHARD_COL, 64, 128, {{1, 64, 128}, {2, 2, 4}, {3, 2, 4}});
    fail_weight = 2;
    spans.clear();
    assert(d.span(nullptr, RAD_SHARD_COL, 64, 128) == RAD_E_STATE && spans.size() == 2);
}
