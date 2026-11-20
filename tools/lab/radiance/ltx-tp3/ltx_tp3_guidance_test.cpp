// LT3 component harness: each guidance evaluation carries identical rank schedules.
// Synthetic trace model; not an engine-level deadlock or timing proof.
#include "ltx_tp3_protocol.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

using namespace bc_ltx_tp3_protocol;
namespace {
template<class F> void refuse(F f) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}
struct Pass {
    std::uint64_t request;
    std::uint64_t epoch;
    int evaluation;
    bool stg;
    bool iso;
};
struct PassKey {
    std::uint64_t request;
    std::uint64_t epoch;
    int evaluation;
    auto operator<=>(const PassKey&) const = default;
};
void simulate(const std::vector<Pass>& passes) {
    assert(!passes.empty());
    std::set<PassKey> started;
    std::uint64_t previous_epoch = 0;
    for (const auto& p : passes) {
        if (!p.request || !p.epoch || p.evaluation < 0 ||
            p.epoch < previous_epoch ||
            !started.insert({p.request, p.epoch, p.evaluation}).second)
            throw std::invalid_argument("stale, repeated, or unordered guidance pass");
        previous_epoch = p.epoch;
        const auto trace = collective_trace(p.stg, p.iso);
        assert(!trace.empty());
        require_agreement({trace, trace, trace});
        assert(trace.size() == std::size_t(960 - (p.stg ? 4 : 0) - (p.iso ? 288 : 0)));
        if (p.stg)
            for (int kind : {0, 1})
                for (int op : {1, 2}) {
                    const int id = kStgBlock * 100 + kind * 10 + op;
                    assert(std::find(trace.begin(), trace.end(), id) == trace.end());
                }
        if (p.iso) {
            for (int block = 0; block < 48; ++block)
                for (int kind : {4, 5}) {
                    const int id = block * 100 + kind * 10 + 3;
                    assert(std::find(trace.begin(), trace.end(), id) == trace.end());
                }
        }
        auto broken = trace;
        broken.pop_back();
        refuse([&] { require_agreement({trace, broken, trace}); });
    }
}
} // namespace
int main() {
    // Both conditional and unconditional evaluations of two diffusion steps;
    // STG and isolation may be enabled in different guidance branches.
    simulate({{3, 1, 0, false, false}, {3, 1, 1, true, false},
              {3, 1, 2, false, true}, {3, 1, 3, true, true},
              {3, 2, 0, true, true}, {3, 2, 1, false, false}});
    refuse([] { simulate({{3, 1, 0, false, false},
                         {3, 1, 0, false, false}}); });
    refuse([] { simulate({{3, 2, 0, false, false},
                         {3, 1, 1, false, false}}); });
    refuse([] { simulate({{0, 1, 0, false, false}}); });
    std::cout << "LT3 guided-pass rank schedules: passed\n";
}
