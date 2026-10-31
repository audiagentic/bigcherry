#include "worker_heat.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace r3::heat;

static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "r3_worker_heat_test: FAILED %s\n", what); std::exit(1); }
}

// 6 experts of 10 bytes; devices 0 (cost 2, 40 B), 1 (cost 1, 40 B), 2 (cost 0.5, 20 B); experts 0-2 on device 0, 3-5 on device 1
static Input fixture() {
    Input in;
    in.bytes = {10, 10, 10, 10, 10, 10};
    in.current = {0, 0, 0, 1, 1, 1};
    in.heat = {100, 90, 80, 70, 60, 50};
    in.devices = {{40, 2.0}, {40, 1.0}, {20, 0.5}};
    in.allowed.assign(6, {1, 1, 1});
    in.margin = 0.1;
    in.max_moves = 2;
    return in;
}

static void test_decay() {
    Heat h(2, 3, 0.5);
    int32_t a[] = {10, 0, 4}, b[] = {2, 1, 0};
    check(h.observe(0, a, 3), "observe");
    check(h.tick(), "tick");
    check(h.observe(0, b, 3, 2.0), "weighted observe");
    check(h.counts[0] == 9.0 && h.counts[1] == 2.0 && h.counts[2] == 2.0, "decayed and weighted counts");
    check(h.counts[3] == 0.0 && h.samples[3] == 0, "another layer is untouched and never observed");
    check(h.samples[0] == 2 && h.samples[1] == 2, "samples count histograms, zero counts included");
    int32_t bad[] = {1, -1, 3};
    check(!h.observe(0, bad, 3) && h.counts[0] == 9.0, "a negative count is refused with no partial update");
    check(!h.observe(2, a, 3) && !h.observe(0, a, 2) && !h.observe(0, nullptr, 3), "bad layer, size or pointer is refused");
    Heat g(1, 1, 1.5);
    check(!g.tick(), "a decay above 1 is refused");
}

static void test_hysteresis() {
    Input in = fixture();
    in.devices = {{40, 2.0}, {40, 1.9}, {20, 1.9}};
    in.max_moves = 6;
    Plan p = propose(in);
    check(p.valid && p.delta.empty(), "a 5% improvement under a 10% margin moves nothing");
    in.margin = 0.04;
    p = propose(in);
    check(p.valid && !p.delta.empty(), "the same costs with a 4% margin move");
    in = fixture();
    in.max_moves = 6;
    p = propose(in);
    check(p.valid && !p.delta.empty(), "a large saving clears the margin");
}

static void test_budgets_and_limits() {
    Input in = fixture();
    in.max_moves = 6;
    Plan p = propose(in);
    check(p.valid && delta_matches(p, in.current), "a valid plan whose delta rebuilds the map");
    for (size_t d = 0; d < in.devices.size(); ++d) check(p.used[d] <= in.devices[d].budget, "no budget exceeded");
    check(p.used[2] == 20, "the cheap device fills to its budget");
    in.max_moves = 1;
    p = propose(in);
    check(p.delta.size() == 1, "the move limit holds");
    in.devices[2].budget = 0;
    p = propose(in);
    check(p.valid && p.used[2] == 0, "a zero-budget device stays empty");
    in = fixture();
    in.devices[0].budget = 10;
    check(!propose(in).valid, "an overfull active map is refused, not repaired");
}

static void test_determinism() {
    Input in = fixture();
    in.max_moves = 6;
    const Plan a = propose(in), b = propose(in);
    check(a.valid && b.valid && a.owner == b.owner && a.used == b.used && a.delta.size() == b.delta.size(), "same input, same plan");
    for (size_t i = 0; i < a.delta.size(); ++i)
        check(a.delta[i].expert == b.delta[i].expert && a.delta[i].to == b.delta[i].to, "same move order");
    in.max_moves = 0;
    const Plan f = propose(in);
    check(f.valid && f.owner == in.current && f.delta.empty(), "zero moves freezes the map");
    // equal savings break on expert index
    Input t = fixture();
    t.heat = {50, 50, 50, 50, 50, 50};
    t.current = {0, 0, 0, 0, 0, 0};
    t.devices = {{60, 2.0}, {10, 1.0}, {0, 0.5}};
    t.max_moves = 1;
    const Plan q = propose(t);
    check(q.delta.size() == 1 && q.delta[0].expert == 0 && q.delta[0].to == 1, "ties break on the lower expert index");
}

static void test_zero_share() {
    std::vector<int64_t> bytes(12, 10);
    uint32_t shares[] = {1, 0, 1};
    uint64_t budgets[] = {40, 0, 20};
    const auto owners = static_tail(bytes, 3, shares, budgets);
    for (int32_t o : owners) check(o != 1, "a head helper with share 0 gets nothing");
    Input in = fixture();
    in.devices[2].budget = 0;
    in.max_moves = 6;
    const Plan p = propose(in);
    check(p.valid && p.used[2] == 0, "a zero-capacity helper receives nothing");
}

static void test_static_tail_identity() {
    std::vector<int64_t> bytes(10, 10);
    uint32_t shares[] = {2, 1, 1};
    uint64_t budgets[] = {30, 20, 20};
    const auto expected = r3::worker_share::plan_tail(bytes, 3, shares, budgets);
    check(static_tail(bytes, 3, shares, budgets) == expected, "static_tail is plan_tail");
    Input in;
    in.bytes.assign(bytes.begin(), bytes.end());
    in.current = expected;
    in.heat.assign(bytes.size(), 100.0);
    in.devices = {{30, 3.0}, {20, 2.0}, {20, 1.0}};
    in.allowed.assign(bytes.size(), {1, 1, 1});
    in.max_moves = 0;
    const Plan p = propose(in);
    check(p.valid && p.owner == expected, "with max_moves 0 the static map is preserved exactly");
}

static void test_protection() {
    Input in = fixture();
    in.allowed[0] = {1, 0, 0};
    in.max_moves = 6;
    Plan p = propose(in);
    check(p.valid && p.owner[0] == 0, "a protected expert stays");
    in.allowed[0] = {0, 1, 1};
    check(!propose(in).valid, "an incumbent on a device it is not allowed on is refused");
    in = fixture();
    in.current[0] = -1;
    in.max_moves = 6;
    p = propose(in);
    check(p.valid && p.owner[0] == -1, "an engine-owned expert is never moved");
    check(!propose(Input{}).valid, "an empty input has no devices and is refused");
}

int main() {
    test_decay();
    test_hysteresis();
    test_budgets_and_limits();
    test_determinism();
    test_zero_share();
    test_static_tail_identity();
    test_protection();
    std::puts("r3_worker_heat_test: ok");
    return 0;
}
