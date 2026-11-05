#include "worker_policy_replay.h"
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

using namespace r3::policy;
static void check(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "r3_policy_replay_test: FAILED %s\n", why); std::exit(1); }
}

// One layer of 40 experts of 10 bytes on three helpers: two heads holding 15 experts each and a tail holding 10.
static ReplayConfig config() {
    ReplayConfig c;
    c.layers = 1; c.experts = 40; c.stride = 10;
    c.budgets = {150, 150, 100};
    c.shares = {1, 1, 1};
    c.half_life_observations = 1e9;   // no decay inside the test
    c.margin = 0.01;
    c.max_moves = 64; c.max_moved_bytes = 1 << 20;
    c.min_samples = 1;
    c.period_ns = 1'000'000'000ULL;
    c.min_residency_ns = 0;
    c.cadence_snapshots = 1;
    return c;
}

// Experts 0-14 on helper 0, 15-29 on helper 1, 30-39 (the tail) on helper 2.
static std::vector<int32_t> start_owner() {
    std::vector<int32_t> o(40);
    for (int i = 0; i < 40; ++i) o[size_t(i)] = i < 15 ? 0 : i < 30 ? 1 : 2;
    return o;
}

// `hot` experts gain 100 a snapshot, the rest 1.
static std::vector<std::vector<uint32_t>> series(size_t snapshots, const std::vector<size_t>& hot, std::vector<uint32_t> base = {}) {
    std::vector<std::vector<uint32_t>> out;
    std::vector<uint32_t> cur = base.empty() ? std::vector<uint32_t>(40, 0) : base;
    out.push_back(cur);
    for (size_t s = 0; s < snapshots; ++s) {
        for (size_t e = 0; e < 40; ++e) cur[e] += 1;
        for (size_t e : hot) cur[e] += 100;
        out.push_back(cur);
    }
    return out;
}

static std::vector<size_t> range(size_t a, size_t b) { std::vector<size_t> v(b - a); std::iota(v.begin(), v.end(), a); return v; }

static double tail_share(const std::vector<int32_t>& owner, const std::vector<uint32_t>& cum) {
    double total = 0, tail = 0;
    for (size_t i = 0; i < owner.size(); ++i) { total += cum[i]; if (owner[i] == 2) tail += cum[i]; }
    return tail / total;
}

static void test_converges_off_the_tail() {
    const auto snaps = series(8, range(30, 40));   // the ten experts on the tail are the hot ones
    const ReplayResult r = replay(config(), snaps, start_owner());
    check(r.valid && r.transactions >= 1 && r.experts_moved >= 10, "the hot experts leave the tail");
    for (size_t e = 30; e < 40; ++e) check(r.owner[e] != 2, "no hot expert is left on the tail");
    check(tail_share(r.owner, snaps.back()) < 0.05, "the tail ends with the cold experts and almost no heat");
    check(r.steps.front().tail_share_before > 0.9, "the tail held nearly all the heat at the start");
    check(r.steps.back().tail_share_after < 0.05, "the last step ends with almost no heat on the tail");
    size_t on_each[3] = {0, 0, 0};
    for (int32_t o : r.owner) ++on_each[size_t(o)];
    check(on_each[0] <= 15 && on_each[1] <= 15 && on_each[2] <= 10, "every helper stays inside its budget");
    check(r.hot_set_k == 10, "the hot set is as many experts as the tail holds");
}

static void test_a_settled_layout_is_a_fixed_point() {
    const auto snaps = series(8, range(30, 40));
    const ReplayResult first = replay(config(), snaps, start_owner());
    const ReplayResult again = replay(config(), snaps, first.owner);
    check(again.valid && again.transactions == 0 && again.owner == first.owner, "the layout it settled on proposes nothing");
}

static void test_deterministic() {
    const auto snaps = series(8, range(30, 40));
    const ReplayResult a = replay(config(), snaps, start_owner()), b = replay(config(), snaps, start_owner());
    check(a.owner == b.owner && a.transactions == b.transactions && a.experts_moved == b.experts_moved, "same series, same replay");
}

static void test_a_shifting_workload_is_followed_and_measured() {
    // The hot set moves from experts 30-39 to experts 0-9 and then to 20-29, five snapshots each.
    auto a = series(5, range(30, 40));
    auto b = series(5, range(0, 10), a.back());
    auto c = series(5, range(20, 30), b.back());
    std::vector<std::vector<uint32_t>> snaps = a;
    snaps.insert(snaps.end(), b.begin() + 1, b.end());
    snaps.insert(snaps.end(), c.begin() + 1, c.end());
    ReplayConfig cfg = config();
    cfg.half_life_observations = 600;   // decay so a past hot set stops counting
    const ReplayResult r = replay(cfg, snaps, start_owner());
    check(r.valid && r.transactions >= 2, "each shift is followed by a transaction");
    check(r.drift_half_life_observations > 0, "the drift of the hot set was measured");
    check(r.marginal_error > 0 && r.marginal_error < 1, "the marginal expert has a finite sampling error");
}

static void test_residency_holds_experts_in_place() {
    auto snaps = series(4, range(30, 40));
    ReplayConfig cfg = config();
    cfg.min_residency_ns = 3'600'000'000'000ULL;   // an hour: nothing moved may move again within the replay
    ReplayConfig free_cfg = config();
    const ReplayResult held = replay(cfg, snaps, start_owner()), free_run = replay(free_cfg, snaps, start_owner());
    check(held.valid && free_run.valid, "valid replays");
    check(held.experts_moved <= free_run.experts_moved, "a residency never moves more");
    // A second hot set on experts the first transaction just moved cannot move them again within the residency.
    auto more = series(4, range(20, 30), snaps.back());   // the experts the first transaction sent to the tail
    snaps.insert(snaps.end(), more.begin() + 1, more.end());
    const ReplayResult held2 = replay(cfg, snaps, start_owner());
    const ReplayResult free2 = replay(free_cfg, snaps, start_owner());
    check(held2.experts_moved < free2.experts_moved, "residency keeps the just-moved experts where they are");
}

static void test_bad_input_is_refused() {
    const auto snaps = series(4, range(30, 40));
    ReplayConfig cfg = config();
    check(!replay(cfg, {snaps[0]}, start_owner()).valid, "one snapshot is not a series");
    check(!replay(cfg, snaps, std::vector<int32_t>(39, 0)).valid, "an owner map of the wrong size");
    cfg.shares = {1, 1};
    check(!replay(cfg, snaps, start_owner()).valid, "one share for each helper");
    cfg = config(); cfg.budgets = {150};
    check(!replay(cfg, snaps, start_owner()).valid, "a tail needs at least two helpers");
    auto bad = snaps; bad[2].pop_back();
    check(!replay(config(), bad, start_owner()).valid, "a snapshot of the wrong shape");
    auto over = snaps;   // an owner map that overfills helper 0 is a fault, not repaired
    std::vector<int32_t> crowded(40, 0);
    check(!replay(config(), over, crowded).valid, "an owner map over its budget is refused");
}

static void test_a_counter_that_goes_back_credits_nothing() {
    auto snaps = series(4, range(30, 40));
    snaps[3] = snaps[1];   // a restart: the counters fall back
    const ReplayResult r = replay(config(), snaps, start_owner());
    check(r.valid, "a counter regression is tolerated, not a fault");
}

int main() {
    test_converges_off_the_tail();
    test_a_settled_layout_is_a_fixed_point();
    test_deterministic();
    test_a_shifting_workload_is_followed_and_measured();
    test_residency_holds_experts_in_place();
    test_bad_input_is_refused();
    test_a_counter_that_goes_back_credits_nothing();
    std::puts("r3_policy_replay_test: ok");
}
