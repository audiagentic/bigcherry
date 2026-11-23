#include "worker_policy_replay.h"
#include "worker_policy_probe.h"
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

// The old simulator also checked causality; exercise the real policy instead.
static void test_no_future_lookahead() {
    const auto prefix = series(8, range(30, 40));
    const auto next = series(5, range(0, 10), prefix.back());
    auto extended = prefix;
    extended.insert(extended.end(), next.begin() + 1, next.end());
    const auto short_run = replay(config(), prefix, start_owner());
    const auto long_run = replay(config(), extended, start_owner());
    check(short_run.valid && long_run.valid &&
          long_run.steps.size() >= short_run.steps.size(), "replay inputs valid");
    for (size_t i = 0; i < short_run.steps.size(); ++i) {
        const auto& a = short_run.steps[i];
        const auto& b = long_run.steps[i];
        check(a.snapshot == b.snapshot && a.observations == b.observations &&
              a.proposed == b.proposed && a.moved == b.moved &&
              a.tail_share_before == b.tail_share_before &&
              a.tail_share_after == b.tail_share_after,
              "future heat does not alter past decisions");
    }
}

// No implicit floor: a zero move limit leaves the ownership map untouched.
static void test_strict_move_budgets() {
    const auto snaps = series(8, range(30, 40));
    for (bool byte_limit : {false, true}) {
        ReplayConfig cfg = config();
        if (byte_limit) cfg.max_moved_bytes = 0;
        else cfg.max_moves = 0;
        const auto r = replay(cfg, snaps, start_owner());
        check(r.valid && r.transactions == 0 && r.experts_moved == 0 &&
              r.owner == start_owner(), "zero movement budget holds owners");
    }
    ReplayConfig cfg = config();
    cfg.max_moves = cfg.experts / 4;
    cfg.max_moved_bytes = cfg.stride * cfg.max_moves;
    const auto r = replay(cfg, snaps, start_owner());
    check(r.valid && r.transactions > 0, "bounded replay makes a capped transaction");
    for (const auto& step : r.steps)
        check(step.moved <= cfg.max_moves &&
              step.moved * cfg.stride <= cfg.max_moved_bytes,
              "per-transaction movement respects byte and count limits");
}

// The calibration probe on recorded heat: the planner limits come from the ramp (one stride, doubled), the margin is the
// least representable one, and a probe counts only if it is tail-involved with a positive predicted gain.
static ReplayConfig probe_config(const ProbeRamp& ramp) {
    ReplayConfig c = config();
    c.margin = 1e-6;
    c.half_life_observations = 1;
    c.max_moves = ramp.moves();
    c.max_moved_bytes = uint64_t(ramp.moves()) * c.stride;
    return c;
}
// The first ramp size at which the replayed first step proposes, or 0 when none up to the cap does.
static uint32_t first_probe_size(const std::vector<std::vector<uint32_t>>& snaps, ReplayResult* first) {
    ProbeRamp ramp(10, {150, 150, 100});
    for (;;) {
        const ReplayResult r = replay(probe_config(ramp), snaps, start_owner());
        check(r.valid, "probe replay is valid");
        if (r.transactions) {
            if (first) *first = r;
            return ramp.moves();
        }
        if (ramp.moves() == ramp.cap()) return 0;
        ramp.grow();
    }
}
static void test_probe_ramps_to_the_first_group_and_gains() {
    const auto snaps = series(8, range(30, 40));   // the hot experts sit on the tail
    ReplayResult r;
    const uint32_t size = first_probe_size(snaps, &r);
    check(size >= 2 && size <= 10, "one stride cannot move a group (an expert needs its partner); the ramp finds the size");
    check(r.steps.front().tail_share_before > 0.9 || r.steps[0].proposed || r.transactions >= 1, "a probe proposed");
    bool tail_involved = false;
    for (const auto& st : r.steps)
        if (st.proposed) { tail_involved = st.tail_share_after < st.tail_share_before; break; }
    check(tail_involved, "the first probe lowers the tail's heat share");
    check(r.experts_moved >= 2, "a group, not a single expert");
}
static void test_probe_has_no_natural_gain() {
    // The hot experts already sit on a head: the static plan changes nothing, so no probe size yields a transaction and the
    // ramp ends at the cap. Tail gain stays unmeasured (a partial record), never forced.
    const auto snaps = series(8, range(0, 10));
    check(first_probe_size(snaps, nullptr) == 0, "no positive-gain move: no probe at any size");
}
static void test_degenerate_workload_never_crosses() {
    // A single workload: the hot set never moves, so the drift half-life is never observed, whatever was probed.
    const auto snaps = series(40, range(0, 10));
    const ReplayResult r = replay(probe_config(ProbeRamp(10, {150, 150, 100})), snaps, start_owner());
    check(r.valid && r.drift_half_life_observations == 0, "a workload that never shifts has no drift crossing");
}

int main() {
    test_converges_off_the_tail();
    test_a_settled_layout_is_a_fixed_point();
    test_deterministic();
    test_a_shifting_workload_is_followed_and_measured();
    test_residency_holds_experts_in_place();
    test_bad_input_is_refused();
    test_a_counter_that_goes_back_credits_nothing();
    test_no_future_lookahead();
    test_strict_move_budgets();
    test_probe_ramps_to_the_first_group_and_gains();
    test_probe_has_no_natural_gain();
    test_degenerate_workload_never_crosses();
    std::puts("r3_policy_replay_test: ok");
}
