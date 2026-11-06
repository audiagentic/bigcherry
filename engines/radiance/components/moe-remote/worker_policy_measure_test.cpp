#include "worker_policy_measure.h"
#include "worker_policy_effect.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace r3::policy;
static void check(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "r3_policy_measure_test: FAILED %s\n", why); std::exit(1); }
}

static void test_series() {
    Series s(5);
    check(s.count() == 0 && s.quantile(0.5) == 0 && s.max() == 0, "an empty series answers zero");
    for (double x : {5.0, 1.0, 4.0, 2.0, 3.0}) s.add(x);
    check(s.quantile(0.0) == 1 && s.quantile(0.5) == 3 && s.quantile(1.0) == 5, "nearest-rank quantiles");
    check(s.quantile(0.95) == 5 && s.quantile(0.2) == 1 && s.quantile(0.21) == 2, "rank rounds up");
    s.add(10.0);   // the oldest (5) leaves
    check(s.count() == 5 && s.total() == 6 && s.max() == 10 && s.quantile(0.0) == 1, "bounded: the oldest sample leaves");
    s.add(std::nan(""));
    s.add(INFINITY);
    check(s.count() == 5 && s.total() == 6, "non-finite samples are not recorded");
    check(s.quantile(-0.1) == 0 && s.quantile(1.1) == 0, "an out-of-range quantile answers zero");
}

static void test_throughput() {
    Throughput t;
    check(t.bytes_per_ns() == 0, "no transfer is zero bandwidth");
    t.add(1000, 0);
    check(t.bytes == 0 && t.ns == 0, "a zero-length transaction is not a sample");
    t.add(3000, 1000);
    t.add(1000, 1000);
    check(std::abs(t.bytes_per_ns() - 2.0) < 1e-12, "bandwidth is the byte total over the time total");
}

static void test_topk_overlap() {
    const std::vector<double> a = {9, 8, 7, 1, 1, 1}, b = {8, 9, 7, 1, 1, 1}, c = {1, 1, 1, 9, 8, 7};
    check(topk_overlap(a, a, 3) == 1.0, "the same heat is the same hot set");
    check(topk_overlap(a, b, 3) == 1.0, "the order of heat inside the set does not matter");
    check(topk_overlap(a, c, 3) == 0.0, "a disjoint hot set has zero overlap");
    check(std::abs(topk_overlap(a, {9, 8, 1, 7, 1, 1}, 3) - 2.0 / 3.0) < 1e-12, "one expert replaced is two thirds");
    check(std::abs(topk_overlap(a, {9, 1, 1, 8, 7, 1}, 2) - 0.5) < 1e-12, "half of the top two remain");
    const std::vector<double> ties = {5, 5, 5, 5};
    check(topk_overlap(ties, ties, 2) == 1.0, "ties go to the lower index, so equal heat has equal sets");
    check(topk_overlap(a, b, 0) == 0.0 && topk_overlap(a, b, 7) == 0.0 && topk_overlap(a, {1.0}, 1) == 0.0,
          "bad shapes answer zero");
}

// A hot window of width 4 slides one expert per interval over 40 experts (100 observations an interval). The overlap of
// the 4 hottest of the newest interval with the one j intervals back is (4 - j) / 4: 0.75, 0.5, 0.25, 0.
static std::vector<double> window_at(size_t start) {
    std::vector<double> v(40, 1.0);
    for (size_t i = 0; i < 4; ++i) v[start + i] = 100.0 + double(i);
    return v;
}

static void test_drift_half_life() {
    DriftTracker d(4, 16);
    check(d.half_life_observations() == 0 && d.curve().empty() && d.horizon_observations() == 0, "no interval, no drift measured");
    d.add(window_at(0), 100);
    check(d.half_life_observations() == 0, "one interval is not a drift");
    d.add(window_at(1), 200);
    check(d.half_life_observations() == 0 && d.horizon_observations() == 100,
          "adjacent intervals share 3 of 4: nothing to compare against yet, the horizon is one interval");
    d.add(window_at(2), 300);
    check(d.half_life_observations() == 0 && d.horizon_observations() == 200,
          "the overlap at lag 200 (0.5) is still above half of the nearest lag's 0.75 (0.375)");
    d.add(window_at(3), 400);
    // lags 100, 200, 300: overlaps 0.75, 0.5, 0.25; the target 0.375 lies halfway between lag 200 and lag 300
    check(d.half_life_observations() == 250, "half of the nearest lag's overlap falls at 250 observations");
    check(d.horizon_observations() == 300, "the horizon is the oldest retained interval");
    d.add(window_at(4), 500);
    d.add(window_at(5), 600);
    const auto curve = d.curve();
    check(curve.size() == 5 && curve.front().lag_observations == 500 && curve.front().overlap == 0.0,
          "the oldest interval is the widest lag");
    check(std::abs(curve.back().overlap - 0.75) < 1e-12, "the nearest lag is 100 observations, overlap 3 of 4");
    check(d.half_life_observations() == 250, "later intervals do not move the first crossing");
}

static void test_drift_is_relative_to_sampling_noise() {
    // A steady workload whose two adjacent intervals only half-overlap by sampling noise: a hot set of 4 where each
    // interval shows two of them plus two noise experts. The overlap never falls further, so there is no drift.
    DriftTracker d(4, 8);
    for (size_t i = 0; i < 6; ++i) {
        std::vector<double> v(40, 1.0);
        v[0] = 100; v[1] = 99;                       // the stable core
        v[2 + (i % 2) * 2] = 98; v[3 + (i % 2) * 2] = 97;   // the noisy half alternates
        d.add(v, 100 * (i + 1));
    }
    const auto c = d.curve();
    check(!c.empty() && std::abs(c.back().overlap - 0.5) < 1e-12, "adjacent intervals of a steady workload overlap by one half");
    check(d.half_life_observations() == 0, "a steady workload shows no drift however noisy: the half-life is relative to the noise");
}

static void test_drift_discontinuities() {
    DriftTracker d(4, 4);
    d.add(window_at(0), 100);
    d.add(window_at(1), 100);
    check(d.intervals() == 1, "a counter that did not advance is not a new interval");
    d.add(window_at(1), 200);
    d.add(window_at(2), 300);
    d.add(window_at(3), 400);
    d.add(window_at(4), 500);
    check(d.intervals() == 4, "the interval ring is bounded");
    d.add(window_at(5), 50);
    check(d.intervals() == 1, "a counter that goes back (a source restart) resets the tracker");
    d.add(std::vector<double>(10, 1.0), 60);
    check(d.intervals() == 1, "a different shape resets the tracker");
}

static void test_sampling_error() {
    check(relative_share_error(0, 100) == 0 && relative_share_error(5, 4) == 0, "no information is zero");
    check(std::abs(relative_share_error(100, 10000) - std::sqrt(0.99 / 100.0)) < 1e-12, "multinomial relative error");
    check(relative_share_error(10000, 1000000) < relative_share_error(100, 10000), "more samples, less error");
    std::vector<double> counts(10, 0.0);
    for (size_t i = 0; i < 10; ++i) counts[i] = double(10 - i);   // 10, 9, ..., 1; total 55
    check(std::abs(marginal_relative_error(counts, 3) - relative_share_error(8, 55)) < 1e-12,
          "the marginal expert is the k-th hottest");
    check(marginal_relative_error(counts, 0) == 0 && marginal_relative_error(counts, 11) == 0, "bad k answers zero");
}

static void test_tail_series() {
    TailSamples t;
    t.commit(0.4,0.2);
    check(std::abs(t.gain_ppm.quantile(0.5)-500000.0)<1e-6,
          "committed predicted relative tail gain");
    t.commit(0.2,0.21);
    check(t.gain_ppm.total()==1,"nonpositive gain is not observed improvement");
}

static void test_effect_quiet_series() {
    EffectGate gate({2,2,2.0});
    ServiceSnapshot snapshot{};
    uint64_t now=1000;
    gate.quiet(snapshot,now);
    for(unsigned i=0;i<5;++i) {
        auto& lane=snapshot.lane[1];
        lane.samples+=2;lane.rows+=20;
        lane.time_ns+=uint64_t((i%2 ? 110 : 100)*20);
        now+=100;
        gate.quiet(snapshot,now);
    }
    const EffectSamples e=gate.observed();
    check(e.quiet_comparisons==4 && e.normalized_noise_ns.count()==4,
          "signed quiet-window comparisons, not configured counts");
    check(std::abs(e.sample_noise_ns()-100.0)<1e-9,
          "known per-request sigma from signed quiet delta * sqrt(n/2)");
    check(e.quiet_delta_ns.count()==4 && e.quiet_delta_ns.max()==100,
          "effect noise in ns per sampled request, not fractional noise");
    check(std::abs(e.sampled_requests_per_s-20000000.0)<1e-6,
          "sampled request rate from observed samples and wall clock");
}


static void test_effect_variable_windows() {
    EffectGate gate({2,2,2.0});
    ServiceSnapshot snapshot{};
    uint64_t now=1000;
    gate.quiet(snapshot,now);
    for (unsigned i=0; i<5; ++i) {
        auto& lane=snapshot.lane[1];
        const unsigned n = i%2 ? 8 : 2;
        lane.samples+=n;
        lane.rows+=10*n;
        lane.time_ns+=uint64_t((i%2 ? 110 : 100)*10*n);
        now+=100;
        gate.quiet(snapshot,now);
    }
    const auto e=gate.observed();
    check(e.normalized_noise_ns.count()==4,
          "all variable-sized quiet comparisons retained");
    check(std::abs(e.sample_noise_ns()-100.0/std::sqrt(0.625))<1e-9,
          "normalizer uses both actual adjacent n values, not configured n");
}

int main() {
    test_series();
    test_throughput();
    test_topk_overlap();
    test_drift_half_life();
    test_drift_is_relative_to_sampling_noise();
    test_drift_discontinuities();
    test_sampling_error();
    test_tail_series();
    test_effect_quiet_series();
    test_effect_variable_windows();
    std::puts("r3_policy_measure_test: ok");
}
