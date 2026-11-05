#include "worker_policy_measure.h"
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

// A hot window of width 4 slides one expert per 100 observations over 40 experts: after a lag of L observations
// the overlap of the 4 hottest is (4 - L/100) / 4, which is one half at 200.
static std::vector<double> window_at(size_t start) {
    std::vector<double> v(40, 1.0);
    for (size_t i = 0; i < 4; ++i) v[start + i] = 100.0 + double(i);
    return v;
}

static void test_drift_half_life() {
    DriftTracker d(4, 16);
    check(d.half_life_observations() == 0 && d.curve().empty(), "no snapshot, no drift measured");
    d.add(window_at(0), 1000);
    check(d.half_life_observations() == 0, "one snapshot is not a drift");
    d.add(window_at(1), 1100);
    check(d.half_life_observations() == 0, "at a lag of 100 the overlap is still 3 of 4: no half-life yet");
    d.add(window_at(2), 1200);
    check(d.half_life_observations() == 200, "at a lag of 200 the overlap is exactly one half");
    d.add(window_at(3), 1300);
    check(d.half_life_observations() == 200, "wider lags do not move the first crossing");
    d.add(window_at(4), 1400);
    d.add(window_at(5), 1500);
    const auto curve = d.curve();
    check(curve.size() == 5 && curve.front().lag_observations == 500 && curve.front().overlap == 0.0,
          "the oldest snapshot is the widest lag");
    check(std::abs(curve.back().overlap - 0.75) < 1e-12, "the nearest lag is 100 observations, overlap 3 of 4");
    check(d.half_life_observations() == 200, "one half falls at exactly 200 observations of lag");
}

static void test_drift_discontinuities() {
    DriftTracker d(4, 4);
    d.add(window_at(0), 100);
    d.add(window_at(1), 100);
    check(d.snapshots() == 1, "a counter that did not advance is not a new snapshot");
    d.add(window_at(1), 200);
    d.add(window_at(2), 300);
    d.add(window_at(3), 400);
    d.add(window_at(4), 500);
    check(d.snapshots() == 4, "the snapshot ring is bounded");
    d.add(window_at(5), 50);
    check(d.snapshots() == 1, "a counter that goes back (a source restart) resets the tracker");
    d.add(std::vector<double>(10, 1.0), 60);
    check(d.snapshots() == 1, "a different shape resets the tracker");
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

int main() {
    test_series();
    test_throughput();
    test_topk_overlap();
    test_drift_half_life();
    test_drift_discontinuities();
    test_sampling_error();
    std::puts("r3_policy_measure_test: ok");
}
