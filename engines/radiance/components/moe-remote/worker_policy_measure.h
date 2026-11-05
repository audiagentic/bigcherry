#pragma once
// RHP16: the measurements the live policy's settings are derived from. Pure data structures, no clock and no thread:
// the driver feeds them at the moments it already observes (a transaction closing, a heat delta arriving) and reads
// them for evidence. Nothing here chooses a number; a derivation consumes these.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <numeric>
#include <vector>

namespace r3::policy {

// The last `capacity` samples with exact nearest-rank quantiles of what is retained.
class Series {
public:
    explicit Series(size_t capacity = 256) : capacity_(capacity ? capacity : 1) {}
    void add(double x) {
        if (!std::isfinite(x)) return;
        if (values_.size() == capacity_) values_.pop_front();
        values_.push_back(x);
        ++total_;
    }
    size_t count() const { return values_.size(); }
    uint64_t total() const { return total_; }   // every sample ever added, retained or not
    double max() const { return values_.empty() ? 0.0 : *std::max_element(values_.begin(), values_.end()); }
    // Nearest-rank quantile, q in [0, 1]; 0 when there is no sample.
    double quantile(double q) const {
        if (values_.empty() || !(q >= 0.0) || q > 1.0) return 0.0;
        std::vector<double> sorted(values_.begin(), values_.end());
        std::sort(sorted.begin(), sorted.end());
        const size_t rank = q == 0.0 ? 1 : size_t(std::ceil(q * double(sorted.size())));
        return sorted[std::min(rank, sorted.size()) - 1];
    }
private:
    size_t capacity_;
    uint64_t total_ = 0;
    std::deque<double> values_;
};

// Bytes moved over wall time, summed: the host-to-VRAM copy bandwidth the transactions actually achieved.
struct Throughput {
    uint64_t bytes = 0, ns = 0;
    void add(uint64_t b, uint64_t elapsed_ns) {
        if (!elapsed_ns) return;
        bytes += b; ns += elapsed_ns;
    }
    double bytes_per_ns() const { return ns ? double(bytes) / double(ns) : 0.0; }
};

// Share of the k hottest experts of `a` that are also among the k hottest of `b`. Ties go to the lower index, so
// the answer is a function of the two vectors alone. 1 means the same set; 0 means disjoint.
inline double topk_overlap(const std::vector<double>& a, const std::vector<double>& b, size_t k) {
    if (a.size() != b.size() || !k || k > a.size()) return 0.0;
    auto top = [&](const std::vector<double>& v) {
        std::vector<size_t> idx(v.size());
        std::iota(idx.begin(), idx.end(), size_t(0));
        std::partial_sort(idx.begin(), idx.begin() + std::ptrdiff_t(k), idx.end(), [&](size_t x, size_t y) {
            return v[x] != v[y] ? v[x] > v[y] : x < y; });
        idx.resize(k);
        std::sort(idx.begin(), idx.end());
        return idx;
    };
    const std::vector<size_t> ta = top(a), tb = top(b);
    std::vector<size_t> common;
    std::set_intersection(ta.begin(), ta.end(), tb.begin(), tb.end(), std::back_inserter(common));
    return double(common.size()) / double(k);
}

// How fast the hot set drifts, in the unit the heat decays in (observations). Keeps up to `capacity` snapshots of the
// heat counts at the observation counts they were taken, and measures the overlap of the k hottest experts between the
// newest snapshot and each older one.
class DriftTracker {
public:
    struct Point { uint64_t lag_observations = 0; double overlap = 0; };
    DriftTracker(size_t k, size_t capacity = 16) : k_(k), capacity_(capacity ? capacity : 1) {}
    // `counts` is the heat now, `observation_count` the monotone observation counter of the source at that moment.
    // A counter that does not advance is not a new snapshot; one that goes back resets the tracker.
    void add(const std::vector<double>& counts, uint64_t observation_count) {
        if (!snapshots_.empty()) {
            if (observation_count < snapshots_.back().observations || counts.size() != snapshots_.back().counts.size()) {
                snapshots_.clear();
            } else if (observation_count == snapshots_.back().observations) {
                return;
            }
        }
        if (snapshots_.size() == capacity_) snapshots_.pop_front();
        snapshots_.push_back({counts, observation_count});
    }
    size_t snapshots() const { return snapshots_.size(); }
    // Overlap of the newest snapshot with each older one, oldest first.
    std::vector<Point> curve() const {
        std::vector<Point> out;
        if (snapshots_.size() < 2) return out;
        const Snapshot& now = snapshots_.back();
        for (size_t i = 0; i + 1 < snapshots_.size(); ++i)
            out.push_back({now.observations - snapshots_[i].observations,
                           topk_overlap(snapshots_[i].counts, now.counts, k_)});
        return out;
    }
    // The lag at which the overlap first falls to one half (linear between the two lags around it), in observations;
    // 0 when no retained snapshot is that far from the newest, i.e. a drift half-life has not been measured yet.
    uint64_t half_life_observations() const {
        const std::vector<Point> c = curve();
        for (size_t i = c.size(); i-- > 0;) {        // newest lag first, widening
            if (c[i].overlap > 0.5) continue;
            if (i + 1 >= c.size()) return c[i].lag_observations;   // already at or below one half at the nearest lag
            const Point hi = c[i + 1], lo = c[i];    // lo.lag > hi.lag, lo.overlap <= 0.5 < hi.overlap
            const double t = (hi.overlap - 0.5) / (hi.overlap - lo.overlap);
            return hi.lag_observations + uint64_t(t * double(lo.lag_observations - hi.lag_observations));
        }
        return 0;
    }
    size_t k() const { return k_; }
private:
    struct Snapshot { std::vector<double> counts; uint64_t observations = 0; };
    size_t k_, capacity_;
    std::deque<Snapshot> snapshots_;
};

// Relative standard error of an expert's share of the heat, from its own count: the multinomial estimate
// sqrt((1 - p) / count) with p = count / total. 0 for no information (zero count or total).
inline double relative_share_error(double count, double total) {
    if (!(count > 0.0) || !(total >= count)) return 0.0;
    return std::sqrt((1.0 - count / total) / count);
}

// The relative sampling error of the marginal expert, the k-th hottest: the one whose placement the policy is
// least sure of. 0 when there are fewer than k experts or no heat.
inline double marginal_relative_error(const std::vector<double>& counts, size_t k) {
    if (!k || k > counts.size()) return 0.0;
    std::vector<double> sorted = counts;
    std::nth_element(sorted.begin(), sorted.begin() + std::ptrdiff_t(k - 1), sorted.end(), std::greater<double>());
    const double total = std::accumulate(counts.begin(), counts.end(), 0.0);
    return relative_share_error(sorted[k - 1], total);
}

// Everything the driver measures for the derivations, in one place so the evidence line and the tests read it alike.
struct Measured {
    Series transaction_ns{64};   // wall time of committed transactions (start to finish of the journal)
    Series abort_ns{64};         // wall time from start to quiescent of aborted ones
    Series delta_gap_ns{256};    // gap between consecutive heat deltas the driver consumed
    Throughput copy;             // bytes of experts moved by committed transactions over their wall time
    uint64_t drift_half_life_observations = 0;
    double marginal_error = 0;
    size_t hot_set_k = 0;
};

}  // namespace r3::policy
