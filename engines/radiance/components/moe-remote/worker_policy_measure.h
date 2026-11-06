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
    // Population standard deviation of the retained signed observations.
    double stddev() const {
        if (values_.size() < 2) return 0.0;
        const double mean = std::accumulate(values_.begin(), values_.end(), 0.0) / double(values_.size());
        double sum = 0.0;
        for (double x : values_) sum += (x - mean) * (x - mean);
        return std::sqrt(sum / double(values_.size()));
    }
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

// How fast the hot set drifts, in the unit the heat decays in (observations). The input is the route counts of ONE source
// update interval, never the accumulated heat: the accumulated heat is decayed by the half-life this measurement exists
// to inform, so measuring drift on it would measure the setting, not the workload. Keeps up to `capacity` intervals and
// compares the k hottest experts of the newest against each older one. Two intervals of the same traffic do not match
// perfectly (sampling noise), so the half-life is taken relative to the overlap at the nearest lag: the lag at which the
// overlap has fallen to half of what two adjacent intervals share.
class DriftTracker {
public:
    struct Point { uint64_t lag_observations = 0; double overlap = 0; };
    DriftTracker(size_t k, size_t capacity = 16) : k_(k), capacity_(capacity ? capacity : 1) {}
    // `interval_counts` is the route count of the interval that just ended, `observation_count` the monotone
    // observation counter of the source at its end. A counter that does not advance is not a new interval; one that goes
    // back (a source restart) or a different shape resets the tracker.
    void add(const std::vector<double>& interval_counts, uint64_t observation_count) {
        if (!intervals_.empty()) {
            if (observation_count < intervals_.back().observations ||
                interval_counts.size() != intervals_.back().counts.size()) {
                intervals_.clear();
            } else if (observation_count == intervals_.back().observations) {
                return;
            }
        }
        if (intervals_.size() == capacity_) intervals_.pop_front();
        intervals_.push_back({interval_counts, observation_count});
    }
    size_t intervals() const { return intervals_.size(); }
    size_t capacity() const { return capacity_; }
    // Overlap of the newest interval's hot set with each older one's, oldest first.
    std::vector<Point> curve() const {
        std::vector<Point> out;
        if (intervals_.size() < 2) return out;
        const Interval& now = intervals_.back();
        for (size_t i = 0; i + 1 < intervals_.size(); ++i)
            out.push_back({now.observations - intervals_[i].observations,
                           topk_overlap(intervals_[i].counts, now.counts, k_)});
        return out;
    }
    // How far back the retained intervals reach, in observations: a half-life of 0 means "not seen within this".
    uint64_t horizon_observations() const {
        const std::vector<Point> c = curve();
        return c.empty() ? 0 : c.front().lag_observations;
    }
    // The lag at which the overlap has first fallen to half of the nearest lag's overlap (linear between the two lags
    // around the crossing), in observations; 0 when it never fell that far within the horizon.
    uint64_t half_life_observations() const {
        const std::vector<Point> c = curve();
        if (c.empty()) return 0;
        const double target = 0.5 * c.back().overlap;
        for (size_t i = c.size(); i-- > 0;) {        // nearest lag first, widening
            if (c[i].overlap > target) continue;
            if (i + 1 >= c.size()) return c[i].lag_observations;   // the nearest lag itself is at the target (it shares nothing)
            const Point near = c[i + 1], far = c[i];   // far.overlap <= target < near.overlap
            const double t = (near.overlap - target) / (near.overlap - far.overlap);
            return near.lag_observations + uint64_t(t * double(far.lag_observations - near.lag_observations));
        }
        return 0;
    }
    size_t k() const { return k_; }
private:
    struct Interval { std::vector<double> counts; uint64_t observations = 0; };
    size_t k_, capacity_;
    std::deque<Interval> intervals_;
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

// Tail-share gain is measured only from plans that actually commit.
// Between-window changes reflect real workload shifts, not sampling noise.
struct TailSamples {
    Series gain_ppm{64};
    void commit(double baseline,double predicted) {
        if (std::isfinite(baseline) && std::isfinite(predicted) &&
            baseline>predicted && predicted>=0 && baseline>0)
            gain_ppm.add((baseline-predicted)/baseline*1000000.0);
    }
};

// Independent counters from completed quiet windows (never copies of gate
// configuration). The rate counts sampled lane requests, i.e. the same
// observations the effect gate's window consumes.
struct EffectSamples {
    Series quiet_delta_ns{256};
    // Signed quiet-window mean differences normalized by their standard-error
    // scale, restoring per-request dispersion: equal n => delta * sqrt(n/2).
    Series normalized_noise_ns{256};
    uint64_t quiet_comparisons=0;
    double sampled_requests_per_s=0;
    double sample_noise_ns() const { return normalized_noise_ns.stddev(); }
};

// Everything the driver measures for the derivations, in one place so the evidence line and the tests read it alike.
struct Measured {
    Series transaction_ns{64};   // wall time of committed transactions (start to finish of the journal)
    Series abort_ns{64};         // wall time from start to quiescent of aborted ones
    Series delta_gap_ns{256};    // gap between consecutive heat deltas the driver consumed
    TailSamples tail;
    EffectSamples effect;
    uint64_t effect_required_quiet_windows=0; // convergence gate, never a measurement
    uint64_t accepted_heat_deltas=0; // only deltas successfully consumed by the planner
    bool accepted_unwatermarked_heat=false; // sticky across heat baseline resets
    void accepted_heat_delta(bool complete_observations) {
        ++accepted_heat_deltas;
        if (!complete_observations) accepted_unwatermarked_heat=true;
    }
    Series tail_lane_service_ns_per_s{64}; // sampled tail-helper large-lane GPU service / wall second
    Series tail_lane_request_ns{64}; // tail large-lane service ns / completed samples in the same interval
    Series lane_lost_ns{64}; // observed tail lane unavailable from quiesce ACK to reopen
    Series slowest_helper_copy_bps{64}; // minimum achieved READY ACK copy rate across receiving helpers
    Throughput copy;             // bytes of experts moved by committed transactions over their wall time
    uint64_t drift_half_life_observations = 0, drift_horizon_observations = 0;
    double marginal_error = 0;
    size_t hot_set_k = 0, drift_intervals = 0, drift_capacity = 0;
    uint64_t marginal_observations = 0;
    double observation_rate_per_s = 0;
};

}  // namespace r3::policy
