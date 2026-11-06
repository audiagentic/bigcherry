#pragma once
// Measured-effect gate: CPU-only snapshots of OFF-LANE sampled HIP event timing.
// No GPU hot-path store, device/system fence or added HIP query is required.
#include "moe_remote_shm.h"
#include "worker_policy_measure.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>

namespace r3::policy {
struct ServiceTotals {
    uint64_t samples=0, rows=0, time_ns=0;
};
struct ServiceSnapshot {
    std::array<ServiceTotals,2> lane{}; // 0 small, 1 large -- NEVER pooled
};
struct ServiceSource {
    virtual ~ServiceSource()=default;
    virtual bool snapshot(ServiceSnapshot* out) const=0;
};

// Each lane has exactly one collector thread. Its sequence and three relaxed
// counters are published only after a sampled request has completed and the
// existing HIP timing and d_counts reads have returned. The policy thread
// retries a concurrent snapshot (no GPU operation and no shared-memory ABI).
class LaneServiceSource final : public ServiceSource {
    struct alignas(64) Counter {
        std::atomic<uint64_t> sequence{0}, samples{0}, rows{0}, time_ns{0};
    };
    std::array<Counter,r3::moe_shm::kLanes> counters_{};
public:
    void record(uint32_t lane,uint64_t routed_rows,double stage_us) {
        if(lane>=counters_.size() || !routed_rows || !std::isfinite(stage_us) ||
           stage_us<=0 || stage_us>double(std::numeric_limits<int64_t>::max()/1000))
            return;
        const uint64_t ns=uint64_t(std::llround(stage_us*1000.0));
        if(!ns)return;
        auto& c=counters_[lane];
        c.sequence.fetch_add(1,std::memory_order_acq_rel); // single writer; publish odd BEFORE fields
        c.time_ns.fetch_add(ns,std::memory_order_relaxed);
        c.rows.fetch_add(routed_rows,std::memory_order_relaxed);
        c.samples.fetch_add(1,std::memory_order_relaxed);
        c.sequence.fetch_add(1,std::memory_order_release);
    }
    bool snapshot(ServiceSnapshot* out) const override {
        if(!out)return false;
        ServiceSnapshot result{};
        for(uint32_t lane=0;lane<counters_.size();++lane) {
            const auto& c=counters_[lane];
            bool stable=false;
            ServiceTotals value{};
            for(unsigned retry=0;retry<8;++retry) {
                const uint64_t first=c.sequence.load(std::memory_order_acquire);
                if(first&1)continue;
                value.samples=c.samples.load(std::memory_order_relaxed);
                value.rows=c.rows.load(std::memory_order_relaxed);
                value.time_ns=c.time_ns.load(std::memory_order_relaxed);
                const uint64_t last=c.sequence.load(std::memory_order_acquire);
                if(first==last && !(last&1)) {stable=true;break;}
            }
            if(!stable)return false; // no false verdict from a torn read
            auto& dst=result.lane[r3::moe_shm::lane_is_small(lane)?0:1];
            if(UINT64_MAX-dst.samples<value.samples ||
               UINT64_MAX-dst.rows<value.rows ||
               UINT64_MAX-dst.time_ns<value.time_ns)return false;
            dst.samples+=value.samples;dst.rows+=value.rows;dst.time_ns+=value.time_ns;
        }
        *out=result;
        return true;
    }
};

struct EffectConfig {
    uint64_t window_samples=0, min_quiet_windows=0;
    double noise_multiplier=0;
};
enum class EffectVerdict { WAITING, KEPT, WORSE, UNDECIDED, INVALID };

class EffectGate {
    EffectConfig cfg_;
    std::array<ServiceTotals,2> anchor_{},post_anchor_{};
    std::array<double,2> last_rate_{},pre_rate_{},noise_{};
    std::array<uint64_t,2> quiet_pairs_{},last_quiet_ns_{};
    std::array<std::deque<double>,2> quiet_deviations_{};
    std::array<bool,2> valid_last_{},pre_active_{};
    bool anchor_valid_=false,trial_=false,post_=false;
    double last_ratio_=0, judged_noise_=0;
    EffectSamples quiet_stats_;
    std::array<double,2> last_latency_ns_{};
    std::array<uint64_t,2> last_window_samples_{};
    std::array<uint64_t,2> first_sample_ns_{},first_sample_count_{};
    static bool subtract(const ServiceTotals& a,const ServiceTotals& b,ServiceTotals* delta) {
        if(a.samples<b.samples || a.rows<b.rows || a.time_ns<b.time_ns)return false;
        *delta={a.samples-b.samples,a.rows-b.rows,a.time_ns-b.time_ns};
        return true;
    }
    static double rate(const ServiceTotals& t) {
        // The measured quiet sigma and predicted committed effect are ns per
        // sampled request, NOT ns per routed row. Comparing per-row rates
        // would manufacture regressions when the request's row mix changes.
        return t.samples && t.time_ns ? double(t.time_ns)/double(t.samples) : 0;
    }
public:
    explicit EffectGate(EffectConfig c):cfg_(c){}
    EffectSamples observed() const { return quiet_stats_; }
    bool valid() const {
        return cfg_.window_samples && cfg_.min_quiet_windows &&
               std::isfinite(cfg_.noise_multiplier) && cfg_.noise_multiplier>0;
    }
    // A quiet pair is TWO consecutive, non-overlapping sample windows without
    // a proposal or an unclosed transaction. Worst measured absolute fractional
    // change over the most recent N pairs (per lane class) is an observed
    // conservative noise envelope, without an indefinitely sticky outlier.
    void quiet(const ServiceSnapshot& now,uint64_t now_ns) {
        if(trial_)return;
        if(!anchor_valid_) {
            for(unsigned i=0;i<2;++i)anchor_[i]=now.lane[i];
            anchor_valid_=true;return;
        }
        for(unsigned i=0;i<2;++i) {
            ServiceTotals d;
            if(!subtract(now.lane[i],anchor_[i],&d)) {
                anchor_[i]=now.lane[i];valid_last_[i]=false;quiet_pairs_[i]=0;
                quiet_deviations_[i].clear();last_quiet_ns_[i]=0;
                quiet_stats_=EffectSamples{};first_sample_ns_={0,0};first_sample_count_={0,0};
                last_window_samples_={0,0};
                continue; // counter reset: recalibrate from scratch
            }
            if(d.samples<cfg_.window_samples)continue;
            const double next=rate(d);
            anchor_[i]=now.lane[i];
            if(next<=0) {valid_last_[i]=false;quiet_pairs_[i]=0;
                quiet_deviations_[i].clear();last_quiet_ns_[i]=0;continue;}
            if (!first_sample_ns_[i]) {
                first_sample_ns_[i]=now_ns;
                first_sample_count_[i]=now.lane[i].samples;
            } else if (now_ns>first_sample_ns_[i] &&
                       now.lane[i].samples>first_sample_count_[i]) {
                const double rate=double(now.lane[i].samples-first_sample_count_[i])*
                                  1000000000.0/double(now_ns-first_sample_ns_[i]);
                quiet_stats_.sampled_requests_per_s=std::max(quiet_stats_.sampled_requests_per_s,rate);
            }
            const double latency=double(d.time_ns)/double(d.samples);
            if (valid_last_[i] && last_window_samples_[i]) {
                const double delta = latency-last_latency_ns_[i];
                quiet_stats_.quiet_delta_ns.add(std::abs(delta));
                // Var(mean_current - mean_previous) = sigma^2 * (1/n_current + 1/n_previous).
                const double scale = std::sqrt(1.0/double(d.samples) +
                                               1.0/double(last_window_samples_[i]));
                quiet_stats_.normalized_noise_ns.add(delta/scale);
                ++quiet_stats_.quiet_comparisons;
            }
            last_latency_ns_[i]=latency;
            last_window_samples_[i]=d.samples;
            if(valid_last_[i]) {
                const double fractional=std::abs(next/last_rate_[i]-1.0);
                if(std::isfinite(fractional)) {
                    auto& recent=quiet_deviations_[i];
                    recent.push_back(fractional);
                    if(recent.size()>cfg_.min_quiet_windows)recent.pop_front();
                    quiet_pairs_[i]=recent.size();
                    noise_[i]=*std::max_element(recent.begin(),recent.end());
                }
            }
            last_rate_[i]=next;
            valid_last_[i]=true;
            last_quiet_ns_[i]=now_ns;
        }
    }
    bool ready(uint64_t now_ns,uint64_t freshness_ns) const {
        if(trial_)return false;
        for(unsigned i=0;i<2;++i)
            if(valid_last_[i] && quiet_pairs_[i]>=cfg_.min_quiet_windows &&
               last_quiet_ns_[i] && now_ns>=last_quiet_ns_[i] &&
               now_ns-last_quiet_ns_[i]<=freshness_ns)return true;
        return false;
    }
    bool begin(uint64_t now_ns,uint64_t freshness_ns) {
        if(!ready(now_ns,freshness_ns))return false;
        pre_active_={false,false};
        for(unsigned i=0;i<2;++i)
            if(valid_last_[i] && quiet_pairs_[i]>=cfg_.min_quiet_windows &&
               last_quiet_ns_[i] && now_ns>=last_quiet_ns_[i] &&
               now_ns-last_quiet_ns_[i]<=freshness_ns) {
                pre_active_[i]=true;pre_rate_[i]=last_rate_[i];
            }
        trial_=true;post_=false;return true;
    }
    void committed(const ServiceSnapshot& now) {
        if(!trial_)return;
        for(unsigned i=0;i<2;++i)post_anchor_[i]=now.lane[i];
        post_=true;
    }
    void clear_quiet() {
        anchor_valid_=false;last_quiet_ns_={0,0};
        valid_last_={false,false};quiet_pairs_={0,0};
        quiet_deviations_[0].clear();quiet_deviations_[1].clear();
        last_rate_={0,0};noise_={0,0};
    }
    void abort() {trial_=false;post_=false;clear_quiet();}
    bool waiting() const {return trial_;}
    double noise_floor() const {
        double n=0;
        bool measured=false;
        for(unsigned i=0;i<2;++i) {
            if(trial_?pre_active_[i]:(quiet_pairs_[i]>=cfg_.min_quiet_windows &&
                                    valid_last_[i])) {
                measured=true;n=std::max(n,noise_[i]);
            }
        }
        return measured?n:judged_noise_;
    }
    double last_ratio() const {return last_ratio_;}
    EffectVerdict judge(const ServiceSnapshot& now) {
        if(!trial_ || !post_)return EffectVerdict::WAITING;
        bool worse=false,better=false;
        double max_ratio=0;
        for(unsigned i=0;i<2;++i) {
            if(!pre_active_[i])continue;
            ServiceTotals d;
            if(!subtract(now.lane[i],post_anchor_[i],&d))return EffectVerdict::INVALID;
            if(d.samples<cfg_.window_samples)return EffectVerdict::WAITING;
            const double after=rate(d);
            if(after<=0 || pre_rate_[i]<=0)return EffectVerdict::INVALID;
            const double ratio=after/pre_rate_[i];
            if(!std::isfinite(ratio))return EffectVerdict::INVALID;
            max_ratio=std::max(max_ratio,ratio);
            const double threshold=cfg_.noise_multiplier*noise_[i];
            if(ratio>1+threshold)worse=true;
            if(ratio<1-threshold)better=true;
        }
        last_ratio_=max_ratio;
        judged_noise_=noise_floor();
        trial_=false;post_=false;clear_quiet();
        if(worse)return EffectVerdict::WORSE; // a slower lane vetoes a faster one
        return better?EffectVerdict::KEPT:EffectVerdict::UNDECIDED;
    }
};
} // namespace r3::policy
