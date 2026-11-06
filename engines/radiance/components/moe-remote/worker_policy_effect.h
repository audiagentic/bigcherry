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
#include <functional>
#include <cstdio>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace r3::policy {
struct ServiceTotals {
    uint64_t samples=0, rows=0, time_ns=0;
    uint64_t reference_rows=0, reference_time_ns=0; // all other helpers, row weighted
    bool relative=false; // production must not fall back to absolute on a paused peer
};
struct ServiceSnapshot {
    std::array<ServiceTotals,2> lane{}; // 0 small, 1 large -- NEVER pooled
};
struct ServiceSource {
    virtual ~ServiceSource()=default;
    virtual bool snapshot(ServiceSnapshot* out) const=0;
    // The outage clock starts only after active service publication.
    virtual bool active_publishers() const {return true;}
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

// Helper CPU epoch ticks publish the pre-existing sampled lane counters into
// the helper's v24 shared slot. The two lane classes are read together under a
// single-writer sequence. Every field is atomic to avoid racing a CPU reader;
// none of this is on a GPU/route hot path.
class SharedTailServiceSource final : public ServiceSource {
    const moe_shm::PublishedServiceSlot* slot_=nullptr;
    std::vector<std::pair<uint32_t,const moe_shm::PublishedServiceSlot*>> references_;
    uint32_t tail_helper_=0;
    std::function<void(const std::string&)> diagnostic_;
    mutable std::set<std::string> reported_;
    void report(uint32_t helper,const char* condition,int lane=-1,uint64_t age_ns=0) const {
        const std::string key=std::to_string(helper)+":"+condition+":"+std::to_string(lane);
        if(!reported_.insert(key).second)return; // once per helper, lane and failure type
        std::string message="policy relative service unavailable helper="+std::to_string(helper)+
                            " condition="+condition;
        if(lane>=0)message+=" lane="+std::to_string(lane);
        if(age_ns)message+=" age_ns="+std::to_string(age_ns);
        if(diagnostic_)diagnostic_(message);
        else std::fprintf(stderr,"%s\n",message.c_str());
    }
    bool require_reference_=false;
    uint64_t max_age_ns_=0;
    std::function<uint64_t()> clock_;
public:
    SharedTailServiceSource(const void* map,uint32_t tail_helper,
                            uint64_t max_age_ns,std::function<uint64_t()> clock,
                            uint32_t helper_count=0,
                            std::function<void(const std::string&)> diagnostic={})
        :slot_(moe_shm::published_service(map,tail_helper)),
         tail_helper_(tail_helper),diagnostic_(std::move(diagnostic)),
         require_reference_(helper_count>0),max_age_ns_(max_age_ns),
         clock_(std::move(clock)) {
        // Every enabled helper publishes the same v24 slot on its CPU tick.
        // Read all peers; the sum(time)/sum(rows) is the row-weighted mean.
        for(uint32_t h=0;h<helper_count;++h)
            if(h!=tail_helper)references_.push_back({h,moe_shm::published_service(map,h)});
    }

    static bool publish(moe_shm::PublishedServiceSlot* slot,
                        const ServiceSnapshot& local,uint64_t now_ns) {
        if(!slot || !now_ns)return false;
        __atomic_add_fetch(&slot->sequence,uint64_t{1},__ATOMIC_ACQ_REL);
        __atomic_thread_fence(__ATOMIC_SEQ_CST); // odd visible before counters
        __atomic_store_n(&slot->published_ns,now_ns,__ATOMIC_RELAXED);
        for(unsigned i=0;i<2;++i) {
            __atomic_store_n(&slot->samples[i],local.lane[i].samples,__ATOMIC_RELAXED);
            __atomic_store_n(&slot->rows[i],local.lane[i].rows,__ATOMIC_RELAXED);
            __atomic_store_n(&slot->time_ns[i],local.lane[i].time_ns,__ATOMIC_RELAXED);
        }
        __atomic_add_fetch(&slot->sequence,uint64_t{1},__ATOMIC_RELEASE);
        return true;
    }
    enum class ReadProblem { NONE, NO_SLOT, UNPUBLISHED, TORN, STALE, FUTURE, CLOCK };
    static bool read(const moe_shm::PublishedServiceSlot* slot,
                     const std::function<uint64_t()>& clock,uint64_t max_age_ns,
                     ServiceSnapshot* out,ReadProblem* problem=nullptr,
                     uint64_t* age_ns=nullptr) {
        if(problem)*problem=ReadProblem::NONE;
        if(age_ns)*age_ns=0;
        const auto fail=[&](ReadProblem why,uint64_t age=0) {
            if(problem)*problem=why;
            if(age_ns)*age_ns=age;
            return false;
        };
        if(!slot)return fail(ReadProblem::NO_SLOT);
        if(!out || !clock || !max_age_ns)return fail(ReadProblem::CLOCK);
        for(unsigned attempt=0;attempt<8;++attempt) {
            const uint64_t first=__atomic_load_n(&slot->sequence,__ATOMIC_ACQUIRE);
            if(!first)return fail(ReadProblem::UNPUBLISHED);
            if(first&1)continue;
            const uint64_t published=__atomic_load_n(&slot->published_ns,__ATOMIC_RELAXED);
            ServiceSnapshot data{};
            for(unsigned i=0;i<2;++i) {
                data.lane[i].samples=__atomic_load_n(&slot->samples[i],__ATOMIC_RELAXED);
                data.lane[i].rows=__atomic_load_n(&slot->rows[i],__ATOMIC_RELAXED);
                data.lane[i].time_ns=__atomic_load_n(&slot->time_ns[i],__ATOMIC_RELAXED);
            }
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            const uint64_t last=__atomic_load_n(&slot->sequence,__ATOMIC_ACQUIRE);
            if(first!=last || (last&1))continue;
            if(!published)return fail(ReadProblem::UNPUBLISHED);
            // Read clock AFTER slot; peer publication may race prior clock.
            const uint64_t now_ns=clock();
            if(!now_ns)return fail(ReadProblem::CLOCK);
            if(published>now_ns)return fail(ReadProblem::FUTURE,published-now_ns);
            if(now_ns-published>max_age_ns)return fail(ReadProblem::STALE,now_ns-published);
            *out=data;
            return true;
        }
        return fail(ReadProblem::TORN);
    }
    // Numeric overload permits fixed post-read time in deterministic tests.
    static bool read(const moe_shm::PublishedServiceSlot* slot,
                     uint64_t now_ns,uint64_t max_age_ns,ServiceSnapshot* out,
                     ReadProblem* problem=nullptr,uint64_t* age_ns=nullptr) {
        return read(slot,[now_ns] {return now_ns;},max_age_ns,out,problem,age_ns);
    }
    bool active_publishers() const override {
        if(!clock_)return false;
        ServiceSnapshot tail{};
        if(!read(slot_,clock_,max_age_ns_,&tail))return false;
        bool rows=tail.lane[0].rows || tail.lane[1].rows;
        for(const auto& [helper,slot]:references_) {
            ServiceSnapshot peer{};
            if(!read(slot,clock_,max_age_ns_,&peer))return false;
            rows=rows || peer.lane[0].rows || peer.lane[1].rows;
        }
        // Fresh/coherent slots plus actual routed work, not empty startup.
        return rows;
    }
    bool snapshot(ServiceSnapshot* out) const override {
        if(!clock_ || !out || (require_reference_ && references_.empty()))return false;
        const auto check=[&](const moe_shm::PublishedServiceSlot* slot,uint32_t helper,
                            ServiceSnapshot* value) {
            ReadProblem why=ReadProblem::NONE;
            uint64_t age=0;
            if(read(slot,clock_,max_age_ns_,value,&why,&age))return true;
            const char* condition="invalid clock";
            switch(why) {
            case ReadProblem::NO_SLOT: condition="no slot";break;
            case ReadProblem::UNPUBLISHED: condition="no publication";break;
            case ReadProblem::TORN: condition="torn";break;
            case ReadProblem::STALE: condition="stale";break;
            case ReadProblem::FUTURE: condition="future timestamp";break;
            default: break;
            }
            report(helper,condition,-1,age);
            return false;
        };
        ServiceSnapshot tail{};
        if(!check(slot_,tail_helper_,&tail))return false;
        for(const auto& [helper,slot]:references_) {
            ServiceSnapshot peer{};
            if(!check(slot,helper,&peer))return false;
            for(unsigned i=0;i<2;++i) {
                auto& x=tail.lane[i];
                const auto& y=peer.lane[i];
                // A helper with no sampled work in this lane must not poison
                // the denominator for the *other* helpers in that lane.
                if(!y.rows) {
                    if(x.rows)report(helper,"zero rows",int(i));
                    continue;
                }
                if(!y.time_ns) {
                    report(helper,"zero service time",int(i));
                    return false; // inconsistent published measurement
                }
                if(UINT64_MAX-x.reference_rows<y.rows ||
                   UINT64_MAX-x.reference_time_ns<y.time_ns)return false;
                x.reference_rows+=y.rows;
                x.reference_time_ns+=y.time_ns;
            }
        }
        bool usable=false;
        bool any_tail_rows=false;
        // Never use absolute service in a multi-helper run. A lane without
        // any peer rows is unavailable *for that lane*, not for the whole
        // signal when a different lane has a valid row-weighted reference.
        for(unsigned i=0;i<2;++i) {
            auto& x=tail.lane[i];
            x.relative=require_reference_;
            any_tail_rows=any_tail_rows || x.rows>0;
            if(require_reference_ && x.rows && (!x.reference_rows || !x.reference_time_ns)) {
                report(tail_helper_,"zero reference rows",int(i));
                x=ServiceTotals{};
                x.relative=true;
            } else if(x.rows && x.time_ns)usable=true;
        }
        if(require_reference_ && any_tail_rows && !usable)return false;
        *out=tail;
        return true;
    }
};

struct EffectConfig {
    uint64_t window_samples=0, min_quiet_windows=0; // window_samples: routed ROWS (v1 field name)
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
    std::array<uint64_t,2> last_window_rows_{};
    std::array<uint64_t,2> first_sample_ns_{},first_row_count_{};
    static bool subtract(const ServiceTotals& a,const ServiceTotals& b,ServiceTotals* delta) {
        if(a.relative!=b.relative || a.samples<b.samples || a.rows<b.rows || a.time_ns<b.time_ns ||
           a.reference_rows<b.reference_rows || a.reference_time_ns<b.reference_time_ns)return false;
        *delta={a.samples-b.samples,a.rows-b.rows,a.time_ns-b.time_ns,
                a.reference_rows-b.reference_rows,a.reference_time_ns-b.reference_time_ns,a.relative};
        return true;
    }
    static double rate(const ServiceTotals& t) {
        // Compare tail ns/row with the row-weighted ns/row of all other
        // helpers, in exactly the same accumulated sample interval.
        if(!t.rows || !t.time_ns)return 0;
        const double tail=double(t.time_ns)/double(t.rows);
        if(t.relative)
            return t.reference_rows && t.reference_time_ns ?
                tail/(double(t.reference_time_ns)/double(t.reference_rows)) : 0;
        return tail; // legacy local-only synthetic sources; production supplies peers
    }
public:
    explicit EffectGate(EffectConfig c):cfg_(c){}
    EffectSamples observed() const { return quiet_stats_; }
    bool valid() const {
        return cfg_.window_samples && cfg_.min_quiet_windows &&
               std::isfinite(cfg_.noise_multiplier) && cfg_.noise_multiplier>0;
    }
    // A quiet pair is TWO consecutive, non-overlapping routed-row windows without
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
                quiet_stats_=EffectSamples{};first_sample_ns_={0,0};first_row_count_={0,0};
                last_window_rows_={0,0};
                continue; // counter reset: recalibrate from scratch
            }
            if(d.rows<cfg_.window_samples)continue;
            const double next=rate(d);
            anchor_[i]=now.lane[i];
            if(next<=0) {valid_last_[i]=false;quiet_pairs_[i]=0;
                quiet_deviations_[i].clear();last_quiet_ns_[i]=0;continue;}
            if (!first_sample_ns_[i]) {
                first_sample_ns_[i]=now_ns;
                first_row_count_[i]=now.lane[i].rows;
            } else if (now_ns>first_sample_ns_[i] &&
                       now.lane[i].rows>first_row_count_[i]) {
                const double rate=double(now.lane[i].rows-first_row_count_[i])*
                                  1000000000.0/double(now_ns-first_sample_ns_[i]);
                quiet_stats_.sampled_rows_per_s=std::max(quiet_stats_.sampled_rows_per_s,rate);
            }
            const double latency=double(d.time_ns)/double(d.rows);
            if (valid_last_[i] && last_window_rows_[i]) {
                const double delta = latency-last_latency_ns_[i];
                quiet_stats_.quiet_delta_ns.add(std::abs(delta));
                // Var(mean_current - mean_previous) = sigma^2 * (1/n_current + 1/n_previous).
                const double scale = std::sqrt(1.0/double(d.rows) +
                                               1.0/double(last_window_rows_[i]));
                quiet_stats_.normalized_noise_ns.add(delta/scale);
                ++quiet_stats_.quiet_comparisons;
            }
            last_latency_ns_[i]=latency;
            last_window_rows_[i]=d.rows;
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
            if(d.rows<cfg_.window_samples)return EffectVerdict::WAITING;
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
