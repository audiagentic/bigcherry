#pragma once
// Stage-1 live placement: one off-lane CPU worker, one v23 Controller.
// The worker only reads copies, never a mailbox, lane, GPU or private pool.
#include "worker_policy_config.h"
#include "worker_policy_live.h"
#include "worker_epoch.h"
#include <condition_variable>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

namespace r3::policy {
class Driver {
public:
    struct Negative {
        uint64_t observed_ns=0, until_ns=0;
        uint32_t reason=0;
    };
    struct Evidence {
        uint64_t evaluated=0, no_change=0, proposed=0, committed=0, aborted=0;
        uint64_t learned_ineligible=0, bytes_moved=0;
        uint64_t transition_samples=0, transition_ns_total=0, transition_ns_max=0;
        double predicted_saving=0;
        bool unwatermarked_heat=false;
        std::map<std::string,uint64_t> reasons;
    };
private:
    struct Snapshot {
        OwnerSnapshot owners;
        std::vector<uint64_t> last_move;
        std::map<std::pair<uint32_t,uint32_t>, Negative> excluded;
        uint64_t now_ns=0, cooldown_until_ns=0;
    };
    struct Result {
        OwnerSnapshot owners;
        heat::Plan plan;
        std::string reason;
        bool evaluated=false, unwatermarked=false;
    };
    Settings settings_;
    std::unique_ptr<HeatSource> source_;
    std::vector<uint64_t> budgets_;
    uint64_t stride_=0;
    std::vector<uint8_t> protected_;
    heat::Heat heat_;
    uint64_t consumed_=0;
    MoveHistory history_;
    bool seeded_=false, inflight_=false;
    uint64_t due_ns_=0, cooldown_until_ns_=0, txn_=0;
    heat::Plan pending_plan_;
    std::map<std::pair<uint32_t,uint32_t>,Negative> negatives_;
    Evidence evidence_;
    std::mutex lock_;
    std::condition_variable cv_;
    bool stop_=false, queued_=false, ready_=false;
    Snapshot job_;
    Result result_;
    std::thread worker_;
    static uint64_t add_saturating(uint64_t a,uint64_t b) {
        return b > UINT64_MAX-a ? UINT64_MAX : a+b;
    }
    void record_no_change(const std::string& why) {
        ++evidence_.no_change;
        ++evidence_.reasons[why.empty()?"unspecified":why];
    }
    Result evaluate(const Snapshot& copy) {
        Result r;
        r.owners=copy.owners;
        timespec rt{};
        if(clock_gettime(CLOCK_REALTIME,&rt)!=0 || rt.tv_sec<0) {
            r.reason="realtime_clock";return r;
        }
        const uint64_t realtime=uint64_t(rt.tv_sec)*1000000000ULL+uint64_t(rt.tv_nsec);
        HeatDelta delta{};
        std::string why;
        const auto status=source_->poll(copy.now_ns,realtime,&delta,&why);
        if(status==HeatPoll::INVALID) {r.reason="heat_invalid";return r;}
        if(status==HeatPoll::BASELINE) {
            // A discontinuity invalidates prior coverage. Rebase, never
            // pretend missing observations were consumed as a fresh delta.
            heat_=heat::Heat(copy.owners.layers,copy.owners.experts,
                             settings_.controls.half_life_observations);
            consumed_=delta.observation_count;
            r.reason="heat_baseline";return r;
        }
        if(status==HeatPoll::UNCHANGED) {r.reason="heat_unchanged";return r;}
        const bool unwatermarked=!delta.complete_observations;
        if(unwatermarked &&
           !(settings_.accept_unwatermarked_heat &&
             delta.provenance==HeatProvenance::ENGINE_HISTOGRAM_FILE)) {
            r.reason="heat_unwatermarked";return r;
        }
        if(delta.age_ns>settings_.controls.max_heat_age_ns) {
            heat_=heat::Heat(copy.owners.layers,copy.owners.experts,
                             settings_.controls.half_life_observations);
            consumed_=delta.observation_count;
            r.reason="heat_stale";return r;
        }
        if(!consume_heat(delta,&heat_,&consumed_,
                         settings_.accept_unwatermarked_heat)) {
            r.reason="heat_invalid_coverage";return r;
        }
        r.unwatermarked=unwatermarked;
        const size_t n=copy.owners.owner.size();
        std::vector<uint64_t> physical(n,stride_);
        std::vector<LedgerView> ledgers;
        if(!derive_active_ledgers(copy.owners,physical,budgets_,stride_,
                                  settings_.costs,settings_.layer_caps,
                                  &ledgers,&why)) {
            r.reason="ledger_invalid";return r;
        }
        std::vector<ExpertView> experts(n);
        for(size_t i=0;i<n;++i) {
            auto& e=experts[i];
            e.bytes=stride_;
            e.last_move_ns=copy.last_move[i];
            e.protected_expert=!copy.owners.runtime[i] || protected_[i];
            e.target_eligible.assign(copy.owners.helpers,0);
            const int32_t from=copy.owners.owner[i];
            if(from<0) continue; // never move engine-local experts
            // Learned eligibility is optimistic. The receiving helper's
            // immutable frozen slot and private physical reservation remain
            // authoritative: v23 fail-closes a refused destination.
            for(uint32_t d=0;d<copy.owners.helpers;++d) {
                const auto it=copy.excluded.find({uint32_t(i),d});
                const bool banned=it!=copy.excluded.end() && copy.now_ns<it->second.until_ns;
                e.target_eligible[d]=uint8_t(d==uint32_t(from) ||
                   (!e.protected_expert && budgets_[d]>=stride_ && !banned));
            }
        }
        heat::Input input;
        if(!build_input(copy.owners,experts,ledgers,heat_,delta,settings_.controls,
                        copy.now_ns,copy.cooldown_until_ns,false,&input,
                        &why,settings_.accept_unwatermarked_heat)) {
            r.reason="input_invalid";return r;
        }
        r.evaluated=true;
        r.plan=heat::propose(input);
        if(!r.plan.valid)r.reason="plan_invalid";
        else if(r.plan.no_change) {
            if(copy.now_ns<copy.cooldown_until_ns)r.reason="abort_cooldown";
            else if(r.plan.baseline==0)r.reason="flat_heat";
            else r.reason="no_improvement_or_margin";
        }
        return r;
    }
    void loop() {
        for(;;) {
            Snapshot copy;
            {
                std::unique_lock<std::mutex> guard(lock_);
                cv_.wait(guard,[&] {return stop_||queued_;});
                if(stop_)return;
                copy=std::move(job_);
                queued_=false;
            }
            Result res=evaluate(copy);
            {
                std::lock_guard<std::mutex> guard(lock_);
                if(stop_)return;
                result_=std::move(res);
                ready_=true;
            }
        }
    }
public:
    Driver(Settings settings,std::unique_ptr<HeatSource> source,
           std::vector<uint64_t> budgets,uint64_t stride,
           std::vector<uint8_t> protected_experts)
        :settings_(std::move(settings)),source_(std::move(source)),
         budgets_(std::move(budgets)),stride_(stride),
         protected_(std::move(protected_experts)),
         heat_(uint32_t(protected_.size()),1,settings_.controls.half_life_observations) {
        // Heat shape is replaced with the coherent ACTIVE layout at first
        // capture; no constructor accesses shared memory.
    }
    ~Driver() { stop(); }
    Driver(const Driver&)=delete;
    Driver& operator=(const Driver&)=delete;
    bool start() {
        if(!settings_.enabled)return true;
        if(!source_ || !stride_ || budgets_.empty() || !controls_valid(settings_.controls) ||
           settings_.cadence_ns==0 || settings_.abort_cooldown_ns==0 ||
           settings_.learned_eligibility_ttl_ns==0 ||
           budgets_.size()!=settings_.costs.size() ||
           budgets_.size()!=settings_.layer_caps.size()) return false;
        worker_=std::thread([this]{loop();});
        return true;
    }
    bool running() const {return worker_.joinable();}
    void stop() {
        {
            std::lock_guard<std::mutex> guard(lock_);
            stop_=true;
            queued_=false;
        }
        cv_.notify_all();
        if(worker_.joinable())worker_.join();
    }
    const Evidence& evidence() const {return evidence_;}
    size_t negative_count(uint64_t now_ns) const {
        size_t n=0;
        for(const auto& [key,entry]:negatives_)if(now_ns<entry.until_ns)++n;
        return n;
    }
    bool submit(void* map,r3::epoch::Controller& controller,uint64_t now_ns) {
        if(!settings_.enabled || !controller.can_propose() || inflight_)return true;
        // No shared-map walk on every 2 ms control-plane tick while the CPU
        // job is pending, before cadence, or during a post-abort cooldown.
        {
            std::lock_guard<std::mutex> guard(lock_);
            if(!ready_ && (queued_ || now_ns<due_ns_ ||
                           now_ns<cooldown_until_ns_))return true;
        }
        OwnerSnapshot active;
        if(!capture_owners(map,&active)) {record_no_change("active_snapshot");return true;}
        if(active.owner.size()!=protected_.size() || active.helpers!=budgets_.size())
            {record_no_change("shape_mismatch");return true;}
        if(!seeded_) {
            if(!history_.reset(active,controller.control().transaction_id))return false;
            heat_=heat::Heat(active.layers,active.experts,
                             settings_.controls.half_life_observations);
            seeded_=true;
        }
        Result result;
        bool got=false;
        {
            std::lock_guard<std::mutex> guard(lock_);
            if(ready_) {result=std::move(result_);ready_=false;got=true;}
        }
        if(got) {
            if(result.unwatermarked)evidence_.unwatermarked_heat=true;
            if(result.evaluated)++evidence_.evaluated;
            if(!result.reason.empty())record_no_change(result.reason);
            else if(result.owners.epoch!=active.epoch ||
                    result.owners.digest!=active.digest ||
                    result.owners.owner!=active.owner)record_no_change("stale_owner_snapshot");
            else if(!result.plan.valid || result.plan.no_change ||
                    !heat::delta_matches(result.plan,active.owner) ||
                    result.plan.delta.empty())record_no_change("invalid_delta");
            else {
                // Pre-proposal ACTIVE revalidation; one Controller proposal,
                // never direct publication. The full candidate map is staged.
                if(!r3::epoch::stage_same_map_generation(map))
                    return controller.reject_staging(r3::epoch::FaultCode::INVALID_MAP);
                const uint32_t dest_slot=1u-controller.control().active_slot;
                auto* owners=moe_shm::generation_map(map,dest_slot);
                for(const auto& m:result.plan.delta) {
                    if(m.expert>=active.owner.size() || m.to<0 ||
                       m.to>=int32_t(active.helpers) ||
                       active.owner[m.expert]!=m.from ||
                       protected_[m.expert] || !active.runtime[m.expert])
                        return controller.reject_staging(r3::epoch::FaultCode::INVALID_MAP);
                    owners[m.expert]=uint8_t(m.to);
                }
                auto* gh=moe_shm::generation_header(map,dest_slot);
                gh->digest=moe_shm::generation_digest(map,dest_slot);
                if(!gh->digest || gh->digest==active.digest ||
                   !controller.propose(gh->digest))
                    return controller.reject_staging(r3::epoch::FaultCode::DEVICE);
                pending_plan_=std::move(result.plan);
                txn_=controller.control().transaction_id;
                inflight_=true;
                ++evidence_.proposed;
                due_ns_=add_saturating(now_ns,settings_.cadence_ns);
                return true;
            }
        }
        if(now_ns<due_ns_ || now_ns<cooldown_until_ns_)return true;
        {
            std::lock_guard<std::mutex> guard(lock_);
            if(queued_ || ready_)return true;
            job_.owners=std::move(active);
            job_.last_move=history_.times();
            job_.excluded=negatives_;
            job_.now_ns=now_ns;
            job_.cooldown_until_ns=cooldown_until_ns_;
            queued_=true;
        }
        due_ns_=add_saturating(now_ns,settings_.cadence_ns);
        cv_.notify_one();
        return true;
    }
    void observe(const r3::epoch::Controller& controller,void* map,uint64_t now_ns) {
        if(!inflight_ || controller.control().transaction_id!=txn_ ||
           !controller.can_propose())return; // REOPENED + CLEANED or no result
        const auto& c=controller.control();
        if(c.journal_index>=2)return;
        const auto& j=controller.journal().entry[c.journal_index];
        if(j.transaction_id!=txn_ || !j.finish_ns)return;
        OwnerSnapshot active;
        if(!capture_owners(map,&active))return;
        const auto obs=history_.observe_closed(j,active,now_ns);
        if(obs==HistoryObservation::COMMITTED) {
            ++evidence_.committed;
            for(const auto& m:pending_plan_.delta)
                evidence_.bytes_moved=add_saturating(evidence_.bytes_moved,stride_);
            evidence_.predicted_saving+=pending_plan_.baseline-pending_plan_.predicted;
        } else if(obs==HistoryObservation::ABORTED) {
            ++evidence_.aborted;
            cooldown_until_ns_=add_saturating(now_ns,settings_.abort_cooldown_ns);
            const uint32_t reason=j.cleanup_fault?j.cleanup_fault:j.fault;
            for(const auto& m:pending_plan_.delta) {
                auto& entry=negatives_[{m.expert,uint32_t(m.to)}];
                entry={now_ns,add_saturating(now_ns,
                       settings_.learned_eligibility_ttl_ns),reason};
                ++evidence_.learned_ineligible;
            }
        } else return; // evidence remains inflight on unverified journal
        if(j.finish_ns>=j.start_ns) {
            const uint64_t elapsed=j.finish_ns-j.start_ns;
            ++evidence_.transition_samples;
            evidence_.transition_ns_total=add_saturating(evidence_.transition_ns_total,elapsed);
            evidence_.transition_ns_max=std::max(evidence_.transition_ns_max,elapsed);
        }
        inflight_=false;
        pending_plan_={};
    }
    void report(const void* map,uint32_t helper,uint64_t now_ns) const {
        moe_shm::PlacementControl c{};
        if(!moe_shm::placement_snapshot(map,&c))return;
        std::fprintf(stderr,
            "policy placement helper=%u evaluated=%llu no_change=%llu proposed=%llu committed=%llu aborted=%llu learned_ineligible=%llu cache_active=%zu inflight=%u bytes_moved=%llu predicted_saving=%.6f transition_samples=%llu transition_ns_total=%llu transition_ns_max=%llu unwatermarked_heat=%u active_epoch=%llu active_digest=%016llx\n",
            helper,(unsigned long long)evidence_.evaluated,
            (unsigned long long)evidence_.no_change,
            (unsigned long long)evidence_.proposed,
            (unsigned long long)evidence_.committed,
            (unsigned long long)evidence_.aborted,
            (unsigned long long)evidence_.learned_ineligible,
            negative_count(now_ns),unsigned(inflight_),
            (unsigned long long)evidence_.bytes_moved,
            evidence_.predicted_saving,
            (unsigned long long)evidence_.transition_samples,
            (unsigned long long)evidence_.transition_ns_total,
            (unsigned long long)evidence_.transition_ns_max,
            unsigned(evidence_.unwatermarked_heat),
            (unsigned long long)c.active_epoch,
            (unsigned long long)c.active_digest);
        for(const auto& [reason,count]:evidence_.reasons)
            std::fprintf(stderr,"policy no_change_reason=%s count=%llu\n",
                         reason.c_str(),(unsigned long long)count);
        for(const auto& [pair,entry]:negatives_)
            std::fprintf(stderr,"policy learned expert=%u destination=%u reason=%u observed_ns=%llu expires_ns=%llu\n",
                         pair.first,pair.second,entry.reason,
                         (unsigned long long)entry.observed_ns,
                         (unsigned long long)entry.until_ns);
    }
};
} // namespace r3::policy
