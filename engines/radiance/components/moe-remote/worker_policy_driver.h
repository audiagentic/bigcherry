#pragma once
// Stage-1 live placement: one off-lane CPU worker, one v23 Controller.
// The worker only reads copies, never a mailbox, lane, GPU or private pool.
#include "worker_policy_config.h"
#include "worker_policy_live.h"
#include "worker_policy_effect.h"
#include "worker_policy_measure.h"
#include "worker_epoch.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

namespace r3::policy {
class Driver {
    friend struct DriverTestAccess;
public:
    static constexpr uint32_t kMeasuredWorseReason=0x4d57; // distinct from Controller faults
    struct Negative {
        uint64_t observed_ns=0, until_ns=0;
        uint32_t reason=0;
    };
    struct Evidence {
        uint64_t evaluated=0, no_change=0, proposed=0, committed=0, aborted=0;
        uint64_t learned_ineligible=0, bytes_moved=0;
        uint64_t transition_samples=0, transition_ns_total=0, transition_ns_max=0;
        double predicted_saving=0;
        uint64_t effect_windows_judged=0, effect_kept=0, effect_reverted=0;
        uint64_t effect_undecided=0, effect_revert_attempts=0, effect_revert_aborts=0;
        double effect_noise_floor=0, effect_last_ratio=0;
        bool unwatermarked_heat=false;
        std::map<std::string,uint64_t> reasons;
    };
private:
    struct Snapshot {
        OwnerSnapshot owners;
        std::vector<uint64_t> last_move;
        std::map<std::pair<uint32_t,uint32_t>, Negative> excluded;
        uint64_t now_ns=0, cooldown_until_ns=0, terminal_generation=0;
    };
    struct Result {
        OwnerSnapshot owners;
        heat::Plan plan;
        std::string reason;
        bool evaluated=false, unwatermarked=false;
        uint64_t terminal_generation=0, valid_until_ns=0;
    };
    Settings settings_;
    const r3::epoch::Clock& clock_;
    ServiceSource* service_=nullptr;
    EffectGate effect_;
    OwnerSnapshot revert_before_{},revert_after_{};
    bool revert_requested_=false,reverting_=false,revert_failed_=false;
    std::unique_ptr<HeatSource> source_;
    std::vector<uint64_t> budgets_;
    uint64_t stride_=0;
    std::vector<uint8_t> protected_;
    // RHP16: what the setting derivations are made from. Written by the worker thread (heat) and the epoch thread
    // (transactions), read by evidence and tests, so behind their own lock.
    mutable std::mutex measure_lock_;
    Measured measured_;
    DriftTracker drift_{1};
    uint64_t last_delta_poll_ns_=0;
    heat::Heat heat_;
    uint64_t consumed_=0;
    MoveHistory history_;
    bool seeded_=false, inflight_=false;
    uint64_t due_ns_=0, cooldown_until_ns_=0, txn_=0, terminal_generation_=0;
    uint64_t started_ns_=0, next_heartbeat_ns_=0;
    std::atomic<uint64_t> last_feed_advance_ns_{0};
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
    uint64_t feed_window_ns() const {
        // Derived solely from mandatory explicit age and poll cadence controls.
        return add_saturating(settings_.cadence_ns,settings_.controls.max_heat_age_ns);
    }
    bool no_live_feed(uint64_t now_ns) const {
        const uint64_t advanced=last_feed_advance_ns_.load(std::memory_order_acquire);
        const uint64_t since=advanced ? advanced : started_ns_;
        return started_ns_ && now_ns>=add_saturating(since,feed_window_ns());
    }
    void prune_expired(uint64_t now_ns) {
        bool changed=false;
        for(auto it=negatives_.begin();it!=negatives_.end();)
            if(now_ns>=it->second.until_ns) {
                it=negatives_.erase(it);
                changed=true;
            } else ++it;
        if(changed)++terminal_generation_; // expired cache snapshot invalidates queued work
    }
    void record_no_change(const std::string& why) {
        ++evidence_.no_change;
        ++evidence_.reasons[why.empty()?"unspecified":why];
    }
    Result evaluate(const Snapshot& copy) {
        Result r;
        r.owners=copy.owners;
        r.terminal_generation=copy.terminal_generation;
        timespec rt{};
        if(clock_gettime(CLOCK_REALTIME,&rt)!=0 || rt.tv_sec<0) {
            r.reason="realtime_clock";return r;
        }
        const uint64_t realtime=uint64_t(rt.tv_sec)*1000000000ULL+uint64_t(rt.tv_nsec);
        HeatDelta delta{};
        std::string why;
        // Poll timestamp is sampled after queue delays, not at job capture.
        const uint64_t poll_now=clock_.now_ns();
        if(!poll_now) {r.reason="monotonic_clock";return r;}
        r.valid_until_ns=poll_now;
        const auto status=source_->poll(poll_now,realtime,&delta,&why);
        if(source_->authenticated_histogram_file() &&
           delta.file_advanced && status!=HeatPoll::INVALID)
            last_feed_advance_ns_.store(poll_now,std::memory_order_release);
        if(status==HeatPoll::INVALID) {r.reason="heat_invalid";return r;}
        if(status==HeatPoll::BASELINE) {
            // A discontinuity invalidates prior coverage. Rebase, never
            // pretend missing observations were consumed as a fresh delta.
            heat_=heat::Heat(copy.owners.layers,copy.owners.experts,
                             settings_.controls.half_life_observations);
            consumed_=delta.observation_count;
            reset_drift();
            r.reason="heat_baseline";return r;
        }
        if(status==HeatPoll::UNCHANGED) {r.reason="heat_unchanged";return r;}
        const bool unwatermarked=!delta.complete_observations;
        if(unwatermarked &&
           !(settings_.accept_unwatermarked_heat &&
             delta.provenance==HeatProvenance::ENGINE_HISTOGRAM_FILE &&
             source_->authenticated_histogram_file())) {
            r.reason="heat_unwatermarked";return r;
        }
        if(delta.age_ns>settings_.controls.max_heat_age_ns) {
            heat_=heat::Heat(copy.owners.layers,copy.owners.experts,
                             settings_.controls.half_life_observations);
            consumed_=delta.observation_count;
            reset_drift();
            r.reason="heat_stale";return r;
        }
        // Absolute expiry uses original file observation age at the poll.
        r.valid_until_ns=add_saturating(
            poll_now,settings_.controls.max_heat_age_ns-delta.age_ns);
        if(!consume_heat(delta,&heat_,&consumed_,
                         settings_.accept_unwatermarked_heat)) {
            r.reason="heat_invalid_coverage";return r;
        }
        {
            std::lock_guard<std::mutex> guard(measure_lock_);
            if(last_delta_poll_ns_ && poll_now>last_delta_poll_ns_)
                measured_.delta_gap_ns.add(double(poll_now-last_delta_poll_ns_));
            last_delta_poll_ns_=poll_now;
            // the interval's own counts: the accumulated heat is decayed by the very half-life this informs
            drift_.add(std::vector<double>(delta.counts.begin(),delta.counts.end()),delta.observation_count);
            measured_.drift_half_life_observations=drift_.half_life_observations();
            measured_.drift_horizon_observations=drift_.horizon_observations();
            measured_.marginal_error=marginal_relative_error(heat_.counts,drift_.k());
        }
        r.unwatermarked=unwatermarked;
        const size_t n=copy.owners.owner.size();
        std::vector<uint64_t> physical(n,stride_);
        std::vector<LedgerView> ledgers;
        if(!derive_active_ledgers(copy.owners,physical,budgets_,stride_,
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
                const bool banned=it!=copy.excluded.end() && poll_now<it->second.until_ns;
                e.target_eligible[d]=uint8_t(d==uint32_t(from) ||
                   (!e.protected_expert && budgets_[d]>=stride_ && !banned));
            }
        }
        heat::Input input;
        if(!build_input(copy.owners,experts,ledgers,heat_,delta,settings_.controls,
                        poll_now,copy.cooldown_until_ns,false,&input,
                        &why,settings_.accept_unwatermarked_heat)) {
            r.reason="input_invalid";return r;
        }
        r.evaluated=true;
        r.plan=heat::propose(input,settings_.head_shares);
        if(!r.plan.valid)r.reason="plan_invalid";
        else if(r.plan.no_change) {
            if(poll_now<copy.cooldown_until_ns)r.reason="abort_cooldown";
            else if(r.plan.heat_total==0)r.reason="flat_heat";
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
    // Each barrier is independent, so a terminal transaction, cooldown,
    // heat expiry or live ban is rechecked against the CURRENT state.
    static const char* proposal_rejection(
        uint64_t result_generation,uint64_t current_generation,
        uint64_t valid_until_ns,uint64_t now_ns,uint64_t cooldown_until_ns,
        const heat::Plan& plan,
        const std::map<std::pair<uint32_t,uint32_t>,Negative>& negatives) {
        if(result_generation!=current_generation)return "stale_terminal_generation";
        if(now_ns<cooldown_until_ns)return "abort_cooldown";
        if(!valid_until_ns || now_ns>=valid_until_ns)return "heat_expired_before_proposal";
        for(const auto& move:plan.delta) {
            if(move.to<0)return "invalid_destination";
            const auto it=negatives.find({move.expert,uint32_t(move.to)});
            if(it!=negatives.end() && now_ns<it->second.until_ns)
                return "negative_destination";
        }
        return nullptr;
    }
    Driver(Settings settings,std::unique_ptr<HeatSource> source,
           std::vector<uint64_t> budgets,uint64_t stride,
           std::vector<uint8_t> protected_experts,const r3::epoch::Clock& clock,
           ServiceSource* service=nullptr)
        :settings_(std::move(settings)),clock_(clock),service_(service),
         effect_({settings_.effect_window_samples,settings_.effect_min_quiet_windows,
                  settings_.effect_noise_multiplier}),source_(std::move(source)),
         budgets_(std::move(budgets)),stride_(stride),
         protected_(std::move(protected_experts)),
         heat_(uint32_t(protected_.size()),1,settings_.controls.half_life_observations) {
        // Heat shape is replaced with the coherent ACTIVE layout at first
        // capture; no constructor accesses shared memory.
        reset_drift();
    }
    ~Driver() { stop(); }
    // The hot set the drift is measured on: as many experts as the tail helper holds.
    size_t hot_set_k() const {return (budgets_.empty()||!stride_)?0:size_t(budgets_.back()/stride_);}
    void reset_drift() {
        std::lock_guard<std::mutex> guard(measure_lock_);
        drift_=DriftTracker(std::max<size_t>(hot_set_k(),1));
        measured_.drift_half_life_observations=0;measured_.drift_horizon_observations=0;measured_.marginal_error=0;
        measured_.hot_set_k=hot_set_k();last_delta_poll_ns_=0;
    }
    Measured measured() const {std::lock_guard<std::mutex> guard(measure_lock_);return measured_;}
    Driver(const Driver&)=delete;
    Driver& operator=(const Driver&)=delete;
    bool start() {
        if(!settings_.enabled)return true;
        if(!source_ || !stride_ || budgets_.empty() || !controls_valid(settings_.controls) ||
           settings_.cadence_ns==0 || settings_.abort_cooldown_ns==0 ||
           settings_.learned_eligibility_ttl_ns==0 ||
           (settings_.effect_gate &&
               (!service_ || !effect_.valid() || !settings_.effect_revert_ttl_ns)) ||
           budgets_.size()!=settings_.head_shares.size()) return false;
        started_ns_=clock_.now_ns();
        if(!started_ns_)return false;
        next_heartbeat_ns_=add_saturating(started_ns_,feed_window_ns());
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
    void heartbeat(uint64_t now_ns) {
        if(!settings_.enabled || !started_ns_ || now_ns<next_heartbeat_ns_)return;
        if(no_live_feed(now_ns))
            std::fprintf(stderr,
                "policy no_live_feed=1 window_ns=%llu (engine needs R3_MOE_WORKER_HISTOGRAM_PERIOD_S)\n",
                (unsigned long long)feed_window_ns());
        next_heartbeat_ns_=add_saturating(now_ns,feed_window_ns());
    }
    size_t negative_count(uint64_t now_ns) const {
        size_t n=0;
        for(const auto& [key,entry]:negatives_)if(now_ns<entry.until_ns)++n;
        return n;
    }
    bool submit(void* map,r3::epoch::Controller& controller,uint64_t now_ns) {
        if(!settings_.enabled || !controller.can_propose() || inflight_)return true;
        if(settings_.effect_gate) {
            if(revert_failed_)return true; // a failed rollback is terminal until restart
            ServiceSnapshot service{};
            if(!service_ || !service_->snapshot(&service)) {
                record_no_change("effect_signal_unavailable");return true;
            }
            if(effect_.waiting()) {
                const auto verdict=effect_.judge(service);
                if(verdict==EffectVerdict::WAITING)return true; // no other proposal while assessing
                if(verdict==EffectVerdict::INVALID) {
                    revert_failed_=true;
                    record_no_change("effect_post_signal_invalid");
                    return true; // fail closed, never guess a verdict
                }
                ++evidence_.effect_windows_judged;
                evidence_.effect_noise_floor=effect_.noise_floor();
                evidence_.effect_last_ratio=effect_.last_ratio();
                if(verdict==EffectVerdict::WORSE) {
                    // Exclude the destinations that produced the REGRESSION,
                    // not the destinations in the map we are restoring.
                    for(const auto& m:pending_plan_.delta) {
                        auto& entry=negatives_[{m.expert,uint32_t(m.to)}];
                        entry={now_ns,add_saturating(now_ns,
                               settings_.effect_revert_ttl_ns),kMeasuredWorseReason};
                        ++evidence_.learned_ineligible;
                    }
                    ++terminal_generation_;
                    revert_requested_=true;
                } else if(verdict==EffectVerdict::KEPT) {
                    ++evidence_.effect_kept;
                    pending_plan_={};
                } else {
                    ++evidence_.effect_undecided;
                    pending_plan_={};
                }
            } else if(!revert_requested_) {
                effect_.quiet(service,now_ns);
                evidence_.effect_noise_floor=effect_.noise_floor();
            }
            if(revert_requested_) {
                // Restore the ENTIRE prior ACTIVE generation through the same
                // Controller protocol, not a best-effort inverse delta.
                OwnerSnapshot active;
                if(!capture_owners(map,&active) ||
                   active.epoch!=revert_after_.epoch ||
                   active.digest!=revert_after_.digest ||
                   active.owner!=revert_after_.owner ||
                   active.runtime!=revert_before_.runtime ||
                   !r3::epoch::stage_same_map_generation(map)) {
                    revert_failed_=true;revert_requested_=false;
                    record_no_change("effect_revert_active_mismatch");return true;
                }
                const uint32_t dest=1u-controller.control().active_slot;
                auto* owners=moe_shm::generation_map(map,dest);
                for(size_t i=0;i<revert_before_.owner.size();++i) {
                    if(revert_before_.owner[i]<-1 ||
                       revert_before_.owner[i]>=int32_t(active.helpers)) {
                        revert_failed_=true;revert_requested_=false;
                        record_no_change("effect_revert_invalid_owner");
                        return controller.reject_staging(r3::epoch::FaultCode::INVALID_MAP);
                    }
                    owners[i]=revert_before_.owner[i]<0?moe_shm::kOwnerLocal:
                              uint8_t(revert_before_.owner[i]);
                }
                auto* gh=moe_shm::generation_header(map,dest);
                gh->digest=moe_shm::generation_digest(map,dest);
                if(!gh->digest || gh->digest==active.digest ||
                   !controller.propose(gh->digest)) {
                    revert_failed_=true;revert_requested_=false;
                    record_no_change("effect_revert_proposal_refused");
                    return controller.reject_staging(r3::epoch::FaultCode::DEVICE);
                }
                revert_requested_=false;reverting_=true;inflight_=true;
                txn_=controller.control().transaction_id;
                ++evidence_.proposed;
                ++evidence_.effect_revert_attempts;
                due_ns_=add_saturating(now_ns,settings_.cadence_ns);
                return true;
            }
        }
        // No shared-map walk on every 2 ms control-plane tick while the CPU
        // job is pending, before cadence, or during a post-abort cooldown.
        {
            std::lock_guard<std::mutex> guard(lock_);
            if(!ready_ && (queued_ || now_ns<due_ns_ ||
                           now_ns<cooldown_until_ns_))return true;
        }
        prune_expired(now_ns);
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
            else if(const char* reject=proposal_rejection(
                        result.terminal_generation,terminal_generation_,
                        result.valid_until_ns,clock_.now_ns(),cooldown_until_ns_,
                        result.plan,negatives_))
                record_no_change(reject);
            else if(settings_.effect_gate &&
                    !effect_.ready(clock_.now_ns(),settings_.controls.max_heat_age_ns))
                record_no_change("effect_not_calibrated");
            else {
                // Pre-proposal ACTIVE revalidation; one Controller proposal,
                // never direct publication. The full candidate map is staged.
                if(settings_.effect_gate &&
                   !effect_.begin(clock_.now_ns(),settings_.controls.max_heat_age_ns)) {
                    record_no_change("effect_pre_window_stale");return true;
                }
                if(!r3::epoch::stage_same_map_generation(map)) {
                    if(settings_.effect_gate)effect_.abort();
                    return controller.reject_staging(r3::epoch::FaultCode::INVALID_MAP);
                }
                const uint32_t dest_slot=1u-controller.control().active_slot;
                auto* owners=moe_shm::generation_map(map,dest_slot);
                for(const auto& m:result.plan.delta) {
                    if(m.expert>=active.owner.size() || m.to<0 ||
                       m.to>=int32_t(active.helpers) ||
                       active.owner[m.expert]!=m.from ||
                       protected_[m.expert] || !active.runtime[m.expert]) {
                        if(settings_.effect_gate)effect_.abort();
                        return controller.reject_staging(r3::epoch::FaultCode::INVALID_MAP);
                    }
                    owners[m.expert]=uint8_t(m.to);
                }
                auto* gh=moe_shm::generation_header(map,dest_slot);
                gh->digest=moe_shm::generation_digest(map,dest_slot);
                if(!gh->digest || gh->digest==active.digest ||
                   !controller.propose(gh->digest)) {
                    if(settings_.effect_gate)effect_.abort();
                    return controller.reject_staging(r3::epoch::FaultCode::DEVICE);
                }
                if(settings_.effect_gate)revert_before_=active;
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
            job_.terminal_generation=terminal_generation_;
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
            const bool was_revert=reverting_;
            if(reverting_) {
                if(active.owner!=revert_before_.owner ||
                   active.runtime!=revert_before_.runtime ||
                   active.epoch!=revert_after_.epoch+1) {
                    revert_failed_=true;
                    record_no_change("effect_revert_verification_failed");
                } else ++evidence_.effect_reverted;
                reverting_=false;
            } else if(settings_.effect_gate) {
                revert_after_=active;
                ServiceSnapshot service{};
                if(!service_ || !service_->snapshot(&service)) {
                    revert_failed_=true;
                    record_no_change("effect_commit_signal_unavailable");
                    effect_.abort();
                } else effect_.committed(service);
            }
            ++evidence_.committed;
            for(const auto& m:pending_plan_.delta)
                evidence_.bytes_moved=add_saturating(evidence_.bytes_moved,stride_);
            if(!was_revert)
                evidence_.predicted_saving+=pending_plan_.baseline-pending_plan_.predicted;
        } else if(obs==HistoryObservation::ABORTED) {
            const bool was_revert=reverting_;
            if(reverting_) {
                reverting_=false;revert_failed_=true;
                ++evidence_.effect_revert_aborts;
                record_no_change("effect_revert_aborted_no_retry");
            } else if(settings_.effect_gate)effect_.abort();
            ++evidence_.aborted;
            cooldown_until_ns_=add_saturating(now_ns,settings_.abort_cooldown_ns);
            if(!was_revert) {
                const uint32_t reason=j.cleanup_fault?j.cleanup_fault:j.fault;
                for(const auto& m:pending_plan_.delta) {
                    auto& entry=negatives_[{m.expert,uint32_t(m.to)}];
                    entry={now_ns,add_saturating(now_ns,
                           settings_.learned_eligibility_ttl_ns),reason};
                    ++evidence_.learned_ineligible;
                }
            } // preserve measured-worse TTL/reason if rollback itself aborts
        } else return; // evidence remains inflight on unverified journal
        if(j.finish_ns>=j.start_ns) {
            const uint64_t elapsed=j.finish_ns-j.start_ns;
            {
                std::lock_guard<std::mutex> guard(measure_lock_);
                if(obs==HistoryObservation::ABORTED)measured_.abort_ns.add(double(elapsed));
                else {
                    measured_.transaction_ns.add(double(elapsed));
                    measured_.copy.add(uint64_t(pending_plan_.delta.size())*stride_,elapsed);
                }
            }
            ++evidence_.transition_samples;
            evidence_.transition_ns_total=add_saturating(evidence_.transition_ns_total,elapsed);
            evidence_.transition_ns_max=std::max(evidence_.transition_ns_max,elapsed);
        }
        ++terminal_generation_; // invalidate results queued before any closed txn
        prune_expired(now_ns);
        inflight_=false;
        if(!settings_.effect_gate || !effect_.waiting())pending_plan_={};
    }
    void report(const void* map,uint32_t helper,uint64_t now_ns) const {
        moe_shm::PlacementControl c{};
        if(!moe_shm::placement_snapshot(map,&c))return;
        std::fprintf(stderr,
            "policy effect_gate=%u effect_window_samples=%llu effect_min_quiet_windows=%llu effect_noise_multiplier=%.6f effect_revert_ttl_ns=%llu windows_judged=%llu kept=%llu reverted=%llu undecided=%llu noise_floor=%.9f last_signal_ratio=%.9f revert_attempts=%llu revert_aborts=%llu revert_blocked=%u\n",
            unsigned(settings_.effect_gate),
            (unsigned long long)settings_.effect_window_samples,
            (unsigned long long)settings_.effect_min_quiet_windows,
            settings_.effect_noise_multiplier,
            (unsigned long long)settings_.effect_revert_ttl_ns,
            (unsigned long long)evidence_.effect_windows_judged,
            (unsigned long long)evidence_.effect_kept,
            (unsigned long long)evidence_.effect_reverted,
            (unsigned long long)evidence_.effect_undecided,
            evidence_.effect_noise_floor,evidence_.effect_last_ratio,
            (unsigned long long)evidence_.effect_revert_attempts,
            (unsigned long long)evidence_.effect_revert_aborts,
            unsigned(revert_failed_));
        std::fprintf(stderr,
            "policy placement helper=%u evaluated=%llu no_change=%llu proposed=%llu committed=%llu aborted=%llu learned_ineligible=%llu cache_active=%zu inflight=%u bytes_moved=%llu predicted_saving=%.6f transition_samples=%llu transition_ns_total=%llu transition_ns_max=%llu unwatermarked_heat=%u no_live_feed=%u active_epoch=%llu active_digest=%016llx\n",
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
            unsigned(evidence_.unwatermarked_heat),unsigned(no_live_feed(now_ns)),
            (unsigned long long)c.active_epoch,
            (unsigned long long)c.active_digest);
        {
            const Measured m=measured();
            std::fprintf(stderr,
                "policy measure transactions=%llu txn_p50_ns=%.0f txn_p95_ns=%.0f txn_max_ns=%.0f copy_bytes_per_ns=%.6f "
                "aborts=%llu abort_p95_ns=%.0f delta_gaps=%llu delta_gap_p99_ns=%.0f delta_gap_max_ns=%.0f "
                "hot_set_k=%zu drift_half_life_observations=%llu drift_horizon_observations=%llu marginal_relative_error=%.6f\n",
                (unsigned long long)m.transaction_ns.total(),m.transaction_ns.quantile(0.5),
                m.transaction_ns.quantile(0.95),m.transaction_ns.max(),m.copy.bytes_per_ns(),
                (unsigned long long)m.abort_ns.total(),m.abort_ns.quantile(0.95),
                (unsigned long long)m.delta_gap_ns.total(),m.delta_gap_ns.quantile(0.99),
                m.delta_gap_ns.max(),m.hot_set_k,
                (unsigned long long)m.drift_half_life_observations,
                (unsigned long long)m.drift_horizon_observations,m.marginal_error);
        }
        for(const auto& [reason,count]:evidence_.reasons)
            std::fprintf(stderr,"policy no_change_reason=%s count=%llu\n",
                         reason.c_str(),(unsigned long long)count);
        // Synchronous evidence detail is capped by explicitly configured
        // max moves and the actual helper count, regardless of cache shape.
        using Entry=std::pair<std::pair<uint32_t,uint32_t>,Negative>;
        std::vector<Entry> newest;
        const size_t limit=std::min<size_t>(settings_.controls.max_moves,budgets_.size());
        for(const auto& item:negatives_) {
            if(now_ns>=item.second.until_ns || !limit)continue;
            newest.push_back(item);
            std::sort(newest.begin(),newest.end(),[](const Entry& a,const Entry& b) {
                return a.second.observed_ns>b.second.observed_ns ||
                    (a.second.observed_ns==b.second.observed_ns && a.first<b.first);
            });
            if(newest.size()>limit)newest.pop_back();
        }
        std::fprintf(stderr,"policy cache_details=%zu cache_detail_limit=%zu\n",
                     newest.size(),limit);
        for(const auto& [pair,entry]:newest)
            std::fprintf(stderr,"policy learned expert=%u destination=%u reason=%u reason_name=%s observed_ns=%llu expires_ns=%llu\n",
                         pair.first,pair.second,entry.reason,
                         entry.reason==kMeasuredWorseReason?"measured-worse":"controller-abort",
                         (unsigned long long)entry.observed_ns,
                         (unsigned long long)entry.until_ns);
    }
};
} // namespace r3::policy
