#pragma once
// Test-only forced owner-map candidate generator. All changes enter the
// production placement Controller; none run on the request/combiner path.
#include "worker_epoch.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace r3::forced {
using namespace r3::moe_shm;
using r3::epoch::Controller;
using r3::epoch::FaultCode;

struct Move { uint32_t layer=0, expert=0, destination=0; };
struct Step {
    enum class Kind : uint8_t { MOVE, SWAP, AUTO_SWAP };
    Kind kind=Kind::MOVE;
    Move move{};
    uint32_t layer_b=0, expert_b=0;
    Step()=default;
    Step(Move m):move(m){}
    static Step swap(uint32_t a_layer,uint32_t a_expert,
                     uint32_t b_layer,uint32_t b_expert) {
        Step s; s.kind=Kind::SWAP; s.move={a_layer,a_expert,0};
        s.layer_b=b_layer;s.expert_b=b_expert;return s;
    }
    static Step automatic(uint32_t x,uint32_t y) {
        Step s; s.kind=Kind::AUTO_SWAP; s.move={x,y,0};return s;
    }
};
// A schedule is `repeat` passes over its ordered move, explicit swap or automatic swap steps, typed from the manifest
// (spec.experiment.placement.forced). Exactly one proposal is issued per step.
struct Schedule {
    std::vector<Step> steps;
    uint32_t repeat=1;
    uint64_t issued=0;
    uint64_t count() const {return uint64_t(steps.size())*repeat;}
    bool pending() const {return issued<count();}
    Step next() const {return steps[size_t(issued%steps.size())];}
    void advance() {++issued;}
};
using Source=std::function<bool(uint32_t,uint32_t)>;
// The engine is serving once it has published a request ticket. Until then it
// may still be initialising its mailbox kernels, which refuse to start while a
// placement transaction is in flight ("called in the wrong phase"), and a KL
// run would see every swap land before its first scored position.
inline bool engine_serving(const void* map) {
    return map && load(&static_cast<const Header*>(map)->publish_sequence)!=0;
}
inline constexpr std::array<uint32_t,2> kNoPrevious{{UINT32_MAX,UINT32_MAX}};
struct Resolved {
    std::array<Move,2> changes{};
    std::array<uint32_t,2> ids{{UINT32_MAX,UINT32_MAX}};
    uint32_t count=0;
};
// Resolve from ACTIVE only; previous contains the expert IDs moved in the
// last committed transaction. AUTO picks the first eligible distinct pair
// in ascending (layer,expert), excluding *both* previous IDs.
inline FaultCode resolve(const void* map,const Step& step,
    const std::vector<uint64_t>& budgets,uint64_t stride,const Source& source,
    const std::array<uint32_t,2>& previous,Resolved* result) {
    PlacementControl c{};
    if(!result||!map||!placement_snapshot(map,&c))return FaultCode::INVALID_MAP;
    const auto* h=static_cast<const Header*>(map);
    if(c.state!=uint32_t(PlacementState::ACTIVE)||c.fault_code||
       c.active_slot>1||!c.active_digest||!stride||
       generation_digest(map,c.active_slot)!=c.active_digest||
       budgets.size()!=h->helper_count||!h->owner_map_bytes)
        return FaultCode::INVALID_MAP;
    const auto* owners=generation_map(map,c.active_slot);
    auto in_range=[&](uint32_t layer,uint32_t expert) {
        return layer<h->layer_count&&expert<h->expert_count;
    };
    auto runtime=[&](uint32_t layer,uint32_t expert) {
        return load(&h->layer_map_ready[layer]) &&
               layer_slot(map,layer,expert)>=0;
    };
    Resolved r;
    if(step.kind==Step::Kind::MOVE) {
        const auto& m=step.move;
        if(!in_range(m.layer,m.expert)||m.destination>=h->helper_count)
            return FaultCode::INVALID_MAP;
        const size_t id=size_t(m.layer)*h->expert_count+m.expert;
        if(owners[id]>=h->helper_count||owners[id]==m.destination||
           !runtime(m.layer,m.expert))return FaultCode::INVALID_MAP;
        if(!source||!source(m.layer,m.expert))return FaultCode::SOURCE;
        r.count=1;r.ids[0]=uint32_t(id);r.changes[0]=m;
    } else {
        uint32_t al=step.move.layer,ae=step.move.expert;
        uint32_t bl=step.layer_b,be=step.expert_b;
        if(step.kind==Step::Kind::AUTO_SWAP) {
            const uint32_t x=step.move.layer,y=step.move.expert;
            if(x>=h->helper_count||y>=h->helper_count||x==y)
                return FaultCode::INVALID_MAP;
            bool found=false;
            for(uint32_t l=0;l<h->layer_count&&!found;++l) {
                if(!load(&h->layer_map_ready[l]))continue;
                uint32_t ex=UINT32_MAX,ey=UINT32_MAX;
                for(uint32_t e=0;e<h->expert_count;++e) {
                    const size_t id=size_t(l)*h->expert_count+e;
                    if(id==previous[0]||id==previous[1]||
                       layer_slot(map,l,e)<0||!source||!source(l,e))continue;
                    if(owners[id]==x && ex==UINT32_MAX)ex=e;
                    if(owners[id]==y && ey==UINT32_MAX)ey=e;
                    if(ex!=UINT32_MAX&&ey!=UINT32_MAX)break;
                }
                if(ex!=UINT32_MAX&&ey!=UINT32_MAX) {
                    al=bl=l;ae=ex;be=ey;found=true;
                }
            }
            if(!found)return FaultCode::SOURCE;
        }
        if(!in_range(al,ae)||!in_range(bl,be)||
           (al==bl&&ae==be))return FaultCode::INVALID_MAP;
        const size_t a=size_t(al)*h->expert_count+ae;
        const size_t b=size_t(bl)*h->expert_count+be;
        const uint32_t x=owners[a],y=owners[b];
        if(x>=h->helper_count||y>=h->helper_count||x==y||
           !runtime(al,ae)||!runtime(bl,be))return FaultCode::INVALID_MAP;
        if(!source||!source(al,ae)||!source(bl,be))return FaultCode::SOURCE;
        r.count=2;r.ids={uint32_t(a),uint32_t(b)};
        r.changes[0]={al,ae,y};r.changes[1]={bl,be,x};
    }
    // Postcommit residency, not transient peak: subtract all departures
    // before admitting arrivals. A full helper can exchange one-for-one.
    std::vector<uint64_t> counts(h->helper_count,0);
    for(uint32_t id=0;id<h->owner_map_bytes;++id)
        if(owners[id]<h->helper_count)++counts[owners[id]];
    for(uint32_t i=0;i<r.count;++i) {
        const uint32_t src=owners[r.ids[i]];
        if(!counts[src])return FaultCode::INVALID_MAP;
        --counts[src];
    }
    for(uint32_t i=0;i<r.count;++i)++counts[r.changes[i].destination];
    for(uint32_t i=0;i<h->helper_count;++i)
        if(counts[i]>budgets[i]/stride)return FaultCode::CAPACITY;
    *result=r;
    return FaultCode::NONE;
}
inline FaultCode eligible(const void* map,const Step& step,
    const std::vector<uint64_t>& budgets,uint64_t stride,const Source& source) {
    Resolved ignored;
    return resolve(map,step,budgets,stride,source,kNoPrevious,&ignored);
}
inline bool propose(void* map,Controller& controller,const Step& step,
    const std::vector<uint64_t>& budgets,uint64_t stride,const Source& source,
    const std::array<uint32_t,2>& previous=kNoPrevious,FaultCode* outcome=nullptr,
    Resolved* chosen=nullptr) {
    if(!controller.can_propose())return false;
    Resolved r;
    FaultCode fault=resolve(map,step,budgets,stride,source,previous,&r);
    if(fault==FaultCode::NONE&&!r3::epoch::stage_same_map_generation(map))
        fault=FaultCode::INVALID_MAP;
    if(fault==FaultCode::NONE) {
        const uint32_t next=1u-controller.control().active_slot;
        for(uint32_t i=0;i<r.count;++i)
            generation_map(map,next)[r.ids[i]]=uint8_t(r.changes[i].destination);
        auto* gh=generation_header(map,next);
        gh->digest=generation_digest(map,next);
        if(!gh->digest||gh->digest==controller.control().active_digest||
           !controller.propose(gh->digest))fault=FaultCode::DEVICE;
    }
    if(outcome)*outcome=fault;
    if(chosen&&fault==FaultCode::NONE)*chosen=r;
    return fault==FaultCode::NONE||controller.reject_staging(fault);
}
// Fixed 64-bin, logarithmic duration histogram. Memory is independent of
// churn count; median is explicitly approximate (lower bucket boundary).
struct Duration {
    uint64_t min=UINT64_MAX,max=0,n=0;
    std::array<uint64_t,64> bins{};
    void add(uint64_t x) {
        min=std::min(min,x); max=std::max(max,x); ++n;
        ++bins[x?63u-unsigned(__builtin_clzll(x)):0u];
    }
    uint64_t median() const {
        uint64_t count=0;
        for(unsigned i=0;i<64;++i) if((count+=bins[i])>=(n+1)/2)
            return n ? (i ? (uint64_t(1)<<i) : min) : 0;
        return 0;
    }
};
class Driver {
    Schedule schedule_;
    std::vector<uint64_t> budgets_;
    uint64_t stride_;
    Source source_;
    uint64_t txn_=0,started_=0,pause_=0,attempted_=0,committed_=0,aborted_=0,bytes_=0;
    uint32_t phase_=0;
    bool inflight_=false;
    std::array<uint32_t,2> previous_{{UINT32_MAX,UINT32_MAX}};
    std::array<uint32_t,2> pending_ids_{{UINT32_MAX,UINT32_MAX}};
    std::array<Duration,8> durations_{};
    Duration pauses_{};
    std::array<uint64_t,18> faults_{};
public:
    Driver(Schedule s,std::vector<uint64_t> b,uint64_t stride,Source source)
        : schedule_(std::move(s)),budgets_(std::move(b)),stride_(stride),
          source_(std::move(source)) {}
    bool pending() const {return schedule_.pending();}
    uint64_t attempted() const {return attempted_;}
    uint64_t committed() const {return committed_;}
    uint64_t aborted() const {return aborted_;}
    bool submit(void* map,Controller& c,uint64_t now) {
        if(inflight_||!pending()||!c.can_propose()||!engine_serving(map)) return true;
        FaultCode fault=FaultCode::NONE;
        Resolved chosen;
        const Step step=schedule_.next();
        if(!propose(map,c,step,budgets_,stride_,source_,previous_,&fault,&chosen))
            return false;
        schedule_.advance(); ++attempted_;
        pending_ids_=chosen.ids;
        if(fault==FaultCode::NONE) {
            bytes_+=stride_*chosen.count;
            if(chosen.count==2)
                std::fprintf(stderr,
                    "forced resolved transaction=%llu epoch=%llu swap=%u:%u:%u:%u helpers=%u:%u\n",
                    (unsigned long long)c.control().transaction_id,
                    (unsigned long long)c.control().pending_epoch,
                    chosen.changes[0].layer,chosen.changes[0].expert,
                    chosen.changes[1].layer,chosen.changes[1].expert,
                    chosen.changes[1].destination,chosen.changes[0].destination);
        }
        inflight_=true;txn_=c.control().transaction_id;
        phase_=c.control().state;started_=now;pause_=0;
        return true;
    }
    void observe(const Controller& controller,uint64_t now) {
        if(!inflight_)return;
        const auto& c=controller.control();
        if(c.transaction_id!=txn_) return;
        if(c.state!=phase_) {
            if(phase_<durations_.size())durations_[phase_].add(now-started_);
            if(c.state==uint32_t(PlacementState::QUIESCING))pause_=now;
            phase_=c.state;started_=now;
        }
        // ACTIVE alone is insufficient. Require exact REOPENED and CLEANED
        // identities through Controller::can_propose before counting success.
        if(!controller.failed_closed() &&
           (c.state!=uint32_t(PlacementState::ACTIVE)||!controller.can_propose()))
            return;
        const auto& j=controller.journal().entry[c.journal_index];
        if(j.transaction_id!=txn_)return;
        if(j.commit_ns) {++committed_;previous_=pending_ids_;}
        else ++aborted_;
        const uint32_t fault=j.cleanup_fault ? j.cleanup_fault : j.fault;
        if(fault<faults_.size()&&fault)++faults_[fault];
        if(pause_)pauses_.add(now-pause_);
        inflight_=false;
    }
    void report(const void* map,uint64_t charged,uint64_t local_bytes_staged,uint32_t helper) const {
        PlacementControl c{};
        if(!placement_snapshot(map,&c))return;
        std::fprintf(stderr,
          "forced placement helper=%u attempted=%llu committed=%llu aborted=%llu inflight=%u bytes_planned=%llu local_bytes_staged=%llu active_epoch=%llu active_digest=%016llx active_slot=%u pending_epoch=%llu pending_digest=%016llx ledger_charged=%llu\n",
          helper,(unsigned long long)attempted_,(unsigned long long)committed_,
          (unsigned long long)aborted_,unsigned(inflight_),(unsigned long long)bytes_,
          (unsigned long long)local_bytes_staged,(unsigned long long)c.active_epoch,(unsigned long long)c.active_digest,
          c.active_slot,(unsigned long long)c.pending_epoch,
          (unsigned long long)c.pending_digest,(unsigned long long)charged);
        for(size_t i=0;i<durations_.size();++i)if(durations_[i].n)
            std::fprintf(stderr,"forced phase=%zu n=%llu min_ns=%llu approx_median_ns=%llu max_ns=%llu\n",
             i,(unsigned long long)durations_[i].n,(unsigned long long)durations_[i].min,
             (unsigned long long)durations_[i].median(),(unsigned long long)durations_[i].max);
        if(pauses_.n)
            std::fprintf(stderr,"forced admission_pause n=%llu min_ns=%llu approx_median_ns=%llu max_ns=%llu\n",
             (unsigned long long)pauses_.n,(unsigned long long)pauses_.min,
             (unsigned long long)pauses_.median(),(unsigned long long)pauses_.max);
        for(size_t i=0;i<faults_.size();++i)if(faults_[i])
            std::fprintf(stderr,"forced fault=%zu count=%llu\n",i,(unsigned long long)faults_[i]);
        for(uint32_t i=0;i<kMaxHelpers+kMaxEngineRanks;++i) {
            auto a=r3::epoch::read_shared_ack(map,i);
            if(a.transaction_id)
                std::fprintf(stderr,"forced ack_actor=%u txn=%llu phase=%u epoch=%llu slot=%u digest=%016llx consumed_step=%llu fault=%u\n",
                  i,(unsigned long long)a.transaction_id,a.phase,
                  (unsigned long long)a.epoch,a.generation_slot,(unsigned long long)a.digest,
                  (unsigned long long)a.drained_step,a.fault_code);
        }
    }
};
} // namespace r3::forced
