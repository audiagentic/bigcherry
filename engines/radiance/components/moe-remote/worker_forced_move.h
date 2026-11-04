#pragma once
// Test-only forced owner-map candidate generator. All changes enter the
// production placement Controller; none run on the request/combiner path.
#include "worker_epoch.h"
#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace r3::forced {
using namespace r3::moe_shm;
using r3::epoch::Controller;
using r3::epoch::FaultCode;

struct Move { uint32_t layer=0, expert=0, destination=0; };
inline bool number(const std::string& s,uint32_t* p) {
    if (!p || s.empty()) return false;
    for(char c:s) if(c<'0'||c>'9') return false;
    auto [end,err]=std::from_chars(s.data(),s.data()+s.size(),*p);
    return err==std::errc{} && end==s.data()+s.size();
}
inline bool parse_move(const std::string& s,Move* m) {
    if(!m) return false;
    const auto a=s.find(':'),b=s.find(':',a==std::string::npos ? 0 : a+1);
    return a!=std::string::npos && b!=std::string::npos &&
        s.find(':',b+1)==std::string::npos &&
        number(s.substr(0,a),&m->layer) &&
        number(s.substr(a+1,b-a-1),&m->expert) &&
        number(s.substr(b+1),&m->destination);
}
// A file contains one layer:expert:destination per line. Optional FIRST
// directive repeat=N repeats all lines. For churn list both move and return:
// repeat=500, 0:3:1, 0:3:0 => 1,000 legal proposals.
struct Schedule {
    std::vector<Move> steps;
    uint32_t repeat=1;
    uint64_t issued=0;
    uint64_t count() const {return uint64_t(steps.size())*repeat;}
    bool pending() const {return issued<count();}
    Move next() const {return steps[size_t(issued%steps.size())];}
    void advance() {++issued;}
};
inline bool read_schedule(const std::string& file,Schedule* out,std::string* err) {
    if(!out||file.empty()) return false;
    std::ifstream in(file);
    if(!in) {if(err)*err="cannot open schedule "+file;return false;}
    Schedule s; std::string line; uint32_t no=0; bool first=true;
    while(std::getline(in,line)) {
        ++no;
        auto start=line.find_first_not_of(" \t\r");
        if(start==std::string::npos||line[start]=='#') continue;
        auto end=line.find_last_not_of(" \t\r");
        line=line.substr(start,end-start+1);
        if(first && line.compare(0,7,"repeat=")==0) {
            if(!number(line.substr(7),&s.repeat)||!s.repeat) {
                if(err)*err="invalid repeat on line "+std::to_string(no);return false;
            }
        } else {
            Move m;
            if(!parse_move(line,&m)) {
                if(err)*err="invalid triple on line "+std::to_string(no);return false;
            }
            s.steps.push_back(m);
        }
        first=false;
    }
    // The transaction/epoch sequence is bounded by the v23 32-bit epoch.
    if(!in.eof()||s.steps.empty()||s.steps.size()>UINT32_MAX/s.repeat) {
        if(err)*err="unreadable, empty or epoch-exhausting schedule";return false;
    }
    *out=std::move(s);return true;
}
using Source = std::function<bool(uint32_t,uint32_t)>;
// Validate the sealed map, frozen runtime slot, source and steady-state budget
// BEFORE staging. Transient physical capacity is checked again by the real
// Participant, which can reject RESERVING without changing ownership.
inline FaultCode eligible(const void* map,const Move& m,
             const std::vector<uint64_t>& budgets,uint64_t stride,
             const Source& source) {
    PlacementControl c{};
    if(!map||!placement_snapshot(map,&c)) return FaultCode::INVALID_MAP;
    const auto* h=static_cast<const Header*>(map);
    if(c.state!=uint32_t(PlacementState::ACTIVE)||c.fault_code||
       c.active_slot>1||!c.active_digest||!stride||
       generation_digest(map,c.active_slot)!=c.active_digest||
       budgets.size()!=h->helper_count||m.layer>=h->layer_count||
       m.expert>=h->expert_count||m.destination>=h->helper_count)
        return FaultCode::INVALID_MAP;
    const size_t id=size_t(m.layer)*h->expert_count+m.expert;
    const auto* owners=generation_map(map,c.active_slot);
    if(owners[id]==kOwnerLocal||owners[id]>=h->helper_count||
       owners[id]==m.destination||!load(&h->layer_map_ready[m.layer])||
       layer_slot(map,m.layer,m.expert)<0) return FaultCode::INVALID_MAP;
    if(!source||!source(m.layer,m.expert)) return FaultCode::SOURCE;
    uint64_t held=0;
    for(uint32_t i=0;i<h->owner_map_bytes;++i) held+=owners[i]==m.destination;
    if(held>=budgets[m.destination]/stride) return FaultCode::CAPACITY;
    return FaultCode::NONE;
}
inline bool propose(void* map,Controller& controller,const Move& m,
            const std::vector<uint64_t>& budgets,uint64_t stride,
            const Source& source,FaultCode* outcome=nullptr) {
    if(!controller.can_propose()) return false;
    FaultCode fault=eligible(map,m,budgets,stride,source);
    if(fault==FaultCode::NONE && !r3::epoch::stage_same_map_generation(map))
        fault=FaultCode::INVALID_MAP;
    if(fault==FaultCode::NONE) {
        const auto* h=static_cast<const Header*>(map);
        const uint32_t next=1u-controller.control().active_slot;
        generation_map(map,next)[size_t(m.layer)*h->expert_count+m.expert]=
            uint8_t(m.destination);
        auto* gh=generation_header(map,next);
        gh->digest=generation_digest(map,next);
        if(!gh->digest||gh->digest==controller.control().active_digest||
           !controller.propose(gh->digest)) fault=FaultCode::DEVICE;
    }
    if(outcome)*outcome=fault;
    return fault==FaultCode::NONE || controller.reject_staging(fault);
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
        if(inflight_||!pending()||!c.can_propose()) return true;
        FaultCode fault=FaultCode::NONE;
        if(!propose(map,c,schedule_.next(),budgets_,stride_,source_,&fault))
            return false;
        schedule_.advance(); ++attempted_;
        if(fault==FaultCode::NONE) bytes_+=stride_;
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
        if(j.commit_ns)++committed_; else ++aborted_;
        if(j.fault<faults_.size()&&j.fault)++faults_[j.fault];
        if(pause_)pauses_.add(now-pause_);
        inflight_=false;
    }
    void report(const void* map,uint64_t charged,uint32_t helper) const {
        PlacementControl c{};
        if(!placement_snapshot(map,&c))return;
        std::fprintf(stderr,
          "forced placement helper=%u attempted=%llu committed=%llu aborted=%llu inflight=%u bytes_staged=%llu active_epoch=%llu active_digest=%016llx active_slot=%u pending_epoch=%llu pending_digest=%016llx ledger_charged=%llu\n",
          helper,(unsigned long long)attempted_,(unsigned long long)committed_,
          (unsigned long long)aborted_,unsigned(inflight_),(unsigned long long)bytes_,
          (unsigned long long)c.active_epoch,(unsigned long long)c.active_digest,
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
