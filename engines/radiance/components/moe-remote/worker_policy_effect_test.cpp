// Card-free measured service gate: production code exercised by synthetic samples.
#include "worker_policy_effect.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
using namespace r3::policy;
static void check(bool ok,const char* why) {
    if(!ok) {std::fprintf(stderr,"effect gate FAIL: %s\n",why);std::exit(1);}
}
struct Synthetic {
    ServiceSnapshot data{};
    uint64_t t=1000;
    void push(double ns_per_row,bool small=true,unsigned n=2) {
        auto& x=data.lane[small?0:1];
        x.samples+=n;x.rows+=10*n;x.time_ns+=uint64_t(std::llround(ns_per_row*10*n));
        t+=100;
    }
    void quiet(EffectGate& gate,double rate,bool small=true) {
        push(rate,small);gate.quiet(data,t);
    }
};
static void calibrate(EffectGate& gate,Synthetic& s) {
    gate.quiet(s.data,s.t); // anchor
    s.quiet(gate,100);
    check(!gate.ready(s.t,1000),"one quiet window is insufficient");
    s.quiet(gate,102);
    check(!gate.ready(s.t,1000),"one pair is insufficient");
    s.quiet(gate,101);
    check(gate.ready(s.t,1000),"two consecutive quiet comparisons");
}
int main() {
    LaneServiceSource source;
    ServiceSnapshot observed{};
    check(source.snapshot(&observed),"empty snapshot");
    source.record(0,25,100);
    source.record(r3::moe_shm::lane_of(0,r3::moe_shm::kSmallClasses),50,200);
    check(source.snapshot(&observed),"stable source snapshot");
    check(observed.lane[0].samples==1&&observed.lane[0].rows==25&&
          observed.lane[1].samples==1&&observed.lane[1].rows==50,
          "small and large lane samples never pooled");
    source.record(0,0,120);
    check(source.snapshot(&observed)&&observed.lane[0].samples==1,
          "zero routed work cannot create a misleading sample");

    for(unsigned scenario=0;scenario<3;++scenario) {
        EffectGate g({2,2,2.0});
        Synthetic s;
        check(g.valid(),"valid declared controls");
        check(!g.begin(s.t,1000),"cannot propose without measured noise");
        calibrate(g,s);
        check(g.noise_floor()>0,"quiet pair noise is visible before a trial");
        check(!g.ready(s.t+1001,1000),"stale quiet signal cannot qualify a future move");
        check(g.begin(s.t,1000),"can propose only after noise and pre-window");
        check(!g.begin(s.t,1000),"cannot overlap a trial");
        g.committed(s.data);
        check(g.judge(s.data)==EffectVerdict::WAITING,"no post observations no verdict");
        s.push(scenario==0?116.0:(scenario==1?84.0:102.0),true,1);
        check(g.judge(s.data)==EffectVerdict::WAITING,"partial post window never judged");
        s.push(scenario==0?116.0:(scenario==1?84.0:102.0),true,1);
        const EffectVerdict expect=scenario==0?EffectVerdict::WORSE:
                                   scenario==1?EffectVerdict::KEPT:EffectVerdict::UNDECIDED;
        check(g.judge(s.data)==expect,"regression, improvement and noise classification");
        check(g.last_ratio()>0&&g.noise_floor()>0,"record measured ratio and floor");
        check(!g.ready(s.t,1000),"old calibration cannot straddle a move");
    }
    {
        EffectGate g({2,1,2});
        Synthetic s;
        g.quiet(s.data,s.t);
        s.quiet(g,100);
        s.quiet(g,101);
        check(g.begin(s.t,1000),"minimum one pair calibrated");
        g.committed(s.data);
        s.data.lane[0].samples=0; // restart/reset of the counter
        check(g.judge(s.data)==EffectVerdict::INVALID,
              "counter reset produces no guessed verdict");
        g.abort();
        check(!g.ready(s.t,1000),"abort clears all pre-move noise");
    }
    {
        EffectGate bad({0,0,0});
        check(!bad.valid(),"missing controls fail closed");
    }
    {
        // Both source noise and the committed effect are ns/request; variable
        // routed rows per request must not turn an improvement into WORSE.
        EffectGate g({2,1,2.576});
        ServiceSnapshot snap{};
        g.quiet(snap,1000);
        snap.lane[1]={2,20,2000000};
        g.quiet(snap,2000);
        snap.lane[1]={4,220,4004000};
        g.quiet(snap,3000);
        check(g.begin(3000,2000),"request-normalised baseline available");
        g.committed(snap);
        snap.lane[1]={6,320,5804000};
        check(g.judge(snap)==EffectVerdict::KEPT,
              "row-mix shift cannot reverse a per-request improvement");
    }
    puts("effect gate: class isolation, noise, good/bad/noise, partial, reset passed");
}
