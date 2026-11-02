#include "worker_heat.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace r3::heat;
static void check(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "r3_worker_heat_test: FAILED %s\n", why); std::exit(1); }
}
static Input fixture() {
    Input in;
    in.bytes={10,10,10,10};
    in.current={0,0,0,1};
    in.heat={80,60,40,20};
    in.layer={0,0,0,0};
    in.devices={{40,1.0,0},{40,1.0,0}};
    in.allowed.assign(4,{1,1});
    in.max_moves=2; in.now_ns=2'000'000'000'000ULL;
    return in;
}
static void test_half_life() {
    Heat h(2,3,100);
    int32_t a[]={10,0,4}, b[]={2,1,0};
    check(h.observe(0,a,3), "observe");
    check(h.tick(100), "half life");
    check(h.observe(0,b,3,2.0), "weighted observe");
    check(h.counts[0]==9 && h.counts[1]==2 && h.counts[2]==2, "half life independent of policy tick size");
    check(h.tick(0) && h.counts[0]==9, "zero observations never decay");
    check(h.samples[0]==2 && h.samples[3]==0, "coverage");
    int32_t invalid[]={1,-1,2};
    check(!h.observe(0,invalid,3) && h.counts[0]==9, "atomic invalid histogram rejection");
    check(!h.observe(2,a,3) && !h.observe(0,nullptr,3) && !h.observe(0,a,2), "bad shape");
    Heat bad(1,1,-1);
    check(!bad.tick(100), "reject half life <=0");
    Heat equivalent(1,1,100);
    int32_t c[]={100};
    check(equivalent.observe(0,c,1) && equivalent.tick(40) && equivalent.tick(60),
          "multiple uneven observations");
    check(std::abs(equivalent.counts[0]-50)<1e-9, "constant half life over variable intervals");
}
static void test_equal_cost_peak() {
    Input in=fixture();
    const Plan p=propose(in);
    check(p.valid && !p.no_change && p.owner!=in.current && p.predicted<p.baseline*0.9,
          "equal cost helpers balance per-layer maximum");
    check(delta_matches(p,in.current), "complete delta");
    for (size_t d=0;d<in.devices.size();++d) check(p.used[d]<=in.devices[d].budget, "hard final budget");
    check(propose(in).owner==p.owner, "deterministic proposal");
    in.margin=0.8;
    check(propose(in).valid && propose(in).no_change, "hysteresis rejects subthreshold gain");
    in.margin=0.1; in.heat.assign(4,0);
    check(propose(in).no_change, "stable zero demand is no_change");
}
static void test_swap_and_budget() {
    Input in;
    in.bytes={10,10,10,10}; in.current={0,0,1,1};
    in.heat={100,90,1,1}; in.layer={0,0,0,0};
    in.devices={{20,1,0},{20,1,0}};
    in.allowed.assign(4,{1,1}); in.now_ns=2'000'000'000'000ULL;
    const Plan p=propose(in);
    check(p.valid && !p.no_change && p.delta.size()==2 &&
          (p.owner[0]==1 || p.owner[1]==1) && (p.owner[2]==0 || p.owner[3]==0),
          "atomic full-helper two-way swap");
    check(p.used==std::vector<uint64_t>({20,20}), "swap respects both full budgets");
    in.max_moves=1;
    check(propose(in).no_change, "single move cannot fit full destination");
    in.max_moves=2; in.max_moved_bytes=19;
    check(propose(in).no_change, "byte budget counts both swap legs");
    in.max_moved_bytes=20;
    in.allowed[0][1]=0;
    const Plan blocked=propose(in);
    check(blocked.valid && blocked.owner[0]==0, "protected source stays");
    in.allowed[0][0]=0;
    check(!propose(in).valid, "illegal active owner is fault, not repaired");
}
static void test_caps_coverage_dwell() {
    Input in=fixture();
    in.devices[1].layer_cap=60;
    const Plan p=propose(in);
    double load=0;
    for (size_t i=0;i<p.owner.size();++i) if (p.owner[i]==1) load+=in.heat[i];
    check(p.valid && load<=60, "per-helper per-layer heat cap");
    in.devices[1].layer_cap=-1;
    check(!propose(in).valid, "invalid cap");
    in=fixture();
    in.samples={1,0,0,0}; in.min_samples=2;
    check(propose(in).no_change, "insufficient per-expert observation coverage");
    in.samples={2,2,2,2}; in.observed_at_ns=1;
    check(propose(in).no_change, "stale interval snapshot");
    in.observed_at_ns=in.now_ns;
    in.cooldown_until_ns=in.now_ns+1;
    check(propose(in).no_change, "failure cooldown");
    in.cooldown_until_ns=0;
    in.last_move_ns.assign(4,in.now_ns-1);
    check(propose(in).no_change, "minimum residence");
    in.last_move_ns.assign(4,0);
    check(!propose(in).no_change, "dwell expired / never moved");
    in.max_moves=0;
    check(propose(in).no_change, "zero move budget");
    in=fixture(); in.deterministic=true;
    check(propose(in).no_change, "deterministic mode freezes policy");
    in=fixture(); in.devices[0].budget=10;
    check(!propose(in).valid, "overbudget incumbent");
    in=fixture(); in.allowed[2]={1};
    check(!propose(in).valid, "eligibility shape");
    in=fixture(); in.current[0]=-1; in.allowed[0]={0,0};
    check(propose(in).valid && propose(in).owner[0]==-1, "local protected expert fixed");
}
static void test_large_bottleneck_swap() {
    Input in;
    in.bytes.assign(12,10);
    in.current={0,0,0,0,0,0,1,1,1,1,1,1};
    in.heat={200,150,120,90,60,30,1,1,1,1,1,1};
    in.layer.assign(12,0);
    in.devices={{60,1,0},{60,1,0}};
    in.allowed.assign(12,{1,1});
    in.max_moves=2;in.now_ns=2'000'000'000'000ULL;
    const Plan p=propose(in);
    check(p.valid && !p.no_change && p.delta.size()==2 &&
          p.owner[0]==1 && p.used[0]==60 && p.used[1]==60,
          "bounded large-instance greedy finds atomic full-helper exchange");
    const Plan again=propose(in);
    check(again.owner==p.owner && delta_matches(p,in.current),
          "large-instance deterministic whole-map delta");
}
static void test_zero_owner_and_static() {
    Input in=fixture(); in.devices.push_back({40,0.5,0});
    for (auto& a:in.allowed) a.push_back(1);
    const Plan p=propose(in);
    bool has2=false;
    for (int o:p.owner) has2|=(o==2);
    check(p.valid && has2, "zero-owner helper eligible to receive expert");
    const std::vector<int64_t> b(12,10);
    uint32_t shares[]={1,0,1};uint64_t budgets[]={40,0,20};
    check(static_tail(b,3,shares,budgets)==r3::worker_share::plan_tail(b,3,shares,budgets),
          "static tail unchanged");
}
int main() {
    test_half_life();
    test_equal_cost_peak();
    test_swap_and_budget();
    test_caps_coverage_dwell();
    test_zero_owner_and_static();
    test_large_bottleneck_swap();
    std::puts("r3_worker_heat_test: ok");
}
