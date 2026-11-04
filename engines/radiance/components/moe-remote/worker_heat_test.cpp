#include "worker_heat.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace r3::heat;
static void check(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "r3_worker_heat_test: FAILED %s\n", why); std::exit(1); }
}
// Two helpers, helper 0 the head and helper 1 the tail. The tail holds the three hottest experts; the static placer
// (hottest first, head up to its budget, the tail the coldest leftovers) would put them on the head.
static Input fixture() {
    Input in;
    in.bytes={10,10,10,10,10,10};
    in.current={0,0,0,1,1,1};
    in.heat={5,1,0,90,60,40};
    in.devices={{40},{40}};
    in.allowed.assign(6,{1,1});
    in.max_moves=8; in.max_moved_bytes=1ULL<<20; in.now_ns=2'000'000'000'000ULL;
    in.margin=0.1;
    return in;
}
static const std::vector<uint32_t> kShares2={1,1};
static std::vector<int32_t> static_plan(const Input& in, const std::vector<uint32_t>& shares) {
    std::vector<size_t> r(in.bytes.size());
    for (size_t i=0;i<r.size();++i) r[i]=i;
    std::sort(r.begin(),r.end(),[&](size_t a,size_t b){
        return in.heat[a]!=in.heat[b] ? in.heat[a]>in.heat[b] : a<b; });
    std::vector<int64_t> bytes;
    for (size_t i : r) bytes.push_back(int64_t(in.bytes[i]));
    uint64_t budgets[8]={};
    for (size_t d=0;d<in.devices.size();++d) budgets[d]=in.devices[d].budget;
    const auto placed=r3::worker_share::plan_tail(bytes,uint32_t(in.devices.size()),shares.data(),budgets);
    std::vector<int32_t> out=in.current;
    for (size_t k=0;k<r.size();++k) if (placed[k]>=0) out[r[k]]=placed[k];
    return out;
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
static void test_converges_to_the_static_plan() {
    Input in=fixture();
    const std::vector<int32_t> want=static_plan(in,kShares2);
    check(want==std::vector<int32_t>({0,1,1,0,0,0}), "fixture's static plan is as computed by hand");
    const Plan p=propose(in,kShares2);
    check(p.valid && !p.no_change && p.owner==want, "live plan equals the static placer's plan for the live heat");
    check(delta_matches(p,in.current), "complete delta");
    check(p.baseline>0.9 && p.predicted<0.01 && p.heat_total==196, "tail heat share before and after");
    for (size_t d=0;d<in.devices.size();++d) check(p.used[d]<=in.devices[d].budget, "hard final budget");
    check(p.delta.size()==5 && p.delta[0].expert==3 && p.delta[1].expert==4 && p.delta[2].expert==5,
          "moves ordered hottest first");
    check(propose(in,kShares2).owner==p.owner, "deterministic proposal");
    // Fixed point: the static layout of its own heat proposes nothing.
    in.current=want;
    const Plan fixed=propose(in,kShares2);
    check(fixed.valid && fixed.no_change && fixed.delta.empty(), "the static layout is a fixed point");
    in=fixture(); in.heat.assign(6,0);
    const Plan flat=propose(in,kShares2);
    check(flat.valid && flat.no_change && flat.heat_total==0, "no heat is flat, not a plan");
}
static void test_groups_are_atomic() {
    Input in=fixture();
    in.devices={{30},{30}};   // both helpers exactly full: the head can only take a hot expert by sending one out
    const Plan full=propose(in,kShares2);
    check(full.valid && !full.no_change && full.delta.size()==6 &&
          full.owner==std::vector<int32_t>({1,1,1,0,0,0}) && full.used==std::vector<uint64_t>({30,30}),
          "full helpers exchange whole groups");
    in.max_moves=2;
    const Plan two=propose(in,kShares2);
    check(two.valid && two.delta.size()==2 && two.owner[3]==0 && two.owner[2]==1 &&
          two.used==std::vector<uint64_t>({30,30}), "a group of two (hot in, coldest out) fits two moves");
    in.max_moves=1;
    check(propose(in,kShares2).no_change, "a group larger than the move budget is skipped whole");
    in.max_moves=8; in.max_moved_bytes=19;
    check(propose(in,kShares2).no_change, "the byte budget counts every leg of the group");
    in.max_moved_bytes=20;
    check(propose(in,kShares2).delta.size()==2, "the byte budget at exactly one group");
}
static void test_chain_across_three_helpers() {
    Input in;
    in.bytes.assign(6,10);
    in.current={0,0,1,1,2,2};
    in.heat={1,0,0,0,90,80};
    in.devices={{20},{20},{20}};
    in.allowed.assign(6,{1,1,1});
    in.max_moves=8; in.max_moved_bytes=1ULL<<20; in.now_ns=2'000'000'000'000ULL; in.margin=0.1;
    const std::vector<uint32_t> shares={1,1,1};
    const Plan all=propose(in,shares);
    check(all.valid && !all.no_change && all.owner==std::vector<int32_t>({0,1,2,2,0,1}) &&
          all.used==std::vector<uint64_t>({20,20,20}) && delta_matches(all,in.current),
          "a chain through three full helpers moves as one transaction");
    in.max_moves=3;
    const Plan part=propose(in,shares);
    check(part.valid && part.delta.size()==3 && part.owner[4]==0 && part.used==std::vector<uint64_t>({20,20,20}),
          "only whole groups: the hottest chain of three, never half of the next");
}
static void test_gates_coverage_dwell() {
    Input in=fixture();
    in.allowed[3][0]=0;
    const Plan blocked=propose(in,kShares2);
    check(blocked.valid && blocked.owner[3]==1, "an expert not allowed on its target stays");
    in=fixture(); in.margin=0.999;
    check(propose(in,kShares2).valid && propose(in,kShares2).no_change, "margin rejects a sub-threshold improvement");
    in=fixture();
    in.samples={1,0,0,0,0,0}; in.min_samples=2;
    check(propose(in,kShares2).no_change, "insufficient per-expert observation coverage");
    in.samples.assign(6,2); in.observed_at_ns=1;
    check(propose(in,kShares2).no_change, "stale interval snapshot");
    in.observed_at_ns=in.now_ns;
    in.cooldown_until_ns=in.now_ns+1;
    check(propose(in,kShares2).no_change, "failure cooldown");
    in.cooldown_until_ns=0;
    in.last_move_ns.assign(6,in.now_ns-1);
    check(propose(in,kShares2).no_change, "minimum residence");
    in.last_move_ns.assign(6,0);
    check(!propose(in,kShares2).no_change, "dwell expired / never moved");
    in.max_moves=0;
    check(propose(in,kShares2).no_change, "zero move budget");
    in=fixture(); in.deterministic=true;
    check(propose(in,kShares2).no_change, "deterministic mode freezes policy");
    in=fixture(); in.devices[0].budget=10;
    check(!propose(in,kShares2).valid, "overbudget incumbent");
    in=fixture(); in.allowed[2]={1};
    check(!propose(in,kShares2).valid, "eligibility shape");
    in=fixture(); in.allowed[3][1]=0;
    check(!propose(in,kShares2).valid, "illegal active owner is a fault, not repaired");
    in=fixture(); in.current[3]=-1; in.allowed[3]={0,0};
    const Plan local=propose(in,kShares2);
    check(local.valid && local.owner[3]==-1, "engine-local expert fixed");
    in=fixture();
    check(!propose(in,{1}).valid && !propose(in,{1,1,1}).valid, "one share for each helper");
    in.devices.pop_back(); in.allowed.assign(6,{1}); in.current={0,0,0,0,0,0}; in.devices={{100}};
    check(!propose(in,{1}).valid, "a single helper has no tail to plan");
}
static void test_zero_owner_and_static() {
    const std::vector<int64_t> b(12,10);
    uint32_t shares[]={1,0,1};uint64_t budgets[]={40,0,20};
    check(static_tail(b,3,shares,budgets)==r3::worker_share::plan_tail(b,3,shares,budgets),
          "static tail unchanged");
}
int main() {
    test_half_life();
    test_converges_to_the_static_plan();
    test_groups_are_atomic();
    test_chain_across_three_helpers();
    test_gates_coverage_dwell();
    test_zero_owner_and_static();
    std::puts("r3_worker_heat_test: ok");
}
