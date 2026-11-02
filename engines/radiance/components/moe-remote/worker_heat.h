#pragma once
// CPU-only stage-1 owner-map policy. A proposal is never a live mailbox update.
// Helper-local weight sources, eligibility and hard budgets are caller-validated inputs.
#include "worker_share.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <vector>

namespace r3::heat {

// Advance by the actual number of completed observations; interval length does not
// change the half-life. Observers supply coherent, uniquely credited route deltas.
struct Heat {
    uint32_t layers = 0, experts = 0;
    double half_life_observations = 1000.0;
    std::vector<double> counts;
    std::vector<uint64_t> samples;
    Heat(uint32_t l, uint32_t e, double half_life = 1000.0)
        : layers(l), experts(e), half_life_observations(half_life),
          counts(size_t(l) * e, 0.0), samples(size_t(l) * e, 0) {}
    bool tick(uint64_t observations) {
        if (!std::isfinite(half_life_observations) || half_life_observations <= 0) return false;
        const double factor = std::exp2(-double(observations) / half_life_observations);
        for (double& x : counts) x *= factor;
        return true;
    }
    bool observe(uint32_t layer, const int32_t* hist, size_t n, double weight = 1.0) {
        if (layer >= layers || !hist || n != experts || !std::isfinite(weight) || weight < 0) return false;
        for (size_t e = 0; e < n; ++e)
            if (hist[e] < 0 || !std::isfinite(counts[size_t(layer)*experts+e] + double(hist[e])*weight))
                return false;
        for (size_t e = 0; e < n; ++e) {
            const size_t i = size_t(layer)*experts+e;
            counts[i] += double(hist[e])*weight;
            ++samples[i];
        }
        return true;
    }
};
struct Device {
    uint64_t budget = 0;
    double cost = 1.0;     // static relative factor; calibrated curves come later
    double layer_cap = 0; // maximum unweighted heat per layer; 0 means unlimited
};
struct Move { uint32_t expert = 0; int32_t from = -1, to = -1; double saving = 0; };
struct Plan {
    bool valid = false;
    bool no_change = true;
    std::vector<int32_t> owner;
    std::vector<Move> delta;
    std::vector<uint64_t> used;
    double baseline = 0, predicted = 0;
};
struct Input {
    std::vector<uint64_t> bytes;
    std::vector<int32_t> current;              // -1 is engine-local, fixed in Stage 1
    std::vector<double> heat;
    std::vector<uint32_t> layer;
    std::vector<Device> devices;
    std::vector<std::vector<uint8_t>> allowed; // exact source/encoding/eligibility checks supplied by caller
    std::vector<uint64_t> samples;             // optional per-expert coverage
    std::vector<uint64_t> last_move_ns;        // optional; zero denotes no prior move
    uint64_t now_ns = 0;                      // monotonic snapshot age/dwell clock
    uint64_t observed_at_ns = 0;              // 0 = coverage age unavailable
    uint64_t max_heat_age_ns = 300'000'000'000ULL;
    uint64_t min_residency_ns = 1'800'000'000'000ULL; // 30 minutes
    uint64_t cooldown_until_ns = 0;
    uint64_t max_moved_bytes = 256ULL << 20;
    uint64_t min_samples = 0;                // chosen from coherent RHP02 coverage
    double margin = 0.10;                   // minimum fractional *global* peak-load improvement
    uint32_t max_moves = 8;
    bool deterministic = false;
};
inline bool validate(const Input& in, std::vector<uint64_t>* used) {
    const size_t n = in.bytes.size(), h = in.devices.size();
    if (!used || !n || !h || h > worker_share::kMaxHelperShares ||
        in.current.size() != n || in.heat.size() != n || in.layer.size() != n ||
        in.allowed.size() != n || (!in.samples.empty() && in.samples.size() != n) ||
        (!in.last_move_ns.empty() && in.last_move_ns.size() != n) ||
        !std::isfinite(in.margin) || in.margin < 0 || in.margin >= 1 ||
        in.observed_at_ns > in.now_ns) return false;
    used->assign(h, 0);
    for (const Device& d : in.devices)
        if (!std::isfinite(d.cost) || d.cost <= 0 || !std::isfinite(d.layer_cap) || d.layer_cap < 0)
            return false;
    std::map<uint32_t, std::vector<double>> loads;
    for (size_t i = 0; i < n; ++i) {
        const int32_t o = in.current[i];
        if (!in.bytes[i] || !std::isfinite(in.heat[i]) || in.heat[i] < 0 ||
            in.allowed[i].size() != h || o < -1 || o >= int32_t(h) ||
            (!in.last_move_ns.empty() && in.last_move_ns[i] > in.now_ns)) return false;
        if (o < 0) continue;
        const size_t d = size_t(o);
        if (!in.allowed[i][d]) return false;
        if ((*used)[d] > in.devices[d].budget ||
            in.bytes[i] > in.devices[d].budget - (*used)[d]) return false;
        (*used)[d] += in.bytes[i];
        auto& row = loads[in.layer[i]];
        if (row.empty()) row.resize(h);
        row[d] += in.heat[i];
        if (!std::isfinite(row[d])) return false;
    }
    for (const auto& [ly, row] : loads)
        for (size_t d = 0; d < h; ++d)
            if (in.devices[d].layer_cap > 0 && row[d] > in.devices[d].layer_cap + 1e-8)
                return false;
    return true;
}
inline bool delta_matches(const Plan& p, const std::vector<int32_t>& before) {
    if (!p.valid || p.owner.size() != before.size()) return false;
    std::vector<int32_t> copy = before;
    for (const Move& m : p.delta) {
        if (m.expert >= copy.size() || copy[m.expert] != m.from) return false;
        copy[m.expert] = m.to;
    }
    return copy == p.owner;
}
inline std::vector<int32_t> static_tail(const std::vector<int64_t>& bytes, uint32_t helpers,
                                        const uint32_t* shares, const uint64_t* budgets) {
    return worker_share::plan_tail(bytes, helpers, shares, budgets);
}
inline Plan propose(const Input& in) {
    Plan out;
    if (!validate(in, &out.used)) return out;
    out.valid = true; out.owner = in.current;
    const size_t n = in.bytes.size(), h = in.devices.size();
    std::map<uint32_t, size_t> layers;
    for (uint32_t ly : in.layer) layers.emplace(ly, layers.size());
    const size_t lcount = layers.size();
    std::vector<size_t> li(n);
    for (size_t i=0;i<n;++i) li[i] = layers.at(in.layer[i]);
    std::vector<double> loads(h*lcount, 0);
    for (size_t i=0;i<n;++i)
        if (in.current[i] >= 0) loads[size_t(in.current[i])*lcount+li[i]] += in.heat[i];
    auto peak = [&](size_t ly, const std::vector<double>& a) {
        double mx = 0;
        for (size_t d=0; d<h; ++d) mx = std::max(mx, a[d*lcount+ly]*in.devices[d].cost);
        return mx;
    };
    for (size_t ly=0;ly<lcount;++ly) out.baseline += peak(ly, loads);
    out.predicted = out.baseline;
    if (!out.baseline || in.deterministic || !in.max_moves || !in.max_moved_bytes ||
        in.now_ns < in.cooldown_until_ns ||
        (in.observed_at_ns && in.now_ns - in.observed_at_ns > in.max_heat_age_ns))
        return out;
    auto movable = [&](size_t i) {
        return in.current[i] >= 0 && in.heat[i] > 0 &&
               (in.samples.empty() || in.samples[i] >= in.min_samples) &&
               (in.last_move_ns.empty() || !in.last_move_ns[i] ||
                in.now_ns - in.last_move_ns[i] >= in.min_residency_ns);
    };
    // Assess a group of 1 or 2 changes atomically (including capacity and per-layer caps).
    // The objective is the sum over layers of the maximum *weighted* helper load.
    auto evaluation = [&](const std::vector<int32_t>& owner, const std::vector<uint64_t>& used,
                          const std::vector<double>& load, size_t a, int32_t to_a,
                          size_t b, int32_t to_b) -> double {
        const bool two = b < n;
        if (!movable(a) || owner[a] == to_a || to_a < 0 || size_t(to_a) >= h ||
            !in.allowed[a][size_t(to_a)] ||
            (two && (!movable(b) || a == b || owner[b] == to_b ||
                     to_b < 0 || size_t(to_b) >= h || !in.allowed[b][size_t(to_b)])))
            return -1;
        std::vector<uint64_t> u=used;
        std::array<size_t,2> ids={a,b};
        std::array<int32_t,2> dest={to_a,to_b};
        const size_t count = two ? 2 : 1;
        for (size_t k=0;k<count;++k) u[size_t(owner[ids[k]])] -= in.bytes[ids[k]];
        for (size_t k=0;k<count;++k) {
            size_t d=size_t(dest[k]);
            if (u[d] > in.devices[d].budget || in.bytes[ids[k]] > in.devices[d].budget-u[d]) return -1;
            u[d] += in.bytes[ids[k]];
        }
        double improvement=0;
        for (size_t k=0;k<count;++k) {
            const size_t ly=li[ids[k]];
            if (k && ly==li[a]) continue;
            double oldmax=peak(ly,load), newmax=0;
            for (size_t d=0;d<h;++d) {
                double v=load[d*lcount+ly];
                for (size_t j=0;j<count;++j) if (li[ids[j]]==ly) {
                    if (owner[ids[j]]==int32_t(d)) v-=in.heat[ids[j]];
                    if (dest[j]==int32_t(d)) v+=in.heat[ids[j]];
                }
                if (v < -1e-6 || (in.devices[d].layer_cap > 0 &&
                    v > in.devices[d].layer_cap + 1e-8)) return -1;
                newmax=std::max(newmax, v*in.devices[d].cost);
            }
            improvement+=oldmax-newmax;
        }
        return improvement;
    };
    auto finish = [&](const std::vector<int32_t>& owner, double objective) {
        if (out.baseline - objective <= out.baseline * in.margin + 1e-9) return;
        out.owner = owner; out.predicted = objective; out.no_change = false;
        out.used.assign(h,0);
        for (size_t i=0;i<n;++i) if (owner[i]>=0) out.used[size_t(owner[i])] += in.bytes[i];
        for (size_t i=0;i<n;++i) if (owner[i]!=in.current[i])
            out.delta.push_back({uint32_t(i),in.current[i],owner[i],out.baseline-objective});
    };
    // Exact branch: full assignment oracle for small instances, with whole-transaction
    // budget/caps, including swaps whose individual legs are not improvements.
    if (n <= 8) {
        std::vector<int32_t> choice=in.current, best=in.current;
        double best_cost=out.baseline;
        auto search = [&](auto&& self,size_t idx,uint32_t moves,uint64_t bytes)->void {
            if (idx==n) {
                std::vector<uint64_t> u(h,0);
                std::vector<double> a(h*lcount,0);
                for (size_t i=0;i<n;++i) if (choice[i]>=0) {
                    size_t d=size_t(choice[i]);
                    if (u[d]>in.devices[d].budget ||
                        in.bytes[i]>in.devices[d].budget-u[d]) return;
                    u[d]+=in.bytes[i]; a[d*lcount+li[i]]+=in.heat[i];
                }
                for (size_t d=0;d<h;++d) for (size_t ly=0;ly<lcount;++ly)
                    if (in.devices[d].layer_cap>0 &&
                        a[d*lcount+ly]>in.devices[d].layer_cap+1e-8) return;
                double score=0;
                for (size_t ly=0;ly<lcount;++ly) score+=peak(ly,a);
                if (score<best_cost-1e-9 || (std::abs(score-best_cost)<=1e-9 && score<out.baseline-1e-9 &&
                    std::lexicographical_compare(choice.begin(),choice.end(),best.begin(),best.end()))) {
                    best_cost=score; best=choice;
                }
                return;
            }
            self(self,idx+1,moves,bytes);
            if (!movable(idx)) return;
            for (size_t d=0;d<h;++d) {
                if (int32_t(d)==in.current[idx] || !in.allowed[idx][d] ||
                    moves>=in.max_moves || in.bytes[idx]>in.max_moved_bytes-bytes) continue;
                choice[idx]=int32_t(d);
                self(self,idx+1,moves+1,bytes+in.bytes[idx]);
            }
            choice[idx]=in.current[idx];
        };
        search(search,0,0,0);
        finish(best,best_cost);
        return out;
    }
    // Deterministic bounded local search for real model sizes. Choose high-impact
    // bottleneck owners plus destination partners; never reuse a moved expert.
    std::vector<size_t> candidates;
    std::vector<uint8_t> seen(n,0);
    auto add=[&](size_t i) {
        if (i<n && !seen[i] && candidates.size()<768) {
            candidates.push_back(i); seen[i]=1;
        }
    };
    std::vector<size_t> ranked(n);
    std::iota(ranked.begin(),ranked.end(),0);
    auto importance=[&](size_t i) {
        return in.current[i]<0 ? 0.0 :
            in.heat[i]*in.devices[size_t(in.current[i])].cost;
    };
    std::sort(ranked.begin(),ranked.end(),[&](size_t a,size_t b){
        if (importance(a)!=importance(b)) return importance(a)>importance(b);
        return a<b;
    });
    for (size_t i=0;i<std::min(n,size_t(64));++i) add(ranked[i]);
    // Include each layer bottleneck's hot source AND every helper's cold
    // exchange partner: a global top-heat shortlist misses full-device swaps.
    std::vector<size_t> hot(h*lcount,n), cold(h*lcount,n);
    for (size_t i=0;i<n;++i) if (in.current[i]>=0) {
        size_t k=size_t(in.current[i])*lcount+li[i];
        if (hot[k]==n || in.heat[i]>in.heat[hot[k]]) hot[k]=i;
        if (cold[k]==n || in.heat[i]<in.heat[cold[k]]) cold[k]=i;
    }
    for (size_t ly=0;ly<lcount;++ly) {
        size_t worst=0;
        for (size_t d=1;d<h;++d)
            if (loads[d*lcount+ly]*in.devices[d].cost >
                loads[worst*lcount+ly]*in.devices[worst].cost) worst=d;
        add(hot[worst*lcount+ly]);
    }
    for (size_t ly=0;ly<lcount;++ly)
        for (size_t d=0;d<h;++d) add(cold[d*lcount+ly]);
    std::sort(candidates.begin(),candidates.end());
    std::vector<int32_t> owner=in.current;
    std::vector<uint64_t> used=out.used;
    uint64_t bytes=0;
    uint32_t moves=0;
    double objective=out.baseline;
    while (moves<in.max_moves) {
        double best_gain=1e-9;
        size_t ba=n, bb=n;
        int32_t da=-1, db=-1;
        auto prefer=[&](double gain,size_t a,size_t b,int32_t ta,int32_t tb) {
            if (gain>best_gain+1e-9 ||
                (std::abs(gain-best_gain)<=1e-9 && gain>1e-9 &&
                 std::array<int64_t,4>{int64_t(a),int64_t(b),ta,tb} <
                 std::array<int64_t,4>{int64_t(ba),int64_t(bb),da,db})) {
                best_gain=gain;ba=a;bb=b;da=ta;db=tb;
            }
        };
        for (size_t a : candidates) {
            if (owner[a]!=in.current[a] || !movable(a) ||
                in.bytes[a]>in.max_moved_bytes-bytes) continue;
            for (size_t d=0;d<h;++d) {
                if (int32_t(d)==owner[a]) continue;
                prefer(evaluation(owner,used,loads,a,int32_t(d),n,-1),a,n,int32_t(d),-1);
            }
            if (moves+2>in.max_moves) continue;
            for (size_t b : candidates) {
                if (b<=a || owner[b]==owner[a] || owner[b]!=in.current[b] ||
                    in.bytes[b]>in.max_moved_bytes-bytes-in.bytes[a]) continue;
                // Exchange exactly two whole experts in one transaction.
                prefer(evaluation(owner,used,loads,a,owner[b],b,owner[a]),
                       a,b,owner[b],owner[a]);
            }
        }
        if (ba==n) break;
        std::array<size_t,2> ix={ba,bb};
        std::array<int32_t,2> dst={da,db};
        const size_t count=bb<n?2:1;
        for (size_t k=0;k<count;++k) {
            const size_t i=ix[k];
            loads[size_t(owner[i])*lcount+li[i]]-=in.heat[i];
            used[size_t(owner[i])]-=in.bytes[i];
        }
        for (size_t k=0;k<count;++k) {
            const size_t i=ix[k];
            owner[i]=dst[k]; loads[size_t(owner[i])*lcount+li[i]]+=in.heat[i];
            used[size_t(owner[i])]+=in.bytes[i]; bytes+=in.bytes[i]; ++moves;
        }
        objective-=best_gain;
    }
    finish(owner,objective);
    return out;
}
} // namespace r3::heat
