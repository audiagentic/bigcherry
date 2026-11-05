#pragma once
// Offline replay of the live placement policy: a recorded series of heat snapshots (the engine's cumulative route
// counts, one file per period) is fed through the same heat accounting and the same planner the driver uses, with each
// proposed transaction applied at once, so a policy or a derivation can be tried in seconds with no GPU and no server.
// What it does NOT model: transaction time and its disturbance of service, aborts, the effect gate, eligibility
// learned from refused destinations. It answers "what would the plan do to the tail's heat share, how many experts
// would move, and how fast does the hot set drift", which is what the derivations in RHP16 are made from.
#include "worker_heat.h"
#include "worker_policy_measure.h"
#include <cstdint>
#include <vector>

namespace r3::policy {

struct ReplayConfig {
    uint32_t layers = 0, experts = 0;       // the shape of one snapshot: layers x experts cumulative uint32 counts
    std::vector<uint64_t> budgets;          // bytes of helper-owned experts each helper may hold
    uint64_t stride = 0;                    // bytes of one expert
    std::vector<uint32_t> shares;           // one weight a helper, as the launch placed with
    double half_life_observations = 0;
    double margin = 0;
    uint32_t max_moves = 0;
    uint64_t max_moved_bytes = 0;
    uint64_t min_samples = 0;
    uint64_t period_ns = 0;                 // time between snapshots
    uint64_t min_residency_ns = 0;
    uint64_t cadence_snapshots = 1;         // propose every this many snapshots
};

struct ReplayStep {
    size_t snapshot = 0;
    uint64_t observations = 0;              // cumulative routed selections at this snapshot
    double tail_share_before = 0, tail_share_after = 0;
    size_t moved = 0;                       // experts that changed owner at this step
    bool proposed = false;
};

struct ReplayResult {
    bool valid = false;
    std::vector<ReplayStep> steps;
    size_t transactions = 0, experts_moved = 0;
    std::vector<int32_t> owner;             // after the last snapshot
    uint64_t drift_half_life_observations = 0, drift_horizon_observations = 0;
    double marginal_error = 0;
    size_t hot_set_k = 0;
};

// `snapshots[i]` is the cumulative count of every (layer, expert), row-major, at period i. `owner` is the starting
// owner map (index layer * experts + expert; -1 is engine-local and fixed), which must fit the budgets.
inline ReplayResult replay(const ReplayConfig& cfg, const std::vector<std::vector<uint32_t>>& snapshots,
                           std::vector<int32_t> owner) {
    ReplayResult out;
    const size_t n = size_t(cfg.layers) * cfg.experts, h = cfg.budgets.size();
    if (!cfg.layers || !cfg.experts || !cfg.stride || h < 2 || cfg.shares.size() != h || !cfg.period_ns ||
        !cfg.cadence_snapshots || owner.size() != n || snapshots.size() < 2) return out;
    for (const auto& s : snapshots) if (s.size() != n) return out;
    heat::Heat heat(cfg.layers, cfg.experts, cfg.half_life_observations);
    const size_t k = size_t(cfg.budgets.back() / cfg.stride);
    DriftTracker drift(std::max<size_t>(k, 1));
    std::vector<uint64_t> last_move(n, 0);
    uint64_t cumulative = 0;
    auto tail_share = [&](const std::vector<int32_t>& o) {
        double total = 0, tail = 0;
        for (size_t i = 0; i < n; ++i) if (o[i] >= 0) { total += heat.counts[i]; if (o[i] == int32_t(h) - 1) tail += heat.counts[i]; }
        return total > 0 ? tail / total : 0.0;
    };
    for (size_t s = 1; s < snapshots.size(); ++s) {
        uint64_t interval = 0;
        std::vector<int32_t> delta(cfg.experts);
        std::vector<double> interval_counts(n, 0.0);   // this interval's own counts, for the drift measurement
        for (uint32_t l = 0; l < cfg.layers; ++l) {
            for (uint32_t e = 0; e < cfg.experts; ++e) {
                const size_t i = size_t(l) * cfg.experts + e;
                const uint32_t a = snapshots[s - 1][i], b = snapshots[s][i];
                delta[e] = b >= a ? int32_t(std::min<uint32_t>(b - a, uint32_t(INT32_MAX))) : 0;   // a counter that went back credits nothing
                interval += uint64_t(delta[e]);
                interval_counts[i] = double(delta[e]);
            }
            if (!heat.observe(l, delta.data(), delta.size())) return out;
        }
        if (!interval) continue;
        heat.tick(interval);
        cumulative += interval;
        drift.add(interval_counts, cumulative);   // never the decayed heat: that depends on the half-life being informed
        ReplayStep step;
        step.snapshot = s; step.observations = cumulative;
        step.tail_share_before = step.tail_share_after = tail_share(owner);
        if (s % cfg.cadence_snapshots == 0) {
            heat::Input in;
            in.bytes.assign(n, cfg.stride);
            in.current = owner;
            in.heat = heat.counts;
            in.devices.resize(h);
            for (size_t d = 0; d < h; ++d) in.devices[d].budget = cfg.budgets[d];
            in.allowed.assign(n, std::vector<uint8_t>(h, 1));
            in.samples = heat.samples;
            in.last_move_ns = last_move;
            in.now_ns = uint64_t(s) * cfg.period_ns;
            in.min_residency_ns = cfg.min_residency_ns;
            in.max_moved_bytes = cfg.max_moved_bytes;
            in.min_samples = cfg.min_samples;
            in.margin = cfg.margin;
            in.max_moves = cfg.max_moves;
            const heat::Plan plan = heat::propose(in, cfg.shares);
            if (!plan.valid) return out;
            step.proposed = !plan.no_change;
            if (!plan.no_change) {
                for (const heat::Move& m : plan.delta) last_move[m.expert] = in.now_ns;
                step.moved = plan.delta.size();
                owner = plan.owner;
                ++out.transactions;
                out.experts_moved += step.moved;
                step.tail_share_after = tail_share(owner);
            }
        }
        out.steps.push_back(step);
    }
    out.valid = true;
    out.owner = owner;
    out.drift_half_life_observations = drift.half_life_observations();
    out.drift_horizon_observations = drift.horizon_observations();
    out.marginal_error = marginal_relative_error(heat.counts, std::max<size_t>(k, 1));
    out.hot_set_k = k;
    return out;
}

}  // namespace r3::policy
