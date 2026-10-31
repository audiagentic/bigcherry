#pragma once

// Decayed route heat and a deterministic, budgeted ownership proposal (RHP04 seed, RAD23). Pure CPU policy: no GPU, environment or
// filesystem access, no allocation beyond its own vectors, and nothing here changes a live ownership map. A plan is a proposal for a
// restart or idle-point transaction (RHP05); publishing one into a running mailbox is a protocol change (RHP11), not this file.
//
// Experts use one canonical model-global order (layer * experts + expert). Owner -1 is "outside the helper set" (the engine's own), and
// this slice never moves an expert to or from -1: engine residency is not described by Device.

#include "worker_share.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace r3::heat {

// Decayed per-(layer, expert) route counts. One decay tick is one policy interval: call tick() once per interval, not once per histogram.
struct Heat {
    uint32_t layers = 0, experts = 0;
    double decay = 0.95;
    std::vector<double> counts;
    std::vector<uint64_t> samples;   // histograms folded in per expert: a count of 0 with samples 0 is "never observed", not "no demand"

    Heat(uint32_t l, uint32_t e, double d = 0.95) : layers(l), experts(e), decay(d), counts(size_t(l) * e, 0.0), samples(size_t(l) * e, 0) {}

    bool tick() {
        if (!std::isfinite(decay) || decay < 0.0 || decay > 1.0) return false;
        for (double& x : counts) x *= decay;
        return true;
    }
    // An already completed, uniquely credited histogram (the caller deduplicates ranks and says its multiplicity as weight). All or nothing.
    bool observe(uint32_t layer, const int32_t* hist, size_t n, double weight = 1.0) {
        if (layer >= layers || !hist || n != experts || !std::isfinite(weight) || weight < 0.0) return false;
        for (size_t e = 0; e < n; ++e)
            if (hist[e] < 0 || !std::isfinite(counts[size_t(layer) * experts + e] + double(hist[e]) * weight)) return false;
        for (size_t e = 0; e < n; ++e) {
            const size_t i = size_t(layer) * experts + e;
            counts[i] += double(hist[e]) * weight;
            ++samples[i];
        }
        return true;
    }
};

struct Device {
    uint64_t budget = 0;   // bytes
    double cost = 1.0;     // per routed row, in one consistent unit across the devices
};

struct Move { uint32_t expert = 0; int32_t from = -1, to = -1; double saving = 0.0; };

struct Plan {
    bool valid = false;
    std::vector<int32_t> owner;     // the whole proposed map
    std::vector<Move> delta;        // current -> owner
    std::vector<uint64_t> used;     // bytes per device in owner
};

struct Input {
    std::vector<uint64_t> bytes;                  // exact laid-out size of each expert (> 0)
    std::vector<int32_t> current;                 // owner now: -1 or a device index
    std::vector<double> heat;                     // decayed demand
    std::vector<Device> devices;
    std::vector<std::vector<uint8_t>> allowed;    // allowed[expert][device]: eligibility (protected / unsupported experts are all 0 and stay)
    double margin = 0.10;                         // least fractional cost improvement that justifies a move (hysteresis)
    uint32_t max_moves = 0;                       // 0 freezes the map
};

// An invalid active map is a control-plane fault: report it, never repair it.
inline bool validate(const Input& in, std::vector<uint64_t>* used) {
    const size_t n = in.bytes.size(), h = in.devices.size();
    if (!used || h == 0 || h > worker_share::kMaxHelperShares || in.current.size() != n || in.heat.size() != n || in.allowed.size() != n ||
        !std::isfinite(in.margin) || in.margin < 0.0) return false;
    used->assign(h, 0);
    for (const Device& d : in.devices)
        if (!std::isfinite(d.cost) || d.cost <= 0.0) return false;
    for (size_t i = 0; i < n; ++i) {
        const int32_t o = in.current[i];
        if (in.bytes[i] == 0 || !std::isfinite(in.heat[i]) || in.heat[i] < 0.0 || in.allowed[i].size() != h || o < -1 || o >= int32_t(h)) return false;
        if (o < 0) continue;
        if (!in.allowed[i][size_t(o)] || in.bytes[i] > in.devices[size_t(o)].budget - (*used)[size_t(o)]) return false;
        (*used)[size_t(o)] += in.bytes[i];
    }
    return true;
}

// Greedy, single-expert moves, each individually beneficial (new cost lower by more than `margin`) and fitting when chosen; no swaps, so a
// full faster device does not take an expert in exchange (RHP04 adds the bounded swap search). Ties break on expert index, then device.
inline Plan propose(const Input& in) {
    Plan p;
    if (!validate(in, &p.used)) return p;
    p.valid = true;
    p.owner = in.current;
    struct Candidate { uint32_t expert; int32_t from, to; double saving; };
    std::vector<Candidate> candidates;
    const size_t h = in.devices.size();
    for (size_t i = 0; i < in.bytes.size(); ++i) {
        const int32_t from = in.current[i];
        if (from < 0 || in.heat[i] == 0.0) continue;
        const double old_cost = in.devices[size_t(from)].cost;
        for (size_t j = 0; j < h; ++j) {
            if (int32_t(j) == from || !in.allowed[i][j] || in.devices[j].budget == 0) continue;
            const double new_cost = in.devices[j].cost;
            if (!(new_cost < old_cost) || (old_cost - new_cost) / old_cost <= in.margin) continue;
            const double saving = in.heat[i] * (old_cost - new_cost);
            if (!std::isfinite(saving) || saving <= 0.0) continue;
            candidates.push_back({uint32_t(i), from, int32_t(j), saving});
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        if (a.saving != b.saving) return a.saving > b.saving;
        if (a.expert != b.expert) return a.expert < b.expert;
        return a.to < b.to;
    });
    for (const Candidate& c : candidates) {
        if (p.delta.size() >= in.max_moves) break;
        if (p.owner[c.expert] != c.from) continue;
        const uint64_t b = in.bytes[c.expert];
        const size_t dst = size_t(c.to);
        if (b > in.devices[dst].budget - p.used[dst]) continue;
        p.used[size_t(c.from)] -= b;
        p.used[dst] += b;
        p.owner[c.expert] = c.to;
        p.delta.push_back({c.expert, c.from, c.to, c.saving});
    }
    return p;
}

// The static layout this slice is measured against: the same plan_tail the helpers compute at start, so a shadow run starts from the real map.
inline std::vector<int32_t> static_tail(const std::vector<int64_t>& bytes, uint32_t helpers, const uint32_t* shares, const uint64_t* budgets) {
    return worker_share::plan_tail(bytes, helpers, shares, budgets);
}

// A delta is a logical transition only: it says nothing about weights loaded, streams idle or helpers having acknowledged anything.
inline bool delta_matches(const Plan& p, const std::vector<int32_t>& before) {
    if (!p.valid || p.owner.size() != before.size()) return false;
    std::vector<int32_t> copy = before;
    for (const Move& m : p.delta) {
        if (m.expert >= copy.size() || copy[m.expert] != m.from) return false;
        copy[m.expert] = m.to;
    }
    return copy == p.owner;
}

}  // namespace r3::heat
