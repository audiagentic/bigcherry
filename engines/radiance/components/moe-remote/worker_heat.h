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
    uint64_t budget = 0;   // bytes of helper-owned experts this helper may hold
};
struct Move { uint32_t expert = 0; int32_t from = -1, to = -1; double saving = 0; };
struct Plan {
    bool valid = false;
    bool no_change = true;
    std::vector<int32_t> owner;
    std::vector<Move> delta;
    std::vector<uint64_t> used;
    // Share of the helper-owned heat that sits on the LAST helper (the tail) before and after the transaction: the
    // static placer puts the coldest experts there, so lower is closer to its plan. heat_total is the denominator.
    double baseline = 0, predicted = 0, heat_total = 0;
};
struct Input {
    std::vector<uint64_t> bytes;
    std::vector<int32_t> current;              // -1 is engine-local, fixed
    std::vector<double> heat;
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
    double margin = 0.10;                   // minimum fractional improvement of the tail's heat share
    uint32_t max_moves = 8;
    bool deterministic = false;
};
inline bool validate(const Input& in, std::vector<uint64_t>* used) {
    const size_t n = in.bytes.size(), h = in.devices.size();
    if (!used || !n || !h || h > worker_share::kMaxHelperShares ||
        in.current.size() != n || in.heat.size() != n ||
        in.allowed.size() != n || (!in.samples.empty() && in.samples.size() != n) ||
        (!in.last_move_ns.empty() && in.last_move_ns.size() != n) ||
        !std::isfinite(in.margin) || in.margin < 0 || in.margin >= 1 ||
        in.observed_at_ns > in.now_ns) return false;
    used->assign(h, 0);
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
    }
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
// The static placer's own plan (worker_share::plan_tail: hottest first, head helpers by their shares up to their budgets,
// the tail helper the coldest that fit) applied to the LIVE decayed heat is the target owner map. Moves toward it are taken
// hottest first, each as a whole group: a move that overflows its destination at the end of the transaction pulls in the coldest
// move out of that destination, and so on along the chain, so budgets hold at the one atomic epoch and never per leg. A group
// that does not fit max_moves / max_moved_bytes is skipped whole. `head_shares` has one weight a helper, the launch's
// R3_MOE_WORKER_SHARES (the last, the tail's, is not read by the placer).
inline Plan propose(const Input& in, const std::vector<uint32_t>& head_shares) {
    Plan out;
    if (!validate(in, &out.used) || in.devices.size() < 2 || head_shares.size() != in.devices.size()) return out;
    out.valid = true; out.owner = in.current;
    const size_t n = in.bytes.size(), h = in.devices.size();
    const int32_t tail = int32_t(h) - 1;
    for (size_t i = 0; i < n; ++i)
        if (in.current[i] >= 0) {
            out.heat_total += in.heat[i];
            if (in.current[i] == tail) out.baseline += in.heat[i];
        }
    if (out.heat_total > 0) out.baseline /= out.heat_total;
    out.predicted = out.baseline;
    if (!out.heat_total || in.deterministic || !in.max_moves || !in.max_moved_bytes ||
        in.now_ns < in.cooldown_until_ns ||
        (in.observed_at_ns && in.now_ns - in.observed_at_ns > in.max_heat_age_ns))
        return out;
    // Target map: the static plan over the helper-owned experts ranked by live heat.
    std::vector<size_t> ranked;
    for (size_t i = 0; i < n; ++i) if (in.current[i] >= 0) ranked.push_back(i);
    std::sort(ranked.begin(), ranked.end(), [&](size_t a, size_t b) {
        if (in.heat[a] != in.heat[b]) return in.heat[a] > in.heat[b];
        return a < b;
    });
    std::vector<int64_t> ranked_bytes(ranked.size());
    for (size_t k = 0; k < ranked.size(); ++k) ranked_bytes[k] = int64_t(in.bytes[ranked[k]]);
    uint64_t budgets[worker_share::kMaxHelperShares] = {};
    for (size_t d = 0; d < h; ++d) budgets[d] = in.devices[d].budget;
    const std::vector<int32_t> placed = worker_share::plan_tail(ranked_bytes, uint32_t(h), head_shares.data(), budgets);
    std::vector<int32_t> target = in.current;
    for (size_t k = 0; k < ranked.size(); ++k) if (placed[k] >= 0) target[ranked[k]] = placed[k];
    auto movable = [&](size_t i) {
        return in.current[i] >= 0 &&
               (in.samples.empty() || in.samples[i] >= in.min_samples) &&
               (in.last_move_ns.empty() || !in.last_move_ns[i] ||
                in.now_ns - in.last_move_ns[i] >= in.min_residency_ns);
    };
    auto wanted = [&](size_t i) {
        return target[i] != in.current[i] && target[i] >= 0 && movable(i) && in.allowed[i][size_t(target[i])];
    };
    // Seeds are the diff moves of experts that carry heat, hottest first; a cold expert only moves as a group's partner.
    std::vector<size_t> seeds;
    std::vector<std::vector<size_t>> out_of(h);   // diff moves by source helper, coldest first
    for (size_t i = 0; i < n; ++i) if (wanted(i)) {
        if (in.heat[i] > 0) seeds.push_back(i);
        out_of[size_t(in.current[i])].push_back(i);
    }
    std::sort(seeds.begin(), seeds.end(), [&](size_t a, size_t b) {
        if (in.heat[a] != in.heat[b]) return in.heat[a] > in.heat[b];
        return a < b;
    });
    for (auto& v : out_of)
        std::sort(v.begin(), v.end(), [&](size_t a, size_t b) {
            if (in.heat[a] != in.heat[b]) return in.heat[a] < in.heat[b];
            return a < b;
        });
    std::vector<size_t> cursor(h, 0);
    std::vector<uint8_t> taken(n, 0);
    std::vector<int32_t> owner = in.current;
    std::vector<uint64_t> used = out.used;
    uint64_t moved_bytes = 0;
    uint32_t moves = 0;
    for (size_t seed : seeds) {
        if (moves >= in.max_moves) break;
        if (taken[seed]) continue;
        std::vector<size_t> group{seed};
        std::vector<uint64_t> u = used;
        std::vector<size_t> probe = cursor;   // partner scans advance on a copy until the group is accepted
        auto apply = [&](size_t i) {
            u[size_t(in.current[i])] -= in.bytes[i];
            u[size_t(target[i])] += in.bytes[i];
        };
        apply(seed);
        uint64_t group_bytes = in.bytes[seed];
        bool ok = true;
        for (;;) {
            size_t over = h;
            for (size_t d = 0; d < h; ++d) if (u[d] > in.devices[d].budget) { over = d; break; }
            if (over == h) break;
            size_t& at = probe[over];
            size_t partner = n;
            for (; at < out_of[over].size(); ++at) {
                const size_t c = out_of[over][at];
                if (taken[c] || std::find(group.begin(), group.end(), c) != group.end()) continue;
                partner = c; break;
            }
            if (partner == n || group.size() >= in.max_moves) { ok = false; break; }
            group.push_back(partner);
            group_bytes += in.bytes[partner];
            apply(partner);
        }
        if (!ok || group.size() > in.max_moves - moves ||
            group_bytes > in.max_moved_bytes - moved_bytes) continue;
        for (size_t i : group) {
            taken[i] = 1;
            owner[i] = target[i];
            ++moves;
        }
        used = u;
        moved_bytes += group_bytes;
        cursor = probe;
    }
    double tail_heat = 0;
    for (size_t i = 0; i < n; ++i) if (owner[i] == tail) tail_heat += in.heat[i];
    const double predicted = tail_heat / out.heat_total;
    if (out.baseline - predicted <= out.baseline * in.margin + 1e-9) return out;
    out.owner = owner; out.predicted = predicted; out.no_change = false; out.used = used;
    for (size_t i = 0; i < n; ++i) if (owner[i] != in.current[i])
        out.delta.push_back({uint32_t(i), in.current[i], owner[i], out.baseline - predicted});
    std::sort(out.delta.begin(), out.delta.end(), [&](const Move& a, const Move& b) {
        if (in.heat[a.expert] != in.heat[b.expert]) return in.heat[a.expert] > in.heat[b.expert];
        return a.expert < b.expert;
    });
    return out;
}
} // namespace r3::heat
