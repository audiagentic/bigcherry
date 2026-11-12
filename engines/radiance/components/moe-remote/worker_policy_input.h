#pragma once
// Stage-1 policy input adapter. Pure CPU, on detached snapshots only. No
// machine-specific budgets, step-size fallbacks or inferred source eligibility.
#include "worker_heat.h"
#include "worker_heat_source.h"
#include "worker_residency.h"
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace r3::policy {

// Owner map + engine's exact runtime slot eligibility, captured from v23's
// ACTIVE generation. This is NOT the engine-local/helper partition planner.
struct OwnerSnapshot {
    uint32_t layers = 0, experts = 0, helpers = 0;
    uint64_t epoch = 0, digest = 0;
    std::vector<int32_t> owner;      // -1 = local; helper indices otherwise
    std::vector<uint8_t> runtime;   // 1 only for ready nonnegative slot
};
inline bool capture_owners(const void* map, OwnerSnapshot* out) {
    using namespace moe_shm;
    if (!map || !out) return false;
    const Header* h = static_cast<const Header*>(map);
    if (!h->layer_count || h->layer_count > kMaxLayers ||
        !h->expert_count || h->layer_map_slot_bytes != uint64_t(kMaxLayers) * h->expert_count * sizeof(int32_t) ||
        !h->helper_count || h->helper_count > kMaxHelpers ||
        h->owner_map_bytes != uint64_t(h->layer_count) * h->expert_count)
        return false;
    PlacementControl a{}, b{};
    if (!placement_snapshot(map, &a) || a.state != uint32_t(PlacementState::ACTIVE) ||
        a.fault_code || a.active_slot > 1 || !a.active_digest)
        return false;
    const GenerationHeader* g = generation_header(map, a.active_slot);
    if (!g || g->state != uint32_t(GenerationState::ACTIVE) ||
        g->epoch != a.active_epoch || g->digest != a.active_digest ||
        g->layer_count != h->layer_count || g->expert_count != h->expert_count)
        return false;
    OwnerSnapshot copy;
    copy.layers = h->layer_count; copy.experts = h->expert_count;
    copy.helpers = h->helper_count;
    copy.epoch = a.active_epoch; copy.digest = a.active_digest;
    const size_t n = size_t(h->layer_count) * h->expert_count;
    copy.owner.reserve(n); copy.runtime.reserve(n);
    const uint8_t* source = generation_map(map, a.active_slot);
    for (uint32_t l = 0; l < h->layer_count; ++l) {
        const bool ready = load(&h->layer_map_ready[l]) != 0;
        for (uint32_t e = 0; e < h->expert_count; ++e) {
            const uint8_t owner = __atomic_load_n(source + size_t(l) * h->expert_count + e, __ATOMIC_ACQUIRE);
            if (owner != kOwnerLocal && owner >= h->helper_count) return false;
            const bool accessible = ready && layer_slot(map, l, e) >= 0;
            if (owner != kOwnerLocal && !accessible) return false;
            copy.owner.push_back(owner == kOwnerLocal ? -1 : int32_t(owner));
            copy.runtime.push_back(uint8_t(accessible));
        }
    }
    if (!placement_snapshot(map, &b) || b.seq != a.seq ||
        b.state != a.state || b.active_slot != a.active_slot ||
        b.active_epoch != a.active_epoch || b.active_digest != a.active_digest ||
        generation_digest(map, a.active_slot) != a.active_digest)
        return false;
    *out = std::move(copy);
    return true;
}

struct ExpertView {
    uint64_t bytes = 0;                // measured physical resident bytes
    uint64_t last_move_ns = 0;         // timestamp from committed migration
    bool protected_expert = false;     // includes protected BF16 and MTP
    // Each byte must come from exact per-helper source, format, GPU ISA,
    // min/max rows and width/TP eligibility. No default "all eligible".
    std::vector<uint8_t> target_eligible;
};
struct LedgerView {
    uint64_t hard = 0, scratch = 0, headroom = 0, charged = 0;
    std::vector<residency::Slot> slots;
};
inline LedgerView copy_ledger(const residency::Ledger& ledger) {
    return {ledger.hard(), ledger.total() - ledger.charged(), ledger.headroom(),
            ledger.charged(), ledger.slots()};
}
// Apply one coherent interval atomically. A repeated, missing or regressing
// observation sequence never decays heat twice. Incomplete diagnostic-file
// windows are intentionally not accepted as policy evidence.
inline bool consume_heat(const HeatDelta& delta, heat::Heat* heat,
                         uint64_t* previously_consumed, bool accept_unwatermarked_heat = false) {
    if (!heat || !previously_consumed || !delta.complete_cells ||
        !(delta.complete_observations || (accept_unwatermarked_heat &&
          delta.provenance == HeatProvenance::ENGINE_HISTOGRAM_FILE)) ||
        delta.layers != heat->layers || delta.experts != heat->experts ||
        delta.counts.size() != heat->counts.size() ||
        !delta.interval_observations ||
        delta.observation_count <= *previously_consumed ||
        delta.observation_count - *previously_consumed != delta.interval_observations)
        return false;
    uint64_t total = 0;
    for (int32_t count : delta.counts) {
        if (count < 0 || total > UINT64_MAX - uint64_t(count)) return false;
        total += uint64_t(count);
    }
    if (total != delta.interval_observations) return false;
    heat::Heat updated = *heat;
    if (!updated.tick(delta.interval_observations)) return false;
    for (uint32_t l = 0; l < heat->layers; ++l)
        if (!updated.observe(l, delta.counts.data() + size_t(l) * heat->experts,
                             heat->experts))
            return false;
    *heat = std::move(updated);
    *previously_consumed = delta.observation_count;
    return true;
}

struct PolicyControls {
    // All required from explicit configuration. Zero/unset fields fail closed.
    double half_life_observations = 0;
    double margin = 0;
    uint64_t min_residency_ns = 0;
    uint64_t max_heat_age_ns = 0;
    uint64_t max_moved_bytes = 0;
    uint64_t min_samples = 0;
    uint32_t max_moves = 0;
};
inline bool controls_valid(const PolicyControls& c) {
    return std::isfinite(c.half_life_observations) && c.half_life_observations > 0 &&
           std::isfinite(c.margin) && c.margin > 0 && c.margin < 1 &&
           c.min_residency_ns && c.max_heat_age_ns && c.max_moved_bytes &&
           c.min_samples && c.max_moves;
}

inline bool build_input(const OwnerSnapshot& owners,
                        const std::vector<ExpertView>& experts,
                        const std::vector<LedgerView>& ledgers,
                        const heat::Heat& heat,
                        const HeatDelta& observation,
                        const PolicyControls& controls,
                        uint64_t now_ns, uint64_t cooldown_until_ns,
                        bool deterministic, heat::Input* result,
                        std::string* why = nullptr,
                        bool accept_unwatermarked_heat = false) {
    auto reject = [&](const char* explanation) {
        if (why) *why = explanation;
        return false;
    };
    const size_t n = size_t(owners.layers) * owners.experts;
    if (!result || !controls_valid(controls) || !owners.layers ||
        !owners.experts || !owners.helpers || owners.helpers > moe_shm::kMaxHelpers ||
        !owners.epoch || !owners.digest || !now_ns ||
        n != owners.owner.size() || n != owners.runtime.size() ||
        n != experts.size() || ledgers.size() != owners.helpers ||
        heat.layers != owners.layers || heat.experts != owners.experts ||
        heat.counts.size() != n || heat.samples.size() != n ||
        observation.layers != owners.layers || observation.experts != owners.experts ||
        observation.counts.size() != n || !observation.complete_cells)
        return reject("policy: incomplete or mismatched live snapshot");
    if (!observation.complete_observations &&
        !(accept_unwatermarked_heat &&
          observation.provenance == HeatProvenance::ENGINE_HISTOGRAM_FILE))
        return reject("policy: heat source lacks coherent observation coverage");
    if (!observation.observation_count || !observation.interval_observations ||
        observation.age_ns > controls.max_heat_age_ns ||
        observation.age_ns >= now_ns)
        return reject("policy: stale or insufficient observation window");
    if (heat.half_life_observations != controls.half_life_observations)
        return reject("policy: decay settings changed without resetting heat");
    heat::Input in;
    in.now_ns = now_ns;
    in.observed_at_ns = now_ns - observation.age_ns;
    in.cooldown_until_ns = cooldown_until_ns;
    in.min_residency_ns = controls.min_residency_ns;
    in.max_heat_age_ns = controls.max_heat_age_ns;
    in.max_moved_bytes = controls.max_moved_bytes;
    in.min_samples = controls.min_samples;
    in.margin = controls.margin;
    in.max_moves = controls.max_moves;
    in.deterministic = deterministic;
    in.current = owners.owner;
    in.heat = heat.counts;
    in.samples = heat.samples;
    in.bytes.reserve(n); in.allowed.reserve(n);
    in.last_move_ns.reserve(n);
    std::vector<uint64_t> sums(owners.helpers, 0);
    std::vector<uint8_t> ledger_owners(n, 0);
    for (size_t i = 0; i < n; ++i) {
        const auto& e = experts[i];
        const int32_t owner = owners.owner[i];
        if (!e.bytes || e.target_eligible.size() != owners.helpers ||
            e.last_move_ns > now_ns || owner < -1 || owner >= int32_t(owners.helpers))
            return reject("policy: invalid expert metadata");
        if (owner >= 0 && (!owners.runtime[i] ||
                           !e.target_eligible[size_t(owner)]))
            return reject("policy: active owner violates runtime eligibility");
        in.bytes.push_back(e.bytes);
        in.last_move_ns.push_back(e.last_move_ns);
        std::vector<uint8_t> allowed(owners.helpers, 0);
        // Local experts are fixed in stage 1. Protected experts may never
        // be moved, even if a source file happens to exist on a helper.
        if (owner >= 0 && owners.runtime[i])
            for (uint32_t d = 0; d < owners.helpers; ++d)
                allowed[d] = uint8_t(e.target_eligible[d] == 1 &&
                                     (!e.protected_expert || d == uint32_t(owner)));
        in.allowed.push_back(std::move(allowed));
        if (owner >= 0) {
            const size_t d = size_t(owner);
            if (sums[d] > UINT64_MAX - e.bytes)
                return reject("policy: resident byte count overflow");
            sums[d] += e.bytes;
        }
    }
    for (size_t d = 0; d < ledgers.size(); ++d) {
        const auto& ledger = ledgers[d];
        if (ledger.hard < ledger.scratch ||
            ledger.headroom > ledger.hard - ledger.scratch ||
            ledger.charged > ledger.hard - ledger.scratch - ledger.headroom)
            return reject("policy: invalid helper ledger");
        uint64_t charged = 0;
        for (const auto& s : ledger.slots) {
            if (s.state == residency::State::FREE) continue;
            if (s.state != residency::State::ACTIVE || !s.bytes ||
                s.expert >= n || owners.owner[s.expert] != int32_t(d) ||
                s.bytes != experts[s.expert].bytes ||
                ledger_owners[s.expert]++)
                return reject("policy: helper ledger not an active exclusive owner snapshot");
            if (charged > UINT64_MAX - s.bytes)
                return reject("policy: ledger slot overflow");
            charged += s.bytes;
        }
        if (charged != ledger.charged || charged != sums[d])
            return reject("policy: ledger and active owner map disagree");
        in.devices.push_back({ledger.hard - ledger.scratch - ledger.headroom});
    }
    std::vector<uint64_t> used;
    if (!heat::validate(in, &used))
        return reject("policy: candidate violates hard budget or policy invariants");
    *result = std::move(in);
    return true;
}
} // namespace r3::policy
