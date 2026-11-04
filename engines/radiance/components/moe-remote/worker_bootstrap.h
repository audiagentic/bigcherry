#pragma once
#include "moe_remote_shm.h"
#include "worker_epoch.h"
#include "worker_generation_tables.h"
#include <cstring>
#include <vector>

namespace r3::epoch {
using namespace r3::moe_shm;

/* Only helper startup publishes slot 0. Once its controller seal is visible
 * there can be no additional updates to the active canonical owner bytes. */
template<class Owners>
bool publish_owned(void* map, const Owners& owned, uint32_t helper) {
    if (!map) return false;
    Header* h = static_cast<Header*>(map);
    GenerationHeader* g = generation_header(map, 0);
    uint8_t* owners = generation_map(map, 0);
    if (!owners || !g || helper >= h->helper_count ||
        __atomic_load_n(&g->ready_helpers, __ATOMIC_ACQUIRE)) return false;
    for (auto [l, e] : owned.expert) {
        if (l >= h->layer_count || e >= h->expert_count) return false;
        uint8_t* entry = owners + (size_t)l * h->expert_count + e;
        uint8_t expected = kOwnerLocal;
        if (!__atomic_compare_exchange_n(entry, &expected, (uint8_t)helper, false,
                                         __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) return false;
    }
    return true;
}

/* Filtering is allowed until the one-time bootstrap seal. */
inline bool drop_bootstrap_owner(void* map, uint32_t l, uint32_t e, uint32_t helper) {
    if (!map) return false;
    Header* h = static_cast<Header*>(map);
    GenerationHeader* g = generation_header(map, 0);
    if (!g || l >= h->layer_count || e >= h->expert_count ||
        helper >= h->helper_count ||
        __atomic_load_n(&g->ready_helpers, __ATOMIC_ACQUIRE)) return false;
    uint8_t* entry = generation_map(map, 0) + (size_t)l * h->expert_count + e;
    uint8_t expected = (uint8_t)helper;
    return __atomic_compare_exchange_n(entry, &expected, kOwnerLocal, false,
                                       __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}

/* Release only after this helper's *last* drop and layer load. No ownership
 * writer remains after setting its completion bit; helper 0 waits for all. */
inline bool mark_bootstrap_filtered(void* map, uint32_t helper) {
    if (!map) return false;
    Header* h = static_cast<Header*>(map);
    auto* g = generation_header(map, 0);
    if (!g || helper >= h->helper_count ||
        __atomic_load_n(&g->ready_helpers, __ATOMIC_ACQUIRE)) return false;
    const uint32_t bit = 1u << helper;
    const uint32_t old = __atomic_fetch_or(&h->bootstrap_filtered_helpers, bit, __ATOMIC_ACQ_REL);
    if (old & bit) return false;
    notify(h);
    return true;
}

/* Serving and replay share runtime filtering, device table loading, and
 * map-consumption publication. */
template<class Selected, class Loader>
bool bootstrap_runtime_layer(void* map, const Selected& owned, uint32_t helper,
                             uint32_t layer, Loader&& loader, const char** why = nullptr) {
    auto fail = [&](const char* reason) { if (why) *why = reason; return false; };
    if (!map) return fail("missing bootstrap map");
    auto* h = static_cast<Header*>(map);
    if (helper >= h->helper_count || layer >= h->layer_count)
        return fail("bootstrap helper or layer out of range");
    if (!load(&h->layer_map_ready[layer])) {
        for (auto [l, e] : owned.expert)
            if (l == layer && !drop_bootstrap_owner(map, l, e, helper))
                return fail("could not drop undeclared layer owner");
        return true;
    }
    const int32_t* slots = layer_slots(map, layer);
    if (!slots) return fail("missing runtime layer slots");
    for (auto [l, e] : owned.expert)
        if (l == layer && (e >= h->expert_count ||
                          (slots[e] < 0 && !drop_bootstrap_owner(map, l, e, helper))))
            return fail("could not filter protected runtime owner");
    if (!loader(layer, slots)) return fail("device runtime layer/table load failed");
    store(&h->layer_map_consumed[layer], 1);
    notify(h);
    return true;
}

/* Compare and replay use the same single-helper production bootstrap.
 * Runner::allocate has already prepared the expert pool before this call. */
template<class Selected, class Loader>
bool prepare_compare_bootstrap(void* map, uint32_t layers,
                               const LayoutShape& shape, const Selected& owned,
                               uint32_t layer, const int32_t* runtime_slots,
                               uint32_t class_count, const uint32_t* class_rows,
                               Loader&& loader, PlacementControl* active,
                               const char** why = nullptr) {
    auto fail = [&](const char* reason) { if (why) *why = reason; return false; };
    if (!map || !active || !runtime_slots || !class_rows ||
        !class_count || class_count > kSmallClasses || layer >= layers)
        return fail("invalid compare bootstrap arguments");
    if (!init(map, layers, shape, 1)) return fail("compare segment init failed");
    auto* h = static_cast<Header*>(map);
    h->small_class_count = class_count;
    for (uint32_t c = 0; c < class_count; ++c) h->small_class_rows[c] = class_rows[c];
    /* init leaves helper_count zero; serving publishes it before ownership. */
    store(&h->helper_count, 1);
    if (!publish_owned(map, owned, 0)) return fail("compare owner publication failed");
    std::memcpy(layer_slots(map, layer), runtime_slots, size_t(shape.experts) * sizeof(int32_t));
    store(&h->layer_map_ready[layer], 1);
    for (uint32_t l = 0; l < layers; ++l) {
        const char* cause = nullptr;
        if (!bootstrap_runtime_layer(map, owned, 0, l, loader, &cause))
            return fail(cause ? cause : "compare runtime layer bootstrap failed");
    }
    if (!mark_bootstrap_filtered(map, 0)) return fail("compare filtered ACK failed");
    SharedTransport transport(map);
    if (!transport.seal_bootstrap()) return fail("compare bootstrap seal refused");
    if (!bootstrap_generation_sealed(map) || !placement_snapshot(map, active) ||
        active->state != uint32_t(PlacementState::ACTIVE) || active->active_slot >= r3::worker_tables::kGenerations)
        return fail("compare active generation is not sealed");
    return true;
}

/* Refuse to issue a replay request against an unsealed or stale generation.
 * The request identity is copied from the controller, never hard-coded. */
inline bool bind_compare_request(void* map, Request* request,
                                 const PlacementControl& expected) {
    PlacementControl now{};
    if (!map || !request || !bootstrap_generation_sealed(map) ||
        !placement_snapshot(map, &now) ||
        now.state != uint32_t(PlacementState::ACTIVE) ||
        now.active_epoch != expected.active_epoch ||
        now.active_slot != expected.active_slot ||
        now.active_digest != expected.active_digest) return false;
    request->generation_slot = now.active_slot;
    request->placement_epoch = now.active_epoch;
    return true;
}


/* The segment reserves two physical slots even for one TP rank. Replay must
 * submit exactly the configured number of engine-rank requests; a one-rank
 * engine computes the full expert width, while two ranks each compute a
 * distinct partial-width contribution. */
inline uint32_t compare_request_ranks(uint32_t engine_ranks, uint32_t slots) {
    return (engine_ranks == 1 || engine_ranks == kMaxEngineRanks) &&
           engine_ranks <= slots ? engine_ranks : 0;
}

/* The captured routes, not the selected candidate count, are the work
 * contract. Before launching, require each captured slot to have an active
 * sealed owner and a fully resident weight for EVERY requested TP slice.
 * Refuse invalid or duplicate routes instead of silently omitting work. */
template<class Routes, class Resident>
bool validate_compare_routes(const void* map, uint32_t layer,
                             uint32_t rows, const Routes& routes,
                             uint32_t ranks, Resident&& resident,
                             const char** why = nullptr) {
    auto fail = [&](const char* reason) { if (why) *why = reason; return false; };
    if (!map || !bootstrap_generation_sealed(map))
        return fail("replay generation not sealed");
    const auto* h = static_cast<const Header*>(map);
    if (layer >= h->layer_count || !rows || rows > h->max_rows ||
        !compare_request_ranks(ranks, h->slot_count) ||
        routes.empty() || routes.size() > size_t(rows) * h->top_k ||
        !load(&h->layer_map_ready[layer]) ||
        !load(&h->layer_map_consumed[layer]))
        return fail("replay shape or runtime layer not ready");
    PlacementControl active{};
    if (!placement_snapshot(map, &active) ||
        active.state != uint32_t(PlacementState::ACTIVE) ||
        active.active_slot >= r3::worker_tables::kGenerations)
        return fail("replay placement control not active");
    const uint8_t* owners = generation_map(map, active.active_slot);
    const int32_t* slots = layer_slots(map, layer);
    if (!owners || !slots) return fail("missing replay ownership or slot map");
    std::vector<uint8_t> seen(size_t(rows) * h->top_k);
    for (const Route& r : routes) {
        if (r.token < 0 || uint32_t(r.token) >= rows || r.expert < 0 ||
            uint32_t(r.expert) >= h->expert_count || r.slot >= h->top_k)
            return fail("captured route outside model or request");
        const size_t flat = size_t(r.token) * h->top_k + r.slot;
        if (seen[flat]++) return fail("duplicate captured top-k slot");
        const uint32_t expert = uint32_t(r.expert);
        if (owners[size_t(layer) * h->expert_count + expert] != 0 ||
            slots[expert] < 0 || uint32_t(slots[expert]) >= h->expert_count)
            return fail("captured route has no resident runtime owner");
        for (uint32_t rank = 0; rank < ranks; ++rank)
            if (!resident(layer, expert, rank, uint32_t(slots[expert])))
                return fail("captured route weights not resident for requested TP slice");
    }
    return true;
}

} // namespace r3::epoch
