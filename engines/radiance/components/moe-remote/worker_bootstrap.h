#pragma once
#include "moe_remote_shm.h"

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
} // namespace r3::epoch
