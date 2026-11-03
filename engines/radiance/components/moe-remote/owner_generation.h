/* Host-only canonical map -> GPU bitmap conversion. The fake-device tests
 * exercise the same validation and bit packing used by the HIP uploader. */
#pragma once
#include "admission_ledger.h"
#include "moe_remote_shm.h"
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace r3 { namespace owner_generation {
using namespace moe_shm;

/* Engine rank construction can follow plugin registration while helpers are
 * still filtering the runtime layer maps. Wait for helper 0's completed
 * seqlock publication, not for a provisional digest or one helper's bit.
 * This boundary is also used verbatim in card-free startup-order tests. */
inline WaitResult wait_bootstrap_seal(void* map, const timespec* deadline) {
    if (!map || !deadline) return WaitResult::Error;
    Header* h = static_cast<Header*>(map);
    const WaitResult waited = wait_for(h, deadline, [&] {
        const uint32_t seq = load(&placement_control(map)->seq);
        return seq != 0 && !(seq & 1u); // helper 0 publishes even only after the final digest and mask
    });
    if (waited != WaitResult::Ready) return waited;
    return bootstrap_generation_sealed(map) ? WaitResult::Ready : WaitResult::Error;
}
inline bool build_bitmap(const void* map, uint32_t slot, uint64_t epoch,
                         uint64_t digest, std::vector<uint8_t>* out) {
    if (!map || !out || slot > 1 || !epoch || epoch > UINT32_MAX || !digest)
        return false;
    const auto* h = static_cast<const Header*>(map);
    if (!h->layer_count || !h->expert_count || !h->helper_count ||
        h->helper_count > kMaxHelpers || h->slot_count < h->helper_count ||
        h->owner_map_bytes != (uint64_t)h->layer_count * h->expert_count)
        return false;
    PlacementControl control{};
    if (!placement_snapshot(map, &control) || control.fault_code) return false;
    const bool active = control.active_slot == slot &&
                        control.active_epoch == epoch && control.active_digest == digest &&
                        control.state <= uint32_t(PlacementState::QUIESCING);
    const bool pending = control.pending_slot == slot &&
                         control.pending_epoch == epoch && control.pending_digest == digest &&
                         control.state >= uint32_t(PlacementState::COPYING) &&
                         control.state <= uint32_t(PlacementState::PREPARED);
    if (!active && !pending) return false;
    if (active && epoch == 1 && slot == 0 && !bootstrap_generation_sealed(map)) return false;
    const auto* g = generation_header(map, slot);
    if (!g || g->slot != slot || g->epoch != epoch || g->digest != digest ||
        g->map_bytes != h->owner_map_bytes || g->layer_count != h->layer_count ||
        g->expert_count != h->expert_count ||
        (active && g->state != uint32_t(GenerationState::ACTIVE)) ||
        (pending && g->state != uint32_t(GenerationState::STAGING) &&
                    g->state != uint32_t(GenerationState::READY)) ||
        generation_digest(map, slot) != digest) return false;
    const size_t stride = ((h->owner_map_bytes + 7u) / 8u);
    std::vector<uint8_t> result((size_t)h->helper_count * stride, 0);
    const uint8_t* owners = generation_map(map, slot);
    for (uint32_t l = 0; l < h->layer_count; ++l)
        for (uint32_t e = 0; e < h->expert_count; ++e) {
            const uint8_t owner = owners[(size_t)l * h->expert_count + e];
            if (owner == kOwnerLocal) continue;
            if (owner >= h->helper_count) return false;
            const size_t bit = (size_t)l * h->expert_count + e;
            result[(size_t)owner * stride + (bit >> 3)] |= uint8_t(1u << (bit & 7u));
        }
    *out = std::move(result);
    return true;
}
inline bool may_upload(const admission::Ledger& ledger, uint32_t slot) {
    return ledger.can_replace_generation(slot);
}
}} // namespace r3::owner_generation
