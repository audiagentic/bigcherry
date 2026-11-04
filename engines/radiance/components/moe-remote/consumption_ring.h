/* The per-parent consumption counting slot, shared by the GPU kernels
 * (step_ticket_kernel.h, moe_mailbox_ops.hip) and the card-free tests.
 *
 * Each rank owns a device-memory ring of ConsumptionStamp, ticket_id %
 * capacity. The ticket setter arms a parent's slot on its first pass on the
 * rank; every publish of that parent adds to published_count and every finish
 * adds to consumed_count (device-scope atomics); the parent's seal copies the
 * slot to the host-mapped ring once, ahead of its terminal event. See
 * ADMISSION.md for why the counts are exact under run-ahead and MTP passes. */
#pragma once
#include "moe_remote_hook.h"

#if defined(__HIP__) || defined(__HIPCC__)
#define R3_CONSUMPTION_HD __host__ __device__
#else
#define R3_CONSUMPTION_HD
#endif

namespace r3 {
namespace consumption {

/* The first pass of a parent resets the slot; later passes of the same parent
 * keep its counts. The previous owner (id - capacity) has retired, so nothing
 * of it is still running or unsealed. */
R3_CONSUMPTION_HD inline void arm(ConsumptionStamp* stamps, uint32_t capacity, const StepTicket& t) {
    ConsumptionStamp* s = stamps + t.ticket_id % capacity;
    if (s->ticket_id != t.ticket_id)
        *s = ConsumptionStamp{t.ticket_id, t.placement_epoch, t.generation_slot, 0u, 0u};
}

/* A publish or finish may count only into its own parent's armed slot. */
R3_CONSUMPTION_HD inline bool owns(const ConsumptionStamp& s, uint64_t id, uint32_t epoch, uint32_t slot) {
    return id && s.ticket_id == id && s.placement_epoch == epoch && s.generation_slot == slot;
}

/* Seal is the sole per-parent result-consumed ACK, after every add/finish
 * (including the last combined-plane reader) on this rank's ordered stream.
 * The terminal rank event follows the seal. Incomplete counts MUST NOT
 * become an ACK even if a scheduler host-completes the parent early. */
R3_CONSUMPTION_HD inline bool seal_ready(const ConsumptionStamp& s,
                                         const StepTicket& t) {
    return t.generation_slot < 2 &&
           !t.reserved[0] && !t.reserved[1] && !t.reserved[2] &&
           owns(s, t.ticket_id, t.placement_epoch, t.generation_slot) &&
           s.published_count == s.consumed_count;
}


} // namespace consumption
} // namespace r3
