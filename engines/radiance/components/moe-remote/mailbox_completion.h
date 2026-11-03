/* Engine GPU route-mark identity: each pass has a device-local ticket copy.
 * The route-mark validates engine-owned Request identity before either add
 * or shadow reads helper output; wait only acquires the helper's exact
 * done_sequence and checks its status, as on main. The helper's independent
 * last epoch/slot identity is corroborated at drain ACK. The predicate is
 * shared with card-free tests to detect removal of the route-mark guard. */
#pragma once
#include "moe_remote_hook.h"
#include "moe_remote_shm.h"
#include <cstdint>
#if defined(__HIPCC__)
#define R3_MAILBOX_HD __host__ __device__
#else
#define R3_MAILBOX_HD
#endif

namespace r3 { namespace mailbox_completion {
using namespace moe_shm;
constexpr uint32_t kPublishedTicketOffset = kMaxSlots + 1u + kCombineChunks;

/* The pass copy, not the rank's mutable ticket cell. Reject a 32-bit slot
 * before narrowing to the wire ticket's uint8_t generation_slot. */
R3_MAILBOX_HD inline bool pass_ticket(const uint32_t* published, StepTicket* out) {
    if (!published || !out) return false;
    const uint32_t slot = published[kPublishedTicketOffset + 3];
    const uint64_t id = (uint64_t)published[kPublishedTicketOffset] |
                        ((uint64_t)published[kPublishedTicketOffset + 1] << 32);
    const uint32_t epoch = published[kPublishedTicketOffset + 2];
    if (!id || !epoch || slot > 1) return false;
    StepTicket t{};
    t.ticket_id = id;
    t.placement_epoch = epoch;
    t.generation_slot = (uint8_t)slot;
    *out = t;
    return true;
}

/* These are engine-owned Request words, immutable from release of
 * request_sequence until the existing ledger's final consumption event. */
R3_MAILBOX_HD inline bool matches_request(const StepTicket& t, uint32_t sequence,
                                           const Request& q) {
    return t.ticket_id && t.placement_epoch && t.generation_slot < 2 &&
           !t.reserved[0] && !t.reserved[1] && !t.reserved[2] &&
           sequence && q.sequence == sequence &&
           q.placement_epoch == t.placement_epoch &&
           q.generation_slot == t.generation_slot;
}

}} // namespace r3::mailbox_completion
#undef R3_MAILBOX_HD
