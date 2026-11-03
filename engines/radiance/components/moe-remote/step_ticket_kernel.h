#pragma once
#include "consumption_ring.h"
#include <hip/hip_runtime.h>

/* Exactly one ordinary kernel launch per pass. Neither chaining nor overlap
 * attributes are attached: AQL's normal barrier-bit packet orders it after
 * the preceding pass's lane-1 join and before the replay's foreign packet.
 *
 * It also arms the parent's device-memory counting slot (ticket_id % capacity)
 * on the parent's first pass on this rank: later passes of the same parent
 * (history, MTP/draft, issue-ahead) find their own id and keep the counts.
 * The slot's previous owner (id - capacity) has retired, so its kernels and
 * its seal are complete before this stream-ordered reset. */
static __global__ void mailbox_set_ticket_kernel(StepTicket* dst, ConsumptionStamp* stamps, uint32_t capacity,
                                                 StepTicket value) {
    if (blockIdx.x || threadIdx.x) return;
    *dst = value;
    r3::consumption::arm(stamps, capacity, value);
}
