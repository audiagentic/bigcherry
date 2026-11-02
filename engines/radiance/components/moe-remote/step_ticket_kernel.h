#pragma once
#include "moe_remote_hook.h"
#include <hip/hip_runtime.h>

/* Exactly one ordinary kernel launch per pass. Neither chaining nor overlap
 * attributes are attached: AQL's normal barrier-bit packet orders it after
 * the preceding pass's lane-1 join and before the replay's foreign packet. */
static __global__ void mailbox_set_ticket_kernel(StepTicket* dst, StepTicket value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *dst = value;
}
