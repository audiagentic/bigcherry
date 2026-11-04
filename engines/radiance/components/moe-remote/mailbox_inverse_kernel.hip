#include "mailbox_inverse_kernel.h"

namespace r3_mailbox_inverse {
// Intentionally the same operation and launch shape as the original in
// moe_mailbox_ops.hip: invert only valid row indices, leave all others alone.
__global__ void mailbox_inverse_kernel(const int32_t* sorted, int32_t* inv, uint32_t slots) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= slots) return;
    const int32_t flat = sorted[i];
    if (flat >= 0 && (uint32_t)flat < slots) inv[flat] = (int32_t)i;
}
}
