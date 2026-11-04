#pragma once
#include <cstdint>
#include <hip/hip_runtime.h>

// BU01: extracted from the monolithic mailbox device-code unit. A change to
// this small inverse-map kernel does not require rebuilding the unrelated
// publish, wait, route-mark, merge and seal kernels.
namespace r3_mailbox_inverse {
__global__ void mailbox_inverse_kernel(const int32_t* sorted, int32_t* inv, uint32_t slots);
}
