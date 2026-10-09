// kdev: device-side helpers shared by candidates. Device code only (include from a .hip).
#ifndef KDEV_DEVICE_H
#define KDEV_DEVICE_H

#include "rad_abi.h"
#include "rad_plugin.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>

// bf16 <-> f32 exactly as the reference defines them (rad_plugin.h): widening is a shift, narrowing rounds to
// nearest even and keeps a NaN a NaN. Restated here because the header's inline functions are host functions.
__device__ __forceinline__ float kdev_bf16_to_f32(uint16_t h) {
    return __uint_as_float(static_cast<uint32_t>(h) << 16);
}

__device__ __forceinline__ uint16_t kdev_f32_to_bf16(float f) {
    const uint32_t v = __float_as_uint(f);
    if ((v & 0x7f800000u) == 0x7f800000u && (v & 0x007fffffu)) {
        return static_cast<uint16_t>((v >> 16) | 0x0040u);  // NaN stays NaN, quiet
    }
    return static_cast<uint16_t>((v + 0x7fffu + ((v >> 16) & 1u)) >> 16);
}

// The engine hands a kernel its own queue as a RadStream; hipLaunchKernelGGL on it is answered by the engine.
static inline hipStream_t kdev_stream(RadStream s) { return reinterpret_cast<hipStream_t>(s); }

static inline int kdev_launch_status() { return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE; }

// A row operand: rows of `n` elements at a pitch the caller narrowed it to. Returns the pitch in elements and the
// row count, or -1 when the operand is not rows of n. Rank 1 is one row.
static inline long long kdev_rows_pitch(const RadTensor* t, long long n, long long* rows) {
    if (!t || t->rank == 0 || n <= 0) return -1;
    const int last = static_cast<int>(t->rank) - 1;
    if (t->shape[last] != n || (n != 1 && t->stride[last] != 1)) return -1;
    if (t->rank == 1) { *rows = 1; return n; }
    long long count = 1;
    for (int i = 0; i < last; ++i) count *= t->shape[i];
    // leading axes must be dense over the row axis: only the innermost of them may carry a pitch wider than n
    const long long pitch = t->stride[last - 1];
    long long acc = pitch;
    for (int i = last - 1; i >= 0; --i) {
        if (t->shape[i] != 1 && t->stride[i] != acc) return -1;
        acc *= t->shape[i];
    }
    if (pitch < n) return -1;
    *rows = count;
    return pitch;
}

#endif
