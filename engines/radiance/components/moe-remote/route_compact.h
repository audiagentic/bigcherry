/* RAD29 (shelved, evaluated 2026-10-28): wave-aggregated route compaction, one shared-memory reservation a wave rather
 * than one a route. Correct on gfx1100/gfx1201 but 0-7% slower than the per-route atomic loop the mailbox publish kernel
 * keeps (the loop is bound by reading ids/ownership, not by LDS atomics); kept with route_compact_test, which times it
 * against that loop, so the idea can be revisited (see RAD29).
 *
 * Every thread of the block calls this with the same arguments; the loop bound is uniform, so the ballot and the
 * shuffle always see the whole wave and an out-of-range lane just votes no. A wave's accepted routes take one contiguous
 * span, in lane order; spans of different waves interleave in reservation order, as single routes did before. Consumers
 * rebuild their order from (token, slot), so no route order is promised. `take(i)` returns the local expert of candidate
 * `i` when it is routed here, else -1 (it may have side effects such as the heat histogram: it runs once a candidate).
 * The caller zeroes `*nr` and synchronises the block before, and synchronises after. */
#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

#include "moe_remote_shm.h"

template <class Take>
__device__ inline void compact_routes(uint32_t n_cand, uint32_t top_k, uint32_t* nr, r3::moe_shm::Route* route, const uint16_t* ew,
                                      int32_t* ids, bool clear, Take take) {
    for (uint32_t base_i = 0; base_i < n_cand; base_i += blockDim.x) {
        const uint32_t i = base_i + threadIdx.x;
        const int32_t le = i < n_cand ? take(i) : -1;
        const uint64_t mask = __ballot(le >= 0);
        const uint32_t lane = __lane_id();
        uint32_t span = 0;
        if (lane == 0 && mask) span = atomicAdd(nr, (uint32_t)__popcll(mask));
        span = __shfl(span, 0);
        if (le >= 0) {
            const uint32_t p = span + (uint32_t)__popcll(mask & ((1ull << lane) - 1ull));
            route[p].token = (int32_t)(i / top_k);
            route[p].expert = le;
            route[p].weight_bf16 = ew[i];
            route[p].slot = (uint16_t)(i % top_k);
            if (clear) ids[i] = -1;
        }
    }
}
