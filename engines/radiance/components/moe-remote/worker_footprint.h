#pragma once
// The helper's own device memory besides its expert arena (RAD49): the prepared lanes, the two pointer-table banks, the frozen
// runtime slot map and the one incoming expert slot. One card-free definition for the worker (moe_remote_gpu.h sizes and checks
// its allocations against it) and for bc-serve (serving.cpp derives a layout's omitted budget-mib from it), so the expert budget
// the owner planner fills is the one the helper can hold: budget = card VRAM - archs.toml reserve-mib - fixed_bytes(...).
//
// Not in it: the kernel plugins' per-lane scratch (the plugin sizes it at run time) and the HIP context and code objects; the
// arch's reserve-mib covers those, and the worker's start-up check (worker_slot_pool.h check_device_memory) still compares the
// whole of it with the card's real free VRAM.
#include "moe_remote_shm.h"
#include "worker_generation_tables.h"

#include <array>
#include <cstdint>

namespace r3::worker_footprint {

using r3::moe_shm::kLanes;
using r3::moe_shm::kMaxSlots;
using r3::moe_shm::kSmallClasses;

/* The model shape the buffers are sized from. */
struct Shape {
    uint64_t layers = 0, experts = 0, hidden = 0, intermediate = 0, top_k = 0, act_group = 0;
};
/* The lanes a helper prepares: the small classes' fixed row counts, the large lane's capacity, and which lanes are open. */
struct Lanes {
    std::array<uint32_t, kSmallClasses> small_rows{};
    uint32_t small_count = 0, large_rows = 0;
    std::array<bool, kLanes> open{};
};

/* Lanes this helper serves: its own slots, slot_of(helper_index, r, world) for every engine rank r; a small class beyond `classes`
 * has none. With power save off the worker opens every protocol lane (r3_moe_worker.cpp lanes_open). */
inline std::array<bool, kLanes> served_lanes(uint32_t helper_count, uint32_t helper_index, uint32_t world, uint32_t classes,
                                             bool power_save = true) {
    using namespace r3::moe_shm;
    std::array<bool, kLanes> s{};
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        const uint32_t slot = lane % kMaxSlots;
        const bool mine = !power_save ||
                          (world && slot < slots_for(helper_count, world) && slot_helper(slot, world) == helper_index);
        s[lane] = mine && !(lane_is_small(lane) && lane_class(lane) >= classes);
    }
    return s;
}

/* One lane's work buffers (moe_remote_gpu.h ensure_lane_work, its stats, publish limit, the large lane's chunk counters and the
 * eight per-parity expert pointer tables) for `rows` rows and `routes` dense route slots. */
inline uint64_t lane_bytes(const Shape& s, uint64_t rows, uint64_t routes, bool large) {
    const uint64_t tr = routes, I = s.intermediate, H = s.hidden, E = s.experts;
    uint64_t n = 0;
    n += E * 4 + rows + E * 4;                    /* counts, row mask, cursor */
    n += rows * H * 2;                            /* bf16 activations */
    n += routes * 4 + tr * 4 + (E + 1) * 4;       /* flat, sorted, offsets */
    n += tr * 2 + tr * 4;                         /* route weights, inverse */
    n += rows * H + rows * (H / s.act_group) * 4; /* exact E4M3 activations and their scales */
    n += tr * 2 * I * 2;                          /* gate/up */
    n += tr * I + tr * (I / s.act_group) * 4;     /* the quantised middle and its scales */
    n += tr * H * 2;                              /* down rows */
    n += tr * 2 * I * 2;                          /* the GEMV debug comparison buffer (allocated with the lane) */
    n += 3 * sizeof(uint64_t) + sizeof(uint32_t); /* stats, publish limit */
    if (large) n += r3::moe_shm::kCombineChunks * sizeof(uint32_t);
    n += 8 * (E / 2) * sizeof(void*);             /* gate/up and down code/scale tables, two parities */
    return n;
}

/* Every open lane's work buffers (the plugin scratch is not included; see the header comment). */
inline uint64_t lanes_bytes(const Shape& s, const Lanes& l) {
    using namespace r3::moe_shm;
    uint64_t n = 0;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        if (!l.open[lane]) continue;
        const uint32_t cls = lane_class(lane);
        if (cls < kSmallClasses && cls >= l.small_count) continue;
        const uint64_t cap = cls < kSmallClasses ? l.small_rows[cls] : l.large_rows;
        n += lane_bytes(s, cap, cap * s.top_k, cls == kSmallClasses);
    }
    return n;
}

/* The two generation banks of per-layer pointer tables (DeviceLayerTables: four parity pairs and the slot-map pointer) and their
 * per-parity expert tables. */
constexpr uint64_t kDeviceLayerTablesBytes = 9 * sizeof(void*);
inline uint64_t pointer_bank_bytes(const Shape& s) {
    return uint64_t(r3::worker_tables::kGenerations) *
           (s.layers * kDeviceLayerTablesBytes + s.layers * 2u * 4u * (s.experts / 2u) * sizeof(void*));
}
inline uint64_t slot_map_bytes(const Shape& s) { return s.layers * s.experts * sizeof(uint16_t); }

/* An expert slot's stride bound from its device-layout bytes: the worker rounds each of its four planes (gate/up and down, code
 * and scale) to 256 bytes (moe_remote_gpu.h arena_round), so the stride is at most the bytes plus 4 x 255. */
constexpr uint64_t kPlaneRound = 256;
inline uint64_t stride_bound(uint64_t expert_bytes) { return expert_bytes + 4 * (kPlaneRound - 1); }

/* The helper's device memory besides its expert arena: lanes, pointer banks, slot map and one incoming slot of `stride`. */
inline uint64_t fixed_bytes(const Shape& s, const Lanes& l, uint64_t stride) {
    return lanes_bytes(s, l) + pointer_bank_bytes(s) + slot_map_bytes(s) + stride;
}

/* The expert budget a card can hold: what the arch reserve leaves of its VRAM, less the helper's own fixed memory; 0 when that
 * leaves nothing. */
inline uint64_t expert_budget_bytes(uint64_t card_bytes, uint64_t reserve_bytes, uint64_t fixed) {
    if (card_bytes <= reserve_bytes) return 0;
    const uint64_t avail = card_bytes - reserve_bytes;
    return avail > fixed ? avail - fixed : 0;
}

}  // namespace r3::worker_footprint
