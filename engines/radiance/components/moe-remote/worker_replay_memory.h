#pragma once

// Card-free sizing decisions for --compare and --replay-sweep. Serving uses
// the manifest's full lane/weight budget; captures use only their live work.
#include "moe_remote_shm.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace r3::replay {

inline bool expert_budget(uint64_t stride, size_t selected, uint64_t manifest_budget, uint64_t* out) {
    if (!out || !stride || !selected || selected > UINT32_MAX ||
        selected > std::numeric_limits<uint64_t>::max() / stride)
        return false;
    const uint64_t bytes = uint64_t(selected) * stride;
    if (bytes > manifest_budget) return false;
    *out = bytes;
    return true;
}

struct LanePlan {
    uint32_t large_rows = 0;
    uint32_t dense_routes = 0;
    std::array<bool, r3::moe_shm::kLanes> open{};
};

// Only the rank(s) and row class exercised by this request need work buffers.
// The lane kernels use dense rows * top_k indexing, even when the capture
// publishes fewer routes. Preserve that full per-request capacity; it is not
// safe to size device buffers to the sparse published-route count. The
// expert-indexed tables still span the model's logical expert IDs.
inline bool lane_plan(const std::array<uint32_t, r3::moe_shm::kSmallClasses>& classes,
                      uint32_t class_count, uint32_t rows, uint32_t published_routes,
                      uint32_t top_k, uint32_t ranks, LanePlan* out) {
    using namespace r3::moe_shm;
    if (!out || !class_count || class_count > kSmallClasses ||
        !rows || rows > kMaxRows || !top_k || top_k > kMaxTopK ||
        published_routes > uint64_t(rows) * top_k || !ranks || ranks > kMaxEngineRanks)
        return false;
    for (uint32_t c = 0; c < class_count; ++c)
        if (!classes[c] || classes[c] > kMaxRows || (c && classes[c] <= classes[c - 1]))
            return false;
    LanePlan plan{};
    plan.large_rows = std::max(rows, classes[class_count - 1]);
    uint32_t cls = kSmallClasses;
    for (uint32_t c = 0; c < class_count; ++c)
        if (rows <= classes[c]) { cls = c; break; }
    plan.dense_routes = (cls == kSmallClasses ? rows : classes[cls]) * top_k;
    for (uint32_t rank = 0; rank < ranks; ++rank)
        plan.open[lane_of(rank, cls)] = true;
    *out = plan;
    return true;
}

} // namespace r3::replay
