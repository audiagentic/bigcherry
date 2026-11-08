#pragma once

// Expert description, model shape, backend acceptance and the shared-memory segment shape. Included only by r3_moe_worker.cpp, at the position the code had in the single file (RCS03, pure move).

namespace {


uint64_t helper_budget_bytes() {
    return r3_serving::settings().budget_mib * 1024ull * 1024ull;
}
uint32_t helper_max_rows() { return r3_serving::settings().handoff.max_rows; }
uint32_t helper_small_rows() {
    return std::min(r3_serving::settings().handoff.small_rows, helper_max_rows());
}

/* The pre-queued small-lane sizes (RAD12): R3_MOE_WORKER_SMALL_LANES="4,8,16,24" (up to kSmallClasses, strictly ascending,
 * each within the large-lane capacity), else the single size R3_MOE_WORKER_SMALL_ROWS gives (the production behaviour). */
struct SmallClasses {
    std::array<uint32_t, r3::moe_shm::kSmallClasses> rows{};
    uint32_t count = 0;
};
SmallClasses helper_small_classes() {
    SmallClasses sc;
    const auto& lanes = r3_serving::settings().small_lanes;
    if (lanes.empty()) {
        sc.rows[0] = helper_small_rows();
        sc.count = 1;
    } else {
        for (uint32_t v : lanes) sc.rows[sc.count++] = v;
    }
    return sc;
}

/* The lanes this helper serves: its own slots, slot_of(helper_index, r, world) for every engine rank r (RAD49), and a
 * disabled small class (beyond `classes`) has none. Only these get a stream, a hardware queue, buffers and a worker thread. */
std::array<bool, r3::moe_shm::kLanes> served_lanes(uint32_t helper_count, uint32_t helper_index, uint32_t world,
                                                   uint32_t classes) {
    return r3::worker_footprint::served_lanes(helper_count, helper_index, world, classes);   /* bc-serve's footprint uses it */
}

/* Expert description, sizing and backend acceptance are bc::experts' (components/serving/experts.h), the code bc-serve
 * plans ownership with. The card's weight formats come from the manifest (helper formats). */
using bc::experts::canonical_expert_bytes;
using bc::experts::container_shape;
using bc::experts::describe_expert;
bool backend_accepts(const Container& f, const char* arch, bool verbose) {
    return bc::experts::backend_accepts(f, arch, r3_serving::settings().formats.c_str(), verbose);
}
int64_t laid_out_bytes(const Container& f, r3_moe_gpu::Runner* gpu, uint32_t l, uint32_t e) {
    return bc::experts::laid_out_bytes(f, gpu != nullptr, l, e);
}

/* The shared-memory shape and layout for this container (RCS09-03): hidden, experts and expert width are the container's own; the
 * row capacity, top-k and activation group are the protocol's until the engine negotiates them. A shape other than the one the
 * execution paths run is refused by the backend probe before this (moe_backend.h), not here. */
bool segment_shape(const Container& f, uint32_t helpers, uint32_t world, r3::moe_shm::LayoutShape* shape,
                   r3::moe_shm::Layout* lay) {
    MoeShape cs{};
    if (!container_shape(f, &cs)) {
        std::fprintf(stderr, "r3_moe_worker: the container's shape could not be read\n");
        return false;
    }
    *shape = r3::moe_shm::LayoutShape{cs.hidden, cs.top_k ? cs.top_k : r3::moe_shm::kTopK, cs.experts, r3::moe_shm::kMaxRows,
                                      r3::moe_shm::kActGroup, cs.intermediate};
    if (!r3::moe_shm::slots_fit(helpers, world) ||
        !r3::moe_shm::layout_for(*shape, f.layers, r3::moe_shm::slots_for(helpers, world), lay)) {
        std::fprintf(stderr, "r3_moe_worker: the shared-memory layout for this shape is invalid\n");
        return false;
    }
    return true;
}
} // namespace
