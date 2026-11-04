#pragma once

// Expert description, model shape, backend acceptance and the shared-memory segment shape. Included only by r3_moe_worker.cpp, at the position the code had in the single file (RCS03, pure move).

namespace {


uint64_t helper_budget_bytes() {
    const char* e = std::getenv("R3_MOE_WORKER_MIB");
    if (!e || !*e) return 28672ull * 1024ull * 1024ull;
    char* end = nullptr;
    errno = 0;
    const unsigned long long mib = std::strtoull(e, &end, 10);
    if (errno || !end || *end || mib == 0 ||
        mib > std::numeric_limits<uint64_t>::max() / (1024ull * 1024ull)) return 0;
    return (uint64_t)mib * 1024ull * 1024ull;
}

uint32_t env_rows(const char* e, uint32_t fallback) {
    if (!e || !*e) return fallback;
    char* end = nullptr;
    errno = 0;
    const unsigned long v = std::strtoul(e, &end, 10);
    if (errno || !end || *end || v == 0) return fallback;
    return (uint32_t)std::min<unsigned long>(v, kMaxRows);
}
/* The largest step handed off (the engine reads the same variable) and the small, pre-queued lane size. */
uint32_t helper_max_rows() { return env_rows(std::getenv("R3_MOE_WORKER_MAX_ROWS"), 64); }
uint32_t helper_small_rows() { return std::min(env_rows(std::getenv("R3_MOE_WORKER_SMALL_ROWS"), 64), helper_max_rows()); }

/* The pre-queued small-lane sizes (RAD12): R3_MOE_WORKER_SMALL_LANES="4,8,16,24" (up to kSmallClasses, strictly ascending,
 * each within the large-lane capacity), else the single size R3_MOE_WORKER_SMALL_ROWS gives (the production behaviour). */
struct SmallClasses {
    std::array<uint32_t, r3::moe_shm::kSmallClasses> rows{};
    uint32_t count = 0;
};
SmallClasses helper_small_classes() {
    SmallClasses sc;
    const char* list = std::getenv("R3_MOE_WORKER_SMALL_LANES");
    if (!list || !*list) {
        sc.rows[0] = helper_small_rows();
        sc.count = 1;
        return sc;
    }
    const uint32_t cap = helper_max_rows();
    const char* p = list;
    while (*p && sc.count < r3::moe_shm::kSmallClasses) {
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p || v <= 0 || (uint32_t)v > cap || (sc.count && (uint32_t)v <= sc.rows[sc.count - 1])) {
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_SMALL_LANES=%s must be up to %u ascending sizes within %u rows\n",
                         list, (unsigned)r3::moe_shm::kSmallClasses, cap);
            return SmallClasses{};
        }
        sc.rows[sc.count++] = (uint32_t)v;
        p = end;
        if (*p == ',') ++p;
        else break;
    }
    if (*p) {
        std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_SMALL_LANES=%s has more than %u sizes or trailing text\n", list,
                     (unsigned)r3::moe_shm::kSmallClasses);
        return SmallClasses{};
    }
    return sc;
}

/* The lanes this helper serves: with several helpers each serves only its own rank slot's lanes, and a disabled small
 * class (beyond `classes`) has none. Only these get a stream, a hardware queue, buffers and a worker thread. */
std::array<bool, r3::moe_shm::kLanes> served_lanes(uint32_t helper_count, uint32_t helper_index, uint32_t classes) {
    using namespace r3::moe_shm;
    std::array<bool, kLanes> s{};
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        s[lane] = (helper_count <= 1 || lane % kMaxSlots == helper_index) &&
                  !(lane_is_small(lane) && lane_class(lane) >= classes);
    return s;
}

/* Expert description, sizing and backend acceptance are bc::experts' (components/serving/experts.h), the code bc-serve
 * plans ownership with. The card's weight formats come from R3_MOE_WORKER_FORMATS, as before. */
using bc::experts::canonical_expert_bytes;
using bc::experts::container_shape;
using bc::experts::describe_expert;
bool backend_accepts(const Container& f, const char* arch, bool verbose) {
    return bc::experts::backend_accepts(f, arch, std::getenv("R3_MOE_WORKER_FORMATS"), verbose);
}
int64_t laid_out_bytes(const Container& f, r3_moe_gpu::Runner* gpu, uint32_t l, uint32_t e) {
    return bc::experts::laid_out_bytes(f, gpu != nullptr, l, e);
}

/* The shared-memory shape and layout for this container (RCS09-03): hidden, experts and expert width are the container's own; the
 * row capacity, top-k and activation group are the protocol's until the engine negotiates them. A shape other than the one the
 * execution paths run is refused by the backend probe before this (moe_backend.h), not here. */
bool segment_shape(const Container& f, uint32_t helpers, r3::moe_shm::LayoutShape* shape, r3::moe_shm::Layout* lay) {
    MoeShape cs{};
    if (!container_shape(f, &cs)) {
        std::fprintf(stderr, "r3_moe_worker: the container's shape could not be read\n");
        return false;
    }
    *shape = r3::moe_shm::LayoutShape{cs.hidden, cs.top_k ? cs.top_k : r3::moe_shm::kTopK, cs.experts, r3::moe_shm::kMaxRows,
                                      r3::moe_shm::kActGroup, cs.intermediate};
    if (!r3::moe_shm::layout_for(*shape, f.layers, r3::moe_shm::slots_for(helpers), lay)) {
        std::fprintf(stderr, "r3_moe_worker: the shared-memory layout for this shape is invalid\n");
        return false;
    }
    return true;
}
} // namespace
