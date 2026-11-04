#pragma once
// Fixed-stride expert weights use the configured EXPERT budget only. The spare
// incoming GPU slot uses the card's free VRAM, not that budget. Occupancy is
// governed by the residency ledger; no hot-path tracking is added here.
#include "worker_residency.h"
#include <algorithm>
#include <cstdint>
#include <limits>

namespace r3::worker_slots {

inline uint32_t slot_capacity(uint64_t budget, uint64_t stride, uint64_t model_experts) {
    if (!stride) return 0;
    return uint32_t(std::min<uint64_t>({model_experts, budget / stride, UINT32_MAX - 1ull}));
}

struct BudgetPlan {
    uint64_t stride = 0, expert_bytes = 0;
    uint32_t expert_slots = 0, physical_slots = 0;
};
inline BudgetPlan budget_plan(uint64_t budget, uint64_t stride, uint64_t model_experts) {
    const uint32_t n = slot_capacity(budget, stride, model_experts);
    return {stride, uint64_t(n) * stride, n, uint32_t(n + 1u)};
}

// The actual VRAM check is made with hipMemGetInfo at startup. 'already' is
// memory allocated *after* the initial free-memory snapshot (prepared lanes,
// including real plugin scratch). 'future' is not allocated at the second
// snapshot (selected arena, both pointer table banks, frozen slot map, spare).
struct DeviceMemory {
    uint64_t lanes = 0, tables = 0, slot_map = 0, staging = 0, arena = 0;
};
struct MemoryCheck { uint64_t total = 0, future = 0, shortfall = 0; bool fits = false; };
inline MemoryCheck check_device_memory(const DeviceMemory& d, uint64_t initial_free, uint64_t current_free) {
    auto add = [](uint64_t a, uint64_t b, uint64_t* out) {
        if (b > UINT64_MAX - a) return false;
        *out = a + b;
        return true;
    };
    uint64_t future = 0, total = 0;
    if (!add(d.tables, d.slot_map, &future) ||
        !add(future, d.staging, &future) ||
        !add(future, d.arena, &future) ||
        !add(future, d.lanes, &total))
        return {UINT64_MAX, UINT64_MAX, UINT64_MAX, false};
    const uint64_t shortfall = std::max(total > initial_free ? total - initial_free : 0,
                                        future > current_free ? future - current_free : 0);
    return {total, future, shortfall, shortfall == 0};
}

template<class Allocator>
class Pool {
    residency::Ledger ledger_;
    Allocator& allocator_;
    uint32_t experts_ = 0;
    uint64_t stride_ = 0;
    bool valid_ = false;
    uint8_t* arena_ = nullptr;
    uint8_t* incoming_ = nullptr;
public:
    Pool(uint64_t expert_budget, uint64_t stride, uint32_t capacity, Allocator& allocator)
        : ledger_(stride && expert_budget <= UINT64_MAX - stride ? expert_budget + stride : 0,
                  0, stride, capacity < UINT32_MAX ? capacity + 1 : 0),
          allocator_(allocator), experts_(capacity), stride_(stride),
          valid_(stride && capacity < UINT32_MAX &&
                 capacity <= expert_budget / stride &&
                 expert_budget <= UINT64_MAX - stride) {}
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    ~Pool() {
        if (arena_) allocator_.release(arena_);
        if (incoming_) allocator_.release(incoming_);
    }

    bool valid() const { return valid_ && ledger_.valid(); }
    residency::Ledger& ledger() { return ledger_; }
    const residency::Ledger& ledger() const { return ledger_; }
    uint32_t capacity() const { return experts_; }
    uint32_t staging_id() const { return experts_; }
    uint64_t stride() const { return stride_; }
    uint64_t arena_bytes() const { return uint64_t(experts_) * stride_; }
    void* address(uint32_t id) const {
        if (id < experts_) return arena_ ? arena_ + uint64_t(id) * stride_ : nullptr;
        return id == experts_ ? incoming_ : nullptr;
    }

    // Allocate the budget-sized arena and incoming slot at startup, including
    // helpers without bootstrap residents. No GPU weight allocation is needed
    // for a subsequent arrival.
    bool prepare() {
        if (!valid()) return false;
        if (arena_ && incoming_) return true;
        void *arena = nullptr, *incoming = nullptr;
        if (!arena_bytes() || arena_bytes() > SIZE_MAX ||
            !allocator_.allocate(arena_bytes(), &arena) || !arena)
            return false;
        if (stride_ > SIZE_MAX || !allocator_.allocate(stride_, &incoming) || !incoming) {
            allocator_.release(arena);
            return false;
        }
        arena_ = static_cast<uint8_t*>(arena);
        incoming_ = static_cast<uint8_t*>(incoming);
        return true;
    }

    residency::Result bootstrap(uint32_t expert, uint64_t bytes, uint64_t epoch,
                                uint32_t* id) {
        if (!id || !bytes || bytes > stride_ || !epoch || !valid())
            return residency::Result::INVALID;
        const auto possible = ledger_.bootstrap_possible(expert, stride_, epoch);
        if (possible != residency::Result::OK) return possible;
        // The staging slot is never consumed by bootstrap even when the
        // expert budget happens to have unused bytes after selection.
        if (ledger_.slots().size() >= experts_) return residency::Result::DEFERRED_CAPACITY;
        if (!prepare()) return residency::Result::DEFERRED_CAPACITY;
        return ledger_.bootstrap(expert, stride_, epoch, id);
    }
};
} // namespace r3::worker_slots
