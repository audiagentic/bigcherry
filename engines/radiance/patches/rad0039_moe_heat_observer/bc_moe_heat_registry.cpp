/* Engine-owned read-only callback registry. Publication and invocation are
 * lock-free on the inference path; registration/unregistration is control-plane.
 */
#include "rad_moe_heat.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>

namespace {
std::mutex registration_mu;
std::atomic<RadMoeHeatObserverV1> observer{nullptr};
std::atomic<void*> observer_user{nullptr};
std::atomic<uint32_t> in_flight{0};
std::atomic<uint64_t> failures{0};
std::atomic<uint64_t> run_sequence{0};
}

extern "C" int rad_moe_heat_register_v1(uint32_t abi_version, uint32_t row_size,
                                         RadMoeHeatObserverV1 callback, void* user) {
    if (abi_version != RAD_MOE_HEAT_ABI_V1 || row_size != sizeof(RadMoeHeatRowV1))
        return RAD_MOE_HEAT_REGISTER_BAD_ABI;
    if (!callback && user) return RAD_MOE_HEAT_REGISTER_BAD_ARGUMENT;
    std::lock_guard<std::mutex> lock(registration_mu);
    if (callback) {
        if (observer.load(std::memory_order_seq_cst))
            return RAD_MOE_HEAT_REGISTER_OCCUPIED;
        observer_user.store(user, std::memory_order_relaxed);
        observer.store(callback, std::memory_order_seq_cst);
    } else {
        observer.store(nullptr, std::memory_order_seq_cst);
        /* A reader that raced with the store re-checks observer after joining
         * in_flight. No callback may still hold user when this returns. */
        while (in_flight.load(std::memory_order_seq_cst))
            std::this_thread::yield();
        observer_user.store(nullptr, std::memory_order_relaxed);
    }
    return RAD_MOE_HEAT_REGISTER_OK;
}

extern "C" uint64_t rad_moe_heat_callback_errors_v1(void) {
    return failures.load(std::memory_order_relaxed);
}

bool bc_moe_heat_enabled() noexcept {
    return observer.load(std::memory_order_seq_cst) != nullptr;
}

uint64_t bc_moe_heat_new_run() noexcept {
    uint64_t previous = run_sequence.load(std::memory_order_relaxed);
    while (previous != UINT64_MAX) {
        if (run_sequence.compare_exchange_weak(previous, previous + 1,
                                               std::memory_order_relaxed))
            return previous + 1;
    }
    return 0;
}

void bc_moe_heat_emit(const RadMoeHeatRowV1* row) noexcept {
    if (!observer.load(std::memory_order_seq_cst)) return;
    in_flight.fetch_add(1, std::memory_order_seq_cst);
    const auto fn = observer.load(std::memory_order_seq_cst);
    if (fn) {
        const bool valid = row && row->abi_version == RAD_MOE_HEAT_ABI_V1 &&
            row->struct_size == sizeof(RadMoeHeatRowV1) &&
            row->counts && row->run_generation && row->bank_generation &&
            row->n_expert && row->pass_reports && row->reserved == 0 &&
            row->phase >= RAD_MOE_HEAT_PREFILL && row->phase <= RAD_MOE_HEAT_UNKNOWN &&
            row->expert_base <= UINT32_MAX - row->n_expert;
        if (!valid) {
            failures.fetch_add(1, std::memory_order_relaxed);
        } else {
            try {
                if (fn(observer_user.load(std::memory_order_relaxed), row) != 0)
                    failures.fetch_add(1, std::memory_order_relaxed);
            } catch (...) {
                failures.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    in_flight.fetch_sub(1, std::memory_order_seq_cst);
}
