/* Admission and completion authority for remote-MoE placement epochs.
 *
 * Pure CPU bookkeeping; no HIP, no shared-memory side effects. The hook
 * coordinator owns one of these per engine, not per rank. The engine supplies
 * the ceiling from its RouteSlot ring plus its explicit issue-ahead depth.
 * The deque grows on demand to that ceiling; draft passes share their parent.
 * Full fails closed without a serving-path GPU wait. See ADMISSION.md.
 */
#pragma once
#include "moe_remote_hook.h"
#include "moe_remote_shm.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <new>

namespace r3 {
namespace admission {

class Ledger {
public:
    enum class Result { Ok, Closed, Full, Interrupted, Fault, Invalid, Timeout };
    struct Entry {
        StepTicket ticket{};
        int64_t scheduler_step = -1;
        bool bound = false;
        bool host_done = false;
        uint32_t ranks = 0;
        uint32_t gpu_mask = 0;
        uint32_t consumed_mask = 0;
    };
    struct Snapshot {
        uint64_t latest_ticket = 0;
        uint32_t old_epoch = 0;
        size_t outstanding = 0;
        size_t high_water = 0;
        size_t capacity = 0;
    };

    explicit Ledger(size_t capacity) : capacity_(capacity) {}
    bool configure(size_t capacity) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!capacity || interrupted_ || fault_) return false;
        if (capacity_) return capacity_ == capacity;
        if (!q_.empty()) return false;
        capacity_ = capacity;
        return true;
    }
    bool initialize(uint32_t epoch, uint8_t slot) {
        std::lock_guard<std::mutex> lock(mu_);
        if (active_epoch_ || !epoch || slot > 1) return false;
        active_epoch_ = epoch; active_slot_ = slot;
        return true;
    }
    bool stopped() const {
        std::lock_guard<std::mutex> lock(mu_);
        return interrupted_ || fault_;
    }

    /* Observer is only consulted while waiting, never on the normal admit
     * fast path: it reads the helper fault word or shutdown flag. */
    void interrupt() {
        std::lock_guard<std::mutex> lock(mu_);
        interrupted_ = true;
        cv_.notify_all();
    }
    void fault() {
        std::lock_guard<std::mutex> lock(mu_);
        fault_ = true;
        cv_.notify_all();
    }
    Result try_admit(uint32_t ranks, StepTicket* out) {
        std::lock_guard<std::mutex> lock(mu_);
        return allocate(active_epoch_, active_slot_, ranks, out);
    }
    Result admit(uint32_t ranks, StepTicket* out,
                 const std::function<bool()>& external_fault = {},
                 const std::function<void()>& progress = {}) {
        std::unique_lock<std::mutex> lock(mu_);
        for (;;) {
            if (interrupted_) return Result::Interrupted;
            if (fault_ || (external_fault && external_fault())) {
                fault_ = true;
                return Result::Fault;
            }
            if (!closed_) return allocate(active_epoch_, active_slot_, ranks, out);
            lock.unlock();
            if (progress) progress();
            lock.lock();
            cv_.wait_for(lock, std::chrono::milliseconds(5));
        }
    }
    bool bind(uint64_t id, int64_t scheduler_step) {
        std::lock_guard<std::mutex> lock(mu_);
        Entry* e = get(id);
        if (!e || e->bound || scheduler_step < 0) return false;
        e->bound = true; e->scheduler_step = scheduler_step;
        return true;
    }
    bool valid(const StepTicket& ticket, uint32_t rank) const {
        std::lock_guard<std::mutex> lock(mu_);
        const Entry* e = get(ticket.ticket_id);
        return e && e->bound && e->ticket.ticket_id == ticket.ticket_id &&
               e->ticket.placement_epoch == ticket.placement_epoch &&
               e->ticket.generation_slot == ticket.generation_slot && rank < e->ranks;
    }
    bool is_live(uint64_t id) const {
        std::lock_guard<std::mutex> lock(mu_);
        return get(id) != nullptr;
    }
    bool cancel(uint64_t id) {
        std::lock_guard<std::mutex> lock(mu_);
        /* Only unused, unbound tail reservations can be returned. */
        if (q_.empty() || q_.back().ticket.ticket_id != id || q_.back().bound) return false;
        q_.pop_back();
        /* The cancelled reservation was never bound or handed to a GPU.
         * Reclaim its provisional sequence number so outstanding IDs stay
         * consecutive: the fixed per-rank consumption ring uses id%capacity
         * and MUST NOT alias two simultaneously admitted parents. */
        --last_id_;
        cv_.notify_all();
        return true;
    }
    bool host_complete(uint64_t id) {
        std::lock_guard<std::mutex> lock(mu_);
        Entry* e = get(id);
        if (!e || !e->bound) return false;
        e->host_done = true;
        retire_locked();
        return true;
    }
    /* Rank signals are monotone, repeatable and may complete out of order. A
     * rank is complete only after its per-parent event is observed AND its
     * request-consumed stamps are verified. */
    bool gpu_complete(uint64_t id, uint32_t rank) {
        std::lock_guard<std::mutex> lock(mu_);
        Entry* e = get(id);
        if (!e || rank >= e->ranks) return false;
        e->gpu_mask |= (1u << rank);
        retire_locked();
        return true;
    }
    bool consumed(uint64_t id, uint32_t rank) {
        std::lock_guard<std::mutex> lock(mu_);
        Entry* e = get(id);
        if (!e || rank >= e->ranks) return false;
        e->consumed_mask |= (1u << rank);
        retire_locked();
        return true;
    }
    Snapshot close_admission() {
        std::lock_guard<std::mutex> lock(mu_);
        closed_ = true;
        Snapshot s{last_id_, active_epoch_,
                   q_.size(), high_water_, capacity_};
        cv_.notify_all();
        return s;
    }
    /* Only reopen once every older epoch has retired. */
    bool reopen(uint32_t epoch, uint8_t slot) {
        std::lock_guard<std::mutex> lock(mu_);
        if (interrupted_ || fault_ || !closed_ || !q_.empty() || epoch == 0 || epoch < active_epoch_ || slot > 1 ||
            (epoch == active_epoch_ && slot != active_slot_)) return false;
        active_epoch_ = epoch;
        active_slot_ = slot;
        closed_ = false;
        cv_.notify_all();
        return true;
    }
    bool drained(uint32_t old_epoch, uint64_t cutoff_id) const {
        std::lock_guard<std::mutex> lock(mu_);
        for (const Entry& e : q_)
            if (e.ticket.ticket_id <= cutoff_id && e.ticket.placement_epoch == old_epoch)
                return false;
        return true;
    }
    Result wait_drained(uint32_t old_epoch, uint64_t cutoff_id,
                        std::chrono::steady_clock::time_point deadline,
                        const std::function<void()>& progress = {},
                        const std::function<bool()>& external_fault = {}) {
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mu_);
                if (interrupted_) return Result::Interrupted;
                if (fault_ || (external_fault && external_fault())) {
                    fault_ = true;
                    return Result::Fault;
                }
                bool pending = false;
                for (const Entry& e : q_)
                    pending |= e.ticket.ticket_id <= cutoff_id &&
                               e.ticket.placement_epoch == old_epoch;
                if (!pending) return Result::Ok;
                if (std::chrono::steady_clock::now() >= deadline) {
                    fault_ = true;
                    cv_.notify_all();
                    return Result::Timeout;
                }
            }
            if (progress) progress();
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait_until(lock, std::min(deadline,
                           std::chrono::steady_clock::now() + std::chrono::milliseconds(2)));
        }
    }
    size_t high_water() const {
        std::lock_guard<std::mutex> lock(mu_);
        return high_water_;
    }
    size_t capacity() const {
        std::lock_guard<std::mutex> lock(mu_);
        return capacity_;
    }
    size_t outstanding() const {
        std::lock_guard<std::mutex> lock(mu_);
        return q_.size();
    }
    /* Snapshot copies never expose mutable internal pointers. */
    std::deque<Entry> entries() const {
        std::lock_guard<std::mutex> lock(mu_);
        return q_;
    }
private:
    Result allocate(uint32_t epoch, uint8_t slot, uint32_t ranks, StepTicket* out) {
        if (interrupted_) return Result::Interrupted;
        if (fault_) return Result::Fault;
        if (closed_) return Result::Closed;
        if (!capacity_ || !out || !epoch || slot > 1 || ranks == 0 ||
            ranks > r3::moe_shm::kMaxEngineRanks ||
            (active_epoch_ && epoch != active_epoch_)) return Result::Invalid;
        if (last_id_ == UINT64_MAX) return Result::Fault;
        if (q_.size() >= capacity_) return Result::Full;
        if (!active_epoch_) active_epoch_ = epoch; /* no silent epoch drift before close/reopen */
        Entry e{};
        e.ticket.placement_epoch = epoch;
        e.ticket.generation_slot = slot;
        e.ticket.ticket_id = last_id_ + 1;
        e.ranks = ranks;
        try { q_.push_back(e); }
        catch (const std::bad_alloc&) { fault_ = true; cv_.notify_all(); return Result::Fault; }
        last_id_ = e.ticket.ticket_id;
        *out = e.ticket;
        high_water_ = std::max(high_water_, q_.size());
        return Result::Ok;
    }
    Entry* get(uint64_t id) {
        for (Entry& e : q_) if (e.ticket.ticket_id == id) return &e;
        return nullptr;
    }
    const Entry* get(uint64_t id) const {
        for (const Entry& e : q_) if (e.ticket.ticket_id == id) return &e;
        return nullptr;
    }
    void retire_locked() {
        while (!q_.empty()) {
            const Entry& e = q_.front();
            const uint32_t all = (1u << e.ranks) - 1u;
            if (!e.host_done || e.gpu_mask != all || e.consumed_mask != all) break;
            q_.pop_front();
            cv_.notify_all();
        }
    }
    size_t capacity_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Entry> q_;
    uint64_t last_id_ = 0;
    uint32_t active_epoch_ = 0;
    uint8_t active_slot_ = 0;
    size_t high_water_ = 0;
    bool closed_ = false, interrupted_ = false, fault_ = false;
};
} // namespace admission
} // namespace r3
