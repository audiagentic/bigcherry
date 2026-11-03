/* Admission and completion authority for remote-MoE placement epochs.
 *
 * Pure CPU bookkeeping; no HIP, no shared-memory side effects. The hook
 * coordinator owns one of these per engine, not per rank. The engine supplies
 * the ceiling from its RouteSlot ring plus its explicit issue-ahead depth.
 * Outstanding IDs are consecutive, so entries live in a fixed ring allocated
 * once at configure and indexed by ticket_id % capacity, the same modulus as
 * the per-rank ConsumptionStamp ring: no per-step allocation or search. Draft
 * passes share their parent. Full fails closed without a serving-path GPU
 * wait. See ADMISSION.md.
 */
#pragma once
#include "moe_remote_hook.h"
#include "moe_remote_shm.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <new>
#include <vector>

namespace r3 {
namespace admission {

class Ledger {
public:
    enum class Result { Ok, Closed, Full, Interrupted, Fault, Invalid, Timeout };
    /* One rank's completion proof for one host-complete parent. */
    enum class Proof { Pending, Proven, Fault };
    struct Entry {
        StepTicket ticket{};
        int64_t scheduler_step = -1;
        bool bound = false;
        bool host_done = false;
        uint32_t ranks = 0;
        uint32_t proven_mask = 0;   /* ranks whose event and consumption proof both held (poll) */
    };
    struct Snapshot {
        uint64_t latest_ticket = 0;
        uint32_t old_epoch = 0;
        size_t outstanding = 0;
        size_t high_water = 0;
        size_t capacity = 0;
    };

    explicit Ledger(size_t capacity) { if (capacity) reserve(capacity); }
    bool configure(size_t capacity) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!capacity || interrupted_ || fault_) return false;
        if (capacity_) return capacity_ == capacity;
        return reserve(capacity);
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
        /* Only unused, unbound tail reservations can be returned. Anything
         * else means the bridge lost track of a reservation: fail closed. */
        if (!count_ || at(last_id_).bound || at(last_id_).ticket.ticket_id != id) {
            fault_ = true;
            cv_.notify_all();
            return false;
        }
        at(last_id_) = Entry{};
        --count_;
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
    /* The only way a rank completes a parent: one lock for every host-complete
     * parent. prove(ticket, rank) proves a rank only after its per-parent event
     * is observed AND its request-consumed stamp is verified (RankCompletion);
     * ranks may complete out of order. prove runs under the ledger lock and
     * must not re-enter it. A rank's terminal events are recorded in admission
     * order on one stream, so once a rank is pending no younger parent is asked
     * for that rank; the scan ends when every rank is pending. Cost is
     * O(outstanding ranks newly proven + one pending query per rank). */
    template<class Prove>
    bool poll(Prove&& prove) {
        std::lock_guard<std::mutex> lock(mu_);
        if (interrupted_ || fault_) return false;
        uint32_t pending = 0;
        for (size_t i = 0; i < count_; ++i) {
            Entry& e = at(last_id_ - count_ + 1 + i);
            const uint32_t all = (1u << e.ranks) - 1u;
            if (!e.host_done) continue; /* never treat an intermediate event as terminal */
            for (uint32_t rank = 0; rank < e.ranks; ++rank) {
                const uint32_t bit = 1u << rank;
                if ((pending & bit) || (e.proven_mask & bit)) continue;
                const Proof p = prove(static_cast<const StepTicket&>(e.ticket), rank);
                if (p == Proof::Fault) { fault_ = true; cv_.notify_all(); return false; }
                if (p == Proof::Pending) { pending |= bit; continue; }
                e.proven_mask |= bit;
            }
            if ((pending & all) == all) break;
        }
        retire_locked();
        return true;
    }
    Snapshot close_admission() {
        std::lock_guard<std::mutex> lock(mu_);
        closed_ = true;
        Snapshot s{last_id_, active_epoch_,
                   count_, high_water_, capacity_};
        cv_.notify_all();
        return s;
    }
    /* Only reopen once every older epoch has retired. */
    bool reopen(uint32_t epoch, uint8_t slot) {
        std::lock_guard<std::mutex> lock(mu_);
        if (interrupted_ || fault_ || !closed_ || count_ || epoch == 0 || epoch < active_epoch_ || slot > 1 ||
            (epoch == active_epoch_ && slot != active_slot_)) return false;
        active_epoch_ = epoch;
        active_slot_ = slot;
        closed_ = false;
        cv_.notify_all();
        return true;
    }
    bool drained(uint32_t old_epoch, uint64_t cutoff_id) const {
        std::lock_guard<std::mutex> lock(mu_);
        return !pending_locked(old_epoch, cutoff_id);
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
                if (!pending_locked(old_epoch, cutoff_id)) return Result::Ok;
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
        return count_;
    }
    /* Snapshot copies, oldest first, never expose mutable internal pointers. */
    std::vector<Entry> entries() const {
        std::lock_guard<std::mutex> lock(mu_);
        std::vector<Entry> out;
        out.reserve(count_);
        for (size_t i = 0; i < count_; ++i) out.push_back(at(last_id_ - count_ + 1 + i));
        return out;
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
        if (count_ >= capacity_) return Result::Full;
        if (!active_epoch_) active_epoch_ = epoch; /* no silent epoch drift before close/reopen */
        Entry& e = at(last_id_ + 1);
        e = Entry{};
        e.ticket.placement_epoch = epoch;
        e.ticket.generation_slot = slot;
        e.ticket.ticket_id = last_id_ + 1;
        e.ranks = ranks;
        last_id_ = e.ticket.ticket_id;
        ++count_;
        *out = e.ticket;
        high_water_ = std::max(high_water_, count_);
        return Result::Ok;
    }
    /* The ring is sized once to the derived ceiling; serving never allocates. */
    bool reserve(size_t capacity) {
        try { ring_.assign(capacity, Entry{}); }
        catch (const std::bad_alloc&) { fault_ = true; return false; }
        capacity_ = capacity;
        return true;
    }
    Entry& at(uint64_t id) { return ring_[id % capacity_]; }
    const Entry& at(uint64_t id) const { return ring_[id % capacity_]; }
    /* Live IDs are exactly (last_id_ - count_, last_id_]. */
    Entry* get(uint64_t id) {
        if (!count_ || id > last_id_ || id <= last_id_ - count_) return nullptr;
        return &at(id);
    }
    const Entry* get(uint64_t id) const {
        if (!count_ || id > last_id_ || id <= last_id_ - count_) return nullptr;
        return &at(id);
    }
    bool pending_locked(uint32_t old_epoch, uint64_t cutoff_id) const {
        for (size_t i = 0; i < count_; ++i) {
            const Entry& e = at(last_id_ - count_ + 1 + i);
            if (e.ticket.ticket_id <= cutoff_id && e.ticket.placement_epoch == old_epoch) return true;
        }
        return false;
    }
    void retire_locked() {
        bool retired = false;
        while (count_) {
            Entry& e = at(last_id_ - count_ + 1);
            const uint32_t all = (1u << e.ranks) - 1u;
            if (!e.host_done || e.proven_mask != all) break;
            e = Entry{};
            --count_;
            retired = true;
        }
        if (retired) cv_.notify_all();
    }
    size_t capacity_ = 0;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::vector<Entry> ring_;
    size_t count_ = 0;
    uint64_t last_id_ = 0;
    uint32_t active_epoch_ = 0;
    uint8_t active_slot_ = 0;
    size_t high_water_ = 0;
    bool closed_ = false, interrupted_ = false, fault_ = false;
};
} // namespace admission
} // namespace r3
