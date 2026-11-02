/* One engine rank's per-parent completion evidence: a terminal event and an
 * arming record per admission-ring slot, beside that rank's host-mapped
 * ConsumptionStamp ring. The GPU counts in a device-memory ring and the
 * per-parent seal kernel copies the parent's slot here, ahead of the event. Card-free; the event API is a policy so the hook
 * binds radiance's rad_event_* and tests bind a stub.
 *
 * Everything is sized once from the ledger's derived capacity, before the
 * rank's graphs are captured, and indexed by ticket_id % capacity like the
 * stamps. Slot reuse is safe because the ledger never admits id + capacity
 * while id is live, and it retires id only after this rank's event for id
 * completed. The steady state creates, destroys and allocates nothing.
 *
 * Threads: arm() runs on the rank thread before each pass, record() on the
 * scheduler thread after the parent's seal kernel, prove() under the ledger
 * lock from whichever thread polls. The arming/recording words are
 * release-stored and acquire-loaded by prove(). See ADMISSION.md.
 */
#pragma once
#include "admission_ledger.h"
#include "moe_remote_hook.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

namespace r3 {
namespace admission {

/* Ev: using Event; static int create(Event*) (>= 0 ok); static void destroy(Event);
 *     static int record(Event, void* stream) (>= 0 ok);
 *     static int query(Event) (1 complete, 0 pending, < 0 error). */
template<class Ev>
class RankCompletion {
public:
    using Event = typename Ev::Event;
    RankCompletion() = default;
    RankCompletion(const RankCompletion&) = delete;
    RankCompletion& operator=(const RankCompletion&) = delete;
    ~RankCompletion() { release(); }

    /* Creates `capacity` events. On failure everything made so far is undone. */
    bool prepare(size_t capacity, ConsumptionStamp* stamps) {
        if (ready_.load(std::memory_order_acquire) || !capacity || !stamps) return false;
        event_.reset(new (std::nothrow) Event[capacity]());
        armed_.reset(new (std::nothrow) std::atomic<uint64_t>[capacity]());
        recorded_.reset(new (std::nothrow) std::atomic<uint64_t>[capacity]());
        if (!event_ || !armed_ || !recorded_) { reset_arrays(); return false; }
        capacity_ = capacity;
        for (size_t i = 0; i < capacity; ++i)
            if (Ev::create(&event_[i]) < 0 || !event_[i]) { destroy_events(); reset_arrays(); return false; }
        stamps_ = stamps;
        ready_.store(true, std::memory_order_release);
        return true;
    }
    void release() {
        ready_.store(false, std::memory_order_release);
        destroy_events();
        reset_arrays();
    }
    bool ready() const { return ready_.load(std::memory_order_acquire); }

    /* Notes that this rank ran a pass of the parent; the ticket setter arms
     * the device counting slot in the same stream order. */
    bool arm(const StepTicket& t) {
        if (!ready()) return false;
        const size_t i = t.ticket_id % capacity_;
        if (armed_[i].load(std::memory_order_relaxed) != t.ticket_id)
            armed_[i].store(t.ticket_id, std::memory_order_release);
        return true;
    }
    /* Exactly one terminal record per parent and rank. */
    bool record(uint64_t id, void* stream) {
        if (!ready() || !stream) return false;
        const size_t i = id % capacity_;
        if (recorded_[i].load(std::memory_order_relaxed) == id) return false;
        if (Ev::record(event_[i], stream) < 0) return false;
        recorded_[i].store(id, std::memory_order_release);
        return true;
    }
    /* Proven needs: this parent's own terminal event complete, the rank armed
     * for it, and its sealed stamp matching ticket/epoch/slot with published
     * equal to consumed. A complete event without the matching stamp stays
     * pending. */
    Ledger::Proof prove(const StepTicket& t) const {
        using P = Ledger::Proof;
        if (!ready()) return P::Pending;
        const size_t i = t.ticket_id % capacity_;
        if (recorded_[i].load(std::memory_order_acquire) != t.ticket_id) return P::Pending;
        const int status = Ev::query(event_[i]);
        if (status == 0) return P::Pending;
        if (status < 0) return P::Fault;
        if (armed_[i].load(std::memory_order_acquire) != t.ticket_id) return P::Fault;
        const ConsumptionStamp* s = stamps_ + i;
        if (__atomic_load_n(&s->ticket_id, __ATOMIC_ACQUIRE) != t.ticket_id ||
            s->placement_epoch != t.placement_epoch || s->generation_slot != t.generation_slot ||
            __atomic_load_n(&s->published_count, __ATOMIC_ACQUIRE) !=
            __atomic_load_n(&s->consumed_count, __ATOMIC_ACQUIRE))
            return P::Pending;
        return P::Proven;
    }
private:
    void destroy_events() {
        if (!event_) return;
        for (size_t i = 0; i < capacity_; ++i)
            if (event_[i]) { Ev::destroy(event_[i]); event_[i] = Event{}; }
    }
    void reset_arrays() {
        event_.reset(); armed_.reset(); recorded_.reset();
        capacity_ = 0; stamps_ = nullptr;
    }
    std::atomic<bool> ready_{false};
    size_t capacity_ = 0;
    ConsumptionStamp* stamps_ = nullptr;
    std::unique_ptr<Event[]> event_;
    std::unique_ptr<std::atomic<uint64_t>[]> armed_, recorded_;
};

} // namespace admission
} // namespace r3
