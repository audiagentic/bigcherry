#pragma once
// CPU-only, per-helper residency accounting. No device pointers or asynchronous side effects.
// Copy completion and retirement are *proofs supplied by the caller*, never inferred here.
#include <cstdint>
#include <limits>
#include <vector>

namespace r3::residency {

enum class State : uint8_t { FREE, RESERVED, COPYING, READY, ACTIVE, RETIRING };
enum class Result : uint8_t { OK, INVALID, DEFERRED_CAPACITY, SOURCE_UNAVAILABLE, BUSY };
struct Slot {
    uint32_t id = 0;
    uint32_t expert = 0;
    uint64_t bytes = 0;
    uint64_t epoch = 0;
    State state = State::FREE;
    uint8_t completed_planes = 0; // four whole-expert planes, never publish partial data
};

class Ledger {
    uint64_t hard_, scratch_, headroom_, charged_ = 0;
    uint32_t copying_ = 0;
    std::vector<Slot> slots_;
    static bool fits(uint64_t add, uint64_t used, uint64_t hard) {
        return used <= hard && add <= hard - used;
    }
    Slot* find(uint32_t id) {
        for (auto& s : slots_) if (s.id == id) return &s;
        return nullptr;
    }
    Result insert(uint32_t expert, uint64_t bytes, uint64_t epoch,
                  State state, uint8_t planes, uint32_t* id) {
        for (auto& s : slots_) if (s.state == State::FREE) {
            *id = s.id;
            s = {*id, expert, bytes, epoch, state, planes};
            charged_ += bytes;
            return Result::OK;
        }
        if (slots_.size() >= UINT32_MAX) return Result::INVALID;
        *id = static_cast<uint32_t>(slots_.size());
        slots_.push_back({*id, expert, bytes, epoch, state, planes});
        charged_ += bytes;
        return Result::OK;
    }
public:
    // Reserve headroom for an incoming expert even after a steady-state bootstrap.
    // Scratch includes lane workspace and all pinned/transfer staging charged to this helper.
    Ledger(uint64_t hard, uint64_t scratch, uint64_t incoming_reserve)
        : hard_(hard), scratch_(scratch), headroom_(incoming_reserve) {}
    uint64_t hard() const { return hard_; }
    uint64_t charged() const { return charged_; }
    uint64_t total() const { return charged_ + scratch_; }
    uint64_t headroom() const { return headroom_; }
    const std::vector<Slot>& slots() const { return slots_; }
    bool valid() const { return scratch_ <= hard_ && charged_ <= hard_ - scratch_; }
    // Only bootstrap may bypass state transitions. It must still leave incoming headroom.
    Result bootstrap(uint32_t expert, uint64_t bytes, uint64_t epoch, uint32_t* id) {
        if (!id || !bytes || !epoch || !valid()) return Result::INVALID;
        if (bytes > hard_ - scratch_ - charged_ ||
            headroom_ > hard_ - scratch_ - charged_ - bytes) return Result::DEFERRED_CAPACITY;
        for (const auto& s : slots_)
            if (s.expert == expert && s.state != State::FREE) return Result::INVALID;
        return insert(expert, bytes, epoch, State::ACTIVE, 15, id);
    }
    Result reserve(uint32_t expert, uint64_t bytes, uint64_t epoch, bool source_available, uint32_t* id) {
        if (!source_available) return Result::SOURCE_UNAVAILABLE;
        if (!id || !bytes || !epoch || !valid()) return Result::INVALID;
        for (const auto& s : slots_)
            if (s.expert == expert && s.state != State::FREE) return Result::INVALID;
        if (!fits(bytes, total(), hard_)) return Result::DEFERRED_CAPACITY;
        // A helper with zero incumbent slots can still accept the first incoming expert.
        return insert(expert, bytes, epoch, State::RESERVED, 0, id);
    }
    Result start_copy(uint32_t id) {
        Slot* s = find(id);
        if (!s || s->state != State::RESERVED) return Result::INVALID;
        if (copying_) return Result::BUSY; // one incoming upload at a time
        s->state = State::COPYING; ++copying_;
        return Result::OK;
    }
    Result plane_complete(uint32_t id, uint32_t plane) {
        Slot* s = find(id);
        if (!s || s->state != State::COPYING || plane >= 4 || (s->completed_planes & (1u << plane)))
            return Result::INVALID;
        s->completed_planes |= uint8_t(1u << plane);
        return Result::OK;
    }
    Result ready(uint32_t id) {
        Slot* s = find(id);
        if (!s || s->state != State::COPYING || s->completed_planes != 15) return Result::INVALID;
        s->state = State::READY; --copying_;
        return Result::OK;
    }
    Result activate(uint32_t id, uint64_t epoch) {
        Slot* s = find(id);
        if (!s || s->state != State::READY || s->epoch != epoch) return Result::INVALID;
        s->state = State::ACTIVE;
        return Result::OK;
    }
    Result retire(uint32_t id) {
        Slot* s = find(id);
        if (!s || s->state != State::ACTIVE) return Result::INVALID;
        s->state = State::RETIRING;
        return Result::OK;
    }
    Result reclaim(uint32_t id, bool all_references_retired) {
        Slot* s = find(id);
        if (!s || s->state != State::RETIRING || !all_references_retired) return Result::INVALID;
        charged_ -= s->bytes; s->state = State::FREE;
        return Result::OK;
    }
    // Precommit cancellation: caller has already proven any copy was stopped/drained.
    Result cancel(uint32_t id, bool copy_drained) {
        Slot* s = find(id);
        if (!s || (s->state != State::RESERVED && s->state != State::COPYING &&
                    s->state != State::READY) || (s->state == State::COPYING && !copy_drained))
            return Result::INVALID;
        if (s->state == State::COPYING) --copying_;
        charged_ -= s->bytes; s->state = State::FREE;
        return Result::OK;
    }
};
} // namespace r3::residency
