#pragma once
// Pure CPU placement-epoch coordinator. The transport owns any publication mechanism;
// neither this state machine nor its tests touch shared memory, HIP or a worker thread.
#include "moe_remote_shm.h"
#include <cstdint>
#include <limits>

namespace r3::epoch {
using namespace r3::moe_shm;

enum class FaultCode : uint32_t {
    NONE=0, INVALID_MAP=1, CAPACITY=2, SOURCE=3, UPLOAD=4, CHECKSUM=5,
    DEVICE=6, HELPER_DEAD=7, READY_TIMEOUT=8, QUIESCE_TIMEOUT=9,
    EPOCH_MISMATCH=10, COMMIT_TIMEOUT=11, RETIRE_TIMEOUT=12,
    RESULT_NOT_CONSUMED=13, SEQUENCE_EXHAUSTED=14, INCOMPATIBLE_LAYOUT=15,
    RANK_DEAD=16
};
enum class ActorKind : uint16_t { COORDINATOR=0, HELPER=1, ENGINE_RANK=2 };
inline uint32_t actor_id(ActorKind kind,uint16_t index) {
    return (uint32_t(kind)<<16)|uint32_t(index);
}
struct Clock {
    virtual ~Clock() = default;
    virtual uint64_t now_ns() const = 0; // monotonic, injectable for tests
};
struct Transport {
    virtual ~Transport() = default;
    virtual PlacementAck read_ack(uint32_t actor) const = 0;
    // Atomically publish a coherent snapshot and two-record diagnostic journal.
    // The concrete implementation of seqlock/notify lives in a later slice.
    virtual void publish(const PlacementControl&, const PlacementJournal&) = 0;
};
struct Deadlines {
    uint64_t reserve_ns = 60'000'000'000ULL;
    uint64_t copy_ns = 180'000'000'000ULL;
    uint64_t ready_ns = 30'000'000'000ULL;
    uint64_t quiesce_ns = 30'000'000'000ULL;
    uint64_t commit_ns = 10'000'000'000ULL;
    uint64_t retire_ns = 120'000'000'000ULL;
};
enum class Poll : uint8_t { WAITING, ADVANCED, FAILED };
// Validated active-generation seed, e.g. reconstructed after a coordinator restart.
struct Bootstrap {
    uint64_t epoch = 1;
    uint32_t slot = 0;
    uint32_t sequence = 0;
    uint64_t transaction_id = 0;
};
class Controller {
    Transport& transport_;
    const Clock& clock_;
    Deadlines deadlines_;
    PlacementControl c_{};
    PlacementJournal journal_{};
    bool failed_closed_ = false;
    bool deterministic_ = false;
    bool quiesced_ = false;
    bool pending_published_ = false;
    uint32_t reserved_helpers_ = 0, reserved_ranks_ = 0;
    uint32_t last_committed_index_ = UINT32_MAX;
    bool timeout(uint64_t duration) {
        const uint64_t now = clock_.now_ns();
        if (duration > UINT64_MAX - now) { fail(FaultCode::SEQUENCE_EXHAUSTED); return false; }
        c_.deadline_ns = now + duration;
        return true;
    }
    bool emit() {
        if (c_.seq > UINT32_MAX - 2) {
            fail_without_emit(FaultCode::SEQUENCE_EXHAUSTED, 0);
            failed_closed_ = true;
            return false;
        }
        c_.seq += 2; // transport supplies the actual odd/even writer protocol
        transport_.publish(c_, journal_);
        return true;
    }
    void fail_without_emit(FaultCode code, uint32_t actor) {
        if (c_.fault_code == 0) {
            c_.fault_code = uint32_t(code);
            c_.fault_actor = actor;
        }
        JournalRecord& j = journal_.entry[c_.journal_index];
        if (!j.fault) j.fault = c_.fault_code;
        j.state = c_.state;
        j.finish_ns = clock_.now_ns();
        if (c_.state >= uint32_t(PlacementState::QUIESCING)) failed_closed_ = true;
    }
    void reset_masks() {
        c_.ready_helpers = c_.ready_ranks = c_.quiesced_helpers = c_.quiesced_ranks = 0;
        c_.commit_helpers = c_.commit_ranks = c_.retired_helpers = c_.retired_ranks = 0;
    }
    bool enter(PlacementState state, uint64_t duration) {
        c_.state = uint32_t(state);
        journal_.entry[c_.journal_index].state = c_.state;
        if (!timeout(duration)) return false;
        return emit();
    }
    void journal_masks() {
        auto& j = journal_.entry[c_.journal_index];
        j.ready_helpers = c_.ready_helpers; j.ready_ranks = c_.ready_ranks;
        j.commit_helpers = c_.commit_helpers; j.commit_ranks = c_.commit_ranks;
        j.retire_helpers = c_.retired_helpers; j.retire_ranks = c_.retired_ranks;
    }
    static bool exact(const PlacementAck& a, const PlacementControl& c, AckPhase phase) {
        return a.transaction_id == c.transaction_id && a.epoch == c.pending_epoch &&
               a.digest == c.pending_digest && a.generation_slot == c.pending_slot &&
               a.phase == uint32_t(phase) && a.fault_code == 0;
    }
    bool collect(AckPhase phase, uint32_t& hm, uint32_t& rm) {
        hm = rm = 0;
        for (uint32_t i = 0; i < kMaxHelpers + kMaxEngineRanks; ++i) {
            const bool helper = i < kMaxHelpers;
            const uint32_t idx = helper ? i : i - kMaxHelpers;
            const uint32_t bit = uint32_t(1) << idx;
            if (!(helper ? (c_.expected_helpers & bit) : (c_.expected_ranks & bit))) continue;
            const PlacementAck a = transport_.read_ack(i);
            if (!exact(a, c_, phase)) continue; // stale/duplicate/lost never count
            if (helper) hm |= bit; else rm |= bit;
        }
        return hm == c_.expected_helpers && rm == c_.expected_ranks;
    }
    // Only a current-transaction, expected-phase ACK can signal a fatal fault.
    // A delayed old transaction or previous phase is ignored, never counted.
    bool check_phase_faults(AckPhase phase) {
        for (uint32_t i=0; i<kMaxHelpers+kMaxEngineRanks; ++i) {
            const bool helper=i<kMaxHelpers;
            const uint32_t bit=1u<<(helper ? i : i-kMaxHelpers);
            if (!(helper ? c_.expected_helpers & bit : c_.expected_ranks & bit)) continue;
            const PlacementAck a=transport_.read_ack(i);
            if (a.transaction_id != c_.transaction_id || a.phase != uint32_t(phase)) continue;
            if (a.fault_code) {
                fail(a.fault_code <= uint32_t(FaultCode::RANK_DEAD) ?
                    FaultCode(a.fault_code) : FaultCode::DEVICE,
                    actor_id(helper?ActorKind::HELPER:ActorKind::ENGINE_RANK,
                             uint16_t(helper?i:i-kMaxHelpers)));
                return false;
            }
            if (a.epoch != c_.pending_epoch || a.generation_slot != c_.pending_slot) {
                fail(FaultCode::EPOCH_MISMATCH,
                     actor_id(helper?ActorKind::HELPER:ActorKind::ENGINE_RANK,
                              uint16_t(helper?i:i-kMaxHelpers)));
                return false;
            }
            if (a.digest != c_.pending_digest) {
                fail(FaultCode::CHECKSUM,
                     actor_id(helper?ActorKind::HELPER:ActorKind::ENGINE_RANK,
                              uint16_t(helper?i:i-kMaxHelpers)));
                return false;
            }
        }
        return true;
    }
    bool run_phase(AckPhase phase, uint32_t& hm, uint32_t& rm) {
        uint32_t h, r;
        const bool all = collect(phase, h, r);
        if (h != hm || r != rm) {
            hm = h; rm = r;
            if (phase != AckPhase::RESERVED) { journal_masks(); emit(); }
        }
        return all;
    }
public:
    Controller(Transport& t, const Clock& clock, uint64_t initial_digest,
               uint32_t helper_count, uint32_t actual_rank_count, Deadlines d = {}, Bootstrap seed = {}, bool deterministic = false)
        : transport_(t), clock_(clock), deadlines_(d), deterministic_(deterministic) {
        // Bad configuration is a terminal local fault, not a partially published map.
        if (!helper_count || helper_count > kMaxHelpers ||
            !actual_rank_count || actual_rank_count > kMaxEngineRanks || !initial_digest ||
            !seed.epoch || seed.slot > 1 || (seed.sequence & 1u)) {
            c_.fault_code = uint32_t(FaultCode::INCOMPATIBLE_LAYOUT);
            failed_closed_ = true;
            return;
        }
        c_.state = uint32_t(PlacementState::ACTIVE);
        c_.seq = seed.sequence; c_.transaction_id = seed.transaction_id;
        c_.active_epoch = seed.epoch; c_.active_slot = seed.slot;
        c_.active_digest = initial_digest;
        c_.pending_slot = UINT32_MAX;
        c_.expected_helpers = (1u << helper_count) - 1u;
        c_.expected_ranks = (1u << actual_rank_count) - 1u;
        c_.journal_index = 1; // first transaction occupies entry 0
        emit();
    }
    const PlacementControl& control() const { return c_; }
    const PlacementJournal& journal() const { return journal_; }
    bool failed_closed() const { return failed_closed_; }
    bool can_propose() const {
        return !failed_closed_ && !deterministic_ && !c_.fault_code && c_.state == uint32_t(PlacementState::ACTIVE) &&
               c_.active_epoch != UINT64_MAX && c_.transaction_id != UINT64_MAX &&
               c_.seq <= UINT32_MAX - 128;
    }
    // The caller validates canonical map, source, eligibility and residency before proposing.
    // A zero/unchanged digest is a no-change, never an epoch increment.
    bool propose(uint64_t candidate_digest) {
        if (!can_propose() || !candidate_digest || candidate_digest == c_.active_digest) return false;
        ++c_.transaction_id;
        c_.pending_epoch = c_.active_epoch + 1;
        c_.pending_slot = 1u - c_.active_slot;
        c_.pending_digest = candidate_digest;
        c_.fault_code = c_.fault_actor = 0;
        c_.journal_index = last_committed_index_ < 2 ? 1u - last_committed_index_ : 0u;
        journal_.entry[c_.journal_index] = {};
        auto& j = journal_.entry[c_.journal_index];
        j.transaction_id = c_.transaction_id;
        j.before_epoch = c_.active_epoch; j.after_epoch = c_.pending_epoch;
        j.before_digest = c_.active_digest; j.candidate_digest = candidate_digest;
        j.old_slot = c_.active_slot; j.new_slot = c_.pending_slot;
        j.start_ns = clock_.now_ns();
        reset_masks(); reserved_helpers_ = reserved_ranks_ = 0;
        quiesced_ = pending_published_ = false;
        return enter(PlacementState::PROPOSED, deadlines_.ready_ns);
    }
    bool start_reserving() {
        return !failed_closed_ && c_.state == uint32_t(PlacementState::PROPOSED) &&
               enter(PlacementState::RESERVING, deadlines_.reserve_ns);
    }
    bool begin_quiesce() {
        return !failed_closed_ && c_.state == uint32_t(PlacementState::PREPARED) &&
               enter(PlacementState::QUIESCING, deadlines_.quiesce_ns);
    }
    // This is the explicit externally supplied proof of drained admission/GPU streams.
    // It is never inferred from READY, a chunk stamp, or the passage of time.
    bool commit() {
        if (failed_closed_ || deterministic_ || !quiesced_ ||
            c_.state != uint32_t(PlacementState::QUIESCING) ||
            c_.quiesced_helpers != c_.expected_helpers || c_.quiesced_ranks != c_.expected_ranks)
            return false;
        c_.active_epoch = c_.pending_epoch;
        c_.active_slot = c_.pending_slot;
        c_.active_digest = c_.pending_digest;
        pending_published_ = true;
        journal_.entry[c_.journal_index].commit_ns = clock_.now_ns();
        return enter(PlacementState::COMMIT, deadlines_.commit_ns);
    }
    Poll poll() {
        if (failed_closed_) return Poll::FAILED;
        const auto state = PlacementState(c_.state);
        if (state == PlacementState::ACTIVE) return Poll::WAITING;
        if (clock_.now_ns() >= c_.deadline_ns) {
            const auto code = state == PlacementState::COMMIT ? FaultCode::COMMIT_TIMEOUT :
                state == PlacementState::RETIRING ? FaultCode::RETIRE_TIMEOUT :
                state == PlacementState::QUIESCING ? FaultCode::QUIESCE_TIMEOUT :
                FaultCode::READY_TIMEOUT;
            fail(code);
            return Poll::FAILED;
        }
        if (state == PlacementState::PROPOSED || state == PlacementState::PREPARED)
            return Poll::WAITING;
        const AckPhase phase = state == PlacementState::RESERVING ? AckPhase::RESERVED :
            state == PlacementState::COPYING ? AckPhase::READY :
            state == PlacementState::QUIESCING ? AckPhase::QUIESCED :
            state == PlacementState::COMMIT ? AckPhase::COMMITTED : AckPhase::RETIRED;
        if (!check_phase_faults(phase)) return Poll::FAILED;
        bool all = false;
        if (state == PlacementState::RESERVING)
            all = run_phase(AckPhase::RESERVED, reserved_helpers_, reserved_ranks_);
        else if (state == PlacementState::COPYING)
            all = run_phase(AckPhase::READY, c_.ready_helpers, c_.ready_ranks);
        else if (state == PlacementState::QUIESCING)
            all = run_phase(AckPhase::QUIESCED, c_.quiesced_helpers, c_.quiesced_ranks);
        else if (state == PlacementState::COMMIT)
            all = run_phase(AckPhase::COMMITTED, c_.commit_helpers, c_.commit_ranks);
        else if (state == PlacementState::RETIRING)
            all = run_phase(AckPhase::RETIRED, c_.retired_helpers, c_.retired_ranks);
        if (!all) return Poll::WAITING;
        if (state == PlacementState::RESERVING)
            return enter(PlacementState::COPYING, deadlines_.copy_ns) ? Poll::ADVANCED : Poll::FAILED;
        if (state == PlacementState::COPYING)
            return enter(PlacementState::PREPARED, deadlines_.ready_ns) ? Poll::ADVANCED : Poll::FAILED;
        if (state == PlacementState::QUIESCING) {
            quiesced_ = true; emit();
            return Poll::ADVANCED; // external admission authority must explicitly commit
        }
        if (state == PlacementState::COMMIT)
            return enter(PlacementState::RETIRING, deadlines_.retire_ns) ? Poll::ADVANCED : Poll::FAILED;
        if (state == PlacementState::RETIRING) {
            auto& j = journal_.entry[c_.journal_index];
            j.finish_ns = clock_.now_ns(); j.state = uint32_t(PlacementState::ACTIVE);
            last_committed_index_ = c_.journal_index;
            c_.state = uint32_t(PlacementState::ACTIVE);
            c_.pending_epoch = 0; c_.pending_digest = 0; c_.pending_slot = UINT32_MAX;
            c_.deadline_ns = 0; reset_masks(); quiesced_ = pending_published_ = false;
            return emit() ? Poll::ADVANCED : Poll::FAILED;
        }
        return Poll::FAILED;
    }
    // First failure is retained. Precommit abort preserves ACTIVE(N); after publication
    // cannot roll back in place: caller must close admission and prove old residency safe.
    bool fail(FaultCode code, uint32_t actor = 0) {
        if (code == FaultCode::NONE) return false;
        const bool committed = pending_published_;
        const bool quiescing = c_.state >= uint32_t(PlacementState::QUIESCING);
        fail_without_emit(code, actor);
        if (!committed) {
            c_.state = uint32_t(PlacementState::ACTIVE);
            c_.pending_epoch = 0; c_.pending_digest = 0; c_.pending_slot = UINT32_MAX;
            c_.deadline_ns = 0; reset_masks(); quiesced_ = false;
            // Quiescence faults can imply partial output: never re-open admission.
            if (!quiescing) c_.fault_code = 0; // retain actor as last-fault diagnostic
        } else failed_closed_ = true;
        emit();
        return !committed;
    }
    // Rollback is a proposal of the last committed map, requiring a *new* epoch.
    // A postcommit fault is not recoverable here without external stream/residency proof.
    bool propose_rollback() {
        if (!can_propose()) return false;
        const auto& j = journal_.entry[c_.journal_index];
        if (!j.commit_ns || !j.before_digest || j.before_digest == c_.active_digest) return false;
        return propose(j.before_digest);
    }
};
} // namespace r3::epoch
