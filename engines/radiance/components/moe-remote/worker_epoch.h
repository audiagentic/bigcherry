#pragma once
// CPU placement-epoch coordinator plus the v23 shared-memory cycle transport.
// The state machine uses injectable clocks/transport; no HIP or GPU work here.
#include "moe_remote_shm.h"
#include <cstdint>
#include <atomic>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <memory>
#include <array>
#include <limits>
#include <functional>

namespace r3::epoch {
using namespace r3::moe_shm;

enum class FaultCode : uint32_t {
    NONE=0, INVALID_MAP=1, CAPACITY=2, SOURCE=3, UPLOAD=4, CHECKSUM=5,
    DEVICE=6, HELPER_DEAD=7, READY_TIMEOUT=8, QUIESCE_TIMEOUT=9,
    EPOCH_MISMATCH=10, COMMIT_TIMEOUT=11, RETIRE_TIMEOUT=12,
    RESULT_NOT_CONSUMED=13, SEQUENCE_EXHAUSTED=14, INCOMPATIBLE_LAYOUT=15,
    RANK_DEAD=16, REOPEN_TIMEOUT=17, CLEANUP_TIMEOUT=18
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
    // SharedTransport below implements the v23 seqlock/notify publication.
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
    struct HelperMeasurements {
        uint64_t transaction=0, service_ns=0, service_wall_ns=0, service_rows=0, copy_bytes=0, copy_ns=0;
    };
    std::array<HelperMeasurements,kMaxHelpers> helper_measurements_{};
    bool failed_closed_ = false;
    bool deterministic_ = false;
    bool quiesced_ = false;
    bool pending_published_ = false;
    uint32_t reserved_helpers_ = 0, reserved_ranks_ = 0;
    uint32_t last_committed_index_ = UINT32_MAX;
    uint64_t awaiting_cleanup_ = 0;
    bool cleanup_committed_ = false;
    uint64_t cleanup_deadline_ns_ = 0;
    void await_cleanup(bool committed) {
        awaiting_cleanup_ = c_.transaction_id;
        cleanup_committed_ = committed;
        const uint64_t now = clock_.now_ns();
        // Committed cleanup includes reopen and reclamation; aborted cleanup
        // includes outstanding rank DMA and helper cancellation.
        const uint64_t window = committed ? deadlines_.retire_ns : deadlines_.copy_ns;
        cleanup_deadline_ns_ = window > UINT64_MAX - now ? now : now + window;
    }
    bool close_ack(uint32_t actor, AckPhase phase) const {
        const auto a = transport_.read_ack(actor);
        return a.phase == uint32_t(phase) && !a.fault_code &&
               a.transaction_id == awaiting_cleanup_ &&
               a.epoch == c_.active_epoch && a.digest == c_.active_digest &&
               a.generation_slot == c_.active_slot;
    }
    uint32_t missing_cleanup_actor() const {
        if (cleanup_committed_) {
            if (!close_ack(kMaxHelpers,AckPhase::REOPENED))
                return actor_id(ActorKind::ENGINE_RANK,0);
        } else {
            for (uint32_t i=0; i<kMaxEngineRanks; ++i)
                if ((c_.expected_ranks & (1u<<i)) &&
                    !close_ack(kMaxHelpers+i,AckPhase::CLEANED))
                    return actor_id(ActorKind::ENGINE_RANK,uint16_t(i));
        }
        for (uint32_t i=0; i<kMaxHelpers; ++i)
            if ((c_.expected_helpers & (1u<<i)) &&
                !close_ack(i,AckPhase::CLEANED))
                return actor_id(ActorKind::HELPER,uint16_t(i));
        return actor_id(ActorKind::COORDINATOR,0);
    }
    // No new proposal may overwrite a generation or its ACK identity until
    // engine reopen and every helper reclaim (or all aborts) have completed.
    bool previous_closed() const {
        if (!awaiting_cleanup_) return true;
        if (cleanup_committed_) {
            if (!close_ack(kMaxHelpers,AckPhase::REOPENED)) return false;
        } else {
            for (uint32_t i=0;i<kMaxEngineRanks;++i)
                if ((c_.expected_ranks & (1u<<i)) &&
                    !close_ack(kMaxHelpers+i,AckPhase::CLEANED)) return false;
        }
        for (uint32_t i=0;i<kMaxHelpers;++i)
            if ((c_.expected_helpers & (1u<<i)) &&
                !close_ack(i,AckPhase::CLEANED)) return false;
        return true;
    }
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
            if (helper && (phase==AckPhase::RESERVED || phase==AckPhase::READY)) {
                auto& m=helper_measurements_[i];
                if (m.transaction!=c_.transaction_id) {
                    m={};
                    m.transaction=c_.transaction_id;
                }
                if (phase==AckPhase::RESERVED) {
                    m.service_ns=a.reserved[0];
                    m.service_wall_ns=a.reserved[1];
                    m.service_rows=a.drained_step; // RESERVED-only: completed large-lane routed rows
                } else {
                    m.copy_bytes=a.reserved[0];
                    m.copy_ns=a.reserved[1];
                }
            }
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
        if (seed.transaction_id) await_cleanup(true);
        emit();
    }
    const PlacementControl& control() const { return c_; }
    const PlacementJournal& journal() const { return journal_; }
    HelperMeasurements helper_measurement(uint32_t helper) const {
        return helper<kMaxHelpers ? helper_measurements_[helper] : HelperMeasurements{};
    }
    PlacementAck helper_ack(uint32_t helper) const {
        return helper < kMaxHelpers ? transport_.read_ack(helper) : PlacementAck{};
    }
    bool failed_closed() const { return failed_closed_; }
    bool can_propose() const {
        return !failed_closed_ && !deterministic_ && !c_.fault_code &&
               c_.state == uint32_t(PlacementState::ACTIVE) && previous_closed() &&
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
    /* Invalid precommit staging is a rejected transaction, not a serving
     * failure. Preserve ACTIVE(N), issue a fresh journal identity, and leave
     * the ledger and current generation untouched. */
    bool reject_staging(FaultCode code) {
        if (!can_propose() || code == FaultCode::NONE) return false;
        ++c_.transaction_id;
        c_.journal_index = last_committed_index_ < 2 ? 1u-last_committed_index_ : 0u;
        auto& j=journal_.entry[c_.journal_index];
        j={};
        j.transaction_id=c_.transaction_id;
        j.before_epoch=c_.active_epoch;
        j.before_digest=c_.active_digest;
        j.start_ns=clock_.now_ns();
        fail_without_emit(code,actor_id(ActorKind::COORDINATOR,0));
        await_cleanup(false);
        c_.fault_code=0;
        return emit();
    }
    // Diagnostic no-move transaction: identical owner bytes, but a genuinely
    // NEW epoch and alternate generation slot. Unlike the previous shortcut,
    // it passes RESERVED, COPYING, PREPARED, QUIESCING, COMMIT and RETIRING;
    // the engine ranks must upload and prove the inactive bitmap's event.
    // The caller stages the canonical same-map generation before publishing.
    bool begin_epoch_cycle() {
        if (!can_propose() || c_.active_epoch >= UINT32_MAX) return false;
        ++c_.transaction_id;
        c_.pending_epoch = c_.active_epoch + 1;
        c_.pending_slot = 1u - c_.active_slot;
        c_.pending_digest = c_.active_digest;
        c_.journal_index = last_committed_index_ < 2 ? 1u - last_committed_index_ : 0u;
        auto& j = journal_.entry[c_.journal_index];
        j = {};
        j.transaction_id = c_.transaction_id;
        j.before_epoch = c_.active_epoch;
        j.after_epoch = c_.pending_epoch;
        j.before_digest = j.candidate_digest = c_.active_digest;
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
    // Live transport integration: r3_moe_close_admission snapshots the hook
    // ledger; r3_moe_wait_drained(snapshot, deadline_ns) proves per-rank GPU
    // completion, result consumption and helper queue/combine drain. Only
    // after that proof may QUIESCED ACKs/commit proceed. Reopen with the
    // committed epoch/slot after all COMMITTED ACKs; failures stay closed.
    // These APIs live in moe_remote_hook.h, keeping this model card-free.
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
            await_cleanup(true);
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
            if (!quiescing) {
                await_cleanup(false);
            }
            c_.state = uint32_t(PlacementState::ACTIVE);
            c_.pending_epoch = 0; c_.pending_digest = 0; c_.pending_slot = UINT32_MAX;
            c_.deadline_ns = 0; reset_masks(); quiesced_ = false;
            // Quiescence faults can imply partial output: never re-open admission.
            if (!quiescing) c_.fault_code = 0; // retain actor as last-fault diagnostic
        } else failed_closed_ = true;
        emit();
        return !committed;
    }
    // Cleanup failure is terminal for placement, NOT serving. The inactive
    // generation remains protected because can_propose() never passes again.
    // Do not set control.fault_code: the live admission ledger must keep serving.
    bool stop_placement_after_cleanup(FaultCode code,uint32_t actor=0) {
        if (code==FaultCode::NONE || c_.state!=uint32_t(PlacementState::ACTIVE) ||
            c_.journal_index>1 || failed_closed_) return false;
        auto& j=journal_.entry[c_.journal_index];
        if (!j.transaction_id) j.transaction_id=c_.transaction_id;
        if (!j.fault) j.fault=uint32_t(code);
        j.cleanup_fault=uint32_t(code);
        j.cleanup_actor=actor;
        j.finish_ns=clock_.now_ns();
        c_.fault_actor=actor;
        failed_closed_=true;
        return emit();
    }
    bool stop_placement_after_commit(FaultCode code) {
        return cleanup_committed_ && stop_placement_after_cleanup(code);
    }
    // Helper zero calls this on every ACTIVE control-plane tick. Only exact
    // current-transaction CLEANED faults stop placement immediately; missing,
    // duplicate, stale and mismatched ACKs never constitute cleanup proof.
    bool poll_cleanup() {
        if (failed_closed_) return false;
        if (!awaiting_cleanup_ || c_.state!=uint32_t(PlacementState::ACTIVE))
            return true;
        auto check_fault = [&](uint32_t transport_actor,uint32_t actor,
                               AckPhase phase) {
            const auto a=transport_.read_ack(transport_actor);
            if (a.phase==uint32_t(phase) && a.fault_code &&
                a.transaction_id==awaiting_cleanup_ &&
                a.epoch==c_.active_epoch && a.digest==c_.active_digest &&
                a.generation_slot==c_.active_slot) {
                (void)stop_placement_after_cleanup(FaultCode::DEVICE,actor);
                return false;
            }
            return true;
        };
        if (!cleanup_committed_)
            for (uint32_t i=0;i<kMaxEngineRanks;++i)
                if ((c_.expected_ranks & (1u<<i)) &&
                    !check_fault(kMaxHelpers+i,actor_id(ActorKind::ENGINE_RANK,uint16_t(i)),
                                 AckPhase::CLEANED)) return false;
        for (uint32_t i=0;i<kMaxHelpers;++i)
            if ((c_.expected_helpers & (1u<<i)) &&
                !check_fault(i,actor_id(ActorKind::HELPER,uint16_t(i)),
                             AckPhase::CLEANED)) return false;
        if (previous_closed()) return true;
        if (clock_.now_ns() >= cleanup_deadline_ns_) {
            (void)stop_placement_after_cleanup(FaultCode::CLEANUP_TIMEOUT,
                                               missing_cleanup_actor());
            return false;
        }
        return true;
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
// The diagnostic is a real two-slot epoch transition with the SAME owner
// bytes/digest. No new owner placement or helper weight/table migration is
// inferred: moved-owner transactions remain outside slice D.
inline bool same_placement_cycle(const PlacementControl& c) {
    if (!c.transaction_id || !c.active_epoch || !c.active_digest ||
        c.active_epoch > UINT32_MAX || c.pending_slot > 1 ||
        c.active_slot > 1 || c.pending_digest != c.active_digest) return false;
    if (c.state < uint32_t(PlacementState::COMMIT))
        return c.pending_epoch == c.active_epoch + 1 &&
               c.pending_slot == 1u - c.active_slot;
    return c.pending_epoch == c.active_epoch &&
           c.pending_slot == c.active_slot;
}

/* Stage the unchanged canonical map into a distinct generation, before
 * PROPOSED publication. Previously retired generations may be overwritten
 * only when the later ledger guard also permits upload. Active bytes and
 * header are never modified here. */
inline bool stage_same_map_generation(void* map) {
    enum Reason : uint32_t { SNAPSHOT, CONTROL, ACTIVE_HEADER, INACTIVE_HEADER,
                             LAYOUT, ACTIVE_DIGEST, COPIED_DIGEST };
    auto fail = [](Reason reason, const char* why) {
        static std::atomic<uint32_t> seen{0};
        const uint32_t bit=1u << uint32_t(reason);
        if (!(seen.fetch_or(bit, std::memory_order_relaxed) & bit))
            std::fprintf(stderr, "moe-remote: stage_same_map_generation refused: %s\n", why);
        return false;
    };
    PlacementControl c{};
    if (!map || !placement_snapshot(map, &c))
        return fail(SNAPSHOT, "missing map or control seqlock snapshot");
    if (c.state != uint32_t(PlacementState::ACTIVE) || c.fault_code ||
        !c.active_epoch || c.active_epoch >= UINT32_MAX ||
        !c.active_digest || c.active_slot > 1)
        return fail(CONTROL, "invalid ACTIVE control epoch/digest/slot/state");
    const Header* h = static_cast<const Header*>(map);
    GenerationHeader* active = generation_header(map,c.active_slot);
    GenerationHeader* inactive = generation_header(map,1u-c.active_slot);
    if (!active || active->epoch != c.active_epoch ||
        active->digest != c.active_digest ||
        active->state != uint32_t(GenerationState::ACTIVE) ||
        !active->ready_helpers || active->ready_helpers != c.expected_helpers)
        return fail(ACTIVE_HEADER, "active generation epoch/digest/state/ready_helpers mismatch");
    if (!inactive ||
        (inactive->state != uint32_t(GenerationState::EMPTY) &&
         inactive->state != uint32_t(GenerationState::RETIRED) &&
         inactive->state != uint32_t(GenerationState::STAGING) &&
         inactive->state != uint32_t(GenerationState::READY)))
        return fail(INACTIVE_HEADER, "inactive generation state invalid");
    if (!h->owner_map_bytes ||
        active->map_bytes != h->owner_map_bytes ||
        inactive->map_bytes != h->owner_map_bytes)
        return fail(LAYOUT, "generation owner map length invalid");
    if (generation_digest(map,c.active_slot) != c.active_digest)
        return fail(ACTIVE_DIGEST, "active owner bytes differ from published digest");
    std::memcpy(generation_map(map,1u-c.active_slot),generation_map(map,c.active_slot),
                h->owner_map_bytes);
    inactive->epoch = c.active_epoch+1;
    inactive->digest = c.active_digest;
    inactive->ready_helpers = inactive->ready_ranks = 0;
    inactive->state = uint32_t(GenerationState::STAGING);
    if (generation_digest(map,1u-c.active_slot)!=c.active_digest)
        return fail(COPIED_DIGEST, "staged owner bytes differ from published digest");
    return true;
}

// Structural predicate shared by the production rank and helper participants.
// Unlike the diagnostic-only same_map_cycle, it permits a *different* map
// through the existing controller proposal and verifies both complete digests.
inline bool placement_epoch_valid(const PlacementControl& c) {
    if (!c.transaction_id || !c.active_epoch || !c.active_digest ||
        !c.pending_digest || c.active_epoch > UINT32_MAX ||
        c.pending_slot > 1 || c.active_slot > 1) return false;
    if (c.state < uint32_t(PlacementState::COMMIT))
        return c.pending_epoch == c.active_epoch+1 &&
               c.pending_slot == 1u-c.active_slot;
    return c.pending_epoch == c.active_epoch &&
           c.pending_slot == c.active_slot;
}
inline bool placement_candidate_valid(const void* map,const PlacementControl& c) {
    if (!map || !placement_epoch_valid(c) || c.journal_index > 1) return false;
    const auto& j=placement_journal(map)->entry[c.journal_index];
    if (j.transaction_id!=c.transaction_id || j.old_slot>1 ||
        j.new_slot!=1u-j.old_slot ||
        !j.before_epoch || j.before_epoch>=UINT32_MAX ||
        j.after_epoch!=j.before_epoch+1 ||
        j.before_digest==0 || j.candidate_digest==0 ||
        j.candidate_digest!=c.pending_digest) return false;
    const bool committed=c.state>=uint32_t(PlacementState::COMMIT);
    if (committed ?
        (c.active_slot!=j.new_slot || c.active_epoch!=j.after_epoch ||
         c.active_digest!=j.candidate_digest || c.pending_slot!=j.new_slot) :
        (c.active_slot!=j.old_slot || c.active_epoch!=j.before_epoch ||
         c.active_digest!=j.before_digest || c.pending_slot!=j.new_slot ||
         c.pending_epoch!=j.after_epoch)) return false;
    const auto* h=static_cast<const Header*>(map);
    const auto* oldg=generation_header(map,j.old_slot);
    const auto* newg=generation_header(map,j.new_slot);
    if (!oldg || !newg || !h->owner_map_bytes ||
        oldg->epoch!=j.before_epoch || oldg->digest!=j.before_digest ||
        oldg->state!=uint32_t(GenerationState::ACTIVE) ||
        newg->epoch!=j.after_epoch || newg->digest!=j.candidate_digest ||
        (newg->state!=uint32_t(GenerationState::STAGING) &&
         newg->state!=uint32_t(GenerationState::READY) &&
         newg->state!=uint32_t(GenerationState::ACTIVE)) ||
        (!committed && newg->state==uint32_t(GenerationState::ACTIVE)) ||
        (c.state==uint32_t(PlacementState::RETIRING) &&
         newg->state!=uint32_t(GenerationState::ACTIVE)) ||
        oldg->map_bytes!=h->owner_map_bytes ||
        newg->map_bytes!=h->owner_map_bytes ||
        generation_digest(map,j.old_slot)!=j.before_digest ||
        generation_digest(map,j.new_slot)!=j.candidate_digest)
        return false;
    const auto* candidate=generation_map(map,j.new_slot);
    const auto* original=generation_map(map,j.old_slot);
    for (uint32_t i=0;i<h->owner_map_bytes;++i) {
        if ((original[i]!=kOwnerLocal && original[i]>=h->helper_count) ||
            (candidate[i]!=kOwnerLocal && candidate[i]>=h->helper_count) ||
            // Bootstrap determines which experts' weights exist on the engine.
            // Placement can exchange helpers, never engine/local residency.
            ((original[i]==kOwnerLocal)!=(candidate[i]==kOwnerLocal)))
            return false;
    }
    return true;
}

/* Shared no-move control-map proof, used by both helper CPU participant and
 * engine coordinator. It also checks the transaction's journal identities,
 * so a stale same-digest generation cannot masquerade as this transition. */
inline bool same_map_cycle(const void* map,const PlacementControl& c) {
    if (!map || !same_placement_cycle(c) || c.journal_index > 1) return false;
    const auto& j=placement_journal(map)->entry[c.journal_index];
    const uint32_t old_slot=j.old_slot, new_slot=j.new_slot;
    if (j.transaction_id != c.transaction_id || old_slot > 1 ||
        new_slot != 1u-old_slot || j.before_digest != j.candidate_digest ||
        j.before_digest != c.active_digest || j.before_epoch == 0 ||
        j.before_epoch >= UINT32_MAX || j.after_epoch != j.before_epoch+1 ||
        j.new_slot != c.pending_slot || j.after_epoch != c.pending_epoch ||
        (c.state < uint32_t(PlacementState::COMMIT) ?
             (c.active_slot != old_slot || c.active_epoch != j.before_epoch) :
             (c.active_slot != new_slot || c.active_epoch != j.after_epoch)))
        return false;
    const auto* h=static_cast<const Header*>(map);
    const auto* oldg=generation_header(map,old_slot);
    const auto* newg=generation_header(map,new_slot);
    if (!oldg || !newg || !h->owner_map_bytes ||
        oldg->epoch != j.before_epoch || newg->epoch != j.after_epoch ||
        oldg->digest != j.before_digest || newg->digest != j.candidate_digest ||
        oldg->state != uint32_t(GenerationState::ACTIVE) ||
        (newg->state != uint32_t(GenerationState::STAGING) &&
         newg->state != uint32_t(GenerationState::READY) &&
         newg->state != uint32_t(GenerationState::ACTIVE)) ||
        (c.state < uint32_t(PlacementState::COMMIT) &&
         newg->state == uint32_t(GenerationState::ACTIVE)) ||
        (c.state == uint32_t(PlacementState::RETIRING) &&
         newg->state != uint32_t(GenerationState::ACTIVE)) ||
        oldg->map_bytes != h->owner_map_bytes ||
        newg->map_bytes != h->owner_map_bytes ||
        generation_digest(map,old_slot)!=c.active_digest ||
        generation_digest(map,new_slot)!=c.pending_digest)
        return false;
    return std::memcmp(generation_map(map,old_slot),
                       generation_map(map,new_slot),h->owner_map_bytes)==0;
}

/* Production engine-rank ACK predicates, also run in card-free fake-device
 * tests. An ACK is not a claim about attempted uploads: every rank's event
 * must be complete with this exact immutable epoch/digest/slot. */
/* The engine-rank reservation participant is shared with fake-device tests.
 * Registered means its serving device is live; active_event_ready proves its
 * current generation finished H2D before the new one can be reserved. */
inline bool rank_reservation_ready(const void* map,const PlacementControl& c,
                                   uint32_t rank,uint32_t actual_rank_mask,
                                   bool registered,bool active_event_ready) {
    if (!map || c.state != uint32_t(PlacementState::RESERVING) ||
        c.fault_code || rank >= kMaxEngineRanks || !actual_rank_mask ||
        (actual_rank_mask & (actual_rank_mask + 1u)) ||
        c.expected_ranks != actual_rank_mask ||
        !(actual_rank_mask & (1u<<rank)) || !registered ||
        !active_event_ready || !placement_candidate_valid(map,c)) return false;
    const auto* active = generation_header(map,c.active_slot);
    const auto* candidate = generation_header(map,c.pending_slot);
    return active && candidate &&
           active->state == uint32_t(GenerationState::ACTIVE) &&
           active->epoch == c.active_epoch && active->digest == c.active_digest &&
           candidate->state == uint32_t(GenerationState::STAGING) &&
           candidate->epoch == c.pending_epoch &&
           candidate->digest == c.pending_digest;
}

inline bool rank_upload_prepared(const PlacementControl& c,uint32_t rank,
                                bool uploaded,bool event_ready,
                                uint64_t epoch,uint64_t digest,uint32_t slot) {
    return c.state == uint32_t(PlacementState::COPYING) &&
           rank < kMaxEngineRanks && (c.expected_ranks & (1u<<rank)) &&
           placement_epoch_valid(c) && uploaded && event_ready &&
           epoch == c.pending_epoch && digest == c.pending_digest &&
           slot == c.pending_slot;
}

/* RETIRING requires the *same* closed-epoch consumption and helper-drain
 * proof used in QUIESCING, plus an unreferenced old generation. Until
 * admission reopens on N+1, the Part-1 guard still prevents uploading the
 * ledger's active (old) slot. This gate does not weaken that guard. */
inline bool rank_retirement_proved(const PlacementControl& c,
                                   uint32_t old_epoch,uint32_t old_slot,
                                   bool all_consumed,bool drain_verified,
                                   bool helper_quiesced,bool no_old_refs,
                                   bool ready_on_all_ranks,bool committed_on_all_ranks) {
    return c.state == uint32_t(PlacementState::RETIRING) &&
           placement_epoch_valid(c) && old_epoch && old_epoch < UINT32_MAX &&
           old_epoch+1 == c.active_epoch && old_slot < 2 &&
           old_slot == 1u-c.active_slot &&
           all_consumed && drain_verified && helper_quiesced &&
           no_old_refs && ready_on_all_ranks && committed_on_all_ranks;
}

/* A current-phase fault is an explicit actor ACK, not a sticky helper fault
 * before COMMIT. The owner controller can abort PROPOSED/RESERVING/COPYING
 * forward, retain ACTIVE(N), then retry with a new transaction. */
inline void publish_fault_ack(void* map,uint32_t actor,const PlacementControl& c,
                              AckPhase phase,FaultCode code) {
    auto* a=placement_ack(map,actor);
    if (!a || code == FaultCode::NONE) return;
    __atomic_store_n(&a->phase,uint32_t(AckPhase::NONE),__ATOMIC_SEQ_CST);
    __atomic_store_n(&a->transaction_id,c.transaction_id,__ATOMIC_RELAXED);
    __atomic_store_n(&a->epoch,c.pending_epoch,__ATOMIC_RELAXED);
    __atomic_store_n(&a->digest,c.pending_digest,__ATOMIC_RELAXED);
    __atomic_store_n(&a->generation_slot,c.pending_slot,__ATOMIC_RELAXED);
    __atomic_store_n(&a->fault_code,uint32_t(code),__ATOMIC_RELAXED);
    __atomic_store_n(&a->phase,uint32_t(phase),__ATOMIC_RELEASE);
    notify(static_cast<Header*>(map));
}

// One writer per actor. Phase is the release publication word; invalidate it
// before updating identity, and acquire/recheck it when collecting an ACK.
inline PlacementAck read_shared_ack(const void* map, uint32_t actor) {
    const auto* a = placement_ack(map, actor);
    PlacementAck out{};
    if (!a) return out;
    const uint32_t phase = __atomic_load_n(&a->phase, __ATOMIC_ACQUIRE);
    if (!phase) return out;
    out.transaction_id = __atomic_load_n(&a->transaction_id, __ATOMIC_ACQUIRE);
    out.epoch = __atomic_load_n(&a->epoch, __ATOMIC_RELAXED);
    out.digest = __atomic_load_n(&a->digest, __ATOMIC_RELAXED);
    out.drained_step = __atomic_load_n(&a->drained_step, __ATOMIC_RELAXED);
    out.generation_slot = __atomic_load_n(&a->generation_slot, __ATOMIC_RELAXED);
    out.fault_code = __atomic_load_n(&a->fault_code, __ATOMIC_RELAXED);
    out.reserved[0] = __atomic_load_n(&a->reserved[0], __ATOMIC_RELAXED);
    out.reserved[1] = __atomic_load_n(&a->reserved[1], __ATOMIC_RELAXED);
    if (__atomic_load_n(&a->phase, __ATOMIC_ACQUIRE) != phase ||
        __atomic_load_n(&a->transaction_id, __ATOMIC_ACQUIRE) != out.transaction_id)
        return PlacementAck{};
    out.phase = phase;
    return out;
}
inline void publish_ack(void* map, uint32_t actor, const PlacementControl& c,
                        AckPhase phase, uint64_t cutoff = 0,
                        uint64_t copy_bytes = 0, uint64_t copy_ns = 0) {
    auto* a = placement_ack(map, actor);
    if (!a) return;
    const auto previous = read_shared_ack(map, actor);
    if (previous.transaction_id == c.transaction_id && previous.phase == uint32_t(phase))
        return;
    __atomic_store_n(&a->phase, uint32_t(AckPhase::NONE), __ATOMIC_SEQ_CST);
    __atomic_store_n(&a->transaction_id, c.transaction_id, __ATOMIC_RELAXED);
    __atomic_store_n(&a->epoch, c.pending_epoch, __ATOMIC_RELAXED);
    __atomic_store_n(&a->digest, c.pending_digest, __ATOMIC_RELAXED);
    __atomic_store_n(&a->drained_step, cutoff, __ATOMIC_RELAXED);
    __atomic_store_n(&a->generation_slot, c.pending_slot, __ATOMIC_RELAXED);
    __atomic_store_n(&a->fault_code, uint32_t(0), __ATOMIC_RELAXED);
    // The two pre-existing reserved words carry per-participant copy evidence
    // for the current transaction. READY publishes exact bytes and elapsed
    // nanoseconds; later phases preserve both until CLEANED.
    if (phase == AckPhase::READY || phase == AckPhase::RESERVED ||
        previous.transaction_id != c.transaction_id) {
        __atomic_store_n(&a->reserved[0], (phase == AckPhase::READY || phase == AckPhase::RESERVED) ? copy_bytes : 0, __ATOMIC_RELAXED);
        __atomic_store_n(&a->reserved[1], (phase == AckPhase::READY || phase == AckPhase::RESERVED) ? copy_ns : 0, __ATOMIC_RELAXED);
    }
    __atomic_store_n(&a->phase, uint32_t(phase), __ATOMIC_RELEASE);
    notify(static_cast<Header*>(map));
}

// Cleanup ACKs use ACTIVE's epoch/digest/slot, not the cleared pending
// identity. One writer per participant and release publication as above.
inline void publish_cleanup_fault_ack(void* map,uint32_t actor,
                                      const PlacementControl& c,FaultCode code) {
    if (!map || code==FaultCode::NONE || !c.transaction_id ||
        c.state!=uint32_t(PlacementState::ACTIVE) || c.fault_code) return;
    auto* a=placement_ack(map,actor);
    if (!a) return;
    __atomic_store_n(&a->phase,uint32_t(AckPhase::NONE),__ATOMIC_SEQ_CST);
    __atomic_store_n(&a->transaction_id,c.transaction_id,__ATOMIC_RELAXED);
    __atomic_store_n(&a->epoch,c.active_epoch,__ATOMIC_RELAXED);
    __atomic_store_n(&a->digest,c.active_digest,__ATOMIC_RELAXED);
    __atomic_store_n(&a->generation_slot,c.active_slot,__ATOMIC_RELAXED);
    __atomic_store_n(&a->fault_code,uint32_t(code),__ATOMIC_RELAXED);
    __atomic_store_n(&a->phase,uint32_t(AckPhase::CLEANED),__ATOMIC_RELEASE);
    notify(static_cast<Header*>(map));
}
inline void publish_cleanup_ack(void* map,uint32_t actor,const PlacementControl& c,
                                uint64_t lane_unavailable_ns=0) {
    if (!map || !c.transaction_id ||
        c.state!=uint32_t(PlacementState::ACTIVE) || c.fault_code) return;
    auto* a=placement_ack(map,actor);
    if (!a) return;
    const auto previous=read_shared_ack(map,actor);
    if (previous.phase==uint32_t(AckPhase::CLEANED) &&
        previous.transaction_id==c.transaction_id &&
        previous.epoch==c.active_epoch && previous.digest==c.active_digest &&
        previous.generation_slot==c.active_slot) return;
    __atomic_store_n(&a->phase,uint32_t(AckPhase::NONE),__ATOMIC_SEQ_CST);
    __atomic_store_n(&a->transaction_id,c.transaction_id,__ATOMIC_RELAXED);
    __atomic_store_n(&a->epoch,c.active_epoch,__ATOMIC_RELAXED);
    __atomic_store_n(&a->digest,c.active_digest,__ATOMIC_RELAXED);
    __atomic_store_n(&a->generation_slot,c.active_slot,__ATOMIC_RELAXED);
    __atomic_store_n(&a->fault_code,0u,__ATOMIC_RELAXED);
    // Complete calibration-only downtime evidence, from quiesce ACK to reopen.
    __atomic_store_n(&a->reserved[0],lane_unavailable_ns,__ATOMIC_RELAXED);
    __atomic_store_n(&a->reserved[1],1ull,__ATOMIC_RELAXED);
    __atomic_store_n(&a->phase,uint32_t(AckPhase::CLEANED),__ATOMIC_RELEASE);
    notify(static_cast<Header*>(map));
}

/* Both the real rank thread and card-free fake devices invoke this exact
 * participant: validation and release publication cannot drift apart. */
inline bool service_rank_reservation(void* map,const PlacementControl& c,
                                     uint32_t rank,uint32_t actual_rank_mask,
                                     bool registered,bool active_event_ready) {
    if (!rank_reservation_ready(map,c,rank,actual_rank_mask,registered,active_event_ready))
        return false;
    publish_ack(map,kMaxHelpers+rank,c,AckPhase::RESERVED);
    return true;
}

/* The coordinator's one global ledger publishes a transaction-specific ACK
 * AFTER admission reopens. The rank-0 retired ACK slot is no longer collected
 * by the controller after ACTIVE, so it carries this completion fence. */
inline void publish_reopen_ack(void* map,const PlacementControl& c) {
    if (!map || c.state != uint32_t(PlacementState::ACTIVE) ||
        !c.transaction_id || !c.active_epoch || c.active_slot > 1 ||
        !c.active_digest) return;
    auto* a = placement_ack(map,kMaxHelpers);
    if (!a) return;
    __atomic_store_n(&a->phase,uint32_t(AckPhase::NONE),__ATOMIC_SEQ_CST);
    __atomic_store_n(&a->transaction_id,c.transaction_id,__ATOMIC_RELAXED);
    __atomic_store_n(&a->epoch,c.active_epoch,__ATOMIC_RELAXED);
    __atomic_store_n(&a->digest,c.active_digest,__ATOMIC_RELAXED);
    __atomic_store_n(&a->generation_slot,c.active_slot,__ATOMIC_RELAXED);
    __atomic_store_n(&a->fault_code,0u,__ATOMIC_RELAXED);
    __atomic_store_n(&a->phase,uint32_t(AckPhase::REOPENED),__ATOMIC_RELEASE);
    notify(static_cast<Header*>(map));
}
inline bool reopened_exactly(const void* map,const PlacementControl& c,
                             uint64_t transaction) {
    if (!map || !transaction || c.state != uint32_t(PlacementState::ACTIVE) ||
        c.transaction_id != transaction || c.fault_code) return false;
    const PlacementAck a = read_shared_ack(map,kMaxHelpers);
    return a.phase == uint32_t(AckPhase::REOPENED) &&
           a.transaction_id == transaction && !a.fault_code &&
           a.epoch == c.active_epoch && a.digest == c.active_digest &&
           a.generation_slot == c.active_slot;
}

/* One absolute monotonic deadline covers ALL lane parks, not one duration
 * per lane. Timeout does not authorize reopening. */
inline WaitResult wait_parks_until(Header* h,const timespec* deadline) {
    if (!h) return WaitResult::Error;
    for (uint32_t lane=0;lane<kLanes;++lane) {
        const auto r=wait_for(h,deadline,[&] {
            return __atomic_load_n(&h->lane[lane].park,__ATOMIC_SEQ_CST) != kParkDraining;
        });
        if (r != WaitResult::Ready) return r;
    }
    return WaitResult::Ready;
}
/* Production and card-free test share the same sticky-fault publication.
 * A park timeout cannot be misread as a completed drain/reopen. */
inline WaitResult wait_parks_checked(Header* h,const timespec* deadline) {
    const auto result=wait_parks_until(h,deadline);
    if (result!=WaitResult::Ready && h) raise_fault(h);
    return result;
}
class SharedTransport : public Transport {
    void* map_;
public:
    explicit SharedTransport(void* map) : map_(map) {}
    /* Exactly once, after all helpers have finished runtime filtering and
     * before any engine rank uploads the initial bitmap or admits a parent.
     * Helper 0 alone writes the control's seqlock (the future controller's
     * only writer). Include the generation header update inside odd/even. */
    bool seal_bootstrap() {
        if (!map_) {
            std::fprintf(stderr, "r3_moe_worker: seal_bootstrap refused: missing mapping\n");
            return false;
        }
        auto* h = static_cast<Header*>(map_);
        PlacementControl c{};
        auto* g = generation_header(map_, 0);
        /* Give the failing predicate AND both sides of its comparison.
         * Startup is one-shot; this logging is never on a serving path. */
        auto reject = [&](const char* reason) {
            std::fprintf(stderr,
                "r3_moe_worker: seal_bootstrap refused: %s "
                "(seq=%u raw_seq=%u state=%u slot=%u epoch=%llu fault=%u "
                "helpers=%u expected=0x%x filtered=0x%x "
                "gen_epoch=%llu gen_state=%u gen_ready=0x%x "
                "gen_bytes=%u owner_bytes=%u gen_digest=0x%llx "
                "control_digest=0x%llx actual_digest=0x%llx)\n",
                reason, c.seq, load(&placement_control(map_)->seq),
                c.state, c.active_slot, (unsigned long long)c.active_epoch,
                c.fault_code, load(&h->helper_count), c.expected_helpers,
                load(&h->bootstrap_filtered_helpers),
                (unsigned long long)(g ? g->epoch : 0u),
                g ? g->state : 0u,
                g ? load(&g->ready_helpers) : 0u,
                g ? g->map_bytes : 0u, h->owner_map_bytes,
                (unsigned long long)(g ? g->digest : 0u),
                (unsigned long long)c.active_digest,
                (unsigned long long)generation_digest(map_, 0));
            return false;
        };
        if (!placement_snapshot(map_, &c)) return reject("placement_snapshot");
        if (c.seq != 0) return reject("control.seq != 0");
        if (c.state != uint32_t(PlacementState::ACTIVE)) return reject("control.state != ACTIVE");
        if (c.active_slot != 0) return reject("control.active_slot != 0");
        if (c.active_epoch != 1) return reject("control.active_epoch != 1");
        if (c.fault_code) return reject("control.fault_code != 0");
        if (!h->helper_count || h->helper_count > kMaxHelpers) return reject("header.helper_count");
        if (c.expected_helpers != ((1u << h->helper_count) - 1u))
            return reject("control.expected_helpers");
        if (__atomic_load_n(&h->bootstrap_filtered_helpers, __ATOMIC_ACQUIRE) != c.expected_helpers)
            return reject("header.bootstrap_filtered_helpers");
        if (!g) return reject("generation[0] missing");
        if (g->epoch != 1) return reject("generation[0].epoch != 1");
        if (g->state != uint32_t(GenerationState::ACTIVE)) return reject("generation[0].state");
        if (load(&g->ready_helpers)) return reject("generation[0].ready_helpers already set");
        if (g->map_bytes != h->owner_map_bytes) return reject("generation[0].map_bytes");
        if (g->digest != c.active_digest) return reject("generation[0].digest != control.active_digest");
        const uint64_t digest = generation_digest(map_, 0);
        if (!digest) return reject("generation_digest == 0");
        c.active_digest = digest;
        c.seq = 2;
        auto* target = placement_control(map_);
        __atomic_store_n(&target->seq, 1u, __ATOMIC_SEQ_CST);
        g->digest = digest;
        __atomic_store_n(&g->ready_helpers, c.expected_helpers, __ATOMIC_RELEASE);
        std::memcpy(reinterpret_cast<uint8_t*>(target) + sizeof(c.seq),
                    reinterpret_cast<const uint8_t*>(&c) + sizeof(c.seq), sizeof(c) - sizeof(c.seq));
        __atomic_store_n(&target->seq, c.seq, __ATOMIC_RELEASE);
        notify(h);
        return true;
    }
    PlacementAck read_ack(uint32_t actor) const override { return read_shared_ack(map_, actor); }
    void publish(const PlacementControl& c, const PlacementJournal& j) override {
        auto* target = placement_control(map_);
        __atomic_store_n(&target->seq, c.seq - 1u, __ATOMIC_SEQ_CST);
        std::memcpy(reinterpret_cast<uint8_t*>(target) + sizeof(c.seq),
                    reinterpret_cast<const uint8_t*>(&c) + sizeof(c.seq), sizeof(c) - sizeof(c.seq));
        *placement_journal(map_) = j;
        __atomic_store_n(&target->seq, c.seq, __ATOMIC_RELEASE);
        notify(static_cast<Header*>(map_));
    }
};

enum class HelperCopy : uint8_t { WAITING, READY, FAILED };
// Called by the existing epoch driver in its ordinary controller phases.
// Implemented by the HIP-independent slot participant, backed by Runner.
struct HelperMigration {
    virtual ~HelperMigration() = default;
    virtual FaultCode reserve(const void*, const PlacementControl&) = 0;
    virtual HelperCopy copy() = 0;
    virtual bool commit(const PlacementControl&) = 0;
    virtual bool reclaim(const void*, const PlacementControl&) = 0;
    virtual bool abort() = 0;
    virtual uint64_t transaction() const = 0;
    // Bytes actually uploaded by this participant in the currently reserved transaction.
    virtual uint64_t copy_bytes() const { return 0; }
};

// Helper-side periodic driver. The caller polls it beside the serving lanes;
// only helper zero publishes PlacementControl. No environment settings here.
class EpochCycle {
    void* map_;
    uint32_t helper_;
    const Clock& clock_;
    std::chrono::nanoseconds period_;
    uint64_t next_ns_ = 0;
    uint64_t awaiting_reopen_ = 0;
    uint64_t reopen_deadline_ns_ = 0;
    Deadlines deadlines_{};
    uint64_t completed_ = 0;   // full close-drain-recommit-reopen cycles, for the run's evidence
    SharedTransport transport_;
    std::unique_ptr<Controller> controller_;
    bool failed_ = false;
    bool placement_halted_ = false; // terminal placement diagnostic, not a helper failure
    FaultCode placement_fault_ = FaultCode::NONE;
    HelperMigration* migration_ = nullptr;
    uint64_t copy_start_transaction_ = 0, copy_start_ns_ = 0;
    std::function<bool(uint64_t*,uint64_t*)> lane_service_probe_;
    uint64_t service_baseline_ns_=0,service_baseline_total_ns_=0,service_baseline_rows_=0;
    uint64_t quiesced_at_ns_=0,quiesced_transaction_=0;
    std::function<bool(void*,Controller&)> forced_proposal_;
    bool schedule() {
        const uint64_t now = clock_.now_ns();
        const uint64_t period = uint64_t(period_.count());
        if (period > UINT64_MAX - now) return false;
        next_ns_ = now + period;
        return true;
    }
public:
    EpochCycle(void* map, uint32_t helper, const Clock& clock,
               std::chrono::nanoseconds period, Deadlines deadlines = {},
               HelperMigration* migration = nullptr)
        : map_(map), helper_(helper), clock_(clock), period_(period),
          deadlines_(deadlines), transport_(map), migration_(migration) {
        failed_ = !map || helper >= static_cast<Header*>(map)->helper_count || period.count() < 0;
    }
    bool enabled() const { return period_.count() != 0 || migration_ != nullptr; }
    // The tests inject their candidate through Controller::propose, not a
    // second path or a synthetic forced-move policy.
    Controller* proposal_controller() { return controller_.get(); }
    void set_lane_service_probe(std::function<bool(uint64_t*,uint64_t*)> probe) {
        lane_service_probe_=std::move(probe);
    }
    void set_forced_proposal(std::function<bool(void*,Controller&)> proposal) {
        forced_proposal_=std::move(proposal);
    }
    uint64_t completed() const { return completed_; }
    bool placement_halted() const { return placement_halted_; }
    FaultCode placement_fault() const { return placement_fault_; }
    bool tick() {
        if (failed_) return false;
        if (!enabled()) return true; // no diagnostic cycle or migration participant
        auto* h = static_cast<Header*>(map_);
        if (load(&h->helper_fault)) return false;
        if (placement_halted_) return true; // keep serving, but still respect real helper faults
        PlacementControl c{};
        if (!placement_snapshot(map_, &c)) return true;
        // Wait for the engine to publish its actual participating rank mask.
        if (!c.expected_ranks) return true;
        // Reuse the helper's existing sampled lane timings on the CPU tick.
        // Baseline predates the move and requires no extra HIP event or thread.
        if (lane_service_probe_ && !service_baseline_ns_) {
            uint64_t total=0,rows=0;
            if (lane_service_probe_(&total,&rows)) {
                service_baseline_total_ns_=total;
                service_baseline_rows_=rows;
                service_baseline_ns_=clock_.now_ns();
            }
        }
        if (helper_ == 0 && !controller_) {
            if (c.state != uint32_t(PlacementState::ACTIVE) || c.fault_code ||
                c.expected_ranks >= (1u << kMaxEngineRanks) ||
                (c.expected_ranks & (c.expected_ranks + 1u))) return false;
            uint32_t ranks = 0;
            for (uint32_t mask = c.expected_ranks; mask; mask >>= 1) ++ranks;
            controller_ = std::make_unique<Controller>(transport_, clock_, c.active_digest,
                h->helper_count, ranks, deadlines_,
                Bootstrap{c.active_epoch, c.active_slot, c.seq, c.transaction_id});
            if (!schedule()) return false;
            c = controller_->control();
        }
        if (c.fault_code) return false;
        const auto phase = PlacementState(c.state);
        if (phase != PlacementState::ACTIVE && !placement_candidate_valid(map_,c)) {
            /* Before quiescence a bad candidate must abort the transaction,
             * not stop serving epoch N. Controller::fail publishes ACTIVE(N)
             * unchanged, with a fault recorded in the journal. */
            if (phase < PlacementState::QUIESCING) {
                if (helper_ == 0) {
                    (void)controller_->fail(FaultCode::CHECKSUM);
                    return schedule();
                }
                return true;
            }
            return false; /* after the admission gate closes, fail closed */
        }
        if (phase == PlacementState::RESERVING) {
            const FaultCode issue=migration_ ? migration_->reserve(map_,c) :
                (same_map_cycle(map_,c) ? FaultCode::NONE : FaultCode::CAPACITY);
            if (issue==FaultCode::NONE) {
                uint64_t serviced_ns=0,wall_ns=0,total_ns=0,total_rows=0,rows=0;
                const uint64_t now=clock_.now_ns();
                if (lane_service_probe_ && service_baseline_ns_ && now>service_baseline_ns_ &&
                    lane_service_probe_(&total_ns,&total_rows) &&
                    total_ns>=service_baseline_total_ns_ && total_rows>=service_baseline_rows_) {
                    serviced_ns=total_ns-service_baseline_total_ns_;
                    rows=total_rows-service_baseline_rows_;
                    wall_ns=now-service_baseline_ns_;
                }
                // RESERVED uses drained_step for the sampled routed-row count; later phases
                // have their own cutoff. Both deltas come from one atomic service snapshot.
                publish_ack(map_,helper_,c,AckPhase::RESERVED,rows,serviced_ns,wall_ns);
            }
            else publish_fault_ack(map_,helper_,c,AckPhase::RESERVED,issue);
        }
        if (phase == PlacementState::COPYING) {
            if (copy_start_transaction_ != c.transaction_id) {
                copy_start_transaction_ = c.transaction_id;
                copy_start_ns_ = clock_.now_ns();
            }
            const HelperCopy result=migration_ ? migration_->copy() :
                (same_map_cycle(map_,c) ? HelperCopy::READY : HelperCopy::FAILED);
            if (result==HelperCopy::READY) {
                const uint64_t end=clock_.now_ns();
                const uint64_t elapsed=end>copy_start_ns_ ? end-copy_start_ns_ : 0;
                publish_ack(map_,helper_,c,AckPhase::READY,0,
                            migration_ ? migration_->copy_bytes() : 0,elapsed);
            } else if (result==HelperCopy::FAILED)
                publish_fault_ack(map_,helper_,c,AckPhase::READY,FaultCode::UPLOAD);
        }
        if (phase == PlacementState::COMMIT && migration_ &&
            !migration_->commit(c)) return false; // cannot ACK COMMITTED without actual slots
        if (phase == PlacementState::QUIESCING || phase == PlacementState::COMMIT ||
            phase == PlacementState::RETIRING) {
            // Slice C drain identities are published once on the lane CPU
            // at the closure ACK, not as per-request helper GPU stores.
            uint64_t closure = 0;
            bool drained = true;
            for (uint32_t lane = 0; lane < kLanes; ++lane) {
                const uint32_t slot=lane % kMaxSlots;
                const bool served=h->helper_count > 1 ? slot == helper_ :
                    (c.expected_ranks & (1u<<slot)) != 0;
                if (!served ||
                    (lane_is_small(lane) && lane_class(lane) >= h->small_class_count)) continue;
                const auto& signal = h->lane[lane];
                const uint64_t requested = __atomic_load_n(&signal.drain_request, __ATOMIC_ACQUIRE);
                const uint64_t ack = __atomic_load_n(&signal.drain_ack, __ATOMIC_ACQUIRE);
                if (!requested || ack != requested || (closure && closure != requested)) drained = false;
                closure = requested;
            }
            if (drained) {
                const auto previous=read_shared_ack(map_,helper_);
                if (phase == PlacementState::QUIESCING) {
                    if (quiesced_transaction_ != c.transaction_id) {
                        quiesced_transaction_=c.transaction_id;
                        quiesced_at_ns_=clock_.now_ns();
                    }
                    publish_ack(map_,helper_,c,AckPhase::QUIESCED);
                }
                else if (phase == PlacementState::COMMIT &&
                         previous.transaction_id == c.transaction_id &&
                         previous.phase == uint32_t(AckPhase::QUIESCED))
                    publish_ack(map_,helper_,c,AckPhase::COMMITTED);
                else if (phase == PlacementState::RETIRING &&
                         previous.transaction_id == c.transaction_id &&
                         previous.phase == uint32_t(AckPhase::COMMITTED))
                    publish_ack(map_,helper_,c,AckPhase::RETIRED);
                if (phase == PlacementState::RETIRING &&
                    previous.transaction_id == c.transaction_id &&
                    previous.phase == uint32_t(AckPhase::COMMITTED))
                    {
                        awaiting_reopen_=c.transaction_id;
                        const uint64_t now=clock_.now_ns();
                        if (deadlines_.retire_ns > UINT64_MAX-now) {
                            // A non-leader has no controller to journal this.
                            // Keep the committed residency and do not raise helper_fault.
                            reopen_deadline_ns_=now;
                            placement_halted_=true;
                            placement_fault_=FaultCode::SEQUENCE_EXHAUSTED;
                        } else reopen_deadline_ns_=now+deadlines_.retire_ns;
                    }
            }
        }
        if (phase == PlacementState::ACTIVE) {
            // A peer must retain staged slots after terminal placement failure;
            // do not turn a cleanup timeout into a serving/helper fault.
            if (c.transaction_id && c.journal_index<=1) {
                const auto& j=placement_journal(map_)->entry[c.journal_index];
                if (j.transaction_id==c.transaction_id && j.cleanup_fault) {
                    placement_halted_=true;
                    placement_fault_=FaultCode(j.cleanup_fault);
                    return true;
                }
            }
            // A precommit abort may have an in-flight helper upload. Until
            // drain_upload proves completion, do NOT publish CLEANED and do
            // NOT kill the serving worker. The leader's cleanup deadline bounds
            // retry and records the participant that never finished.
            if (migration_ && migration_->transaction() && !awaiting_reopen_ &&
                !migration_->abort()) {
                const auto& j=placement_journal(map_)->entry[
                    c.journal_index<=1 ? c.journal_index : 0];
                if (c.journal_index>1 || j.transaction_id!=c.transaction_id ||
                    !j.fault || j.commit_ns) return false;
                if (helper_==0 && !controller_->poll_cleanup()) {
                    placement_halted_=true;
                    placement_fault_=FaultCode::CLEANUP_TIMEOUT;
                }
                return true;
            }
            if (!awaiting_reopen_ && c.transaction_id && c.journal_index<=1) {
                const auto& j=placement_journal(map_)->entry[c.journal_index];
                if (j.transaction_id==c.transaction_id && j.fault && !j.commit_ns)
                    publish_cleanup_ack(map_,helper_,c);
            }
            // ACTIVE publication follows all RETIRED rank/helper ACKs; wait
            // for engine reopen before a new cycle can stage the old slot.
            if (awaiting_reopen_) {
                // Drain words clear BEFORE burst restoration, parks and the
                // ledger gate. Only the exact post-ledger ACK permits staging.
                if (!reopened_exactly(map_,c,awaiting_reopen_)) {
                    if (clock_.now_ns() < reopen_deadline_ns_) return true;
                    if (helper_ == 0)
                        (void)controller_->stop_placement_after_commit(FaultCode::REOPEN_TIMEOUT);
                    placement_halted_ = true; // every helper keeps old residency on timeout
                    placement_fault_ = FaultCode::REOPEN_TIMEOUT;
                    return true; // worker loop must keep serving the committed generation
                }
                const uint64_t reopened=clock_.now_ns(); // admission proven reopened
                if (migration_ && !migration_->reclaim(map_,c)) return false;
                const uint64_t lost=(quiesced_transaction_==c.transaction_id &&
                                     reopened>=quiesced_at_ns_) ?
                                     reopened-quiesced_at_ns_ : 0;
                publish_cleanup_ack(map_,helper_,c,lost);
                if (lane_service_probe_) {
                    uint64_t total=0,rows=0;
                    if (lane_service_probe_(&total,&rows)) {
                        service_baseline_total_ns_=total;
                        service_baseline_rows_=rows;
                        service_baseline_ns_=reopened;
                    }
                }
                quiesced_transaction_=quiesced_at_ns_=0;
                awaiting_reopen_ = 0;
                reopen_deadline_ns_ = 0;
                if (helper_ == 0) {
                    ++completed_;
                    if (!schedule()) return false;
                }
            }
            if (helper_ == 0 && !controller_->poll_cleanup()) {
                placement_halted_ = true;
                placement_fault_ = FaultCode(controller_->journal().entry[
                    controller_->control().journal_index].cleanup_fault);
                return true;
            }
            if (helper_ != 0) return true;
            if (forced_proposal_) {
                // Exactly the existing Controller, after previous REOPENED
                // and CLEANED acknowledgments; no synthetic policy path.
                return controller_->can_propose() ?
                    forced_proposal_(map_,*controller_) : true;
            }
            if (period_.count() == 0) return true;
            if (!controller_->can_propose() ||
                !(load(&h->burst) & 1u) || clock_.now_ns() < next_ns_) return true;
            if (!stage_same_map_generation(map_)) {
                if (!controller_->reject_staging(FaultCode::INVALID_MAP)) return false;
                return schedule(); // forward abort on N; serving is untouched
            }
            return controller_->begin_epoch_cycle();
        }
        if (helper_ != 0) return true;
        if (phase == PlacementState::PROPOSED) {
            if (controller_->start_reserving()) return true;
            (void)controller_->fail(FaultCode::DEVICE);
            return !controller_->failed_closed() && schedule();
        }
        if (phase == PlacementState::PREPARED) {
            auto* candidate=generation_header(map_,c.pending_slot);
            if (!candidate || c.ready_helpers != c.expected_helpers ||
                c.ready_ranks != c.expected_ranks) {
                (void)controller_->fail(FaultCode::EPOCH_MISMATCH);
                return !controller_->failed_closed() && schedule();
            }
            candidate->ready_helpers=c.ready_helpers;
            candidate->ready_ranks=c.ready_ranks;
            candidate->state=uint32_t(GenerationState::READY);
            return controller_->begin_quiesce();
        }
        if (phase == PlacementState::COMMIT) {
            /* Commit has already published active_epoch/slot under the
             * existing shared-control seqlock. Admission stays closed. */
            generation_header(map_,c.active_slot)->state=uint32_t(GenerationState::ACTIVE);
        }
        const Poll result = controller_->poll();
        if (result == Poll::FAILED) {
            /* A precommit failure leaves ACTIVE(old) and permits a later
             * fresh transaction; after QUIESCING we fail closed. */
            if (!controller_->failed_closed() &&
                controller_->control().state == uint32_t(PlacementState::ACTIVE))
                return schedule();
            return false;
        }
        if (result == Poll::ADVANCED && phase == PlacementState::QUIESCING)
            return controller_->commit();
        if (result == Poll::ADVANCED && phase == PlacementState::RETIRING) {
            const auto committed = controller_->control();
            if (committed.state != uint32_t(PlacementState::ACTIVE)) return false;
            generation_header(map_,1u-committed.active_slot)->state=
                uint32_t(GenerationState::RETIRED);
            awaiting_reopen_ = c.transaction_id;
            const uint64_t now = clock_.now_ns();
            if (deadlines_.retire_ns > UINT64_MAX - now) {
                (void)controller_->stop_placement_after_commit(FaultCode::SEQUENCE_EXHAUSTED);
                placement_halted_ = true;
                placement_fault_ = FaultCode::SEQUENCE_EXHAUSTED;
                return true; // journal a terminal diagnostic, do not kill serving
            }
            reopen_deadline_ns_ = now + deadlines_.retire_ns;
        }
        return true;
    }
};
} // namespace r3::epoch
