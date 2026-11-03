#pragma once
// CPU placement-epoch coordinator plus the v23 shared-memory cycle transport.
// The state machine uses injectable clocks/transport; no HIP or GPU work here.
#include "moe_remote_shm.h"
#include <cstdint>
#include <chrono>
#include <cstring>
#include <memory>
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
    PlacementControl c{};
    if (!map || !placement_snapshot(map, &c) ||
        c.state != uint32_t(PlacementState::ACTIVE) || c.fault_code ||
        !c.active_epoch || c.active_epoch >= UINT32_MAX ||
        !c.active_digest || c.active_slot > 1) return false;
    const Header* h = static_cast<const Header*>(map);
    GenerationHeader* active = generation_header(map,c.active_slot);
    GenerationHeader* inactive = generation_header(map,1u-c.active_slot);
    if (!active || !inactive || !h->owner_map_bytes ||
        active->epoch != c.active_epoch || active->digest != c.active_digest ||
        active->state != uint32_t(GenerationState::ACTIVE) ||
        (inactive->state != uint32_t(GenerationState::EMPTY) &&
         inactive->state != uint32_t(GenerationState::RETIRED) &&
         inactive->state != uint32_t(GenerationState::STAGING) &&
         inactive->state != uint32_t(GenerationState::READY)) ||
        active->map_bytes != h->owner_map_bytes ||
        inactive->map_bytes != h->owner_map_bytes ||
        generation_digest(map,c.active_slot) != c.active_digest) return false;
    std::memcpy(generation_map(map,1u-c.active_slot),generation_map(map,c.active_slot),
                h->owner_map_bytes);
    inactive->epoch = c.active_epoch+1;
    inactive->digest = c.active_digest;
    inactive->ready_helpers = inactive->ready_ranks = 0;
    inactive->state = uint32_t(GenerationState::STAGING);
    return generation_digest(map,1u-c.active_slot)==c.active_digest;
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
inline bool rank_upload_prepared(const PlacementControl& c,uint32_t rank,
                                bool uploaded,bool event_ready,
                                uint64_t epoch,uint64_t digest,uint32_t slot) {
    return c.state == uint32_t(PlacementState::COPYING) &&
           rank < kMaxEngineRanks && (c.expected_ranks & (1u<<rank)) &&
           same_placement_cycle(c) && uploaded && event_ready &&
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
           same_placement_cycle(c) && old_epoch && old_epoch < UINT32_MAX &&
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
    if (__atomic_load_n(&a->phase, __ATOMIC_ACQUIRE) != phase ||
        __atomic_load_n(&a->transaction_id, __ATOMIC_ACQUIRE) != out.transaction_id)
        return PlacementAck{};
    out.phase = phase;
    return out;
}
inline void publish_ack(void* map, uint32_t actor, const PlacementControl& c,
                        AckPhase phase, uint64_t cutoff = 0) {
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
    __atomic_store_n(&a->phase, uint32_t(phase), __ATOMIC_RELEASE);
    notify(static_cast<Header*>(map));
}
class SharedTransport : public Transport {
    void* map_;
public:
    explicit SharedTransport(void* map) : map_(map) {}
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

// Helper-side periodic driver. The caller polls it beside the serving lanes;
// only helper zero publishes PlacementControl. No environment settings here.
class EpochCycle {
    void* map_;
    uint32_t helper_;
    const Clock& clock_;
    std::chrono::nanoseconds period_;
    uint64_t next_ns_ = 0;
    uint64_t awaiting_reopen_ = 0;
    uint64_t completed_ = 0;   // full close-drain-recommit-reopen cycles, for the run's evidence
    SharedTransport transport_;
    std::unique_ptr<Controller> controller_;
    bool failed_ = false;
    bool schedule() {
        const uint64_t now = clock_.now_ns();
        const uint64_t period = uint64_t(period_.count());
        if (period > UINT64_MAX - now) return false;
        next_ns_ = now + period;
        return true;
    }
public:
    EpochCycle(void* map, uint32_t helper, const Clock& clock,
               std::chrono::nanoseconds period)
        : map_(map), helper_(helper), clock_(clock), period_(period), transport_(map) {
        failed_ = !map || helper >= static_cast<Header*>(map)->helper_count || period.count() < 0;
    }
    bool enabled() const { return period_.count() != 0; }
    uint64_t completed() const { return completed_; }
    bool tick() {
        if (failed_) return false;
        if (period_.count() == 0) return true; // typed setting disables the experiment
        auto* h = static_cast<Header*>(map_);
        if (load(&h->helper_fault)) return false;
        PlacementControl c{};
        if (!placement_snapshot(map_, &c)) return true;
        // Wait for the engine to publish its actual participating rank mask.
        if (!c.expected_ranks) return true;
        if (helper_ == 0 && !controller_) {
            if (c.state != uint32_t(PlacementState::ACTIVE) || c.fault_code ||
                c.expected_ranks >= (1u << kMaxEngineRanks) ||
                (c.expected_ranks & (c.expected_ranks + 1u))) return false;
            uint32_t ranks = 0;
            for (uint32_t mask = c.expected_ranks; mask; mask >>= 1) ++ranks;
            controller_ = std::make_unique<Controller>(transport_, clock_, c.active_digest,
                h->helper_count, ranks, Deadlines{},
                Bootstrap{c.active_epoch, c.active_slot, c.seq, c.transaction_id});
            if (!schedule()) return false;
            c = controller_->control();
        }
        if (c.fault_code) return false;
        const auto phase = PlacementState(c.state);
        if (phase != PlacementState::ACTIVE && !same_map_cycle(map_,c)) {
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
        /* Every helper confirms the identical generation; there is no
         * helper-side upload or table switch in this no-move diagnostic. */
        if (phase == PlacementState::RESERVING)
            publish_ack(map_, helper_, c, AckPhase::RESERVED);
        if (phase == PlacementState::COPYING)
            publish_ack(map_, helper_, c, AckPhase::READY);
        if (phase == PlacementState::QUIESCING || phase == PlacementState::COMMIT ||
            phase == PlacementState::RETIRING) {
            // Slice C drain identities are published once on the lane CPU
            // at the closure ACK, not as per-request helper GPU stores.
            uint64_t closure = 0;
            bool drained = true;
            for (uint32_t lane = 0; lane < kLanes; ++lane) {
                if ((h->helper_count > 1 && lane % kMaxSlots != helper_) ||
                    (lane_is_small(lane) && lane_class(lane) >= h->small_class_count)) continue;
                const auto& signal = h->lane[lane];
                const uint64_t requested = __atomic_load_n(&signal.drain_request, __ATOMIC_ACQUIRE);
                const uint64_t ack = __atomic_load_n(&signal.drain_ack, __ATOMIC_ACQUIRE);
                if (!requested || ack != requested || (closure && closure != requested)) drained = false;
                closure = requested;
            }
            if (drained) {
                const auto previous=read_shared_ack(map_,helper_);
                if (phase == PlacementState::QUIESCING)
                    publish_ack(map_,helper_,c,AckPhase::QUIESCED);
                else if (phase == PlacementState::COMMIT &&
                         previous.transaction_id == c.transaction_id &&
                         previous.phase == uint32_t(AckPhase::QUIESCED))
                    publish_ack(map_,helper_,c,AckPhase::COMMITTED);
                else if (phase == PlacementState::RETIRING &&
                         previous.transaction_id == c.transaction_id &&
                         previous.phase == uint32_t(AckPhase::COMMITTED))
                    publish_ack(map_,helper_,c,AckPhase::RETIRED);
            }
        }
        if (helper_ != 0) return true;
        if (phase == PlacementState::ACTIVE) {
            // ACTIVE publication follows all RETIRED rank/helper ACKs; wait
            // for engine reopen before a new cycle can stage the old slot.
            if (awaiting_reopen_) {
                for (const auto& lane : h->lane)
                    if (__atomic_load_n(&lane.drain_request, __ATOMIC_ACQUIRE)) return true;
                awaiting_reopen_ = 0;
                ++completed_;
                if (!schedule()) return false;
            }
            if (!(load(&h->burst) & 1u) || clock_.now_ns() < next_ns_) return true;
            return stage_same_map_generation(map_) && controller_->begin_epoch_cycle();
        }
        if (phase == PlacementState::PROPOSED)
            return controller_->start_reserving();
        if (phase == PlacementState::PREPARED) {
            auto* candidate=generation_header(map_,c.pending_slot);
            if (!candidate || c.ready_helpers != c.expected_helpers ||
                c.ready_ranks != c.expected_ranks) return false;
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
        }
        return true;
    }
};
} // namespace r3::epoch
