#pragma once
// Engine placement participant. No GPU dependency: the production loop supplies
// rank operations (which own the mutex, HIP event queries and admission ledger).
// CPU tests drive this SAME state machine with fake rank devices.
#include "worker_epoch.h"

namespace r3::epoch {

enum class RankResult : uint8_t { Ready, Pending, DeviceError };

struct RankOperations {
    virtual ~RankOperations() = default;
    virtual uint32_t actual_ranks() = 0;
    virtual bool registered(uint32_t rank) = 0;
    virtual RankResult upload(uint32_t rank, uint32_t slot, uint64_t epoch, uint64_t digest) = 0;
    virtual RankResult ready(uint32_t rank, uint32_t slot, uint64_t epoch, uint64_t digest) = 0;
    virtual bool exact_binding(uint32_t rank, uint32_t slot, uint64_t epoch, uint64_t digest) = 0;
    // Owns the actual admission close, ledger/event wait and helper lane drain;
    // returns the sealed snapshot's cutoff and old epoch, never a fabricated ACK.
    virtual bool close_and_wait(uint64_t deadline_ns, uint64_t* cutoff, uint32_t* old_epoch) = 0;
    virtual bool consumed() = 0;
    virtual bool drain_verified() = 0;
    virtual bool helper_quiesced() = 0;
    virtual bool generation_referenced(uint32_t slot) = 0;
    virtual bool can_replace(uint32_t slot) = 0;
    virtual bool reopen(uint32_t epoch, uint32_t slot) = 0;
    virtual bool stopped() = 0;
    virtual void fatal() = 0;
};

class EngineRankCoordinator {
    uint64_t uploaded_ = 0, reserved_ = 0, closed_ = 0;
    uint64_t cutoff_ = 0;
    uint32_t old_epoch_ = 0;
    uint32_t reserved_mask_ = 0, issued_mask_ = 0, ready_mask_ = 0, committed_mask_ = 0;
public:
    bool step(void* map, const PlacementControl& c, RankOperations& ops) {
        if (!map || ops.stopped()) { ops.fatal(); return false; }
        const auto state = PlacementState(c.state);
        const uint32_t ranks = c.expected_ranks;
        const uint32_t actual = ops.actual_ranks();
        const bool current = actual && actual <= kMaxEngineRanks &&
            ranks == ((1u << actual) - 1u) && same_map_cycle(map,c);
        auto fail = [&]() -> bool { ops.fatal(); return false; };
        auto fault_ranks = [&](AckPhase phase, FaultCode code) {
            for (uint32_t rank=0; rank<kMaxEngineRanks; ++rank)
                if (ranks & (1u<<rank))
                    publish_fault_ack(map,kMaxHelpers+rank,c,phase,code);
        };

        if (state == PlacementState::RESERVING) {
            if (reserved_ != c.transaction_id) {
                reserved_ = c.transaction_id;
                reserved_mask_ = 0;
            }
            if (closed_ || !current) {
                fault_ranks(AckPhase::RESERVED,FaultCode::CHECKSUM);
            } else {
                for (uint32_t rank=0; rank<kMaxEngineRanks; ++rank) {
                    const uint32_t bit=1u<<rank;
                    if (!(ranks & bit) || (reserved_mask_ & bit)) continue;
                    const bool present = ops.registered(rank);
                    if (!present) continue; // bounded by controller reservation deadline
                    const RankResult result=ops.ready(rank,c.active_slot,c.active_epoch,c.active_digest);
                    if (result == RankResult::DeviceError) {
                        publish_fault_ack(map,kMaxHelpers+rank,c,AckPhase::RESERVED,FaultCode::DEVICE);
                    } else if (service_rank_reservation(map,c,rank,ranks,present,
                                                         result==RankResult::Ready)) {
                        reserved_mask_ |= bit;
                    } else if (result == RankResult::Pending) {
                        publish_fault_ack(map,kMaxHelpers+rank,c,AckPhase::RESERVED,FaultCode::CHECKSUM);
                    }
                }
            }
        } else if (state == PlacementState::COPYING) {
            if (uploaded_ != c.transaction_id) {
                uploaded_ = c.transaction_id;
                issued_mask_ = ready_mask_ = committed_mask_ = 0;
            }
            if (closed_ || !current) {
                fault_ranks(AckPhase::READY,FaultCode::CHECKSUM);
            } else {
                for (uint32_t rank=0; rank<kMaxEngineRanks; ++rank) {
                    const uint32_t bit=1u<<rank;
                    if (!(ranks & bit)) continue;
                    if (!(issued_mask_ & bit)) {
                        const RankResult result=ops.upload(rank,c.pending_slot,c.pending_epoch,c.pending_digest);
                        if (result != RankResult::Ready) {
                            fault_ranks(AckPhase::READY,result==RankResult::DeviceError ?
                                        FaultCode::DEVICE : FaultCode::UPLOAD);
                            break;
                        }
                        issued_mask_ |= bit;
                    }
                    if (!(ready_mask_ & bit)) {
                        // A pending event is retryable, but wrong upload identity
                        // is a CHECKSUM fault before any rank may publish READY.
                        if (!ops.exact_binding(rank,c.pending_slot,c.pending_epoch,c.pending_digest)) {
                            fault_ranks(AckPhase::READY,FaultCode::CHECKSUM);
                            break;
                        }
                        const RankResult result=ops.ready(rank,c.pending_slot,c.pending_epoch,c.pending_digest);
                        if (result == RankResult::Ready &&
                            rank_upload_prepared(c,rank,true,true,c.pending_epoch,
                                                 c.pending_digest,c.pending_slot)) {
                            ready_mask_ |= bit;
                            publish_ack(map,kMaxHelpers+rank,c,AckPhase::READY);
                        } else if (result == RankResult::DeviceError) {
                            fault_ranks(AckPhase::READY,FaultCode::DEVICE);
                            break;
                        }
                    }
                }
            }
        } else if (state == PlacementState::QUIESCING && closed_ != c.transaction_id) {
            if (uploaded_ != c.transaction_id || ready_mask_ != ranks || !current ||
                !ops.close_and_wait(c.deadline_ns,&cutoff_,&old_epoch_))
                return fail();
            closed_ = c.transaction_id;
            for (uint32_t rank=0; rank<kMaxEngineRanks; ++rank)
                if (ranks & (1u<<rank))
                    publish_ack(map,kMaxHelpers+rank,c,AckPhase::QUIESCED,cutoff_);
        } else if (state == PlacementState::COMMIT && closed_ == c.transaction_id && closed_) {
            if (!current || uint64_t(old_epoch_)+1u != c.active_epoch ||
                c.journal_index > 1 ||
                c.active_slot != 1u-placement_journal(map)->entry[c.journal_index].old_slot ||
                ready_mask_ != ranks) return fail();
            // The helper must mark the new header ACTIVE before COMMITTED.
            const auto* active=generation_header(map,c.active_slot);
            if (active && active->state == uint32_t(GenerationState::ACTIVE)) {
                for (uint32_t rank=0; rank<kMaxEngineRanks; ++rank)
                    if (ranks & (1u<<rank)) {
                        publish_ack(map,kMaxHelpers+rank,c,AckPhase::COMMITTED,cutoff_);
                        committed_mask_ |= 1u<<rank;
                    }
            }
        } else if (state == PlacementState::RETIRING && closed_ == c.transaction_id && closed_) {
            const uint32_t old_slot=1u-c.active_slot;
            const bool consumed=ops.consumed();
            const bool drained=ops.drain_verified();
            const bool quiescent=drained && ops.helper_quiesced();
            const bool unreferenced=!ops.generation_referenced(old_slot);
            if (!current ||
                !rank_retirement_proved(c,old_epoch_,old_slot,consumed,drained,quiescent,
                                        unreferenced,ready_mask_==ranks,committed_mask_==ranks)) {
                if (!current || ops.stopped()) return fail();
            } else {
                // The ledger's active slot is still the OLD generation.
                if (ops.can_replace(old_slot)) return fail();
                for (uint32_t rank=0; rank<kMaxEngineRanks; ++rank)
                    if (ranks & (1u<<rank))
                        publish_ack(map,kMaxHelpers+rank,c,AckPhase::RETIRED,cutoff_);
            }
        } else if (state == PlacementState::ACTIVE && closed_) {
            if (c.active_slot > 1) return fail();
            const uint32_t old_slot=1u-c.active_slot;
            const auto* retired=generation_header(map,old_slot);
            if (retired && retired->state == uint32_t(GenerationState::RETIRED)) {
                bool events_ready=true;
                for (uint32_t rank=0; rank<kMaxEngineRanks; ++rank)
                    if ((ranks & (1u<<rank)) &&
                        ops.ready(rank,c.active_slot,c.active_epoch,c.active_digest)!=RankResult::Ready)
                        events_ready=false;
                if (c.transaction_id != closed_ || !c.active_epoch ||
                    c.active_epoch != uint64_t(old_epoch_)+1u ||
                    ready_mask_ != ranks || committed_mask_ != ranks || !events_ready ||
                    !ops.consumed() || !ops.reopen((uint32_t)c.active_epoch,c.active_slot) ||
                    !ops.can_replace(old_slot)) return fail();
                // One production publication, after the real reopen operation.
                // Fake rank ops use this same boundary, not an independent ACK.
                publish_reopen_ack(map,c);
                cutoff_ = 0;
                old_epoch_ = 0;
                closed_ = 0;
            }
        } else if (state == PlacementState::ACTIVE && !closed_ &&
                   uploaded_ && uploaded_ == c.transaction_id) {
            // Precommit abort: keep serving old generation; retry new txn.
            issued_mask_ = ready_mask_ = committed_mask_ = 0;
            uploaded_ = 0;
        }
        return true;
    }
};
} // namespace r3::epoch
