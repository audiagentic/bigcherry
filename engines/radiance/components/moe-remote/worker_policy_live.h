#pragma once
// Slice H live-input derivations. CPU only: never inspect other helpers' private
// Ledger objects. Read the v23 ACTIVE map only when Controller::can_propose()
// proves the preceding transaction has been fully CLEANED and REOPENED.
#include "worker_policy_input.h"
#include "worker_slot_pool.h"
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace r3::policy {

// worker_slots::Pool charges exactly its container-derived maximum slot stride
// for EVERY occupied expert, not the unpadded container file size. Its Ledger
// has hard=expert_budget+stride, scratch=0, incoming_reserve=stride. This is
// the same budget in R3_MOE_WORKER_MIBS passed to Runner::allocate and to the
// forced-move driver. A synthetic ACTIVE slot is only an accounting witness:
// its id is NOT the remote process's private physical slot identifier.
//
// This reconstruction is valid solely at a fully closed ACTIVE placement
// boundary (Controller::can_propose()); RESERVING..CLEANED may have additional
// private charged slots and MUST NOT be reconstructed from owners.
inline bool derive_active_ledgers(const OwnerSnapshot& owners,
    const std::vector<uint64_t>& physical_bytes,
    const std::vector<uint64_t>& expert_budgets, uint64_t stride,
    const std::vector<double>& costs, const std::vector<double>& caps,
    std::vector<LedgerView>* out, std::string* why = nullptr) {
    auto reject = [&](const char* s) { if (why) *why=s; return false; };
    if (!out || !owners.layers || !owners.experts ||
        !owners.helpers || owners.helpers > moe_shm::kMaxHelpers ||
        !owners.epoch || !owners.digest || !stride ||
        size_t(owners.layers)*owners.experts != owners.owner.size() ||
        physical_bytes.size() != owners.owner.size() ||
        owners.runtime.size() != owners.owner.size() ||
        expert_budgets.size() != owners.helpers ||
        costs.size() != owners.helpers || caps.size() != owners.helpers)
        return reject("policy: cannot reconstruct incomplete ACTIVE accounting");
    std::vector<LedgerView> result(owners.helpers);
    for (size_t d=0; d<result.size(); ++d) {
        const uint64_t budget=expert_budgets[d];
        if (budget < stride || budget > UINT64_MAX-stride ||
            !std::isfinite(costs[d]) || costs[d] <= 0 ||
            !std::isfinite(caps[d]) || caps[d] < 0)
            return reject("policy: invalid declared helper budget or cost");
        auto& v=result[d];
        v.hard=budget+stride;
        v.scratch=0;
        v.headroom=stride;
        v.cost=costs[d];
        v.layer_cap=caps[d];
    }
    for (size_t i=0; i<owners.owner.size(); ++i) {
        const int32_t d=owners.owner[i];
        if (d < -1 || d >= int32_t(owners.helpers))
            return reject("policy: invalid ACTIVE owner");
        if (d < 0) continue; // engine-local never charged to a helper
        if (!owners.runtime[i] || physical_bytes[i] != stride)
            return reject("policy: missing runtime slot or wrong physical stride");
        auto& v=result[size_t(d)];
        const uint64_t budget=expert_budgets[size_t(d)];
        if (v.charged > budget || stride > budget-v.charged ||
            v.slots.size() >= budget/stride)
            return reject("policy: reconstructed helper budget exceeded");
        const uint32_t id=uint32_t(v.slots.size());
        if (v.slots.size() >= UINT32_MAX)
            return reject("policy: synthetic slot index overflow");
        v.slots.push_back({id,uint32_t(i),stride,owners.epoch,
                           residency::State::ACTIVE,residency::kCompletePlaneMask});
        v.charged += stride;
    }
    *out=std::move(result);
    return true;
}

enum class HistoryObservation : uint8_t { UNCHANGED, COMMITTED, ABORTED, INVALID };

// Helper 0 proposes every migration. Record moves only from a fully closed,
// verified transaction journal, *never* from a planned candidate or an ACK.
// The history is intentionally process-local: after restart zero timestamps
// mean "last moved long ago" (stage-1 policy's existing convention).
class MoveHistory {
    OwnerSnapshot previous_{};
    std::vector<uint64_t> last_move_;
    uint64_t seen_transaction_=0;
    bool initialized_=false;
public:
    bool reset(const OwnerSnapshot& initial, uint64_t transaction_id=0) {
        if (!initial.layers || !initial.experts || !initial.epoch ||
            !initial.digest || !initial.helpers ||
            initial.owner.size()!=size_t(initial.layers)*initial.experts ||
            initial.runtime.size()!=initial.owner.size())
            return false;
        previous_=initial;
        last_move_.assign(initial.owner.size(),0);
        seen_transaction_=transaction_id;
        initialized_=true;
        return true;
    }
    const std::vector<uint64_t>& times() const { return last_move_; }
    HistoryObservation observe_closed(const moe_shm::JournalRecord& j,
        const OwnerSnapshot& active, uint64_t now_ns) {
        if (!initialized_ || !now_ns ||
            active.layers!=previous_.layers ||
            active.experts!=previous_.experts ||
            active.helpers!=previous_.helpers ||
            active.owner.size()!=last_move_.size() ||
            active.runtime.size()!=last_move_.size() ||
            !active.digest || !active.epoch)
            return HistoryObservation::INVALID;
        if (j.transaction_id <= seen_transaction_)
            return (active.digest==previous_.digest &&
                    active.epoch==previous_.epoch &&
                    active.owner==previous_.owner)
                ? HistoryObservation::UNCHANGED : HistoryObservation::INVALID;
        // Caller must first establish Controller::can_propose() and capture
        // this exact ACTIVE generation. A journal with no completed cleanup
        // is NEVER proof that a move is resident and safely committed.
        // A precommit abort preserves the failing phase in the journal
        // (RESERVING/COPYING/PREPARED), while the controller returns ACTIVE.
        // Controller::can_propose() plus exact CLEANED acks is the close proof.
        if (!j.finish_ns ||
            (j.commit_ns && j.state!=uint32_t(moe_shm::PlacementState::ACTIVE)) ||
            (!j.commit_ns && j.state>uint32_t(moe_shm::PlacementState::PREPARED)) ||
            j.before_epoch!=previous_.epoch ||
            j.before_digest!=previous_.digest)
            return HistoryObservation::INVALID;
        if (j.commit_ns) {
            if (j.after_epoch!=previous_.epoch+1 ||
                active.epoch!=j.after_epoch ||
                active.digest!=j.candidate_digest)
                return HistoryObservation::INVALID;
            for (size_t i=0; i<last_move_.size(); ++i)
                if (previous_.owner[i]!=active.owner[i])
                    last_move_[i]=now_ns;
            previous_=active;
            seen_transaction_=j.transaction_id;
            return HistoryObservation::COMMITTED;
        }
        if (!j.fault || active.epoch!=previous_.epoch ||
            active.digest!=previous_.digest ||
            active.owner!=previous_.owner)
            return HistoryObservation::INVALID;
        seen_transaction_=j.transaction_id;
        return HistoryObservation::ABORTED;
    }
};

} // namespace r3::policy
