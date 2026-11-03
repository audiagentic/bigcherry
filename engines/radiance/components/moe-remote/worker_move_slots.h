#pragma once
// Helper-side placement migration. All orchestration is CPU-only; production
// supplies HIP operations, tests supply fake streams and event completions.
#include "worker_epoch.h"
#include "worker_residency.h"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace r3::worker_moves {
using namespace r3::epoch;
using namespace r3::moe_shm;

enum class Event : uint8_t { PENDING, COMPLETE, FAILED };
struct Operations {
    virtual ~Operations() = default;
    // start_upload queues all planes into an unreferenced physical slot on a
    // private stream and records a distinct event for this expert.
    virtual bool start_upload(uint32_t expert, uint32_t physical, uint32_t local) = 0;
    virtual Event upload_event() = 0;
    virtual bool build_bank(uint32_t generation, const std::vector<int32_t>& locals) = 0;
    virtual bool drain_upload() = 0;
    virtual void forget(uint32_t expert) = 0;
    virtual void activate_bank(uint32_t generation) = 0;
    // Immutable engine-defined runtime index, shared by both generations.
    virtual int32_t preferred_local(uint32_t expert) const = 0;
};

class Participant : public r3::epoch::HelperMigration {
    residency::Ledger& ledger_;
    Operations& operations_;
    uint32_t helper_, layers_, experts_;
    uint64_t stride_;
    std::vector<int32_t> local_[2], physical_[2];
    std::vector<uint32_t> gained_, lost_, reserved_;
    uint64_t transaction_ = 0, candidate_epoch_ = 0;
    uint32_t candidate_slot_ = UINT32_MAX, old_slot_ = UINT32_MAX;
    size_t uploaded_ = 0;
    bool copying_ = false, bank_built_ = false, committed_ = false;
    bool initialized_ = false;
    FaultCode staging_fault_ = FaultCode::NONE;
    static constexpr int32_t none = -1;

    bool is_mine(const void* map, uint32_t slot, uint32_t index) const {
        return generation_map(map, slot)[index] == helper_;
    }
    void reset() {
        gained_.clear(); lost_.clear(); reserved_.clear();
        uploaded_ = 0; copying_ = bank_built_ = committed_ = false;
        transaction_ = candidate_epoch_ = 0;
        staging_fault_ = FaultCode::NONE;
        candidate_slot_ = old_slot_ = UINT32_MAX;
    }
public:
    Participant(residency::Ledger& ledger, Operations& operations,
                uint32_t helper, uint32_t layers, uint32_t experts, uint64_t stride)
        : ledger_(ledger), operations_(operations), helper_(helper),
          layers_(layers), experts_(experts), stride_(stride) {}

    bool initialize(const void* map, uint32_t active, const std::vector<int32_t>& locals) {
        if (initialized_ || !map || active > 1 || !stride_ || !layers_ || !experts_ ||
            locals.size() != size_t(layers_) * experts_) return false;
        const auto* h = static_cast<const Header*>(map);
        if (h->layer_count != layers_ || h->expert_count != experts_ ||
            helper_ >= h->helper_count) return false;
        local_[active] = locals;
        physical_[active].assign(locals.size(), none);
        for (size_t i=0; i<locals.size(); ++i) {
            if (is_mine(map,active,uint32_t(i)) != (locals[i] >= 0)) return false;
            if (locals[i] < 0) continue;
            if (locals[i]!=operations_.preferred_local(uint32_t(i))) return false;
            bool found = false;
            for (const auto& s: ledger_.slots())
                if (s.expert == i && s.state == residency::State::ACTIVE &&
                    s.bytes == stride_) {
                    physical_[active][i] = int32_t(s.id); found=true; break;
                }
            if (!found) return false;
        }
        initialized_ = true;
        return true;
    }
    bool initialized() const { return initialized_; }
    uint64_t transaction() const override { return transaction_; }
    bool bank_built() const { return bank_built_; }
    const std::vector<int32_t>& local(uint32_t generation) const { return local_[generation]; }
    const std::vector<int32_t>& physical(uint32_t generation) const { return physical_[generation]; }

    // Must be called only for the existing Controller::propose/begin_epoch_cycle
    // transaction. No alternate control transaction or proposed map is made here.
    FaultCode reserve(const void* map, const PlacementControl& c) override {
        if (!initialized_ || !map || !placement_candidate_valid(map,c) ||
            c.state != uint32_t(PlacementState::RESERVING))
            return FaultCode::CHECKSUM;
        if (transaction_ == c.transaction_id)
            return candidate_slot_ == c.pending_slot ? staging_fault_ : FaultCode::CHECKSUM;
        if (transaction_ || c.active_slot > 1 || c.pending_slot > 1)
            return FaultCode::CAPACITY; // previous generation still held
        old_slot_=c.active_slot; candidate_slot_=c.pending_slot;
        transaction_=c.transaction_id; candidate_epoch_=c.pending_epoch;
        const auto fail = [&](FaultCode code) { staging_fault_=code; return code; };
        const size_t count=size_t(layers_)*experts_;
        local_[candidate_slot_].assign(count,none);
        physical_[candidate_slot_].assign(count,none);
        const auto* candidates = generation_map(map,candidate_slot_);
        const auto* active = generation_map(map,old_slot_);
        for (size_t i=0; i<count; ++i) {
            if (active[i]==helper_ && candidates[i]==helper_) {
                local_[candidate_slot_][i]=local_[old_slot_][i];
                physical_[candidate_slot_][i]=physical_[old_slot_][i];
                if (local_[candidate_slot_][i]<0 || physical_[candidate_slot_][i]<0)
                    return fail(FaultCode::CHECKSUM);
            } else if (active[i]==helper_ && candidates[i]!=helper_) {
                if (physical_[old_slot_][i]<0) return fail(FaultCode::CHECKSUM);
                lost_.push_back(uint32_t(i));
            } else if (active[i]!=helper_ && candidates[i]==helper_) {
                gained_.push_back(uint32_t(i));
            }
        }
        // The engine's layer_slot map is immutable and was sealed at
        // bootstrap. A newly owned expert uses its predeclared index, not a
        // different free index (which would change routing on the hot path).
        for (uint32_t i : gained_) {
            const int32_t local=operations_.preferred_local(i);
            if (local<0 || uint32_t(local)>=experts_)
                return fail(FaultCode::SOURCE); // protected/absent engine layer
            for (uint32_t e=0;e<experts_;++e) {
                const size_t index=size_t(i/experts_)*experts_+e;
                if (local_[candidate_slot_][index]==local)
                    return fail(FaultCode::CHECKSUM);
            }
            local_[candidate_slot_][i]=local;
        }
        for (uint32_t i : gained_) {
            uint32_t id=UINT32_MAX;
            const auto result=ledger_.reserve(i,stride_,candidate_epoch_,true,&id);
            if (result!=residency::Result::OK) {
                // The caller invokes abort() before publishing a fault ACK.
                return fail(result==residency::Result::DEFERRED_CAPACITY ?
                    FaultCode::CAPACITY : FaultCode::DEVICE);
            }
            physical_[candidate_slot_][i]=int32_t(id);
            reserved_.push_back(id);
        }
        return FaultCode::NONE;
    }

    HelperCopy copy() override {
        if (!transaction_ || committed_ || candidate_slot_>1) return HelperCopy::FAILED;
        if (bank_built_) return HelperCopy::READY;
        if (copying_) {
            const Event state=operations_.upload_event();
            if (state==Event::FAILED) return HelperCopy::FAILED;
            if (state==Event::PENDING) return HelperCopy::WAITING;
            const auto slot=reserved_[uploaded_];
            for (uint32_t plane=0; plane<4; ++plane)
                if (ledger_.plane_complete(slot,plane)!=residency::Result::OK) return HelperCopy::FAILED;
            if (ledger_.ready(slot)!=residency::Result::OK) return HelperCopy::FAILED;
            ++uploaded_; copying_=false;
        }
        if (uploaded_<gained_.size()) {
            const uint32_t expert=gained_[uploaded_], slot=reserved_[uploaded_];
            if (ledger_.start_copy(slot)!=residency::Result::OK ||
                !operations_.start_upload(expert,slot,
                                          uint32_t(local_[candidate_slot_][expert])))
                return HelperCopy::FAILED;
            copying_=true;
            return HelperCopy::WAITING; // never READY in the upload-issuing tick
        }
        if (!operations_.build_bank(candidate_slot_,local_[candidate_slot_]))
            return HelperCopy::FAILED;
        bank_built_=true;
        return HelperCopy::READY;
    }

    bool commit(const PlacementControl& c) override {
        if (!transaction_ || !bank_built_ ||
            c.transaction_id!=transaction_ || c.active_slot!=candidate_slot_ ||
            c.active_epoch!=candidate_epoch_ ||
            c.state < uint32_t(PlacementState::COMMIT)) return false;
        if (committed_) return true;
        for (uint32_t id : reserved_)
            if (ledger_.activate(id,candidate_epoch_)!=residency::Result::OK) return false;
        for (uint32_t i : lost_) {
            const uint32_t id=uint32_t(physical_[old_slot_][i]);
            if (ledger_.retire(id)!=residency::Result::OK) return false;
        }
        operations_.activate_bank(candidate_slot_);
        committed_=true;
        return true;
    }

    // Caller must provide the genuine post-all-ranks-consumed REOPENED ACK,
    // not merely RETIRING or an in-memory ACTIVE control snapshot.
    bool reclaim(const void* map, const PlacementControl& c) override {
        if (!committed_ || !reopened_exactly(map,c,transaction_)) return false;
        for (uint32_t i : lost_) {
            const auto id=uint32_t(physical_[old_slot_][i]);
            if (ledger_.reclaim(id,true)!=residency::Result::OK) return false;
            operations_.forget(i);
        }
        local_[old_slot_].clear();
        physical_[old_slot_].clear();
        reset();
        return true;
    }
    bool abort() override {
        if (!transaction_ || committed_) return false;
        if (!operations_.drain_upload()) return false;
        for (uint32_t id : reserved_)
            if (ledger_.cancel(id,true)!=residency::Result::OK) return false;
        for (uint32_t i : gained_) operations_.forget(i);
        local_[candidate_slot_].clear();
        physical_[candidate_slot_].clear();
        reset();
        return true;
    }
};
} // namespace r3::worker_moves
