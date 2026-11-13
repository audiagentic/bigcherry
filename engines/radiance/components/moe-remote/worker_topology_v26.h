#pragma once
// RCS14 gate C: counted CPU-side access, epoch proofs, and resource admission.
// Not yet the live worker: v25 GPU Request/Result/Combine and the producer must
// switch with gate D before this becomes the production coordinator.
#include "moe_remote_topology_v26.h"
#include "worker_footprint.h"
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <utility>
#include <limits>
#include <vector>

namespace r3::moe_shm::v26::cpu {

enum class Refusal : uint8_t {
    None, InvalidSegment, WrongTopology, InvalidRole, InvalidPhase,
    WrongAckIdentity, ParticipantFault, SequenceExhausted,
    Undrained, InvalidShape, ResourceOverflow, BudgetExceeded
};

// No consumers derive dynamic pointers from an unvalidated Header.
class Segment {
    void* base_ = nullptr;
    Layout layout_{};
    Topology topology_{};
public:
    Refusal attach(void* base, size_t mapped, const LayoutShape& shape,
                   uint32_t layers, Topology topology, uint64_t manifest) {
        base_ = nullptr;
        Layout accepted{};
        if (v26::valid(base, mapped, shape, layers, topology, manifest, &accepted)
                != v26::Refusal::None) return Refusal::InvalidSegment;
        base_ = base;
        layout_ = accepted;
        topology_ = topology;
        return Refusal::None;
    }
    bool attached() const { return base_ != nullptr; }
    Topology topology() const { return topology_; }
    const Layout& layout() const { return layout_; }
    const Header* header() const { return static_cast<const Header*>(base_); }
    bool actor(bool helper, uint32_t index, uint32_t* out) const {
        if (!base_ || !out) return false;
        if (helper) {
            if (index >= topology_.helpers) return false;
            *out = index;
            return true;
        }
        return ack_for_rank(topology_, index, out);
    }
    PlacementAck* ack(uint32_t actor_id) const {
        return base_ ? element<PlacementAck>(base_, layout_, Region::Acks, actor_id) : nullptr;
    }
    PlacementAck* helper_ack(uint32_t helper) const {
        uint32_t actor_id = 0;
        return actor(true, helper, &actor_id) ? ack(actor_id) : nullptr;
    }
    PlacementAck* rank_ack(uint32_t rank) const {
        uint32_t actor_id = 0;
        return actor(false, rank, &actor_id) ? ack(actor_id) : nullptr;
    }
    HelperDescriptor* helper(uint32_t index) const {
        return base_ ? element<HelperDescriptor>(base_, layout_, Region::Helpers, index) : nullptr;
    }
    PublishedServiceSlot* service(uint32_t helper_index) const {
        return base_ ? element<PublishedServiceSlot>(base_, layout_, Region::Services, helper_index) : nullptr;
    }
    PlacementControl* control() const {
        return base_ ? element<PlacementControl>(base_, layout_, Region::Placement, 0) : nullptr;
    }
    PlacementJournal* journal() const {
        return base_ ? element<PlacementJournal>(base_, layout_, Region::Journal, 0) : nullptr;
    }
    GenerationHeader* generation(uint32_t slot) const {
        return base_ ? element<GenerationHeader>(base_, layout_, Region::Generations, slot) : nullptr;
    }
    LaneSignal* lane(uint32_t id) const {
        return base_ ? element<LaneSignal>(base_, layout_, Region::Lanes, id) : nullptr;
    }
    Request* request(uint32_t id) const {
        return base_ ? element<Request>(base_, layout_, Region::Requests, id) : nullptr;
    }
    SlotDescriptor* slot(uint32_t id) const {
        return base_ ? element<SlotDescriptor>(base_, layout_, Region::Slots, id) : nullptr;
    }
    bool owner(uint32_t generation_slot, uint32_t layer, uint32_t expert, uint8_t* out) const {
        if (!base_ || !out || generation_slot >= kGenerationSlots ||
            layer >= header()->layer_count || expert >= header()->shape.experts) return false;
        const auto* g = generation(generation_slot);
        const uint64_t index = uint64_t(layer) * header()->shape.experts + expert;
        if (!g || index >= g->map_bytes) return false;
        *out = *(reinterpret_cast<const uint8_t*>(g) + sizeof(GenerationHeader) + index);
        return true;
    }
};

// The same resources the worker and the native P1 planner must use once gate D
// changes their fixed wire arrays. No compiled topology capacities or queues.
struct ResourcePlan {
    std::vector<uint8_t> opened; // dynamic lane count from validated segment
    uint32_t threads = 0, queued = 0;
    uint64_t lane_bytes = 0, pointer_bytes = 0, map_bytes = 0;
    uint64_t fixed_bytes = 0, expert_budget = 0, host_segment_bytes = 0;
};
inline Refusal resources(const Segment& segment, const worker_footprint::Shape& s,
                         const worker_footprint::SmallClasses& classes,
                         uint32_t large_rows, uint32_t helper_index,
                         bool power_save, uint64_t incoming_stride,
                         uint64_t card_bytes, uint64_t reserve_bytes,
                         ResourcePlan* out) {
    if (!out || !segment.attached() || helper_index >= segment.topology().helpers)
        return Refusal::InvalidRole;
    const auto& l = segment.layout();
    const auto t = segment.topology();
    const auto* header = segment.header();
    if (!s.layers || !s.experts || !s.hidden || !s.intermediate || !s.top_k ||
        !s.act_group || s.hidden % s.act_group || s.intermediate % s.act_group ||
        s.experts != header->shape.experts || s.hidden != header->shape.hidden ||
        s.intermediate != header->shape.intermediate || s.top_k != header->shape.top_k ||
        s.act_group != header->shape.act_group || s.layers != header->layer_count ||
        !large_rows || large_rows > header->shape.max_rows ||
        classes.count > kSmallClasses)
        return Refusal::InvalidShape;
    for (uint32_t c = 0; c < classes.count; ++c)
        if (!classes.rows[c] || classes.rows[c] > large_rows ||
            (c && classes.rows[c] <= classes.rows[c-1]))
            return Refusal::InvalidShape;
    ResourcePlan plan{};
    plan.opened.resize(l.counts.lanes);
    plan.host_segment_bytes = l.bytes;
    auto plus = [](uint64_t* value, uint64_t add) {
        return checked_add(*value, add, value);
    };
    auto product = [&](uint64_t* total, std::initializer_list<uint64_t> factors) {
        uint64_t term = 1;
        for (uint64_t f : factors)
            if (!checked_mul(term, f, &term)) return false;
        return plus(total, term);
    };
    for (uint32_t lane = 0; lane < l.counts.lanes; ++lane) {
        const uint32_t cls = lane / l.counts.slots;
        const uint32_t slot_index = lane % l.counts.slots;
        const uint32_t helper = slot_index / t.ranks;
        const bool mine = !power_save || helper == helper_index;
        if (!mine || (cls < kSmallClasses && cls >= classes.count)) continue;
        if (cls > kSmallClasses) return Refusal::InvalidShape;
        const uint64_t rows = cls < kSmallClasses ? classes.rows[cls] : large_rows;
        uint64_t routes = 0, bytes = 0;
        if (!checked_mul(rows, s.top_k, &routes)) return Refusal::ResourceOverflow;
        // Exact GPU Runner lane buffers (worker_footprint::lane_bytes), with
        // checked arithmetic rather than wrapping uint64_t in budget admission.
        if (!product(&bytes, {s.experts, 4}) ||
            !product(&bytes, {rows}) ||
            !product(&bytes, {s.experts, 4}) ||
            !product(&bytes, {rows, s.hidden, 2}) ||
            !product(&bytes, {routes, 4}) ||
            !product(&bytes, {routes, 4}) ||
            !product(&bytes, {s.experts + 1, 4}) ||
            !product(&bytes, {routes, 2}) ||
            !product(&bytes, {routes, 4}) ||
            !product(&bytes, {rows, s.hidden}) ||
            !product(&bytes, {rows, s.hidden / s.act_group, 4}) ||
            !product(&bytes, {routes, 2, s.intermediate, 2}) ||
            !product(&bytes, {routes, s.intermediate}) ||
            !product(&bytes, {routes, s.intermediate / s.act_group, 4}) ||
            !product(&bytes, {routes, s.hidden, 2}) ||
            !product(&bytes, {routes, 2, s.intermediate, 2}) ||
            !product(&bytes, {3, sizeof(uint64_t)}) ||
            !product(&bytes, {sizeof(uint32_t)}) ||
            !product(&bytes, {8, s.experts / 2, sizeof(void*)}) ||
            (cls == kSmallClasses &&
             !product(&bytes, {l.counts.chunks, sizeof(uint32_t)})) ||
            !plus(&plan.lane_bytes, bytes))
            return Refusal::ResourceOverflow;
        plan.opened[lane] = 1;
        ++plan.threads;
    }
    uint64_t pointer_per_layer = 0, per_gen = 0;
    if (!product(&pointer_per_layer, {worker_footprint::kDeviceLayerTablesBytes}) ||
        !product(&pointer_per_layer, {2, 4, s.experts / 2, sizeof(void*)}) ||
        !checked_mul(s.layers, pointer_per_layer, &per_gen) ||
        !checked_mul(per_gen, worker_tables::kGenerations, &plan.pointer_bytes) ||
        !checked_mul(s.layers, s.experts, &plan.map_bytes) ||
        !checked_mul(plan.map_bytes, sizeof(uint16_t), &plan.map_bytes))
        return Refusal::ResourceOverflow;
    plan.fixed_bytes = plan.lane_bytes;
    if (!plus(&plan.fixed_bytes, plan.pointer_bytes) ||
        !plus(&plan.fixed_bytes, plan.map_bytes) ||
        !plus(&plan.fixed_bytes, incoming_stride))
        return Refusal::ResourceOverflow;
    uint64_t queue_count = 0;
    if (!checked_mul(plan.threads, kLaneSlots, &queue_count) || queue_count > UINT32_MAX)
        return Refusal::ResourceOverflow;
    plan.queued = uint32_t(queue_count);
    if (card_bytes <= reserve_bytes || card_bytes - reserve_bytes <= plan.fixed_bytes)
        return Refusal::BudgetExceeded;
    plan.expert_budget = card_bytes - reserve_bytes - plan.fixed_bytes;
    *out = std::move(plan);
    return Refusal::None;
}

enum class Progress : uint8_t { Waiting, Advanced, Refused };

// CPU-only coordinator over the validated ACK/placement tables. The runtime
// publisher must still provide seqlock publication and deadline transport in
// gate D; this proof model deliberately does NOT attach to v25.
class Epoch {
    Segment& segment_;
    PlacementControl c_{};
    Refusal refusal_ = Refusal::None;
    // Single-writer seqlock publication to the counted v26 control region.
    bool publish() {
        auto* dst = segment_.control();
        if (!dst) return false;
        const uint32_t seq = __atomic_load_n(&dst->seq, __ATOMIC_ACQUIRE);
        if ((seq & 1u) || seq > UINT32_MAX - 2u) return false;
        __atomic_store_n(&dst->seq, seq + 1u, __ATOMIC_RELEASE);
        std::memcpy(reinterpret_cast<uint8_t*>(dst) + sizeof(uint32_t),
                    reinterpret_cast<const uint8_t*>(&c_) + sizeof(uint32_t),
                    sizeof(c_) - sizeof(uint32_t));
        __atomic_store_n(&dst->seq, seq + 2u, __ATOMIC_RELEASE);
        c_.seq = seq + 2u;
        return true;
    }
    bool cleanup_ = false;
    uint32_t all_helpers_ = 0, all_ranks_ = 0;
    Refusal fail(Refusal reason) {
        refusal_ = reason;
        c_.fault_code = uint32_t(reason);
        (void)publish();
        return reason;
    }
    bool identity(const PlacementAck& a, AckPhase phase, bool cleanup) const {
        return a.transaction_id == c_.transaction_id &&
            a.epoch == (cleanup ? c_.active_epoch : c_.pending_epoch) &&
            a.digest == (cleanup ? c_.active_digest : c_.pending_digest) &&
            a.generation_slot == (cleanup ? c_.active_slot : c_.pending_slot) &&
            a.phase == uint32_t(phase) && !a.fault_code;
    }
    Progress collect(AckPhase phase, PlacementState next) {
        if (refusal_ != Refusal::None) return Progress::Refused;
        uint32_t helpers = 0, ranks = 0;
        for (uint32_t actor = 0; actor < segment_.layout().counts.acks; ++actor) {
            const PlacementAck* a = segment_.ack(actor);
            const bool helper = actor < segment_.topology().helpers;
            const uint32_t index = helper ? actor : actor - segment_.topology().helpers;
            const uint32_t bit = uint32_t(1) << index;
            if (a->transaction_id != c_.transaction_id || a->phase != uint32_t(phase))
                continue; // late/old phase is not evidence
            if (!identity(*a, phase, false)) {
                fail(a->fault_code ? Refusal::ParticipantFault : Refusal::WrongAckIdentity);
                return Progress::Refused;
            }
            (helper ? helpers : ranks) |= bit;
        }
        if (helpers != all_helpers_ || ranks != all_ranks_) return Progress::Waiting;
        if (phase == AckPhase::QUIESCED) {
            // This is the explicit host-ledger / rank drain barrier, not a
            // device-counter shortcut. All participants must attest one cutoff.
            const uint64_t cutoff = segment_.ack(0)->drained_step;
            for (uint32_t actor = 0; actor < segment_.layout().counts.acks; ++actor)
                if (segment_.ack(actor)->drained_step != cutoff) {
                    fail(Refusal::Undrained);
                    return Progress::Refused;
                }
        }
        if (phase == AckPhase::COMMITTED) {
            const auto* g = segment_.generation(c_.pending_slot);
            if (!g || g->state != uint32_t(GenerationState::ACTIVE) ||
                g->epoch != c_.pending_epoch || g->digest != c_.pending_digest) {
                fail(Refusal::WrongAckIdentity);
                return Progress::Refused;
            }
            c_.active_slot = c_.pending_slot;
            c_.active_epoch = c_.pending_epoch;
            c_.active_digest = c_.pending_digest;
        }
        if (phase == AckPhase::RETIRED) cleanup_ = true;
        c_.state = uint32_t(next);
        if (!publish()) { fail(Refusal::SequenceExhausted); return Progress::Refused; }
        return Progress::Advanced;
    }
public:
    Epoch(Segment& segment, uint64_t active_digest) : segment_(segment) {
        if (!segment.attached() || !active_digest ||
            !role_masks(segment.topology(), &all_helpers_, &all_ranks_)) {
            fail(Refusal::InvalidSegment);
            return;
        }
        c_.state = uint32_t(PlacementState::ACTIVE);
        c_.active_epoch = 1;
        c_.active_digest = active_digest;
        c_.expected_helpers = all_helpers_;
        c_.expected_ranks = all_ranks_;
        if (!publish()) fail(Refusal::SequenceExhausted);
    }
    const PlacementControl& control() const { return c_; }
    Refusal refusal() const { return refusal_; }
    bool cleaned() const {
        if (!cleanup_) return true;
        for (uint32_t actor = 0; actor < segment_.layout().counts.acks; ++actor) {
            const bool helper = actor < segment_.topology().helpers;
            const AckPhase phase = !helper && actor == segment_.topology().helpers ?
                                   AckPhase::REOPENED : AckPhase::CLEANED;
            if (!identity(*segment_.ack(actor), phase, true)) return false;
        }
        return true;
    }
    bool begin(uint64_t digest) {
        if (refusal_ != Refusal::None || !cleaned() ||
            c_.state != uint32_t(PlacementState::ACTIVE) || !digest ||
            digest == c_.active_digest || c_.active_epoch == UINT64_MAX ||
            c_.transaction_id == UINT64_MAX) return false;
        ++c_.transaction_id;
        c_.pending_epoch = c_.active_epoch + 1;
        c_.pending_slot = 1u - c_.active_slot;
        c_.pending_digest = digest;
        c_.state = uint32_t(PlacementState::RESERVING);
        cleanup_ = false;
        if (!publish()) { fail(Refusal::SequenceExhausted); return false; }
        return true;
    }
    Progress step() {
        switch (PlacementState(c_.state)) {
            case PlacementState::RESERVING: return collect(AckPhase::RESERVED, PlacementState::COPYING);
            case PlacementState::COPYING: return collect(AckPhase::READY, PlacementState::PREPARED);
            case PlacementState::QUIESCING: return collect(AckPhase::QUIESCED, PlacementState::COMMIT);
            case PlacementState::COMMIT: return collect(AckPhase::COMMITTED, PlacementState::RETIRING);
            case PlacementState::RETIRING: return collect(AckPhase::RETIRED, PlacementState::ACTIVE);
            default: return Progress::Waiting;
        }
    }
    bool quiesce() {
        if (refusal_ != Refusal::None || c_.state != uint32_t(PlacementState::PREPARED))
            return false;
        c_.state = uint32_t(PlacementState::QUIESCING);
        if (!publish()) { fail(Refusal::SequenceExhausted); return false; }
        return true;
    }
};

} // namespace r3::moe_shm::v26::cpu
