#pragma once

// RCS14 gate D: counted, host-testable GPU wire and scratch arithmetic.
// The final launch sites must consume the same descriptors in the coordinated
// engine+helper cutover. No GPU can safely read an unvalidated v26 mapping.
#include "moe_remote_topology_v26.h"

namespace r3::moe_shm::v26::gpu {

// The device's per-pass published ticket is four uint32 words: 64-bit ticket
// identity, placement epoch and generation slot, published in this order.
struct TicketWords {
    uint32_t id_low, id_high, epoch, generation_slot;
};
static_assert(sizeof(TicketWords) == 4 * sizeof(uint32_t));

struct Scratch {
    uint32_t sequence_count;
    uint32_t slot_flags;          // one selected lane class per (helper, rank)
    uint32_t combine_word;        // separate; never aliases a slot flag
    uint32_t chunk_mirror_begin;  // one device word per distinct combine chunk
    uint32_t ticket_begin;        // complete four-word ticket follows mirrors
    uint32_t published_words;
    uint32_t consumed_records;    // one per lane plus a separate combine record
    uint32_t helper_words;        // per-block from/row_of scratch size
    uint64_t sequence_bytes, published_bytes, consumed_bytes, helper_bytes;
};

enum class Refusal : uint8_t {
    None, InvalidTopology, SizeOverflow, InvalidBlockLimit, InvalidRequest
};

inline Refusal scratch_for(Topology t, uint32_t rows,
                           Scratch* out) {
    if (!out) return Refusal::InvalidTopology;
    Counts counts{};
    if (counts_for(t, &counts) != v26::Refusal::None || !rows)
        return Refusal::InvalidTopology;

    // kCombineChunkRows is a protocol chunk width (not a topology bound).
    const uint64_t chunks = (uint64_t(rows) + kCombineChunkRows - 1u) /
                            kCombineChunkRows;
    const uint64_t flags = counts.slots;
    const uint64_t combine = flags;
    const uint64_t mirrors = flags + 1u;
    const uint64_t ticket = mirrors + chunks;
    const uint64_t published = ticket + sizeof(TicketWords) / sizeof(uint32_t);
    const uint64_t consumed = uint64_t(counts.lanes) + 1u;
    uint64_t sequence_bytes = 0, published_bytes = 0, consumed_bytes = 0;
    uint64_t helper_bytes = 0;
    if (published > UINT32_MAX || consumed > UINT32_MAX ||
        chunks > UINT32_MAX ||
        !checked_mul(counts.lanes, sizeof(uint32_t), &sequence_bytes) ||
        !checked_mul(published, sizeof(uint32_t), &published_bytes) ||
        !checked_mul(consumed, sizeof(ConsumedRecord), &consumed_bytes) ||
        !checked_mul(t.helpers, sizeof(uint32_t), &helper_bytes))
        return Refusal::SizeOverflow;
    *out = Scratch{counts.lanes, counts.slots, uint32_t(combine),
                   uint32_t(mirrors), uint32_t(ticket), uint32_t(published),
                   uint32_t(consumed), t.helpers, sequence_bytes,
                   published_bytes, consumed_bytes, helper_bytes};
    return Refusal::None;
}

// The finish pass must make all consumed records visible before the seal.
// Until GPU execution has a separately sequenced seal, one block is required:
// blocks cannot synchronize globally. Refuse if actual maxThreadsPerBlock
// cannot cover all used slots and the combine record.
struct FinishLaunch {
    uint32_t threads, blocks, work_items;
};
inline Refusal finish_launch(Topology t, uint32_t device_max_threads,
                             FinishLaunch* out) {
    Counts c{};
    if (!out || counts_for(t, &c) != v26::Refusal::None)
        return Refusal::InvalidTopology;
    if (!device_max_threads) return Refusal::InvalidBlockLimit;
    const uint64_t work = uint64_t(c.slots) + 1u;
    if (work > UINT32_MAX) return Refusal::SizeOverflow;
    if (work > device_max_threads) return Refusal::InvalidBlockLimit;
    *out = FinishLaunch{uint32_t(work), 1u, uint32_t(work)};
    return Refusal::None;
}

// Caller must supply a validated Layout and its matching Topology. These
// checks run before device publication, not on already-consumed requests.
inline bool request_identity_valid(const Request& q, const Layout& layout,
                                   const LayoutShape& shape, Topology t,
                                   uint32_t layers, uint32_t expected_lane) {
    Counts c{};
    if (counts_for(t, &c) != v26::Refusal::None ||
        layout.counts.slots != c.slots ||
        layout.counts.lanes != c.lanes ||
        q.slot >= c.slots || q.world != t.ranks ||
        q.helper >= t.helpers || q.consumer >= t.ranks ||
        q.slot != q.helper * t.ranks + q.consumer ||
        q.layer >= layers || q.layer % t.ranks != q.consumer ||
        !q.rows || q.rows > shape.max_rows || q.hidden != shape.hidden ||
        !q.sequence || q.generation_slot >= kGenerationSlots ||
        !q.placement_epoch || !shape.act_group ||
        q.lane != expected_lane ||
        q.lane >= c.lanes || q.lane % c.slots != q.slot ||
        q.lane_slot >= kLaneSlots ||
        q.lane_slot != (q.sequence - 1u) % kLaneSlots ||
        q.route_count > uint64_t(q.rows) * shape.top_k ||
        q.reserved)
        return false;

    const uint64_t slot_activation =
        element_offset(layout, Region::Activations, q.slot);
    const uint64_t rows_hidden = uint64_t(shape.max_rows) * shape.hidden;
    const uint64_t scale_stride =
        uint64_t(shape.hidden / shape.act_group) * sizeof(float);
    const uint64_t activation_q = slot_activation +
                                  rows_hidden * sizeof(uint16_t);
    const uint64_t activation_s = activation_q + rows_hidden;
    const uint64_t result_stride =
        uint64_t(shape.hidden) * sizeof(uint16_t);
    return shape.act_group && shape.hidden % shape.act_group == 0 &&
        result_stride <= UINT32_MAX &&
        scale_stride <= UINT32_MAX &&
        q.route_offset == element_offset(layout, Region::Routes, q.slot) &&
        q.activation_offset == slot_activation &&
        q.activation_q_offset == activation_q &&
        q.activation_s_offset == activation_s &&
        q.activation_stride == result_stride &&
        q.activation_q_stride == shape.hidden &&
        q.activation_s_stride == scale_stride &&
        q.result_offset == element_offset(layout, Region::ResultPlanes, q.slot) &&
        q.result_stride == result_stride;
}

// The producer and helper consume exactly the same per-slot offset contract.
inline bool populate_request(Request* q, const Layout& layout,
                             const LayoutShape& shape, Topology t,
                             uint32_t layer, uint32_t layers, uint32_t helper,
                             uint32_t lane_class, uint32_t lane_slot,
                             uint32_t rows, uint32_t route_count,
                             uint64_t placement_epoch,
                             uint32_t generation_slot,
                             uint32_t sequence) {
    Counts c{};
    uint32_t slot = 0, lane = 0;
    if (!q || counts_for(t, &c) != v26::Refusal::None ||
        !shape.act_group || shape.hidden % shape.act_group ||
        layer >= layers ||
        !role_slot(t, helper, layer % t.ranks, &slot) ||
        !role_lane(t, slot, lane_class, &lane) ||
        lane_slot >= kLaneSlots || !sequence ||
        lane_slot != (sequence - 1u) % kLaneSlots ||
        !placement_epoch || generation_slot >= kGenerationSlots ||
        !rows || rows > shape.max_rows ||
        uint64_t(route_count) > uint64_t(rows) * shape.top_k)
        return false;
    Request r{};
    r.sequence = sequence;
    r.layer = layer;
    r.slot = slot;
    r.world = t.ranks;
    r.rows = rows;
    r.hidden = shape.hidden;
    r.route_count = route_count;
    r.generation_slot = generation_slot;
    r.route_offset = element_offset(layout, Region::Routes, slot);
    r.activation_offset = element_offset(layout, Region::Activations, slot);
    const uint64_t rows_hidden = uint64_t(shape.max_rows) * shape.hidden;
    r.activation_q_offset = r.activation_offset + rows_hidden * sizeof(uint16_t);
    r.activation_s_offset = r.activation_q_offset + rows_hidden;
    r.activation_stride = shape.hidden * sizeof(uint16_t);
    r.activation_q_stride = shape.hidden;
    r.activation_s_stride = (shape.hidden / shape.act_group) * sizeof(float);
    r.result_offset = element_offset(layout, Region::ResultPlanes, slot);
    r.result_stride = shape.hidden * sizeof(uint16_t);
    r.consumer = layer % t.ranks;
    r.helper = helper;
    r.lane = lane;
    r.lane_slot = lane_slot;
    r.placement_epoch = placement_epoch;
    if (!request_identity_valid(r, layout, shape, t, layers, lane))
        return false;
    *q = r;
    return true;
}

} // namespace r3::moe_shm::v26::gpu
