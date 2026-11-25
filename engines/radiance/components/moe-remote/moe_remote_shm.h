#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cstring>
#include <ctime>
#include <initializer_list>
#include <type_traits>
#include <string_view>
#include <limits>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>


namespace r3::moe_shm {

// Protocol v26: every topology count is validated from the segment; only the
// ownership/combine masks, lane-class count and ring depth are representation
// constants. The old v25 offsets and fixed-capacity header do not exist.
constexpr uint64_t kMagic = 0x31454f4d33485253ull;
constexpr uint32_t kVersion = 26;
constexpr uint32_t kPage = 4096;
constexpr int kStartupWaitSeconds = 1800;
constexpr uint16_t kCombineFallback = 0x7E01;
constexpr uint32_t kCombineMaskBits = std::numeric_limits<uint8_t>::digits; // typed uint8_t per-row contributor mask
constexpr uint32_t kSmallClasses = 4;
constexpr uint32_t kClasses = kSmallClasses + 1;
constexpr uint32_t kLaneSlots = 4;
constexpr uint32_t kBurstStop = 0x80000000u;
constexpr uint32_t kParkDraining = 1u;
constexpr uint32_t kCombineChunkRows = 256;
constexpr uint32_t kMaxTopK = 16;
constexpr uint32_t kMaxHeatLdsBytes = 64u * 1024u;
// Existing model fixture dimensions, not topology capacity.
constexpr uint32_t kHidden = 2560, kTopK = 10, kExperts = 512;
constexpr uint32_t kMaxRows = 4096;
constexpr uint32_t kMaxLayers = 128;
constexpr uint32_t kMaxRoutes = kMaxRows * kTopK;
constexpr uint32_t kActGroup = 128;
constexpr uint32_t kActivationBf16Bytes = kMaxRows * kHidden * sizeof(uint16_t);
constexpr uint32_t kActivationQBytes = kMaxRows * kHidden;
constexpr uint32_t kActivationScaleBytes = kMaxRows * (kHidden / kActGroup) * sizeof(float);

constexpr uint32_t slots_for(uint32_t helpers, uint32_t world) { return helpers * world; }
constexpr uint32_t slot_of(uint32_t helper, uint32_t consumer, uint32_t world) {
    return helper * world + consumer;
}
constexpr uint32_t slot_helper(uint32_t slot, uint32_t world) { return world == 1u ? slot : slot / world; }
constexpr uint32_t slot_consumer(uint32_t slot, uint32_t world) { return world == 1u ? 0u : slot % world; }
constexpr uint32_t consumer_of(uint32_t layer, uint32_t world) { return world > 1u ? layer % world : 0u; }
constexpr uint32_t result_index(uint32_t slot, uint32_t /*out_rank*/) { return slot; }
constexpr uint32_t combine_index(uint32_t consumer, uint32_t slots) { return slots + consumer; }
constexpr uint32_t lane_of(uint32_t slot, uint32_t cls, uint32_t slots) { return cls * slots + slot; }
constexpr uint32_t lane_class(uint32_t lane, uint32_t slots) { return slots ? lane / slots : UINT32_MAX; }
constexpr bool lane_is_small(uint32_t lane, uint32_t slots) {
    return lane_class(lane, slots) < kSmallClasses;
}
struct RouteRole {
    bool count_heat, mask_local, publish;
};
/* `consumer`: this rank is the layer's consumer, consumer_of(layer, world) == rank, decided once a pass (the publish kernel's
 * route loop runs rows x top-k x helpers times on one block, so nothing per route may cost more than the route itself). */
constexpr RouteRole route_role(bool consumer, bool in_range, bool helper_owned, bool shadow) {
    return RouteRole{in_range, in_range && helper_owned && !shadow, in_range && helper_owned && consumer};
}
/* RAD49 fused gather (moe_mailbox_inject): on the layer's consumer, helper k's row for token t joins the engine's fused
 * MoE gather + reducing write as one more sorted row, at the first of t's slots whose route helper k computed
 * (rmask[t*K+s] == k+1, as route-mark writes it). Those slots were masked out of the local sort, so the slot is unclaimed,
 * and two helpers never share one. -1: helper k has no route of t. One definition for the inject kernel and the card-free
 * exactly-once test. */
constexpr int32_t inject_slot(const uint32_t* rmask_row, uint32_t top_k, uint32_t helper) {
    for (uint32_t s = 0; s < top_k; ++s)
        if (rmask_row[s] == helper + 1u) return (int32_t)s;
    return -1;
}
/* The router weight the injected slot takes: bf16 1.0, so the gather adds the helper's already weighted row unchanged. */
constexpr uint16_t kInjectWeightBf16 = 0x3F80;
struct Route {
    int32_t token;       /* activation/result row to which this expert contribution belongs. */
    int32_t expert;      /* global expert id; addresses the .rad expert group directly. */
    uint16_t weight_bf16;/* router weight exactly as supplied by the patch ABI. */
    uint16_t slot;        /* original top-k slot; gather arithmetic is defined in slot order. */
};
static_assert(sizeof(Route) == 12);

constexpr uint8_t kOwnerLocal = 0xff;

enum class PlacementState : uint32_t {
    ACTIVE=0, PROPOSED=1, RESERVING=2, COPYING=3,
    PREPARED=4, QUIESCING=5, COMMIT=6, RETIRING=7
};
enum class AckPhase : uint32_t { NONE=0, RESERVED=1, READY=2, QUIESCED=3, COMMITTED=4, RETIRED=5, REOPENED=6, CLEANED=7 };
enum class GenerationState : uint32_t { EMPTY=0, STAGING=1, READY=2, ACTIVE=3, RETIRED=4 };

struct alignas(64) PlacementControl {
    uint32_t seq;
    uint32_t state;
    uint32_t active_slot;
    uint32_t pending_slot;
    uint64_t active_epoch;
    uint64_t pending_epoch;
    uint64_t active_digest;
    uint64_t pending_digest;
    uint64_t transaction_id;
    uint64_t deadline_ns;
    uint32_t ready_helpers;
    uint32_t ready_ranks;
    uint32_t quiesced_helpers;
    uint32_t quiesced_ranks;
    uint32_t commit_helpers;
    uint32_t commit_ranks;
    uint32_t retired_helpers;
    uint32_t retired_ranks;
    uint32_t fault_code;
    uint32_t fault_actor;
    uint32_t notify;
    uint32_t expected_helpers;
    uint32_t expected_ranks;
    uint32_t journal_index;
    uint32_t reserved[34];
};
struct alignas(64) GenerationHeader {
    uint64_t epoch;
    uint64_t digest;
    uint32_t map_bytes;
    uint32_t layer_count;
    uint32_t expert_count;
    uint32_t slot;
    uint32_t ready_helpers;
    uint32_t ready_ranks;
    uint32_t state;
    uint32_t reserved[5];
};
struct alignas(64) PlacementAck {
    uint64_t transaction_id;
    uint64_t epoch;
    uint64_t digest;
    uint64_t drained_step;
    uint32_t phase;
    uint32_t generation_slot;
    uint32_t fault_code;
    uint32_t reserved32;
    uint64_t reserved[2];
};
/* Each helper's CPU epoch thread is the sole writer of one coherent two-lane
 * service snapshot. These slots occupy the previously unused ACK page tail.
 * The engine/other helpers never write a foreign helper's slot. */
struct alignas(64) PublishedServiceSlot {
    uint64_t sequence;       /* odd while the helper publishes */
    uint64_t published_ns;   /* CLOCK_MONOTONIC, for rejecting stalled helpers */
    uint64_t samples[2];     /* small and large lanes, never mixed */
    uint64_t rows[2];
    uint64_t time_ns[2];
};
static_assert(sizeof(PublishedServiceSlot)==64);

struct alignas(64) JournalRecord {
    uint64_t transaction_id;
    uint64_t before_epoch;
    uint64_t after_epoch;
    uint64_t before_digest;
    uint64_t candidate_digest;
    uint64_t start_ns;
    uint64_t commit_ns;
    uint64_t finish_ns;
    uint32_t state;
    uint32_t fault;
    uint32_t old_slot;
    uint32_t new_slot;
    uint32_t ready_helpers;
    uint32_t ready_ranks;
    uint32_t commit_helpers;
    uint32_t commit_ranks;
    uint32_t retire_helpers;
    uint32_t retire_ranks;
    // Terminal placement-only cleanup diagnostic, distinct from a precommit
    // abort's original fault; these occupy previously reserved journal words.
    uint32_t cleanup_fault;
    uint32_t cleanup_actor;
    uint32_t reserved[4];
};
struct alignas(64) PlacementJournal {
    JournalRecord entry[2];
};

/* One fixed request slot. Offsets are from the shm base so neither process exchanges pointers. */
struct alignas(64) LaneSignal {
    uint32_t request_sequence; /* engine GPU writes after route/activation data is visible. */
    uint32_t done_sequence;    /* helper GPU writes after the result is visible. */
    uint32_t park;             /* helper CPU: kParkDraining while it releases this lane's waits (RAD20, Header::burst). */
    uint32_t consumed_sequence;   /* engine GPU completion of last result-plane read */
    uint32_t done_slot;
    uint32_t consumed_slot;
    uint64_t done_epoch;
    uint64_t consumed_epoch;
    /* Engine closes admission and retires parents before requesting a drain.
     * Helpers ACK only after their queued waits/compute/combine work drained.
     * Closure IDs prevent an old park from certifying a newer quiescence. */
    uint64_t drain_request;
    uint64_t drain_ack;
    uint32_t reserved[2];
};
static_assert(sizeof(LaneSignal) == 64);
static_assert(offsetof(LaneSignal, drain_request) == 40 && offsetof(LaneSignal, drain_ack) == 48);

/* A helper ACK is a post-drain proof for exactly one closure. The callback
 * owns its stream/CPU-combine drain; failures and stale requests never ACK.
 * Kept card-free so cancellation and stale-proof behavior are unit tested. */
template<class Drain>
inline bool acknowledge_lane_drain(LaneSignal& lane, uint64_t closure, Drain&& drain) {
    if (!closure || __atomic_load_n(&lane.drain_request, __ATOMIC_ACQUIRE) != closure)
        return false;
    if (!drain() || __atomic_load_n(&lane.drain_request, __ATOMIC_ACQUIRE) != closure)
        return false;
    __atomic_store_n(&lane.drain_ack, closure, __ATOMIC_RELEASE);
    return true;
}

struct ConsumedRecord {
    uint32_t sequence;
    uint32_t slot;
    uint64_t epoch;
    uint32_t flushed;
    uint32_t reserved;
};
static_assert(sizeof(ConsumedRecord) == 24);
enum class CombineChunkWait : uint8_t { None, Poll, Mirror };
constexpr CombineChunkWait combine_chunk_wait(bool combined, uint32_t token, bool routed) {
    if (!combined) return CombineChunkWait::None;
    if (token % kCombineChunkRows == 0u) return CombineChunkWait::Poll;
    return routed ? CombineChunkWait::Mirror : CombineChunkWait::None;
}

struct LayoutShape {
    uint32_t hidden, top_k, experts, max_rows, act_group, intermediate;
};
constexpr bool operator==(const LayoutShape& a, const LayoutShape& b) {
    return a.hidden == b.hidden && a.top_k == b.top_k && a.experts == b.experts && a.max_rows == b.max_rows &&
           a.act_group == b.act_group && a.intermediate == b.intermediate;
}
constexpr bool operator!=(const LayoutShape& a, const LayoutShape& b) { return !(a == b); }
/* FNV-1a over the shape fields: both sides compute it from their own model and compare it first; the fields are compared too. */
constexpr uint64_t shape_hash(const LayoutShape& s) {
    uint64_t h = 1469598103934665603ull;
    const uint32_t v[] = {s.hidden, s.top_k, s.experts, s.max_rows, s.act_group, s.intermediate};
    for (uint32_t x : v)
        for (int i = 0; i < 4; ++i) { h ^= (x >> (8 * i)) & 0xffu; h *= 1099511628211ull; }
    return h;
}

/* FNV-1a 64 over explicitly little-endian fields followed by exactly the canonical map. */
inline uint64_t placement_digest(uint64_t model_shape_hash, uint32_t layers,
                                 uint32_t experts, uint32_t helpers, const uint8_t* owners) {
    uint64_t hash = 14695981039346656037ull;
    auto u8 = [&](uint8_t byte) { hash = (hash ^ byte) * 1099511628211ull; };
    auto u32 = [&](uint32_t value) {
        for (uint32_t i = 0; i != 4; ++i) u8((uint8_t)(value >> (i * 8)));
    };
    for (uint32_t i = 0; i != 8; ++i) u8((uint8_t)(model_shape_hash >> (i * 8)));
    u32(layers); u32(experts); u32(helpers);
    if (owners)
        for (uint64_t i = 0, n = (uint64_t)layers * experts; i < n; ++i) u8(owners[i]);
    return hash;
}


} // namespace r3::moe_shm

namespace r3::moe_shm::v26 {
using ::r3::moe_shm::kVersion;

// Compact nonzero wire identity of one owner-selected segment attempt: its unique name (/r3-serve-<instance uuid>,
// journaled by the owner before any helper starts) and whether it serves the measuring start, so a segment left by
// the measuring helpers is never taken for the serving one. Both the helper that creates the segment and the owner
// that reads its telemetry know exactly these two values; no manifest is involved. Like any 64-bit projection it
// cannot guarantee a collision-free identity.
inline uint64_t segment_identity(std::string_view shm_name, bool measuring) {
    if (shm_name.size() < 2 || shm_name.front() != '/') return 0;
    uint64_t hash = 14695981039346656037ull; // FNV-1a offset
    auto mix = [&hash](unsigned char byte) {
        hash ^= byte;
        hash *= 1099511628211ull; // FNV-1a prime
    };
    for (char c : shm_name) mix(static_cast<unsigned char>(c));
    mix(measuring ? 'm' : 's');
    return hash ? hash : 1ull;
}
constexpr uint32_t kGenerationSlots = 2; // active and candidate generations
enum class Refusal : uint8_t {
    None, NullOutput, ZeroTopology, SlotOverflow, LaneOverflow, RingOverflow,
    AckOverflow, ResultOverflow, ParticipantMask, OwnerId, CombineMask,
    InvalidShape, SizeOverflow, AddressOverflow, BadMapping, Unpublished,
    WrongVersion, WrongIdentity, CorruptDescriptor
};
struct Topology { uint32_t helpers, ranks; };
constexpr bool operator==(Topology a, Topology b) {
    return a.helpers == b.helpers && a.ranks == b.ranks;
}
struct Counts {
    uint32_t slots, lanes, ring_entries, acks, results, chunks;
    uint32_t chunk_sequences, chunk_completions, combine_inputs;
};

inline Refusal counts_for(Topology t, Counts* out) {
    if (!out) return Refusal::NullOutput;
    if (!t.helpers || !t.ranks) return Refusal::ZeroTopology;
    const uint64_t slots = uint64_t(t.helpers) * t.ranks;
    if (slots > UINT32_MAX) return Refusal::SlotOverflow;
    const uint64_t lanes = slots * kClasses;
    if (lanes > UINT32_MAX) return Refusal::LaneOverflow;
    const uint64_t ring = lanes * kLaneSlots;
    if (ring > UINT32_MAX) return Refusal::RingOverflow;
    const uint64_t acks = uint64_t(t.helpers) + t.ranks;
    if (acks > UINT32_MAX) return Refusal::AckOverflow;
    const uint64_t results = slots + t.ranks;
    if (results > UINT32_MAX) return Refusal::ResultOverflow;
    // PlacementControl/Journal participant masks are uint32_t; never shift 32.
    // uint8_t ownership reserves 0xff for engine-local experts.
    if (t.helpers > uint32_t(kOwnerLocal)) return Refusal::OwnerId;
    // worker_combine still uses a uint8_t mask of per-row contributors.
    if (t.helpers > std::numeric_limits<uint8_t>::digits) return Refusal::CombineMask;
    if (acks > std::numeric_limits<uint32_t>::digits) return Refusal::ParticipantMask;
    *out = Counts{uint32_t(slots), uint32_t(lanes), uint32_t(ring),
                  uint32_t(acks), uint32_t(results), 0, 0, 0, uint32_t(slots)};
    return Refusal::None;
}
inline bool role_slot(Topology t, uint32_t helper, uint32_t rank, uint32_t* out) {
    Counts c{};
    if (!out || counts_for(t, &c) != Refusal::None || helper >= t.helpers || rank >= t.ranks) return false;
    *out = helper * t.ranks + rank;
    return true;
}
inline bool role_lane(Topology t, uint32_t slot, uint32_t cls, uint32_t* out) {
    Counts c{};
    if (!out || counts_for(t, &c) != Refusal::None || slot >= c.slots || cls >= kClasses) return false;
    *out = cls * c.slots + slot;
    return true;
}
inline bool ring_entry(Topology t, uint32_t lane, uint32_t sequence, uint32_t* out) {
    Counts c{};
    if (!out || counts_for(t, &c) != Refusal::None || lane >= c.lanes || !sequence) return false;
    *out = lane * kLaneSlots + (sequence - 1u) % kLaneSlots;
    return true;
}
inline bool ack_for_rank(Topology t, uint32_t rank, uint32_t* out) {
    Counts c{};
    if (!out || counts_for(t, &c) != Refusal::None || rank >= t.ranks) return false;
    *out = t.helpers + rank;
    return true;
}
inline bool result_for_slot(Topology t, uint32_t slot, uint32_t* out) {
    Counts c{};
    if (!out || counts_for(t, &c) != Refusal::None || slot >= c.slots) return false;
    *out = slot;
    return true;
}
inline bool combine_for_rank(Topology t, uint32_t rank, uint32_t* out) {
    Counts c{};
    if (!out || counts_for(t, &c) != Refusal::None || rank >= t.ranks) return false;
    *out = c.slots + rank; // dedicated output, never a second rank of a slot
    return true;
}
inline bool role_masks(Topology t, uint32_t* helper_mask, uint32_t* rank_mask) {
    Counts c{};
    if (!helper_mask || !rank_mask || counts_for(t, &c) != Refusal::None) return false;
    *helper_mask = uint32_t((uint64_t{1} << t.helpers) - 1);
    *rank_mask = uint32_t((uint64_t{1} << t.ranks) - 1);
    return true;
}

enum class Region : uint32_t {
    Lanes, Slots, Helpers, CombineSignals, ChunkDescriptors, CombineInputs, ChunkSequences,
    ChunkCompletions, Requests, Acks, Services, Placement, Journal,
    Generations, LayerMap, LayerMapReady, LayerMapConsumed, Routes, Activations, ResultPlanes, Count
};
constexpr size_t kRegions = static_cast<size_t>(Region::Count);
constexpr size_t index(Region r) { return static_cast<size_t>(r); }
struct Span {
    uint64_t offset, bytes;
    uint32_t count, stride;
};
// Every mapped table uses relative offsets (no process-local pointers in the
// wire). These counted views allow CPU and HIP readers to address the same
// independently validated layout. They are not compiled capacity arrays.
#if defined(__HIPCC__) || defined(__CUDACC__)
#define R3_MOE_WIRE_HD __host__ __device__
#else
#define R3_MOE_WIRE_HD
#endif
template<class T> struct WireTable {
    uint64_t relative;
    uint32_t count, stride;
    R3_MOE_WIRE_HD T& operator[](uint32_t i) const {
        return *reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(const_cast<WireTable*>(this)) +
                                     relative + uint64_t(i) * stride);
    }
};
template<class T> inline WireTable<T> wire_table(void* base, void* member, const Span& span) {
    const uint64_t member_offset = uint64_t(reinterpret_cast<uint8_t*>(member) -
                                             reinterpret_cast<uint8_t*>(base));
    return {span.offset - member_offset, span.count, span.stride};
}
// A field projection of a count-sized descriptor table, not another list.
template<class T> struct WireField {
    uint64_t relative;
    uint32_t count, stride, field;
    R3_MOE_WIRE_HD T& operator[](uint32_t i) const {
        return *reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(const_cast<WireField*>(this)) +
                                     relative + uint64_t(i) * stride + field);
    }
};
struct WireOffsets {
    uint64_t start;
    uint32_t count, stride;
    R3_MOE_WIRE_HD uint64_t operator[](uint32_t i) const { return start + uint64_t(i) * stride; }
};
struct WireValues {
    uint32_t value, count;
    R3_MOE_WIRE_HD uint32_t operator[](uint32_t) const { return value; }
};
template<class T> constexpr bool same(WireTable<T> a, WireTable<T> b) {
    return a.relative == b.relative && a.count == b.count && a.stride == b.stride;
}
template<class T> constexpr bool same(WireField<T> a, WireField<T> b) {
    return a.relative == b.relative && a.count == b.count &&
           a.stride == b.stride && a.field == b.field;
}
constexpr bool same(WireOffsets a, WireOffsets b) {
    return a.start == b.start && a.count == b.count && a.stride == b.stride;
}
constexpr bool same(WireValues a, WireValues b) {
    return a.value == b.value && a.count == b.count;
}
struct WireMatrix {
    uint64_t relative;
    uint32_t outer, inner;
    struct Row {
        uint32_t* p;
        R3_MOE_WIRE_HD uint32_t& operator[](uint32_t j) const { return p[j]; }
    };
    R3_MOE_WIRE_HD Row operator[](uint32_t i) const {
        return {reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(const_cast<WireMatrix*>(this)) +
                                             relative) + uint64_t(i) * inner};
    }
};
constexpr bool same(WireMatrix a, WireMatrix b) {
    return a.relative == b.relative && a.outer == b.outer && a.inner == b.inner;
}
constexpr bool operator==(Span a, Span b) {
    return a.offset == b.offset && a.bytes == b.bytes &&
           a.count == b.count && a.stride == b.stride;
}
// Variable-length tables and buffers always start after the fixed header.
struct alignas(64) SlotDescriptor {
    uint32_t helper, consumer;
    uint32_t route_bytes, activation_bytes, result_bytes;
    uint32_t reserved;
    uint64_t route, activation, result;
};
struct alignas(64) HelperDescriptor {
    uint32_t helper, pid;
    uint64_t start_ticks;
};
struct alignas(64) CombineDescriptor {
    uint32_t consumer, enabled, request, done;
    uint64_t inputs, chunk_sequences, chunk_completions;
    uint32_t min_rows, rows;
    uint32_t result_consumed_generation, result_consumed_slot;
    uint64_t result_consumed_epoch;
    WireField<uint32_t> lane, seq;
};
struct alignas(64) ChunkDescriptor {
    WireMatrix seq;
    WireTable<uint32_t> done;
};
struct alignas(64) CombineInput {
    uint32_t helper, consumer, lane, sequence;
};
// The v26 request keeps the v25 first cache line offsets used by GPU lane
// kernels (sequence, layer, slot, world, generation, route and activation).
// One validated full-width output plane replaces inline rank/plane arrays.
// Host reference, helper kernels and GPU publisher must switch together.
struct alignas(64) Request {
    uint32_t sequence, completion_sequence;
    int32_t status;
    uint32_t layer, slot, world, rows, hidden, route_count, generation_slot;
    uint64_t route_offset, activation_offset, activation_q_offset, activation_s_offset;
    uint32_t activation_stride, activation_q_stride, activation_s_stride, result_stride;
    uint64_t result_offset;
    uint32_t consumer, combine_generation, row_base, completion_slot;
    uint64_t trace[5];
    uint64_t stage_trace[3];
    uint64_t placement_epoch, completion_epoch;
    uint32_t helper, lane, lane_slot, reserved;
    // ENG05 debug stamps occupy the old 48-byte tail padding: the v26 wire
    // size, offsets, routing fields and helper contract are unchanged.
    // Written only for sequence % experiment.diagnostics.lanes.trace_level == 0.
    // All six stamps are engine GPU wall_clock64 ticks, not helper GPU ticks.
    uint64_t mailbox_trace[6];
};
static_assert(sizeof(Request) == 256 && alignof(Request) == 64);
static_assert(offsetof(Request, layer) == 12 &&
              offsetof(Request, slot) == 16 &&
              offsetof(Request, world) == 20 &&
              offsetof(Request, generation_slot) == 36 &&
              offsetof(Request, route_offset) == 40 &&
              offsetof(Request, activation_offset) == 48 &&
              offsetof(Request, activation_q_offset) == 56);
static_assert(offsetof(Request, mailbox_trace) == 208);
static_assert(offsetof(Request, result_offset) > 64 &&
              offsetof(Request, result_stride) > 64);
static_assert(std::is_standard_layout_v<Request> &&
              std::is_trivially_copyable_v<Request>);
static_assert(sizeof(SlotDescriptor) == 64 && sizeof(HelperDescriptor) == 64);
static_assert(sizeof(CombineDescriptor) == 128 && sizeof(CombineInput) == 64);
static_assert(sizeof(ChunkDescriptor) == 64);

struct Layout {
    Span regions[kRegions];
    Counts counts;
    uint64_t bytes;
    uint32_t route_bytes, activation_bytes, result_bytes, owner_map_bytes;
};
struct alignas(kPage) Header {
    uint64_t magic;
    uint32_t version, header_bytes;
    uint64_t segment_bytes;
    LayoutShape shape;
    uint32_t layer_count;
    Topology topology;
    Counts counts;
    uint64_t model_shape_hash, manifest_digest, identity_hash;
    Span regions[kRegions];
    uint32_t small_class_count;
    uint32_t small_class_rows[kSmallClasses];
    uint32_t next_sequence, publish_sequence, burst, helper_fault, notify;
    uint32_t bootstrap_filtered_helpers;
    // Runtime tables are counted relative-offset views, never inline capacity arrays.
    WireTable<LaneSignal> lane;
    WireTable<CombineDescriptor> combine;
    WireTable<ChunkDescriptor> chunks;
    WireTable<uint32_t> layer_map_ready, layer_map_consumed;
    WireField<uint64_t> route_offset, activation_offset;
    WireOffsets result_offset, generation_offset;
    WireField<int32_t> helper_pid;
    WireField<uint64_t> helper_start;
    WireValues route_slot_bytes, activation_slot_bytes, result_slot_bytes;
    uint32_t rank_count, slot_count, helper_count;
    uint32_t hidden, top_k, expert_count, max_rows, intermediate, activation_group;
    uint32_t owner_map_bytes, layer_map_slot_bytes, generation_slot_bytes;
    uint32_t placement_ack_count;
    uint32_t published_helpers; // release-published LAST, after every table
    uint32_t reserved[8];
};
static_assert(sizeof(Header) == kPage && alignof(Header) == kPage);
static_assert(std::is_standard_layout_v<Header> && std::is_trivially_copyable_v<Header>);

inline bool checked_add(uint64_t a, uint64_t b, uint64_t* out) {
    if (b > UINT64_MAX - a) return false;
    *out = a + b; return true;
}
inline bool checked_mul(uint64_t a, uint64_t b, uint64_t* out) {
    if (a && b > UINT64_MAX / a) return false;
    *out = a * b; return true;
}
inline bool checked_align(uint64_t n, uint64_t alignment, uint64_t* out) {
    uint64_t sum = 0;
    if (!alignment || (alignment & (alignment - 1)) || !checked_add(n, alignment - 1, &sum)) return false;
    *out = sum & ~(alignment - 1); return true;
}
inline uint64_t identity_hash(const LayoutShape& shape, Topology topology, uint64_t manifest_digest) {
    uint64_t hash = shape_hash(shape);
    for (uint64_t word : {uint64_t(topology.helpers), uint64_t(topology.ranks), manifest_digest}) {
        for (uint32_t j = 0; j != 8; ++j) {
            hash ^= (word >> (j * 8)) & 0xffu;
            hash *= 1099511628211ull;
        }
    }
    return hash;
}

inline Refusal layout_for(const LayoutShape& s, uint32_t layers, Topology t, Layout* out) {
    if (!out) return Refusal::NullOutput;
    Layout x{};
    if (const Refusal e = counts_for(t, &x.counts); e != Refusal::None) return e;
    if (!layers || !s.hidden || !s.top_k || !s.experts || !s.max_rows ||
        !s.act_group || !s.intermediate || s.top_k > s.experts ||
        s.hidden % s.act_group) return Refusal::InvalidShape;

    uint64_t routes, rows_hidden, groups, scalars, scale_bytes;
    uint64_t route_bytes, result_bytes, act_bytes, owner_bytes, generation_bytes, tmp;
    if (!checked_mul(s.max_rows, s.top_k, &routes) ||
        !checked_mul(routes, sizeof(Route), &route_bytes) ||
        !checked_mul(s.max_rows, s.hidden, &rows_hidden) ||
        !checked_mul(rows_hidden, sizeof(uint16_t), &result_bytes) ||
        !checked_mul(s.max_rows, s.hidden / s.act_group, &groups) ||
        !checked_mul(groups, sizeof(float), &scale_bytes) ||
        !checked_mul(rows_hidden, 3, &scalars) ||
        !checked_add(scalars, scale_bytes, &act_bytes) ||
        !checked_mul(layers, s.experts, &owner_bytes) ||
        !checked_add(owner_bytes, sizeof(GenerationHeader), &generation_bytes))
        return Refusal::SizeOverflow;
    uint64_t layer_map_bytes = 0;
    if (!checked_mul(layers, uint64_t(s.experts) * sizeof(int32_t), &layer_map_bytes))
        return Refusal::SizeOverflow;
    for (uint64_t n : {route_bytes, result_bytes, act_bytes, owner_bytes, layer_map_bytes})
        if (n > UINT32_MAX) return Refusal::SizeOverflow;
    x.route_bytes = uint32_t(route_bytes);
    x.result_bytes = uint32_t(result_bytes);
    x.activation_bytes = uint32_t(act_bytes);
    x.owner_map_bytes = uint32_t(owner_bytes);
    x.counts.chunks = uint32_t((uint64_t(s.max_rows) + kCombineChunkRows - 1) / kCombineChunkRows);
    if (uint64_t(t.helpers) * t.ranks * x.counts.chunks > UINT32_MAX ||
        uint64_t(t.ranks) * x.counts.chunks > UINT32_MAX) return Refusal::SizeOverflow;
    x.counts.chunk_sequences = t.helpers * t.ranks * x.counts.chunks;
    x.counts.chunk_completions = t.ranks * x.counts.chunks;

    uint64_t at = sizeof(Header);
    auto reserve = [&](Region which, uint32_t count, uint64_t unit, uint64_t alignment) -> bool {
        uint64_t start, stride, length, end;
        if (!count || !checked_align(at, alignment, &start) ||
            !checked_align(unit, alignment, &stride) || stride > UINT32_MAX ||
            !checked_mul(count, stride, &length) || !checked_add(start, length, &end) ||
            end > SIZE_MAX) return false;
        x.regions[index(which)] = Span{start, length, count, uint32_t(stride)};
        at = end;
        return true;
    };
    if (!reserve(Region::Lanes, x.counts.lanes, sizeof(LaneSignal), alignof(LaneSignal)) ||
        !reserve(Region::Slots, x.counts.slots, sizeof(SlotDescriptor), alignof(SlotDescriptor)) ||
        !reserve(Region::Helpers, t.helpers, sizeof(HelperDescriptor), alignof(HelperDescriptor)) ||
        !reserve(Region::CombineSignals, t.ranks, sizeof(CombineDescriptor), alignof(CombineDescriptor)) ||
        !reserve(Region::ChunkDescriptors, t.ranks, sizeof(ChunkDescriptor), alignof(ChunkDescriptor)) ||
        !reserve(Region::CombineInputs, x.counts.combine_inputs, sizeof(CombineInput), alignof(CombineInput)) ||
        !reserve(Region::ChunkSequences, x.counts.chunk_sequences, sizeof(uint32_t), alignof(uint32_t)) ||
        !reserve(Region::ChunkCompletions, x.counts.chunk_completions, sizeof(uint32_t), alignof(uint32_t)) ||
        !reserve(Region::Requests, x.counts.ring_entries, sizeof(Request), alignof(Request)) ||
        !reserve(Region::Acks, x.counts.acks, sizeof(PlacementAck), alignof(PlacementAck)) ||
        !reserve(Region::Services, t.helpers, sizeof(PublishedServiceSlot), alignof(PublishedServiceSlot)) ||
        !reserve(Region::Placement, 1, sizeof(PlacementControl), alignof(PlacementControl)) ||
        !reserve(Region::Journal, 1, sizeof(PlacementJournal), alignof(PlacementJournal)) ||
        !reserve(Region::Generations, kGenerationSlots, generation_bytes, kPage) ||
        !reserve(Region::LayerMap, layers, uint64_t(s.experts) * sizeof(int32_t), alignof(int32_t)) ||
        !reserve(Region::LayerMapReady, layers, sizeof(uint32_t), alignof(uint32_t)) ||
        !reserve(Region::LayerMapConsumed, layers, sizeof(uint32_t), alignof(uint32_t)) ||
        !reserve(Region::Routes, x.counts.slots, route_bytes, kPage) ||
        !reserve(Region::Activations, t.ranks, act_bytes, kPage) ||
        !reserve(Region::ResultPlanes, x.counts.results, result_bytes, kPage) ||
        !checked_align(at, kPage, &tmp) || tmp > SIZE_MAX) return Refusal::AddressOverflow;
    x.bytes = tmp;
    *out = x;
    return Refusal::None;
}
inline uint64_t element_offset(const Layout& l, Region region, uint32_t n) {
    const Span& span = l.regions[index(region)];
    return span.offset + uint64_t(n) * span.stride; // only after layout_for/valid, n<count
}
template <class T> inline T* element(void* base, const Layout& l, Region region, uint32_t n) {
    if (!base || n >= l.regions[index(region)].count) return nullptr;
    return reinterpret_cast<T*>(static_cast<uint8_t*>(base) + element_offset(l, region, n));
}
template <class T> inline const T* element(const void* base, const Layout& l, Region region, uint32_t n) {
    if (!base || n >= l.regions[index(region)].count) return nullptr;
    return reinterpret_cast<const T*>(static_cast<const uint8_t*>(base) + element_offset(l, region, n));
}

inline void bind_header_views(Header* h, const Layout& l, Topology t, const LayoutShape& s) {
    void* base = h;
    auto span = [&](Region r) -> const Span& { return l.regions[index(r)]; };
    auto rel = [&](void* member, uint64_t absolute) -> uint64_t {
        return absolute - uint64_t(reinterpret_cast<uint8_t*>(member) -
                                   reinterpret_cast<uint8_t*>(base));
    };
    h->lane = wire_table<LaneSignal>(base, &h->lane, span(Region::Lanes));
    h->combine = wire_table<CombineDescriptor>(base, &h->combine, span(Region::CombineSignals));
    h->chunks = wire_table<ChunkDescriptor>(base, &h->chunks, span(Region::ChunkDescriptors));
    h->layer_map_ready = wire_table<uint32_t>(base, &h->layer_map_ready, span(Region::LayerMapReady));
    h->layer_map_consumed = wire_table<uint32_t>(base, &h->layer_map_consumed, span(Region::LayerMapConsumed));
    h->route_offset = {rel(&h->route_offset, span(Region::Slots).offset),
                       l.counts.slots, span(Region::Slots).stride, offsetof(SlotDescriptor, route)};
    h->activation_offset = {rel(&h->activation_offset, span(Region::Slots).offset),
                            l.counts.slots, span(Region::Slots).stride, offsetof(SlotDescriptor, activation)};
    h->result_offset = {span(Region::ResultPlanes).offset,
                        span(Region::ResultPlanes).count, span(Region::ResultPlanes).stride};
    h->generation_offset = {span(Region::Generations).offset,
                            span(Region::Generations).count, span(Region::Generations).stride};
    h->helper_pid = {rel(&h->helper_pid, span(Region::Helpers).offset),
                     t.helpers, span(Region::Helpers).stride, offsetof(HelperDescriptor, pid)};
    h->helper_start = {rel(&h->helper_start, span(Region::Helpers).offset),
                       t.helpers, span(Region::Helpers).stride, offsetof(HelperDescriptor, start_ticks)};
    h->route_slot_bytes = {l.route_bytes, l.counts.slots};
    h->activation_slot_bytes = {l.activation_bytes, l.counts.slots};
    h->result_slot_bytes = {l.result_bytes, l.counts.results};
    h->helper_count = t.helpers;
    h->rank_count = t.ranks;
    h->slot_count = l.counts.slots;
    h->hidden = s.hidden;
    h->top_k = s.top_k;
    h->expert_count = s.experts;
    h->max_rows = s.max_rows;
    h->intermediate = s.intermediate;
    h->activation_group = s.act_group;
    h->owner_map_bytes = l.owner_map_bytes;
    h->layer_map_slot_bytes = uint32_t(span(Region::LayerMap).bytes);
    h->generation_slot_bytes = span(Region::Generations).stride;
    h->placement_ack_count = l.counts.acks;
}
inline void bind_combine_views(void* base, const Layout& l, Topology t, uint32_t rank) {
    auto* c = element<CombineDescriptor>(base, l, Region::CombineSignals, rank);
    auto* d = element<ChunkDescriptor>(base, l, Region::ChunkDescriptors, rank);
    const Span& inputs = l.regions[index(Region::CombineInputs)];
    [[maybe_unused]] const Span &sequences =
        l.regions[index(Region::ChunkSequences)];
    const Span& completions = l.regions[index(Region::ChunkCompletions)];
    auto absolute = [&](const void* p, uint64_t region_offset) {
        return region_offset - uint64_t(reinterpret_cast<const uint8_t*>(p) -
                                        reinterpret_cast<const uint8_t*>(base));
    };
    const uint64_t first_input = element_offset(l, Region::CombineInputs, rank * t.helpers);
    c->lane = {absolute(&c->lane, first_input), t.helpers, inputs.stride,
               offsetof(CombineInput, lane)};
    c->seq = {absolute(&c->seq, first_input), t.helpers, inputs.stride,
              offsetof(CombineInput, sequence)};
    d->seq = {absolute(&d->seq, element_offset(l, Region::ChunkSequences,
                                              rank * t.helpers * l.counts.chunks)),
              t.helpers, l.counts.chunks};
    d->done = {absolute(&d->done, element_offset(l, Region::ChunkCompletions,
                                                rank * l.counts.chunks)),
               l.counts.chunks, completions.stride};
}

inline Refusal init(void* base, size_t mapped, const LayoutShape& shape, uint32_t layers,
                    Topology t, uint64_t manifest_digest, bool defer_publication = false) {
    Layout l{};
    const Refusal e = layout_for(shape, layers, t, &l);
    if (e != Refusal::None) return e;
    if (!base || reinterpret_cast<uintptr_t>(base) % kPage || mapped < l.bytes)
        return Refusal::BadMapping;
    if (!manifest_digest) return Refusal::WrongIdentity;
    std::memset(base, 0, size_t(l.bytes));
    auto* h = static_cast<Header*>(base);
    h->magic = kMagic; h->version = kVersion; h->header_bytes = sizeof(Header);
    h->segment_bytes = l.bytes; h->shape = shape; h->layer_count = layers;
    h->topology = t; h->counts = l.counts;
    h->model_shape_hash = shape_hash(shape); h->manifest_digest = manifest_digest;
    h->identity_hash = identity_hash(shape, t, manifest_digest);
    for (size_t i = 0; i < kRegions; ++i) h->regions[i] = l.regions[i];
    bind_header_views(h, l, t, shape);
    h->small_class_count = 1;
    h->small_class_rows[0] = shape.max_rows < 8u ? shape.max_rows : 8u; // bootstrap default
    h->next_sequence = 1; // sequence zero is never a published request
    std::memset(static_cast<uint8_t*>(base) + l.regions[index(Region::LayerMap)].offset,
                0xff, size_t(l.regions[index(Region::LayerMap)].bytes));
    for (uint32_t slot = 0; slot < l.counts.slots; ++slot) {
        auto* d = element<SlotDescriptor>(base, l, Region::Slots, slot);
        d->helper = slot / t.ranks; d->consumer = slot % t.ranks;
        d->route = element_offset(l, Region::Routes, slot);
        // Every helper of a consumer shares the one activation staging area.
        d->activation = element_offset(l, Region::Activations, slot % t.ranks);
        d->result = element_offset(l, Region::ResultPlanes, slot);
        d->route_bytes = l.route_bytes; d->activation_bytes = l.activation_bytes;
        d->result_bytes = l.result_bytes;
    }
    for (uint32_t helper = 0; helper < t.helpers; ++helper)
        element<HelperDescriptor>(base, l, Region::Helpers, helper)->helper = helper;
    for (uint32_t rank = 0; rank < t.ranks; ++rank) {
        auto* c = element<CombineDescriptor>(base, l, Region::CombineSignals, rank);
        c->consumer = rank;
        c->inputs = element_offset(l, Region::CombineInputs, rank * t.helpers);
        c->chunk_sequences = element_offset(l, Region::ChunkSequences, rank * t.helpers * l.counts.chunks);
        c->chunk_completions = element_offset(l, Region::ChunkCompletions, rank * l.counts.chunks);
        bind_combine_views(base, l, t, rank);
        for (uint32_t helper = 0; helper < t.helpers; ++helper) {
            auto* i = element<CombineInput>(base, l, Region::CombineInputs, rank * t.helpers + helper);
            i->helper = helper; i->consumer = rank;
        }
    }
    for (uint32_t gen = 0; gen < kGenerationSlots; ++gen) {
        auto* g = element<GenerationHeader>(base, l, Region::Generations, gen);
        g->slot = gen; g->layer_count = layers; g->expert_count = shape.experts;
        g->map_bytes = l.owner_map_bytes;
        g->state = uint32_t(gen == 0 ? GenerationState::ACTIVE : GenerationState::EMPTY);
        std::memset(reinterpret_cast<uint8_t*>(g) + sizeof(GenerationHeader), kOwnerLocal, l.owner_map_bytes);
    }
    auto* active = element<GenerationHeader>(base, l, Region::Generations, 0);
    active->epoch = 1;
    active->digest = placement_digest(shape_hash(shape), layers, shape.experts, t.helpers,
                                      reinterpret_cast<uint8_t*>(active) + sizeof(GenerationHeader));
    auto* control = element<PlacementControl>(base, l, Region::Placement, 0);
    control->state = uint32_t(PlacementState::ACTIVE);
    control->active_slot = 0;
    control->pending_slot = UINT32_MAX;
    control->active_epoch = active->epoch;
    control->active_digest = active->digest;
    uint32_t helper_mask = 0, rank_mask = 0;
    if (!role_masks(t, &helper_mask, &rank_mask)) return Refusal::ParticipantMask;
    control->expected_helpers = helper_mask;
    // An attacher cannot validate before the last release store.
    if (!defer_publication)
        __atomic_store_n(&h->published_helpers, t.helpers, __ATOMIC_RELEASE);
    return Refusal::None;
}
// Caller invokes only after all custom lane classes and launch policy fields
// have been populated. No waiter uses the early descriptive helper_count.
inline void publish(Header* h) {
    __atomic_store_n(&h->published_helpers, h->topology.helpers, __ATOMIC_RELEASE);
}
inline Refusal valid(const void* base, size_t mapped, const LayoutShape& expected_shape,
                     uint32_t expected_layers, Topology expected_topology,
                     uint64_t expected_manifest, Layout* accepted = nullptr) {
    if (!base || reinterpret_cast<uintptr_t>(base) % kPage || mapped < sizeof(Header))
        return Refusal::BadMapping;
    const auto* h = static_cast<const Header*>(base);
    const uint32_t published = __atomic_load_n(&h->published_helpers, __ATOMIC_ACQUIRE);
    if (!published) return Refusal::Unpublished;
    if (h->magic != kMagic || h->version != kVersion || h->header_bytes != sizeof(Header))
        return Refusal::WrongVersion;
    if (published != h->topology.helpers || !(h->topology == expected_topology) ||
        h->layer_count != expected_layers || h->shape != expected_shape ||
        !expected_manifest || h->manifest_digest != expected_manifest ||
        h->model_shape_hash != shape_hash(expected_shape) ||
        h->identity_hash != identity_hash(expected_shape, expected_topology, expected_manifest))
        return Refusal::WrongIdentity;
    if (!h->small_class_count || h->small_class_count > kSmallClasses)
        return Refusal::CorruptDescriptor;
    for (uint32_t c = 0; c < h->small_class_count; ++c)
        if (!h->small_class_rows[c] || h->small_class_rows[c] > expected_shape.max_rows ||
            (c && h->small_class_rows[c] <= h->small_class_rows[c - 1]))
            return Refusal::CorruptDescriptor;
    uint32_t helper_mask = 0, rank_mask = 0;
    if (!role_masks(expected_topology, &helper_mask, &rank_mask) ||
        (h->bootstrap_filtered_helpers & ~helper_mask))
        return Refusal::CorruptDescriptor;
    for (uint32_t r : h->reserved) if (r) return Refusal::CorruptDescriptor;
    Layout l{};
    const Refusal e = layout_for(expected_shape, expected_layers, expected_topology, &l);
    if (e != Refusal::None) return e;
    if (mapped < l.bytes) return Refusal::BadMapping;
    const auto* control = element<PlacementControl>(base, l, Region::Placement, 0);
    if (!control || control->expected_helpers != helper_mask ||
        control->active_slot >= kGenerationSlots ||
        (control->pending_slot >= kGenerationSlots && control->pending_slot != UINT32_MAX) ||
        !control->active_epoch || control->state > uint32_t(PlacementState::RETIRING))
        return Refusal::CorruptDescriptor;
    const auto* active = element<GenerationHeader>(base, l, Region::Generations,
                                                    control->active_slot);
    if (!active || active->state != uint32_t(GenerationState::ACTIVE) ||
        active->epoch != control->active_epoch ||
        !active->digest || active->digest != control->active_digest)
        return Refusal::CorruptDescriptor;
    if (h->segment_bytes != l.bytes || std::memcmp(&h->counts, &l.counts, sizeof(Counts)) != 0)
        return Refusal::CorruptDescriptor;
    for (size_t i = 0; i < kRegions; ++i)
        if (!(h->regions[i] == l.regions[i])) return Refusal::CorruptDescriptor;
    Header expected{};
    bind_header_views(&expected, l, expected_topology, expected_shape);
    if (!same(h->lane, expected.lane) || !same(h->combine, expected.combine) ||
        !same(h->chunks, expected.chunks) ||
        !same(h->layer_map_ready, expected.layer_map_ready) ||
        !same(h->layer_map_consumed, expected.layer_map_consumed) ||
        !same(h->route_offset, expected.route_offset) ||
        !same(h->activation_offset, expected.activation_offset) ||
        !same(h->result_offset, expected.result_offset) ||
        !same(h->generation_offset, expected.generation_offset) ||
        !same(h->helper_pid, expected.helper_pid) ||
        !same(h->helper_start, expected.helper_start) ||
        !same(h->route_slot_bytes, expected.route_slot_bytes) ||
        !same(h->activation_slot_bytes, expected.activation_slot_bytes) ||
        !same(h->result_slot_bytes, expected.result_slot_bytes) ||
        h->helper_count != expected_topology.helpers ||
        h->rank_count != expected_topology.ranks || h->slot_count != l.counts.slots ||
        h->hidden != expected_shape.hidden || h->top_k != expected_shape.top_k ||
        h->expert_count != expected_shape.experts || h->max_rows != expected_shape.max_rows ||
        h->intermediate != expected_shape.intermediate ||
        h->activation_group != expected_shape.act_group ||
        h->owner_map_bytes != l.owner_map_bytes ||
        h->layer_map_slot_bytes != l.regions[index(Region::LayerMap)].bytes ||
        h->generation_slot_bytes != l.regions[index(Region::Generations)].stride ||
        h->placement_ack_count != l.counts.acks)
        return Refusal::CorruptDescriptor;
    const Topology t = expected_topology;
    for (uint32_t slot = 0; slot < l.counts.slots; ++slot) {
        const auto* d = element<SlotDescriptor>(base, l, Region::Slots, slot);
        if (d->helper != slot / t.ranks || d->consumer != slot % t.ranks || d->reserved ||
            d->route != element_offset(l, Region::Routes, slot) ||
            d->activation != element_offset(l, Region::Activations, slot % t.ranks) ||
            d->result != element_offset(l, Region::ResultPlanes, slot) ||
            d->route_bytes != l.route_bytes || d->activation_bytes != l.activation_bytes ||
            d->result_bytes != l.result_bytes) return Refusal::CorruptDescriptor;
    }
    for (uint32_t helper = 0; helper < t.helpers; ++helper)
        if (element<HelperDescriptor>(base, l, Region::Helpers, helper)->helper != helper)
            return Refusal::CorruptDescriptor;
    for (uint32_t rank = 0; rank < t.ranks; ++rank) {
        const auto* c = element<CombineDescriptor>(base, l, Region::CombineSignals, rank);
        if (c->consumer != rank ||
            c->inputs != element_offset(l, Region::CombineInputs, rank * t.helpers) ||
            c->chunk_sequences != element_offset(l, Region::ChunkSequences, rank * t.helpers * l.counts.chunks) ||
            c->chunk_completions != element_offset(l, Region::ChunkCompletions, rank * l.counts.chunks))
            return Refusal::CorruptDescriptor;
        const uint64_t first_input = element_offset(l, Region::CombineInputs, rank * t.helpers);
        const auto rel = [&](const void* member, uint64_t target) {
            return target - uint64_t(reinterpret_cast<const uint8_t*>(member) -
                                      reinterpret_cast<const uint8_t*>(base));
        };
        const WireField<uint32_t> lane_expected{rel(&c->lane, first_input), t.helpers,
            l.regions[index(Region::CombineInputs)].stride, offsetof(CombineInput, lane)};
        const WireField<uint32_t> seq_expected{rel(&c->seq, first_input), t.helpers,
            l.regions[index(Region::CombineInputs)].stride, offsetof(CombineInput, sequence)};
        const auto* d = element<ChunkDescriptor>(base, l, Region::ChunkDescriptors, rank);
        const WireMatrix chunks_expected{rel(&d->seq, c->chunk_sequences), t.helpers, l.counts.chunks};
        const WireTable<uint32_t> done_expected{
            rel(&d->done, c->chunk_completions), l.counts.chunks,
            l.regions[index(Region::ChunkCompletions)].stride};
        if (!same(c->lane, lane_expected) || !same(c->seq, seq_expected) ||
            !same(d->seq, chunks_expected) || !same(d->done, done_expected))
            return Refusal::CorruptDescriptor;
        for (uint32_t helper = 0; helper < t.helpers; ++helper) {
            const auto* i = element<CombineInput>(base, l, Region::CombineInputs, rank * t.helpers + helper);
            if (i->helper != helper || i->consumer != rank) return Refusal::CorruptDescriptor;
        }
    }
    for (uint32_t gen = 0; gen < kGenerationSlots; ++gen) {
        const auto* g = element<GenerationHeader>(base, l, Region::Generations, gen);
        if (g->slot != gen || g->layer_count != expected_layers ||
            g->expert_count != expected_shape.experts || g->map_bytes != l.owner_map_bytes ||
            g->state > uint32_t(GenerationState::RETIRED) ||
            (g->ready_helpers & ~helper_mask))
            return Refusal::CorruptDescriptor;
        for (uint32_t reserved : g->reserved)
            if (reserved) return Refusal::CorruptDescriptor;
        if (g->state == uint32_t(GenerationState::EMPTY) && (g->epoch || g->digest))
            return Refusal::CorruptDescriptor;
        if (gen == control->active_slot) {
            const auto* owners = reinterpret_cast<const uint8_t*>(g) + sizeof(GenerationHeader);
            for (uint32_t j = 0; j < g->map_bytes; ++j) {
                const uint8_t owner = __atomic_load_n(owners + j, __ATOMIC_ACQUIRE);
                if (owner != kOwnerLocal && owner >= t.helpers)
                    return Refusal::CorruptDescriptor;
            }
            if (g->ready_helpers == helper_mask &&
                placement_digest(h->model_shape_hash, expected_layers, expected_shape.experts,
                                 t.helpers, owners) != g->digest)
                return Refusal::CorruptDescriptor;
        }
    }
    if (accepted) *accepted = l;
    return Refusal::None;
}


} // namespace r3::moe_shm::v26


namespace r3::moe_shm {
using v26::Header;
using v26::Request;
using v26::Layout;
using v26::Topology;
using v26::Counts;
using v26::Refusal;
using v26::Region;
using v26::SlotDescriptor;
using v26::HelperDescriptor;
using v26::CombineInput;
using v26::CombineDescriptor;
using v26::ChunkDescriptor;
using CombineSignal = v26::CombineDescriptor;
using CombineChunks = v26::ChunkDescriptor;
inline bool slots_fit(uint32_t helpers, uint32_t world) {
    v26::Counts c{};
    return v26::counts_for({helpers, world}, &c) == v26::Refusal::None;
}

constexpr bool request_lane_identity_valid(uint32_t slot, uint32_t world, uint32_t layer,
                                           uint32_t lane_slot, uint32_t segment_world,
                                           uint32_t segment_slots) {
    return segment_world && world == segment_world && slot == lane_slot &&
           slot < segment_slots && slot_consumer(slot, world) == consumer_of(layer, world);
}
constexpr bool request_slot_valid(uint32_t slot, uint32_t world, uint32_t consumer,
                                  uint32_t layer, uint32_t lane_slot,
                                  uint32_t segment_world, uint32_t segment_slots) {
    return request_lane_identity_valid(slot, world, layer, lane_slot,
                                       segment_world, segment_slots) &&
           consumer == slot_consumer(slot, world);
}
inline bool valid_self(const Header* h, size_t mapped) {
    return h && v26::valid(h, mapped, h->shape, h->layer_count, h->topology,
                           h->manifest_digest) == Refusal::None;
}
constexpr bool combine_shape_wanted(bool asked, uint32_t helpers, uint32_t cls,
                                    bool helper0_routes, bool other_routes) {
    return asked && helpers >= 2 && helpers <= std::numeric_limits<uint8_t>::digits &&
           cls == kSmallClasses && helper0_routes && other_routes;
}
inline bool combine_quiesced(const CombineSignal& c) {
    const uint32_t request = __atomic_load_n(&c.request, __ATOMIC_ACQUIRE);
    return !request || (__atomic_load_n(&c.done, __ATOMIC_ACQUIRE) == request &&
                        __atomic_load_n(&c.result_consumed_generation, __ATOMIC_ACQUIRE) == request);
}

/* The activation slot's sub-regions and the route cap from the segment's own shape (Header::max_rows/hidden/top_k), the same sums
 * layout_for() uses: BF16 reference rows, then the exact E4M3 q rows, then the F32 group scales. They replace the Flash-Next constants
 * kActivationBf16Bytes/kActivationQBytes/kMaxRoutes wherever a segment is already open, so a model with another hidden width or top-k
 * addresses its own slot rather than the 2560/10 one. */
constexpr uint64_t seg_activation_q_off(const Header* h) { return uint64_t(h->max_rows) * h->hidden * sizeof(uint16_t); }
constexpr uint64_t seg_activation_scale_off(const Header* h) { return seg_activation_q_off(h) + uint64_t(h->max_rows) * h->hidden; }
constexpr uint32_t seg_max_routes(const Header* h) { return h->max_rows * h->top_k; }

constexpr uint64_t align_page(uint64_t n) { return (n + kPage - 1) & ~(uint64_t(kPage) - 1); }
constexpr uint32_t route_slot_bytes() { return kMaxRoutes * sizeof(Route); }
constexpr uint32_t activation_slot_bytes() { return kActivationBf16Bytes + kActivationQBytes + kActivationScaleBytes; }
constexpr uint32_t result_slot_bytes() { return kMaxRows * kHidden * sizeof(uint16_t); }



template <class T> inline T* ptr(void* base, uint64_t off) {
    return reinterpret_cast<T*>(static_cast<uint8_t*>(base) + off);
}
template <class T> inline const T* ptr(const void* base, uint64_t off) {
    return reinterpret_cast<const T*>(static_cast<const uint8_t*>(base) + off);
}
inline Request* request(void* base, uint32_t seq) {
    auto* h = static_cast<Header*>(base);
    return ptr<Request>(base, h->regions[v26::index(Region::Requests)].offset) + ((seq - 1) % h->counts.ring_entries);
}
inline const Request* request(const void* base, uint32_t seq) {
    auto* h = static_cast<const Header*>(base);
    return ptr<Request>(base, h->regions[v26::index(Region::Requests)].offset) + ((seq - 1) % h->counts.ring_entries);
}

/* The lane class a step of `rows` rows takes: the first small class that fits it, else the large lane (kSmallClasses). */
inline uint32_t step_class(const Header* h, uint32_t rows) {
    for (uint32_t c = 0; c < h->small_class_count && c < kSmallClasses; ++c)
        if (rows <= h->small_class_rows[c]) return c;
    return kSmallClasses;
}

inline Request* lane_request(void* base, uint32_t lane, uint32_t seq) {
    auto* h = static_cast<Header*>(base);
    if (lane >= h->counts.lanes || seq == 0) return nullptr;
    return ptr<Request>(base, h->regions[v26::index(Region::Requests)].offset) + lane * kLaneSlots + ((seq - 1) % kLaneSlots);
}
inline const Request* lane_request(const void* base, uint32_t lane, uint32_t seq) {
    auto* h = static_cast<const Header*>(base);
    if (lane >= h->counts.lanes || seq == 0) return nullptr;
    return ptr<Request>(base, h->regions[v26::index(Region::Requests)].offset) + lane * kLaneSlots + ((seq - 1) % kLaneSlots);
}

inline PlacementControl* placement_control(void* base) {
    auto* h = static_cast<Header*>(base);
    return ptr<PlacementControl>(base, h->regions[v26::index(Region::Placement)].offset);
}
inline const PlacementControl* placement_control(const void* base) {
    auto* h = static_cast<const Header*>(base);
    return ptr<PlacementControl>(base, h->regions[v26::index(Region::Placement)].offset);
}
/* Bounded seqlock snapshot: a writer publishes odd seq, data, even seq with release.
 * No coordinator writes these words after initialization in slice A. */
inline bool placement_snapshot(const void* base, PlacementControl* out) {
    if (!base || !out) return false;
    const PlacementControl* c = placement_control(base);
    for (uint32_t tries = 0; tries < 64; ++tries) {
        const uint32_t before = __atomic_load_n(&c->seq, __ATOMIC_ACQUIRE);
        if (before & 1u) continue;
        std::memcpy(out, c, sizeof(*out));
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        if (before == __atomic_load_n(&c->seq, __ATOMIC_ACQUIRE)) return true;
    }
    return false;
}
inline PlacementJournal* placement_journal(void* base) {
    auto* h = static_cast<Header*>(base);
    return ptr<PlacementJournal>(base, h->regions[v26::index(Region::Journal)].offset);
}
inline const PlacementJournal* placement_journal(const void* base) {
    auto* h = static_cast<const Header*>(base);
    return ptr<PlacementJournal>(base, h->regions[v26::index(Region::Journal)].offset);
}
/* The v24 service slots are part of the control page, directly after ACKs.
 * No extra mapping or dynamic allocator; layout_for validates the page fit. */
inline PublishedServiceSlot* published_service(void* base, uint32_t helper) {
    auto* h = static_cast<Header*>(base);
    return h && helper < h->helper_count
        ? ptr<PublishedServiceSlot>(base, h->regions[v26::index(Region::Services)].offset) + helper : nullptr;
}
inline const PublishedServiceSlot* published_service(const void* base, uint32_t helper) {
    return published_service(const_cast<void*>(base), helper);
}

inline PlacementAck* placement_ack(void* base, uint32_t actor) {
    auto* h = static_cast<Header*>(base);
    return actor < h->placement_ack_count ? ptr<PlacementAck>(base, h->regions[v26::index(Region::Acks)].offset) + actor : nullptr;
}
inline const PlacementAck* placement_ack(const void* base, uint32_t actor) {
    auto* h = static_cast<const Header*>(base);
    return actor < h->placement_ack_count ? ptr<PlacementAck>(base, h->regions[v26::index(Region::Acks)].offset) + actor : nullptr;
}
inline GenerationHeader* generation_header(void* base, uint32_t slot) {
    auto* h = static_cast<Header*>(base);
    return slot < v26::kGenerationSlots ? ptr<GenerationHeader>(base, h->generation_offset[slot]) : nullptr;
}
inline const GenerationHeader* generation_header(const void* base, uint32_t slot) {
    auto* h = static_cast<const Header*>(base);
    return slot < v26::kGenerationSlots ? ptr<GenerationHeader>(base, h->generation_offset[slot]) : nullptr;
}
inline uint8_t* generation_map(void* base, uint32_t slot) {
    auto* g = generation_header(base, slot);
    return g ? reinterpret_cast<uint8_t*>(g) + sizeof(GenerationHeader) : nullptr;
}
inline const uint8_t* generation_map(const void* base, uint32_t slot) {
    auto* g = generation_header(base, slot);
    return g ? reinterpret_cast<const uint8_t*>(g) + sizeof(GenerationHeader) : nullptr;
}
inline uint8_t owner_at(const void* base, uint32_t slot, uint32_t layer, uint32_t expert) {
    const auto* h = static_cast<const Header*>(base);
    const uint8_t* map = h ? generation_map(base, slot) : nullptr;
    if (!map || layer >= h->layer_count || expert >= h->expert_count) return kOwnerLocal;
    return __atomic_load_n(map + (size_t)layer * h->expert_count + expert, __ATOMIC_ACQUIRE);
}
inline uint64_t generation_digest(const void* base, uint32_t slot) {
    const auto* h = static_cast<const Header*>(base);
    const uint8_t* map = h ? generation_map(base, slot) : nullptr;
    return map ? placement_digest(h->model_shape_hash, h->layer_count, h->expert_count,
                                  h->helper_count, map) : 0;
}

/* The bootstrap publication is a distinct controller seqlock transition.
 * The initial init() digest alone is not a seal, and must never be uploaded. */
inline bool bootstrap_generation_sealed(const void* base) {
    if (!base) return false;
    const auto* h = static_cast<const Header*>(base);
    PlacementControl c{};
    if (!placement_snapshot(base, &c) || c.seq < 2 || (c.seq & 1u) ||
        !h->helper_count || !slots_fit(h->helper_count,h->rank_count) ||
        c.active_slot != 0 || c.active_epoch != 1 || c.fault_code ||
        c.expected_helpers != ((1u << h->helper_count) - 1u) ||
        __atomic_load_n(&h->bootstrap_filtered_helpers, __ATOMIC_ACQUIRE) != c.expected_helpers)
        return false;
    const auto* g = generation_header(base, 0);
    return g && g->state == uint32_t(GenerationState::ACTIVE) && g->epoch == 1 &&
           __atomic_load_n(&g->ready_helpers, __ATOMIC_ACQUIRE) == c.expected_helpers &&
           g->digest == c.active_digest && c.active_digest &&
           generation_digest(base, 0) == c.active_digest;
}

inline int32_t* layer_slots(void* base, uint32_t layer) {
    auto* h = static_cast<Header*>(base);
    if (!h || layer >= h->layer_count) return nullptr;
    return ptr<int32_t>(base, h->regions[v26::index(Region::LayerMap)].offset) + (size_t)layer * h->expert_count;
}
inline const int32_t* layer_slots(const void* base, uint32_t layer) {
    auto* h = static_cast<const Header*>(base);
    if (!h || layer >= h->layer_count) return nullptr;
    return ptr<int32_t>(base, h->regions[v26::index(Region::LayerMap)].offset) + (size_t)layer * h->expert_count;
}
inline int32_t layer_slot(const void* base, uint32_t layer, uint32_t expert) {
    const int32_t* p = layer_slots(base, layer);
    return p && expert < static_cast<const Header*>(base)->expert_count ? p[expert] : -1;
}
/* Sequence order modulo 2^32: a is after b when it is less than half the space ahead (defined unsigned arithmetic, no
 * signed cast). Holds while fewer than 2^31 sequences are in flight between the two, which the protocol keeps. */
constexpr bool seq_after(uint32_t a, uint32_t b) { return a != b && (uint32_t)(a - b) < 0x80000000u; }

inline uint32_t load(const uint32_t* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
inline void store(uint32_t* p, uint32_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
inline uint32_t fetch_add(uint32_t* p, uint32_t v) { return __atomic_fetch_add(p, v, __ATOMIC_ACQ_REL); }
inline int futex_wait(uint32_t* p, uint32_t expected) {
    return static_cast<int>(::syscall(SYS_futex, p, FUTEX_WAIT, expected, nullptr, nullptr, 0));
}
/* Every waiter: two helpers (and the engine) can wait on the same word of a shared segment. */
inline int futex_wake(uint32_t* p) {
    return static_cast<int>(::syscall(SYS_futex, p, FUTEX_WAKE, INT_MAX, nullptr, nullptr, 0));
}
/* Header::notify: publish a change CPU waiters wait for (after the change itself is stored). */
inline void notify(Header* h) {
    (void)__atomic_fetch_add(&h->notify, 1u, __ATOMIC_SEQ_CST);
    (void)futex_wake(&h->notify);
}

/* Helper drain (after the lane's stream is drained): publish the identity of the lane's last completed request
 * (epoch/slot) for the engine's quiescence check. The ring slot must still hold that request (admission closed and
 * every parent retired, so nothing newer was published) and carry a valid epoch/slot; otherwise refuse the ACK
 * rather than copy another request's identity. acknowledge_lane_drain's release of drain_ack publishes the words. */
inline bool publish_done_identity(void* base, uint32_t lane) {
    LaneSignal& l = static_cast<Header*>(base)->lane[lane];
    const uint32_t seq = load(&l.done_sequence);
    if (!seq) return true;
    const Request* q = lane_request(base, lane, seq);
    if (!q || load(&q->sequence) != seq) return false;
    const uint64_t epoch = __atomic_load_n(&q->placement_epoch, __ATOMIC_ACQUIRE);
    const uint32_t slot = load(&q->generation_slot);
    if (!epoch || slot >= v26::kGenerationSlots) return false;
    __atomic_store_n(&l.done_epoch, epoch, __ATOMIC_RELAXED);
    __atomic_store_n(&l.done_slot, slot, __ATOMIC_RELAXED);
    return true;
}

/* Engine quiescence, one lane: a served lane acknowledged exactly this closure, every published request is done and
 * consumed, and the helper's done identity equals the engine's consumed identity for the last one. */
inline bool lane_quiesced(const Header* h, uint32_t lane, uint64_t closure, bool served) {
    const LaneSignal& l = h->lane[lane];
    if (served && __atomic_load_n(&l.drain_ack, __ATOMIC_ACQUIRE) != closure) return false;
    const uint32_t seq = load(&l.request_sequence);
    if (load(&l.done_sequence) != seq || load(&l.consumed_sequence) != seq) return false;
    return !seq || (__atomic_load_n(&l.done_epoch, __ATOMIC_ACQUIRE) == __atomic_load_n(&l.consumed_epoch, __ATOMIC_ACQUIRE) &&
                    load(&l.done_slot) == load(&l.consumed_slot));
}
/* Raise helper_fault (sticky) and wake every CPU waiter. */
inline void raise_fault(Header* h) {
    __atomic_store_n(&h->helper_fault, 1u, __ATOMIC_SEQ_CST);
    notify(h);
}

enum class WaitResult { Ready, Fault, Error, Timeout };
/* Sleep until done() holds, or until `deadline` (CLOCK_MONOTONIC; nullptr: none) passes. helper_fault wins over done():
 * once a helper has failed, nothing it published since counts. The generation is read before both checks, so a notify()
 * between the checks and the sleep makes the futex return EAGAIN instead of sleeping (no lost wake-up). */
template <class Done>
inline WaitResult wait_for(Header* h, const timespec* deadline, Done done) {
    for (;;) {
        const uint32_t gen = __atomic_load_n(&h->notify, __ATOMIC_SEQ_CST);
        if (__atomic_load_n(&h->helper_fault, __ATOMIC_SEQ_CST)) return WaitResult::Fault;
        if (done()) return WaitResult::Ready;
        timespec left{}, *rel = nullptr;
        if (deadline) {
            timespec now{};
            clock_gettime(CLOCK_MONOTONIC, &now);
            int64_t ns = (int64_t)(deadline->tv_sec - now.tv_sec) * 1000000000 + (deadline->tv_nsec - now.tv_nsec);
            if (ns <= 0) return WaitResult::Timeout;
            left = {(time_t)(ns / 1000000000), (long)(ns % 1000000000)};
            rel = &left;
        }
        if (::syscall(SYS_futex, &h->notify, FUTEX_WAIT, gen, rel, nullptr, 0) < 0 && errno != EAGAIN &&
            errno != EINTR && errno != ETIMEDOUT)
            return WaitResult::Error;
    }
}

/* A process's start time in clock ticks (/proc/<pid>/stat field 22), 0 when it cannot be read. With the pid it names
 * one process for the life of the system. */
inline uint64_t process_start_ticks(int pid) {
    char path[64], buf[1024];
    std::snprintf(path, sizeof(path), "/proc/%d/stat", pid);
    FILE* f = std::fopen(path, "r");
    if (!f) return 0;
    const size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
    std::fclose(f);
    buf[n] = '\0';
    const char* p = std::strrchr(buf, ')');   /* comm may contain spaces; the fields after it do not */
    if (!p) return 0;
    for (int field = 2; field < 22 && *p; ++p)
        if (*p == ' ') ++field;
    return std::strtoull(p, nullptr, 10);
}

/* Sleep until *p == value: true then; false when *fault becomes non-zero (whoever sets it wakes p) or the futex
 * fails, so a failed wait is never taken for the value having arrived. */
inline bool wait_until(uint32_t* p, uint32_t value, const uint32_t* fault) {
    for (;;) {
        const uint32_t seen = load(p);
        if (seen == value) return true;
        if (load(fault)) return false;
        if (futex_wait(p, seen) < 0 && errno != EAGAIN && errno != EINTR) return false;
    }
}


} // namespace r3::moe_shm
