#pragma once

// Gates A/B: inactive v26 host geometry. The v25 GPU and worker protocol
// stays intact until the coordinated C-E migration. No v25 fallback reader.
#include "moe_remote_shm.h"
#include <limits>

namespace r3::moe_shm::v26 {

constexpr uint32_t kVersion = 26;
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
    Lanes, Slots, Helpers, CombineSignals, CombineInputs, ChunkSequences,
    ChunkCompletions, Requests, Acks, Services, Placement, Journal,
    Generations, LayerMap, Routes, Activations, ResultPlanes, Count
};
constexpr size_t kRegions = static_cast<size_t>(Region::Count);
constexpr size_t index(Region r) { return static_cast<size_t>(r); }
struct Span {
    uint64_t offset, bytes;
    uint32_t count, stride;
};
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
    uint32_t consumer, enabled, generation, done;
    uint64_t inputs, chunk_sequences, chunk_completions;
    uint32_t min_rows, rows;
};
struct alignas(64) CombineInput {
    uint32_t helper, consumer, lane, sequence;
};
// Prototype v26 request has one consumer result, not inline fixed-rank arrays.
// The identity is on the first cache line; no code uses this on the GPU yet.
struct alignas(64) Request {
    uint32_t sequence, completion_sequence;
    int32_t status;
    uint32_t layer, slot, world, consumer, helper, lane, lane_slot, rows;
    uint32_t generation_slot;
    uint64_t result_offset, result_stride;
    uint64_t route_offset, activation_offset;
    uint64_t placement_epoch, completion_epoch;
};
static_assert(sizeof(Request) == 128);
static_assert(offsetof(Request, consumer) < 64 && offsetof(Request, layer) < 64);
static_assert(sizeof(SlotDescriptor) == 64 && sizeof(HelperDescriptor) == 64);
static_assert(sizeof(CombineDescriptor) == 64 && sizeof(CombineInput) == 64);

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
    for (uint64_t n : {route_bytes, result_bytes, act_bytes, owner_bytes})
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
        !reserve(Region::Routes, x.counts.slots, route_bytes, kPage) ||
        !reserve(Region::Activations, x.counts.slots, act_bytes, kPage) ||
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

inline Refusal init(void* base, size_t mapped, const LayoutShape& shape, uint32_t layers,
                    Topology t, uint64_t manifest_digest) {
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
    for (uint32_t slot = 0; slot < l.counts.slots; ++slot) {
        auto* d = element<SlotDescriptor>(base, l, Region::Slots, slot);
        d->helper = slot / t.ranks; d->consumer = slot % t.ranks;
        d->route = element_offset(l, Region::Routes, slot);
        d->activation = element_offset(l, Region::Activations, slot);
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
        for (uint32_t helper = 0; helper < t.helpers; ++helper) {
            auto* i = element<CombineInput>(base, l, Region::CombineInputs, rank * t.helpers + helper);
            i->helper = helper; i->consumer = rank;
        }
    }
    for (uint32_t gen = 0; gen < kGenerationSlots; ++gen) {
        auto* g = element<GenerationHeader>(base, l, Region::Generations, gen);
        g->slot = gen; g->layer_count = layers; g->expert_count = shape.experts;
        g->map_bytes = l.owner_map_bytes;
        std::memset(reinterpret_cast<uint8_t*>(g) + sizeof(GenerationHeader), kOwnerLocal, l.owner_map_bytes);
    }
    // An attacher cannot validate before the last release store.
    __atomic_store_n(&h->published_helpers, t.helpers, __ATOMIC_RELEASE);
    return Refusal::None;
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
    for (uint32_t r : h->reserved) if (r) return Refusal::CorruptDescriptor;
    Layout l{};
    const Refusal e = layout_for(expected_shape, expected_layers, expected_topology, &l);
    if (e != Refusal::None) return e;
    if (mapped < l.bytes) return Refusal::BadMapping;
    if (h->segment_bytes != l.bytes || std::memcmp(&h->counts, &l.counts, sizeof(Counts)) != 0)
        return Refusal::CorruptDescriptor;
    for (size_t i = 0; i < kRegions; ++i)
        if (!(h->regions[i] == l.regions[i])) return Refusal::CorruptDescriptor;
    const Topology t = expected_topology;
    for (uint32_t slot = 0; slot < l.counts.slots; ++slot) {
        const auto* d = element<SlotDescriptor>(base, l, Region::Slots, slot);
        if (d->helper != slot / t.ranks || d->consumer != slot % t.ranks || d->reserved ||
            d->route != element_offset(l, Region::Routes, slot) ||
            d->activation != element_offset(l, Region::Activations, slot) ||
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
        for (uint32_t helper = 0; helper < t.helpers; ++helper) {
            const auto* i = element<CombineInput>(base, l, Region::CombineInputs, rank * t.helpers + helper);
            if (i->helper != helper || i->consumer != rank) return Refusal::CorruptDescriptor;
        }
    }
    for (uint32_t gen = 0; gen < kGenerationSlots; ++gen) {
        const auto* g = element<GenerationHeader>(base, l, Region::Generations, gen);
        if (g->slot != gen || g->layer_count != expected_layers ||
            g->expert_count != expected_shape.experts || g->map_bytes != l.owner_map_bytes)
            return Refusal::CorruptDescriptor;
    }
    if (accepted) *accepted = l;
    return Refusal::None;
}

} // namespace r3::moe_shm::v26
