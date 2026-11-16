#include "moe_remote_shm.h"
#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>
#include <vector>

namespace v = r3::moe_shm::v26;
using r3::moe_shm::LayoutShape;

static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "r3_topology_v26_test: FAILED %s\n", what); std::exit(1); }
}
static constexpr uint64_t manifest = 0x82374ae0951bcf16ull;

static size_t alignment(v::Region r) {
    switch (r) {
    case v::Region::Generations:
    case v::Region::Routes:
    case v::Region::Activations:
    case v::Region::ResultPlanes: return r3::moe_shm::kPage;
    case v::Region::ChunkSequences:
    case v::Region::ChunkCompletions:
    case v::Region::LayerMap:
    case v::Region::LayerMapReady:
    case v::Region::LayerMapConsumed: return alignof(uint32_t);
    default: return 64;
    }
}
static void* map_layout(const v::Layout& l) {
    void* p = mmap(nullptr, size_t(l.bytes), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    check(p != MAP_FAILED, "map");
    return p;
}
static void expect_valid(void* p, const v::Layout& l, LayoutShape s, uint32_t layers, v::Topology t) {
    v::Layout accepted{};
    check(v::valid(p, size_t(l.bytes), s, layers, t, manifest, &accepted) == v::Refusal::None,
          "expected valid v26 segment");
    check(accepted.bytes == l.bytes, "consumer reconstructed producer bytes");
    for (size_t i = 0; i < v::kRegions; ++i)
        check(accepted.regions[i] == l.regions[i], "consumer independently reconstructs every region");
}
static void geometry(LayoutShape shape, uint32_t layers, v::Topology t) {
    v::Counts counts{};
    check(v::counts_for(t, &counts) == v::Refusal::None, "counts valid");
    check(counts.slots == t.helpers * t.ranks &&
          counts.lanes == counts.slots * r3::moe_shm::kClasses &&
          counts.ring_entries == counts.lanes * r3::moe_shm::kLaneSlots &&
          counts.acks == t.helpers + t.ranks && counts.results == counts.slots + t.ranks,
          "all counts derive from topology");

    v::Layout l{};
    check(v::layout_for(shape, layers, t, &l) == v::Refusal::None, "layout valid");
    check(l.bytes % r3::moe_shm::kPage == 0, "mapped bytes page-aligned");
    uint64_t last_end = sizeof(v::Header);
    for (uint32_t i = 0; i < v::kRegions; ++i) {
        const auto r = static_cast<v::Region>(i);
        const auto sp = l.regions[i];
        check(sp.offset >= last_end && sp.offset % alignment(r) == 0 &&
              sp.stride % alignment(r) == 0 && sp.count && sp.stride &&
              sp.bytes == uint64_t(sp.stride) * sp.count && sp.offset <= l.bytes &&
              sp.bytes <= l.bytes - sp.offset, "regions in bounds, aligned, disjoint");
        last_end = sp.offset + sp.bytes;
    }
    check(l.regions[v::index(v::Region::Lanes)].offset >= sizeof(v::Header), "metadata outside header");
    check(l.regions[v::index(v::Region::ResultPlanes)].count == counts.slots + t.ranks,
          "dedicated combined plane counted");
    check(l.regions[v::index(v::Region::Activations)].count == t.ranks,
          "one staged activation buffer per consumer, not per helper");

    std::vector<bool> seen_slot(counts.slots), seen_lane(counts.lanes), seen_ring(counts.ring_entries);
    for (uint32_t helper = 0; helper < t.helpers; ++helper) {
        for (uint32_t rank = 0; rank < t.ranks; ++rank) {
            uint32_t slot = UINT32_MAX;
            check(v::role_slot(t, helper, rank, &slot) && slot < counts.slots && !seen_slot[slot],
                  "unique helper/consumer slot");
            seen_slot[slot] = true;
            for (uint32_t cls = 0; cls < r3::moe_shm::kClasses; ++cls) {
                uint32_t lane = UINT32_MAX, entry = UINT32_MAX;
                check(v::role_lane(t, slot, cls, &lane) && lane < counts.lanes && !seen_lane[lane],
                      "unique lane / runtime class stride");
                seen_lane[lane] = true;
                for (uint32_t sequence = 1; sequence <= r3::moe_shm::kLaneSlots; ++sequence) {
                    check(v::ring_entry(t, lane, sequence, &entry) && entry < counts.ring_entries &&
                          !seen_ring[entry], "unique ring entry");
                    seen_ring[entry] = true;
                }
                check(v::ring_entry(t, lane, r3::moe_shm::kLaneSlots + 1u, &entry) &&
                      entry == lane * r3::moe_shm::kLaneSlots, "ring wrap");
            }
        }
    }
    for (bool b : seen_slot) check(b, "no missing slots");
    for (bool b : seen_lane) check(b, "no missing lanes");
    for (bool b : seen_ring) check(b, "no missing ring entries");

    uint32_t helper_mask = 0, rank_mask = 0;
    check(v::role_masks(t, &helper_mask, &rank_mask) &&
          helper_mask == ((1u << t.helpers) - 1u) &&
          rank_mask == ((1u << t.ranks) - 1u), "role masks");
    for (uint32_t rank = 0; rank < t.ranks; ++rank) {
        uint32_t ack = UINT32_MAX, combine = UINT32_MAX;
        check(v::ack_for_rank(t, rank, &ack) && ack == t.helpers + rank &&
              ack < counts.acks, "rank ACK follows actual helper count");
        check(v::combine_for_rank(t, rank, &combine) && combine == counts.slots + rank &&
              combine < counts.results, "combine is a separate result plane");
        for (uint32_t slot = 0; slot < counts.slots; ++slot) {
            uint32_t result = UINT32_MAX;
            check(v::result_for_slot(t, slot, &result) && result != combine &&
                  v::element_offset(l, v::Region::ResultPlanes, result) !=
                  v::element_offset(l, v::Region::ResultPlanes, combine),
                  "TP1 and all TP counts: combine never aliases a result");
        }
    }
    uint32_t invalid = 0;
    check(!v::role_slot(t, t.helpers, 0, &invalid) &&
          !v::role_slot(t, 0, t.ranks, &invalid) &&
          !v::role_lane(t, counts.slots, 0, &invalid) &&
          !v::role_lane(t, 0, r3::moe_shm::kClasses, &invalid) &&
          !v::ring_entry(t, counts.lanes, 1, &invalid) &&
          !v::ring_entry(t, 0, 0, &invalid) &&
          !v::ack_for_rank(t, t.ranks, &invalid) &&
          !v::combine_for_rank(t, t.ranks, &invalid) &&
          !v::result_for_slot(t, counts.slots, &invalid), "invalid role identities refused");

    void* p = map_layout(l);
    check(v::valid(p, size_t(l.bytes), shape, layers, t, manifest) == v::Refusal::Unpublished,
          "bootstrap not yet published");
    check(v::init(p, size_t(l.bytes) - 1, shape, layers, t, manifest) == v::Refusal::BadMapping,
          "producer refuses truncated map");
    check(v::init(p, size_t(l.bytes), shape, layers, t, manifest) == v::Refusal::None,
          "producer emits v26");
    expect_valid(p, l, shape, layers, t);
    check(v::valid(p, size_t(l.bytes) - 1, shape, layers, t, manifest) == v::Refusal::BadMapping &&
          v::valid(p, size_t(l.bytes), shape, layers, {t.helpers, t.ranks + 1}, manifest) != v::Refusal::None &&
          v::valid(p, size_t(l.bytes), shape, layers, t, manifest + 1) == v::Refusal::WrongIdentity &&
          v::valid(p, size_t(l.bytes), shape, layers + 1, t, manifest) == v::Refusal::WrongIdentity,
          "consumer checks mapped length, topology, manifest and layers");
    auto* h = static_cast<v::Header*>(p);
    check(h->version == 26 && h->header_bytes == r3::moe_shm::kPage, "v26 page header");
    const auto* control = v::element<r3::moe_shm::PlacementControl>(p, l, v::Region::Placement, 0);
    check(control && control->active_slot == 0 && control->pending_slot == UINT32_MAX &&
          control->active_epoch == 1 && control->expected_helpers == ((1u << t.helpers) - 1u) &&
          control->state == uint32_t(r3::moe_shm::PlacementState::ACTIVE) &&
          h->next_sequence == 1 && h->small_class_count == 1,
          "v26 bootstrap control and request allocator match production baseline");
    for (uint32_t layer = 0; layer < layers; ++layer) {
        check(*v::element<uint32_t>(p, l, v::Region::LayerMapReady, layer) == 0 &&
              *v::element<uint32_t>(p, l, v::Region::LayerMapConsumed, layer) == 0,
              "per-layer publication and consumption begin unissued");
        const int32_t* owners = v::element<int32_t>(p, l, v::Region::LayerMap, layer);
        for (uint32_t expert = 0; expert < shape.experts; ++expert)
            check(owners[expert] == -1, "unpublished regular layer map never routes to helper zero");
    }
    check(&h->lane[0] == v::element<r3::moe_shm::LaneSignal>(p, l, v::Region::Lanes, 0),
          "lane indexed view addresses its validated region");
    for (uint32_t slot = 0; slot < counts.slots; ++slot) {
        auto* d = v::element<v::SlotDescriptor>(p, l, v::Region::Slots, slot);
        check(d && d->helper == slot / t.ranks && d->consumer == slot % t.ranks &&
              d->result == v::element_offset(l, v::Region::ResultPlanes, slot) &&
              d->activation == v::element_offset(l, v::Region::Activations, slot % t.ranks),
              "producer slot identity and shared activation");
    }
    for (uint32_t rank = 0; rank < t.ranks; ++rank) {
        auto* c = v::element<v::CombineDescriptor>(p, l, v::Region::CombineSignals, rank);
        auto* d = v::element<v::ChunkDescriptor>(p, l, v::Region::ChunkDescriptors, rank);
        check(&h->combine[rank] == c && &h->chunks[rank] == d,
              "combine and chunk descriptors use counted views");
        for (uint32_t helper = 0; helper < t.helpers; ++helper) {
            const uint32_t ix = rank * t.helpers + helper;
            auto* i = v::element<v::CombineInput>(p, l, v::Region::CombineInputs, ix);
            check(&c->lane[helper] == &i->lane && &c->seq[helper] == &i->sequence,
                  "combine helper view addresses rank's exact contributor words");
            for (uint32_t chunk = 0; chunk < l.counts.chunks; ++chunk)
                check(&d->seq[helper][chunk] == v::element<uint32_t>(p, l, v::Region::ChunkSequences,
                            ix * l.counts.chunks + chunk) &&
                      &d->done[chunk] == v::element<uint32_t>(p, l, v::Region::ChunkCompletions,
                            rank * l.counts.chunks + chunk),
                      "dynamic combine sequence and completion views never alias");
        }
    }
    for (uint32_t gen = 0; gen < v::kGenerationSlots; ++gen) {
        auto* g = v::element<r3::moe_shm::GenerationHeader>(p, l, v::Region::Generations, gen);
        check(g && g->map_bytes == l.owner_map_bytes &&
              g->state == uint32_t(gen == 0 ? r3::moe_shm::GenerationState::ACTIVE :
                                              r3::moe_shm::GenerationState::EMPTY),
              "owner map correctly sized and active generation identified");
        if (gen == 0)
            check(g->epoch == 1 && g->digest == control->active_digest && g->digest != 0,
                  "generation digest agrees with active placement control");
        auto* bytes = reinterpret_cast<const uint8_t*>(g) + sizeof(*g);
        for (uint32_t i = 0; i < l.owner_map_bytes; ++i)
            check(bytes[i] == r3::moe_shm::kOwnerLocal, "generation starts local");
    }
    munmap(p, size_t(l.bytes));
}

template<class T> static void corrupt(T& field, void* p, const v::Layout& l,
                                      LayoutShape s, uint32_t layers, v::Topology t) {
    const T saved = field;
    ++field;
    check(v::valid(p, size_t(l.bytes), s, layers, t, manifest) != v::Refusal::None,
          "mutation detected");
    field = saved;
    expect_valid(p, l, s, layers, t);
}
static void corruptions() {
    LayoutShape s{128, 2, 3, 8, 128, 64};
    constexpr uint32_t layers = 2;
    const v::Topology t{2, 3};
    v::Layout l{};
    check(v::layout_for(s, layers, t, &l) == v::Refusal::None, "mutation layout");
    void* p = map_layout(l);
    check(v::init(p, size_t(l.bytes), s, layers, t, manifest) == v::Refusal::None, "mutation init");
    auto* h = static_cast<v::Header*>(p);
    corrupt(h->magic, p, l, s, layers, t);
    corrupt(h->version, p, l, s, layers, t);
    corrupt(h->header_bytes, p, l, s, layers, t);
    corrupt(h->segment_bytes, p, l, s, layers, t);
    corrupt(h->shape.hidden, p, l, s, layers, t);
    corrupt(h->shape.top_k, p, l, s, layers, t);
    corrupt(h->shape.experts, p, l, s, layers, t);
    corrupt(h->shape.max_rows, p, l, s, layers, t);
    corrupt(h->shape.act_group, p, l, s, layers, t);
    corrupt(h->shape.intermediate, p, l, s, layers, t);
    corrupt(h->layer_count, p, l, s, layers, t);
    corrupt(h->topology.helpers, p, l, s, layers, t);
    corrupt(h->topology.ranks, p, l, s, layers, t);
    corrupt(h->model_shape_hash, p, l, s, layers, t);
    corrupt(h->manifest_digest, p, l, s, layers, t);
    corrupt(h->identity_hash, p, l, s, layers, t);
    corrupt(h->published_helpers, p, l, s, layers, t);
    corrupt(h->rank_count, p, l, s, layers, t);
    corrupt(h->slot_count, p, l, s, layers, t);
    corrupt(h->route_offset.relative, p, l, s, layers, t);
    corrupt(h->activation_offset.stride, p, l, s, layers, t);
    corrupt(h->result_offset.start, p, l, s, layers, t);
    corrupt(h->lane.count, p, l, s, layers, t);
    corrupt(h->combine.relative, p, l, s, layers, t);
    corrupt(h->chunks.count, p, l, s, layers, t);
    corrupt(h->layer_map_ready.relative, p, l, s, layers, t);
    corrupt(h->helper_pid.field, p, l, s, layers, t);
    corrupt(h->small_class_count, p, l, s, layers, t);
    corrupt(h->small_class_rows[0], p, l, s, layers, t);
    auto* control = v::element<r3::moe_shm::PlacementControl>(p, l, v::Region::Placement, 0);
    corrupt(control->expected_helpers, p, l, s, layers, t);
    corrupt(control->active_slot, p, l, s, layers, t);
    for (auto& field : h->reserved) corrupt(field, p, l, s, layers, t);
    auto* counts = &h->counts.slots;
    for (size_t i = 0; i < sizeof(v::Counts) / sizeof(uint32_t); ++i)
        corrupt(counts[i], p, l, s, layers, t);
    for (size_t i = 0; i < v::kRegions; ++i) {
        auto& sp = h->regions[i];
        corrupt(sp.offset, p, l, s, layers, t);
        corrupt(sp.bytes, p, l, s, layers, t);
        corrupt(sp.count, p, l, s, layers, t);
        corrupt(sp.stride, p, l, s, layers, t);
    }
    for (uint32_t slot = 0; slot < l.counts.slots; ++slot) {
        auto* d = v::element<v::SlotDescriptor>(p, l, v::Region::Slots, slot);
        corrupt(d->helper, p, l, s, layers, t);
        corrupt(d->consumer, p, l, s, layers, t);
        corrupt(d->route, p, l, s, layers, t);
        corrupt(d->activation, p, l, s, layers, t);
        corrupt(d->result, p, l, s, layers, t);
        corrupt(d->route_bytes, p, l, s, layers, t);
        corrupt(d->activation_bytes, p, l, s, layers, t);
        corrupt(d->result_bytes, p, l, s, layers, t);
        corrupt(d->reserved, p, l, s, layers, t);
    }
    for (uint32_t helper = 0; helper < t.helpers; ++helper) {
        auto* d = v::element<v::HelperDescriptor>(p, l, v::Region::Helpers, helper);
        corrupt(d->helper, p, l, s, layers, t);
    }
    for (uint32_t rank = 0; rank < t.ranks; ++rank) {
        auto* c = v::element<v::CombineDescriptor>(p, l, v::Region::CombineSignals, rank);
        corrupt(c->consumer, p, l, s, layers, t);
        corrupt(c->inputs, p, l, s, layers, t);
        corrupt(c->chunk_sequences, p, l, s, layers, t);
        corrupt(c->chunk_completions, p, l, s, layers, t);
        corrupt(c->lane.relative, p, l, s, layers, t);
        corrupt(c->seq.stride, p, l, s, layers, t);
        auto* d = v::element<v::ChunkDescriptor>(p, l, v::Region::ChunkDescriptors, rank);
        corrupt(d->seq.relative, p, l, s, layers, t);
        corrupt(d->done.count, p, l, s, layers, t);
        for (uint32_t helper = 0; helper < t.helpers; ++helper) {
            auto* input = v::element<v::CombineInput>(p, l, v::Region::CombineInputs, rank*t.helpers+helper);
            corrupt(input->helper, p, l, s, layers, t);
            corrupt(input->consumer, p, l, s, layers, t);
        }
    }
    for (uint32_t gen = 0; gen < v::kGenerationSlots; ++gen) {
        auto* g = v::element<r3::moe_shm::GenerationHeader>(p, l, v::Region::Generations, gen);
        corrupt(g->slot, p, l, s, layers, t);
        corrupt(g->map_bytes, p, l, s, layers, t);
        corrupt(g->layer_count, p, l, s, layers, t);
        corrupt(g->expert_count, p, l, s, layers, t);
    }
    // v25 is incompatible even when every other byte matches.
    h->version = 25;
    check(v::valid(p, size_t(l.bytes), s, layers, t, manifest) == v::Refusal::WrongVersion,
          "v25 explicitly rejected");
    h->version = v::kVersion;
    h->published_helpers = 0;
    check(v::valid(p, size_t(l.bytes), s, layers, t, manifest) == v::Refusal::Unpublished,
          "unpublished descriptors rejected");
    h->published_helpers = t.helpers;
    expect_valid(p, l, s, layers, t);
    munmap(p, size_t(l.bytes));
}
static void hostile() {
    v::Counts c{};
    check(v::counts_for({0, 1}, &c) == v::Refusal::ZeroTopology &&
          v::counts_for({1, 0}, &c) == v::Refusal::ZeroTopology,
          "zero topology refused");
    check(v::counts_for({UINT32_MAX, UINT32_MAX}, &c) == v::Refusal::SlotOverflow,
          "multiplication overflowing uint32 slot count refused");
    check(v::counts_for({UINT32_MAX, 1}, &c) != v::Refusal::None &&
          v::counts_for({1, UINT32_MAX}, &c) != v::Refusal::None,
          "UINT32 boundary refused");
    check(v::counts_for({256, 1}, &c) == v::Refusal::OwnerId,
          "reserved local owner id is a real bound");
    check(v::counts_for({9, 1}, &c) == v::Refusal::CombineMask,
          "uint8 contributor mask is a real bound");
    check(v::counts_for({1, 32}, &c) == v::Refusal::ParticipantMask,
          "uint32 participant namespace is a real bound");
    check(v::counts_for({1, 31}, &c) == v::Refusal::None,
          "32-bit namespace at legal boundary");
    check(v::counts_for({1, 31}, nullptr) == v::Refusal::NullOutput,
          "null output refused");
    check(!v::role_masks({1,32}, &c.slots, &c.acks), "role mask refuses overflowing shift");
    v::Layout l{};
    LayoutShape s{128, 2, 3, 8, 128, 64};
    check(v::layout_for(s, 0, {1,1}, &l) == v::Refusal::InvalidShape &&
          v::layout_for({128, 4, 3, 8, 128, 64}, 2, {1,1}, &l) == v::Refusal::InvalidShape &&
          v::layout_for({127, 2, 3, 8, 128, 64}, 2, {1,1}, &l) == v::Refusal::InvalidShape &&
          v::layout_for({UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, 1, UINT32_MAX},
                        UINT32_MAX, {1,1}, &l) != v::Refusal::None,
          "invalid shapes and huge products refused");
    check(v::checked_add(UINT64_MAX, 1, &l.bytes) == false &&
          v::checked_mul(UINT64_MAX, 2, &l.bytes) == false &&
          v::checked_align(UINT64_MAX, 4096, &l.bytes) == false,
          "overflow predicates fail closed");
}

int main() {
    // Freeze the v25 four supported outcomes without editing its active ABI.
    check(r3::moe_shm::slots_fit(1,1) && r3::moe_shm::slots_fit(2,1) &&
          r3::moe_shm::slots_fit(1,2) && r3::moe_shm::slots_fit(2,2) &&
          r3::moe_shm::slots_fit(1,3) && r3::moe_shm::slots_fit(2,3),
          "v25 baseline and old TP3 refusals");
    check(r3::moe_shm::kVersion == 26 && v::kVersion == 26,
          "old active and new staged versions distinguishable");
    hostile();
    const v::Topology topologies[] = {{1,1},{2,1},{1,2},{2,2},{1,3},{2,3},{3,2},{4,1}};
    struct Case { LayoutShape shape; uint32_t layers; };
    const Case shapes[] = {
        {{2560,10,512,64,128,640},49}, // Flash-Next
        {{2048,4,256,64,128,768},32},  // RCS12 synthetic 1
        {{2560,16,1024,64,128,3072},64} // RCS12 synthetic 2
    };
    for (const auto& s : shapes)
        for (const auto& t : topologies)
            geometry(s.shape, s.layers, t);
    corruptions();
    std::puts("r3_topology_v26_test: PASS");
}
