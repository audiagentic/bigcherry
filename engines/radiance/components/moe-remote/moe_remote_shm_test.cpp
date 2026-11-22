#include "moe_remote_shm.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace r3::moe_shm;
namespace wire = r3::moe_shm::v26;

static void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "r3_moe_shm_test: FAILED %s\n", what);
        std::exit(1);
    }
}
constexpr uint64_t kManifest = 0x13572680ae691d31ull;
constexpr LayoutShape kShape{128, 2, 3, 8, 128, 64};

struct Segment {
    Layout layout{};
    Topology topology{};
    uint32_t layers = 2;
    void* data = nullptr;
    explicit Segment(Topology t, LayoutShape s = kShape, uint32_t n_layers = 2)
        : topology(t), layers(n_layers) {
        check(wire::layout_for(s, layers, t, &layout) == Refusal::None, "counted layout");
        data = mmap(nullptr, size_t(layout.bytes), PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        check(data != MAP_FAILED, "map host shm");
        check(wire::init(data, size_t(layout.bytes), s, layers, t, kManifest) == Refusal::None,
              "v26 segment init");
        check(valid_self(header(), size_t(layout.bytes)), "v26 live self-validation");
        check(wire::valid(data, size_t(layout.bytes), s, layers, t, kManifest) ==
                  Refusal::None, "external contract validation");
    }
    ~Segment() { if (data && data != MAP_FAILED) munmap(data, size_t(layout.bytes)); }
    Segment(const Segment&) = delete;
    Segment& operator=(const Segment&) = delete;
    Header* header() const { return static_cast<Header*>(data); }
};

static void Geometry_RuntimeCountsAndABI() {
    check(kVersion == 26 && sizeof(Header) == kPage && alignof(Header) == kPage &&
          sizeof(Request) == 256 && sizeof(LaneSignal) == 64 &&
          sizeof(CombineSignal) == 128 && sizeof(CombineChunks) == 64,
          "one v26 wire ABI and no inline fixed topology arrays");
    check(offsetof(Request, layer) == 12 && offsetof(Request, slot) == 16 &&
          offsetof(Request, world) == 20 &&
          offsetof(Request, generation_slot) == 36 &&
          offsetof(Request, route_offset) == 40 &&
          offsetof(Request, activation_offset) == 48 &&
          offsetof(Request, activation_q_offset) == 56,
          "the GPU Request first cache line identity is unchanged");
    const Topology cases[] = {{1,1}, {2,1}, {1,2}, {2,2}, {1,3}, {2,3}, {3,2}, {4,1}};
    for (const Topology t : cases) {
        wire::Counts c{};
        check(wire::counts_for(t, &c) == Refusal::None && slots_fit(t.helpers,t.ranks),
              "legal topology is counted and accepted");
        check(c.slots == t.helpers * t.ranks && c.lanes == c.slots * kClasses &&
              c.ring_entries == c.lanes * kLaneSlots && c.acks == t.helpers + t.ranks &&
              c.results == c.slots + t.ranks, "exact topology-derived capacities");
        Segment s(t);
        Header* h = s.header();
        check(h->counts.slots == c.slots && h->rank_count == t.ranks &&
              h->helper_count == t.helpers && h->placement_ack_count == c.acks,
              "native layout published the same counts");
        check(h->regions[wire::index(Region::Requests)].count == c.ring_entries &&
              h->regions[wire::index(Region::ResultPlanes)].count == c.results &&
              h->regions[wire::index(Region::Activations)].count == t.ranks,
              "ring and distinct slot/combine planes, shared activation by consumer");
        for (uint32_t helper = 0; helper < t.helpers; ++helper) {
            check(h->helper_pid[helper] == 0 && h->helper_start[helper] == 0,
                  "helper descriptor is initialized");
            for (uint32_t rank = 0; rank < t.ranks; ++rank) {
                uint32_t slot = slot_of(helper,rank,t.ranks);
                uint32_t lane = lane_of(slot,kSmallClasses,c.slots);
                check(lane < c.lanes && lane_class(lane,c.slots) == kSmallClasses &&
                      slot_helper(slot,t.ranks) == helper &&
                      slot_consumer(slot,t.ranks) == rank,
                      "rank/slot/lane are inverses");
                check(h->route_offset[slot] == wire::element_offset(s.layout,Region::Routes,slot) &&
                      h->activation_offset[slot] ==
                          wire::element_offset(s.layout,Region::Activations,rank) &&
                      h->result_offset[result_index(slot,rank)] ==
                          wire::element_offset(s.layout,Region::ResultPlanes,slot),
                      "all view offsets derive from counted regions");
                check(h->combine[rank].consumer == rank &&
                      h->combine[rank].lane[helper] == 0 &&
                      h->chunks[rank].done[0] == 0, "rank combine and contributor views");
            }
        }
        for (uint32_t rank = 0; rank < t.ranks; ++rank) {
            const uint32_t combined = combine_index(rank,c.slots);
            check(combined >= c.slots && combined < c.results &&
                  h->result_offset[combined] ==
                      wire::element_offset(s.layout,Region::ResultPlanes,combined),
                  "dedicated rank combine never aliases a helper result");
        }
        check(placement_ack(s.data,c.acks) == nullptr &&
              placement_ack(s.data,c.acks-1) != nullptr &&
              published_service(s.data,t.helpers) == nullptr &&
              published_service(s.data,t.helpers-1) != nullptr,
              "ACKs, services are bounded by actual participants");
    }
    wire::Counts count{};
    check(wire::counts_for({9,1},&count) == Refusal::CombineMask &&
          wire::counts_for({1,32},&count) == Refusal::ParticipantMask &&
          !slots_fit(9,1) && !slots_fit(1,32) && slots_fit(2,3),
          "only typed mask representation bounds refuse otherwise-countable topologies");
}

static void Descriptors_MutationAndDigest() {
    Segment s({2,3});
    Header* h = s.header();
    const auto l = s.layout;
    check(placement_control(s.data)->active_slot == 0 &&
          placement_control(s.data)->active_epoch == 1 &&
          placement_control(s.data)->pending_slot == UINT32_MAX &&
          generation_header(s.data,0)->state == uint32_t(GenerationState::ACTIVE) &&
          generation_header(s.data,1)->state == uint32_t(GenerationState::EMPTY),
          "initial generation, placement state, and slots");
    for (uint32_t layer = 0; layer < s.layers; ++layer) {
        check(h->layer_map_ready[layer] == 0 && h->layer_map_consumed[layer] == 0,
              "all layer publication words begin clear");
        for (uint32_t expert = 0; expert < kShape.experts; ++expert)
            check(layer_slot(s.data,layer,expert) == -1,
                  "unpublished layer owner is never helper zero");
    }
    for (uint32_t g = 0; g < wire::kGenerationSlots; ++g)
        for (uint32_t layer = 0; layer < s.layers; ++layer)
            for (uint32_t e = 0; e < kShape.experts; ++e)
                check(owner_at(s.data,g,layer,e) == kOwnerLocal, "initial ownership local");

    auto* c = placement_control(s.data);
    PlacementControl snapshot{};
    check(placement_snapshot(s.data,&snapshot) && snapshot.active_epoch == 1,
          "initial seqlock snapshot");
    c->seq = 1;
    check(!placement_snapshot(s.data,&snapshot), "odd seqlock cannot be read");
    c->seq = 2;
    check(placement_snapshot(s.data,&snapshot) && snapshot.seq == 2,
          "even seqlock snapshot");

    auto* g = generation_header(s.data,0);
    uint8_t* owners = generation_map(s.data,0);
    owners[0] = 0; owners[1] = 1;
    g->digest = generation_digest(s.data,0);
    c->active_digest = g->digest;
    g->ready_helpers = c->expected_helpers;
    store(&h->bootstrap_filtered_helpers,c->expected_helpers);
    check(bootstrap_generation_sealed(s.data) &&
          valid_self(h,size_t(l.bytes)), "sealed valid ownership");
    owners[2] = 1;
    check(!valid_self(h,size_t(l.bytes)), "sealed digest mutation is rejected");
    owners[2] = kOwnerLocal;
    owners[2] = 9;
    check(!valid_self(h,size_t(l.bytes)), "invalid owner id rejected");
    owners[2] = kOwnerLocal;
    check(valid_self(h,size_t(l.bytes)), "restored owners validated");

    const Header keep = *h;
    ++h->route_offset.relative;
    check(!valid_self(h,size_t(l.bytes)), "misdirected route view refused"); *h = keep;
    ++h->result_offset.start;
    check(!valid_self(h,size_t(l.bytes)), "misdirected result view refused"); *h = keep;
    ++h->lane.count;
    check(!valid_self(h,size_t(l.bytes)), "misreported lane count refused"); *h = keep;
    ++h->helper_count;
    check(!valid_self(h,size_t(l.bytes)), "helper count mismatch refused"); *h = keep;
    h->version = 25;
    check(!valid_self(h,size_t(l.bytes)), "retired v25 header refused"); *h = keep;
    check(wire::valid(s.data,l.bytes-1,kShape,s.layers,s.topology,kManifest) ==
              Refusal::BadMapping, "short mapping fails closed");
    check(wire::valid(s.data,l.bytes,kShape,s.layers,s.topology,kManifest+1) ==
              Refusal::WrongIdentity, "foreign placement digest rejected");
}

static void Digest_DeterministicAndSeparated() {
    const uint8_t owners[] = {0,0xff,1,0xff,1,0};
    const uint64_t a = placement_digest(0x0102030405060708ull,2,3,2,owners);
    check(a == 0x5e0154deaf149dd6ull &&
          a != placement_digest(0x0102030405060709ull,2,3,2,owners) &&
          a != placement_digest(0x0102030405060708ull,2,3,1,owners),
          "digest input domains are immutable");
    uint8_t changed[6]; std::memcpy(changed, owners, sizeof(changed)); changed[1]=0;
    check(a != placement_digest(0x0102030405060708ull,2,3,2,changed),
          "every owner byte affects placement digest");
}

static void CombineChunkWait_FirstBlockAlwaysPolls() {
    constexpr uint32_t chunks = (kMaxRows + kCombineChunkRows-1)/kCombineChunkRows;
    for (uint32_t pattern=0;pattern<16;++pattern)
        for (uint32_t chunk=0;chunk<chunks;++chunk) {
            const uint32_t first=chunk*kCombineChunkRows;
            uint32_t pollers=0;
            for (uint32_t t=0;t<kCombineChunkRows;++t) {
                const bool routed = t<4 ? (pattern>>t)&1u : t==kCombineChunkRows-1;
                const auto w=combine_chunk_wait(true,first+t,routed);
                pollers += w==CombineChunkWait::Poll;
                if (!routed && t)
                    check(w==CombineChunkWait::None,"unrouted non-first block does not wait");
            }
            check(pollers==1,"every chunk has exactly one poller");
        }
    check(combine_chunk_wait(true,0,false)==CombineChunkWait::Poll &&
          combine_chunk_wait(true,1,true)==CombineChunkWait::Mirror &&
          combine_chunk_wait(false,0,true)==CombineChunkWait::None,
          "chunk mirror wait follows initial publication");
}

static void Lane_DrainAndRequestIdentity() {
    Segment s({2,2});
    Header* h=s.header();
    const uint32_t slot=slot_of(1,1,2);
    const uint32_t lane=lane_of(slot,kSmallClasses,h->counts.slots);
    LaneSignal& ls=h->lane[lane];
    const uint64_t closure=27;
    ls.drain_request=closure;
    check(!acknowledge_lane_drain(ls,closure,[&] { return false; }) &&
          ls.drain_ack==0, "failed drain cannot certify");
    check(acknowledge_lane_drain(ls,closure,[&] { return true; }) &&
          ls.drain_ack==closure, "successful drain certifies exact closure");
    check(!acknowledge_lane_drain(ls,closure+1,[&] { return true; }),
          "stale closure cannot certify a newer drain");
    check(request_slot_valid(slot,2,1,1,slot,2,h->counts.slots) &&
          !request_slot_valid(slot,2,1,0,slot,2,h->counts.slots) &&
          !request_slot_valid(slot,1,0,1,slot,2,h->counts.slots) &&
          !request_slot_valid(h->counts.slots,2,0,0,h->counts.slots,2,h->counts.slots),
          "slot, consumer, layer, width and live count all checked");
    Request* q=lane_request(s.data,lane,1);
    check(q && q==wire::element<Request>(s.data,s.layout,Region::Requests,lane*kLaneSlots),
          "metadata ring indexed by counted lane");
    store(&ls.done_sequence,1);
    check(!publish_done_identity(s.data,lane),"unpublished request cannot certify last-done identity");
    q->sequence=1; q->generation_slot=1; q->placement_epoch=17;
    check(publish_done_identity(s.data,lane) && ls.done_epoch==17 && ls.done_slot==1,
          "done identity follows only the completed ring entry");
    q->generation_slot=wire::kGenerationSlots;
    check(!publish_done_identity(s.data,lane),"out-of-range generation never certifies");
    q->generation_slot=1;
    ls.request_sequence=1; ls.consumed_sequence=1; ls.done_sequence=1;
    ls.consumed_epoch=17; ls.consumed_slot=1;
    check(lane_quiesced(h,lane,closure,true), "exact done and consumed identity is quiescent");
    ls.consumed_epoch=16;
    check(!lane_quiesced(h,lane,closure,true), "stale consumption identity not quiescent");
}

// A model shape for the layout tests (the shipped paths read theirs from the container).
static constexpr LayoutShape fixture_shape() { return LayoutShape{kHidden, kTopK, kExperts, kMaxRows, kActGroup, 640}; }

static void SharedMapping_ReleaseAcquire() {
    const Topology t{1,2};
    Layout l{};
    check(wire::layout_for(fixture_shape(),48,t,&l)==Refusal::None,"mapped model layout");
    const std::string name="/r3-moe-shm-v26-test-"+std::to_string((long long)getpid());
    const int fd=shm_open(name.c_str(),O_CREAT|O_EXCL|O_RDWR,0600);
    check(fd>=0 && ftruncate(fd,off_t(l.bytes))==0, "shared memory create");
    void* base=mmap(nullptr,size_t(l.bytes),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    check(base!=MAP_FAILED &&
          wire::init(base,size_t(l.bytes),fixture_shape(),48,t,kManifest)==Refusal::None,
          "shared memory init");
    Header* h=static_cast<Header*>(base);
    const uint32_t slot=slot_of(0,1,t.ranks);
    const uint32_t lane=lane_of(slot,kSmallClasses,h->counts.slots);
    Request* q=lane_request(base,lane,1);
    Route* routes=ptr<Route>(base,h->route_offset[slot]);
    routes[0]={0,23,0x3f80,7};
    std::thread helper([&] {
        while (load(&h->lane[lane].request_sequence)!=1) {
            uint32_t seen=load(&h->publish_sequence);
            if (seen<1) futex_wait(&h->publish_sequence,seen);
        }
        check(q->slot==slot && q->consumer==1 && q->layer==7 &&
              q->route_count==1 && q->result_offset==h->result_offset[slot] &&
              request_slot_valid(q->slot,q->world,q->consumer,q->layer,
                                 slot,t.ranks,h->counts.slots),
              "helper reads exact live request identity");
        const Route* r=ptr<Route>(base,q->route_offset);
        check(r[0].expert==23 && r[0].weight_bf16==0x3f80 && r[0].slot==7,
              "published route is visible before request");
        ptr<uint16_t>(base,q->result_offset)[0]=0x4000;
        q->status=0;
        store(&q->completion_sequence,1);
        futex_wake(&q->completion_sequence);
    });
    const uint32_t seq=fetch_add(&h->next_sequence,1);
    check(seq==1,"first sequence is nonzero");
    q->layer=7; q->slot=slot; q->world=t.ranks; q->consumer=1; q->rows=1;
    q->hidden=h->hidden; q->route_count=1;
    q->route_offset=h->route_offset[slot];
    q->activation_offset=h->activation_offset[slot];
    q->result_offset=h->result_offset[slot];
    q->activation_stride=h->hidden*2;
    q->result_stride=h->hidden*2;
    store(&q->sequence,seq);
    store(&h->lane[lane].request_sequence,seq);
    store(&h->publish_sequence,seq);
    futex_wake(&h->publish_sequence);
    check(wait_until(&q->completion_sequence,seq,&h->helper_fault) &&
          ptr<uint16_t>(base,q->result_offset)[0]==0x4000,
          "helper result becomes visible through done release");
    helper.join();
    munmap(base,size_t(l.bytes)); close(fd); shm_unlink(name.c_str());
}

static void DeferredPublication_ManifestIdentity() {
    const std::string m1 = "sha256:0123456789abcdef" + std::string(48, '1');
    const std::string m2 = "sha256:0123456789abcdef" + std::string(47, '1') + "2";
    const uint64_t d1 = wire::manifest_identity(m1);
    const uint64_t d2 = wire::manifest_identity(m2);
    check(d1 && d2 && d1 != d2 && wire::manifest_identity("sha256:" + std::string(64, '0')) != 0 &&
          wire::manifest_identity("sha256:bad") == 0 &&
          wire::manifest_identity("sha256:" + std::string(64, 'G')) == 0,
          "all SHA-256 bytes contribute to nonzero strict manifest identity");
    constexpr Topology t{2,3};
    Layout l{};
    check(wire::layout_for(kShape, 2, t, &l) == Refusal::None, "deferred v26 layout");
    void* base = mmap(nullptr, size_t(l.bytes), PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    check(base != MAP_FAILED, "deferred v26 map");
    check(wire::init(base, size_t(l.bytes), kShape, 2, t, d1, true) == Refusal::None,
          "deferred init succeeds without allowing attachment");
    auto* h = static_cast<Header*>(base);
    check(h->published_helpers == 0 &&
          wire::valid(base, size_t(l.bytes), kShape, 2, t, d1) == Refusal::Unpublished,
          "attacher cannot validate a half-configured shared-memory segment");
    h->small_class_count = 2;
    h->small_class_rows[0] = 2;
    h->small_class_rows[1] = 8;
    wire::publish(h);
    check(wire::valid(base, size_t(l.bytes), kShape, 2, t, d1) == Refusal::None &&
          wire::valid(base, size_t(l.bytes), kShape, 2, t, d2) == Refusal::WrongIdentity,
          "published segment refuses a foreign manifest even with same shape and topology");
    munmap(base, size_t(l.bytes));
}

static void Schema_RepresentationParity() {
    // One versioned schema for the Python layout resolver, checked against
    // *native* v26::counts_for representation types on every card-free run.
    const auto path = std::filesystem::path(__FILE__).parent_path() / "topology-v26.schema.json";
    std::ifstream in(path);
    check(bool(in), "v26 topology schema accessible");
    const std::string doc{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    const auto field = [&](const char* name) -> uint32_t {
        const std::string key = std::string("\"") + name + "\"";
        size_t pos = doc.find(key);
        check(pos != std::string::npos, "v26 topology schema field exists");
        pos = doc.find(':', pos + key.size());
        check(pos != std::string::npos, "v26 topology schema field has numeric value");
        char* end = nullptr;
        const auto value = std::strtoul(doc.c_str() + pos + 1, &end, 10);
        check(end != doc.c_str() + pos + 1 && value <= UINT32_MAX,
              "v26 topology schema field is a uint32");
        return uint32_t(value);
    };
    check(field("wire_version") == kVersion &&
          field("owner_id_bits") == std::numeric_limits<uint8_t>::digits &&
          field("combine_mask_bits") == kCombineMaskBits &&
          field("participant_mask_bits") == std::numeric_limits<uint32_t>::digits &&
          field("count_bits") == std::numeric_limits<uint32_t>::digits &&
          field("lane_classes") == kClasses &&
          field("ring_slots") == kLaneSlots,
          "native/Python v26 representation schema parity");
    v26::Counts c{};
    check(v26::counts_for({8,24},&c)==Refusal::None && c.slots==192,
          "typed v26 mask boundary accepted");
    check(v26::counts_for({9,1},&c)==Refusal::CombineMask,
          "typed contributor mask boundary refused");
    check(v26::counts_for({8,25},&c)==Refusal::ParticipantMask,
          "typed participant mask boundary refused");
}

int main() {
    DeferredPublication_ManifestIdentity();
    Schema_RepresentationParity();
    Geometry_RuntimeCountsAndABI();
    Descriptors_MutationAndDigest();
    Digest_DeterministicAndSeparated();
    CombineChunkWait_FirstBlockAlwaysPolls();
    Lane_DrainAndRequestIdentity();
    SharedMapping_ReleaseAcquire();
    std::puts("r3_moe_shm_test: PASS");
    return 0;
}
