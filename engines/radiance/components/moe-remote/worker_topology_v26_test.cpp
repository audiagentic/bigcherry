// CPU-only RCS14 gate C: real v26 descriptor mapping, participant lifecycle,
// and budget planning for TP3/greater than the retired slot capacity.
#include "worker_topology_v26.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>

namespace v = r3::moe_shm::v26;
namespace cpu = r3::moe_shm::v26::cpu;
using r3::moe_shm::AckPhase;
using r3::moe_shm::GenerationState;
using r3::moe_shm::LayoutShape;
using r3::moe_shm::PlacementState;

static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "r3_worker_topology_v26_test: FAILED %s\n", what); std::exit(1); }
}

static constexpr uint64_t manifest_digest = 0x82374ae0951bcf16ull;
struct Fixture {
    LayoutShape shape;
    uint32_t layers;
    v::Topology topology;
    v::Layout layout{};
    void* base = nullptr;
    cpu::Segment segment;
    Fixture(LayoutShape s, uint32_t layer_count, v::Topology t)
        : shape(s), layers(layer_count), topology(t) {
        check(v::layout_for(shape, layers, t, &layout) == v::Refusal::None, "geometry");
        base = mmap(nullptr, layout.bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        check(base != MAP_FAILED, "anonymous mapped CPU segment");
        check(v::init(base, layout.bytes, shape, layers, t, manifest_digest) == v::Refusal::None,
              "v26 initialized and published");
        check(segment.attach(base, layout.bytes, shape, layers, t, manifest_digest) == cpu::Refusal::None,
              "validated attach");
    }
    ~Fixture() { munmap(base, layout.bytes); }
};
static void ack_phase(Fixture& f, cpu::Epoch& e, AckPhase phase, uint64_t cutoff = 0) {
    const auto& c = e.control();
    for (uint32_t actor = 0; actor < f.layout.counts.acks; ++actor) {
        auto* a = f.segment.ack(actor);
        *a = {};
        a->transaction_id = c.transaction_id;
        a->epoch = c.pending_epoch;
        a->digest = c.pending_digest;
        a->generation_slot = c.pending_slot;
        a->phase = uint32_t(phase);
        a->drained_step = cutoff;
    }
}
static void cleanup(Fixture& f, cpu::Epoch& e) {
    const auto& c = e.control();
    for (uint32_t actor = 0; actor < f.layout.counts.acks; ++actor) {
        auto* a = f.segment.ack(actor);
        *a = {};
        a->transaction_id = c.transaction_id;
        a->epoch = c.active_epoch;
        a->digest = c.active_digest;
        a->generation_slot = c.active_slot;
        a->phase = uint32_t(actor == f.topology.helpers ?
                            AckPhase::REOPENED : AckPhase::CLEANED);
    }
}

static void actor_and_resource_geometry(Fixture& f) {
    check(f.layout.counts.acks == f.topology.helpers + f.topology.ranks,
          "ACK region uses the actual helper + rank count");
    check(f.layout.regions[v::index(v::Region::Services)].count == f.topology.helpers,
          "service records follow counted ACKs");
    for (uint32_t helper = 0; helper < f.topology.helpers; ++helper) {
        uint32_t actor = UINT32_MAX;
        check(f.segment.actor(true, helper, &actor) && actor == helper &&
              f.segment.helper_ack(helper) == f.segment.ack(helper) &&
              f.segment.service(helper) != nullptr && f.segment.helper(helper) != nullptr,
              "helper actor, PID and service use counted descriptors");
        f.segment.helper(helper)->pid = helper + 100;
        check(f.segment.helper(helper)->pid == helper + 100, "PID stored per actual helper");
    }
    for (uint32_t rank = 0; rank < f.topology.ranks; ++rank) {
        uint32_t actor = UINT32_MAX;
        check(f.segment.actor(false, rank, &actor) &&
              actor == f.topology.helpers + rank &&
              f.segment.rank_ack(rank) == f.segment.ack(actor),
              "rank ACK index follows actual number of helper ACKs");
    }
    uint32_t actor = 0;
    check(!f.segment.actor(true, f.topology.helpers, &actor) &&
          !f.segment.actor(false, f.topology.ranks, &actor) &&
          !f.segment.actor(true, 0, nullptr) &&
          !f.segment.ack(f.layout.counts.acks) &&
          !f.segment.helper(f.topology.helpers) &&
          !f.segment.service(f.topology.helpers),
          "unrecognized role/index cannot read beyond mapped records");
    uint8_t owner = 0;
    check(f.segment.owner(0, 0, 0, &owner) &&
          owner == r3::moe_shm::kOwnerLocal &&
          !f.segment.owner(0, f.layers, 0, &owner) &&
          !f.segment.owner(0, 0, f.shape.experts, &owner) &&
          !f.segment.owner(v::kGenerationSlots, 0, 0, &owner),
          "owner tables are counted by model and generation");

    cpu::Segment foreign;
    check(foreign.attach(f.base, f.layout.bytes, f.shape, f.layers,
                         {f.topology.helpers+1, f.topology.ranks}, manifest_digest)
              == cpu::Refusal::InvalidSegment &&
          foreign.attach(f.base, f.layout.bytes, f.shape, f.layers,
                         f.topology, manifest_digest+1) == cpu::Refusal::InvalidSegment &&
          foreign.attach(f.base, f.layout.bytes-1, f.shape, f.layers,
                         f.topology, manifest_digest) == cpu::Refusal::InvalidSegment &&
          !foreign.ack(0),
          "changed live helper count / manifest / truncated map refused");
    auto* header = static_cast<v::Header*>(f.base);
    const auto version = header->version;
    header->version = r3::moe_shm::kVersion;
    check(foreign.attach(f.base, f.layout.bytes, f.shape, f.layers,
                         f.topology, manifest_digest) == cpu::Refusal::InvalidSegment,
          "no v25 fallback");
    header->version = version;

    const r3::worker_footprint::Shape s{
        f.layers, f.shape.experts, f.shape.hidden, f.shape.intermediate,
        f.shape.top_k, f.shape.act_group
    };
    r3::worker_footprint::SmallClasses classes{};
    classes.count = 2;
    classes.rows[0] = 4;
    classes.rows[1] = 8;
    const uint32_t large_rows = f.shape.max_rows;
    cpu::ResourcePlan save, all;
    // Test hardware budget deliberately derived from a model-independent
    // high upper bound, never used as a default in production.
    const uint64_t capacity = UINT64_C(1) << 48;
    check(cpu::resources(f.segment, s, classes, large_rows, 0, true, 4096,
                         capacity, 0, &save) == cpu::Refusal::None &&
          cpu::resources(f.segment, s, classes, large_rows, 0, false, 4096,
                         capacity, 0, &all) == cpu::Refusal::None,
          "card-free counted footprint passes");
    const uint32_t open_save = f.topology.ranks * (classes.count + 1u);
    const uint32_t open_all = f.layout.counts.slots * (classes.count + 1u);
    check(save.opened.size() == f.layout.counts.lanes &&
          all.opened.size() == f.layout.counts.lanes &&
          save.threads == open_save && all.threads == open_all &&
          save.queued == save.threads * r3::moe_shm::kLaneSlots &&
          all.queued == all.threads * r3::moe_shm::kLaneSlots,
          "threads and request queue come from actual open lane count / ring depth");
    check(save.lane_bytes < all.lane_bytes || open_save == open_all,
          "power-save off reserves all real topology slots");
    check(save.pointer_bytes == r3::worker_footprint::pointer_bank_bytes(s) &&
          save.map_bytes == r3::worker_footprint::slot_map_bytes(s),
          "v26 planner matches immutable model-only pointer and owner storage");
    check(save.fixed_bytes == save.lane_bytes + save.pointer_bytes +
          save.map_bytes + 4096 && save.expert_budget == capacity - save.fixed_bytes &&
          save.host_segment_bytes == f.layout.bytes,
          "budget subtracts the runtime topology footprint");
    check(cpu::resources(f.segment, s, classes, large_rows, 0, false, 4096,
                         all.fixed_bytes, 0, &save) == cpu::Refusal::BudgetExceeded,
          "refuse a card that cannot hold the topology");
    check(cpu::resources(f.segment, s, classes, large_rows,
                         f.topology.helpers, true, 4096, capacity, 0, &save)
              == cpu::Refusal::InvalidRole,
          "invalid helper cannot budget another helper's lanes");
    check(cpu::resources(f.segment, s, classes, large_rows+1, 0, true, 4096,
                         capacity, 0, &save) == cpu::Refusal::InvalidShape,
          "oversized lane rows refused");
}

static void full_cycle(Fixture& f) {
    cpu::Epoch e(f.segment, 100);
    check(e.refusal() == cpu::Refusal::None &&
          e.control().expected_helpers == (uint32_t(1) << f.topology.helpers)-1 &&
          e.control().expected_ranks == (uint32_t(1) << f.topology.ranks)-1,
          "expected roles are counted from topology");
    check(f.segment.control()->state == uint32_t(PlacementState::ACTIVE) &&
          f.segment.control()->seq == e.control().seq && !(e.control().seq & 1u),
          "control is coherently published to counted v26 placement region");
    check(e.begin(200) && e.control().pending_epoch == 2 &&
          e.control().state == uint32_t(PlacementState::RESERVING), "begin txn");
    check(e.step() == cpu::Progress::Waiting, "no RESERVED is not approval");
    ack_phase(f, e, AckPhase::RESERVED);
    check(e.step() == cpu::Progress::Advanced &&
          e.control().state == uint32_t(PlacementState::COPYING),
          "counted RESERVED from every role");
    ack_phase(f, e, AckPhase::READY);
    check(e.step() == cpu::Progress::Advanced &&
          f.segment.control()->state == uint32_t(PlacementState::PREPARED) &&
          f.segment.control()->seq == e.control().seq &&
          e.control().state == uint32_t(PlacementState::PREPARED),
          "counted READY produces PREPARED only");
    check(e.step() == cpu::Progress::Waiting && e.quiesce(), "explicit quiesce gate");
    ack_phase(f, e, AckPhase::QUIESCED, 42);
    check(e.step() == cpu::Progress::Advanced &&
          e.control().state == uint32_t(PlacementState::COMMIT),
          "all ranks and helpers attest same drained cutoff");
    ack_phase(f, e, AckPhase::COMMITTED);
    check(e.step() == cpu::Progress::Refused &&
          e.refusal() == cpu::Refusal::WrongAckIdentity,
          "cannot commit before true generation is ACTIVE");
    // Independently complete a clean transaction to test COMMIT/RETIRING.
    cpu::Epoch b(f.segment, 100);
    check(b.begin(200), "fresh exact transaction");
    ack_phase(f, b, AckPhase::RESERVED);
    check(b.step() == cpu::Progress::Advanced, "reserve");
    ack_phase(f, b, AckPhase::READY);
    check(b.step() == cpu::Progress::Advanced && b.quiesce(), "prepare");
    ack_phase(f, b, AckPhase::QUIESCED, 42);
    check(b.step() == cpu::Progress::Advanced, "quiesce");
    auto* gen = f.segment.generation(b.control().pending_slot);
    gen->epoch = b.control().pending_epoch;
    gen->digest = b.control().pending_digest;
    gen->state = uint32_t(GenerationState::ACTIVE);
    ack_phase(f, b, AckPhase::COMMITTED);
    check(b.step() == cpu::Progress::Advanced &&
          b.control().active_epoch == 2 &&
          b.control().state == uint32_t(PlacementState::RETIRING),
          "commit updates active generation only after published ACTIVE");
    ack_phase(f, b, AckPhase::RETIRED);
    check(b.step() == cpu::Progress::Advanced &&
          b.control().state == uint32_t(PlacementState::ACTIVE),
          "all participants RETIRED");
    check(!b.begin(300), "next proposal awaits cleanup");
    cleanup(f, b);
    check(b.cleaned() && b.begin(300), "counted helper CLEANED / rank REOPENED");
}

static void missing_and_mismatched(Fixture& f) {
    const uint32_t rank_actor = f.topology.helpers + f.topology.ranks - 1;
    const uint32_t helper_actor = f.topology.helpers - 1;
    for (uint32_t broken : {helper_actor, rank_actor}) {
        for (int variant = 0; variant < 5; ++variant) {
            cpu::Epoch e(f.segment, 11);
            check(e.begin(22), "begin ACK fault injection");
            ack_phase(f, e, AckPhase::RESERVED);
            auto* a = f.segment.ack(broken);
            if (variant == 0) *a = {}; // lost
            if (variant == 1) --a->transaction_id; // stale
            if (variant == 2) ++a->epoch; // wrong generation
            if (variant == 3) a->digest ^= 1; // mismatched map
            if (variant == 4) a->fault_code = 1; // producer fault
            const auto state = e.step();
            if (variant <= 1) {
                check(state == cpu::Progress::Waiting && e.refusal() == cpu::Refusal::None,
                      "lost/stale helper or rank never satisfies RESERVED");
            } else {
                check(state == cpu::Progress::Refused &&
                      e.refusal() == (variant == 4 ? cpu::Refusal::ParticipantFault :
                                                  cpu::Refusal::WrongAckIdentity),
                      "current participant mismatch or fault fails closed");
            }
            ack_phase(f, e, AckPhase::RESERVED);
            if (variant <= 1) check(e.step() == cpu::Progress::Advanced, "retry with exact ACK");
            else check(e.step() == cpu::Progress::Refused, "terminal refusal not cleared");
        }
    }
    // Deliberately publish rank's ACK at the old compiled base: never pass.
    if (f.topology.helpers != r3::moe_shm::kMaxHelpers) {
        cpu::Epoch e(f.segment, 30);
        check(e.begin(31), "old-base mutation setup");
        ack_phase(f, e, AckPhase::RESERVED);
        *f.segment.rank_ack(0) = {};
        auto* wrong = f.segment.ack(r3::moe_shm::kMaxHelpers);
        if (wrong) {
            wrong->phase = uint32_t(AckPhase::RESERVED);
            wrong->transaction_id = e.control().transaction_id - 1u;
        }
        check(e.step() == cpu::Progress::Waiting, "old fixed ACK base cannot prove rank READY");
    }
}

#include "worker_topology_v26_phase_test.h"

int main() {
    const v::Topology topologies[] = {{1,3}, {2,3}, {3,2}, {4,1}, {2,2}};
    struct Shape { LayoutShape shape; uint32_t layers; };
    const Shape shapes[] = {
        {{2048, 4, 256, 64, 128, 768}, 32},
        {{2560, 16, 1024, 64, 128, 3072}, 64}
    };
    for (auto s : shapes) for (auto t : topologies) {
        Fixture f(s.shape, s.layers, t);
        actor_and_resource_geometry(f);
        full_cycle(f);
        missing_and_mismatched(f);
        all_phase_faults(f);
    }
    std::puts("r3_worker_topology_v26_test: PASS");
}
