// Card-free production combine path: old epoch/slot, participant identity,
// stalled chunks and faulted publication must never combine across a flip.
#include "moe_remote_shm.h"
#include "worker_epoch.h"
#include "worker_bootstrap.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <thread>
#include <vector>
namespace { inline bool stopping() { return false; } }
#include "worker_combine.h"

static void check(bool ok,const char* why) {
    if(!ok) {std::fprintf(stderr,"r3_combine_epoch_test: %s\n",why);std::exit(1);}
}
// Mirrors the publisher's multi-helper large-lane layout behind an engine of `world`
// ranks (RAD49): a request exists only for a helper with routes; the consumer rank's
// slot of each helper, slot_of(k, consumer, world), is the Request::slot, every request
// names the consumer, and every participating helper shares the pass row_base. The
// layer is the consumer, so consumer_of(layer, world) names it.
static void publisher_shaped_combine(uint32_t helpers, uint32_t world, uint32_t consumer, bool empty_tail) {
    using namespace r3::moe_shm;
    constexpr uint32_t rows = 2 * kCombineChunkRows + 9;
    const LayoutShape shape{128, 2, 3, rows, 128, 64};
    Layout layout{};
    check(slots_fit(helpers, world) && consumer < world, "publisher shape");
    check(layout_for(shape, 2, slots_for(helpers, world), &layout), "publisher layout");
    void* map = mmap(nullptr, layout.bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    check(map != MAP_FAILED && init(map, 2, shape, helpers, world), "publisher init");
    auto* h = static_cast<Header*>(map);
    h->helper_count = helpers;
    auto* ctrl = placement_control(map);
    ctrl->expected_ranks = (1u << world) - 1u;
    const uint32_t layer = world == 1 ? 1u : consumer;
    CombineSignal& cs = h->combine[consumer];
    CombineChunks& ch = h->chunks[consumer];
    for (uint32_t k = 0; k < helpers; ++k)
        check(r3::epoch::mark_bootstrap_filtered(map, k), "publisher bootstrap filtered");
    r3::epoch::SharedTransport bootstrap(map);
    check(bootstrap.seal_bootstrap(), "publisher bootstrap sealed");

    // Distinct real route lists overlap in every chunk and also leave
    // unrouted rows. The third helper has zero routes in one three-helper step.
    const std::array<std::array<uint32_t, 6>, 3> routed{{
        {{0, 1, 256, 257, 512, rows - 1}},
        {{1, 255, 256, 512, rows - 1, rows - 2}},
        {{1, 2, 255, 257, 512, rows - 1}}
    }};
    constexpr uint16_t bf16[3] = {0x3f80, 0x4000, 0x4080}; // 1, 2, 4
    constexpr uint16_t fp16[8] = {
        0xbeef, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700
    };
    const uint32_t generation = 7;
    cs.request = generation;
    cs.rows = rows;
    std::vector<uint8_t> expected(rows, 0);
    Request* q[kCombineHelpers] = {};
    for (uint32_t k = 0; k < helpers; ++k) {
        const uint32_t slot = slot_of(k, consumer, world);
        const uint32_t lane = lane_of(slot, kSmallClasses);
        cs.lane[k] = lane;
        const uint32_t seq = k + 5; // independent producer lane counters
        if (empty_tail && k == helpers - 1) {
            cs.seq[k] = 0; // published by the kernel, no Request
            continue;
        }
        cs.seq[k] = seq;
        q[k] = lane_request(map, lane, seq);
        check(q[k] != nullptr, "publisher ring");
        q[k]->sequence = seq;
        q[k]->layer = layer;
        q[k]->slot = slot;
        q[k]->world = world;
        q[k]->consumer = consumer;
        q[k]->rows = rows;
        q[k]->hidden = h->hidden;
        q[k]->route_count = (uint32_t)routed[k].size();
        q[k]->row_base = 19;
        q[k]->route_offset = h->route_offset[slot];
        q[k]->activation_offset = h->activation_offset[slot];
        q[k]->activation_stride = h->hidden * sizeof(uint16_t);
        q[k]->activation_q_stride = h->hidden;
        q[k]->activation_s_stride = h->hidden / shape.act_group * sizeof(float);
        q[k]->placement_epoch = ctrl->active_epoch;
        q[k]->generation_slot = ctrl->active_slot;
        q[k]->combine_generation = k == 0 ? generation : 0;
        auto* routes = ptr<Route>(map, q[k]->route_offset);
        for (uint32_t i = 0; i < q[k]->route_count; ++i) {
            const uint32_t token = routed[k][i];
            routes[i] = Route{(int32_t)token, (int32_t)k, 0x3f80, (uint16_t)i};
            expected[token] |= (uint8_t)(1u << k);
        }
        auto* plane = ptr<uint16_t>(map, h->result_offset[result_index(slot, consumer)]);
        std::fill_n(plane, (size_t)rows * h->hidden, bf16[k]);
        for (uint32_t c = 0; c < (rows + kCombineChunkRows - 1) / kCombineChunkRows; ++c)
            ch.seq[k][c] = seq;
        h->lane[lane].request_sequence = seq;
        h->lane[lane].done_sequence = seq;
    }
    check(q[0] && q[1] && (!empty_tail || !q[helpers - 1]), "publisher participation");
    check(expected[0] == 1 && expected[1] == (empty_tail || helpers == 2 ? 3 : 7) &&
          expected[3] == 0 && expected[256] == 3 &&
          expected[512] == (empty_tail || helpers == 2 ? 3 : 7), "route mask geometry");
    auto* out = ptr<uint16_t>(map, h->result_offset[combine_index(consumer)]);
    std::fill_n(out, (size_t)rows * h->hidden, fp16[0]);
    /* The combine plane is never a result plane of any slot (RAD49). */
    for (uint32_t s = 0; s < h->slot_count; ++s)
        check(combine_index(consumer) != result_index(s, slot_consumer(s, world)), "combine plane is free");
    /* The other consumer's combine state stays untouched. */
    if (world > 1) {
        h->combine[1u - consumer].request = 0;
        h->combine[1u - consumer].done = 0;
    }
    CombinePool pool(3);
    std::vector<uint8_t> mask(h->max_rows);
    CombineStats stats{};
    auto run = [&] { return combine_step(map, q[0], helpers, &pool, &mask, &stats); };

    // Every validation executes before the first chunk is published. Each
    // mutated request represents a corruption of an actual publisher field.
    auto reject = [&](uint32_t& field, uint32_t wrong, const char* why) {
        const uint32_t original = field;
        field = wrong;
        cs.done = 0;
        for (auto& done : ch.done) done = 0;
        check(!run() && !cs.done && !ch.done[0], why);
        field = original;
    };
    const uint32_t other_world = world == 1 ? 2u : 1u;
    reject(q[0]->slot, slot_of(1, consumer, world), "helper zero must use its own slot for the consumer");
    reject(q[1]->slot, slot_of(0, consumer, world), "helper one's request must be in its own slot");
    reject(q[0]->world, other_world, "helper zero's world must be the segment's");
    reject(q[1]->world, other_world, "helper one's world must match helper zero's");
    if (world > 1) {
        reject(q[0]->consumer, 1u - consumer, "helper zero's consumer must match its slot");
        reject(q[1]->consumer, 1u - consumer, "helper one's consumer must match helper zero's");
        reject(q[1]->layer, q[1]->layer + 1, "every helper's layer must be the consumer's");
    }
    reject(q[1]->row_base, q[1]->row_base + 1, "shared pass row_base");
    reject(q[0]->row_base, q[0]->row_base + 1, "helper zero pass row_base");
    if (q[2]) reject(q[2]->slot, slot_of(1, consumer, world), "third helper's request must be in its own slot");

    // The real lane completion is required even when all chunks are present.
    // A full three-helper step exercises the second peer's terminal word.
    if (helpers == 3 && !empty_tail) {
        const uint32_t lane = cs.lane[2];
        const uint32_t seq = cs.seq[2];
        h->lane[lane].done_sequence = 0;
        std::atomic<bool> completed{false};
        bool success = false;
        std::thread pending([&] {
            success = run();
            completed.store(true, std::memory_order_release);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        check(!completed.load(std::memory_order_acquire) && !cs.done,
              "third helper completion gates combined done");
        store(&h->lane[lane].done_sequence, seq);
        pending.join();
        check(success, "three-helper production completion");
    } else {
        check(run(), "publisher-shaped combine success");
    }
    check(cs.done == generation && stats.steps == 1,
          "combined generation and stats");
    for (uint32_t c = 0; c < (rows + kCombineChunkRows - 1) / kCombineChunkRows; ++c)
        check(ch.done[c] == generation, "every chunk completion generation");
    if (world > 1)
        check(!h->combine[1u - consumer].done && !h->chunks[1u - consumer].done[0],
              "the other consumer's combine words are never written");
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t j = 0; j < h->hidden; ++j)
            check(out[(size_t)r * h->hidden + j] == fp16[expected[r]],
                  "combined fp16 output matches actual route masks");
    if (empty_tail)
        check(cs.seq[helpers - 1] == 0 &&
              h->lane[lane_of(slot_of(helpers - 1, consumer, world), kSmallClasses)].done_sequence == 0,
              "empty tail never publishes a request or completion");
    check(munmap(map, layout.bytes) == 0, "publisher unmap");
}

int main() {
    publisher_shaped_combine(2, 1, 0, false);
    publisher_shaped_combine(3, 1, 0, false);
    publisher_shaped_combine(3, 1, 0, true);
    /* RAD49: two helpers behind a TP2 engine, each consumer rank with its own slots, signal, chunks and plane. */
    publisher_shaped_combine(2, 2, 0, false);
    publisher_shaped_combine(2, 2, 1, false);
    using namespace r3::moe_shm;
    const LayoutShape shape{128,2,3,8,128,64};
    Layout layout{};
    check(layout_for(shape,2,slots_for(2, 1),&layout),"layout");
    void* map=mmap(nullptr,layout.bytes,PROT_READ|PROT_WRITE,
                   MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    check(map!=MAP_FAILED && init(map,2,shape,2, 1),"init");
    auto* h=static_cast<Header*>(map);
    auto* ctrl=placement_control(map);
    h->helper_count=2;
    ctrl->expected_ranks=1;
    check(r3::epoch::mark_bootstrap_filtered(map,0) &&
          r3::epoch::mark_bootstrap_filtered(map,1),"bootstrap filtered");
    r3::epoch::SharedTransport bootstrap(map);
    check(bootstrap.seal_bootstrap(),"bootstrap sealed");

    const uint32_t lane0=lane_of(0,kSmallClasses),lane1=lane_of(1,kSmallClasses);
    Request* q0=lane_request(map,lane0,1);
    Request* q1=lane_request(map,lane1,1);
    for(Request* q : {q0,q1}) {
        q->sequence=1; q->rows=1; q->hidden=h->hidden;
        q->world=1; q->consumer=0; q->slot=(q==q0?0:1); q->layer=0;   /* as publish writes it: slot_of(helper, 0, 1) */
        q->placement_epoch=ctrl->active_epoch;
        q->generation_slot=ctrl->active_slot;
        q->route_count=0; q->route_offset=h->route_offset[q==q0?0:1];
    }
    q0->combine_generation=1;
    CombineSignal& cs=h->combine[0];
    CombineChunks& ch=h->chunks[0];
    cs.request=1; cs.rows=1;
    cs.seq[0]=cs.seq[1]=1;
    cs.lane[0]=lane0; cs.lane[1]=lane1;
    ch.seq[0][0]=ch.seq[1][0]=1;
    h->lane[lane1].done_sequence=1;
    CombinePool pool(2);
    std::vector<uint8_t> mask(h->max_rows);
    CombineStats stats{};
    auto run=[&] { return combine_step(map,q0,2,&pool,&mask,&stats); };

    check(!pool.combine(nullptr,2,nullptr,nullptr,1,128,CombineIdentity{1,2,1,3}),
          "pool rejects invalid ticket");
    uint16_t first[128]{}, second[128]{}, dest[128]{};
    const uint16_t* inputs[2]={first,second};
    uint8_t no_rows[1]{};
    check(!pool.combine(inputs,2,dest,no_rows,1,128,CombineIdentity{1,0,1,2}),
          "pool rejects participant without helper zero");

    const uint64_t epoch=q1->placement_epoch;
    q1->placement_epoch=epoch+1;
    check(!run() && !cs.done,"wrong helper epoch rejected");
    q1->placement_epoch=epoch;
    q1->generation_slot=1;
    check(!run() && !cs.done,"wrong helper slot rejected");
    q1->generation_slot=0;
    q1->layer=1;
    check(!run() && !cs.done,"wrong helper layer rejected");
    q1->layer=0;
    q1->world=2;
    check(!run() && !cs.done,"wrong helper participant shape rejected");
    q1->world=1;
    cs.lane[1]=lane_of(1,0);
    check(!run() && !cs.done,"wrong helper lane class rejected");
    cs.lane[1]=lane1;
    cs.request=2;
    check(!run() && !cs.done,"stale combine generation rejected");
    cs.request=1;
    q0->placement_epoch=epoch+1;
    check(!run() && !cs.done,"helper zero wrong active epoch rejected");
    q0->placement_epoch=epoch;
    // Both helpers can agree with each other yet carry a non-active epoch.
    q0->placement_epoch=q1->placement_epoch=epoch+1;
    check(!run() && !cs.done && !ch.done[0],
          "non-active epoch never publishes an early chunk");
    q0->placement_epoch=q1->placement_epoch=epoch;
    // This single-row fixture tests rejected protocol states only. Positive
    // combine cases above use publisher-shaped routes, lanes and real chunks.
    check(!cs.done && !ch.done[0], "negative-only fixture stays uncompleted");


    // Fake non-power-saving small lane: all kLaneSlots waits remain queued
    // after a burst. The production drain predicate must refuse to ACK until
    // their release+stream synchronization and the consumed parent proof.
    constexpr uint32_t small=lane_of(0,0);
    auto& signal=h->lane[small];
    const uint64_t closure=99;
    signal.drain_request=closure;
    signal.request_sequence=1;
    signal.done_sequence=1;
    signal.consumed_sequence=1;
    signal.consumed_epoch=ctrl->active_epoch;
    signal.consumed_slot=ctrl->active_slot;
    auto* parent=lane_request(map,small,1);
    parent->sequence=1;
    parent->placement_epoch=ctrl->active_epoch;
    parent->generation_slot=ctrl->active_slot;
    std::array<bool,kLaneSlots> queued{{true,true,true,true}};
    auto sync=[&] {
        return std::none_of(queued.begin(),queued.end(),[](bool live){return live;});
    };
    auto drain=[&]{return sync() && publish_done_identity(map,small);};
    check(!acknowledge_lane_drain(signal,closure,drain) &&
          !lane_quiesced(h,small,closure,true),
          "power-save-off cannot ACK with four old prequeued waits");
    for(auto& live:queued) live=false;
    check(acknowledge_lane_drain(signal,closure,drain) &&
          lane_quiesced(h,small,closure,true),
          "exact stream drain and consumed identity prove quiescence");
    signal.consumed_epoch=parent->placement_epoch+1;
    check(!lane_quiesced(h,small,closure,true),
          "stale consumed epoch cannot certify RETIRED");
    signal.consumed_epoch=parent->placement_epoch;
    signal.drain_request=closure+1;
    check(!acknowledge_lane_drain(signal,closure,drain),
          "late closure cannot reuse old drain ACK");
    check(!lane_quiesced(h,small,closure+1,true),
          "new closure never inherits old ACK");

    // A missing chunk keeps helper 0's pool/lane live. The control plane must
    // not infer RETIRED from the other helper's last completion word or an
    // unrelated/stale ACK. Inject a fault to release the stalled combiner.
    ch.seq[1][0]=0;
    std::atomic<bool> finished{false};
    bool success=true;
    std::thread blocked([&] {success=run();finished.store(true,std::memory_order_release);});
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    check(!finished.load(std::memory_order_acquire) && !cs.done,
          "missing chunk stalls real combine_step");
    struct Clock : r3::epoch::Clock {
        uint64_t now_ns() const override {return 1;}
    } clock;
    struct Transport : r3::epoch::Transport {
        PlacementAck ack{};
        void publish(const PlacementControl&,const PlacementJournal&) override {}
        PlacementAck read_ack(uint32_t) const override {return ack;}
    } transport;
    r3::epoch::Controller controller(transport,clock,ctrl->active_digest,2,1);
    check(controller.propose(ctrl->active_digest+1) && controller.start_reserving(),
          "test controller transaction");
    // Missing/faulted ACKs cannot advance RESERVING, and the old output
    // remains live; the phase deadline is the bounded escape.
    check(controller.poll()==r3::epoch::Poll::WAITING,"no reserved ACK");
    transport.ack.transaction_id=controller.control().transaction_id;
    transport.ack.epoch=controller.control().pending_epoch;
    transport.ack.generation_slot=controller.control().pending_slot;
    transport.ack.digest=controller.control().pending_digest;
    transport.ack.phase=uint32_t(AckPhase::RESERVED);
    transport.ack.fault_code=uint32_t(r3::epoch::FaultCode::DEVICE);
    check(controller.poll()==r3::epoch::Poll::FAILED &&
          controller.control().active_epoch==1,
          "faulted ACK aborts before any ownership flip");
    store(&h->helper_fault,1u);
    blocked.join();
    check(finished.load() && !success && !cs.done,
          "stalled combine exits without publishing done on helper fault");
    check(munmap(map,layout.bytes)==0,"unmap");
    std::puts("r3_combine_epoch_test: ok");
}
