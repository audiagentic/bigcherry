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
int main() {
    using namespace r3::moe_shm;
    const LayoutShape shape{128,2,3,8,128,64};
    Layout layout{};
    check(layout_for(shape,2,slots_for(2),&layout),"layout");
    void* map=mmap(nullptr,layout.bytes,PROT_READ|PROT_WRITE,
                   MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    check(map!=MAP_FAILED && init(map,2,shape,2),"init");
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
        q->rank_count=1; q->rank=(q==q0?0:1); q->layer=0;   /* as publish writes it: rank = this helper's slot */
        q->placement_epoch=ctrl->active_epoch;
        q->generation_slot=ctrl->active_slot;
        q->route_count=0; q->route_offset=h->route_offset[q==q0?0:1];
    }
    q0->combine_generation=1;
    h->combine.request=1; h->combine.rows=1;
    h->combine.seq[0]=h->combine.seq[1]=1;
    h->combine.lane[0]=lane0; h->combine.lane[1]=lane1;
    h->chunks.seq[0][0]=h->chunks.seq[1][0]=1;
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
    check(!run() && !h->combine.done,"wrong helper epoch rejected");
    q1->placement_epoch=epoch;
    q1->generation_slot=1;
    check(!run() && !h->combine.done,"wrong helper slot rejected");
    q1->generation_slot=0;
    q1->layer=1;
    check(!run() && !h->combine.done,"wrong helper layer rejected");
    q1->layer=0;
    q1->rank_count=2;
    check(!run() && !h->combine.done,"wrong helper participant shape rejected");
    q1->rank_count=1;
    h->combine.lane[1]=lane_of(1,0);
    check(!run() && !h->combine.done,"wrong helper lane class rejected");
    h->combine.lane[1]=lane1;
    h->combine.request=2;
    check(!run() && !h->combine.done,"stale combine generation rejected");
    h->combine.request=1;
    q0->placement_epoch=epoch+1;
    check(!run() && !h->combine.done,"helper zero wrong active epoch rejected");
    q0->placement_epoch=epoch;
    // Both helpers can agree with each other yet carry a non-active epoch.
    q0->placement_epoch=q1->placement_epoch=epoch+1;
    check(!run() && !h->combine.done && !h->chunks.done[0],
          "non-active epoch never publishes an early chunk");
    q0->placement_epoch=q1->placement_epoch=epoch;
    check(run() && h->combine.done==1 && h->chunks.done[0]==1,
          "production combine success with matching participants");
    h->combine.done=h->chunks.done[0]=0;


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
    h->chunks.seq[1][0]=0;
    std::atomic<bool> finished{false};
    bool success=true;
    std::thread blocked([&] {success=run();finished.store(true,std::memory_order_release);});
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    check(!finished.load(std::memory_order_acquire) && !h->combine.done,
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
    check(finished.load() && !success && !h->combine.done,
          "stalled combine exits without publishing done on helper fault");
    check(munmap(map,layout.bytes)==0,"unmap");
    std::puts("r3_combine_epoch_test: ok");
}
