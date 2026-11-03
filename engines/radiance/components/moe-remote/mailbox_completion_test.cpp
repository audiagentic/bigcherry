/* Card-free boundary tests for the GPU wait/finish identity predicates,
 * the once-per-parent result-consumed seal, and the existing event ledger's
 * all-rank closed-epoch query. No HIP context is created or used. */
#include "mailbox_completion.h"
#include "consumption_ring.h"
#include "admission_ledger.h"
#include "admission_completion.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using r3::admission::Ledger;
using r3::admission::RankCompletion;
using namespace r3::mailbox_completion;

static void check(bool good, const char* name) {
    if (!good) { std::fprintf(stderr, "r3_mailbox_completion_test: FAIL %s\n", name); std::exit(1); }
}

static std::array<uint32_t, kPublishedTicketOffset+4> published_ticket(const StepTicket& t) {
    std::array<uint32_t, kPublishedTicketOffset+4> out{};
    out[kPublishedTicketOffset] = uint32_t(t.ticket_id);
    out[kPublishedTicketOffset+1] = uint32_t(t.ticket_id >> 32);
    out[kPublishedTicketOffset+2] = t.placement_epoch;
    out[kPublishedTicketOffset+3] = t.generation_slot;
    return out;
}

static void V23RouteMark_StaleCompletionIdentityRejected() {
    const StepTicket expected{7, 1, {0,0,0}, (uint64_t(1)<<32)+91};
    auto data = published_ticket(expected);
    StepTicket pass{};
    check(pass_ticket(data.data(), &pass) && pass.ticket_id == expected.ticket_id &&
          pass.placement_epoch == expected.placement_epoch &&
          pass.generation_slot == expected.generation_slot, "full parent ticket decoded from immutable pass copy");
    r3::moe_shm::Request q{};
    const uint32_t seq = 19;
    q.sequence = seq; q.status = 0;
    q.placement_epoch = pass.placement_epoch;
    q.generation_slot = pass.generation_slot;
    check(matches_request(pass, seq, q), "route mark accepts matching Request identity");
    q.placement_epoch++;
    check(!matches_request(pass, seq, q),
          "stale completion matching sequence but wrong epoch rejected");
    q.placement_epoch = pass.placement_epoch;
    q.generation_slot ^= 1u;
    check(!matches_request(pass, seq, q),
          "stale completion matching sequence but wrong slot rejected");
    q.generation_slot = pass.generation_slot;
    check(!matches_request(pass, seq+1, q), "wrong sequence rejected by route-mark");
    q.sequence++;
    check(!matches_request(pass, seq, q), "wrong Request sequence rejected");
    q.sequence = seq;
    /* Wait checks status after the existing done_sequence acquire. The
     * route-mark identity guard must not add a mapped status read to
     * combined chunks before the helper has completed. */
    data[kPublishedTicketOffset+3] = 257u; /* must reject before narrowing to uint8_t */
    check(!pass_ticket(data.data(), &pass), "invalid wide slot rejected");
    data[kPublishedTicketOffset+3] = expected.generation_slot;
    data[kPublishedTicketOffset+2] = 0u;
    check(!pass_ticket(data.data(), &pass), "zero epoch rejected");
    data[kPublishedTicketOffset+2] = expected.placement_epoch;
    data[kPublishedTicketOffset] = 0u;
    data[kPublishedTicketOffset+1] = 0u;
    check(!pass_ticket(data.data(), &pass), "zero ticket id rejected");
    std::puts("V23RouteMark_StaleCompletionIdentityRejected: PASS");
}

/* Fake Radiance terminal events: a recorded event starts incomplete; the fake
 * stream turns it complete only after the last GPU result-plane consumer. */
struct FakeEvent {
    using Event = uint8_t*;
    static uint8_t states[64];
    static unsigned created;
    static Event last_recorded;
    static int create(Event* e) {
        if (created >= 64) return -1;
        *e = &states[created++]; **e = 0;
        return 0;
    }
    static void destroy(Event) {}
    static int record(Event e, void*) { *e = 0; last_recorded = e; return 0; }
    static int query(Event e) { return *e ? 1 : 0; }
};
uint8_t FakeEvent::states[64]{};
unsigned FakeEvent::created = 0;
FakeEvent::Event FakeEvent::last_recorded = nullptr;
using Completion = RankCompletion<FakeEvent>;

/* Production seal_ready and RankCompletion::prove enforce the two barriers.
 * The model exposes add, combined tail, finish, seal and terminal event in
 * their actual stream order; an attempted seal before finish cannot ACK. */
struct FakeCombinedStream {
    StepTicket ticket{};
    ConsumptionStamp device[3]{};
    ConsumptionStamp host[3]{};
    bool add_read = false;
    bool tail_wait = false;
    bool finish = false;
    bool seal = false;
    void begin(const StepTicket& t) {
        ticket = t;
        r3::consumption::arm(device, 3, t);
        ++device[t.ticket_id % 3].published_count;
    }
    void read_combined_plane() { add_read = true; }
    void wait_for_helper_tail() { tail_wait = true; }
    bool finish_add() {
        if (!add_read || !tail_wait) return false;
        ++device[ticket.ticket_id % 3].consumed_count;
        finish = true;
        return true;
    }
    bool seal_parent() {
        if (!r3::consumption::seal_ready(device[ticket.ticket_id % 3], ticket))
            return false;
        host[ticket.ticket_id % 3] = device[ticket.ticket_id % 3];
        seal = true;
        return true;
    }
};
static void V23Consumption_SealFollowsFinalCombinedPlaneReader() {
    FakeCombinedStream gpu;
    Ledger ledger(3);
    Completion rank;
    StepTicket t{};
    check(ledger.initialize(5,1) && rank.prepare(3,gpu.host) &&
          ledger.try_admit(1,&t)==Ledger::Result::Ok &&
          ledger.bind(t.ticket_id,42) && rank.arm(t), "fake rank setup");
    gpu.begin(t);
    check(!gpu.seal_parent(), "no consumed ACK before combined-plane reader");
    gpu.read_combined_plane();
    check(!gpu.finish_add() && !gpu.seal_parent(),
          "no consumed ACK while tail wait is still outstanding");
    gpu.wait_for_helper_tail();
    check(gpu.finish_add() && gpu.seal_parent(), "seal only after true last reader and tail wait");
    void* stream = &gpu;
    check(rank.record(t.ticket_id,stream), "record terminal event behind seal");
    check(ledger.host_complete(t.ticket_id), "host logical completion");
    check(ledger.poll([&](const StepTicket& v,uint32_t) { return rank.prove(v); }) &&
          ledger.outstanding()==1, "sealed stamp alone cannot ACK ahead of terminal event");
    *FakeEvent::last_recorded = 1;
    check(ledger.poll([&](const StepTicket& v,uint32_t) { return rank.prove(v); }) &&
          ledger.outstanding()==0, "event completion makes per-parent seal consumable");
    std::puts("V23Consumption_SealFollowsFinalCombinedPlaneReader: PASS");
}
static void V23Consumption_AllRanksWaitForLastTicketSeal() {
    Ledger ledger(3);
    check(ledger.initialize(11,0), "two-rank ledger epoch");
    ConsumptionStamp host[2][3]{};
    Completion ranks[2];
    check(ranks[0].prepare(3,host[0]) && ranks[1].prepare(3,host[1]), "two rank events");
    StepTicket t[2]{};
    for (uint32_t i=0;i<2;++i)
        check(ledger.try_admit(2,&t[i])==Ledger::Result::Ok &&
              ledger.bind(t[i].ticket_id,i), "two admitted parents");
    check(!ledger.all_ranks_consumed(11,t[1].ticket_id),
          "open admission cannot certify all tickets");
    const auto snap = ledger.close_admission();
    check(snap.old_epoch==11 && snap.latest_ticket==t[1].ticket_id &&
          !ledger.all_ranks_consumed(snap.old_epoch,snap.latest_ticket),
          "closed admission still has live tickets");
    FakeEvent::Event event[2][2]{};
    for (uint32_t i=0;i<2;++i) {
        for (uint32_t r=0;r<2;++r) {
            check(ranks[r].arm(t[i]), "arm exact parent per rank");
            /* Last rank's last ticket is still waiting for its seal.
             * Event recording is enqueued after seal, but its completion is
             * impossible until that preceding seal has executed. */
            if (!(i == 1 && r == 1))
                host[r][t[i].ticket_id % 3] =
                    ConsumptionStamp{t[i].ticket_id,t[i].placement_epoch,t[i].generation_slot,2,2};
            check(ranks[r].record(t[i].ticket_id, &ledger), "rank terminal event record");
            event[i][r] = FakeEvent::last_recorded;
        }
        check(ledger.host_complete(t[i].ticket_id), "host done for parent");
    }
    auto prove=[&](const StepTicket& v,uint32_t rank) { return ranks[rank].prove(v); };
    check(ledger.poll(prove) && !ledger.all_ranks_consumed(11,snap.latest_ticket),
          "host done and sealed stamps alone are not consumed ACKs");
    *event[0][0]=1; *event[0][1]=1; *event[1][0]=1;
    check(ledger.poll(prove) && ledger.outstanding()==1 &&
          host[1][t[1].ticket_id % 3].ticket_id == 0 &&
          !ledger.all_ranks_consumed(11,snap.latest_ticket),
          "all-ranks query false until last rank's last ticket seal");
    /* The fake GPU now finishes the final rank's seal, but its terminal
     * event has not yet reached completion. Neither alone suffices. */
    host[1][t[1].ticket_id % 3] =
        ConsumptionStamp{t[1].ticket_id,t[1].placement_epoch,t[1].generation_slot,2,2};
    check(ledger.poll(prove) && !ledger.all_ranks_consumed(11,snap.latest_ticket),
          "last seal without terminal event is still pending");
    *event[1][1]=1;
    check(ledger.poll(prove) && ledger.outstanding()==0 &&
          ledger.all_ranks_consumed(11,snap.latest_ticket),
          "every rank terminal event and matching seal completes epoch query");
    check(!ledger.all_ranks_consumed(12,snap.latest_ticket) &&
          !ledger.all_ranks_consumed(11,snap.latest_ticket-1),
          "wrong epoch and cutoff not accepted");
    std::puts("V23Consumption_AllRanksWaitForLastTicketSeal: PASS");
}
int main() {
    V23RouteMark_StaleCompletionIdentityRejected();
    V23Consumption_SealFollowsFinalCombinedPlaneReader();
    V23Consumption_AllRanksWaitForLastTicketSeal();
    return 0;
}
