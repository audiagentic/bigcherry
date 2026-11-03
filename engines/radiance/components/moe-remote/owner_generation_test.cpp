#include "owner_generation.h"
#include "worker_epoch.h"
#include "worker_bootstrap.h"
#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>
using namespace r3::moe_shm;
using r3::admission::Ledger;
using r3::owner_generation::build_bitmap;
using r3::owner_generation::may_upload;

static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "r3_owner_generation_test: FAIL %s\n", what); std::exit(1); }
}

/* Two different full canonical maps with a real v23 layout, no GPU device.
 * Each upload uses the production converter and ticket-selector function. */
struct Fixture {
    const LayoutShape shape{128, 2, 3, 8, 128, 64};
    Layout l{};
    void* base = nullptr;
    uint64_t digest[2]{};
    Fixture() {
        check(layout_for(shape, 2, slots_for(2), &l), "shape layout");
        base = mmap(nullptr, l.bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        check(base != MAP_FAILED && init(base, 2, shape, 2), "initialize shared segment");
        static_cast<Header*>(base)->helper_count = 2; /* helper bootstrap publication */
        uint8_t* b = generation_map(base, 1);
        struct Selected { std::vector<std::pair<uint32_t,uint32_t>> expert; };
        std::vector<uint8_t> premature;
        check(!build_bitmap(base,0,1,placement_control(base)->active_digest,&premature),
              "no initial GPU upload before seal");
        check(r3::epoch::publish_owned(base,Selected{{{0,0},{1,0}}},0) &&
              r3::epoch::publish_owned(base,Selected{{{0,1}}},1),
              "real owner publish for bitmap test");
        for (uint32_t helper=0;helper<2;++helper)
            check(r3::epoch::mark_bootstrap_filtered(base,helper),"bitmap helpers filtered");
        r3::epoch::SharedTransport transport(base);
        check(transport.seal_bootstrap() && bootstrap_generation_sealed(base),
              "real bootstrap seal before bitmap conversion");
        digest[0] = placement_control(base)->active_digest;
        b[0] = 1; b[1] = 0; b[3] = 1;
        auto* c = placement_control(base);
        auto* pending = generation_header(base,1);
        pending->epoch = 2;
        pending->state = uint32_t(GenerationState::STAGING);
        digest[1] = generation_digest(base,1);
        pending->digest = digest[1];
        c->pending_slot = 1; c->pending_epoch = 2; c->pending_digest = digest[1];
        c->state = uint32_t(PlacementState::COPYING);
    }
    ~Fixture() { munmap(base, l.bytes); }
};

static bool has(const uint8_t* bits, uint32_t helper, uint32_t layer, uint32_t expert,
                uint32_t layers, uint32_t experts) {
    const uint32_t stride = (layers * experts + 7u) / 8u;
    return r3_moe_owner_has(bits+(size_t)helper*stride,layer,expert,experts);
}

static void V23EngineOwners_NoStepMixesGenerations() {
    Fixture f;
    std::vector<uint8_t> a, b;
    check(build_bitmap(f.base, 0, 1, f.digest[0], &a) &&
          build_bitmap(f.base, 1, 2, f.digest[1], &b), "each generation derives independently");
    check(a.size() == 2 && b.size() == 2, "helper bitmap size from real layers/expert count");
    const R3MoeOwnerBinding binds[2]{{1,f.digest[0]},{2,f.digest[1]}};
    StepTicket t0{}; t0.ticket_id = 101; t0.placement_epoch = 1; t0.generation_slot = 0;
    StepTicket t1{}; t1.ticket_id = 102; t1.placement_epoch = 2; t1.generation_slot = 1;
    /* All layers and all helpers of one parent select a single immutable table.
     * Changing the other slot is invisible to the original ticket. */
    for (int pass = 0; pass < 4; ++pass) {
        const uint8_t* x = r3_moe_select_owner_bitmap(t0, binds, a.data(), b.data());
        const uint8_t* y = r3_moe_select_owner_bitmap(t1, binds, a.data(), b.data());
        check(x == a.data() && y == b.data() &&
              has(x,0,0,0,2,3) && !has(x,1,0,0,2,3) &&
              !has(y,0,0,0,2,3) && has(y,1,0,0,2,3) &&
              has(x,0,1,0,2,3) && has(y,1,1,0,2,3), "ticket locks every pass to its own bitmap");
    }
    t1.placement_epoch = 1;
    check(!r3_moe_select_owner_bitmap(t1, binds, a.data(), b.data()), "stale epoch denied");
    t1.placement_epoch = 2; t1.generation_slot = 2;
    check(!r3_moe_select_owner_bitmap(t1, binds, a.data(), b.data()), "invalid slot denied");
    t1.generation_slot = 1; t1.reserved[0] = 1;
    check(!r3_moe_select_owner_bitmap(t1, binds, a.data(), b.data()), "nonzero ticket reserve denied");
    t1.reserved[0] = 0;
    check(!r3_moe_select_owner_bitmap(t1, nullptr, a.data(), b.data()), "missing generation bindings denied");
    check(!r3_moe_select_owner_bitmap(t1, binds, a.data(), nullptr), "missing selected bitmap denied");
    check(!build_bitmap(f.base, 1, 2, f.digest[0], &b), "wrong digest refused");
    generation_map(f.base,1)[0] = kOwnerLocal;
    check(!build_bitmap(f.base,1,2,f.digest[1],&b), "changed map rejected by digest");
    generation_map(f.base,1)[0] = 7;
    generation_header(f.base,1)->digest = generation_digest(f.base,1);
    placement_control(f.base)->pending_digest = generation_header(f.base,1)->digest;
    check(!build_bitmap(f.base,1,2,generation_header(f.base,1)->digest,&b),
          "out of range helper rejected even with matching digest");
    std::puts("V23EngineOwners_NoStepMixesGenerations: PASS");
}

static void V23EngineOwners_NonByteAlignedHelperStrides() {
    const LayoutShape shape{128,3,5,8,128,64};
    Layout layout{};
    check(layout_for(shape,3,slots_for(3),&layout),"odd bitmap layout");
    void* map=mmap(nullptr,layout.bytes,PROT_READ|PROT_WRITE,
                   MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    check(map!=MAP_FAILED && init(map,3,shape,3),"odd bitmap segment");
    auto* h=static_cast<Header*>(map);
    h->helper_count=3;
    struct Selected { std::vector<std::pair<uint32_t,uint32_t>> expert; };
    check(r3::epoch::publish_owned(map,Selected{{{0,0}}},0) &&
          r3::epoch::publish_owned(map,Selected{{{1,2},{1,3}}},1) &&
          r3::epoch::publish_owned(map,Selected{{{2,4}}},2),
          "nonbyte real bootstrap publishes owners");
    for (uint32_t helper=0;helper<3;++helper)
        check(r3::epoch::mark_bootstrap_filtered(map,helper),"nonbyte helpers filtered");
    r3::epoch::SharedTransport transport(map);
    check(transport.seal_bootstrap(),"nonbyte bootstrap seal");
    const uint64_t digest=generation_digest(map,0);
    std::vector<uint8_t> bits;
    check(build_bitmap(map,0,1,digest,&bits) && bits.size()==6,
          "three helpers times ceil(15/8) stride");
    const uint32_t stride=(3*5+7)/8;
    for (uint32_t owner=0;owner<3;++owner)
        for (uint32_t layer=0;layer<3;++layer)
            for (uint32_t expert=0;expert<5;++expert) {
                const uint32_t index=layer*5+expert;
                const bool expected=(index==0 && owner==0) ||
                    ((index==7 || index==8) && owner==1) ||
                    (index==14 && owner==2);
                check(r3_moe_owner_has(bits.data()+owner*stride,layer,expert,5)==expected,
                      "bit and helper stride exact across byte boundary");
            }
    munmap(map,layout.bytes);
}
struct FakeOwnerStream {
    std::vector<uint8_t> owner[2];
    R3MoeOwnerBinding binding[2]{};
    bool event[2]{};
    bool queue(const Fixture& f, const Ledger& ledger, uint32_t slot, uint64_t epoch,
               uint64_t digest) {
        if (!may_upload(ledger,slot)) return false;
        std::vector<uint8_t> bits;
        if (!build_bitmap(f.base,slot,epoch,digest,&bits)) return false;
        if (!event[slot] && binding[slot].digest) return false; /* pinned staging in flight */
        owner[slot] = std::move(bits);
        binding[slot] = {epoch,digest};
        event[slot] = false;
        return true;
    }
    void complete(uint32_t slot) { event[slot] = true; }
    bool ready(uint32_t slot, uint64_t epoch, uint64_t digest) const {
        return slot < 2 && event[slot] &&
               binding[slot].epoch == epoch && binding[slot].digest == digest;
    }
};
static void V23EngineOwners_UnretiredTicketRefusesUpload() {
    Fixture f;
    Ledger ledger(3);
    check(ledger.initialize(1,0), "ledger initialization");
    FakeOwnerStream uploader;
    check(uploader.queue(f,ledger,1,2,f.digest[1]), "inactive upload while serving slot0");
    check(!uploader.ready(1,2,f.digest[1]), "upload event incomplete");
    uploader.complete(1);
    check(uploader.ready(1,2,f.digest[1]), "event completes private stream upload");
    StepTicket t{};
    check(ledger.try_admit(1,&t) == Ledger::Result::Ok &&
          ledger.bind(t.ticket_id,10), "admit old slot");
    check(ledger.host_complete(t.ticket_id), "logical host completion only");
    check(ledger.generation_referenced(0), "host-done ticket still pins old slot");
    check(!may_upload(ledger,0), "old slot cannot be uploaded while ticket unretired");
    check(ledger.poll([](const StepTicket&,uint32_t) {return Ledger::Proof::Pending;}),
          "incomplete GPU event does not retire ticket");
    check(ledger.generation_referenced(0), "rank proof pending still pins");
    check(!uploader.queue(f,ledger,0,1,f.digest[0]), "actual upload denied while unretired");
    std::puts("V23EngineOwners_UnretiredTicketRefusesUpload: PASS");
}
static void V23EngineOwners_RetirementUnlocksUpload() {
    Fixture f;
    Ledger ledger(3);
    check(ledger.initialize(1,0), "ledger init");
    StepTicket t{};
    check(ledger.try_admit(1,&t) == Ledger::Result::Ok &&
          ledger.bind(t.ticket_id,11) && ledger.host_complete(t.ticket_id), "admit and host complete");
    check(ledger.generation_referenced(0), "old generation held before proof");
    check(ledger.poll([](const StepTicket&,uint32_t) {return Ledger::Proof::Proven;}) &&
          !ledger.generation_referenced(0), "consumption/event proof retires ticket");
    const auto snap = ledger.close_admission();
    check(ledger.drained(snap.old_epoch,snap.latest_ticket) &&
          ledger.reopen(2,1), "only a drained ledger changes active slot");
    check(may_upload(ledger,0) && !may_upload(ledger,1), "retired inactive slot now replaceable");
    /* Recycle the old slot for the next staged generation with its own digest
     * and header; the fake event mirrors the real private-stream event. */
    auto* c = placement_control(f.base);
    c->active_slot = 1; c->active_epoch = 2; c->active_digest = f.digest[1];
    c->pending_slot = 0; c->pending_epoch = 3; c->pending_digest = f.digest[0];
    generation_header(f.base,0)->epoch = 3;
    generation_header(f.base,0)->digest = generation_digest(f.base,0);
    generation_header(f.base,0)->state = uint32_t(GenerationState::STAGING);
    FakeOwnerStream uploader;
    check(uploader.queue(f,ledger,0,3,f.digest[0]), "upload succeeds after retirement");
    check(!uploader.ready(0,3,f.digest[0]), "not ready before event");
    uploader.complete(0);
    check(uploader.ready(0,3,f.digest[0]) &&
          !uploader.ready(0,2,f.digest[0]), "ready requires exact generation and completed event");
    std::puts("V23EngineOwners_RetirementUnlocksUpload: PASS");
}
int main() {
    V23EngineOwners_NoStepMixesGenerations();
    V23EngineOwners_NonByteAlignedHelperStrides();
    V23EngineOwners_UnretiredTicketRefusesUpload();
    V23EngineOwners_RetirementUnlocksUpload();
    return 0;
}
