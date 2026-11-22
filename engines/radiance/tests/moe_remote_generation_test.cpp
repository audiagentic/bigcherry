// Card-free exercise of the engine-owned remote-MoE generation (patch rad0002): explicit open/close/destroy, the
// slot-keyed init-time lookup, stale/foreign refusals, per-generation state, and concurrent lifetimes.
#include "bc_moe_remote.h"
#include <atomic>
#include <cassert>
#include <cstdint>
#include <thread>
#include <vector>

static int64_t policy_rows = 4;
static uint32_t policy_skip = 0, policy_shadow = 0, policy_mask = 0;
extern "C" int bc_serving_remote_settings(int64_t* rows, uint32_t* skip, uint32_t* shadow, uint32_t* mask) {
    *rows = policy_rows; *skip = policy_skip; *shadow = policy_shadow; *mask = policy_mask;
    return 0;
}

struct Fake {
    int id = 0;
    std::atomic<int> admits{0}, bursts_on{0}, bursts_off{0}, closes{0};
    uint64_t next_ticket = 0;
    std::vector<int> maps;
};
static std::atomic<int> opened{0}, closed{0}, ids{0}, self_cleaned{0};
static bool factory_fails = false;
static std::atomic<int> factory_calls{0};

static int f_admit(void* self, BcMoeRemoteTicket* out, int, int) {
    auto* f = static_cast<Fake*>(self);
    ++f->admits;
    *out = BcMoeRemoteTicket{7, 0, {}, ++f->next_ticket};
    return 0;
}
static int f_ok_ticket(void*, BcMoeRemoteTicket) { return 0; }
static int f_bind(void*, BcMoeRemoteTicket, int64_t) { return 0; }
static int f_rank_done(void*, int, BcMoeRemoteTicket, void*) { return 0; }
static int f_before(void*, int, BcMoeRemoteTicket, void*) { return 0; }
static int f_wait(void*, int64_t) { return 0; }
static void f_layer_map(void* self, int layer, int n, const int32_t*) {
    static_cast<Fake*>(self)->maps.push_back(layer * 1000 + n);
}
static int f_owns(void*, int, int expert) { return expert % 2 == 0; }
static void f_burst(void* self, int on) {
    auto* f = static_cast<Fake*>(self);
    (on ? f->bursts_on : f->bursts_off)++;
}
static void f_noop(void*) {}
static void f_close(void* self) {
    auto* f = static_cast<Fake*>(self);
    ++closed;
    delete f;
}
static int factory(const BcMoeRemoteOpenArgs* a, BcMoeRemotePlugin* out) {
    ++factory_calls;
    if (!a || a->struct_size != sizeof(BcMoeRemoteOpenArgs) || a->capacity_limit == 0 ||
        out->struct_size != sizeof(BcMoeRemotePlugin))   // the engine presets its table size: refuse before allocating
        return BC_MOE_REMOTE_REFUSED;
    if (factory_fails) return BC_MOE_REMOTE_REFUSED;
    auto* f = new Fake;
    f->id = ++ids;
    ++opened;
    *out = BcMoeRemotePlugin{sizeof(BcMoeRemotePlugin), f, f_close, f_layer_map, f_owns, f_burst, f_before, f_admit,
                             f_bind, f_ok_ticket, f_ok_ticket, f_rank_done, f_noop, f_noop, f_wait};
    return 0;
}
/* Failure-atomic factory protocol: a factory that fails has cleaned up after itself and the engine must not call its
 * close; a factory that succeeds transfers the state, and if the table it returned is the wrong size the engine
 * closes it exactly once through the head every table starts with (struct_size, self, close). */
static int factory_alloc_then_fail(const BcMoeRemoteOpenArgs*, BcMoeRemotePlugin* out) {
    ++factory_calls;
    auto* f = new Fake;
    ++opened;
    delete f;               // the factory unwinds its own allocation before returning the failure
    ++self_cleaned;
    (void)out;
    return BC_MOE_REMOTE_REFUSED;
}
static int factory_wrong_size_alloc(const BcMoeRemoteOpenArgs*, BcMoeRemotePlugin* out) {
    ++factory_calls;
    auto* f = new Fake;     // really allocates, then hands back a table of the wrong size
    ++opened;
    *out = BcMoeRemotePlugin{};
    out->struct_size = sizeof(BcMoeRemotePlugin) - 8;
    out->self = f;
    out->close = f_close;
    return 0;
}
static int factory_tiny_size(const BcMoeRemoteOpenArgs*, BcMoeRemotePlugin* out) {
    ++factory_calls;        // returns success without even the head: nothing the engine can close; refused, nothing bound
    out->struct_size = 4;
    return 0;
}
static BcMoeRemoteGeneration* open_at(int base, int ranks) {
    BcMoeRemoteGeneration* g = nullptr;
    assert(bc_moe_remote_open_with(factory, base, ranks, 5, &g) == BC_MOE_REMOTE_OK && g);
    return g;
}

int main() {
    BcMoeRemoteTicket t{};

    // Nothing is bound at the start; unclaimed, negative and out-of-range slots find nothing.
    for (int s : {-1, 0, 5, 31, 32, 99}) assert(!bc_moe_remote_lookup(s));

    // A generation binds exactly its engine's slot span, and every consumer finds it there.
    BcMoeRemoteGeneration* a = open_at(4, 2);
    assert(bc_moe_remote_lookup(4) == a && bc_moe_remote_lookup(5) == a);
    assert(!bc_moe_remote_lookup(3) && !bc_moe_remote_lookup(6));
    assert(bc_moe_remote_wanted(a, 4) && !bc_moe_remote_wanted(a, 5) && bc_moe_remote_wanted_upto(a, 100));
    assert(bc_moe_remote_admit(a, &t, 0, 2) == 0 && t.ticket_id == 1);

    // A second engine cannot take a bound slot; the factory does not run (no side effect before the refusal).
    factory_calls = 0;
    BcMoeRemoteGeneration* none = nullptr;
    assert(bc_moe_remote_open_with(factory, 5, 2, 5, &none) == BC_MOE_REMOTE_BUSY && !none && factory_calls == 0);
    assert(bc_moe_remote_open_with(factory, 3, 2, 5, &none) == BC_MOE_REMOTE_BUSY && !none && factory_calls == 0);
    assert(bc_moe_remote_open_with(factory, 31, 2, 5, &none) == BC_MOE_REMOTE_REFUSED && factory_calls == 0);
    assert(bc_moe_remote_open_with(factory, -1, 2, 5, &none) == BC_MOE_REMOTE_REFUSED && factory_calls == 0);
    assert(bc_moe_remote_open_with(factory, 8, 0, 5, &none) == BC_MOE_REMOTE_REFUSED && factory_calls == 0);

    // Stale holder: close, a second engine re-claims the same slots, the old holder is refused with a typed result and
    // never reaches the new plugin; the new generation is untouched by the old one's repeated close.
    bc_moe_remote_close(a);
    assert(!bc_moe_remote_lookup(4) && !bc_moe_remote_lookup(5));
    BcMoeRemoteGeneration* b = open_at(4, 2);
    assert(b != a && bc_moe_remote_lookup(4) == b && bc_moe_remote_lookup(5) == b);
    assert(bc_moe_remote_admit(a, &t, 0, 2) == BC_MOE_REMOTE_STALE && t.ticket_id == 0);
    t.ticket_id = 1;
    assert(bc_moe_remote_bind(a, t, 1) == BC_MOE_REMOTE_STALE);
    assert(bc_moe_remote_cancel(a, t) == BC_MOE_REMOTE_STALE);
    assert(bc_moe_remote_host_done(a, t) == BC_MOE_REMOTE_STALE);
    assert(bc_moe_remote_rank_done(a, 0, t, (void*)1) == BC_MOE_REMOTE_STALE);
    assert(bc_moe_remote_before_pass(a, 0, t, (void*)1) == BC_MOE_REMOTE_STALE);
    assert(bc_moe_remote_wait_finished(a, 1) == BC_MOE_REMOTE_STALE);
    bc_moe_remote_close(a);   // idempotent: does not unbind b, does not run the old plugin close twice
    assert(bc_moe_remote_lookup(4) == b && closed == 1);
    assert(bc_moe_remote_admit(b, &t, 0, 2) == 0 && t.ticket_id == 1);   // b's own counter, not a's
    bc_moe_remote_destroy(a);
    assert(bc_moe_remote_lookup(4) == b);

    // No generation (policy off or unclaimed slot): every query answers "no remote", nothing faults.
    assert(!bc_moe_remote_wanted(nullptr, 1) && !bc_moe_remote_wanted_upto(nullptr, 1));
    assert(!bc_moe_remote_shadow(nullptr) && !bc_moe_remote_mask_only(nullptr) && !bc_moe_remote_skip_unit(nullptr, 0, 0, 1));
    assert(bc_moe_remote_admit(nullptr, &t, 0, 2) == 0 && t.ticket_id == 0);
    assert(bc_moe_remote_before_pass(nullptr, 0, BcMoeRemoteTicket{}, nullptr) == 0);
    assert(bc_moe_remote_finish_admission(nullptr) == 0);
    bc_moe_remote_engine_step(nullptr);
    bc_moe_remote_engine_idle(nullptr);
    bc_moe_remote_close(nullptr);
    bc_moe_remote_destroy(nullptr);

    // Per-generation state: layer maps and bursts do not leak across engines, and a later generation replays nothing.
    const int32_t map[4] = {0, 1, 2, 3};
    bc_moe_remote_layer_map(b, 3, 4, map);
    BcMoeRemoteGeneration* c = open_at(10, 1);
    assert(bc_moe_remote_skip_unit(c, 3, 0, 1) == 0);   // c never saw b's layer 3
    policy_skip = 1;
    BcMoeRemoteGeneration* d = open_at(12, 1);          // skip_owned captured at open
    policy_skip = 0;
    bc_moe_remote_layer_map(d, 3, 4, map);
    assert(bc_moe_remote_skip_unit(d, 3, 2, 1) == 1 && bc_moe_remote_skip_unit(d, 3, 1, 1) == 0);
    assert(bc_moe_remote_skip_unit(d, 3, 2, 5) == 0);     // the hand-off cap must cover the largest step
    assert(bc_moe_remote_skip_unit(b, 3, 2, 1) == 0);     // b opened with skip_owned off
    const int32_t protected_map[4] = {0, -1, 2, 3};
    bc_moe_remote_layer_map(d, 4, 4, protected_map);
    assert(bc_moe_remote_skip_unit(d, 4, 2, 1) == 0);     // a protected expert keeps the layer local
    bc_moe_remote_engine_step(b);
    bc_moe_remote_engine_step(b);
    bc_moe_remote_engine_idle(b);
    bc_moe_remote_engine_idle(b);
    bc_moe_remote_engine_step(c);
    bc_moe_remote_destroy(d);
    assert(!bc_moe_remote_lookup(12));

    // The manifest asking for nothing opens nothing and never calls the factory; one that needs a hand-off has to have a
    // factory; a refusing or mis-sized factory leaves no binding behind.
    factory_calls = 0;
    policy_rows = 0;
    assert(bc_moe_remote_open_with(factory, 14, 1, 5, &none) == BC_MOE_REMOTE_OK && !none && factory_calls == 0);
    policy_rows = 4;
    assert(bc_moe_remote_open_with(nullptr, 14, 1, 5, &none) == BC_MOE_REMOTE_ABSENT && !none);
    assert(bc_moe_remote_open(nullptr, 14, 1, 5, &none) == BC_MOE_REMOTE_ABSENT && !none);
    factory_fails = true;
    assert(bc_moe_remote_open_with(factory, 14, 1, 5, &none) == BC_MOE_REMOTE_REFUSED && !none && !bc_moe_remote_lookup(14));
    factory_fails = false;
    {
        // Every failure path of the negotiation releases whatever the factory allocated, exactly once, and leaves
        // nothing bound: a failing factory cleans itself (the engine never calls its close); a wrong-size success is
        // closed by the engine through the table head.
        const int open0 = opened, closed0 = closed, cleaned0 = self_cleaned;
        assert(bc_moe_remote_open_with(factory_alloc_then_fail, 14, 1, 5, &none) == BC_MOE_REMOTE_REFUSED && !none &&
               !bc_moe_remote_lookup(14));
        assert(opened == open0 + 1 && self_cleaned == cleaned0 + 1 && closed == closed0);
        assert(bc_moe_remote_open_with(factory_wrong_size_alloc, 14, 1, 5, &none) == BC_MOE_REMOTE_REFUSED && !none &&
               !bc_moe_remote_lookup(14));
        assert(opened == open0 + 2 && closed == closed0 + 1);                // closed once, by the engine
        assert(bc_moe_remote_open_with(factory_tiny_size, 14, 1, 5, &none) == BC_MOE_REMOTE_REFUSED && !none &&
               !bc_moe_remote_lookup(14) && closed == closed0 + 1);
        // and the span is open for a good factory right after: no stuck BUSY
        BcMoeRemoteGeneration* again = open_at(14, 1);
        assert(bc_moe_remote_lookup(14) == again);
        bc_moe_remote_destroy(again);
        assert(!bc_moe_remote_lookup(14));
    }
    policy_shadow = 1;   // a production build cannot honour a diagnostic mode: off, said once, no generation
    factory_calls = 0;
    assert(bc_moe_remote_open_with(factory, 14, 1, 5, &none) == BC_MOE_REMOTE_OK && !none && factory_calls == 0);
    policy_shadow = 0;

    // Repeated engine lifetimes on one slot span: each open binds, each destroy unbinds and closes the plugin once.
    const int closed_before = closed;
    for (int i = 0; i < 50; ++i) {
        BcMoeRemoteGeneration* g = open_at(20, 3);
        assert(bc_moe_remote_lookup(20) == g && bc_moe_remote_lookup(22) == g);
        bc_moe_remote_destroy(g);
        assert(!bc_moe_remote_lookup(20) && !bc_moe_remote_lookup(22));
    }
    assert(closed == closed_before + 50);

    // Concurrent engines on disjoint spans, with readers looking slots up while they open and close.
    std::atomic<bool> stop{false};
    std::atomic<int> bad{0};
    std::vector<std::thread> threads;
    for (int w = 0; w < 4; ++w)
        threads.emplace_back([w, &bad] {
            for (int i = 0; i < 200; ++i) {
                BcMoeRemoteGeneration* g = nullptr;
                if (bc_moe_remote_open_with(factory, 24 + w * 2, 2, 5, &g) != BC_MOE_REMOTE_OK || !g) { ++bad; continue; }
                if (bc_moe_remote_lookup(24 + w * 2) != g || bc_moe_remote_lookup(25 + w * 2) != g) ++bad;
                BcMoeRemoteTicket tk{};
                if (bc_moe_remote_admit(g, &tk, 0, 2) != 0 || tk.ticket_id != 1) ++bad;
                bc_moe_remote_destroy(g);
            }
        });
    std::thread reader([&] { while (!stop) for (int s = 0; s < 32; ++s) (void)bc_moe_remote_lookup(s); });
    for (auto& th : threads) th.join();
    stop = true;
    reader.join();
    assert(bad == 0);
    // Contending openers for ONE span: exactly one wins at a time, the rest are refused BUSY and leave nothing behind.
    std::atomic<int> won{0}, busy{0};
    std::atomic<bool> go{false};
    std::vector<BcMoeRemoteGeneration*> winners(8, nullptr);
    std::vector<std::thread> racers;
    for (int i = 0; i < 8; ++i)
        racers.emplace_back([&, i] {
            while (!go) {}
            BcMoeRemoteGeneration* g = nullptr;
            const int rc = bc_moe_remote_open_with(factory, 16, 2, 5, &g);
            if (rc == BC_MOE_REMOTE_OK) { ++won; winners[i] = g; } else if (rc == BC_MOE_REMOTE_BUSY) ++busy;
        });
    go = true;
    for (auto& th : racers) th.join();
    assert(won == 1 && busy == 7);
    for (auto* g : winners) bc_moe_remote_destroy(g);
    assert(!bc_moe_remote_lookup(16) && !bc_moe_remote_lookup(17));

    bc_moe_remote_destroy(b);
    bc_moe_remote_destroy(c);
    for (int s = 0; s < 32; ++s) assert(!bc_moe_remote_lookup(s));
    assert(opened == closed + self_cleaned);   // every opened plugin state was closed exactly once (no leak, no double close)
    return 0;
}
