// RUM06 S3: the identity of the one live hook. A handle is a serial, not an address, so every wrong handle is refused
// before any side effect: from a closed hook, from one that never existed, from another engine, and from a hook that was
// closed and whose address a later hook then reused (bind -> close -> reopen in the same slot, old holder still holding
// its old value).
#include "hook_identity.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <new>
#include <thread>
#include <vector>

struct Hook { uint64_t serial = 0; int slot_base = -1; int ranks = 0; int touched = 0; };

int main() {
    LiveHook<Hook> live;
    alignas(Hook) static unsigned char storage[sizeof(Hook)];   // the same address for every hook below

    assert(!live.busy() && !live.borrow(0) && !live.borrow(1));
    assert(live.serial_if([](const Hook&) { return true; }) == 0);

    auto* a = new (storage) Hook;
    a->slot_base = 4;
    const uint64_t ida = live.publish(a);
    assert(ida != 0 && a->serial == ida && live.busy());
    assert(live.borrow(ida).get() == a);
    assert(live.serial_if([](const Hook& h) { return h.slot_base == 4; }) == ida);
    assert(live.serial_if([](const Hook& h) { return h.slot_base == 5; }) == 0);   // wrong span: not found
    assert(!live.borrow(0) && !live.borrow(ida + 1) && !live.borrow(0x1000));  // never issued / address-looking

    // A second engine while one is live: refused, and the live hook is untouched.
    Hook other;
    assert(live.publish(&other) == 0 && !live.unpublish(&other));
    assert(live.borrow(ida).get() == a);

    // Close the first and reopen in the same place: the old holder keeps its old serial and is refused; the new
    // holder is served; the old serial never resolves again.
    assert(live.unpublish(a));
    assert(!live.borrow(ida) && !live.busy());
    a->~Hook();
    auto* b = new (storage) Hook;
    assert(static_cast<void*>(b) == static_cast<void*>(a));   // same address, new generation
    b->slot_base = 4;
    const uint64_t idb = live.publish(b);
    assert(idb != 0 && idb != ida && live.borrow(idb).get() == b);
    assert(!live.borrow(ida));
    b->~Hook();
    live.unpublish(b);
    assert(!live.busy() && !live.borrow(idb));

    // Many racing openers: exactly one wins each round, serials are unique and nonzero.
    for (int round = 0; round < 50; ++round) {
        std::vector<Hook> hooks(8);
        std::atomic<int> winners{0};
        std::atomic<bool> go{false};
        std::vector<std::thread> ts;
        for (auto& h : hooks)
            ts.emplace_back([&] {
                while (!go.load()) {}
                if (live.publish(&h)) winners.fetch_add(1);
            });
        go = true;
        for (auto& t : ts) t.join();
        assert(winners == 1);
        int live_count = 0;
        for (auto& h : hooks)
        {
            const bool is_live = live.borrow(h.serial).get() == &h;   // the borrow ends before the close
            if (is_live) { ++live_count; const bool ok = live.unpublish(&h); assert(ok); }
        }
        assert(live_count == 1 && !live.busy());
    }

    // A resolved handle is a borrow: unpublish (the close) waits for every borrower, so the hook cannot be freed under
    // a reader that already resolved it. Here a reader is paused after resolving while the close runs.
    {
        LiveHook<Hook> l2;
        Hook hk; hk.slot_base = 9;
        const uint64_t id = l2.publish(&hk);
        std::atomic<bool> borrowed{false}, release{false}, closed{false};
        std::thread reader([&] {
            auto b = l2.borrow(id);
            assert(b && b.get() == &hk);
            borrowed = true;
            while (!release) std::this_thread::yield();
            hk.touched = 1;               // still valid: the closer has not freed it
            assert(!closed);
        });
        while (!borrowed) std::this_thread::yield();
        std::thread closer([&] { const bool ok = l2.unpublish(&hk); assert(ok); closed = true; });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        assert(!closed);                  // the close is waiting for the paused reader
        release = true;
        reader.join();
        closer.join();
        assert(closed && hk.touched == 1);
        assert(!l2.borrow(id) && !l2.busy());   // after the close the handle is refused
    }

    // Serial exhaustion fails closed: a loser of a BUSY race consumes no serial, the last serial is issued once, and
    // the next publish is refused (never zero, never reused) with nothing published.
    {
        LiveHook<Hook> l3(UINT64_MAX - 1);
        Hook a1, a2, a3, a4;
        assert(l3.publish(&a1) == UINT64_MAX - 1);
        assert(l3.publish(&a2) == 0 && l3.busy() && !l3.exhausted());   // BUSY loser: no serial consumed
        assert(l3.unpublish(&a1));
        assert(l3.publish(&a3) == UINT64_MAX && l3.unpublish(&a3));
        assert(l3.publish(&a4) == 0 && l3.exhausted() && !l3.busy() && !l3.borrow(0));
    }

    // The hook for a numeric group answers only for its exact span and width, and not after it closed.
    {
        LiveHook<Hook> l4;
        Hook hk; hk.slot_base = 4; hk.ranks = 2;
        const uint64_t id = l4.publish(&hk);
        auto match = [](int base, int world) { return [=](const Hook& h) { return h.slot_base == base && h.ranks == world; }; };
        assert(l4.serial_if(match(4, 2)) == id);
        assert(l4.serial_if(match(4, 3)) == 0 && l4.serial_if(match(4, 1)) == 0 && l4.serial_if(match(5, 2)) == 0);
        l4.unpublish(&hk);
        assert(l4.serial_if(match(4, 2)) == 0);
    }
    return 0;
}
