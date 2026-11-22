// RUM06 S3: the identity of the one live hook. A handle is a serial, not an address, so every wrong handle is refused
// before any side effect: from a closed hook, from one that never existed, from another engine, and from a hook that was
// closed and whose address a later hook then reused (bind -> close -> reopen in the same slot, old holder still holding
// its old value).
#include "hook_identity.h"
#include <atomic>
#include <cassert>
#include <new>
#include <thread>
#include <vector>

struct Hook { uint64_t serial = 0; int slot_base = -1; int touched = 0; };

int main() {
    LiveHook<Hook> live;
    alignas(Hook) static unsigned char storage[sizeof(Hook)];   // the same address for every hook below

    assert(!live.busy() && !live.resolve(0) && !live.resolve(1));
    assert(live.serial_if([](const Hook&) { return true; }) == 0);

    auto* a = new (storage) Hook;
    a->slot_base = 4;
    const uint64_t ida = live.publish(a);
    assert(ida != 0 && a->serial == ida && live.busy());
    assert(live.resolve(ida) == a);
    assert(live.serial_if([](const Hook& h) { return h.slot_base == 4; }) == ida);
    assert(live.serial_if([](const Hook& h) { return h.slot_base == 5; }) == 0);   // wrong span: not found
    assert(!live.resolve(0) && !live.resolve(ida + 1) && !live.resolve(0x1000));  // never issued / address-looking

    // A second engine while one is live: refused, and the live hook is untouched.
    Hook other;
    assert(live.publish(&other) == 0 && live.resolve(ida) == a && !live.unpublish(&other));

    // Close the first and reopen in the same place: the old holder keeps its old serial and is refused; the new
    // holder is served; the old serial never resolves again.
    assert(live.unpublish(a));
    assert(!live.resolve(ida) && !live.busy());
    a->~Hook();
    auto* b = new (storage) Hook;
    assert(static_cast<void*>(b) == static_cast<void*>(a));   // same address, new generation
    b->slot_base = 4;
    const uint64_t idb = live.publish(b);
    assert(idb != 0 && idb != ida && live.resolve(idb) == b);
    assert(!live.resolve(ida));
    b->~Hook();
    live.unpublish(b);
    assert(!live.busy() && !live.resolve(idb));

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
            if (live.resolve(h.serial) == &h) { ++live_count; assert(live.unpublish(&h)); }
        assert(live_count == 1 && !live.busy());
    }
    return 0;
}
