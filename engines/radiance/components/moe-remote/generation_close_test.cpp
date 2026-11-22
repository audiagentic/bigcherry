// RUM06 re-review P1-1: the production close orchestration (close_generation) and the direct drain wait (await_drain),
// driven with the real Ledger, LiveHook and shared seal wait. Two direct borrowers are paused (A in the bootstrap seal
// wait, B in the helper-acknowledgment wait after the tickets reached zero) while a worker thread runs and a clean close
// happens. The close must join the worker before it cancels anything, cancel both borrowers promptly without touching
// the ledger (clean_stop() stays true until the resources are freed), free nothing until both have left, and refuse
// later borrows. A real helper fault still faults the ledger.
#include "generation_close.h"
#include "hook_identity.h"
#include "owner_generation.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sys/mman.h>
#include <thread>

using namespace r3::moe_shm;
namespace wire = r3::moe_shm::v26;
using std::chrono::milliseconds;
using std::chrono::seconds;
using r3::admission::Ledger;

static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "r3_generation_close_test: FAILED %s\n", what); std::exit(1); }
}
constexpr uint64_t kManifest = 0x13572680ae691d31ull;
constexpr LayoutShape kShape{128, 2, 3, 8, 128, 64};

struct Gen {
    uint64_t serial = 0; int slot_base = 0, ranks = 1; bool opened = true;
    std::atomic<bool> closing{false};
    Ledger ledger{2};
    void* map = nullptr; size_t bytes = 0;
};

static void* make_map(size_t* bytes) {
    Layout layout{};
    check(wire::layout_for(kShape, 2, {1, 1}, &layout) == Refusal::None, "layout");
    *bytes = size_t(layout.bytes);
    void* m = mmap(nullptr, *bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    check(m != MAP_FAILED, "map");
    check(wire::init(m, *bytes, kShape, 2, {1, 1}, kManifest) == Refusal::None, "init");   // never sealed
    return m;
}

int main() {
    Gen g;
    g.map = make_map(&g.bytes);
    Header* h = static_cast<Header*>(g.map);
    check(g.ledger.initialize(1, 0), "ledger");
    const auto snap = g.ledger.close_admission();       // zero outstanding tickets: the epoch drains at once
    check(snap.outstanding == 0 && g.ledger.clean_stop(), "healthy ledger before the close");

    LiveHook<Gen> live;
    const uint64_t id = live.publish(&g);
    check(id != 0, "publish");

    // The hook's own worker: it uses the entry points (borrows) until stopped, like the placement thread.
    std::atomic<bool> worker_stop{false}, worker_joined{false};
    std::thread worker([&] { while (!worker_stop) { auto b = live.borrow(id); std::this_thread::sleep_for(milliseconds(1)); } });

    std::mutex mu;
    std::atomic<uint32_t> ready{0};
    std::atomic<int> a_in{0}, b_in{0}, a_left{0}, b_left{0};
    std::atomic<int> a_result{-1};
    std::atomic<DrainWait> b_result{DrainWait::Drained};
    std::thread borrower_a([&] {
        auto b = live.borrow(id);
        check(static_cast<bool>(b), "A borrow");
        timespec deadline{}; clock_gettime(CLOCK_MONOTONIC, &deadline); deadline.tv_sec += 60;
        a_in = 1;
        a_result = r3::owner_generation::with_bootstrap_rank_lock(
            g.map, ready, 0, mu, &deadline, []() -> int { return int(WaitResult::Ready); },
            [](WaitResult e) -> int { return int(e); }, &g.closing);
        (void)load(&h->helper_count);                    // the mapping must still be there
        a_left = 1;
    });
    std::thread borrower_b([&] {
        auto b = live.borrow(id);
        check(static_cast<bool>(b), "B borrow");
        b_in = 1;
        b_result = await_drain(g, 1, snap.latest_ticket, std::chrono::steady_clock::now() + seconds(60), [] {},
                               [] { return false; }, [] { return true; }, [] { return false; });   // helpers never ack
        (void)g.ledger.outstanding();
        b_left = 1;
    });
    while (!a_in || !b_in) std::this_thread::yield();
    std::this_thread::sleep_for(milliseconds(60));
    check(a_result.load() == -1 && !b_left, "both borrowers are blocked");

    bool joined_before_wake = false, clean_at_free = false, none_inside_at_free = false, ledger_healthy_at_free = false;
    const auto t0 = std::chrono::steady_clock::now();
    close_generation(
        live, g,
        [&](bool clean) { check(clean, "a healthy ledger is a clean stop"); worker_stop = true; worker.join(); worker_joined = true; },
        [&] { joined_before_wake = worker_joined.load(); notify(h); },
        [&] {
            none_inside_at_free = a_left.load() && b_left.load();
            clean_at_free = g.ledger.clean_stop();
            ledger_healthy_at_free = !g.ledger.stopped();
            munmap(g.map, g.bytes); g.map = nullptr;
        });
    borrower_a.join(); borrower_b.join();
    check(std::chrono::steady_clock::now() - t0 < milliseconds(10000), "cancelled promptly, not timed out");
    check(joined_before_wake, "the worker is joined before any waiter is cancelled");
    check(none_inside_at_free, "nothing is freed until both borrowers have left");
    check(a_result.load() == int(WaitResult::Error), "A: the seal wait is cancelled with an error");
    check(b_result.load() == DrainWait::Cancelled, "B: the drain wait is cancelled, not failed");
    check(clean_at_free && ledger_healthy_at_free, "the cancel left the ledger healthy (clean_stop still true)");
    check(!g.ledger.stopped(), "the ledger was neither faulted nor interrupted by the cancel");
    check(!live.borrow(id) && !live.busy(), "later borrows are refused and the slot is free");

    // A cancel that arrives while tickets are still pending (the ledger wait itself, not the helper wait) is also not a
    // fault: feeding the close to wait_drained as an external fault would stop the ledger here.
    {
        Gen p; check(p.ledger.initialize(1, 0), "ledger");
        StepTicket tk{};
        check(p.ledger.try_admit(1, &tk) == Ledger::Result::Ok && p.ledger.bind(tk.ticket_id, 1), "pending ticket");
        const auto ps = p.ledger.close_admission();
        p.closing = true;
        check(await_drain(p, 1, ps.latest_ticket, std::chrono::steady_clock::now() + seconds(5), [] {}, [] { return false; },
                          [] { return true; }, [] { return true; }) == DrainWait::Cancelled, "cancel while tickets are pending");
        check(!p.ledger.stopped(), "a cancelled ledger wait neither faults nor interrupts the ledger");
    }

    // A genuine helper fault or a missed deadline still faults the ledger.
    Gen f; f.map = nullptr;
    check(f.ledger.initialize(1, 0), "ledger");
    const auto fs = f.ledger.close_admission();
    check(await_drain(f, 1, fs.latest_ticket, std::chrono::steady_clock::now() + seconds(5), [] {}, [] { return false; },
                      [] { return true; }, [] { return true; }) == DrainWait::Drained, "drains when helpers ack");
    Gen f2; check(f2.ledger.initialize(1, 0), "ledger");
    const auto f2s = f2.ledger.close_admission();
    check(await_drain(f2, 1, f2s.latest_ticket, std::chrono::steady_clock::now() + seconds(5), [] {}, [] { return true; },
                      [] { return true; }, [] { return false; }) == DrainWait::Failed && f2.ledger.stopped(),
          "a helper fault fails the drain and faults the ledger");
    Gen f3; check(f3.ledger.initialize(1, 0), "ledger");
    const auto f3s = f3.ledger.close_admission();
    check(await_drain(f3, 1, f3s.latest_ticket, std::chrono::steady_clock::now() + milliseconds(30), [] {}, [] { return false; },
                      [] { return true; }, [] { return false; }) == DrainWait::Failed && f3.ledger.stopped(),
          "a missed acknowledgment deadline faults the ledger");

    std::puts("r3_generation_close_test: OK");
}
