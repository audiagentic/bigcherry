// RUM06 re-review P1-A: closing a generation cancels the waits borrowers are blocked in, waits for them to leave, and
// only then frees. A borrower paused inside the REAL bootstrap-seal wait (the 1800 s one in r3_moe_mailbox_device) is
// woken by the close, leaves with an error (not by timeout), and the segment is not unmapped until it has left.
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

static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "r3_hook_retire_test: FAILED %s\n", what); std::exit(1); }
}
constexpr uint64_t kManifest = 0x13572680ae691d31ull;
constexpr LayoutShape kShape{128, 2, 3, 8, 128, 64};

struct Hook {
    uint64_t serial = 0; int slot_base = 0, ranks = 1; bool opened = true;
    std::atomic<bool> closing{false};
    void* map = nullptr; size_t bytes = 0;
    std::atomic<bool> freed{false};
};

int main() {
    Layout layout{};
    check(wire::layout_for(kShape, 2, {1, 1}, &layout) == Refusal::None, "layout");
    Hook hook;
    hook.bytes = size_t(layout.bytes);
    hook.map = mmap(nullptr, hook.bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    check(hook.map != MAP_FAILED, "map");
    check(wire::init(hook.map, hook.bytes, kShape, 2, {1, 1}, kManifest) == Refusal::None, "init");   // never sealed
    Header* h = static_cast<Header*>(hook.map);

    LiveHook<Hook> live;
    const uint64_t id = live.publish(&hook);
    check(id != 0, "publish");

    std::mutex mu;
    std::atomic<uint32_t> ready{0};
    std::atomic<bool> in_wait{false}, freed_while_inside{false};
    std::atomic<int> result{-1};
    std::thread borrower([&] {
        auto b = live.borrow(id);
        check(static_cast<bool>(b), "borrow");
        timespec deadline{}; clock_gettime(CLOCK_MONOTONIC, &deadline);
        deadline.tv_sec += 60;                    // the production bound is 1800 s: only the cancel may end this
        in_wait = true;
        const int r = r3::owner_generation::with_bootstrap_rank_lock(
            hook.map, ready, 0, mu, &deadline, []() -> int { return int(WaitResult::Ready); },
            [](WaitResult e) -> int { return int(e); }, &hook.closing);
        freed_while_inside = hook.freed.load();   // still holding the borrow: nothing may be freed yet
        (void)load(&h->helper_count);             // the header must still be mapped (ASan/segv otherwise)
        result = r;
    });
    while (!in_wait) std::this_thread::yield();
    std::this_thread::sleep_for(milliseconds(50));
    check(result.load() == -1, "the borrower is blocked in the seal wait");

    const auto t0 = std::chrono::steady_clock::now();
    const bool retired = live.retire(
        &hook,
        [&] { hook.closing = true; notify(h); },                         // refuse + cancel + wake
        [&] { hook.freed = true; munmap(hook.map, hook.bytes); hook.map = nullptr; });   // only after every borrow ended
    borrower.join();
    check(retired, "retire");
    check(std::chrono::steady_clock::now() - t0 < milliseconds(10000), "cancelled, not timed out");
    check(result.load() == int(WaitResult::Error), "cancelled wait returns an error");
    check(!freed_while_inside.load(), "nothing freed while a borrower was inside");
    check(hook.freed.load(), "freed after the borrower left");
    check(!live.borrow(id) && !live.busy(), "retired handle is refused and the slot is free");

    // Retiring something that is not live changes nothing and frees nothing.
    Hook other; bool released = false;
    check(!live.retire(&other, [] {}, [&] { released = true; }) && !released, "retire of a non-live hook is refused");
    // New borrows are refused as soon as intake is closed, before the borrowers have drained.
    Hook second; second.opened = true;
    const uint64_t id2 = live.publish(&second);
    std::atomic<bool> inside{false}, go{false};
    std::thread holder([&] { auto b = live.borrow(id2); inside = true; while (!go) std::this_thread::yield(); });
    while (!inside) std::this_thread::yield();
    std::atomic<bool> refused{false};
    std::thread closer([&] {
        live.retire(&second, [&] { second.closing = true; }, [] {});
    });
    for (int i = 0; i < 2000 && !refused; ++i) {
        if (!live.borrow(id2)) refused = true; else std::this_thread::sleep_for(milliseconds(1));
    }
    check(refused, "intake closes before the drain finishes");
    go = true; holder.join(); closer.join();
    std::puts("r3_hook_retire_test: OK");
}
