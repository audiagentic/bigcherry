// RUM06 review P1-B: the helper registration barrier, driven with fake helpers (real pidfd watcher, real header).
// delayed-but-valid registration passes; a helper that never registers ends at its bound with the sticky fault; a
// helper that dies fails the barrier at once; and the production bound is the startup protocol's, not the placement
// readiness deadline (swap the source and this fails).
#include "helper_registration_wait.h"
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

using namespace r3::moe_shm;
namespace wire = r3::moe_shm::v26;
using std::chrono::milliseconds;

static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "r3_helper_registration_test: FAILED %s\n", what); std::exit(1); }
}
constexpr uint64_t kManifest = 0x13572680ae691d31ull;
constexpr LayoutShape kShape{128, 2, 3, 8, 128, 64};

struct Seg {
    Layout layout{}; void* data = nullptr;
    Seg() {
        check(wire::layout_for(kShape, 2, {1, 1}, &layout) == Refusal::None, "layout");
        data = mmap(nullptr, size_t(layout.bytes), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);   // shared: forked fakes see it
        check(data != MAP_FAILED, "map");
        check(wire::init(data, size_t(layout.bytes), kShape, 2, {1, 1}, kManifest) == Refusal::None, "init");
    }
    ~Seg() { munmap(data, size_t(layout.bytes)); }
    Header* h() const { return static_cast<Header*>(data); }
};

static void publish_helper(Header* h, int pid) {
    h->helper_start[0] = process_start_ticks(pid);
    __atomic_store_n(&h->helper_pid[0], (int32_t)pid, __ATOMIC_RELEASE);
    notify(h);
}
static pid_t fake_helper() {
    const pid_t p = fork();
    if (p == 0) { for (;;) pause(); }
    return p;
}

int main() {
    // The deadline sources: registration is the startup protocol's, and is not the placement readiness bound.
    check(registration_bound() == std::chrono::seconds(kStartupWaitSeconds), "registration bound is the startup deadline");
    check(registration_bound() != layer_map_bound(), "registration bound is not the placement readiness deadline");

    {   // delayed but valid: the helper publishes its identity after the engine started waiting
        Seg s; HelperWatch w; check(w.start(s.h()), "watch");
        const pid_t p = fake_helper();
        std::thread late([&] { std::this_thread::sleep_for(milliseconds(60)); publish_helper(s.h(), p); });
        check(await_helpers_registered(s.h(), w, milliseconds(20000)) == WaitResult::Ready, "late valid registration passes");
        late.join();
        check(!load(&s.h()->helper_fault), "no fault after a valid registration");
        w.stop(); kill(p, SIGKILL); waitpid(p, nullptr, 0);
    }
    {   // never registers: bounded, sticky fault
        Seg s; HelperWatch w; check(w.start(s.h()), "watch");
        const auto t0 = std::chrono::steady_clock::now();
        check(await_helpers_registered(s.h(), w, milliseconds(50)) == WaitResult::Timeout, "never registers: times out");
        check(std::chrono::steady_clock::now() - t0 < milliseconds(5000), "bounded by the given bound");
        check(load(&s.h()->helper_fault) != 0, "timeout raises the sticky helper fault");
        const auto t1 = std::chrono::steady_clock::now();
        check(await_helpers_registered(s.h(), w, milliseconds(100000)) == WaitResult::Fault, "after the fault nothing waits");
        check(std::chrono::steady_clock::now() - t1 < milliseconds(1000), "fault ends later waits at once");
        w.stop();
    }
    {   // helper dies before it is watched: the barrier fails at once, not at the bound. The death is complete before
        // the watcher starts, so the watcher can only find an exited process (no race with its first pass).
        Seg s;
        const pid_t p = fake_helper();
        publish_helper(s.h(), p);
        kill(p, SIGKILL); waitpid(p, nullptr, 0);
        HelperWatch w; check(w.start(s.h()), "watch");
        const auto t0 = std::chrono::steady_clock::now();
        const WaitResult r = await_helpers_registered(s.h(), w, milliseconds(20000));
        check(r == WaitResult::Fault, "a dead helper fails the barrier");
        check(std::chrono::steady_clock::now() - t0 < milliseconds(5000), "a dead helper does not wait out the bound");
        check(load(&s.h()->helper_fault) != 0, "death leaves the sticky fault");
        w.stop();
    }
    {   // helper dies after it is watched: the watcher raises the fault; the barrier then fails at once. The test waits
        // for the event it asserts (the watcher holding the helper, then the fault word), each bounded by the
        // registration bound, never a fixed sleep.
        Seg s; HelperWatch w; check(w.start(s.h()), "watch");
        const pid_t p = fake_helper();
        publish_helper(s.h(), p);
        const auto limit = std::chrono::steady_clock::now() + registration_bound();
        while (!w.watching(0) && std::chrono::steady_clock::now() < limit) { w.notice(); std::this_thread::yield(); }
        check(w.watching(0), "the watcher holds the helper");
        kill(p, SIGKILL); waitpid(p, nullptr, 0);
        while (!load(&s.h()->helper_fault) && std::chrono::steady_clock::now() < limit) std::this_thread::yield();
        check(load(&s.h()->helper_fault) != 0, "the watcher raised the fault for the dead helper");
        check(await_helpers_registered(s.h(), w, milliseconds(20000)) == WaitResult::Fault, "a helper that died while watched fails the barrier");
        w.stop();
    }
    std::puts("r3_helper_registration_test: OK");
}
