// RUM06 review P1-1: the wait for a helper to consume a published layer map is bounded. A helper that is alive but
// never consumes ends the wait at its bound with a sticky helper fault (every later wait returns at once), a consumed
// map and a faulted helper end it at once, and teardown after the fault is not blocked.
#include "layer_map_wait.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>
#include <thread>

using namespace r3::moe_shm;
namespace wire = r3::moe_shm::v26;
using std::chrono::milliseconds;

static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "r3_layer_map_wait_test: FAILED %s\n", what); std::exit(1); }
}
constexpr uint64_t kManifest = 0x13572680ae691d31ull;
constexpr LayoutShape kShape{128, 2, 3, 8, 128, 64};

int main() {
    const Topology topo{1, 1};
    Layout layout{};
    check(wire::layout_for(kShape, 2, topo, &layout) == Refusal::None, "layout");
    void* data = mmap(nullptr, size_t(layout.bytes), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    check(data != MAP_FAILED, "map");
    check(wire::init(data, size_t(layout.bytes), kShape, 2, topo, kManifest) == Refusal::None, "init");
    Header* h = static_cast<Header*>(data);

    // Consumed already: returns at once.
    __atomic_store_n(&h->layer_map_consumed[0], 1u, __ATOMIC_RELEASE);
    check(await_layer_consumed(h, 0, milliseconds(1000)) == MapWait::Consumed, "consumed map ends the wait");

    // Consumed by the helper while the engine waits: woken, not timed out.
    std::thread helper([&] {
        std::this_thread::sleep_for(milliseconds(30));
        __atomic_store_n(&h->layer_map_consumed[1], 1u, __ATOMIC_RELEASE);
        notify(h);
    });
    check(await_layer_consumed(h, 1, milliseconds(5000)) == MapWait::Consumed, "a late consume wakes the waiter");
    helper.join();

    // A helper that is alive but never consumes: the wait ends at its bound, the fault is raised and sticky.
    check(!load(&h->helper_fault), "no fault yet");
    const auto t0 = std::chrono::steady_clock::now();
    check(await_layer_consumed(h, 1, milliseconds(1)) == MapWait::Consumed, "layer 1 already consumed");
    __atomic_store_n(&h->layer_map_consumed[1], 0u, __ATOMIC_RELEASE);
    check(await_layer_consumed(h, 1, milliseconds(40)) == MapWait::Timeout, "alive but never consuming: bounded");
    const auto waited = std::chrono::steady_clock::now() - t0;
    check(waited >= milliseconds(40) && waited < milliseconds(4000), "the wait lasted its bound, not forever");
    check(load(&h->helper_fault) != 0, "a missed bound raises the sticky helper fault");

    // Teardown after the fault is not blocked: every later wait returns at once, whatever the bound.
    const auto t1 = std::chrono::steady_clock::now();
    check(await_layer_consumed(h, 1, milliseconds(100000)) == MapWait::Faulted, "faulted helper ends the wait");
    check(await_layer_consumed(h, 0, milliseconds(100000)) == MapWait::Faulted, "fault wins over a consumed map");
    check(std::chrono::steady_clock::now() - t1 < milliseconds(1000), "no wait after the fault");

    // The helper-registration wait uses the same derived bound, expressed as an absolute monotonic deadline.
    timespec now{}; clock_gettime(CLOCK_MONOTONIC, &now);
    const timespec d = deadline_after(milliseconds(1500));
    const int64_t delta_ms = (int64_t(d.tv_sec) - now.tv_sec) * 1000 + (d.tv_nsec - now.tv_nsec) / 1000000;
    check(delta_ms >= 1490 && delta_ms <= 1600 && d.tv_nsec >= 0 && d.tv_nsec < 1000000000, "absolute deadline after a bound");
    check(layer_map_bound() == std::chrono::nanoseconds(r3::epoch::Deadlines{}.ready_ns), "bound is the protocol readiness deadline");

    // The engine's wait total and max per layer (MapWaitStats), with a fake helper that consumes each layer late.
    // A fresh segment: the fault above is sticky.
    void* data2 = mmap(nullptr, size_t(layout.bytes), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    check(data2 != MAP_FAILED && wire::init(data2, size_t(layout.bytes), kShape, 2, topo, kManifest) == Refusal::None, "init 2");
    Header* h2 = static_cast<Header*>(data2);
    MapWaitStats stats;
    check(stats.total().calls == 0 && stats.line() == "libr3: layer_map_wait {\"calls\":0,\"total_ns\":0,\"max_ns\":0,\"per_layer\":[]}",
          "empty stats line");
    const int delays_ms[2] = {30, 60};
    std::thread fake([&] {
        for (int l = 0; l < 2; ++l) {
            std::this_thread::sleep_for(milliseconds(delays_ms[l]));
            __atomic_store_n(&h2->layer_map_consumed[l], 1u, __ATOMIC_RELEASE);
            notify(h2);
        }
    });
    for (int l = 0; l < 2; ++l)
        check(await_layer_consumed_timed(h2, l, milliseconds(5000), &stats) == MapWait::Consumed, "timed wait consumed");
    fake.join();
    // A wait that is already satisfied still counts as a call, with a tiny wall.
    check(await_layer_consumed_timed(h2, 0, milliseconds(5000), &stats) == MapWait::Consumed, "timed wait, already consumed");
    check(stats.layer[0].calls == 2 && stats.layer[1].calls == 1, "calls per layer");
    check(stats.layer[0].max_ns >= 30000000ull && stats.layer[1].max_ns >= 30000000ull, "waits cover the helper's delay");
    const auto tot = stats.total();
    check(tot.calls == 3 && tot.total_ns >= 60000000ull && tot.max_ns == std::max(stats.layer[0].max_ns, stats.layer[1].max_ns) &&
          tot.total_ns == stats.layer[0].total_ns + stats.layer[1].total_ns, "totals and max");
    const std::string line = stats.line();
    check(line.rfind("libr3: layer_map_wait {\"calls\":3,\"total_ns\":", 0) == 0 &&
          line.find("\"per_layer\":[[0,2,") != std::string::npos && line.find(",[1,1,") != std::string::npos &&
          line.find('\n') == std::string::npos, "stats line format");
    // A bounded timeout is recorded too (the wait wall is the bound), and a null sink is allowed.
    MapWaitStats late;
    __atomic_store_n(&h2->layer_map_consumed[1], 0u, __ATOMIC_RELEASE);
    check(await_layer_consumed_timed(h2, 1, milliseconds(40), &late) == MapWait::Timeout, "timeout through timed wait");
    check(late.layer[1].calls == 1 && late.layer[1].total_ns >= 40000000ull, "timeout wall recorded");
    check(await_layer_consumed_timed(h2, 1, milliseconds(40), nullptr) == MapWait::Faulted, "null sink allowed");
    munmap(data2, size_t(layout.bytes));

    munmap(data, size_t(layout.bytes));
    std::puts("r3_layer_map_wait_test: OK");
}
