// The generation's close orchestration and the direct drain wait (RUM06), header-only so the production order and the
// cancel semantics are tested on the host with the real Ledger, LiveHook and shared waits. moe_remote_hook.cpp's
// cb_close and r3_moe_wait_drained are thin callers of these.
#pragma once
#include "admission_ledger.h"
#include <atomic>
#include <chrono>
#include <thread>

/* Close `g` (live in `live`). The order is the safety argument:
 *  1. the kind of stop is decided first: cancelling waiters below must not change it;
 *  2. the hook's own workers are stopped and joined (they use the direct entry points, so this must precede retire,
 *     which waits for borrowers);
 *  3. LiveHook::retire: refuse new borrows, set g.closing and wake every wait a borrower may be blocked in (a
 *     Ledger::wake, never an interrupt or fault on a clean stop), wait for the borrowers to leave, then free;
 * g needs `ledger` (r3::admission::Ledger) and `std::atomic<bool> closing`. */
template <class Live, class G, class StopWorkers, class WakeWaiters, class Release>
void close_generation(Live& live, G& g, StopWorkers stop_workers, WakeWaiters wake_waiters, Release release) {
    stop_workers(g.ledger.clean_stop());
    live.retire(&g,
                [&] {
                    g.closing.store(true, std::memory_order_release);
                    wake_waiters();
                    g.ledger.wake();
                },
                release);
}

enum class DrainWait { Drained, Cancelled, Failed };

/* The direct drain wait: the ledger drains the epoch's tickets up to the cutoff, then every helper acknowledges.
 * A close of the generation CANCELS the wait (Cancelled, no ledger state touched, nothing proceeds to COMMIT); a genuine
 * helper fault, a missed deadline or a stopped ledger fails it and faults the ledger. */
template <class G, class Poll, class HelperFaulted, class RequestDrain, class HelperDrained>
DrainWait await_drain(G& g, uint32_t old_epoch, uint64_t cutoff, std::chrono::steady_clock::time_point limit,
                      Poll poll, HelperFaulted helper_faulted, RequestDrain request_drain, HelperDrained helper_drained) {
    using Result = r3::admission::Ledger::Result;
    auto closing = [&] { return g.closing.load(std::memory_order_acquire); };
    const Result drained = g.ledger.wait_drained(old_epoch, cutoff, limit, [&] { poll(); },
                                                 [&] { return helper_faulted(); }, closing);
    if (drained == Result::Cancelled) return DrainWait::Cancelled;
    if (drained != Result::Ok) { g.ledger.fault(); return DrainWait::Failed; }
    if (!request_drain()) return DrainWait::Failed;   /* the closure changed under us: a state error, not a fault */
    while (!helper_drained()) {
        if (closing()) return DrainWait::Cancelled;
        if (g.ledger.stopped() || helper_faulted() || std::chrono::steady_clock::now() >= limit) {
            g.ledger.fault();
            return DrainWait::Failed;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return DrainWait::Drained;
}
