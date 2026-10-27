#pragma once
// Engine side (libr3moe). RAD20 helper liveness: a helper that stops sets Header::helper_fault itself, but one that is
// killed, crashes or loses its GPU cannot, and the engine's GPU wait (mailbox_add_wait_kernel) then spins on a done word
// that never moves (an 8.7 h hang, 2026-10-26). This watcher holds a pidfd for each helper process
// (Header::helper_pid) and sleeps in poll(); when a helper exits it sets helper_fault and wakes every CPU waiter, so the
// engine's GPU wait traps and its CPU waits return. Event-driven: one persistent thread, blocked, no timeout.
//
// Helpers publish their pid after attaching, which may be after the engine has registered; notice() (called on the
// engine's burst and layer-map paths, a few loads) hands newly published pids to the thread through an eventfd.
#include "moe_remote_shm.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>

class HelperWatch {
public:
    HelperWatch() = default;
    HelperWatch(const HelperWatch&) = delete;
    HelperWatch& operator=(const HelperWatch&) = delete;
    ~HelperWatch() { stop(); }

    /* False (logged) when the watch cannot run; the engine then has no liveness check, as before. */
    bool start(r3::moe_shm::Header* h) {
        h_ = h;
        wake_ = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
        if (wake_ < 0) return fail("eventfd");
        try {
            thread_ = std::thread([this] { run(); });
        } catch (const std::exception&) {
            ::close(wake_);
            wake_ = -1;
            return fail("thread");
        }
        return true;
    }

    /* Before the segment is unmapped. */
    void stop() {
        if (thread_.joinable()) {
            __atomic_store_n(&quit_, 1u, __ATOMIC_RELEASE);
            signal();
            thread_.join();
        }
        if (wake_ >= 0) ::close(wake_);
        wake_ = -1;
    }

    /* A helper has published a pid the thread is not watching yet: wake it to open a pidfd. */
    void notice() {
        if (wake_ < 0) return;
        for (uint32_t i = 0; i < r3::moe_shm::kMaxRanks; ++i)
            if (__atomic_load_n(&h_->helper_pid[i], __ATOMIC_ACQUIRE) > 0 &&
                __atomic_load_n(&watched_[i], __ATOMIC_ACQUIRE) == 0) {
                signal();
                return;
            }
    }

private:
    static bool fail(const char* what) {
        std::fprintf(stderr, "libr3: R3_MOE_WORKER helper watch: %s failed (%s); a killed helper will not be detected\n",
                     what, std::strerror(errno));
        return false;
    }
    void signal() {
        const uint64_t one = 1;
        (void)!::write(wake_, &one, sizeof(one));   /* EAGAIN: already pending, which is all we need */
    }

    void run() {
        int pidfd[r3::moe_shm::kMaxRanks];
        for (auto& f : pidfd) f = -1;
        while (watch_once(pidfd)) {}
        for (int f : pidfd)
            if (f >= 0) ::close(f);
    }

    /* One wait. False when the watch ends: stop(), a helper exited (helper_fault raised), or poll failed. */
    bool watch_once(int* pidfd) {
        using namespace r3::moe_shm;
        /* Pick up helpers that have published since the last pass. A pid already gone fails pidfd_open with ESRCH:
         * that helper died before we saw it. */
        for (uint32_t i = 0; i < kMaxRanks; ++i) {
            const int32_t pid = __atomic_load_n(&h_->helper_pid[i], __ATOMIC_ACQUIRE);
            if (pid <= 0 || pidfd[i] >= 0) continue;
            pidfd[i] = (int)::syscall(SYS_pidfd_open, pid, 0);
            if (pidfd[i] < 0 && errno == ESRCH) return fault(i, pid);
            if (pidfd[i] < 0)
                std::fprintf(stderr, "libr3: R3_MOE_WORKER helper watch: pidfd_open(%d) failed (%s)\n", pid,
                             std::strerror(errno));
            __atomic_store_n(&watched_[i], 1u, __ATOMIC_RELEASE);
        }
        pollfd fds[kMaxRanks + 1];
        uint32_t who[kMaxRanks + 1];
        nfds_t n = 0;
        fds[n] = {wake_, POLLIN, 0};
        who[n++] = UINT32_MAX;
        for (uint32_t i = 0; i < kMaxRanks; ++i)
            if (pidfd[i] >= 0) { fds[n] = {pidfd[i], POLLIN, 0}; who[n++] = i; }
        if (::poll(fds, n, -1) < 0) {
            if (errno == EINTR) return true;
            std::fprintf(stderr, "libr3: R3_MOE_WORKER helper watch: poll failed (%s)\n", std::strerror(errno));
            return false;
        }
        if (fds[0].revents) {
            uint64_t v;
            (void)!::read(wake_, &v, sizeof(v));
            if (__atomic_load_n(&quit_, __ATOMIC_ACQUIRE)) return false;
        }
        for (nfds_t k = 1; k < n; ++k)
            if (fds[k].revents) return fault(who[k], __atomic_load_n(&h_->helper_pid[who[k]], __ATOMIC_ACQUIRE));
        return true;
    }

    /* A helper process is gone. A helper that stopped cleanly has already set helper_fault; this covers the rest.
     * Returns false: the watch ends (the engine is failing; there is nothing left to watch for). */
    bool fault(uint32_t helper, int32_t pid) {
        using namespace r3::moe_shm;
        if (__atomic_load_n(&quit_, __ATOMIC_ACQUIRE)) return false;   /* engine shutting down: helpers may exit first */
        std::fprintf(stderr, "libr3: R3_MOE_WORKER helper %u (pid %d) exited; failing the engine's helper waits\n",
                     helper, pid);
        store(&h_->helper_fault, 1u);
        (void)futex_wake(&h_->burst);
        for (uint32_t lane = 0; lane < kLanes; ++lane) (void)futex_wake(&h_->lane[lane].park);
        for (uint32_t layer = 0; layer < kMaxLayers; ++layer) (void)futex_wake(&h_->layer_map_consumed[layer]);
        return false;
    }

    r3::moe_shm::Header* h_ = nullptr;
    int wake_ = -1;
    uint32_t quit_ = 0;
    uint32_t watched_[r3::moe_shm::kMaxRanks] = {};
    std::thread thread_;
};
