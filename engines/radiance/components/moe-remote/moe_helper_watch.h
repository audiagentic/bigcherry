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

    /* False (logged) when the watch cannot run: the caller must not use the helpers then. */
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

    /* True once helper i's process is held by a validated pidfd. */
    bool watching(uint32_t i) const { return __atomic_load_n(&watched_[i], __ATOMIC_ACQUIRE) != 0; }

    /* A helper has published a pid the thread is not watching yet: wake it to open a pidfd. */
    void notice() {
        if (wake_ < 0) return;
        for (uint32_t i = 0; i < r3::moe_shm::kMaxHelpers; ++i)
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
        int pidfd[r3::moe_shm::kMaxHelpers];
        for (auto& f : pidfd) f = -1;
        while (watch_once(pidfd)) {}
        for (int f : pidfd)
            if (f >= 0) ::close(f);
    }

    /* One wait. False when the watch ends: stop(), or a helper is gone or cannot be watched (helper_fault raised).
     * Fails closed: the engine never keeps serving with a helper it cannot watch. */
    bool watch_once(int* pidfd) {
        using namespace r3::moe_shm;
        /* Pick up helpers that have published since the last pass. The pidfd names the process only if its start
         * time still matches the helper's own record: a helper that died before we opened it (ESRCH, or its pid
         * already reused by another process) is a dead helper. */
        for (uint32_t i = 0; i < kMaxHelpers; ++i) {
            const int32_t pid = __atomic_load_n(&h_->helper_pid[i], __ATOMIC_ACQUIRE);
            if (pid <= 0 || pidfd[i] >= 0) continue;
            pidfd[i] = (int)::syscall(SYS_pidfd_open, pid, 0);
            if (pidfd[i] < 0)
                return fault(i, pid, errno == ESRCH ? "exited" : std::strerror(errno));
            const uint64_t start = process_start_ticks(pid);
            if (!start || start != h_->helper_start[i]) return fault(i, pid, "exited (pid reused)");
            __atomic_store_n(&watched_[i], 1u, __ATOMIC_RELEASE);
            notify(h_);   /* the engine's registration barrier waits for every helper to be watched */
        }
        pollfd fds[kMaxHelpers + 1];
        uint32_t who[kMaxHelpers + 1];
        nfds_t n = 0;
        fds[n] = {wake_, POLLIN, 0};
        who[n++] = UINT32_MAX;
        for (uint32_t i = 0; i < kMaxHelpers; ++i)
            if (pidfd[i] >= 0) { fds[n] = {pidfd[i], POLLIN, 0}; who[n++] = i; }
        if (::poll(fds, n, -1) < 0) {
            if (errno == EINTR) return true;
            return fault(UINT32_MAX, 0, std::strerror(errno));
        }
        if (fds[0].revents) {
            uint64_t v;
            (void)!::read(wake_, &v, sizeof(v));
            if (__atomic_load_n(&quit_, __ATOMIC_ACQUIRE)) return false;
        }
        for (nfds_t k = 1; k < n; ++k)
            if (fds[k].revents)
                return fault(who[k], __atomic_load_n(&h_->helper_pid[who[k]], __ATOMIC_ACQUIRE), "exited");
        return true;
    }

    /* A helper is gone, or the watch itself failed (helper UINT32_MAX). A helper that stopped cleanly has already set
     * helper_fault; this covers the rest. Raising the fault wakes every engine CPU wait (Header::notify) and makes the
     * engine's next GPU wait trap. Returns false: the watch ends. */
    bool fault(uint32_t helper, int32_t pid, const char* why) {
        using namespace r3::moe_shm;
        if (__atomic_load_n(&quit_, __ATOMIC_ACQUIRE)) return false;   /* engine shutting down: helpers may exit first */
        if (helper == UINT32_MAX)
            std::fprintf(stderr, "libr3: R3_MOE_WORKER helper watch failed (%s); failing the engine's helper waits\n", why);
        else
            std::fprintf(stderr, "libr3: R3_MOE_WORKER helper %u (pid %d) %s; failing the engine's helper waits\n",
                         helper, pid, why);
        raise_fault(h_);
        return false;
    }

    r3::moe_shm::Header* h_ = nullptr;
    int wake_ = -1;
    uint32_t quit_ = 0;
    uint32_t watched_[r3::moe_shm::kMaxHelpers] = {};
    std::thread thread_;
};
