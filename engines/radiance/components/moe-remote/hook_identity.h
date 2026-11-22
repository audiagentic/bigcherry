// The identity of the one live remote-MoE hook (RUM06), kept apart from the HIP/shared-memory hook so it is tested on
// the host. The helper segment is a single named, process-wide resource, so at most one hook is live; this holds it as a
// non-owning mutual-exclusion and lookup pointer and nothing else (no policy or state hangs off it).
//  * Callers hold a serial, never an address. A serial is nonzero and never reused; the counter fails closed at its
//    end (no serial is issued after the last one), and a publisher that loses to a live hook consumes none.
//  * A resolved handle is a BORROW: borrow() takes the shared side of a lock for as long as the Borrow lives, and
//    retire() (the close: refuse new borrows, cancel the borrowers' waits, wait for them, free, unpublish) takes the
//    exclusive side, so the hook cannot be freed under a reader that already resolved it. unpublish() is only the
//    failed-open path, where no borrower exists. Borrowers are the cold paths (kernel init, the placement thread's cycles, direct entry points), never
//    a decode or prefill step.
// T must have a `uint64_t serial` member.
#pragma once
#include <atomic>
#include <cstdint>
#include <mutex>
#include <shared_mutex>

template <class T>
class LiveHook {
public:
    class Borrow {
    public:
        Borrow() = default;
        Borrow(Borrow&&) = default;
        Borrow& operator=(Borrow&&) = default;
        explicit operator bool() const { return p_ != nullptr; }
        T* get() const { return p_; }
        T& operator*() const { return *p_; }
    private:
        friend class LiveHook;
        Borrow(std::shared_lock<std::shared_mutex> lock, T* p) : lock_(std::move(lock)), p_(p) {}
        std::shared_lock<std::shared_mutex> lock_;
        T* p_ = nullptr;
    };

    explicit LiveHook(uint64_t first_serial = 1) : next_(first_serial) {}

    // True while a hook is published: the check an opener makes before it maps or allocates anything.
    bool busy() const { return live_.load(std::memory_order_acquire) != nullptr; }
    // True once every serial has been issued: no further hook can be published (fails closed, never wraps to 0).
    bool exhausted() const { return next_.load(std::memory_order_acquire) == 0; }

    // Publish `h` as the live hook and give it a fresh serial. 0 when another hook is live or the serials are
    // exhausted (nothing changed, no serial consumed).
    uint64_t publish(T* h) {
        std::unique_lock<std::shared_mutex> lock(mu_);
        if (live_.load(std::memory_order_relaxed)) return 0;
        const uint64_t id = next_.load(std::memory_order_relaxed);
        if (id == 0) return 0;
        next_.store(id + 1, std::memory_order_release);   // after UINT64_MAX this is 0: exhausted
        h->serial = id;
        live_.store(h, std::memory_order_release);
        return id;
    }
    // Unpublish `h` if it is the live hook, after every borrower has finished; later lookups by its serial find
    // nothing. Must not be called while holding a Borrow.
    bool unpublish(T* h) {
        std::unique_lock<std::shared_mutex> lock(mu_);
        if (live_.load(std::memory_order_relaxed) != h) return false;
        live_.store(nullptr, std::memory_order_release);
        return true;
    }
    // Close the generation `h` in the order that keeps every borrower safe: (1) refuse new borrows, (2) `wake()` -- the
    // caller cancels and wakes whatever a borrower may be blocked in (its own closing flag, a futex notify), (3) wait
    // for every existing borrow to end, (4) `release()` frees the state's resources while no borrower can be inside,
    // (5) unpublish. Not called from a thread holding a Borrow, and the hook's own worker threads (which borrow) must
    // already be stopped and joined. False, with nothing done, when `h` is not the live hook.
    template <class Wake, class Release>
    bool retire(T* h, Wake wake, Release release) {
        if (live_.load(std::memory_order_acquire) != h) return false;
        closing_.store(true, std::memory_order_release);
        wake();
        std::unique_lock<std::shared_mutex> lock(mu_);
        release();
        live_.store(nullptr, std::memory_order_release);
        closing_.store(false, std::memory_order_release);
        return true;
    }
    // The live hook for `id`, or an empty Borrow: the check every direct entry point makes before any side effect.
    Borrow borrow(uint64_t id) const {
        std::shared_lock<std::shared_mutex> lock(mu_);
        T* now = live_.load(std::memory_order_acquire);
        if (id == 0 || !now || now->serial != id || closing_.load(std::memory_order_acquire)) return {};
        return Borrow(std::move(lock), now);
    }
    // The serial of the live hook when `match(hook)` holds, else 0.
    template <class F>
    uint64_t serial_if(F match) const {
        std::shared_lock<std::shared_mutex> lock(mu_);
        T* now = live_.load(std::memory_order_acquire);
        return (now && match(*now)) ? now->serial : 0;
    }

    // The serial a kernel init is entitled to: only the exact generation that owns it (`owner` is the serial the engine's
    // generation was opened with and put in its ops), and only for that generation's own span, width and rank. A foreign
    // generation on the same numeric group, or a closed owner whose span was reopened by a later generation, gets 0.
    uint64_t serial_for_owner(uint64_t owner, long long group, int rank, int world) const {
        return serial_if([&](const T& h) {
            return owner != 0 && h.serial == owner && h.opened && h.slot_base == group && world == h.ranks &&
                   rank >= 0 && rank < h.ranks;
        });
    }

private:
    mutable std::shared_mutex mu_;
    std::atomic<T*> live_{nullptr};
    std::atomic<bool> closing_{false};
    std::atomic<uint64_t> next_;
};
