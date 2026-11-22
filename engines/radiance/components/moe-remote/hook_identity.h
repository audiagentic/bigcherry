// The identity of the one live remote-MoE hook (RUM06 S3), kept apart from the HIP/shared-memory hook so it is tested on
// the host. The helper segment is a single named, process-wide resource, so at most one hook is live; this holds it as a
// non-owning mutual-exclusion and lookup pointer and nothing else (no policy or state hangs off it). Callers hold a
// serial, never an address: the serial is nonzero and never reused, so a handle from a closed hook, another engine's, or
// one that never existed no longer matches and is refused, even when a later hook is allocated at the same address.
// T must have a `uint64_t serial` member. Whoever closes a hook does so after the threads that use its handle stopped.
#pragma once
#include <atomic>
#include <cstdint>

template <class T>
class LiveHook {
public:
    // True while a hook is published: the check an opener makes before it maps or allocates anything.
    bool busy() const { return live_.load(std::memory_order_acquire) != nullptr; }
    // Publish `h` as the live hook and give it a fresh serial. 0 when another hook is live (nothing changed).
    uint64_t publish(T* h) {
        T* none = nullptr;
        const uint64_t id = next_.fetch_add(1, std::memory_order_relaxed);
        h->serial = id;
        if (!live_.compare_exchange_strong(none, h, std::memory_order_acq_rel)) return 0;
        return id;
    }
    // Unpublish `h` if it is the live hook; later lookups by its serial find nothing.
    bool unpublish(T* h) {
        T* expected = h;
        return live_.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel);
    }
    // The live hook for `id`, or null: the check every direct entry point makes before any side effect.
    T* resolve(uint64_t id) const {
        T* now = live_.load(std::memory_order_acquire);
        return (id != 0 && now && now->serial == id) ? now : nullptr;
    }
    // The serial of the live hook when `match(hook)` holds, else 0.
    template <class F>
    uint64_t serial_if(F match) const {
        T* now = live_.load(std::memory_order_acquire);
        return (now && match(*now)) ? now->serial : 0;
    }

private:
    std::atomic<T*> live_{nullptr};
    std::atomic<uint64_t> next_{1};
};
