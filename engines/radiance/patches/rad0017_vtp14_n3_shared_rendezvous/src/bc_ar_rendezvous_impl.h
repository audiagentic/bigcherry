/* rad0017: typed process-wide rendezvous and poison; frozen 2.2 projection. */
#pragma once

#include "r4d_ar_links.h"
#include "r4d_chain.h"

#include <dlfcn.h> /* BC_VTP14_N3_INCLUDE: the engine's shared-slot service */
#include <new>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include <chrono>

static_assert(R4D_AR_TWOSHOT_WIDE_MAX_ELEMS == 268435456,
              "r4d_rows.cpp restates this bound; libr4d changed it");
static_assert(R4D_MAX_RANKS == 8, "the 8-rank entry point names exactly seven peers");

/* ------------------------------------------------------------------ throw conversion
 *
 * The all-reduce entry points throw; the ABI returns error codes, so every call is wrapped. The
 * shim pre-checks everything the entry points throw about, so an escaping throw is a library or
 * device fault: it becomes RAD_E_DEVICE and is printed once. */
static int r4d_caught(const char* who, const char* what) {
    static bool said = false;
    if (!said) {
        said = true;
        std::fprintf(stderr, "[r4d] %s threw: %s\n", who, what ? what : "(null)");
    }
    return RAD_E_DEVICE;
}

#define R4D_TRY(who, ...)                                                       \
    do {                                                                        \
        try { __VA_ARGS__; }                                                    \
        catch (const std::exception& e_) { return r4d_caught(who, e_.what()); } \
        catch (...) { return r4d_caught(who, "unknown exception"); }            \
    } while (0)

/* ------------------------------------------------------------------ the rendezvous */

namespace {

struct Rendezvous {
    std::mutex              m;
    std::condition_variable cv;
    int  arrived = 0;
    int  epoch   = 0;
    long scratch[R4D_MAX_RANKS] = { 0 };
    long flags  [R4D_MAX_RANKS] = { 0 };
    long seq    [R4D_MAX_RANKS] = { 0 };
    int  dev    [R4D_MAX_RANKS] = { 0 };
    /* Each rank's card's link, which the eight-rank two-shot's weights are derived from. */
    R4DArLink ar8_link[R4D_MAX_RANKS];
    int  failed = 0;
    int  owner = 0;  /* BC_VTP14_N3_OWNER: engine lifetime within this group */
};

/* One rendezvous per tensor-parallel group. Several engines may run in one process
 * (docs/MULTIMODEL.md), each with its own ranks 0..world-1, so the ranks that meet are the ranks of
 * one group: the core names it in init's `tp_group` parameter (absent, the one group 0). Entries
 * live for the process; a group's id is reused only by a later engine with the same ranks. */
std::mutex                                       g_rv_mu;
std::map<long long, std::unique_ptr<Rendezvous>> g_rvs;

Rendezvous& rv_for(long long group) {
    using SlotFn = int (*)(const char*, size_t, size_t, void (*)(void*), void**, char*, size_t);
    using OwnerFn = int (*)();
    static const auto slot = reinterpret_cast<SlotFn>(dlsym(RTLD_DEFAULT, "rad_bc_shared_slot"));
    static const auto current_owner =
        reinterpret_cast<OwnerFn>(dlsym(RTLD_DEFAULT, "rad_bc_rendezvous_owner"));
    // BC_VTP14_N3_SHARED_RV: standalone tools use their local per-group rendezvous.
    if (slot && current_owner) {
        char key[96], why[320] = {};
        std::snprintf(key, sizeof key, "libr4d.ar.rendezvous/v2/group=%lld", group);
        void* storage = nullptr;
        if (slot(key, sizeof(Rendezvous), alignof(Rendezvous),
                 [](void* p) { new (p) Rendezvous(); }, &storage, why, sizeof why) == RAD_OK && storage) {
            auto& rv = *static_cast<Rendezvous*>(storage);
            const int owner = current_owner();
            std::lock_guard<std::mutex> lk(rv.m);
            if (rv.owner != owner) {
                // A slot base can be claimed again after all prior ranks have joined.
                rv.owner = owner;
                rv.failed = 0;
                rv.arrived = 0;
                rv.epoch = 0;
                std::memset(rv.scratch, 0, sizeof rv.scratch);
                std::memset(rv.flags, 0, sizeof rv.flags);
                std::memset(rv.seq, 0, sizeof rv.seq);
                std::memset(rv.dev, 0, sizeof rv.dev);
            }
            return rv;
        }
        std::fprintf(stderr, "[r4d] shared all-reduce group %lld: %s\n", group, why);
    }
    std::lock_guard<std::mutex> lk(g_rv_mu);
    std::unique_ptr<Rendezvous>& r = g_rvs[group];
    if (!r) {
        r.reset(new Rendezvous());
        if (slot || current_owner) r->failed = 1;
    }
    return *r;
}

/* Sense-reversing barrier on a monotone epoch, so a rank that wakes and re-enters for the next
 * instance cannot be counted twice. The timeout is not recovery; it turns a startup deadlock into
 * a named error instead of a silent hang. */
int rv_barrier(Rendezvous& rv, int world) {
    std::unique_lock<std::mutex> lk(rv.m);
    if (rv.failed) return RAD_E_STATE;  /* BC_VTP14_N3_POISON */
    const int e = rv.epoch;
    if (++rv.arrived == world) {
        rv.arrived = 0;
        rv.epoch++;
        rv.cv.notify_all();
        return RAD_OK;
    }
    const bool ok = rv.cv.wait_for(lk, std::chrono::seconds(60),
                                   [&] { return rv.epoch != e || rv.failed; });
    /* A rank that gave up takes its arrival back: a peer that failed before arriving would
     * otherwise leave it counted, and the group's next engine (its id is reused) would be released
     * one rank early. */
    if (rv.failed) return RAD_E_STATE;
    if (!ok) {
        --rv.arrived;
        if (rv.owner != 0) {
            std::fprintf(stderr, "[r4d] all-reduce group timed out at %d of %d ranks\n",
                         rv.arrived + 1, world);
            rv.failed = 1;
            rv.cv.notify_all();
        }
    }
    return ok ? RAD_OK : RAD_E_STATE;
}
