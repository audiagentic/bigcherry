#include "r4d_ar_entry.hip"
#include "bc_shared_slot.h"

#include <cassert>
#include <stdexcept>
#include <thread>

thread_local int fake_hip_device = 0;

namespace {
thread_local int owner_id = 0;
std::mutex gate_mu;
std::condition_variable gate_cv;
bool fail_peer = false;
int fail_device = 1;   // the rank (device = rank + 1) whose peer enabling faults; the other holds until released
bool held_peer = false;
bool release_peer = false;

enum class Failure { allocate, memset, synchronize };
Failure failure;
int fail_at;
thread_local int allocations;
std::mutex live_mu;
struct Allocation { bool host; int device; };
std::map<long, Allocation> live;

size_t live_count() {
    std::lock_guard<std::mutex> lock(live_mu);
    return live.size();
}

long allocate(long size, bool host) {
    if (++allocations == fail_at && failure == Failure::allocate)
        return 0;
    auto address = reinterpret_cast<long>(std::malloc(size));
    std::lock_guard<std::mutex> lock(live_mu);
    assert(address && live.emplace(address, Allocation{host, fake_hip_device}).second);
    return address;
}

void deallocate(long address, bool host) {
    if (!address) return;
    std::lock_guard<std::mutex> lock(live_mu);
    const auto it = live.find(address);
    assert(it != live.end() && it->second.host == host && it->second.device == fake_hip_device);
    live.erase(it);
    std::free(reinterpret_cast<void*>(address));
}

using hipIpcMemHandle_t = char;
constexpr int hipDeviceMallocFinegrained = 1;
int hipMalloc(void** p, long size) {
    *p = reinterpret_cast<void*>(allocate(size, false));
    return *p ? hipSuccess : 1;
}
int hipExtMallocWithFlags(void** p, long size, int) { return hipMalloc(p, size); }
int hipMemset(void*, int, long) {
    return allocations == fail_at && failure == Failure::memset ? 1 : hipSuccess;
}
int hipStreamSynchronize(hipStream_t) {
    return allocations == fail_at && failure == Failure::synchronize ? 1 : hipSuccess;
}
int hipFree(void* p) {
    deallocate(reinterpret_cast<long>(p), false);
    return hipSuccess;
}
int hipIpcGetMemHandle(hipIpcMemHandle_t*, void*) { return 1; }
int hipGetLastError() { return hipSuccess; }

}  // namespace

#define HIPOK(expr) do { if ((expr) != hipSuccess) throw std::runtime_error(#expr); } while (0)
#include "ar_ipc_alloc.inc"
#undef HIPOK

long r4d_ar_host_alloc(long size) {
    const long address = allocate(size, true);
    if (!address) throw std::runtime_error("host allocation fault");
    return address;
}
void r4d_ar_host_free(long address) { deallocate(address, true); }
void r4d_ar_ipc_free(long address) { deallocate(address, false); }
void r4d_ar_ipc_enable_peer(long) {
    if (!fail_peer) return;
    std::unique_lock<std::mutex> lock(gate_mu);
    if (fake_hip_device == fail_device) {
        gate_cv.wait(lock, [] { return held_peer; });
        throw std::runtime_error("peer enable fault");
    }
    held_peer = true;
    gate_cv.notify_all();
    gate_cv.wait(lock, [] { return release_peer; });
}
long r4d_ar_coded2_bytes(long n16) { return 16 * n16; }
int r4d_ar_max_blocks() { return 8; }
void r4d_ar_wide_dims(int*, int*, int*) { assert(false); }

namespace rad {
int engine_owner() { return owner_id; }
}  // namespace rad

void test_unpublished() {
    const RadParam params[] = { RAD_STR("dtype", "bf16"), RAD_INT("numel", 128),
                               RAD_INT("tp_group", 42) };
    for (bool host : { false, true }) {
        const auto init = host ? r4d_rad_ar_host_init : r4d_rad_ar_init;
        for (auto mode : { Failure::allocate, Failure::memset, Failure::synchronize }) {
            failure = mode;
            // Host scratch and flags are pinned buffers, bypassing the IPC initialization calls.
            const int first = host && mode != Failure::allocate ? 3 : 1;
            const int last = host ? 5 : 4;
            for (fail_at = first; fail_at <= last; ++fail_at) {
                allocations = 0;
                void* instance = nullptr;
                assert(init(params, 3, 0, 2, &instance) == RAD_E_DEVICE);
                assert(allocations == fail_at && instance == nullptr && live_count() == 0);
                assert(g_rvs.empty() && g_cache.empty());
            }
        }
    }
}

// `bad` is the rank whose publication fails after both ranks published their peer addresses; `good` is the peer that
// still holds them. Group, owners and device ordinals are per case so a failed slot cannot leak into the next case.
void test_published(int group, int owner_a, int owner_b, int owner_c, int bad) {
    const int good = 1 - bad;
    held_peer = false;
    release_peer = false;
    fail_device = bad + 1;
    fail_at = 0;
    fail_peer = true;
    const RadParam params[] = { RAD_STR("dtype", "bf16"), RAD_INT("numel", 128),
                               RAD_INT("tp_group", group) };
    int status[2]{};
    void* instance[2]{};
    auto start_rank = [&](int rank, int owner) {
        return std::thread([&, rank, owner] {
            owner_id = owner;
            fake_hip_device = rank + 1;
            status[rank] = r4d_rad_ar_init(params, 3, rank, 2, &instance[rank]);
        });
    };
    auto first = start_rank(0, owner_a);
    auto second = start_rank(1, owner_a);
    auto& failing = bad == 0 ? first : second;
    auto& holding = bad == 0 ? second : first;
    failing.join();
    assert(status[bad] == RAD_E_DEVICE && !instance[bad]);
    assert(live_count() == 8);  // The other rank still holds the published peer address.

    int retired = 0;
    void* other_slot = nullptr;
    owner_id = owner_b;
    assert(rad_bc_shared_slot("test.other-owner", sizeof(int), alignof(int), nullptr,
                              &other_slot, nullptr, 0) == RAD_OK);
    assert(rad_bc_defer_rank_cleanup(other_slot, &retired, [](void* ptr) {
        ++*static_cast<int*>(ptr);
    }) == RAD_OK);
    owner_id = 0;
    rad::bc_shared_slot_retire_owner(owner_b);
    assert(retired == 1 && live_count() == 8);

    {
        std::lock_guard<std::mutex> lock(gate_mu);
        release_peer = true;
    }
    gate_cv.notify_all();
    holding.join();
    assert(status[good] == RAD_E_STATE && !instance[good]);
    assert(live_count() == 8);
    // The failed slot is not reused: another init in the same group by the same engine is refused with no allocation
    // and no change to what the peers still hold.
    {
        void* again = nullptr;
        owner_id = owner_a;
        fake_hip_device = 5;
        assert(r4d_rad_ar_init(params, 3, 0, 2, &again) == RAD_E_STATE && !again && live_count() == 8);
        owner_id = 0;
        fake_hip_device = 0;
    }
    rad::bc_shared_slot_retire_owner(owner_a);
    assert(live_count() == 0 && fake_hip_device == 0);
    rad::bc_shared_slot_retire_owner(owner_a);
    assert(live_count() == 0);

    // A replacement engine can reuse the group and its ranks. Its successful instances
    // transfer to the ordinary fini path with no outstanding retirement callback.
    fail_peer = false;
    auto replacement_first = start_rank(0, owner_c);
    auto replacement_second = start_rank(1, owner_c);
    replacement_first.join();
    replacement_second.join();
    assert(status[0] == RAD_OK && status[1] == RAD_OK);
    assert(instance[0] && instance[1] && live_count() == 8);
    for (int rank = 0; rank < 2; ++rank) {
        fake_hip_device = rank + 1;
        r4d_rad_ar_fini(instance[rank]);
    }
    rad::bc_shared_slot_retire_owner(owner_c);
    assert(live_count() == 0);
}

// Capacity reuse must not borrow another engine group's scratch, even when it fits.
void test_capacity_reuse() {
    assert(g_cache.empty());
    ArInst foreign{}, small{}, matching{};
    g_cache.push_back(CacheSlot{CacheKey{101, 0, 2, 0, 256, 0}, &foreign, 1});
    g_cache.push_back(CacheSlot{CacheKey{102, 0, 2, 0, 64, 0}, &small, 1});
    g_cache.push_back(CacheSlot{CacheKey{102, 0, 2, 0, 256, 0}, &matching, 1});
    ArInst* out = nullptr;
    assert(acquire_at_least(CacheKey{102, 0, 2, 0, 128, 0}, &out) == RAD_OK);
    assert(out == &matching && g_cache[0].refs == 1 && g_cache[1].refs == 1 && g_cache[2].refs == 2);
    g_cache.clear();
}

void test_native_traffic() {
    ArInst in{};
    in.rank = 1;
    in.world = 3;
    in.esize = 2;
    traffic_all(&in, 32, kTrOneShot);
    assert(g_ar_sent[1][0].load() == 64 && g_ar_sent[1][2].load() == 64);
    assert(g_ar_sent[1][1].load() == 0);
    assert(g_ar_route_calls[1][kTrOneShot].load() == 1);
    assert(g_ar_route_bytes[1][kTrOneShot].load() == 64);
}

int main() {
    test_capacity_reuse();
    test_native_traffic();
    test_unpublished();
    test_published(55, 91, 92, 93, 0);   // the first rank fails, the second holds its published addresses
    test_published(56, 94, 95, 96, 1);   // the second rank fails after the first published
}
