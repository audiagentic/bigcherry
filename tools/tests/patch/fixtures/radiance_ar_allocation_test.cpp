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
    if (fake_hip_device == 1) {
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

void test_published() {
    fail_at = 0;
    fail_peer = true;
    const RadParam params[] = { RAD_STR("dtype", "bf16"), RAD_INT("numel", 128),
                               RAD_INT("tp_group", 55) };
    int status[2]{};
    void* instance[2]{};
    auto start_rank = [&](int rank, int owner) {
        return std::thread([&, rank, owner] {
            owner_id = owner;
            fake_hip_device = rank + 1;
            status[rank] = r4d_rad_ar_init(params, 3, rank, 2, &instance[rank]);
        });
    };
    auto first = start_rank(0, 91);
    auto second = start_rank(1, 91);
    first.join();
    assert(status[0] == RAD_E_DEVICE && !instance[0]);
    assert(live_count() == 8);  // The other rank still holds the published peer address.

    int retired = 0;
    void* other_slot = nullptr;
    owner_id = 92;
    assert(rad_bc_shared_slot("test.other-owner", sizeof(int), alignof(int), nullptr,
                              &other_slot, nullptr, 0) == RAD_OK);
    assert(rad_bc_defer_rank_cleanup(other_slot, &retired, [](void* ptr) {
        ++*static_cast<int*>(ptr);
    }) == RAD_OK);
    owner_id = 0;
    rad::bc_shared_slot_retire_owner(92);
    assert(retired == 1 && live_count() == 8);

    {
        std::lock_guard<std::mutex> lock(gate_mu);
        release_peer = true;
    }
    gate_cv.notify_all();
    second.join();
    assert(status[1] == RAD_E_STATE && !instance[1]);
    assert(live_count() == 8);
    rad::bc_shared_slot_retire_owner(91);
    assert(live_count() == 0 && fake_hip_device == 0);
    rad::bc_shared_slot_retire_owner(91);
    assert(live_count() == 0);

    // A replacement engine can reuse the group and its ranks. Its successful instances
    // transfer to the ordinary fini path with no outstanding retirement callback.
    fail_peer = false;
    auto replacement_first = start_rank(0, 93);
    auto replacement_second = start_rank(1, 93);
    replacement_first.join();
    replacement_second.join();
    assert(status[0] == RAD_OK && status[1] == RAD_OK);
    assert(instance[0] && instance[1] && live_count() == 8);
    for (int rank = 0; rank < 2; ++rank) {
        fake_hip_device = rank + 1;
        r4d_rad_ar_fini(instance[rank]);
    }
    rad::bc_shared_slot_retire_owner(93);
    assert(live_count() == 0);
}

int main() {
    test_unpublished();
    test_published();
}
