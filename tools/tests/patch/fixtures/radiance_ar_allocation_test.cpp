#include "r4d_ar_entry.hip"

#include <cassert>
#include <stdexcept>

namespace {

enum class Failure { allocate, memset, synchronize };
Failure failure;
int fail_at, allocations;
std::map<long, bool> live;

long allocate(long size, bool host) {
    if (++allocations == fail_at && failure == Failure::allocate)
        return 0;
    auto address = reinterpret_cast<long>(std::malloc(size));
    assert(address && live.emplace(address, host).second);
    return address;
}

void deallocate(long address, bool host) {
    if (!address) return;
    const auto it = live.find(address);
    assert(it != live.end() && it->second == host);
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
void r4d_ar_ipc_enable_peer(long) { assert(false); }
long r4d_ar_coded2_bytes(long n16) { return 16 * n16; }
int r4d_ar_max_blocks() { return 8; }
void r4d_ar_wide_dims(int*, int*, int*) { assert(false); }

int main() {
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
                assert(allocations == fail_at && instance == nullptr && live.empty());
                assert(g_rvs.empty() && g_cache.empty());
            }
        }
    }
}
