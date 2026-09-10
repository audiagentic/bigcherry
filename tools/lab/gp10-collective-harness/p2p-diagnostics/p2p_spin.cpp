// Can ANY in-kernel load observe a peer write made by a concurrently running
// kernel? Tries several load flavours against the same live-writer scenario.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHK(c) do { hipError_t r=(c); if(r!=hipSuccess) printf("  !! %d: %s\n", __LINE__, hipGetErrorString(r)); } while(0)

enum { L_ACQ = 0, L_RMW = 1, L_NT = 2, L_VOL = 3, L_INV = 4 };

__global__ void k_spin(int * flag, int token, long long budget,
                       int * result, long long * iters, int kind) {
    long long i = 0; int seen = -12345;
    for (; i < budget; ++i) {
        switch (kind) {
        case L_ACQ: seen = __hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM); break;
        // atomic RMW must go to memory, cannot be served by a stale cache line
        case L_RMW: seen = __hip_atomic_fetch_add(flag, 0, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_SYSTEM); break;
        case L_NT:  seen = __builtin_nontemporal_load(flag); break;
        case L_VOL: seen = *(volatile int *) flag; break;
        // explicit system-scope fence each iteration, then plain load
        case L_INV: __threadfence_system(); seen = *(volatile int *) flag; break;
        }
        if (seen == token) break;
    }
    *result = seen; *iters = i;
}

__global__ void k_write_then_linger(int * peer_flag, int token, long long linger) {
    __hip_atomic_store(peer_flag, token, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    __threadfence_system();
    for (volatile long long i = 0; i < linger; ++i) { }
}

int main() {
    const int D0 = 0, D1 = 1;
    CHK(hipSetDevice(D0)); CHK(hipDeviceEnablePeerAccess(D1, 0));
    CHK(hipSetDevice(D1)); CHK(hipDeviceEnablePeerAccess(D0, 0));

    const char * names[] = {"sys-acquire", "atomic-RMW", "nontemporal", "volatile", "fence+vol"};
    const long long budget = 2000000LL;
    const int token = 7;

    for (int kind = 0; kind <= L_INV; ++kind) {
        CHK(hipSetDevice(D1));
        int *flag=nullptr,*res=nullptr; long long *its=nullptr;
        CHK(hipMalloc(&flag,4)); CHK(hipMalloc(&res,4)); CHK(hipMalloc(&its,8));
        CHK(hipMemset(flag,0,4)); CHK(hipMemset(res,0,4)); CHK(hipMemset(its,0,8));
        CHK(hipDeviceSynchronize());

        hipStream_t s1; CHK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
        k_spin<<<1,1,0,s1>>>(flag, token, budget, res, its, kind);

        CHK(hipSetDevice(D0));
        hipStream_t s0; CHK(hipStreamCreateWithFlags(&s0, hipStreamNonBlocking));
        k_write_then_linger<<<1,1,0,s0>>>(flag, token, 200000000LL);

        CHK(hipSetDevice(D1)); CHK(hipStreamSynchronize(s1));
        int r=0; long long i=0;
        CHK(hipMemcpy(&r,res,4,hipMemcpyDeviceToHost));
        CHK(hipMemcpy(&i,its,8,hipMemcpyDeviceToHost));
        printf("  %-12s saw=%-8d iters=%-10lld %s\n", names[kind], r, i,
               r == token ? "*** SAW IT (live visibility works) ***" : "never saw it");
        CHK(hipSetDevice(D0)); CHK(hipStreamSynchronize(s0));
        CHK(hipSetDevice(D1)); CHK(hipFree(flag)); CHK(hipFree(res)); CHK(hipFree(its));
    }
    return 0;
}
