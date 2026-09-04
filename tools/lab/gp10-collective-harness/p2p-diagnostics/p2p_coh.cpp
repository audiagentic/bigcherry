// Does a kernel on the DESTINATION device observe a peer's PCIe write?
// (host readback already proven to work; this tests device-side visibility)
#include <hip/hip_runtime.h>
#include <cstdio>

#define CHK(c) do { hipError_t r=(c); if(r!=hipSuccess) printf("  !! %d: %s\n", __LINE__, hipGetErrorString(r)); } while(0)

__global__ void k_write_plain(int * dst, int v)  { dst[0] = v; }
__global__ void k_write_sys(int * dst, int v) {
    __hip_atomic_store(dst, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__global__ void k_read_plain(int * out, const int * src)    { out[0] = src[0]; }
__global__ void k_read_vol(int * out, const int * src)      { out[0] = *(const volatile int *) src; }
__global__ void k_read_sys(int * out, const int * src) {
    out[0] = __hip_atomic_load(src, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

int main() {
    const int D0 = 0, D1 = 1;
    CHK(hipSetDevice(D0)); CHK(hipDeviceEnablePeerAccess(D1, 0));
    CHK(hipSetDevice(D1)); CHK(hipDeviceEnablePeerAccess(D0, 0));

    CHK(hipSetDevice(D1));
    int *flag = nullptr, *out1 = nullptr;
    CHK(hipMalloc(&flag, sizeof(int)));
    CHK(hipMalloc(&out1, sizeof(int)));

    auto reset = [&](int writer_sys) {
        CHK(hipSetDevice(D1));
        CHK(hipMemset(flag, 0, sizeof(int)));
        CHK(hipMemset(out1, 0xFF, sizeof(int)));
        CHK(hipDeviceSynchronize());
        // touch flag from a dev1 kernel first, so dev1's L2 caches the line
        // (this is what makes the spin-wait case realistic)
        k_read_plain<<<1,1>>>(out1, flag);
        CHK(hipDeviceSynchronize());
        // now dev0 writes across PCIe
        CHK(hipSetDevice(D0));
        if (writer_sys) k_write_sys<<<1,1>>>(flag, 42);
        else            k_write_plain<<<1,1>>>(flag, 42);
        CHK(hipDeviceSynchronize());
    };
    auto host_sees = [&]() { int v=-1; CHK(hipSetDevice(D1)); CHK(hipMemcpy(&v, flag, 4, hipMemcpyDeviceToHost)); return v; };
    auto dev_sees = [&](int which) {
        int v=-1; CHK(hipSetDevice(D1));
        if (which==0) k_read_plain<<<1,1>>>(out1, flag);
        if (which==1) k_read_vol<<<1,1>>>(out1, flag);
        if (which==2) k_read_sys<<<1,1>>>(out1, flag);
        CHK(hipDeviceSynchronize());
        CHK(hipMemcpy(&v, out1, 4, hipMemcpyDeviceToHost));
        return v;
    };

    const char * rd[] = {"plain", "volatile", "sys-acquire"};
    for (int w = 0; w < 2; ++w) {
        printf("\n== peer writer: %s ==\n", w ? "sys-release atomic" : "plain store");
        for (int r = 0; r < 3; ++r) {
            reset(w);
            int hv = host_sees();
            int dv = dev_sees(r);
            printf("  dev1 %-12s read -> %-6d (host readback -> %d) %s\n",
                   rd[r], dv, hv, dv == 42 ? "VISIBLE" : "*** NOT VISIBLE ***");
        }
    }
    return 0;
}
