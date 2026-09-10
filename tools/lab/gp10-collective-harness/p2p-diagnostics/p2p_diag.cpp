// GP11 deep-dive: characterize exactly what does and does not work for
// kernel-side peer access on the dual-XTX pair.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHK(call)                                                              \
    do {                                                                       \
        hipError_t _rc = (call);                                               \
        if (_rc != hipSuccess) {                                               \
            printf("  !! HIP error at %d: %s (%s)\n", __LINE__,                \
                   hipGetErrorString(_rc), #call);                             \
        }                                                                      \
    } while (0)

__global__ void k_read(float * out, const float * src)  { out[0] = src[0]; }
__global__ void k_write(float * dst, float v)           { dst[0] = v; }
__global__ void k_read_volatile(float * out, const float * src) {
    out[0] = *(const volatile float *) src;
}

static void dump_attrs(const char * tag, const void * p) {
    hipPointerAttribute_t a;
    hipError_t rc = hipPointerGetAttributes(&a, p);
    if (rc != hipSuccess) {
        printf("  %s: hipPointerGetAttributes FAILED: %s\n", tag, hipGetErrorString(rc));
        return;
    }
    printf("  %s: ptr=%p device=%d type=%d devicePointer=%p hostPointer=%p\n",
           tag, p, a.device, (int) a.type, a.devicePointer, a.hostPointer);
}

int main() {
    const int D0 = 0, D1 = 1;

    int can01 = 0, can10 = 0;
    CHK(hipDeviceCanAccessPeer(&can01, D0, D1));
    CHK(hipDeviceCanAccessPeer(&can10, D1, D0));
    printf("canAccessPeer: %d->%d = %d, %d->%d = %d\n", D0, D1, can01, D1, D0, can10);

    CHK(hipSetDevice(D0));
    hipError_t e0 = hipDeviceEnablePeerAccess(D1, 0);
    printf("enablePeerAccess %d->%d: %s\n", D0, D1, hipGetErrorString(e0));
    CHK(hipSetDevice(D1));
    hipError_t e1 = hipDeviceEnablePeerAccess(D0, 0);
    printf("enablePeerAccess %d->%d: %s\n", D1, D0, hipGetErrorString(e1));

    // buffers
    CHK(hipSetDevice(D0));
    float *b0 = nullptr, *out0 = nullptr;
    CHK(hipMalloc(&b0, sizeof(float)));
    CHK(hipMalloc(&out0, sizeof(float)));
    CHK(hipSetDevice(D1));
    float *b1 = nullptr, *out1 = nullptr;
    CHK(hipMalloc(&b1, sizeof(float)));
    CHK(hipMalloc(&out1, sizeof(float)));

    printf("\n-- pointer attributes --\n");
    CHK(hipSetDevice(D0));
    dump_attrs("b0 (own, from dev0 ctx)", b0);
    dump_attrs("b1 (peer, from dev0 ctx)", b1);
    CHK(hipSetDevice(D1));
    dump_attrs("b1 (own, from dev1 ctx)", b1);
    dump_attrs("b0 (peer, from dev1 ctx)", b0);

    auto set_val = [&](int dev, float * p, float v) {
        CHK(hipSetDevice(dev));
        CHK(hipMemcpy(p, &v, sizeof(float), hipMemcpyHostToDevice));
        CHK(hipDeviceSynchronize());
    };
    auto get_val = [&](int dev, float * p) {
        float v = -999.0f;
        CHK(hipSetDevice(dev));
        CHK(hipMemcpy(&v, p, sizeof(float), hipMemcpyDeviceToHost));
        return v;
    };

    printf("\n-- CONTROL: dev0 kernel reads dev0 LOCAL buffer --\n");
    set_val(D0, b0, 11.0f);
    set_val(D0, out0, 0.0f);
    CHK(hipSetDevice(D0));
    k_read<<<1, 1>>>(out0, b0);
    CHK(hipDeviceSynchronize());
    printf("  wrote 11.0 to b0, kernel read -> %.3f  %s\n", get_val(D0, out0),
           get_val(D0, out0) == 11.0f ? "OK" : "*** FAIL (harness bug!) ***");

    printf("\n-- TEST: dev0 kernel reads dev1 PEER buffer --\n");
    set_val(D1, b1, 22.0f);
    set_val(D0, out0, -1.0f);
    CHK(hipSetDevice(D0));
    k_read<<<1, 1>>>(out0, b1);
    CHK(hipDeviceSynchronize());
    printf("  wrote 22.0 to b1(dev1), dev0 kernel read -> %.3f\n", get_val(D0, out0));

    printf("\n-- TEST: same, volatile load --\n");
    set_val(D0, out0, -1.0f);
    CHK(hipSetDevice(D0));
    k_read_volatile<<<1, 1>>>(out0, b1);
    CHK(hipDeviceSynchronize());
    printf("  dev0 volatile peer read -> %.3f\n", get_val(D0, out0));

    printf("\n-- TEST: reverse direction, dev1 kernel reads dev0 PEER buffer --\n");
    set_val(D0, b0, 33.0f);
    set_val(D1, out1, -1.0f);
    CHK(hipSetDevice(D1));
    k_read<<<1, 1>>>(out1, b0);
    CHK(hipDeviceSynchronize());
    printf("  wrote 33.0 to b0(dev0), dev1 kernel read -> %.3f\n", get_val(D1, out1));

    printf("\n-- TEST: dev0 kernel WRITES to dev1 peer buffer --\n");
    set_val(D1, b1, 0.0f);
    CHK(hipSetDevice(D0));
    k_write<<<1, 1>>>(b1, 44.0f);
    CHK(hipDeviceSynchronize());
    CHK(hipSetDevice(D1));
    CHK(hipDeviceSynchronize());
    printf("  dev0 kernel wrote 44.0 to b1(dev1); dev1 reads back -> %.3f\n", get_val(D1, b1));

    printf("\n-- TEST: hipMemcpyPeer (DMA path) with the SAME pointers --\n");
    set_val(D1, b1, 55.0f);
    set_val(D0, out0, -1.0f);
    CHK(hipMemcpyPeer(out0, D0, b1, D1, sizeof(float)));
    CHK(hipDeviceSynchronize());
    printf("  hipMemcpyPeer dev1->dev0 -> %.3f\n", get_val(D0, out0));

    printf("\n-- TEST: hipMemcpy(hipMemcpyDeviceToDevice) with the SAME pointers --\n");
    set_val(D1, b1, 66.0f);
    set_val(D0, out0, -1.0f);
    CHK(hipSetDevice(D0));
    CHK(hipMemcpy(out0, b1, sizeof(float), hipMemcpyDeviceToDevice));
    CHK(hipDeviceSynchronize());
    printf("  hipMemcpy D2D dev1->dev0 -> %.3f\n", get_val(D0, out0));

    return 0;
}
