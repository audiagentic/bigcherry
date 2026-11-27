// Minimal independent HIP wrong-architecture launch reproducer.
// Build for gfx1201 only; on gfx1100 it MUST return an error, not SIGSEGV.
// Intentionally no BigCherry headers and no production libraries.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void write_flag(int* p) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *p = 42;
}

int main() {
    int count = -1;
    hipError_t r = hipGetDeviceCount(&count);
    if (r != hipSuccess) {
        std::fprintf(stderr, "hipGetDeviceCount: %s (%d)\n", hipGetErrorString(r), int(r));
        return 3;
    }
    hipDeviceProp_t props{};
    r = hipGetDeviceProperties(&props, 0);
    if (r != hipSuccess) {
        std::fprintf(stderr, "hipGetDeviceProperties: %s (%d)\n", hipGetErrorString(r), int(r));
        return 4;
    }
    std::printf("device_count=%d device0=%s arch=%s\n", count, props.name, props.gcnArchName);
    std::fflush(stdout);
    int* p = nullptr;
    r = hipMalloc(&p, sizeof(int));
    if (r != hipSuccess) {
        std::fprintf(stderr, "hipMalloc: %s (%d)\n", hipGetErrorString(r), int(r));
        return 5;
    }
    std::puts("before_launch");
    std::fflush(stdout);
    hipLaunchKernelGGL(write_flag, dim3(1), dim3(64), 0, 0, p);
    r = hipGetLastError();
    std::printf("launch=%s (%d)\n", hipGetErrorString(r), int(r));
    std::fflush(stdout);
    if (r != hipSuccess) {
        hipFree(p);
        return 10;
    }
    r = hipDeviceSynchronize();
    std::printf("sync=%s (%d)\n", hipGetErrorString(r), int(r));
    if (r == hipSuccess) {
        int value = 0;
        r = hipMemcpy(&value, p, sizeof(int), hipMemcpyDeviceToHost);
        std::printf("readback=%d copy=%s (%d)\n", value, hipGetErrorString(r), int(r));
        hipFree(p);
        return r == hipSuccess && value == 42 ? 0 : 11;
    }
    hipFree(p);
    return 12;
}
