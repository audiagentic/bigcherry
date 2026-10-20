#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>
__global__ void touch(char* p, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) p[i] += 1; }
__global__ void rd(const uint4* p, size_t n, uint4* o) { uint4 a = {0,0,0,0}; for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { uint4 v = p[i]; a.x ^= v.x; a.y ^= v.y; } if (a.x == 12345) *o = a; }
int main() {
  hipSetDevice(0);
  for (size_t kb : {5, 30, 300}) {
    size_t n = kb * 1024; char *h, *d; hipHostMalloc(&h, n); hipMalloc(&d, n);
    for (int w = 0; w < 20; ++w) { hipMemcpy(d, h, n, hipMemcpyHostToDevice); touch<<<(n+255)/256,256>>>(d, n); hipMemcpy(h, d, n, hipMemcpyDeviceToHost); }
    const int it = 200; auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < it; ++k) { hipMemcpy(d, h, n, hipMemcpyHostToDevice); touch<<<(n+255)/256,256>>>(d, n); hipMemcpy(h, d, n, hipMemcpyDeviceToHost); }
    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
    std::printf("round trip %zu KB (H2D + kernel + D2H): %.1f us\n", kb, us);
    hipFree(d); hipHostFree(h);
  }
  size_t n = (size_t)1 << 30; uint4* d; uint4* o; hipMalloc(&d, n); hipMalloc(&o, 16); hipMemset(d, 1, n);
  hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
  rd<<<4096,256>>>(d, n / 16, o); hipEventRecord(a); for (int k = 0; k < 10; ++k) rd<<<4096,256>>>(d, n / 16, o); hipEventRecord(b); hipEventSynchronize(b);
  float ms; hipEventElapsedTime(&ms, a, b); std::printf("read bandwidth 1 GiB x10: %.0f GB/s\n", 10.0 * n / (ms * 1e6));
  return 0;
}
