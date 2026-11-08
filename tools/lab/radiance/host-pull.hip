// RAD49: a kernel streaming reads of mapped, coherent pinned host memory until SIGINT/SIGTERM, as a remote-MoE helper reads
// activations. Run beside a measurement on another card to see what a concurrent host pull costs it (RAD40: one XTX pulling
// halves the other CPU-attached cards). Build: hipcc -O3 --offload-arch=<arch> host-pull.hip -o host-pull
#include <hip/hip_runtime.h>
#include <cstdio>
#include <csignal>
static volatile sig_atomic_t stop = 0;
__global__ void rd(const uint4* p, size_t n, unsigned* out) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, s = (size_t)gridDim.x * blockDim.x; unsigned a = 0;
  for (; i < n; i += s) { uint4 v = p[i]; a ^= v.x ^ v.y ^ v.z ^ v.w; }
  if (a == 0x12345678u) out[0] = a;
}
int main() {
  signal(SIGINT, [](int) { stop = 1; }); signal(SIGTERM, [](int) { stop = 1; });
  const size_t bytes = 64ull << 20, n = bytes / 16;
  void* h; if (hipHostMalloc(&h, bytes, hipHostMallocMapped | hipHostMallocCoherent) != hipSuccess) return 1;
  void* dh; hipHostGetDevicePointer(&dh, h, 0); unsigned* o; hipMalloc(&o, 4);
  hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b); double gb = 0, ms_all = 0; long k = 0;
  while (!stop) {
    hipEventRecord(a); for (int i = 0; i < 8; ++i) rd<<<1024, 256>>>((const uint4*)dh, n, o); hipEventRecord(b);
    hipEventSynchronize(b); float ms; hipEventElapsedTime(&ms, a, b); gb += 8.0 * bytes / 1e9; ms_all += ms; ++k;
  }
  printf("host-pull: %.1f GB/s over %ld rounds\n", gb / (ms_all / 1e3), k); return 0;
}
