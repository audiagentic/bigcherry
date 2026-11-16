// duplex.hip: can one card move bytes to and from pinned host memory at once? (VTP36)
// Kernel stores (card -> host), kernel loads (host -> card) and hipMemcpyAsync (the copy engines), alone and in pairs.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

__global__ void copy16(const int4* __restrict__ src, int4* __restrict__ dst, int n16) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  const int w = (n16 + nb - 1) / nb, start = b * w;
  int end = start + w; if (end > n16) end = n16;
  for (int k = start + tid; k < end; k += nt) dst[k] = src[k];
  __threadfence_system();
}

// the same loop without restrict: may the compiler's doubt about aliasing cost throughput?
__global__ void copy16_na(const int4* src, int4* dst, int n16) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  const int w = (n16 + nb - 1) / nb, start = b * w;
  int end = start + w; if (end > n16) end = n16;
  for (int k = start + tid; k < end; k += nt) dst[k] = src[k];
  __threadfence_system();
}
// two interleaved copies in one loop, as a fused first hop + assemble: card -> host and host -> card
__global__ void copy16_x2(const int4* __restrict__ s_dev, int4* __restrict__ d_host,
                          const int4* __restrict__ s_host, int4* __restrict__ d_dev, int n16) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  const int w = (n16 + nb - 1) / nb, start = b * w;
  int end = start + w; if (end > n16) end = n16;
  for (int k = start + tid; k < end; k += nt) {
    const int4 vin = s_host[k];
    d_host[k] = s_dev[k];
    d_dev[k] = vin;
  }
  __threadfence_system();
}
__global__ void copy16_x2_na(const int4* s_dev, int4* d_host, const int4* s_host, int4* d_dev, int n16) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  const int w = (n16 + nb - 1) / nb, start = b * w;
  int end = start + w; if (end > n16) end = n16;
  for (int k = start + tid; k < end; k += nt) {
    const int4 vin = s_host[k];
    d_host[k] = s_dev[k];
    d_dev[k] = vin;
  }
  __threadfence_system();
}

#define OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
  const int dev = argc > 1 ? std::atoi(argv[1]) : 0;
  const size_t mib = argc > 2 ? std::atol(argv[2]) : 32;
  const int nb = argc > 3 ? std::atoi(argv[3]) : 4, nt = argc > 4 ? std::atoi(argv[4]) : 256;
  const size_t N = mib << 20; const int n16 = (int)(N / 16);
  OK(hipSetDevice(dev));
  char bus[64] = {0}; OK(hipDeviceGetPCIBusId(bus, sizeof bus, dev));
  hipDeviceProp_t pr; OK(hipGetDeviceProperties(&pr, dev));
  void *hout, *hin, *hout2, *hin2; int4 *d1, *d2, *d3, *d4, *dho, *dhi, *dho2, *dhi2;
  const unsigned fl = hipHostMallocCoherent | hipHostMallocMapped | hipHostMallocPortable;
  OK(hipHostMalloc(&hout, N, fl)); OK(hipHostMalloc(&hin, N, fl)); OK(hipHostMalloc(&hout2, N, fl)); OK(hipHostMalloc(&hin2, N, fl));
  std::memset(hout2, 5, N); std::memset(hin2, 6, N);
  std::memset(hout, 1, N); std::memset(hin, 2, N);
  OK(hipMalloc((void**)&d1, N)); OK(hipMalloc((void**)&d2, N)); OK(hipMalloc((void**)&d3, N)); OK(hipMalloc((void**)&d4, N));
  OK(hipMemset(d3, 7, N)); OK(hipMemset(d4, 8, N));
  OK(hipMemset(d1, 3, N)); OK(hipMemset(d2, 4, N));
  OK(hipHostGetDevicePointer((void**)&dho, hout, 0)); OK(hipHostGetDevicePointer((void**)&dhi, hin, 0));
  OK(hipHostGetDevicePointer((void**)&dho2, hout2, 0)); OK(hipHostGetDevicePointer((void**)&dhi2, hin2, 0));
  hipStream_t s1, s2; OK(hipStreamCreate(&s1)); OK(hipStreamCreate(&s2));
  auto kpush = [&](hipStream_t s) { copy16<<<nb, nt, 0, s>>>(d1, dho, n16); };          // card -> host, shader
  auto kpull = [&](hipStream_t s) { copy16<<<nb, nt, 0, s>>>(dhi, d2, n16); };          // host -> card, shader
  auto mout  = [&](hipStream_t s) { OK(hipMemcpyAsync(hout, d1, N, hipMemcpyDeviceToHost, s)); };  // copy engine
  auto min_  = [&](hipStream_t s) { OK(hipMemcpyAsync(d2, hin, N, hipMemcpyHostToDevice, s)); };
  struct Op { const char* name; int a, b; };   // 0 kpush 1 kpull 2 mout 3 min ; -1 none
  auto kpush2 = [&](hipStream_t s) { copy16<<<nb, nt, 0, s>>>(d3, dho2, n16); };        // a second, distinct pair
  auto kpull2 = [&](hipStream_t s) { copy16<<<nb, nt, 0, s>>>(dhi2, d4, n16); };
  auto kboth  = [&](hipStream_t s) { copy16<<<nb, nt, 0, s>>>(dhi, dho, n16); };        // host -> host in one kernel
  auto run = [&](int op, hipStream_t s) { if (op == 0) kpush(s); else if (op == 1) kpull(s); else if (op == 2) mout(s); else if (op == 3) min_(s); else if (op == 4) kpush2(s); else if (op == 5) kpull2(s); else if (op == 6) kboth(s);
    else if (op == 7) copy16_na<<<nb, nt, 0, s>>>(dhi, dho, n16);
    else if (op == 8) copy16_x2<<<nb, nt, 0, s>>>(d1, dho, dhi, d2, n16);
    else if (op == 9) copy16_x2_na<<<nb, nt, 0, s>>>(d1, dho, dhi, d2, n16); };
  const Op ops[] = {
    {"kernel out alone", 0, -1}, {"kernel in alone", 1, -1}, {"memcpy out alone", 2, -1}, {"memcpy in alone", 3, -1},
    {"kernel out + kernel in", 0, 1}, {"memcpy out + kernel in", 2, 1}, {"kernel out + memcpy in", 0, 3},
    {"memcpy out + memcpy in", 2, 3}, {"kernel out + kernel out", 0, 4}, {"kernel in + kernel in", 1, 5},
    {"one kernel, in and out", 6, -1}, {"the same, no restrict", 7, -1},
    {"one kernel, out and in fused", 8, -1}, {"the same, no restrict", 9, -1},
  };
  std::printf("device %d %s (%s), %zu MiB a transfer, kernel grid %d x %d\n", dev, bus, pr.name, mib, nb, nt);
  for (const Op& o : ops) {
    std::vector<double> t;
    for (int it = 0; it < 14; ++it) {
      OK(hipStreamSynchronize(s1)); OK(hipStreamSynchronize(s2));
      const double t0 = now();
      run(o.a, s1); if (o.b >= 0) run(o.b, s2);
      OK(hipStreamSynchronize(s1)); OK(hipStreamSynchronize(s2));
      if (it >= 2) t.push_back(now() - t0);
    }
    std::sort(t.begin(), t.end());
    const double med = t[t.size() / 2];
    const double bytes = (double)N * ((o.b >= 0 || o.a >= 6) ? 2 : 1);
    std::printf("  %-26s %8.0f us  %6.2f GB/s total\n", o.name, med * 1e6, bytes / med / 1e9);
  }
  return 0;
}
