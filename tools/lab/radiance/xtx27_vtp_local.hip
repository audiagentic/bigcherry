// VTP04: GPU-local two-logical-shard scheduling probe (NOT a real GEMM).
// Test fused physical execution against sequential and independent HIP streams.
// Every mode reads exactly the same two device-resident synthetic weight buffers,
// computes the identical UINT32 per-output-lane checksum and is oracle checked.
// No P2P, transport, Q6/Q8 decode or production inference is claimed.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define HC(x) do { const hipError_t err_ = (x); if (err_ != hipSuccess) { \
  std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err_)); \
  std::exit(2); } } while (0)

__global__ void make_codes(uint32_t* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
       i < n; i += gridDim.x * (size_t)blockDim.x)
    p[i] = (uint32_t)i * 2654435761u + seed;
}
__global__ void scan_one(const uint32_t* p, size_t n, uint32_t* out, int lanes) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= lanes) return;
  uint32_t s = 0;
  for (size_t j = (size_t)k; j < n; j += (size_t)lanes) s += p[j];
  out[k] = s;
}
__global__ void scan_fused(const uint32_t* a, const uint32_t* b, size_t n,
                           uint32_t* out, int lanes) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= lanes) return;
  uint32_t sa = 0, sb = 0;
  for (size_t j = (size_t)k; j < n; j += (size_t)lanes) {
    sa += a[j];
    sb += b[j];
  }
  out[k] = sa + sb;
}
__global__ void add_parts(const uint32_t* a, const uint32_t* b, uint32_t* out, int lanes) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < lanes) out[k] = a[k] + b[k];
}

static double quantile(std::vector<double> v, double fraction) {
  std::sort(v.begin(), v.end());
  const size_t i = (size_t)((v.size() - 1) * fraction);
  return v[i];
}
static int arg_int(int argc, char** argv, const char* key, int d) {
  for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], key) == 0) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing ") + key);
    char* end = nullptr;
    const long n = std::strtol(argv[i + 1], &end, 10);
    if (!end || *end || n < 1 || n > 1000000)
      throw std::runtime_error(std::string("invalid ") + key);
    return (int)n;
  }
  return d;
}
static int r9700_device() {
  int count = 0;
  HC(hipGetDeviceCount(&count));
  int id = -1, matches = 0;
  for (int i = 0; i < count; ++i) {
    hipDeviceProp_t p{};
    HC(hipGetDeviceProperties(&p, i));
    if (std::strncmp(p.gcnArchName, "gfx1201", 7) == 0) { id = i; ++matches; }
  }
  if (matches != 1) throw std::runtime_error("require exactly one visible gfx1201 R9700");
  return id;
}
int main(int argc, char** argv) {
  try {
    const int mib = arg_int(argc, argv, "--mib-per-shard", 16);
    const int lanes = arg_int(argc, argv, "--lanes", 32768);
    const int iterations = arg_int(argc, argv, "--iterations", 9);
    bool reverse = false;
    for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], "--reverse") == 0) reverse = true;
    if (mib > 256 || lanes > 262144 || (lanes % 256))
      throw std::runtime_error("bounds: mib-per-shard<=256, lanes<=262144 and multiple of 256");
    const size_t words = (size_t)mib * (1 << 20) / sizeof(uint32_t);
    const int gpu = r9700_device();
    HC(hipSetDevice(gpu));
    hipDeviceProp_t info{}; HC(hipGetDeviceProperties(&info, gpu));
    uint32_t *wa = nullptr, *wb = nullptr, *pa = nullptr, *pb = nullptr, *out = nullptr;
    HC(hipMalloc((void**)&wa, words * 4));
    HC(hipMalloc((void**)&wb, words * 4));
    HC(hipMalloc((void**)&pa, (size_t)lanes * 4));
    HC(hipMalloc((void**)&pb, (size_t)lanes * 4));
    HC(hipMalloc((void**)&out, (size_t)lanes * 4));
    hipStream_t sa, sb, sc;
    HC(hipStreamCreateWithFlags(&sa, hipStreamNonBlocking));
    HC(hipStreamCreateWithFlags(&sb, hipStreamNonBlocking));
    HC(hipStreamCreateWithFlags(&sc, hipStreamNonBlocking));
    hipEvent_t start, doneA, doneB, stop;
    HC(hipEventCreate(&start)); HC(hipEventCreate(&doneA));
    HC(hipEventCreate(&doneB)); HC(hipEventCreate(&stop));
    make_codes<<<512, 256, 0, sc>>>(wa, words, 113);
    make_codes<<<512, 256, 0, sc>>>(wb, words, 269);
    HC(hipGetLastError()); HC(hipStreamSynchronize(sc));
    const int blocks = lanes / 256;

    auto one = [&](int mode) -> std::pair<double, double> {
      const auto before = std::chrono::steady_clock::now();
      HC(hipEventRecord(start, sc));
      HC(hipStreamWaitEvent(sa, start, 0));
      if (mode == 2) HC(hipStreamWaitEvent(sb, start, 0));
      if (mode == 0) {
        scan_fused<<<blocks, 256, 0, sa>>>(wa, wb, words, out, lanes);
      } else {
        scan_one<<<blocks, 256, 0, sa>>>(wa, words, pa, lanes);
        if (mode == 1) scan_one<<<blocks, 256, 0, sa>>>(wb, words, pb, lanes);
        else scan_one<<<blocks, 256, 0, sb>>>(wb, words, pb, lanes);
      }
      HC(hipGetLastError());
      HC(hipEventRecord(doneA, sa));
      HC(hipStreamWaitEvent(sc, doneA, 0));
      if (mode == 2) {
        HC(hipEventRecord(doneB, sb));
        HC(hipStreamWaitEvent(sc, doneB, 0));
      }
      if (mode != 0) {
        add_parts<<<blocks, 256, 0, sc>>>(pa, pb, out, lanes);
        HC(hipGetLastError());
      }
      HC(hipEventRecord(stop, sc));
      HC(hipEventSynchronize(stop));
      const auto after = std::chrono::steady_clock::now();
      float ms = 0.0f;
      HC(hipEventElapsedTime(&ms, start, stop));
      return {ms * 1000.0, std::chrono::duration<double, std::micro>(after - before).count()};
    };
    std::vector<uint32_t> golden, check((size_t)lanes);
    std::printf("gpu=%d arch=%s mib_per_shard=%d lanes=%d words=%zu\n",
                gpu, info.gcnArchName, mib, lanes, words);
    std::puts("mode,gpu_us_p50,gpu_us_p95,host_us_p50,host_us_p95,oracle");
    for (int pass = 0; pass < 3; ++pass) {
      const int mode = reverse ? 2 - pass : pass;
      for (int warm = 0; warm < 3; ++warm) one(mode);
      std::vector<double> gpu_us, host_us;
      for (int i = 0; i < iterations; ++i) {
        const auto [g, h] = one(mode);
        gpu_us.push_back(g); host_us.push_back(h);
      }
      HC(hipMemcpy(check.data(), out, (size_t)lanes * 4, hipMemcpyDeviceToHost));
      // Independent CPU oracle, including overflow modulo 2^32.
      for (int k = 0; k < lanes; k += std::max(1, lanes / 37)) {
        uint32_t expected = 0;
        for (size_t j = (size_t)k; j < words; j += (size_t)lanes)
          expected += (uint32_t)j * 2654435761u + 113u +
                      ((uint32_t)j * 2654435761u + 269u);
        if (expected != check[k])
          throw std::runtime_error("independent CPU sample oracle mismatch");
      }
      if (golden.empty()) golden = check;
      else if (check != golden) throw std::runtime_error("mode output differs from fused");
      std::printf("%s,%.2f,%.2f,%.2f,%.2f,PASS\n",
                  mode == 0 ? "fused" : mode == 1 ? "sequential" : "two_stream",
                  quantile(gpu_us, 0.50), quantile(gpu_us, 0.95),
                  quantile(host_us, 0.50), quantile(host_us, 0.95));
    }
    HC(hipEventDestroy(start)); HC(hipEventDestroy(doneA));
    HC(hipEventDestroy(doneB)); HC(hipEventDestroy(stop));
    HC(hipStreamDestroy(sa)); HC(hipStreamDestroy(sb)); HC(hipStreamDestroy(sc));
    HC(hipFree(wa)); HC(hipFree(wb)); HC(hipFree(pa)); HC(hipFree(pb)); HC(hipFree(out));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "VTP04 local FAILED: %s\n", e.what());
    return 1;
  }
}
