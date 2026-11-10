// VTP04: three-PHYSICAL-GPU CPU-root exact F32 collective, no P2P.
// XTX0 + XTX1 + R9700 (two local logical ranks). Compare R9700 local
// GPU combine and ONE D2H packet against two separate R9700 D2H packets.
// This is a safe host-staged transport lower-bound harness, NOT Radiance's
// all-reduce kernel, graph, residency system, or a throughput prediction.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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

__global__ void make_float(float* dst, size_t n, float base) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
       i < n; i += gridDim.x * (size_t)blockDim.x)
    dst[i] = base + (float)(i & 7) * 0.125f;
}
__global__ void local_add(const float* a, const float* b, float* out, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
       i < n; i += gridDim.x * (size_t)blockDim.x)
    out[i] = a[i] + b[i];
}
struct Dev {
  int id = -1;
  hipStream_t stream = nullptr;
  hipEvent_t ready = nullptr;
  float* input0 = nullptr;
  float* input1 = nullptr;  // only present on the R9700
  float* combined = nullptr; // only present on the R9700
  float* result = nullptr;
};
static double percentile(std::vector<double> v, double frac) {
  std::sort(v.begin(), v.end());
  return v[(size_t)((v.size() - 1) * frac)];
}
static int read_arg(int argc, char** argv, const char* name, int def) {
  for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], name) == 0) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value: ") + name);
    char* end = nullptr;
    const long v = std::strtol(argv[i + 1], &end, 10);
    if (!end || *end || v < 1 || v > 1000000)
      throw std::runtime_error(std::string("bad value: ") + name);
    return (int)v;
  }
  return def;
}
static std::array<int,3> physical_devices() {
  int count = 0; HC(hipGetDeviceCount(&count));
  std::vector<int> xtx, r9700;
  for (int i = 0; i < count; ++i) {
    hipDeviceProp_t p{}; HC(hipGetDeviceProperties(&p, i));
    if (std::strncmp(p.gcnArchName, "gfx1100", 7) == 0) xtx.push_back(i);
    if (std::strncmp(p.gcnArchName, "gfx1201", 7) == 0) r9700.push_back(i);
  }
  if (xtx.size() != 2 || r9700.size() != 1)
    throw std::runtime_error("requires exactly two visible gfx1100 and one gfx1201");
  return {xtx[0], xtx[1], r9700[0]};
}
using Clock = std::chrono::steady_clock;
static double micro(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}
struct Times {
  double total_us, from_gpu_us, host_reduce_us, to_gpu_us;
};
int main(int argc, char** argv) {
  try {
    const int rows = read_arg(argc, argv, "--rows", 4);
    const int iterations = read_arg(argc, argv, "--iterations", 9);
    bool reverse = false;
    for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], "--reverse") == 0) reverse = true;
    if (rows > 2048 || iterations > 100)
      throw std::runtime_error("limits: rows <= 2048, iterations <= 100");
    constexpr int hidden = 5120;
    const size_t count = (size_t)rows * hidden;
    const size_t bytes = count * sizeof(float);
    const int blocks = (int)std::min<size_t>((count + 255) / 256, 1024);
    const auto ids = physical_devices();
    std::array<Dev,3> d;
    for (int i = 0; i < 3; ++i) {
      Dev& p = d[i]; p.id = ids[i];
      HC(hipSetDevice(p.id));
      HC(hipStreamCreateWithFlags(&p.stream, hipStreamNonBlocking));
      HC(hipEventCreateWithFlags(&p.ready, hipEventDisableTiming));
      HC(hipMalloc((void**)&p.input0, bytes));
      HC(hipMalloc((void**)&p.result, bytes));
      if (i == 2) {
        HC(hipMalloc((void**)&p.input1, bytes));
        HC(hipMalloc((void**)&p.combined, bytes));
      }
      make_float<<<blocks, 256, 0, p.stream>>>(p.input0, count, (float)(i + 1));
      if (i == 2) make_float<<<blocks, 256, 0, p.stream>>>(p.input1, count, 4.0f);
      HC(hipGetLastError());
      HC(hipStreamSynchronize(p.stream));
    }
    std::array<float*,5> host{};
    for (float*& p : host) HC(hipHostMalloc((void**)&p, bytes, hipHostMallocDefault));
    // h0, h1, h2, h3, result; all pinned and separately addressed.
    auto once = [&](bool local) -> Times {
      const auto total = Clock::now();
      for (int i = 0; i < 3; ++i) {
        HC(hipSetDevice(d[i].id));
        if (i != 2) {
          HC(hipMemcpyAsync(host[i], d[i].input0, bytes,
                            hipMemcpyDeviceToHost, d[i].stream));
        } else if (local) {
          local_add<<<blocks, 256, 0, d[i].stream>>>(d[i].input0, d[i].input1,
                                                      d[i].combined, count);
          HC(hipGetLastError());
          HC(hipMemcpyAsync(host[2], d[i].combined, bytes,
                            hipMemcpyDeviceToHost, d[i].stream));
        } else {
          HC(hipMemcpyAsync(host[2], d[i].input0, bytes,
                            hipMemcpyDeviceToHost, d[i].stream));
          HC(hipMemcpyAsync(host[3], d[i].input1, bytes,
                            hipMemcpyDeviceToHost, d[i].stream));
        }
        HC(hipEventRecord(d[i].ready, d[i].stream));
      }
      for (int i = 0; i < 3; ++i) {
        HC(hipSetDevice(d[i].id));
        HC(hipEventSynchronize(d[i].ready));
      }
      const double from_gpu = micro(total);
      const auto cpu = Clock::now();
      for (size_t k = 0; k < count; ++k)
        host[4][k] = local ? (host[0][k] + host[1][k] + host[2][k])
                            : ((host[0][k] + host[1][k] + host[2][k]) + host[3][k]);
      const double cpu_us = micro(cpu);
      const auto back = Clock::now();
      for (int i = 0; i < 3; ++i) {
        HC(hipSetDevice(d[i].id));
        HC(hipMemcpyAsync(d[i].result, host[4], bytes, hipMemcpyHostToDevice, d[i].stream));
      }
      for (int i = 0; i < 3; ++i) {
        HC(hipSetDevice(d[i].id));
        HC(hipStreamSynchronize(d[i].stream));
      }
      return {micro(total), from_gpu, cpu_us, micro(back)};
    };
    std::printf("physical_device_ids=%d,%d,%d rows=%d hidden=%d bytes_per_packet=%zu\n",
                ids[0], ids[1], ids[2], rows, hidden, bytes);
    std::puts("mode,wall_us_p50,wall_us_p95,from_gpu_us_p50,host_sum_us_p50,to_gpu_us_p50,host_packets,oracle");
    std::vector<float> check(count);
    for (int mode = 0; mode < 2; ++mode) {
      const bool local = reverse ? mode == 1 : mode == 0;
      for (int warm = 0; warm < 2; ++warm) (void)once(local);
      std::vector<double> wall, from, cpu, back;
      for (int n = 0; n < iterations; ++n) {
        Times t = once(local);
        wall.push_back(t.total_us); from.push_back(t.from_gpu_us);
        cpu.push_back(t.host_reduce_us); back.push_back(t.to_gpu_us);
      }
      for (int i = 0; i < 3; ++i) {
        HC(hipSetDevice(d[i].id));
        HC(hipMemcpy(check.data(), d[i].result, bytes, hipMemcpyDeviceToHost));
        for (size_t k = 0; k < count; ++k) {
          const float expected = 10.0f + (float)(k & 7) * 0.5f;
          if (check[k] != expected)
            throw std::runtime_error("exact F32 host collective oracle mismatch");
        }
      }
      std::printf("%s,%.2f,%.2f,%.2f,%.2f,%.2f,%d,PASS\n",
                  local ? "r9700_gpu_local" : "r9700_host_two_packets",
                  percentile(wall, 0.50), percentile(wall, 0.95),
                  percentile(from, 0.50), percentile(cpu, 0.50),
                  percentile(back, 0.50), local ? 3 : 4);
    }
    for (auto& p : d) {
      HC(hipSetDevice(p.id));
      HC(hipStreamDestroy(p.stream)); HC(hipEventDestroy(p.ready));
      HC(hipFree(p.input0)); HC(hipFree(p.result));
      if (p.input1) HC(hipFree(p.input1));
      if (p.combined) HC(hipFree(p.combined));
    }
    for (float* p : host) HC(hipHostFree(p));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "VTP04 host collective FAILED: %s\n", e.what());
    return 1;
  }
}
