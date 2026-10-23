// RAD18: is combining the two XTX helpers' bf16 partials on the CPU, so the R9700 reads one plane over its x4 link
// instead of two, cheaper than the read it saves? Measures, on pinned host memory shaped like a real prefill step
// (rows x 2560 bf16 per helper):
//   saving: R9700 kernel reading 1 vs 2 mapped host planes with nontemporal loads (the engine merge's access pattern)
//   cost:   CPU combine of the two planes into one (f32 add, one bf16 round), 1..N threads
// The combined value is not bit-identical to the engine's current merge (one extra bf16 rounding); this harness only
// measures time.
// Build: hipcc --offload-arch=gfx1201 -O3 -pthread helper-combine-bench.hip -o helper-combine-bench
// Run:   HIP_VISIBLE_DEVICES=<R9700> ./helper-combine-bench [rows...]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); std::exit(1); } } while (0)

constexpr uint32_t kHidden = 2560;

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

__global__ void read_planes(const u32x4* a, const u32x4* b, u32x4* out, uint64_t n16, int planes) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n16; i += (uint64_t)gridDim.x * blockDim.x) {
        u32x4 v = __builtin_nontemporal_load(a + i);
        if (planes == 2) v ^= __builtin_nontemporal_load(b + i);
        out[i] = v;
    }
}

static inline float bf(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
static inline uint16_t rn(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }

static void combine(const uint16_t* a, const uint16_t* b, uint16_t* o, size_t lo, size_t hi) {
    for (size_t i = lo; i < hi; ++i) o[i] = rn(bf(a[i]) + bf(b[i]));
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
    std::vector<uint32_t> rows_list;
    for (int i = 1; i < argc; ++i) rows_list.push_back((uint32_t)std::atoi(argv[i]));
    if (rows_list.empty()) rows_list = {1152, 2048};
    const int reps = 20;
    for (uint32_t rows : rows_list) {
        const size_t n = (size_t)rows * kHidden, bytes = n * 2;
        uint16_t *a, *b, *c;
        CK(hipHostMalloc(&a, bytes, hipHostMallocMapped));
        CK(hipHostMalloc(&b, bytes, hipHostMallocMapped));
        CK(hipHostMalloc(&c, bytes, hipHostMallocMapped));
        for (size_t i = 0; i < n; ++i) { a[i] = (uint16_t)(0x3f80 + (i % 97)); b[i] = (uint16_t)(0x3f00 + (i % 89)); }
        u32x4* dev_out;
        CK(hipMalloc(&dev_out, bytes));
        u32x4 *da, *db;
        CK(hipHostGetDevicePointer((void**)&da, a, 0));
        CK(hipHostGetDevicePointer((void**)&db, b, 0));
        hipEvent_t e0, e1;
        CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
        double gpu_ms[3] = {0, 0, 0};
        for (int planes = 1; planes <= 2; ++planes) {
            std::vector<double> t;
            for (int r = 0; r < reps + 2; ++r) {
                CK(hipEventRecord(e0));
                hipLaunchKernelGGL(read_planes, dim3(256), dim3(256), 0, 0, da, db, dev_out, (uint64_t)(bytes / 16), planes);
                CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1));
                if (r >= 2) t.push_back(ms);
            }
            gpu_ms[planes] = median(t);
        }
        std::printf("rows %u (%.2f MB a plane): R9700 reads 1 plane %.3f ms, 2 planes %.3f ms -> saving %.3f ms (%.2f GB/s)\n",
                    rows, bytes / 1e6, gpu_ms[1], gpu_ms[2], gpu_ms[2] - gpu_ms[1], 2 * bytes / gpu_ms[2] / 1e6);
        for (unsigned th : {1u, 2u, 4u, 8u, 12u}) {
            std::vector<double> t;
            for (int r = 0; r < reps + 2; ++r) {
                const auto s = std::chrono::steady_clock::now();
                std::vector<std::thread> pool;
                for (unsigned k = 0; k < th; ++k)
                    pool.emplace_back(combine, a, b, c, n * k / th, n * (k + 1) / th);
                for (auto& p : pool) p.join();
                const auto e = std::chrono::steady_clock::now();
                if (r >= 2) t.push_back(std::chrono::duration<double, std::milli>(e - s).count());
            }
            std::printf("  CPU combine %2u thread(s): %.3f ms (incl. thread start)\n", th, median(t));
        }
        CK(hipFree(dev_out)); CK(hipHostFree(a)); CK(hipHostFree(b)); CK(hipHostFree(c));
    }
    return 0;
}
