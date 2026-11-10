// HL01: does a card's host link carry both directions at once, and does moving host data slow a compute kernel on the same card?
// The tail's layer call is pull (host -> GPU), GEMM, push (GPU -> host); PIPE01 would stagger two token halves so one half's
// transfer runs under the other half's GEMM. This measures that on one card, no GEMM replay: the compute stand-in is a CU-bound
// FMA kernel sized to about one gate/up GEMM, so a slowdown here is an upper bound on contention, not the real kernel's.
// Build: hipcc -O3 --offload-arch=gfx1030,gfx1100,gfx1201 tools/lab/radiance/host-link-duplex.hip -o host-link-duplex
// Run:   for g in 0 1 2 3; do HIP_VISIBLE_DEVICES=$g ./host-link-duplex 5 10; done      (in MiB: inbound, outbound)
// Cells: serial (pull then push), duplex by kernels, duplex by copy engines (hipMemcpyAsync), mixed, each alone and beside compute.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1); } } while (0)

__global__ void kread(const uint4* p, size_t n, unsigned* out) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, s = (size_t)gridDim.x * blockDim.x; unsigned a = 0;
    for (; i < n; i += s) { uint4 v = p[i]; a += v.x ^ v.y ^ v.z ^ v.w; }
    if (a == 0x12345678u) *out = a;
}
__global__ void kwrite(uint4* p, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, s = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += s) p[i] = make_uint4((unsigned)i, (unsigned)i, (unsigned)i, (unsigned)i);
}
__global__ void kfma(float* out, int iters) {
    float a = threadIdx.x * 1e-3f + 1.f, b = 1.0001f, c = 0.5f;
    for (int i = 0; i < iters; ++i) { a = a * b + c; c = c * b + a; }
    if (a == 12345.f) *out = a + c;
}

static float quantile(std::vector<float>& v, double q) { std::sort(v.begin(), v.end()); return v[(size_t)(q * (v.size() - 1))]; }

int main(int argc, char** argv) {
    size_t in_mib = argc > 1 ? atoi(argv[1]) : 5, out_mib = argc > 2 ? atoi(argv[2]) : 10;
    const size_t in_b = in_mib << 20, out_b = out_mib << 20;
    int dev; CK(hipGetDevice(&dev)); hipDeviceProp_t pr; CK(hipGetDeviceProperties(&pr, dev));
    void *h_in, *h_out, *g_in, *g_out, *d_in, *d_out; unsigned* o; float* fo;
    CK(hipHostMalloc(&h_in, in_b, hipHostMallocMapped | hipHostMallocCoherent));
    CK(hipHostMalloc(&h_out, out_b, hipHostMallocMapped | hipHostMallocCoherent));
    CK(hipHostGetDevicePointer(&g_in, h_in, 0)); CK(hipHostGetDevicePointer(&g_out, h_out, 0));
    CK(hipMalloc(&d_in, in_b)); CK(hipMalloc(&d_out, out_b)); CK(hipMalloc(&o, 4)); CK(hipMalloc(&fo, 4));
    CK(hipMemset(d_out, 1, out_b)); CK(hipMemset(h_in, 2, in_b));
    hipStream_t s1, s2, s3; CK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
    CK(hipStreamCreateWithFlags(&s2, hipStreamNonBlocking)); CK(hipStreamCreateWithFlags(&s3, hipStreamNonBlocking));
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    const size_t n_in = in_b / 16, n_out = out_b / 16;
    const int cus = pr.multiProcessorCount;
    auto pull_k = [&](hipStream_t s) { kread<<<1024, 256, 0, s>>>((const uint4*)g_in, n_in, o); };
    auto push_k = [&](hipStream_t s) { kwrite<<<1024, 256, 0, s>>>((uint4*)g_out, n_out); };
    auto pull_c = [&](hipStream_t s) { CK(hipMemcpyAsync(d_in, h_in, in_b, hipMemcpyHostToDevice, s)); };
    auto push_c = [&](hipStream_t s) { CK(hipMemcpyAsync(h_out, d_out, out_b, hipMemcpyDeviceToHost, s)); };
    int fma_iters = 2000;
    auto comp = [&](hipStream_t s) { kfma<<<cus * 8, 256, 0, s>>>(fo, fma_iters); };
    // time `body` (enqueues on streams) as wall time between two events on s3 around a full device sync
    auto run = [&](auto body, int iters) {
        std::vector<float> t;
        for (int i = -10; i < iters; ++i) {
            CK(hipDeviceSynchronize()); CK(hipEventRecord(a, s3)); body(); CK(hipDeviceSynchronize()); CK(hipEventRecord(b, s3));
            CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); if (i >= 0) t.push_back(ms * 1000.f);
        }
        return t;
    };
    // tune the compute stand-in to ~ 2 ms alone (a tail gate/up GEMM call is 4-6 ms)
    for (int k = 0; k < 6; ++k) { auto t = run([&] { comp(s3); }, 5); float m = quantile(t, 0.5); if (m > 1800 && m < 2600) break; fma_iters = (int)(fma_iters * 2000.0 / std::max(m, 50.f)); }
    auto row = [&](const char* name, auto body) {
        std::vector<float> t = run(body, 100); std::printf("%-44s p50 %8.0f us  p95 %8.0f us\n", name, quantile(t, 0.5), quantile(t, 0.95)); return quantile(t, 0.5); };
    std::printf("device %s  CUs %d  asyncEngines %d  inbound %zu MiB outbound %zu MiB  compute stand-in %d fma iters\n", pr.name, cus, pr.asyncEngineCount, in_mib, out_mib, fma_iters);
    float c_alone = row("compute alone", [&] { comp(s3); });
    float pk = row("pull kernel alone", [&] { pull_k(s1); });
    float sk = row("push kernel alone", [&] { push_k(s2); });
    float pc = row("pull copy (H2D) alone", [&] { pull_c(s1); });
    float sc = row("push copy (D2H) alone", [&] { push_c(s2); });
    row("pull+push kernels, serial", [&] { pull_k(s1); push_k(s1); });
    float dk = row("pull+push kernels, duplex (2 streams)", [&] { pull_k(s1); push_k(s2); });
    row("pull+push copies, serial", [&] { pull_c(s1); push_c(s1); });
    float dc = row("pull+push copies, duplex (2 streams)", [&] { pull_c(s1); push_c(s2); });
    row("pull copy + push kernel, duplex", [&] { pull_c(s1); push_k(s2); });
    row("pull kernel + push copy, duplex", [&] { pull_k(s1); push_c(s2); });
    float ck = row("compute + pull/push kernels duplex", [&] { comp(s3); pull_k(s1); push_k(s2); });
    float cc = row("compute + pull/push copies duplex", [&] { comp(s3); pull_c(s1); push_c(s2); });
    std::printf("duplex kernels %.2fx the slower direction (serial sum %.0f us); duplex copies %.2fx (sum %.0f us)\n",
                dk / std::max(pk, sk), pk + sk, dc / std::max(pc, sc), pc + sc);
    std::printf("makespan beside compute: kernels %.0f us vs max(compute alone, duplex kernels) %.0f us; copies %.0f us vs %.0f us\n",
                ck, std::max(c_alone, dk), cc, std::max(c_alone, dc));
    return 0;
}
