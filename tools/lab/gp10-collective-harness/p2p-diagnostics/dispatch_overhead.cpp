// GP11 follow-up: raw proof of per-dispatch launch overhead on gfx1100, and
// whether HIP graph replay removes it.
//
#include <chrono>
// Motivation: a fresh rocprofv3 decode capture shows the GPU idle 30.2% of
// decode wall, and ~56% of that idle is sub-20us gaps BETWEEN kernels
// (290k of them, ~107k dispatches/sec). That is dispatch overhead, not
// compute. This measures it directly and tests the fix.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>

#define CHK(c) do { hipError_t r=(c); if(r!=hipSuccess){printf("!! %d %s\n",__LINE__,hipGetErrorString(r));} } while(0)

// deliberately trivial: we are measuring launch cost, not kernel cost
__global__ void tiny(float * p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = p[i] * 1.000001f + 1e-7f;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
}

int main(int argc, char** argv) {
    const int N       = argc > 1 ? atoi(argv[1]) : 4096;   // elements (small = launch-bound)
    const int CHAIN   = argc > 2 ? atoi(argv[2]) : 200;    // kernels per iteration
    const int ITERS   = argc > 3 ? atoi(argv[3]) : 50;

    CHK(hipSetDevice(0));
    float * d = nullptr;
    CHK(hipMalloc(&d, N * sizeof(float)));
    CHK(hipMemset(d, 0, N * sizeof(float)));

    hipStream_t s;
    CHK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    printf("elements=%d  chain=%d kernels/iter  iters=%d  grid=%dx%d\n",
           N, CHAIN, ITERS, blocks, threads);

    // ---- warmup
    for (int i = 0; i < CHAIN; ++i) tiny<<<blocks, threads, 0, s>>>(d, N);
    CHK(hipStreamSynchronize(s));

    // ---- A: ordinary stream launches
    std::vector<double> a_us;
    for (int it = 0; it < ITERS; ++it) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < CHAIN; ++i) tiny<<<blocks, threads, 0, s>>>(d, N);
        CHK(hipStreamSynchronize(s));
        auto t1 = std::chrono::high_resolution_clock::now();
        a_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    const double a = median(a_us);

    // ---- B: same chain captured into a HIP graph and replayed
    hipGraph_t graph; hipGraphExec_t exec;
    CHK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
    for (int i = 0; i < CHAIN; ++i) tiny<<<blocks, threads, 0, s>>>(d, N);
    CHK(hipStreamEndCapture(s, &graph));
    CHK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CHK(hipGraphLaunch(exec, s));           // warm the exec
    CHK(hipStreamSynchronize(s));

    std::vector<double> b_us;
    for (int it = 0; it < ITERS; ++it) {
        auto t0 = std::chrono::high_resolution_clock::now();
        CHK(hipGraphLaunch(exec, s));
        CHK(hipStreamSynchronize(s));
        auto t1 = std::chrono::high_resolution_clock::now();
        b_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    const double b = median(b_us);

    // ---- C: pure launch cost with no sync (submission throughput)
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < CHAIN * ITERS; ++i) tiny<<<blocks, threads, 0, s>>>(d, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    CHK(hipStreamSynchronize(s));
    const double submit_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / (CHAIN * ITERS);

    printf("\n%-34s %10s %14s\n", "mode", "us/chain", "us/kernel");
    printf("%-34s %10.1f %14.3f\n", "A: stream launches", a, a / CHAIN);
    printf("%-34s %10.1f %14.3f\n", "B: HIP graph replay", b, b / CHAIN);
    printf("%-34s %10s %14.3f\n", "C: submit-only (no sync)", "-", submit_us);
    printf("\ngraph speedup: %.2fx   per-kernel saving: %.3f us\n",
           a / b, (a - b) / CHAIN);

    // Scale the saving to the real workload measured in the decode trace.
    const double DISPATCHES = 632620.0, DECODE_MS = 5900.0;
    const double saved_ms = (a - b) / CHAIN * DISPATCHES / 1000.0;
    printf("\nExtrapolated to the real decode window (%.0f dispatches, %.0f ms):\n",
           DISPATCHES, DECODE_MS);
    printf("  would remove ~%.1f ms = %.1f%% of decode wall\n",
           saved_ms, 100.0 * saved_ms / DECODE_MS);
    printf("  (upper bound: assumes every dispatch is graph-capturable)\n");

    CHK(hipGraphExecDestroy(exec)); CHK(hipGraphDestroy(graph));
    CHK(hipStreamDestroy(s)); CHK(hipFree(d));
    return 0;
}
