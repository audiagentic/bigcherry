// RAD20 decode diagnosis: does a completion event recorded after every request slow the next request's kernels?
// The live helper's small-lane GEMV doubled (gate_up 73 -> 156 us a request) once RAD20 recorded a blocking-sync event
// after every request. This repeats a weight-streaming kernel back to back on one GPU, with each event variant between
// launches, and times every kernel on the GPU's own clock (wall_clock64 at start and end), so gaps and host waits do
// not count. Depth-4 in-flight queue with a host wait on the oldest event, as the lane thread does.
//
// Build: hipcc -O3 --offload-arch=gfx1100 event-fence-bench.hip -o event-fence-bench
// Run:   HIP_VISIBLE_DEVICES=<xtx> ./event-fence-bench [working-set MiB=64] [iterations=3000] [blockingsync-device-flag 0|1]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)

// Each launch reads a different slice of the working set (like different experts), sums it, and stamps its own time.
__global__ void stream_read(const uint4* w, size_t n_vec, size_t offset, unsigned long long* stamps, float* sink) {
    __shared__ unsigned long long t0;
    if (threadIdx.x == 0) t0 = wall_clock64();
    __syncthreads();
    float acc = 0.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += (size_t)gridDim.x * blockDim.x) {
        const uint4 v = w[(offset + i) % (n_vec * 8)];
        acc += __uint_as_float(v.x) + __uint_as_float(v.y) + __uint_as_float(v.z) + __uint_as_float(v.w);
    }
    if (acc == 1234.5f) *sink = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicMin(&stamps[0], t0);
        atomicMax(&stamps[1], wall_clock64());
    }
}

int main(int argc, char** argv) {
    const size_t set_mib = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 64;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 3000;
    const bool dev_flag = argc > 3 && std::atoi(argv[3]) != 0;
    CK(hipSetDevice(0));
    if (dev_flag) CK(hipSetDeviceFlags(hipDeviceScheduleBlockingSync));
    int clock_khz = 0;
    CK(hipDeviceGetAttribute(&clock_khz, hipDeviceAttributeWallClockRate, 0));
    // A request reads ~1/8 of the set (one expert group); the whole set is 8x that, so consecutive requests differ.
    const size_t per_req = set_mib * 1024 * 1024 / 8, n_vec = per_req / sizeof(uint4);
    uint4* w; float* sink; unsigned long long* stamps;
    CK(hipMalloc(&w, per_req * 8));
    CK(hipMemset(w, 1, per_req * 8));
    CK(hipMalloc(&sink, 4));
    CK(hipMalloc(&stamps, sizeof(unsigned long long) * 2 * iters));
    hipStream_t s;
    CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

    struct Variant { const char* name; bool record; unsigned flags; bool host_wait; };
    const Variant variants[] = {
        {"no-event", false, 0, false},
        {"event-default", true, hipEventDefault, true},
        {"event-disabletiming", true, hipEventDisableTiming, true},
        {"event-blocking", true, hipEventBlockingSync | hipEventDisableTiming, true},
        {"event-blocking-nofence", true, hipEventBlockingSync | hipEventDisableTiming | hipEventDisableSystemFence, true},
        {"event-nofence", true, hipEventDisableTiming | hipEventDisableSystemFence, true},
    };
    std::printf("working set %zu MiB (%zu MiB a request), %d iterations, device BlockingSync flag %d\n", set_mib,
                per_req >> 20, iters, (int)dev_flag);
    for (int rep = 0; rep < 2; ++rep)
        for (const Variant& v : variants) {
            hipEvent_t ev[4] = {};
            if (v.record) for (auto& e : ev) CK(hipEventCreateWithFlags(&e, v.flags));
            std::vector<unsigned long long> init(2 * iters);
            for (int i = 0; i < iters; ++i) { init[2 * i] = ~0ull; init[2 * i + 1] = 0; }
            CK(hipMemcpy(stamps, init.data(), init.size() * 8, hipMemcpyHostToDevice));
            CK(hipDeviceSynchronize());
            for (int i = 0; i < iters; ++i) {
                if (v.host_wait && i >= 4) CK(hipEventSynchronize(ev[i % 4]));   // oldest of four in flight
                hipLaunchKernelGGL(stream_read, dim3(192), dim3(256), 0, s, w, n_vec, (size_t)(i % 8) * n_vec,
                                   stamps + 2 * i, sink);
                if (v.record) CK(hipEventRecord(ev[i % 4], s));
            }
            CK(hipStreamSynchronize(s));
            std::vector<unsigned long long> st(2 * iters);
            CK(hipMemcpy(st.data(), stamps, st.size() * 8, hipMemcpyDeviceToHost));
            std::vector<double> dur, gap;
            for (int i = 100; i < iters; ++i) {   // skip warm-up
                dur.push_back((st[2 * i + 1] - st[2 * i]) * 1000.0 / clock_khz);
                gap.push_back(((long long)st[2 * i] - (long long)st[2 * i - 1]) * 1000.0 / clock_khz);
            }
            auto pct = [](std::vector<double> x, double p) { std::sort(x.begin(), x.end()); return x[(size_t)(p * (x.size() - 1))]; };
            std::printf("%-24s kernel p50 %7.1f us p90 %7.1f | gap between kernels p50 %6.1f us p90 %6.1f\n", v.name,
                        pct(dur, .5), pct(dur, .9), pct(gap, .5), pct(gap, .9));
            if (v.record) for (auto& e : ev) CK(hipEventDestroy(e));
        }
    return 0;
}
