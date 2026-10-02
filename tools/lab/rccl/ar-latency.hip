// Small-message AllReduce latency on the visible GPUs: RCCL (ncclAllReduce, f32 sum, single process,
// ncclCommInitAll) for sizes 2.5K..10M floats; prints mean microseconds per call over N iterations after
// warmup. Build: hipcc -O2 ar-latency.hip -lrccl -o ar-latency ; run with HIP_VISIBLE_DEVICES set.
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <chrono>
#include <cstdio>
#include <vector>
int main() {
    int n = 0; hipGetDeviceCount(&n);
    std::vector<int> devs(n); for (int i = 0; i < n; i++) devs[i] = i;
    std::vector<ncclComm_t> comms(n); ncclCommInitAll(comms.data(), n, devs.data());
    std::vector<hipStream_t> streams(n); std::vector<float *> bufs(n);
    const size_t max_elems = 10u << 20;
    for (int i = 0; i < n; i++) { hipSetDevice(i); hipStreamCreate(&streams[i]); hipMalloc(&bufs[i], max_elems * sizeof(float)); hipMemset(bufs[i], 0, max_elems * sizeof(float)); }
    const size_t sizes[] = {2560, 5120, 10240, 2560 * 64, 2560 * 1024, 2560 * 4096};
    for (size_t elems : sizes) {
        const int iters = elems <= 10240 ? 2000 : 100;
        for (int rep = 0; rep < 2; rep++) {  // rep 0 = warmup
            auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < iters; it++) {
                ncclGroupStart();
                for (int i = 0; i < n; i++) ncclAllReduce(bufs[i], bufs[i], elems, ncclFloat, ncclSum, comms[i], streams[i]);
                ncclGroupEnd();
                for (int i = 0; i < n; i++) { hipSetDevice(i); hipStreamSynchronize(streams[i]); }
            }
            auto us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
            if (rep) printf("ranks=%d elems=%zu bytes=%zu  %.1f us/call (synchronised each call)\n", n, elems, elems * 4, us);
        }
        // Stream-ordered chain, one sync at the end: how the graph issues back-to-back AllReduces.
        {
            auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < iters; it++) {
                ncclGroupStart();
                for (int i = 0; i < n; i++) ncclAllReduce(bufs[i], bufs[i], elems, ncclFloat, ncclSum, comms[i], streams[i]);
                ncclGroupEnd();
            }
            for (int i = 0; i < n; i++) { hipSetDevice(i); hipStreamSynchronize(streams[i]); }
            auto us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
            printf("ranks=%d elems=%zu bytes=%zu  %.1f us/call (stream-ordered chain)\n", n, elems, elems * 4, us);
        }
    }
    for (int i = 0; i < n; i++) ncclCommDestroy(comms[i]);
    return 0;
}
