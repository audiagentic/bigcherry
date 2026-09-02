// GP10: base-level (sub-llama.cpp) test harness for collective AllReduce
// transforms. First primitive: an N-way generalization of patch
// 1001_hip_internal_allreduce's pairwise pinned-host chunked-kernel
// mechanism (see vendor/llama.cpp/ggml/src/ggml-cuda/allreduce.cu).
//
// The existing internal pipeline hard-requires n_devices == 2: each GPU
// writes its contribution to its own pinned host slot, signals an arrival
// token, spins on its ONE peer's token, then reads the ONE peer's slot and
// sums. This harness generalizes that to N devices with a "star" pattern:
// each rank still writes exactly once and spins on all N-1 peers' tokens,
// but then reads and sums all N-1 peer slots instead of just one. This is
// the most direct N-way extension of the existing mechanism (no ring/tree
// staging), and is what GP11 needs a real go/no-go signal on before
// committing to a full ggml-cuda.cu implementation.
//
// Correctness is checked against an independent CPU-computed reference sum.
// Latency is measured with HIP events (not wall-clock-around-the-process).
//
// This is NOT production code and does not touch the patched llama.cpp
// source tree at all -- it links directly against ROCm/HIP, compiles in
// seconds, and runs in well under a minute for the whole sweep.
//
// Build (on a box with HIP/ROCm):
//   hipcc -O3 -std=c++17 -o nway_star_allreduce nway_star_allreduce.cpp
// Run:
//   ./nway_star_allreduce --devices 0,1,2 --elements 4096,30720,2621440 --reps 20

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t _rc = (call);                                               \
        if (_rc != hipSuccess) {                                               \
            fprintf(stderr, "HIP error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, hipGetErrorString(_rc));                         \
            exit(1);                                                          \
        }                                                                      \
    } while (0)

static constexpr int MAX_DEVICES = 8;
static constexpr int ARRIVAL_STRIDE_INTS = 16; // cache-line pad, mirrors allreduce.cu

// ---------------------------------------------------------------------------
// Cross-GPU signal mechanism -- same volatile+threadfence_system design as
// allreduce.cu's ggml_cuda_ar_signal_set/get, generalized to N peers.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void ar_signal_set(int * p, int token) {
    *(volatile int *) p = token;
}
static __device__ __forceinline__ int ar_signal_get(const int * p) {
    return *(const volatile int *) p;
}

// One kernel launch per rank. host_mine is this rank's pinned staging slot;
// host_peers[k] (k != rank) are every other rank's slot (host_peers[rank] is
// unused/nullptr). arrival is a flat n_devices-length array of per-rank
// tokens in pinned host memory.
__global__ void nway_star_kernel(
        const float * sendbuf,
        float       * recvbuf,
        float       * host_mine,
        float * const * host_peers,   // device array of n_devices host pointers
        int             n_devices,
        int             rank,
        int             count,
        int           * arrival,      // device array of n_devices ints (pinned)
        int             token) {
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;

    // Phase 1: stage local contribution to our own pinned slot.
    for (int i = tid; i < count; i += nt) {
        host_mine[i] = sendbuf[i];
    }
    __threadfence_system();
    __syncthreads();

    // Phase 2: signal arrival, then spin on every other rank's token.
    if (tid == 0) {
        ar_signal_set(&arrival[rank], token);
        __threadfence_system();
        for (int peer = 0; peer < n_devices; ++peer) {
            if (peer == rank) continue;
            while (ar_signal_get(&arrival[peer]) != token) {
#if defined(__HIP_DEVICE_COMPILE__)
                __builtin_amdgcn_s_sleep(4);
#endif
            }
        }
    }
    __syncthreads();
    __threadfence_system();

    // Phase 3: sum local + every peer's slot.
    for (int i = tid; i < count; i += nt) {
        float acc = sendbuf[i];
        for (int peer = 0; peer < n_devices; ++peer) {
            if (peer == rank) continue;
            acc += host_peers[peer][i];
        }
        recvbuf[i] = acc;
    }
}

struct DeviceState {
    int      device_id;
    float *  d_send   = nullptr;
    float *  d_recv   = nullptr;
    float *  h_stage  = nullptr;  // pinned, mapped
    float *  d_stage  = nullptr;  // device-mapped pointer to h_stage
    hipStream_t stream = nullptr;
};

static std::vector<int> parse_int_list(const std::string & s) {
    std::vector<int> out;
    size_t start = 0;
    while (start < s.size()) {
        size_t comma = s.find(',', start);
        std::string tok = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

int main(int argc, char ** argv) {
    std::vector<int> devices = {0, 1};
    std::vector<int> element_counts = {4096, 30720, 2621440};
    int reps = 20;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--devices" && i + 1 < argc) {
            devices = parse_int_list(argv[++i]);
        } else if (arg == "--elements" && i + 1 < argc) {
            element_counts = parse_int_list(argv[++i]);
        } else if (arg == "--reps" && i + 1 < argc) {
            reps = std::atoi(argv[++i]);
        } else {
            fprintf(stderr, "unknown arg: %s\n", arg.c_str());
            return 2;
        }
    }

    const int n = (int) devices.size();
    if (n < 2 || n > MAX_DEVICES) {
        fprintf(stderr, "need 2..%d devices, got %d\n", MAX_DEVICES, n);
        return 2;
    }

    printf("nway_star_allreduce: devices=[");
    for (int i = 0; i < n; ++i) printf("%d%s", devices[i], i + 1 < n ? "," : "");
    printf("] reps=%d\n", reps);

    for (int elems : element_counts) {
        const size_t bytes = (size_t) elems * sizeof(float);

        std::vector<DeviceState> ds(n);
        // Host-side reference input, one vector per rank.
        std::vector<std::vector<float>> input(n, std::vector<float>(elems));
        std::mt19937 rng(1234 + elems);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int r = 0; r < n; ++r) {
            for (int i = 0; i < elems; ++i) input[r][i] = dist(rng);
        }
        std::vector<float> reference(elems, 0.0f);
        for (int i = 0; i < elems; ++i) {
            float acc = 0.0f;
            for (int r = 0; r < n; ++r) acc += input[r][i];
            reference[i] = acc;
        }

        // Per-device alloc.
        for (int r = 0; r < n; ++r) {
            ds[r].device_id = devices[r];
            HIP_CHECK(hipSetDevice(devices[r]));
            HIP_CHECK(hipStreamCreateWithFlags(&ds[r].stream, hipStreamNonBlocking));
            HIP_CHECK(hipMalloc(&ds[r].d_send, bytes));
            HIP_CHECK(hipMalloc(&ds[r].d_recv, bytes));
            HIP_CHECK(hipHostMalloc((void **) &ds[r].h_stage, bytes,
                                     hipHostMallocPortable | hipHostMallocMapped));
            HIP_CHECK(hipHostGetDevicePointer((void **) &ds[r].d_stage, ds[r].h_stage, 0));
            HIP_CHECK(hipMemcpy(ds[r].d_send, input[r].data(), bytes, hipMemcpyHostToDevice));
        }

        // Pinned arrival ring (one int per rank) and a device-array of peer
        // staging pointers, visible from every device via mapped memory.
        int * h_arrival = nullptr;
        HIP_CHECK(hipHostMalloc((void **) &h_arrival, n * sizeof(int),
                                 hipHostMallocPortable | hipHostMallocMapped));
        memset(h_arrival, 0, n * sizeof(int));
        int * d_arrival = nullptr;
        HIP_CHECK(hipHostGetDevicePointer((void **) &d_arrival, h_arrival, 0));

        float ** h_peer_ptrs = nullptr;
        HIP_CHECK(hipHostMalloc((void **) &h_peer_ptrs, n * sizeof(float *),
                                 hipHostMallocPortable | hipHostMallocMapped));
        for (int r = 0; r < n; ++r) h_peer_ptrs[r] = ds[r].d_stage;
        float ** d_peer_ptrs = nullptr;
        HIP_CHECK(hipHostGetDevicePointer((void **) &d_peer_ptrs, h_peer_ptrs, 0));

        // Warmup (also validates correctness once before timing).
        int token = 1;
        std::vector<hipEvent_t> start_ev(n), stop_ev(n);
        for (int r = 0; r < n; ++r) {
            HIP_CHECK(hipSetDevice(devices[r]));
            HIP_CHECK(hipEventCreate(&start_ev[r]));
            HIP_CHECK(hipEventCreate(&stop_ev[r]));
            nway_star_kernel<<<1, 256, 0, ds[r].stream>>>(
                ds[r].d_send, ds[r].d_recv, ds[r].d_stage, d_peer_ptrs,
                n, r, elems, d_arrival, token);
        }
        for (int r = 0; r < n; ++r) {
            HIP_CHECK(hipSetDevice(devices[r]));
            HIP_CHECK(hipStreamSynchronize(ds[r].stream));
        }

        std::vector<float> out0(elems);
        HIP_CHECK(hipSetDevice(devices[0]));
        HIP_CHECK(hipMemcpy(out0.data(), ds[0].d_recv, bytes, hipMemcpyDeviceToHost));
        double max_abs_err = 0.0;
        for (int i = 0; i < elems; ++i) {
            max_abs_err = std::max(max_abs_err, (double) std::fabs(out0[i] - reference[i]));
        }
        // F32 accumulation in a different order than the CPU reference; allow
        // a generous but real epsilon (not "close enough to hide a bug").
        const double eps = 1e-4 * n;
        bool correct = max_abs_err < eps;

        // Timed reps.
        std::vector<double> elapsed_ms;
        elapsed_ms.reserve(reps);
        for (int rep = 0; rep < reps; ++rep) {
            token++;
            for (int r = 0; r < n; ++r) {
                HIP_CHECK(hipSetDevice(devices[r]));
                HIP_CHECK(hipEventRecord(start_ev[r], ds[r].stream));
                nway_star_kernel<<<1, 256, 0, ds[r].stream>>>(
                    ds[r].d_send, ds[r].d_recv, ds[r].d_stage, d_peer_ptrs,
                    n, r, elems, d_arrival, token);
                HIP_CHECK(hipEventRecord(stop_ev[r], ds[r].stream));
            }
            for (int r = 0; r < n; ++r) {
                HIP_CHECK(hipSetDevice(devices[r]));
                HIP_CHECK(hipStreamSynchronize(ds[r].stream));
            }
            float max_ms = 0.0f;
            for (int r = 0; r < n; ++r) {
                float ms = 0.0f;
                HIP_CHECK(hipEventElapsedTime(&ms, start_ev[r], stop_ev[r]));
                max_ms = std::max(max_ms, ms);
            }
            elapsed_ms.push_back(max_ms);
        }

        std::vector<double> sorted = elapsed_ms;
        std::sort(sorted.begin(), sorted.end());
        double median_us = sorted[sorted.size() / 2] * 1000.0;
        double p90_us = sorted[(size_t)(sorted.size() * 0.9)] * 1000.0;

        printf("elements=%-9d bytes=%-9zu correct=%s max_abs_err=%.6g median_us=%.2f p90_us=%.2f\n",
               elems, bytes, correct ? "true" : "FALSE", max_abs_err, median_us, p90_us);

        for (int r = 0; r < n; ++r) {
            HIP_CHECK(hipSetDevice(devices[r]));
            hipEventDestroy(start_ev[r]);
            hipEventDestroy(stop_ev[r]);
            hipFree(ds[r].d_send);
            hipFree(ds[r].d_recv);
            hipHostFree(ds[r].h_stage);
            hipStreamDestroy(ds[r].stream);
        }
        hipHostFree(h_arrival);
        hipHostFree(h_peer_ptrs);
    }

    return 0;
}
