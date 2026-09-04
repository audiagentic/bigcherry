// GP11 follow-up: P2P-direct-load AllReduce prototype for the dual-XTX pair.
//
// Context: a separate agent's concurrent work enabled real hipDeviceCanAccessPeer
// between the two RX 7900 XTXs on this box (was 0 for every GPU pair before,
// now 1 for GPU0<->GPU1 specifically -- see GP11.md notes, 2026-09-04). Raw
// hipMemcpy P2P bandwidth measured at 10.4 GB/s direct vs 6.85 GB/s host-staged
// (1.5x steady-state). RCCL itself does NOT exploit this (RCCL 2.27.7 hard-caps
// Intel x86 P2P eligibility at PATH_PXB; this box's topology classifies as
// PATH_PHB, so RCCL falls back to SHM regardless of canAccessPeer -- confirmed
// via NCCL_DEBUG=INFO showing "via SHM/direct/direct", per gpt-dev-agent
// req_9d9d432819ae46f3).
//
// This harness tests gpt's recommended design for bigcherry's OWN internal
// AllReduce mechanism (patches 1001/1244): keep the existing tiny pinned-host
// arrival-token handshake (cheap, already proven), but replace PAYLOAD staging
// with a direct peer-device load instead of a pinned-host round-trip. This is
// NOT hipMemcpyPeerAsync (gpt flagged that as adding avoidable event/launch
// overhead) -- it's a kernel on rank A directly dereferencing a P2P-mapped
// pointer into rank B's device memory.
//
// --mode host   -- baseline: today's production design (root_kernel from
//                  nway_star_allreduce.cpp, N=2, staged through pinned host
//                  memory both directions).
// --mode p2p    -- direct peer-device load: each rank reads the OTHER rank's
//                  d_send buffer directly (no host staging for the payload),
//                  still uses the tiny host-mapped arrival array for the
//                  ready/wait handshake (same cost as today, not the thing
//                  being tested).
//
// Build:
//   hipcc -O3 -std=c++17 -o p2p_direct_allreduce p2p_direct_allreduce.cpp
// Run:
//   ./p2p_direct_allreduce --devices 0,1 --elements 7680,15360,30720,61440,122880 --reps 500 --mode p2p
//   ./p2p_direct_allreduce --devices 0,1 --elements 7680,15360,30720,61440,122880 --reps 500 --mode host

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

static constexpr int MAX_BLOCKS = 32;
static constexpr int ARRIVAL_STRIDE_INTS = 16; // one cache line per (rank,block)

enum Mode { MODE_HOST = 0, MODE_P2P = 1 };

static __device__ __forceinline__ void ar_signal_set(int * p, int token) {
    *(volatile int *) p = token;
}
static __device__ __forceinline__ int ar_signal_get(const int * p) {
    return *(const volatile int *) p;
}
static __device__ __forceinline__ int * arrival_slot(int * arrival, int rank, int block) {
    return arrival + (rank * MAX_BLOCKS + block) * ARRIVAL_STRIDE_INTS;
}

// --- host-staged baseline (today's production design, N=2 specialization of
// root_kernel from nway_star_allreduce.cpp) ---------------------------------
__global__ void host_staged_kernel(
        const float * sendbuf,
        float       * recvbuf,
        float       * host_mine,
        const float * host_peer,
        int             rank,
        int             peer_rank,
        int             count,
        int           * arrival,
        int             token) {
    // stage local payload into pinned host memory
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gnt  = gridDim.x * blockDim.x;
    const int count4 = count / 4;
    const float4 * src4 = reinterpret_cast<const float4 *>(sendbuf);
    float4       * dst4 = reinterpret_cast<float4 *>(host_mine);
    for (int i = gtid; i < count4; i += gnt) dst4[i] = src4[i];
    const int tail = count4 * 4;
    for (int i = tail + gtid; i < count; i += gnt) host_mine[i] = sendbuf[i];

    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        ar_signal_set(arrival_slot(arrival, rank, blockIdx.x), token);
        __threadfence_system();
        while (ar_signal_get(arrival_slot(arrival, peer_rank, blockIdx.x)) != token) {
#if defined(__HIP_DEVICE_COMPILE__)
            __builtin_amdgcn_s_sleep(4);
#endif
        }
    }
    __syncthreads();
    __threadfence_system();

    const float4 * peer4 = reinterpret_cast<const float4 *>(host_peer);
    float4       * recv4 = reinterpret_cast<float4 *>(recvbuf);
    for (int i = gtid; i < count4; i += gnt) {
        const float4 a = src4[i];
        const float4 b = peer4[i];
        float4 o; o.x = a.x + b.x; o.y = a.y + b.y; o.z = a.z + b.z; o.w = a.w + b.w;
        recv4[i] = o;
    }
    for (int i = tail + gtid; i < count; i += gnt) {
        recvbuf[i] = sendbuf[i] + host_peer[i];
    }
}

// --- P2P-direct-load: no host staging for the payload. Signal readiness
// (sendbuf is already fully written before kernel launch, so this is
// immediate) via the same tiny host-mapped arrival array, then directly
// dereference a P2P-mapped pointer into the peer device's own d_send buffer.
__global__ void p2p_direct_kernel(
        const float * sendbuf,
        float       * recvbuf,
        const float * peer_sendbuf,  // P2P-mapped pointer into the OTHER device's memory
        int             rank,
        int             peer_rank,
        int             count,
        int           * arrival,
        int             token) {
    if (threadIdx.x == 0) {
        ar_signal_set(arrival_slot(arrival, rank, blockIdx.x), token);
        __threadfence_system();
        while (ar_signal_get(arrival_slot(arrival, peer_rank, blockIdx.x)) != token) {
#if defined(__HIP_DEVICE_COMPILE__)
            __builtin_amdgcn_s_sleep(4);
#endif
        }
    }
    __syncthreads();
    __threadfence_system();

    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gnt  = gridDim.x * blockDim.x;
    const int count4 = count / 4;
    const float4 * src4  = reinterpret_cast<const float4 *>(sendbuf);
    const float4 * peer4 = reinterpret_cast<const float4 *>(peer_sendbuf);
    float4       * recv4 = reinterpret_cast<float4 *>(recvbuf);
    for (int i = gtid; i < count4; i += gnt) {
        const float4 a = src4[i];
        const float4 b = peer4[i];
        float4 o; o.x = a.x + b.x; o.y = a.y + b.y; o.z = a.z + b.z; o.w = a.w + b.w;
        recv4[i] = o;
    }
    const int tail = count4 * 4;
    for (int i = tail + gtid; i < count; i += gnt) {
        recvbuf[i] = sendbuf[i] + peer_sendbuf[i];
    }
}

struct DeviceState {
    int      device_id;
    float *  d_send   = nullptr;
    float *  d_recv   = nullptr;
    float *  h_stage  = nullptr;  // pinned, mapped (host mode only)
    float *  d_stage  = nullptr;  // device-mapped pointer to h_stage (host mode only)
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
    std::vector<int> element_counts = {7680, 15360, 30720, 61440, 122880};
    int reps = 500;
    int blocks = 8;
    Mode mode = MODE_HOST;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--devices" && i + 1 < argc) {
            devices = parse_int_list(argv[++i]);
        } else if (arg == "--elements" && i + 1 < argc) {
            element_counts = parse_int_list(argv[++i]);
        } else if (arg == "--reps" && i + 1 < argc) {
            reps = std::atoi(argv[++i]);
        } else if (arg == "--blocks" && i + 1 < argc) {
            blocks = std::atoi(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "host") mode = MODE_HOST;
            else if (m == "p2p") mode = MODE_P2P;
            else { fprintf(stderr, "unknown --mode %s (want host|p2p)\n", m.c_str()); return 2; }
        } else {
            fprintf(stderr, "unknown arg: %s\n", arg.c_str());
            return 2;
        }
    }

    if (devices.size() != 2) {
        fprintf(stderr, "this harness is N=2 only (dual-XTX P2P prototype), got %zu devices\n", devices.size());
        return 2;
    }
    if (blocks < 1 || blocks > MAX_BLOCKS) {
        fprintf(stderr, "--blocks must be 1..%d\n", MAX_BLOCKS);
        return 2;
    }

    const int n = 2;
    const int rankA = 0, rankB = 1;

    if (mode == MODE_P2P) {
        int can01 = 0, can10 = 0;
        HIP_CHECK(hipDeviceCanAccessPeer(&can01, devices[0], devices[1]));
        HIP_CHECK(hipDeviceCanAccessPeer(&can10, devices[1], devices[0]));
        if (!can01 || !can10) {
            fprintf(stderr, "--mode p2p requires hipDeviceCanAccessPeer in both directions "
                             "(got %d->%d=%d, %d->%d=%d) -- this device pair is not P2P-capable\n",
                    devices[0], devices[1], can01, devices[1], devices[0], can10);
            return 2;
        }
        HIP_CHECK(hipSetDevice(devices[0]));
        hipError_t e0 = hipDeviceEnablePeerAccess(devices[1], 0);
        if (e0 != hipSuccess && e0 != hipErrorPeerAccessAlreadyEnabled) HIP_CHECK(e0);
        HIP_CHECK(hipSetDevice(devices[1]));
        hipError_t e1 = hipDeviceEnablePeerAccess(devices[0], 0);
        if (e1 != hipSuccess && e1 != hipErrorPeerAccessAlreadyEnabled) HIP_CHECK(e1);
    }

    const char * mode_name = mode == MODE_HOST ? "host-staged" : "p2p-direct-load";
    printf("p2p_direct_allreduce: mode=%s devices=[%d,%d] blocks=%d reps=%d\n",
           mode_name, devices[0], devices[1], blocks, reps);

    for (int elems : element_counts) {
        if (elems % 4 != 0) {
            fprintf(stderr, "elements must be divisible by 4 (float4 kernels, no tail handling beyond count4 boundary)\n");
            return 2;
        }
        const size_t bytes = (size_t) elems * sizeof(float);

        std::vector<DeviceState> ds(n);
        std::vector<std::vector<float>> input(n, std::vector<float>(elems));
        std::mt19937 rng(1234 + elems);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int r = 0; r < n; ++r) {
            for (int i = 0; i < elems; ++i) input[r][i] = dist(rng);
        }
        std::vector<float> reference(elems, 0.0f);
        for (int i = 0; i < elems; ++i) reference[i] = input[0][i] + input[1][i];

        for (int r = 0; r < n; ++r) {
            ds[r].device_id = devices[r];
            HIP_CHECK(hipSetDevice(devices[r]));
            HIP_CHECK(hipStreamCreateWithFlags(&ds[r].stream, hipStreamNonBlocking));
            HIP_CHECK(hipMalloc(&ds[r].d_send, bytes));
            HIP_CHECK(hipMalloc(&ds[r].d_recv, bytes));
            if (mode == MODE_HOST) {
                HIP_CHECK(hipHostMalloc((void **) &ds[r].h_stage, bytes,
                                         hipHostMallocPortable | hipHostMallocMapped));
                HIP_CHECK(hipHostGetDevicePointer((void **) &ds[r].d_stage, ds[r].h_stage, 0));
            }
            HIP_CHECK(hipMemcpy(ds[r].d_send, input[r].data(), bytes, hipMemcpyHostToDevice));
        }

        const size_t arrival_ints = (size_t) n * MAX_BLOCKS * ARRIVAL_STRIDE_INTS;
        int * h_arrival = nullptr;
        HIP_CHECK(hipHostMalloc((void **) &h_arrival, arrival_ints * sizeof(int),
                                 hipHostMallocPortable | hipHostMallocMapped));
        memset(h_arrival, 0, arrival_ints * sizeof(int));

        // per-consumer arrival alias (same pattern as nway_star_allreduce.cpp --
        // hipHostGetDevicePointer resolved with the CONSUMING device current).
        std::vector<int *> d_arrival_by_rank(n);
        // per-consumer stage alias (host mode only)
        std::vector<std::vector<float *>> stage_alias_by_rank(n, std::vector<float *>(n));
        for (int c = 0; c < n; ++c) {
            HIP_CHECK(hipSetDevice(devices[c]));
            HIP_CHECK(hipHostGetDevicePointer((void **) &d_arrival_by_rank[c], h_arrival, 0));
            if (mode == MODE_HOST) {
                for (int owner = 0; owner < n; ++owner) {
                    HIP_CHECK(hipHostGetDevicePointer((void **) &stage_alias_by_rank[c][owner], ds[owner].h_stage, 0));
                }
            }
        }

        std::vector<hipEvent_t> start_ev(n), stop_ev(n);
        for (int r = 0; r < n; ++r) {
            HIP_CHECK(hipSetDevice(devices[r]));
            HIP_CHECK(hipEventCreate(&start_ev[r]));
            HIP_CHECK(hipEventCreate(&stop_ev[r]));
        }

        auto launch = [&](int token) {
            for (int r = 0; r < n; ++r) {
                const int peer = 1 - r;
                HIP_CHECK(hipSetDevice(devices[r]));
                HIP_CHECK(hipEventRecord(start_ev[r], ds[r].stream));
                if (mode == MODE_HOST) {
                    host_staged_kernel<<<blocks, 256, 0, ds[r].stream>>>(
                        ds[r].d_send, ds[r].d_recv, ds[r].d_stage, stage_alias_by_rank[r][peer],
                        r, peer, elems, d_arrival_by_rank[r], token);
                } else {
                    // direct P2P pointer: the peer's own d_send, dereferenced
                    // from THIS device's kernel via HIP's peer-access mapping
                    // (no hipHostGetDevicePointer involved -- this is real
                    // device-to-device addressing, enabled by
                    // hipDeviceEnablePeerAccess above).
                    p2p_direct_kernel<<<blocks, 256, 0, ds[r].stream>>>(
                        ds[r].d_send, ds[r].d_recv, ds[peer].d_send,
                        r, peer, elems, d_arrival_by_rank[r], token);
                }
                HIP_CHECK(hipEventRecord(stop_ev[r], ds[r].stream));
            }
            for (int r = 0; r < n; ++r) {
                HIP_CHECK(hipSetDevice(devices[r]));
                HIP_CHECK(hipStreamSynchronize(ds[r].stream));
            }
        };

        auto validate = [&](const char * tag) -> double {
            double max_abs_err = 0.0;
            std::vector<float> out(elems);
            for (int r = 0; r < n; ++r) {
                HIP_CHECK(hipSetDevice(devices[r]));
                HIP_CHECK(hipMemcpy(out.data(), ds[r].d_recv, bytes, hipMemcpyDeviceToHost));
                int wrong_count = 0;
                int first_bad_i = -1;
                float first_bad_got = 0.0f, first_bad_want = 0.0f;
                const double per_elem_eps = 2e-4;
                for (int i = 0; i < elems; ++i) {
                    double err = std::fabs(out[i] - reference[i]);
                    max_abs_err = std::max(max_abs_err, err);
                    if (err >= per_elem_eps) {
                        wrong_count++;
                        if (first_bad_i < 0) { first_bad_i = i; first_bad_got = out[i]; first_bad_want = reference[i]; }
                    }
                }
                if (wrong_count > 0) {
                    fprintf(stderr, "[%s] rank=%d(device=%d) wrong_count=%d/%d first_bad_i=%d got=%.6f want=%.6f\n",
                        tag, r, devices[r], wrong_count, elems, first_bad_i, first_bad_got, first_bad_want);
                }
            }
            return max_abs_err;
        };

        int token = 1;
        launch(token); // warmup
        double max_abs_err = validate("warmup");
        const double eps = 2e-4;

        std::vector<double> elapsed_ms;
        elapsed_ms.reserve(reps);
        for (int rep = 0; rep < reps; ++rep) {
            token++;
            launch(token);
            char tag[32];
            snprintf(tag, sizeof(tag), "rep%d", rep);
            max_abs_err = std::max(max_abs_err, validate(tag));

            float max_ms = 0.0f;
            for (int r = 0; r < n; ++r) {
                float ms = 0.0f;
                HIP_CHECK(hipEventElapsedTime(&ms, start_ev[r], stop_ev[r]));
                max_ms = std::max(max_ms, ms);
            }
            elapsed_ms.push_back(max_ms);
        }
        bool correct = max_abs_err < eps;

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
            if (mode == MODE_HOST) hipHostFree(ds[r].h_stage);
            hipStreamDestroy(ds[r].stream);
        }
        hipHostFree(h_arrival);
    }

    return 0;
}
