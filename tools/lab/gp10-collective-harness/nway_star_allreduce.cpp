// GP10: base-level (sub-llama.cpp) test harness for collective AllReduce
// transforms. See vendor/llama.cpp/ggml/src/ggml-cuda/allreduce.cu (patch
// 1001_hip_internal_allreduce) for the real production 2-GPU mechanism this
// generalizes from, and docs/planning/active/gpu-collectives/GP11.md for
// full context.
//
// Two topologies, selected with --mode:
//
//   star  -- every rank spins on and reads every other rank (N-1 serial
//            blocking peer reads each). Direct N-way extension of 1001's
//            pairwise mechanism. First experiment; real hardware showed
//            latency scaling roughly linearly in N-1.
//
//   root  -- fused reduce-to-root + broadcast: leaves each do exactly one
//            handshake with a single root rank; root reduces and republishes
//            its result for leaves to read back, in ONE kernel launch per
//            rank (no CPU-side phase transition), per gpt-dev-agent's
//            2026-09-02 design review. Root's PCIe endpoint is a real
//            bottleneck (root reads every leaf's contribution through its
//            own single link), so this does NOT use per-leaf concurrent
//            streams on the root side -- see GP11 notes for why that
//            wouldn't help.
//
// 2026-09-02 revisions per reviewer-gpt-agent's adversarial pass on the
// original star-only version (req_25e6393fd0264cd0):
//   - arrival slots are now genuinely cache-line-separated per (rank,
//     block) -- the original ARRIVAL_STRIDE_INTS constant was defined but
//     never applied to the actual allocation/indexing, so all ranks'
//     tokens shared one cache line and could manufacture false-sharing
//     cost that inflates N=3 vs N=2 results.
//   - correctness is now checked for EVERY rank and EVERY timed rep (was:
//     rank 0 only, warmup rep only).
//   - --blocks lets the kernel launch with a production-shaped multi-block
//     grid (default 8, matching GGML_CUDA_AR_KERNEL_BLOCKS in allreduce.cu)
//     with vectorized float4 copies, so the star-vs-root comparison isn't
//     confounded by a deliberately naive single-block kernel.
//
// This is NOT production code and does not touch the patched llama.cpp
// source tree at all -- it links directly against ROCm/HIP, compiles in
// seconds, and runs in well under a minute for a full sweep.
//
// Build:
//   hipcc -O3 -std=c++17 -o nway_star_allreduce nway_star_allreduce.cpp
// Run:
//   ./nway_star_allreduce --mode star --devices 0,1,2 --elements 30720,2621440 --reps 20 --blocks 8
//   ./nway_star_allreduce --mode root --root 0 --devices 0,1,2 --elements 30720,2621440 --reps 20 --blocks 8

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
static constexpr int MAX_BLOCKS  = 32;
// 64 bytes = one cache line; each (rank, block) token gets its own line so
// polling ranks don't false-share a line with another rank's writer.
static constexpr int ARRIVAL_STRIDE_INTS = 16;

enum Mode { MODE_STAR = 0, MODE_ROOT = 1, MODE_ROOT3 = 2 };

static __device__ __forceinline__ void ar_signal_set(int * p, int token) {
    *(volatile int *) p = token;
}
static __device__ __forceinline__ int ar_signal_get(const int * p) {
    return *(const volatile int *) p;
}
static __device__ __forceinline__ int * arrival_slot(int * arrival, int n_devices, int rank, int block) {
    (void) n_devices;
    return arrival + (rank * MAX_BLOCKS + block) * ARRIVAL_STRIDE_INTS;
}

// Vectorized copy loop: float4 for the bulk, scalar tail. gridDim.x blocks x
// blockDim.x threads stripe the full [0, count) range.
static __device__ __forceinline__ void vec_copy(const float * src, float * dst, int count) {
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gnt  = gridDim.x * blockDim.x;
    const int count4 = count / 4;
    const float4 * src4 = reinterpret_cast<const float4 *>(src);
    float4       * dst4 = reinterpret_cast<float4 *>(dst);
    for (int i = gtid; i < count4; i += gnt) {
        dst4[i] = src4[i];
    }
    const int tail_start = count4 * 4;
    for (int i = tail_start + gtid; i < count; i += gnt) {
        dst[i] = src[i];
    }
}

// --- star: every rank reads every other rank -------------------------------
__global__ void star_kernel(
        const float * sendbuf,
        float       * recvbuf,
        float       * host_mine,
        float * const * host_peers,
        int             n_devices,
        int             rank,
        int             count,
        int           * arrival,
        int             token) {
    vec_copy(sendbuf, host_mine, count);
    __threadfence_system();
    __syncthreads();

    if (threadIdx.x == 0) {
        ar_signal_set(arrival_slot(arrival, n_devices, rank, blockIdx.x), token);
        __threadfence_system();
        for (int peer = 0; peer < n_devices; ++peer) {
            if (peer == rank) continue;
            while (ar_signal_get(arrival_slot(arrival, n_devices, peer, blockIdx.x)) != token) {
#if defined(__HIP_DEVICE_COMPILE__)
                __builtin_amdgcn_s_sleep(4);
#endif
            }
        }
    }
    __syncthreads();
    __threadfence_system();

    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gnt  = gridDim.x * blockDim.x;
    for (int i = gtid; i < count; i += gnt) {
        float acc = sendbuf[i];
        for (int peer = 0; peer < n_devices; ++peer) {
            if (peer == rank) continue;
            acc += host_peers[peer][i];
        }
        recvbuf[i] = acc;
    }
}

// --- root: leaves handshake with root only; root reduces + republishes -----
// Leaf: stage -> signal -> wait for root's token -> read root's stage.
// Root: wait for every leaf's token -> sum local + all leaf stages -> write
//       recvbuf AND republish the same sum into its own stage (so leaves can
//       read it back) -> signal.
__global__ void root_kernel(
        const float * sendbuf,
        float       * recvbuf,
        float       * host_mine,       // this rank's pinned stage slot
        float * const * host_peers,    // all ranks' stage slots
        int             n_devices,
        int             rank,
        int             root,
        int             count,
        int           * arrival,
        int             token) {
    const bool is_root = (rank == root);

    if (!is_root) {
        vec_copy(sendbuf, host_mine, count);
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            ar_signal_set(arrival_slot(arrival, n_devices, rank, blockIdx.x), token);
            __threadfence_system(); // publish the token itself system-wide (matches production)
        }
    } else {
        __syncthreads();
    }

    if (threadIdx.x == 0 && is_root) {
        for (int leaf = 0; leaf < n_devices; ++leaf) {
            if (leaf == root) continue;
            while (ar_signal_get(arrival_slot(arrival, n_devices, leaf, blockIdx.x)) != token) {
#if defined(__HIP_DEVICE_COMPILE__)
                __builtin_amdgcn_s_sleep(4);
#endif
            }
        }
    }
    __syncthreads();
    __threadfence_system();

    if (is_root) {
        const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
        const int gnt  = gridDim.x * blockDim.x;
        // N=3 fast path (per gpt-dev-agent's optimization, req_27af96122a244da5):
        // load both leaf buffers independently before summing, so both mapped-
        // host reads can be outstanding simultaneously instead of a serial
        // dependent-add loop over leaves.
        if (n_devices == 3) {
            int leaf0 = -1, leaf1 = -1;
            for (int p = 0; p < 3; ++p) {
                if (p == root) continue;
                if (leaf0 < 0) leaf0 = p; else leaf1 = p;
            }
            const int count4 = count / 4;
            const float4 * __restrict__ src4 = reinterpret_cast<const float4 *>(sendbuf);
            const float4 * __restrict__ l04  = reinterpret_cast<const float4 *>(host_peers[leaf0]);
            const float4 * __restrict__ l14  = reinterpret_cast<const float4 *>(host_peers[leaf1]);
            float4       * __restrict__ dst4 = reinterpret_cast<float4 *>(recvbuf);
            float4       * __restrict__ pub4 = reinterpret_cast<float4 *>(host_mine);
            for (int i = gtid; i < count4; i += gnt) {
                const float4 a = l04[i];
                const float4 b = l14[i];
                const float4 s = src4[i];
                float4 out;
                out.x = s.x + a.x + b.x;
                out.y = s.y + a.y + b.y;
                out.z = s.z + a.z + b.z;
                out.w = s.w + a.w + b.w;
                dst4[i] = out;
                pub4[i] = out;
            }
            const int tail = count4 * 4;
            for (int i = tail + gtid; i < count; i += gnt) {
                const float out = sendbuf[i] + host_peers[leaf0][i] + host_peers[leaf1][i];
                recvbuf[i] = out;
                host_mine[i] = out;
            }
        } else {
            for (int i = gtid; i < count; i += gnt) {
                float acc = sendbuf[i];
                for (int leaf = 0; leaf < n_devices; ++leaf) {
                    if (leaf == root) continue;
                    acc += host_peers[leaf][i];
                }
                recvbuf[i] = acc;
                host_mine[i] = acc; // republish for leaves
            }
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            ar_signal_set(arrival_slot(arrival, n_devices, root, blockIdx.x), token);
            __threadfence_system(); // publish the token itself system-wide (matches production)
        }
    } else {
        if (threadIdx.x == 0) {
            while (ar_signal_get(arrival_slot(arrival, n_devices, root, blockIdx.x)) != token) {
#if defined(__HIP_DEVICE_COMPILE__)
                __builtin_amdgcn_s_sleep(4);
#endif
            }
        }
        __syncthreads();
        __threadfence_system();
        vec_copy(host_peers[root], recvbuf, count);
    }
}

// --- root3: specialized N=3-only root/leaf kernels (P1+P2, gpt-dev-agent -----
// design, req_f4cb36cb22fd48a6 / req_1f050a8abef749a2) -- no generic
// is_root/n_devices/root branching in the hot path; fixed float4 signatures;
// root's two leaf-token waits are polled by threadIdx 0 and 1 concurrently
// instead of serially (P2). Uses the SAME arrival_slot() per-block indexing
// as the generic kernels above -- this is the exact invariant production's
// kernel3 initially missed, confirmed still correct here by gpt's review.
__global__ void root3_root_kernel(
        const float4 * __restrict__ send,
        float4       * __restrict__ recv,
        float4       * __restrict__ publish,
        const float4 * __restrict__ leaf0,
        const float4 * __restrict__ leaf1,
        int                          count4,
        int *                        leaf0_ready,
        int *                        leaf1_ready,
        int *                        root_ready,
        int                          token) {
    if (threadIdx.x == 0) {
        while (ar_signal_get(leaf0_ready + blockIdx.x * ARRIVAL_STRIDE_INTS) != token) {
#if defined(__HIP_DEVICE_COMPILE__)
            __builtin_amdgcn_s_sleep(4);
#endif
        }
    }
    if (threadIdx.x == 1) {
        while (ar_signal_get(leaf1_ready + blockIdx.x * ARRIVAL_STRIDE_INTS) != token) {
#if defined(__HIP_DEVICE_COMPILE__)
            __builtin_amdgcn_s_sleep(4);
#endif
        }
    }
    __syncthreads();
    __threadfence_system();

    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gnt  = gridDim.x * blockDim.x;
    for (int i = gtid; i < count4; i += gnt) {
        const float4 a = leaf0[i];
        const float4 b = leaf1[i];
        const float4 s = send[i];
        float4 o;
        o.x = s.x + a.x + b.x;
        o.y = s.y + a.y + b.y;
        o.z = s.z + a.z + b.z;
        o.w = s.w + a.w + b.w;
        recv[i]    = o;
        publish[i] = o;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        ar_signal_set(root_ready + blockIdx.x * ARRIVAL_STRIDE_INTS, token);
        __threadfence_system();
    }
}

__global__ void root3_leaf_kernel(
        const float4 * __restrict__ send,
        float4       * __restrict__ recv,
        float4       * __restrict__ mine,
        const float4 * __restrict__ root_stage,
        int                          count4,
        int *                        mine_ready,
        int *                        root_ready,
        int                          token) {
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gnt  = gridDim.x * blockDim.x;
    for (int i = gtid; i < count4; i += gnt) {
        mine[i] = send[i];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        ar_signal_set(mine_ready + blockIdx.x * ARRIVAL_STRIDE_INTS, token);
        __threadfence_system();
    }

    if (threadIdx.x == 0) {
        while (ar_signal_get(root_ready + blockIdx.x * ARRIVAL_STRIDE_INTS) != token) {
#if defined(__HIP_DEVICE_COMPILE__)
            __builtin_amdgcn_s_sleep(4);
#endif
        }
    }
    __syncthreads();
    __threadfence_system();

    for (int i = gtid; i < count4; i += gnt) {
        recv[i] = root_stage[i];
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
    int blocks = 8;
    int root = 0;
    Mode mode = MODE_STAR;

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
        } else if (arg == "--root" && i + 1 < argc) {
            root = std::atoi(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "star") mode = MODE_STAR;
            else if (m == "root") mode = MODE_ROOT;
            else if (m == "root3") mode = MODE_ROOT3;
            else { fprintf(stderr, "unknown --mode %s\n", m.c_str()); return 2; }
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
    if (blocks < 1 || blocks > MAX_BLOCKS) {
        fprintf(stderr, "--blocks must be 1..%d\n", MAX_BLOCKS);
        return 2;
    }
    if ((mode == MODE_ROOT || mode == MODE_ROOT3) && (root < 0 || root >= n)) {
        fprintf(stderr, "--root must index into --devices\n");
        return 2;
    }
    if (mode == MODE_ROOT3 && n != 3) {
        fprintf(stderr, "--mode root3 requires exactly 3 --devices\n");
        return 2;
    }

    const char * mode_name = mode == MODE_STAR ? "star" : mode == MODE_ROOT ? "root" : "root3";
    printf("nway_%s_allreduce: devices=[", mode_name);
    for (int i = 0; i < n; ++i) printf("%d%s", devices[i], i + 1 < n ? "," : "");
    printf("] blocks=%d reps=%d%s\n", blocks, reps,
           (mode == MODE_ROOT || mode == MODE_ROOT3) ? (" root_rank_index=" + std::to_string(root)).c_str() : "");

    for (int elems : element_counts) {
        if (mode == MODE_ROOT3 && elems % 4 != 0) {
            fprintf(stderr, "--mode root3 requires elements divisible by 4 (no tail handling in this specialized kernel)\n");
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
        for (int i = 0; i < elems; ++i) {
            float acc = 0.0f;
            for (int r = 0; r < n; ++r) acc += input[r][i];
            reference[i] = acc;
        }

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

        // arrival: n_devices * MAX_BLOCKS slots, ARRIVAL_STRIDE_INTS apart --
        // real cache-line separation per (rank, block).
        const size_t arrival_ints = (size_t) n * MAX_BLOCKS * ARRIVAL_STRIDE_INTS;
        int * h_arrival = nullptr;
        HIP_CHECK(hipHostMalloc((void **) &h_arrival, arrival_ints * sizeof(int),
                                 hipHostMallocPortable | hipHostMallocMapped));
        memset(h_arrival, 0, arrival_ints * sizeof(int));

        // 2026-09-02 fix (per gpt-dev-agent root-cause analysis, req_27af96122a244da5):
        // hipHostGetDevicePointer() must be called with the CONSUMING device
        // current -- an alias resolved once under one device's context is not
        // guaranteed valid when dereferenced by a kernel on a DIFFERENT
        // device. The original version called it once (with whichever device
        // happened to be current from the alloc loop above) and shared that
        // single alias across every rank's kernel launch -- a real, rare
        // cross-device mapped-pointer bug, not a fence/ordering race. Resolve
        // a separate alias table per CONSUMING device instead.
        std::vector<int *>   d_arrival_by_rank(n);
        std::vector<float **> d_peer_table_by_rank(n);
        // host-side mirror of the same per-consumer aliases, for root3's
        // specialized kernels which take direct pointer args (no on-device
        // table lookup) -- stage_alias_by_rank[c][owner].
        std::vector<std::vector<float *>> stage_alias_by_rank(n, std::vector<float *>(n));
        for (int c = 0; c < n; ++c) {
            HIP_CHECK(hipSetDevice(devices[c]));
            HIP_CHECK(hipHostGetDevicePointer((void **) &d_arrival_by_rank[c], h_arrival, 0));

            std::vector<float *> aliases(n);
            for (int owner = 0; owner < n; ++owner) {
                HIP_CHECK(hipHostGetDevicePointer((void **) &aliases[owner], ds[owner].h_stage, 0));
            }
            stage_alias_by_rank[c] = aliases;
            HIP_CHECK(hipMalloc(&d_peer_table_by_rank[c], n * sizeof(float *)));
            HIP_CHECK(hipMemcpy(d_peer_table_by_rank[c], aliases.data(), n * sizeof(float *),
                                 hipMemcpyHostToDevice));
        }

        std::vector<hipEvent_t> start_ev(n), stop_ev(n);
        for (int r = 0; r < n; ++r) {
            HIP_CHECK(hipSetDevice(devices[r]));
            HIP_CHECK(hipEventCreate(&start_ev[r]));
            HIP_CHECK(hipEventCreate(&stop_ev[r]));
        }

        auto launch = [&](int token) {
            for (int r = 0; r < n; ++r) {
                HIP_CHECK(hipSetDevice(devices[r]));
                HIP_CHECK(hipEventRecord(start_ev[r], ds[r].stream));
                if (mode == MODE_STAR) {
                    star_kernel<<<blocks, 256, 0, ds[r].stream>>>(
                        ds[r].d_send, ds[r].d_recv, ds[r].d_stage, d_peer_table_by_rank[r],
                        n, r, elems, d_arrival_by_rank[r], token);
                } else if (mode == MODE_ROOT) {
                    root_kernel<<<blocks, 256, 0, ds[r].stream>>>(
                        ds[r].d_send, ds[r].d_recv, ds[r].d_stage, d_peer_table_by_rank[r],
                        n, r, root, elems, d_arrival_by_rank[r], token);
                } else {
                    // root3: specialized fixed-signature kernels, float4-only,
                    // direct pointer args (no on-device table lookup). Base
                    // arrival pointer per rank = arrival + rank*MAX_BLOCKS*STRIDE
                    // (the kernel itself adds the block offset), matching
                    // arrival_slot()'s own split.
                    auto rank_base = [&](int rank) {
                        return d_arrival_by_rank[r] + rank * MAX_BLOCKS * ARRIVAL_STRIDE_INTS;
                    };
                    const int count4 = elems / 4;
                    if (r == root) {
                        int leaf0 = -1, leaf1 = -1;
                        for (int p = 0; p < 3; ++p) {
                            if (p == root) continue;
                            if (leaf0 < 0) leaf0 = p; else leaf1 = p;
                        }
                        root3_root_kernel<<<blocks, 256, 0, ds[r].stream>>>(
                            reinterpret_cast<const float4 *>(ds[r].d_send),
                            reinterpret_cast<float4 *>(ds[r].d_recv),
                            reinterpret_cast<float4 *>(ds[r].d_stage),
                            reinterpret_cast<const float4 *>(stage_alias_by_rank[r][leaf0]),
                            reinterpret_cast<const float4 *>(stage_alias_by_rank[r][leaf1]),
                            count4,
                            rank_base(leaf0), rank_base(leaf1), rank_base(root),
                            token);
                    } else {
                        root3_leaf_kernel<<<blocks, 256, 0, ds[r].stream>>>(
                            reinterpret_cast<const float4 *>(ds[r].d_send),
                            reinterpret_cast<float4 *>(ds[r].d_recv),
                            reinterpret_cast<float4 *>(ds[r].d_stage),
                            reinterpret_cast<const float4 *>(stage_alias_by_rank[r][root]),
                            count4,
                            rank_base(r), rank_base(root),
                            token);
                    }
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
                const double per_elem_eps = 1e-4 * n;
                for (int i = 0; i < elems; ++i) {
                    double err = std::fabs(out[i] - reference[i]);
                    max_abs_err = std::max(max_abs_err, err);
                    if (err >= per_elem_eps) {
                        wrong_count++;
                        if (first_bad_i < 0) {
                            first_bad_i = i;
                            first_bad_got = out[i];
                            first_bad_want = reference[i];
                        }
                    }
                }
                if (wrong_count > 0) {
                    const int elems_per_block = (blocks * 256 == 0) ? elems : std::max(1, elems / blocks);
                    fprintf(stderr,
                        "[%s] rank=%d(device=%d) wrong_count=%d/%d first_bad_i=%d got=%.6f want=%.6f "
                        "(approx block owner=%d, elems_per_block~%d)\n",
                        tag, r, devices[r], wrong_count, elems, first_bad_i, first_bad_got, first_bad_want,
                        first_bad_i / elems_per_block, elems_per_block);
                }
            }
            return max_abs_err;
        };

        int token = 1;
        launch(token); // warmup
        double max_abs_err = validate("warmup");
        const double eps = 1e-4 * n;

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
            hipHostFree(ds[r].h_stage);
            hipFree(d_peer_table_by_rank[r]);
            hipStreamDestroy(ds[r].stream);
        }
        hipHostFree(h_arrival);
    }

    return 0;
}
