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
// --- 2026-09-04 update: --mode p2p produced a deterministic wrong result
// (remote contribution reads back as exactly zero, not corrupted noise --
// see GP11.md). Escalated to gpt-dev-agent (req_877e043460c94bd8), which
// pointed out ROCm's own kernel-side-P2P tests (RCCL's p2p_latency_test)
// allocate the peer-visible buffer with hipExtMallocWithFlags(...,
// hipDeviceMallocFinegrained) under HSA_FORCE_FINE_GRAIN_PCIE=1, NOT plain
// coarse-grain hipMalloc -- so the failure may be a coarse-memory
// peer-visibility/coherency gap rather than "kernel-side P2P doesn't work
// on this stack." Added minimal single-thread discriminator probes below
// per gpt's recommended test matrix, run BEFORE building any memcpy-based
// fallback design:
//
// --mode probe-coarse -- <<<1,1>>> kernel does out[0] = peer[0] against an
//                  ordinary coarse-grain hipMalloc peer buffer. Peer buffer's
//                  element 0 is overwritten with a fresh sentinel (host
//                  hipMemcpy, synchronous) immediately before every rep, so
//                  the read result classifies as current / stale / zero
//                  instead of just "wrong" (the AllReduce test above reuses
//                  static input every rep, which can't distinguish those).
// === RESOLVED 2026-09-04: why every P2P design fails on this box ===========
//
// Systematic diagnosis (scratch programs p2p_diag / p2p_write / p2p_coh /
// p2p_spin / p2p_d2d / p2p_bwval, run on brutus 0<->1) established a clean and
// consistent rule, with a local-read control arm proving the harness itself is
// sound:
//
//   PULL operations silently return ZEROS (no error, no fault):
//     - kernel-side peer reads, both directions
//     - hipMemcpy(...) where the CURRENT device is the DESTINATION
//     - hipMemcpyPeer(), in BOTH directions, at every size tested
//   PUSH operations work correctly:
//     - kernel-side peer writes (validated exact to 67 MB, scalar and float4)
//     - hipMemcpy(...) where the CURRENT device is the SOURCE
//
// This is the classic PCIe split: posted writes traverse the path, non-posted
// reads (which need a completion round-trip) do not. It is exactly why
// NCCL/RCCL are built around pushing into remote buffers, never pulling.
//
// BUT the push direction is still unusable for a low-latency collective:
// peer-directed writes do NOT become visible to the other GPU until the
// WRITING KERNEL RETIRES. __threadfence_system() does not flush them. Proven
// by having one GPU spin on a flag while a concurrently-resident kernel on the
// other GPU wrote it: 2,000,000 iterations each of sys-scope acquire load,
// system-scope atomic RMW, non-temporal load, volatile load, and
// fence-then-load ALL failed to observe the write, while a host readback after
// the writer retired saw it immediately. The atomic-RMW arm is the decisive
// one -- an RMW cannot be served from a stale cache line, so the data is
// genuinely not in the destination's memory yet; this is a writer-side flush
// problem, not a reader-side cache-coherency problem.
//
// Consequence: NO single-kernel, fine-grained handshake collective is possible
// across these two GPUs -- pull-based (--mode p2p) or push-based
// (--mode p2p-push) alike. P2P here can only work as
// push-kernel -> kernel boundary -> reduce-kernel, whose launch and
// synchronisation overhead exceeds the entire theoretical saving at decode
// message sizes. The production pinned-host design works precisely BECAUSE
// host-memory writes are immediately visible to the peer.
//
// Corollary worth recording: an earlier "10.4 GB/s P2P, 1.5x faster than
// host-staged" measurement from a separate agent's benchmark is INVALID -- that
// benchmark did no correctness validation and used the pull configuration
// (current device = destination), so it was timing a copy that transferred
// nothing. Re-running its exact configuration with validation added shows
// 262144/262144 elements wrong (all zeros).
//
// --mode p2p-push -- NCCL-style push design (write into the peer's exchange
//                  buffer, signal a flag in peer VRAM, spin on our own local
//                  flag, then reduce from two local operands). Correct in
//                  principle and in the direction that works, but deadlocks
//                  here for the flush reason above; the spin is bounded so it
//                  reports the failure instead of hanging the box.
//
// --mode probe-fine -- identical probe, but the peer buffer is allocated
//                  with hipExtMallocWithFlags(..., hipDeviceMallocFinegrained).
//                  Run with HSA_FORCE_FINE_GRAIN_PCIE=1 in the environment
//                  (the harness warns if it isn't set).
// --peer-order enable-first|alloc-first -- order of hipDeviceEnablePeerAccess
//                  vs the peer buffer allocation. Default enable-first
//                  (matches today's --mode p2p and probe-* default). Pass
//                  alloc-first to test gpt's Arm C: rule out a runtime bug
//                  where only allocations made AFTER peer-enable get mapped
//                  (not required by the HIP API -- CLR is documented to
//                  update pre-existing allocations too -- but worth ruling
//                  out empirically). Applies to --mode p2p and both probes.
//
// Build:
//   hipcc -O3 -std=c++17 -o p2p_direct_allreduce p2p_direct_allreduce.cpp
// Run:
//   ./p2p_direct_allreduce --devices 0,1 --elements 7680,15360,30720,61440,122880 --reps 500 --mode p2p
//   ./p2p_direct_allreduce --devices 0,1 --elements 7680,15360,30720,61440,122880 --reps 500 --mode host
//   ./p2p_direct_allreduce --devices 0,1 --reps 50 --mode probe-coarse
//   HSA_FORCE_FINE_GRAIN_PCIE=1 ./p2p_direct_allreduce --devices 0,1 --reps 50 --mode probe-fine
//   ./p2p_direct_allreduce --devices 0,1 --reps 50 --mode probe-coarse --peer-order alloc-first

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

enum Mode { MODE_HOST = 0, MODE_P2P = 1, MODE_PROBE_COARSE = 2, MODE_PROBE_FINE = 3,
            MODE_P2P_PUSH = 4 };
enum PeerOrder { ORDER_ENABLE_FIRST = 0, ORDER_ALLOC_FIRST = 1 };

static __device__ __forceinline__ void ar_signal_set(int * p, int token) {
    *(volatile int *) p = token;
}
static __device__ __forceinline__ int ar_signal_get(const int * p) {
    return *(const volatile int *) p;
}
static __device__ __forceinline__ int * arrival_slot(int * arrival, int rank, int block) {
    return arrival + (rank * MAX_BLOCKS + block) * ARRIVAL_STRIDE_INTS;
}

// System-scope signalling for the p2p-push path. The KFD topology reports this
// GPU<->GPU link as NON_COHERENT (io_link flags=3), so a peer's PCIe write
// lands in our VRAM while our L2 can keep serving a stale line -- a plain
// volatile load spins forever. System-scope acquire/release atomics emit the
// cache invalidate/writeback needed to actually observe the peer's write.
static __device__ __forceinline__ void ar_signal_set_sys(int * p, int token) {
    __hip_atomic_store(p, token, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
static __device__ __forceinline__ int ar_signal_get_sys(const int * p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
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

// --- P2P PUSH design (2026-09-04). Diagnostics established that on this box
// peer READS return zero (kernel-side and via hipMemcpyPeer/D2D alike) while
// kernel-side peer WRITES work correctly and validate at every size. This is
// the classic PCIe posted-write-works / non-posted-read-fails split, and it is
// exactly why NCCL/RCCL are architected around pushing into remote buffers
// rather than pulling from them.
//
// Design (NCCL-style, all operations in the direction that actually works):
//   1. each rank vector-stores its payload INTO the peer's exchange buffer
//      (peer VRAM) -- a peer WRITE.
//   2. __threadfence_system(), then the same block writes a per-block flag
//      token INTO the peer's flag array (also peer VRAM). PCIe posted writes
//      to the same target stay ordered, so the flag cannot land before the
//      payload it guards.
//   3. each rank spins on its OWN LOCAL flag (a local read -- fast, and works)
//      until the peer's token arrives.
//   4. sum: recvbuf[i] = sendbuf[i] + my_exchange[i]. BOTH operands are in
//      local VRAM, so the reduction runs at full local bandwidth with no
//      peer reads at all.
//
// Note this also removes the pinned-host arrival handshake entirely -- the
// flag lives in device memory rather than host memory.
__global__ void p2p_push_kernel(
        const float * sendbuf,
        float       * recvbuf,
        float       * peer_exchange,   // peer VRAM: where WE write our payload
        const float * my_exchange,     // local VRAM: where the PEER wrote its payload
        int         * peer_flag,       // peer VRAM: flag array we signal into
        const int   * my_flag,         // local VRAM: flag array we spin on
        int             count,
        int             token,
        int         *   deadlock_flag) {
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gnt  = gridDim.x * blockDim.x;
    const int count4 = count / 4;
    const int tail   = count4 * 4;

    // 1. push our payload into the peer's exchange buffer (peer WRITE)
    const float4 * src4  = reinterpret_cast<const float4 *>(sendbuf);
    float4       * peer4 = reinterpret_cast<float4 *>(peer_exchange);
    for (int i = gtid; i < count4; i += gnt) peer4[i] = src4[i];
    for (int i = tail + gtid; i < count; i += gnt) peer_exchange[i] = sendbuf[i];

    // 2. fence, then signal the peer by writing our token into ITS flag array
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        ar_signal_set_sys(peer_flag + blockIdx.x * ARRIVAL_STRIDE_INTS, token);
        // 3. spin on our OWN local flag (system-scope acquire load -- see note
        // on ar_signal_get_sys: the link is NON_COHERENT, a volatile load
        // would spin on a stale L2 line forever)
        // Bounded spin: on this box the peer's write does NOT become visible
        // while the writing kernel is still resident (see file header), so an
        // unbounded spin deadlocks both GPUs. Bail out and report instead.
        long long budget = 20000000LL;
        while (ar_signal_get_sys(my_flag + blockIdx.x * ARRIVAL_STRIDE_INTS) != token) {
            if (--budget <= 0) {
                __hip_atomic_store(deadlock_flag, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                break;
            }
#if defined(__HIP_DEVICE_COMPILE__)
            __builtin_amdgcn_s_sleep(4);
#endif
        }
    }
    __syncthreads();
    __threadfence_system();

    // 4. reduce -- both operands local
    const float4 * mine4 = reinterpret_cast<const float4 *>(my_exchange);
    float4       * recv4 = reinterpret_cast<float4 *>(recvbuf);
    for (int i = gtid; i < count4; i += gnt) {
        const float4 a = src4[i];
        const float4 b = mine4[i];
        float4 o; o.x = a.x + b.x; o.y = a.y + b.y; o.z = a.z + b.z; o.w = a.w + b.w;
        recv4[i] = o;
    }
    for (int i = tail + gtid; i < count; i += gnt) {
        recvbuf[i] = sendbuf[i] + my_exchange[i];
    }
}

// --- minimal single-thread peer-read discriminator (gpt-dev-agent
// req_877e043460c94bd8) -- no handshake, no vectorization, just:
// out[0] = peer[0]. Isolates whether the AllReduce kernel's failure is a
// mapping/coherency problem versus a race/vectorization artifact.
__global__ void probe_read_kernel(float * out, const float * peer) {
    out[0] = peer[0];
}

struct DeviceState {
    int      device_id;
    float *  d_send   = nullptr;
    float *  d_recv   = nullptr;
    float *  h_stage  = nullptr;  // pinned, mapped (host mode only)
    float *  d_stage  = nullptr;  // device-mapped pointer to h_stage (host mode only)
    float *  d_exch   = nullptr;  // local VRAM exchange buffer (p2p-push mode)
    int   *  d_flag   = nullptr;  // local VRAM flag array (p2p-push mode)
    int   *  d_dead   = nullptr;  // set by the kernel if its spin budget expires
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

// Runs gpt's minimal peer-read discriminator: rankA's out[0] = rankB's
// peer[0], where peer[0] is set to a fresh sentinel via host hipMemcpy
// immediately before every rep. Classifies each rep's result as
// current (mapping/coherency fine), stale (sees a PREVIOUS sentinel --
// visibility/coherency lag), zero (permanently reads 0 -- bad/inaccessible
// mapping), or other (neither -- report raw value).
static int run_probe(const std::vector<int> & devices, bool fine_grain,
                      PeerOrder peer_order, int reps) {
    if (fine_grain && getenv("HSA_FORCE_FINE_GRAIN_PCIE") == nullptr) {
        fprintf(stderr, "WARNING: --mode probe-fine without HSA_FORCE_FINE_GRAIN_PCIE=1 set "
                         "in the environment -- gpt-dev-agent's recommended repro requires it.\n");
    }

    const int rankA = 0, rankB = 1; // A reads B's buffer

    auto enable_peer = [&]() {
        HIP_CHECK(hipSetDevice(devices[rankA]));
        hipError_t eA = hipDeviceEnablePeerAccess(devices[rankB], 0);
        if (eA != hipSuccess && eA != hipErrorPeerAccessAlreadyEnabled) HIP_CHECK(eA);
        HIP_CHECK(hipSetDevice(devices[rankB]));
        hipError_t eB = hipDeviceEnablePeerAccess(devices[rankA], 0);
        if (eB != hipSuccess && eB != hipErrorPeerAccessAlreadyEnabled) HIP_CHECK(eB);
    };

    if (peer_order == ORDER_ENABLE_FIRST) enable_peer();

    HIP_CHECK(hipSetDevice(devices[rankA]));
    float * d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, sizeof(float)));

    HIP_CHECK(hipSetDevice(devices[rankB]));
    float * d_peer = nullptr;
    if (fine_grain) {
        HIP_CHECK(hipExtMallocWithFlags((void **) &d_peer, sizeof(float), hipDeviceMallocFinegrained));
    } else {
        HIP_CHECK(hipMalloc(&d_peer, sizeof(float)));
    }

    if (peer_order == ORDER_ALLOC_FIRST) enable_peer();

    printf("probe: mode=%s peer_order=%s devices=[%d,%d] reps=%d\n",
           fine_grain ? "probe-fine" : "probe-coarse",
           peer_order == ORDER_ENABLE_FIRST ? "enable-first" : "alloc-first",
           devices[rankA], devices[rankB], reps);

    int n_current = 0, n_stale = 0, n_zero = 0, n_other = 0;
    float prev_sentinel = 0.0f;
    for (int rep = 0; rep < reps; ++rep) {
        const float sentinel = 1.0f + (float) rep; // 1,2,3,... -- never 0
        HIP_CHECK(hipSetDevice(devices[rankB]));
        HIP_CHECK(hipMemcpy(d_peer, &sentinel, sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipSetDevice(devices[rankA]));
        probe_read_kernel<<<1, 1>>>(d_out, d_peer);
        HIP_CHECK(hipDeviceSynchronize());

        float got = 0.0f;
        HIP_CHECK(hipMemcpy(&got, d_out, sizeof(float), hipMemcpyDeviceToHost));

        const char * cls;
        if (got == sentinel) { cls = "current"; n_current++; }
        else if (rep > 0 && got == prev_sentinel) { cls = "stale"; n_stale++; }
        else if (got == 0.0f) { cls = "zero"; n_zero++; }
        else { cls = "other"; n_other++; }

        if (rep < 5 || cls[0] != 'c') {
            printf("  rep=%-4d sentinel=%.1f got=%.6f class=%s\n", rep, sentinel, got, cls);
        }
        prev_sentinel = sentinel;
    }

    printf("probe summary: current=%d stale=%d zero=%d other=%d / %d reps\n",
           n_current, n_stale, n_zero, n_other, reps);

    hipSetDevice(devices[rankA]);
    hipFree(d_out);
    hipSetDevice(devices[rankB]);
    hipFree(d_peer);

    return (n_current == reps) ? 0 : 1;
}

int main(int argc, char ** argv) {
    std::vector<int> devices = {0, 1};
    std::vector<int> element_counts = {7680, 15360, 30720, 61440, 122880};
    int reps = 500;
    int blocks = 8;
    Mode mode = MODE_HOST;
    PeerOrder peer_order = ORDER_ENABLE_FIRST;

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
            else if (m == "probe-coarse") mode = MODE_PROBE_COARSE;
            else if (m == "probe-fine") mode = MODE_PROBE_FINE;
            else if (m == "p2p-push") mode = MODE_P2P_PUSH;
            else { fprintf(stderr, "unknown --mode %s (want host|p2p|p2p-push|probe-coarse|probe-fine)\n", m.c_str()); return 2; }
        } else if (arg == "--peer-order" && i + 1 < argc) {
            std::string o = argv[++i];
            if (o == "enable-first") peer_order = ORDER_ENABLE_FIRST;
            else if (o == "alloc-first") peer_order = ORDER_ALLOC_FIRST;
            else { fprintf(stderr, "unknown --peer-order %s (want enable-first|alloc-first)\n", o.c_str()); return 2; }
        } else {
            fprintf(stderr, "unknown arg: %s\n", arg.c_str());
            return 2;
        }
    }

    if (devices.size() != 2) {
        fprintf(stderr, "this harness is N=2 only (dual-XTX P2P prototype), got %zu devices\n", devices.size());
        return 2;
    }

    if (mode == MODE_PROBE_COARSE || mode == MODE_PROBE_FINE) {
        int can01 = 0, can10 = 0;
        HIP_CHECK(hipDeviceCanAccessPeer(&can01, devices[0], devices[1]));
        HIP_CHECK(hipDeviceCanAccessPeer(&can10, devices[1], devices[0]));
        if (!can01 || !can10) {
            fprintf(stderr, "probe modes require hipDeviceCanAccessPeer in both directions "
                             "(got %d->%d=%d, %d->%d=%d)\n",
                    devices[0], devices[1], can01, devices[1], devices[0], can10);
            return 2;
        }
        return run_probe(devices, mode == MODE_PROBE_FINE, peer_order, reps);
    }

    if (blocks < 1 || blocks > MAX_BLOCKS) {
        fprintf(stderr, "--blocks must be 1..%d\n", MAX_BLOCKS);
        return 2;
    }

    const int n = 2;
    const int rankA = 0, rankB = 1;

    if (mode == MODE_P2P || mode == MODE_P2P_PUSH) {
        int can01 = 0, can10 = 0;
        HIP_CHECK(hipDeviceCanAccessPeer(&can01, devices[0], devices[1]));
        HIP_CHECK(hipDeviceCanAccessPeer(&can10, devices[1], devices[0]));
        if (!can01 || !can10) {
            fprintf(stderr, "p2p modes require hipDeviceCanAccessPeer in both directions "
                             "(got %d->%d=%d, %d->%d=%d) -- this device pair is not P2P-capable\n",
                    devices[0], devices[1], can01, devices[1], devices[0], can10);
            return 2;
        }
        if (peer_order == ORDER_ENABLE_FIRST) {
            HIP_CHECK(hipSetDevice(devices[0]));
            hipError_t e0 = hipDeviceEnablePeerAccess(devices[1], 0);
            if (e0 != hipSuccess && e0 != hipErrorPeerAccessAlreadyEnabled) HIP_CHECK(e0);
            HIP_CHECK(hipSetDevice(devices[1]));
            hipError_t e1 = hipDeviceEnablePeerAccess(devices[0], 0);
            if (e1 != hipSuccess && e1 != hipErrorPeerAccessAlreadyEnabled) HIP_CHECK(e1);
        }
        // (ORDER_ALLOC_FIRST for --mode p2p: peer-enable is deferred to just
        // after the per-element hipMalloc calls below, inside the loop.)
    }

    const char * mode_name = mode == MODE_HOST     ? "host-staged"
                           : mode == MODE_P2P_PUSH ? "p2p-push"
                                                   : "p2p-direct-load";
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
            if (mode == MODE_P2P_PUSH) {
                // exchange buffer + flag array live in LOCAL VRAM; the peer
                // writes into them across PCIe (the direction that works).
                HIP_CHECK(hipMalloc(&ds[r].d_exch, bytes));
                HIP_CHECK(hipMalloc(&ds[r].d_flag, (size_t) MAX_BLOCKS * ARRIVAL_STRIDE_INTS * sizeof(int)));
                HIP_CHECK(hipMemset(ds[r].d_flag, 0, (size_t) MAX_BLOCKS * ARRIVAL_STRIDE_INTS * sizeof(int)));
                HIP_CHECK(hipMalloc(&ds[r].d_dead, sizeof(int)));
                HIP_CHECK(hipMemset(ds[r].d_dead, 0, sizeof(int)));
            }
            HIP_CHECK(hipMemcpy(ds[r].d_send, input[r].data(), bytes, hipMemcpyHostToDevice));
        }
        if (mode == MODE_P2P && peer_order == ORDER_ALLOC_FIRST) {
            // Arm C (gpt-dev-agent req_877e043460c94bd8): enable peer access
            // AFTER these buffers are already allocated, to rule out a
            // runtime bug where only future allocations get mapped.
            HIP_CHECK(hipSetDevice(devices[0]));
            hipError_t e0 = hipDeviceEnablePeerAccess(devices[1], 0);
            if (e0 != hipSuccess && e0 != hipErrorPeerAccessAlreadyEnabled) HIP_CHECK(e0);
            HIP_CHECK(hipSetDevice(devices[1]));
            hipError_t e1 = hipDeviceEnablePeerAccess(devices[0], 0);
            if (e1 != hipSuccess && e1 != hipErrorPeerAccessAlreadyEnabled) HIP_CHECK(e1);
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
                } else if (mode == MODE_P2P_PUSH) {
                    // push our payload into the PEER's exchange buffer and
                    // signal the PEER's flag; spin on our own local flag.
                    p2p_push_kernel<<<blocks, 256, 0, ds[r].stream>>>(
                        ds[r].d_send, ds[r].d_recv,
                        ds[peer].d_exch, ds[r].d_exch,
                        ds[peer].d_flag, ds[r].d_flag,
                        elems, token, ds[r].d_dead);
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
        if (mode == MODE_P2P_PUSH) {
            for (int r = 0; r < n; ++r) {
                int dead = 0;
                HIP_CHECK(hipSetDevice(devices[r]));
                HIP_CHECK(hipMemcpy(&dead, ds[r].d_dead, sizeof(int), hipMemcpyDeviceToHost));
                if (dead) {
                    fprintf(stderr,
                        "rank=%d(device=%d): handshake spin budget EXHAUSTED -- the peer's flag write\n"
                        "  never became visible while this kernel was resident. On this box peer-directed\n"
                        "  writes only drain across PCIe when the writing kernel RETIRES;\n"
                        "  __threadfence_system() does not flush them. A single-kernel in-kernel\n"
                        "  handshake between these two GPUs is therefore not possible.\n", r, devices[r]);
                }
            }
        }
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
            if (mode == MODE_P2P_PUSH) { hipFree(ds[r].d_exch); hipFree(ds[r].d_flag); hipFree(ds[r].d_dead); }
            hipStreamDestroy(ds[r].stream);
        }
        hipHostFree(h_arrival);
    }

    return 0;
}
