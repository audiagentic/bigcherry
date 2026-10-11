// QFP49: does an RCCL all-reduce on a second stream run under a compute kernel on the same card?
// The stagger test of hip_overlap_probe.cpp routes its exchange through the host; production uses RCCL, whose
// exchange is a kernel on the card. This probe repeats the staggered half-batch schedule with ncclAllReduce.
//
// Arms, each 96 sums' worth of compute on every card, one host thread as in the tensor-split backend:
//   compute    no sums
//   today      [compute, sum of one full message on the compute stream] x 96
//   halves     [half compute, sum of a half message on the compute stream] x 192          (what -ub 256 does)
//   (transfer) today and halves again with the sum on the transfer stream, fenced by events, nothing overlapping:
//              what the second stream costs by itself
//   staggered  halves A and B alternate; a half's sum runs on the transfer stream, fenced by events, while the
//              card computes the other half
// Then a short staggered run from known values to check the sums.
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CK(x)                                                                                    \
    do {                                                                                         \
        hipError_t e_ = (x);                                                                     \
        if (e_ != hipSuccess) {                                                                  \
            std::fprintf(stderr, "HIP error %s at line %d\n", hipGetErrorString(e_), __LINE__); \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)
#define NK(x)                                                                                     \
    do {                                                                                          \
        ncclResult_t r_ = (x);                                                                    \
        if (r_ != ncclSuccess) {                                                                  \
            std::fprintf(stderr, "RCCL error %s at line %d\n", ncclGetErrorString(r_), __LINE__); \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (0)

// One prefill sum of the Flash-Next production profile on its bf16 wire: 512 tokens x 2560 wide x 2 bytes, as f32 elements.
static const size_t kMsgElems = 512ull * 2560ull / 2ull;
static const size_t kBurnElems = 64ull << 20;  // 256 MiB of f32; one pass reads and writes it once

__global__ void burn_kernel(float * x, size_t n, float a) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = x[i] * a + 1.0f;
    }
}

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct Card {
    int id = 0;
    hipStream_t compute = nullptr, transfer = nullptr;
    float * buf = nullptr;
    float * msg[2] = {nullptr, nullptr};  // one message per half; msg[0] is also the full message
    hipEvent_t burned[2], summed[2];
};

static std::vector<Card> g_cards;
static std::vector<ncclComm_t> g_comms;

// PROBE_SPLIT=N: each pass is N kernels over a 1/N slice, the same work as one kernel (the engine runs some 40
// small kernels between two sums, not two large ones)
static int g_split = 1;

static void burn(Card & c, float * x, size_t n, int passes) {
    const unsigned block = 256;
    const size_t slice = n / (size_t) g_split;
    const unsigned grid = (unsigned) ((slice + block - 1) / block);
    CK(hipSetDevice(c.id));
    for (int p = 0; p < passes; ++p) {
        for (int k = 0; k < g_split; ++k) {
            hipLaunchKernelGGL(burn_kernel, dim3(grid), dim3(block), 0, c.compute, x + (size_t) k * slice, slice, 1.0f);
        }
    }
}

static void sum(int half, size_t elems, bool on_transfer) {
    NK(ncclGroupStart());
    for (size_t i = 0; i < g_cards.size(); ++i) {
        Card & c = g_cards[i];
        NK(ncclAllReduce(c.msg[half], c.msg[half], elems, ncclFloat, ncclSum, g_comms[i], on_transfer ? c.transfer : c.compute));
    }
    NK(ncclGroupEnd());
}

static void sync_all() {
    for (Card & c : g_cards) {
        CK(hipSetDevice(c.id));
        CK(hipStreamSynchronize(c.compute));
        CK(hipStreamSynchronize(c.transfer));
    }
}

static void fill(float v) {
    std::vector<float> host(kMsgElems, v);
    for (Card & c : g_cards) {
        CK(hipSetDevice(c.id));
        for (int h = 0; h < 2; ++h) {
            CK(hipMemcpy(c.msg[h], host.data(), kMsgElems * sizeof(float), hipMemcpyHostToDevice));
        }
    }
}

static double run_compute(int layers, int passes) {
    const double t0 = now_ms();
    for (int l = 0; l < layers; ++l) {
        for (Card & c : g_cards) {
            burn(c, c.buf, kBurnElems, passes);
        }
    }
    sync_all();
    return now_ms() - t0;
}

// [compute over `burn_elems`, sum of `msg_elems`] x rounds, everything on the compute stream
static double run_serial(int rounds, int passes, size_t burn_elems, size_t msg_elems) {
    const double t0 = now_ms();
    for (int r = 0; r < rounds; ++r) {
        for (Card & c : g_cards) {
            burn(c, c.buf, burn_elems, passes);
        }
        sum(0, msg_elems, false);
    }
    sync_all();
    return now_ms() - t0;
}

// today's order, but the sum on the transfer stream, fenced by events both ways (what 1360 does back to back):
// nothing overlaps it, so any difference from run_serial is the cost of the second stream itself
static double run_serial_transfer(int rounds, int passes, size_t burn_elems, size_t msg_elems) {
    const double t0 = now_ms();
    for (int r = 0; r < rounds; ++r) {
        for (Card & c : g_cards) {
            CK(hipSetDevice(c.id));
            if (r > 0) {
                CK(hipStreamWaitEvent(c.compute, c.summed[0], 0));
            }
            burn(c, c.buf, burn_elems, passes);
            CK(hipEventRecord(c.burned[0], c.compute));
            CK(hipStreamWaitEvent(c.transfer, c.burned[0], 0));
        }
        sum(0, msg_elems, true);
        for (Card & c : g_cards) {
            CK(hipSetDevice(c.id));
            CK(hipEventRecord(c.summed[0], c.transfer));
        }
    }
    for (Card & c : g_cards) {
        CK(hipSetDevice(c.id));
        CK(hipStreamWaitEvent(c.compute, c.summed[0], 0));
    }
    sync_all();
    return now_ms() - t0;
}

static double run_staggered(int layers, int passes) {
    bool have_sum[2] = {false, false};
    const double t0 = now_ms();
    for (int step = 0; step < 2 * layers; ++step) {
        const int h = step & 1;
        for (Card & c : g_cards) {
            CK(hipSetDevice(c.id));
            if (have_sum[h]) {
                CK(hipStreamWaitEvent(c.compute, c.summed[h], 0));  // this half's next layer needs its sum
            }
            burn(c, c.buf + (h ? kBurnElems / 2 : 0), kBurnElems / 2, passes);
            CK(hipEventRecord(c.burned[h], c.compute));
            CK(hipStreamWaitEvent(c.transfer, c.burned[h], 0));
        }
        sum(h, kMsgElems / 2, true);
        for (Card & c : g_cards) {
            CK(hipSetDevice(c.id));
            CK(hipEventRecord(c.summed[h], c.transfer));
        }
        have_sum[h] = true;
    }
    sync_all();
    return now_ms() - t0;
}

int main(int argc, char ** argv) {
    const int layers = argc > 1 ? std::atoi(argv[1]) : 96;
    const int passes = argc > 2 ? std::atoi(argv[2]) : 2;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 5;
    const bool nonblocking = std::getenv("PROBE_NONBLOCKING") != nullptr && std::atoi(std::getenv("PROBE_NONBLOCKING")) != 0;
    g_split = std::getenv("PROBE_SPLIT") != nullptr ? std::max(1, std::atoi(std::getenv("PROBE_SPLIT"))) : 1;
    std::printf("transfer stream and events %s; %d kernel(s) a compute pass
", nonblocking ? "as 1360 creates them (non-blocking, timing off)" : "default", g_split);
    int n = 0;
    CK(hipGetDeviceCount(&n));
    g_cards.resize(n);
    g_comms.resize(n);
    std::vector<int> devs(n);
    for (int d = 0; d < n; ++d) {
        Card & c = g_cards[d];
        c.id = devs[d] = d;
        CK(hipSetDevice(d));
        hipDeviceProp_t props;
        CK(hipGetDeviceProperties(&props, d));
        CK(hipStreamCreate(&c.compute));
        // PROBE_NONBLOCKING=1: the transfer stream and the events as patch 1360 creates them
        if (nonblocking) {
            CK(hipStreamCreateWithFlags(&c.transfer, hipStreamNonBlocking));
        } else {
            CK(hipStreamCreate(&c.transfer));
        }
        CK(hipMalloc(&c.buf, kBurnElems * sizeof(float)));
        CK(hipMemset(c.buf, 0, kBurnElems * sizeof(float)));
        for (int h = 0; h < 2; ++h) {
            CK(hipMalloc(&c.msg[h], kMsgElems * sizeof(float)));
            if (nonblocking) {
                CK(hipEventCreateWithFlags(&c.burned[h], hipEventDisableTiming));
                CK(hipEventCreateWithFlags(&c.summed[h], hipEventDisableTiming));
            } else {
                CK(hipEventCreate(&c.burned[h]));
                CK(hipEventCreate(&c.summed[h]));
            }
        }
        std::printf("card %d: %s (%s)\n", d, props.name, props.gcnArchName);
    }
    NK(ncclCommInitAll(g_comms.data(), n, devs.data()));
    fill(0.0f);
    run_serial(4, passes, kBurnElems, kMsgElems);  // code load and first touch, outside every timing
    run_staggered(4, passes);

    std::printf("\n== %d sums of %.2f MB on %d cards, %d compute passes a sum; ms, %d repeats\n", layers,
                kMsgElems * sizeof(float) / 1e6, n, passes, reps);
    for (int r = 0; r < reps; ++r) {
        const double compute = run_compute(layers, passes);
        const double today = run_serial(layers, passes, kBurnElems, kMsgElems);
        const double halves = run_serial(2 * layers, passes, kBurnElems / 2, kMsgElems / 2);
        const double today_t = run_serial_transfer(layers, passes, kBurnElems, kMsgElems);
        const double halves_t = run_serial_transfer(2 * layers, passes, kBurnElems / 2, kMsgElems / 2);
        const double staggered = run_staggered(layers, passes);
        std::printf("  compute %.1f  today %.1f (+%.0f%%)  halves %.1f (+%.0f%%)  staggered %.1f (+%.0f%%)  hidden %.0f%% of today's sum cost\n",
                    compute, today, 100.0 * (today - compute) / compute, halves, 100.0 * (halves - compute) / compute, staggered,
                    100.0 * (staggered - compute) / compute, 100.0 * (today - staggered) / (today - compute));
        std::printf("    sum on the transfer stream, nothing overlapping: today %.1f (%+.3f ms a sum against the compute stream)  halves %.1f (%+.3f ms a sum)\n",
                    today_t, (today_t - today) / layers, halves_t, (halves_t - halves) / (2 * layers));
        std::fflush(stdout);
    }

    // the sums themselves: 5 staggered sums a half from 1.0 on n cards give n^5
    fill(1.0f);
    run_staggered(5, 1);
    double want = 1.0;
    for (int i = 0; i < 5; ++i) {
        want *= n;
    }
    long wrong = 0;
    std::vector<float> host(kMsgElems / 2);
    for (Card & c : g_cards) {
        CK(hipSetDevice(c.id));
        for (int h = 0; h < 2; ++h) {
            CK(hipMemcpy(host.data(), c.msg[h], host.size() * sizeof(float), hipMemcpyDeviceToHost));
            for (float v : host) {
                wrong += v != (float) want;
            }
        }
    }
    std::printf("check: %ld wrong values (want %.0f everywhere)\n", wrong, want);
    for (ncclComm_t comm : g_comms) {
        NK(ncclCommDestroy(comm));
    }
    return wrong != 0;
}
