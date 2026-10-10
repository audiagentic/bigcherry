// QFP41 / QFP49: standalone HIP probes, outside llama.cpp and radiance, for the questions the threading and
// cross-card-sum work rests on. Each test prints what it measured; nothing here is tuned to pass.
//
//   links    how fast one card exchanges a prefill-sized message with pinned host memory: up, down, both at once,
//            and with every card doing it at the same time (the cards have no peer-to-peer path)
//   overlap  does a transfer on its own stream run beside kernels on the compute stream, and does either slow down
//   submit   many small kernels on every card: one host thread feeding all cards against one thread a card
//   graphs   one thread a card capturing, instantiating and launching HIP graphs at the same time, with and without
//            a process-wide lock: are the results still right (the 1356 dispatch-worker race, in isolation)
//   stagger  a layer loop of compute + exchange on every card: as today (the card waits for the exchange) against
//            two half-batches staggered so one half's exchange runs under the other half's compute
//
// Build: hipcc -O2 -std=c++17 -pthread hip_overlap_probe.cpp -o hip_overlap_probe   (run.sh does it)
// Usage: hip_overlap_probe [test...]      default: every test.   Cards: HIP_VISIBLE_DEVICES.
#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define CK(call)                                                                                                    \
    do {                                                                                                            \
        hipError_t e_ = (call);                                                                                     \
        if (e_ != hipSuccess) {                                                                                     \
            std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", hipGetErrorString(e_), __FILE__, __LINE__, #call);  \
            std::exit(2);                                                                                           \
        }                                                                                                           \
    } while (0)

// One prefill sum of the Flash-Next production profile: 512 tokens x 2560 wide x f32.
static const size_t kMsg = 512ull * 2560ull * 4ull;
// The buffer a "compute" kernel streams over; one pass reads and writes it once.
static const size_t kBurnElems = 64ull << 20;  // 256 MiB of f32

__global__ void burn_kernel(float * x, size_t n, float a) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = x[i] * a + 1.0f;
    }
}

__global__ void add_kernel(float * x, int n, float c) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += c;
    }
}

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static void burn(hipStream_t s, float * x, size_t n, int passes) {
    const unsigned block = 256;
    const unsigned grid = (unsigned) ((n + block - 1) / block);
    for (int p = 0; p < passes; ++p) {
        hipLaunchKernelGGL(burn_kernel, dim3(grid), dim3(block), 0, s, x, n, 1.0f);
    }
}

// All threads wait here until `count` of them have arrived (the cards arriving at one sum).
class Rendezvous {
  public:
    explicit Rendezvous(int count) : count_(count) {}
    void arrive() {
        std::unique_lock<std::mutex> lock(mutex_);
        const long generation = generation_;
        if (++waiting_ == count_) {
            waiting_ = 0;
            ++generation_;
            cv_.notify_all();
        } else {
            cv_.wait(lock, [&] { return generation_ != generation; });
        }
    }

  private:
    std::mutex mutex_;
    std::condition_variable cv_;
    int count_;
    int waiting_ = 0;
    long generation_ = 0;
};

struct Card {
    int id = 0;
    std::string name;
    hipStream_t compute = nullptr, transfer = nullptr, transfer2 = nullptr;
    float * buf = nullptr;      // kBurnElems
    char * msg = nullptr;       // 2 x kMsg on the card
    char * host = nullptr;      // 2 x kMsg pinned
};

static std::vector<Card> g_cards;

static void for_each_card_thread(const std::function<void(Card &)> & body) {
    std::vector<std::thread> threads;
    for (auto & card : g_cards) {
        threads.emplace_back([&card, &body] {
            CK(hipSetDevice(card.id));
            body(card);
        });
    }
    for (auto & t : threads) {
        t.join();
    }
}

static void setup() {
    int n = 0;
    CK(hipGetDeviceCount(&n));
    g_cards.resize(n);
    for (int d = 0; d < n; ++d) {
        Card & c = g_cards[d];
        c.id = d;
        CK(hipSetDevice(d));
        hipDeviceProp_t props;
        CK(hipGetDeviceProperties(&props, d));
        c.name = std::string(props.name) + " (" + props.gcnArchName + ")";
        CK(hipStreamCreate(&c.compute));
        CK(hipStreamCreate(&c.transfer));
        CK(hipStreamCreate(&c.transfer2));
        CK(hipMalloc(&c.buf, kBurnElems * sizeof(float)));
        CK(hipMalloc(&c.msg, 2 * kMsg));
        CK(hipHostMalloc(&c.host, 2 * kMsg, hipHostMallocDefault));
        std::memset(c.host, 1, 2 * kMsg);
        CK(hipMemset(c.buf, 0, kBurnElems * sizeof(float)));
        CK(hipMemset(c.msg, 0, 2 * kMsg));
        burn(c.compute, c.buf, kBurnElems, 2);  // first-touch and code load, outside every timing
        CK(hipStreamSynchronize(c.compute));
        std::printf("card %d: %s\n", d, c.name.c_str());
    }
}

// ------------------------------------------------------------------------------------------------ links
static double exchange_ms(Card & c, size_t bytes, bool up, bool down, int reps) {
    const double t0 = now_ms();
    for (int r = 0; r < reps; ++r) {
        if (up) {
            CK(hipMemcpyAsync(c.host, c.msg, bytes, hipMemcpyDeviceToHost, c.transfer));
        }
        if (down) {
            CK(hipMemcpyAsync(c.msg + kMsg, c.host + kMsg, bytes, hipMemcpyHostToDevice, up ? c.transfer2 : c.transfer));
        }
        CK(hipStreamSynchronize(c.transfer));
        if (up && down) {
            CK(hipStreamSynchronize(c.transfer2));
        }
    }
    return (now_ms() - t0) / reps;
}

static void test_links() {
    std::printf("\n== links: one message is %.2f MB (512 x 2560 f32); ms a message and GB/s one way\n", kMsg / 1e6);
    const int reps = 200;
    const size_t sizes[] = {kMsg / 4, kMsg / 2, kMsg, 4 * kMsg / 2};
    for (auto & c : g_cards) {
        CK(hipSetDevice(c.id));
        for (size_t bytes : sizes) {
            if (bytes > kMsg) {
                continue;
            }
            const double up = exchange_ms(c, bytes, true, false, reps);
            const double down = exchange_ms(c, bytes, false, true, reps);
            const double both = exchange_ms(c, bytes, true, true, reps);
            std::printf("  card %d alone   %5.2f MB: up %.3f ms (%.1f GB/s)  down %.3f ms (%.1f GB/s)  both at once %.3f ms\n",
                        c.id, bytes / 1e6, up, bytes / up / 1e6, down, bytes / down / 1e6, both);
        }
    }
    std::vector<double> up(g_cards.size()), down(g_cards.size()), both(g_cards.size());
    Rendezvous start(static_cast<int>(g_cards.size()));
    for_each_card_thread([&](Card & c) {
        start.arrive();
        up[c.id] = exchange_ms(c, kMsg, true, false, reps);
        start.arrive();
        down[c.id] = exchange_ms(c, kMsg, false, true, reps);
        start.arrive();
        both[c.id] = exchange_ms(c, kMsg, true, true, reps);
    });
    for (auto & c : g_cards) {
        std::printf("  card %d with all  %5.2f MB: up %.3f ms (%.1f GB/s)  down %.3f ms (%.1f GB/s)  both at once %.3f ms\n",
                    c.id, kMsg / 1e6, up[c.id], kMsg / up[c.id] / 1e6, down[c.id], kMsg / down[c.id] / 1e6, both[c.id]);
    }
}

// ---------------------------------------------------------------------------------------------- overlap
static void test_overlap() {
    std::printf("\n== overlap: 8 compute passes over 256 MiB on the compute stream, 8 round trips of one message on the transfer stream\n");
    const int passes = 8, trips = 8, reps = 20;
    for (auto & c : g_cards) {
        CK(hipSetDevice(c.id));
        double compute = 0, transfer = 0, together = 0, together_compute = 0;
        for (int r = 0; r < reps; ++r) {
            double t0 = now_ms();
            burn(c.compute, c.buf, kBurnElems, passes);
            CK(hipStreamSynchronize(c.compute));
            compute += now_ms() - t0;

            t0 = now_ms();
            for (int i = 0; i < trips; ++i) {
                CK(hipMemcpyAsync(c.host, c.msg, kMsg, hipMemcpyDeviceToHost, c.transfer));
                CK(hipMemcpyAsync(c.msg + kMsg, c.host + kMsg, kMsg, hipMemcpyHostToDevice, c.transfer));
            }
            CK(hipStreamSynchronize(c.transfer));
            transfer += now_ms() - t0;

            t0 = now_ms();
            burn(c.compute, c.buf, kBurnElems, passes);
            for (int i = 0; i < trips; ++i) {
                CK(hipMemcpyAsync(c.host, c.msg, kMsg, hipMemcpyDeviceToHost, c.transfer));
                CK(hipMemcpyAsync(c.msg + kMsg, c.host + kMsg, kMsg, hipMemcpyHostToDevice, c.transfer));
            }
            CK(hipStreamSynchronize(c.compute));
            together_compute += now_ms() - t0;
            CK(hipStreamSynchronize(c.transfer));
            together += now_ms() - t0;
        }
        std::printf("  card %d: compute alone %.2f ms, transfers alone %.2f ms, both started together: compute done at %.2f ms, all done at %.2f ms"
                    "  (one after the other would be %.2f)\n",
                    c.id, compute / reps, transfer / reps, together_compute / reps, together / reps, (compute + transfer) / reps);
    }
}

// ----------------------------------------------------------------------------------------------- submit
static void test_submit() {
    const int launches = 4000;
    std::printf("\n== submit: %d small kernels a card (256 elements each)\n", launches);
    auto feed = [&](Card & c, int count) {
        for (int i = 0; i < count; ++i) {
            hipLaunchKernelGGL(add_kernel, dim3(1), dim3(256), 0, c.compute, c.buf, 256, 1.0f);
        }
    };
    double t0 = now_ms();
    for (int i = 0; i < launches; i += 8) {  // one host thread, the cards in turn, 8 launches at a time
        for (auto & c : g_cards) {
            CK(hipSetDevice(c.id));
            feed(c, 8);
        }
    }
    const double one_submit = now_ms() - t0;
    for (auto & c : g_cards) {
        CK(hipSetDevice(c.id));
        CK(hipStreamSynchronize(c.compute));
    }
    const double one_total = now_ms() - t0;

    Rendezvous start(static_cast<int>(g_cards.size()));
    std::vector<double> submit(g_cards.size());
    t0 = now_ms();
    for_each_card_thread([&](Card & c) {
        start.arrive();
        const double t1 = now_ms();
        feed(c, launches);
        submit[c.id] = now_ms() - t1;
        CK(hipStreamSynchronize(c.compute));
    });
    const double many_total = now_ms() - t0;
    double worst = 0;
    for (double s : submit) {
        worst = s > worst ? s : worst;
    }
    std::printf("  one host thread for all cards: submitted in %.1f ms, finished at %.1f ms (%.1f us a launch)\n", one_submit, one_total,
                one_submit * 1e3 / (launches * g_cards.size()));
    std::printf("  one host thread a card:        slowest thread submitted in %.1f ms, finished at %.1f ms (%.1f us a launch)\n", worst,
                many_total, worst * 1e3 / launches);
}

// ----------------------------------------------------------------------------------------------- graphs
static void test_graphs() {
    const int rounds = 300, nodes = 12, elems = 4096;
    std::printf("\n== graphs: one thread a card, %d rounds of capture + instantiate + launch of a %d-kernel graph\n", rounds, nodes);
    for (int locked = 0; locked < 2; ++locked) {
        std::mutex graph_mutex;
        std::atomic<long> wrong{0}, errors{0};
        Rendezvous start(static_cast<int>(g_cards.size()));
        const double t0 = now_ms();
        for_each_card_thread([&](Card & c) {
            float * x = nullptr;
            CK(hipMalloc(&x, elems * sizeof(float)));
            std::vector<float> back(elems);
            start.arrive();
            for (int r = 0; r < rounds; ++r) {
                CK(hipMemsetAsync(x, 0, elems * sizeof(float), c.compute));
                hipGraph_t graph = nullptr;
                hipGraphExec_t exec = nullptr;
                float expect = 0;
                bool ok = true;
                {
                    std::unique_lock<std::mutex> lock(graph_mutex, std::defer_lock);
                    if (locked) {
                        lock.lock();
                    }
                    ok = hipStreamBeginCapture(c.compute, hipStreamCaptureModeGlobal) == hipSuccess;
                    for (int k = 0; ok && k < nodes; ++k) {
                        const float add = static_cast<float>(1 + (r + k + c.id) % 7);
                        expect += add;
                        hipLaunchKernelGGL(add_kernel, dim3(elems / 256), dim3(256), 0, c.compute, x, elems, add);
                    }
                    ok = ok && hipStreamEndCapture(c.compute, &graph) == hipSuccess;
                    ok = ok && hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == hipSuccess;
                }
                ok = ok && hipGraphLaunch(exec, c.compute) == hipSuccess;
                ok = ok && hipMemcpyAsync(back.data(), x, elems * sizeof(float), hipMemcpyDeviceToHost, c.compute) == hipSuccess;
                ok = ok && hipStreamSynchronize(c.compute) == hipSuccess;
                if (!ok) {
                    ++errors;
                } else {
                    for (int i = 0; i < elems; i += 97) {
                        if (back[i] != expect) {
                            ++wrong;
                            break;
                        }
                    }
                }
                if (exec) {
                    hipGraphExecDestroy(exec);
                }
                if (graph) {
                    hipGraphDestroy(graph);
                }
            }
            hipFree(x);
        });
        std::printf("  %s: %ld of %ld rounds gave a wrong result, %ld returned an error, %.0f ms\n",
                    locked ? "capture and instantiate under one process-wide lock" : "no lock", wrong.load(),
                    static_cast<long>(rounds) * static_cast<long>(g_cards.size()), errors.load(), now_ms() - t0);
    }
}

// ---------------------------------------------------------------------------------------------- stagger
static void test_stagger() {
    const int layers = 96, passes = 2;  // 96 sums a batch in production; `passes` full passes of compute a sum
    std::printf("\n== stagger: %d rounds of [compute, then every card exchanges one %.2f MB message through the host]\n", layers, kMsg / 1e6);
    const int ncards = static_cast<int>(g_cards.size());

    // compute only, for scale
    Rendezvous go0(ncards);
    double t0 = now_ms();
    for_each_card_thread([&](Card & c) {
        go0.arrive();
        for (int l = 0; l < layers; ++l) {
            burn(c.compute, c.buf, kBurnElems, passes);
        }
        CK(hipStreamSynchronize(c.compute));
    });
    const double compute_only = now_ms() - t0;

    // today: the card computes, then waits while the message goes up, every card has arrived, and it comes down
    Rendezvous go1(ncards), sum1(ncards);
    t0 = now_ms();
    for_each_card_thread([&](Card & c) {
        go1.arrive();
        for (int l = 0; l < layers; ++l) {
            burn(c.compute, c.buf, kBurnElems, passes);
            CK(hipMemcpyAsync(c.host, c.msg, kMsg, hipMemcpyDeviceToHost, c.compute));
            CK(hipStreamSynchronize(c.compute));
            sum1.arrive();  // the host has every card's part
            CK(hipMemcpyAsync(c.msg + kMsg, c.host + kMsg, kMsg, hipMemcpyHostToDevice, c.compute));
        }
        CK(hipStreamSynchronize(c.compute));
    });
    const double today = now_ms() - t0;

    // staggered: two half-batches; a half's exchange runs on the transfer stream under the other half's compute.
    // The same compute in total: each half streams over half the buffer.
    Rendezvous go2(ncards), sum2(ncards);
    t0 = now_ms();
    for_each_card_thread([&](Card & c) {
        hipEvent_t burned[2], uploaded[2], downloaded[2];
        bool have_down[2] = {false, false}, pending[2] = {false, false};
        for (int h = 0; h < 2; ++h) {
            CK(hipEventCreate(&burned[h]));
            CK(hipEventCreate(&uploaded[h]));
            CK(hipEventCreate(&downloaded[h]));
        }
        float * half[2] = {c.buf, c.buf + kBurnElems / 2};
        auto finish_exchange = [&](int h) {  // the host side of a sum for half h: wait for the upload, meet, send down
            CK(hipEventSynchronize(uploaded[h]));
            sum2.arrive();
            CK(hipMemcpyAsync(c.msg + kMsg, c.host + kMsg, kMsg / 2, hipMemcpyHostToDevice, c.transfer));
            CK(hipEventRecord(downloaded[h], c.transfer));
            have_down[h] = true;
            pending[h] = false;
        };
        go2.arrive();
        for (int step = 0; step < 2 * layers; ++step) {
            const int h = step & 1, other = h ^ 1;
            if (have_down[h]) {
                CK(hipStreamWaitEvent(c.compute, downloaded[h], 0));  // this half's next layer needs its sum
            }
            burn(c.compute, half[h], kBurnElems / 2, passes);
            CK(hipEventRecord(burned[h], c.compute));
            if (pending[other]) {
                finish_exchange(other);  // the card is busy with this half's compute meanwhile
            }
            CK(hipStreamWaitEvent(c.transfer, burned[h], 0));
            CK(hipMemcpyAsync(c.host, c.msg, kMsg / 2, hipMemcpyDeviceToHost, c.transfer));
            CK(hipEventRecord(uploaded[h], c.transfer));
            pending[h] = true;
        }
        for (int h = 0; h < 2; ++h) {
            if (pending[h]) {
                finish_exchange(h);
            }
        }
        CK(hipStreamSynchronize(c.compute));
        CK(hipStreamSynchronize(c.transfer));
        for (int h = 0; h < 2; ++h) {
            hipEventDestroy(burned[h]);
            hipEventDestroy(uploaded[h]);
            hipEventDestroy(downloaded[h]);
        }
    });
    const double staggered = now_ms() - t0;
    std::printf("  compute only %.1f ms; today %.1f ms (the exchange adds %.1f ms, %.0f%%); staggered halves %.1f ms (adds %.1f ms, %.0f%%)\n",
                compute_only, today, today - compute_only, 100.0 * (today - compute_only) / compute_only, staggered,
                staggered - compute_only, 100.0 * (staggered - compute_only) / compute_only);
    std::printf("  staggering recovers %.0f%% of what the exchange costs today\n", 100.0 * (today - staggered) / (today - compute_only));
}

int main(int argc, char ** argv) {
    setup();
    std::vector<std::string> tests(argv + 1, argv + argc);
    if (tests.empty()) {
        tests = {"links", "overlap", "submit", "graphs", "stagger"};
    }
    for (const auto & t : tests) {
        if (t == "links") {
            test_links();
        } else if (t == "overlap") {
            test_overlap();
        } else if (t == "submit") {
            test_submit();
        } else if (t == "graphs") {
            test_graphs();
        } else if (t == "stagger") {
            test_stagger();
        } else {
            std::fprintf(stderr, "unknown test %s\n", t.c_str());
            return 2;
        }
        std::fflush(stdout);
    }
    return 0;
}
