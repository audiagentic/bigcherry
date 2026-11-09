// RAD49 step 1: the two-rank cross-card exchange alone, timed through the kernel plugin's own ABI rows -- no model, no engine.
//
// One process, one thread per rank, as radiance runs a TP engine (libr4d's all-reduce instances rendezvous inside the plugin on
// arrival order, r4d_ar_entry.hip). The plugin (libr3.so on gfx1100) is dlopen()ed and its pinned-host-memory rows are called
// exactly as the engine calls them: init() with the frozen geometry, launch() with RadArgs, fini().
//
//   ar_hc_write_host               the gated-residual write with the all-reduce in it (every Flash-Next mixer/MoE output at TP2)
//   ar_oneshot_2rank_exact_host    the standalone exact all_reduce (the embedding, and the MoE output while handed off, rad0002)
//   ar_oneshot_2rank_wht6_host     the standalone rotated 6-bit all_reduce (min_bytes 0: the lossy wire forced at every height)
//
// For each (op, wire, M rows of the model's hidden width n, hc connections) it reports, per rank:
//   loop us   -- `iters` launches back to back on the rank's stream, event-timed: the steady exchange cost a step pays
//   iso us    -- one launch at a time, both ranks released by a barrier, event pair around the launch; median and p90 over iters.
//                A rank's time includes waiting for the later rank, so the smaller rank's median is the closer transfer cost.
// and checks the exact sum (rank 0 sends 1.0, rank 1 sends 2.0: every rank must read 3.0 back).
//
// Build (card-free): g++ -O2 -std=c++20 -D__HIP_PLATFORM_AMD__ -I<rad include> -I<ROCm root>/include r3-arbench.cpp
//                    -L<ROCm root>/lib -lamdhip64 -ldl -lpthread -o r3-arbench
// Run (needs the gpu lease; tools/lab/radiance/rad49-arbench.sh takes it):
//   HIP_VISIBLE_DEVICES=<xtx0>,<xtx1> r3-arbench --lib <radiance_home>/kernels/libr3.so [--rows 1,4,24,32,2048] [--iters 200]
//        [--ops hc,ar] [--wires exact,wht6] --n HIDDEN --hc HC_COUNT
#include <hip/hip_runtime.h>

#include "rad_abi.h"

#include <dlfcn.h>

#include <algorithm>
#include <barrier>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Opts {
    std::string lib;
    std::vector<long> rows{1, 4, 24, 32, 2048};
    std::vector<std::string> ops{"hc", "ar"};
    std::vector<std::string> wires{"exact", "wht6"};
    long n = 0, hc = 0;  /* the model's hidden width and hyper-connection count: required, from the container (the runner reads rad-info) */
    int iters = 200, warm = 20;
};

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    for (std::string x; std::getline(ss, x, ',');) if (!x.empty()) out.push_back(x);
    return out;
}

#define HIP_OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "r3-arbench: %s: %s\n", #x, hipGetErrorString(e_)); std::exit(3); } } while (0)

using CountFn = int (*)(void);
using AtFn = const RadKernelInfo* (*)(int);
using VerFn = uint32_t (*)(void);

RadTensor tensor2(void* p, uint32_t dtype, long r, long c) {
    RadTensor t{};
    t.data = p; t.dtype = dtype; t.rank = 2;
    t.shape[0] = r; t.shape[1] = c; t.stride[0] = c; t.stride[1] = 1;
    return t;
}
RadTensor tensor1(void* p, uint32_t dtype, long n) {
    RadTensor t{};
    t.data = p; t.dtype = dtype; t.rank = 1; t.shape[0] = n; t.stride[0] = 1;
    return t;
}

double pct(std::vector<float> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, size_t(q * (v.size() - 1) + 0.5))];
}

struct Case { std::string op, wire; long M; const RadKernelInfo* k; };
struct Result { double loop_us = 0, iso_med = 0, iso_p90 = 0; bool sum_ok = true; int rc = 0; };

}  // namespace

int main(int argc, char** argv) {
    Opts o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--lib") o.lib = next();
        else if (a == "--rows") { o.rows.clear(); for (auto& x : split(next())) o.rows.push_back(std::atol(x.c_str())); }
        else if (a == "--ops") o.ops = split(next());
        else if (a == "--wires") o.wires = split(next());
        else if (a == "--iters") o.iters = std::atoi(next().c_str());
        else if (a == "--warm") o.warm = std::atoi(next().c_str());
        else if (a == "--n") o.n = std::atol(next().c_str());
        else if (a == "--hc") o.hc = std::atol(next().c_str());
        else { std::fprintf(stderr, "r3-arbench: unknown argument %s\n", a.c_str()); return 2; }
    }
    if (o.lib.empty()) { std::fprintf(stderr, "r3-arbench: --lib PATH (libr3.so) is required\n"); return 2; }
    if (o.n <= 0 || o.hc <= 0) { std::fprintf(stderr, "r3-arbench: --n HIDDEN and --hc HC_COUNT (the model's n_embd and hc_count) are required\n"); return 2; }

    int ndev = 0;
    HIP_OK(hipGetDeviceCount(&ndev));
    if (ndev < 2) { std::fprintf(stderr, "r3-arbench: needs two visible cards (HIP_VISIBLE_DEVICES=a,b), sees %d\n", ndev); return 2; }

    void* so = dlopen(o.lib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!so) { std::fprintf(stderr, "r3-arbench: dlopen %s: %s\n", o.lib.c_str(), dlerror()); return 2; }
    auto ver = reinterpret_cast<VerFn>(dlsym(so, "rad_plugin_abi_version"));
    auto count = reinterpret_cast<CountFn>(dlsym(so, "rad_kernel_count"));
    auto at = reinterpret_cast<AtFn>(dlsym(so, "rad_kernel_at"));
    if (!ver || !count || !at) { std::fprintf(stderr, "r3-arbench: %s is not a kernel plugin\n", o.lib.c_str()); return 2; }
    if (ver() != RAD_ABI_VERSION) {
        std::fprintf(stderr, "r3-arbench: %s has ABI %u, this harness was built against %u: rebuild it against that build's headers\n",
                     o.lib.c_str(), ver(), RAD_ABI_VERSION);
        return 2;
    }
    if (auto open = reinterpret_cast<RadPluginOpenFn>(dlsym(so, "rad_plugin_open")); open && open() != RAD_OK) {
        std::fprintf(stderr, "r3-arbench: rad_plugin_open declined this machine\n");
        return 2;
    }
    auto row = [&](const char* name) -> const RadKernelInfo* {
        for (int i = 0; i < count(); ++i) if (const RadKernelInfo* k = at(i); k && k->name && !std::strcmp(k->name, name)) return k;
        return nullptr;
    };

    std::vector<Case> cases;
    for (const auto& op : o.ops)
        for (const auto& wire : o.wires)
            for (long M : o.rows) {
                const char* name = op == "hc" ? "ar_hc_write_host"
                                 : wire == "exact" ? "ar_oneshot_2rank_exact_host" : "ar_oneshot_2rank_wht6_host";
                const RadKernelInfo* k = row(name);
                if (!k) {
                    std::fprintf(stderr, "r3-arbench: %s has no row %s; its collective rows are:\n", o.lib.c_str(), name);
                    for (int i = 0; i < count(); ++i)
                        if (const RadKernelInfo* r = at(i); r && r->family && (!std::strcmp(r->family, "ar") || std::strstr(r->name, "_host")))
                            std::fprintf(stderr, "  %s (%s)\n", r->name, r->op);
                    return 2;
                }
                cases.push_back({op, wire, M, k});
            }

    std::barrier sync(2);
    std::vector<std::vector<Result>> res(2, std::vector<Result>(cases.size()));
    std::mutex log_mu;

    auto rank_main = [&](int rank) {
        HIP_OK(hipSetDevice(rank));
        hipDeviceProp_t prop{};
        HIP_OK(hipGetDeviceProperties(&prop, rank));
        hipStream_t st;
        HIP_OK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
        hipEvent_t e0, e1;
        HIP_OK(hipEventCreate(&e0));
        HIP_OK(hipEventCreate(&e1));
        for (size_t ci = 0; ci < cases.size(); ++ci) {
            const Case& c = cases[ci];
            Result& r = res[rank][ci];
            const long M = c.M, n = o.n, hc = o.hc, numel = M * n;
            void *y = nullptr, *inj = nullptr, *h = nullptr;
            HIP_OK(hipMalloc(&y, numel * 2));
            HIP_OK(hipMalloc(&inj, std::max<long>(M * hc * 2, 16)));
            HIP_OK(hipMalloc(&h, M * hc * n * 2));
            HIP_OK(hipMemsetD16(reinterpret_cast<hipDeviceptr_t>(inj), 0x3F80, M * hc));   // inj = 1.0
            HIP_OK(hipMemset(h, 0, M * hc * n * 2));
            std::vector<RadParam> p;
            if (c.op == "hc") {
                p = {RAD_INT("M", M), RAD_INT("n", n), RAD_INT("hc", hc), RAD_INT("world_size", 2), RAD_STR("dtype", "bf16"),
                     RAD_INT("wire", c.wire == "wht6" ? 6 : 0)};
            } else {
                p = {RAD_INT("world_size", 2), RAD_INT("numel", numel), RAD_STR("dtype", "bf16"),
                     RAD_INT("exact", c.wire == "wht6" ? 0 : 1), RAD_INT("min_bytes", 0)};
            }
            std::vector<RadTensor> t;
            if (c.op == "hc") t = {tensor2(y, RAD_BF16, M, n), tensor2(inj, RAD_BF16, M, hc), tensor2(h, RAD_BF16, M, hc * n)};
            else t = {tensor1(y, RAD_BF16, numel)};
            void* inst = nullptr;
            sync.arrive_and_wait();
            r.rc = c.k->init ? c.k->init(p.data(), (int)p.size(), rank, 2, &inst) : RAD_OK;
            RadArgs a{};
            a.t = t.data(); a.n_t = (int)t.size(); a.p = p.data(); a.n_p = (int)p.size();
            a.instance = inst; a.rank = rank; a.world_size = 2; a.kv_used_max_h = -1; a.wgp_count = prop.multiProcessorCount;
            void* scratch = nullptr;
            if (r.rc == RAD_OK && c.k->scratch) {
                const int64_t b = c.k->scratch(&a);
                if (b > 0) { HIP_OK(hipMalloc(&scratch, b)); a.scratch = scratch; a.scratch_bytes = b; }
            }
            // Both ranks must agree to go on: a rank whose init failed would leave the other's first launch waiting forever.
            static int agreed[2];
            agreed[rank] = r.rc;
            sync.arrive_and_wait();
            const bool go = agreed[0] == RAD_OK && agreed[1] == RAD_OK;
            auto fill = [&] { HIP_OK(hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(y), rank == 0 ? 0x3F80 : 0x4000, numel, st)); };
            auto launch = [&] {
                const int rc = c.k->launch(&a, st);
                if (rc != RAD_OK && rc != RAD_LINK_ENDED) { std::fprintf(stderr, "r3-arbench: launch %s rc %d\n", c.k->name, rc); std::exit(4); }
            };
            auto step = [&](const char* what) {
                if (rank) return;
                std::lock_guard<std::mutex> lk(log_mu);
                std::fprintf(stderr, "r3-arbench: %s %s M=%ld: %s\n", c.k->name, c.wire.c_str(), M, what);
            };
            if (go) {
                // the exact sum, once: 1.0 + 2.0 = 3.0 on both ranks (0x4040)
                step("first launch");
                fill(); launch();
                HIP_OK(hipStreamSynchronize(st));
                uint16_t got = 0;
                HIP_OK(hipMemcpy(&got, y, 2, hipMemcpyDeviceToHost));
                r.sum_ok = c.wire != "exact" || got == 0x4040;
                step("sum checked; warm and timed loops");
                for (int w = 0; w < o.warm; ++w) launch();
                HIP_OK(hipStreamSynchronize(st));
                sync.arrive_and_wait();
                HIP_OK(hipEventRecord(e0, st));
                for (int k = 0; k < o.iters; ++k) launch();
                HIP_OK(hipEventRecord(e1, st));
                HIP_OK(hipEventSynchronize(e1));
                float ms = 0;
                HIP_OK(hipEventElapsedTime(&ms, e0, e1));
                r.loop_us = ms * 1000.0 / o.iters;
                std::vector<float> iso;
                for (int k = 0; k < o.iters; ++k) {
                    sync.arrive_and_wait();
                    HIP_OK(hipEventRecord(e0, st));
                    launch();
                    HIP_OK(hipEventRecord(e1, st));
                    HIP_OK(hipEventSynchronize(e1));
                    HIP_OK(hipEventElapsedTime(&ms, e0, e1));
                    iso.push_back(ms * 1000.0f);
                }
                step("done");
                r.iso_med = pct(iso, 0.5);
                r.iso_p90 = pct(iso, 0.9);
            }
            HIP_OK(hipStreamSynchronize(st));
            sync.arrive_and_wait();
            if (inst && c.k->fini) c.k->fini(inst);
            if (scratch) HIP_OK(hipFree(scratch));
            HIP_OK(hipFree(y)); HIP_OK(hipFree(inj)); HIP_OK(hipFree(h));
            if (!go) {
                std::lock_guard<std::mutex> lk(log_mu);
                std::fprintf(stderr, "r3-arbench: %s M=%ld: init refused (rank 0 rc %d, rank 1 rc %d)\n", c.k->name, M, agreed[0], agreed[1]);
            }
        }
    };
    std::thread t0(rank_main, 0), t1(rank_main, 1);
    t0.join();
    t1.join();

    std::printf("op\twire\trows\tbytes_bf16\trow\tloop_us_r0\tloop_us_r1\tiso_med_us_r0\tiso_med_us_r1\tiso_p90_us_r0\tiso_p90_us_r1\tiso_med_us_min\tsum_ok\n");
    bool all_ok = true;
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const Case& c = cases[ci];
        const Result &a = res[0][ci], &b = res[1][ci];
        const bool ok = a.rc == RAD_OK && b.rc == RAD_OK && a.sum_ok && b.sum_ok;
        all_ok = all_ok && ok;
        std::printf("%s\t%s\t%ld\t%ld\t%s\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%s\n", c.op.c_str(), c.wire.c_str(), c.M,
                    c.M * o.n * 2, c.k->name, a.loop_us, b.loop_us, a.iso_med, b.iso_med, a.iso_p90, b.iso_p90,
                    std::min(a.iso_med, b.iso_med), a.rc || b.rc ? "init-refused" : (ok ? "yes" : "NO"));
    }
    return all_ok ? 0 : 1;
}
