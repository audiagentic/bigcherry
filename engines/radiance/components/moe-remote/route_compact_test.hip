/* RAD29: compact_routes against the per-route-atomic loop it replaced, on random routing, and its timing.
 *
 * The reference below is the pre-RAD29 publish loop, kept here (test-only) as the equivalence oracle and the timing
 * baseline; it is not built into the engine. Each case runs both on fresh copies of the same inputs and compares the
 * route count, the route multiset (token, expert, weight, slot), the ids left behind, and that every route's candidate
 * really was routed here. Shapes cover tails below, at and around the wave and block sizes. `--bench` adds a timing table.
 * Exit 0 = every case equal. */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <tuple>
#include <vector>

#include "route_compact.h"

using r3::moe_shm::Route;

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)

constexpr uint32_t kBlock = 256;

/* own[e] != 0: expert e is routed here (stands in for slot_map + owner_has). */
__device__ inline int32_t routed(const int32_t* ids, const uint8_t* own, int32_t expert_base, uint32_t n_expert,
                                 uint32_t i) {
    const int32_t le = ids[i] - expert_base;
    return le >= 0 && (uint32_t)le < n_expert && own[le] ? le : -1;
}

__global__ void new_kernel(uint32_t n_cand, uint32_t top_k, Route* route, uint32_t* count, const uint16_t* ew,
                           int32_t* ids, const uint8_t* own, int32_t expert_base, uint32_t n_expert, uint32_t clear) {
    __shared__ uint32_t nr;
    if (threadIdx.x == 0) nr = 0;
    __syncthreads();
    compact_routes(n_cand, top_k, &nr, route, ew, ids, clear != 0,
                   [&](uint32_t i) { return routed(ids, own, expert_base, n_expert, i); });
    __syncthreads();
    if (threadIdx.x == 0) *count = nr;
}

__global__ void ref_kernel(uint32_t n_cand, uint32_t top_k, Route* route, uint32_t* count, const uint16_t* ew,
                           int32_t* ids, const uint8_t* own, int32_t expert_base, uint32_t n_expert, uint32_t clear) {
    __shared__ uint32_t nr;
    if (threadIdx.x == 0) nr = 0;
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < n_cand; i += blockDim.x) {
        const int32_t le = routed(ids, own, expert_base, n_expert, i);
        if (le < 0) continue;
        const uint32_t p = atomicAdd(&nr, 1u);
        route[p].token = (int32_t)(i / top_k);
        route[p].expert = le;
        route[p].weight_bf16 = ew[i];
        route[p].slot = (uint16_t)(i % top_k);
        if (clear) ids[i] = -1;
    }
    __syncthreads();
    if (threadIdx.x == 0) *count = nr;
}

struct Case {
    uint32_t rows, top_k, n_expert;
    double owned;      /* fraction of experts routed here */
    int32_t base;      /* expert_base */
    bool clear;
};

struct Dev {
    int32_t* ids; uint16_t* ew; uint8_t* own; Route* route; uint32_t* count;
};

using Key = std::tuple<int32_t, int32_t, uint16_t, uint16_t>;

static std::vector<Key> keys(const std::vector<Route>& r, uint32_t n) {
    std::vector<Key> k;
    for (uint32_t i = 0; i < n; ++i) k.emplace_back(r[i].token, r[i].expert, r[i].weight_bf16, r[i].slot);
    std::sort(k.begin(), k.end());
    return k;
}

template <class K>
static float run(K kernel, const Case& c, const Dev& d, const std::vector<int32_t>& ids, int iters,
                 std::vector<Route>* out_route, uint32_t* out_count, std::vector<int32_t>* out_ids) {
    const uint32_t n = c.rows * c.top_k;
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    float ms = 0;
    for (int it = 0; it < iters; ++it) {
        CK(hipMemcpy(d.ids, ids.data(), n * sizeof(int32_t), hipMemcpyHostToDevice));
        CK(hipMemset(d.route, 0xff, (n ? n : 1) * sizeof(Route)));
        CK(hipEventRecord(a));
        hipLaunchKernelGGL(kernel, dim3(1), dim3(kBlock), 0, 0, n, c.top_k, d.route, d.count, d.ew, d.ids, d.own,
                           c.base, c.n_expert, c.clear ? 1u : 0u);
        CK(hipEventRecord(b));
        CK(hipEventSynchronize(b));
        float t = 0;
        CK(hipEventElapsedTime(&t, a, b));
        if (it) ms += t;   /* first launch is warm-up */
    }
    CK(hipGetLastError());
    out_route->resize(n ? n : 1);
    out_ids->resize(n);
    CK(hipMemcpy(out_route->data(), d.route, out_route->size() * sizeof(Route), hipMemcpyDeviceToHost));
    CK(hipMemcpy(out_count, d.count, sizeof(uint32_t), hipMemcpyDeviceToHost));
    CK(hipMemcpy(out_ids->data(), d.ids, n * sizeof(int32_t), hipMemcpyDeviceToHost));
    CK(hipEventDestroy(a));
    CK(hipEventDestroy(b));
    return iters > 1 ? ms / (iters - 1) : 0.f;
}

int main(int argc, char** argv) {
    const bool bench = argc > 1 && !std::strcmp(argv[1], "--bench");
    int wave = 0;
    CK(hipDeviceGetAttribute(&wave, hipDeviceAttributeWarpSize, 0));
    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, 0));
    std::printf("device %s wave%d\n", prop.gcnArchName, wave);

    std::vector<Case> cases;
    const uint32_t rows[] = {1, 2, 3, 4, 8, 9, 16, 31, 32, 33, 63, 64, 65, 128, 255, 256, 257, 1024, 4096};
    const uint32_t topk[] = {1, 2, 4, 8, 10};
    const double owned[] = {0.0, 0.01, 0.1, 0.5, 0.9, 1.0};
    for (uint32_t r : rows)
        for (uint32_t k : topk)
            for (double o : owned) cases.push_back({r, k, 128, o, (r & 1) ? 64 : 0, ((r + k) & 1) != 0});

    const uint32_t max_n = 4096 * 10;
    Dev d{};
    CK(hipMalloc(&d.ids, max_n * sizeof(int32_t)));
    CK(hipMalloc(&d.ew, max_n * sizeof(uint16_t)));
    CK(hipMalloc(&d.own, 1024));
    CK(hipMalloc(&d.route, max_n * sizeof(Route)));
    CK(hipMalloc(&d.count, sizeof(uint32_t)));

    std::mt19937 rng(29);
    int failed = 0;
    if (bench) std::printf("%6s %4s %6s %10s %10s %8s\n", "rows", "topk", "owned", "ref_us", "new_us", "delta");
    for (const Case& c : cases) {
        const uint32_t n = c.rows * c.top_k;
        std::vector<uint8_t> own(c.n_expert);
        for (auto& o : own) o = std::uniform_real_distribution<double>(0, 1)(rng) < c.owned;
        std::vector<int32_t> ids(n);
        /* experts in and around this rank's range, including invalid ones */
        std::uniform_int_distribution<int32_t> pick(c.base - 8, c.base + (int32_t)c.n_expert + 8);
        for (auto& v : ids) v = pick(rng);
        std::vector<uint16_t> ew(n);
        for (uint32_t i = 0; i < n; ++i) ew[i] = (uint16_t)(rng() & 0xffff);
        CK(hipMemcpy(d.own, own.data(), own.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy(d.ew, ew.data(), n * sizeof(uint16_t), hipMemcpyHostToDevice));

        const int iters = bench ? 51 : 1;
        std::vector<Route> rr, nr_;
        std::vector<int32_t> ri, ni;
        uint32_t rc = 0, nc = 0;
        const float rt = run(ref_kernel, c, d, ids, iters, &rr, &rc, &ri);
        const float nt = run(new_kernel, c, d, ids, iters, &nr_, &nc, &ni);

        bool ok = rc == nc && keys(rr, rc) == keys(nr_, nc) && ri == ni;
        /* independently: every route names a candidate routed here, once */
        std::vector<uint8_t> seen(n, 0);
        for (uint32_t p = 0; ok && p < nc; ++p) {
            const uint32_t i = (uint32_t)nr_[p].token * c.top_k + nr_[p].slot;
            const int32_t le = ids[i] - c.base;
            ok = i < n && !seen[i] && le == nr_[p].expert && le >= 0 && (uint32_t)le < c.n_expert && own[le] &&
                 nr_[p].weight_bf16 == ew[i];
            if (ok) seen[i] = 1;
        }
        if (!ok) {
            ++failed;
            std::printf("FAIL rows=%u topk=%u owned=%.2f clear=%d: ref %u routes, new %u\n", c.rows, c.top_k, c.owned,
                        c.clear, rc, nc);
        }
        if (bench && c.owned >= 0.1)
            std::printf("%6u %4u %6.2f %10.2f %10.2f %+7.1f%%\n", c.rows, c.top_k, c.owned, rt * 1e3f, nt * 1e3f,
                        rt > 0 ? (nt / rt - 1) * 100 : 0.0);
    }
    std::printf("%zu cases, %d failed\n", cases.size(), failed);
    return failed ? 1 : 0;
}
