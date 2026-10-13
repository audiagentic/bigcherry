#pragma once

#include "rad_abi.h"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <initializer_list>
#include <limits>
#include <string>
#include <utility>
#include <vector>

/* GPU half of r3_moe_helper. This header is included after Container is defined in
 * r3_moe_helper.cpp, so it can consume the installed container ABI without linking core/private
 * Radiance code. The only executable ABI it uses is libr4d's rad_plugin surface. */
namespace r3_moe_gpu {

constexpr int64_t kExpertWidth = 640;
constexpr int64_t kGateUpWidth = 2 * kExpertWidth;
constexpr int64_t kGroup = 128;
constexpr const char* kDtype = "w4nl64a8h";

__global__ void lane_route_count(const r3::moe_shm::Route* route, uint32_t n, uint32_t rows,
                                 uint32_t* counts, uint32_t* token_slot, int32_t* flat,
                                 uint16_t* ew) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const r3::moe_shm::Route r = route[i];
    if (r.token < 0 || (uint32_t)r.token >= rows || r.expert < 0 ||
        r.expert >= (int32_t)r3::moe_shm::kExperts) {
        flat[i] = -1;
        return;
    }
    const uint32_t slot = atomicAdd(token_slot + r.token, 1u);
    if (slot >= r3::moe_shm::kTopK) {
        flat[i] = -1;
        return;
    }
    const uint32_t f = (uint32_t)r.token * r3::moe_shm::kTopK + slot;
    flat[i] = (int32_t)f;
    ew[f] = r.weight_bf16;
    atomicAdd(counts + (uint32_t)r.expert, 1u);
}

__global__ void lane_route_prefix(const uint32_t* counts, int32_t* off, uint32_t* cursor) {
    if (blockIdx.x || threadIdx.x) return;
    uint32_t at = 0;
    for (uint32_t e = 0; e < r3::moe_shm::kExperts; ++e) {
        off[e] = (int32_t)at;
        cursor[e] = at;
        at += counts[e];
    }
    off[r3::moe_shm::kExperts] = (int32_t)at;
}

__global__ void lane_route_scatter(const r3::moe_shm::Route* route, uint32_t n,
                                   const int32_t* flat, uint32_t* cursor,
                                   int32_t* sorted, int32_t* sorted_expert) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || flat[i] < 0) return;
    const uint32_t e = (uint32_t)route[i].expert;
    const uint32_t pos = atomicAdd(cursor + e, 1u);
    sorted[pos] = flat[i];
    sorted_expert[pos] = (int32_t)e;
}

__global__ void lane_mask_middle(const uint8_t* src, uint8_t* dst, const int32_t* sorted_expert,
                                 uint32_t routes, uint32_t rank) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = routes * (uint32_t)kExpertWidth;
    if (i >= total) return;
    const uint32_t r = i / (uint32_t)kExpertWidth;
    const uint32_t k = i - r * (uint32_t)kExpertWidth;
    const int32_t e = sorted_expert[r];
    const bool wide = e >= 0 && rank == ((uint32_t)e & 1u);
    const bool keep = e >= 0 && (wide ? k < 384u : k >= 384u);
    dst[i] = keep ? src[i] : 0;
}

inline bool hip_ok(hipError_t e, const char* what) {
    if (e == hipSuccess) return true;
    std::fprintf(stderr, "r3_moe_helper: %s: %s\n", what, hipGetErrorString(e));
    return false;
}

inline RadParam pi(const char* key, int64_t v) {
    RadParam p{};
    p.key = key; p.kind = RAD_P_INT; p.ival = v;
    return p;
}
inline RadParam ps(const char* key, const char* v) {
    RadParam p{};
    p.key = key; p.kind = RAD_P_STR; p.sval = v;
    return p;
}
inline RadTensor tensor(void* data, uint32_t dtype, std::initializer_list<int64_t> dims) {
    RadTensor t{};
    t.data = data; t.dtype = dtype; t.rank = (uint32_t)dims.size();
    size_t i = 0;
    for (int64_t d : dims) t.shape[i++] = d;
    int64_t stride = 1;
    for (int j = (int)t.rank - 1; j >= 0; --j) {
        t.stride[j] = stride;
        stride *= t.shape[j];
    }
    return t;
}
inline RadTensor absent() { return RadTensor{}; }

struct Buffer {
    void* p = nullptr;
    size_t bytes = 0;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (p) (void)hipFree(p); }
    bool ensure(size_t need) {
        if (need <= bytes) return true;
        if (p) { if (!hip_ok(hipFree(p), "hipFree grow buffer")) return false; p = nullptr; bytes = 0; }
        if (!need) need = 16;
        if (!hip_ok(hipMalloc(&p, need), "hipMalloc work buffer")) return false;
        bytes = need;
        return true;
    }
};

struct HostBounce {
    void* p = nullptr;
    size_t bytes = 0;
    ~HostBounce() { if (p) (void)hipHostFree(p); }
    bool ensure(size_t need) {
        if (need <= bytes) return true;
        if (p) { (void)hipHostFree(p); p = nullptr; bytes = 0; }
        if (!hip_ok(hipHostMalloc(&p, need, hipHostMallocPortable), "hipHostMalloc load bounce")) return false;
        bytes = need;
        return true;
    }
};

struct StoredWeight {
    void* code = nullptr;
    void* scale = nullptr;
    uint32_t code_dtype = 0;
    uint32_t scale_dtype = 0;
    int64_t code_bytes = 0;
    int64_t scale_bytes = 0;
};

struct Expert {
    StoredWeight gu;
    StoredWeight dn;
    bool live = false;
};

struct LayerTables {
    void* gu_code[2]{};
    void* gu_scale[2]{};
    void* dn_code[2]{};
    void* dn_scale[2]{};
    uint32_t gu_code_dtype = 0, gu_scale_dtype = 0;
    uint32_t dn_code_dtype = 0, dn_scale_dtype = 0;
};

struct LaneWork {
    hipStream_t stream = nullptr;
    Buffer d_counts, d_cursor, d_token_slot, d_flat, d_sorted, d_sorted_expert, d_off, d_ew;
    Buffer d_aq, d_as, d_gu, d_mid, d_mid_s, d_mid_rank, d_down, scratch;
};

struct WeightPlan {
    RadLayout code{};
    RadLayout scale{};
    std::vector<RadParam> params;
    std::array<int, 2> code_sel{};
    std::array<int, 2> scale_sel{};
    std::array<RadTensor, 2> code_planes{};
    std::array<RadTensor, 2> scale_planes{};
};

class Runner {
public:
    explicit Runner(const Container& model) : model_(model) {}
    ~Runner() {
        if (stream_) (void)hipStreamDestroy(stream_);
        for (auto& l : lane_) if (l.stream) (void)hipStreamDestroy(l.stream);
        for (void* p : persistent_) if (p) (void)hipFree(p);
        if (plugin_opened_ && plugin_close_) plugin_close_();
        if (so_) dlclose(so_);
    }

    bool open() {
        if (!hip_ok(hipSetDevice(0), "hipSetDevice(0)")) return false;
        hipDeviceProp_t prop{};
        if (!hip_ok(hipGetDeviceProperties(&prop, 0), "hipGetDeviceProperties")) return false;
        arch_ = prop.gcnArchName;
        const size_t colon = arch_.find(':');
        if (colon != std::string::npos) arch_.resize(colon);
        if (arch_ != "gfx1201") {
            std::fprintf(stderr,
                "r3_moe_helper: visible GPU is %s, want gfx1201 (set R3_MOE_HELPER_CPU=1 for host path)\n",
                arch_.c_str());
            return false;
        }
        if (!load_plugin()) return false;
        if (!hip_ok(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "hipStreamCreate")) return false;
        for (uint32_t r = 0; r < r3::moe_shm::kMaxRanks; ++r)
            if (!hip_ok(hipStreamCreateWithFlags(&lane_[r].stream, hipStreamNonBlocking), "hipStreamCreate lane"))
                return false;
        experts_.resize((size_t)model_.layers * r3::moe_shm::kExperts);
        tables_.resize(model_.layers);
        return true;
    }

    const std::string& arch() const { return arch_; }

    int64_t expert_bytes(uint32_t layer, uint32_t expert) {
        const RadFileEntry* gu = model_.weight(layer, expert, 0);
        const RadFileEntry* dn = model_.weight(layer, expert, 1);
        if (!gu || !dn) return -1;
        WeightPlan pg, pd;
        if (!plan_weight(gu, true, &pg) || !plan_weight(dn, false, &pd)) return -1;
        return pg.code.bytes + pg.scale.bytes + pd.code.bytes + pd.scale.bytes;
    }

    bool upload(const std::vector<std::pair<uint32_t, uint32_t>>& owned) {
        constexpr size_t kLoadBounceBytes = 64ull * 1024ull * 1024ull;
        if (!load_bounce_.ensure(kLoadBounceBytes)) return false;
        /* One arena for every resident expert: tens of thousands of small hipMallocs exhaust the
         * device on allocation overhead long before the budget does. */
        size_t total = 0;
        for (auto [l, e] : owned) {
            WeightPlan pg, pd;
            if (!plan_weight(model_.weight(l, e, 0), true, &pg) || !plan_weight(model_.weight(l, e, 1), false, &pd))
                return false;
            total += arena_round(pg.code.bytes) + arena_round(pg.scale.bytes) +
                     arena_round(pd.code.bytes) + arena_round(pd.scale.bytes);
        }
        if (!hip_ok(hipMalloc(&arena_, total ? total : 16), "hipMalloc resident expert arena")) return false;
        persistent_.push_back(arena_);
        arena_size_ = total; arena_used_ = 0;
        for (auto [l, e] : owned) {
            Expert& x = experts_[(size_t)l * r3::moe_shm::kExperts + e];
            if (!upload_weight(model_.weight(l, e, 0), true, &x.gu) ||
                !upload_weight(model_.weight(l, e, 1), false, &x.dn)) return false;
            x.live = true;
        }
        return build_tables();
    }

    int compute(void* map, r3::moe_shm::Request* q) {
        using namespace r3::moe_shm;
        if (!map || !q || q->rank >= kMaxRanks || q->rank_count != kMaxRanks || q->layer >= model_.layers ||
            q->rows == 0 || q->rows > kMaxRows || q->hidden != kHidden ||
            q->route_count == 0 || q->route_count > kMaxRoutes ||
            q->activation_stride != kHidden * (uint32_t)sizeof(uint16_t)) return -EINVAL;

        const Route* route = ptr<Route>(map, q->route_offset);
        const uint16_t* hact = ptr<uint16_t>(map, q->activation_offset);
        const uint32_t rows = q->rows;
        const uint32_t tr = rows * kTopK;

        std::vector<std::vector<uint32_t>> by_exp(kExperts);
        std::vector<uint16_t> ew((size_t)rows * kTopK, 0);
        std::vector<uint8_t> next_slot(rows, 0);
        for (uint32_t i = 0; i < q->route_count; ++i) {
            const Route& r = route[i];
            if (r.token < 0 || (uint32_t)r.token >= rows || r.expert < 0 ||
                r.expert >= (int32_t)kExperts || !owns(q->layer, (uint32_t)r.expert)) return -EINVAL;
            uint8_t& slot = next_slot[(uint32_t)r.token];
            if (slot >= kTopK) return -EINVAL;
            const uint32_t flat = (uint32_t)r.token * kTopK + slot++;
            ew[flat] = r.weight_bf16;
            by_exp[(uint32_t)r.expert].push_back(flat);
        }

        std::vector<int32_t> sorted(tr, -1);
        std::array<int32_t, kExperts + 1> off{};
        uint32_t at = 0;
        for (uint32_t e = 0; e < kExperts; ++e) {
            off[e] = (int32_t)at;
            for (uint32_t flat : by_exp[e]) sorted[at++] = (int32_t)flat;
        }
        off[kExperts] = (int32_t)at;
        if (at != q->route_count) return -EINVAL;

        if (!ensure_work(rows, tr)) return -ENOMEM;
        const size_t act_bytes = (size_t)rows * kHidden * sizeof(uint16_t);
        if (!hip_ok(hipMemcpyAsync(d_act_.p, hact, act_bytes, hipMemcpyHostToDevice, stream_),
                    "upload activations")) return -EIO;
        if (!hip_ok(hipMemcpyAsync(d_sorted_.p, sorted.data(), (size_t)tr * sizeof(int32_t),
                                   hipMemcpyHostToDevice, stream_), "upload sorted routes") ||
            !hip_ok(hipMemcpyAsync(d_off_.p, off.data(), sizeof(off),
                                   hipMemcpyHostToDevice, stream_), "upload expert offsets") ||
            !hip_ok(hipMemcpyAsync(d_ew_.p, ew.data(), ew.size() * sizeof(uint16_t),
                                   hipMemcpyHostToDevice, stream_), "upload route weights"))
            return -EIO;

        /* The container's w4nl64a8h expert is Hadamard-rotated. Quantise the block input once,
         * then one grouped gate/up handles every helper-owned expert selected in this layer. */
        std::vector<RadParam> aqp = {
            pi("M", rows), pi("n", kHidden), pi("group", kGroup), ps("dtype", "bf16")
        };
        if (launch(had_quant_,
                   { tensor(d_act_.p, RAD_BF16, {rows, kHidden}),
                     tensor(d_aq_.p, RAD_F8E4M3, {rows, kHidden}),
                     tensor(d_as_.p, RAD_F32, {rows, kHidden / kGroup}) }, aqp) < 0)
            return -EIO;

        LayerTables& lt = tables_[q->layer];
        std::vector<RadParam> gp = gemm_params(rows, kGateUpWidth, kHidden, 2, "token");
        if (launch(moe_,
                   { tensor(d_aq_.p, RAD_F8E4M3, {rows, kHidden}),
                     tensor(d_as_.p, RAD_F32, {rows, kHidden / kGroup}),
                     table_tensor(lt.gu_code[0], lt.gu_code_dtype, kGateUpWidth, kHidden),
                     table_tensor(lt.gu_scale[0], lt.gu_scale_dtype, kGateUpWidth, kHidden / 64),
                     tensor(d_sorted_.p, RAD_I32, {tr}),
                     tensor(d_off_.p, RAD_I32, {kExperts + 1}),
                     table_tensor(lt.gu_code[1], lt.gu_code_dtype, kGateUpWidth, kHidden),
                     table_tensor(lt.gu_scale[1], lt.gu_scale_dtype, kGateUpWidth, kHidden / 64),
                     tensor(d_gu_.p, RAD_BF16, {tr, kGateUpWidth}) }, gp) < 0)
            return -EIO;

        std::vector<RadParam> gqp = {
            pi("M", q->route_count), pi("n", kExpertWidth), pi("group", kGroup),
            ps("dtype", "bf16"), ps("act", "silu")
        };
        if (launch(gated_quant_,
                   { tensor(d_gu_.p, RAD_BF16, {q->route_count, kGateUpWidth}),
                     tensor(d_mid_.p, RAD_F8E4M3, {q->route_count, kExpertWidth}),
                     tensor(d_mid_s_.p, RAD_F32, {q->route_count, kExpertWidth / kGroup}),
                     absent() }, gqp) < 0)
            return -EIO;

        /* One full expert is resident on the helper. The two TP partials are the two complementary
         * 128-wide K-group subsets used by the engine: 3+2 groups, parity-swapped by expert. */
        for (uint32_t rank = 0; rank < kMaxRanks; ++rank) {
            if (!hip_ok(hipMemcpyAsync(d_mid_rank_.p, d_mid_.p,
                                       (size_t)q->route_count * kExpertWidth,
                                       hipMemcpyDeviceToDevice, stream_), "copy routed middle"))
                return -EIO;
            for (uint32_t e = 0; e < kExperts; ++e) {
                const uint32_t begin = (uint32_t)off[e], end = (uint32_t)off[e + 1];
                if (begin == end) continue;
                const bool wide = rank == (e & 1u);
                const size_t lo = wide ? 384u : 0u;
                const size_t width = wide ? 256u : 384u;
                uint8_t* base = static_cast<uint8_t*>(d_mid_rank_.p) +
                                (size_t)begin * kExpertWidth + lo;
                if (!hip_ok(hipMemset2DAsync(base, kExpertWidth, 0, width, end - begin, stream_),
                            "mask TP middle groups")) return -EIO;
            }

            std::vector<RadParam> dp = gemm_params(rows, kHidden, kExpertWidth, 1, "sorted");
            if (launch(moe_,
                       { tensor(d_mid_rank_.p, RAD_F8E4M3, {tr, kExpertWidth}),
                         tensor(d_mid_s_.p, RAD_F32, {tr, kExpertWidth / kGroup}),
                         table_tensor(lt.dn_code[0], lt.dn_code_dtype, kHidden, kExpertWidth),
                         table_tensor(lt.dn_scale[0], lt.dn_scale_dtype, kHidden, kExpertWidth / 64),
                         tensor(d_sorted_.p, RAD_I32, {tr}),
                         tensor(d_off_.p, RAD_I32, {kExperts + 1}),
                         table_tensor(lt.dn_code[1], lt.dn_code_dtype, kHidden, kExpertWidth),
                         table_tensor(lt.dn_scale[1], lt.dn_scale_dtype, kHidden, kExpertWidth / 64),
                         tensor(d_down_.p, RAD_BF16, {tr, kHidden}) }, dp) < 0)
                return -EIO;

            std::vector<RadParam> mp = {
                pi("M", rows), pi("n", kHidden), pi("top_k", kTopK), ps("dtype", "bf16")
            };
            if (launch(gather_,
                       { tensor(d_down_.p, RAD_BF16, {tr, kHidden}),
                         tensor(d_ew_.p, RAD_BF16, {rows, kTopK}),
                         tensor(d_sorted_.p, RAD_I32, {tr}),
                         absent(), absent(),
                         tensor(d_result_.p, RAD_BF16, {rows, kHidden}) }, mp) < 0)
                return -EIO;

            void* dst = ptr<void>(map, q->result_offset[rank]);
            if (!hip_ok(hipMemcpyAsync(dst, d_result_.p, act_bytes,
                                       hipMemcpyDeviceToHost, stream_), "download TP partial"))
                return -EIO;
        }
        return hip_ok(hipStreamSynchronize(stream_), "helper stream sync") ? 0 : -EIO;
    }


    int enqueue_lane(void* host_map, void* device_map, r3::moe_shm::Request* q,
                     uint32_t rank, uint32_t sequence) {
        using namespace r3::moe_shm;
        if (!host_map || !device_map || !q || rank >= kMaxRanks || q->rank != rank ||
            q->rank_count != kMaxRanks || q->layer >= model_.layers || q->rows == 0 ||
            q->rows > kMaxRows || q->hidden != kHidden ||
            q->route_count > q->rows * kTopK || q->activation_stride != kHidden * sizeof(uint16_t))
            return -EINVAL;
        LaneWork& w = lane_[rank];
        const uint32_t rows = q->rows;
        const uint32_t tr = rows * kTopK;
        if (!ensure_lane_work(w, rows, q->route_count, tr)) return -ENOMEM;
        auto* hhost = static_cast<Header*>(host_map);
        auto* dbase = static_cast<uint8_t*>(device_map);
        auto dev_at = [&](const void* hp) -> void* {
            return dbase + (static_cast<const uint8_t*>(hp) - static_cast<const uint8_t*>(host_map));
        };
        auto* request_seq = static_cast<uint32_t*>(dev_at(&hhost->lane[rank].request_sequence));
        auto* done_seq = static_cast<uint32_t*>(dev_at(&hhost->lane[rank].done_sequence));
        auto* route = static_cast<const Route*>(dev_at(ptr<Route>(host_map, q->route_offset)));
        auto* act = static_cast<uint16_t*>(dev_at(ptr<uint16_t>(host_map, q->activation_offset)));
        auto* dst = static_cast<uint16_t*>(dev_at(ptr<uint16_t>(host_map, q->result_offset[rank])));

        if (!hip_ok(hipStreamWaitValue32(w.stream, request_seq, sequence,
                                         hipStreamWaitValueGte, 0xffffffffu),
                    "lane wait request")) return -EIO;
        if (q->route_count == 0) {
            const size_t result_bytes = (size_t)rows * kHidden * sizeof(uint16_t);
            if (!hip_ok(hipMemsetAsync(dst, 0, result_bytes, w.stream), "zero empty lane result") ||
                !hip_ok(hipStreamWriteValue32(w.stream, done_seq, sequence, 0), "lane write empty done"))
                return -EIO;
            return 0;
        }
        if (!hip_ok(hipMemsetAsync(w.d_counts.p, 0, kExperts * sizeof(uint32_t), w.stream), "clear route counts") ||
            !hip_ok(hipMemsetAsync(w.d_token_slot.p, 0, rows * sizeof(uint32_t), w.stream), "clear token slots") ||
            !hip_ok(hipMemsetAsync(w.d_ew.p, 0, tr * sizeof(uint16_t), w.stream), "clear route weights") ||
            !hip_ok(hipMemsetAsync(w.d_sorted.p, 0xff, tr * sizeof(int32_t), w.stream), "clear sorted routes") ||
            !hip_ok(hipMemsetAsync(w.d_sorted_expert.p, 0xff, tr * sizeof(int32_t), w.stream), "clear sorted experts"))
            return -EIO;
        const dim3 block(256);
        const dim3 route_grid((q->route_count + block.x - 1) / block.x);
        hipLaunchKernelGGL(lane_route_count, route_grid, block, 0, w.stream,
                           route, q->route_count, rows,
                           static_cast<uint32_t*>(w.d_counts.p),
                           static_cast<uint32_t*>(w.d_token_slot.p),
                           static_cast<int32_t*>(w.d_flat.p),
                           static_cast<uint16_t*>(w.d_ew.p));
        hipLaunchKernelGGL(lane_route_prefix, dim3(1), dim3(1), 0, w.stream,
                           static_cast<const uint32_t*>(w.d_counts.p),
                           static_cast<int32_t*>(w.d_off.p),
                           static_cast<uint32_t*>(w.d_cursor.p));
        hipLaunchKernelGGL(lane_route_scatter, route_grid, block, 0, w.stream,
                           route, q->route_count,
                           static_cast<const int32_t*>(w.d_flat.p),
                           static_cast<uint32_t*>(w.d_cursor.p),
                           static_cast<int32_t*>(w.d_sorted.p),
                           static_cast<int32_t*>(w.d_sorted_expert.p));
        if (!hip_ok(hipGetLastError(), "lane route kernels")) return -EIO;

        std::vector<RadParam> aqp = {
            pi("M", rows), pi("n", kHidden), pi("group", kGroup), ps("dtype", "bf16")
        };
        if (launch_on(had_quant_,
                      { tensor(act, RAD_BF16, {rows, kHidden}),
                        tensor(w.d_aq.p, RAD_F8E4M3, {rows, kHidden}),
                        tensor(w.d_as.p, RAD_F32, {rows, kHidden / kGroup}) },
                      aqp, w.stream, w.scratch) < 0) return -EIO;

        LayerTables& lt = tables_[q->layer];
        std::vector<RadParam> gp = gemm_params(rows, kGateUpWidth, kHidden, 2, "token");
        if (launch_on(moe_,
                      { tensor(w.d_aq.p, RAD_F8E4M3, {rows, kHidden}),
                        tensor(w.d_as.p, RAD_F32, {rows, kHidden / kGroup}),
                        table_tensor(lt.gu_code[0], lt.gu_code_dtype, kGateUpWidth, kHidden),
                        table_tensor(lt.gu_scale[0], lt.gu_scale_dtype, kGateUpWidth, kHidden / 64),
                        tensor(w.d_sorted.p, RAD_I32, {tr}),
                        tensor(w.d_off.p, RAD_I32, {kExperts + 1}),
                        table_tensor(lt.gu_code[1], lt.gu_code_dtype, kGateUpWidth, kHidden),
                        table_tensor(lt.gu_scale[1], lt.gu_scale_dtype, kGateUpWidth, kHidden / 64),
                        tensor(w.d_gu.p, RAD_BF16, {tr, kGateUpWidth}) },
                      gp, w.stream, w.scratch) < 0) return -EIO;

        std::vector<RadParam> gqp = {
            pi("M", q->route_count), pi("n", kExpertWidth), pi("group", kGroup),
            ps("dtype", "bf16"), ps("act", "silu")
        };
        if (launch_on(gated_quant_,
                      { tensor(w.d_gu.p, RAD_BF16, {q->route_count, kGateUpWidth}),
                        tensor(w.d_mid.p, RAD_F8E4M3, {q->route_count, kExpertWidth}),
                        tensor(w.d_mid_s.p, RAD_F32, {q->route_count, kExpertWidth / kGroup}),
                        absent() }, gqp, w.stream, w.scratch) < 0) return -EIO;

        const uint32_t middle_n = q->route_count * (uint32_t)kExpertWidth;
        const dim3 middle_grid((middle_n + block.x - 1) / block.x);
        hipLaunchKernelGGL(lane_mask_middle, middle_grid, block, 0, w.stream,
                           static_cast<const uint8_t*>(w.d_mid.p),
                           static_cast<uint8_t*>(w.d_mid_rank.p),
                           static_cast<const int32_t*>(w.d_sorted_expert.p),
                           q->route_count, rank);
        if (!hip_ok(hipGetLastError(), "lane TP mask")) return -EIO;

        std::vector<RadParam> dp = gemm_params(rows, kHidden, kExpertWidth, 1, "sorted");
        if (launch_on(moe_,
                      { tensor(w.d_mid_rank.p, RAD_F8E4M3, {tr, kExpertWidth}),
                        tensor(w.d_mid_s.p, RAD_F32, {tr, kExpertWidth / kGroup}),
                        table_tensor(lt.dn_code[0], lt.dn_code_dtype, kHidden, kExpertWidth),
                        table_tensor(lt.dn_scale[0], lt.dn_scale_dtype, kHidden, kExpertWidth / 64),
                        tensor(w.d_sorted.p, RAD_I32, {tr}),
                        tensor(w.d_off.p, RAD_I32, {kExperts + 1}),
                        table_tensor(lt.dn_code[1], lt.dn_code_dtype, kHidden, kExpertWidth),
                        table_tensor(lt.dn_scale[1], lt.dn_scale_dtype, kHidden, kExpertWidth / 64),
                        tensor(w.d_down.p, RAD_BF16, {tr, kHidden}) },
                      dp, w.stream, w.scratch) < 0) return -EIO;

        std::vector<RadParam> mp = {
            pi("M", rows), pi("n", kHidden), pi("top_k", kTopK), ps("dtype", "bf16")
        };
        if (launch_on(gather_,
                      { tensor(w.d_down.p, RAD_BF16, {tr, kHidden}),
                        tensor(w.d_ew.p, RAD_BF16, {rows, kTopK}),
                        tensor(w.d_sorted.p, RAD_I32, {tr}),
                        absent(), absent(), tensor(dst, RAD_BF16, {rows, kHidden}) },
                      mp, w.stream, w.scratch) < 0) return -EIO;

        if (!hip_ok(hipStreamWriteValue32(w.stream, done_seq, sequence, 0), "lane write done"))
            return -EIO;
        return 0;
    }

private:
    const Container& model_;
    std::string arch_;
    void* so_ = nullptr;
    void (*plugin_close_)() = nullptr;
    bool plugin_opened_ = false;
    hipStream_t stream_ = nullptr;
    std::array<LaneWork, r3::moe_shm::kMaxRanks> lane_{};
    uint32_t gu_code_dtype_ = 0, gu_scale_dtype_ = 0;
    uint32_t dn_code_dtype_ = 0, dn_scale_dtype_ = 0;
    const RadKernelInfo* moe_ = nullptr;
    const RadKernelInfo* had_quant_ = nullptr;
    const RadKernelInfo* gated_quant_ = nullptr;
    const RadKernelInfo* gather_ = nullptr;
    std::vector<Expert> experts_;
    std::vector<LayerTables> tables_;
    std::vector<void*> persistent_;
    void* arena_ = nullptr;
    size_t arena_size_ = 0, arena_used_ = 0;

    HostBounce load_bounce_;
    Buffer d_act_, d_aq_, d_as_, d_sorted_, d_off_, d_ew_, d_gu_;
    Buffer d_mid_, d_mid_s_, d_mid_rank_, d_down_, d_result_, scratch_;

    template <class T> bool symbol(const char* name, T* out, bool required = true) {
        dlerror();
        void* p = dlsym(so_, name);
        const char* e = dlerror();
        if (!p || e) {
            if (required) std::fprintf(stderr, "r3_moe_helper: libr4d missing %s: %s\n",
                                       name, e ? e : "null");
            return !required;
        }
        std::memcpy(out, &p, sizeof(p));
        return true;
    }

    std::string plugin_path() const {
        if (const char* e = std::getenv("R3_MOE_HELPER_R4D"); e && *e) return e;
        if (const char* e = std::getenv("RADIANCE_HOME"); e && *e) {
            std::string h(e);
            const size_t colon = h.find(':');
            if (colon != std::string::npos) h.resize(colon);
            return h + "/kernels/libr4d.so";
        }
#ifdef RAD_DEFAULT_HOME
        return std::string(RAD_DEFAULT_HOME) + "/kernels/libr4d.so";
#else
        return "libr4d.so";
#endif
    }

    bool load_plugin() {
        const std::string path = plugin_path();
        so_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!so_) {
            std::fprintf(stderr, "r3_moe_helper: dlopen %s: %s\n", path.c_str(), dlerror());
            return false;
        }
        uint32_t (*abi)() = nullptr;
        const RadPluginInfo* (*info)() = nullptr;
        int (*count)() = nullptr;
        const RadKernelInfo* (*at)(int) = nullptr;
        int (*open_fn)() = nullptr;
        if (!symbol("rad_plugin_abi_version", &abi) || !symbol("rad_plugin_info", &info) ||
            !symbol("rad_kernel_count", &count) || !symbol("rad_kernel_at", &at)) return false;
        /* rad_plugin_open/close are optional in the ABI (rad_abi.h); libr4d exports neither. */
        (void)symbol("rad_plugin_open", &open_fn, false);
        (void)symbol("rad_plugin_close", &plugin_close_, false);
        if (abi() != RAD_ABI_VERSION) {
            std::fprintf(stderr, "r3_moe_helper: libr4d ABI %u, helper ABI %u\n",
                         abi(), (unsigned)RAD_ABI_VERSION);
            return false;
        }
        const RadPluginInfo* inf = info();
        if (!inf || !inf->build_target || !std::strstr(inf->build_target, "gfx1201")) {
            std::fprintf(stderr, "r3_moe_helper: libr4d is not a gfx1201 plugin\n");
            return false;
        }
        if (open_fn && open_fn() != RAD_OK) {
            std::fprintf(stderr, "r3_moe_helper: libr4d rad_plugin_open refused this device\n");
            return false;
        }
        plugin_opened_ = true;
        auto find = [&](const char* op, const char* preferred) -> const RadKernelInfo* {
            const RadKernelInfo* fallback = nullptr;
            for (int i = 0; i < count(); ++i) {
                const RadKernelInfo* r = at(i);
                if (!r || !r->op || std::strcmp(r->op, op) != 0) continue;
                if (!fallback) fallback = r;
                if (preferred && r->name && std::strcmp(r->name, preferred) == 0) return r;
            }
            return fallback;
        };
        moe_ = find("moe_gemm_q", "moe_gemm_w4a8");
        had_quant_ = find("had_quant_act_fp8", "had_quant_act_fp8");
        gated_quant_ = find("gated_had_quant_fp8", "gated_had_quant_fp8");
        gather_ = find("moe_gather", "moe_gather_bf16");
        if (!moe_ || !moe_->layout || !moe_->relayout || !had_quant_ || !gated_quant_ || !gather_) {
            std::fprintf(stderr, "r3_moe_helper: installed libr4d lacks required MoE/quant rows\n");
            return false;
        }
        return true;
    }

    static size_t arena_round(int64_t n) { return ((size_t)(n > 0 ? n : 1) + 255u) & ~(size_t)255u; }

    void* arena_take(size_t n) {
        const size_t r = arena_round((int64_t)n);
        if (!arena_ || arena_used_ + r > arena_size_) {
            std::fprintf(stderr, "r3_moe_helper: resident expert arena overflow\n");
            return nullptr;
        }
        void* p = (uint8_t*)arena_ + arena_used_;
        arena_used_ += r;
        return p;
    }

    void* alloc_persistent(size_t n) {
        void* p = nullptr;
        if (!hip_ok(hipMalloc(&p, n ? n : 16), "hipMalloc resident expert")) return nullptr;
        persistent_.push_back(p);
        return p;
    }

    bool canonical_pair(const RadFileEntry* e, const char* role0, int level0,
                        const char* role1, int level1,
                        std::array<int,2>* sel, std::array<RadTensor,2>* out) const {
        const RadEncoding* enc = model_.encoding(e);
        int s[2] = {-1, -1};
        s[0] = level0 >= 0 ? rad_enc_find_level(enc, role0, level0) : rad_enc_find(enc, role0);
        s[1] = level1 >= 0 ? rad_enc_find_level(enc, role1, level1) : rad_enc_find(enc, role1);
        if (s[0] < 0 || s[1] < 0) return false;
        int64_t rows = 1, cols = 1;
        rad_enc_view(e->rank, e->shape, &rows, &cols);
        for (int i = 0; i < 2; ++i) {
            const RadEncPlane& ep = enc->plane[s[i]];
            int64_t pr = 0, pc = 0;
            rad_enc_plane_dims(&ep, rows, cols, &pr, &pc);
            const RadFilePlane* fp = model_.plane(e, s[i]);
            (*sel)[i] = s[i];
            (*out)[i] = tensor(nullptr, ep.dtype, {pr, pc});
        }
        return true;
    }

    std::vector<RadParam> layout_params(const RadFileEntry* e, bool gate_up) const {
        int64_t nr = 1, nc = 1;
        rad_enc_view(e->rank, e->shape, &nr, &nc);
        return {
            pi("M", 1), pi("N", nr), pi("K", nc), pi("N_odd", nr), pi("K_odd", nc),
            pi("parts", gate_up ? 2 : 1), pi("n_expert", r3::moe_shm::kExperts),
            pi("top_k", r3::moe_shm::kTopK), pi("group", kGroup),
            ps("a_order", gate_up ? "token" : "sorted"), ps("dtype", kDtype)
        };
    }

    bool plan_weight(const RadFileEntry* e, bool gate_up, WeightPlan* out) {
        if (!e || !out || e->enc >= model_.h->enc_count) return false;
        const RadEncoding* enc = model_.encoding(e);
        if (!enc || !rad_enc_valid(enc)) return false;
        out->params = layout_params(e, gate_up);
        if (!canonical_pair(e, "codes", -1, "table", -1, &out->code_sel, &out->code_planes) ||
            !canonical_pair(e, "scale", 0, "scale", 1, &out->scale_sel, &out->scale_planes))
            return false;
        const int rc0 = moe_->layout(out->params.data(), (int)out->params.size(), 2, enc,
                                     out->code_sel.data(), out->code_planes.data(), 2, &out->code);
        const int rc1 = moe_->layout(out->params.data(), (int)out->params.size(), 3, enc,
                                     out->scale_sel.data(), out->scale_planes.data(), 2, &out->scale);
        return rc0 == RAD_OK && rc1 == RAD_OK && out->code.bytes > 0 && out->scale.bytes > 0;
    }

    bool load_plane_data(const RadFileEntry* e, int sel, RadTensor* t, std::vector<uint8_t>* out) {
        if (!e || !t || !out || sel < 0 || (uint32_t)sel >= e->n_planes) return false;
        const RadFilePlane* fp = model_.plane(e, sel);
        out->resize((size_t)fp->bytes);
        size_t done = 0;
        while (done < out->size()) {
            const size_t n = std::min(load_bounce_.bytes, out->size() - done);
            if (!model_.pread_full(fp->offset + done, load_bounce_.p, n)) return false;
            std::memcpy(out->data() + done, load_bounce_.p, n);
            done += n;
        }
        t->data = out->data();
        return true;
    }

    bool upload_weight(const RadFileEntry* e, bool gate_up, StoredWeight* out) {
        WeightPlan p;
        if (!plan_weight(e, gate_up, &p)) {
            std::fprintf(stderr, "r3_moe_helper: expert weight is not %s-compatible\n", kDtype);
            return false;
        }
        if ((size_t)p.code.bytes > load_bounce_.bytes || (size_t)p.scale.bytes > load_bounce_.bytes) {
            std::fprintf(stderr, "r3_moe_helper: relayout output exceeds 64 MiB load bounce\n");
            return false;
        }
        std::array<std::vector<uint8_t>, 2> code_src, scale_src;
        auto code_planes = p.code_planes;
        auto scale_planes = p.scale_planes;
        for (int i = 0; i < 2; ++i) {
            if (!load_plane_data(e, p.code_sel[i], &code_planes[i], &code_src[i]) ||
                !load_plane_data(e, p.scale_sel[i], &scale_planes[i], &scale_src[i])) {
                std::fprintf(stderr, "r3_moe_helper: pread expert plane failed\n");
                return false;
            }
        }
        const RadEncoding* enc = model_.encoding(e);
        out->code = arena_take((size_t)p.code.bytes);
        out->scale = arena_take((size_t)p.scale.bytes);
        if (!out->code || !out->scale) return false;
        if (moe_->relayout(p.params.data(), (int)p.params.size(), 2, enc,
                           p.code_sel.data(), code_planes.data(), 2,
                           load_bounce_.p, p.code.bytes) != RAD_OK ||
            !hip_ok(hipMemcpyAsync(out->code, load_bounce_.p, (size_t)p.code.bytes,
                                   hipMemcpyHostToDevice, stream_), "upload expert codes") ||
            !hip_ok(hipStreamSynchronize(stream_), "upload expert codes sync") ||
            moe_->relayout(p.params.data(), (int)p.params.size(), 3, enc,
                           p.scale_sel.data(), scale_planes.data(), 2,
                           load_bounce_.p, p.scale.bytes) != RAD_OK ||
            !hip_ok(hipMemcpyAsync(out->scale, load_bounce_.p, (size_t)p.scale.bytes,
                                   hipMemcpyHostToDevice, stream_), "upload expert scales") ||
            !hip_ok(hipStreamSynchronize(stream_), "upload expert scales sync")) {
            std::fprintf(stderr, "r3_moe_helper: libr4d refused expert relayout/upload\n");
            return false;
        }
        out->code_dtype = p.code.dtype; out->scale_dtype = p.scale.dtype;
        out->code_bytes = p.code.bytes; out->scale_bytes = p.scale.bytes;
        uint32_t& cd = gate_up ? gu_code_dtype_ : dn_code_dtype_;
        uint32_t& sd = gate_up ? gu_scale_dtype_ : dn_scale_dtype_;
        if ((cd && cd != out->code_dtype) || (sd && sd != out->scale_dtype)) {
            std::fprintf(stderr, "r3_moe_helper: owned expert stored layouts disagree\n");
            return false;
        }
        cd = out->code_dtype; sd = out->scale_dtype;
        return true;
    }

    bool build_one_table(const std::vector<void*>& host, void** dev) {
        *dev = alloc_persistent(host.size() * sizeof(void*));
        return *dev && hip_ok(hipMemcpy(*dev, host.data(), host.size() * sizeof(void*),
                                        hipMemcpyHostToDevice), "upload expert pointer table");
    }

    bool build_tables() {
        constexpr size_t nclass = r3::moe_shm::kExperts / 2;
        for (uint32_t l = 0; l < model_.layers; ++l) {
            LayerTables& lt = tables_[l];
            lt.gu_code_dtype = gu_code_dtype_; lt.gu_scale_dtype = gu_scale_dtype_;
            lt.dn_code_dtype = dn_code_dtype_; lt.dn_scale_dtype = dn_scale_dtype_;
            for (int parity = 0; parity < 2; ++parity) {
                std::vector<void*> guc(nclass, nullptr), gus(nclass, nullptr);
                std::vector<void*> dnc(nclass, nullptr), dns(nclass, nullptr);
                for (uint32_t e = (uint32_t)parity; e < r3::moe_shm::kExperts; e += 2) {
                    const Expert& x = experts_[(size_t)l * r3::moe_shm::kExperts + e];
                    if (!x.live) continue;
                    const size_t ix = e / 2;
                    guc[ix] = x.gu.code; gus[ix] = x.gu.scale;
                    dnc[ix] = x.dn.code; dns[ix] = x.dn.scale;
                }
                if (!build_one_table(guc, &lt.gu_code[parity]) ||
                    !build_one_table(gus, &lt.gu_scale[parity]) ||
                    !build_one_table(dnc, &lt.dn_code[parity]) ||
                    !build_one_table(dns, &lt.dn_scale[parity])) return false;
            }
        }
        return true;
    }

    bool owns(uint32_t layer, uint32_t expert) const {
        return layer < model_.layers && expert < r3::moe_shm::kExperts &&
               experts_[(size_t)layer * r3::moe_shm::kExperts + expert].live;
    }

    RadTensor table_tensor(void* table, uint32_t dtype, int64_t n, int64_t k) const {
        RadTensor t = tensor(table, dtype, {r3::moe_shm::kExperts / 2, n, k});
        t.stride[0] = 0; /* ABI table form: data points to a device array of per-entry pointers. */
        return t;
    }

    std::vector<RadParam> gemm_params(uint32_t rows, int64_t n, int64_t k,
                                      int64_t parts, const char* order) const {
        return {
            pi("M", rows), pi("N", n), pi("K", k), pi("N_odd", n), pi("K_odd", k),
            pi("parts", parts), pi("n_expert", r3::moe_shm::kExperts),
            pi("top_k", r3::moe_shm::kTopK), pi("group", kGroup),
            ps("a_order", order), ps("dtype", kDtype)
        };
    }

    int launch_on(const RadKernelInfo* r, std::vector<RadTensor> ts, std::vector<RadParam> ps,
                  hipStream_t stream, Buffer& scratch) {
        if (!r || !r->launch) return RAD_E_UNSUPPORTED;
        RadArgs a{};
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        a.rank = 0; a.world_size = 1;
        if (r->scratch) {
            const int64_t need = r->scratch(&a);
            if (need < 0 || !scratch.ensure((size_t)std::max<int64_t>(need, 1))) return RAD_E_STATE;
            a.scratch = scratch.p; a.scratch_bytes = (int64_t)scratch.bytes;
        }
        const int rc = r->launch(&a, (RadStream)stream);
        if (rc != RAD_OK)
            std::fprintf(stderr, "r3_moe_helper: libr4d %s launch rc=%d\n", r->name, rc);
        return rc;
    }

    int launch(const RadKernelInfo* r, std::vector<RadTensor> ts, std::vector<RadParam> ps) {
        return launch_on(r, std::move(ts), std::move(ps), stream_, scratch_);
    }

    bool ensure_lane_work(LaneWork& w, uint32_t rows, uint32_t routes, uint32_t tr) {
        return w.d_counts.ensure((size_t)r3::moe_shm::kExperts * 4) &&
               w.d_cursor.ensure((size_t)r3::moe_shm::kExperts * 4) &&
               w.d_token_slot.ensure((size_t)rows * 4) &&
               w.d_flat.ensure((size_t)routes * 4) &&
               w.d_sorted.ensure((size_t)tr * 4) &&
               w.d_sorted_expert.ensure((size_t)tr * 4) &&
               w.d_off.ensure((r3::moe_shm::kExperts + 1ull) * 4) &&
               w.d_ew.ensure((size_t)tr * 2) &&
               w.d_aq.ensure((size_t)rows * r3::moe_shm::kHidden) &&
               w.d_as.ensure((size_t)rows * (r3::moe_shm::kHidden / kGroup) * 4) &&
               w.d_gu.ensure((size_t)tr * kGateUpWidth * 2) &&
               w.d_mid.ensure((size_t)tr * kExpertWidth) &&
               w.d_mid_s.ensure((size_t)tr * (kExpertWidth / kGroup) * 4) &&
               w.d_mid_rank.ensure((size_t)tr * kExpertWidth) &&
               w.d_down.ensure((size_t)tr * r3::moe_shm::kHidden * 2);
    }

    bool ensure_work(uint32_t rows, uint32_t tr) {
        return d_act_.ensure((size_t)rows * r3::moe_shm::kHidden * 2) &&
               d_aq_.ensure((size_t)rows * r3::moe_shm::kHidden) &&
               d_as_.ensure((size_t)rows * (r3::moe_shm::kHidden / kGroup) * 4) &&
               d_sorted_.ensure((size_t)tr * 4) &&
               d_off_.ensure((r3::moe_shm::kExperts + 1ull) * 4) &&
               d_ew_.ensure((size_t)rows * r3::moe_shm::kTopK * 2) &&
               d_gu_.ensure((size_t)tr * kGateUpWidth * 2) &&
               d_mid_.ensure((size_t)tr * kExpertWidth) &&
               d_mid_s_.ensure((size_t)tr * (kExpertWidth / kGroup) * 4) &&
               d_mid_rank_.ensure((size_t)tr * kExpertWidth) &&
               d_down_.ensure((size_t)tr * r3::moe_shm::kHidden * 2) &&
               d_result_.ensure((size_t)rows * r3::moe_shm::kHidden * 2);
    }
};

} // namespace r3_moe_gpu
