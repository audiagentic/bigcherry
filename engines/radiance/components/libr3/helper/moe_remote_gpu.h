#pragma once

#include "rad_abi.h"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cerrno>
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
constexpr int64_t kRankWidth = 384;
constexpr int64_t kRankGateUpWidth = 2 * kRankWidth;
constexpr int64_t kGroup = 128;
constexpr const char* kDtype = "w4nl64a8h";

inline int64_t rank_slice_lo(uint32_t rank, uint32_t parity) {
    return rank == parity ? 0 : kRankWidth;
}
inline int64_t rank_slice_width(uint32_t rank, uint32_t parity) {
    return rank == parity ? kRankWidth : kExpertWidth - kRankWidth;
}

struct DeviceLayerTables {
    void* gu_code[r3::moe_shm::kMaxRanks][2]{};
    void* gu_scale[r3::moe_shm::kMaxRanks][2]{};
    void* dn_code[r3::moe_shm::kMaxRanks][2]{};
    void* dn_scale[r3::moe_shm::kMaxRanks][2]{};
    const uint16_t* expert_slot = nullptr;
};

__device__ __forceinline__ void lane_store_sys_rel(uint32_t* p, uint32_t v) {
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ uint32_t lane_load_sys_acq(const uint32_t* p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ bool lane_request_valid(
    const r3::moe_shm::Request* q, uint32_t rank, uint32_t sequence,
    uint32_t max_rows, uint32_t layers) {
    using namespace r3::moe_shm;
    /* A one-rank engine stamps rank_count 1 and uses the rank slot as the helper index (each slot's
     * request asks for full expert width); a two-rank engine's slot is its rank. */
    return q && q->sequence == sequence && q->rank == rank &&
           q->rank_count >= 1 && q->rank_count <= kMaxRanks && (q->rank_count == 1 || rank < q->rank_count) &&
           q->layer < layers && q->rows > 0 && q->rows <= max_rows &&
           q->hidden == kHidden && q->route_count > 0 &&
           q->route_count <= q->rows * kTopK &&
           q->activation_stride == kHidden * (uint32_t)sizeof(uint16_t) &&
           q->activation_q_stride == kHidden &&
           q->activation_s_stride == (kHidden / kActGroup) * (uint32_t)sizeof(float);
}

__global__ void lane_prepare_activation(
    r3::moe_shm::Request* q, const uint8_t* src_q, const float* src_s,
    uint8_t* dst_q, float* dst_s,
    uint32_t rank, uint32_t sequence, uint32_t max_rows, uint32_t layers,
    uint64_t* stats) {
    using namespace r3::moe_shm;
    const bool valid = lane_request_valid(q, rank, sequence, max_rows, layers);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        q->status = valid ? 0 : -EINVAL;
        if (valid) {
            atomicAdd(reinterpret_cast<unsigned long long*>(stats) + 0, 1ull);
            atomicAdd(reinterpret_cast<unsigned long long*>(stats) + 1,
                      (unsigned long long)q->rows);
            atomicAdd(reinterpret_cast<unsigned long long*>(stats) + 2,
                      (unsigned long long)q->route_count);
        }
    }
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t q_total = (uint64_t)max_rows * kHidden;
    if (i < q_total) {
        const uint32_t row = (uint32_t)(i / kHidden);
        dst_q[i] = valid && row < q->rows ? src_q[i] : 0;
    }
    const uint32_t scale_cols = kHidden / kActGroup;
    const uint64_t s_total = (uint64_t)max_rows * scale_cols;
    if (i < s_total) {
        const uint32_t row = (uint32_t)(i / scale_cols);
        dst_s[i] = valid && row < q->rows ? src_s[i] : 0.0f;
    }
}

__global__ void lane_select_tables(
    const r3::moe_shm::Request* q, const DeviceLayerTables* layers_table,
    uint32_t layers, uint32_t rank, uint32_t slice, uint32_t sequence, uint32_t max_rows,
    void** guc0, void** gus0, void** dnc0, void** dns0,
    void** guc1, void** gus1, void** dnc1, void** dns1) {
    using namespace r3::moe_shm;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kExperts / 2) return;
    const bool valid = lane_request_valid(q, rank, sequence, max_rows, layers);
    if (!valid) {
        guc0[i] = gus0[i] = dnc0[i] = dns0[i] = nullptr;
        guc1[i] = gus1[i] = dnc1[i] = dns1[i] = nullptr;
        return;
    }
    const DeviceLayerTables t = layers_table[q->layer];
    guc0[i] = static_cast<void**>(t.gu_code[slice][0])[i];
    gus0[i] = static_cast<void**>(t.gu_scale[slice][0])[i];
    dnc0[i] = static_cast<void**>(t.dn_code[slice][0])[i];
    dns0[i] = static_cast<void**>(t.dn_scale[slice][0])[i];
    guc1[i] = static_cast<void**>(t.gu_code[slice][1])[i];
    gus1[i] = static_cast<void**>(t.gu_scale[slice][1])[i];
    dnc1[i] = static_cast<void**>(t.dn_code[slice][1])[i];
    dns1[i] = static_cast<void**>(t.dn_scale[slice][1])[i];
}

__global__ void lane_route_count(
    const r3::moe_shm::Route* route, r3::moe_shm::Request* q,
    const DeviceLayerTables* layers_table,
    uint32_t rank, uint32_t sequence, uint32_t max_rows, uint32_t layers,
    uint32_t* counts, int32_t* flat, uint16_t* ew) {
    using namespace r3::moe_shm;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!lane_request_valid(q, rank, sequence, max_rows, layers) || i >= q->route_count) return;
    const Route r = route[i];
    if (r.token < 0 || (uint32_t)r.token >= q->rows || r.expert < 0 ||
        r.expert >= (int32_t)kExperts || r.slot >= kTopK) {
        flat[i] = -1;
        return;
    }
    const uint16_t es = layers_table[q->layer].expert_slot[(uint32_t)r.expert];
    if (es >= kExperts) {
        flat[i] = -1;
        atomicExch(&q->status, -EINVAL);
        return;
    }
    const uint32_t f = (uint32_t)r.token * kTopK + r.slot;
    flat[i] = (int32_t)f;
    ew[f] = r.weight_bf16;
    atomicAdd(counts + (uint32_t)es, 1u);
}

/* Exclusive prefix over the 512 expert counts in one block of kExperts threads (was one thread walking all 512:
 * ~43 us a request). */
__global__ void lane_route_prefix(const uint32_t* counts, int32_t* off, uint32_t* cursor) {
    constexpr uint32_t n = r3::moe_shm::kExperts;
    __shared__ uint32_t s[n];
    const uint32_t e = threadIdx.x;
    if (blockIdx.x || e >= n) return;
    const uint32_t mine = counts[e];
    s[e] = mine;
    __syncthreads();
    for (uint32_t d = 1; d < n; d <<= 1) {
        const uint32_t add = e >= d ? s[e - d] : 0u;
        __syncthreads();
        s[e] += add;
        __syncthreads();
    }
    const uint32_t excl = s[e] - mine;
    off[e] = (int32_t)excl;
    cursor[e] = excl;
    if (e == n - 1) off[n] = (int32_t)s[e];
}

__global__ void lane_route_scatter(
    const r3::moe_shm::Route* route, const r3::moe_shm::Request* q,
    const DeviceLayerTables* layers_table,
    uint32_t rank, uint32_t sequence, uint32_t max_rows, uint32_t layers,
    const int32_t* flat, uint32_t* cursor, int32_t* sorted, int32_t* sorted_expert) {
    using namespace r3::moe_shm;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!lane_request_valid(q, rank, sequence, max_rows, layers) ||
        i >= q->route_count || flat[i] < 0) return;
    const uint32_t e = (uint32_t)route[i].expert;
    const uint32_t es = layers_table[q->layer].expert_slot[e];
    if (es >= kExperts) return;
    const uint32_t pos = atomicAdd(cursor + es, 1u);
    sorted[pos] = flat[i];
    if (sorted_expert) sorted_expert[pos] = (int32_t)e;
}

__global__ void lane_route_inverse(const int32_t* sorted, int32_t* inv, uint32_t routes, uint32_t slots) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= routes) return;
    const int32_t flat = sorted[i];
    if (flat >= 0 && (uint32_t)flat < slots) inv[flat] = (int32_t)i;
}

/* One row a token with any helper route: the router-weighted sum of its helper-computed experts, slot
 * order, f32 accumulator, narrowed once to bf16 (moe_gather arithmetic over the slots this helper owns).
 * Tokens without a route are not written; the engine's merge reads only tokens it routed here. */
__global__ void lane_reduce_tokens(const uint16_t* down, const uint16_t* down2, const int32_t* inv,
                                   const uint16_t* ew, uint16_t* dst, uint32_t rows, uint32_t hidden,
                                   uint32_t poison) {
    using namespace r3::moe_shm;
    const uint32_t token = blockIdx.x;
    if (token >= rows) return;
    __shared__ int32_t row[kTopK];
    __shared__ float w[kTopK];
    __shared__ uint32_t any;
    if (threadIdx.x == 0) {
        any = 0;
        for (uint32_t slot = 0; slot < kTopK; ++slot) {
            const uint32_t flat = token * kTopK + slot;
            row[slot] = inv[flat];
            w[slot] = __uint_as_float((uint32_t)ew[flat] << 16);
            any |= row[slot] >= 0 ? 1u : 0u;
        }
    }
    __syncthreads();
    if (!any) return;
    uint16_t* out = dst + (uint64_t)token * hidden;
    for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t slot = 0; slot < kTopK; ++slot) {
            if (row[slot] < 0) continue;
            const uint64_t at = (uint64_t)(uint32_t)row[slot] * hidden + col;
            /* One-rank engine: the expert's two TP slices are partial sums of the same down product. */
            float v = __uint_as_float((uint32_t)down[at] << 16);
            if (down2) v += __uint_as_float((uint32_t)down2[at] << 16);
            acc += w[slot] * v;
        }
        uint32_t u = __float_as_uint(acc);
        u += 0x7fffu + ((u >> 16) & 1u);
        out[col] = poison == 1u ? (uint16_t)0x447au : poison == 2u ? (uint16_t)0 : (uint16_t)(u >> 16);
    }
    /* Mapped host memory is fine-grained/uncached. Drain every writer wave's posted stores
     * before the later release flag, matching libr4d's no-P2P all-reduce host path. */
#if defined(__gfx1100__)
    asm volatile("s_waitcnt_vscnt null, 0x0" ::: "memory");   /* gfx11 spelling of the store drain */
#else
    asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
}

/* Every mapped result writer has drained before this kernel can run. Publish completion with a
 * system-scope release; the engine-side join waits with a system-scope acquire. */
__global__ void lane_publish_done(uint32_t* done_sequence, uint32_t sequence) {
    if (blockIdx.x || threadIdx.x) return;
    lane_store_sys_rel(done_sequence, sequence);
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
    std::array<StoredWeight, r3::moe_shm::kMaxRanks> gu;
    std::array<StoredWeight, r3::moe_shm::kMaxRanks> dn;
    bool live = false;
};

struct LayerTables {
    void* gu_code[r3::moe_shm::kMaxRanks][2]{};
    void* gu_scale[r3::moe_shm::kMaxRanks][2]{};
    void* dn_code[r3::moe_shm::kMaxRanks][2]{};
    void* dn_scale[r3::moe_shm::kMaxRanks][2]{};
    uint32_t gu_code_dtype = 0, gu_scale_dtype = 0;
    uint32_t dn_code_dtype = 0, dn_scale_dtype = 0;
};

struct LaneWork {
    hipStream_t stream = nullptr;
    hipEvent_t sample_start = nullptr;
    hipEvent_t sample_stop = nullptr;
    uint32_t sample_sequence = 0;
    bool sample_pending = false;
    uint32_t cap_rows = 0;
    Buffer d_counts, d_cursor, d_flat, d_sorted, d_off, d_ew, d_inv, d_down2;
    Buffer d_act, d_aq, d_as, d_gu, d_mid, d_mid_s, d_down, d_stats, scratch;
    Buffer gu_code[2], gu_scale[2], dn_code[2], dn_scale[2];
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
        for (auto& l : lane_) {
            if (l.sample_start) (void)hipEventDestroy(l.sample_start);
            if (l.sample_stop) (void)hipEventDestroy(l.sample_stop);
            if (l.stream) (void)hipStreamDestroy(l.stream);
        }
        for (void* p : persistent_) if (p) (void)hipFree(p);
        if (plugin_opened_ && plugin_close_) plugin_close_();
        if (so_) dlclose(so_);
    }

    bool open() {
        const char* poison = std::getenv("R3_MOE_HELPER_POISON");
        /* "zero" returns zeros (the same output as dropping the remote routes); anything else non-"0" is 1e3. */
        poison_ = !poison || !*poison || std::strcmp(poison, "0") == 0 ? 0u
                : std::strcmp(poison, "zero") == 0 ? 2u : 1u;
        if (!hip_ok(hipSetDevice(0), "hipSetDevice(0)")) return false;
        hipDeviceProp_t prop{};
        if (!hip_ok(hipGetDeviceProperties(&prop, 0), "hipGetDeviceProperties")) return false;
        arch_ = prop.gcnArchName;
        const size_t colon = arch_.find(':');
        if (colon != std::string::npos) arch_.resize(colon);
        /* The expert kernels come from the plugin built for this card: libr4d on gfx1201 (the R9700 helping two
         * XTX ranks), libr3 on gfx1100 (an XTX helping an R9700 engine, RAD02). Both export the same rows. */
        if (arch_ != "gfx1201" && arch_ != "gfx1100") {
            std::fprintf(stderr,
                "r3_moe_helper: visible GPU is %s, want gfx1201 or gfx1100 (set R3_MOE_HELPER_CPU=1 for host path)\n",
                arch_.c_str());
            return false;
        }
        if (!load_plugin()) return false;
        if (!hip_ok(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "hipStreamCreate")) return false;
        for (uint32_t r = 0; r < r3::moe_shm::kLanes; ++r) {
            if (!hip_ok(hipStreamCreateWithFlags(&lane_[r].stream, hipStreamNonBlocking), "hipStreamCreate lane") ||
                !hip_ok(hipEventCreate(&lane_[r].sample_start), "hipEventCreate lane start") ||
                !hip_ok(hipEventCreate(&lane_[r].sample_stop), "hipEventCreate lane stop"))
                return false;
        }
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

    bool allocate(const std::vector<std::pair<uint32_t, uint32_t>>& owned) {
        constexpr size_t kLoadBounceBytes = 64ull * 1024ull * 1024ull;
        if (!load_bounce_.ensure(kLoadBounceBytes)) return false;
        size_t total = 0;
        for (auto [l, e] : owned) {
            for (uint32_t rank = 0; rank < r3::moe_shm::kMaxRanks; ++rank) {
                int64_t cb = 0, sb = 0;
                if (!slice_bytes(model_.weight(l, e, 0), true, 0, rank, &cb, &sb)) return false;
                total += arena_round(cb) + arena_round(sb);
                if (!slice_bytes(model_.weight(l, e, 1), false, 0, rank, &cb, &sb)) return false;
                total += arena_round(cb) + arena_round(sb);
            }
        }
        if (!hip_ok(hipMalloc(&arena_, total ? total : 16), "hipMalloc resident expert arena")) return false;
        persistent_.push_back(arena_);
        arena_size_ = total; arena_used_ = 0;
        layer_loaded_.assign(model_.layers, 0);
        const size_t slot_bytes = (size_t)model_.layers * r3::moe_shm::kExperts * sizeof(uint16_t);
        const size_t table_bytes = (size_t)model_.layers * sizeof(DeviceLayerTables);
        return d_expert_slot_.ensure(slot_bytes) &&
               d_layer_tables_.ensure(table_bytes) &&
               hip_ok(hipMemset(d_expert_slot_.p, 0xff, slot_bytes), "initialize expert slot map") &&
               hip_ok(hipMemset(d_layer_tables_.p, 0, table_bytes), "initialize layer table index");
    }

    bool load_layer(uint32_t layer,
                    const std::vector<std::pair<uint32_t, uint32_t>>& owned,
                    const int32_t* slot_of_expert) {
        using namespace r3::moe_shm;
        if (layer >= model_.layers || !slot_of_expert || layer_loaded_.empty()) return false;
        if (layer_loaded_[layer]) return true;

        std::array<uint8_t, kExperts> seen{};
        uint32_t nreg = 0;
        for (uint32_t e = 0; e < kExperts; ++e) {
            const int32_t es = slot_of_expert[e];
            if (es < 0) continue;
            if (es >= (int32_t)kExperts || seen[(uint32_t)es]) return false;
            seen[(uint32_t)es] = 1;
            nreg = std::max(nreg, (uint32_t)es + 1u);
        }
        for (uint32_t es = 0; es < nreg; ++es) if (!seen[es]) return false;

        for (auto [l, e] : owned) {
            if (l != layer || slot_of_expert[e] < 0) continue;
            const uint32_t es = (uint32_t)slot_of_expert[e];
            Expert& x = experts_[(size_t)l * kExperts + e];
            if (!upload_weight(model_.weight(l, e, 0), true, es, &x.gu) ||
                !upload_weight(model_.weight(l, e, 1), false, es, &x.dn))
                return false;
            x.live = true;
        }
        if (!build_layer_table(layer, slot_of_expert)) return false;
        layer_loaded_[layer] = 1;
        return true;
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

        std::vector<std::vector<uint32_t>> by_slot(kExperts);
        std::vector<uint16_t> ew((size_t)rows * kTopK, 0);
        for (uint32_t i = 0; i < q->route_count; ++i) {
            const Route& r = route[i];
            if (r.token < 0 || (uint32_t)r.token >= rows || r.expert < 0 ||
                r.expert >= (int32_t)kExperts || !owns(q->layer, (uint32_t)r.expert)) return -EINVAL;
            if (r.slot >= kTopK) return -EINVAL;
            const int32_t es = layer_slot(map, q->layer, (uint32_t)r.expert);
            if (es < 0 || es >= (int32_t)kExperts) return -EINVAL;
            const uint32_t flat = (uint32_t)r.token * kTopK + r.slot;
            ew[flat] = r.weight_bf16;
            by_slot[(uint32_t)es].push_back(flat);
        }

        std::vector<int32_t> sorted(tr, -1);
        std::array<int32_t, kExperts + 1> off{};
        uint32_t at = 0;
        for (uint32_t es = 0; es < kExperts; ++es) {
            off[es] = (int32_t)at;
            for (uint32_t flat : by_slot[es]) sorted[at++] = (int32_t)flat;
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

        std::vector<RadParam> aqp = {
            pi("M", rows), pi("n", kHidden), pi("group", kGroup), ps("dtype", "bf16")
        };
        if (launch(had_quant_,
                   { tensor(d_act_.p, RAD_BF16, {rows, kHidden}),
                     tensor(d_aq_.p, RAD_F8E4M3, {rows, kHidden}),
                     tensor(d_as_.p, RAD_F32, {rows, kHidden / kGroup}) }, aqp) < 0)
            return -EIO;

        for (uint32_t rank = 0; rank < kMaxRanks; ++rank) {
            LayerTables& lt = tables_[q->layer];
            const int64_t w0 = rank_slice_width(rank, 0);
            const int64_t w1 = rank_slice_width(rank, 1);

            std::vector<RadParam> gp =
                gemm_params(rows, 2 * w0, kHidden, 2, "token", 2 * w1, kHidden);
            if (launch(moe_,
                       { tensor(d_aq_.p, RAD_F8E4M3, {rows, kHidden}),
                         tensor(d_as_.p, RAD_F32, {rows, kHidden / kGroup}),
                         table_tensor(lt.gu_code[rank][0], lt.gu_code_dtype, 2 * w0, kHidden),
                         table_tensor(lt.gu_scale[rank][0], lt.gu_scale_dtype, 2 * w0, kHidden / 64),
                         tensor(d_sorted_.p, RAD_I32, {tr}),
                         tensor(d_off_.p, RAD_I32, {kExperts + 1}),
                         table_tensor(lt.gu_code[rank][1], lt.gu_code_dtype, 2 * w1, kHidden),
                         table_tensor(lt.gu_scale[rank][1], lt.gu_scale_dtype, 2 * w1, kHidden / 64),
                         tensor(d_gu_.p, RAD_BF16, {tr, kRankGateUpWidth}) }, gp) < 0)
                return -EIO;

            std::vector<RadParam> gqp = {
                pi("M", q->route_count), pi("n", kRankWidth), pi("group", kGroup),
                ps("dtype", "bf16"), ps("act", "silu")
            };
            if (launch(gated_quant_,
                       { tensor(d_gu_.p, RAD_BF16, {q->route_count, kRankGateUpWidth}),
                         tensor(d_mid_.p, RAD_F8E4M3, {q->route_count, kRankWidth}),
                         tensor(d_mid_s_.p, RAD_F32, {q->route_count, kRankWidth / kGroup}),
                         absent() }, gqp) < 0)
                return -EIO;

            std::vector<RadParam> dp =
                gemm_params(rows, kHidden, w0, 1, "sorted", kHidden, w1);
            if (launch(moe_,
                       { tensor(d_mid_.p, RAD_F8E4M3, {tr, kRankWidth}),
                         tensor(d_mid_s_.p, RAD_F32, {tr, kRankWidth / kGroup}),
                         table_tensor(lt.dn_code[rank][0], lt.dn_code_dtype, kHidden, w0),
                         table_tensor(lt.dn_scale[rank][0], lt.dn_scale_dtype, kHidden, w0 / 64),
                         tensor(d_sorted_.p, RAD_I32, {tr}),
                         tensor(d_off_.p, RAD_I32, {kExperts + 1}),
                         table_tensor(lt.dn_code[rank][1], lt.dn_code_dtype, kHidden, w1),
                         table_tensor(lt.dn_scale[rank][1], lt.dn_scale_dtype, kHidden, w1 / 64),
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

    /* Small lanes are sized to small_rows and pre-queued; large lanes to large_rows, launched per request. */
    bool prepare_lanes(uint32_t small_rows, uint32_t large_rows) {
        using namespace r3::moe_shm;
        if (small_rows == 0 || small_rows > kMaxRows || large_rows < small_rows || large_rows > kMaxRows ||
            !d_layer_tables_.p)
            return false;
        lane_rows_ = small_rows;
        large_rows_ = large_rows;
        constexpr size_t nclass = kExperts / 2;
        for (uint32_t lane = 0; lane < kLanes; ++lane) {
            const uint32_t rank = lane % kMaxRanks;
            const uint32_t cap = lane < kMaxRanks ? lane_rows_ : large_rows_;
            LaneWork& w = lane_[lane];
            w.cap_rows = cap;
            if (!ensure_lane_work(w, cap, cap * kTopK, cap * kTopK) ||
                !w.d_stats.ensure(3 * sizeof(uint64_t)))
                return false;
            for (uint32_t parity = 0; parity < 2; ++parity) {
                if (!w.gu_code[parity].ensure(nclass * sizeof(void*)) ||
                    !w.gu_scale[parity].ensure(nclass * sizeof(void*)) ||
                    !w.dn_code[parity].ensure(nclass * sizeof(void*)) ||
                    !w.dn_scale[parity].ensure(nclass * sizeof(void*)))
                    return false;
            }
            if (!prepare_lane_scratch(w, rank, cap, cap * kTopK) ||
                !hip_ok(hipMemset(w.d_stats.p, 0, 3 * sizeof(uint64_t)), "zero lane stats"))
                return false;
        }
        return true;
    }

    bool stage_compare_activation(void* host_map, uint32_t lane,
                                  const uint16_t* activation, uint32_t rows) {
        using namespace r3::moe_shm;
        if (!host_map || !activation || lane >= kLanes || rows == 0 || rows > lane_[lane].cap_rows)
            return false;
        const uint32_t rank = lane % kMaxRanks;
        LaneWork& w = lane_[lane];
        const size_t xbytes = (size_t)rows * kHidden * sizeof(uint16_t);
        if (!hip_ok(hipMemcpyAsync(w.d_act.p, activation, xbytes,
                                   hipMemcpyHostToDevice, w.stream),
                    "--compare upload BF16 activation"))
            return false;
        std::vector<RadParam> aqp = {
            pi("M", rows), pi("n", kHidden), pi("group", kGroup), ps("dtype", "bf16")
        };
        if (launch_on(had_quant_,
                      { tensor(w.d_act.p, RAD_BF16, {rows, kHidden}),
                        tensor(w.d_aq.p, RAD_F8E4M3, {rows, kHidden}),
                        tensor(w.d_as.p, RAD_F32, {rows, kHidden / kGroup}) },
                      aqp, w.stream, w.scratch) < 0)
            return false;
        Header* h = static_cast<Header*>(host_map);
        auto* hq = ptr<uint8_t>(host_map, h->activation_offset[rank] + kActivationBf16Bytes);
        auto* hs = ptr<float>(host_map,
            h->activation_offset[rank] + kActivationBf16Bytes + kActivationQBytes);
        if (!hip_ok(hipMemcpyAsync(hq, w.d_aq.p, (size_t)rows * kHidden,
                                   hipMemcpyDeviceToHost, w.stream),
                    "--compare stage exact q") ||
            !hip_ok(hipMemcpyAsync(hs, w.d_as.p,
                                   (size_t)rows * (kHidden / kGroup) * sizeof(float),
                                   hipMemcpyDeviceToHost, w.stream),
                    "--compare stage exact scales"))
            return false;
        return hip_ok(hipStreamSynchronize(w.stream), "--compare stage q/s sync");
    }

    /* rows: the fixed small-lane size (pre-queued), or the request's own row count on a large lane. */
    int enqueue_lane(void* host_map, void* device_map, uint32_t lane, uint32_t sequence,
                     uint32_t rows, bool sample) {
        using namespace r3::moe_shm;
        if (!host_map || !device_map || lane >= kLanes || sequence == 0 || rows == 0 ||
            rows > lane_[lane].cap_rows)
            return -EINVAL;
        const uint32_t rank = lane % kMaxRanks;
        LaneWork& w = lane_[lane];
        auto* hhost = static_cast<Header*>(host_map);
        auto* dbase = static_cast<uint8_t*>(device_map);
        auto dev_at = [&](const void* hp) -> void* {
            return dbase + (static_cast<const uint8_t*>(hp) -
                            static_cast<const uint8_t*>(host_map));
        };

        Request* qhost = lane_request(host_map, lane, sequence);
        if (!qhost) return -EINVAL;
        auto* q = static_cast<Request*>(dev_at(qhost));
        auto* request_seq =
            static_cast<uint32_t*>(dev_at(&hhost->lane[lane].request_sequence));
        auto* done_seq =
            static_cast<uint32_t*>(dev_at(&hhost->lane[lane].done_sequence));
        auto* route = static_cast<const Route*>(
            dev_at(ptr<Route>(host_map, hhost->route_offset[rank])));
        auto* act_q = static_cast<const uint8_t*>(
            dev_at(ptr<uint8_t>(host_map,
                hhost->activation_offset[rank] + kActivationBf16Bytes)));
        auto* act_s = static_cast<const float*>(
            dev_at(ptr<float>(host_map,
                hhost->activation_offset[rank] + kActivationBf16Bytes + kActivationQBytes)));
        auto* dst = static_cast<uint16_t*>(
            dev_at(ptr<uint16_t>(
                host_map, hhost->result_offset[rank * kMaxRanks + rank])));

        if (!hip_ok(hipStreamWaitValue32(w.stream, request_seq, sequence,
                                         hipStreamWaitValueGte, 0xffffffffu),
                    "lane wait request"))
            return -EIO;
        if (sample && !w.sample_pending) {
            if (!hip_ok(hipEventRecord(w.sample_start, w.stream), "lane timing start"))
                return -EIO;
            w.sample_sequence = sequence;
            w.sample_pending = true;
        }

        const uint32_t routes = rows * kTopK;
        const uint32_t slots = routes;
        /* No clear of the result slot: it is mapped host memory, and the merge reads only the tokens this
         * request's routes name, each of which the reduction writes. */
        if (!hip_ok(hipMemsetAsync(w.d_counts.p, 0, kExperts * sizeof(uint32_t), w.stream),
                    "clear route counts") ||
            !hip_ok(hipMemsetAsync(w.d_ew.p, 0, routes * sizeof(uint16_t), w.stream),
                    "clear route weights") ||
            !hip_ok(hipMemsetAsync(w.d_sorted.p, 0xff, routes * sizeof(int32_t), w.stream),
                    "clear sorted routes") ||
            !hip_ok(hipMemsetAsync(w.d_inv.p, 0xff, slots * sizeof(int32_t), w.stream),
                    "clear slot inverse"))
            return -EIO;

        const dim3 block(256);
        const dim3 act_grid(
            ((uint64_t)rows * kHidden + block.x - 1) / block.x);
        const dim3 route_grid((routes + block.x - 1) / block.x);
        hipLaunchKernelGGL(lane_prepare_activation, act_grid, block, 0, w.stream,
                           q, act_q, act_s,
                           static_cast<uint8_t*>(w.d_aq.p), static_cast<float*>(w.d_as.p),
                           rank, sequence, rows, model_.layers,
                           static_cast<uint64_t*>(w.d_stats.p));
        hipLaunchKernelGGL(lane_route_count, route_grid, block, 0, w.stream,
                           route, q, static_cast<const DeviceLayerTables*>(d_layer_tables_.p),
                           rank, sequence, rows, model_.layers,
                           static_cast<uint32_t*>(w.d_counts.p),
                           static_cast<int32_t*>(w.d_flat.p),
                           static_cast<uint16_t*>(w.d_ew.p));
        hipLaunchKernelGGL(lane_route_prefix, dim3(1), dim3(kExperts), 0, w.stream,
                           static_cast<const uint32_t*>(w.d_counts.p),
                           static_cast<int32_t*>(w.d_off.p),
                           static_cast<uint32_t*>(w.d_cursor.p));
        hipLaunchKernelGGL(lane_route_scatter, route_grid, block, 0, w.stream,
                           route, q, static_cast<const DeviceLayerTables*>(d_layer_tables_.p),
                           rank, sequence, rows, model_.layers,
                           static_cast<const int32_t*>(w.d_flat.p),
                           static_cast<uint32_t*>(w.d_cursor.p),
                           static_cast<int32_t*>(w.d_sorted.p),
                           nullptr);
        if (!hip_ok(hipGetLastError(), "lane request kernels")) return -EIO;

        /* A two-rank engine asks for its own TP slice; a one-rank engine for both, each into its own rows. */
        const uint32_t nslice = engine_ranks_ == 1 ? 2u : 1u;
        for (uint32_t k = 0; k < nslice; ++k) {
            const uint32_t slice = nslice == 2 ? k : rank;
            void* out_down = k == 0 ? w.d_down.p : w.d_down2.p;
            hipLaunchKernelGGL(lane_select_tables,
                               dim3((kExperts / 2 + block.x - 1) / block.x), block, 0, w.stream,
                               q, static_cast<const DeviceLayerTables*>(d_layer_tables_.p),
                               model_.layers, rank, slice, sequence, rows,
                               static_cast<void**>(w.gu_code[0].p),
                               static_cast<void**>(w.gu_scale[0].p),
                               static_cast<void**>(w.dn_code[0].p),
                               static_cast<void**>(w.dn_scale[0].p),
                               static_cast<void**>(w.gu_code[1].p),
                               static_cast<void**>(w.gu_scale[1].p),
                               static_cast<void**>(w.dn_code[1].p),
                               static_cast<void**>(w.dn_scale[1].p));
            const int64_t w0 = rank_slice_width(slice, 0);
            const int64_t w1 = rank_slice_width(slice, 1);
            std::vector<RadParam> gp =
                gemm_params(rows, 2 * w0, kHidden, 2, "token", 2 * w1, kHidden);
            if (launch_on(moe_,
                          { tensor(w.d_aq.p, RAD_F8E4M3, {rows, kHidden}),
                            tensor(w.d_as.p, RAD_F32, {rows, kHidden / kGroup}),
                            table_tensor(w.gu_code[0].p, gu_code_dtype_, 2 * w0, kHidden),
                            table_tensor(w.gu_scale[0].p, gu_scale_dtype_, 2 * w0, kHidden / 64),
                            tensor(w.d_sorted.p, RAD_I32, {routes}),
                            tensor(w.d_off.p, RAD_I32, {kExperts + 1}),
                            table_tensor(w.gu_code[1].p, gu_code_dtype_, 2 * w1, kHidden),
                            table_tensor(w.gu_scale[1].p, gu_scale_dtype_, 2 * w1, kHidden / 64),
                            tensor(w.d_gu.p, RAD_BF16, {routes, kRankGateUpWidth}) },
                          gp, w.stream, w.scratch) < 0)
                return -EIO;

            std::vector<RadParam> gqp = {
                pi("M", routes), pi("n", kRankWidth), pi("group", kGroup),
                ps("dtype", "bf16"), ps("act", "silu")
            };
            if (launch_on(gated_quant_,
                          { tensor(w.d_gu.p, RAD_BF16, {routes, kRankGateUpWidth}),
                            tensor(w.d_mid.p, RAD_F8E4M3, {routes, kRankWidth}),
                            tensor(w.d_mid_s.p, RAD_F32, {routes, kRankWidth / kGroup}),
                            absent() }, gqp, w.stream, w.scratch) < 0)
                return -EIO;

            std::vector<RadParam> dp =
                gemm_params(rows, kHidden, w0, 1, "sorted", kHidden, w1);
            if (launch_on(moe_,
                          { tensor(w.d_mid.p, RAD_F8E4M3, {routes, kRankWidth}),
                            tensor(w.d_mid_s.p, RAD_F32, {routes, kRankWidth / kGroup}),
                            table_tensor(w.dn_code[0].p, dn_code_dtype_, kHidden, w0),
                            table_tensor(w.dn_scale[0].p, dn_scale_dtype_, kHidden, w0 / 64),
                            tensor(w.d_sorted.p, RAD_I32, {routes}),
                            tensor(w.d_off.p, RAD_I32, {kExperts + 1}),
                            table_tensor(w.dn_code[1].p, dn_code_dtype_, kHidden, w1),
                            table_tensor(w.dn_scale[1].p, dn_scale_dtype_, kHidden, w1 / 64),
                            tensor(out_down, RAD_BF16, {routes, kHidden}) },
                          dp, w.stream, w.scratch) < 0)
                return -EIO;
        }

        hipLaunchKernelGGL(lane_route_inverse, route_grid, block, 0, w.stream,
                           static_cast<const int32_t*>(w.d_sorted.p),
                           static_cast<int32_t*>(w.d_inv.p), routes, slots);
        hipLaunchKernelGGL(lane_reduce_tokens, dim3(rows), block, 0, w.stream,
                           static_cast<const uint16_t*>(w.d_down.p),
                           nslice == 2 ? static_cast<const uint16_t*>(w.d_down2.p) : nullptr,
                           static_cast<const int32_t*>(w.d_inv.p),
                           static_cast<const uint16_t*>(w.d_ew.p),
                           dst, rows, kHidden, poison_);
        if (!hip_ok(hipGetLastError(), "lane reduce tokens")) return -EIO;

        if (sample && w.sample_pending && w.sample_sequence == sequence &&
            !hip_ok(hipEventRecord(w.sample_stop, w.stream), "lane timing stop"))
            return -EIO;
        hipLaunchKernelGGL(lane_publish_done, dim3(1), dim3(1), 0, w.stream,
                           done_seq, sequence);
        if (!hip_ok(hipGetLastError(), "lane publish done")) return -EIO;
        return 0;
    }

    bool collect_lane_timing(uint32_t lane, uint32_t done_sequence,
                             double* sampled_us, uint64_t* samples) {
        if (lane >= r3::moe_shm::kLanes || !sampled_us || !samples) return false;
        LaneWork& w = lane_[lane];
        if (!w.sample_pending ||
            (int32_t)(done_sequence - w.sample_sequence) < 0)
            return true;
        float ms = 0.0f;
        if (!hip_ok(hipEventElapsedTime(&ms, w.sample_start, w.sample_stop),
                    "lane timing elapsed"))
            return false;
        *sampled_us += (double)ms * 1000.0;
        ++*samples;
        w.sample_pending = false;
        return true;
    }

    bool sync_lane(uint32_t lane) {
        return lane < r3::moe_shm::kLanes &&
               hip_ok(hipStreamSynchronize(lane_[lane].stream), "lane stream sync");
    }

    bool lane_stats(uint32_t lane, uint64_t out[3]) {
        if (lane >= r3::moe_shm::kLanes || !out) return false;
        return hip_ok(hipMemcpy(out, lane_[lane].d_stats.p, 3 * sizeof(uint64_t),
                                hipMemcpyDeviceToHost), "read lane stats");
    }

private:
    const Container& model_;
    std::string arch_;
    void* so_ = nullptr;
    void (*plugin_close_)() = nullptr;
    bool plugin_opened_ = false;
    hipStream_t stream_ = nullptr;
    std::array<LaneWork, r3::moe_shm::kLanes> lane_{};
    uint32_t gu_code_dtype_ = 0, gu_scale_dtype_ = 0;
    uint32_t dn_code_dtype_ = 0, dn_scale_dtype_ = 0;
    const RadKernelInfo* moe_ = nullptr;
    const RadKernelInfo* had_quant_ = nullptr;
    const RadKernelInfo* gated_quant_ = nullptr;
    const RadKernelInfo* gather_ = nullptr;
    std::vector<Expert> experts_;
    std::vector<LayerTables> tables_;
    std::vector<uint8_t> layer_loaded_;
    std::vector<void*> persistent_;
    void* arena_ = nullptr;
    size_t arena_size_ = 0, arena_used_ = 0;

    HostBounce load_bounce_;
    Buffer d_act_, d_aq_, d_as_, d_sorted_, d_off_, d_ew_, d_gu_;
    Buffer d_mid_, d_mid_s_, d_down_, d_result_, scratch_;
    Buffer d_layer_tables_, d_expert_slot_;
    uint32_t lane_rows_ = 0;    /* small-lane (pre-queued) step size */
public:
    /* Engine tensor-parallel ranks (R3_MOE_HELPER_ENGINE_RANKS): 2 = two XTX ranks, each asking for its TP
     * slice; 1 = one engine (the R9700, RAD02), asking for the full expert width -- both slices, summed. */
    uint32_t engine_ranks_ = r3::moe_shm::kMaxRanks;
private:
    uint32_t large_rows_ = 0;   /* large-lane capacity */
    uint32_t poison_ = 0;

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
        if (const char* e = std::getenv("R3_MOE_HELPER_PLUGIN"); e && *e) return e;
        const std::string so = arch_ == "gfx1100" ? "libr3.so" : "libr4d.so";
        if (const char* e = std::getenv("RADIANCE_HOME"); e && *e) {
            std::string h(e);
            const size_t colon = h.find(':');
            if (colon != std::string::npos) h.resize(colon);
            return h + "/kernels/" + so;
        }
#ifdef RAD_DEFAULT_HOME
        return std::string(RAD_DEFAULT_HOME) + "/kernels/" + so;
#else
        return so;
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
        if (!inf || !inf->build_target || !std::strstr(inf->build_target, arch_.c_str())) {
            std::fprintf(stderr, "r3_moe_helper: %s is not a %s plugin\n", path.c_str(), arch_.c_str());
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

    bool slice_bytes(const RadFileEntry* e, bool gate_up, uint32_t expert_slot, uint32_t rank,
                     int64_t* code, int64_t* scale) const {
        if (!e || !code || !scale || rank >= r3::moe_shm::kMaxRanks) return false;
        int64_t nr = 1, nc = 1;
        rad_enc_view(e->rank, e->shape, &nr, &nc);
        const int64_t width = rank_slice_width(rank, expert_slot & 1u);
        if (gate_up) {
            if (nr != kGateUpWidth || nc != r3::moe_shm::kHidden) return false;
            *code = (2 * width) * nc / 2;
            *scale = (2 * width) * (nc / 64);
        } else {
            if (nr != r3::moe_shm::kHidden || nc != kExpertWidth) return false;
            *code = nr * width / 2;
            *scale = nr * (width / 64);
        }
        return true;
    }

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

    bool upload_weight(const RadFileEntry* e, bool gate_up, uint32_t expert_slot,
                       std::array<StoredWeight, r3::moe_shm::kMaxRanks>* out) {
        WeightPlan p;
        if (!plan_weight(e, gate_up, &p)) {
            std::fprintf(stderr, "r3_moe_helper: expert weight is not %s-compatible\n", kDtype);
            return false;
        }
        int64_t nr = 1, nc = 1;
        rad_enc_view(e->rank, e->shape, &nr, &nc);
        if (p.code.bytes != nr * nc / 2 || p.scale.bytes != nr * (nc / 64)) {
            std::fprintf(stderr, "r3_moe_helper: unexpected %s relayout geometry\n",
                         gate_up ? "gate/up" : "down");
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
        std::vector<uint8_t> full_code((size_t)p.code.bytes);
        std::vector<uint8_t> full_scale((size_t)p.scale.bytes);
        if (moe_->relayout(p.params.data(), (int)p.params.size(), 2, enc,
                           p.code_sel.data(), code_planes.data(), 2,
                           full_code.data(), p.code.bytes) != RAD_OK ||
            moe_->relayout(p.params.data(), (int)p.params.size(), 3, enc,
                           p.scale_sel.data(), scale_planes.data(), 2,
                           full_scale.data(), p.scale.bytes) != RAD_OK) {
            std::fprintf(stderr, "r3_moe_helper: libr4d refused expert relayout\n");
            return false;
        }

        for (uint32_t rank = 0; rank < r3::moe_shm::kMaxRanks; ++rank) {
            const int64_t lo = rank_slice_lo(rank, expert_slot & 1u);
            const int64_t width = rank_slice_width(rank, expert_slot & 1u);
            int64_t cb = 0, sb = 0;
            if (!slice_bytes(e, gate_up, expert_slot, rank, &cb, &sb)) return false;
            std::vector<uint8_t> code((size_t)cb);
            std::vector<uint8_t> scale((size_t)sb);
            if (gate_up) {
                const size_t crb = (size_t)nc / 2;
                const size_t srb = (size_t)nc / 64;
                for (int part = 0; part < 2; ++part) {
                    const int64_t src0 = part * kExpertWidth + lo;
                    const int64_t dst0 = part * width;
                    std::memcpy(code.data() + (size_t)dst0 * crb,
                                full_code.data() + (size_t)src0 * crb,
                                (size_t)width * crb);
                    std::memcpy(scale.data() + (size_t)dst0 * srb,
                                full_scale.data() + (size_t)src0 * srb,
                                (size_t)width * srb);
                }
            } else {
                const size_t fcrb = (size_t)nc / 2;
                const size_t fsrb = (size_t)nc / 64;
                const size_t crb = (size_t)width / 2;
                const size_t srb = (size_t)width / 64;
                for (int64_t row = 0; row < nr; ++row) {
                    std::memcpy(code.data() + (size_t)row * crb,
                                full_code.data() + (size_t)row * fcrb + (size_t)lo / 2,
                                crb);
                    std::memcpy(scale.data() + (size_t)row * srb,
                                full_scale.data() + (size_t)row * fsrb + (size_t)lo / 64,
                                srb);
                }
            }

            StoredWeight& w = (*out)[rank];
            w.code = arena_take(code.size());
            w.scale = arena_take(scale.size());
            if (!w.code || !w.scale ||
                !hip_ok(hipMemcpy(w.code, code.data(), code.size(), hipMemcpyHostToDevice),
                        "upload sliced expert codes") ||
                !hip_ok(hipMemcpy(w.scale, scale.data(), scale.size(), hipMemcpyHostToDevice),
                        "upload sliced expert scales"))
                return false;
            w.code_dtype = p.code.dtype;
            w.scale_dtype = p.scale.dtype;
            w.code_bytes = cb;
            w.scale_bytes = sb;
            uint32_t& cd = gate_up ? gu_code_dtype_ : dn_code_dtype_;
            uint32_t& sd = gate_up ? gu_scale_dtype_ : dn_scale_dtype_;
            if ((cd && cd != w.code_dtype) || (sd && sd != w.scale_dtype)) {
                std::fprintf(stderr, "r3_moe_helper: owned expert stored layouts disagree\n");
                return false;
            }
            cd = w.code_dtype;
            sd = w.scale_dtype;
        }
        return true;
    }

    bool build_one_table(const std::vector<void*>& host, void** dev) {
        *dev = alloc_persistent(host.size() * sizeof(void*));
        return *dev && hip_ok(hipMemcpy(*dev, host.data(), host.size() * sizeof(void*),
                                        hipMemcpyHostToDevice), "upload expert pointer table");
    }

    bool build_layer_table(uint32_t l, const int32_t* slot_of_expert) {
        using namespace r3::moe_shm;
        if (l >= model_.layers || !slot_of_expert) return false;
        constexpr size_t nclass = kExperts / 2;
        std::array<uint16_t, kExperts> slot_map{};
        slot_map.fill(0xffffu);
        for (uint32_t e = 0; e < kExperts; ++e) {
            const int32_t es = slot_of_expert[e];
            if (es >= 0) slot_map[e] = (uint16_t)es;
        }

        LayerTables& lt = tables_[l];
        lt.gu_code_dtype = gu_code_dtype_; lt.gu_scale_dtype = gu_scale_dtype_;
        lt.dn_code_dtype = dn_code_dtype_; lt.dn_scale_dtype = dn_scale_dtype_;
        for (uint32_t rank = 0; rank < kMaxRanks; ++rank) {
            for (uint32_t parity = 0; parity < 2; ++parity) {
                std::vector<void*> guc(nclass, nullptr), gus(nclass, nullptr);
                std::vector<void*> dnc(nclass, nullptr), dns(nclass, nullptr);
                for (uint32_t e = 0; e < kExperts; ++e) {
                    const int32_t es = slot_of_expert[e];
                    if (es < 0 || ((uint32_t)es & 1u) != parity) continue;
                    const Expert& x = experts_[(size_t)l * kExperts + e];
                    if (!x.live) continue;
                    const size_t ix = (size_t)es / 2;
                    guc[ix] = x.gu[rank].code; gus[ix] = x.gu[rank].scale;
                    dnc[ix] = x.dn[rank].code; dns[ix] = x.dn[rank].scale;
                }
                if (!build_one_table(guc, &lt.gu_code[rank][parity]) ||
                    !build_one_table(gus, &lt.gu_scale[rank][parity]) ||
                    !build_one_table(dnc, &lt.dn_code[rank][parity]) ||
                    !build_one_table(dns, &lt.dn_scale[rank][parity]))
                    return false;
            }
        }

        const size_t slot_off = (size_t)l * kExperts * sizeof(uint16_t);
        if (!hip_ok(hipMemcpy(static_cast<uint8_t*>(d_expert_slot_.p) + slot_off,
                              slot_map.data(), sizeof(slot_map), hipMemcpyHostToDevice),
                    "upload runtime expert slot map"))
            return false;

        DeviceLayerTables dt{};
        for (uint32_t rank = 0; rank < kMaxRanks; ++rank) {
            for (uint32_t parity = 0; parity < 2; ++parity) {
                dt.gu_code[rank][parity] = lt.gu_code[rank][parity];
                dt.gu_scale[rank][parity] = lt.gu_scale[rank][parity];
                dt.dn_code[rank][parity] = lt.dn_code[rank][parity];
                dt.dn_scale[rank][parity] = lt.dn_scale[rank][parity];
            }
        }
        dt.expert_slot = static_cast<const uint16_t*>(d_expert_slot_.p) + (size_t)l * kExperts;
        return hip_ok(hipMemcpy(static_cast<uint8_t*>(d_layer_tables_.p) +
                                    (size_t)l * sizeof(DeviceLayerTables),
                                &dt, sizeof(dt), hipMemcpyHostToDevice),
                      "upload runtime layer table index");
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
                                      int64_t parts, const char* order,
                                      int64_t n_odd = -1, int64_t k_odd = -1) const {
        if (n_odd < 0) n_odd = n;
        if (k_odd < 0) k_odd = k;
        return {
            pi("M", rows), pi("N", n), pi("K", k), pi("N_odd", n_odd), pi("K_odd", k_odd),
            pi("parts", parts), pi("n_expert", r3::moe_shm::kExperts),
            pi("top_k", r3::moe_shm::kTopK), pi("group", kGroup),
            ps("a_order", order), ps("dtype", kDtype)
        };
    }

    bool scratch_need(const RadKernelInfo* r, std::vector<RadTensor> ts,
                      std::vector<RadParam> ps, size_t* need) {
        if (!r || !need) return false;
        if (!r->scratch) return true;
        RadArgs a{};
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        a.rank = 0; a.world_size = 1;
        const int64_t n = r->scratch(&a);
        if (n < 0) return false;
        *need = std::max(*need, (size_t)std::max<int64_t>(n, 1));
        return true;
    }

    bool prepare_lane_scratch(LaneWork& w, uint32_t rank, uint32_t rows, uint32_t routes) {
        using namespace r3::moe_shm;
        size_t need = 1;
        std::vector<RadParam> aqp = {
            pi("M", rows), pi("n", kHidden), pi("group", kGroup), ps("dtype", "bf16")
        };
        if (!scratch_need(had_quant_,
                          { tensor(w.d_act.p, RAD_BF16, {rows, kHidden}),
                            tensor(w.d_aq.p, RAD_F8E4M3, {rows, kHidden}),
                            tensor(w.d_as.p, RAD_F32, {rows, kHidden / kGroup}) },
                          aqp, &need))
            return false;

        const int64_t w0 = rank_slice_width(rank, 0);
        const int64_t w1 = rank_slice_width(rank, 1);
        std::vector<RadParam> gp =
            gemm_params(rows, 2 * w0, kHidden, 2, "token", 2 * w1, kHidden);
        if (!scratch_need(moe_,
                          { tensor(w.d_aq.p, RAD_F8E4M3, {rows, kHidden}),
                            tensor(w.d_as.p, RAD_F32, {rows, kHidden / kGroup}),
                            table_tensor(w.gu_code[0].p, gu_code_dtype_, 2 * w0, kHidden),
                            table_tensor(w.gu_scale[0].p, gu_scale_dtype_, 2 * w0, kHidden / 64),
                            tensor(w.d_sorted.p, RAD_I32, {routes}),
                            tensor(w.d_off.p, RAD_I32, {kExperts + 1}),
                            table_tensor(w.gu_code[1].p, gu_code_dtype_, 2 * w1, kHidden),
                            table_tensor(w.gu_scale[1].p, gu_scale_dtype_, 2 * w1, kHidden / 64),
                            tensor(w.d_gu.p, RAD_BF16, {routes, kRankGateUpWidth}) },
                          gp, &need))
            return false;

        std::vector<RadParam> gqp = {
            pi("M", routes), pi("n", kRankWidth), pi("group", kGroup),
            ps("dtype", "bf16"), ps("act", "silu")
        };
        if (!scratch_need(gated_quant_,
                          { tensor(w.d_gu.p, RAD_BF16, {routes, kRankGateUpWidth}),
                            tensor(w.d_mid.p, RAD_F8E4M3, {routes, kRankWidth}),
                            tensor(w.d_mid_s.p, RAD_F32, {routes, kRankWidth / kGroup}),
                            absent() },
                          gqp, &need))
            return false;

        std::vector<RadParam> dp =
            gemm_params(rows, kHidden, w0, 1, "sorted", kHidden, w1);
        if (!scratch_need(moe_,
                          { tensor(w.d_mid.p, RAD_F8E4M3, {routes, kRankWidth}),
                            tensor(w.d_mid_s.p, RAD_F32, {routes, kRankWidth / kGroup}),
                            table_tensor(w.dn_code[0].p, dn_code_dtype_, kHidden, w0),
                            table_tensor(w.dn_scale[0].p, dn_scale_dtype_, kHidden, w0 / 64),
                            tensor(w.d_sorted.p, RAD_I32, {routes}),
                            tensor(w.d_off.p, RAD_I32, {kExperts + 1}),
                            table_tensor(w.dn_code[1].p, dn_code_dtype_, kHidden, w1),
                            table_tensor(w.dn_scale[1].p, dn_scale_dtype_, kHidden, w1 / 64),
                            tensor(w.d_down.p, RAD_BF16, {routes, kHidden}) },
                          dp, &need))
            return false;

        std::vector<RadParam> mp = {
            pi("M", rows), pi("n", kHidden), pi("top_k", kTopK), ps("dtype", "bf16")
        };
        if (!scratch_need(gather_,
                          { tensor(w.d_down.p, RAD_BF16, {routes, kHidden}),
                            tensor(w.d_ew.p, RAD_BF16, {rows, kTopK}),
                            tensor(w.d_sorted.p, RAD_I32, {routes}),
                            absent(), absent(), tensor(w.d_act.p, RAD_BF16, {rows, kHidden}) },
                          mp, &need))
            return false;
        return w.scratch.ensure(need);
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
               w.d_act.ensure((size_t)rows * r3::moe_shm::kHidden * 2) &&
               w.d_flat.ensure((size_t)routes * 4) &&
               w.d_sorted.ensure((size_t)tr * 4) &&
               w.d_off.ensure((r3::moe_shm::kExperts + 1ull) * 4) &&
               w.d_ew.ensure((size_t)tr * 2) &&
               w.d_inv.ensure((size_t)tr * 4) &&
               w.d_aq.ensure((size_t)rows * r3::moe_shm::kHidden) &&
               w.d_as.ensure((size_t)rows * (r3::moe_shm::kHidden / kGroup) * 4) &&
               w.d_gu.ensure((size_t)tr * kRankGateUpWidth * 2) &&
               w.d_mid.ensure((size_t)tr * kRankWidth) &&
               w.d_mid_s.ensure((size_t)tr * (kRankWidth / kGroup) * 4) &&
               w.d_down.ensure((size_t)tr * r3::moe_shm::kHidden * 2) &&
               (engine_ranks_ != 1 || w.d_down2.ensure((size_t)tr * r3::moe_shm::kHidden * 2));
    }

    bool ensure_work(uint32_t rows, uint32_t tr) {
        return d_act_.ensure((size_t)rows * r3::moe_shm::kHidden * 2) &&
               d_aq_.ensure((size_t)rows * r3::moe_shm::kHidden) &&
               d_as_.ensure((size_t)rows * (r3::moe_shm::kHidden / kGroup) * 4) &&
               d_sorted_.ensure((size_t)tr * 4) &&
               d_off_.ensure((r3::moe_shm::kExperts + 1ull) * 4) &&
               d_ew_.ensure((size_t)rows * r3::moe_shm::kTopK * 2) &&
               d_gu_.ensure((size_t)tr * kRankGateUpWidth * 2) &&
               d_mid_.ensure((size_t)tr * kRankWidth) &&
               d_mid_s_.ensure((size_t)tr * (kRankWidth / kGroup) * 4) &&
               d_down_.ensure((size_t)tr * r3::moe_shm::kHidden * 2) &&
               d_result_.ensure((size_t)rows * r3::moe_shm::kHidden * 2);
    }
};

} // namespace r3_moe_gpu
