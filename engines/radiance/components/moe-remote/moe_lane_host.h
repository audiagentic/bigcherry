#pragma once

// Host-side lane types and helpers: hip_ok, RadTensor builders, Buffer, StoredWeight, WeightPlan (pure move of moe_remote_gpu.h, RCS02).
#include "moe_lane_common.h"

namespace r3_moe_gpu {

inline bool hip_ok(hipError_t e, const char* what) {
    if (e == hipSuccess) return true;
    std::fprintf(stderr, "r3_moe_worker: %s: %s\n", what, hipGetErrorString(e));
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
    std::array<StoredWeight, r3::moe_shm::kMaxEngineRanks> gu;
    std::array<StoredWeight, r3::moe_shm::kMaxEngineRanks> dn;
    bool live = false;
};

struct LayerTables {
    void* gu_code[r3::moe_shm::kMaxEngineRanks][2]{};
    void* gu_scale[r3::moe_shm::kMaxEngineRanks][2]{};
    void* dn_code[r3::moe_shm::kMaxEngineRanks][2]{};
    void* dn_scale[r3::moe_shm::kMaxEngineRanks][2]{};
    uint32_t gu_code_dtype = 0, gu_scale_dtype = 0;
    uint32_t dn_code_dtype = 0, dn_scale_dtype = 0;
};

struct LaneWork {
    hipStream_t stream = nullptr;
    hipEvent_t sample_start = nullptr;
    hipEvent_t sample_stop = nullptr;
    /* Stage marks inside a sampled request: routes, gate/up GEMM, gated quant, down GEMM (then reduce = stop). */
    static constexpr int kStages = 4;
    hipEvent_t stage[kStages] = {};
    double stage_us[kStages + 1] = {};
    /* Sampled requests: experts by routed rows - 1, 2, 3-4, 5-8, 9-16, 17-32, 33-64, 65+ (RAD06 tile sizing). */
    uint64_t run_hist[8] = {};
    uint32_t sample_sequence = 0;
    bool sample_pending = false;
    uint32_t cap_rows = 0;
    Buffer d_counts, d_cursor, d_flat, d_sorted, d_off, d_ew, d_inv, d_dbg;
    Buffer d_rowmask;   // one byte a row: the rows this request's routes name (sparse activation copy, RAD23)
    uint32_t dbg_routes = 0;
    Buffer d_act, d_aq, d_as, d_gu, d_mid, d_mid_s, d_down, d_stats, scratch;
    Buffer d_chunk_cnt;       /* uint32[kCombineChunks]: the large-lane reduce's per-chunk row counts (RAD42) */
    Buffer d_publish_limit;   /* uint32: newest sequence lane_publish_done may publish (RAD20 park); UINT32_MAX otherwise */
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


}  // namespace r3_moe_gpu
