#pragma once

#include "rad_abi.h"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
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
#include <type_traits>
#include <utility>
#include <vector>

/* GPU half of r3_moe_worker. This header is included after Container is defined in
 * r3_moe_worker.cpp, so it can consume the installed container ABI without linking core/private
 * Radiance code. The only executable ABI it uses is libr4d's rad_plugin surface. */
namespace r3_moe_gpu {

/* The model shape the lane kernels run with (RCS09-06): set once per process by Runner::set_shape, read by every kernel, so a kernel
 * takes no shape argument and a second model is a second set_shape, not a recompile. */
struct LaneShape {
    uint32_t hidden, top_k, act_group, experts;
};
__constant__ LaneShape g_shape;

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
           q->hidden == g_shape.hidden && q->route_count > 0 &&
           q->route_count <= q->rows * g_shape.top_k &&
           q->activation_stride == g_shape.hidden * (uint32_t)sizeof(uint16_t) &&
           q->activation_q_stride == g_shape.hidden &&
           q->activation_s_stride == (g_shape.hidden / g_shape.act_group) * (uint32_t)sizeof(float);
}

}  // namespace r3_moe_gpu
