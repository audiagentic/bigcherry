#pragma once

#include "rad_abi.h"
#include "worker_generation_tables.h"
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

/* The model shape the lane kernels run with (RCS09-06, RCS05): Runner::set_shape stores it and every kernel that reads it takes it as an
 * argument, so no device-side global is shared between runners or backends and a second model is a second Runner. */
struct LaneShape {
    uint32_t hidden, top_k, act_group, experts;
    uint32_t world;   /* the segment's engine width (Header::rank_count): a request of another width is refused (RAD49) */
};

constexpr int64_t kExpertWidth = 640;
constexpr int64_t kGateUpWidth = 2 * kExpertWidth;
constexpr int64_t kGroup = 128;
constexpr const char* kDtype = "w4nl64a8h";

/* One stored form an expert: full width, whatever the engine's TP width (RAD49), in two tables by expert-slot parity. */
struct DeviceLayerTables {
    void* gu_code[2]{};
    void* gu_scale[2]{};
    void* dn_code[2]{};
    void* dn_scale[2]{};
    const uint16_t* expert_slot = nullptr;
};

__device__ __forceinline__ void lane_store_sys_rel(uint32_t* p, uint32_t v) {
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ uint32_t lane_load_sys_acq(const uint32_t* p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ bool lane_request_valid(
    const r3::moe_shm::Request* q, uint32_t slot, uint32_t sequence,
    uint32_t max_rows, uint32_t layers, const LaneShape& shape) {
    using namespace r3::moe_shm;
    /* The request is this lane's slot, for the consumer its slot and its layer both name (RAD49); it always asks for the
     * full expert width. */
    return q && q->sequence == sequence &&
           request_lane_identity_valid(q->slot, q->world, q->layer, slot, shape.world) &&
           q->layer < layers && q->rows > 0 && q->rows <= max_rows &&
           q->hidden == shape.hidden && q->route_count > 0 &&
           q->route_count <= q->rows * shape.top_k &&
           q->activation_stride == shape.hidden * (uint32_t)sizeof(uint16_t) &&
           q->activation_q_stride == shape.hidden &&
           q->activation_s_stride == (shape.hidden / shape.act_group) * (uint32_t)sizeof(float);
}

}  // namespace r3_moe_gpu
