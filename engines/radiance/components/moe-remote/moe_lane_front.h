#pragma once

// Lane front end: activation prep, table select, route count/prefix/scatter, fused small front (pure move of moe_remote_gpu.h, RCS02).
#include "moe_lane_common.h"

namespace r3_moe_gpu {

__global__ void lane_prepare_activation(
    r3::moe_shm::Request* q, const uint8_t* src_q, const float* src_s,
    uint8_t* dst_q, float* dst_s,
    uint32_t rank, uint32_t sequence, uint32_t max_rows, uint32_t layers,
    uint64_t* stats) {
    using namespace r3::moe_shm;
    const bool valid = lane_request_valid(q, rank, sequence, max_rows, layers);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        q->status = valid ? 0 : -EINVAL;
        q->trace[3] = wall_clock64();   /* R3_MOE_WORKER_TRACE: this device's clock */
        if (valid) {
            atomicAdd(reinterpret_cast<unsigned long long*>(stats) + 0, 1ull);
            atomicAdd(reinterpret_cast<unsigned long long*>(stats) + 1,
                      (unsigned long long)q->rows);
            atomicAdd(reinterpret_cast<unsigned long long*>(stats) + 2,
                      (unsigned long long)q->route_count);
        }
    }
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t q_total = (uint64_t)max_rows * g_shape.hidden;
    if (i < q_total) {
        const uint32_t row = (uint32_t)(i / g_shape.hidden);
        dst_q[i] = valid && row < q->rows ? src_q[i] : 0;
    }
    const uint32_t scale_cols = g_shape.hidden / g_shape.act_group;
    const uint64_t s_total = (uint64_t)max_rows * scale_cols;
    if (i < s_total) {
        const uint32_t row = (uint32_t)(i / scale_cols);
        dst_s[i] = valid && row < q->rows ? src_s[i] : 0.0f;
    }
}

/* The same copy, 16 bytes a thread. The byte-a-thread form above reads the engine's mapped host memory (uncached,
 * over PCIe) one byte at a time: 5 MB took ~0.35-0.8 ms. Rows past q->rows or an invalid request zero-fill as before;
 * the host checks the sources are 16-byte aligned. */
__global__ void lane_prepare_activation_v(
    r3::moe_shm::Request* q, const uint4* src_q, const uint4* src_s,
    uint4* dst_q, uint4* dst_s,
    uint32_t rank, uint32_t sequence, uint32_t max_rows, uint32_t layers,
    uint64_t* stats) {
    using namespace r3::moe_shm;
    const bool valid = lane_request_valid(q, rank, sequence, max_rows, layers);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        q->status = valid ? 0 : -EINVAL;
        q->trace[3] = wall_clock64();   /* R3_MOE_WORKER_TRACE: this device's clock */
        if (valid) {
            atomicAdd(reinterpret_cast<unsigned long long*>(stats) + 0, 1ull);
            atomicAdd(reinterpret_cast<unsigned long long*>(stats) + 1,
                      (unsigned long long)q->rows);
            atomicAdd(reinterpret_cast<unsigned long long*>(stats) + 2,
                      (unsigned long long)q->route_count);
        }
    }
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t q_per_row = g_shape.hidden / 16;                              // 16-byte pieces of a row of codes
    const uint32_t s_per_row = (g_shape.hidden / g_shape.act_group) * 4 / 16;    // and of a row of scales (set_shape checks whole pieces)
    if (i < (uint64_t)max_rows * q_per_row) {
        const uint32_t row = (uint32_t)(i / q_per_row);
        dst_q[i] = valid && row < q->rows ? src_q[i] : make_uint4(0, 0, 0, 0);
    }
    if (i < (uint64_t)max_rows * s_per_row) {
        const uint32_t row = (uint32_t)(i / s_per_row);
        dst_s[i] = valid && row < q->rows ? src_s[i] : make_uint4(0, 0, 0, 0);
    }
}

__global__ void lane_select_tables(
    const r3::moe_shm::Request* q, const DeviceLayerTables* layers_table,
    uint32_t layers, uint32_t rank, uint32_t slice, uint32_t sequence, uint32_t max_rows, uint32_t experts,
    void** guc0, void** gus0, void** dnc0, void** dns0,
    void** guc1, void** gus1, void** dnc1, void** dns1) {
    using namespace r3::moe_shm;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= experts / 2) return;
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
    uint32_t rank, uint32_t sequence, uint32_t max_rows, uint32_t layers, uint32_t experts,
    uint32_t* counts, int32_t* flat, uint16_t* ew) {
    using namespace r3::moe_shm;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!lane_request_valid(q, rank, sequence, max_rows, layers) || i >= q->route_count) return;
    const Route r = route[i];
    if (r.token < 0 || (uint32_t)r.token >= q->rows || r.expert < 0 ||
        r.expert >= (int32_t)experts || r.slot >= g_shape.top_k) {
        flat[i] = -1;
        return;
    }
    const uint16_t es = layers_table[q->layer].expert_slot[(uint32_t)r.expert];
    if (es >= experts) {
        flat[i] = -1;
        atomicExch(&q->status, -EINVAL);
        return;
    }
    const uint32_t f = (uint32_t)r.token * g_shape.top_k + r.slot;
    flat[i] = (int32_t)f;
    ew[f] = r.weight_bf16;
    atomicAdd(counts + (uint32_t)es, 1u);
}

/* Exclusive prefix over the expert counts in one block of N threads (N a power of two, at least the expert count: threads past it
 * count zero; was one thread walking all 512: ~43 us a request). off holds experts + 1 entries, the last the total. */
template <uint32_t N>
__global__ __launch_bounds__(N) void lane_route_prefix(const uint32_t* counts, int32_t* off, uint32_t* cursor, uint32_t experts) {
    constexpr uint32_t n = N;
    __shared__ uint32_t s[n];
    const uint32_t e = threadIdx.x;
    if (blockIdx.x || e >= n) return;
    const uint32_t mine = e < experts ? counts[e] : 0u;
    s[e] = mine;
    __syncthreads();
    for (uint32_t d = 1; d < n; d <<= 1) {
        const uint32_t add = e >= d ? s[e - d] : 0u;
        __syncthreads();
        s[e] += add;
        __syncthreads();
    }
    const uint32_t excl = s[e] - mine;
    if (e < experts) {
        off[e] = (int32_t)excl;
        cursor[e] = excl;
    }
    if (e == n - 1) off[experts] = (int32_t)s[e];
}

__global__ void lane_route_scatter(
    const r3::moe_shm::Route* route, const r3::moe_shm::Request* q,
    const DeviceLayerTables* layers_table,
    uint32_t rank, uint32_t sequence, uint32_t max_rows, uint32_t layers, uint32_t experts,
    const int32_t* flat, uint32_t* cursor, int32_t* sorted, int32_t* sorted_expert) {
    using namespace r3::moe_shm;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!lane_request_valid(q, rank, sequence, max_rows, layers) ||
        i >= q->route_count || flat[i] < 0) return;
    const uint32_t e = (uint32_t)route[i].expert;
    const uint32_t es = layers_table[q->layer].expert_slot[e];
    if (es >= experts) return;
    const uint32_t pos = atomicAdd(cursor + es, 1u);
    sorted[pos] = flat[i];
    if (sorted_expert) sorted_expert[pos] = (int32_t)e;
}

/* R3_MOE_WORKER_FUSEROUTE: a small (decode) lane request's whole front end in one block: activation copy, table
 * selection and the routing prologue. A request has at most
 * 8 rows x top-k routes, so one block of N threads (N a power of two, at least 256 and the expert count) does what four memsets and lane_route_count / _prefix /
 * _scatter / _inverse did across ten launches: clear the sorted order, its inverse and the weights; count the routes
 * of each expert slot; scan the counts into the expert offsets; place every route and its inverse. A route's place in
 * its expert's run is the number of earlier routes with the same slot (the atomic cursor made that order arbitrary), so
 * the sorted order is now deterministic. A route's output does not depend on its place and the reduce reads through the
 * inverse, so the result is the same. */
constexpr uint32_t kFuseMaxRoutes = 256;
template <uint32_t N>
__global__ __launch_bounds__(N) void lane_front_small(
    const r3::moe_shm::Route* route, r3::moe_shm::Request* q, const DeviceLayerTables* layers_table,
    const uint4* src_q, const uint4* src_s, uint4* dst_q, uint4* dst_s, uint64_t* stats,
    uint32_t rank, uint32_t slice, uint32_t sequence, uint32_t max_rows, uint32_t layers, uint32_t routes_cap,
    uint32_t experts, int32_t* off, int32_t* sorted, int32_t* inv, uint16_t* ew,
    void** guc0, void** gus0, void** dnc0, void** dns0, void** guc1, void** gus1, void** dnc1, void** dns1) {
    using namespace r3::moe_shm;
    constexpr uint32_t n = N;
    static_assert(N >= kFuseMaxRoutes, "one thread a route");
    __shared__ uint32_t s_cnt[n];
    __shared__ uint32_t s_excl[n];
    __shared__ uint16_t r_es[kFuseMaxRoutes];
    __shared__ int32_t r_f[kFuseMaxRoutes];
    const uint32_t t = threadIdx.x;
    /* What lane_prepare_activation_v and lane_select_tables did, ahead of the routing: the request's own status and
     * stats, the activation rows copied from the engine's mapped host memory (their PCIe reads in flight together
     * with the routes' below), and the lane's expert tables. */
    const bool valid0 = lane_request_valid(q, rank, sequence, max_rows, layers);
    if (t == 0) {
        q->status = valid0 ? 0 : -EINVAL;
        q->trace[3] = wall_clock64();   /* R3_MOE_WORKER_TRACE: this device's clock */
        if (valid0) {
            atomicAdd(reinterpret_cast<unsigned long long*>(stats) + 0, 1ull);
            atomicAdd(reinterpret_cast<unsigned long long*>(stats) + 1, (unsigned long long)q->rows);
            atomicAdd(reinterpret_cast<unsigned long long*>(stats) + 2, (unsigned long long)q->route_count);
        }
    }
    {
        const uint32_t q_per_row = g_shape.hidden / 16;
        const uint32_t s_per_row = (g_shape.hidden / g_shape.act_group) * 4 / 16;
        for (uint32_t i = t; i < max_rows * q_per_row; i += n)
            dst_q[i] = valid0 && i / q_per_row < q->rows ? src_q[i] : make_uint4(0, 0, 0, 0);
        for (uint32_t i = t; i < max_rows * s_per_row; i += n)
            dst_s[i] = valid0 && i / s_per_row < q->rows ? src_s[i] : make_uint4(0, 0, 0, 0);
    }
    if (t < experts / 2) {
        const uint32_t i = t;
        if (!valid0) {
            guc0[i] = gus0[i] = dnc0[i] = dns0[i] = nullptr;
            guc1[i] = gus1[i] = dnc1[i] = dns1[i] = nullptr;
        } else {
            const DeviceLayerTables tl = layers_table[q->layer];
            guc0[i] = static_cast<void**>(tl.gu_code[slice][0])[i];
            gus0[i] = static_cast<void**>(tl.gu_scale[slice][0])[i];
            dnc0[i] = static_cast<void**>(tl.dn_code[slice][0])[i];
            dns0[i] = static_cast<void**>(tl.dn_scale[slice][0])[i];
            guc1[i] = static_cast<void**>(tl.gu_code[slice][1])[i];
            gus1[i] = static_cast<void**>(tl.gu_scale[slice][1])[i];
            dnc1[i] = static_cast<void**>(tl.dn_code[slice][1])[i];
            dns1[i] = static_cast<void**>(tl.dn_scale[slice][1])[i];
        }
    }
    __syncthreads();   /* the status above is written before a route error below can overwrite it */
    for (uint32_t i = t; i < routes_cap; i += n) {
        sorted[i] = -1;
        inv[i] = -1;
        ew[i] = 0;
    }
    s_cnt[t] = 0;
    const bool valid = lane_request_valid(q, rank, sequence, max_rows, layers) &&
                       q->route_count <= kFuseMaxRoutes && q->route_count <= routes_cap;
    const uint32_t nr = valid ? q->route_count : 0u;
    __syncthreads();
    if (t < nr) {
        const Route r = route[t];
        r_f[t] = -1;
        r_es[t] = 0;
        if (!(r.token < 0 || (uint32_t)r.token >= q->rows || r.expert < 0 || r.expert >= (int32_t)experts ||
              r.slot >= g_shape.top_k)) {
            const uint16_t es = layers_table[q->layer].expert_slot[(uint32_t)r.expert];
            if (es >= experts) {
                atomicExch(&q->status, -EINVAL);
            } else {
                const uint32_t f = (uint32_t)r.token * g_shape.top_k + r.slot;
                r_f[t] = (int32_t)f;
                r_es[t] = es;
                ew[f] = r.weight_bf16;
                atomicAdd(&s_cnt[es], 1u);
            }
        }
    }
    __syncthreads();
    const uint32_t mine = s_cnt[t];
    for (uint32_t d = 1; d < n; d <<= 1) {
        const uint32_t add = t >= d ? s_cnt[t - d] : 0u;
        __syncthreads();
        s_cnt[t] += add;
        __syncthreads();
    }
    const uint32_t excl = s_cnt[t] - mine;
    if (t < experts) off[t] = (int32_t)excl;
    s_excl[t] = excl;
    if (t == n - 1) off[experts] = (int32_t)s_cnt[t];
    __syncthreads();
    if (t < nr && r_f[t] >= 0) {
        uint32_t before = 0;
        for (uint32_t j = 0; j < t; ++j) before += (r_f[j] >= 0 && r_es[j] == r_es[t]) ? 1u : 0u;
        const uint32_t pos = s_excl[r_es[t]] + before;
        sorted[pos] = r_f[t];
        inv[r_f[t]] = (int32_t)pos;
    }
}

}  // namespace r3_moe_gpu
